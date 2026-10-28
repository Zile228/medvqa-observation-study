"""Pinned multimodal model backends, greedy decoding and answer-label probabilities."""

import json
import math
import re
from pathlib import Path

MODELS = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/models.json").read_text(
        encoding="utf-8"
    )
)
QWEN_MIN_PIXELS, QWEN_MAX_PIXELS = (3136, 448 * 448)
PRECISIONS = ("PREC-0", "PREC-1", "PREC-2")
YESNO_LABELS, MCQ_LABELS = (("Yes", "No"), ("A", "B", "C", "D"))


def build_label_map(decode_one, vocab_size, labels=YESNO_LABELS + MCQ_LABELS):
    """Find all single-token realizations of the accepted answer labels.

    Args:
        decode_one: Callable decoding one vocabulary token without stripping whitespace.
        vocab_size: Number of vocabulary token IDs to inspect.
        labels: Canonical answer labels to match after stripping token whitespace.

    Returns:
        dict[str, list[int]]: Accepted vocabulary token IDs for each label.

    Raises:
        RuntimeError: At least one answer label has no single-token realization.
    """
    out = {c: [] for c in labels}
    for t in range(vocab_size):
        s = decode_one(t).strip()
        for c in labels:
            if (
                c in YESNO_LABELS
                and s in (c, c.lower())
                or (c in MCQ_LABELS and s == c)
            ):
                out[c].append(t)
    empty = [c for c, ids in out.items() if not ids]
    if empty:
        raise RuntimeError(f"No single-token realization found for labels {empty}")
    return out


def label_logprobs(logits_1d, label_map, labels):
    """Compute normalized answer-label probabilities from first-token logits.

    Args:
        logits_1d: One-dimensional vocabulary logits tensor.
        label_map: Accepted token IDs for each canonical label.
        labels: Ordered labels for the current question.

    Returns:
        dict: Label log probabilities, normalized probabilities and finite-status
            metadata.
    """
    import torch

    lsm = torch.log_softmax(logits_1d.detach().float().cpu(), dim=-1)
    n = lsm.shape[0]
    lp = {}
    for c in labels:
        ids = [i for i in label_map[c] if i < n]
        lp[c] = float(torch.logsumexp(lsm[ids], dim=0)) if ids else float("-inf")
    finite = all((math.isfinite(v) for v in lp.values())) and bool(
        torch.isfinite(logits_1d.float()).all()
    )
    m = max(lp.values())
    if math.isfinite(m):
        z = sum((math.exp(v - m) for v in lp.values()))
        p = {c: math.exp(v - m) / z for c, v in lp.items()}
        raw = math.exp(m) * z
    else:
        p, raw = ({c: None for c in lp}, None)
    top = torch.topk(lsm, 5)
    return {
        "labels": list(labels),
        "lp": lp,
        "p_tilde": p,
        "raw_mass": raw,
        "finite": finite,
        "top5": [[int(i), float(v)] for v, i in zip(top.values, top.indices)],
    }


def _cast(x, dtype):
    import torch

    if torch.is_tensor(x):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, tuple):
        return tuple((_cast(v, dtype) for v in x))
    if isinstance(x, list):
        return [_cast(v, dtype) for v in x]
    if isinstance(x, dict):
        try:
            return type(x)(**{k: _cast(v, dtype) for k, v in x.items()})
        except Exception:
            return {k: _cast(v, dtype) for k, v in x.items()}
    return x


def upcast_fp32(module, low_dtype):
    """Keep a module in float32 while adapting its input/output dtypes.

    Keep `module` in fp32: inputs cast to fp32, outputs cast back to the model's low
    dtype.
    """
    import torch

    module.float()
    module.register_forward_pre_hook(
        lambda m, a, k: (_cast(a, torch.float32), _cast(k, torch.float32)),
        with_kwargs=True,
    )
    module.register_forward_hook(lambda m, a, out: _cast(out, low_dtype))


def find_vision_tower(model):
    for name in ("vision_tower", "visual", "model.vision_tower", "model.visual"):
        obj = model
        try:
            for part in name.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (list, tuple)):
            obj = obj[0]
        if obj is not None:
            return (name, obj)
    raise RuntimeError("vision tower not found")


def apply_prec1(model, has_vision=True):
    """Keep vision and normalization modules in float32.

    PREC-1: fp16 model; vision tower + every LayerNorm/RMSNorm kept in fp32 (via dtype
    hooks).
    """
    import torch

    low = torch.float16
    vt_ids = set()
    if has_vision:
        _, vt = find_vision_tower(model)
        upcast_fp32(vt, low)
        vt_ids = {id(m) for m in vt.modules()}
    n = 0
    for m in model.modules():
        if id(m) in vt_ids:
            continue
        cname = type(m).__name__
        if (
            isinstance(m, torch.nn.LayerNorm)
            or "RMSNorm" in cname
            or cname.endswith("LayerNorm")
        ):
            upcast_fp32(m, low)
            n += 1
    return {"fp32_norm_modules": n, "vision_fp32": has_vision}


PREC2_KEEP_VISION_FP32 = False
PREC2_SKIP_MODULES = ["vision_tower", "visual", "multi_modal_projector", "mm_projector"]


def load_kwargs(precision):
    import torch

    if precision == "PREC-0" or precision == "PREC-1":
        return {"torch_dtype": torch.float16}
    if precision == "PREC-2":
        from transformers import BitsAndBytesConfig

        extra = (
            {"llm_int8_skip_modules": PREC2_SKIP_MODULES}
            if PREC2_KEEP_VISION_FP32
            else {}
        )
        q = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            **extra,
        )
        return {"torch_dtype": torch.float32, "quantization_config": q}
    raise ValueError(precision)


GREEDY_KW = {
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}
_NEUTRALISE = {
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "encoder_repetition_penalty": 1.0,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "typical_p": None,
    "length_penalty": 1.0,
    "bad_words_ids": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "sequence_bias": None,
    "min_new_tokens": None,
    "min_length": 0,
}


def force_greedy(model):
    """Neutralise sampling/penalty fields of model.generation_config in place.

    Returns (original, effective) dicts.
    """
    gc = model.generation_config
    original = gc.to_dict()
    for k, v in _NEUTRALISE.items():
        if hasattr(gc, k):
            setattr(gc, k, v)
    assert_greedy(gc)
    return (original, gc.to_dict())


def assert_greedy(gc):
    bad = {}
    if getattr(gc, "do_sample", False):
        bad["do_sample"] = gc.do_sample
    if (getattr(gc, "num_beams", 1) or 1) != 1:
        bad["num_beams"] = gc.num_beams
    if getattr(gc, "repetition_penalty", 1.0) not in (None, 1.0):
        bad["repetition_penalty"] = gc.repetition_penalty
    if getattr(gc, "no_repeat_ngram_size", 0) not in (None, 0):
        bad["no_repeat_ngram_size"] = gc.no_repeat_ngram_size
    if bad:
        raise RuntimeError(f"generation config is not plain greedy: {bad}")


def _gen_result(tokenizer, new_ids, max_new_tokens, eos_ids, first_logits, method):
    import torch

    ids = [int(i) for i in new_ids]
    n_raw = len(ids)
    while ids and ids[-1] in eos_ids:
        ids.pop()
    hit_eos = len(ids) < n_raw
    truncated = n_raw >= max_new_tokens and (not hit_eos)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    finite = (
        None
        if first_logits is None
        else bool(torch.isfinite(first_logits.float()).all())
    )
    return {
        "text": text,
        "n_output_tokens": len(ids),
        "truncated": truncated,
        "finish_reason": "length" if truncated else "stop",
        "first_logits": first_logits,
        "first_step_finite": finite,
        "logits_method": method,
    }


class HFVisionBackend:
    """LLaVA-1.5 (HF) and Qwen2.5-VL (HF) in ENV-1.

    Image first, then text, default chat template, no system prompt.
    """

    def __init__(self, key, precision, device_map="auto"):
        import torch
        import transformers as T

        self.key, self.precision, self.spec = (key, precision, MODELS[key])
        mid, rev = (self.spec["id"], self.spec["revision"])
        kw = load_kwargs(precision)
        if self.spec["family"] == "llava_hf":
            self.processor = T.AutoProcessor.from_pretrained(mid, revision=rev)
            self.model = T.LlavaForConditionalGeneration.from_pretrained(
                mid, revision=rev, device_map=device_map, **kw
            )
        elif self.spec["family"] == "qwen25vl":
            self.processor = T.AutoProcessor.from_pretrained(
                mid,
                revision=rev,
                min_pixels=QWEN_MIN_PIXELS,
                max_pixels=QWEN_MAX_PIXELS,
            )
            self.model = T.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                mid, revision=rev, device_map=device_map, **kw
            )
        else:
            raise ValueError(key)
        self.model.eval()
        self.gen_config_original, self.gen_config_effective = force_greedy(self.model)
        self.prec_info = apply_prec1(self.model) if precision == "PREC-1" else {}
        self.tokenizer = self.processor.tokenizer
        self.pixel_dtype = (
            torch.float32 if precision in ("PREC-1", "PREC-2") else torch.float16
        )
        self.eos_ids = {
            i
            for i in [
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                getattr(self.model.generation_config, "eos_token_id", None),
            ]
            if isinstance(i, int)
        }
        ge = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(ge, list):
            self.eos_ids |= set(ge)
        self.identity = {
            "model": mid,
            "revision": rev,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "precision": precision,
            "prec_info": self.prec_info,
            "env": self.spec["env"],
            "transformers": T.__version__,
            "logits_method": "output_logits",
            "chat_template_sha256": _sha(self.processor.chat_template or ""),
            "generate_kwargs": GREEDY_KW,
            "generation_config_original": self.gen_config_original,
            "generation_config_effective": self.gen_config_effective,
        }

    def _inputs(self, image, prompt):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        if prompt not in text:
            raise RuntimeError("chat template dropped the prompt text")
        if self.spec["family"] == "qwen25vl":
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        else:
            inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.pixel_dtype)
        return inputs

    def generate(self, image, prompt, max_new_tokens):
        import torch

        inputs = self._inputs(image, prompt)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **GREEDY_KW,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new = out.sequences[0, n_in:].tolist()
        first = out.logits[0][0].detach().float().cpu() if out.logits else None
        return _gen_result(
            self.tokenizer, new, max_new_tokens, self.eos_ids, first, "output_logits"
        )

    def decode_one(self, t):
        return self.tokenizer.decode(
            [t], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def vocab_size(self):
        return len(self.tokenizer)


class LlavaMedBackend:
    """Load LLaVA-Med using its pinned original implementation.

    LLaVA-Med v1.5 (Mistral) with the ORIGINAL LLaVA-Med code (ENV-2). conv_mode from
    the official eval script at the pinned commit (parameter, recorded in run_config).
    Log-probs: separate forward on the prompt.
    """

    def __init__(self, precision, conv_mode, device_map="auto"):
        import torch
        from llava import constants as C
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
        from transformers import AutoTokenizer

        self.C, self.conv_templates, self.process_images, self.tokenizer_image_token = (
            C,
            conv_templates,
            process_images,
            tokenizer_image_token,
        )
        self.key, self.precision, self.spec, self.conv_mode = (
            "M",
            precision,
            MODELS["M"],
            conv_mode,
        )
        mid, rev = (self.spec["id"], self.spec["revision"])
        kw = load_kwargs(precision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            mid, revision=rev, use_fast=False
        )
        self.model = LlavaMistralForCausalLM.from_pretrained(
            mid, revision=rev, low_cpu_mem_usage=True, device_map=device_map, **kw
        )
        cfg = self.model.config
        if getattr(cfg, "mm_use_im_patch_token", True):
            self.tokenizer.add_tokens(
                [C.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True
            )
        if getattr(cfg, "mm_use_im_start_end", False):
            self.tokenizer.add_tokens(
                [C.DEFAULT_IM_START_TOKEN, C.DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        self.model.resize_token_embeddings(len(self.tokenizer))
        vt = self.model.get_vision_tower()
        if not vt.is_loaded:
            vt.load_model()
        self.vision_dtype = (
            torch.float32 if precision in ("PREC-1", "PREC-2") else torch.float16
        )
        vt.to(device=self.model.device, dtype=self.vision_dtype)
        self.image_processor = vt.image_processor
        self.prec_info = apply_prec1(self.model) if precision == "PREC-1" else {}
        self.model.eval()
        self.gen_config_original, self.gen_config_effective = force_greedy(self.model)
        self.eos_ids = {
            i
            for i in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
            if isinstance(i, int)
        }
        import transformers

        self.identity = {
            "model": mid,
            "revision": rev,
            "precision": precision,
            "conv_mode": conv_mode,
            "env": "ENV-2",
            "transformers": transformers.__version__,
            "logits_method": "forward",
            "prec_info": self.prec_info,
            "generate_kwargs": GREEDY_KW,
            "generation_config_original": self.gen_config_original,
            "generation_config_effective": self.gen_config_effective,
        }

    def _inputs(self, image, prompt):
        C = self.C
        img_tok = (
            C.DEFAULT_IM_START_TOKEN + C.DEFAULT_IMAGE_TOKEN + C.DEFAULT_IM_END_TOKEN
            if getattr(self.model.config, "mm_use_im_start_end", False)
            else C.DEFAULT_IMAGE_TOKEN
        )
        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], img_tok + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        full = conv.get_prompt()
        ids = (
            self.tokenizer_image_token(
                full, self.tokenizer, C.IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        pix = self.process_images([image], self.image_processor, self.model.config).to(
            self.model.device, dtype=self.vision_dtype
        )
        return (ids, pix, full)

    def generate(self, image, prompt, max_new_tokens, want_logits=True):
        import torch

        ids, pix, _ = self._inputs(image, prompt)
        first = None
        with torch.inference_mode():
            if want_logits:
                first = (
                    self.model(input_ids=ids, images=pix, image_sizes=[image.size])
                    .logits[0, -1]
                    .detach()
                    .float()
                    .cpu()
                )
            out = self.model.generate(
                ids,
                images=pix,
                image_sizes=[image.size],
                **GREEDY_KW,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new = out[0].tolist()
        if len(new) > ids.shape[1] and new[: ids.shape[1]] == ids[0].tolist():
            new = new[ids.shape[1] :]
        return _gen_result(
            self.tokenizer, new, max_new_tokens, self.eos_ids, first, "forward"
        )

    def decode_one(self, t):
        return self.tokenizer.decode(
            [t], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def vocab_size(self):
        return len(self.tokenizer)


class HFTextBackend:
    """Text-only judges (J1 InternLM3, J2 Granite) and EDIT-ALT (Phi-4-mini).

    Greedy, default chat template.
    """

    def __init__(self, key, precision, device_map="auto"):
        import transformers as T

        self.key, self.precision, self.spec = (key, precision, MODELS[key])
        mid, rev, trc = (
            self.spec["id"],
            self.spec["revision"],
            self.spec.get("trust_remote_code", False),
        )
        self.tokenizer = T.AutoTokenizer.from_pretrained(
            mid, revision=rev, trust_remote_code=trc
        )
        self.model = T.AutoModelForCausalLM.from_pretrained(
            mid,
            revision=rev,
            trust_remote_code=trc,
            device_map=device_map,
            **load_kwargs(precision),
        )
        self.model.eval()
        self.gen_config_original, self.gen_config_effective = force_greedy(self.model)
        self.prec_info = (
            apply_prec1(self.model, has_vision=False) if precision == "PREC-1" else {}
        )
        self.eos_ids = {
            i
            for i in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
            if isinstance(i, int)
        }
        ge = getattr(self.model.generation_config, "eos_token_id", None)
        self.eos_ids |= (
            set(ge) if isinstance(ge, list) else {ge} if isinstance(ge, int) else set()
        )
        self.identity = {
            "model": mid,
            "revision": rev,
            "precision": precision,
            "transformers": T.__version__,
            "trust_remote_code": trc,
            "prec_info": self.prec_info,
            "generate_kwargs": GREEDY_KW,
            "generation_config_original": self.gen_config_original,
            "generation_config_effective": self.gen_config_effective,
        }

    def generate_text(self, prompt, max_new_tokens):
        import torch

        msgs = [{"role": "user", "content": prompt}]
        ids = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                **GREEDY_KW,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new = out[0, ids.shape[1] :].tolist()
        r = _gen_result(self.tokenizer, new, max_new_tokens, self.eos_ids, None, None)
        r.pop("first_logits")
        return r


def _sha(s):
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class StubTokenizer:
    VOCAB = [
        "<s>",
        "</s>",
        "Yes",
        "▁Yes",
        "yes",
        "No",
        "▁No",
        "no",
        "A",
        "B",
        "C",
        "D",
        "▁A",
        "▁B",
        "x",
        "Maybe",
        "▁C",
    ]

    def decode(
        self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ):
        return "".join(
            (
                self.VOCAB[i].replace("▁", " ")
                for i in ids
                if not (skip_special_tokens and i < 2)
            )
        )

    def __len__(self):
        return len(self.VOCAB)


class StubVLM:
    """Provide a deterministic model substitute for offline tests.

    Deterministic fake model: observer prompts -> one line per requested item; answer
    prompts -> 'Yes' / 'A' / text. First-step logits are a fixed function of the prompt
    so log-prob code paths are exercised.
    """

    def __init__(self, key="STUB", precision="PREC-0", fail_on=None):
        self.key, self.precision = (key, precision)
        self.tokenizer = StubTokenizer()
        self.identity = {
            "model": f"stub-{key}",
            "revision": "none",
            "precision": precision,
            "logits_method": "output_logits",
            "generate_kwargs": GREEDY_KW,
            "generation_config_effective": None,
        }
        self.fail_on = fail_on
        self.calls = 0

    def generate(self, image, prompt, max_new_tokens):
        import torch

        self.calls += 1
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("simulated crash")
        assert image is not None and hasattr(image, "size")
        if "Report each item below" in prompt:
            items = re.findall(
                "^(\\d+)\\. ([^:\\n]+):", prompt.split("Items:")[1], re.M
            )
            text = "\n".join(
                (
                    (
                        f"{n}"
                        ". "
                        f"{name}"
                        " | visible | image-left area looks small and "
                        "slightly bright "
                        f"{n}"
                    )
                    for n, name in items
                )
            )
        elif "one word: Yes or No" in prompt:
            text = "Yes"
        elif "letter of one option" in prompt:
            text = "A"
        else:
            text = "liver"
        n_tok = max(1, len(text.split()))
        h = int(_sha(prompt)[:8], 16)
        logits = torch.tensor(
            [float((h >> i % 24) % 7) for i in range(len(self.tokenizer))]
        )
        trunc = n_tok >= max_new_tokens
        return {
            "text": text,
            "n_output_tokens": min(n_tok, max_new_tokens),
            "truncated": trunc,
            "finish_reason": "length" if trunc else "stop",
            "first_logits": logits,
            "first_step_finite": True,
            "logits_method": "output_logits",
        }

    def generate_text(self, prompt, max_new_tokens):
        if 'Return JSON only: {"labels"' in prompt:
            n = len(re.findall("^\\d+\\. ", prompt.split("Statements:")[1], re.M))
            body = prompt.split("Statements:")[1]
            sts = re.findall("^\\d+\\. (.*)$", body, re.M)[:n]
            labs = ("supported", "contradicted", "not_verifiable")
            text = json.dumps(
                {
                    "labels": [
                        {"id": i, "label": labs[int(_sha(st)[:6], 16) % 3]}
                        for i, st in enumerate(sts, 1)
                    ]
                }
            )
        elif "items_addressed" in prompt:
            text = (
                '{"items_addressed": 3, "diagnosis": "no", "unsupported_claim": "no"}'
            )
        elif "relevance" in prompt:
            text = '{"relevance": 1}'
        elif "verdict" in prompt:
            text = '{"verdict": "correct"}'
        else:
            text = prompt.split("Text:\n", 1)[-1]
        return {
            "text": text,
            "n_output_tokens": len(text.split()),
            "truncated": False,
            "finish_reason": "stop",
        }

    def decode_one(self, t):
        return self.tokenizer.decode([t])

    def vocab_size(self):
        return len(self.tokenizer)
