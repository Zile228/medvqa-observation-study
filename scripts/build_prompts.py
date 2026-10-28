"""Build observation prompt banks and verify anatomy/length constraints."""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "resources/protocol/prompts"
DEFAULT_CATALOG = (
    Path(__file__).resolve().parents[1] / "resources/catalog/field_catalog.json"
)
DEFAULT_DELTA = Path(__file__).resolve().parents[1] / "resources/catalog/ct_fields.json"
TOL_FULL = 0.05
TOL_ITEMS = 0.1
TOL_TERM_DENSITY = 0.2
TOL_MEAN_WORD_LEN = 0.5
EXCLUDED = {
    "neuro_head_face:MR_BRAIN_sequence_specific_signal",
    "neuro_head_face:MR_BRAIN_contrast_change",
    "body:MRTHORAX.T1_T2_relative_signal",
    "body:MRTHORAX.fat_suppression_signal_change",
    "body:MRTHORAX.pre_post_gadolinium_change",
    "body:MRABD.chemical_shift_change",
    "body:MRABD.diffusion_context",
    "body:MRLIVER.pre_post_contrast_change",
    "body:MRLIVER.chemical_shift_change",
    "msk_spine:mr_hip_arthrogram_acetabular_labrum",
    "msk_spine:mr_hip_arthrogram_articular_cartilage",
    "chest_skin_mammo:skin.dermoscopy.shiny_white_streaks",
}
ALLOWED_STATUS = {"source_backed_candidate", "author_adaptation", "gate"}
FORBIDDEN_PLACEHOLDERS = ("{QUESTION}", "{OPTIONS", "{GOLD", "{ANSWER")
O200K_SHA = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/tokenizers.json").read_text(
        encoding="utf-8"
    )
)["o200k_base_sha256"]


def approx_tokens(text):
    """Estimate text length using word lengths and punctuation counts."""
    n = 0
    for tok in re.findall("[A-Za-z0-9]+|[^\\sA-Za-z0-9]", text):
        n += max(1, math.ceil(len(tok) / 4)) if tok[0].isalnum() else 1
    return n


def load_tokenizers(tdir, pylib):
    if pylib:
        sys.path.insert(0, str(pylib))
    out, missing = ({}, [])
    if not tdir:
        return (
            out,
            ["llava-1.5-7b", "qwen2.5-vl-7b", "llava-med-v1.5", "gpt-4o-mini-o200k"],
        )
    tdir = Path(tdir)
    try:
        from tokenizers import Tokenizer

        for key, fn in (
            ("llava-1.5-7b", "llava15_tokenizer.json"),
            ("qwen2.5-vl-7b", "qwen25vl_tokenizer.json"),
        ):
            p = tdir / fn
            if p.is_file():
                t = Tokenizer.from_file(str(p))
                out[key] = lambda s, t=t: len(t.encode(s, add_special_tokens=False).ids)
            else:
                missing.append(key)
    except ImportError:
        missing += ["llava-1.5-7b", "qwen2.5-vl-7b"]
    try:
        import sentencepiece as spm

        p = tdir / "llavamed_tokenizer.model"
        if p.is_file():
            sp = spm.SentencePieceProcessor(model_file=str(p))
            out["llava-med-v1.5"] = lambda s, sp=sp: len(sp.encode(s))
        else:
            missing.append("llava-med-v1.5")
    except ImportError:
        missing.append("llava-med-v1.5")
    try:
        import tiktoken
        import tiktoken_ext.openai_public as op
        from tiktoken import load as tkload

        p = tdir / "o200k_base.tiktoken"
        if p.is_file():
            real = tkload.load_tiktoken_bpe
            op.load_tiktoken_bpe = lambda url, expected_hash=None: real(
                str(p), expected_hash=O200K_SHA
            )
            enc = tiktoken.Encoding(**op.o200k_base())
            out["gpt-4o-mini-o200k"] = lambda s, enc=enc: len(enc.encode(s))
        else:
            missing.append("gpt-4o-mini-o200k")
    except ImportError:
        missing.append("gpt-4o-mini-o200k")
    return (out, missing)


def counts(text, toks):
    c = {
        "approx_bpe_v1": approx_tokens(text),
        "chars": len(text),
        "words": len(text.split()),
    }
    for k, f in toks.items():
        c[k] = f(text)
    return c


def vocab_stats(block, terms):
    words = re.findall("[a-z]+", block.lower())
    words = [w for w in words if not w.isdigit()]
    n_term = sum((w in terms for w in words))
    return {
        "n_words": len(words),
        "n_terms": n_term,
        "term_density": round(n_term / len(words), 4) if words else 0.0,
        "mean_word_len": round(sum(map(len, words)) / len(words), 3) if words else 0.0,
    }


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def build_field_index(catalog, delta):
    idx = {}
    for f in catalog["fields"]:
        idx["catalog:" + f["catalog_field_id"]] = {
            "status": f["status"],
            "source_id": f.get("source_id"),
            "source_anchor": f.get("source_anchor"),
            "family": f["family_modality_region"],
            "unsupported_inferences": f.get("unsupported_inferences", []),
        }
    for f in delta["fields"]:
        idx["ctdelta:" + f["field_id"]] = {
            "status": f["status"],
            "source_id": f.get("source_ids"),
            "source_anchor": f.get("relation_to_catalog_v2"),
            "family": f.get("family_modality_region"),
            "unsupported_inferences": f.get("unsupported_inferences", []),
        }
    return idx


def render(frame, context_line, items):
    block = "\n".join(
        (f"{i}. {it['name']}: {it['ask']}" for i, it in enumerate(items, 1))
    )
    text = frame.replace("{CONTEXT_LINE}", context_line).replace("{ITEMS}", block)
    for ph in FORBIDDEN_PLACEHOLDERS:
        assert ph not in text, ph
    return (text, block)


def render_empty(items):
    return "\n".join((f"{i}. {it['name']} | — | —" for i, it in enumerate(items, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    ap.add_argument("--delta", default=str(DEFAULT_DELTA))
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--pylib", default=None)
    ap.add_argument("--require-real", action="store_true")
    a = ap.parse_args()
    catalog, delta = (read_json(a.catalog), read_json(a.delta))
    routes = read_json(HERE / "specs" / "routes.json")["routes"]
    grd = read_json(HERE / "specs" / "grounded_spec.json")["routes"]
    ana = read_json(HERE / "specs" / "author_anatomy_spec.json")["routes"]
    terms = set(read_json(HERE / "specs" / "anatomy_terms.json")["terms"])
    gen = read_json(HERE / "templates" / "generic_items.json")
    frame = (
        (HERE / "templates" / "observer_frame.txt")
        .read_text(encoding="utf-8")
        .rstrip("\n")
    )
    idx = build_field_index(catalog, delta)
    toks, missing = load_tokenizers(a.tokenizer_dir, a.pylib)
    if a.require_real and missing:
        sys.exit(f"Thiếu tokenizer thật: {missing}")
    errors, trace, lengths = ([], {}, {})
    out = HERE / "generated"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.txt"):
        old.unlink()
    fb_text, _ = render(frame, gen["context_line"], gen["items"])
    for arm in ("GEN", "ANA", "GRD"):
        (out / f"OBS-{arm}__RT-FALLBACK.txt").write_text(
            fb_text + "\n", encoding="utf-8"
        )
    (out / "OBS-EMPTY__RT-FALLBACK.txt").write_text(
        render_empty(gen["items"]) + "\n", encoding="utf-8"
    )
    for rid, meta in routes.items():
        if rid == "RT-FALLBACK":
            continue
        if rid not in grd or rid not in ana:
            errors.append(f"{rid}: thiếu trong grounded hoặc author-anatomy spec")
            continue
        g_items, a_items = (grd[rid], ana[rid])
        if len(g_items) != len(a_items):
            errors.append(f"{rid}: số item GRD {len(g_items)} != ANA {len(a_items)}")
        trace[rid] = []
        for k, it in enumerate(g_items, 1):
            rec = {"item_no": k, "name": it["name"], "fields": []}
            for fid in it["fields"]:
                f = idx.get(fid)
                if f is None:
                    errors.append(f"{rid}#{k}: field không tồn tại {fid}")
                    continue
                if (
                    f["status"] not in ALLOWED_STATUS
                    or fid.split(":", 1)[1] in EXCLUDED
                ):
                    errors.append(
                        f"{rid}#{k}: field không được dùng ({f['status']}) {fid}"
                    )
                rec["fields"].append(
                    {
                        "field_id": fid,
                        "status": f["status"],
                        "source_id": f["source_id"],
                        "source_anchor": f["source_anchor"],
                        "family": f["family"],
                        "unsupported_inferences": f["unsupported_inferences"],
                    }
                )
            st = [x["status"] for x in rec["fields"]]
            rec["item_label"] = (
                "author_adaptation"
                if all((s == "author_adaptation" for s in st))
                else "mixed"
                if "author_adaptation" in st
                else "gate"
                if all((s == "gate" for s in st))
                else "source_backed"
            )
            trace[rid].append(rec)
        ctx = f"Image context: {meta['header']}."
        g_text, g_block = render(frame, ctx, g_items)
        a_text, a_block = render(frame, ctx, a_items)
        gen_text, _ = render(frame, ctx, gen["items"])
        (out / f"OBS-GRD__{rid}.txt").write_text(g_text + "\n", encoding="utf-8")
        (out / f"OBS-ANA__{rid}.txt").write_text(a_text + "\n", encoding="utf-8")
        (out / f"OBS-GEN__{rid}.txt").write_text(gen_text + "\n", encoding="utf-8")
        (out / f"OBS-EMPTY__{rid}.txt").write_text(
            render_empty(g_items) + "\n", encoding="utf-8"
        )
        cg, ca = (counts(g_text, toks), counts(a_text, toks))
        bg, ba = (counts(g_block, toks), counts(a_block, toks))
        tk = [k for k in cg if k not in ("chars", "words")]
        gate_keys = [k for k in tk if k != "approx_bpe_v1"] or ["approx_bpe_v1"]
        rel_full = {k: round((ca[k] - cg[k]) / cg[k], 4) for k in tk}
        rel_items = {k: round((ba[k] - bg[k]) / bg[k], 4) for k in tk}
        vg, va = (vocab_stats(g_block, terms), vocab_stats(a_block, terms))
        rel_dens = (
            round((va["term_density"] - vg["term_density"]) / vg["term_density"], 4)
            if vg["term_density"]
            else None
        )
        d_wlen = round(va["mean_word_len"] - vg["mean_word_len"], 3)
        len_ok = all(
            (
                abs(rel_full[k]) <= TOL_FULL and abs(rel_items[k]) <= TOL_ITEMS
                for k in gate_keys
            )
        )
        voc_ok = (
            rel_dens is not None
            and abs(rel_dens) <= TOL_TERM_DENSITY
            and (abs(d_wlen) <= TOL_MEAN_WORD_LEN)
        )
        lengths[rid] = {
            "n_items": len(g_items),
            "grd_full": cg,
            "ana_full": ca,
            "grd_items_block": bg,
            "ana_items_block": ba,
            "rel_diff_full": rel_full,
            "rel_diff_items": rel_items,
            "gate_tokenizers": gate_keys,
            "length_pass": len_ok,
            "vocab_grd": vg,
            "vocab_ana": va,
            "rel_diff_term_density": rel_dens,
            "diff_mean_word_len": d_wlen,
            "vocab_pass": voc_ok,
            "pass": len_ok and voc_ok,
        }
        if not len_ok:
            errors.append(
                f"{rid}: độ dài ngoài dung sai full={rel_full} items={rel_items}"
            )
        if not voc_ok:
            errors.append(
                f"{rid}: từ vựng ngoài dung sai density={rel_dens} wlen={d_wlen}"
            )
    sha = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.glob("*.txt"))
    }
    (HERE / "field_trace.json").write_text(
        json.dumps(
            {
                "catalog": str(a.catalog),
                "delta": str(a.delta),
                "catalog_version": catalog.get("catalog_version"),
                "delta_version": delta.get("delta_version"),
                "excluded_fields": sorted(EXCLUDED),
                "routes": trace,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    (HERE / "length_check.json").write_text(
        json.dumps(
            {
                "tokenizers_real": sorted(toks),
                "tokenizers_missing": missing,
                ("tokenizer_note"): (
                    "Tiêu chí độ dài áp cho mọi tokenizer thật có mặt"
                    " (gate_tokenizers); approx_bpe_v1 chỉ để tham "
                    "khảo, trừ khi không có tokenizer thật nào."
                ),
                "tolerance": {
                    "full_prompt_rel": TOL_FULL,
                    "items_block_rel": TOL_ITEMS,
                    "term_density_rel": TOL_TERM_DENSITY,
                    "mean_word_len_abs": TOL_MEAN_WORD_LEN,
                },
                "vocab_list": "specs/anatomy_terms.json",
                "all_pass": not errors,
                "routes": lengths,
                "generated_sha256": sha,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    print(
        (
            "OK: "
            f"{len(lengths)}"
            " route, "
            f"{len(sha)}"
            " file prompt, tokenizer thật: "
            f"{sorted(toks)}"
        )
    )


if __name__ == "__main__":
    main()
