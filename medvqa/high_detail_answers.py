"""Run closed-question answering with high-detail external observations."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import time
from pathlib import Path

try:
    from . import common as U
    from . import models as B
    from . import prompts as P
except ImportError:
    import common as U
    import models as B
    import prompts as P

ANSWER_CAP = {"yesno": 8, "mcq": 8, "closed_short": 24, "open": 24}


def read_inputs(path):
    """Validate the fixed questions, observations and generation settings.

    Args:
        path: Path to high_detail_answer_inputs.json.

    Returns:
        tuple[dict, dict]: Validated input document and observations by image ID.

    Raises:
        U.LockError: Membership, observation hashes, template or settings differ.
    """
    data = U.read_json(path)
    if data.get("schema") != "HIGH-DETAIL-ANSWER-INPUTS-v1":
        raise U.LockError("Unknown high-detail answer input schema.")
    rows = data["questions"]
    bank = {r["neutral_id"]: r for r in data["observations"]}
    if (
        len(rows) != 988
        or len({r["qid"] for r in rows}) != 988
        or len({r["neutral_id"] for r in rows}) != 542
        or len(bank) != 599
        or len(data["observations"]) != 599
        or any(r["answer_kind"] not in ("yesno", "closed_short", "mcq") for r in rows)
    ):
        raise U.LockError(
            "The frozen closed endpoint or observation membership differs."
        )
    for row in rows:
        obs = bank[row["neutral_id"]]
        if (
            obs["status"] != "ok"
            or not isinstance(obs["text"], str)
            or not obs["text"].strip()
        ):
            raise U.LockError("Required high-detail observation is unavailable.")
        if U.sha256_text(obs["text"]) != obs["text_sha256"]:
            raise U.LockError("High-detail observation text checksum mismatch.")
    if U.sha256_text(data["answer_template"]) != data["answer_template_sha256"]:
        raise U.LockError("Answer template checksum mismatch.")
    if data["generate_kwargs"] != B.GREEDY_KW or data["answer_cap"] != ANSWER_CAP:
        raise U.LockError("Generation settings changed.")
    return data, bank


def units_for(data, answerer):
    """Expand questions into the fixed answer and option-order units.

    Args:
        data: Validated high-detail answer input document.
        answerer: Q for Qwen or L for LLaVA.

    Returns:
        list[tuple]: Execution key, question record and option order for each turn.

    Raises:
        U.LockError: The 1,824 unique answer units cannot be reconstructed.
    """
    cell = f"CELL-{answerer}-G-GRD-HI"
    units = []
    for row in data["questions"]:
        for order in row["orders"] if row["answer_kind"] == "mcq" else ["-"]:
            units.append((f"ANS|{cell}|{row['qid']}|{order}", row, order))
    if len(units) != len({key for key, _, _ in units}) or len(units) != 1824:
        raise U.LockError("Answer turn membership differs from the frozen endpoint.")
    return units


def preflight(path, images, answerer):
    """Verify the closed-question sample and every required image checksum.

    Args:
        path: Path to the fixed answer input JSON document.
        images: Directory containing the normalized neutral-ID PNG files.
        answerer: Q or L, selecting the measured answering model.

    Returns:
        tuple: Validated document, observation bank and planned answer units.

    Raises:
        ValueError: The selected model code is unsupported.
        U.LockError: Input membership, content or image checksums differ.
    """
    if answerer not in ("Q", "L"):
        raise ValueError("Select Q or L.")
    data, bank = read_inputs(path)
    units = units_for(data, answerer)
    for image_id in {r["neutral_id"] for _, r, _ in units}:
        if (
            U.sha256_file(Path(images) / f"{image_id}.png")
            != bank[image_id]["image_sha256"]
        ):
            raise U.LockError("An endpoint image differs from the frozen input.")
    return data, bank, units


def verify_environment(data, answerer):
    """Require the measured package versions and GPU configuration.

    Args:
        data: Input document containing the baseline package pins.
        answerer: Q requires one L4; L requires two T4 GPUs.

    Returns:
        dict: Verified package versions and GPU names.

    Raises:
        U.LockError: A package version or GPU configuration differs.
    """
    pins = data["baselines"][answerer]["package_pins"]
    for package, expected in pins.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            raise U.LockError(f"Environment mismatch: {package} needs {expected}.")
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if answerer == "Q" and (len(names) != 1 or "L4" not in names[0]):
        raise U.LockError("Q requires the measured Colab L4 environment.")
    if answerer == "L" and (len(names) != 2 or any("T4" not in name for name in names)):
        raise U.LockError("L requires the measured Kaggle T4x2 environment.")
    return {"packages": pins, "gpu": names}


def validate_identity(data, answerer, model):
    """Check the loaded model against the recorded baseline identity.

    Args:
        data: Input document with baseline identities.
        answerer: Q or L.
        model: Loaded model object exposing its identity mapping.

    Raises:
        U.LockError: Checkpoint, precision, template or decoding settings differ.
    """
    expected = data["baselines"][answerer]["identity"]
    for field in (
        "model",
        "revision",
        "resolved_revision",
        "precision",
        "chat_template_sha256",
        "logits_method",
        "generate_kwargs",
    ):
        if model.identity.get(field) != expected.get(field):
            raise U.LockError(f"Backend differs from the measured baseline: {field}.")


def run(
    inputs,
    images,
    output_root,
    answerer,
    *,
    backend_factory=None,
    environment_checker=verify_environment,
    scorer=None,
    max_items=None,
    progress=print,
):
    """Run or resume answers using the fixed high-detail observations.

    Args:
        inputs: Path to the fixed answer input JSON document.
        images: Directory containing normalized PNG files.
        output_root: Parent directory of the deterministic per-model run folder.
        answerer: Q or L.
        backend_factory: Optional test backend factory; outputs are marked synthetic.
        environment_checker: Callable that verifies the measured environment before
            inference.
        scorer: Optional label-probability callable used in offline tests.
        max_items: Optional execution limit for tests; None processes all remaining
            units.
        progress: Callable receiving progress messages.

    Returns:
        dict: Run directory and count of remaining answer units.

    Raises:
        U.LockError: Inputs, model identity or resume configuration differ.
        U.IntegrityError: Completed answer keys are duplicated or unexpected.
    """
    inputs, images, output_root = map(Path, (inputs, images, output_root))
    data, bank, units = preflight(inputs, images, answerer)
    environment = environment_checker(data, answerer)
    run_id = f"high_detail_answers_{answerer}_20261026"
    run_root = output_root / run_id
    static = {
        "schema": "HIGH-DETAIL-ANSWERS-v1",
        "mode": "TEST",
        "answerer": f"ANS-{answerer}",
        "cell": f"CELL-{answerer}-G-GRD-HI",
        "observation_detail": "high",
        "observation_artifact_sha256": data["observation_artifact_sha256"],
        "inputs_sha256": U.sha256_file(inputs),
        "model": data["baselines"][answerer]["identity"],
        "precision": "PREC-0",
        "generate_kwargs": B.GREEDY_KW,
        "answer_cap": ANSWER_CAP,
        "planned_units": len(units),
        "synthetic_backend": backend_factory is not None,
        "source_sha256": {
            name: U.sha256_file(path)
            for name, path in {
                "runner": __file__,
                "models": B.__file__,
                "prompts": P.__file__,
                "common": U.__file__,
            }.items()
        },
    }
    run_root.mkdir(parents=True, exist_ok=True)
    frozen = run_root / "execution.json"
    if frozen.exists() and U.read_json(frozen) != static:
        raise U.LockError("Resume configuration differs from this experiment.")
    U.write_json_once(frozen, static)
    template_file = run_root / "artifacts/answerer_template.txt"
    template_file.parent.mkdir(parents=True, exist_ok=True)
    if template_file.exists():
        if U.sha256_file(template_file) != data["answer_template_sha256"]:
            raise U.LockError("Stored answer template changed.")
    else:
        with template_file.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data["answer_template"])
    template = P.AnswerTemplate(template_file)
    keys = {key for key, _, _ in units}
    done, duplicates = U.collect_done_keys([run_root])
    if duplicates or not done.issubset(keys):
        raise U.IntegrityError("Duplicate or unexpected completed answer units.")
    if done == keys:
        return {
            "status": "ALREADY_COMPLETE",
            "completed": len(done),
            "run_dir": str(run_root),
        }
    model = (backend_factory or B.HFVisionBackend)(answerer, "PREC-0")
    validate_identity(data, answerer, model)
    label_map = B.build_label_map(model.decode_one, model.vocab_size())
    if label_map != data["baselines"][answerer]["label_map"]:
        raise U.LockError("Answer label token map differs from the measured baseline.")
    scorer = scorer or B.label_logprobs
    session = U.ShardSession(
        "high_detail_answers",
        run_id,
        "TEST",
        output_root,
        0,
        1,
        time_limit_s=3 * 3600,
        max_items=max_items,
    )
    config = dict(static, environment=environment, actual_model=model.identity)
    U.write_json(session.out_dir / "run_config.json", config)
    from PIL import Image

    image_cache = {}
    for key, row, order in units:
        if key in done:
            continue
        if session.should_stop():
            break
        image_id, kind = row["neutral_id"], row["answer_kind"]
        options = (
            P.cyclic_order([o["text"] for o in row["options"]], int(order))
            if kind == "mcq"
            else None
        )
        obs_text = bank[image_id]["text"].strip()
        prompt = template.render(obs_text, row["question"], kind, options)
        P.check6_answer_prompt(template, prompt, kind, len(options or []))
        P.check3_no_original_names(
            prompt.replace(obs_text, "<OBS>").replace(row["question"], "<Q>")
        )
        if image_id not in image_cache:
            with Image.open(images / f"{image_id}.png") as original:
                image_cache = {image_id: original.convert("RGB")}
        started = time.time()
        generated = model.generate(image_cache[image_id], prompt, ANSWER_CAP[kind])
        lp = None
        if kind in ("yesno", "mcq"):
            labels = (
                list(B.YESNO_LABELS)
                if kind == "yesno"
                else [o["label"] for o in options]
            )
            lp = scorer(generated["first_logits"], label_map, labels)
            lp["method"] = generated["logits_method"]
        record = session.base("ANS-v1", key)
        record.update(
            {
                "cell": static["cell"],
                "answerer": f"ANS-{answerer}",
                "obs_observer": "OBS-G",
                "obs_arm": "ARM-GRD",
                "intervention": None,
                "qid": row["qid"],
                "neutral_id": image_id,
                "cohort": row["cohort"],
                "answer_kind": kind,
                "order": order if kind == "mcq" else None,
                "K": row["K"],
                "options_presented": options,
                "image_condition": "original",
                "max_new_tokens": ANSWER_CAP[kind],
                "status": "ok",
                "text": generated["text"],
                "finish_reason": generated["finish_reason"],
                "truncated": generated["truncated"],
                "n_output_tokens": generated["n_output_tokens"],
                "first_step_finite": generated["first_step_finite"],
                "logprob": lp,
                "quality": U.quality_flags(generated["text"]),
                "timing_s": round(time.time() - started, 3),
                "prompt_sha256": U.sha256_text(prompt),
                "obs_text_sha256": U.sha256_text(obs_text),
                "obs_ref": {
                    "key": f"OBS|OBS-G|ARM-GRD|{image_id}",
                    "obs_status": "ok",
                    "artifact_sha256": data["observation_artifact_sha256"],
                },
            }
        )
        session.write(record)
        if session.done_this % 50 == 0:
            progress(
                f"{answerer}: {len(done) + session.done_this}/{len(units)} answers"
            )
    report = session.integrity(
        sorted(keys),
        {
            "run_config_sha256": U.sha256_file(session.out_dir / "run_config.json"),
            "observation_artifact_sha256": data["observation_artifact_sha256"],
            "synthetic_backend": backend_factory is not None,
        },
    )
    return dict(report, run_dir=str(run_root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--answerer", choices=("Q", "L"), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.inputs, args.images, args.output_root, args.answerer), indent=2
        )
    )


if __name__ == "__main__":
    main()
