"""Paired sensitivity analysis of high-detail external observations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from medvqa import high_detail_answers as H
from medvqa.analysis import io_adapter as io
from medvqa.analysis.bootstrap import summarize
from medvqa.analysis.contrasts import Engine, Spec, required_cells
from medvqa.analysis.metrics import observation_checks
from medvqa.analysis.parsers import parse_lead
from medvqa.analysis.run_all import PROTOCOL_DIR, _clean, build_boots
from medvqa.analysis.scoring import (
    int_missing_qids,
    prepare_items,
    question_table,
    score_rows,
)

CELLS = ["Q-G-GRD-HI", "L-G-GRD-HI"]


def validate_source_provenance(static, current_sources):
    """Accept current source files or an exact registered measured version.

    Args:
        static: Execution metadata containing input and source checksums.
        current_sources: Current source-file checksums indexed by module role.

    Raises:
        ValueError: The source and input combination is not registered.
    """
    if static.get("source_sha256") == current_sources:
        return
    registry_path = (
        Path(__file__).resolve().parents[2] / "configs/inference_provenance.json"
    )
    registry = io.read_json(str(registry_path))
    measured = {
        "inputs_sha256": static.get("inputs_sha256"),
        "source_sha256": static.get("source_sha256"),
    }
    if measured not in registry["high_detail"]:
        raise ValueError("Inference source differs from the measured run.")


def validate_run(inputs, images, run_dir, model):
    """Validate all saved high-detail answers before opening scoring labels.

    Args:
        inputs: Path to the fixed high-detail answer input document.
        images: Directory containing normalized PNG files.
        run_dir: Completed answer run directory.
        model: Q or L.

    Returns:
        tuple[pandas.DataFrame, dict]: Normalized answers and detailed integrity audit.

    Raises:
        ValueError: A configuration, prompt, response or integrity check fails.
        H.U.LockError: The fixed inputs or image checksums differ.
    """
    data, bank, units = H.preflight(inputs, images, model)
    unit_map = {key: (row, order) for key, row, order in units}
    static = io.read_json(str(run_dir / "execution.json"))
    expected_sources = {
        "runner": H.__file__,
        "common": H.U.__file__,
        "models": H.B.__file__,
        "prompts": H.P.__file__,
    }
    if (
        static.get("synthetic_backend") is not False
        or static.get("mode") != "TEST"
        or static.get("cell") != f"CELL-{model}-G-GRD-HI"
        or static.get("inputs_sha256") != H.U.sha256_file(inputs)
        or static.get("observation_artifact_sha256")
        != data["observation_artifact_sha256"]
        or static.get("generate_kwargs") != H.B.GREEDY_KW
        or static.get("answer_cap") != H.ANSWER_CAP
        or static.get("precision") != "PREC-0"
        or static.get("model") != data["baselines"][model]["identity"]
        or static.get("planned_units") != 1824
    ):
        raise ValueError("Evaluation configuration differs from the released run.")
    validate_source_provenance(
        static,
        {key: H.U.sha256_file(path) for key, path in expected_sources.items()},
    )
    template_file = run_dir / "artifacts/answerer_template.txt"
    if H.U.sha256_file(template_file) != data["answer_template_sha256"]:
        raise ValueError("Measured answer template differs.")
    template = H.P.AnswerTemplate(template_file)
    received, raw_hashes = {}, {}
    quality = dict(invalid=0, truncated=0, empty=0, nonfinite=0)
    for shard in sorted(run_dir.glob("shard_*")):
        config = io.read_json(str(shard / "run_config.json"))
        integrity = io.read_json(str(shard / "integrity.json"))
        if any(config.get(k) != v for k, v in static.items()):
            raise ValueError("Session configuration differs from execution manifest.")
        if (
            integrity["output_sha256"] != H.U.sha256_file(shard / "records.jsonl")
            or integrity["run_config_sha256"]
            != H.U.sha256_file(shard / "run_config.json")
            or integrity.get("synthetic_backend") is not False
            or integrity.get("duplicate_keys")
            or integrity.get("errors")
        ):
            raise ValueError("Raw output integrity check failed.")
        identity = config["actual_model"]
        for key in (
            "model",
            "revision",
            "resolved_revision",
            "precision",
            "chat_template_sha256",
            "logits_method",
            "generate_kwargs",
        ):
            if identity.get(key) != static["model"].get(key):
                raise ValueError("Actual backend identity changed.")
        if (
            config["environment"]["packages"]
            != data["baselines"][model]["package_pins"]
        ):
            raise ValueError("Measured package versions changed.")
        for record in io.read_jsonl(str(shard / "records.jsonl")):
            if record.get("key") not in unit_map or record["key"] in received:
                raise ValueError("Unexpected or duplicate completed answer.")
            row, order = unit_map[record["key"]]
            if record.get("mode") != "TEST" or record.get("status") != "ok":
                raise ValueError("An answer unit is not a completed evaluation output.")
            for key in ("qid", "neutral_id", "cohort", "answer_kind", "K"):
                if record.get(key) != row[key]:
                    raise ValueError(
                        "Answer metadata differs from planned public input."
                    )
            if (
                record.get("cell") != static["cell"]
                or record.get("image_condition") != "original"
            ):
                raise ValueError("Condition or image input differs.")
            if str(
                record.get("order") if record.get("order") is not None else "-"
            ) != str(order):
                raise ValueError("Choice order differs.")
            options = (
                H.P.cyclic_order([o["text"] for o in row["options"]], int(order))
                if row["answer_kind"] == "mcq"
                else None
            )
            text = bank[row["neutral_id"]]["text"].strip()
            prompt = template.render(text, row["question"], row["answer_kind"], options)
            if (
                record.get("prompt_sha256") != H.U.sha256_text(prompt)
                or record.get("obs_text_sha256") != H.U.sha256_text(text)
                or record.get("options_presented") != options
                or record.get("max_new_tokens") != H.ANSWER_CAP[row["answer_kind"]]
                or record.get("obs_ref", {}).get("artifact_sha256")
                != data["observation_artifact_sha256"]
            ):
                raise ValueError(
                    "Prompt or high-detail observation provenance differs."
                )
            lp = record.get("logprob")
            if row["answer_kind"] in ("yesno", "mcq"):
                labels = (
                    list(H.B.YESNO_LABELS)
                    if options is None
                    else [o["label"] for o in options]
                )
                if (
                    not lp
                    or not lp.get("finite")
                    or lp.get("labels") != labels
                    or lp.get("method") != "output_logits"
                    or set(lp["lp"]) != set(labels)
                    or set(lp["p_tilde"]) != set(labels)
                    or any(not math.isfinite(v) for v in lp["lp"].values())
                    or any(
                        not math.isfinite(v) or not 0 <= v <= 1
                        for v in lp["p_tilde"].values()
                    )
                    or abs(sum(lp["p_tilde"].values()) - 1) > 1e-8
                ):
                    raise ValueError("Answer-label probabilities are invalid.")
            lead = parse_lead(
                record["text"],
                row["answer_kind"],
                [o["text"] for o in options] if options else [],
            )
            quality["invalid"] += lead is None
            quality["truncated"] += bool(record["truncated"])
            quality["empty"] += not bool(record["text"].strip())
            quality["nonfinite"] += record.get("first_step_finite") is False
            received[record["key"]] = record
        for path in (
            shard / "run_config.json",
            shard / "integrity.json",
            shard / "records.jsonl",
        ):
            raw_hashes[path.relative_to(run_dir).as_posix()] = H.U.sha256_file(path)
    if set(received) != set(unit_map):
        raise ValueError("Evaluation is incomplete; reference labels remain closed.")
    return io.answers_table(pd.DataFrame(received.values())), {
        "model": model,
        "records": len(received),
        "expected_records": len(unit_map),
        "rebuilt_prompt_hashes_checked": len(received),
        "quality": quality,
        "raw_files_sha256": raw_hashes,
        "public_checks_before_reference_labels": True,
    }


def calculate(engine):
    """Calculate twelve paired contrasts for high-detail sensitivity.

    Args:
        engine: Paired engine with original and high-detail conditions for Q and L.

    Returns:
        pandas.DataFrame: Three contrasts for each model and cohort.
    """
    records = []
    for model in ("Q", "L"):
        hi = model + "-G-GRD-HI"
        comparisons = {
            "C2-high": (model + "-NONE", "G-GRD-high minus NONE"),
            "C3-high": (model + "-SELF-GRD", "G-GRD-high minus SELF-GRD"),
            "high-minus-low": (model + "-G-GRD", "G-GRD-high minus G-GRD-low"),
        }
        for group in ("RAD", "NAT"):
            for name, (reference, description) in comparisons.items():
                point, replicates, count = engine.evaluate(
                    Spec(name, "acc", group, [(1, hi), (-1, reference)])
                )
                records.append(
                    {
                        "contrast": name,
                        "description": description,
                        "model": model,
                        "cohort": group,
                        "n_questions": count,
                        **summarize(point, replicates, scale=100),
                    }
                )
    return pd.DataFrame(records)


def analyze(base_runs, q_run, l_run, inputs, observations, output):
    """Validate measured runs and export high-detail sensitivity tables.

    Args:
        base_runs: Original run collection with prepared data and separate scoring
            labels.
        q_run: Completed high-detail Qwen answer run.
        l_run: Completed high-detail LLaVA answer run.
        inputs: Path to the fixed high-detail answer input JSON document.
        observations: JSONL file containing the high-detail observation records.
        output: New directory for aggregate CSV tables and analysis metadata.

    Returns:
        dict: Input audits, analysis settings, warnings and output checksums.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: A saved run or its sample membership fails validation.
    """
    if output.exists():
        raise FileExistsError("Choose a new analysis output directory.")
    frames, audits = [], []
    for model, run in (("Q", q_run), ("L", l_run)):
        frame, audit = validate_run(
            inputs, base_runs / "prep/prep_out/images", run, model
        )
        frames.append(frame)
        audits.append(audit)
    data, warnings = io.load_all(str(base_runs), required_cells)
    items, extra = prepare_items(data["items"], io.load_routes_table(PROTOCOL_DIR))
    base_scored, integrity = score_rows(
        items,
        data["answers"],
        sorted(set(data["answers"]["cell"])),
        required=required_cells(data["design_state"]["design"]),
    )
    high_scored, high_integrity = score_rows(
        items[items["kind"].isin(["yesno", "closed_short", "mcq"])],
        pd.concat(frames, ignore_index=True),
        CELLS,
        required=CELLS,
    )
    for check in (integrity, high_integrity):
        if check["missing_rows"] or check["unexpected_rows"]:
            raise ValueError("Scoring membership differs from the required endpoint.")
    scored = pd.concat([base_scored, high_scored], ignore_index=True)
    qids, bootstraps = build_boots(items)
    engine = Engine(
        items,
        scored,
        question_table(scored),
        int_missing_qids(scored),
        bootstraps,
        qids,
    )
    contrasts = calculate(engine)
    endpoint_cells = [
        f"{m}-{condition}"
        for m in ("Q", "L")
        for condition in ("NONE", "SELF-GRD", "G-GRD", "G-GRD-HI")
    ]
    endpoint_rows = []
    for cell in endpoint_cells:
        for group, endpoint in (("RAD", "E-P1"), ("NAT", "E-P2")):
            point, replicates, count = engine.evaluate(
                Spec(endpoint, "acc", group, [(1, cell)])
            )
            endpoint_rows.append(
                {
                    "endpoint": endpoint,
                    "cell": cell,
                    "cohort": group,
                    "n_questions": count,
                    **summarize(point, replicates, scale=100),
                }
            )
    endpoints = pd.DataFrame(endpoint_rows).drop(columns=["p"])
    obs_rows = io.read_jsonl(str(observations))
    obs_frame = io.observations_table(pd.DataFrame(obs_rows))

    # Expected item counts are reconstructed from the same frozen observer prompts.
    def expected_items(arm, route):
        routes = str(route or "RT-FALLBACK").split("+")
        return sum(len(io.route_items(PROTOCOL_DIR, arm, r)) for r in routes)

    high_checks = observation_checks(obs_frame, expected_items).assign(detail="high")
    low_obs = data["observations"]
    low_checks = observation_checks(
        low_obs[(low_obs["observer"] == "G") & (low_obs["arm"] == "GRD")],
        expected_items,
    ).assign(detail="low")
    checks = pd.concat([low_checks, high_checks], ignore_index=True)
    output.mkdir(parents=True)
    for name, table in {
        "contrasts": contrasts,
        "endpoints": endpoints,
        "observation_checks": checks,
    }.items():
        table.to_csv(output / f"{name}.csv", index=False)
    report = {
        "study_type": "exploratory_sensitivity",
        "bootstrap_replicates": 10000,
        "seed": 20261016,
        "RAD_role": "primary",
        "NAT_role": "descriptive",
        "multiplicity": (
            "Unadjusted exploratory estimates; confirmatory families unchanged; "
            "no new Holm family introduced."
        ),
        "audits": audits,
        "warnings": warnings + extra,
        "scoring_integrity": high_integrity,
        "input_sha256": H.U.sha256_file(inputs),
        "observation_records_sha256": H.U.sha256_file(observations),
        "code_sha256": H.U.sha256_file(Path(__file__)),
        "base_record_files_sha256": {
            p.relative_to(base_runs).as_posix(): H.U.sha256_file(p)
            for p in sorted(base_runs.glob("*/shard_*/records.jsonl"))
        },
        "output_sha256": {
            p.name: H.U.sha256_file(p) for p in sorted(output.glob("*.csv"))
        },
    }
    with (output / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(_clean(report), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-runs", "q-run", "l-run", "inputs", "observations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    analyze(
        args.base_runs,
        args.q_run,
        args.l_run,
        args.inputs,
        args.observations,
        args.output,
    )


if __name__ == "__main__":
    main()
