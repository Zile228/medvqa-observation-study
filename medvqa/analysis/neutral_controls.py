"""Exploratory paired comparisons for fixed neutral-list observation controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from medvqa import neutral_controls as controls
from medvqa.analysis import io_adapter as io
from medvqa.analysis.bootstrap import holm, summarize, tost
from medvqa.analysis.contrasts import Engine, Spec, required_cells
from medvqa.analysis.parsers import norm
from medvqa.analysis.run_all import PROTOCOL_DIR, _clean, build_boots
from medvqa.analysis.scoring import (
    int_missing_qids,
    prepare_items,
    question_table,
    score_rows,
)
from medvqa.conditions import ANSWER_CAP
from medvqa.models import GREEDY_KW, MODELS

SOURCE_ROOT = Path(__file__).resolve().parents[2]
CONTROL_CELLS = tuple("Q-" + arm for arm in controls.ARMS)
CONTRASTS = {
    "T1": [(1, "Q-INT-EMPTY"), (-1, "Q-NONE")],
    "T2": [(1, "Q-INT-EMPTY"), (-1, "Q-ITEMS")],
    "T3": [(1, "Q-INT-EMPTY"), (-1, "Q-NEUTRAL-DASH")],
    "T4": [(1, "Q-NEUTRAL-DASH"), (-1, "Q-NONE")],
    "T5": [(1, "Q-SELF-GRD"), (-1, "Q-INT-EMPTY")],
    "T6": [(1, "Q-ITEMS"), (-1, "Q-NEUTRAL-LIST")],
    "T7": [
        (1, "Q-INT-EMPTY"),
        (-1, "Q-ITEMS"),
        (-1, "Q-NEUTRAL-DASH"),
        (1, "Q-NEUTRAL-LIST"),
    ],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_state(result: dict) -> str:
    """Assign one of the four prespecified difference and equivalence states.

    Args:
        result: Contrast summary containing 95% and 90% confidence bounds.

    Returns:
        str: Difference, equivalence, small-difference or inconclusive state.
    """
    different = result["ci_low"] > 0 or result["ci_high"] < 0
    equivalent = result["tost_ci_low"] > -3 and result["tost_ci_high"] < 3
    if different and equivalent:
        return "different_but_within_margin"
    if different:
        return "different"
    if equivalent:
        return "equivalent_within_margin"
    return "inconclusive"


def summary(point, replicates, *, equivalence=False, scale=100):
    result = summarize(point, replicates, scale=scale)
    if equivalence:
        if result["B_eff"]:
            result.update(tost(replicates, m=1, margin=3))
        else:
            result.update(tost_ci_low=np.nan, tost_ci_high=np.nan, equivalent=False)
        result["state"] = interval_state(result)
    return result


def validate_control_run(base: Path, run: Path) -> tuple[pd.DataFrame, dict]:
    """Validate every control response before opening scoring labels.

    Args:
        base: Original run collection used for the public question definitions.
        run: Directory containing the three-condition control answers.

    Returns:
        tuple[pandas.DataFrame, dict]: Normalized answer rows and integrity audit.

    Raises:
        ValueError: Membership, prompts, model identity or integrity checks fail.
    """
    public_rows = []
    for cohort in ("COH-RAD-TEST", "COH-NAT-TEST"):
        public_rows += io.read_jsonl(
            str(base / "prep/prep_out/cohorts" / f"{cohort}.jsonl")
        )
    closed = [r for r in public_rows if r["answer_kind"] != "open"]
    bank = controls.load_bank(SOURCE_ROOT)
    units = controls.plan(closed, "TEST", bank)
    expected_units = {key: (cell, row, order) for key, cell, row, order in units}
    expected = set(expected_units)
    received = {}
    inputs = {}
    for shard in sorted(run.glob("shard_*")):
        records = shard / "records.jsonl"
        config = shard / "run_config.json"
        integrity = shard / "integrity.json"
        if not all(p.is_file() for p in (records, config, integrity)):
            raise ValueError("Every session needs records, config, and integrity.")
        check = io.read_json(str(integrity))
        cfg = io.read_json(str(config))
        if cfg.get("mode") != "TEST":
            raise ValueError("Control analysis accepts evaluation runs only.")
        identity = cfg.get("model", {})
        if (
            identity.get("model") != MODELS["Q"]["id"]
            or identity.get("revision") != MODELS["Q"]["revision"]
            or cfg.get("precision") != "PREC-0"
            or cfg.get("generate_kwargs") != GREEDY_KW
            or cfg.get("answer_cap") != ANSWER_CAP
            or set(cfg.get("cells", [])) != {"CELL-" + cell for cell in CONTROL_CELLS}
        ):
            raise ValueError("Control model, generation settings, or cells changed.")
        if check.get("output_sha256") != sha(records):
            raise ValueError("Control record checksum mismatch.")
        if check.get("run_config_sha256") != sha(config):
            raise ValueError("Control configuration checksum mismatch.")
        for path in (records, config, integrity):
            inputs[path.relative_to(run).as_posix()] = sha(path)
        for row in io.read_jsonl(str(records)):
            if row.get("key") not in expected or row.get("mode") != "TEST":
                raise ValueError("Unexpected control unit or non-evaluation record.")
            if row.get("status") == "error":
                continue
            if row["key"] in received:
                raise ValueError("Duplicate completed control unit.")
            expected_cell, question, order = expected_units[row["key"]]
            if (
                row.get("cell") != expected_cell["cell"]
                or row.get("qid") != question["qid"]
                or row.get("neutral_id") != question["neutral_id"]
                or row.get("answer_kind") != question["answer_kind"]
                or str(row.get("order") if row.get("order") is not None else "-")
                != str(order)
                or row.get("image_condition") != "original"
            ):
                raise ValueError("Answer metadata differs from its planned unit.")
            arm = io.canonical_cell(row["cell"])[2:]
            text = bank[arm]["TEST", row["neutral_id"]]
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            if row.get("obs_text_sha256") != text_hash:
                raise ValueError("Control observation text does not match the bank.")
            received[row["key"]] = row
    if set(received) != expected:
        raise ValueError("Control run is incomplete; reference labels remain closed.")
    frame = pd.DataFrame(received.values())
    return io.answers_table(frame), {
        "raw_files_sha256": inputs,
        "completed_units": len(received),
        "expected_units": len(expected),
        "public_scope_checked_before_reference_labels": True,
    }


def calculate(engine: Engine, rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate the seven contrasts and answer-direction diagnostics.

    Args:
        engine: Paired analysis engine with fixed image-cluster bootstrap samples.
        rows: Scored baseline and control-condition answer rows.

    Returns:
        dict[str, pandas.DataFrame]: Contrasts, answer changes and answer direction.
    """
    comparisons, changes, directions = [], [], []
    cells = ("Q-NONE", "Q-INT-EMPTY", "Q-SELF-GRD", *CONTROL_CELLS)
    for cohort in ("RAD", "NAT"):
        for name, terms in CONTRASTS.items():
            point, reps, count = engine.evaluate(Spec(name, "acc", cohort, terms))
            comparisons.append(
                {
                    "contrast": name,
                    "cohort": cohort,
                    "n_questions": count,
                    "terms": json.dumps(terms),
                    **summary(point, reps, equivalence=True),
                }
            )
        for cell in cells[1:]:
            for metric in ("flip", "tv"):
                point, reps, count = engine.evaluate(
                    Spec(metric, metric, cohort, [(1, cell, "Q-NONE")])
                )
                changes.append(
                    {
                        "cell": cell,
                        "reference": "Q-NONE",
                        "metric": metric,
                        "cohort": cohort,
                        "n_questions": count,
                        **summary(point, reps),
                    }
                )
    table = pd.DataFrame(comparisons)
    rad = table["cohort"].eq("RAD")
    adjusted, rejected = holm(table.loc[rad, "p"].fillna(1).to_numpy())
    table.loc[rad, "p_holm_sensitivity"] = adjusted
    table.loc[rad, "holm_reject_sensitivity"] = rejected
    qids = engine.qids["RAD"]
    yesno = engine.kind["RAD"] == "yesno"
    gold = engine.items.loc[qids, "gold"].map(norm).to_numpy()
    for cell in cells:
        selected = rows[rows["cell"].eq(cell) & rows["cohort"].eq("RAD")].set_index(
            "qid"
        )
        lead = selected["lead"].reindex(qids)
        valid = selected["valid_lead"].reindex(qids).fillna(False)
        probability = (
            selected["ptilde"]
            .map(
                lambda value: (
                    value.get("No", np.nan) if isinstance(value, dict) else np.nan
                )
            )
            .reindex(qids)
        )
        measurements = {
            "No_rate": (lead.eq("No").to_numpy(float), yesno, 100),
            "mean_P_No": (probability.to_numpy(float), yesno, 1),
            "invalid_rate": ((~valid.astype(bool)).to_numpy(float), yesno, 100),
            "accuracy_gold_Yes": (
                engine.vec("correct_lead", cell, "RAD"),
                yesno & (gold == "yes"),
                100,
            ),
            "accuracy_gold_No": (
                engine.vec("correct_lead", cell, "RAD"),
                yesno & (gold == "no"),
                100,
            ),
        }
        for metric, (vector, mask, scale) in measurements.items():
            mask = mask & engine.missing_mask("RAD", [cell]) & np.isfinite(vector)
            point, reps = engine.boots["RAD"].endpoint(vector, mask, "mean")
            directions.append(
                {
                    "cell": cell,
                    "cohort": "RAD",
                    "metric": metric,
                    "n_questions": int(mask.sum()),
                    "scale": scale,
                    **summary(point, reps, scale=scale),
                }
            )
    # Paired changes in No output are separate descriptive analyses, not Holm terms.
    for left, right in [
        ("Q-INT-EMPTY", "Q-ITEMS"),
        ("Q-NEUTRAL-DASH", "Q-ITEMS"),
        ("Q-NEUTRAL-DASH", "Q-NONE"),
    ]:

        def no_vector(cell):
            subset = rows[rows["cell"].eq(cell) & rows["cohort"].eq("RAD")].set_index(
                "qid"
            )
            return subset["lead"].reindex(qids).eq("No").to_numpy(float)

        mask = yesno & engine.missing_mask("RAD", [left, right])
        point, reps = engine.boots["RAD"].endpoint(
            no_vector(left) - no_vector(right), mask, "mean"
        )
        directions.append(
            {
                "cell": left,
                "reference": right,
                "cohort": "RAD",
                "metric": "paired_No_difference",
                "n_questions": int(mask.sum()),
                "scale": 100,
                **summary(point, reps),
            }
        )
    return {
        "contrasts": table,
        "answer_changes": pd.DataFrame(changes),
        "answer_direction": pd.DataFrame(directions),
    }


def analyze(base_runs: Path, control_run: Path, output: Path) -> dict:
    """Validate control outputs and export their supplementary paired analyses.

    Args:
        base_runs: Original runs including prepared questions and separate scoring
            labels.
        control_run: Complete three-condition Qwen answer run.
        output: New directory for aggregate CSV tables and analysis metadata.

    Returns:
        dict: Audit results, analysis settings and output checksums.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: The runs fail validation or answer membership is incomplete.
    """
    if output.exists():
        raise FileExistsError("Choose a new output directory.")
    control_answers, audit = validate_control_run(base_runs, control_run)
    data, warnings = io.load_all(str(base_runs), required_cells)
    items, extra_warnings = prepare_items(
        data["items"], io.load_routes_table(PROTOCOL_DIR)
    )
    base_cells = sorted(set(data["answers"]["cell"]))
    scored, base_integrity = score_rows(
        items,
        data["answers"],
        base_cells,
        required=required_cells(data["design_state"]["design"]),
    )
    closed_items = items[items["kind"].isin(["yesno", "closed_short", "mcq"])]
    new_scored, control_integrity = score_rows(
        closed_items,
        control_answers,
        list(CONTROL_CELLS),
        required=list(CONTROL_CELLS),
    )
    for integrity in (base_integrity, control_integrity):
        if integrity["missing_rows"] or integrity["unexpected_rows"]:
            raise ValueError("Incomplete or unexpected answer membership.")
    rows = pd.concat([scored, new_scored], ignore_index=True)
    qids, bootstraps = build_boots(items)
    engine = Engine(
        items, rows, question_table(rows), int_missing_qids(rows), bootstraps, qids
    )
    tables = calculate(engine, rows)
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    report = {
        "study_type": "exploratory",
        "bootstrap_replicates": 10000,
        "seed": 20261016,
        "equivalence_margin_percentage_points": 3,
        "Holm_family": "Seven RAD contrasts; sensitivity only",
        "NAT_role": "descriptive",
        "audit": audit,
        "warnings": warnings + extra_warnings,
        "control_integrity": control_integrity,
        "base_record_files_sha256": {
            path.relative_to(base_runs).as_posix(): sha(path)
            for path in sorted(base_runs.glob("*/shard_*/records.jsonl"))
        },
        "control_text_files_sha256": {
            path.name: sha(path)
            for path in sorted(
                (SOURCE_ROOT / "resources/neutral_controls").glob("*.jsonl")
            )
        },
        "code_sha256": sha(Path(__file__)),
        "output_sha256": {p.name: sha(p) for p in output.glob("*.csv")},
    }
    (output / "analysis.json").write_text(
        json.dumps(_clean(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runs", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.base_runs, args.control_run, args.output)


if __name__ == "__main__":
    main()
