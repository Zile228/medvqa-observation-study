"""Supplementary paired comparisons and answer-direction analyses.

These analyses are exploratory, specified after inspection of the main study.
They do not replace the confirmatory contrast families or their multiplicity rules.
Only aggregate statistics and input-file digests are exported.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from medvqa.analysis import io_adapter as IO
from medvqa.analysis.bootstrap import summarize, tost
from medvqa.analysis.constants import SEED, B
from medvqa.analysis.contrasts import Engine, Spec, required_cells
from medvqa.analysis.run_all import PROJECT_ROOT, PROTOCOL_DIR, build_boots
from medvqa.analysis.scoring import (
    int_missing_qids,
    prepare_items,
    question_table,
    score_rows,
)


def analyze(runs_dir, out_dir, protocol_dir=PROTOCOL_DIR):
    """Calculate supplementary paired comparisons from the original saved runs.

    Args:
        runs_dir: Original run collection with prepared data and scoring labels.
        out_dir: Destination for supplementary aggregate tables.
        protocol_dir: Directory of fixed prompt and routing specifications.

    Returns:
        dict: Comparison table, arithmetic cross-checks and analysis metadata.
    """
    runs_dir, out_dir = (Path(runs_dir), Path(out_dir))
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    data, warns = IO.load_all(str(runs_dir), required_cells)
    items, w2 = prepare_items(data["items"], IO.load_routes_table(str(protocol_dir)))
    cells = sorted(set(data["answers"]["cell"]))
    rows, integ = score_rows(
        items,
        data["answers"],
        cells,
        required=required_cells(data["design_state"]["design"]),
    )
    assert integ["missing_rows"] == 0 and integ["unexpected_rows"] == 0
    qt = question_table(rows)
    miss = int_missing_qids(rows)
    qids, boots = build_boots(items)
    e = Engine(items, rows, qt, miss, boots, qids)
    out = []
    equations = []

    def add(task, label, coh, terms, mask=None, do_tost=False, code="new"):
        sp = Spec(label, "acc", coh, terms)
        p, r, n = e.evaluate(sp, extra_mask=mask)
        row = dict(
            task=task,
            contrast=label,
            code_contrast=code,
            cohort=coh,
            n_questions=n,
            **summarize(p, r),
            source="Recorded answers and interventions; image-cluster bootstrap",
        )
        if do_tost:
            row.update(tost(r, m=1))
        out.append(row)
        return (p, r, n)

    for coh in ["RAD", "NAT"]:
        a = add(
            "paired_empty_control",
            "EMPTY − NONE",
            coh,
            [(1, "Q-INT-EMPTY"), (-1, "Q-NONE")],
        )
        b = add(
            "paired_empty_control",
            "SELF-GRD − EMPTY",
            coh,
            [(1, "Q-SELF-GRD"), (-1, "Q-INT-EMPTY")],
        )
        c = add(
            "paired_empty_control",
            "SELF-GRD − NONE",
            coh,
            [(1, "Q-SELF-GRD"), (-1, "Q-NONE")],
            code="C1-Q",
        )
        residual = a[0] + b[0] - c[0]
        rep_error = float(np.max(np.abs(a[1] + b[1] - c[1])))
        assert abs(residual) < 1e-12 and rep_error < 1e-12
        equations.append(
            {
                "cohort": coh,
                "identity": "(EMPTY−NONE)+(SELF-GRD−EMPTY)=C1-Q",
                "residual_pp": residual * 100,
                "max_replicate_residual_pp": rep_error * 100,
                "n": c[2],
            }
        )
        for cid, o in [("C1", "SELF"), ("C2", "G")]:
            add(
                "model_interaction",
                f"{cid}-Q − {cid}-L model interaction",
                coh,
                [(1, f"Q-{o}-GRD"), (-1, "Q-NONE"), (-1, f"L-{o}-GRD"), (1, "L-NONE")],
                code=f"{cid}-Q − {cid}-L",
            )
        for label, ca, cb, code in [
            ("GRD − ANA", "GRD", "ANA", "C7"),
            ("GRD − GEN", "GRD", "GEN", "C8"),
            ("ANA − GEN", "ANA", "GEN", "C9"),
        ]:
            for k, o in [("L", "SELF"), ("Q", "SELF"), ("L", "G"), ("Q", "G")]:
                add(
                    "observation_structure",
                    f"{label} [{k}-{o}]",
                    coh,
                    [(1, f"{k}-{o}-{ca}"), (-1, f"{k}-{o}-{cb}")],
                    do_tost=True,
                    code=code,
                )
            add(
                "observation_structure",
                f"{label} mean L-G,Q-G",
                coh,
                [
                    (0.5, f"L-G-{ca}"),
                    (-0.5, f"L-G-{cb}"),
                    (0.5, f"Q-G-{ca}"),
                    (-0.5, f"Q-G-{cb}"),
                ],
                do_tost=True,
                code=code,
            )
        add(
            "image_removal",
            "BLANK − SELF-GRD",
            coh,
            [(1, "Q-INT-BLANK"), (-1, "Q-SELF-GRD")],
            do_tost=True,
            code="C16",
        )
        if coh == "NAT":
            for g in sorted(items.loc[qids[coh], "group"].unique()):
                add(
                    "image_removal",
                    f"BLANK − SELF-GRD [{g}]",
                    coh,
                    [(1, "Q-INT-BLANK"), (-1, "Q-SELF-GRD")],
                    mask=e.qmask(coh, lambda r, g=g: r["group"] == g),
                    do_tost=True,
                    code="C16",
                )
        for available in [True, False]:
            mask = np.array(
                [
                    (q not in miss.get("Q-INT-CORR", set())) == available
                    for q in qids[coh]
                ]
            )
            for c in ["Q-NONE", "Q-SELF-GRD"]:
                add(
                    "correction_subset",
                    f"{c} accuracy CORR " + ("available" if available else "remainder"),
                    coh,
                    [(1, c)],
                    mask=mask,
                )
    for available in [True, False]:
        mask = np.array(
            [
                (q not in miss.get("Q-INT-CORR", set())) == available
                for q in qids["POOL"]
            ]
        )
        for c in ["Q-NONE", "Q-SELF-GRD"]:
            add(
                "correction_subset",
                f"{c} accuracy CORR " + ("available" if available else "remainder"),
                "POOL",
                [(1, c)],
                mask=mask,
            )
    yn = items.loc[qids["RAD"], "kind"].eq("yesno").to_numpy()
    rr = rows[(rows["cohort"] == "RAD") & (rows["kind"] == "yesno")]
    for c in [
        "Q-NONE",
        "Q-INT-EMPTY",
        "Q-SELF-GRD",
        "Q-G-GRD",
        "Q-INT-PARA",
        "Q-INT-SHUF",
    ]:
        cr = rr[rr.cell.eq(c)].set_index("qid")
        validmask = e.missing_mask("RAD", [c]) & yn
        for label in ["model No rate", "accuracy gold=Yes", "accuracy gold=No"]:
            mask = validmask.copy()
            if label == "model No rate":
                v = (
                    cr["lead"]
                    .eq("No")
                    .reindex(qids["RAD"], fill_value=False)
                    .astype(float)
                    .to_numpy()
                )
            else:
                gold = label.split("=")[1]
                mask &= (
                    cr["gold_label"]
                    .eq(gold)
                    .reindex(qids["RAD"], fill_value=False)
                    .to_numpy()
                )
                v = e.vec("correct_lead", c, "RAD")
            p, r = boots["RAD"].endpoint(v, mask, "mean")
            out.append(
                dict(
                    task="answer_direction",
                    contrast=f"{c} {label}",
                    code_contrast="new",
                    cohort="RAD yes/no",
                    n_questions=int(mask.sum()),
                    **summarize(p, r),
                    source="io_adapter + locked parser + locked bootstrap",
                )
            )
    cr = rr[rr.cell.eq("Q-NONE")].set_index("qid")
    v = (
        cr.gold_label.eq("No")
        .reindex(qids["RAD"], fill_value=False)
        .astype(float)
        .to_numpy()
    )
    p, r = boots["RAD"].endpoint(v, yn, "mean")
    out.append(
        dict(
            task="answer_direction",
            contrast="gold No rate",
            code_contrast="new",
            cohort="RAD yes/no",
            n_questions=int(yn.sum()),
            **summarize(p, r),
            source="gold read programmatically only via io_adapter; no labels exported",
        )
    )
    out_dir.mkdir(parents=True)
    pd.DataFrame(out).to_csv(
        out_dir / "supplementary_comparisons.csv", index=False, encoding="utf-8"
    )
    reports = {
        "additive_equation_checks.json": equations,
        "analysis_metadata.json": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "warnings": warns + w2,
            "scoring_integrity": integ,
            "bootstrap_replicates": B,
            "seed": SEED,
            "analysis_type": "post-hoc exploratory",
        },
    }
    sources = {}
    for path in sorted(runs_dir.glob("*/shard_*/*")):
        if path.name in {"records.jsonl", "integrity.json", "run_config.json"}:
            sources[path.relative_to(runs_dir).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    reports["input_sha256.json"] = sources
    for filename, value in reports.items():
        (out_dir / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return {
        "comparisons": pd.DataFrame(out),
        "equations": equations,
        "metadata": reports["analysis_metadata.json"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Supplementary paired medical VQA analyses."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path(PROJECT_ROOT) / "runs")
    parser.add_argument(
        "--out-dir", type=Path, default=Path(PROJECT_ROOT) / "results" / "supplementary"
    )
    args = parser.parse_args(argv)
    result = analyze(args.runs_dir, args.out_dir)
    print(
        f"Exported {len(result['comparisons'])} aggregate comparisons to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
