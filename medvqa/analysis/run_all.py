"""Reproduce the predefined statistical tables from recorded experiment outputs."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

from medvqa.analysis import io_adapter as IO
from medvqa.analysis import mixed_models as MM
from medvqa.analysis import rq2 as RQ2
from medvqa.analysis.bootstrap import ClusterBootstrap
from medvqa.analysis.constants import ALPHA, SEED, TOST_MARGIN_PP, B
from medvqa.analysis.contrasts import (
    Engine,
    confirmatory_specs,
    corr_by_type,
    descriptive_specs,
    design_cells,
    required_cells,
    routeq_report,
    sensitivity,
)
from medvqa.analysis.leakage import check13
from medvqa.analysis.metrics import (
    cell_endpoints,
    invalid_rates,
    judge_status_table,
    observation_checks,
    open_judge,
)
from medvqa.analysis.scoring import (
    int_missing_qids,
    prepare_items,
    question_table,
    score_rows,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROTOCOL_DIR = os.path.join(PROJECT_ROOT, "resources/protocol")
CODE_FILES = [
    "constants.py",
    "parsers.py",
    "scoring.py",
    "bootstrap.py",
    "contrasts.py",
    "metrics.py",
    "mixed_models.py",
    "rq2.py",
    "leakage.py",
    "io_adapter.py",
    "run_all.py",
    "../judge_scope.py",
]


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_boots(items: pd.DataFrame):
    """Build paired image-cluster bootstrap samples for all cohorts.

    Args:
        items: Question table with qid, cohort, dataset, group and cluster_id.

    Returns:
        tuple[dict, dict]: Ordered question IDs and bootstrap objects by cohort.
    """
    qids, boots, parts = ({}, {}, {})
    for coh in ("RAD", "NAT"):
        g = items[items["cohort"] == coh].sort_values("qid")
        qids[coh] = list(g["qid"])
        strat = g["dataset"] if coh == "RAD" else g["group"]
        boots[coh] = ClusterBootstrap(
            list(g["cluster_id"]), list(strat), list(g["group"]), B=B, seed=SEED
        )
        parts[coh] = {"cluster": list(g["cluster_id"]), "group": list(g["group"])}
    qids["POOL"] = qids["RAD"] + qids["NAT"]
    boots["POOL"] = ClusterBootstrap.concat(
        boots["RAD"], boots["NAT"], parts["RAD"], parts["NAT"]
    )
    return (qids, boots)


def analyze(runs_dir: str, out_dir: str) -> dict:
    """Score saved runs and calculate the study's statistical tables.

    Args:
        runs_dir: Run collection including prepared questions and separate scoring
            labels.
        out_dir: Destination directory for aggregate statistical tables and audit files.

    Returns:
        dict: Tables, integrity results, question scores and analysis objects.

    Raises:
        ValueError: Run membership or configuration fails validation.
        OSError: Required inputs or output files cannot be accessed.
    """
    data, warns = IO.load_all(runs_dir, required_cells)
    design = data["design_state"]["design"]
    routes_tbl = IO.load_routes_table(PROTOCOL_DIR)
    items, w2 = prepare_items(data["items"], routes_tbl)
    warns += w2
    mism = items[
        items["answer_kind_schema"].notna()
        & (items["answer_kind_schema"] != items["kind"])
    ]
    if len(mism):
        warns.append(
            (
                "INTEGRITY: "
                f"{len(mism)}"
                " questions where SCHEMA answer_kind != protocol "
                "question kind (classification rules applied)"
            )
        )
    answers = data["answers"]
    present = set(answers["cell"])
    req = required_cells(design)
    cells = sorted(present)
    rows, integ = score_rows(items, answers, cells, required=req)
    qtab = question_table(rows)
    intmiss = int_missing_qids(rows)
    qids, boots = build_boots(items)
    eng = Engine(items, rows, qtab, intmiss, boots, qids)
    conf_specs = confirmatory_specs(design)
    conf, _ = eng.run_confirmatory(conf_specs)
    desc = eng.run_descriptive(descriptive_specs(design), present)
    rq = routeq_report(eng, conf, conf_specs)
    plan = enrich_corr_plan(data["corruption_plan"], items, warns)
    sens = sensitivity(eng, conf_specs, plan)
    corr_t = corr_by_type(eng, plan, {})
    endpoints = cell_endpoints(eng, cells)
    inval = invalid_rates(rows)
    leak = check13(items)
    obs = data["observations"]

    def n_items(arm, route_key):
        rl = str(route_key or "RT-FALLBACK").split("+")
        return sum((len(IO.route_items(PROTOCOL_DIR, arm, r)) for r in rl))

    es9 = observation_checks(obs, n_items)
    mm = MM.run_models(rows, items, conf_specs)
    slen = MM.s_len(rows, items, conf_specs, obs)
    MM.write_r_script(rows, items, conf_specs, os.path.join(out_dir, "mixed_models_R"))
    strata = dict(zip(items["image_id"], items["dataset"]))
    img_route = dict(zip(items["image_id"], items["routes_l"]))
    rq2_level = data["design_state"].get("rq2", "FULL")
    rq2 = RQ2.run_rq2(
        data["judge"],
        obs,
        data["jprobe_set"],
        strata,
        lambda img: sum(
            (
                len(IO.route_items(PROTOCOL_DIR, "GRD", r))
                for r in img_route.get(img) or ["RT-FALLBACK"]
            )
        ),
        level=rq2_level,
        ax1_scope=data.get("ax1_scope"),
    )
    dc = design_cells(design)
    if rq2_level == "NONE":
        c17 = RQ2.not_run_table()
    else:
        eligible = (
            set(data["ax1_scope"]["eligible_ids"]) if data.get("ax1_scope") else None
        )
        c17 = RQ2.c17(
            rows,
            items,
            rq2.get("support_by_obs"),
            dc["answerers"],
            dc["ext"],
            rq2.get("probe_rule_pass"),
            eligible,
        )
    rq1_cells = [
        c
        for c in cells
        if c.endswith(("-NONE", "-SELF-GRD", "-G-GRD", "-Q-GRD")) and "INT" not in c
    ]
    es8 = (
        open_judge(data["judge"], rq1_cells)
        if design != "MAIN"
        else RQ2.not_run_table()
    )
    jstat = judge_status_table(data["judge"])
    rq2_check = RQ2.rq2_integrity(
        data["judge"], rq2, c17, data["jprobe_set"] is not None, level=rq2_level
    )
    tables = {
        "T01_cell_endpoints": endpoints,
        "T02_confirmatory_holm_tost": conf,
        "T03_descriptive": desc,
        "T04_R-ROUTEQ": rq,
        "T05_sensitivity": sens,
        "T06_E-S1_invalid": inval,
        "T07_INT-CORR_by_type": corr_t,
        "T08_mixed_models_GEE": mm,
        "T09_S-LEN": slen,
        "T10_check13_leakage": leak,
        "T11_E-S9_observation": es9,
        "T12_E-S8_open_judge": es8,
        "T13_C17": c17,
        "T14_judge_status": jstat,
    }
    if data.get("ax1_scope") and rq2_level != "NONE":
        tables["T13_C17_scope_attrition"] = pd.concat(
            [
                RQ2.c17_sample(
                    rows,
                    items,
                    rq2.get("support_by_obs"),
                    k,
                    dc["ext"].get(k),
                    eligible,
                )[1]
                for k in dc["answerers"]
            ],
            ignore_index=True,
        )
    for name, t in rq2.get("tables", {}).items():
        tables[f"T2x_RQ2_{name}"] = t
    integ.update(
        {
            "design": design,
            "design_state": data["design_state"],
            "cells_present": cells,
            "required_cells": req,
            "warnings": warns,
            "n_questions": {c: len(q) for c, q in qids.items()},
            "n_closed_RAD": int(
                (
                    (items["cohort"] == "RAD")
                    & items["kind"].isin(["yesno", "closed_short"])
                ).sum()
            ),
            "kind_counts": items.groupby(["cohort", "kind"])
            .size()
            .rename("n")
            .reset_index()
            .to_dict("records"),
            "int_missing_images": {
                c: len({items.loc[q, "image_id"] for q in s})
                for c, s in intmiss.items()
            },
            "bootstrap_notes": boots["RAD"].notes + boots["NAT"].notes,
        }
    )
    integ["judge_roles"] = data.get("judge_roles")
    integ["rq2_status"] = rq2_check["status"]
    integ["rq2_problems"] = rq2_check["problems"]
    ok_rows = integ["missing_rows"] == 0 and integ["unexpected_rows"] == 0
    integ["status"] = "PASS" if ok_rows and rq2_check["status"] != "FAIL" else "FAIL"
    if rq2_check["status"] not in ("PASS", "NOT_RUN_MAIN"):
        banner = "!" * 78
        msg = f"RQ2 {rq2_check['status']}: " + "; ".join(rq2_check["problems"])
        print("\n".join([banner, msg, banner]), file=sys.stderr)
        warns.append(msg)
    write_outputs(out_dir, tables, integ, rq2, runs_dir)
    return {
        "tables": tables,
        "integrity": integ,
        "rq2": rq2,
        "rows": rows,
        "items": items,
        "engine": eng,
    }


def enrich_corr_plan(plan, items, warns=None) -> Dict[str, dict]:
    """Attach edited item names and synonyms to the intervention plan.

    corruption_plan.json -> {image: edit_type, item_name, synonyms} for
    S-CORR-REL.

    The plan's own `item_name` takes precedence; the item lookup by `item_no` is a
    fallback, and disagreements are warned.
    """
    if not plan:
        return {}
    asg = plan.get("assignments", plan)
    routes = dict(zip(items["image_id"], items["routes_l"]))
    out, mismatch = ({}, 0)
    for img, a in asg.items():
        names = []
        for r in routes.get(img) or []:
            names += IO.route_items(PROTOCOL_DIR, "GRD", r)
        ino = a.get("item_no")
        full = names[ino - 1] if ino and 0 < ino <= len(names) else ""
        looked = full.split(":")[0].strip()
        name = a.get("item_name") or looked
        if a.get("item_name") and looked and (P_norm(a["item_name"]) != P_norm(looked)):
            mismatch += 1
        out[img] = {
            "edit_type": a.get("edit_type"),
            "item_name": name,
            "synonyms": a.get("synonyms", []),
        }
    if mismatch and warns is not None:
        warns.append(
            (
                "corruption_plan: "
                f"{mismatch}"
                " images where item_name != GRD item at item_no "
                "(plan used)"
            )
        )
    return out


def P_norm(x):
    from medvqa.analysis.parsers import norm

    return norm(str(x).split(":")[0])


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict("records"))
    return o


def to_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(empty)_\n"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = [
        "| " + " | ".join((str(v).replace("|", "/") for v in r)) + " |"
        for r in d.itertuples(index=False)
    ]
    return "\n".join([head, sep] + body) + "\n"


def write_outputs(
    out_dir: str, tables: dict, integ: dict, rq2: dict, runs_dir: str
) -> None:
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    for name, t in tables.items():
        if t is None:
            continue
        t.to_csv(
            os.path.join(out_dir, "tables", f"{name}.csv"),
            index=False,
            encoding="utf-8",
        )
        with open(
            os.path.join(out_dir, "tables", f"{name}.md"), "w", encoding="utf-8"
        ) as f:
            f.write(f"# {name}\n\n" + to_md(t))
    manifest = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "locked": {
            "SEED": SEED,
            "B": B,
            "ALPHA": ALPHA,
            "TOST_MARGIN_PP": TOST_MARGIN_PP,
            "parsers": ["PARSE-LEAD-v1", "PARSE-EXACT-v1"],
        },
        "code_sha256": {
            f: sha256_file(os.path.join(HERE, f))
            for f in CODE_FILES
            if os.path.exists(os.path.join(HERE, f))
        },
        "runs_dir": os.path.abspath(runs_dir),
    }
    res = {
        "manifest": manifest,
        "integrity": integ,
        "rq2": {k: v for k, v in rq2.items() if k != "tables"},
        "tables": {k: v for k, v in tables.items()},
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(_clean(res), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "integrity.json"), "w", encoding="utf-8") as f:
        json.dump(_clean(integ), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(_clean(manifest), f, ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Analyze saved experiment runs and write result tables."
    )
    ap.add_argument("--runs-dir", default=os.path.join(PROJECT_ROOT, "runs"))
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results"))
    a = ap.parse_args(argv)
    if os.path.abspath(a.out_dir).startswith(os.path.abspath(PROTOCOL_DIR)):
        raise SystemExit("refusing to write inside resources/protocol/")
    res = analyze(a.runs_dir, a.out_dir)
    print(f"done: integrity={res['integrity']['status']} -> {a.out_dir}")


if __name__ == "__main__":
    main()
