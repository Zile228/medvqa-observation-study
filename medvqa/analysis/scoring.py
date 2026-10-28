"""Score answers and aggregate option rotations with the predefined endpoint rules."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import parsers as P
from .constants import ROUTEQ_OMNI_TYPES, ROUTEQ_SLAKE_CONTENT, ROUTEQ_VQARAD_TOKENS
from .missing import is_missing_scalar

INVALID = "<INVALID>"


def routeq_flag(dataset: str, qtype_meta) -> Optional[bool]:
    """Identify routing questions from metadata labels without reading answers."""
    if is_missing_scalar(qtype_meta):
        return None
    t = str(qtype_meta)
    if dataset == "SLAKE":
        return t.strip().lower() in ROUTEQ_SLAKE_CONTENT
    if dataset == "VQA-RAD":
        up = t.upper()
        return any((tok in up for tok in ROUTEQ_VQARAD_TOKENS))
    return t.strip().lower() in ROUTEQ_OMNI_TYPES


def _as_list(v) -> List[str]:
    if is_missing_scalar(v):
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def prepare_items(
    items: pd.DataFrame, routes_table: Optional[dict] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Normalize question metadata and attach the fixed routing groups.

    Args:
        items: Raw question metadata table.
        routes_table: Optional routing table derived from the fixed study
            specifications.

    Returns:
        tuple[pandas.DataFrame, list]: Normalized questions and data-quality warnings.
    """
    warn: List[str] = []
    it = items.copy()
    it["kind"] = [
        P.answer_kind(q, at, c)
        for q, at, c in zip(it["question"], it["answer_type"], it["cohort"])
    ]
    if "group" not in it or it["group"].isna().all():
        it["group"] = None
    it.loc[it["cohort"] == "RAD", "group"] = it.loc[it["cohort"] == "RAD", "dataset"]
    ks, gidx = ([], [])
    for _, r in it.iterrows():
        if r["kind"] != "mcq":
            ks.append(0)
            gidx.append(-1)
            continue
        opts = list(r["options"])
        ks.append(len(opts))
        gi = r["gold_index"]
        if is_missing_scalar(gi):
            hits = [i for i, o in enumerate(opts) if P.norm(o) == P.norm(r["gold"])]
            if len(hits) != 1:
                raise ValueError(f"{r['qid']}: cannot locate gold option uniquely")
            gi = hits[0]
        gidx.append(int(gi))
    it["K"] = ks
    it["gold_index"] = gidx
    it["routeq"] = [routeq_flag(d, m) for d, m in zip(it["dataset"], it["qtype_meta"])]
    if it["routeq"].isna().any():
        warn.append(
            f"R-ROUTEQ: {int(it['routeq'].isna().sum())} questions lack qtype_meta"
        )
    it["routes_l"] = [_as_list(v) for v in it["routes"]]
    if it["weak_group"].isna().any():
        warn.append(
            (
                "weak_group missing for some items: derived from "
                "routes.json (misses MR-chest FALLBACK)"
            )
        )
        tbl = routes_table or {}
        derived = [
            any((tbl.get(rt, {}).get("weak_group", False) for rt in rl))
            for rl in it["routes_l"]
        ]
        it["weak_group"] = [
            bool(w) if not is_missing_scalar(w) else d
            for w, d in zip(it["weak_group"], derived)
        ]
    it["weak_group"] = it["weak_group"].astype(bool)
    if "route_header" not in it.columns:
        tbl = routes_table or {}
        hdr = []
        for rl in it["routes_l"]:
            if not rl or rl == ["RT-FALLBACK"]:
                hdr.append("Image context: not provided.")
            else:
                hdr.append(
                    "Image context: "
                    + " ; ".join((tbl.get(rt, {}).get("header", "") for rt in rl))
                    + "."
                )
        it["route_header"] = hdr
    it = it.set_index("qid", drop=False)
    it.index.name = None
    return (it, warn)


def _label_set(kind: str, k: int) -> List[str]:
    if kind == "yesno":
        return ["Yes", "No"]
    if kind == "mcq":
        return list(P.LETTERS[:k])
    return []


def _ptilde(lp, labels: List[str]) -> Optional[np.ndarray]:
    """Label probability: p~(c) = exp lp(c) / sum_c' exp lp(c') over the label set."""
    if not labels or lp is None or (not isinstance(lp, dict)):
        return None
    try:
        v = np.array([float(lp[c]) for c in labels])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(v)):
        return None
    v = np.exp(v - v.max())
    return v / v.sum()


def expected_rows(items: pd.DataFrame, cells: List[str]) -> pd.DataFrame:
    """Enumerate all required condition, question and option-order rows.

    One row per (cell, qid, order) the design requires: RAD order 0, NAT orders 0..K-1.
    """
    recs = []
    for q, r in items.iterrows():
        orders = range(r["K"]) if r["cohort"] == "NAT" else [0]
        for o in orders:
            recs.append((q, o))
    base = pd.DataFrame(recs, columns=["qid", "order"])
    return pd.concat([base.assign(cell=c) for c in cells], ignore_index=True)


def score_rows(
    items: pd.DataFrame,
    answers: pd.DataFrame,
    cells: List[str],
    required: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Score answer turns and account for missing required executions.

    Args:
        items: Prepared question metadata including reference answers.
        answers: Normalized raw answer turns.
        cells: Conditions present or expected in this analysis.
        required: Conditions requiring complete membership; None requires all cells.

    Returns:
        tuple[pandas.DataFrame, dict]: Scored turns and missing/unexpected-row counts.
    """
    req = [c for c in cells if required is None or c in required]
    opt = [c for c in cells if c not in req]
    exp = expected_rows(items, req)
    a = answers[answers["cell"].isin(req)]
    df = exp.merge(a, on=["cell", "qid", "order"], how="left", indicator=True)
    extra = answers[answers["cell"].isin(req)].merge(
        exp, on=["cell", "qid", "order"], how="left", indicator=True
    )
    integrity = {
        "missing_rows": int((df["_merge"] == "left_only").sum()),
        "missing_rows_by_cell": df[df["_merge"] == "left_only"]
        .groupby("cell")
        .size()
        .to_dict(),
        "unexpected_rows": int((extra["_merge"] == "left_only").sum()),
        "optional_cells_partial": {},
    }
    df["missing_row"] = df["_merge"] == "left_only"
    df = df.drop(columns="_merge")
    if opt:
        exp_o = expected_rows(items, opt)
        o = exp_o.merge(
            answers[answers["cell"].isin(opt)], on=["cell", "qid", "order"], how="inner"
        )
        o["missing_row"] = False
        integrity["optional_cells_partial"] = {
            c: f"{int((o['cell'] == c).sum())}/{int((exp_o['cell'] == c).sum())}"
            for c in opt
        }
        df = pd.concat([df, o], ignore_index=True)
    df["gen_error"] = (
        df["gen_error"].map(lambda v: True if is_missing_scalar(v) else bool(v))
        | df["missing_row"]
    )
    df["int_missing"] = df["int_missing"].map(
        lambda v: False if is_missing_scalar(v) else bool(v)
    )
    itd = items.to_dict("index")
    out = {
        k: []
        for k in (
            "kind",
            "lead",
            "exact",
            "gold_label",
            "correct_lead",
            "correct_exact",
            "valid_lead",
            "valid_exact",
            "ptilde",
            "p_gold",
            "argmax_correct",
            "brier",
            "b_is_correct",
            "em",
            "f1",
            "cohort",
            "image_id",
        )
    }
    for r in df.itertuples(index=False):
        ir = itd[r.qid]
        kind = ir["kind"]
        forced_invalid = bool(r.gen_error) or bool(ir["image_broken"])
        presented = P.shift_options(ir["options"], r.order) if kind == "mcq" else None
        if kind == "mcq":
            gold_label = P.gold_letter_for_order(ir["gold_index"], ir["K"], r.order)
        elif kind == "yesno":
            g = P.norm(ir["gold"])
            gold_label = {"yes": "Yes", "no": "No"}.get(g, ir["gold"])
        else:
            gold_label = ir["gold"]
        raw = (
            None if forced_invalid or is_missing_scalar(r.raw_output) else r.raw_output
        )
        lead = P.parse_lead(raw, kind, presented)
        exact = P.parse_exact(raw, kind, presented) if kind != "open" else None
        labels = _label_set(kind, ir["K"])
        pt = _ptilde(r.label_logprobs, labels) if not bool(ir["image_broken"]) else None
        if pt is not None:
            onehot = np.array([1.0 if c == gold_label else 0.0 for c in labels])
            p_gold = float((pt * onehot).sum())
            amax = float(labels[int(np.argmax(pt))] == gold_label)
            brier = float(((pt - onehot) ** 2).sum())
        else:
            p_gold = amax = brier = np.nan
        out["kind"].append(kind)
        out["lead"].append(lead)
        out["exact"].append(exact)
        out["gold_label"].append(gold_label)
        out["correct_lead"].append(
            float(P.is_correct(lead, kind, gold_label)) if kind != "open" else np.nan
        )
        out["correct_exact"].append(
            float(P.is_correct(exact, kind, gold_label)) if kind != "open" else np.nan
        )
        out["valid_lead"].append(float(lead is not None))
        out["valid_exact"].append(float(exact is not None))
        out["ptilde"].append(None if pt is None else dict(zip(labels, pt.tolist())))
        out["p_gold"].append(p_gold)
        out["argmax_correct"].append(amax)
        out["brier"].append(brier)
        out["b_is_correct"].append(
            float(kind == "mcq" and lead is None and P.b_is_correct_type(raw))
        )
        out["em"].append(
            float(lead is not None and P.norm(lead) == P.norm(ir["gold"]))
            if kind == "open"
            else np.nan
        )
        out["f1"].append(P.token_f1(lead, ir["gold"]) if kind == "open" else np.nan)
        out["cohort"].append(ir["cohort"])
        out["image_id"].append(ir["image_id"])
    for k, v in out.items():
        df[k] = v
    df["correct_valid"] = np.where(df["valid_lead"] == 1, df["correct_lead"], np.nan)
    df["label_key"] = [
        INVALID
        if is_missing_scalar(line)
        else P.norm(line)
        if k in ("closed_short", "open")
        else line
        for line, k in zip(df["lead"], df["kind"])
    ]
    df["label_key_exact"] = [
        INVALID
        if is_missing_scalar(line)
        else P.norm(line)
        if k in ("closed_short", "open")
        else line
        for line, k in zip(df["exact"], df["kind"])
    ]
    return (df, integrity)


QCOLS = [
    "correct_lead",
    "correct_exact",
    "correct_valid",
    "valid_lead",
    "p_gold",
    "argmax_correct",
    "brier",
    "em",
    "f1",
]


def question_table(rows: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Average each question's scores over its option-order executions.

    Args:
        rows: Scored answer turns, including cyclic option orders.

    Returns:
        dict[str, pandas.DataFrame]: Metric names mapped to question-by-condition
            tables.
    """
    g = rows.groupby(["cell", "qid"])[QCOLS].mean()
    return {c: g[c].unstack("cell") for c in QCOLS}


def int_missing_qids(rows: pd.DataFrame) -> Dict[str, set]:
    """Identify questions excluded by missing image interventions.

    Images flagged int_missing are dropped from the contrasts using that intervention.
    """
    m = rows[rows["int_missing"]]
    img = m.groupby("cell")["image_id"].apply(set).to_dict()
    res = {}
    for cell, imgs in img.items():
        res[cell] = set(
            rows.loc[(rows["cell"] == cell) & rows["image_id"].isin(imgs), "qid"]
        )
    return res


def pair_table(
    rows: pd.DataFrame, int_cell: str, ref_cell: str, parser: str = "lead"
) -> pd.DataFrame:
    """Calculate paired answer flips and probability changes.

    Per-question E-S4 flip (invalid is a label), S-FLIPVALID flip, and E-S5 TV /
    delta p~(gold).
    """
    key = "label_key" if parser == "lead" else "label_key_exact"
    cols = ["qid", "order", key, "valid_lead", "ptilde", "p_gold", "kind"]
    a = rows.loc[rows["cell"] == int_cell, cols]
    b = rows.loc[rows["cell"] == ref_cell, cols]
    m = a.merge(b, on=["qid", "order"], suffixes=("_i", "_r"))
    m["flip"] = (m[f"{key}_i"].fillna(INVALID) != m[f"{key}_r"].fillna(INVALID)).astype(
        float
    )
    both_valid = (m["valid_lead_i"] == 1) & (m["valid_lead_r"] == 1)
    m["flip_valid"] = np.where(both_valid, m["flip"], np.nan)
    tv = []
    for pi, pr in zip(m["ptilde_i"], m["ptilde_r"]):
        if is_missing_scalar(pi) or is_missing_scalar(pr):
            tv.append(np.nan)
        else:
            tv.append(0.5 * sum((abs(pi[c] - pr[c]) for c in pr)))
    m["tv"] = tv
    m["dpgold"] = m["p_gold_i"] - m["p_gold_r"]
    return m.groupby("qid")[["flip", "flip_valid", "tv", "dpgold"]].mean()
