"""Question-level endpoints, validity rates and structured observation summaries."""

from __future__ import annotations

import json
import re
from typing import List, Optional

import pandas as pd

from .bootstrap import summarize
from .constants import JUDGE_JSON_FENCE_REGEX, JUDGE_STRIP_ONE_JSON_FENCE
from .contrasts import Engine, Spec

_STMT = re.compile("^\\s*(\\d+)\\.\\s*(.+?)\\s*\\|\\s*(.+?)\\s*\\|\\s*(.*)$")


def cell_endpoints(eng: Engine, cells: List[str]) -> pd.DataFrame:
    """Calculate primary and secondary outcomes for each condition.

    E-P1 (RAD closed, mean), E-P2 (NAT, K orders -> group -> macro) and secondary per
    cell, 95% CI.
    """
    plan = [
        ("E-P1", "RAD", "correct_lead"),
        ("E-P2", "NAT", "correct_lead"),
        ("S-EXACT-acc", "RAD", "correct_exact"),
        ("S-EXACT-acc", "NAT", "correct_exact"),
        ("S-VALID", "RAD", "correct_valid"),
        ("S-VALID", "NAT", "correct_valid"),
        ("E-S2", "RAD", "p_gold"),
        ("E-S2", "NAT", "p_gold"),
        ("E-S2b", "RAD", "argmax_correct"),
        ("E-S2b", "NAT", "argmax_correct"),
        ("E-S3", "RAD", "brier"),
        ("E-S3", "NAT", "brier"),
        ("E-S6", "RAD", "em"),
        ("E-S7", "RAD", "f1"),
    ]
    recs = []
    for cell in cells:
        for ep, coh, metric in plan:
            sp = Spec(f"{ep}[{cell}]", "acc", coh, [(1, cell)])
            try:
                point, reps, n = eng.evaluate(sp, metric=metric)
            except KeyError:
                continue
            scale = 1.0 if metric == "brier" else 100.0
            recs.append(
                {
                    "endpoint": ep,
                    "cell": cell,
                    "cohort": coh,
                    "metric": metric,
                    "n_questions": n,
                    **summarize(point, reps, scale=scale),
                }
            )
    df = pd.DataFrame(recs)
    return df.drop(columns=["p"], errors="ignore")


def invalid_rates(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize invalid answers and generation errors.

    E-S1: invalid rate per cell x cohort x kind (rows), incl. 'B is correct'-type count.
    """
    r = rows[rows["kind"] != "open"].copy()
    r["invalid"] = 1 - r["valid_lead"]
    g = r.groupby(["cell", "cohort", "kind"]).agg(
        n_rows=("invalid", "size"),
        invalid_rate=("invalid", "mean"),
        n_gen_error=("gen_error", "sum"),
        n_missing_row=("missing_row", "sum"),
        n_b_is_correct=("b_is_correct", "sum"),
        n_int_missing=("int_missing", "sum"),
    )
    return g.reset_index()


def observation_checks(obs: pd.DataFrame, expected_items) -> pd.DataFrame:
    """Summarize observation format, length and truncation.

    E-S9 manipulation check: format compliance, output length, truncated rate per
    observer x arm x route.
    """
    if obs is None or obs.empty:
        return pd.DataFrame()
    recs = []
    for r in obs.itertuples(index=False):
        lines = [line for line in str(r.text).splitlines() if line.strip()]
        n_fmt = sum((bool(_STMT.match(line)) for line in lines))
        exp = expected_items(r.arm, r.route_key)
        recs.append(
            {
                "observer": r.observer,
                "arm": r.arm,
                "route": r.route_key,
                "format_ok": float(
                    bool(lines) and n_fmt == len(lines) and (exp == 0 or n_fmt == exp)
                ),
                "lines_fmt_share": n_fmt / len(lines) if lines else 0.0,
                "n_output_tokens": r.n_output_tokens,
                "truncated": float(r.truncated),
                "obs_error": float(r.obs_error),
                "empty": float(not lines),
            }
        )
    d = pd.DataFrame(recs)
    g = (
        d.groupby(["observer", "arm", "route"])
        .agg(
            n=("format_ok", "size"),
            format_ok_rate=("format_ok", "mean"),
            lines_fmt_share=("lines_fmt_share", "mean"),
            mean_output_tokens=("n_output_tokens", "mean"),
            truncated_rate=("truncated", "mean"),
            obs_error_rate=("obs_error", "mean"),
            empty_rate=("empty", "mean"),
        )
        .reset_index()
    )
    tot = (
        d.groupby(["observer", "arm"])
        .agg(
            n=("format_ok", "size"),
            format_ok_rate=("format_ok", "mean"),
            lines_fmt_share=("lines_fmt_share", "mean"),
            mean_output_tokens=("n_output_tokens", "mean"),
            truncated_rate=("truncated", "mean"),
            obs_error_rate=("obs_error", "mean"),
            empty_rate=("empty", "mean"),
        )
        .reset_index()
        .assign(route="ALL")
    )
    return pd.concat([tot, g], ignore_index=True)


JUDGE_FENCE_RE = re.compile(JUDGE_JSON_FENCE_REGEX, re.S)
AX1_LABEL_SET = ("supported", "contradicted", "not_verifiable")


def _judge_obj(text: Optional[str], strip_fence: bool) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if strip_fence:
        m = JUDGE_FENCE_RE.match(s)
        if m:
            s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _schema_ok(task: str, obj: dict, ids: Optional[set]) -> bool:
    """Task schemas of templates/judge_prompts.md (J-AX1, J-AX2, J-AX3, J-OPEN)."""
    if task == "J-AX1":
        labs = obj.get("labels")
        if not isinstance(labs, list):
            return False
        try:
            got = {int(e["id"]): e["label"] for e in labs}
        except (KeyError, TypeError, ValueError):
            return False
        if len(got) != len(labs) or any((v not in AX1_LABEL_SET for v in got.values())):
            return False
        return ids is None or set(got) == set(ids)
    if task == "J-AX2":
        ia = obj.get("items_addressed")
        return (
            isinstance(ia, int)
            and (not isinstance(ia, bool))
            and (obj.get("diagnosis") in ("yes", "no"))
            and (obj.get("unsupported_claim") in ("yes", "no"))
        )
    if task == "J-AX3":
        r = obj.get("relevance")
        return isinstance(r, int) and (not isinstance(r, bool)) and (r in (0, 1, 2))
    if task == "J-OPEN":
        return obj.get("verdict") in ("correct", "incorrect")
    return False


def _judge_payload(
    task: str, text: Optional[str], ids: Optional[set], strip_fence: bool
) -> Optional[dict]:
    obj = _judge_obj(text, strip_fence)
    return obj if obj is not None and _schema_ok(task, obj, ids) else None


def parse_judge_json(
    task: str, text: Optional[str], ids: Optional[set] = None
) -> Optional[dict]:
    """Parse judge outputs with the scoring format rules.

    The same parsing rules are used for scoring and technical checks.

    Fence handling follows constants.JUDGE_STRIP_ONE_JSON_FENCE only; schema checked per
    task; for J-AX1 the label ids must equal the statement ids of the record. None ==
    judge_invalid.
    """
    return _judge_payload(task, text, ids, JUDGE_STRIP_ONE_JSON_FENCE)


def ax1_ids(refs) -> Optional[set]:
    st = (refs or {}).get("statements") if isinstance(refs, dict) else None
    try:
        return {int(x["id"]) for x in st} if st else None
    except (KeyError, TypeError, ValueError):
        return None


def open_judge(judge: pd.DataFrame, cells_rq1: List[str]) -> pd.DataFrame:
    """Summarize open-answer correctness from both judges.

    E-S8: J-OPEN correctness per cell for J1, J2 and 'both agree correct' (11 RQ1
    cells).
    """
    if judge is None or judge.empty:
        return pd.DataFrame()
    j = judge[judge["task"] == "J-OPEN"]
    if j.empty:
        return pd.DataFrame()
    recs = []
    for r in j.itertuples(index=False):
        refs = r.refs if isinstance(r.refs, dict) else {}
        ak = str(refs.get("answer_key", ""))
        parts = ak.split("|")
        cell = (
            parts[1][5:]
            if len(parts) > 1 and parts[1].startswith("CELL-")
            else parts[1]
            if len(parts) > 1
            else None
        )
        st = getattr(r, "status", "ok")
        if st == "empty_prediction":
            recs.append(
                {
                    "judge": r.judge,
                    "cell": cell,
                    "qid": refs.get("qid"),
                    "valid": True,
                    "correct": 0.0,
                }
            )
            continue
        obj = parse_judge_json("J-OPEN", r.text)
        recs.append(
            {
                "judge": r.judge,
                "cell": cell,
                "qid": refs.get("qid"),
                "valid": obj is not None,
                "correct": float(obj is not None and obj["verdict"] == "correct"),
            }
        )
    d = pd.DataFrame(recs)
    d = d[d["cell"].isin(cells_rq1)]
    out = (
        d.groupby(["cell", "judge"])
        .agg(
            n=("correct", "size"),
            acc=("correct", "mean"),
            invalid_rate=("valid", lambda v: 1 - v.mean()),
        )
        .reset_index()
    )
    piv = d.pivot_table(index=["cell", "qid"], columns="judge", values="correct")
    if {"J1", "J2"} <= set(piv.columns):
        both = ((piv["J1"] == 1) & (piv["J2"] == 1)).groupby(level=0).mean()
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    {
                        "cell": both.index,
                        "judge": "both_agree_correct",
                        "n": piv.groupby(level=0).size().values,
                        "acc": both.values,
                    }
                ),
            ]
        )
    out["acc"] = out["acc"] * 100
    return out.reset_index(drop=True)


def judge_status_table(judge: pd.DataFrame) -> pd.DataFrame:
    """Summarize judge calls, skipped units and valid JSON outputs.

    Per judge x task: records, skipped (no call), schema-valid via parse_judge_json (the
    scoring parser), judge_invalid, and how many invalid outputs WOULD be valid with one
    outer fence removed (same regex JUDGE_JSON_FENCE_REGEX; diagnostic for NQ-1 only,
    never used for scoring).
    """
    if judge is None or judge.empty:
        return pd.DataFrame()
    recs = []
    for r in judge.to_dict("records"):
        st = r.get("status") or "ok"
        skipped = st in ("obs_error_skipped", "no_statements", "empty_prediction")
        ids = ax1_ids(r.get("refs")) if r.get("task") == "J-AX1" else None
        ok = not skipped and parse_judge_json(r["task"], r.get("text"), ids) is not None
        fenced = (
            not skipped
            and (not ok)
            and (_judge_payload(r["task"], r.get("text"), ids, True) is not None)
        )
        recs.append(
            {
                "judge": r["judge"],
                "task": r["task"],
                "skipped": skipped,
                "valid": ok,
                "invalid": not skipped and (not ok),
                "fence_would_parse": fenced,
            }
        )
    d = pd.DataFrame(recs)
    out = (
        d.groupby(["judge", "task"])
        .agg(
            n=("valid", "size"),
            n_skipped=("skipped", "sum"),
            n_valid_schema=("valid", "sum"),
            n_invalid=("invalid", "sum"),
            n_fence_would_parse=("fence_would_parse", "sum"),
        )
        .reset_index()
    )
    out["strip_one_fence_in_force"] = JUDGE_STRIP_ONE_JSON_FENCE
    return out
