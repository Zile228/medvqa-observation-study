"""Technical validation of pilot outputs without reference-answer scoring."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd

from medvqa.analysis import io_adapter as IO
from medvqa.analysis import parsers as P
from medvqa.analysis.metrics import judge_status_table

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "resources/protocol"))
INVALID_GATE = 0.05
TRUNC_THRESHOLD = 0.02
EMPTY_THRESHOLD = 0.01
REPEAT_THRESHOLD = 0.01
_FORBIDDEN = ("prep_sealed", "gold_", "id_map", "source_labels", "coordinator_key")
_STMT = re.compile("^\\s*(\\d+)\\.\\s*(.+?)\\s*\\|\\s*(.+?)\\s*\\|\\s*(.*)$")
_REPEAT20 = re.compile("(.)\\1{19,}", re.S)


def _guard(path: str) -> str:
    low = path.replace("\\", "/").lower()
    if any((f in low for f in _FORBIDDEN)):
        raise PermissionError(f"pilot_gate must not read reference labels: {path}")
    return path


def load_pilot_records(run_dirs: List[str]) -> (Dict[str, pd.DataFrame], List[str]):
    rows, notes = ([], [])
    for rd in run_dirs:
        for p in sorted(glob.glob(os.path.join(rd, "shard_*", "records.jsonl"))):
            for r in IO.read_jsonl(_guard(p)):
                rows.append(r)
    if not rows:
        raise SystemExit("no records found under <run-dir>/shard_*/records.jsonl")
    df = pd.DataFrame(rows)
    mode = df.get("mode", pd.Series("PILOT", index=df.index)).fillna("PILOT")
    if (mode != "PILOT").any():
        notes.append(f"ignored {int((mode != 'PILOT').sum())} non-PILOT records")
    df = df[mode == "PILOT"]
    if "key" in df:
        st = df.get("status", pd.Series("ok", index=df.index)).fillna("ok")
        df = df.assign(
            _err=st.eq("error"),
            _t=df.get("created_utc", pd.Series("", index=df.index)).fillna(""),
        )
        df = df.sort_values(
            ["key", "_err", "_t"], ascending=[True, False, True]
        ).drop_duplicates("key", keep="last")
    return ({str(k): g.reset_index(drop=True) for k, g in df.groupby("schema")}, notes)


def _flag(q, name):
    return bool((q or {}).get(name)) if isinstance(q, dict) else False


def answer_table(ans: pd.DataFrame) -> pd.DataFrame:
    """Check answer validity without reading reference answers.

    Row-level PARSE-LEAD validity (no gold): invalid = parser None, error, broken image.
    """
    recs = []
    for r in ans.to_dict("records"):
        st = r.get("status") or "ok"
        if st == "skipped_int_missing":
            continue
        kind = r.get("answer_kind")
        op = r.get("options_presented")
        opts = [
            o["text"] if isinstance(o, dict) else o
            for o in (op if isinstance(op, list) else [])
        ]
        text = r.get("text") if st == "ok" else None
        lead = P.parse_lead(text, kind, opts) if kind else None
        lp = r.get("logprob")
        recs.append(
            {
                "answerer": str(
                    r.get("answerer") or IO.canonical_cell(r["cell"])[0]
                ).replace("ANS-", ""),
                "cell": IO.canonical_cell(r["cell"]),
                "kind": kind,
                "status": st,
                "invalid": float(lead is None),
                "b_is_correct": float(
                    kind == "mcq" and lead is None and P.b_is_correct_type(text)
                ),
                "empty": float(not str(text or "").strip()),
                "repeat_char20": float(bool(_REPEAT20.search(str(text or "")))),
                "nonfinite": float(isinstance(lp, dict) and lp.get("finite") is False),
                "logprob_missing": float(
                    kind in ("yesno", "mcq")
                    and st == "ok"
                    and (not isinstance(lp, dict))
                ),
                "truncated": float(bool(r.get("truncated"))),
                "timing_s": r.get("timing_s"),
            }
        )
    return pd.DataFrame(recs)


def _rates(d: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    return (
        d.groupby(by)
        .agg(
            n=("invalid", "size"),
            invalid_rate=("invalid", "mean"),
            b_is_correct_rate=("b_is_correct", "mean"),
            empty_rate=("empty", "mean"),
            repeat_char20_rate=("repeat_char20", "mean"),
            nonfinite_rate=("nonfinite", "mean"),
            logprob_missing_rate=("logprob_missing", "mean"),
            truncated_rate=("truncated", "mean"),
            median_timing_s=("timing_s", "median"),
        )
        .reset_index()
    )


def observation_table(
    obs: pd.DataFrame, protocol_dir: str = PROTOCOL_DIR
) -> pd.DataFrame:
    cache: Dict[tuple, int] = {}

    def n_items(arm, rk):
        key = (arm, rk)
        if key not in cache:
            cache[key] = sum(
                (
                    len(IO.route_items(protocol_dir, arm, r))
                    for r in str(rk or "RT-FALLBACK").split("+")
                )
            )
        return cache[key]

    recs = []
    for r in obs.to_dict("records"):
        arm = str(r.get("arm", "")).replace("ARM-", "")
        text = str(r.get("text") or "")
        lines = [line for line in text.splitlines() if line.strip()]
        nfmt = sum((bool(_STMT.match(line)) for line in lines))
        exp = n_items(arm, r.get("route_key"))
        recs.append(
            {
                "observer": str(r.get("observer", "")).replace("OBS-", ""),
                "arm": arm,
                "route": r.get("route_key"),
                "status": r.get("status") or "ok",
                "format_ok": float(
                    bool(lines) and nfmt == len(lines) and (exp == 0 or nfmt == exp)
                ),
                "truncated": float(bool(r.get("truncated"))),
                "empty": float(not text.strip()),
                "repeat_char20": float(bool(_REPEAT20.search(text))),
                "nonfinite": float(r.get("first_step_finite") is False),
                "obs_error": float((r.get("status") or "ok") in ("obs_error", "error")),
                "n_output_tokens": r.get("n_output_tokens"),
                "timing_s": r.get("timing_s"),
            }
        )
    return pd.DataFrame(recs)


def _obs_rates(d, by):
    return (
        d.groupby(by)
        .agg(
            n=("format_ok", "size"),
            format_ok_rate=("format_ok", "mean"),
            truncated_rate=("truncated", "mean"),
            empty_rate=("empty", "mean"),
            repeat_char20_rate=("repeat_char20", "mean"),
            nonfinite_rate=("nonfinite", "mean"),
            obs_error_rate=("obs_error", "mean"),
            mean_output_tokens=("n_output_tokens", "mean"),
            median_timing_s=("timing_s", "median"),
        )
        .reset_index()
    )


def label_map_check(run_dirs: List[str]) -> dict:
    """Check that every answer label has an accepted token ID.

    T(c) must be non-empty for every label of every tokenizer; empty ->
    configuration error.
    """
    out = {"files": [], "empty_labels": [], "status": "MISSING"}
    for rd in run_dirs:
        for p in sorted(
            glob.glob(os.path.join(rd, "artifacts", "logprob_label_map*.json"))
        ):
            m = IO.read_json(_guard(p))
            out["files"].append(os.path.basename(p))
            for tok, v in m.items():
                labels = (v or {}).get("labels", {}) if isinstance(v, dict) else {}
                for lab in ("Yes", "No", "A", "B", "C", "D"):
                    if not labels.get(lab):
                        out["empty_labels"].append(f"{tok}:{lab}")
    if out["files"]:
        out["status"] = "FAIL" if out["empty_labels"] else "PASS"
    return out


def timing_table(recs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sch, df in recs.items():
        if "timing_s" not in df:
            continue
        model = (
            df["answerer"]
            if "answerer" in df
            else df["observer"]
            if "observer" in df
            else df["judge"]
            if "judge" in df
            else df.get("editor", pd.Series("?", index=df.index))
        )
        t = pd.DataFrame(
            {
                "schema": sch,
                "model": model,
                "timing_s": pd.to_numeric(df["timing_s"], errors="coerce"),
            }
        )
        rows.append(t.dropna())
    if not rows:
        return pd.DataFrame()
    t = pd.concat(rows)
    return (
        t.groupby(["schema", "model"])["timing_s"]
        .agg(
            n="size",
            mean="mean",
            median="median",
            p90=lambda v: float(np.percentile(v, 90)),
            total_h=lambda v: v.sum() / 3600,
        )
        .reset_index()
    )


def run_gate(
    run_dirs: List[str], out_dir: str, protocol_dir: str = PROTOCOL_DIR
) -> dict:
    recs, notes = load_pilot_records(run_dirs)
    res: dict = {"notes": notes, "tables": {}, "gates": {}}
    T = res["tables"]
    if "ANS-v1" in recs:
        a = answer_table(recs["ANS-v1"])
        T["ans_by_answerer_cell_kind"] = _rates(a, ["answerer", "cell", "kind"])
        T["ans_by_answerer_kind"] = _rates(a, ["answerer", "kind"])
        per = _rates(a, ["answerer"])
        T["ans_by_answerer"] = per
        res["gates"]["invalid_lt_5pct_per_answerer"] = {
            r.answerer: {
                "invalid_rate": r.invalid_rate,
                "n": int(r.n),
                "pass": bool(r.invalid_rate < INVALID_GATE),
            }
            for r in per.itertuples()
        }
        res["gates"]["precision_ladder_answer"] = {
            r.answerer: {
                "nonfinite_rows": int(round(r.nonfinite_rate * r.n)),
                "empty_rate": r.empty_rate,
                "repeat_char20_rate": r.repeat_char20_rate,
                "escalate": bool(
                    r.nonfinite_rate > 0
                    or r.empty_rate > EMPTY_THRESHOLD
                    or r.repeat_char20_rate > REPEAT_THRESHOLD
                ),
            }
            for r in per.itertuples()
        }
    if "OBS-v1" in recs:
        o = observation_table(recs["OBS-v1"], protocol_dir)
        cells = _obs_rates(o, ["observer", "arm", "route"])
        T["obs_by_observer_arm_route"] = cells
        T["obs_by_observer_arm"] = _obs_rates(o, ["observer", "arm"])
        per = _obs_rates(o, ["observer"])
        T["obs_by_observer"] = per
        over = cells[cells["truncated_rate"] > TRUNC_THRESHOLD]
        res["gates"]["truncation_cap"] = {
            ("rule"): (
                "any observer x arm x route with truncated > 2% "
                "-> cap 900 for every observer/arm "
            ),
            "raise_cap_to_900": bool(len(over)),
            "cells_over": over[
                ["observer", "arm", "route", "n", "truncated_rate"]
            ].to_dict("records"),
        }
        res["gates"]["observation_format_ok_rate"] = {
            r.observer: r.format_ok_rate for r in per.itertuples()
        }
        res["gates"]["precision_ladder_obs"] = {
            r.observer: {
                "nonfinite_rows": int(round(r.nonfinite_rate * r.n)),
                "empty_rate": r.empty_rate,
                "repeat_char20_rate": r.repeat_char20_rate,
                "escalate": bool(
                    r.nonfinite_rate > 0
                    or r.empty_rate > EMPTY_THRESHOLD
                    or r.repeat_char20_rate > REPEAT_THRESHOLD
                ),
            }
            for r in per.itertuples()
        }
    res["gates"]["T(c)_non_empty"] = label_map_check(run_dirs)
    T["timing_per_turn"] = timing_table(recs)
    if "JUDGE-v1" in recs:
        T["judge_json_validity"] = judge_status_table(recs["JUDGE-v1"])
    if "EDIT-v1" in recs:
        e = recs["EDIT-v1"]
        e = e.assign(
            int_type=e["key"].str.split("|").str[1],
            editor=e["editor"] if "editor" in e else "?",
            status=e["status"] if "status" in e else "?",
        )
        T["edit_status"] = (
            e.groupby(["int_type", "editor", "status"]).size().rename("n").reset_index()
        )
    write(out_dir, res)
    return res


def write(out_dir: str, res: dict) -> None:
    from medvqa.analysis.run_all import _clean, to_md

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "pilot_gate.json"), "w", encoding="utf-8") as f:
        json.dump(
            _clean(
                {"gates": res["gates"], "notes": res["notes"], "tables": res["tables"]}
            ),
            f,
            ensure_ascii=False,
            indent=1,
        )
    L = [
        (
            "# Kiểm tra kỹ thuật đầu ra PILOT (không đọc đáp án tham chiếu,"
            " không tính độ chính xác)\n"
        )
    ]
    L.append(
        "## Kết quả kiểm tra\n```json\n"
        + json.dumps(_clean(res["gates"]), ensure_ascii=False, indent=1)
        + "\n```\n"
    )
    for k, t in res["tables"].items():
        L.append(f"## {k}\n\n" + to_md(t))
    with open(os.path.join(out_dir, "pilot_gate.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="validation pilot gate (no gold, no accuracy)."
    )
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = a.out_dir or os.path.join(a.run_dir[0], "_pilot_gate")
    res = run_gate(a.run_dir, out)
    print(
        json.dumps({k: v for k, v in res["gates"].items()}, default=str, indent=1)[
            :4000
        ]
    )


if __name__ == "__main__":
    main()
