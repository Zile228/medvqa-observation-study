"""Check pilot outputs, support-analysis integrity and judge data formats."""

import inspect
import json
import os

import pandas as pd
import pytest

from medvqa.analysis import constants as K
from medvqa.analysis import io_adapter as IO
from medvqa.analysis import pilot_gate as PG
from medvqa.analysis import rq2 as RQ2
from medvqa.analysis.metrics import (
    _judge_payload,
    judge_status_table,
    open_judge,
    parse_judge_json,
)
from medvqa.analysis.scoring import prepare_items, score_rows


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _pilot_run(tmp_path, empty_label=False):
    run = tmp_path / "pilot_T0"
    ans = []
    texts_L = ["Yes", "No", "Yes no", "B is correct", "A"]
    kinds = ["yesno", "yesno", "yesno", "mcq", "mcq"]
    for i, (t, k) in enumerate(zip(texts_L, kinds)):
        ans.append(
            {
                "schema": "ANS-v1",
                "key": f"ANS|CELL-L-NONE|q{i}|-",
                "mode": "PILOT",
                "cell": "CELL-L-NONE",
                "answerer": "ANS-L",
                "answer_kind": k,
                "status": "ok",
                "text": t,
                "timing_s": 1.0 + i,
                "options_presented": [
                    {"label": "A", "text": "CT"},
                    {"label": "B", "text": "MRI"},
                ]
                if k == "mcq"
                else None,
                "logprob": {"lp": {"Yes": -0.1, "No": -2}, "finite": True},
            }
        )
    for i in range(40):
        ans.append(
            {
                "schema": "ANS-v1",
                "key": f"ANS|CELL-Q-NONE|q{i}|-",
                "mode": "PILOT",
                "cell": "CELL-Q-NONE",
                "answerer": "ANS-Q",
                "answer_kind": "yesno",
                "status": "ok",
                "text": "Yes",
                "timing_s": 0.5,
                "logprob": {"lp": {"Yes": -0.1, "No": -2}, "finite": True},
            }
        )
    ans.append(
        {
            "schema": "ANS-v1",
            "key": "ANS|CELL-Q-INT-PARA|qx|-",
            "mode": "PILOT",
            "cell": "CELL-Q-INT-PARA",
            "answer_kind": "yesno",
            "status": "skipped_int_missing",
            "text": None,
        }
    )
    ans.append(
        {
            "schema": "ANS-v1",
            "key": "ANS|CELL-Q-NONE|t|-",
            "mode": "TEST",
            "cell": "CELL-Q-NONE",
            "answer_kind": "yesno",
            "status": "ok",
            "text": "garbage",
        }
    )
    obs = []
    for i in range(3):
        obs.append(
            {
                "schema": "OBS-v1",
                "key": f"OBS|OBS-L|ARM-GEN|RD-{i}",
                "mode": "PILOT",
                "observer": "OBS-L",
                "arm": "ARM-GEN",
                "route_key": "RT-FALLBACK",
                "status": "ok",
                "truncated": i == 0,
                "first_step_finite": True,
                "text": "1. a | visible | x",
                "n_output_tokens": 100,
                "timing_s": 9.0,
            }
        )
    _write(str(run / "shard_00_s1" / "records.jsonl"), ans + obs)
    labels = {
        "Yes": [1],
        "No": [2],
        "A": [3],
        "B": [4],
        "C": [5],
        "D": [] if empty_label else [6],
    }
    os.makedirs(run / "artifacts", exist_ok=True)
    json.dump(
        {"llava": {"labels": labels}},
        open(run / "artifacts" / "logprob_label_map_llava.json", "w"),
    )
    return str(run)


def test_pilot_gate(tmp_path):
    run = _pilot_run(tmp_path)
    res = PG.run_gate([run], str(tmp_path / "out"))
    g = res["gates"]
    assert g["invalid_lt_5pct_per_answerer"]["L"]["invalid_rate"] == pytest.approx(
        2 / 5
    )
    assert not g["invalid_lt_5pct_per_answerer"]["L"]["pass"]
    assert g["invalid_lt_5pct_per_answerer"]["Q"]["pass"]
    assert g["invalid_lt_5pct_per_answerer"]["Q"]["n"] == 40
    t = res["tables"]["ans_by_answerer_cell_kind"]
    assert t.loc[
        (t["answerer"] == "L") & (t["kind"] == "mcq"), "b_is_correct_rate"
    ].iloc[0] == pytest.approx(0.5)
    assert g["truncation_cap"]["raise_cap_to_900"]
    assert g["truncation_cap"]["cells_over"][0]["n"] == 3
    assert g["T(c)_non_empty"]["status"] == "PASS"
    assert os.path.exists(tmp_path / "out" / "pilot_gate.md")
    assert (
        "timing_per_turn" in res["tables"]
        and len(res["tables"]["timing_per_turn"]) >= 2
    )


def test_pilot_gate_empty_label(tmp_path):
    run = _pilot_run(tmp_path, empty_label=True)
    res = PG.run_gate([run], str(tmp_path / "out"))
    assert res["gates"]["T(c)_non_empty"]["status"] == "FAIL"
    assert "llava:D" in res["gates"]["T(c)_non_empty"]["empty_labels"]


def test_pilot_gate_never_reads_gold(tmp_path):
    src = inspect.getsource(PG)
    assert (
        "P.is_correct(" not in src
        and "gold_answer" not in src
        and ("load_items" not in src)
    )
    assert "correct_lead" not in src and "accuracy" not in src.split('"""')[2]
    with pytest.raises(PermissionError):
        PG._guard(str(tmp_path / "prep" / "prep_sealed" / "gold_RAD_TEST.jsonl"))


def _ax1(layout, judge="J1", status="ok"):
    ok = "OBS|OBS-L|ARM-GRD|RT-0001"
    if layout == "schema":
        refs = {
            "statements": [
                {"id": 1, "sid": f"{ok}#0", "obs_key": ok, "is_probe": False},
                {"id": 2, "sid": f"{ok}#1", "obs_key": ok, "is_probe": False},
            ],
            "skipped_evasion": ["x"],
        }
    else:
        refs = {
            "obs_key": ok,
            "is_probe_block": False,
            "probe_ids": None,
            "evasion": 1,
            "statements": [
                {"id": 1, "sid": f"{ok}#0", "text": "a"},
                {"id": 2, "sid": f"{ok}#1", "text": "b"},
            ],
        }
    return {
        "judge": judge,
        "task": "J-AX1",
        "variant": "main",
        "status": status,
        "refs": refs,
        "text": json.dumps(
            {
                "labels": [
                    {"id": 1, "label": "supported"},
                    {"id": 2, "label": "contradicted"},
                ]
            }
        ),
    }


def _probe(judge="J1"):
    return {
        "judge": judge,
        "task": "J-AX1",
        "variant": "main",
        "status": "ok",
        "refs": {
            "obs_key": None,
            "is_probe_block": True,
            "probe_ids": ["PROBE-RT-0001-0"],
            "statements": [{"id": 1, "sid": "PROBE-RT-0001-0", "text": "s"}],
        },
        "text": json.dumps({"labels": [{"id": 1, "label": "supported"}]}),
    }


@pytest.mark.parametrize("layout", ["schema", "record"])
def test_ax1_both_layouts(layout):
    s = RQ2.ax1_statements(pd.DataFrame([_ax1(layout)]))
    assert list(s["label"]) == ["supported", "contradicted"]
    assert (
        set(s["observer"]) == {"L"}
        and set(s["arm"]) == {"GRD"}
        and (set(s["image_id"]) == {"RT-0001"})
    )
    ev = RQ2.evasion_rate(pd.DataFrame([_ax1(layout)]))
    assert ev["n_skipped"].iloc[0] == 1 and ev["n_judged"].iloc[0] == 2
    p = RQ2.ax1_statements(pd.DataFrame([_probe()]))
    assert bool(p["is_probe"].iloc[0]) and p["probe_id"].iloc[0] == "PROBE-RT-0001-0"


def test_rq2_integrity_fails_loudly_when_tables_empty():
    judge = pd.DataFrame([_ax1("record", "J1"), _ax1("record", "J2")])
    chk = RQ2.rq2_integrity(
        judge,
        {"tables": {}, "probe_rule_pass": None},
        pd.DataFrame(),
        has_probe_set=False,
    )
    assert chk["status"] == "FAIL"
    joined = " ".join(chk["problems"])
    assert "E-R2a" in joined and "probe" in joined.lower() and ("C17" in joined)
    assert RQ2.rq2_integrity(pd.DataFrame(), {}, None, False)["status"] == "NOT_RUN"


def test_rq2_integrity_pass_on_complete_tables():
    judge = pd.DataFrame(
        [_ax1("schema", "J1"), _ax1("schema", "J2"), _probe("J1"), _probe("J2")]
    )
    one = pd.DataFrame([{"x": 1}])
    res = {
        "tables": {"E-R2a": one, "E-R2b": one, "E-R2-evasion": one, "J-PROBE": one},
        "probe_rule_pass": True,
    }
    chk = RQ2.rq2_integrity(
        judge, res, pd.DataFrame([{"contrast": "C17-L"}]), has_probe_set=True
    )
    assert chk["status"] == "PASS", chk


def test_run_rq2_end_to_end_record_layout():
    judge = pd.DataFrame(
        [_ax1("record", "J1"), _ax1("record", "J2"), _probe("J1"), _probe("J2")]
    )
    probe_set = {
        "probes": [
            {
                "probe_id": "PROBE-RT-0001-0",
                "neutral_id": "RT-0001",
                "template": "A",
                "expected_label": "supported",
            }
        ]
    }
    res = RQ2.run_rq2(
        judge, pd.DataFrame(), probe_set, {"RT-0001": "SLAKE"}, lambda img: 5
    )
    assert len(res["tables"]["E-R2a"]) > 0 and len(res["tables"]["J-PROBE"]) > 0
    assert len(res["tables"]["E-R2-evasion"]) == 1


def test_skip_status_is_not_judge_invalid():
    j = pd.DataFrame(
        [
            {
                "judge": "J1",
                "task": "J-AX2",
                "status": "obs_error_skipped",
                "refs": {"obs_key": "OBS|OBS-L|ARM-GRD|RT-1"},
                "text": None,
            },
            {
                "judge": "J1",
                "task": "J-OPEN",
                "status": "empty_prediction",
                "refs": {"answer_key": "ANS|CELL-L-NONE|RT-1-Q01|-", "qid": "RT-1-Q01"},
                "text": None,
            },
            {
                "judge": "J1",
                "task": "J-OPEN",
                "status": "ok",
                "refs": {"answer_key": "ANS|CELL-L-NONE|RT-1-Q02|-", "qid": "RT-1-Q02"},
                "text": '```json\n{"verdict": "correct"}\n```',
            },
        ]
    )
    assert RQ2.ax2_table(j, lambda i: 5).empty
    o = open_judge(j, ["L-NONE"])
    row = o[o["judge"] == "J1"].iloc[0]
    assert row["n"] == 2 and row["invalid_rate"] == pytest.approx(0.0)
    st = judge_status_table(j)
    jo = st[st["task"] == "J-OPEN"].iloc[0]
    assert (
        jo["n_skipped"] == 1
        and jo["n_invalid"] == 0
        and (jo["n_fence_would_parse"] == 0)
    )


def test_judge_roles():
    assert IO.judge_roles({}) == {"J1": "J1", "J2": "J2"}
    assert IO.judge_roles({"judge_fallback": "J2"}) == {"J1": "J1", "J-ALT": "J2"}
    assert IO.judge_roles({"judge_fallback": {"replaces": "J1"}}) == {
        "J2": "J2",
        "J-ALT": "J1",
    }


def test_optional_cells_do_not_fail_integrity():
    items = pd.DataFrame(
        [
            {
                "qid": "RT-1-Q01",
                "image_id": "RT-1",
                "cohort": "RAD",
                "dataset": "SLAKE",
                "group": None,
                "cluster_id": "c",
                "answer_type": "CLOSED",
                "question": "Is it?",
                "options": None,
                "gold": "yes",
                "gold_index": None,
                "qtype_meta": "Organ",
                "routes": ["RT-CT-CHEST"],
                "weak_group": False,
                "image_broken": False,
            },
            {
                "qid": "RT-1-Q02",
                "image_id": "RT-1",
                "cohort": "RAD",
                "dataset": "SLAKE",
                "group": None,
                "cluster_id": "c",
                "answer_type": "CLOSED",
                "question": "Is it?",
                "options": None,
                "gold": "no",
                "gold_index": None,
                "qtype_meta": "Organ",
                "routes": ["RT-CT-CHEST"],
                "weak_group": False,
                "image_broken": False,
            },
        ]
    )
    it, _ = prepare_items(items, {})
    a = pd.DataFrame(
        [
            {
                "cell": c,
                "qid": q,
                "order": 0,
                "raw_output": "Yes",
                "gen_error": False,
                "int_missing": False,
                "label_logprobs": None,
            }
            for c in ("L-NONE",)
            for q in ("RT-1-Q01", "RT-1-Q02")
        ]
        + [
            {
                "cell": "L-G-GRD",
                "qid": "RT-1-Q01",
                "order": 0,
                "raw_output": "Yes",
                "gen_error": False,
                "int_missing": False,
                "label_logprobs": None,
            }
        ]
    )
    rows, integ = score_rows(it, a, ["L-G-GRD", "L-NONE"], required=["L-NONE"])
    assert integ["missing_rows"] == 0 and integ["optional_cells_partial"] == {
        "L-G-GRD": "1/2"
    }
    assert (rows["cell"] == "L-G-GRD").sum() == 1


FENCE_CASES = [
    ('{"verdict": "correct"}', True, True),
    ('  {"verdict": "incorrect"}\n', True, True),
    ('```json\n{"verdict": "correct"}\n```', False, True),
    ('```\n{"verdict": "correct"}\n```', False, True),
    ('```json\r\n{"verdict": "correct"}\r\n```', False, True),
    ('```json {"verdict": "correct"} ```', False, False),
    ('```python\n{"verdict": "correct"}\n```', False, False),
    ('Here: ```json\n{"verdict": "correct"}\n```', False, False),
    ('```json\n```json\n{"verdict": "correct"}\n```\n```', False, False),
    ('{"verdict": "maybe"}', False, False),
    ('{"Verdict": "correct"}', False, False),
]


@pytest.mark.parametrize("text,strict,fence", FENCE_CASES)
def test_single_fence_rule(text, strict, fence):
    assert (_judge_payload("J-OPEN", text, None, False) is not None) == strict
    assert (_judge_payload("J-OPEN", text, None, True) is not None) == fence
    assert (parse_judge_json("J-OPEN", text) is not None) == fence


def test_fence_constant_default_and_single_parser():
    assert K.JUDGE_STRIP_ONE_JSON_FENCE is True
    assert parse_judge_json("J-OPEN", '```json\n{"verdict": "correct"}\n```') == {
        "verdict": "correct"
    }
    import inspect

    from medvqa.analysis import metrics, pilot_gate, rq2

    for mod in (rq2, pilot_gate):
        src = inspect.getsource(mod)
        assert "json.loads(" not in src.replace("IO.read_json", "")
    assert "JUDGE_JSON_FENCE_REGEX" in inspect.getsource(metrics)


def test_schema_checked_in_status_table():
    ok = json.dumps({"labels": [{"id": 1, "label": "supported"}]})
    bad_schema = json.dumps({"labels": [{"id": 1, "label": "true"}]})
    wrong_ids = json.dumps({"labels": [{"id": 2, "label": "supported"}]})
    refs = {
        "statements": [{"id": 1, "sid": "x#0", "obs_key": "OBS|OBS-L|ARM-GRD|RT-1"}]
    }
    j = pd.DataFrame(
        [
            {"judge": "J1", "task": "J-AX1", "status": "ok", "refs": refs, "text": t}
            for t in (ok, bad_schema, wrong_ids)
        ]
        + [
            {
                "judge": "J1",
                "task": "J-AX2",
                "status": "ok",
                "refs": {},
                "text": json.dumps(
                    {
                        "items_addressed": True,
                        "diagnosis": "no",
                        "unsupported_claim": "no",
                    }
                ),
            }
        ]
    )
    st = judge_status_table(j).set_index("task")
    assert st.loc["J-AX1", "n_valid_schema"] == 1 and st.loc["J-AX1", "n_invalid"] == 2
    assert st.loc["J-AX2", "n_valid_schema"] == 0
    s = RQ2.ax1_statements(j[j["task"] == "J-AX1"])
    assert s["label"].notna().sum() == 1


def test_c17_not_estimable_is_explicit():
    items = pd.DataFrame(
        [
            {
                "qid": "RT-1-Q01",
                "image_id": "RT-1",
                "dataset": "SLAKE",
                "kind": "yesno",
                "cluster_id": "c1",
            }
        ]
    ).set_index("qid", drop=False)
    rows = pd.DataFrame(
        [
            {"cell": c, "qid": "RT-1-Q01", "image_id": "RT-1", "correct_lead": 1.0}
            for c in ("L-NONE", "L-SELF-GRD", "L-G-GRD")
        ]
    )
    sup = pd.DataFrame(
        [
            {"observer": o, "arm": "GRD", "image_id": "RT-1", "support_mean": 0.0}
            for o in ("L", "G")
        ]
    )
    out = RQ2.c17(rows, items, sup, ["L"], {"L": "G"}, None)
    assert len(out) == 1 and "not estimable" in out["note"].iloc[0]
    judge = pd.DataFrame([_ax1("schema", "J1"), _ax1("schema", "J2")])
    chk = RQ2.rq2_integrity(
        judge, {"tables": {}, "probe_rule_pass": None}, out, has_probe_set=False
    )
    assert any(("C17-L not estimable" in p for p in chk["problems"]))
