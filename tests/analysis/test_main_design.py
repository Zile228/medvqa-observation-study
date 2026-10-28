"""Check study metadata, condition coverage, test families and judge outcomes."""

import json
import os

import pandas as pd
import pytest

from medvqa.analysis import io_adapter as IO
from medvqa.analysis import rq2 as RQ2
from medvqa.analysis.constants import NOT_RUN_MAIN
from medvqa.analysis.contrasts import (
    confirmatory_specs,
    descriptive_specs,
    required_cells,
)

MAIN_STATE = {
    "branch": "MAIN",
    "fb": True,
    "u3": "A",
    "u4": "A",
    "rq2": "AX1_SLAKE",
    "configuration_ref": "MAIN",
}
MAIN_CELLS = sorted(
    [
        f"{k}-{x}"
        for k in "LQ"
        for x in ("NONE", "SELF-GEN", "SELF-ANA", "SELF-GRD", "G-GEN", "G-ANA", "G-GRD")
    ]
    + [f"Q-INT-{x}" for x in ("SHUF", "CORR", "EMPTY", "PARA", "BLANK")]
)


def test_parse_design_state_main():
    d = IO.parse_design_state(MAIN_STATE)
    assert d["design"] == "MAIN" and d["rq2"] == "AX1_SLAKE"
    assert IO.parse_design_state(dict(MAIN_STATE, rq2="NONE"))["rq2"] == "NONE"
    for bad in (
        dict(MAIN_STATE, u3="B"),
        dict(MAIN_STATE, u4="B"),
        dict(MAIN_STATE, fb=False),
        dict(MAIN_STATE, rq2="FULL"),
        dict(MAIN_STATE, branch=""),
        {"design": "MAIN"},
    ):
        with pytest.raises(ValueError):
            IO.parse_design_state(bad)
    assert IO.parse_design_state({"design": "FULL"})["design"] == "FULL"


def test_archived_design_is_identified_by_experiment_traits():
    metadata = dict(MAIN_STATE, branch="archived-study")
    metadata.pop("configuration_ref")
    result = IO.parse_design_state(metadata)
    assert result["design"] == "MAIN"
    assert result["branch"] == "MAIN"
    assert result["configuration_ref"] == "MAIN"
    assert result["rq2"] == "AX1_SLAKE"


def test_main_required_cells():
    assert required_cells("MAIN") == MAIN_CELLS
    assert len(MAIN_CELLS) == 19


def test_main_families():
    specs = confirmatory_specs("MAIN")
    fam = {}
    for s in specs:
        fam[s.family] = fam.get(s.family, 0) + 1
    assert fam == {"F1-RAD": 6, "F1-NAT": 6, "F3-RAD": 3, "F3-NAT": 3, "F-INT": 7}
    ids = {s.id for s in specs}
    assert not {"C4", "C4'", "C18", "C19", "C20", "C21"} & ids
    assert {
        "C1-L",
        "C1-Q",
        "C2-L",
        "C2-Q",
        "C3-L",
        "C3-Q",
        "C7",
        "C8",
        "C9",
        "C10",
        "C16",
    } <= ids
    c7 = [s for s in specs if s.id == "C7" and s.cohort == "RAD"][0]
    assert sorted((c for _, c in c7.terms)) == sorted(
        (f"{k}-{o}-{a}" for k in "LQ" for o in ("SELF", "G") for a in ("GRD", "ANA"))
    )
    assert all((abs(abs(w) - 0.25) < 1e-12 for w, _ in c7.terms))
    desc = {s.id for s in descriptive_specs("MAIN")}
    assert not any((i.startswith(("C22", "C23", "C24", "C5", "C6")) for i in desc))
    for s in specs + descriptive_specs("MAIN"):
        cells = {t[1] for t in s.terms} | {t[2] for t in s.terms if len(t) > 2}
        assert cells <= set(MAIN_CELLS), (s.id, cells - set(MAIN_CELLS))


def _runs_dir(tmp_path, cells):
    rd = tmp_path / "runs"
    os.makedirs(rd / "ans" / "shard_00_s1", exist_ok=True)
    with open(rd / "ans" / "shard_00_s1" / "records.jsonl", "w") as f:
        for i, c in enumerate(cells):
            f.write(
                json.dumps(
                    {
                        "schema": "ANS-v1",
                        "key": f"ANS|CELL-{c}|q|-",
                        "mode": "TEST",
                        "cell": f"CELL-{c}",
                        "qid": "RT-0001-Q01",
                        "order": "-",
                        "status": "ok",
                        "text": "Yes",
                    }
                )
                + "\n"
            )
    json.dump(MAIN_STATE, open(rd / "design_state.json", "w"))
    return str(rd)


def test_missing_main_cell_stops_before_gold(tmp_path):
    rd = _runs_dir(tmp_path, [c for c in MAIN_CELLS if c != "Q-INT-BLANK"])
    with pytest.raises(SystemExit) as e:
        IO.load_all(rd, required_cells)
    assert "Q-INT-BLANK" in str(e.value)


def test_complete_main_cells_pass_cell_check(tmp_path):
    rd = _runs_dir(tmp_path, MAIN_CELLS)
    with pytest.raises(FileNotFoundError):
        IO.load_all(rd, required_cells)


def _ax1(judge, probe=False):
    if probe:
        refs = {
            "statements": [
                {
                    "id": 1,
                    "sid": "PROBE-RT-0001-0",
                    "is_probe": True,
                    "probe_id": "PROBE-RT-0001-0",
                }
            ]
        }
    else:
        ok = "OBS|OBS-L|ARM-GRD|RT-0001"
        refs = {
            "statements": [
                {"id": 1, "sid": f"{ok}#0", "obs_key": ok, "is_probe": False},
                {"id": 2, "sid": f"{ok}#1", "obs_key": ok, "is_probe": False},
            ],
            "skipped_evasion": [],
        }
    labels = [{"id": 1, "label": "supported"}] + (
        [] if probe else [{"id": 2, "label": "contradicted"}]
    )
    return {
        "judge": judge,
        "task": "J-AX1",
        "variant": "main",
        "status": "ok",
        "refs": refs,
        "text": json.dumps({"labels": labels}),
    }


def test_rq2_none_never_fails():
    res = RQ2.run_rq2(
        pd.DataFrame(), pd.DataFrame(), None, {}, lambda i: 5, level="NONE"
    )
    assert set(res["tables"]) == set(RQ2.RQ2_TABLES)
    assert all((t.iloc[0]["status"] == NOT_RUN_MAIN for t in res["tables"].values()))
    for judge in (pd.DataFrame(), pd.DataFrame([_ax1("J1")])):
        chk = RQ2.rq2_integrity(judge, res, RQ2.not_run_table(), False, level="NONE")
        assert chk == {"status": "NOT_RUN_MAIN", "problems": []}


def test_rq2_ax1_slake_only_ax1_required():
    judge = pd.DataFrame(
        [
            _ax1("J1"),
            _ax1("J2"),
            _ax1("J1", True),
            _ax1("J2", True),
            {
                "judge": "J1",
                "task": "J-AX2",
                "status": "ok",
                "refs": {"obs_key": "OBS|OBS-L|ARM-GRD|RT-0001"},
                "text": json.dumps(
                    {"items_addressed": 3, "diagnosis": "no", "unsupported_claim": "no"}
                ),
            },
        ]
    )
    probes = {
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
        judge,
        pd.DataFrame(),
        probes,
        {"RT-0001": "SLAKE"},
        lambda i: 5,
        level="AX1_SLAKE",
    )
    for t in ("E-R2c", "E-R2d-diagnosis", "E-R2e"):
        assert res["tables"][t].iloc[0]["status"] == NOT_RUN_MAIN
    assert len(res["tables"]["E-R2a"]) > 0 and len(res["tables"]["J-PROBE"]) > 0
    assert "J-AX1" in res["agreement"] and "gwet_ac1" in res["agreement"]["J-AX1"]
    chk = RQ2.rq2_integrity(judge, res, pd.DataFrame(), True, level="AX1_SLAKE")
    assert chk["status"] == "PASS", chk
    bad = dict(res, tables=dict(res["tables"], **{"E-R2a": pd.DataFrame()}))
    assert (
        RQ2.rq2_integrity(judge, bad, pd.DataFrame(), True, level="AX1_SLAKE")["status"]
        == "FAIL"
    )
