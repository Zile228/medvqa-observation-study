"""Synthetic provenance regressions: TEST artifact discovery must exclude PILOT."""

import json

import pandas as pd
import pytest

from medvqa import judge_scope as AX1
from medvqa.analysis import io_adapter as IO


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def artifact(root, owner, mode, value, config=True, records=True):
    path = root / owner / "artifacts/corruption_plan.json"
    write(path, value)
    shard = path.parents[1] / "shard_00"
    if config:
        write(
            shard / "run_config.json",
            {"mode": mode, "corruption_plan_sha256": AX1.file_sha256(path)},
        )
    if records:
        write(
            shard / "records.jsonl",
            {"mode": mode, "run_id": "opaque-owner", "schema": "EDIT-v1", "key": owner},
        )
    return path


def test_pilot_difference_excluded_by_provenance_not_folder_name(tmp_path):
    artifact(tmp_path, "looks_like_test", "PILOT", {"assignments": {"pilot": 1}})
    expected = {"assignments": {"test": 2}}
    artifact(tmp_path, "looks_like_pilot", "TEST", expected)
    warnings = []
    assert IO.find_artifact(str(tmp_path), "corruption_plan.json", warnings) == expected
    assert len(warnings) == 1 and "ignored PILOT" in warnings[0]


def test_conflicting_test_copies_rejected(tmp_path):
    artifact(tmp_path, "a", "TEST", {"x": 1})
    artifact(tmp_path, "b", "TEST", {"x": 2})
    with pytest.raises(ValueError, match="conflicting copies"):
        IO.find_artifact(str(tmp_path), "corruption_plan.json", [])


@pytest.mark.parametrize("pilot_only", [False, True])
def test_missing_test_artifact_returns_none(tmp_path, pilot_only):
    if pilot_only:
        artifact(tmp_path, "arbitrary", "PILOT", {"x": 1})
    assert IO.find_artifact(str(tmp_path), "corruption_plan.json", []) is None


def test_equal_test_json_copies_keep_existing_contract(tmp_path):
    artifact(tmp_path, "a", "TEST", {"x": 1, "y": 2})
    artifact(tmp_path, "b", "TEST", {"y": 2, "x": 1})
    assert IO.find_artifact(str(tmp_path), "corruption_plan.json", []) == {
        "x": 1,
        "y": 2,
    }


@pytest.mark.parametrize("config,records", [(True, False), (False, True)])
def test_supported_single_provenance_source(tmp_path, config, records):
    artifact(tmp_path, "arbitrary", "TEST", {"x": 1}, config, records)
    assert IO.find_artifact(str(tmp_path), "corruption_plan.json", []) == {"x": 1}


@pytest.mark.parametrize(
    "kind", ["absent", "invalid_mode", "record_mode", "shard_mode", "run_id", "sha"]
)
def test_ambiguous_or_tampered_provenance_rejected(tmp_path, kind):
    p = artifact(
        tmp_path, "arbitrary", "TEST", {"x": 1}, kind != "absent", kind != "absent"
    )
    shard = p.parents[1] / "shard_00"
    if kind == "invalid_mode":
        write(shard / "run_config.json", {"mode": "test"})
    elif kind == "record_mode":
        write(shard / "records.jsonl", {"mode": "PILOT"})
    elif kind == "shard_mode":
        write(p.parents[1] / "shard_01/run_config.json", {"mode": "PILOT"})
    elif kind == "run_id":
        write(
            p.parents[1] / "shard_01/records.jsonl",
            {"mode": "TEST", "run_id": "other-owner"},
        )
    elif kind == "sha":
        write(p, {"x": 99})
    with pytest.raises(ValueError, match="provenance|SHA mismatch"):
        IO.find_artifact(str(tmp_path), "corruption_plan.json", [])


def test_load_all_routes_only_test_plan_and_keeps_reference_panel_hook(
    tmp_path, monkeypatch
):
    """Exercise the production entry point; no gold/scoring is opened."""
    artifact(tmp_path, "alpha", "PILOT", {"x": "pilot"})
    expected = {"x": "test"}
    artifact(tmp_path, "beta", "TEST", expected)
    monkeypatch.setattr(
        IO,
        "load_records",
        lambda *a: {
            "ANS-v1": pd.DataFrame(
                [
                    {
                        "cell": "CELL-Q-NONE",
                        "qid": "fake-q",
                        "text": "synthetic",
                        "status": "ok",
                    }
                ]
            )
        },
    )
    monkeypatch.setattr(IO, "load_items", lambda *a: pd.DataFrame())
    scope_marker = {"eligible_ids": ["synthetic"]}
    calls = []

    def scope(*args):
        calls.append("REFERENCE_PANEL")
        return scope_marker

    monkeypatch.setattr(IO, "load_ax1_scope", scope)
    result, warnings = IO.load_all(str(tmp_path))
    assert result["corruption_plan"] == expected
    assert result["ax1_scope"] is scope_marker and calls == ["REFERENCE_PANEL"]
    assert any(("ignored PILOT" in w for w in warnings))
