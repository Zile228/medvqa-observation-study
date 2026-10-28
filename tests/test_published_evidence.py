"""Check that published aggregate tables are intact and self-contained."""

import json
import shutil
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("verify_evidence", ROOT / "scripts/verify_evidence.py")
VERIFIER = module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def test_published_tables_are_intact():
    files = json.loads((ROOT / "evidence/manifest.json").read_text())["files"]
    assert VERIFIER.verify(ROOT) == len(files)
    assert {
        "main_routing_sensitivity.csv",
        "judge_calibration.csv",
        "answer_execution_counts.csv",
        "fallback_answer_changes.csv",
    }.issubset(files)
    assert files["supplementary_comparisons.csv"]["rows"] == 76
    assert files["neutral_list_contrasts.csv"]["rows"] == 14
    assert files["high_detail_contrasts.csv"]["rows"] == 12
    assert files["high_detail_endpoints.csv"]["rows"] == 16


def test_changed_result_is_rejected(tmp_path):
    shutil.copytree(ROOT / "evidence", tmp_path / "evidence")
    with (tmp_path / "evidence/high_detail_contrasts.csv").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        VERIFIER.verify(tmp_path)


def test_missing_result_is_rejected(tmp_path):
    shutil.copytree(ROOT / "evidence", tmp_path / "evidence")
    (tmp_path / "evidence/high_detail_endpoints.csv").unlink()
    with pytest.raises(ValueError, match="table list"):
        VERIFIER.verify(tmp_path)
