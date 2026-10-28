"""Reference-panel manifests remain strict across archive filename changes."""

import hashlib

import pytest

from medvqa.analysis.io_adapter import _verify_reference_manifest
from medvqa.common import LockError


def panel_files(tmp_path):
    names = ["scope.json", "judge_plan.json", "references.json"]
    files = {}
    for name in names:
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        files[f"artifacts/{name}"] = path
    return files


def manifest(tmp_path, filename, files, corrupt=False):
    destination = tmp_path / "locks" / filename
    destination.parent.mkdir(exist_ok=True)
    lines = []
    for logical, path in files.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{('0' * 64 if corrupt else digest)}  {logical}\n")
    destination.write_text("".join(lines), encoding="utf-8")
    return destination


def test_manifest_filename_is_not_an_experiment_identity(tmp_path):
    files = panel_files(tmp_path)
    manifest(tmp_path, "locked_inputs_archived_panel.sha256", files)
    _verify_reference_manifest(str(tmp_path), files)


def test_partial_manifest_cannot_authorize_panel(tmp_path):
    files = panel_files(tmp_path)
    manifest(tmp_path, "locked_inputs_panel.sha256", dict(list(files.items())[:1]))
    with pytest.raises(LockError, match="complete reference panel"):
        _verify_reference_manifest(str(tmp_path), files)


def test_changed_input_is_rejected(tmp_path):
    files = panel_files(tmp_path)
    manifest(tmp_path, "locked_inputs_panel.sha256", files)
    next(iter(files.values())).write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(LockError, match="SHA mismatch"):
        _verify_reference_manifest(str(tmp_path), files)


def test_conflicting_manifest_is_not_silently_ignored(tmp_path):
    files = panel_files(tmp_path)
    manifest(tmp_path, "locked_inputs_current.sha256", files)
    manifest(tmp_path, "locked_inputs_archived.sha256", files, corrupt=True)
    with pytest.raises(LockError):
        _verify_reference_manifest(str(tmp_path), files)
