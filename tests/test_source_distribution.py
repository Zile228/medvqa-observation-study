"""Check source configuration, notebook imports and run provenance."""

import ast
import json
import re
from pathlib import Path

import pytest

from medvqa import inference_environment, models
from medvqa.analysis.high_detail import validate_source_provenance

ROOT = Path(__file__).resolve().parents[1]


def test_no_fixed_hash_literals_or_internal_identifiers_in_source():
    for path in [*ROOT.rglob("*.py"), *ROOT.rglob("*.ipynb")]:
        if path.suffix == ".ipynb":
            data = json.loads(path.read_text(encoding="utf-8"))
            sources = [
                "".join(c["source"]) for c in data["cells"] if c["cell_type"] == "code"
            ]
            for c in data["cells"]:
                if c["cell_type"] == "code":
                    assert not c.get("outputs")
                    assert c.get("execution_count") is None
        else:
            sources = [path.read_text(encoding="utf-8")]
        for source in sources:
            assert not re.search(r"P0[49]|D(?:14|55|56)(?!\d)", source), path
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert not re.search(
                        r"(?<![a-fA-F0-9])(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})"
                        r"(?![a-fA-F0-9])",
                        node.value,
                    ), (path, node.lineno)


def test_models_are_loaded_from_public_configuration():
    expected = json.loads((ROOT / "configs/models.json").read_text())
    assert models.MODELS == expected
    assert set(expected) == {"Q", "L", "M", "J1", "J2", "PHI"}


def test_measured_provenance_accepts_only_registered_sources_and_inputs():
    registry = json.loads((ROOT / "configs/inference_provenance.json").read_text())
    measured = registry["high_detail"][0]
    validate_source_provenance(measured, {})
    validate_source_provenance(
        {"source_sha256": {"runner": "current"}}, {"runner": "current"}
    )
    for key in ("inputs_sha256", "source_sha256"):
        bad = dict(measured, **{key: "unknown"})
        with pytest.raises(ValueError, match="measured run"):
            validate_source_provenance(bad, {})


def test_environment_installer_preserves_measured_versions(monkeypatch):
    calls = []
    monkeypatch.setattr(
        inference_environment.metadata, "version", lambda name: "different"
    )
    monkeypatch.setattr(
        inference_environment.subprocess,
        "run",
        lambda args, check: calls.append(args),
    )
    data = {
        "baselines": {
            "Q": {"package_pins": {"torch": "2.11.0+cu130", "transformers": "4.49.0"}}
        }
    }
    inference_environment.install_measured_environment(data, "Q")
    assert "torch==2.11.0+cu130" in calls[0]
    assert calls[0][-1].endswith("/cu130")
    assert "transformers==4.49.0" in calls[1]


def test_extra_notebooks_import_package_without_embedded_module_loaders():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb"))[7:]:
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        code = "\n".join(
            "".join(c["source"]) for c in cells if c["cell_type"] == "code"
        )
        assert "from medvqa" in code
        assert "EXECUTE = False" in code
        for forbidden in ("files.upload", "spec_from_file_location", "%%writefile"):
            assert forbidden not in code
