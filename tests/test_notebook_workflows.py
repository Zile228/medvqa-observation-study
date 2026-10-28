"""Run notebook controllers on synthetic inputs without model downloads or API calls."""

import json
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image

from medvqa import common, execution, interventions, models, observers, prompts, routing

SOURCE_ROOT = Path(__file__).resolve().parents[1]


class InMemoryVisionModel(models.StubVLM):
    """Substitute a CPU model to test notebook input and output handling."""

    def generate(self, image, prompt, max_new_tokens):
        assert image.mode == "RGB"
        if "Report each item below" in prompt:
            items = re.findall(r"^(\d+)\. ([^:\n]+):", prompt.split("Items:")[1], re.M)
            text = "\n".join(
                f"{number}. {name} | visible | a marked image region"
                for number, name in items
            )
        else:
            text = "synthetic structure"
        return {
            "text": text,
            "n_output_tokens": len(text.split()),
            "truncated": False,
            "finish_reason": "stop",
            "first_logits": None,
            "first_step_finite": True,
            "logits_method": "controller_test",
        }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def tiny_inputs(tmp_path):
    root = tmp_path / "inputs"
    prompt_root = root / "resources/protocol/prompts"
    shutil.copytree(SOURCE_ROOT / "resources/protocol/prompts", prompt_root)
    prep = root / "prep_out"
    image_ids = ["RT-9001", "RT-9002"]
    rows = [
        {
            "qid": f"QUESTION-{index}",
            "neutral_id": image_id,
            "cohort": "COH-PILOT",
            "question": "Which structure appears in the marked region?",
            "answer_kind": "closed_short",
            "options": [],
            "orders": [],
            "K": 1,
            "strata": {"dataset": "SLAKE"},
        }
        for index, image_id in enumerate(image_ids)
    ]
    cohort_path = prep / "cohorts/COH-PILOT.jsonl"
    cohort_path.parent.mkdir(parents=True)
    cohort_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    metadata = {
        image_id: routing.slake_record([{"modality": "CT", "location": "Abdomen"}])
        for image_id in image_ids
    }
    route_pairs = routing.build_route_pairs(
        metadata, routing.load_routes_spec(prompt_root)
    )
    route_records = {
        image_id: routing.route_for(image_id, metadata[image_id], route_pairs)
        for image_id in image_ids
    }
    bank = prompts.ObserverPromptBank(prompt_root)
    manifest = {
        f"{arm}|{image_id}": {
            "prompt_sha256": common.sha256_text(
                bank.prompt(arm, route_records[image_id]["routes"])[0]
            )
        }
        for arm in prompts.ARMS
        for image_id in image_ids
    }
    write_json(prep / "route_metadata.json", {"records": metadata})
    write_json(prep / "route_pairs.json", route_pairs)
    write_json(prep / "observer_prompt_manifest.json", {"entries": manifest})
    write_json(prep / "images_index.json", {})
    write_json(
        prep / "panel_rq2.json",
        {"neutral_ids": image_ids, "strata": {"SLAKE": image_ids}},
    )
    write_json(
        prep / "judge/annotations_slake_dev.json",
        {
            image_id: {"text": "A synthetic image region is marked.", "entries": [{}]}
            for image_id in image_ids
        },
    )
    write_json(prep / "judge/jprobe_set_dev.json", {"probes": []})
    donor_table = interventions.shuf_table(
        image_ids,
        {image_id: route_records[image_id]["route_key"] for image_id in image_ids},
        {image_id: "CT" for image_id in image_ids},
        {image_id: "COH-PILOT" for image_id in image_ids},
    )
    write_json(prep / "shuf_donors.json", {"by_mode": {"PILOT": donor_table}})
    (prep / "images").mkdir()
    for image_id in image_ids:
        Image.new("RGB", (8, 8), color=(120, 120, 120)).save(
            prep / "images" / f"{image_id}.png"
        )
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "locked_inputs.sha256").write_text(
        "".join(
            f"{common.sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def cpu_backends(monkeypatch):
    monkeypatch.setenv("MEDVQA_SOURCE_ROOT", str(SOURCE_ROOT))
    monkeypatch.setattr(models, "StubVLM", InMemoryVisionModel)

    def disallow_external_service(*args, **kwargs):
        raise AssertionError("A controller test attempted to use an external service")

    monkeypatch.setattr(models, "HFVisionBackend", disallow_external_service)
    monkeypatch.setattr(models, "HFTextBackend", disallow_external_service)
    monkeypatch.setattr(observers, "openai_sender", disallow_external_service)
    monkeypatch.setattr(common, "get_secret", disallow_external_service)


def run_notebook(filename, overrides):
    notebook = json.loads(
        (SOURCE_ROOT / "notebooks" / filename).read_text(encoding="utf-8")
    )
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{filename}:cell-{index}", "exec"), namespace)
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            namespace.update(overrides)
    return namespace


def configuration(input_root, output_root, run_id):
    return {
        "INPUT_ROOT": input_root,
        "OUT_ROOT": output_root,
        "LOCK_FILE": input_root / "locked_inputs.sha256",
        "RUN_ID": run_id,
        "MODE": "PILOT",
        "BACKEND": "stub",
        "INSTALL": False,
    }


def test_observation_and_answer_controllers_resume_without_duplicate_records(
    tiny_inputs, tmp_path, cpu_backends
):
    output = tmp_path / "output"
    options = configuration(tiny_inputs, output, "synthetic_observations")
    options.update(OBSERVER="Q", ARMS=["ARM-GRD"], MAX_ITEMS_THIS_SESSION=1)
    first = run_notebook("02_generate_observations.ipynb", options)
    assert first["rep"]["remaining_in_shard"] == 1
    options["MAX_ITEMS_THIS_SESSION"] = None
    resumed = run_notebook("02_generate_observations.ipynb", options)
    assert resumed["rep"]["done_total_in_shard"] == 2
    assert resumed["rep"]["duplicate_keys"] == []

    options = configuration(tiny_inputs, output, "synthetic_answers")
    options.update(
        ANSWERER="Q",
        CELLS=["NONE", "SELF-GRD"],
        OBS_DIRS=[output / "synthetic_observations"],
        MAX_ITEMS_THIS_SESSION=1,
    )
    first = run_notebook("04_answer_questions.ipynb", options)
    assert first["rep"]["remaining_in_shard"] == 3
    options["MAX_ITEMS_THIS_SESSION"] = None
    resumed = run_notebook("04_answer_questions.ipynb", options)
    assert resumed["rep"]["done_total_in_shard"] == 4
    assert resumed["rep"]["duplicate_keys"] == []
    answers = execution.load_records([output / "synthetic_answers"], "ANS-v1")
    assert len(answers) == 4
    assert {row["status"] for row in answers.values()} == {"ok"}
    assert {row["cell"] for row in answers.values()} == {
        "CELL-Q-NONE",
        "CELL-Q-SELF-GRD",
    }
    assert all(len(row["prompt_sha256"]) == 64 for row in answers.values())
    assert all(row["logprob"] is None for row in answers.values())


def test_external_observation_and_judge_controllers_use_only_stub_services(
    tiny_inputs, tmp_path, cpu_backends
):
    output = tmp_path / "output"
    options = configuration(tiny_inputs, output, "synthetic_external_observations")
    options.update(TASK="OBS", RETRY_WAIT_S=0)
    observed = run_notebook("03_generate_external_observations.ipynb", options)
    assert observed["rep"]["done_this_session"] == 6
    assert observed["rep"]["errors"] == 0
    options = configuration(tiny_inputs, output, "synthetic_judge")
    options.update(
        JUDGE="J1",
        TASKS=["J-AX1"],
        OBSERVERS=["OBS-G"],
        OBS_DIRS=[output / "synthetic_external_observations"],
    )
    judged = run_notebook("05_evaluate_observations.ipynb", options)
    assert judged["rep"]["remaining_in_shard"] == 0
    assert judged["rep"]["errors"] == 0
    assert judged["rep"]["done_total_in_shard"] > 0
    results = execution.load_records([output / "synthetic_judge"], "JUDGE-v1")
    assert all(row["diag_json_valid"] for row in results.values())
    assert all(row["refs"]["statements"] for row in results.values())


def test_changed_prompt_bytes_stop_execution_before_model_initialization(
    tiny_inputs, tmp_path, cpu_backends, monkeypatch
):
    frame = tiny_inputs / "resources/protocol/prompts/templates/observer_frame.txt"
    frame.write_bytes(frame.read_bytes() + b"\nchanged")

    def unexpected_model(*args, **kwargs):
        raise AssertionError("The model was initialized before input validation")

    monkeypatch.setattr(models, "StubVLM", unexpected_model)
    options = configuration(tiny_inputs, tmp_path / "output", "invalid_inputs")
    with pytest.raises(common.LockError, match="SHA mismatch"):
        run_notebook("02_generate_observations.ipynb", options)


def test_api_transport_failure_preserves_incomplete_run_for_resume(
    tiny_inputs, tmp_path, cpu_backends
):
    output = tmp_path / "output"
    options = configuration(tiny_inputs, output, "interrupted_external_observations")
    options.update(TASK="OBS", RETRY_WAIT_S=0, STUB_FAIL_FIRST=1, STUB_FAIL_STATUS=503)
    with pytest.raises(observers.SystemAPIError):
        run_notebook("03_generate_external_observations.ipynb", options)
    reports = list((output / options["RUN_ID"]).glob("shard_*/integrity.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["stopped_reason"] == "system_api_error"
    assert report["done_total_in_shard"] == 0
    assert report["remaining_in_shard"] == 6

    options["STUB_FAIL_FIRST"] = 0
    resumed = run_notebook("03_generate_external_observations.ipynb", options)
    assert resumed["rep"]["done_total_in_shard"] == 6
    assert resumed["rep"]["remaining_in_shard"] == 0
    assert resumed["rep"]["duplicate_keys"] == []
