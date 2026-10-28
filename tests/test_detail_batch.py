"""Offline high-detail generation, budget, resume and interruption checks."""

import json
from types import SimpleNamespace

import pytest

from medvqa import detail_batch as batch
from medvqa import detail_probe as probe
from tests.test_detail_probe import inventory as probe_inventory


def fixture(tmp_path):
    inv = probe_inventory()
    inv["dashboard_reconciled"] = True
    inv["batch_approved"] = True
    images = tmp_path / "images"
    images.mkdir()
    for row in inv["records"]:
        (images / f"{row['neutral_id']}.png").write_bytes(b"synthetic-image-bytes")
        row["observation_metadata"] = {
            "routes": ["SYNTHETIC"],
            "route_key": "SYNTHETIC",
            "route_gaps": [],
            "cohort": "TEST",
            "prompt_source": "synthetic",
        }
    response = {
        "model": probe.MODEL,
        "usage": {"prompt_tokens": 8501, "completion_tokens": 10},
        "choices": [
            {"finish_reason": "stop", "message": {"content": "1. Synthetic | visible"}}
        ],
    }
    inv["probe_response"] = response
    inv["probe_request_sha256"] = probe.digest(
        json.dumps(
            batch.payload(inv, inv["records"][0], b"synthetic-image-bytes"),
            sort_keys=True,
        ).encode()
    )
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inv), encoding="utf-8")
    return inv, path, images, response


def sender(response, *, fail=False):
    state = {"calls": 0, "settings": None}

    def create(**kwargs):
        state["calls"] += 1
        assert kwargs["messages"][0]["content"][0]["image_url"]["detail"] == "high"
        if fail:
            raise TimeoutError("DO_NOT_PERSIST_PRIVATE_ERROR")
        return SimpleNamespace(model_dump=lambda: response)

    def factory(**kwargs):
        state["settings"] = kwargs
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    return factory, state


def execute(path, images, output, factory):
    return batch.run(
        path,
        images,
        output,
        11.75,
        "synthetic-key",
        counter=lambda text: 1,
        client_factory=factory,
        sleep=lambda seconds: None,
        progress=lambda text: None,
    )


def test_full_run_reuses_trial_and_resume_makes_no_calls(tmp_path):
    inv, path, images, response = fixture(tmp_path)
    factory, state = sender(response)
    output = tmp_path / "run"
    result = execute(path, images, output, factory)
    assert result["completed"] == 599
    assert state["calls"] == 598
    assert state["settings"]["max_retries"] == 0
    original = (output / "shard_00/records.jsonl").read_bytes()
    assert execute(path, images, output, factory) == result
    assert state["calls"] == 598
    assert original == (output / "shard_00/records.jsonl").read_bytes()
    assert b"synthetic-key" not in (output / "run_config.json").read_bytes()


def test_missing_response_never_automatically_repeats(tmp_path):
    inv, path, images, response = fixture(tmp_path)
    factory, state = sender(response, fail=True)
    output = tmp_path / "run"
    with pytest.raises(batch.ReviewRequired, match="API failed"):
        execute(path, images, output, factory)
    assert state["calls"] == 1
    with pytest.raises(batch.ReviewRequired, match="without response"):
        execute(path, images, output, factory)
    assert state["calls"] == 1
    assert all(
        "DO_NOT_PERSIST_PRIVATE_ERROR" not in p.read_text()
        for p in output.rglob("*.json")
    )


def test_all_images_are_checked_before_network(tmp_path):
    inv, path, images, response = fixture(tmp_path)
    (images / f"{inv['records'][-1]['neutral_id']}.png").write_bytes(b"changed")
    factory, state = sender(response)
    with pytest.raises(ValueError, match="image differs"):
        execute(path, images, tmp_path / "run", factory)
    assert state["calls"] == 0


@pytest.mark.parametrize(
    "change", ["dashboard", "approval", "model", "usage", "calibration", "tokens"]
)
def test_gates_fail_closed(tmp_path, change):
    inv, path, images, response = fixture(tmp_path)
    if change == "dashboard":
        inv["dashboard_reconciled"] = False
    elif change == "approval":
        inv["batch_approved"] = False
    elif change == "model":
        inv["probe_response"]["model"] = "another-model"
    elif change == "usage":
        inv["probe_response"]["usage"].pop("prompt_tokens")
    elif change == "tokens":
        inv["probe_response"]["usage"]["prompt_tokens"] = 100000
    else:
        inv["probe_request_sha256"] = "0" * 64
    path.write_text(json.dumps(inv), encoding="utf-8")
    factory, state = sender(response)
    with pytest.raises(batch.ReviewRequired):
        execute(path, images, tmp_path / "run", factory)
    assert state["calls"] == 0


def test_completed_response_tampering_is_rejected(tmp_path):
    inv, path, images, response = fixture(tmp_path)
    factory, state = sender(response)
    output = tmp_path / "run"
    execute(path, images, output, factory)
    saved = next((output / "calls").rglob("response.json"))
    saved.write_text("{}", encoding="utf-8")
    with pytest.raises(batch.ReviewRequired, match="checksum"):
        execute(path, images, output, factory)
    assert state["calls"] == 598


def test_budget_rejects_before_network(tmp_path):
    inv, path, images, response = fixture(tmp_path)
    factory, state = sender(response)
    with pytest.raises(batch.ReviewRequired, match="budget"):
        batch.run(
            path,
            images,
            tmp_path / "run",
            0.1,
            "synthetic-key",
            counter=lambda text: 1,
            client_factory=factory,
        )
    assert state["calls"] == 0


def test_content_failures_retry_three_times_then_stop(tmp_path):
    inv, path, images, response = fixture(tmp_path)
    invalid = {**response, "choices": [{"message": {"content": ""}}]}
    factory, state = sender(invalid)
    output = tmp_path / "run"
    with pytest.raises(batch.ReviewRequired, match="Three content failures"):
        execute(path, images, output, factory)
    assert state["calls"] == 3
    with pytest.raises(batch.ReviewRequired, match="Three content failures"):
        execute(path, images, output, factory)
    assert state["calls"] == 3
