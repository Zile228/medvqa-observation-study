"""Offline request-count, duplicate-prevention and image-budget checks."""

import base64
from types import SimpleNamespace

import pytest

from medvqa.detail_probe import (
    digest,
    estimate,
    existing_probes,
    image_tokens,
    run_probe,
)


def inventory():
    prompt = "A fixed observation prompt."
    prompt_hash = digest(prompt.encode())
    png = b"synthetic-image-bytes"
    return {
        "payload_config": {
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0,
            "seed": 20261016,
            "max_tokens": 900,
        },
        "probe_neutral_id": "IMAGE-0000",
        "prompts": {prompt_hash: prompt},
        "probe_png_base64": base64.b64encode(png).decode(),
        "records": [
            {
                "neutral_id": f"IMAGE-{i:04d}",
                "width": 512,
                "height": 512,
                "prompt_sha256": prompt_hash,
                "image_sha256": digest(png),
                "original_completion_tokens": 100,
            }
            for i in range(599)
        ],
    }


def test_image_token_rules():
    assert image_tokens(512, 512) == 8500
    assert image_tokens(1024, 1024) == 25501
    assert image_tokens(2048, 4096) == 36835
    assert image_tokens(100, 100) == 8500
    with pytest.raises(ValueError):
        image_tokens(0, 512)


def test_forecast_uses_all_images_and_current_balance():
    forecast = estimate(inventory(), 11.75, len)
    assert forecast["formula_image_tokens"] == 599 * 8500
    assert forecast["old_mean_output_tokens"] == 100
    assert forecast["budget_limit_usd"] == 8
    assert not forecast["batch_approved"]
    assert estimate(inventory(), 0.1, len)["budget_stop"]


def test_existing_legacy_ledger_is_found_without_opening_it(tmp_path):
    folder = tmp_path / "previous_work" / "runs"
    folder.mkdir(parents=True)
    ledger = folder / "image_trial_one_call_ledger.json"
    ledger.write_text("{}", encoding="utf-8")
    assert existing_probes([tmp_path]) == [ledger.resolve()]
    result = run_probe({}, tmp_path / "new", 11.75, "", [tmp_path])
    assert result["status"] == "PREVIOUS_PROBE_FOUND"


def test_single_call_and_reexecution_never_calls_twice(tmp_path):
    calls = []

    def factory(**kwargs):
        assert kwargs["max_retries"] == 0

        def create(**payload):
            calls.append(payload)
            response = {
                "model": "gpt-4o-mini-2024-07-18",
                "usage": {"prompt_tokens": 8526, "completion_tokens": 50},
            }
            return SimpleNamespace(model_dump=lambda: response)

        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    result = run_probe(
        inventory(),
        tmp_path,
        11.75,
        "synthetic-only",
        [],
        counter=len,
        client_factory=factory,
    )
    assert result["status"] == "PROBE_SAVED_PENDING_DASHBOARD"
    assert len(calls) == 1
    again = run_probe({}, tmp_path, 11.75, "", [], counter=len, client_factory=factory)
    assert again["status"] == "PREVIOUS_PROBE_FOUND"
    assert len(calls) == 1


def test_failed_request_keeps_ledger_and_does_not_retry(tmp_path):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("synthetic service timeout")

    def factory(**kwargs):
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    result = run_probe(
        inventory(),
        tmp_path,
        11.75,
        "synthetic-only",
        [],
        counter=len,
        client_factory=factory,
    )
    assert result["status"] == "FAILED_NO_RETRY"
    assert len(calls) == 1
    assert (tmp_path / "image_detail_probe_ledger.json").is_file()


def test_budget_stop_precedes_client_creation(tmp_path):
    def forbidden_client(**kwargs):
        raise AssertionError("No client should be created.")

    result = run_probe(
        inventory(),
        tmp_path,
        0.1,
        "",
        [],
        counter=len,
        client_factory=forbidden_client,
    )
    assert result["status"] == "STOP_BUDGET_REVIEW"
    assert not (tmp_path / "image_detail_probe_ledger.json").exists()


@pytest.mark.parametrize("usage", [{}, {"prompt_tokens": 50000}])
def test_bad_usage_stops_without_batch_permission(tmp_path, usage):
    response = {"model": "gpt-4o-mini-2024-07-18", "usage": usage}

    def factory(**kwargs):
        def create(**payload):
            return SimpleNamespace(model_dump=lambda: response)

        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    result = run_probe(
        inventory(),
        tmp_path,
        11.75,
        "synthetic-only",
        [],
        counter=len,
        client_factory=factory,
    )
    assert result["status"] in {"STOP_MISSING_USAGE", "STOP_REVIEW"}
