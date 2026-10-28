"""Generate high-detail observations and save requests for resumable runs."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import re
import time
from pathlib import Path

try:
    from medvqa import detail_probe as probe
except ModuleNotFoundError:
    import detail_probe as probe


class ReviewRequired(RuntimeError):
    """Execution stopped with saved evidence; do not automatically repeat calls."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload(inventory, row, png):
    """Build one high-detail multimodal request without sending it.

    Args:
        inventory: Fixed request settings and prompt text mapping.
        row: Metadata for the selected image.
        png: Original PNG bytes for the selected image.

    Returns:
        dict: Request arguments with one image and its observation prompt.
    """
    return dict(
        inventory["payload_config"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(png).decode(),
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": inventory["prompts"][row["prompt_sha256"]],
                    },
                ],
            }
        ],
    )


def preflight(inventory, images, balance, counter=probe.count_text):
    """Check every image checksum and estimate the allowed batch cost.

    Args:
        inventory: Fixed 599-image inventory.
        images: Path to the directory containing neutral-ID PNG files.
        balance: Current API credit balance in USD.
        counter: Text-token counting callable.

    Returns:
        dict: Cost forecast and calibration approval status.

    Raises:
        ReviewRequired: The forecast exceeds the study's spending ceiling.
        ValueError: An image ID, checksum or frozen request setting is invalid.
    """
    probe.validate_inputs(inventory)
    forecast = probe.estimate(inventory, balance, counter)
    if forecast["budget_stop"]:
        raise ReviewRequired("Forecast exceeds the approved budget threshold.")
    forecast["batch_approved"] = inventory.get("batch_approved") is True
    for row in inventory["records"]:
        image_id = row["neutral_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
            raise ValueError("Unsafe image identifier.")
        if sha(images / f"{image_id}.png") != row["image_sha256"]:
            raise ValueError("An image differs from the frozen input inventory.")
    return forecast


def response_usage(response):
    if response.get("model") != probe.MODEL:
        raise ReviewRequired("The response model differs from the fixed snapshot.")
    usage = response.get("usage") or {}
    for key in ("prompt_tokens", "completion_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise ReviewRequired("Response usage is missing or invalid.")
    return usage


def response_text(response):
    choices = response.get("choices") or []
    if len(choices) != 1:
        return None
    message = choices[0].get("message") or {}
    text = message.get("content")
    if (
        message.get("refusal")
        or choices[0].get("finish_reason") == "content_filter"
        or not isinstance(text, str)
        or not text.strip()
    ):
        return None
    return text


def cost(usage):
    return (
        usage["prompt_tokens"] * probe.INPUT_PRICE
        + usage["completion_tokens"] * probe.OUTPUT_PRICE
    ) / 1e6


def validate_response(response, inventory, row, counter):
    """Check a saved response against the expected image-token usage.

    Args:
        response: Decoded API response with model identity and usage.
        inventory: Fixed prompt and request inventory.
        row: Image metadata including width, height and prompt checksum.
        counter: Text-token counting callable.

    Returns:
        dict: Validated input and output token usage from the response.

    Raises:
        ReviewRequired: The response model, usage or image-token residual is invalid.
    """
    usage = response_usage(response)
    prompt = inventory["prompts"][row["prompt_sha256"]]
    image_estimate = probe.image_tokens(row["width"], row["height"])
    residual = usage["prompt_tokens"] - counter(prompt)
    if abs(residual - image_estimate) / image_estimate > 0.15:
        raise ReviewRequired("Actual image-token residual differs by over 15%.")
    return usage


def observation(row, response, run_id, attempts, source):
    text = response_text(response)
    return {
        **row["observation_metadata"],
        "schema": "OBS-v1",
        "key": f"OBS|OBS-G|ARM-GRD|{row['neutral_id']}",
        "run_id": run_id,
        "mode": "TEST",
        "shard_index": 0,
        "observer": "OBS-G",
        "arm": "ARM-GRD",
        "neutral_id": row["neutral_id"],
        "prompt_sha256": row["prompt_sha256"],
        "image_sha256": row["image_sha256"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "ok",
        "text": text,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "truncated": response["choices"][0].get("finish_reason") == "length",
        "n_output_tokens": response["usage"]["completion_tokens"],
        "api": {
            k: response.get(k)
            for k in ("id", "model", "created", "system_fingerprint", "usage")
        },
        "gen": {
            "temperature": 0,
            "seed": 20261016,
            "max_tokens": 900,
            "detail": "high",
        },
        "quality": {
            "empty": False,
            "repeat_char20": bool(re.search(r"(.)\1{19,}", text, re.S)),
        },
        "attempts": attempts,
        "source": source,
    }


def run(
    inventory_path,
    images,
    output,
    balance,
    api_key,
    *,
    counter=probe.count_text,
    client_factory=None,
    sleep=time.sleep,
    progress=print,
):
    """Generate or resume observations while preserving every saved request.

    Args:
        inventory_path: Path to the fixed inventory with reconciled calibration usage.
        images: Directory of normalized input PNG files.
        output: Persistent run directory; reuse it only for the same configuration.
        balance: Current API balance in USD.
        api_key: API credential read from the environment or hidden input.
        counter: Text-token counting callable, injectable for offline tests.
        client_factory: Optional API client constructor for offline tests.
        sleep: Waiting callable used between content retries.
        progress: Callable receiving progress messages.

    Returns:
        dict: Completion and integrity summary with token-based cost.

    Raises:
        ReviewRequired: Calibration, budget, response or resume checks require review.
        ValueError: Input data or credentials are invalid.
        OSError: The persistent call ledger cannot be read or saved.
    """
    inventory_path, images, output = map(Path, (inventory_path, images, output))
    inventory = read(inventory_path)
    if (
        inventory.get("dashboard_reconciled") is not True
        or inventory.get("batch_approved") is not True
    ):
        raise ReviewRequired("The calibration usage has not been reconciled.")
    forecast = preflight(inventory, images, balance, counter)
    limit = min(forecast["output_cap_usd_with_reserve"], 8.0, 0.7 * balance)
    config = {
        "schema": "HIGH-DETAIL-OBSERVATIONS-v1",
        "mode": "TEST",
        "model": probe.MODEL,
        "detail": "high",
        "payload_config": inventory["payload_config"],
        "inventory_sha256": sha(inventory_path),
        "script_sha256": sha(Path(__file__)),
        "helper_sha256": sha(Path(probe.__file__)),
        "images": 599,
        "SDK_max_retries": 0,
        "content_attempts": 3,
        "content_retry_wait_seconds": 60,
        "budget_ceiling_usd": limit,
        "balance_before_usd_user_reported": balance,
        "probe_reused": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists():
        if read(config_path) != config:
            raise ReviewRequired(
                "Run configuration changed; existing calls are frozen."
            )
    else:
        probe.write_json(config_path, config)
    completed = output / "shard_00/integrity.json"
    if completed.exists():
        saved = read(completed)
        if (
            saved.get("status") != "COMPLETE"
            or saved.get("output_sha256") != sha(output / "shard_00/records.jsonl")
            or saved.get("run_config_sha256")
            != sha(output / "shard_00/run_config.json")
            or any(
                sha(output / name) != value
                for name, value in saved["response_files_sha256"].items()
            )
        ):
            raise ReviewRequired("Completed output checksum mismatch.")
        return saved
    lock = output / "running.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ReviewRequired(
            "Another run or interrupted session needs review."
        ) from None
    os.close(descriptor)
    try:
        return _execute(
            inventory,
            images,
            output,
            config,
            api_key,
            counter,
            client_factory,
            sleep,
            progress,
        )
    finally:
        lock.unlink()


def _execute(inv, images, output, config, key, counter, factory, sleep, progress):
    calls = output / "calls"
    calls.mkdir(exist_ok=True)
    records = []
    spent = 0.0
    client = None
    probe_id = inv["probe_neutral_id"]
    for row in inv["records"]:
        image_id = row["neutral_id"]
        png = (images / f"{image_id}.png").read_bytes()
        request = payload(inv, row, png)
        request_hash = probe.digest(json.dumps(request, sort_keys=True).encode())
        attempts = []
        response = None
        for attempt in range(1, 4):
            folder = calls / image_id / f"attempt_{attempt}"
            folder.mkdir(parents=True, exist_ok=True)
            response_path = folder / "response.json"
            request_path = folder / "request.json"
            if response_path.exists():
                if read(request_path)["payload_sha256"] != request_hash:
                    raise ReviewRequired("Saved response belongs to another request.")
                response = read(response_path)
            elif request_path.exists():
                raise ReviewRequired(
                    "Request without response: reconcile before retry."
                )
            elif image_id == probe_id and attempt == 1:
                if inv["probe_request_sha256"] != request_hash:
                    raise ReviewRequired("The calibration payload differs.")
                response = inv["probe_response"]
                probe.write_json(
                    request_path,
                    {"payload_sha256": request_hash, "reused_calibration": True},
                )
                probe.write_json(response_path, response)
            else:
                upper_prompt = (
                    probe.image_tokens(row["width"], row["height"]) * 1.15
                    + counter(inv["prompts"][row["prompt_sha256"]])
                    + 32
                )
                upper_cost = cost(
                    {"prompt_tokens": upper_prompt, "completion_tokens": 900}
                )
                if spent + upper_cost > config["budget_ceiling_usd"]:
                    raise ReviewRequired(
                        "Next request may exceed the spending ceiling."
                    )
                if attempt > 1:
                    sleep(60)
                if client is None:
                    if not key:
                        raise ValueError(
                            "Read the API key from Secrets or hidden input."
                        )
                    if factory is None:
                        from openai import OpenAI

                        factory = OpenAI
                    client = factory(api_key=key, max_retries=0, timeout=120)
                probe.write_json(request_path, {"payload_sha256": request_hash})
                try:
                    response = client.chat.completions.create(**request).model_dump()
                except Exception as error:
                    probe.write_json(
                        folder / "failure.json",
                        {
                            "exception_type": type(error).__name__,
                            "http_status": getattr(error, "status_code", None),
                        },
                    )
                    raise ReviewRequired(
                        "API failed; saved call needs review."
                    ) from None
                probe.write_json(response_path, response)
            usage = validate_response(response, inv, row, counter)
            spent += cost(usage)
            text = response_text(response)
            attempts.append({"attempt": attempt, "ok": text is not None})
            if text is not None:
                break
        if response_text(response) is None:
            raise ReviewRequired("Three content failures; review before continuing.")
        records.append(
            observation(
                row,
                response,
                output.name,
                attempts,
                "calibration" if image_id == probe_id else "batch",
            )
        )
        progress(f"{len(records)}/599 observations; token-based cost ${spent:.4f}")
    shard = output / "shard_00"
    shard.mkdir(exist_ok=True)
    encoded = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    record_path = shard / "records.jsonl"
    if record_path.exists():
        if record_path.read_text(encoding="utf-8") != encoded:
            # Only the generated audit timestamp may differ after a successful resume.
            old = [
                json.loads(s)
                for s in record_path.read_text(encoding="utf-8").splitlines()
            ]
            for before, after in zip(old, records, strict=True):
                before.pop("created_utc", None)
                after.pop("created_utc", None)
                if before != after:
                    raise ReviewRequired("Completed observation output differs.")
    else:
        with record_path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    if not (shard / "run_config.json").exists():
        probe.write_json(shard / "run_config.json", config)
    integrity = {
        "status": "COMPLETE",
        "mode": "TEST",
        "stopped_reason": "complete",
        "completed": len(records),
        "expected": 599,
        "remaining_in_shard": 0,
        "output_sha256": sha(record_path),
        "run_config_sha256": sha(shard / "run_config.json"),
        "response_files_sha256": {
            p.relative_to(output).as_posix(): sha(p)
            for p in sorted(calls.rglob("*.json"))
        },
        "token_based_total_cost_usd_including_calibration": spent,
        "dashboard_cost_reconciliation": "Required after completion",
    }
    if not (shard / "integrity.json").exists():
        probe.write_json(shard / "integrity.json", integrity)
    return integrity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--balance", type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    inventory = read(args.inventory)
    forecast = preflight(inventory, args.images, args.balance)
    print(json.dumps(forecast, indent=2))
    if args.execute:
        import getpass

        key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key: ")
        print(run(args.inventory, args.images, args.output, args.balance, key))


if __name__ == "__main__":
    main()
