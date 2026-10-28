"""One-request calibration of high-detail image input tokens and study cost."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import math
import uuid
from pathlib import Path

MODEL = "gpt-4o-mini-2024-07-18"
INPUT_PRICE = 0.15
OUTPUT_PRICE = 0.60
RESERVE = 1.20


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def image_tokens(width: int, height: int) -> int:
    """Estimate high-detail image tokens with the study's tile formula.

    Args:
        width: Original image width in pixels.
        height: Original image height in pixels.

    Returns:
        int: Estimated image input tokens for the pinned GPT snapshot.

    Raises:
        ValueError: Either image dimension is nonpositive.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if max(width, height) > 2048:
        scale = 2048 / max(width, height)
        width, height = math.floor(width * scale), math.floor(height * scale)
    if min(width, height) > 768:
        scale = 768 / min(width, height)
        width, height = math.floor(width * scale), math.floor(height * scale)
    return 2833 + 5667 * math.ceil(width / 512) * math.ceil(height / 512)


def count_text(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("o200k_base").encode(text))


def estimate(inventory: dict, balance: float, counter=count_text) -> dict:
    """Estimate all image and text costs with a twenty-percent reserve.

    Args:
        inventory: Fixed 599-image inventory with prompts and earlier output lengths.
        balance: Current positive API balance in USD.
        counter: Callable returning the token count of a text string.

    Returns:
        dict: Token totals, projected costs, budget ceiling and stop decision.

    Raises:
        ValueError: The balance is nonfinite or nonpositive.
    """
    if not math.isfinite(balance) or balance <= 0:
        raise ValueError("Enter the current positive API credit balance.")
    rows = inventory["records"]
    texts = {key: counter(text) for key, text in inventory["prompts"].items()}
    image_total = sum(image_tokens(row["width"], row["height"]) for row in rows)
    text_total = sum(texts[row["prompt_sha256"]] for row in rows)
    prompt_total = image_total + text_total + 7 * len(rows)
    old_outputs = sum(row["original_completion_tokens"] for row in rows)
    expected = (prompt_total * INPUT_PRICE + old_outputs * OUTPUT_PRICE) / 1e6
    cap = (prompt_total * INPUT_PRICE + len(rows) * 900 * OUTPUT_PRICE) / 1e6
    limit = min(8.0, 0.7 * balance)
    return {
        "images": len(rows),
        "formula_image_tokens": image_total,
        "encoded_text_tokens": text_total,
        "chat_wrapper_tokens_per_request_estimated": 7,
        "chat_wrapper_exact": False,
        "old_output_tokens": old_outputs,
        "old_mean_output_tokens": old_outputs / len(rows),
        "estimated_usd_with_reserve": expected * RESERVE,
        "output_cap_usd_with_reserve": cap * RESERVE,
        "reserve_fraction": 0.2,
        "budget_limit_usd": limit,
        "budget_stop": cap * RESERVE > limit,
        "batch_approved": False,
    }


def existing_probes(roots: list[Path]) -> list[Path]:
    """Find saved calibration ledgers before creating a new API request.

    Args:
        roots: Directories to search for the supported ledger filenames.

    Returns:
        list[Path]: Sorted unique paths to existing calibration evidence.
    """
    patterns = (
        "*probe*ledger*.json",
        "*one_call_ledger.json",
        "*/runs/*one_call_ledger.json",
        "*runs/*one_call_ledger.json",
        "*/runs/*probe*/request_summary.json",
        "*runs/*probe*/request_summary.json",
        "*runs/*detail_probe*/request_summary.json",
    )
    found = set()
    for root in roots:
        if root.exists():
            for pattern in patterns:
                found.update(path.resolve() for path in root.glob(pattern))
    return sorted(found)


def validate_inputs(inventory: dict) -> tuple[dict, bytes]:
    """Validate the pinned request settings and calibration image bytes.

    Args:
        inventory: Calibration inventory loaded from the supplied JSON resource.

    Returns:
        tuple[dict, bytes]: Selected image metadata and decoded PNG bytes.

    Raises:
        ValueError: Settings, membership, prompt hashes or image hashes differ.
    """
    expected = {"model": MODEL, "temperature": 0, "seed": 20261016, "max_tokens": 900}
    if inventory["payload_config"] != expected:
        raise ValueError("The model snapshot or request settings changed.")
    rows = inventory["records"]
    if len(rows) != 599 or len({row["neutral_id"] for row in rows}) != 599:
        raise ValueError("The forecast must cover the fixed 599 images.")
    for expected_hash, text in inventory["prompts"].items():
        if digest(text.encode()) != expected_hash:
            raise ValueError("Observation prompt hash mismatch.")
    probe = next(
        row for row in rows if row["neutral_id"] == inventory["probe_neutral_id"]
    )
    png = base64.b64decode(inventory["probe_png_base64"], validate=True)
    if digest(png) != probe["image_sha256"]:
        raise ValueError("The selected trial image changed.")
    return probe, png


def run_probe(
    inventory: dict,
    output_root: Path,
    balance_before: float,
    api_key: str,
    search_roots: list[Path],
    *,
    counter=count_text,
    client_factory=None,
) -> dict:
    """Send at most one calibration request and preserve its call ledger.

    Args:
        inventory: Validated inventory containing the calibration image and prompts.
        output_root: Persistent directory in which to save the new trial.
        balance_before: Current API balance in USD before the request.
        api_key: API credential supplied by the caller; never saved in the ledger.
        search_roots: Additional directories to check for earlier trial ledgers.
        counter: Text-token counting callable; may be replaced in offline tests.
        client_factory: Optional client constructor for offline tests; None uses OpenAI.

    Returns:
        dict: Saved trial result, previous-trial notice or budget-stop notice.

    Raises:
        ValueError: Inputs, balance or API credential are invalid.
        OSError: The durable trial files cannot be saved.
    """
    previous = existing_probes(search_roots + [output_root])
    if previous:
        return {"status": "PREVIOUS_PROBE_FOUND", "paths": [str(p) for p in previous]}
    probe, png = validate_inputs(inventory)
    forecast = estimate(inventory, balance_before, counter)
    if forecast["budget_stop"]:
        return {"status": "STOP_BUDGET_REVIEW", "forecast": forecast}
    if not api_key:
        raise ValueError("An API key is required for the single trial request.")
    prompt = inventory["prompts"][probe["prompt_sha256"]]
    payload = dict(
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
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    if client_factory is None:
        from openai import OpenAI

        client_factory = OpenAI
    client = client_factory(api_key=api_key, max_retries=0, timeout=120)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"detail_probe_{now}_{uuid.uuid4().hex[:6]}"
    destination.mkdir()
    write_json(
        output_root / "image_detail_probe_ledger.json",
        {"model": MODEL, "started_utc": now, "output_dir": str(destination)},
    )
    request = {
        "model": MODEL,
        "detail": "high",
        "neutral_id": probe["neutral_id"],
        "image_sha256": probe["image_sha256"],
        "prompt_sha256": probe["prompt_sha256"],
        "request_sha256": digest(json.dumps(payload, sort_keys=True).encode()),
        "balance_before_usd_user_reported": balance_before,
        "SDK_max_retries": 0,
    }
    write_json(destination / "request_summary.json", request)
    try:
        response = client.chat.completions.create(**payload).model_dump()
    except Exception as error:
        result = {
            "status": "FAILED_NO_RETRY",
            "exception_type": type(error).__name__,
            "http_status": getattr(error, "status_code", None),
            "batch_approved": False,
        }
    else:
        write_json(destination / "response.json", response)
        usage = response.get("usage") or {}
        if not isinstance(usage.get("prompt_tokens"), int):
            result = {"status": "STOP_MISSING_USAGE", "batch_approved": False}
        else:
            residual = usage["prompt_tokens"] - counter(prompt)
            predicted = image_tokens(probe["width"], probe["height"])
            difference = abs(residual - predicted) / predicted
            stop = difference > 0.15 or response.get("model") != MODEL
            result = {
                "status": "STOP_REVIEW" if stop else "PROBE_SAVED_PENDING_DASHBOARD",
                "request": request,
                "response": response,
                "forecast": forecast,
                "calibration": {
                    "formula_image_tokens": predicted,
                    "text_tokens": counter(prompt),
                    "image_plus_wrapper_residual": residual,
                    "image_tokens_exact": False,
                    "relative_difference": difference,
                    "stop_over_15_percent": difference > 0.15,
                },
                "probe_token_based_cost_usd": (
                    usage["prompt_tokens"] * INPUT_PRICE
                    + usage.get("completion_tokens", 0) * OUTPUT_PRICE
                )
                / 1e6,
                "dashboard": {
                    "balance_after_usd": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "reconciled": False,
                },
                "batch_approved": False,
            }
    write_json(destination / "probe_result.json", result)
    files = {
        p.name: digest(p.read_bytes()) for p in destination.iterdir() if p.is_file()
    }
    write_json(destination / "integrity.json", {"files_sha256": files})
    return {"status": result["status"], "output_dir": str(destination)}
