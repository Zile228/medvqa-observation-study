"""Frozen neutral-label controls for supplementary Qwen answer experiments."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ARMS = ("ITEMS", "NEUTRAL-DASH", "NEUTRAL-LIST")
COUNTS = {"PILOT": (70, 199, 279), "TEST": (542, 988, 1824)}


def load_bank(source_root: Path) -> dict:
    """Load the three fixed control-text banks after checksum verification.

    Args:
        source_root: Source directory containing configs and resources.

    Returns:
        dict: Condition names mapped to text indexed by mode and image ID.

    Raises:
        ValueError: A checksum, image membership or mode count differs.
    """
    manifest = json.loads(
        (source_root / "configs/resource_manifest.json").read_text(encoding="utf-8")
    )
    bank = {}
    for arm in ARMS:
        name = arm.lower().replace("-", "_") + ".jsonl"
        path = source_root / "resources/neutral_controls" / name
        relative = path.relative_to(source_root).as_posix()
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[relative]:
            raise ValueError(f"Condition text hash mismatch: {arm}")
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        indexed = {(row["mode"], row["neutral_id"]): row["text"] for row in rows}
        if len(rows) != len(indexed) or len(rows) != 669:
            raise ValueError("Condition texts must cover 669 unique image/mode pairs.")
        if Counter(mode for mode, _ in indexed) != {"PILOT": 70, "TEST": 599}:
            raise ValueError("Unexpected pilot/evaluation condition membership.")
        bank[arm] = indexed
    return bank


def cell_specs() -> list[dict]:
    """Describe the three supplementary Qwen text conditions.

    Returns:
        list[dict]: Answer-runner condition records for ITEMS and both neutral lists.
    """
    return [
        {
            "cell": f"CELL-Q-{arm}",
            "answerer": "ANS-Q",
            "obs_observer": None,
            "obs_arm": None,
            "intervention": None,
            "image_condition": "original",
            "obs_kind": "neutral_control",
        }
        for arm in ARMS
    ]


def plan(rows: list[dict], mode: str, bank: dict) -> list[tuple]:
    """Expand closed questions into all three control-condition executions.

    Args:
        rows: Question records for one complete PILOT or TEST sample.
        mode: PILOT or TEST, selecting the fixed sample and turn counts.
        bank: Validated control texts from load_bank.

    Returns:
        list[tuple]: Execution key, condition, question and option order.

    Raises:
        ValueError: Membership, question kinds or turn counts differ.
        KeyError: The requested mode or condition text is absent.
    """
    images, questions, turns = COUNTS[mode]
    if len(rows) != questions or len({r["neutral_id"] for r in rows}) != images:
        raise ValueError("Question/image membership differs from the study design.")
    if any(r["answer_kind"] == "open" for r in rows):
        raise ValueError("Neutral-control evaluation uses closed questions only.")
    units = []
    for condition in cell_specs():
        for row in rows:
            observation(bank, mode, condition, row["neutral_id"])
            orders = row["orders"] if row["answer_kind"] == "mcq" else ["-"]
            for order in orders:
                key = f"ANS|{condition['cell']}|{row['qid']}|{order}"
                units.append((key, condition, row, order))
    if len(units) != turns * len(ARMS) or len({u[0] for u in units}) != len(units):
        raise ValueError("Planned answer units are missing or duplicated.")
    return units


def observation(bank: dict, mode: str, condition: dict, neutral_id: str) -> tuple:
    """Retrieve one exact control text and its reproducibility metadata.

    Args:
        bank: Validated condition text banks.
        mode: PILOT or TEST.
        condition: Condition record returned by cell_specs.
        neutral_id: Normalized image identifier.

    Returns:
        tuple[str, dict]: Fixed text and metadata containing its checksum.

    Raises:
        KeyError: The selected text is absent from the fixed bank.
    """
    arm = condition["cell"][len("CELL-Q-") :]
    text = bank[arm][mode, neutral_id]
    return text, {
        "key": f"NEUTRAL-CONTROL|{mode}|{arm}|{neutral_id}",
        "obs_status": "ok",
        "obs_text_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
