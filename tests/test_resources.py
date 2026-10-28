"""Checks for fixed prompts and supplementary condition texts."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from medvqa import neutral_controls

ROOT = Path(__file__).resolve().parents[1]


def test_resource_manifest_matches_all_packaged_resources():
    manifest = json.loads(
        (ROOT / "configs/resource_manifest.json").read_text(encoding="utf-8")
    )
    files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "resources").rglob("*")
        if path.is_file()
    }
    assert files == set(manifest)
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_neutral_conditions_have_identical_image_membership():
    bank = neutral_controls.load_bank(ROOT)
    membership = set(bank["ITEMS"])
    assert all(set(condition) == membership for condition in bank.values())
    assert Counter(mode for mode, _ in membership) == {"PILOT": 70, "TEST": 599}


def test_neutral_list_is_exact_placeholder_removal_for_every_image():
    bank = neutral_controls.load_bank(ROOT)
    suffix = " | — | —"
    for key, text in bank["NEUTRAL-DASH"].items():
        lines = text.splitlines()
        assert all(line.endswith(suffix) for line in lines)
        assert (
            "\n".join(line[: -len(suffix)] for line in lines)
            == (bank["NEUTRAL-LIST"][key])
        )
        assert len(lines) == len(bank["ITEMS"][key].splitlines())
        assert all(
            not line.endswith(suffix) for line in bank["ITEMS"][key].splitlines()
        )


def test_control_text_hash_is_recorded_without_modification():
    bank = neutral_controls.load_bank(ROOT)
    for condition in neutral_controls.cell_specs():
        for mode, neutral_id in bank[condition["cell"][len("CELL-Q-") :]]:
            text, reference = neutral_controls.observation(
                bank, mode, condition, neutral_id
            )
            assert (
                reference["obs_text_source_sha256"]
                == hashlib.sha256(text.encode("utf-8")).hexdigest()
            )
