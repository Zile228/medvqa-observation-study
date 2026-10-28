"""Assemble an input directory without mixing reference labels into inference."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {"prep_sealed", "id_map.json", "source_labels.jsonl"}


def public_preparation(path: Path) -> bool:
    """Return whether a prepared-data tree contains inference inputs only."""
    return not any(
        p.name in FORBIDDEN_NAMES or p.name.startswith("gold_") for p in path.rglob("*")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output: Path, prepared_data: Path | None, vqa_rad: Path | None) -> None:
    """Copy fixed resources and optionally add prepared images and questions."""
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if prepared_data and not public_preparation(prepared_data):
        raise ValueError("Prepared inference inputs must exclude reference labels.")
    output.mkdir(parents=True)
    for name in ("resources", "prep", "artifacts", "configs"):
        shutil.copytree(ROOT / name, output / name)
    for name in ("run_params.json", "design_state.json"):
        shutil.copy2(ROOT / name, output / name)
    if prepared_data:
        shutil.copytree(prepared_data, output / "prep_out")
    if vqa_rad:
        shutil.copytree(vqa_rad, output / "vqarad_original")
    entries = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            continue
        entries[path.relative_to(output).as_posix()] = sha256(path)
    entries["run_params.json"] = sha256(output / "run_params.json")
    text = "".join(f"{digest}  {name}\n" for name, digest in entries.items())
    (output / "locked_inputs.sha256").write_text(text, encoding="utf-8")
    (output / "locked_inputs_judge_panel.sha256").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--vqa-rad", type=Path)
    args = parser.parse_args()
    build(args.output, args.prepared_data, args.vqa_rad)


if __name__ == "__main__":
    main()
