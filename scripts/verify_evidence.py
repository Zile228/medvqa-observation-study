"""Verify the checksums, columns and row counts of published result tables."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> int:
    """Check each CSV against the aggregate-evidence manifest.

    Args:
        root: Directory containing the source distribution and evidence folder.

    Returns:
        int: Number of verified result tables.

    Raises:
        ValueError: A path, checksum, row count or column definition differs.
        OSError: A required manifest or result table cannot be read.
    """
    folder = (root / "evidence").resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "AGGREGATE-STUDY-EVIDENCE-v1":
        raise ValueError("Unknown aggregate evidence schema.")
    expected = manifest["files"]
    if set(expected) != {p.name for p in folder.glob("*.csv")}:
        raise ValueError("The result table list differs from the manifest.")
    for name, record in expected.items():
        path = (folder / name).resolve()
        if path.parent != folder or path.suffix != ".csv":
            raise ValueError("Invalid result table path.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Result checksum mismatch: {name}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            if reader.fieldnames != record["columns"] or len(rows) != record["rows"]:
                raise ValueError(f"Result dimensions differ: {name}")
    return len(expected)


def main() -> None:
    """Print the number of result tables that pass all evidence checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    print(f"Verified {verify(args.root)} aggregate result tables.")


if __name__ == "__main__":
    main()
