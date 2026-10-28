"""Verify the source distribution against its SHA-256 manifest."""

import argparse
import hashlib
from pathlib import Path


def verify(root: Path) -> int:
    """Reject absent, changed, duplicated, or unsafe manifest members."""
    root = root.resolve()
    checked = set()
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name in checked:
            raise ValueError(f"Invalid manifest member: {name}")
        if not path.is_file():
            raise FileNotFoundError(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Hash mismatch: {name}")
        checked.add(name)
    return len(checked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    print(f"Verified {verify(args.root)} source files.")


if __name__ == "__main__":
    main()
