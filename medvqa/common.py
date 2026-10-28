"""File I/O, checksums and execution-session tracking."""

import builtins
import contextlib
import datetime as _dt
import hashlib
import io
import json
import os
import platform
import re
import sys
import time
import uuid
from pathlib import Path

SEED = 20261016
SCHEMA_VERSION = "SCHEMA-v1"
_api_cutoff = os.environ.get("MEDVQA_API_CUTOFF_UTC")
GPT_DEADLINE_UTC = (
    _dt.datetime.fromisoformat(_api_cutoff.replace("Z", "+00:00"))
    if _api_cutoff
    else None
)
if GPT_DEADLINE_UTC is not None and GPT_DEADLINE_UTC.tzinfo is None:
    raise ValueError("MEDVQA_API_CUTOFF_UTC must include a timezone")
SHARD_TIME_LIMIT_S = 9 * 3600 - 15 * 60


class LockError(RuntimeError):
    """Signal a missing or inconsistent fixed input or configuration.

    Raised when a locked input is missing from locked_inputs.sha256 or its SHA differs.
    """


class IntegrityError(RuntimeError):
    pass


def utc_now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def new_session_id():
    return (
        _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:6]
    )


def sha256_bytes(b):
    """Compute the SHA-256 digest of a byte sequence.

    Args:
        b: Bytes to hash.

    Returns:
        str: Lowercase hexadecimal digest.
    """
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    """Hash UTF-8 text without changing its whitespace.

    Args:
        s: Text to encode and hash.

    Returns:
        str: Lowercase hexadecimal digest.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path):
    """Hash a file in bounded-size binary chunks.

    Args:
        path: Path to the input file.

    Returns:
        str: Lowercase hexadecimal digest.

    Raises:
        OSError: The file cannot be opened or read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(obj):
    return sha256_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def rgb_hash(image):
    """Hash the decoded RGB pixels together with the image dimensions.

    This definition is shared by image deduplication and bootstrap grouping.
    """
    rgb = image.convert("RGB")
    return "rgb:" + hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def read_json(path):
    """Read a UTF-8 JSON document.

    Args:
        path: Path to the JSON file.

    Returns:
        object: Decoded JSON value.

    Raises:
        OSError: The file cannot be read.
        json.JSONDecodeError: The file is not valid JSON.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    """Write JSON atomically through a temporary sibling file.

    Args:
        path: Destination path; missing parent directories are created.
        obj: JSON-serializable value to save.

    Raises:
        OSError: The destination cannot be written.
        TypeError: The value is not JSON-serializable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def write_json_once(path, obj):
    """Save an immutable JSON value or verify an identical existing copy.

    Args:
        path: Destination path.
        obj: JSON-serializable value to preserve.

    Raises:
        IntegrityError: An existing file contains a different value.
        OSError: The file cannot be read or written.
    """
    path = Path(path)
    if path.exists():
        if read_json(path) != json.loads(json.dumps(obj, ensure_ascii=False)):
            raise IntegrityError(
                f"Frozen artifact differs: {path}. Use a new run directory."
            )
        return
    write_json(path, obj)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"Invalid JSONL {path}:{n}") from exc
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def get_secret(name):
    """Environment -> Kaggle secrets -> Colab userdata.

    Returns the value; callers must not log it.
    """
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        value = (UserSecretsClient().get_secret(name) or "").strip()
        if value:
            return value
    except Exception:
        pass
    try:
        from google.colab import userdata

        value = (userdata.get(name) or "").strip()
        if value:
            return value
    except Exception:
        pass
    raise RuntimeError(
        f"Secret {name} not found: add it in Kaggle Add-ons > Secrets or Colab Secrets."
    )


PLACEHOLDER_SHA = {"0" * 64, "TO_FILL", "PENDING"}


def parse_sha256_file(path):
    """Read a checksum manifest in sha256sum format.

    Parse `sha256sum` format: '<sha> <relpath>' (also '<sha> *<relpath>'). '#' lines are
    comments.
    """
    table = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match("^(\\S+)\\s+\\*?(.+)$", s)
        if not m:
            raise LockError(f"Bad line {n} in {path}: {line!r}")
        sha, rel = (m.group(1), m.group(2).strip())
        if rel.startswith("./"):
            rel = rel[2:]
        if rel in table:
            raise LockError(f"Duplicate entry {rel} in {path}")
        table[rel] = sha.lower() if sha.upper() not in PLACEHOLDER_SHA else "TO_FILL"
    return table


def verify_locked_inputs(lock_file, files, mode):
    """Check input files against their recorded checksums before execution.

    Args:
        lock_file: Checksum manifest path.
        files: Mapping from manifest names to local paths.
        mode: PILOT allows unlisted checksums; TEST requires the complete lock.

    Returns:
        dict: Verification results, including verified status and unlisted files.

    Raises:
        ValueError: The execution mode is unsupported.
        LockError: A file is missing, a checksum differs or a TEST lock is incomplete.
    """
    if mode not in ("PILOT", "TEST", "DRYRUN"):
        raise ValueError(mode)
    table = (
        parse_sha256_file(lock_file)
        if lock_file and Path(lock_file).is_file()
        else None
    )
    if table is None and mode == "TEST":
        raise LockError(f"locked_inputs.sha256 not found: {lock_file}")
    table = table or {}
    report = {
        "lock_file": str(lock_file),
        "lock_file_sha256": sha256_file(lock_file)
        if lock_file and Path(lock_file).is_file()
        else None,
        "files": {},
        "unlisted_used": [],
        "verified": True,
    }
    problems = []
    for rel, actual in sorted(files.items()):
        actual = Path(actual)
        if not actual.is_file():
            problems.append(f"missing file {rel} -> {actual}")
            continue
        got = sha256_file(actual)
        report["files"][rel] = got
        want = table.get(rel)
        if want is None or want == "TO_FILL":
            if mode == "TEST":
                problems.append(f"not locked (no SHA in lock file): {rel}")
            else:
                report["unlisted_used"].append(rel)
        elif want != got:
            problems.append(
                f"SHA mismatch {rel}: locked {want[:12]}…, actual {got[:12]}…"
            )
    if problems:
        report["verified"] = False
        raise LockError(
            "Input checksum verification failed:\n  " + "\n  ".join(problems)
        )
    return report


SEALED_PATTERNS = (
    "id_map.json",
    "prep_sealed",
    "gold_RAD_TEST",
    "gold_NAT_TEST",
    "gold_DEV",
)


@contextlib.contextmanager
def sealed_guard(allow=()):
    """Block access to scoring labels except explicitly allowed files.

    Inside this context any open() of a sealed path raises PermissionError (id_map /
    gold files). `allow`: exact basenames that are permitted (e.g. judge J-OPEN may read
    gold_RAD_TEST.jsonl).
    """
    real_open, real_io_open = (builtins.open, io.open)

    def guarded(file, *args, **kwargs):
        name = str(file)
        if isinstance(file, (str, os.PathLike)) and any(
            (p in name for p in SEALED_PATTERNS)
        ):
            if Path(name).name not in allow:
                raise PermissionError(
                    f"Reference-label access is blocked: {Path(name).name}"
                )
        return real_open(file, *args, **kwargs)

    builtins.open = guarded
    io.open = guarded
    try:
        yield
    finally:
        builtins.open = real_open
        io.open = real_io_open


def assert_no_sealed_inputs(dirs):
    """Reject scoring-label files in inference input directories.

    Fail if an input directory of a GPU/API notebook contains sealed files (they should
    not be uploaded).
    """
    bad = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and any((s in str(p) for s in SEALED_PATTERNS)):
                bad.append(str(p))
    if bad:
        raise IntegrityError(
            (
                "Sealed files found in notebook inputs; remove "
                "them from the uploaded dataset: "
            )
            + "; ".join(bad[:5])
        )


def environment_info():
    pk = {}
    try:
        import importlib.metadata as md

        for name in (
            "torch",
            "transformers",
            "accelerate",
            "bitsandbytes",
            "tokenizers",
            "openai",
            "numpy",
            "pillow",
            "huggingface-hub",
            "sentencepiece",
            "qwen-vl-utils",
        ):
            try:
                pk[name] = md.version(name)
            except Exception:
                pk[name] = None
    except Exception:
        pass
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": pk,
        "gpu": gpu,
        "kaggle": Path("/kaggle/working").is_dir(),
        "colab": "google.colab" in sys.modules,
        "kaggle_env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("KAGGLE_") and ("IMAGE" in k or "DOCKER" in k)
        },
    }


def plan_shard(keys, num_shards, shard_index):
    """Assign sorted unique execution keys to one deterministic shard.

    Args:
        keys: Complete iterable of planned execution keys.
        num_shards: Positive total number of shards.
        shard_index: Zero-based shard to select.

    Returns:
        list: Keys assigned to the selected shard.

    Raises:
        ValueError: The shard count or index is invalid.
        IntegrityError: The input contains duplicate keys.
    """
    if not (
        isinstance(num_shards, int)
        and num_shards >= 1
        and (0 <= shard_index < num_shards)
    ):
        raise ValueError("bad shard spec")
    keys = list(keys)
    if len(set(keys)) != len(keys):
        raise IntegrityError("duplicate planned keys")
    n = len(keys)
    lo = n * shard_index // num_shards
    hi = n * (shard_index + 1) // num_shards
    return keys[lo:hi]


def collect_done_keys(dirs, filename="records.jsonl", exclude_status=("error",)):
    """Union of keys already written in prior/current output dirs (recursive).

    Rows whose status is in exclude_status (transient errors) do not count as done, so
    resume retries them. Two non- error rows with the same key are reported as
    duplicates (integrity failure).
    """
    seen, dups, files_seen = ({}, [], set())
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(d.rglob(filename)):
            rf = f.resolve()
            if rf in files_seen:
                continue
            files_seen.add(rf)
            for row in read_jsonl(f):
                if row.get("status") in exclude_status:
                    continue
                k = row["key"]
                if k in seen:
                    dups.append(k)
                seen[k] = str(f)
    return (set(seen), dups)


class ShardSession:
    """One Kaggle/Colab session working on one shard.

    Append-only output, resume by key, time guard.
    """

    def __init__(
        self,
        notebook,
        run_id,
        mode,
        out_root,
        shard_index,
        num_shards,
        prior_dirs=(),
        time_limit_s=SHARD_TIME_LIMIT_S,
        max_items=None,
    ):
        self.notebook, self.run_id, self.mode = (notebook, run_id, mode)
        self.shard_index, self.num_shards = (shard_index, num_shards)
        self.session_id = new_session_id()
        self.out_dir = (
            Path(out_root) / run_id / f"shard_{shard_index:02d}_{self.session_id}"
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records = self.out_dir / "records.jsonl"
        self.prior_dirs = [Path(p) for p in prior_dirs] + [Path(out_root) / run_id]
        self.time_limit_s, self.max_items = (time_limit_s, max_items)
        self.t0 = time.time()
        self.started = utc_now()
        self.done_this = 0
        self.errors = 0
        self.stopped_reason = None

    def done_keys(self):
        return collect_done_keys(self.prior_dirs)

    def should_stop(self):
        if self.max_items is not None and self.done_this >= self.max_items:
            self.stopped_reason = "max_items"
            return True
        if time.time() - self.t0 > self.time_limit_s:
            self.stopped_reason = "time_limit"
            return True
        return False

    def base(self, schema, key):
        return {
            "schema": schema,
            "key": key,
            "run_id": self.run_id,
            "mode": self.mode,
            "shard_index": self.shard_index,
            "session_id": self.session_id,
            "created_utc": utc_now(),
        }

    def write(self, row):
        append_jsonl(self.records, row)
        self.done_this += 1
        if row.get("status") == "error":
            self.errors += 1

    def integrity(self, planned, extra=None):
        done, dups = collect_done_keys(self.prior_dirs)
        planned = list(planned)
        remaining = [k for k in planned if k not in done]
        rep = {
            "schema": "INTEGRITY-v1",
            "notebook": self.notebook,
            "run_id": self.run_id,
            "mode": self.mode,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "session_id": self.session_id,
            "started_utc": self.started,
            "ended_utc": utc_now(),
            "stopped_reason": self.stopped_reason
            or ("complete" if not remaining else "incomplete"),
            "planned_keys_in_shard": len(planned),
            "done_this_session": self.done_this,
            "done_total_in_shard": len(planned) - len(remaining),
            "remaining_in_shard": len(remaining),
            "errors": self.errors,
            "duplicate_keys": sorted(set(dups)),
            "output_sha256": sha256_file(self.records)
            if self.records.exists()
            else None,
        }
        if extra:
            rep.update(extra)
        write_json(self.out_dir / "integrity.json", rep)
        return rep


_REPEAT20 = re.compile("(.)\\1{19,}", re.S)


def quality_flags(text):
    return {
        "empty": not str(text).strip(),
        "repeat_char20": bool(_REPEAT20.search(str(text))),
    }


def summarize_quality(rows, group_fields=("observer", "arm", "route_key")):
    """Calculate grouped output-quality and numerical-error rates.

    Rates per group: truncation, empty text, character repetition and invalid logits.
    """
    from collections import defaultdict

    acc = defaultdict(
        lambda: {"n": 0, "truncated": 0, "empty": 0, "repeat_char20": 0, "nonfinite": 0}
    )
    for r in rows:
        g = "|".join((str(r.get(f)) for f in group_fields))
        a = acc[g]
        a["n"] += 1
        a["truncated"] += bool(r.get("truncated"))
        q = r.get("quality") or {}
        a["empty"] += bool(q.get("empty"))
        a["repeat_char20"] += bool(q.get("repeat_char20"))
        a["nonfinite"] += (
            r.get("first_step_finite") is False
            or (r.get("logprob") or {}).get("finite") is False
        )
    out = {}
    for g, a in sorted(acc.items()):
        n = max(a["n"], 1)
        out[g] = dict(
            a,
            **{
                f"{k}_rate": round(a[k] / n, 4)
                for k in ("truncated", "empty", "repeat_char20", "nonfinite")
            },
        )
    return out


def installed_versions(packages):
    import importlib.metadata as md

    out = {}
    for p in packages:
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


def check_env_pins(pins, mode):
    """Compare installed package versions with the recorded execution environment.

    TEST rejects missing or mismatched versions; PILOT and DRYRUN report them."""
    got = installed_versions(pins)
    report = {p: {"pinned": v, "installed": got[p]} for p, v in pins.items()}
    if mode == "TEST":
        bad = [
            f"{p}: pinned {v!r}, installed {got[p]!r}"
            for p, v in pins.items()
            if not v or str(v).upper().startswith(("TO_", "PENDING")) or got[p] != v
        ]
        if bad:
            raise LockError(
                "Environment differs from locked pins (run_params.json env_pins):\n  "
                + "\n  ".join(bad)
            )
    return report
