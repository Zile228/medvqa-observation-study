"""Execution environments, input manifests and reusable experiment helpers."""

from pathlib import Path

from medvqa import common as U

OBS_BANK_FILES = [
    "templates/observer_frame.txt",
    "templates/generic_items.json",
    "specs/routes.json",
    "specs/grounded_spec.json",
    "specs/author_anatomy_spec.json",
]
CANDIDATE_PINS = {
    "ENV-1": {
        "transformers": "4.49.0",
        "accelerate": "1.3.0",
        "bitsandbytes": "0.45.2",
        "tokenizers": "0.21.0",
        "torch": None,
        "pillow": None,
    },
    "ENV-2": {
        "transformers": "4.36.2",
        "tokenizers": "0.15.1",
        "accelerate": "0.21.0",
        "bitsandbytes": "0.41.0",
        "torch": None,
        "pillow": None,
    },
    "ENV-3": {
        "transformers": None,
        "accelerate": None,
        "bitsandbytes": None,
        "torch": None,
    },
    "ENV-API": {"openai": None, "pillow": None},
    "ENV-COLAB-L4": {
        "transformers": "4.49.0",
        "accelerate": "1.3.0",
        "bitsandbytes": "0.45.2",
        "tokenizers": "0.21.0",
        "torch": None,
        "pillow": None,
    },
}
PLATFORMS = ("kaggle", "colab")
PLATFORM_DEFAULTS = {
    "kaggle": {
        "input_root": "/kaggle/input/medvqa-inputs",
        "out_root": "/kaggle/working/out",
        "time_limit_s": 9 * 3600 - 15 * 60,
        "gpu": "T4 x2",
    },
    "colab": {
        "input_root": "/content/drive/MyDrive/medvqa_inputs",
        "out_root": "/content/drive/MyDrive/medvqa_runs",
        "time_limit_s": 3 * 3600,
        "gpu": "L4",
    },
}
MODEL_PLATFORM = {
    "Q": "colab",
    "L": "kaggle",
    "M": "kaggle",
    "J1": "kaggle",
    "J2": "kaggle",
    "PHI": "kaggle",
    "G": "any",
}


def platform_setup(platform):
    """Set platform defaults and mount Colab Drive when available.

    Defaults for a platform; mounts Google Drive when running inside Colab (outputs
    survive a cut session).
    """
    if platform not in PLATFORMS:
        raise ValueError(f"PLATFORM must be one of {PLATFORMS}")
    cfg = dict(PLATFORM_DEFAULTS[platform], platform=platform)
    if platform == "colab":
        try:
            from pathlib import Path as _P

            from google.colab import drive

            if not _P("/content/drive/MyDrive").is_dir():
                drive.mount("/content/drive")
        except ImportError:
            pass
    return cfg


def check_platform(model_key, platform, mode, run_params_file=None):
    """Require the recorded platform for the selected model.

    Require the measured platform for each model and verify the runtime configuration.
    """
    want = MODEL_PLATFORM[model_key]
    if want != "any" and platform != want:
        raise U.LockError(
            f"MAIN: model {model_key} must run on {want!r}, not {platform!r}"
        )
    if mode == "TEST":
        rp = U.read_json(run_params_file)
        locked = rp_get(rp, "platform_lock", model_key)
        if locked != "any" and locked != platform:
            raise U.LockError(
                (
                    "run_params.platform_lock["
                    f"{model_key}"
                    "] = "
                    f"{locked!r}"
                    " differs from PLATFORM "
                    f"{platform!r}"
                )
            )
    return {"model": model_key, "platform": platform, "lock": want}


def check_design(design, mode, run_params_file=None):
    """Require the configured experiment matrix before TEST execution."""
    from medvqa import conditions as CL

    if design not in CL.DESIGNS:
        raise ValueError(f"DESIGN must be one of {CL.DESIGNS}")
    if mode == "TEST" and rp_get(U.read_json(run_params_file), "design") != design:
        raise U.LockError("DESIGN differs from locked run_params.design")
    return design


def env_for(model_key, platform):
    if model_key == "M":
        return "ENV-2"
    if model_key == "Q" and platform == "colab":
        return "ENV-COLAB-L4"
    return "ENV-1"


def pip_install_cmd(pins):
    pk = [f"{p}=={v}" for p, v in pins.items() if v and p not in ("torch",)]
    return "pip install -q " + " ".join(pk) if pk else None


def verify_run_params_first(lock_file, run_params_file, mode):
    """Check runtime-configuration bytes before installation.

    Verify runtime-configuration bytes before reading versions or installing packages.
    """
    if mode == "TEST":
        return U.verify_locked_inputs(
            lock_file, {"run_params.json": run_params_file}, "TEST"
        )
    return None


def rp_get(rp, *path):
    """Read a required configuration value and reject unconfirmed placeholders."""
    v = rp
    for k in path:
        if not isinstance(v, dict) or k not in v:
            raise U.LockError(f"run_params.json has no {'.'.join(path)}")
        v = v[k]
    if v is None or (isinstance(v, str) and v.upper().startswith(("TO_", "PENDING"))):
        raise U.LockError(f"run_params.json {'.'.join(path)} is not confirmed ({v!r})")
    return v


def env_pins_for(mode, run_params_file, env):
    """TEST: pins from locked run_params.json; PILOT/DRYRUN: candidate pins."""
    if mode == "TEST":
        return U.read_json(run_params_file)["env_pins"][env]
    return dict(CANDIDATE_PINS[env])


def git_head(repo):
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


LLAVAMED_REPO = "/tmp/LLaVA-Med"


def install_env(env, pins, llavamed_commit=None):
    """Explicit installs shown in the notebook output.

    ENV-2 = original LLaVA-Med code at a pinned commit, installed with --no-deps so it
    cannot silently change torch; its dependencies come from the explicit pins.
    """
    import subprocess

    if env == "ENV-2":
        if not llavamed_commit:
            raise ValueError(
                "LLAVAMED_CODE_COMMIT must be set (pin) before installing ENV-2"
            )
        subprocess.run(
            (
                "rm -rf "
                f"{LLAVAMED_REPO}"
                " && git clone "
                "https://github.com/microsoft/LLaVA-Med.git "
                f"{LLAVAMED_REPO}"
                " && git -C "
                f"{LLAVAMED_REPO}"
                " checkout "
                f"{llavamed_commit}"
                " && pip install -q --no-deps -e "
                f"{LLAVAMED_REPO}"
            ),
            shell=True,
            check=True,
        )
    cmd = pip_install_cmd(pins)
    if cmd:
        subprocess.run(cmd, shell=True, check=True)


def verify_env(env, pins, mode, llavamed_commit=None, check_git=True):
    """Check package versions and the model source revision.

    Validate package versions and the pinned model code revision.
    """
    rep = {"env": env, "pins": U.check_env_pins(pins, mode)}
    if env == "ENV-2" and check_git:
        head = git_head(LLAVAMED_REPO)
        rep["llavamed_commit"] = {"pinned": llavamed_commit, "checked_out": head}
        if mode == "TEST" and (not llavamed_commit or head != llavamed_commit):
            raise U.LockError(
                (
                    "LLaVA-Med code commit "
                    f"{head!r}"
                    " differs from locked "
                    f"{llavamed_commit!r}"
                )
            )
    return rep


PILOT_COHORTS = ["COH-PILOT"]
TEST_COHORTS = ["COH-RAD-TEST", "COH-NAT-TEST"]


def cohorts_for(mode):
    return TEST_COHORTS if mode == "TEST" else PILOT_COHORTS


def prompt_lock_files(input_root, extra=()):
    """Map observer-prompt manifest names to local files.

    Logical relpath -> actual path for the observer prompt bank (+ extra protocol
    files).
    """
    root = Path(input_root)
    pd = root / "resources/protocol" / "prompts"
    files = {
        f"resources/protocol/prompts/generated/{p.name}": p
        for p in sorted((pd / "generated").glob("*.txt"))
    }
    for rel in list(OBS_BANK_FILES) + list(extra):
        files[f"resources/protocol/prompts/{rel}"] = pd / rel
    return files


def prep_lock_files(input_root, rels):
    return {f"prep_out/{r}": Path(input_root) / "prep_out" / r for r in rels}


def load_cohort_rows(prep_dir, cohorts):
    rows = []
    for c in cohorts:
        rows += U.read_jsonl(Path(prep_dir) / "cohorts" / f"{c}.jsonl")
    if len({r["qid"] for r in rows}) != len(rows):
        raise U.IntegrityError("duplicate qid in cohorts")
    return rows


def open_image(prep_dir, neutral_id):
    from PIL import Image

    p = Path(prep_dir) / "images" / f"{neutral_id}.png"
    if not p.is_file():
        return None
    with Image.open(p) as im:
        return im.convert("RGB")


def load_records(dirs, schema):
    """Read recorded outputs and detect duplicate execution keys.

    All non-error records of a schema from output dirs; duplicate keys are an integrity
    error.
    """
    out = {}
    done, dups = U.collect_done_keys(dirs)
    if dups:
        raise U.IntegrityError(
            f"duplicate keys in {schema} outputs: {sorted(set(dups))[:5]}"
        )
    for d in dirs:
        for f in sorted(Path(d).rglob("records.jsonl")):
            for r in U.read_jsonl(f):
                if r.get("schema") == schema and r.get("status") != "error":
                    out[r["key"]] = r
    return out


def write_run_config(session, cfg):
    path = session.out_dir / "run_config.json"
    U.write_json(path, cfg)
    return U.sha256_file(path)
