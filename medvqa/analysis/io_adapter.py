"""Load and validate experiment records before constructing analysis tables."""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from medvqa import judge_scope as AX1

from .constants import DESIGNS, MAIN_RQ2_LEVELS
from .missing import is_missing_scalar

COHORT_MAP = {"COH-RAD-TEST": "RAD", "COH-NAT-TEST": "NAT"}
OBSERVER_MAP = {"OBS-L": "L", "OBS-Q": "Q", "OBS-M": "M", "OBS-G": "G"}


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def canonical_cell(cell: str) -> str:
    """Remove the CELL- prefix from a recorded condition identifier.

    Recorded IDs have the form `CELL-<answerer>-<observer>-<arm>`.
    """
    c = str(cell).strip()
    return c[5:] if c.upper().startswith("CELL-") else c


def load_records(runs_dir: str, warnings: List[str]) -> Dict[str, pd.DataFrame]:
    """Read recorded outputs and detect duplicate execution keys.

    All TEST-mode records grouped by schema; one record per `key`.

    A key may appear again only after an `error` record (resume); the latest non-error
    record wins. Two non-error records with the same key are an integrity error (kept:
    latest; warned).
    """
    rows: List[dict] = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "shard_*", "records.jsonl"))):
        if os.sep + "prep" + os.sep in p:
            continue
        for r in read_jsonl(p):
            r["_file"] = os.path.relpath(p, runs_dir)
            rows.append(r)
    out: Dict[str, pd.DataFrame] = {}
    if not rows:
        return out
    ax1_records = [
        r
        for r in rows
        if r.get("mode", "TEST") == "TEST"
        and r.get("schema") == "JUDGE-v1"
        and (r.get("task") == "J-AX1")
    ]
    scope_marks = {r.get("ax1_scope_sha256") or "legacy" for r in ax1_records}
    if len(scope_marks) > 1:
        raise ValueError(
            "mixed old/new J-AX1 scope outputs; separate the analysis input directories"
        )
    df = pd.DataFrame(rows)
    if "mode" in df:
        df = df[df["mode"].fillna("TEST") == "TEST"]
    df["_err"] = df.get("status", pd.Series(index=df.index, dtype=object)).eq("error")
    df["_t"] = df.get("created_utc", pd.Series("", index=df.index)).fillna("")
    df = df.sort_values(["key", "_err", "_t"], ascending=[True, False, True])
    ok_dups = df[~df["_err"]].duplicated("key", keep=False)
    if ok_dups.any():
        warnings.append(
            (
                "INTEGRITY: "
                f"{int(ok_dups.sum())}"
                " non-error records share a key; latest kept"
            )
        )
    df = df.drop_duplicates("key", keep="last")
    for sch, g in df.groupby("schema"):
        out[str(sch)] = g.reset_index(drop=True)
    return out


def _strata(r: dict, k: str):
    s = r.get("strata") or {}
    return s.get(k)


def load_items(
    runs_dir: str, routes_by_image: Dict[str, List[str]], warnings: List[str]
) -> pd.DataFrame:
    prep = os.path.join(runs_dir, "prep")
    recs = []
    for coh_file, coh in (("COH-RAD-TEST", "RAD"), ("COH-NAT-TEST", "NAT")):
        qs = read_jsonl(os.path.join(prep, "prep_out", "cohorts", f"{coh_file}.jsonl"))
        gold_name = "gold_RAD_TEST.jsonl" if coh == "RAD" else "gold_NAT_TEST.jsonl"
        gold = {
            g["qid"]: g
            for g in read_jsonl(os.path.join(prep, "prep_sealed", gold_name))
        }
        for r in qs:
            g = gold.get(r["qid"])
            if g is None:
                raise ValueError(f"no gold for {r['qid']}")
            ds = _strata(r, "dataset")
            opts = r.get("options")
            gold_index = None
            texts = None
            if opts:
                texts = [o["text"] for o in opts]
                labels = [o["label"] for o in opts]
                if g.get("gold_label") is not None:
                    gold_index = labels.index(g["gold_label"])
            qmeta = {
                "SLAKE": _strata(r, "slake_content_type"),
                "VQA-RAD": _strata(r, "vqarad_question_type"),
            }.get(ds, _strata(r, "omni_question_type"))
            recs.append(
                {
                    "qid": r["qid"],
                    "image_id": r["neutral_id"],
                    "cohort": coh,
                    "dataset": ds,
                    "group": _strata(r, "native_group") if coh == "NAT" else ds,
                    "cluster_id": r["cluster_id"],
                    "answer_type": r.get("answer_type_official"),
                    "answer_kind_schema": r.get("answer_kind"),
                    "question": r["question"],
                    "options": texts,
                    "gold": texts[gold_index]
                    if texts and gold_index is not None
                    else g["gold_answer"],
                    "gold_index": gold_index,
                    "qtype_meta": qmeta,
                    "routes": routes_by_image.get(r["neutral_id"]),
                }
            )
    items = pd.DataFrame(recs)
    if items["qid"].duplicated().any():
        raise ValueError("duplicate qid across cohorts")
    broken = set()
    bp = os.path.join(prep, "prep_out", "reports", "broken_images.json")
    if os.path.exists(bp):
        b = read_json(bp)
        broken = set(
            b if isinstance(b, list) else b.get("neutral_ids", b.get("images", []))
        )
    items["image_broken"] = items["image_id"].isin(broken)
    items["weak_group"] = derive_weak(runs_dir, items, warnings)
    if items["routes"].isna().any():
        warnings.append(
            (
                f"{int(items['routes'].isna().sum())}"
                " questions have no route (no OBS record found)"
            )
        )
    return items


def derive_weak(
    runs_dir: str, items: pd.DataFrame, warnings: List[str]
) -> List[Optional[bool]]:
    """Weak-source routing groups: CT neck, MR abdomen, MR chest images (radiology).

    SCHEMA-v1 has no flag and derives the flag from route metadata; MR chest images are
    weak-reference cases.
    """
    meta_p = os.path.join(runs_dir, "prep", "prep_out", "route_metadata.json")
    meta = read_json(meta_p).get("records", {}) if os.path.exists(meta_p) else {}
    if not meta:
        warnings.append(
            (
                "route_metadata.json missing: S-WEAK uses routes "
                "only (MR chest not captured)"
            )
        )
    out = []
    for img, routes, coh in zip(items["image_id"], items["routes"], items["cohort"]):
        rl = routes or []
        m = meta.get(img, {})
        mr_chest = str(m.get("modality", "")).upper().startswith("MR") and any(
            (
                "chest" in str(x).lower() or "lung" in str(x).lower()
                for x in m.get("regions") or []
            )
        )
        out.append(
            coh == "RAD" and (bool({"RT-CT-NECK", "RT-MR-ABD"} & set(rl)) or mr_chest)
        )
    return out


def answers_table(recs: pd.DataFrame) -> pd.DataFrame:
    """Normalize recorded answer text and label probabilities.

    ANS-v1 -> (cell, qid, order, raw_output, gen_error, int_missing, label_logprobs).
    """

    def order(v):
        return 0 if is_missing_scalar(v) or v == "-" else int(v)

    def lp(v):
        if not isinstance(v, dict) or v.get("finite") is False:
            return None
        return v.get("lp")

    st = recs.get("status", pd.Series("ok", index=recs.index)).fillna("ok")
    return pd.DataFrame(
        {
            "cell": recs["cell"].map(canonical_cell),
            "qid": recs["qid"],
            "order": recs["order"].map(order) if "order" in recs else 0,
            "raw_output": pd.Series(
                [
                    None if s != "ok" or is_missing_scalar(t) else t
                    for s, t in zip(st, recs["text"])
                ],
                index=recs.index,
                dtype=object,
            ),
            "gen_error": st == "error",
            "int_missing": st == "skipped_int_missing",
            "label_logprobs": recs["logprob"].map(lp) if "logprob" in recs else None,
        }
    )


def observations_table(recs: pd.DataFrame) -> pd.DataFrame:
    if recs is None or recs.empty:
        return pd.DataFrame(
            columns=[
                "observer",
                "arm",
                "image_id",
                "text",
                "n_output_tokens",
                "truncated",
                "obs_error",
                "routes",
                "route_key",
            ]
        )
    return pd.DataFrame(
        {
            "observer": recs["observer"].map(lambda o: OBSERVER_MAP.get(o, o)),
            "arm": recs["arm"].str.replace("ARM-", "", regex=False),
            "image_id": recs["neutral_id"],
            "cohort": recs.get("cohort"),
            "text": recs["text"].fillna(""),
            "n_output_tokens": recs.get("n_output_tokens"),
            "truncated": recs["truncated"].map(bool) if "truncated" in recs else False,
            "obs_error": recs.get("status", "ok").eq("obs_error"),
            "routes": recs.get("routes"),
            "route_key": recs.get("route_key"),
        }
    )


def routes_by_image(obs: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for img, rl, rk in zip(obs["image_id"], obs["routes"], obs["route_key"]):
        if img in out:
            continue
        if isinstance(rl, list) and rl:
            out[img] = list(rl)
        elif isinstance(rk, str) and rk:
            out[img] = rk.split("+")
    return out


def _artifact_run_mode(path: str) -> str:
    """Resolve owner MODE from shard provenance, never from a directory name.

    SCHEMA-v1 records and run_config declare MODE. Accept either source (legacy
    synthetic fixtures have records only), but require all declarations to agree.
    A declared artifact digest must also match, including on ignored PILOT runs.
    """
    run_dir = os.path.dirname(os.path.dirname(path))
    modes, run_ids = (set(), set())
    digest_field = {"corruption_plan.json": "corruption_plan_sha256"}.get(
        os.path.basename(path)
    )
    for shard in sorted(glob.glob(os.path.join(run_dir, "shard_*"))):
        cfg_path = os.path.join(shard, "run_config.json")
        records_path = os.path.join(shard, "records.jsonl")
        declarations = []
        if os.path.isfile(cfg_path):
            cfg = read_json(cfg_path)
            declarations.append(cfg)
            if digest_field and cfg.get(digest_field) is not None:
                if cfg[digest_field] != AX1.file_sha256(path):
                    raise ValueError(f"artifact SHA mismatch: {path} ({cfg_path})")
        if os.path.isfile(records_path):
            declarations.extend(read_jsonl(records_path))
        for declaration in declarations:
            mode = declaration.get("mode")
            if mode not in ("TEST", "PILOT"):
                raise ValueError(
                    f"missing/invalid artifact MODE provenance: {path} ({shard})"
                )
            modes.add(mode)
            if declaration.get("run_id") is not None:
                run_ids.add(declaration["run_id"])
    if len(modes) != 1 or len(run_ids) > 1:
        raise ValueError(f"missing/conflicting artifact run provenance: {path}")
    return modes.pop()


def find_artifact(runs_dir: str, name: str, warnings: List[str]):
    """TEST artifacts only; missing -> None, equal JSON copies -> one object.

    REFERENCE_PANEL scope/plans deliberately use load_ax1_scope's separate byte/SHA
    manifest and record-binding checks instead of JSON-equivalence checks.
    """
    paths = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "artifacts", name))):
        if _artifact_run_mode(path) == "TEST":
            paths.append(path)
        else:
            warnings.append(f"ignored PILOT artifact: {path}")
    if not paths:
        return None
    objs = [read_json(p) for p in paths]
    if any(
        (
            json.dumps(o, sort_keys=True) != json.dumps(objs[0], sort_keys=True)
            for o in objs[1:]
        )
    ):
        raise ValueError(f"conflicting copies of {name}: {paths}")
    return objs[0]


def load_design_state(runs_dir: str) -> dict:
    """Read the study design metadata; an absent file selects the full design.

    Two accepted layouts: * {"design": "FULL"|"FB"|"LATE"|"FB_LATE",
    "configuration_ref": "configuration"}; * {"branch": "MAIN",
    "fb": true, "u3": "A", "u4": "A", "rq2": "AX1_SLAKE"|"NONE", "configuration_ref":
    "MAIN"} The main design excludes the medical model and extended intervention branch.
    Returned dict always has "design" and "rq2" ("FULL" unless MAIN).
    """
    p = os.path.join(runs_dir, "design_state.json")
    d = read_json(p) if os.path.exists(p) else {"design": "FULL"}
    return parse_design_state(d)


def parse_design_state(d: dict) -> dict:
    d = dict(d)
    if "branch" in d:
        if not isinstance(d["branch"], str) or not d["branch"].strip():
            raise ValueError("study branch must be a nonempty identifier")
        bad = [k for k, v in (("fb", True), ("u3", "A"), ("u4", "A")) if d.get(k) != v]
        if bad:
            raise ValueError(
                f"MAIN design_state must have fb=true, u3='A', u4='A' (wrong: {bad})"
            )
        if d.get("rq2") not in MAIN_RQ2_LEVELS:
            raise ValueError(f"MAIN rq2 must be one of {MAIN_RQ2_LEVELS}")
        d["branch"] = "MAIN"
        d["configuration_ref"] = "MAIN"
        d["design"] = "MAIN"
        return d
    if d.get("design") not in DESIGNS or d.get("design") == "MAIN":
        raise ValueError(
            (
                "design_state.design must be one of "
                f"{DESIGNS[:-1]}"
                " (MAIN uses the 'branch' layout)"
            )
        )
    d.setdefault("rq2", "FULL")
    return d


def load_routes_table(protocol_dir: str) -> dict:
    with open(
        os.path.join(protocol_dir, "prompts", "specs", "routes.json"), encoding="utf-8"
    ) as f:
        return json.load(f)["routes"]


_ITEM_LINE = re.compile("^(\\d+)\\.\\s+(.*)$")


def route_items(protocol_dir: str, arm: str, route: str) -> List[str]:
    """Read item names from a fixed observer prompt.

    Item names of a locked generated observer prompt
    (resources/protocol/prompts/generated, read-only).
    """
    p = os.path.join(protocol_dir, "prompts", "generated", f"OBS-{arm}__{route}.txt")
    if not os.path.exists(p):
        return []
    items, on = ([], False)
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("Items:"):
                on = True
                continue
            if on:
                m = _ITEM_LINE.match(line)
                if m:
                    items.append(m.group(2))
                elif items:
                    break
    return items


def load_run_params(runs_dir: str) -> dict:
    """Load the recorded runtime parameters.

    Read calibrated run parameters from the run directory, then the project
    configuration.
    """
    for p in (
        os.path.join(runs_dir, "run_params.json"),
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "configs",
            "run_params.json",
        ),
    ):
        if os.path.exists(p):
            return read_json(p)
    return {}


def judge_roles(run_params: dict) -> Dict[str, str]:
    """Map recorded judge identifiers to the specified J1/J2 roles.

    `judge_fallback` may be null, "J1"/"J2" (the role J-ALT replaces) or {"replaces":
    "J1"|"J2"}.
    """
    fb = run_params.get("judge_fallback")
    if isinstance(fb, dict):
        fb = fb.get("replaces")
    roles = {"J1": "J1", "J2": "J2"}
    if fb in ("J1", "J2"):
        roles = {k: v for k, v in roles.items() if k != fb}
        roles["J-ALT"] = fb
    return roles


def load_headers(runs_dir: str) -> Dict[str, str]:
    """Read the rendered routing header for each image.

    Rendered route header per image from observer_prompt_manifest.json (R-2 / OQ-19).
    """
    p = os.path.join(runs_dir, "prep", "prep_out", "observer_prompt_manifest.json")
    if not os.path.exists(p):
        return {}
    out = {}
    for k, v in (read_json(p).get("entries") or {}).items():
        nid = k.split("|", 1)[1] if "|" in k else k
        if v.get("header"):
            out.setdefault(nid, v["header"])
    return out


def load_all(
    runs_dir: str, required_cells_fn: Optional[Callable[[str], List[str]]] = None
) -> Tuple[dict, List[str]]:
    """Load experiment records and reference labels for analysis.

    Check the study design and required conditions before opening reference labels.
    """
    warnings: List[str] = []
    recs = load_records(runs_dir, warnings)
    if "ANS-v1" not in recs:
        raise ValueError(
            "no ANS-v1 records found under runs/<run_id>/shard_*/records.jsonl"
        )
    design_state = load_design_state(runs_dir)
    answers = answers_table(recs["ANS-v1"])
    if required_cells_fn is not None:
        missing = [
            c
            for c in required_cells_fn(design_state["design"])
            if c not in set(answers["cell"])
        ]
        if missing:
            raise SystemExit(
                (
                    "required cells missing for design "
                    f"{design_state['design']}"
                    ": "
                    f"{missing}"
                    " (reference labels have not been opened)"
                )
            )
    obs = observations_table(recs.get("OBS-v1"))
    ax1_scope = load_ax1_scope(runs_dir, recs.get("JUDGE-v1", pd.DataFrame()))
    items = load_items(runs_dir, routes_by_image(obs), warnings)
    headers = load_headers(runs_dir)
    if headers:
        items["route_header"] = items["image_id"].map(headers)
        if items["route_header"].isna().any():
            warnings.append(
                (
                    f"{int(items['route_header'].isna().sum())}"
                    " questions without manifest header; rebuilt"
                )
            )
            items = items.drop(columns="route_header")
    run_params = load_run_params(runs_dir)
    roles = judge_roles(run_params)
    judge = recs.get("JUDGE-v1", pd.DataFrame())
    if not judge.empty:
        unknown = sorted(set(judge["judge"]) - set(roles))
        if unknown:
            warnings.append(f"judge ids without a role (ignored in RQ2): {unknown}")
        judge = judge[judge["judge"].isin(roles)].copy()
        judge["judge_id"] = judge["judge"]
        judge["judge"] = judge["judge"].map(roles)
    probe_p = os.path.join(runs_dir, "prep", "prep_out", "judge", "jprobe_set.json")
    data = {
        "items": items,
        "answers": answers,
        "observations": obs,
        "edits": recs.get("EDIT-v1", pd.DataFrame()),
        "judge": judge,
        "judge_roles": roles,
        "corruption_plan": find_artifact(runs_dir, "corruption_plan.json", warnings),
        "jprobe_set": read_json(probe_p) if os.path.exists(probe_p) else None,
        "design_state": design_state,
        "ax1_scope": ax1_scope,
    }
    return (data, warnings)


def load_ax1_scope(runs_dir: str, judge: pd.DataFrame):
    """Validate the independent reference-panel manifest.

    Validate the reference-panel manifest independently of answer and intervention
    inputs.
    """

    def candidates(name):
        return sorted(glob.glob(os.path.join(runs_dir, "*", "artifacts", name))) + [
            p for p in [os.path.join(runs_dir, "locks", name)] if os.path.isfile(p)
        ]

    def one(name):
        paths = candidates(name)
        if not paths or len({AX1.file_sha256(p) for p in paths}) != 1:
            raise ValueError(f"REFERENCE_PANEL missing/conflicting artifact: {name}")
        return paths[0]

    marked = (
        not judge.empty
        and "ax1_scope_sha256" in judge
        and judge["ax1_scope_sha256"].notna().any()
    )
    if not candidates("ax1_scope.json") and (not marked):
        return None
    scope_path = one("ax1_scope.json")
    scope, scope_sha = (read_json(scope_path), AX1.file_sha256(scope_path))
    files = {"artifacts/ax1_scope.json": scope_path}
    plans = {}
    for j in ("J1", "J2"):
        p = one(f"ax1_plan_{j}.json")
        files[f"artifacts/ax1_plan_{j}.json"] = p
        plans[j] = (read_json(p), AX1.file_sha256(p))
    prep = os.path.join(runs_dir, "prep", "prep_out")
    source_paths = {
        "panel_rq2.json": "panel_rq2.json",
        "annotations_slake_test.json": "judge/annotations_slake_test.json",
        "jprobe_set.json": "judge/jprobe_set.json",
    }
    for rel in source_paths.values():
        files[f"prep_out/{rel}"] = os.path.join(prep, rel)
    _verify_reference_manifest(runs_dir, files)
    panel, ann, probe = [
        read_json(os.path.join(prep, source_paths[k])) for k in source_paths
    ]
    cohort_ids = {
        r["neutral_id"]
        for r in read_jsonl(os.path.join(prep, "cohorts", "COH-RAD-TEST.jsonl"))
    }
    AX1.validate_scope(
        scope,
        panel,
        ann,
        *(AX1.file_sha256(os.path.join(prep, rel)) for rel in source_paths.values()),
        probe["probes"],
        cohort_ids,
    )
    for j, (plan, _) in plans.items():
        AX1.validate_plan(
            plan,
            scope_sha,
            scope["source_sha256"]["jprobe_set.json"],
            j,
            scope["eligible_ids"],
            {p["probe_id"]: p["neutral_id"] for p in probe["probes"]},
        )
    records = judge.to_dict("records") if not judge.empty else []
    AX1.validate_record_scopes(records, scope, scope_sha, plans)
    return scope


def _verify_reference_manifest(runs_dir: str, files: dict) -> None:
    """Validate reference-panel files against their manifest.

    Match manifests by their required file entries.

    Archived experiments may use a differently named manifest. The candidate must
    contain every required logical path, and every matching candidate must verify the
    exact bytes of all files. A partial manifest does not cover the panel.
    """
    from medvqa.common import LockError, parse_sha256_file, verify_locked_inputs

    candidates = sorted(
        glob.glob(os.path.join(runs_dir, "locks", "locked_inputs*.sha256"))
    )
    matching = [p for p in candidates if set(files) <= set(parse_sha256_file(p))]
    if not matching:
        raise LockError("no locked-input manifest covers the complete reference panel")
    for manifest in matching:
        verify_locked_inputs(manifest, files, "TEST")
