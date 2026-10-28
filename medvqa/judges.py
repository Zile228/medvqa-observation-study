"""Statement-level and answer-level evaluation prompts and record validation."""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from medvqa.common import SEED, sha256_text
from medvqa.interventions import LINE_RE
from medvqa.judge_scope import PLAN_VERSION, digest_json

GENERIC_UNSUPPORTED = [
    "whole-examination normality",
    "size in physical units without a scale",
    "change over time",
]
EVASION = {"not visible", "not assessable"}
MAX_TOKENS = {"J-AX1": 400, "J-AX2": 200, "J-AX3": 20, "J-OPEN": 20}
STRIP_ONE_JSON_FENCE = True
JUDGE_JSON_FENCE_REGEX = "^```(?:json)?[ \\t]*\\r?\\n(.*?)\\r?\\n```$"
FENCE_RE = re.compile(JUDGE_JSON_FENCE_REGEX, re.S)
try:
    from medvqa.analysis import constants as _B

    if (_B.JUDGE_JSON_FENCE_REGEX, _B.JUDGE_STRIP_ONE_JSON_FENCE) != (
        JUDGE_JSON_FENCE_REGEX,
        STRIP_ONE_JSON_FENCE,
    ):
        raise RuntimeError(
            (
                "judge JSON rule differs from "
                "analysis/constants.py (validation): sync both "
                "sides"
            )
        )
except ImportError:
    pass
AX1_LABEL_SET = ("supported", "contradicted", "not_verifiable")


class AnnotationCoverageError(ValueError):
    """J-AX1 cannot run without a nonempty source reference for every selected image."""

    def __init__(self, report):
        self.report = report
        super().__init__(
            (
                "J-AX1 annotation preflight ("
                f"{report['mode']}"
                ") failed before model load: missing="
                f"{report['missing_ids']}"
                ", unusable="
                f"{report['unusable_ids']}"
                ", absent_from_cohort="
                f"{report['absent_from_cohort']}"
                ". Keep the panel unchanged; resolve source "
                "references and locks before retry."
            )
        )


def validate_ax1_annotations(annotations, required_ids, mode, cohort_ids=None):
    """Check annotation coverage before loading the judge model.

    PILOT callers must supply SLAKE IDs from cohort metadata, not annotation keys.
    Probe references must also be included in required_ids by the caller.
    """
    if mode not in ("TEST", "PILOT"):
        raise ValueError(f"unknown judge mode {mode!r}")
    required = sorted(set(required_ids))
    missing, unusable = ([], [])
    for nid in required:
        if nid not in annotations:
            missing.append(nid)
            continue
        ref = annotations[nid]
        if (
            not isinstance(ref, dict)
            or not isinstance(ref.get("text"), str)
            or (not ref["text"].strip())
            or (not isinstance(ref.get("entries"), list))
            or (not ref["entries"])
        ):
            unusable.append(nid)
    report = {
        "mode": mode,
        "required_images": len(required),
        "missing_ids": missing,
        "unusable_ids": unusable,
        "absent_from_cohort": sorted(set(required) - set(cohort_ids))
        if cohort_ids is not None
        else [],
    }
    if missing or unusable or report["absent_from_cohort"]:
        raise AnnotationCoverageError(report)
    return report


class JudgePrompts:
    def __init__(self, md_path):
        md = Path(md_path).read_text(encoding="utf-8")
        blocks = re.findall("```\\n(.*?)\\n```", md, re.S)

        def pick(start):
            return next((b for b in blocks if b.startswith(start)))

        self.ax1 = pick("You compare statements")
        self.ax2 = pick("Below is a list of requested")
        self.ax3 = pick("Here is a question about a medical image")
        self.open = pick("A question about a medical image has a reference")

    def render_ax1(self, annotation_text, statements):
        numbered = "\n".join((f"{i}. {s}" for i, s in enumerate(statements, 1)))
        return self.ax1.replace("{ANNOTATION_TEXT}", annotation_text).replace(
            "{NUMBERED_STATEMENTS}", numbered
        )

    def render_ax2(self, item_list, unsupported, observation):
        return (
            self.ax2.replace("{UNSUPPORTED_LIST}", "; ".join(unsupported))
            .replace("{GRD_ITEM_LIST_FOR_ROUTE}", item_list)
            .replace("{OBSERVATION}", observation)
        )

    def render_ax3(self, question, observation):
        return self.ax3.replace("{QUESTION}", question).replace(
            "{OBSERVATION}", observation
        )

    def render_open(self, question, gold, prediction):
        return (
            self.open.replace("{QUESTION}", question)
            .replace("{GOLD}", gold)
            .replace("{PREDICTION}", prediction)
        )


def split_statements(observation):
    """Split observations into statements and skip lines without content."""
    stmts, evasion = ([], [])
    for ln in observation.strip().split("\n"):
        if not ln.strip():
            continue
        m = LINE_RE.match(ln)
        if m:
            name, status, obs = (
                m.group(2).strip(),
                m.group(3).strip().lower(),
                m.group(4).strip(),
            )
            obs_core = obs.strip(" .").lower()
            if status in EVASION and (
                not obs_core
                or obs_core in EVASION
                or obs_core in {"-", "—", "n/a", "none"}
            ):
                evasion.append(ln.strip())
                continue
            stmts.append(f"{name}: {obs}")
        else:
            stmts += [s for s in re.split("(?<=[.!?])\\s+", ln.strip()) if s]
    return (stmts, evasion)


def unsupported_for_routes(field_trace, routes):
    out = []
    for r in routes:
        for item in field_trace["routes"].get(r, []):
            for f in item["fields"]:
                for u in f.get("unsupported_inferences") or []:
                    if u not in out:
                        out.append(u)
    for g in GENERIC_UNSUPPORTED:
        if g not in out:
            out.append(g)
    return out


def shuffled(units, tag, seed=SEED):
    """Shuffle judge calls across models and observation arms.

    Seeded order of judge calls (blinding: observation order mixed across
    observers/arms).
    """
    units = sorted(units, key=lambda u: u["key"])
    random.Random(f"{seed}|{tag}").shuffle(units)
    return units


def bias_subset(ax1_units, frac=0.1, seed=SEED):
    """Select support-judge calls for reversed-statement checks.

    10% of J-AX1 observation calls, stratified by arm, re-run with reversed statement
    order.
    """
    by_arm = {}
    for u in ax1_units:
        if not u.get("is_probe_block"):
            by_arm.setdefault(u["arm"], []).append(u)
    pick = []
    for arm, us in sorted(by_arm.items()):
        us = sorted(us, key=lambda u: sha256_text(f"{seed}|bias|{u['key']}"))
        pick += us[: max(1, round(frac * len(us)))] if us else []
    return pick


def build_ax1_units(slake_ids, probes, judge, observers, arms, get_obs):
    """Build support, probe and statement-order check units.

    Original main/probe/bias algorithm, shared with the CPU plan-only tool
    (REFERENCE_PANEL).
    """
    units = []
    for n in slake_ids:
        for o in observers:
            for a in arms:
                r = get_obs(o, a, n)
                if r is None:
                    continue
                st, ev = (
                    split_statements(r["text"]) if r["status"] == "ok" else ([], [])
                )
                stm = [
                    {
                        "sid": f"{r['key']}#{i}",
                        "obs_key": r["key"],
                        "is_probe": False,
                        "probe_id": None,
                        "text": t,
                    }
                    for i, t in enumerate(st, 1)
                ]
                units.append(
                    {
                        "key": f"J|{judge}|J-AX1|{r['key']}|main",
                        "task": "J-AX1",
                        "arm": a,
                        "nid": n,
                        "obs_key": r["key"],
                        "stm": stm,
                        "skipped_evasion": ev,
                        "obs_status": r["status"],
                    }
                )
    by_img = defaultdict(list)
    for p in probes:
        by_img[p["neutral_id"]].append(p)
    for n, ps in sorted(by_img.items()):
        for b in range(0, len(ps), 10):
            blk = ps[b : b + 10]
            stm = [
                {
                    "sid": p["probe_id"],
                    "obs_key": None,
                    "is_probe": True,
                    "probe_id": p["probe_id"],
                    "text": p["statement"],
                }
                for p in blk
            ]
            units.append(
                {
                    "key": f"J|{judge}|J-AX1|PROBE-{n}-{b // 10}|main",
                    "task": "J-AX1",
                    "is_probe_block": True,
                    "nid": n,
                    "obs_key": None,
                    "stm": stm,
                    "skipped_evasion": [],
                    "obs_status": "ok",
                }
            )
    for u in bias_subset(units):
        units.append(
            dict(
                u,
                key=u["key"][: -len("|main")] + "|reversed",
                stm=list(reversed(u["stm"])),
                variant="reversed",
            )
        )
    return units


def build_ax1_plan(units, observations, judge, scope_sha256, probe_sha256):
    """Record support-judge unit metadata without statement text.

    Metadata only: preserve shuffled order, skip/call counts and statement identities,
    never text/labels.
    """
    metadata = []
    used = sorted({u["obs_key"] for u in units if u.get("obs_key")})
    for u in units:
        status = (
            "obs_error_skipped"
            if u["obs_status"] != "ok"
            else "no_statements"
            if not u["stm"]
            else "needs_generation"
        )
        kind = (
            "probe"
            if u.get("is_probe_block")
            else "reversed"
            if u.get("variant")
            else "main"
        )
        observer = u["obs_key"].split("|")[1] if u.get("obs_key") else None
        metadata.append(
            {
                "key": u["key"],
                "neutral_id": u["nid"],
                "kind": kind,
                "observer": observer,
                "arm": u.get("arm"),
                "planned_status": status,
                "n_statements": len(u["stm"]),
                "n_skipped_evasion": len(u["skipped_evasion"]),
                "sids": [s["sid"] for s in u["stm"]],
            }
        )
    sources = [
        (k, observations[k]["status"], sha256_text(observations[k].get("text") or ""))
        for k in used
    ]
    return {
        "version": PLAN_VERSION,
        "configuration_ref": "REFERENCE_PANEL",
        "judge": judge,
        "scope_sha256": scope_sha256,
        "probe_sha256": probe_sha256,
        "observation_digest": digest_json(sources),
        "units": metadata,
        "units_sha256": digest_json(metadata),
        "counts_by_kind": dict(Counter((u["kind"] for u in metadata))),
        "counts_by_status": dict(Counter((u["planned_status"] for u in metadata))),
    }


def strip_one_fence(text):
    s = text.strip()
    m = FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def json_valid(task, text, strip_fence=None, ids=None):
    """Check judge JSON format for diagnostic reporting only.

    DIAGNOSTIC: schema-valid JSON under the NQ-1 rule (default STRIP_ONE_JSON_FENCE).
    Not used for scoring.
    """
    if not isinstance(text, str):
        return False
    strip_fence = STRIP_ONE_JSON_FENCE if strip_fence is None else strip_fence
    return json_valid_strict(task, strip_one_fence(text) if strip_fence else text, ids)


AX1_STATEMENT_FIELDS = ("id", "sid", "obs_key", "is_probe", "probe_id", "text")


def assert_ax1_refs(refs):
    """Validate statement references and skipped-evasion fields.

    Validate statement identifiers and the skipped-evasion fields in judge records.
    """
    assert isinstance(refs.get("skipped_evasion"), list), (
        "skipped_evasion must be a list"
    )
    for st in refs["statements"]:
        missing = [f for f in AX1_STATEMENT_FIELDS if f not in st]
        assert not missing, f"J-AX1 statement missing {missing}"
        assert (
            st["is_probe"] == (st["probe_id"] is not None)
            and (st["obs_key"] is None) == st["is_probe"]
        ), "probe/obs_key mismatch"
    return True


def json_valid_strict(task, text, ids=None):
    """Check strict JSON parsing and task format diagnostically.

    DIAGNOSTIC: strict json.loads on text.strip() + task schema (mirrors
    analysis/metrics._schema_ok).
    """
    if not isinstance(text, str):
        return False
    try:
        obj = json.loads(text.strip())
    except (ValueError, TypeError):
        return False
    if not isinstance(obj, dict):
        return False
    if task == "J-AX1":
        labs = obj.get("labels")
        if not isinstance(labs, list):
            return False
        try:
            got = {int(e["id"]): e["label"] for e in labs}
        except (KeyError, TypeError, ValueError):
            return False
        if len(got) != len(labs) or any((v not in AX1_LABEL_SET for v in got.values())):
            return False
        return ids is None or set(got) == set(ids)
    if task == "J-AX2":
        ia = obj.get("items_addressed")
        return (
            isinstance(ia, int)
            and (not isinstance(ia, bool))
            and (obj.get("diagnosis") in ("yes", "no"))
            and (obj.get("unsupported_claim") in ("yes", "no"))
        )
    if task == "J-AX3":
        r = obj.get("relevance")
        return isinstance(r, int) and (not isinstance(r, bool)) and (r in (0, 1, 2))
    if task == "J-OPEN":
        return obj.get("verdict") in ("correct", "incorrect")
    return False
