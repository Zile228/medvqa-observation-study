"""Paired observation interventions and deterministic donor selection."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile("^\\s*(\\d+)\\.\\s*(.+?)\\s*\\|\\s*(.+?)\\s*\\|\\s*(.*)$")
STATUSES_VISIBLE = ("visible", "partly visible")
PARA_KEYWORDS = (
    "left",
    "right",
    "image-left",
    "image-right",
    "upper",
    "lower",
    "anterior",
    "posterior",
    "medial",
    "lateral",
    "no",
    "not",
    "without",
    "absent",
)
LAT_PAIRS = [
    ("left", "right"),
    ("image-left", "image-right"),
    ("upper", "lower"),
    ("anterior", "posterior"),
    ("medial", "lateral"),
]
SEV_PAIRS = [
    ("small", "large"),
    ("mild", "marked"),
    ("slight", "severe"),
    ("narrow", "wide"),
    ("thin", "thick"),
    ("few", "many"),
]
CORR_TYPES = ("PRES", "LAT", "SEV")
CORR_SYNONYMS = {}


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _count_word(text, w):
    return len(re.findall("(?<![\\w-])" + re.escape(w) + "(?![\\w-])", text, re.I))


def parse_lines(observation):
    """Physical lines of the stripped observation with their parsed fields (or None)."""
    out = []
    for i, ln in enumerate(observation.strip().split("\n"), 1):
        m = LINE_RE.match(ln)
        out.append(
            {
                "line_no": i,
                "raw": ln,
                "item_no": int(m.group(1)) if m else None,
                "name": m.group(2) if m else None,
                "status": m.group(3).strip().lower() if m else None,
                "obs": m.group(4) if m else None,
            }
        )
    return out


def shuf_assign(neutral_ids, route_keys, modality_family, cohort_of):
    """Donor map nid -> nid | None (stratified donor matching).

    Tier 1 = (cohort, route_key); RT-FALLBACK counts as one route. Sort by
    sha256(nid+'SHUF-v1'), donor = next image (cyclic shift 1). Tier 2, only for images
    alone in tier 1 = (cohort, modality family CT/MR/X-ray/skin; RT-MAMMO is X-ray;
    FALLBACK uses route_metadata modality); donor = the next image after it in tier-2
    order. NO third tier: an image still alone in tier 2 gets donor None -> int_missing
    for INT-SHUF / INT2-SHUF (reported).
    """

    def order(n):
        return _sha(n + "SHUF-v1")

    tiers = defaultdict(list)
    for n in neutral_ids:
        tiers[cohort_of[n], route_keys[n]].append(n)
    donor, tier_used = ({}, {})
    for (coh, rk), members in tiers.items():
        if len(members) >= 2:
            ms = sorted(members, key=order)
            for i, n in enumerate(ms):
                donor[n] = ms[(i + 1) % len(ms)]
                tier_used[n] = "route"
    for n in neutral_ids:
        if n in donor:
            continue
        coh = cohort_of[n]
        pool = [
            m
            for m in neutral_ids
            if cohort_of[m] == coh
            and modality_family[m] == modality_family[n]
            and (modality_family[n] != "unknown")
        ]
        if len(pool) >= 2:
            ms = sorted(pool, key=order)
            donor[n] = ms[(ms.index(n) + 1) % len(ms)]
            tier_used[n] = "modality"
        else:
            donor[n] = None
            tier_used[n] = "none_int_missing"
    assert all((donor[n] != n for n in donor)), "self-assignment"
    return (donor, tier_used)


def shuf_table(neutral_ids, route_keys, modality_family, cohort_of):
    """Create the fixed image-to-donor assignment table.

    Precomputed donor table (only depends on route_metadata); written by prep stage B
    and locked.
    """
    donor, tier = shuf_assign(neutral_ids, route_keys, modality_family, cohort_of)
    from collections import Counter

    return {
        "version": "SHUF-DONORS-v2",
        ("rule"): (
            "SHUFFLE_RULE: tier1 (cohort, route_key); tier2 "
            "(cohort, modality family); no tier 3"
        ),
        "donors": {n: donor[n] for n in sorted(donor)},
        "tiers": {n: tier[n] for n in sorted(tier)},
        "tier_counts": dict(Counter(tier.values())),
        "int_missing": sorted((n for n in donor if donor[n] is None)),
    }


def empty_observation(bank, routes):
    return bank.compose_empty(routes)


def gray_image(image):
    """RGB(128,128,128), same size as the original (INT-BLANK, INT-BLANK0)."""
    from PIL import Image

    return Image.new("RGB", image.size, (128, 128, 128))


def feasible_lines(observation):
    lines = [line for line in parse_lines(observation) if line["item_no"] is not None]
    lat = [
        line
        for line in lines
        if any((_count_word(line["obs"], w) for p in LAT_PAIRS for w in p))
    ]
    sev = [
        line
        for line in lines
        if any((_count_word(line["obs"], w) for p in SEV_PAIRS for w in p))
    ]
    pres = [line for line in lines if line["status"] in STATUSES_VISIBLE] or [
        line for line in lines if line["status"] == "not visible"
    ]
    return {"PRES": pres, "LAT": lat, "SEV": sev}


def corruption_plan(observations):
    """observations: {neutral_id: GRD observation text}.

    Deterministic balanced assignment.
    """
    counts = {t: 0 for t in CORR_TYPES}
    plan, none = ({}, [])
    for nid in sorted(observations, key=lambda n: _sha(n + "CORR-v1")):
        feas = feasible_lines(observations[nid])
        types = [t for t in CORR_TYPES if feas[t]]
        if not types:
            none.append(nid)
            continue
        t = min(types, key=lambda x: (counts[x], CORR_TYPES.index(x)))
        counts[t] += 1
        line = min(feas[t], key=lambda line: _sha(nid + str(line["item_no"])))
        plan[nid] = {
            "edit_type": t,
            "item_no": line["item_no"],
            "line_no": line["line_no"],
            "item_name": (line["name"] or "").strip(),
            "synonyms": list(CORR_SYNONYMS.get((line["name"] or "").strip(), [])),
            "feasible": types,
        }
    return {
        "version": "CORR-PLAN-v1",
        "assignments": plan,
        "no_feasible": none,
        "counts": counts,
    }


class EditorPrompts:
    def __init__(self, md_path):
        md = Path(md_path).read_text(encoding="utf-8")
        blocks = re.findall("```\\n(.*?)\\n```", md, re.S)
        self.para = next((b for b in blocks if b.startswith("Rewrite the text below")))
        self.corr = next((b for b in blocks if b.startswith("Edit the text below")))
        self.edit_desc = {}
        for t in CORR_TYPES:
            m = re.search("^- " + t + ": `([^`]+)`", md, re.M)
            self.edit_desc[t] = m.group(1)
        for ph in ("{OBSERVATION}",):
            assert ph in self.para and ph in self.corr

    def para_prompt(self, observation):
        return self.para.replace("{OBSERVATION}", observation)

    def corr_prompt(self, observation, edit_type, line_no):
        return (
            self.corr.replace("{EDIT_TYPE_DESCRIPTION}", self.edit_desc[edit_type])
            .replace("{LINE_NO}", str(line_no))
            .replace("{OBSERVATION}", observation)
        )


def _words(s):
    return len(s.split())


def para_checks(observation, output):
    o, n = (parse_lines(observation), parse_lines(output))
    c = {"n_lines": len(o) == len(n)}
    c["items"] = c["n_lines"] and all(
        (
            a["item_no"] == b["item_no"]
            and (a["name"] or "").strip() == (b["name"] or "").strip()
            for a, b in zip(o, n)
            if a["item_no"] is not None
        )
    )
    c["status"] = c["n_lines"] and all(
        (a["status"] == b["status"] for a, b in zip(o, n) if a["item_no"] is not None)
    )
    c["keywords"] = c["n_lines"] and all(
        (
            _count_word(a["raw"], w) == _count_word(b["raw"], w)
            for a, b in zip(o, n)
            for w in PARA_KEYWORDS
        )
    )
    r = _words(output) / max(1, _words(observation))
    c["length"] = 0.85 <= r <= 1.15
    return c


def _strip_fence(t):
    t = t.strip()
    m = re.match("^```(?:json)?\\s*\\n(.*)\\n```$", t, re.S)
    return m.group(1) if m else t


def corr_checks(observation, output, line_no, edit_type):
    c = {"json": False}
    try:
        obj = json.loads(_strip_fence(output))
        ok = isinstance(obj, dict) and all(
            (
                isinstance(obj.get(k), str)
                for k in ("edited_text", "original_span", "new_span")
            )
        )
        c["json"] = ok
    except Exception:
        ok = False
    if not ok:
        return (c, None)
    o = observation.strip().split("\n")
    n = obj["edited_text"].strip().split("\n")
    c["n_lines"] = len(o) == len(n)
    if not c["n_lines"] or not 1 <= line_no <= len(o):
        return (c, None)
    i = line_no - 1
    c["others_identical"] = all(
        (a == b for k, (a, b) in enumerate(zip(o, n)) if k != i)
    )
    c["line_changed"] = o[i] != n[i]
    c["spans"] = (
        bool(obj["original_span"].strip())
        and obj["original_span"] in o[i]
        and bool(obj["new_span"].strip())
        and (obj["new_span"] in n[i])
    )
    if edit_type in ("LAT", "SEV"):
        pairs = LAT_PAIRS if edit_type == "LAT" else SEV_PAIRS
        both = pairs + [(b, a) for a, b in pairs]
        c["pair_in_list"] = any(
            (
                _count_word(obj["original_span"], a) and _count_word(obj["new_span"], b)
                for a, b in both
            )
        )
    c["length"] = abs(_words(n[i]) - _words(o[i])) <= 10
    return (c, obj["edited_text"].strip())
