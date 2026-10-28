"""Image routing from dataset provenance and image metadata only."""

import json
from pathlib import Path

FALLBACK = "RT-FALLBACK"
SOURCE_PREFIX = {
    "SLAKE": "SLAKE",
    "VQA-RAD": "VQA-RAD",
    "NATIVE-VISUAL": "native visual inventory",
}
NATIVE_MODALITIES = {"X-ray", "mammography", "MR", "skin"}
NATIVE_REGIONS = {
    "X-ray": {
        "chest",
        "wrist",
        "elbow",
        "humerus",
        "shoulder",
        "finger",
        "hand",
        "other",
    },
    "mammography": {"breast", "other"},
    "MR": {
        "brain",
        "abdomen",
        "knee",
        "shoulder",
        "hip",
        "ankle_hindfoot",
        "midfoot_forefoot",
        "toe_MTP_only",
        "foot_unresolved",
        "spine",
        "other",
    },
}
SKIN_ACQ = {"dermoscopy", "clinical_photo", "unresolved"}


class RouteBuildStop(RuntimeError):
    """Signal that a real modality-region pair lacks a routing rule.

    The modality-region lookup table needs an entry for the requested pair.
    """


def _source_family(rule_source):
    s = rule_source.strip()
    if s.startswith("SLAKE"):
        return "SLAKE"
    if s.startswith("VQA-RAD"):
        return "VQA-RAD"
    if s.startswith("native visual inventory"):
        return "NATIVE-VISUAL"
    return "any"


def lookup_rules(routes_spec, supplement=None):
    """Expand the fixed route lookup and optional supplement.

    Expand routes.json 'lookup' (and an optional supplement file) into a dict (family,
    modality, region)->(route, origin).
    """
    table = {}
    family = None
    for idx, rule in enumerate(routes_spec["lookup"]):
        fam = _source_family(rule["source"])
        if fam == "any":
            continue
        family = fam
        m = rule["match"]
        mod = m["modality"]
        regions = m.get("location") or m.get("region") or m.get("acquisition") or []
        for reg in regions:
            route = rule["route"]
            if "<REGION>" in route:
                route = route.replace("<REGION>", reg.upper())
            key = (family, mod, reg)
            if key in table and table[key][0] != route:
                raise ValueError(f"conflicting lookup rows for {key}")
            table[key] = (route, f"routes.json#lookup[{idx}]")
    for row in (supplement or {}).get("rows", []):
        if not row.get("reason"):
            raise ValueError("supplement rows need a reason")
        r = row["route"]
        if r != FALLBACK and r not in routes_spec["routes"]:
            raise ValueError(
                f"supplement may only map to an existing route or FALLBACK: {r}"
            )
        table[row["source_kind"], row["modality"], row["region"]] = (
            r,
            "supplement: " + row["reason"],
        )
    return table


def view(record):
    """Normalize radiology and native routing metadata to a common schema.

    Native records use modality and region/acquisition from the provenance
    specification. They are matched to the native rows of the route table.
    """
    if record is not None and "provenance_rule" in record:
        sub_label = record.get("region", record.get("acquisition"))
        return {
            "source_kind": "NATIVE-VISUAL",
            "modality": record["modality"],
            "regions": [sub_label],
            "status": "ok"
            if record["route_reason"] == "ok"
            else record["route_reason"],
        }
    return record


def record_pairs(record):
    """The (source_kind, modality, region) pairs a route_metadata record needs.

    Empty when the record goes to FALLBACK by rule (unknown / conflict / nonconstant /
    region 'other' / CK-COLOR fail).
    """
    record = view(record)
    if record.get("status") != "ok":
        return []
    out = []
    for reg in record.get("regions") or []:
        if reg in ("other", "unknown", None):
            continue
        out.append((record["source_kind"], record["modality"], reg))
    return out


def build_route_pairs(records, routes_spec, supplement=None):
    """Check 10: every actual pair must be covered.

    Returns the frozen route_pairs dict or raises RouteBuildStop.
    """
    table = lookup_rules(routes_spec, supplement)
    pairs, missing = ({}, {})
    for nid, rec in sorted(records.items()):
        for p in record_pairs(rec):
            if p in table:
                pairs.setdefault(
                    p,
                    {
                        "source_kind": p[0],
                        "modality": p[1],
                        "region": p[2],
                        "route": table[p][0],
                        "origin": table[p][1],
                        "n_images": 0,
                    },
                )
                pairs[p]["n_images"] += 1
            else:
                missing.setdefault(p, []).append(nid)
    if missing:
        lines = [
            f"{k} -> {len(v)} images (e.g. {v[0]})" for k, v in sorted(missing.items())
        ]
        raise RouteBuildStop(
            (
                "Pairs without a lookup row (add rows to "
                "route_lookup_supplement.json, only to an "
                "existing route or RT-FALLBACK, with a reason):"
                "\n  "
            )
            + "\n  ".join(lines)
        )
    return {
        "version": "ROUTE-PAIRS-v1",
        "routes_spec_id": routes_spec.get("id"),
        "route_order": route_order(routes_spec),
        "pairs": [pairs[k] for k in sorted(pairs)],
    }


def route_for(neutral_id, record, route_pairs):
    """Select a route from the fixed image metadata and routing table.

    Only neutral_id + its route_metadata record + frozen route_pairs. Never
    question/answer/path.
    """
    if record is None:
        raise KeyError(f"no route_metadata record for {neutral_id}")
    table = {
        (p["source_kind"], p["modality"], p["region"]): p["route"]
        for p in route_pairs["pairs"]
    }
    record = view(record)
    gaps = list(record.get("gaps") or [])
    if record.get("status") != "ok":
        return {
            "route_key": FALLBACK,
            "routes": [FALLBACK],
            "gaps": gaps,
            "fallback_reason": record.get("status"),
        }
    routes = []
    for p in record_pairs(record):
        if p not in table:
            raise RouteBuildStop(f"{neutral_id}: pair {p} not in route_pairs.json")
        r = table[p]
        if r == FALLBACK:
            gaps.append(f"{p[2]}: no route")
            continue
        if r not in routes:
            routes.append(r)
    for reg in record.get("regions") or []:
        if reg in ("other", "unknown", None):
            gaps.append(f"{reg}: no route")
    if not routes:
        return {
            "route_key": FALLBACK,
            "routes": [FALLBACK],
            "gaps": gaps,
            "fallback_reason": "no_route_region",
        }
    order = route_pairs.get("route_order")
    if order:
        routes.sort(key=order.index)
    return {
        "route_key": "+".join(routes),
        "routes": routes,
        "gaps": gaps,
        "fallback_reason": None,
    }


def modality_family(route_key, record=None):
    """CT / MR / X-ray / skin tier for INT-SHUF singletons."""
    r = route_key.split("+")[0]
    if r.startswith("RT-XR-") or r == "RT-MAMMO":
        return "X-ray"
    if r.startswith("RT-CT-"):
        return "CT"
    if r.startswith("RT-MR-"):
        return "MR"
    if r.startswith("RT-SKIN-"):
        return "skin"
    mod = str((record or {}).get("modality") or "").lower()
    return {
        "x-ray": "X-ray",
        "mammography": "X-ray",
        "ct": "CT",
        "mr": "MR",
        "mri": "MR",
        "skin": "skin",
    }.get(mod, "unknown")


def collapse_constant(values):
    """Check 9 helper: metadata must be constant over all QA records of one image."""
    vs = {str(v).strip() for v in values}
    return (vs.pop(), True) if len(vs) == 1 else (sorted(vs), False)


def slake_record(qa_meta_rows):
    mods, ok1 = collapse_constant((r["modality"] for r in qa_meta_rows))
    locs, ok2 = collapse_constant((r["location"] for r in qa_meta_rows))
    if not (ok1 and ok2):
        return {
            "source_kind": "SLAKE",
            "modality": mods,
            "regions": [locs] if ok2 else locs,
            "status": "nonconstant",
            ("provenance"): (
                "SLAKE metadata (modality, location); not "
                "constant over QA records -> FALLBACK (check 9)"
            ),
        }
    return {
        "source_kind": "SLAKE",
        "modality": mods,
        "regions": [locs],
        "status": "ok",
        ("provenance"): (
            "SLAKE dataset metadata (modality, location), constant over QA records"
        ),
    }


def vqarad_record(organ_rows, modality, modality_source, extra_gaps=()):
    org, ok = collapse_constant(organ_rows)
    if not ok:
        return {
            "source_kind": "VQA-RAD",
            "modality": modality,
            "regions": org,
            "status": "nonconstant",
            "provenance": "VQA-RAD image_organ not constant (check 9)",
        }
    if modality in (None, "unknown", "conflict"):
        return {
            "source_kind": "VQA-RAD",
            "modality": modality,
            "regions": [org],
            "status": modality or "unknown",
            "provenance": modality_source,
        }
    return {
        "source_kind": "VQA-RAD",
        "modality": modality,
        "regions": [org],
        "status": "ok",
        "provenance": f"VQA-RAD image_organ + {modality_source}",
        "gaps": list(extra_gaps),
    }


def route_order(routes_spec):
    return [r for r in routes_spec["routes"] if r != FALLBACK]


def load_routes_spec(prompts_dir):
    return json.loads(
        Path(prompts_dir, "specs", "routes.json").read_text(encoding="utf-8")
    )
