"""Reference-panel coverage and judge execution-plan validation."""

import hashlib
import json

SCOPE_VERSION = "AX1-SCOPE-v1"
PLAN_VERSION = "AX1-PLAN-v1"
REFERENCE_PANEL_MISSING_IDS = (
    "RT-0001",
    "RT-0057",
    "RT-0095",
    "RT-0108",
    "RT-0115",
    "RT-0117",
    "RT-0146",
    "RT-0179",
    "RT-0228",
)


def digest_json(obj):
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def usable_reference(ref):
    return (
        isinstance(ref, dict)
        and isinstance(ref.get("text"), str)
        and bool(ref["text"].strip())
        and isinstance(ref.get("entries"), list)
        and bool(ref["entries"])
    )


def build_scope(
    panel, annotations, panel_sha256, annotation_sha256, probe_sha256, probes
):
    expected = sorted(panel["strata"]["SLAKE"])
    eligible = [n for n in expected if usable_reference(annotations.get(n))]
    return {
        "version": SCOPE_VERSION,
        "configuration_ref": "REFERENCE_PANEL",
        "expected_ids": expected,
        "eligible_ids": eligible,
        "excluded": [
            {"neutral_id": n, "reason": "missing_reference"}
            for n in expected
            if n not in eligible
        ],
        "probe_requested": 250,
        "probe_actual": len(probes),
        "source_sha256": {
            "panel_rq2.json": panel_sha256,
            "annotations_slake_test.json": annotation_sha256,
            "jprobe_set.json": probe_sha256,
        },
    }


def validate_scope_metadata(scope):
    if scope.get("version") != SCOPE_VERSION:
        raise ValueError("J-AX1 scope must declare AX1-SCOPE-v1")
    expected, eligible = (scope.get("expected_ids", []), scope.get("eligible_ids", []))
    excluded = scope.get("excluded", [])
    if (
        not isinstance(expected, list)
        or not isinstance(eligible, list)
        or (not isinstance(excluded, list))
        or (len(expected) != 96)
        or (len(set(expected)) != 96)
        or (len(eligible) != 87)
        or (len(set(eligible)) != 87)
    ):
        raise ValueError(
            "REFERENCE_PANEL requires exactly 96 expected / 87 unique eligible images"
        )
    if any(
        (
            not isinstance(x, dict) or x.get("reason") != "missing_reference"
            for x in excluded
        )
    ):
        raise ValueError(
            "REFERENCE_PANEL exclusions must be explicit missing_reference metadata"
        )
    ids = [x.get("neutral_id") for x in excluded]
    if (
        len(ids) != 9
        or set(ids) != set(REFERENCE_PANEL_MISSING_IDS)
        or set(eligible) & set(ids)
        or (set(expected) != set(eligible) | set(ids))
    ):
        raise ValueError(
            (
                "REFERENCE_PANEL expected/eligible/excluded sets "
                "differ from the approved 96/87/9 partition"
            )
        )
    hashes = scope.get("source_sha256", {})
    if (
        scope.get("probe_requested") != 250
        or type(scope.get("probe_actual")) is not int
        or scope["probe_actual"] < 1
    ):
        raise ValueError(
            "REFERENCE_PANEL must report requested versus actual locked probe counts"
        )
    for k in ("panel_rq2.json", "annotations_slake_test.json", "jprobe_set.json"):
        h = hashes.get(k)
        if (
            not isinstance(h, str)
            or len(h) != 64
            or any((c not in "0123456789abcdef" for c in h))
        ):
            raise ValueError(f"REFERENCE_PANEL missing source SHA256: {k}")
    return set(eligible)


def validate_scope(
    scope,
    panel,
    annotations,
    panel_sha256,
    annotation_sha256,
    probe_sha256,
    probes,
    cohort_ids,
):
    eligible = validate_scope_metadata(scope)
    rebuilt = build_scope(
        panel, annotations, panel_sha256, annotation_sha256, probe_sha256, probes
    )
    identity_fields = set(rebuilt) - {"configuration_ref"}
    if {key: scope.get(key) for key in identity_fields} != {
        key: rebuilt[key] for key in identity_fields
    }:
        raise ValueError(
            (
                "REFERENCE_PANEL scope differs from original "
                "panel/reference/probe hashes or eligible "
                "references"
            )
        )
    if not eligible <= set(cohort_ids):
        raise ValueError("REFERENCE_PANEL eligible images absent from cohort")
    if len(probes) != scope["probe_actual"] or len(
        {p["probe_id"] for p in probes}
    ) != len(probes):
        raise ValueError(
            "REFERENCE_PANEL must retain all unique probes in the unchanged locked file"
        )
    if any((p["neutral_id"] not in eligible for p in probes)):
        raise ValueError(
            (
                "REFERENCE_PANEL probe outside eligible reference"
                " coverage; never filter or rebuild probes"
            )
        )
    return {
        "configuration_ref": "REFERENCE_PANEL",
        "expected_images": 96,
        "eligible_images": 87,
        "missing_reference_images": 9,
        "coverage": 87 / 96,
        "probe_statements": len(probes),
        "probe_requested": 250,
    }


def validate_plan(
    plan, scope_sha256, probe_sha256, judge, eligible_ids=None, probe_images=None
):
    if (
        plan.get("version") != PLAN_VERSION
        or plan.get("judge") != judge
        or (plan.get("scope_sha256") != scope_sha256)
        or (plan.get("probe_sha256") != probe_sha256)
    ):
        raise ValueError("REFERENCE_PANEL judge plan identity/source hash mismatch")
    units = plan.get("units", [])
    if (
        not units
        or len({u["key"] for u in units}) != len(units)
        or digest_json(units) != plan.get("units_sha256")
    ):
        raise ValueError("REFERENCE_PANEL unit plan is empty, duplicated or changed")
    if eligible_ids is not None:
        eligible = set(eligible_ids)
        if any((u["neutral_id"] not in eligible for u in units)):
            raise ValueError("REFERENCE_PANEL planned unit outside eligible coverage")
        main = [u for u in units if u["kind"] == "main"]
        got = {(u["neutral_id"], u["observer"], u["arm"]) for u in main}
        want = {
            (n, o, a)
            for n in eligible
            for o in ("OBS-L", "OBS-Q", "OBS-G")
            for a in ("ARM-GEN", "ARM-ANA", "ARM-GRD")
        }
        if len(main) != len(want) or got != want:
            raise ValueError(
                "REFERENCE_PANEL main plan must cover every eligible observer/arm"
            )
    if probe_images is not None:
        probe_units = [u for u in units if u["kind"] == "probe"]
        pids = [p for u in probe_units for p in u["sids"]]
        if len(pids) != len(probe_images) or set(pids) != set(probe_images):
            raise ValueError(
                "REFERENCE_PANEL plan must preserve every locked probe exactly once"
            )
        if any(
            (
                not 1 <= len(u["sids"]) <= 10
                or any((probe_images[p] != u["neutral_id"] for p in u["sids"]))
                for u in probe_units
            )
        ):
            raise ValueError(
                (
                    "REFERENCE_PANEL probe blocks must retain the "
                    "original image and block cap"
                )
            )
    return {u["key"]: u for u in units}


def validate_record_scopes(records, scope, scope_sha256, plans):
    """Reject incompatible reference-panel scopes before scoring.

    Reject old/new mixtures before de-duplication or scoring; no clinical labels read.
    """
    eligible = validate_scope_metadata(scope)
    unit_maps = {
        j: validate_plan(
            p, scope_sha256, scope["source_sha256"]["jprobe_set.json"], j, eligible
        )
        for j, (p, _) in plans.items()
    }
    for r in records:
        if r.get("task") != "J-AX1":
            continue
        j = r["judge"]
        if j not in plans:
            raise ValueError(f"REFERENCE_PANEL judge without a locked plan: {j}")
        plan, plan_sha256 = plans[j]
        units = unit_maps[j]
        if (
            r.get("ax1_scope_sha256") != scope_sha256
            or r.get("ax1_plan_sha256") != plan_sha256
            or r.get("ax1_image_id") not in eligible
            or (r["key"] not in units)
            or (units[r["key"]]["neutral_id"] != r.get("ax1_image_id"))
        ):
            raise ValueError(
                (
                    "mixed/mismatched REFERENCE_PANEL scope, plan, "
                    "image or unit key in J-AX1 outputs"
                )
            )
        statements = (r.get("refs") or {}).get("statements") or []
        if [s.get("sid") for s in statements] != units[r["key"]]["sids"]:
            raise ValueError(
                (
                    "REFERENCE_PANEL statement identities/order "
                    "differ from the locked unit plan"
                )
            )
