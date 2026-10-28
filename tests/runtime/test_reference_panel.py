"""Check reference-panel membership, source checksums and judge plans."""

import copy

import pytest

from medvqa import judge_scope as scope_checks


@pytest.fixture
def panel():
    eligible = [f"IMAGE-{index:04d}" for index in range(87)]
    expected = eligible + list(scope_checks.REFERENCE_PANEL_MISSING_IDS)
    source = {"strata": {"SLAKE": expected}}
    annotations = {
        image: {"text": "Synthetic reference", "entries": [{}]} for image in eligible
    }
    probes = [{"probe_id": "PROBE-1", "neutral_id": eligible[0]}]
    hashes = ("a" * 64, "b" * 64, "c" * 64)
    scope = scope_checks.build_scope(source, annotations, *hashes, probes)
    scope.pop("configuration_ref")
    return scope, source, annotations, hashes, probes, expected


def validate_panel(fixture, scope):
    _, source, annotations, hashes, probes, expected = fixture
    return scope_checks.validate_scope(
        scope, source, annotations, *hashes, probes, expected
    )


def make_plan(eligible, scope_sha="d" * 64, probe_sha="c" * 64):
    units = [
        {
            "key": f"main|{image}|{observer}|{arm}",
            "neutral_id": image,
            "observer": observer,
            "arm": arm,
            "kind": "main",
            "sids": [f"statement|{image}|{observer}|{arm}"],
        }
        for image in eligible
        for observer in ("OBS-L", "OBS-Q", "OBS-G")
        for arm in ("ARM-GEN", "ARM-ANA", "ARM-GRD")
    ]
    return {
        "version": scope_checks.PLAN_VERSION,
        "judge": "J1",
        "scope_sha256": scope_sha,
        "probe_sha256": probe_sha,
        "units": units,
        "units_sha256": scope_checks.digest_json(units),
    }


def test_scope_uses_scientific_identity_without_a_journal_label(panel):
    scope = panel[0]
    result = validate_panel(panel, scope)
    assert result["eligible_images"] == 87
    assert result["missing_reference_images"] == 9


@pytest.mark.parametrize("change", ["version", "source", "count", "partition"])
def test_scope_still_rejects_changed_scientific_identity(panel, change):
    scope = copy.deepcopy(panel[0])
    if change == "version":
        scope["version"] = "unsupported"
    elif change == "source":
        scope["source_sha256"]["panel_rq2.json"] = "0" * 64
    elif change == "count":
        scope["eligible_ids"].pop()
    else:
        scope["eligible_ids"][0] = scope["excluded"][0]["neutral_id"]
    with pytest.raises(ValueError):
        validate_panel(panel, scope)


def test_plan_does_not_require_a_journal_label(panel):
    eligible = panel[0]["eligible_ids"]
    plan = make_plan(eligible)
    units = scope_checks.validate_plan(plan, "d" * 64, "c" * 64, "J1", eligible)
    assert len(units) == 87 * 3 * 3


@pytest.mark.parametrize(
    "field", ["version", "judge", "scope_sha256", "probe_sha256", "units_sha256"]
)
def test_plan_still_rejects_wrong_scope_sources_or_units(panel, field):
    eligible = panel[0]["eligible_ids"]
    plan = make_plan(eligible)
    plan[field] = "wrong"
    with pytest.raises(ValueError):
        scope_checks.validate_plan(plan, "d" * 64, "c" * 64, "J1", eligible)


def test_record_from_another_scope_is_rejected(panel):
    scope = panel[0]
    eligible = scope["eligible_ids"]
    plan = make_plan(eligible)
    unit = plan["units"][0]
    record = {
        "task": "J-AX1",
        "judge": "J1",
        "key": unit["key"],
        "ax1_scope_sha256": "d" * 64,
        "ax1_plan_sha256": "e" * 64,
        "ax1_image_id": unit["neutral_id"],
        "refs": {"statements": [{"sid": unit["sids"][0]}]},
    }
    scope_checks.validate_record_scopes(
        [record], scope, "d" * 64, {"J1": (plan, "e" * 64)}
    )
    record["ax1_scope_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        scope_checks.validate_record_scopes(
            [record], scope, "d" * 64, {"J1": (plan, "e" * 64)}
        )
