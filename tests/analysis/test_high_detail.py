"""Synthetic paired-estimate identities and reference-label access boundaries."""

import json

import pandas as pd
import pytest

from medvqa.analysis import high_detail as study
from medvqa.analysis.contrasts import Engine
from medvqa.analysis.scoring import question_table
from tests.analysis.test_neutral_controls import synthetic_engine


def test_paired_high_low_identities_and_native_question_units():
    original, rows = synthetic_engine()
    mapping = {
        "Q-NONE": "Q-NONE",
        "Q-SELF-GRD": "Q-SELF-GRD",
        "Q-G-GRD": "Q-ITEMS",
        "Q-G-GRD-HI": "Q-NEUTRAL-DASH",
        "L-NONE": "Q-NEUTRAL-DASH",
        "L-SELF-GRD": "Q-SELF-GRD",
        "L-G-GRD": "Q-NEUTRAL-LIST",
        "L-G-GRD-HI": "Q-SELF-GRD",
    }
    selected = pd.concat(
        [
            rows[rows["cell"].eq(source)].assign(cell=target)
            for target, source in mapping.items()
        ],
        ignore_index=True,
    )
    engine = Engine(
        original.items,
        selected,
        question_table(selected),
        {},
        original.boots,
        original.qids,
    )
    results = study.calculate(engine)
    assert len(results) == 12
    assert set(results["n_questions"]) == {4}
    assert not any("holm" in name or "tost" in name for name in results.columns)
    for model in ("Q", "L"):
        for group in ("RAD", "NAT"):
            table = results[
                results["model"].eq(model) & results["cohort"].eq(group)
            ].set_index("contrast")
            c2, c3, high_low = (
                table.loc[name, "estimate"]
                for name in ("C2-high", "C3-high", "high-minus-low")
            )
            none = engine.vec("correct_lead", model + "-NONE", group).mean()
            self_grd = engine.vec("correct_lead", model + "-SELF-GRD", group).mean()
            low = engine.vec("correct_lead", model + "-G-GRD", group).mean()
            assert c2 - c3 == pytest.approx((self_grd - none) * 100)
            assert c2 - high_low == pytest.approx((low - none) * 100)
    identical = results[results["model"].eq("L") & results["contrast"].eq("C3-high")]
    assert identical[["estimate", "ci_low", "ci_high"]].eq(0).all().all()


def test_synthetic_outputs_rejected_before_reference_labels(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "execution.json").write_text(
        json.dumps({"synthetic_backend": True}), encoding="utf-8"
    )
    monkeypatch.setattr(study.H, "preflight", lambda *args: ({}, {}, []))

    def forbidden(*args):
        raise AssertionError(
            "Reference labels cannot be opened before public validation."
        )

    monkeypatch.setattr(study.io, "load_all", forbidden)
    with pytest.raises(ValueError, match="configuration"):
        study.analyze(
            tmp_path,
            run,
            run,
            tmp_path / "inputs",
            tmp_path / "observations",
            tmp_path / "output",
        )
