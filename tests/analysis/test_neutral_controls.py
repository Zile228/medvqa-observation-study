"""Synthetic checks for exploratory contrasts and reference-label access order."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from medvqa.analysis import neutral_controls as study
from medvqa.analysis.bootstrap import ClusterBootstrap
from medvqa.analysis.contrasts import Engine
from medvqa.analysis.scoring import QCOLS, question_table


@pytest.mark.parametrize(
    "bounds,state",
    [
        ((1, 2, 1.1, 1.9), "different_but_within_margin"),
        ((4, 6, 4.2, 5.8), "different"),
        ((-1, 1, -0.8, 0.8), "equivalent_within_margin"),
        ((-5, 5, -4, 4), "inconclusive"),
        ((0, 5, 0.1, 4), "inconclusive"),
        ((-4, 0, -3, -0.1), "inconclusive"),
    ],
)
def test_four_interval_states_have_no_overlap(bounds, state):
    result = dict(zip(("ci_low", "ci_high", "tost_ci_low", "tost_ci_high"), bounds))
    assert study.interval_state(result) == state


def synthetic_engine():
    cells = ("Q-NONE", "Q-INT-EMPTY", "Q-SELF-GRD", *study.CONTROL_CELLS)
    qids = {"RAD": [f"R{i}" for i in range(4)], "NAT": [f"N{i}" for i in range(4)]}
    items = []
    rows = []
    for cohort, questions in qids.items():
        for index, qid in enumerate(questions):
            kind = (
                "mcq" if cohort == "NAT" else "closed_short" if index == 3 else "yesno"
            )
            gold = (
                "A"
                if kind == "mcq"
                else "finding"
                if kind == "closed_short"
                else "yes"
                if index % 2 == 0
                else "no"
            )
            items.append({"qid": qid, "cohort": cohort, "kind": kind, "gold": gold})
            for cell_index, cell in enumerate(cells):
                for order in range(2 if cohort == "NAT" else 1):
                    correct = (index + cell_index) % 3 != 0
                    lead = (
                        "A"
                        if kind == "mcq"
                        else "finding"
                        if kind == "closed_short"
                        else "Yes"
                        if correct
                        else "No"
                    )
                    valid = not (cell == "Q-ITEMS" and qid == "R0")
                    if not valid:
                        lead = None
                    probability = (
                        None
                        if kind == "closed_short" or (cell == "Q-ITEMS" and qid == "R1")
                        else {"A": 0.7, "B": 0.3}
                        if kind == "mcq"
                        else {"Yes": 0.3, "No": 0.7}
                    )
                    row = {name: float(correct and valid) for name in QCOLS}
                    row.update(
                        qid=qid,
                        cell=cell,
                        order=order,
                        cohort=cohort,
                        kind=kind,
                        lead=lead,
                        valid_lead=float(valid),
                        label_key=lead,
                        ptilde=probability,
                        p_gold=0.7,
                        image_id=qid,
                    )
                    rows.append(row)
    items = pd.DataFrame(items).set_index("qid")
    rows = pd.DataFrame(rows)
    bootstraps = {
        cohort: ClusterBootstrap(
            ids, [cohort] * len(ids), [cohort] * len(ids), B=64, seed=20261016
        )
        for cohort, ids in qids.items()
    }
    engine = Engine(items, rows, question_table(rows), {}, bootstraps, qids)
    return engine, rows


def test_real_engine_handles_native_repeated_orders_and_all_seven_contrasts():
    engine, rows = synthetic_engine()
    result = study.calculate(engine, rows)
    contrasts = result["contrasts"]
    assert len(contrasts) == 14
    rad = contrasts[contrasts["cohort"].eq("RAD")].set_index("contrast")
    assert len(rad["p_holm_sensitivity"].dropna()) == 7
    assert (
        contrasts.loc[contrasts["cohort"].eq("NAT"), "p_holm_sensitivity"].isna().all()
    )
    points = {
        cell: engine.vec("correct_lead", cell, "RAD").mean()
        for cell in ("Q-INT-EMPTY", "Q-ITEMS", "Q-NEUTRAL-DASH", "Q-NEUTRAL-LIST")
    }
    interaction = (
        points["Q-INT-EMPTY"]
        - points["Q-ITEMS"]
        - points["Q-NEUTRAL-DASH"]
        + points["Q-NEUTRAL-LIST"]
    )
    assert rad.loc["T7", "estimate"] == pytest.approx(interaction * 100)
    directions = result["answer_direction"]
    assert set(directions["cohort"]) == {"RAD"}
    no_rate = directions[
        (directions["cell"] == "Q-ITEMS") & (directions["metric"] == "No_rate")
    ].iloc[0]
    probability = directions[
        (directions["cell"] == "Q-ITEMS") & (directions["metric"] == "mean_P_No")
    ].iloc[0]
    assert no_rate["n_questions"] == 3
    assert probability["n_questions"] == 2
    assert len(result["answer_changes"]) == 20


def test_empty_bootstrap_summary_stays_inconclusive():
    result = study.summary(np.nan, np.array([np.nan]), equivalence=True)
    assert result["state"] == "inconclusive"
    assert result["B_nan"] == 1


def test_incomplete_run_is_rejected_before_reference_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(study.controls, "load_bank", lambda root: {})
    monkeypatch.setattr(
        study.controls, "plan", lambda *args: [("EXPECTED", {}, {}, "-")]
    )
    monkeypatch.setattr(study.io, "read_jsonl", lambda path: [])

    def forbidden_labels(*args):
        raise AssertionError("Reference labels must not be opened.")

    monkeypatch.setattr(study.io, "load_all", forbidden_labels)
    with pytest.raises(ValueError, match="incomplete"):
        study.analyze(tmp_path, tmp_path / "missing", tmp_path / "output")


def control_fixture(root, monkeypatch, alteration=None):
    run = root / "control_run"
    shard = run / "shard_00"
    shard.mkdir(parents=True)
    for cohort in ("COH-RAD-TEST", "COH-NAT-TEST"):
        path = root / "prep/prep_out/cohorts" / f"{cohort}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    text = "1. Synthetic observation"
    question = {"qid": "QUESTION", "neutral_id": "IMAGE", "answer_kind": "yesno"}
    condition = {"cell": "CELL-Q-ITEMS"}
    key = "ANS|CELL-Q-ITEMS|QUESTION|-"
    monkeypatch.setattr(
        study.controls, "load_bank", lambda root: {"ITEMS": {("TEST", "IMAGE"): text}}
    )
    monkeypatch.setattr(
        study.controls, "plan", lambda *args: [(key, condition, question, "-")]
    )
    row = {
        "schema": "ANS-v1",
        "mode": "TEST",
        "key": key,
        "cell": "CELL-Q-ITEMS",
        "qid": "QUESTION",
        "neutral_id": "IMAGE",
        "answer_kind": "yesno",
        "order": "-",
        "image_condition": "original",
        "status": "ok",
        "text": "No",
        "obs_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    cfg = {
        "mode": "TEST",
        "precision": "PREC-0",
        "answer_cap": study.ANSWER_CAP,
        "generate_kwargs": study.GREEDY_KW,
        "cells": ["CELL-" + cell for cell in study.CONTROL_CELLS],
        "model": {
            "model": study.MODELS["Q"]["id"],
            "revision": study.MODELS["Q"]["revision"],
        },
    }
    if alteration == "precision":
        cfg["precision"] = "PREC-2"
    elif alteration == "text_hash":
        row["obs_text_sha256"] = "0" * 64
    elif alteration == "qid":
        row["qid"] = "OTHER_QUESTION"
    elif alteration == "no_option_order":
        row["order"] = None
    records = shard / "records.jsonl"
    records.write_text(json.dumps(row) + "\n", encoding="utf-8")
    if alteration == "duplicate":
        with records.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
    config = shard / "run_config.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    integrity = {
        "output_sha256": study.sha(records),
        "run_config_sha256": study.sha(config),
    }
    (shard / "integrity.json").write_text(json.dumps(integrity), encoding="utf-8")
    if alteration == "record_hash":
        with records.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    return run


@pytest.mark.parametrize("alteration", [None, "no_option_order"])
def test_completed_control_session_validates_without_reading_labels(
    tmp_path, monkeypatch, alteration
):
    run = control_fixture(tmp_path, monkeypatch, alteration)
    answers, audit = study.validate_control_run(tmp_path, run)
    assert len(answers) == 1
    assert audit["completed_units"] == 1
    assert audit["public_scope_checked_before_reference_labels"]


@pytest.mark.parametrize(
    "alteration", ["record_hash", "precision", "text_hash", "qid", "duplicate"]
)
def test_control_integrity_and_membership_fail_closed(
    tmp_path, monkeypatch, alteration
):
    run = control_fixture(tmp_path, monkeypatch, alteration)
    with pytest.raises(ValueError):
        study.validate_control_run(tmp_path, run)
