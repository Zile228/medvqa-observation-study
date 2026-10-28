"""Check missing-value handling with synthetic answer records."""

import numpy as np
import pandas as pd
import pytest

from medvqa.analysis import io_adapter as IO
from medvqa.analysis.contrasts import Engine
from medvqa.analysis.missing import is_missing_scalar
from medvqa.analysis.scoring import (
    INVALID,
    int_missing_qids,
    pair_table,
    prepare_items,
    score_rows,
)

MISSING = [None, np.nan, pd.NA]


def _items(kind="yesno"):
    native = kind == "mcq"
    item = {
        "qid": "q",
        "image_id": "img",
        "cohort": "NAT" if native else "RAD",
        "dataset": "OmniMedVQA" if native else "SLAKE",
        "group": "MR" if native else None,
        "cluster_id": "c",
        "answer_type": "OPEN" if kind == "open" else "CLOSED",
        "question": "Is it present?" if kind == "yesno" else "Which option?",
        "options": ["Alpha", "Beta", "Gamma"] if native else None,
        "gold": "Beta" if native else "Yes",
        "gold_index": 1 if native else None,
        "qtype_meta": None,
        "routes": ["RT-FALLBACK"],
        "weak_group": False,
        "image_broken": False,
    }
    return prepare_items(pd.DataFrame([item]))[0]


def _answer(raw, cell="Q-SELF-GRD", order=0, **flags):
    return {
        "cell": cell,
        "qid": "q",
        "order": order,
        "raw_output": raw,
        "gen_error": False,
        "int_missing": False,
        "label_logprobs": None,
        **flags,
    }


@pytest.mark.parametrize("missing", MISSING, ids=["none", "nan", "pdNA"])
@pytest.mark.parametrize("kind", ["yesno", "mcq", "closed_short", "open"])
def test_missing_outputs_are_invalid_at_scoring_boundary(missing, kind):
    items = _items(kind)
    orders = range(3) if kind == "mcq" else [0]
    answers = pd.DataFrame([_answer(missing, order=o) for o in orders], dtype=object)
    rows, integrity = score_rows(items, answers, ["Q-SELF-GRD"])
    assert integrity["missing_rows"] == 0
    assert rows["valid_lead"].eq(0).all() and rows["valid_exact"].eq(0).all()
    assert (
        rows["label_key"].eq(INVALID).all()
        and rows["label_key_exact"].eq(INVALID).all()
    )
    assert not rows["int_missing"].any()
    if kind == "open":
        assert rows["em"].eq(0).all() and rows["f1"].eq(0).all()
    else:
        assert rows["correct_lead"].eq(0).all() and rows["correct_exact"].eq(0).all()


@pytest.mark.parametrize("missing", MISSING, ids=["none", "nan", "pdNA"])
def test_answer_adapter_preserves_text_and_normalizes_missing(missing):
    recs = pd.DataFrame(
        [
            {
                "cell": "CELL-Q-NONE",
                "qid": "q",
                "order": missing,
                "text": missing,
                "status": "ok",
            },
            {
                "cell": "CELL-Q-NONE",
                "qid": "q2",
                "order": "-",
                "text": "nan",
                "status": "ok",
            },
            {
                "cell": "CELL-Q-NONE",
                "qid": "q3",
                "order": 0,
                "text": "Yes",
                "status": "error",
            },
            {
                "cell": "CELL-Q-NONE",
                "qid": "q4",
                "order": 0,
                "text": "Yes",
                "status": "skipped_int_missing",
            },
        ],
        index=[10, 20, 30, 40],
    )
    ans = IO.answers_table(recs)
    assert ans.loc[10, "raw_output"] is None and ans.loc[10, "order"] == 0
    assert ans.loc[20, "raw_output"] == "nan"
    assert ans.loc[30, "raw_output"] is None and ans.loc[30, "gen_error"]
    assert ans.loc[40, "raw_output"] is None and ans.loc[40, "int_missing"]


@pytest.mark.parametrize("missing", MISSING, ids=["none", "nan", "pdNA"])
def test_missing_flags_and_required_rows(missing):
    answers = pd.DataFrame(
        [_answer("Yes", gen_error=missing, int_missing=missing)], dtype=object
    )
    rows, integrity = score_rows(_items(), answers, ["Q-SELF-GRD", "Q-INT-SHUF"])
    assert integrity["missing_rows"] == 1 and integrity["missing_rows_by_cell"] == {
        "Q-INT-SHUF": 1
    }
    assert rows["label_key"].eq(INVALID).all()
    assert rows["gen_error"].all() and (not rows["int_missing"].any())
    pair = pair_table(rows, "Q-INT-SHUF", "Q-SELF-GRD")
    assert pair.loc["q", "flip"] == 0 and pd.isna(pair.loc["q", "flip_valid"])


@pytest.mark.parametrize("left", MISSING, ids=["none", "nan", "pdNA"])
@pytest.mark.parametrize("right", MISSING, ids=["none", "nan", "pdNA"])
@pytest.mark.parametrize("parser", ["lead", "exact"])
def test_pair_table_missing_labels_share_one_invalid_label(left, right, parser):
    key = "label_key" if parser == "lead" else "label_key_exact"
    records = []
    for order, (i, r) in enumerate(
        [(left, right), ("Yes", right), (left, "Yes"), ("Yes", "Yes")]
    ):
        for cell, label in [("Q-INT-SHUF", i), ("Q-SELF-GRD", r)]:
            records.append(
                {
                    "cell": cell,
                    "qid": "q",
                    "order": order,
                    key: label,
                    "valid_lead": float(not is_missing_scalar(label)),
                    "kind": "yesno",
                    "ptilde": pd.NA,
                    "p_gold": np.nan,
                }
            )
    result = pair_table(pd.DataFrame(records), "Q-INT-SHUF", "Q-SELF-GRD", parser)
    assert result.loc["q", "flip"] == 0.5
    assert result.loc["q", "flip_valid"] == 0
    assert pd.isna(result.loc["q", "tv"])


def test_cyclic_flip_and_intervention_missing_are_separate():
    items = _items("mcq")
    records = []
    for o, (ref, edited) in enumerate([("B", "B"), ("A", None), (pd.NA, np.nan)]):
        records += [_answer(ref, order=o), _answer(edited, "Q-INT-SHUF", o)]
        records += [_answer(None, "Q-INT-CORR", o, int_missing=True)]
    rows, integrity = score_rows(
        items, pd.DataFrame(records), ["Q-SELF-GRD", "Q-INT-SHUF", "Q-INT-CORR"]
    )
    assert integrity["missing_rows"] == 0
    assert rows.loc[rows["cell"] == "Q-SELF-GRD", "gold_label"].tolist() == [
        "B",
        "A",
        "C",
    ]
    pair = pair_table(rows, "Q-INT-SHUF", "Q-SELF-GRD")
    assert pair.loc["q", "flip"] == pytest.approx(1 / 3)
    missing = int_missing_qids(rows)
    assert missing == {"Q-INT-CORR": {"q"}}
    eng = Engine(items, rows, {}, missing, {}, {"POOL": ["q"]})
    assert eng.missing_mask("POOL", ["Q-INT-CORR", "Q-SELF-GRD"]).tolist() == [False]
    assert eng.missing_mask("POOL", ["Q-INT-SHUF", "Q-SELF-GRD"]).tolist() == [True]
    assert eng.pair_vec("flip", "Q-INT-SHUF", "Q-SELF-GRD", "POOL")[0] == pytest.approx(
        1 / 3
    )


@pytest.mark.parametrize("text", ["nan", "None", "<NA>", " Yes.\nNo"])
def test_literal_text_is_not_missing(text):
    assert not is_missing_scalar(text)
    items = _items("open")
    rows, _ = score_rows(items, pd.DataFrame([_answer(text)]), ["Q-SELF-GRD"])
    assert rows.iloc[0]["valid_lead"] == 1


def test_missing_helper_does_not_test_container_truth_values():
    assert all((is_missing_scalar(v) for v in MISSING))
    assert all(
        (not is_missing_scalar(v) for v in [[], [None], {}, np.array([None]), False, 0])
    )
