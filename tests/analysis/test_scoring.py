"""Check option rotation, invalid answers and label-probability scores."""

import numpy as np
import pandas as pd
import pytest

from medvqa.analysis.scoring import (
    pair_table,
    prepare_items,
    question_table,
    score_rows,
)


def _items():
    return pd.DataFrame(
        [
            {
                "qid": "NT-0001-Q01",
                "image_id": "NT-0001",
                "cohort": "NAT",
                "dataset": "OmniMedVQA",
                "group": "MR",
                "cluster_id": "c1",
                "answer_type": "MCQ",
                "question": "Which?",
                "options": ["CT", "MRI", "X-ray"],
                "gold": "MRI",
                "gold_index": 1,
                "qtype_meta": "Modality Recognition",
                "routes": ["RT-MR-KNEE"],
                "weak_group": False,
                "image_broken": False,
            },
            {
                "qid": "RT-0001-Q01",
                "image_id": "RT-0001",
                "cohort": "RAD",
                "dataset": "SLAKE",
                "group": None,
                "cluster_id": "c2",
                "answer_type": "CLOSED",
                "question": "Is this a CT?",
                "options": None,
                "gold": "Yes",
                "gold_index": None,
                "qtype_meta": "Modality",
                "routes": ["RT-CT-CHEST"],
                "weak_group": False,
                "image_broken": False,
            },
        ]
    )


def test_scoring_end_to_end():
    items, _ = prepare_items(_items(), {})
    assert (
        items.loc["RT-0001-Q01", "kind"] == "yesno"
        and items.loc["RT-0001-Q01", "routeq"]
    )

    def lp(d):
        return {k: np.log(v) for k, v in d.items()}

    ans = pd.DataFrame(
        [
            {
                "cell": "Q-SELF-GRD",
                "qid": "NT-0001-Q01",
                "order": 0,
                "raw_output": "B",
                "gen_error": False,
                "int_missing": False,
                "label_logprobs": lp({"A": 0.1, "B": 0.8, "C": 0.1}),
            },
            {
                "cell": "Q-SELF-GRD",
                "qid": "NT-0001-Q01",
                "order": 1,
                "raw_output": "MRI",
                "gen_error": False,
                "int_missing": False,
                "label_logprobs": lp({"A": 0.6, "B": 0.2, "C": 0.2}),
            },
            {
                "cell": "Q-SELF-GRD",
                "qid": "NT-0001-Q01",
                "order": 2,
                "raw_output": "B is correct",
                "gen_error": False,
                "int_missing": False,
                "label_logprobs": None,
            },
            {
                "cell": "Q-SELF-GRD",
                "qid": "RT-0001-Q01",
                "order": 0,
                "raw_output": "Yes, it is.",
                "gen_error": False,
                "int_missing": False,
                "label_logprobs": lp({"Yes": 0.7, "No": 0.3}),
            },
            {
                "cell": "Q-INT-SHUF",
                "qid": "NT-0001-Q01",
                "order": 0,
                "raw_output": "B",
                "gen_error": False,
                "int_missing": False,
                "label_logprobs": lp({"A": 0.1, "B": 0.8, "C": 0.1}),
            },
            {
                "cell": "Q-INT-SHUF",
                "qid": "NT-0001-Q01",
                "order": 1,
                "raw_output": "",
                "gen_error": False,
                "int_missing": False,
                "label_logprobs": lp({"A": 0.2, "B": 0.6, "C": 0.2}),
            },
            {
                "cell": "Q-INT-SHUF",
                "qid": "NT-0001-Q01",
                "order": 2,
                "raw_output": "nonsense",
                "gen_error": False,
                "int_missing": False,
                "label_logprobs": None,
            },
            {
                "cell": "Q-INT-SHUF",
                "qid": "RT-0001-Q01",
                "order": 0,
                "raw_output": None,
                "gen_error": True,
                "int_missing": False,
                "label_logprobs": lp({"Yes": 0.3, "No": 0.7}),
            },
        ]
    )
    rows, integ = score_rows(items, ans, ["Q-SELF-GRD", "Q-INT-SHUF"])
    assert integ["missing_rows"] == 0
    r = rows.set_index(["cell", "qid", "order"])
    assert r.loc[("Q-SELF-GRD", "NT-0001-Q01", 1), "gold_label"] == "A"
    assert r.loc[("Q-SELF-GRD", "NT-0001-Q01", 1), "correct_lead"] == 1.0
    assert r.loc[("Q-SELF-GRD", "NT-0001-Q01", 2), "correct_lead"] == 0.0
    assert r.loc[("Q-SELF-GRD", "NT-0001-Q01", 2), "b_is_correct"] == 1.0
    q = question_table(rows)
    assert q["correct_lead"].loc["NT-0001-Q01", "Q-SELF-GRD"] == pytest.approx(2 / 3)
    pt = pair_table(rows, "Q-INT-SHUF", "Q-SELF-GRD")
    assert pt.loc["NT-0001-Q01", "flip"] == pytest.approx(1 / 3)
    assert pt.loc["RT-0001-Q01", "flip"] == 1.0
    assert pt.loc["RT-0001-Q01", "tv"] == pytest.approx(0.4)
    assert pt.loc["NT-0001-Q01", "tv"] == pytest.approx(np.mean([0.0, 0.4]))


def test_missing_rows_are_invalid():
    items, _ = prepare_items(_items(), {})
    ans = pd.DataFrame(
        [
            {
                "cell": "L-NONE",
                "qid": "RT-0001-Q01",
                "order": 0,
                "raw_output": "Yes",
                "gen_error": False,
                "int_missing": False,
                "label_logprobs": None,
            }
        ]
    )
    rows, integ = score_rows(items, ans, ["L-NONE"])
    assert integ["missing_rows"] == 3
    assert rows.loc[rows["missing_row"], "correct_lead"].eq(0).all()
