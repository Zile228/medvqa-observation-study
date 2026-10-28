"""Check answer parsing for valid, invalid and ambiguous outputs."""

import pytest

from medvqa.analysis.parsers import (
    answer_kind,
    b_is_correct_type,
    gold_letter_for_order,
    is_correct,
    norm,
    parse_exact,
    parse_lead,
    shift_options,
    token_f1,
)

OPTS3 = ["Large mass", "Pleural effusion", "Normal"]
OPTS4 = ["CT", "MRI", "X-ray", "Ultrasound"]
YESNO_CASES = [
    ("Yes", "Yes"),
    ("No", "No"),
    ("yes", "Yes"),
    ("NO.", "No"),
    ("Yes, there is no mass.", "Yes"),
    ("Yes no", None),
    ("No, yes", "No"),
    ("No yes", None),
    ("  \n\nYes\nNo", "Yes"),
    ('"Yes"', "Yes"),
    ("**No**", "No"),
    ("(yes)", "Yes"),
    ("`No`", "No"),
    ("Yes.", "Yes"),
    ("Yes!", "Yes"),
    ("Nope", None),
    ("Not visible", None),
    ("None", None),
    ("Yeah", None),
    ("The answer is yes", None),
    ("yes/no", None),
    ("No; yes it is", "No"),
    ("Yes: no abnormality", "Yes"),
    ("", None),
    ("   ", None),
    (None, None),
    ("Yes there is no effusion", None),
]


@pytest.mark.parametrize("raw,expected", YESNO_CASES)
def test_lead_yesno(raw, expected):
    assert parse_lead(raw, "yesno") == expected


MCQ_CASES = [
    ("A", OPTS3, "A"),
    ("B.", OPTS3, "B"),
    ("C)", OPTS3, "C"),
    ("(B)", OPTS3, "B"),
    ("B is correct", OPTS3, None),
    ("A large mass", ["Mass", "Effusion", "Normal"], None),
    ("A large mass", OPTS3, "A"),
    ("The answer is C", OPTS3, "C"),
    ("the answer is: B", OPTS3, "B"),
    ("Answer: A", OPTS3, "A"),
    ("ANSWER - B", OPTS3, "B"),
    ("Option B", OPTS3, "B"),
    ("option C.", OPTS3, "C"),
    ("Answer:B", OPTS3, "B"),
    ("D", OPTS3, None),
    ("D", OPTS4, "D"),
    ("b", OPTS3, None),
    ("A, because it is round", OPTS3, "A"),
    ("A: Large mass", OPTS3, "A"),
    ("B Pleural effusion", OPTS3, "B"),
    ("Pleural effusion", OPTS3, "B"),
    ("pleural effusion.", OPTS3, "B"),
    ("The pleural effusion", OPTS3, "B"),
    ("Effusion", OPTS3, None),
    ("MRI", OPTS4, "B"),
    ("x ray", OPTS4, None),
    ("Xray", OPTS4, "C"),
    ("AB", OPTS3, None),
    ("A or B", OPTS3, None),
    ("", OPTS3, None),
    ("\n\n  C\n", OPTS3, "C"),
    ("*A*", OPTS3, None),
    ("**A**", OPTS3, None),
    ("E", OPTS4, None),
]


@pytest.mark.parametrize("raw,opts,expected", MCQ_CASES)
def test_lead_mcq(raw, opts, expected):
    assert parse_lead(raw, "mcq", opts) == expected


def test_mcq_duplicate_option_text_is_invalid():
    assert parse_lead("Normal", "mcq", ["Normal", "normal.", "Mass"]) is None


SHORT_CASES = [
    ("Liver.", "closed_short", "Liver"),
    ("Left lung", "closed_short", "Left lung"),
    ("the left lung\nbecause", "closed_short", "the left lung"),
    ("one two three four five six", "closed_short", "one two three four five six"),
    ("one two three four five six seven", "closed_short", None),
    ("one two three four five six seven", "open", "one two three four five six seven"),
    ("...", "closed_short", None),
    ("The", "closed_short", None),
    ("", "open", None),
]


@pytest.mark.parametrize("raw,kind,expected", SHORT_CASES)
def test_lead_short(raw, kind, expected):
    assert parse_lead(raw, kind) == expected


EXACT_CASES = [
    ("Yes", "yesno", None, "Yes"),
    (" yes. ", "yesno", None, "Yes"),
    ("(No)", "yesno", None, "No"),
    ('"Yes!"', "yesno", None, "Yes"),
    ("Yes, there is no mass.", "yesno", None, None),
    ("Yes\n", "yesno", None, "Yes"),
    ("Yes\nNo", "yesno", None, None),
    ("B", "mcq", OPTS3, "B"),
    ("(B).", "mcq", OPTS3, "B"),
    ("B is correct", "mcq", OPTS3, None),
    ("The answer is B", "mcq", OPTS3, None),
    ("b", "mcq", OPTS3, None),
    ("D", "mcq", OPTS3, None),
    ("Liver.", "closed_short", None, "Liver"),
    ("Liver\nkidney", "closed_short", None, None),
]


@pytest.mark.parametrize("raw,kind,opts,expected", EXACT_CASES)
def test_exact(raw, kind, opts, expected):
    assert parse_exact(raw, kind, opts) == expected


def test_norm():
    assert norm("The Left  Lung.") == "left lung"
    assert norm("An X-Ray!") == "xray"
    assert norm(None) == ""


@pytest.mark.parametrize(
    "q,at,cohort,kind",
    [
        ("Is this a CT?", "CLOSED", "RAD", "yesno"),
        ("  does the liver look normal", "CLOSED", "RAD", "yesno"),
        ("Is this a CT or an MRI?", "CLOSED", "RAD", "closed_short"),
        ("Which side is abnormal?", "CLOSED", "RAD", "closed_short"),
        ("Island sign present?", "CLOSED", "RAD", "closed_short"),
        ("What organ is shown?", "OPEN", "RAD", "open"),
        ("Is there a mass?", "OPEN", "RAD", "open"),
        ("Which modality?", "", "NAT", "mcq"),
    ],
)
def test_answer_kind(q, at, cohort, kind):
    assert answer_kind(q, at, cohort) == kind


def test_shift_and_gold_letter():
    opts = ["a", "b", "c", "d"]
    assert shift_options(opts, 0) == opts
    assert shift_options(opts, 1) == ["b", "c", "d", "a"]
    assert gold_letter_for_order(2, 4, 1) == "B"
    assert gold_letter_for_order(2, 4, 3) == "D"
    for s in range(4):
        sh = shift_options(opts, s)
        assert sh["ABCD".index(gold_letter_for_order(2, 4, s))] == "c"


def test_is_correct():
    assert is_correct("Yes", "yesno", "yes")
    assert not is_correct("Yes", "yesno", "No")
    assert not is_correct("Yes", "yesno", "left")
    assert is_correct("B", "mcq", "B")
    assert is_correct("The Liver", "closed_short", "liver.")
    assert not is_correct(None, "yesno", "yes")


def test_b_is_correct_counter():
    assert b_is_correct_type("B is correct")
    assert b_is_correct_type("The answer is A because")
    assert not b_is_correct_type("B")
    assert not b_is_correct_type("A Large mass")


def test_token_f1():
    assert token_f1("left lung", "the left lung") == 1.0
    assert token_f1("right lung", "left lung") == pytest.approx(0.5)
    assert token_f1("", "x") == 0.0


def test_parser_case_count():
    n = len(YESNO_CASES) + len(MCQ_CASES) + len(SHORT_CASES) + len(EXACT_CASES)
    assert n >= 40
