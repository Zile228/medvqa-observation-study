"""Answer parsers for yes/no, short text and cyclic multiple-choice questions."""

from __future__ import annotations

import re
from typing import Optional, Sequence

from .constants import YESNO_LEAD

PARSER_IDS = ("PARSE-LEAD-v1", "PARSE-EXACT-v1")
LETTERS = "ABCD"
_ARTICLES = re.compile("\\b(?:a|an|the)\\b")
_PUNCT = re.compile("[^\\w\\s]")
_WS = re.compile("\\s+")


def norm(text: Optional[str]) -> str:
    """Normalize answer text using the study's fixed scoring rules.

    Normalization: lowercase, drop punctuation, drop a/an/the, collapse spaces.

    Punctuation is deleted (not replaced by a space); the resulting characters are
    joined before whitespace normalization.
    """
    if text is None:
        return ""
    s = text.lower()
    s = _PUNCT.sub("", s)
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


_YESNO_RE = re.compile(YESNO_LEAD, re.IGNORECASE)
_OR_RE = re.compile("\\bor\\b", re.IGNORECASE)


def answer_kind(question: str, answer_type: str, cohort: str) -> str:
    """Determine question kind from its text and official answer type.

    Question kind from question text + official answer_type only (never gold).
    """
    if cohort == "NAT":
        return "mcq"
    at = (answer_type or "").strip().upper()
    if at == "OPEN":
        return "open"
    if at != "CLOSED":
        raise ValueError(f"unknown answer_type {answer_type!r}")
    q = (question or "").strip()
    if _YESNO_RE.match(q) and (not _OR_RE.search(q)):
        return "yesno"
    return "closed_short"


def shift_options(options: Sequence[str], s: int) -> list:
    """Rotate option texts left while retaining label positions.

    Option order s is a LEFT cyclic shift opts[s:] + opts[:s]; letters keep
    positions.
    """
    opts = list(options)
    k = len(opts)
    s = s % k
    return opts[s:] + opts[:s]


def gold_letter_for_order(gold_index: int, k: int, s: int) -> str:
    """Map the correct option to its letter after rotation.

    Gold letter after a left rotation (original index g -> (g - s) mod
    K).
    """
    return LETTERS[(gold_index - s) % k]


_LEAD_STRIP = re.compile("^[\\s\\\"'`*(\\[{]+")
_YN_HEAD = re.compile("^(yes|no)\\b", re.IGNORECASE)
_CLAUSE_END = re.compile("[.,;:!?\\n]")
_YES_W = re.compile("\\byes\\b", re.IGNORECASE)
_NO_W = re.compile("\\bno\\b", re.IGNORECASE)
_MCQ_RE = re.compile(
    (
        "^(?:(?i:(?:the\\s+)?(?:answer|option)\\s*(?:is)?\\s"
        "*[:\\-]?\\s*))?([A-D])(?=$|[\\s.):,])(?!\\s+[a-z])"
    )
)


def first_line(raw: Optional[str]) -> Optional[str]:
    r"""Extract and clean the first nonempty answer line.

    Preprocessing: first non-empty line, leading chars in \s"'`*([{
    removed.
    """
    if raw is None:
        return None
    for line in raw.splitlines():
        if line.strip():
            return _LEAD_STRIP.sub("", line)
    return None


def parse_lead_yesno(raw: Optional[str]) -> Optional[str]:
    """Read a leading yes/no label without an opposite-label conflict.

    Yes/no parsing: leading yes/no; opposite label must not occur in the first
    clause.
    """
    s = first_line(raw)
    if not s:
        return None
    m = _YN_HEAD.match(s)
    if not m:
        return None
    label = "Yes" if m.group(1).lower() == "yes" else "No"
    clause = _CLAUSE_END.split(s, maxsplit=1)[0]
    opposite = _NO_W if label == "Yes" else _YES_W
    if opposite.search(clause):
        return None
    return label


def _match_option_text(text: str, options: Sequence[str]) -> Optional[str]:
    """Match normalized text to exactly one presented option.

    Multiple-choice fallback: NORM(output) equals NORM(text) of exactly one option.
    """
    n = norm(text)
    if not n:
        return None
    hits = [i for i, o in enumerate(options) if norm(o) == n]
    if len(hits) == 1:
        return LETTERS[hits[0]]
    return None


def parse_lead_mcq(raw: Optional[str], options: Sequence[str]) -> Optional[str]:
    """Parse a multiple-choice answer from the first nonempty line.

    `options` = options in PRESENTED order (after rotating the options).
    """
    s = first_line(raw)
    if not s:
        return None
    labels = LETTERS[: len(options)]
    m = _MCQ_RE.match(s)
    if m and m.group(1) in labels:
        return m.group(1)
    return _match_option_text(s.rstrip().rstrip("."), options)


def parse_lead_short(raw: Optional[str], kind: str) -> Optional[str]:
    """Parse a short answer from the first output line.

    Short and open answers: first line, trailing period dropped; validity by
    NORM.
    """
    s = first_line(raw)
    if s is None:
        return None
    s = s.strip().rstrip(".").strip()
    n = norm(s)
    if not n:
        return None
    if kind == "closed_short" and len(n.split()) > 6:
        return None
    return s


def parse_lead(
    raw: Optional[str], kind: str, options: Optional[Sequence[str]] = None
) -> Optional[str]:
    """PARSE-LEAD-v1 dispatcher."""
    if kind == "yesno":
        return parse_lead_yesno(raw)
    if kind == "mcq":
        return parse_lead_mcq(raw, options or [])
    if kind in ("closed_short", "open"):
        return parse_lead_short(raw, kind)
    raise ValueError(kind)


_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    '"': '"',
    "'": "'",
    "`": "`",
    "“": "”",
    "‘": "’",
}


def _exact_core(raw: Optional[str]) -> Optional[str]:
    """Remove permitted wrappers before exact-answer parsing.

    Exact parsing: whole output, whitespace stripped, trailing./! and enclosing
    brackets/quotes removed.
    """
    if raw is None:
        return None
    s = raw.strip()
    while True:
        prev = s
        s = s.rstrip(".!").strip()
        if len(s) >= 2 and s[0] in _PAIRS and (s[-1] == _PAIRS[s[0]]):
            s = s[1:-1].strip()
        if s == prev:
            return s


def parse_exact(
    raw: Optional[str], kind: str, options: Optional[Sequence[str]] = None
) -> Optional[str]:
    """Require the complete output to match one valid answer label.

    PARSE-EXACT-v1, used by S-EXACT: the core must be exactly one label.

    yesno: Yes/No, case-insensitive; mcq: one capital letter in the label set;
    closed_short: a single line obeying the PARSE-LEAD validity rule.
    """
    s = _exact_core(raw)
    if not s:
        return None
    if kind == "yesno":
        low = s.lower()
        return {"yes": "Yes", "no": "No"}.get(low)
    if kind == "mcq":
        labels = LETTERS[: len(options or [])]
        return s if len(s) == 1 and s in labels else None
    if kind in ("closed_short", "open"):
        if "\n" in s:
            return None
        n = norm(s)
        if not n or (kind == "closed_short" and len(n.split()) > 6):
            return None
        return s
    raise ValueError(kind)


def is_correct(label: Optional[str], kind: str, gold: str) -> bool:
    """Compare a parsed answer with the appropriate reference label.

    Scoring: yesno/mcq label equality (mcq gold already shifted); closed_short
    NORM equality.
    """
    if label is None:
        return False
    if kind == "yesno":
        return label.lower() == norm(gold)
    if kind == "mcq":
        return label == gold
    return norm(label) == norm(gold)


def b_is_correct_type(raw: Optional[str]) -> bool:
    """Detect a leading option letter followed by lowercase text.

    E-S1 sub-count: leading letter followed by a lowercase
    word.
    """
    s = first_line(raw)
    return (
        bool(s)
        and re.match(
            (
                "^(?i:(?:the\\s+)?(?:answer|option)\\s*(?:is)?\\s*[:"
                "\\-]?\\s*)?[A-D]\\s+[a-z]"
            ),
            s,
        )
        is not None
    )


def token_f1(pred: Optional[str], gold: str) -> float:
    """E-S7: multiset token F1 after NORM."""
    p = norm(pred).split()
    g = norm(gold).split()
    if not p or not g:
        return 0.0
    from collections import Counter

    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = (common / len(p), common / len(g))
    return 2 * prec * rec / (prec + rec)
