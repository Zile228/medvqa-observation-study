"""Audit context-sensitive token lengths of neutral list controls."""

import hashlib
import re
from collections import Counter

from medvqa.models import MODELS

MODEL_ID = MODELS["Q"]["id"]
MODEL_REVISION = MODELS["Q"]["revision"]
SUFFIX = " | — | —"
CANDIDATES = (
    "Alpha",
    "Bravo",
    "Charlie",
    "Delta",
    "Echo",
    "Foxtrot",
    "Golf",
    "Hotel",
    "India",
    "Juliett",
    "Kilo",
    "Lima",
    "Mike",
    "November",
    "Oscar",
    "Papa",
    "Quebec",
    "Romeo",
    "Sierra",
    "Tango",
    "Uniform",
    "Victor",
    "Whiskey",
    "X-ray",
    "Yankee",
    "Zulu",
)
SEMANTIC_EXCLUDE = frozenset(
    ("Alpha", "Delta", "Echo", "X-ray", "Lima", "Uniform", "Kilo")
)


class ContextError(ValueError):
    pass


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def count(tokenizer, s):
    return len(tokenizer.encode(s, add_special_tokens=False))


def audit_regions(tokenizer, prefix, name, suffix):
    """Assign each whole token to its semantic text region.

    Assign whole tokens; never split IDs or subtract independently encoded fragments.
    """
    text = prefix + name + suffix
    start = len(prefix)
    end = start + len(name)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if len(ids) != len(offsets):
        raise ContextError("OFFSET_COUNT")
    owners = []
    tokens = []
    ambiguities = []
    for j, (token_id, span) in enumerate(zip(ids, offsets)):
        lo, hi = map(int, span)
        if not 0 <= lo < hi <= len(text):
            raise ContextError("OFFSET_BOUNDS_OR_ZERO")
        semantic = set()
        for p in range(lo, hi):
            if not text[p].isspace():
                semantic.add("prefix" if p < start else "name" if p < end else "suffix")
        if len(semantic) > 1:
            owner = "AMBIGUOUS"
            ambiguities.append(j)
        elif semantic:
            owner = next(iter(semantic))
        else:
            prefix_semantic_end = len(prefix.rstrip())
            if hi <= prefix_semantic_end:
                owner = "prefix"
            elif lo < end:
                owner = "name"
            else:
                owner = "suffix"
        owners.append(owner)
        tokens.append(
            {
                "index": j,
                "id": int(token_id),
                "offset": [lo, hi],
                "owner": owner,
                "crosses_name_start": lo < start < hi,
                "crosses_name_end": lo < end < hi,
            }
        )
    counts = {r: owners.count(r) for r in ("prefix", "name", "suffix", "AMBIGUOUS")}
    return {
        "text": text,
        "name_span": [start, end],
        "tokens": tokens,
        "counts": counts,
        "total_tokens": len(ids),
        "ambiguous": bool(ambiguities),
    }


def word_context_gate(tokenizer, word):
    audits = []
    for prefix in (". ", " "):
        a = audit_regions(tokenizer, prefix, word, " |")
        audits.append(
            {
                "context": prefix + word + " |",
                "name_token_count": a["counts"]["name"],
                "ambiguous": a["ambiguous"],
                "tokens": a["tokens"],
            }
        )
    return {
        "word": word,
        "pass": all(
            (a["name_token_count"] == 1 and (not a["ambiguous"]) for a in audits)
        ),
        "contexts": audits,
    }


def adjacency_audit(region, label):
    """Every word in an actual multi-word label must use a distinct owned token."""
    start = region["name_span"][0]
    word_to_tokens = []
    for m in re.finditer("\\S+", label):
        lo = start + m.start()
        hi = start + m.end()
        hits = [
            t["index"]
            for t in region["tokens"]
            if t["owner"] == "name" and t["offset"][0] < hi and (t["offset"][1] > lo)
        ]
        word_to_tokens.append(hits)
    flat = [hits[0] for hits in word_to_tokens if len(hits) == 1]
    return {
        "word_token_indices": word_to_tokens,
        "pass": all((len(h) == 1 for h in word_to_tokens))
        and len(set(flat)) == len(word_to_tokens),
    }


def prepare_W(tokenizer, literal_report):
    if (
        literal_report.get("schema") != "NEUTRAL-LABEL-LEAKAGE-v1"
        or literal_report.get("status") != "LITERAL_W_READY"
    ):
        raise ContextError("VALID_LITERAL_LEAKAGE_REPORT_REQUIRED")
    literal = literal_report["W_literal_eligible"]
    possible = [w for w in CANDIDATES if w not in SEMANTIC_EXCLUDE]
    if literal != [w for w in possible if w in set(literal)]:
        raise ContextError("W_LITERAL_ORDER_OR_SOURCE")
    gates = [word_context_gate(tokenizer, w) for w in literal]
    W = [r["word"] for r in gates if r["pass"]]
    return (W, gates)


def audit_row(tokenizer, row, W):
    items = []
    nd = []
    nl = []
    details = []
    errors = []
    for i, line in enumerate(row["empty"].split("\n"), 1):
        if not line.endswith(SUFFIX):
            raise ContextError("EMPTY_SUFFIX")
        item_line = line[: -len(SUFFIX)]
        m = re.fullmatch("(\\d+\\. )(.+)", item_line)
        if not m or int(m[1].split(".")[0]) != i:
            raise ContextError("EMPTY_NUMBERING")
        prefix, original = (m[1], m[2])
        items.append(item_line)
        old = audit_regions(tokenizer, prefix, original, SUFFIX)
        ki = old["total_tokens"] - old["counts"]["prefix"] - old["counts"]["suffix"]
        k_standalone = (
            count(tokenizer, line) - count(tokenizer, prefix) - count(tokenizer, SUFFIX)
        )
        d = {
            "line_index": i,
            "original_name": original,
            "ki_context": ki,
            "ki_standalone_subtraction": k_standalone,
            "ki_difference": ki - k_standalone,
            "original_name_standalone_tokens": count(tokenizer, original),
            "original_encoding": old,
        }
        if old["ambiguous"] or ki <= 0:
            errors.append(
                "ORIGINAL_BOUNDARY_AMBIGUOUS" if old["ambiguous"] else "KI_NONPOSITIVE"
            )
            d["status"] = "FAIL_UNBUILT"
            details.append(d)
            continue
        label = " ".join((W[(i - 1 + j) % len(W)] for j in range(ki)))
        new = audit_regions(tokenizer, prefix, label, SUFFIX)
        adjacency = adjacency_audit(new, label)
        matched = (
            new["counts"]["name"] == ki and (not new["ambiguous"]) and adjacency["pass"]
        )
        if not matched:
            errors.append("LABEL_CONTEXT_TOKEN_MISMATCH")
        nd.append(prefix + label + SUFFIX)
        nl.append(prefix + label)
        d.update(
            label=label,
            label_standalone_tokens=count(tokenizer, label),
            new_encoding=new,
            adjacency=adjacency,
            label_context_gate=matched,
            status="PASS" if matched else "FAIL",
        )
        details.append(d)
    empty_count = count(tokenizer, row["empty"])
    items_text = "\n".join(items)
    if len(nd) != len(items):
        nd_text = nl_text = None
        nd_count = nl_count = None
        row_delta = None
    else:
        nd_text = "\n".join(nd)
        nl_text = "\n".join(nl)
        nd_count = count(tokenizer, nd_text)
        nl_count = count(tokenizer, nl_text)
        row_delta = nd_count - empty_count
        if abs(row_delta) > 2:
            errors.append("FULL_ROW_DELTA_GT2")
    built_labels = [d["label"] for d in details if "label" in d]
    duplicates = {label: n for label, n in Counter(built_labels).items() if n > 1}
    return {
        "mode": row["mode"],
        "neutral_id": row["neutral_id"],
        "route_key": row["route_key"],
        "status": "FAIL" if errors else "PASS",
        "errors": sorted(set(errors)),
        "empty_tokens": empty_count,
        "items_tokens": count(tokenizer, items_text),
        "neutral_dash_tokens": nd_count,
        "neutral_list_tokens": nl_count,
        "ND_minus_EMPTY_tokens": row_delta,
        "ITEMS_minus_NL_tokens": None
        if nl_count is None
        else count(tokenizer, items_text) - nl_count,
        "empty_sha256": sha_text(row["empty"]),
        "items_text": items_text,
        "ND_text": nd_text,
        "NL_text": nl_text,
        "repeated_label_groups": len(duplicates),
        "repeated_label_extra_instances": sum((n - 1 for n in duplicates.values())),
        "repeated_labels_diagnostic": duplicates,
        "repeated_labels_warning_only": True,
        "item_audits": details,
    }


def preflight(tokenizer, assets, literal_report, membership):
    if type(tokenizer).__name__ != "Qwen2TokenizerFast" or not tokenizer.is_fast:
        raise ContextError("EXACT_QWEN_TOKENIZER_REQUIRED")
    W, gates = prepare_W(tokenizer, literal_report)
    if not W:
        return {
            "schema": "NEUTRAL-LABEL-TOKEN-AUDIT-v1",
            "status": "STOP_W_EMPTY",
            "token_gate": "STOP",
            "W": [],
            "word_context_audit": gates,
            "rows": [],
            "final_strings_built": False,
            "pilot_allowed": False,
        }
    rows = [audit_row(tokenizer, r, W) for r in assets["records"]]
    if len(rows) != 669 or len({(r["mode"], r["neutral_id"]) for r in rows}) != 669:
        raise ContextError("ALL669_SCOPE_REQUIRED")
    primary = set(membership["test_primary_ids"])
    if len(primary) != 542:
        raise ContextError("PRIMARY542_MEMBERSHIP_REQUIRED")
    summaries = {}
    for group, rr in [
        ("ALL669", rows),
        ("TEST599", [r for r in rows if r["mode"] == "TEST"]),
        (
            "PRIMARY542",
            [r for r in rows if r["mode"] == "TEST" and r["neutral_id"] in primary],
        ),
        ("PILOT70", [r for r in rows if r["mode"] == "PILOT"]),
    ]:
        summaries[group] = {
            "n_rows": len(rr),
            "n_fail": sum((r["status"] == "FAIL" for r in rr)),
        }
    if [
        summaries[k]["n_rows"] for k in ("ALL669", "TEST599", "PRIMARY542", "PILOT70")
    ] != [669, 599, 542, 70]:
        raise ContextError("SUBGROUP_COVERAGE")
    nfail = summaries["ALL669"]["n_fail"]
    hard = any(
        (
            "ORIGINAL_BOUNDARY_AMBIGUOUS" in r["errors"]
            or "KI_NONPOSITIVE" in r["errors"]
            for r in rows
        )
    )
    token_gate = (
        "STOP" if hard or 100 * nfail > 2 * 669 else "REVIEW" if nfail else "PASS"
    )
    return {
        "schema": "NEUTRAL-LABEL-TOKEN-AUDIT-v1",
        "status": "TOKEN_" + token_gate + "_COMPOSED_LABEL_CHECK_REQUIRED",
        "token_gate": token_gate,
        "W": W,
        "word_context_audit": gates,
        "rows": rows,
        "summaries": summaries,
        ("ki_rule"): (
            "Single full-line encoding; semantic-region-owned"
            " tokens; leading whitespace belongs to word "
            "region"
        ),
        "standalone_subtraction": "Diagnostic only; never selected as production",
        "row_gate": "abs(ND-EMPTY)<=2",
        "review_threshold": "0<fail<=2% REVIEW; >2% STOP; never drop failing images",
        "final_composed_label_leak_gate": "PENDING_COMPOSED_LABEL_LEAKAGE_CHECK",
        "pilot_allowed": False,
        "not_scored": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
