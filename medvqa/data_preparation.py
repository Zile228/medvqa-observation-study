"""Benchmark preparation, neutral identifiers and held-out cohort construction."""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from medvqa import prompts as P
from medvqa.common import SEED, rgb_hash, sha256_text

_ORDERING_NAMESPACE = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/identifiers.json").read_text(
        encoding="utf-8"
    )
)["ordering_namespace"]

NEUTRAL_RE = re.compile("^(RT|NT|RD|ND)-\\d{4}$")
NATIVE_GROUPS = ("MR", "X-Ray", "Dermoscopy")
VQARAD_EXTRA_GAPS = {
    "synpic52932.jpg": [
        "ABD (X-ray abdomen): no route; gap per routes.json multi_region_rule"
    ]
}


def decode_image(path, retries=1):
    """Return (PIL RGB image, rgb hash) or (None, error).

    Radiology test images get one extra decoding attempt.
    """
    from PIL import Image

    err = None
    for _ in range(1 + retries):
        try:
            with Image.open(path) as im:
                im.load()
                rgb = im.convert("RGB")
            return (rgb, rgb_hash(rgb))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
    return (None, err)


def save_png(rgb, path):
    """Lossless RGB PNG without EXIF/text metadata."""
    from PIL import Image

    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean.save(path, format="PNG", optimize=False)


def neutral_rank_key(dataset, split, original_name):
    return hashlib.sha256(
        f"{_ORDERING_NAMESPACE}|{dataset}|{split}|{original_name}".encode("utf-8")
    ).hexdigest()


def assign_neutral_ids(prefix, images):
    """images: list of dicts with dataset/split/original_name.

    Rank by sha256, 1-based.
    """
    keyed = sorted(
        images,
        key=lambda x: neutral_rank_key(x["dataset"], x["split"], x["original_name"]),
    )
    for rank, im in enumerate(keyed, 1):
        im["neutral_id"] = f"{prefix}-{rank:04d}"
    return keyed


def find_slake_json(slake_dir):
    d = Path(slake_dir)
    out = {}
    for p in d.rglob("*.json"):
        n = p.name.lower()
        if n == "train.json":
            out["train"] = p
        elif n in ("validate.json", "validation.json"):
            out["validation"] = p
        elif n == "test.json":
            out["test"] = p
    missing = {"train", "validation", "test"} - set(out)
    if missing:
        raise FileNotFoundError(f"SLAKE json missing: {missing}")
    return out


def load_slake(slake_dir):
    """English QA only. Returns {split: [row,...]} in file order."""
    files = find_slake_json(slake_dir)
    out = {}
    for split, p in files.items():
        rows = json.loads(Path(p).read_text(encoding="utf-8"))
        out[split] = [r for r in rows if r.get("q_lang") == "en"]
    return out


def slake_image_path(slake_dir, img_name):
    d = Path(slake_dir)
    for cand in (d / "imgs" / img_name, d / img_name):
        if cand.is_file():
            return cand
    hits = list(d.rglob(img_name.split("/")[-1]))
    hits = [h for h in hits if str(h).replace("\\", "/").endswith(img_name)]
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(img_name)


def compare_slake_with_rq1(slake_test_rows, rq1_ref, new_rgb_by_imgid):
    """Compare official SLAKE test IDs with the earlier study sample.

    ID-by-ID comparison of official test.json (EN) with the RQ1 SLAKE cohort (qid,
    img_id, RGB).
    """
    ref_q = {r["source_qid"] for r in rq1_ref["slake"]}
    ref_img = {str(r["source_image_id"]) for r in rq1_ref["slake"]}
    new_q = {str(r["qid"]) for r in slake_test_rows}
    new_img = {str(r["img_id"]) for r in slake_test_rows}
    ref_rgb = {str(r["source_image_id"]): r["image_rgb"] for r in rq1_ref["slake"]}
    rgb_mismatch = sorted(
        (
            i
            for i in ref_img & new_img
            if new_rgb_by_imgid.get(i)
            and ref_rgb.get(i)
            and (new_rgb_by_imgid[i] != ref_rgb[i])
        )
    )
    rep = {
        "n_questions_official_en": len(new_q),
        "n_images_official": len(new_img),
        "n_questions_rq1": len(ref_q),
        "n_images_rq1": len(ref_img),
        "questions_only_in_official": sorted(new_q - ref_q),
        "questions_only_in_rq1": sorted(ref_q - new_q),
        "images_only_in_official": sorted(new_img - ref_img),
        "images_only_in_rq1": sorted(ref_img - new_img),
        "images_rgb_differ": rgb_mismatch,
    }
    rep["identical"] = not any(
        (
            rep[k]
            for k in (
                "questions_only_in_official",
                "questions_only_in_rq1",
                "images_only_in_official",
                "images_only_in_rq1",
                "images_rgb_differ",
            )
        )
    )
    rep["claim"] = (
        "RQ1 SLAKE cohort == full official English test (ID-by-ID)"
        if rep["identical"]
        else (
            "Official test set differs from the reference cohort; cohort left unchanged"
        )
    )
    return rep


def load_omni_qa(omni_root):
    """Index Open-access QA rows by 'Omni/<dataset>/<question_id>'."""
    files = sorted(Path(omni_root).rglob("QA_information/Open-access/*.json"))
    if not files:
        raise FileNotFoundError(
            "OmniMedVQA QA_information/Open-access/*.json not found"
        )
    idx = {}
    for f in files:
        rows = json.loads(f.read_text(encoding="utf-8"))
        for r in rows if isinstance(rows, list) else [rows]:
            idx[f"Omni/{r.get('dataset')}/{r.get('question_id')}"] = r
    return idx


def omni_image_path(omni_root, rel):
    root = Path(omni_root)
    for base in (root, root / "OmniMedVQA"):
        if (base / rel).is_file():
            return base / rel
    raise FileNotFoundError(rel)


def omni_item(row):
    """Options and gold label at order 0.

    Raises ValueError for an ineligible row (should not happen: frozen pool).
    """
    opts = [
        str(row[k]).strip()
        for k in ("option_A", "option_B", "option_C", "option_D")
        if row.get(k) is not None and str(row[k]).strip()
    ]
    if not 2 <= len(opts) <= 4:
        raise ValueError("option count")
    gold = str(row["gt_answer"]).strip()

    def norm(s):
        return re.sub("\\s+", " ", s.casefold()).strip(" .")

    match = [i for i, o in enumerate(opts) if norm(o) == norm(gold)]
    if len(match) != 1 or len({norm(o) for o in opts}) != len(opts):
        raise ValueError("gold does not map to exactly one option")
    return {
        "question": str(row["question"]),
        "options": opts,
        "gold_answer": gold,
        "gold_label": "ABCD"[match[0]],
        "question_type": row.get("question_type"),
        "modality_type": row.get("modality_type"),
    }


def select_native_heldout(candidates, qa_index, omni_root, exposed_rgb, per_group=100):
    """Select usable native images in the fixed reserve order.

    Accept the first `per_group` usable entries per modality in reserve order.
    Skip broken images and RGB duplicates of previously exposed or accepted images.
    """
    accepted, ledger, seen = ({}, [], set(exposed_rgb))
    for group in NATIVE_GROUPS:
        accepted[group] = []
        for pos, c in enumerate(candidates["pools"][group]):
            if len(accepted[group]) >= per_group:
                break
            ev = {
                "group": group,
                "pool_position": pos,
                "reserve_index": c["reserve_index"],
                "question_id": c["question_id"],
            }
            try:
                path = omni_image_path(omni_root, c["source_image_path"])
                rgb, h = decode_image(path, retries=0)
            except FileNotFoundError:
                rgb, h = (None, "missing file")
            if rgb is None:
                ledger.append(dict(ev, decision="REPLACED_BROKEN", error=h))
                continue
            if h in seen:
                ledger.append(dict(ev, decision="REPLACED_RGB_DUPLICATE", rgb=h))
                continue
            row = qa_index.get(c["question_id"])
            if row is None:
                raise ValueError(
                    f"QA row not found for frozen candidate {c['question_id']}"
                )
            item = omni_item(row)
            seen.add(h)
            accepted[group].append(
                {
                    "candidate": c,
                    "path": str(path),
                    "rgb": h,
                    "size": list(rgb.size),
                    "item": item,
                }
            )
            ledger.append(dict(ev, decision="ACCEPTED", rgb=h))
        if len(accepted[group]) < per_group:
            raise ValueError(f"reserve exhausted for {group}: {len(accepted[group])}")
    return (accepted, ledger)


COLS = ("image-left", "centre", "image-right")
ROWS = ("upper", "middle", "lower")


def bbox_region(b, W, H):
    x, y, w, h = b
    cx, cy = (x + w / 2, y + h / 2)
    c = min(2, max(0, int(cx / (W / 3))))
    r = min(2, max(0, int(cy / (H / 3))))
    return (c, r)


def annotation_entries(detection, W, H):
    out = []
    for d in detection:
        for label, b in d.items():
            c, r = bbox_region(b, W, H)
            out.append(
                {
                    "label": label,
                    "bbox": [round(float(v), 1) for v in b],
                    "col": c,
                    "row": r,
                }
            )
    return out


def annotation_text(entries, W, H):
    lines = []
    for e in entries:
        x, y, w, h = e["bbox"]
        lines.append(
            (
                f"{e['label']}"
                ": region "
                f"{COLS[e['col']]}"
                ", "
                f"{ROWS[e['row']]}"
                "; bounding box x="
                f"{x:g}"
                ", y="
                f"{y:g}"
                ", w="
                f"{w:g}"
                ", h="
                f"{h:g}"
                " (pixels, image size "
                f"{W}"
                "×"
                f"{H}"
                ")"
            )
        )
    return "\n".join(lines)


def region_phrase(c, r):
    return f"{COLS[c]}, {ROWS[r]} region"


def build_probes(
    annotations, seed=SEED, n_a_true=100, n_a_swap=100, n_b_true=25, n_b_swap=25
):
    """250 known-answer statements (judge_prompts.md J-PROBE).

    Swap = image-left<->image-right, only bbox centres outside the centre column, and
    only if the same label has no bbox in the mirrored column of that image.
    """
    cands = []
    for nid, a in sorted(annotations.items()):
        for k, e in enumerate(a["entries"]):
            cands.append((nid, k, e))
    rng = random.Random(seed)
    true_pool = list(cands)
    rng.shuffle(true_pool)
    swap_pool = []
    for nid, k, e in cands:
        if e["col"] == 1:
            continue
        mirror = 2 - e["col"]
        if any(
            (
                o["label"] == e["label"] and o["col"] == mirror
                for o in annotations[nid]["entries"]
            )
        ):
            continue
        swap_pool.append((nid, k, e))
    rng.shuffle(swap_pool)
    probes = []

    def add(src, n, kind, swap):
        for nid, k, e in src[:n]:
            c = 2 - e["col"] if swap else e["col"]
            if kind == "A":
                st = f"{e['label']} is visible in the {region_phrase(c, e['row'])}"
            else:
                rp = region_phrase(c, e["row"])
                st = (
                    f"{rp[0].upper() + rp[1:]}"
                    ": the "
                    f"{e['label']}"
                    " appears in this part of the image, with no "
                    "other structure described there."
                )
            probes.append(
                {
                    "probe_id": f"P{len(probes) + 1:03d}",
                    "neutral_id": nid,
                    "entry": k,
                    "template": kind,
                    "statement": st,
                    "expected": "contradicted" if swap else "supported",
                    "expected_label": "contradicted" if swap else "supported",
                }
            )

    add(true_pool, n_a_true, "A", False)
    add(swap_pool, n_a_swap, "A", True)
    add(true_pool[n_a_true:], n_b_true, "B", False)
    add(swap_pool[n_a_swap:], n_b_swap, "B", True)
    return {
        "version": "JPROBE-v1",
        "seed": seed,
        "requested": {
            "A_true": n_a_true,
            "A_swap": n_a_swap,
            "B_true": n_b_true,
            "B_swap": n_b_swap,
        },
        "counts": {
            f"{t}_{e}": n
            for (t, e), n in Counter(
                ((p["template"], p["expected"]) for p in probes)
            ).items()
        },
        "probes": probes,
    }


def _h(*parts):
    return hashlib.sha256("|".join((str(p) for p in parts)).encode("utf-8")).hexdigest()


def stratified_pick(ids, stratum_of, n, tag, seed=SEED):
    """Select a deterministic proportional sample within strata.

    Proportional allocation (largest remainder) over strata, then sha256(seed|tag|id)
    order within stratum.
    """
    strata = defaultdict(list)
    for i in ids:
        strata[stratum_of[i]].append(i)
    total = len(ids)
    if n >= total:
        return sorted(ids)
    quotas = {s: n * len(v) / total for s, v in strata.items()}
    alloc = {s: int(q) for s, q in quotas.items()}
    rest = n - sum(alloc.values())
    for s in sorted(
        strata, key=lambda s: (-(quotas[s] - alloc[s]), -len(strata[s]), s)
    )[:rest]:
        alloc[s] += 1
    out = []
    for s, members in strata.items():
        out += sorted(members, key=lambda i: _h(seed, tag, i))[: alloc[s]]
    return sorted(out)


def first_by_hash(ids, n, tag, seed=SEED):
    return sorted(sorted(ids, key=lambda i: _h(seed, tag, i))[:n])


def observer_prompt_manifest(
    bank, routing, questions, options, answers, original_names
):
    """routing: {neutral_id: router result}.

    Runs check 1 (questions/options/gold) and check 3 (original names) on every observer
    prompt that will exist. Returns manifest + a counts-only report (no strings).
    """
    route_keys = sorted({r["route_key"] for r in routing.values()})
    allowed = bank.allowed_texts(route_keys)
    locked_lines = bank.locked_lines()
    manifest, short_hits = ({}, 0)
    checked = {}
    coll = {"question": 0, "option": 0, "answer": 0}
    coll_lines = set()
    for nid, r in sorted(routing.items()):
        for arm in P.ARMS:
            text, src = bank.prompt(arm, r["routes"])
            h = sha256_text(text)
            if h not in checked:
                res = P.check1_observer_prompt(
                    text,
                    allowed,
                    questions,
                    options,
                    answers,
                    locked_lines=locked_lines,
                )
                P.check3_no_original_names(text, original_names)
                checked[h] = res["short_overlaps"]
                for k, v in res["template_collisions"].items():
                    coll[k] += v
                coll_lines.update(res["template_collision_lines_sha256"])
            short_hits += checked[h]
            header = next(
                (ln for ln in text.splitlines() if ln.startswith("Image context:")),
                None,
            )
            manifest[f"{arm}|{nid}"] = {
                "route_key": r["route_key"],
                "prompt_source": src,
                "prompt_sha256": h,
                "header": header,
            }
    return (
        {"version": "OBS-PROMPT-MANIFEST-v1", "entries": manifest},
        {
            "check1": "pass",
            "check3": "pass",
            "distinct_prompts": len(checked),
            "short_string_overlaps_counted_not_failed": short_hits,
            ("check1b_rule"): (
                "strings inside one locked template line are counted, not failed"
            ),
            "check1b_template_collisions_by_distinct_prompt": coll,
            "check1b_template_collision_lines": len(coll_lines),
            "check1b_template_collision_lines_sha256": sorted(coll_lines),
        },
    )
