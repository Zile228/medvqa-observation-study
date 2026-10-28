"""CPU regression tests for prompt isolation, routing and experiment execution."""

import inspect
import json
import random as _random
import re as _re
import sys
from pathlib import Path

import pytest

from medvqa import common as U
from medvqa import conditions as CL
from medvqa import data_preparation as PR
from medvqa import interventions as I
from medvqa import judges as JG
from medvqa import models as B
from medvqa import native_data as NV
from medvqa import observers as G
from medvqa import prompts as P
from medvqa import routing as R

CODE = Path(__file__).resolve().parents[2]
THESIS = CODE
PROMPTS = THESIS / "resources/protocol" / "prompts"


@pytest.fixture(scope="module")
def bank():
    return P.ObserverPromptBank(PROMPTS)


def test_bank_reproduces_all_100_locked_files(bank):
    assert len(bank.files) == 100


def test_multi_region_numbering_continuous(bank):
    t, src = bank.prompt("ARM-GRD", ["RT-XR-CHEST", "RT-CT-ABDPEL"])
    nums = [
        int(line.split(".")[0])
        for line in t.split("Items:\n")[1].split("\n")
        if line[:1].isdigit()
    ]
    assert (
        nums == list(range(1, 26))
        and src.startswith("composed:")
        and ("chest + computed tomography" in t)
    )


def test_fallback_same_for_all_arms(bank):
    assert len({bank.prompt(a, ["RT-FALLBACK"])[0] for a in P.ARMS}) == 1


def test_check1_fails_on_injected_question(bank):
    t, _ = bank.prompt("ARM-GRD", ["RT-MR-KNEE"])
    allowed = bank.allowed_texts(["RT-MR-KNEE"])
    q = "Is there a tear of the medial meniscus in this knee image?"
    P.check1_observer_prompt(t, allowed, [q], ["normal"], ["no"])
    with pytest.raises(P.LeakError, match="check1a"):
        P.check1_observer_prompt(t + "\n" + q, allowed, [q])
    with pytest.raises(P.LeakError, match="check1b"):
        P.check1_observer_prompt(t + "\n" + q, allowed | {t + "\n" + q}, [q])
    with pytest.raises(P.LeakError, match="check1b"):
        P.check1_observer_prompt(
            t + "\nAnswer: anterior cruciate rupture",
            allowed | {t + "\nAnswer: anterior cruciate rupture"},
            [],
            [],
            ["anterior cruciate rupture"],
        )


def test_check1b_template_line_collisions(bank):
    """Count fixed-template collisions while rejecting injected lines."""
    locked = bank.locked_lines()
    t, _ = bank.prompt("ARM-GRD", ["RT-MAMMO"])
    allowed = bank.allowed_texts(["RT-MAMMO"])
    with pytest.raises(P.LeakError, match="check1b"):
        P.check1_observer_prompt(t, allowed, [], ["Architectural distortion"])
    res = P.check1_observer_prompt(
        t,
        allowed,
        [],
        ["Architectural distortion", "Normal tissue"],
        [],
        locked_lines=locked,
    )
    assert res["template_collisions"] == {"question": 0, "option": 2, "answer": 0}
    assert len(res["template_collision_lines_sha256"]) == 1
    ta, _ = bank.prompt("ARM-ANA", ["RT-CT-HEAD"])
    res = P.check1_observer_prompt(
        ta,
        bank.allowed_texts(["RT-CT-HEAD"]),
        ["What is visible in this image?"],
        locked_lines=locked,
    )
    assert res["template_collisions"]["question"] == 1
    tc = bank.compose("ARM-GRD", ["RT-CT-HEAD", "RT-CT-FACE"])
    res = P.check1_observer_prompt(
        tc,
        bank.allowed_texts(["RT-CT-HEAD+RT-CT-FACE"]),
        [],
        ["Soft tissue swelling"],
        locked_lines=locked,
    )
    assert res["template_collisions"]["option"] == 1
    q = "Is there a tear of the medial meniscus in this knee image?"
    tk, _ = bank.prompt("ARM-GRD", ["RT-MR-KNEE"])
    with pytest.raises(P.LeakError, match="check1b"):
        P.check1_observer_prompt(
            tk + "\n" + q, {tk + "\n" + q}, [q], locked_lines=locked
        )
    with pytest.raises(P.LeakError, match="check1b"):
        P.check1_observer_prompt(
            tk + "\nAnswer: anterior cruciate rupture",
            {tk + "\nAnswer: anterior cruciate rupture"},
            [],
            [],
            ["anterior cruciate rupture"],
            locked_lines=locked,
        )


def test_check2_router_inputs_only():
    params = list(inspect.signature(R.route_for).parameters)
    assert params == ["neutral_id", "record", "route_pairs"]
    spec = R.load_routes_spec(PROMPTS)
    rec = R.slake_record([{"modality": "CT", "location": "Abdomen"}] * 3)
    pairs = R.build_route_pairs({"RT-0001": rec}, spec)
    r1 = R.route_for("RT-0001", rec, pairs)
    rec2 = dict(
        rec,
        question="Is this the liver?",
        answer="yes",
        path="Images/liver_lesion/x.png",
    )
    assert (
        R.route_for("RT-0001", rec2, pairs) == r1 and r1["route_key"] == "RT-CT-ABDPEL"
    )


def test_check3_original_names():
    for bad in (
        "look at synpic54610.jpg",
        "Images/RadImageNet/a.png",
        "xmlab102/source.jpg",
    ):
        with pytest.raises(P.LeakError):
            P.check3_no_original_names(bad)
    with pytest.raises(P.LeakError):
        P.check3_no_original_names("image covid_case_17 here", ["covid_case_17"])
    assert P.check3_no_original_names("plain prompt RT-0001 not named")


def test_check4_sealed_guard(tmp_path):
    f = tmp_path / "prep_sealed" / "id_map.json"
    f.parent.mkdir()
    f.write_text("{}")
    with U.sealed_guard():
        with pytest.raises(PermissionError):
            open(f)
        with pytest.raises(PermissionError):
            Path(f).read_text()
    with U.sealed_guard(allow={"id_map.json"}):
        assert open(f).read() == "{}"
    with pytest.raises(U.IntegrityError):
        U.assert_no_sealed_inputs([tmp_path])


def test_check5_signatures():
    ed = I.EditorPrompts(PROMPTS / "templates" / "intervention_editor_prompts.md")
    assert P.check5_intervention_signature(
        [
            I.shuf_assign,
            I.empty_observation,
            I.gray_image,
            I.corruption_plan,
            ed.para_prompt,
            ed.corr_prompt,
            I.para_checks,
            I.corr_checks,
        ]
    )

    def bad_editor(observation, question):
        return observation

    with pytest.raises(P.LeakError):
        P.check5_intervention_signature([bad_editor])


def test_check6_answer_template(tmp_path):
    t = P.AnswerTemplate(PROMPTS / "templates" / "answerer_template.txt")
    pr = t.render("none", "Is this a CT?", "yesno")
    assert pr == (
        "Image observation:\nnone\n\nQuestion: Is this a CT?"
        "\nAnswer with one word: Yes or No."
    )
    assert P.check6_answer_prompt(t, pr, "yesno")
    bad = tmp_path / "t.txt"
    bad.write_text(
        (
            "Benchmark-declared modality: CT\nImage "
            "observation:\n{OBS_TEXT}\n\nQuestion: {QUESTION}\n"
            "{OPTIONS_BLOCK}\n{FORMAT_LINE}\n"
        )
    )
    tb = P.AnswerTemplate(bad)
    with pytest.raises(P.LeakError):
        P.check6_answer_prompt(tb, tb.render("none", "Q?", "yesno"), "yesno")
    with pytest.raises(P.LeakError):
        P.check6_answer_prompt(
            t,
            "Modality: CT\n" + pr.replace("Answer with one word: Yes or No.", ""),
            "yesno",
        )


def test_check7_sha(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    lock = tmp_path / "lock.sha256"
    lock.write_text(f"{U.sha256_file(f)}  a.txt\n" + "0" * 64 + "  b.txt\n")
    assert U.verify_locked_inputs(lock, {"a.txt": f}, "TEST")["verified"]
    f.write_text("y")
    with pytest.raises(U.LockError):
        U.verify_locked_inputs(lock, {"a.txt": f}, "PILOT")
    g = tmp_path / "b.txt"
    g.write_text("z")
    assert U.verify_locked_inputs(lock, {"b.txt": g}, "PILOT")["unlisted_used"] == [
        "b.txt"
    ]
    with pytest.raises(U.LockError):
        U.verify_locked_inputs(lock, {"b.txt": g}, "TEST")


def test_check9_nonconstant_goes_fallback():
    rec = R.slake_record(
        [
            {"modality": "CT", "location": "Abdomen"},
            {"modality": "CT", "location": "Pelvic Cavity"},
        ]
    )
    assert rec["status"] == "nonconstant"
    pairs = {"pairs": [], "route_order": []}
    assert R.route_for("RD-0001", rec, pairs)["route_key"] == "RT-FALLBACK"
    assert R.vqarad_record(["ABD", "CHEST"], "CT", "inv")["status"] == "nonconstant"


def test_check10_unknown_pair_stops_build():
    spec = R.load_routes_spec(PROMPTS)
    rec = R.slake_record([{"modality": "X-Ray", "location": "Chest_lung"}])
    with pytest.raises(R.RouteBuildStop):
        R.build_route_pairs({"RD-0002": rec}, spec)
    sup = {
        "rows": [
            {
                "source_kind": "SLAKE",
                "modality": "X-Ray",
                "region": "Chest_lung",
                "route": "RT-XR-CHEST",
                "reason": "variant",
            }
        ]
    }
    assert (
        R.route_for("RD-0002", rec, R.build_route_pairs({"RD-0002": rec}, spec, sup))[
            "route_key"
        ]
        == "RT-XR-CHEST"
    )
    with pytest.raises(ValueError):
        R.lookup_rules(
            spec,
            {
                "rows": [
                    {
                        "source_kind": "SLAKE",
                        "modality": "X",
                        "region": "Y",
                        "route": "RT-NEW",
                        "reason": "r",
                    }
                ]
            },
        )


def test_c66_gap_and_multiregion():
    spec = R.load_routes_spec(PROMPTS)
    rec = R.vqarad_record(
        ["CHEST"], "X-Ray", "inv", PR.VQARAD_EXTRA_GAPS["synpic52932.jpg"]
    )
    r = R.route_for("RT-0003", rec, R.build_route_pairs({"RT-0003": rec}, spec))
    assert r["route_key"] == "RT-XR-CHEST" and r["gaps"]


def test_check11_payload():
    p = G.obs_payload("Describe the image.", b"\x89PNG fake")
    assert G.check11(p, expect_image=True)
    for bad in (dict(p, user="u1"), dict(p, metadata={"a": 1})):
        with pytest.raises(P.LeakError):
            G.check11(bad, expect_image=True)
    with pytest.raises(P.LeakError):
        G.check11(G.obs_payload("Image RT-0001 here", b"x"), expect_image=True)
    with pytest.raises(P.LeakError):
        G.check11(G.obs_payload("file synpic100.jpg", b"x"), expect_image=True)
    lg = G.loggable(p)
    assert (
        "base64" not in json.dumps(lg).replace("png base64 sha256", "")
        and set(lg) == G.ALLOWED_KEYS
    )


def test_gpt_retry_permanent_error():
    calls = []

    def send(payload):
        calls.append(1)
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "", "refusal": None}}
            ]
        }

    status, resp, log = G.call_observation(
        send, G.obs_payload("x", b"y"), sleep=lambda s: calls.append(("sleep", s))
    )
    assert status == "obs_error" and len(log) == 3 and (("sleep", 60) in calls)


def test_answer_kind_and_cyclic():
    assert P.answer_kind("CLOSED", "Is there a mass?") == "yesno"
    assert P.answer_kind("CLOSED", "Is it CT or MRI?") == "closed_short"
    assert P.answer_kind("CLOSED", "Which side?") == "closed_short"
    assert P.answer_kind("OPEN", "Is there...?") == "open"
    assert [o["text"] for o in P.cyclic_order(["a", "b", "c"], 1)] == ["b", "c", "a"]


def test_neutral_ids_deterministic():
    ims = [
        {"dataset": "SLAKE", "split": "test", "original_name": f"xmlab{i}/source.jpg"}
        for i in range(5)
    ]
    a = [x["neutral_id"] for x in PR.assign_neutral_ids("RT", [dict(i) for i in ims])]
    b = [
        x["neutral_id"]
        for x in PR.assign_neutral_ids("RT", [dict(i) for i in reversed(ims)])
    ]
    assert (
        sorted(a) == ["RT-0001", "RT-0002", "RT-0003", "RT-0004", "RT-0005"] and a == b
    )


def test_shuf_never_self_and_tiers():
    ids = ["A", "B", "C", "D"]
    donor, tier = I.shuf_assign(
        ids,
        {"A": "r1", "B": "r1", "C": "r2", "D": "route_c"},
        {"A": "CT", "B": "CT", "C": "CT", "D": "MR"},
        {i: "X" for i in ids},
    )
    assert all((donor[i] != i for i in ids if donor[i])) and tier["C"] == "modality"
    assert donor["D"] is None and tier["D"] == "none_int_missing"


def test_para_and_corr_checks():
    o = (
        "1. Liver | visible | small area on the "
        "image-left side\n2. Spleen | not visible | not "
        "assessable"
    )
    assert all(
        I.para_checks(
            o,
            (
                "1. Liver | visible | a small region at the "
                "image-left side\n2. Spleen | not visible | not "
                "assessable"
            ),
        ).values()
    )
    assert not I.para_checks(
        o,
        (
            "1. Liver | visible | small area on the "
            "image-right side\n2. Spleen | not visible | not "
            "assessable"
        ),
    )["keywords"]
    out = json.dumps(
        {
            "edited_text": o.replace("image-left", "image-right"),
            "original_span": "image-left",
            "new_span": "image-right",
        }
    )
    c, txt = I.corr_checks(o, out, 1, "LAT")
    assert all(c.values()) and "image-right" in txt
    c, _ = I.corr_checks(o, out, 1, "SEV")
    assert c["pair_in_list"] is False


def test_corruption_plan_balanced():
    obs = {
        (f"N{i}"): (
            "1. A | visible | small left thing\n2. B | partly visible | thin upper part"
        )
        for i in range(9)
    }
    plan = I.corruption_plan(obs)
    assert plan["counts"] == {"PRES": 3, "LAT": 3, "SEV": 3}


def test_cells_36():
    assert len(CL.all_cells()) == 36 and len({c["cell"] for c in CL.all_cells()}) == 36
    assert len(CL.RQ1_OPEN_CELLS) == 11


def test_label_logprobs_math():
    pytest.importorskip("torch")
    import torch

    lm = {"Yes": [2, 3], "No": [5]}
    logits = torch.zeros(8)
    logits[2] = 2.0
    r = B.label_logprobs(logits, lm, ["Yes", "No"])
    z = torch.logsumexp(logits, 0)
    assert abs(r["lp"]["Yes"] - float(torch.logsumexp(logits[[2, 3]], 0) - z)) < 1e-06
    assert abs(sum(r["p_tilde"].values()) - 1) < 1e-09 and r["finite"]


TOK = THESIS / "resources/annotations" / "scratch" / "tok"


@pytest.mark.skipif(
    not (TOK / "llava15_tokenizer.json").is_file(),
    reason="local tokenizer files absent",
)
def test_label_map_real_tokenizers_nonempty():
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer

    out = {}
    for key, fn in (
        ("llava-1.5", "llava15_tokenizer.json"),
        ("qwen2.5-vl", "qwen25vl_tokenizer.json"),
    ):
        t = Tokenizer.from_file(str(TOK / fn))
        m = B.build_label_map(
            lambda i: t.decode([i], skip_special_tokens=False), t.get_vocab_size()
        )
        out[key] = {c: len(v) for c, v in m.items()}
    (CODE / "tests" / "label_map_tokenizer_check.json").write_text(
        json.dumps(out, indent=1)
    )
    assert all((n > 0 for v in out.values() for n in v.values()))


def test_prec1_hooks_cpu():
    pytest.importorskip("torch")
    import torch

    m = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 2)
    ).to(torch.bfloat16)
    B.upcast_fp32(m[1], torch.bfloat16)
    y = m(torch.randn(1, 4, dtype=torch.bfloat16))
    assert m[1].weight.dtype == torch.float32 and y.dtype == torch.bfloat16


def test_shard_plan_partition():
    keys = [f"k{i}" for i in range(23)]
    parts = [U.plan_shard(keys, 4, s) for s in range(4)]
    assert sum(parts, []) == keys


def test_force_greedy_overrides_repo_penalty():
    pytest.importorskip("transformers")
    from transformers import GenerationConfig

    class M:
        generation_config = GenerationConfig(
            repetition_penalty=1.05,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

    orig, eff = B.force_greedy(M)
    assert (
        orig["repetition_penalty"] == 1.05
        and eff["repetition_penalty"] == 1.0
        and (eff["do_sample"] is False)
    )
    assert (
        B.GREEDY_KW["repetition_penalty"] == 1.0
        and B.GREEDY_KW["no_repeat_ngram_size"] == 0
    )
    M.generation_config.repetition_penalty = 1.1
    with pytest.raises(RuntimeError):
        B.assert_greedy(M.generation_config)


class _E(Exception):
    def __init__(self, code):
        super().__init__(str(code))
        self.status_code = code


def test_breaker_classification_and_no_record():
    assert (
        G.classify_exception(_E(401)) == "system"
        and G.classify_exception(_E(503)) == "system"
    )
    assert (
        G.classify_exception(_E(400)) == "content"
        and G.classify_exception(RuntimeError("insufficient_quota")) == "system"
    )

    def send(payload):
        raise _E(401)

    with pytest.raises(G.SystemAPIError):
        G.call_observation(send, G.obs_payload("x", b"y"), sleep=lambda s: None)
    written = []
    buf = G.ObsErrorBuffer(written.append, limit=3)
    err = {"status": "obs_error", "attempts": [{"error": "empty_output"}]}
    buf.add(dict(err))
    buf.add({"status": "ok"})
    assert [r["status"] for r in written] == ["obs_error", "ok"]
    buf.add(dict(err))
    buf.add(dict(err))
    with pytest.raises(G.SystemAPIError):
        buf.add(dict(err))
    assert len(written) == 2 and buf.pending == []


def test_editor_deadline_and_content_errors():
    import datetime as dt

    calls = []

    def send(payload):
        calls.append(1)
        raise _E(400)

    raw, meta = G.call_editor(send, G.edit_payload("t", 1), sleep=lambda s: None)
    assert raw is None and len(calls) == 3 and (len(meta["http_content_errors"]) == 3)
    old = G.GPT_DEADLINE_UTC
    G.GPT_DEADLINE_UTC = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    try:
        with pytest.raises(G.DeadlinePassed):
            G.call_editor(
                lambda p: {"choices": [{"message": {"content": "x"}}]},
                G.edit_payload("t", 1),
                sleep=lambda s: None,
            )
    finally:
        G.GPT_DEADLINE_UTC = old


def test_env_pins():
    v = U.installed_versions(["numpy"])["numpy"]
    assert U.check_env_pins({"numpy": v}, "TEST")["numpy"]["installed"] == v
    with pytest.raises(U.LockError):
        U.check_env_pins({"numpy": "0.0.0"}, "TEST")
    with pytest.raises(U.LockError):
        U.check_env_pins({"numpy": None}, "TEST")
    assert U.check_env_pins({"numpy": "0.0.0"}, "PILOT")


def test_ax1_schema_and_nq1_fence():
    refs = {
        "statements": [
            {
                "id": 1,
                "sid": "OBS|x#1",
                "obs_key": "OBS|x",
                "is_probe": False,
                "probe_id": None,
                "text": "a",
            },
            {
                "id": 2,
                "sid": "P001",
                "obs_key": None,
                "is_probe": True,
                "probe_id": "P001",
                "text": "b",
            },
        ],
        "skipped_evasion": [],
    }
    assert JG.assert_ax1_refs(refs)
    with pytest.raises(AssertionError):
        JG.assert_ax1_refs(
            {"statements": [{"id": 1, "sid": "s", "text": "a"}], "skipped_evasion": []}
        )
    fenced = '```json\n{"relevance": 2}\n```'
    assert JG.json_valid("J-AX3", fenced, strip_fence=True) and (
        not JG.json_valid("J-AX3", fenced, strip_fence=False)
    )
    assert not JG.json_valid(
        "J-AX3", '```json\n```json\n{"relevance": 2}\n```\n```', strip_fence=True
    )
    st, ev = JG.split_statements(
        "1. A | not visible | not assessable\n2. B | visible | small area"
    )
    assert st == ["B: small area"] and ev == ["1. A | not visible | not assessable"]


def test_default_synonyms_default_empty():
    assert I.CORR_SYNONYMS == {}
    plan = I.corruption_plan({"N1": "1. Liver | visible | small left area"})
    assert (
        plan["assignments"]["N1"]["item_name"] == "Liver"
        and plan["assignments"]["N1"]["synonyms"] == []
    )


def test_judge_json_rule_matches_analysis():
    sys.path.insert(0, str(CODE))
    from medvqa.analysis import constants as BC
    from medvqa.analysis import metrics as BM

    assert JG.JUDGE_JSON_FENCE_REGEX == BC.JUDGE_JSON_FENCE_REGEX
    assert JG.STRIP_ONE_JSON_FENCE is True and BC.JUDGE_STRIP_ONE_JSON_FENCE is True
    corpus = [
        '{"relevance": 1}',
        ' {"relevance": 2}\n',
        '```json\n{"relevance": 1}\n```',
        '```\n{"relevance": 0}\n```',
        '```json\r\n{"relevance": 1}\r\n```',
        '```json {"relevance": 1}```',
        'Answer: {"relevance": 1}',
        '```json\n```json\n{"relevance": 1}\n```\n```',
        '```python\n{"relevance": 1}\n```',
        '{"relevance": 3}',
        "",
        '{"verdict": "correct"}',
        '{"items_addressed": 3, "diagnosis": "no", "unsupported_claim": "no"}',
        '```json\n{"labels": [{"id": 1, "label": "supported"}]}\n```',
        '{"labels": [{"id": "1", "label": "supported"}]}',
    ]
    for task in ("J-AX1", "J-AX2", "J-AX3", "J-OPEN"):
        for t in corpus:
            for fence in (False, True):
                ids = {1} if task == "J-AX1" else None
                a = JG.json_valid(task, t, strip_fence=fence, ids=ids)
                b = BM._judge_payload(task, t, ids, fence) is not None
                assert a == b, (task, t, fence, a, b)
            assert JG.json_valid(task, t, ids=ids) == (
                BM.parse_judge_json(task, t, ids) is not None
            )


def test_nq1_outer_fence_preserves_schema_and_ids():
    from medvqa.analysis.metrics import parse_judge_json

    body = '{"labels": [{"id": 1, "label": "supported"}]}'
    fenced = f"```json\n{body}\n```"
    for text in (body, fenced, f"```\r\n{body}\r\n```"):
        assert JG.json_valid("J-AX1", text, ids={1})
        assert parse_judge_json("J-AX1", text, {1}) is not None
        assert not JG.json_valid("J-AX1", text, ids={2})
        assert parse_judge_json("J-AX1", text, {2}) is None
    for text in (
        f"```json\n{fenced}\n```",
        f"Before\n{fenced}",
        f"{fenced}\nAfter",
        '```json\n{"labels": [{"id": 1, "label": "invalid"}]}\n```',
        '```json\n{"labels": [}\n```',
    ):
        assert not JG.json_valid("J-AX1", text, ids={1})
        assert parse_judge_json("J-AX1", text, {1}) is None


def test_breaker_acknowledged_keys(tmp_path):
    written = []
    buf = G.ObsErrorBuffer(
        written.append,
        limit=2,
        acknowledged={"k1", "k2"},
        pending_log=tmp_path / "p.json",
    )
    for k in ("k1", "k2"):
        buf.add({"key": k, "status": "obs_error", "attempts": [{"error": "refusal"}]})
    assert [r["key"] for r in written] == ["k1", "k2"] and all(
        (r["ack_content_error_after_review"] for r in written)
    )
    buf.add({"key": "k3", "status": "obs_error", "attempts": [{"error": "refusal"}]})
    with pytest.raises(G.SystemAPIError):
        buf.add(
            {"key": "k4", "status": "obs_error", "attempts": [{"error": "refusal"}]}
        )
    assert json.loads((tmp_path / "p.json").read_text())["keys"] == ["k3", "k4"]


def test_shuf_fallback_is_one_route_and_mammo_is_xray():
    ids = ["A", "B", "C", "D"]
    rk = {"A": "RT-FALLBACK", "B": "RT-FALLBACK", "C": "RT-MAMMO", "D": "RT-XR-CHEST"}
    fam = {n: R.modality_family(rk[n], {"modality": "MR"}) for n in ids}
    assert fam["C"] == "X-ray" == fam["D"]
    t = I.shuf_table(ids, rk, fam, {n: "X" for n in ids})
    assert (
        t["donors"]["A"] == "B"
        and t["donors"]["B"] == "A"
        and (t["tiers"]["C"] == "modality")
        and (t["int_missing"] == [])
    )


def test_rp_get_rejects_placeholders():
    from medvqa import execution as RUN

    with pytest.raises(U.LockError):
        RUN.rp_get({"precision": {"Q": "TO_CONFIRM (PREC-0)"}}, "precision", "Q")
    assert RUN.rp_get({"precision": {"Q": "PREC-1"}}, "precision", "Q") == "PREC-1"


SPEC = PROMPTS / "specs" / "native_provenance.json"
CAND = THESIS / "resources/inventory" / "native_heldout_candidates.json"
REF = CODE / "prep" / "ref" / "native300_ref.json"


@pytest.fixture(scope="module")
def nat():
    A = NV.NativeAssigner(SPEC)
    table = R.lookup_rules(R.load_routes_spec(PROMPTS))

    def route(path, sd=None):
        lab, _ = A.label(path, sd)
        if lab["route_reason"] != "ok":
            return "RT-FALLBACK"
        sub_label = lab.get("region", lab.get("acquisition"))
        return table.get(
            ("NATIVE-VISUAL", lab["modality"], sub_label), ("RT-FALLBACK",)
        )[0]

    paths = []
    for pool in json.loads(CAND.read_text(encoding="utf-8"))["pools"].values():
        paths += [(x["source_image_path"], x["source_dataset"]) for x in pool]
    paths += [
        (x["source_image_path"], x["source_dataset"])
        for x in json.loads(REF.read_text(encoding="utf-8"))["native300"]
    ]
    return (A, route, paths)


def test_t_no_match(nat):
    A, route, _ = nat
    for bad in (
        "RadImageNet/knee/knee001.png",
        "/data/omni/RadImageNet/knee001.png",
        "images/MIAS/mdb001.png",
        "Images//x.png",
        "Images/MIAS",
    ):
        with pytest.raises(NV.NativeBuildStop):
            A.label(bad)
    rel = "Images/RadImageNet/acl/knee001.png"
    for variant in (
        "/kaggle/input/omni/OmniMedVQA/" + rel,
        "RQ3\\data\\omni_full\\OmniMedVQA\\" + rel.replace("/", "\\"),
        "D:\\x\\OmniMedVQA\\Images\\RadImageNet\\acl\\knee001.png",
    ):
        assert (
            NV.NativeAssigner.normalize(variant) == rel
            and route(variant) == route(rel) == "RT-MR-KNEE"
        )


def _mutations(path, diag_dirs, rnd):
    """Check 2 (i)-(iv) variants of one normalised native path."""
    parts = path.split("/")
    ds, mids, fname = (parts[1], parts[2:-1], parts[-1])

    def keep(c):
        return ds == "Mura" and _re.fullmatch("XR_[A-Z]+", c)

    def rand():
        return "".join(
            (rnd.choice("abcdefghijklmnopqrstuvwxyz0123456789_") for _ in range(9))
        )

    out = []
    for k, c in enumerate(mids):
        if keep(c):
            continue
        for alt in list(diag_dirs.get(ds, ()))[:25] + [rand()]:
            if not _re.fullmatch("XR_[A-Z]+", alt):
                out.append(
                    "/".join(parts[:2] + mids[:k] + [alt] + mids[k + 1 :] + [fname])
                )
    if ds == "Mura":
        swap = {
            "positive": "negative",
            "negative": "positive",
            "valid": "train",
            "train": "valid",
        }
        m2 = [
            _re.sub("positive|negative|valid|train", lambda m: swap[m.group(0)], c)
            if not keep(c)
            else c
            for c in mids
        ]
        m3 = [
            _re.sub("\\d+", lambda m: str(int(m.group(0)) + 7), c) if not keep(c) else c
            for c in m2
        ]
        out += ["/".join(parts[:2] + m2 + [fname]), "/".join(parts[:2] + m3 + [fname])]
    if ds == "RadImageNet":
        m = _re.match(
            "^(mri-abd-normal|mrabd|([a-z]+)-normal|([a-z]+))(\\d+)(\\.\\w+)$", fname
        )
        if m:
            num, ext = (m.group(4), m.group(5))
            if m.group(1) in ("mrabd", "mri-abd-normal"):
                forms = [f"mrabd{num}{ext}", f"mri-abd-normal{num}{ext}"]
            else:
                tok = m.group(2) or m.group(3)
                forms = [f"{tok}{num}{ext}", f"{tok}-normal{num}{ext}"]
            for f in forms:
                out += [
                    "/".join(parts[:2] + mids + [f]),
                    "/".join(parts[:2] + [rand()] + [f]),
                ]
    else:
        stem_ext = fname.rsplit(".", 1)
        out += [
            "/".join(parts[:2] + mids + [rand() + "." + stem_ext[-1]]),
            "/".join(parts[:2] + mids + ["NORMAL-" + fname]),
        ]
    return out


def test_t_diag_invariance_all_native_paths(nat):
    """Keep native routes unchanged when diagnostic path components change."""
    A, route, paths = nat
    diag_dirs = {}
    for pth, _ in paths:
        parts = NV.NativeAssigner.normalize(pth).split("/")
        diag_dirs.setdefault(parts[1], set()).update(
            (c for c in parts[2:-1] if not _re.fullmatch("XR_[A-Z]+", c))
        )
    diag_dirs = {k: sorted(v) for k, v in diag_dirs.items()}
    rnd = _random.Random(20261016)
    n_checked = 0
    for pth, sd in paths:
        norm = NV.NativeAssigner.normalize(pth)
        base = route(norm, sd)
        for v in _mutations(norm, diag_dirs, rnd):
            assert route(v, sd) == base, (norm, v)
            n_checked += 1
    assert len(paths) == 900 and n_checked > 5000
    assert route("Images/RadImageNet/knee_x/foot001.png") == "RT-MR-MIDFOOT"
    assert route("Images/RadImageNet/XR_KNEE/mri-abd-normal5.png") == "RT-MR-ABD"


def test_t_closed_tokens_and_record_fields(nat):
    A, route, paths = nat
    for bad in (
        "Images/RadImageNet/x/ctabd001.png",
        "Images/RadImageNet/x/Knee001.png",
        "Images/RadImageNet/x/mri-knee001.png",
        "Images/RadImageNet/x/knee.png",
        "Images/Mura/valid/XR_WRIST/XR_HAND/p/s/i.png",
        "Images/Mura/valid/p/s/i.png",
        "Images/Mura/valid/XR_KNEE/p/s/i.png",
    ):
        with pytest.raises(NV.NativeBuildStop):
            A.label(bad)
    lab, _ = A.label("Images/RadImageNet/x/mri123.png")
    assert (
        lab["route_reason"] == "unknown_explicit_token"
        and route("Images/RadImageNet/x/mri123.png") == "RT-FALLBACK"
    )
    assert (
        A.label("Images/Mura/valid/XR_FOREARM/p/s/i.png")[0]["route_reason"] == "other"
    )
    assert A.label("Images/MIAS/mdb001.png", "Mura")[0]["route_reason"] == "conflict"
    for pth, sd in paths:
        lab, _ = A.label(pth, sd)
        rec = dict(lab, neutral_id="NT-0001", ck_chroma_frac=0.0)
        assert NV.validate_record(rec)
    with pytest.raises(NV.NativeBuildStop):
        NV.validate_record(
            {
                "neutral_id": "NT-1",
                "modality": "MR",
                "region": "knee",
                "provenance_rule": "dataset_table",
                "ck_chroma_frac": 0.0,
                "route_reason": "ok",
                "dataset": "RadImageNet",
            }
        )
    spec = json.loads(SPEC.read_text(encoding="utf-8"))["extraction"]
    assert [A.re_dataset.pattern, A.re_mura.pattern, A.re_radimg.pattern] == [
        spec[k]["regex"] for k in ("dataset", "mura_region", "radimagenet_prefix")
    ]


def test_t_unknown_dataset(nat):
    A, route, _ = nat
    with pytest.raises(NV.NativeBuildStop, match="T-UNKNOWN-DATASET"):
        A.label("Images/BrandNewSet/a/b.png")
    assert route("Images/OCT & X-Ray 2017/train/NORMAL/NORMAL-1.jpeg") == "RT-FALLBACK"


def test_ck_color_rules_and_mass_fail():
    from PIL import Image

    gray = Image.new("RGB", (10, 10), (90, 90, 90))
    colour = Image.new("RGB", (10, 10), (200, 20, 20))
    assert (
        NV.NativeAssigner.ck_chroma_frac(gray) == 0.0
        and NV.NativeAssigner.ck_chroma_frac(colour) == 1.0
    )
    base = {
        "modality": "X-ray",
        "region": "chest",
        "provenance_rule": "dataset_table",
        "route_reason": "ok",
    }
    assert NV.NativeAssigner.apply_ck(base, 0.051)["route_reason"] == "ck_color"
    assert NV.NativeAssigner.apply_ck(base, 0.05)["route_reason"] == "ok"
    skin = {
        "modality": "skin",
        "acquisition": "dermoscopy",
        "provenance_rule": "dataset_table",
        "route_reason": "ok",
    }
    assert NV.NativeAssigner.apply_ck(skin, 0.019)["route_reason"] == "ck_gray"
    assert NV.NativeAssigner.apply_ck(skin, 0.02)["route_reason"] == "ok"
    for frac, reason in ((0.0500001, "ok"), (0.0500006, "ck_color")):
        r = NV.NativeAssigner.apply_ck(base, frac)
        assert r["route_reason"] == reason and (
            r["ck_chroma_frac"] > NV.CK_GRAY_MODALITIES_MAX
        ) == (reason == "ck_color")
    for frac, reason in ((0.0199999, "ok"), (0.0199994, "ck_gray")):
        r = NV.NativeAssigner.apply_ck(skin, frac)
        assert r["route_reason"] == reason and (
            r["ck_chroma_frac"] < NV.CK_SKIN_MIN
        ) == (reason == "ck_gray")
    table, bad = NV.mass_fail_check(
        [("A", "ok")] * 4
        + [("A", "ck_color")]
        + [("B", "ok")] * 3
        + [("B", "ck_gray")] * 2
    )
    assert bad == ["B"] and table["A"]["frac"] == 0.2


def test_router_reads_native_record_only():
    rec = {
        "neutral_id": "NT-0001",
        "modality": "MR",
        "region": "knee",
        "provenance_rule": "radimagenet_prefix",
        "ck_chroma_frac": 0.0,
        "route_reason": "ok",
    }
    pairs = R.build_route_pairs({"NT-0001": rec}, R.load_routes_spec(PROMPTS))
    assert R.route_for("NT-0001", rec, pairs)["route_key"] == "RT-MR-KNEE"
    assert (
        R.route_for("NT-0001", dict(rec, route_reason="ck_color"), pairs)["route_key"]
        == "RT-FALLBACK"
    )


def test_cells_exact():
    names = CL.required_cells("MAIN")
    assert (
        sorted(names) == sorted(CL.MAIN_REQUIRED_CELLS)
        and len(names) == 19
        and (len(set(names)) == 19)
    )
    assert sorted((c["cell"] for c in CL.answer_cells("L", design="MAIN"))) == sorted(
        CL.MAIN_REQUIRED_CELLS[:7]
    )
    assert sorted((c["cell"] for c in CL.answer_cells("Q", design="MAIN"))) == sorted(
        CL.MAIN_REQUIRED_CELLS[7:]
    )
    assert CL.answer_cells("M", design="MAIN") == []
    assert not any(
        (
            "INT2" in n or "BLANK0" in n or "-Q-GEN" in n or n.startswith("CELL-M")
            for n in names
        )
    )
    assert CL.DEFAULT_DESIGN == "MAIN"
    assert len(CL.all_cells()) == 36 and len(CL.all_cells(design="FB")) == 23


def test_observers_edits_design_state():
    assert CL.observers_for("MAIN") == ["OBS-L", "OBS-Q", "OBS-G"]
    assert CL.edit_types_for("MAIN") == ["INT-PARA", "INT-CORR"]
    assert CL.design_state("MAIN") == {
        "branch": "MAIN",
        "fb": True,
        "u3": "A",
        "u4": "A",
        "rq2": "AX1_SLAKE",
        "configuration_ref": "MAIN",
    }
    assert CL.design_state("MAIN", "NONE")["rq2"] == "NONE"
    with pytest.raises(ValueError):
        CL.design_state("MAIN", "AX2")


def test_design_state_matches_analysis():
    sys.path.insert(0, str(CODE))
    from medvqa.analysis import contrasts as CT
    from medvqa.analysis import io_adapter as IO

    for rq2 in ("AX1_SLAKE", "NONE"):
        assert IO.parse_design_state(CL.design_state("MAIN", rq2))["design"] == "MAIN"
    assert sorted(CT.required_cells("MAIN")) == sorted(
        (n[len("CELL-") :] for n in CL.required_cells("MAIN"))
    )


def test_platform_lock_and_env(tmp_path):
    from medvqa import execution as RUN

    assert RUN.check_platform("Q", "colab", "PILOT")["lock"] == "colab"
    assert RUN.check_platform("L", "kaggle", "PILOT")["lock"] == "kaggle"
    assert RUN.check_platform("G", "colab", "PILOT") and RUN.check_platform(
        "G", "kaggle", "PILOT"
    )
    for m, wrong in (("Q", "kaggle"), ("L", "colab"), ("J1", "colab")):
        with pytest.raises(U.LockError):
            RUN.check_platform(m, wrong, "PILOT")
    rp = tmp_path / "run_params.json"
    rp.write_text(
        json.dumps(
            {
                "design": "MAIN",
                "platform_lock": {"Q": "colab", "L": "kaggle", "G": "any"},
            }
        ),
        encoding="utf-8",
    )
    assert (
        RUN.check_platform("Q", "colab", "TEST", rp)
        and RUN.check_design("MAIN", "TEST", rp) == "MAIN"
    )
    rp.write_text(
        json.dumps({"design": "MAIN", "platform_lock": {"Q": "kaggle"}}),
        encoding="utf-8",
    )
    with pytest.raises(U.LockError):
        RUN.check_platform("Q", "colab", "TEST", rp)
    rp.write_text(
        json.dumps({"design": "FB", "platform_lock": {"Q": "colab"}}), encoding="utf-8"
    )
    with pytest.raises(U.LockError):
        RUN.check_design("MAIN", "TEST", rp)
    assert (
        RUN.env_for("Q", "colab") == "ENV-COLAB-L4"
        and RUN.env_for("L", "kaggle") == "ENV-1"
    )
    assert (
        RUN.CANDIDATE_PINS["ENV-COLAB-L4"]["transformers"]
        == RUN.CANDIDATE_PINS["ENV-1"]["transformers"]
    )
    assert RUN.PLATFORM_DEFAULTS["colab"]["time_limit_s"] == 3 * 3600
    assert RUN.PLATFORM_DEFAULTS["colab"]["input_root"].startswith(
        "/content/drive/MyDrive/"
    )
    assert RUN.PLATFORM_DEFAULTS["colab"]["out_root"].startswith(
        "/content/drive/MyDrive/"
    )
