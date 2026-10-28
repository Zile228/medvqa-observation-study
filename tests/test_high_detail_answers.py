"""CPU-only endpoint, prompt, identity and resume tests for the two answerers."""

import copy
import json

import pytest
from PIL import Image

from medvqa import high_detail_answers as answers


@pytest.fixture
def inputs(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    picture = tmp_path / "picture.png"
    Image.new("RGB", (2, 2), "white").save(picture)
    image_hash = answers.U.sha256_file(picture)
    observations, questions = [], []
    image_ids = (
        [f"RAD{i}" for i in range(242)]
        + [f"NAT{i}" for i in range(300)]
        + [f"UNUSED{i}" for i in range(57)]
    )
    for image_id in image_ids:
        (image_dir / f"{image_id}.png").write_bytes(picture.read_bytes())
        text = "  1. Synthetic | visible | synthetic finding\n"
        observations.append(
            {
                "neutral_id": image_id,
                "status": "ok",
                "text": text,
                "text_sha256": answers.U.sha256_text(text),
                "image_sha256": image_hash,
            }
        )
    for i in range(688):
        questions.append(
            {
                "qid": f"RAD-Q{i}",
                "neutral_id": f"RAD{i % 242}",
                "cohort": "COH-RAD-TEST",
                "question": "Is a synthetic object visible?",
                "answer_kind": "yesno" if i < 603 else "closed_short",
                "K": 1,
                "orders": [0],
                "options": [],
            }
        )
    for i in range(300):
        count = 4 if i < 236 else 3
        questions.append(
            {
                "qid": f"NAT-Q{i}",
                "neutral_id": f"NAT{i}",
                "cohort": "COH-NAT-TEST",
                "question": "Choose a synthetic object.",
                "answer_kind": "mcq",
                "K": count,
                "orders": list(range(count)),
                "options": [{"text": f"Synthetic option {j}"} for j in range(count)],
            }
        )
    text = "{OBS_TEXT}\r\n{QUESTION}\r\n{OPTIONS_BLOCK}\r\n{FORMAT_LINE}\r\n"
    baselines = {}
    for model in ("Q", "L"):
        identity = dict(answers.B.MODELS[model])
        identity = {
            "model": identity["id"],
            "revision": identity["revision"],
            "resolved_revision": identity["revision"],
            "precision": "PREC-0",
            "chat_template_sha256": "synthetic",
            "logits_method": "output_logits",
            "generate_kwargs": answers.B.GREEDY_KW,
        }
        baselines[model] = {
            "identity": identity,
            "label_map": {
                label: [i] for i, label in enumerate(["Yes", "No", "A", "B", "C", "D"])
            },
            "package_pins": {},
        }
    data = {
        "schema": "HIGH-DETAIL-ANSWER-INPUTS-v1",
        "questions": questions,
        "observations": observations,
        "answer_template": text,
        "answer_template_sha256": answers.U.sha256_text(text),
        "answer_cap": answers.ANSWER_CAP,
        "generate_kwargs": answers.B.GREEDY_KW,
        "baselines": baselines,
        "observation_artifact_sha256": "synthetic",
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return data, path, image_dir


class FakeBackend:
    def __init__(self, data, model):
        self.identity = copy.deepcopy(data["baselines"][model]["identity"])
        self.prompts = []

    def decode_one(self, index):
        return ["Yes", "No", "A", "B", "C", "D"][index]

    def vocab_size(self):
        return 6

    def generate(self, image, prompt, cap):
        self.prompts.append(prompt)
        return {
            "text": "Yes" if "one word: Yes or No" in prompt else "A",
            "first_logits": [],
            "logits_method": "output_logits",
            "first_step_finite": True,
            "truncated": False,
            "n_output_tokens": 1,
            "finish_reason": "stop",
        }


def fake_score(logits, label_map, labels):
    return {
        "finite": True,
        "labels": labels,
        "p_tilde": {label: 1 / len(labels) for label in labels},
    }


@pytest.mark.parametrize("model", ["Q", "L"])
def test_full_endpoint_resume_and_prompt_parity(inputs, tmp_path, model):
    data, path, images = inputs
    backend = FakeBackend(data, model)
    kwargs = dict(
        backend_factory=lambda *args: backend,
        environment_checker=lambda *args: {"synthetic": True},
        scorer=fake_score,
        progress=lambda text: None,
    )
    output = tmp_path / "out"
    partial = answers.run(path, images, output, model, max_items=3, **kwargs)
    assert partial["remaining_in_shard"] == 1821
    result = answers.run(path, images, output, model, **kwargs)
    assert result["remaining_in_shard"] == 0
    assert len(backend.prompts) == 1824
    count = len(backend.prompts)
    answers.run(path, images, output, model, **kwargs)
    assert len(backend.prompts) == count
    run_dir = output / f"high_detail_answers_{model}_20261026"
    rows = [
        row
        for p in run_dir.glob("shard_*/records.jsonl")
        for row in answers.U.read_jsonl(p)
    ]
    assert len(rows) == len({r["key"] for r in rows}) == 1824
    assert answers.U.read_json(run_dir / "execution.json")["synthetic_backend"]
    assert rows[0]["obs_text_sha256"] == answers.U.sha256_text(
        data["observations"][0]["text"].strip()
    )
    template = answers.P.AnswerTemplate(run_dir / "artifacts/answerer_template.txt")
    expected = template.render(
        data["observations"][0]["text"].strip(),
        data["questions"][0]["question"],
        "yesno",
    )
    assert backend.prompts[0] == expected
    assert (
        answers.U.sha256_file(run_dir / "artifacts/answerer_template.txt")
        == data["answer_template_sha256"]
    )


def test_missing_image_rejected_before_backend(inputs, tmp_path):
    data, path, images = inputs
    (images / "RAD0.png").unlink()

    def forbidden(*args):
        raise AssertionError("Backend must remain unopened.")

    with pytest.raises(FileNotFoundError):
        answers.run(path, images, tmp_path / "out", "Q", backend_factory=forbidden)


@pytest.mark.parametrize("change", ["endpoint", "observation", "settings", "template"])
def test_frozen_inputs_fail_closed(inputs, change):
    data, path, images = inputs
    if change == "endpoint":
        data["questions"].pop()
    elif change == "observation":
        data["observations"][0]["text"] = "changed"
    elif change == "settings":
        data["generate_kwargs"] = {"do_sample": True}
    else:
        data["answer_template"] = "changed"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(answers.U.LockError):
        answers.preflight(path, images, "Q")


def test_model_identity_is_checked_before_generation(inputs, tmp_path):
    data, path, images = inputs
    backend = FakeBackend(data, "Q")
    backend.identity["revision"] = "different"
    with pytest.raises(answers.U.LockError, match="revision"):
        answers.run(
            path,
            images,
            tmp_path / "out",
            "Q",
            backend_factory=lambda *args: backend,
            environment_checker=lambda *args: {},
        )
    assert not backend.prompts
