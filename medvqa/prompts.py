"""Build observation and answer prompts and check for information leakage."""

import hashlib
import inspect
import json
import re
import unicodedata
from pathlib import Path

ARMS = ("ARM-GEN", "ARM-ANA", "ARM-GRD")
ARM_FILE = {"ARM-GEN": "GEN", "ARM-ANA": "ANA", "ARM-GRD": "GRD"}
FALLBACK = "RT-FALLBACK"
FORMAT_LINES = {
    "yesno": "Answer with one word: Yes or No.",
    "mcq": "Answer with the letter of one option only.",
    "closed_short": "Answer with a single word or short phrase only.",
    "open": "Answer with a single word or short phrase only.",
}
YESNO_RE = re.compile(
    (
        "^(is|are|does|do|did|was|were|can|could|has|have"
        "|had|will|would|should|may|might)\\b"
    ),
    re.I,
)
OR_RE = re.compile("\\bor\\b", re.I)
LABELS = "ABCD"
ITEM_NO_RE = re.compile("^\\s*\\d+\\.\\s*")


def sha256_hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def answer_kind(answer_type_official, question, native=False):
    if native:
        return "mcq"
    t = str(answer_type_official).strip().upper()
    if t == "OPEN":
        return "open"
    if t != "CLOSED":
        raise ValueError(
            f"answer_type must be CLOSED/OPEN, got {answer_type_official!r}"
        )
    q = str(question).strip()
    return "yesno" if YESNO_RE.match(q) and (not OR_RE.search(q)) else "closed_short"


def cyclic_order(options, s):
    """options: list of option texts in original order.

    Order s = left shift opts[s:]+opts[:s]; letters keep position.
    """
    texts = list(options)
    shifted = texts[s:] + texts[:s]
    return [{"label": LABELS[i], "text": t} for i, t in enumerate(shifted)]


class AnswerTemplate:
    def __init__(self, path):
        lines = [ln for ln in read_text(path).split("\n") if not ln.startswith("#")]
        self.text = "\n".join(lines).strip("\n")
        for ph in ("{OBS_TEXT}", "{QUESTION}", "{OPTIONS_BLOCK}", "{FORMAT_LINE}"):
            if ph not in self.text:
                raise ValueError(f"answerer template missing {ph}")

    def render(self, obs_text, question, kind, options_presented=None):
        if kind not in FORMAT_LINES:
            raise ValueError(kind)
        if kind == "mcq":
            if not options_presented:
                raise ValueError("mcq needs options")
            block = "Options:\n" + "\n".join(
                (f"{o['label']}. {o['text']}" for o in options_presented)
            )
        else:
            if options_presented:
                raise ValueError("options only for mcq")
            block = None
        out = (
            self.text.replace("{OBS_TEXT}", obs_text)
            .replace("{QUESTION}", question)
            .replace("{FORMAT_LINE}", FORMAT_LINES[kind])
        )
        if block is None:
            out = out.replace("{OPTIONS_BLOCK}\n", "").replace("\n{OPTIONS_BLOCK}", "")
        else:
            out = out.replace("{OPTIONS_BLOCK}", block)
        return out

    def skeleton(self, kind, n_options=0):
        """Prompt with dynamic parts replaced by fixed tokens; used by check 6."""
        opts = (
            [{"label": LABELS[i], "text": "<OPT>"} for i in range(n_options)]
            if kind == "mcq"
            else None
        )
        return self.render("<OBS>", "<Q>", kind, opts)


class ObserverPromptBank:
    """Load fixed prompts and compose multi-region observations.

    Serves the exact locked text of generated/OBS-<ARM>__<route>.txt for single routes,
    and composes multi-region prompts with the same renderer as build_prompts.py
    (verified against every locked file at load).
    """

    def __init__(self, prompts_dir):
        d = Path(prompts_dir)
        self.dir = d
        self.frame = read_text(d / "templates" / "observer_frame.txt").rstrip("\n")
        self.routes = json.loads(read_text(d / "specs" / "routes.json"))["routes"]
        self.grd = json.loads(read_text(d / "specs" / "grounded_spec.json"))["routes"]
        self.ana = json.loads(read_text(d / "specs" / "author_anatomy_spec.json"))[
            "routes"
        ]
        self.gen = json.loads(read_text(d / "templates" / "generic_items.json"))
        self.files = {}
        for p in sorted((d / "generated").glob("*.txt")):
            self.files[p.name] = read_text(p)
        self._selfcheck()

    @staticmethod
    def _render(frame, context_line, items):
        block = "\n".join(
            (f"{i}. {it['name']}: {it['ask']}" for i, it in enumerate(items, 1))
        )
        return frame.replace("{CONTEXT_LINE}", context_line).replace("{ITEMS}", block)

    def _items(self, arm, route):
        if route == FALLBACK or arm == "ARM-GEN":
            return self.gen["items"]
        return (self.grd if arm == "ARM-GRD" else self.ana)[route]

    def _selfcheck(self):
        for rid in self.routes:
            for arm in ARMS:
                want = self.files[f"OBS-{ARM_FILE[arm]}__{rid}.txt"].rstrip("\n")
                if self.compose(arm, [rid], _check=False) != want:
                    raise RuntimeError(
                        f"Composition differs from locked file for {arm} {rid}"
                    )
            if self.compose_empty([rid], _check=False) != self.files[
                f"OBS-EMPTY__{rid}.txt"
            ].rstrip("\n"):
                raise RuntimeError(f"EMPTY composition differs for {rid}")

    def compose(self, arm, routes, _check=True):
        if arm not in ARMS:
            raise ValueError(arm)
        routes = list(routes)
        if not routes or FALLBACK in routes:
            if routes and routes != [FALLBACK]:
                raise ValueError("FALLBACK cannot be combined")
            return self._render(self.frame, self.gen["context_line"], self.gen["items"])
        ctx = (
            "Image context: "
            + " + ".join((self.routes[r]["header"] for r in routes))
            + "."
        )
        if arm == "ARM-GEN":
            items = self.gen["items"]
        else:
            items = [it for r in routes for it in self._items(arm, r)]
        return self._render(self.frame, ctx, items)

    def compose_empty(self, routes, _check=True):
        routes = list(routes) or [FALLBACK]
        items = (
            self.gen["items"]
            if routes == [FALLBACK]
            else [it for r in routes for it in self.grd[r]]
        )
        return "\n".join(
            (f"{i}. {it['name']} | — | —" for i, it in enumerate(items, 1))
        )

    def prompt(self, arm, routes):
        """Returns (prompt_text, prompt_source)."""
        routes = list(routes) or [FALLBACK]
        text = self.compose(arm, routes)
        if len(routes) == 1:
            fname = f"OBS-{ARM_FILE[arm]}__{routes[0]}.txt"
            assert text == self.files[fname].rstrip("\n")
            return (text, f"generated/{fname}")
        return (text, "composed:" + "+".join(routes))

    def grd_item_list(self, routes):
        """Render the requested grounded-observation items.

        Requested GRD item list for J-AX2 (same rendering as the observer items block).
        """
        routes = list(routes) or [FALLBACK]
        items = (
            self.gen["items"]
            if routes == [FALLBACK]
            else [it for r in routes for it in self.grd[r]]
        )
        return "\n".join(
            (f"{i}. {it['name']}: {it['ask']}" for i, it in enumerate(items, 1))
        )

    def locked_lines(self):
        """Return normalized fixed-template lines without item numbers.

        Return normalized lines of the fixed observer templates, without item numbers.
        """
        out = set()
        for txt in [self.frame, *self.files.values()]:
            for ln in txt.splitlines():
                n = norm_text(ITEM_NO_RE.sub("", ln))
                if n:
                    out.add(n)
        return out

    def allowed_texts(self, route_keys):
        """Enumerate permitted observer prompts for the selected routes.

        Every prompt text that may legitimately be sent to an observer for the given
        route combinations.
        """
        out = set()
        for rk in route_keys:
            for arm in ARMS:
                out.add(self.compose(arm, rk.split("+")))
        return out


def norm_text(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub("[^\\w\\s]", " ", s)
    return re.sub("\\s+", " ", s).strip()


class LeakError(RuntimeError):
    pass


MIN_FAIL_LEN = 12


def check1_observer_prompt(
    prompt, allowed_texts, questions=(), options=(), answers=(), locked_lines=None
):
    """Check that the observer prompt cannot expose question, option or answer text.

    Overlaps fully contained within one fixed template line are counted separately;
    injected content is rejected.
    """
    if prompt not in allowed_texts:
        raise LeakError("check1a: observer prompt is not a locked template text")
    p = " " + norm_text(prompt) + " "
    lines = [
        n
        for n in (norm_text(ITEM_NO_RE.sub("", ln)) for ln in prompt.splitlines())
        if n
    ]
    short_hits = 0
    collisions = {"question": 0, "option": 0, "answer": 0}
    collision_lines = set()

    def template_line(ns):
        if locked_lines is None:
            return None
        return next(
            (
                ln
                for ln in lines
                if ln in locked_lines and " " + ns + " " in " " + ln + " "
            ),
            None,
        )

    for kind, seq in (
        ("question", questions),
        ("option", options),
        ("answer", answers),
    ):
        for s in seq:
            ns = norm_text(s)
            if not ns or " " + ns + " " not in p:
                continue
            if kind != "question" and len(ns) < MIN_FAIL_LEN:
                short_hits += 1
                continue
            ln = template_line(ns)
            if ln is None:
                raise LeakError(
                    f"check1b: a cohort {kind} string occurs in an observer prompt"
                )
            collisions[kind] += 1
            collision_lines.add(sha256_hex(ln))
    return {
        "short_overlaps": short_hits,
        "template_collisions": collisions,
        "template_collision_lines_sha256": sorted(collision_lines),
    }


ORIGINAL_NAME_RE = re.compile(
    (
        "(synpic\\d+|xmlab\\d+|\\.jpe?g\\b|\\.png\\b|\\.bmp\\b|\\."
        "tiff?\\b|Images/|[A-Za-z]:\\\\|/kaggle/|/content/)"
    ),
    re.I,
)


def check3_no_original_names(prompt, original_names=()):
    """Reject original filenames or folder names in prompts.

    Check 3: no original file/folder name in any prompt (generic patterns + explicit
    names when available).
    """
    m = ORIGINAL_NAME_RE.search(prompt)
    if m:
        raise LeakError(f"check3: original file/path pattern in prompt: {m.group(0)!r}")
    low = prompt.casefold()
    for n in original_names:
        n = str(n).strip()
        if len(n) >= 6 and n.casefold() in low:
            raise LeakError("check3: original image name in prompt")
    return True


MODALITY_LINE_RE = re.compile("(?im)^\\s*(benchmark[- ]declared\\s+)?modality\\s*:")


def check6_answer_prompt(template, prompt, kind, n_options=0):
    """Check the answer prompt against its fixed template.

    Check 6: the answer prompt has no benchmark modality line; its skeleton is exactly
    the locked template.
    """
    sk = template.skeleton(kind, n_options)
    if MODALITY_LINE_RE.search(sk):
        raise LeakError("check6: modality line in answer template")
    lines_sk = [
        ln
        for ln in sk.split("\n")
        if "<OBS>" not in ln and "<Q>" not in ln and ("<OPT>" not in ln)
    ]
    for ln in lines_sk:
        if ln not in prompt:
            raise LeakError("check6: answer prompt does not follow the locked template")
    if re.search("(?im)^\\s*benchmark[- ]declared", prompt):
        raise LeakError("check6: benchmark-declared line in answer prompt")
    return True


INTERVENTION_ALLOWED_PARAMS = {
    "observations",
    "observation",
    "neutral_ids",
    "route_keys",
    "modality_family",
    "obs_text",
    "plan",
    "edit_type",
    "item_no",
    "line_no",
    "image",
    "routes",
    "bank",
    "cohort_of",
    "output",
    "int_type",
    "self",
}
FORBIDDEN_PARAM_WORDS = (
    "question",
    "option",
    "answer",
    "gold",
    "label_gold",
    "qid",
    "id_map",
)


def check5_intervention_signature(funcs):
    """Reject intervention functions with forbidden parameters.

    Check 5: intervention generators only receive observations (+ ids/routes), never
    question/options/answers.
    """
    for f in funcs:
        for name in inspect.signature(f).parameters:
            low = name.lower()
            if (
                any((w in low for w in FORBIDDEN_PARAM_WORDS))
                or name not in INTERVENTION_ALLOWED_PARAMS
            ):
                raise LeakError(
                    f"check5: {f.__name__} has forbidden/unknown parameter {name!r}"
                )
    return True
