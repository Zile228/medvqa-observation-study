"""Snapshot-pinned API payloads, bounded retries and response validation."""

import base64
import datetime as _dt
import hashlib
import re
import time

from medvqa import prompts as P
from medvqa.common import GPT_DEADLINE_UTC

GPT_MODEL = "gpt-4o-mini-2024-07-18"
OBS_MAX_TOKENS, EDIT_MAX_TOKENS = (700, 900)
GPT_SEED = 20261016
EDIT_SEEDS = (20261016, 20261017, 20261018)
ALLOWED_KEYS = {"model", "messages", "temperature", "seed", "max_tokens"}
NEUTRAL_ANY = re.compile("\\b(RT|NT|RD|ND)-\\d{4}\\b")


class DeadlinePassed(RuntimeError):
    pass


def check_deadline(now=None):
    """Reject API calls after the optional, timezone-aware execution cutoff."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    if GPT_DEADLINE_UTC is not None and now > GPT_DEADLINE_UTC:
        raise DeadlinePassed("The configured API execution cutoff has passed.")


def call_order(neutral_ids, tags, salt):
    """Order interleaved calls with a deterministic image-and-tag hash.

    Frozen interleaved order: sha256(neutral_id + '|' + tag + '|' + salt) ascending.
    """
    pairs = [(n, t) for n in neutral_ids for t in tags]
    return sorted(
        pairs,
        key=lambda p: hashlib.sha256(
            f"{p[0]}|{p[1]}|{salt}".encode("utf-8")
        ).hexdigest(),
    )


def obs_payload(prompt, png_bytes, max_tokens=OBS_MAX_TOKENS, detail="low"):
    uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    return {
        "model": GPT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri, "detail": detail}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "seed": GPT_SEED,
        "max_tokens": max_tokens,
    }


def edit_payload(prompt, seed, max_tokens=EDIT_MAX_TOKENS):
    return {
        "model": GPT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
    }


def _texts(payload):
    out = []
    for m in payload["messages"]:
        c = m["content"]
        if isinstance(c, str):
            out.append(c)
        else:
            out += [p["text"] for p in c if p.get("type") == "text"]
    return out


def check11(payload, expect_image):
    """Validate the permitted observation-request fields and content.

    Check 11: exact key set; only user messages with text (+ one PNG data URI); no file
    names/ids in text.
    """
    if set(payload) != ALLOWED_KEYS:
        raise P.LeakError(f"check11: payload keys {sorted(payload)}")
    if payload["model"] != GPT_MODEL or payload["temperature"] != 0:
        raise P.LeakError("check11: model/temperature")
    msgs = payload["messages"]
    if (
        len(msgs) != 1
        or set(msgs[0]) != {"role", "content"}
        or msgs[0]["role"] != "user"
    ):
        raise P.LeakError(
            "check11: messages must be one user message with role+content only"
        )
    c = msgs[0]["content"]
    if expect_image:
        imgs = [p for p in c if p.get("type") == "image_url"]
        if (
            len(imgs) != 1
            or not imgs[0]["image_url"]["url"].startswith("data:image/png;base64,")
            or set(imgs[0]["image_url"]) != {"url", "detail"}
        ):
            raise P.LeakError("check11: exactly one PNG data URI expected")
        if any((set(p) - {"type", "text", "image_url"} for p in c)):
            raise P.LeakError("check11: extra content fields")
    elif not isinstance(c, str):
        raise P.LeakError("check11: editor payload must be text only")
    for t in _texts(payload):
        if NEUTRAL_ANY.search(t):
            raise P.LeakError("check11: neutral_id in payload text")
        P.check3_no_original_names(t)
    return True


def loggable(payload):
    """Replace image bytes with checksum metadata for request logging.

    Payload copy with base64 replaced by sha256/length, for payload_log.jsonl (Validator
    audit).
    """
    import copy

    p = copy.deepcopy(payload)
    for m in p["messages"]:
        if isinstance(m["content"], list):
            for part in m["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    b = url.split(",", 1)[1]
                    part["image_url"]["url"] = (
                        "<png base64 sha256="
                        f"{hashlib.sha256(b.encode()).hexdigest()}"
                        " len="
                        f"{len(b)}"
                        ">"
                    )
    return p


def _valid_obs(resp):
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    if msg.get("refusal"):
        return (False, "refusal")
    if ch.get("finish_reason") == "content_filter":
        return (False, "content_filter")
    if not (msg.get("content") or "").strip():
        return (False, "empty_output")
    return (True, None)


class SystemAPIError(RuntimeError):
    """Signal an authentication, quota, transport or server failure.

    Raised when authentication, quota, transport or server failures stop the session.
    """


CONTENT_STATUS = {400, 413, 422}
BREAKER_CONSECUTIVE = 5


def classify_exception(exc):
    """Distinguish content failures from session-stopping API errors.

    'content' (counts as an attempt, may become a permanent error) or 'system' (circuit
    breaker).
    """
    code = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    text = f"{type(exc).__name__} {exc}"
    if (
        code in (401, 403)
        or "insufficient_quota" in text
        or type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError")
    ):
        return "system"
    if code in CONTENT_STATUS:
        return "content"
    return "system"


def _err(exc):
    return (
        f"{type(exc).__name__}[{getattr(exc, 'status_code', None)}]: {str(exc)[:300]}"
    )


def call_observation(send, payload, attempts=3, wait_s=60, sleep=time.sleep):
    """Generate an observation with bounded content-error attempts.

    A GPT observation is 'done' when (i) a valid output is saved or (ii) 3 content-level
    attempts failed (refusal, content_filter, empty output, 4xx tied to the payload) ->
    permanent obs_error. System errors raise SystemAPIError and nothing is
    recorded. Deadline is checked before every attempt.
    """
    log = []
    for k in range(attempts):
        if k > 0:
            sleep(wait_s)
        check_deadline()
        t0 = time.time()
        try:
            resp = send(payload)
        except Exception as exc:
            if classify_exception(exc) == "system":
                raise SystemAPIError(_err(exc)) from exc
            resp, ok, why = (None, False, "http_content:" + _err(exc))
        else:
            ok, why = _valid_obs(resp)
        log.append(
            {
                "attempt": k + 1,
                "ok": ok,
                "error": why,
                "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "latency_s": round(time.time() - t0, 2),
            }
        )
        if ok:
            return ("ok", resp, log)
    return ("obs_error", None, log)


def call_editor(send, payload, wait_s=60, sleep=time.sleep, api_attempts=3):
    """Run one deterministic editor attempt with bounded content-error retries.

    System failures and the optional API cutoff stop execution without recording a
    completed unit.
    """
    errs = []
    for k in range(api_attempts):
        if k > 0:
            sleep(wait_s)
        check_deadline()
        try:
            resp = send(payload)
        except Exception as exc:
            if classify_exception(exc) == "system":
                raise SystemAPIError(_err(exc)) from exc
            errs.append(_err(exc))
            continue
        return (
            (resp["choices"][0].get("message") or {}).get("content") or "",
            dict(api_meta(resp), http_content_errors=errs),
        )
    return (None, {"http_content_errors": errs})


class ObsErrorBuffer:
    """Buffer content failures until success or normal completion.

        Repeated failures stop the session so a service-wide outage
    cannot silently complete the experiment. Acknowledged payload
    failures may be recorded explicitly.
    """

    def __init__(
        self, write, limit=BREAKER_CONSECUTIVE, acknowledged=(), pending_log=None
    ):
        self.write, self.limit, self.pending = (write, limit, [])
        self.acknowledged, self.pending_log = (set(acknowledged), pending_log)

    @staticmethod
    def _reason(rec):
        e = (rec.get("attempts") or [{}])[-1].get("error") or ""
        return e.split(":")[0]

    def add(self, rec):
        if rec["status"] != "obs_error":
            self.flush()
            self.write(rec)
            return
        if rec.get("key") in self.acknowledged:
            self.write(dict(rec, ack_content_error_after_review=True))
            return
        self.pending.append(rec)
        if (
            len(self.pending) >= self.limit
            and len({self._reason(r) for r in self.pending[-self.limit :]}) == 1
        ):
            keys = [r.get("key") for r in self.pending]
            if self.pending_log is not None:
                import json as _json
                from pathlib import Path as _P

                _P(self.pending_log).write_text(
                    _json.dumps(
                        {
                            "reason": self._reason(self.pending[-1]),
                            "keys": keys,
                            "attempts": {
                                str(r.get("key")): r.get("attempts")
                                for r in self.pending
                            },
                        },
                        ensure_ascii=False,
                        indent=1,
                    ),
                    encoding="utf-8",
                )
            self.pending = []
            raise SystemAPIError(
                (
                    f"{len(keys)}"
                    " consecutive obs_error with the same reason: "
                    "circuit breaker, nothing recorded. Review "
                    "breaker_pending_keys.json; only if they are "
                    "genuine content errors, put the keys in "
                    "ACK_CONTENT_ERRORS and resume."
                )
            )

    def flush(self):
        for r in self.pending:
            self.write(r)
        self.pending = []


def api_meta(resp):
    return (
        {
            k: resp.get(k)
            for k in ("id", "system_fingerprint", "model", "usage", "created")
        }
        if resp
        else None
    )


def openai_sender(api_key, max_retries=5, timeout=120):
    """Real sender (used only on Kaggle/Colab).

    Imports openai lazily; transport-level retries (429/5xx) inside one attempt.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, max_retries=max_retries, timeout=timeout)

    def send(payload):
        return client.chat.completions.create(**payload).model_dump()

    return send


class StubHTTPError(Exception):
    def __init__(self, status_code, msg="stub"):
        super().__init__(f"{status_code} {msg}")
        self.status_code = status_code


def stub_sender(fail_first=0, fail_status=503, empty_ids=()):
    """Imitate API responses for offline observation and editing tests.

    CPU dry-run only: imitates Chat Completions responses (observation lines / unchanged
    paraphrase / valid CORR JSON). fail_first: number of initial calls raising
    StubHTTPError(fail_status); empty_ids: substrings of prompts answered with an empty
    output (content-level failure).
    """
    import json as _json
    import re as _re

    from medvqa.interventions import LAT_PAIRS, SEV_PAIRS

    state = {"n": 0}

    def send(payload):
        state["n"] += 1
        if state["n"] <= fail_first:
            raise StubHTTPError(fail_status)
        c = payload["messages"][0]["content"]
        text = (
            c
            if isinstance(c, str)
            else next((p["text"] for p in c if p["type"] == "text"))
        )
        if any((e in text for e in empty_ids)):
            return {
                "id": f"stub-{state['n']}",
                "system_fingerprint": "fp_stub",
                "model": payload["model"],
                "created": 0,
                "usage": {"completion_tokens": 0},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "", "refusal": None},
                    }
                ],
            }
        if "Report each item below" in text:
            items = _re.findall(
                "^(\\d+)\\. ([^:\\n]+):", text.split("Items:")[1], _re.M
            )
            out = "\n".join(
                (
                    f"{n}. {name} | visible | small rounded area on the image-left side"
                    for n, name in items
                )
            )
        elif text.startswith("Rewrite the text below"):
            out = text.split("Text:\n", 1)[1]
        else:
            obs = text.split("Text:\n", 1)[1]
            ln = int(_re.search("Line to edit: line (\\d+)", text).group(1))
            lines = obs.split("\n")
            old = lines[ln - 1]
            new, o_sp, n_sp = (old, None, None)
            for a, b in LAT_PAIRS + SEV_PAIRS:
                m = _re.search("(?<![\\w-])" + a + "(?![\\w-])", old)
                if m:
                    new = old[: m.start()] + b + old[m.end() :]
                    o_sp, n_sp = (a, b)
                    break
            if o_sp is None:
                o_sp = old.split("|")[-1].strip()
                n_sp = "no " + o_sp
                new = old[: old.rfind("|") + 1] + " " + n_sp
            lines[ln - 1] = new
            out = _json.dumps(
                {
                    "edited_text": "\n".join(lines),
                    "original_span": o_sp,
                    "new_span": n_sp,
                }
            )
        return {
            "id": f"stub-{state['n']}",
            "system_fingerprint": "fp_stub",
            "model": payload["model"],
            "created": 0,
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": len(out.split()),
                "total_tokens": 1,
            },
            "choices": [
                {"finish_reason": "stop", "message": {"content": out, "refusal": None}}
            ],
        }

    return send
