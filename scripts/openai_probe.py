#!/usr/bin/env python3
"""openai_probe.py — answer the questions the OpenAI docs do not answer (§14.24).

WHY THIS EXISTS
---------------
Item 3 of the hosted-backend work builds a capability table that decides which
request fields are sent to which model. Most of that table is documented:
`temperature` and `top_p` are rejected while reasoning is on, `top_k`/`min_p` are
not Chat Completions parameters at all, and `max_tokens` is the legacy spelling
that reasoning models refuse.

Two fields are NOT documented anywhere we could find:

  * `seed`  — the reference lists it only as "used to make sampling more
    reproducible WHEN THE PROVIDER SUPPORTS IT", which is a statement about
    providers in general, not about these models.
  * `presence_penalty` — no model page mentions it either way.

Guessing is the one thing this must not do. Sending a field that is rejected
turns every call in a paid run into a 400; dropping a field that works silently
loses the only reproducibility control the pipeline has (§14.23). So we ask the
API directly, once, for a few hundred tokens, before the capability table is
written.

This script is deliberately standalone: stdlib only, no repo imports, no config,
no `Vanta` code, nothing that boots a VM. Run it from anywhere.

USAGE (the key is never echoed, never logged, never written to disk)
-------------------------------------------------------------------
    read -s OPENAI_API_KEY
    export OPENAI_API_KEY
    python3 scripts/openai_probe.py

    # one model only, or a different gateway:
    python3 scripts/openai_probe.py --models gpt-6.1-sol
    python3 scripts/openai_probe.py --base-url https://api.openai.com/v1

    # see exactly what would be sent, calling nothing:
    python3 scripts/openai_probe.py --dry-run

OUTPUT
------
One line per probe (`OK` / `REJECTED` / `ERROR`) naming the field, the HTTP
status, and — when the API refuses — the provider's own `error.code` and
`error.param`. Then a verdict table per model that is meant to be pasted into
the capability table in `src/llm/`, and a usage block that shows whether
`cached_tokens` and `reasoning_tokens` actually arrive.

Cost: ~8-11 calls per model, each capped at 64 completion tokens on a one-line
prompt. Most rows leave `reasoning_effort` unset so they exercise the provider's
own default — the state a default-configured Vanta run is in.

SECURITY
--------
The key is read from `OPENAI_API_KEY` only — never a `VANTA_` variable — and is
passed straight into the `Authorization` header. Every line this script prints
goes through `redact()` first, and the request headers are never printed at all,
so a gateway that echoes the header back in an error body cannot leak it. There
is a built-in self-check that fails loudly if the key ever appears in the output
we are about to print.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# The two models this backend exists for. Kept here rather than imported so the
# probe keeps working on a checkout that has not landed the backend yet.
MODELS = ("gpt-6-luna", "gpt-6.1-sol")

# The general probes send NO `reasoning_effort` at all, because that is what
# item 4 does by default: the provider's own default (medium on both models) is
# the state whose field rejections the capability table has to describe. Only the
# Luna-only comparison rows pin an effort, and they pin `none` — Luna is the one
# model that accepts it, so it is the only place that can answer "does switching
# reasoning OFF re-enable temperature, top_p and seed?". Sol cannot answer it:
# `none` and `minimal` are rejected outright.
REASONING_OFF = "none"

PROMPT = "Reply with the single word: ok"
MAX_COMPLETION_TOKENS = 64

_KEY = ""
_REDACTION_CACHE: list[str] = []


def redact(text: str) -> str:
    """Replace anything key-shaped before it can reach a terminal.

    Two rules, both belt-and-braces: the actual key we hold, and any `sk-`
    prefixed run, which is what the provider's own secret format looks like.
    """
    out = text
    if _KEY:
        out = out.replace(_KEY, "<redacted>")
    for token in _REDACTION_CACHE:
        out = out.replace(token, "<redacted>")
    # A key that arrives inside an error body is not the key we hold; scrub the
    # documented shape too. `sk-proj-...` and `sk-...` both start this way.
    parts = []
    for word in out.split():
        if word.startswith("sk-") and len(word) > 12:
            parts.append("<redacted>")
        else:
            parts.append(word)
    return " ".join(parts)


def say(line: str = "") -> None:
    text = redact(line)
    if _KEY and _KEY in text:  # pragma: no cover - a guard, not a path
        raise SystemExit("REFUSING TO PRINT: the API key survived redaction.")
    print(text, flush=True)


def build_body(model: str, extra: dict, *, effort: str | None, stream: bool) -> dict:
    """The base request plus one candidate field.

    The base is the smallest body that can succeed: the model, one user turn, and
    `max_completion_tokens` rather than the legacy `max_tokens`, so a failure we
    see is attributable to `extra` and not to the base.
    """
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    if effort is not None:
        body["reasoning_effort"] = effort
    body.update(extra)
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def post(base_url: str, body: dict) -> tuple[int, dict | None, str]:
    """One non-streaming or streaming POST. Returns (status, parsed, raw_text)."""
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(body).encode("utf-8")
    # Headers are constructed here and never printed: an error path that echoed
    # them would be the leak this item forbids.
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        status = exc.code
    except Exception as exc:  # network, DNS, timeout
        return 0, None, f"{type(exc).__name__}: {exc}"

    if body.get("stream"):
        return status, parse_sse_usage(raw), raw
    try:
        return status, json.loads(raw), raw
    except json.JSONDecodeError:
        return status, None, raw


def parse_sse_usage(raw: str) -> dict:
    """Pull the usage block out of a streamed response.

    `stream_options.include_usage` puts the whole request's usage in one final
    chunk before `[DONE]`; every earlier chunk carries `usage: null`. This is also
    the cheapest proof that the field is honoured on these models at all.
    """
    usage: dict = {}
    saw_done = False
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk, dict) and chunk.get("usage"):
            usage = chunk["usage"]
    if usage:
        usage["_saw_done"] = saw_done
    return usage


def usage_summary(usage: dict) -> str:
    """Flatten the usage block into the four numbers item 5 has to parse."""
    if not usage:
        return "usage=absent"
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return (
        f"prompt={usage.get('prompt_tokens')} "
        f"completion={usage.get('completion_tokens')} "
        f"cached={prompt_details.get('cached_tokens')} "
        f"reasoning={completion_details.get('reasoning_tokens')} "
        f"done={usage.get('_saw_done')}"
    )


def error_summary(parsed: dict | None) -> str:
    """The provider's own words, which is what fills the capability table."""
    if not isinstance(parsed, dict):
        return ""
    err = parsed.get("error")
    if not isinstance(err, dict):
        return ""
    bits = []
    if err.get("code"):
        bits.append(f"code={err['code']}")
    if err.get("param"):
        bits.append(f"param={err['param']}")
    if err.get("type"):
        bits.append(f"type={err['type']}")
    message = (err.get("message") or "").strip().replace("\n", " ")
    if message:
        bits.append(f'"{message[:160]}"')
    return " ".join(bits)


def finish_summary(parsed: dict | None) -> str:
    if not isinstance(parsed, dict):
        return ""
    choices = parsed.get("choices") or []
    if not choices:
        return ""
    return f"finish={choices[0].get('finish_reason')}"


# ---------------------------------------------------------------------------
# the probes
# ---------------------------------------------------------------------------
def probes_for(model: str) -> list[tuple[str, dict, str | None, bool]]:
    """(label, extra fields, effort, stream) — one row per question we are asking.

    `effort=None` means the key is omitted and the provider default applies. That
    is deliberate: it mirrors the body item 4 sends when `VANTA_REASONING_EFFORT`
    is unset, so a rejection recorded here is a rejection the real run would hit.
    """
    rows: list[tuple[str, dict, str | None, bool]] = [
        ("control (no extra fields)", {}, None, False),
        ("seed", {"seed": 1234}, None, False),
        ("presence_penalty", {"presence_penalty": 0.0}, None, False),
        ("temperature (reasoning on)", {"temperature": 0.6}, None, False),
        ("top_p (reasoning on)", {"top_p": 0.9}, None, False),
        ("max_tokens (legacy spelling)", {"max_tokens": 4096}, None, False),
        (
            "stream + stream_options.include_usage + prompt_cache_key",
            {"prompt_cache_key": "vanta-probe"},
            None,
            True,
        ),
        ("service_tier=flex", {"service_tier": "flex"}, None, False),
    ]
    if model == "gpt-6-luna":
        # Only Luna has a reasoning-off state, so only Luna can say whether these
        # three fields come back when reasoning is switched off. If they do, the
        # capability table needs an effort-dependent column rather than a flat
        # "unsupported" — which is exactly the kind of thing worth knowing before
        # an arm is run under a non-default effort.
        rows.append(("temperature (reasoning off)", {"temperature": 0.6}, REASONING_OFF, False))
        rows.append(("top_p (reasoning off)", {"top_p": 0.9}, REASONING_OFF, False))
        rows.append(("seed (reasoning off)", {"seed": 1234}, REASONING_OFF, False))
    return rows


def probe_model(base_url: str, model: str, *, dry_run: bool) -> dict[str, str]:
    """Run every probe for one model; return {field: verdict} for the table."""
    say(f"\n=== {model} ===")
    verdicts: dict[str, str] = {}
    for label, extra, effort, stream in probes_for(model):
        body = build_body(model, extra, effort=effort, stream=stream)
        if dry_run:
            say(f"  [dry-run ] {label}\n            {redact(json.dumps(body))}")
            continue

        status, parsed, raw = post(base_url, body)
        if status == 200:
            # A streamed call returns its usage block from the SSE tail instead of
            # a body, so both paths arrive here as a dict.
            note = usage_summary(parsed or {})
            finish = finish_summary(parsed)
            say(f"  [OK      ] {label}  status=200 {finish} {note}".rstrip())
            verdicts[label] = "accepted"
        elif status == 0:
            say(f"  [ERROR   ] {label}  transport: {raw[:200]}")
            verdicts[label] = "transport-error"
        else:
            detail = error_summary(parsed) or raw.strip()[:200]
            tag = "RESOURCE" if "resource" in detail.lower() else "REJECTED"
            say(f"  [{tag:<9}] {label}  status={status} {detail}".rstrip())
            if tag == "RESOURCE":
                # Flex capacity, not a field rejection: the request was valid and
                # was not billed. Retryable, so it must not reach the table as a
                # "rejected" verdict.
                verdicts[label] = "retryable (not billed)"
            else:
                verdicts[label] = f"error {status}"
    return verdicts


def field_of(label: str) -> str:
    """`seed` from `seed (reasoning OFF)`; keeps the table one row per field."""
    return label.split(" (")[0].split(" + ")[0].strip()


def main() -> int:
    global _KEY
    ap = argparse.ArgumentParser(
        description="Probe which request fields gpt-6-luna / gpt-6.1-sol accept."
    )
    ap.add_argument("--base-url", default=os.environ.get("VANTA_LLM_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--dry-run", action="store_true", help="print bodies, call nothing")
    args = ap.parse_args()

    _KEY = os.environ.get("OPENAI_API_KEY", "")
    if not args.dry_run:
        if not _KEY:
            print(
                "OPENAI_API_KEY is not set. Load it without echoing it:\n"
                "    read -s OPENAI_API_KEY; export OPENAI_API_KEY\n",
                file=sys.stderr,
            )
            return 2
        # A cheap sanity check that we were handed a key and not, say, a pasted
        # shell line. The length is reported; the value never is.
        say(f"key loaded: yes (length {len(_KEY)}), base_url={args.base_url}")
    else:
        say(f"dry run: base_url={args.base_url} (no key needed, nothing sent)")

    tables: dict[str, dict[str, str]] = {}
    for model in args.models:
        tables[model] = probe_model(args.base_url, model, dry_run=args.dry_run)

    if args.dry_run:
        return 0

    say("\n" + "=" * 74)
    say("VERDICT TABLE — paste into the capability table (src/llm/)")
    say("=" * 74)
    fields: list[str] = []
    for model in args.models:
        for label in tables[model]:
            key = field_of(label)
            if key not in fields:
                fields.append(key)

    for model in args.models:
        say(f"\n{model}")
        by_field: dict[str, list[str]] = {}
        for label, verdict in tables[model].items():
            by_field.setdefault(field_of(label), []).append(verdict)
        for field in fields:
            vals = by_field.get(field)
            if not vals:
                say(f"  {field:<45} — not probed")
                continue
            say(f"  {field:<45} {vals[0]}")

    say("\nNotes for the capability table:")
    say("  * `accepted` means the field was sent and the call succeeded — for `seed`")
    say("    that is still NOT proof the value changed the sample. Reproducibility")
    say("    needs two identical calls compared by hand.")
    say("  * `retryable (not billed)` is flex capacity pressure, not a field")
    say("    rejection; item 6's backoff handles it.")
    say("  * a field accepted ONLY with reasoning off must be table'd per effort,")
    say("    because item 4 sends no effort by default and both models then reason.")
    say("  * `max_tokens` is expected to be REJECTED: item 3 maps it to")
    say("    `max_completion_tokens`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
