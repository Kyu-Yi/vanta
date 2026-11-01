"""One redaction rule, two independent halves — the harness's copy of it.

[VANTA-WHY] ``src/config/mod.rs`` carries the reference implementation
(``is_secret_env_name`` / ``redact_secret``) and this module is its Python
mirror, deliberately kept to the same two halves and the same thresholds:

* the NAME rule — an environment variable whose name contains ``KEY``,
  ``TOKEN``, ``SECRET`` or ``PASSWORD`` has a secret value, whatever it is;
* the VALUE rule — any whitespace-delimited ``sk-`` token longer than 12
  characters is a credential, wherever it is found.

The two halves are independent for the same reason they are in Rust. The name
rule covers the key this process read; the ``sk-`` rule covers a key that
arrived from somewhere else — a provider echoing the credential it just
rejected, a model quoting one back — and is being printed by us. Matching only
names would miss a key inside an error body; matching only ``sk-`` would miss a
provider whose keys carry no such prefix.

[VANTA-WHY] Redaction PRESERVES the surrounding text and whitespace. A helper
that split on whitespace and rejoined would silently reflow a model's raw output
in a log, and the artifacts whose whole purpose is forensic fidelity are exactly
where that must not happen.

[VANTA-WHY] The harness needs its own copy rather than shelling out to Vanta for
it: the places a secret can be printed here (the run log's ``# env:`` header, a
sandbox's ``export`` lines) are written by Python, some of them before any Vanta
process exists, and a redactor that could fail, or that had to be spawned per
line, is not a redactor.

Nothing here is reached by the model calls themselves. The env handed to
``subprocess.run`` for ``vanta_m3`` is the REAL environment — that is the one
place the key must survive, and it does: redaction applies to what is PRINTED,
never to what is passed.
"""

from __future__ import annotations

from typing import Mapping

SECRET_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
"""Substrings that mark an environment variable's NAME as carrying a secret."""

REDACTED = "<redacted>"
"""The token written in place of a redacted secret. Must equal the Rust one."""

MIN_SECRET_LEN = 12
"""A ``sk-`` token must be LONGER than this to be redacted.

The floor exists so a harmless string that merely begins ``sk-`` (``sk-learn``,
``sk-test``, a short path fragment) is not mangled. Real provider keys are far
longer, and mangling a harmless string is a worse failure than printing it.
"""

# ``str.isspace()`` is true for Unicode spaces (U+00A0, U+2028); Rust's
# ``u8::is_ascii_whitespace`` is not. The rule is about a credential, and a
# credential is never printed with a non-breaking space inside it — but the two
# implementations have to agree on where a token ENDS, so the set is written out.
_ASCII_WHITESPACE = frozenset(" \t\n\r\x0b\x0c")


def is_secret_env_name(name: str) -> bool:
    """True when an environment variable's name marks its value as a secret.

    Case-insensitive substring match on the four markers, so
    ``OPENAI_API_KEY``, ``VANTA_API_KEY``, ``GH_TOKEN``, ``HF_TOKEN``,
    ``CLIENT_SECRET`` and ``DB_PASSWORD`` are all covered without a list that has
    to be extended for every new provider.
    """
    upper = name.upper()
    return any(marker in upper for marker in SECRET_NAME_MARKERS)


def redact_secret(text: str) -> str:
    """Replace every ``sk-`` token longer than 12 characters with ``<redacted>``.

    Whitespace-delimited on purpose: the rule is about a credential, and a
    credential is never printed with spaces inside it.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("sk-", i):
            j = i
            while j < n and text[j] not in _ASCII_WHITESPACE:
                j += 1
            if j - i > MIN_SECRET_LEN:
                out.append(REDACTED)
                i = j
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def redact_env(env: Mapping[str, str]) -> dict[str, str]:
    """`env` with every secret VALUE replaced — names and order preserved.

    Both halves apply: a secret-named entry is replaced outright (its value need
    not look like a key at all), and any remaining value is scanned for an ``sk-``
    token. A redacted entry keeps its name, because knowing that the key was set
    (and under which variable) is the fact a reader is after — which is what
    makes this stronger than the name-prefix filter it replaces, where an entry
    simply vanished from the line.
    """
    safe: dict[str, str] = {}
    for key, value in env.items():
        text = "" if value is None else str(value)
        safe[key] = REDACTED if is_secret_env_name(key) else redact_secret(text)
    return safe
