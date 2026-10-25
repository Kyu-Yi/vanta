"""Issue text that never leaks the fix or the hidden test (design §2).

The highest-risk part of the harness: a leaky issue text turns the pilot into
"copy the answer out of the prompt", and the leak is invisible in every metric
afterwards.

**The text is the project's own report, and it is a committed file.** The freeze
step in ``prepare`` writes ``plans/bugsinpy_issues/<project>_<bug>.md`` once, from
the GitHub issue the fix commit references; ``prepare``, a measure-only run and the
run all read that file's gated section, so the text a measure-only run measures and
the text a run sends are the same bytes by construction (§2.4). This module
supplies what that step needs: the token counts, the gate, and the actionability
rule.

[VANTA-WHY] The fix commit's message used to *be* the text. It is written by the
person who fixed the bug, after they fixed it, and on the development split it
names the fix location outright for 7 of 11 bugs — a leak the gate below cannot
take back, because a symbol named by the person who fixed it is by definition
present in the buggy tree. So the commit-message body is dropped from the prompt
entirely, and with it the hidden-test behavioural sentence that used to be appended
to it: a real report's own words are the behaviour. What remains here of that
route is offline only. :func:`strip_diff_and_code` produces
``issue_text_commit`` — the message as the record keeps it (§2.4.6) — and
:func:`default_behavioural_line` and :func:`build_issue_text` survive so the old
assembled text can still be replayed for the write-up's counterfactual. None of
them is ever shown to Vanta; the prompt's text is the frozen report's gated
section.

**The leak gate (hard).** A token is a leak when it appears in the fix diff or in
the hidden test and appears **zero** times in the tree the Creator may read. That
is the precise meaning of "this came from the answer". The gate strips leaks from
the text, logs every removal with its class and reason, and marks a text that is no
longer actionable as needing a decision rather than guessing — and a report whose
project has no issue at all is not a text to fix but a bug that cannot take part
(§2.4).

[VANTA-WHY] Which tokens the gate gates. Identifiers that are *answer-shaped* —
contain ``_`` or a digit, are six characters or longer, or are quoted strings,
numeric literals, or file paths — are gated, because those are the shapes a
function name, a test name or a new constant take. Short unadorned prose words
(``the``, ``fix``, ``bug``) are **listed but not stripped**: gating them would
shred the text into a placeholder soup, and they carry no answer. The leak report
records the ungated set explicitly, so a reviewer can see exactly what was and was
not checked rather than trusting a summary. The gate **removes tokens and does nothing else**:
it never collapses whitespace, never re-wraps and never trims, because the report is the
reporter's own text and a code block in it has to survive as the Python they pasted (§2.4.4).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from . import suite
from .records import RecordError, field_of, strings_of
from .split import sha256_text
from .trees import nodeid_chain, nodeid_file

MIN_ACTIONABLE_WORDS = 8
"""Below this the stripped message says nothing a Creator could act on."""

TOKEN_RE = re.compile(
    r"""
      (?P<path>[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|txt|cfg|toml|json|rst|md|ini))
    | (?P<string>"[^"\n]*"|'[^'\n]*')
    | (?P<number>\b\d+(?:\.\d+)?\b)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)

FENCE_RE = re.compile(r"^```.*?^```\s*$", re.DOTALL | re.MULTILINE)


class IssueError(RuntimeError):
    """The issue text could not be derived."""


# ---------------------------------------------------------------------------
# step 1 + 2 — the fix-commit message, with diff and code removed
# ---------------------------------------------------------------------------
def fix_commit_message(repo: Path, commit: str) -> str:
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(repo), "show", "-s", "--format=%B", commit],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise IssueError(f"git show {commit} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def fix_commit_date(repo: Path, commit: str) -> str:
    """The fix commit's own date, ISO-8601 UTC (``...Z``) — §2.4's comment cut.

    Taken as a Unix timestamp and formatted here rather than with ``--format=%cI``,
    because ``%cI`` keeps the commit's own UTC offset: the GitHub API reports every
    ``created_at`` in ``Z``, and comparing ``2020-06-01T10:34:56+02:00`` against
    ``2020-06-01T10:34:56Z`` as strings would cut the comments in the wrong place.
    """
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(repo), "show", "-s", "--format=%ct", commit],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise IssueError(f"git show {commit} failed: {proc.stderr.strip()}")
    try:
        stamp = int(proc.stdout.strip())
    except ValueError as exc:
        raise IssueError(
            f"git show {commit} did not report a committer timestamp: {proc.stdout!r}"
        ) from exc
    return datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def strip_diff_and_code(message: str) -> str:
    """Remove diff hunks, fenced/indented code blocks and inline code spans."""
    text = FENCE_RE.sub("", message)

    kept: list[str] = []
    in_indented_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("diff --git")
            or stripped.startswith("@@")
            or stripped.startswith("index ")
            or stripped.startswith("--- ")
            or stripped.startswith("+++ ")
            or stripped.startswith("new file mode")
            or stripped.startswith("deleted file mode")
        ):
            in_indented_block = False
            continue
        if (line.startswith("+") and not line.startswith("+++")) or (
            line.startswith("-") and not line.startswith("---")
        ):
            continue
        if line.startswith("    ") or line.startswith("\t"):
            in_indented_block = True
            continue
        in_indented_block = False
        kept.append(line)

    text = "\n".join(kept)
    text = re.sub(r"`[^`\n]+`", "", text)  # inline code spans
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# step 3 — the behavioural line (deterministic by default, LLM-injectable)
# ---------------------------------------------------------------------------
ParaphraseFn = Callable[[str, list[str]], str]


def default_behavioural_line(stripped_message: str, failure_kinds: Sequence[str]) -> str:
    """A behavioural sentence built only from the failure *kind*.

    Never the assertion expression, the expected value, or the nodeid: those
    state the answer. ``AssertionError`` says "an assertion failed"; it does not
    say what the assertion was.

    [VANTA-WHY] This is the deterministic default. Design §2.2 has the local model
    write this sentence; that hook is wired in §11 step 5, where the LLM client
    lands. Until then a template is used, and the frozen text is regenerated when
    the model takes over — the split's text is a derived field either way.
    """
    kinds = sorted({k for k in failure_kinds if k})
    if not kinds:
        return (
            "Observed behaviour: the project's own test suite fails on the reported "
            "scenario. Expected behaviour: that scenario should work as documented."
        )
    return (
        "Observed behaviour: the project's own test suite fails on the reported "
        f"scenario ({', '.join(kinds)}). Expected behaviour: it should not."
    )


def failure_kinds_from_report(report: dict) -> list[str]:
    """Exception class names from a pytest-json-report, for the behavioural line.

    Only the class name is read out of ``longrepr``; the message — which for an
    assertion contains the expected value — is never extracted.

    The entries come from :func:`suite.report_tests`, so the report of a bug whose own
    pytest pin caps pytest-json-report at 0.2 (a flat list, §3.2) names its exceptions
    instead of silently contributing none.
    """
    kinds: list[str] = []
    for test in suite.report_tests(report):
        for phase in ("call", "setup", "teardown"):
            block = test.get(phase) or {}
            longrepr = block.get("longrepr")
            if longrepr is None:
                continue
            text = longrepr if isinstance(longrepr, str) else json.dumps(longrepr)
            found = re.findall(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b", text)
            if found:
                kinds.append(found[-1])
    return kinds


# ---------------------------------------------------------------------------
# step 4 — the leak gate
# ---------------------------------------------------------------------------
IGNORED_DIRS = frozenset({".git", "__pycache__", "node_modules"})
IGNORED_SUFFIXES = (".pyc", ".pyo")
IGNORED_NAMES = ("_manifest.json",)


@dataclass
class LeakRemoval:
    token: str
    token_class: str
    reason: str
    occurrences_in_answer: int
    occurrences_in_vanta_tree: int
    action: str  # "stripped"

    def to_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "token_class": self.token_class,
            "reason": self.reason,
            "occurrences_in_answer": self.occurrences_in_answer,
            "occurrences_in_vanta_tree": self.occurrences_in_vanta_tree,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, payload: object, *, where: str = "") -> "LeakRemoval":
        owner = cls.__name__
        return cls(
            token=str(field_of(payload, "token", owner=owner, where=where)),
            token_class=str(field_of(payload, "token_class", owner=owner, where=where)),
            reason=str(field_of(payload, "reason", owner=owner, where=where)),
            occurrences_in_answer=int(
                field_of(payload, "occurrences_in_answer", owner=owner, where=where)
            ),
            occurrences_in_vanta_tree=int(
                field_of(payload, "occurrences_in_vanta_tree", owner=owner, where=where)
            ),
            action=str(field_of(payload, "action", owner=owner, where=where)),
        )


@dataclass
class LeakResult:
    text: str
    removals: list[LeakRemoval] = field(default_factory=list)
    ungated_prose: list[str] = field(default_factory=list)
    ok: bool = True
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        """Lossless: the round trip is what keeps the reader honest.

        The ungated list is stored verbatim rather than deduplicated here, because
        ``load_prepared`` has to give back the same ``LeakResult`` the gate produced.
        ``write_leak_gate`` sorts and deduplicates for the human-readable report.
        """
        return {
            "text": self.text,
            "ok": self.ok,
            "reason": self.reason,
            "removals": [r.to_dict() for r in self.removals],
            "ungated_prose": list(self.ungated_prose),
        }

    @classmethod
    def from_dict(cls, payload: object, *, where: str = "") -> "LeakResult":
        owner = cls.__name__
        removals = field_of(payload, "removals", owner=owner, where=where)
        if not isinstance(removals, list):
            raise RecordError(
                f"{owner} in {where}: field 'removals' is "
                f"{type(removals).__name__}, not a list"
            )
        return cls(
            text=str(field_of(payload, "text", owner=owner, where=where)),
            removals=[LeakRemoval.from_dict(item, where=where) for item in removals],
            ungated_prose=strings_of(payload, "ungated_prose", owner=owner, where=where),
            ok=bool(field_of(payload, "ok", owner=owner, where=where)),
            reason=field_of(payload, "reason", owner=owner, where=where),
        )


def normalise_token(token: str) -> str:
    """Case-fold, ``_`` ≡ ``-``, drop ``self.`` prefixes and one quote layer."""
    text = token.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    text = text.removeprefix("self.")
    return text.lower().replace("-", "_")


def tokenize(text: str) -> list[tuple[str, str]]:
    """``[(token, class)]`` in order, for the classes the gate cares about."""
    out: list[tuple[str, str]] = []
    for match in TOKEN_RE.finditer(text):
        out.append((match.group(0), match.lastgroup or "ident"))
    return out


def is_answer_shaped(token: str, token_class: str) -> bool:
    """Would this token be gated if it existed only in the answer?"""
    if token_class in ("string", "number", "path"):
        return True
    bare = token.removeprefix("self.")
    return "_" in bare or any(c.isdigit() for c in bare) or len(bare) >= 6


def vanta_tree_token_counts(tree: Path) -> dict[str, int]:
    """Normalised token → occurrence count over the tree Vanta is given."""
    counts: dict[str, int] = {}
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(tree).as_posix()
        parts = rel.split("/")
        if any(part in IGNORED_DIRS for part in parts[:-1]):
            continue
        if rel.endswith(IGNORED_SUFFIXES) or rel.endswith(IGNORED_NAMES):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # The path itself is part of what the Creator can read.
        for token, _klass in tokenize(rel):
            counts[normalise_token(token)] = counts.get(normalise_token(token), 0) + 1
        for token, _klass in tokenize(text):
            key = normalise_token(token)
            counts[key] = counts.get(key, 0) + 1
    return counts


def answer_token_counts(
    bug_patch_txt: str,
    hidden_sources: Iterable[str],
    hidden_nodeids: Sequence[str],
) -> dict[str, int]:
    """Normalised token → occurrence count over the fix diff and the hidden test."""
    counts: dict[str, int] = {}
    blobs = [bug_patch_txt, *hidden_sources]
    # The nodeid's own parts are answer material even if the source moved.
    for nodeid in hidden_nodeids:
        blobs.append(nodeid)
        blobs.extend(nodeid_chain(nodeid))
        blobs.append(nodeid_file(nodeid))
    for blob in blobs:
        for token, _klass in tokenize(blob):
            key = normalise_token(token)
            counts[key] = counts.get(key, 0) + 1
    return counts


def token_counts_sha256(counts: Mapping[str, int]) -> str:
    """A stable hash of a token→count map — the frozen artifact's input provenance.

    [VANTA-WHY] The gate runs once, at freeze time, and its inputs are computed from the
    trees rather than from a suite run (§2.4). Recording what those inputs *were* is the
    only way a later reader can tell whether the frozen text was gated against the same
    oracle it is being run with: a re-derivation that produced a different answer set
    would make the frozen text's removals wrong, and nothing in the text itself shows it.
    Sorted, so the hash is a property of the set and not of the walk's order.
    """
    lines = [f"{token}\t{count}" for token, count in sorted(counts.items())]
    return sha256_text("\n".join(lines) + "\n")


def hidden_test_sources(oracle_tree: Path, nodeids: Sequence[str]) -> list[str]:
    sources: list[str] = []
    for rel in sorted({nodeid_file(n) for n in nodeids}):
        path = oracle_tree / rel
        try:
            sources.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return sources


def leak_gate(
    text: str,
    *,
    answer_tokens: dict[str, int],
    tree_tokens: dict[str, int],
) -> LeakResult:
    """Strip every answer-shaped token that exists only in the answer — and change nothing else.

    [VANTA-WHY] The gate substitutes and never reformats. It used to end with
    ``re.sub(r"[ \\t]{2,}", " ", ...).strip()``, and that collapse turned the four-space
    indentation of a fenced code block in ``sanic/1``'s report into one space: ``    print('1')``
    came out `` print('1')``, which is not Python, in the one part of the prompt the Creator is
    meant to be able to run. A collapse is a *rewrite* of the reporter's text, and the gate's
    job is to remove answer tokens and record what it removed — a stage that also reformats
    what it passes through ships a prompt nobody wrote and no reviewer can compare against the
    report. ``[redacted]`` and the removal table already say exactly what the gate did.

    What it returns is therefore the text it was given, byte-for-byte, except for the
    substitutions it recorded. Runs of spaces, tabs, leading indentation and blank lines all
    survive: skipping a token leaves the text on either side of it untouched. The only
    normalisation anywhere on the way to the prompt is the line-ending fold, and it lives in
    one place — :func:`issue_source._canonical`, which is about *storing* the text, not about
    what the text says (§2.4.3, §2.4.4).
    """
    leaks: dict[str, int] = {
        token: count
        for token, count in answer_tokens.items()
        if tree_tokens.get(token, 0) == 0
    }

    removals: list[LeakRemoval] = []
    ungated: list[str] = []
    out: list[str] = []
    cursor = 0

    for match in TOKEN_RE.finditer(text):
        out.append(text[cursor : match.start()])
        cursor = match.end()
        raw = match.group(0)
        klass = match.lastgroup or "ident"
        key = normalise_token(raw)
        if key not in leaks:
            out.append(raw)
            continue
        if not is_answer_shaped(raw, klass):
            # Listed, not stripped: short prose words carry no answer, and gating
            # them would leave nothing to read. Recorded so the choice is visible.
            ungated.append(raw)
            out.append(raw)
            continue
        removals.append(
            LeakRemoval(
                token=raw,
                token_class=klass,
                reason="token exists in the fix diff or the hidden test and nowhere in "
                "the tree the Creator may read",
                occurrences_in_answer=leaks[key],
                occurrences_in_vanta_tree=tree_tokens.get(key, 0),
                action="stripped",
            )
        )
        out.append("[redacted]")

    out.append(text[cursor:])
    cleaned = "".join(out)

    ok, reason = is_actionable(cleaned)
    return LeakResult(
        text=cleaned, removals=removals, ungated_prose=ungated, ok=ok, reason=reason
    )


def is_actionable(text: str) -> tuple[bool, Optional[str]]:
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < MIN_ACTIONABLE_WORDS:
        return False, (
            f"stripped message has {len(words)} actionable words "
            f"(< {MIN_ACTIONABLE_WORDS}); it says nothing a Creator could work from"
        )
    if text.count("[redacted]") >= len(words):
        return False, "the leak gate removed more text than it left"
    return True, None


# ---------------------------------------------------------------------------
# assembling the issue text
# ---------------------------------------------------------------------------
@dataclass
class IssueText:
    text: str
    pre_gate: str
    leak: LeakResult
    failure_kinds: list[str]
    ok: bool
    needs_decision: Optional[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "pre_gate": self.pre_gate,
            "failure_kinds": list(self.failure_kinds),
            "ok": self.ok,
            "needs_decision": self.needs_decision,
            "leak": self.leak.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object, *, where: str = "") -> "IssueText":
        """The reader for :meth:`to_dict` — the one ``runner.load_prepared`` uses.

        Every field is required, including the ones a previous reader guessed at:
        the crash this pair exists to prevent was a reader constructing an
        ``IssueText`` without ``leak``/``failure_kinds``/``needs_decision`` and
        setting them afterwards, which is exactly how the two sides drifted.
        """
        owner = cls.__name__
        return cls(
            text=str(field_of(payload, "text", owner=owner, where=where)),
            pre_gate=str(field_of(payload, "pre_gate", owner=owner, where=where)),
            leak=LeakResult.from_dict(
                field_of(payload, "leak", owner=owner, where=where), where=where
            ),
            failure_kinds=strings_of(payload, "failure_kinds", owner=owner, where=where),
            ok=bool(field_of(payload, "ok", owner=owner, where=where)),
            needs_decision=field_of(payload, "needs_decision", owner=owner, where=where),
        )


def build_issue_text(
    *,
    stripped_message: str,
    failure_kinds: Sequence[str],
    answer_tokens: dict[str, int],
    tree_tokens: dict[str, int],
    paraphrase: Optional[ParaphraseFn] = None,
) -> IssueText:
    body = strip_diff_and_code(stripped_message)
    writer = paraphrase or default_behavioural_line
    behavioural = writer(body, list(failure_kinds))
    pre_gate = f"{body}\n\n{behavioural}".strip() if body else behavioural

    leak = leak_gate(pre_gate, answer_tokens=answer_tokens, tree_tokens=tree_tokens)
    needs: Optional[str] = None
    if not leak.ok:
        needs = leak.reason or "the leak gate left the text unactionable"
    return IssueText(
        text=leak.text,
        pre_gate=pre_gate,
        leak=leak,
        failure_kinds=sorted(set(failure_kinds)),
        ok=needs is None,
        needs_decision=needs,
    )
