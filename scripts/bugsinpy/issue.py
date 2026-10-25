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

SOURCE_SUFFIXES = (".py", ".pyi", ".pyx", ".pxd")
"""Files that define a name at module level — the same set §1.6.8 hands content for.

[VANTA-WHY] ``.pyx``/``.pxd`` are here because pandas and spaCy have real bugs in Cython
files, and a symbol the buggy tree defines in one of them is as much a name of the tree as
one it defines in a ``.py``. This is a *reporting* question — did the visible issue text name
something the tree defines? — so the answer has to be about the tree, not about the subset of
it :mod:`ast` happens to parse.
"""

TOP_LEVEL_DEF_RE = re.compile(
    r"^(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)"
    r"|^class[ \t]+([A-Za-z_]\w*)"
    r"|^([A-Za-z_]\w*)[ \t]*(?::[^=\n]*)?=(?!=)",
    re.MULTILINE,
)
"""The three shapes that bind a name at module level: ``def``, ``class``, ``x =`` / ``x: T =``.

Column 0 only, which is what "top level" means: a method of a class or a closure body is not a
name of the tree in the sense §2.4.6 asks about. A regex rather than :mod:`ast` so one rule
covers ``.py``, ``.pyi``, ``.pyx`` and ``.pxd`` alike. Its false positive is a binding written
at column 0 *inside a string literal*, which can only matter if the issue text also names that
identical token — and what this feeds is a flag the paper reports, never a selection.
"""


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


# ---------------------------------------------------------------------------
# the reporting flag — what the visible text names that the tree also defines (§2.4.6)
# ---------------------------------------------------------------------------
def _names_one_file(named: str, rel: str) -> bool:
    """Do these two strings name the same file of the tree?

    Exact match, or one is the other with extra leading directories — the shapes a report
    uses when it quotes a path from its own machine. A bare ``mod.py`` is **not** a match for
    ``pkg/mod.py``: a basename is not a repo path, and accepting it would make the flag fire
    on nearly every report that quotes a traceback.
    """
    if named == rel:
        return True
    if "/" not in named:
        return False
    return named.endswith("/" + rel) or rel.endswith("/" + named)


def top_level_names(source: str) -> set[str]:
    """The names one file binds at module level (:data:`TOP_LEVEL_DEF_RE`)."""
    names: set[str] = set()
    for match in TOP_LEVEL_DEF_RE.finditer(source):
        name = match.group(1) or match.group(2) or match.group(3)
        if name:
            names.add(name)
    return names


def tree_names_in_text(text: str, tree: Path) -> tuple[list[str], list[str]]:
    """What the text names that the tree itself defines: repo paths, then top-level names.

    [VANTA-WHY] A reporting fact about the prompt, and nothing else. A report that says
    ``sanic/app.py`` or ``register_named_middleware`` has localised the bug for the Creator to
    a degree ``localization_hit`` never credits it for — that metric is about where the patch
    went, not about what the text already said — and the write-up has to be able to report how
    often the visible text does the harness's localisation for it. So the answer is recorded
    per bug in the record (§2.4.6) and read by **nothing** that builds a prompt, exactly like
    ``issue_text_commit``.

    It is computed over the text a run sends — the frozen report's gated section, title and
    body, post-gate — and never over the comments: the default setting does not send those, so
    a comment naming the file would be a fact about a prompt nobody sends (§2.4.4). The
    ``tree`` is the buggy source the gate already reads, so the flag cannot disagree with the
    text about which tree it is talking about.

    A path is recorded when the tree contains a file for it (:func:`_names_one_file`), and a
    name when some source file of the tree binds it at module level. Both lists are sorted and
    hold the tokens *as the text writes them*, so the paper can quote the report's own words.
    """
    rel_paths: set[str] = set()
    defined: set[str] = set()
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(tree).as_posix()
        parts = rel.split("/")
        if any(part in IGNORED_DIRS for part in parts[:-1]):
            continue
        if rel.endswith(IGNORED_SUFFIXES) or rel.endswith(IGNORED_NAMES):
            continue
        rel_paths.add(rel)
        if path.suffix in SOURCE_SUFFIXES:
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            defined |= top_level_names(source)

    paths: set[str] = set()
    symbols: set[str] = set()
    for token, klass in tokenize(text):
        if klass == "ident":
            if token in defined:
                symbols.add(token)
        elif klass in ("path", "string"):
            # A quoted path tokenizes as a string, not as a path: ``"pkg/mod.py"`` starts with
            # a quote, so the path alternative cannot match at that position — which is how a
            # report's own traceback writes it (§2.4.4).
            named = token.strip("\"'") if klass == "string" else token
            if any(_names_one_file(named, rel) for rel in rel_paths):
                paths.add(named)
    return sorted(paths), sorted(symbols)


def gate_text(
    text: str,
    *,
    answer_tokens: dict[str, int],
    tree_tokens: dict[str, int],
) -> LeakResult:
    """Replace every answer-shaped token that exists only in the answer — and change nothing else.

    **The one substitution.** Both the text a run sends (:func:`leak_gate`) and the comments a
    run does not (:func:`gate_hints`, §2.4.4) come through here, so "gated" means exactly one
    thing in this harness: the tokens the answer and only the answer carries become
    ``[redacted]``, and every replacement is recorded with its occurrence counts.

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

    ``ok`` is ``True`` and ``reason`` is ``None`` — the absence of the question, not an answer
    to it. Whether a text says enough to work from is asked once and only of the text a run
    sends, which is why that question lives in :func:`leak_gate`.
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
    return LeakResult(
        text="".join(out),
        removals=removals,
        ungated_prose=ungated,
        ok=True,
        reason=None,
    )


def leak_gate(
    text: str,
    *,
    answer_tokens: dict[str, int],
    tree_tokens: dict[str, int],
) -> LeakResult:
    """The gate over the text a run sends: :func:`gate_text`, and then its verdict.

    A report the gate has stripped down to nothing is not a prompt — it is a bug that would
    spend an attempt on a text no Creator can work from — so this is the one call site that
    asks :func:`is_actionable`, and its answer is what makes a bug a NEEDS DECISION rather
    than a run. The comments of the same report are gated by :func:`gate_text` alone
    (:func:`gate_hints`): they are never sent in the default setting, so there is no
    attempt for them to waste and nothing to decide about them (§2.4.4).
    """
    gated = gate_text(text, answer_tokens=answer_tokens, tree_tokens=tree_tokens)
    ok, reason = is_actionable(gated.text)
    return LeakResult(
        text=gated.text,
        removals=gated.removals,
        ungated_prose=gated.ungated_prose,
        ok=ok,
        reason=reason,
    )


def gate_hints(
    text: str,
    *,
    answer_tokens: dict[str, int],
    tree_tokens: dict[str, int],
) -> LeakResult:
    """The same substitution over a report's comments, and **no verdict about the result**.

    [VANTA-WHY] The comments are SWE-bench's ``hints_text``: they were posted before the fix
    and they frequently name it. ``matplotlib/1``'s carry a maintainer saying what the cause
    was and a range in ``backend_agg.py``, which is the answer — so they are gated by the same
    rule as the text a run sends, and then not sent (§2.4.4). A hint section that an ablation
    later shows must not be able to carry what the default setting refuses, which is why the
    gate runs here rather than being waived as "not shown, so not a risk".

    The verdict is not asked: :func:`is_actionable` is about whether a text is worth spending
    an attempt on, and a text nobody sends is never such a text. What is stored for this
    section is the gated text and every substitution made in it, and not an ``ok``/``reason``
    nobody asked for.
    """
    return gate_text(text, answer_tokens=answer_tokens, tree_tokens=tree_tokens)


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
