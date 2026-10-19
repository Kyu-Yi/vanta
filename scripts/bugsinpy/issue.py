"""Issue text that never leaks the fix or the hidden test (design §2).

The highest-risk part of the harness: a leaky issue text turns the pilot into
"copy the answer out of the prompt", and the leak is invisible in every metric
afterwards. Two independent mechanisms guard it.

**Step 2 — strip diff and code only.** The fix-commit message is run through a
stripper that removes diff hunks, fenced and indented code blocks, and inline code
spans. Nothing else.

[VANTA-WHY] An earlier draft *also* stripped every line containing a symbol found
in ``bug_patch.txt``. That rule deletes the name of the function being fixed — a
name that is present in the buggy tree and that the Creator needs in order to work
at all — and it would have made many issues unsolvable. Symbol-level leakage is
the leak gate's job, and the gate is strictly better because it fires only on
tokens that exist *nowhere* in the tree the Creator is allowed to read.

**Step 4 — the leak gate (hard).** A token is a leak when it appears in the fix
diff or in the hidden test and appears **zero** times in ``vanta_tree/``. That is
the precise meaning of "this came from the answer". The gate strips leaks from the
text, logs every removal with its class and reason, and marks a text that is no
longer actionable as ``NEEDS DECISION`` so a human decides rather than the harness
guessing.

[VANTA-WHY] Which tokens the gate gates. Identifiers that are *answer-shaped* —
contain ``_`` or a digit, are six characters or longer, or are quoted strings,
numeric literals, or file paths — are gated, because those are the shapes a
function name, a test name or a new constant take. Short unadorned prose words
(``the``, ``fix``, ``bug``) are **listed but not stripped**: gating them would
shred every commit message into a placeholder soup, and they carry no answer. The
leak report records the ungated set explicitly, so a reviewer can see exactly what
was and was not checked rather than trusting a summary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

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
    """
    kinds: list[str] = []
    for test in report.get("tests") or []:
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


@dataclass
class LeakResult:
    text: str
    removals: list[LeakRemoval] = field(default_factory=list)
    ungated_prose: list[str] = field(default_factory=list)
    ok: bool = True
    reason: Optional[str] = None

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "removals": [r.__dict__ for r in self.removals],
            "ungated_prose_only_in_answer": sorted(set(self.ungated_prose)),
        }


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
    """Strip every answer-shaped token that exists only in the answer."""
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
    cleaned = re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()

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

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "pre_gate": self.pre_gate,
            "failure_kinds": self.failure_kinds,
            "ok": self.ok,
            "needs_decision": self.needs_decision,
            "leak": self.leak.as_dict(),
        }


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
