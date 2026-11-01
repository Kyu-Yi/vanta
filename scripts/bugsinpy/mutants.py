"""Item 5 — the reference-fix mutant pool: deliberately broken fixes (§5.4).

A review row needs a *known-bad* patch to review. ``candidate_set`` already supplies
one floor (the no-op) and up to two pooled failed Creator patches, but those only
exist *after* a repair wave has run and only when the Creator failed. The mutant
pool is the other half: broken versions of the developer fix itself, produced
*mechanically* from the one patch that is known to be a real fix.

[VANTA-WHY] Four rules here are the kind that fail silently, and each one has a
cheap failure mode that looks like a result rather than a bug:

* **The mutants are made from the source-only reference fix.** The mutant source is
  ``prep.bug.bug_patch_txt`` through ``runner.test_hunk_free`` — the *same* bytes
  ``oracle.grade_dev_fix`` grades and ``candidate_set``'s ``dev`` candidate carries.
  Every candidate is applied with its test hunks dropped (``oracle.grade``), so a
  mutant made from a test hunk would be a patch the oracle never applies and would
  grade against the wrong tree.
* **A syntax-error mutant is not a wrong patch.** A mutant whose diff applies but
  whose Python does not parse is *trivially* detectable: pytest collects nothing,
  the hidden nodeids read ``absent``, and the oracle grades it ``bad``. That would
  pad the pool with patches no reviewer can be credited for finding, so every
  candidate is preflighted in Shuru (apply **then** ``py_compile``) and a mutant
  that fails either step is discarded and counted, never graded.
* **A mutant that still passes the oracle is equivalent, not bad.** Keeping it
  would put a *correct* patch in a "known-bad" pool and score every review row
  against a lie. Equivalence is dropped and counted, on the same rule that makes
  the no-op floor a floor.
* **Selection is a pure function of the diff.** Sort by ``(operator priority,
  sha256)`` and take the first ≤3 the oracle grades bad. No randomness, no
  filesystem order, no counter: the pool for a bug is reproducible from the
  reference fix plus the frozen generator config alone (M5 freezes it, item 19
  hashes it).

The operators are **pure text functions on the unified diff** — no Python AST, no
project-language knowledge — so the same four work on Java in M4 and the generator
stays language-neutral. Every mutant is a *complete* diff, so it is a drop-in
``write_candidate_patch`` body and the pool's patch files are ordinary review
patches.

The artifact is ``<batch_dir>/pool/<slug>.mutants.json``: a ``#`` title, a
``- key: value`` header carrying a hash of the body, a marker line, then the JSON
body. It is written **once and hash-locked**; a reader recomputes the header hash
and refuses a file that was edited after it was frozen (the same lock form
``issue_source`` uses for a frozen report). Item 6 assembles the review's read-path
``pool/<slug>.json`` from this file plus the reference fix, the no-op and the pooled
failed patches, referring to this one by its ``content-sha256`` — so the two files
never edit each other.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import locked
from . import oracle as oracle_mod
from . import runner as runner_mod
from . import shuru as shuru_mod
from . import suite as suite_mod
from . import trees as tree_mod
from .prepare import BugPrep
from .split import sha256_text

MUTANTS_DIRNAME = "pool"
MUTANTS_SUFFIX = ".mutants.json"
MUTANT_KIND = "mutant"
"""The candidate ``kind`` a pool entry carries; item 6 keys on it."""

MUTANTS_MARKER = "<!-- mutants -->"
"""The line between the lock header and the JSON body."""

SCHEMA = 1

MAX_MUTANTS = 3
"""At most three bad mutants per bug (§Item 5: "at most 3 mutants per bug")."""

GRADE_CAP = 12
"""The most gradings one bug may spend looking for its three bad mutants.

[VANTA-WHY] A cap rather than "grade until three are found", because the number of
candidates is a property of the fix (a 40-line fix yields hundreds) and a bug that
cannot supply three bad mutants would otherwise grade every one of them. Twelve is
four times the target: enough that a bug with a noisy front of equivalents still
finds its three, small enough that an unlucky bug is bounded. It is part of the
frozen generator config — the deviation from "the first that grades bad" is only in
the exhaustion direction and is recorded per bug as ``cap-hit``.
"""

PREFLIGHT_CAP = 48
"""The most preflights one bug may spend, whatever the outcome of each.

[VANTA-WHY] C10 — ``GRADE_CAP`` bounds *gradings*, and a mutant is graded only
after it has applied and compiled, so a candidate that fails either is discarded
before it counts. Those discards still boot a Shuru VM each (the tree copy, the
``git apply`` and the ``py_compile`` all run in the checkpoint), and nothing
bounded them: a fix whose candidates mostly fail to apply could preflight every
mutant the operators produced. This is the missing bound. It is the same shape as
``GRADE_CAP`` — four times it — so a bug may lose up to three quarters of its
attempts to discards and still spend its full grading budget; past that the pool
is starved, the loop stops, and ``cap-hit`` says so.
"""

POOL_COMPILE_TIMEOUT_SECS = 600
"""``py_compile`` is fast; this is a backstop, not an estimate."""


class MutantError(RuntimeError):
    """The pool could not be generated, or a written pool file failed its lock."""


# ---------------------------------------------------------------------------
# the diff, as structures the operators can rewrite
# ---------------------------------------------------------------------------
_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old>\d+)(?:,(?P<oldn>\d+))? "
    r"\+(?P<new>\d+)(?:,(?P<newn>\d+))? @@(?P<tail>.*)$"
)


@dataclass
class Hunk:
    """One ``@@`` hunk: its original old-side start, and its body lines verbatim."""

    old_start: int
    new_start: int
    tail: str
    body: list[str]

    def copy(self) -> "Hunk":
        return Hunk(self.old_start, self.new_start, self.tail, list(self.body))


@dataclass
class FileSection:
    """One ``diff --git`` section: its header lines and its hunks."""

    header: list[str]
    hunks: list[Hunk]

    def copy(self) -> "FileSection":
        return FileSection(list(self.header), [hunk.copy() for hunk in self.hunks])


def parse_diff(text: str) -> list[FileSection]:
    """Split a unified diff into file sections and hunks, keeping every line verbatim.

    Only the ``@@`` header is interpreted; the body is kept as raw lines, so a
    rewrite is a matter of changing one line and re-rendering.
    """
    sections: list[FileSection] = []
    current: Optional[FileSection] = None
    hunk: Optional[Hunk] = None
    for line in text.splitlines():
        match = _HUNK_HEADER.match(line)
        if match:
            if current is None:
                raise MutantError("a hunk header appears before any 'diff --git' line")
            hunk = Hunk(
                old_start=int(match.group("old")),
                new_start=int(match.group("new")),
                tail=match.group("tail"),
                body=[],
            )
            current.hunks.append(hunk)
            continue
        if line.startswith("diff --git "):
            current = FileSection(header=[line], hunks=[])
            sections.append(current)
            hunk = None
            continue
        if current is None:
            # A preamble before the first ``diff --git`` (never produced by the
            # ``git diff`` this harness reads, but kept rather than refused so a
            # stray blank leading line cannot take a whole bug's pool down).
            current = FileSection(header=[line], hunks=[])
            sections.append(current)
            continue
        if hunk is None:
            current.header.append(line)
        else:
            hunk.body.append(line)
    return sections


def _is_file_section(section: FileSection) -> bool:
    return bool(section.header) and section.header[0].startswith("diff --git ")


def _prune(sections: list[FileSection]) -> list[FileSection]:
    """Drop file sections left with no hunks (a header with nothing under it is not a patch)."""
    return [
        section
        for section in sections
        if not _is_file_section(section) or section.hunks
    ]


def _counts(body: Sequence[str]) -> tuple[int, int]:
    """``(old_count, new_count)`` for a hunk body — an empty line counts as context.

    [VANTA-WHY] An empty body line is treated as a context line on **both** sides,
    which is what ``git`` itself does: a context line for a blank source line is
    normally a single space, but a tool that stripped trailing whitespace leaves it
    empty, and git accepts that. Counting it as neither side would render a hunk
    whose header disagrees with its body — a corrupt patch that ``git apply``
    refuses, so a whole operator's candidates would silently vanish into
    ``did-not-apply``.
    """
    old = sum(1 for line in body if line == "" or line[:1] in (" ", "-"))
    new = sum(1 for line in body if line == "" or line[:1] in (" ", "+"))
    return old, new


def render_diff(sections: Sequence[FileSection]) -> str:
    """Re-render the diff, recomputing every ``@@`` header from the mutated bodies.

    [VANTA-WHY] The headers are recomputed rather than edited in place, because two
    operators change a hunk's line counts (dropping a hunk, deleting a ``+`` line)
    and ``suite.git_apply_patch`` runs ``git apply`` **without** ``--recount``: a
    header whose counts disagree with its body makes git refuse the patch as
    corrupt, and the mutant would be discarded as *unappliable* when the real fault
    is arithmetic in this file. The old start is kept (it is a position in the
    unmodified buggy file, which a mutant patch still targets) and the new start is
    derived from the running delta of the retained hunks, which is exactly how a
    unified diff defines it.
    """
    lines: list[str] = []
    for section in sections:
        lines.extend(section.header)
        running = 0
        for hunk in section.hunks:
            old_count, new_count = _counts(hunk.body)
            new_start = hunk.old_start + running
            lines.append(
                f"@@ -{hunk.old_start},{old_count} +{new_start},{new_count} @@{hunk.tail}"
            )
            lines.extend(hunk.body)
            running += new_count - old_count
    # A blank line renders as "" here and as "" in the file — the one place where
    # "".splitlines() and "\n".join are not inverse, and the render is the writer.
    return "".join(line + "\n" for line in lines)


def _plus_lines(
    sections: Sequence[FileSection],
) -> list[tuple[int, int, int, str]]:
    """Every added line as ``(section, hunk, index, content-without-the-plus)``."""
    found: list[tuple[int, int, int, str]] = []
    for si, section in enumerate(sections):
        for hi, hunk in enumerate(section.hunks):
            for li, line in enumerate(hunk.body):
                if line.startswith("+") and not line.startswith("+++"):
                    found.append((si, hi, li, line[1:]))
    return found


# ---------------------------------------------------------------------------
# the four operators — pure text functions on the diff
# ---------------------------------------------------------------------------
def _with_line(
    sections: Sequence[FileSection], si: int, hi: int, li: int, content: str
) -> str:
    copy = [section.copy() for section in sections]
    copy[si].hunks[hi].body[li] = "+" + content
    return render_diff(copy)


def hunk_drop(text: str) -> list[str]:
    """Operator 1 — remove one whole hunk, so a multi-hunk fix becomes a partial one.

    A one-hunk fix yields nothing: "a fix with **several** hunks loses one", and a
    fix with one hunk has no partial version to drop to.
    """
    sections = parse_diff(text)
    if sum(len(section.hunks) for section in sections) < 2:
        return []
    out: list[str] = []
    for si, section in enumerate(sections):
        for hi in range(len(section.hunks)):
            copy = [s.copy() for s in sections]
            del copy[si].hunks[hi]
            out.append(render_diff(_prune(copy)))
    return out


_FLIP_RE = re.compile(
    r"==|!=|<=|>=|(?<![<>=!-])<(?![<=])|(?<![<>=!-])>(?![>=])|\b(?:and|or)\b"
)
_FLIP = {
    "==": "!=",
    "!=": "==",
    "<=": "<",
    ">=": ">",
    "<": "<=",
    ">": ">=",
    "and": "or",
    "or": "and",
}

# [VANTA-WHY] C10 — the negatable heads, and the guard against the rest.
#
# ``_flip_not`` used to prepend ``not `` to *any* non-empty line, so
# ``with open(…) as f:`` became ``not with open(…) as f:`` and ``return x`` became
# ``not return x`` — both syntax errors. A syntax-error mutant is trivially
# detectable (see the module docstring), so it pads the pool with patches no
# reviewer can be credited for finding. The operator now negates only a real
# condition: an ``if``/``elif``/``while`` header, the value of an ``assert`` or a
# ``return``, or a bare expression statement. Everything else is left alone.
_COND_HEADER_RE = re.compile(r"^(if|elif|while)\s+([^:]+):\s*$")
_EXPR_HEAD_RE = re.compile(r"^(assert|return)\s+(\S.*)$")
# A statement head — `not <such a line>` never parses, so it is never touched.
_STATEMENT_START_RE = re.compile(
    r"^(?:def|class|with|for|async|await|import|from|del|pass|break|continue|"
    r"raise|global|nonlocal|yield|try|except|finally|else|if|elif|while|assert|"
    r"return|lambda)\b"
)
# Any `=` that makes the line an assignment rather than an expression, so that
# `not <line>` is a syntax error and the line is never negated: a plain `x = 1`,
# an augmented `x += 1` … `x **= 2` … `x <<= 2` / `x >>= 2`, or the walrus
# `x := 1`. `==`, `!=`, `<=` and `>=` are comparisons, not assignments. The
# shifts get their own alternative because the negative lookbehind that keeps
# `<=`/`>=` out of the plain case would also swallow `<<=`/`>>=` (their `=` is
# preceded by `<`/`>`), and `not x <<= 1` does not parse.
_ASSIGN_RE = re.compile(r"[+\-*/%@&|^]=|<<=|>>=|(?<![=!<>:])=(?!=)")
# [VANTA-WHY] E11 — the characters that can begin an operand. ``_flip_not``
# prefixes ``not ``, so the line it rewrites must be able to start an expression:
# a name or number (``isalnum``), an opening bracket, a quote, or a unary
# ``-``/``+``/``~``. Anything else — a closing bracket, a comma, a colon, ``.``,
# an operator — marks a continuation line or a fragment (``, b)):``,
# ``.append(x)``, ``)``), and ``not <such a line>`` is a syntax error.
_OPERAND_START_CHARS = frozenset("_([{'+~-\"")


def _toggle_not(expr: str) -> str:
    """``expr`` with a leading ``not `` removed, or added when there is none."""
    return expr[len("not ") :] if expr.startswith("not ") else "not " + expr


def _is_bare_expression(line: str) -> bool:
    """Is ``line`` a single expression, so that ``not <line>`` still parses?"""
    if _STATEMENT_START_RE.match(line) or line[0] in "@#":
        # a statement keyword, a decorator or a comment — not an expression
        return False
    if not (line[0].isalnum() or line[0] in _OPERAND_START_CHARS):
        # a continuation line or a fragment — ``not <line>`` does not parse
        return False
    return _ASSIGN_RE.search(line) is None


def _flip_not(content: str) -> Optional[str]:
    """``content`` with its condition negated, or ``None`` when it is not one.

    Three forms are conditions: ``if``/``elif``/``while <cond>:`` (negate the
    condition, keeping the header), ``assert``/``return <value>`` (negate the
    value), and a bare expression statement (negate the whole line). A leading
    ``not`` is removed in each case, so the operator is an involution.
    """
    stripped = content.lstrip()
    if not stripped:
        return None
    indent = content[: len(content) - len(stripped)]
    header = _COND_HEADER_RE.match(stripped)
    if header:
        return f"{indent}{header.group(1)} {_toggle_not(header.group(2))}:"
    head = _EXPR_HEAD_RE.match(stripped)
    if head:
        return f"{indent}{head.group(1)} {_toggle_not(head.group(2))}"
    if _is_bare_expression(stripped):
        return indent + _toggle_not(stripped)
    return None


def condition_flip(text: str) -> list[str]:
    """Operator 2 — invert one comparison or boolean token, or the leading ``not``."""
    sections = parse_diff(text)
    out: list[str] = []
    for si, hi, li, content in _plus_lines(sections):
        for match in _FLIP_RE.finditer(content):
            replacement = _FLIP[match.group(0)]
            out.append(
                _with_line(
                    sections,
                    si,
                    hi,
                    li,
                    content[: match.start()] + replacement + content[match.end() :],
                )
            )
        variant = _flip_not(content)
        if variant is not None and variant != content:
            out.append(_with_line(sections, si, hi, li, variant))
    return out


_INT_RE = re.compile(r"(?<![\w.])\d+(?![\w.])")
_LEN_RE = re.compile(r"\blen\(([^()]*)\)")
_LEN_MINUS_RE = re.compile(r"\blen\(([^()]*)\)\s*-\s*1")


def boundary_shift(text: str) -> list[str]:
    """Operator 3 — shift one integer literal by ±1, or move a ``len(…)`` boundary.

    The integer rule skips an identifier's digit and a float's: ``x1`` and ``1.5``
    are not boundaries, and shifting them would be a different edit than the one
    this operator means.
    """
    sections = parse_diff(text)
    out: list[str] = []
    for si, hi, li, content in _plus_lines(sections):
        for match in _INT_RE.finditer(content):
            value = int(match.group(0))
            for delta in (1, -1):
                out.append(
                    _with_line(
                        sections,
                        si,
                        hi,
                        li,
                        content[: match.start()] + str(value + delta) + content[match.end() :],
                    )
                )
        for match in _LEN_MINUS_RE.finditer(content):
            out.append(
                _with_line(
                    sections,
                    si,
                    hi,
                    li,
                    content[: match.start()] + f"len({match.group(1)})" + content[match.end() :],
                )
            )
        for match in _LEN_RE.finditer(content):
            out.append(
                _with_line(
                    sections,
                    si,
                    hi,
                    li,
                    content[: match.start()]
                    + f"len({match.group(1)}) - 1"
                    + content[match.end() :],
                )
            )
    return out


def statement_deletion(text: str) -> list[str]:
    """Operator 4 — delete one non-trivial added line (not blank, not a comment)."""
    sections = parse_diff(text)
    out: list[str] = []
    for si, hi, li, content in _plus_lines(sections):
        stripped = content.strip()
        if not stripped or stripped.startswith("#"):
            continue
        copy = [section.copy() for section in sections]
        body = copy[si].hunks[hi].body
        del body[li]
        # A "\\ No newline at end of file" marker belongs to the line it followed;
        # leaving it behind would render a hunk git refuses.
        if li < len(body) and body[li].startswith("\\"):
            del body[li]
        out.append(render_diff(_prune(copy)))
    return out


@dataclass(frozen=True)
class Operator:
    """One generator, with the priority the deterministic sort keys on."""

    name: str
    priority: int
    mutate: Callable[[str], list[str]]


OPERATORS: tuple[Operator, ...] = (
    Operator("hunk_drop", 1, hunk_drop),
    Operator("condition_flip", 2, condition_flip),
    Operator("boundary_shift", 3, boundary_shift),
    Operator("statement_deletion", 4, statement_deletion),
)
"""The single registry: the spec's priority order, and the one place a name lives."""

OPERATOR_NAMES: tuple[str, ...] = tuple(op.name for op in OPERATORS)
OPERATOR_ORDER = ">".join(OPERATOR_NAMES)
SORT_KEY = "(operator_priority, sha256)"


@dataclass(frozen=True)
class Mutant:
    """One candidate: its operator, the sort priority, and the complete diff text."""

    operator: str
    priority: int
    text: str
    sha256: str

    @property
    def name(self) -> str:
        """Stable stem, matching ``runner.Candidate.name`` for ``kind='mutant'``."""
        return f"{MUTANT_KIND}-{self.sha256[:12]}"


@dataclass
class Candidates:
    mutants: list[Mutant]
    per_operator: dict[str, int]
    """How many candidates each operator generated, before the cross-operator dedup."""


def generate(source: str) -> Candidates:
    """Every candidate the operators can make of ``source``, deduplicated by text.

    [VANTA-WHY] Deduplicated across operators, because two operators can produce
    byte-identical diffs (an integer shift and a ``len()`` shift can coincide) and
    the same patch graded twice is one experiment counted twice — it would inflate
    ``graded`` and, worse, could put the *same* patch in the pool twice. The
    per-operator ``generated`` counts stay raw, so the record still shows what each
    operator produced.
    """
    per_operator: dict[str, int] = {}
    seen: set[str] = set()
    mutants: list[Mutant] = []
    for op in OPERATORS:
        texts = op.mutate(source)
        per_operator[op.name] = len(texts)
        for text in texts:
            digest = sha256_text(text)
            if digest in seen:
                continue
            seen.add(digest)
            mutants.append(Mutant(op.name, op.priority, text, digest))
    return Candidates(mutants=mutants, per_operator=per_operator)


# ---------------------------------------------------------------------------
# selection — the spec's rule, with the preflight and the cap made injectable
# ---------------------------------------------------------------------------
@dataclass
class Preflight:
    """Did the mutant apply, and does its Python parse?"""

    applied: bool
    compiled: bool
    detail: str = ""


@dataclass
class Tally:
    """Every count the record keeps, per bug (Nathan's grading-budget answer)."""

    per_operator: dict[str, int] = field(default_factory=dict)
    did_not_apply: int = 0
    did_not_compile: int = 0
    graded: int = 0
    equivalent: int = 0
    inconclusive: int = 0
    """Graded mutants whose grading was unreadable (item B4): the patch applied
    but its hidden run wrote no pytest report, so it is neither a definite bad
    mutant nor an equivalent one. Dropped from the pool and counted here, so the
    exclusion is visible rather than silent."""
    selected: int = 0
    cap_hit: bool = False


@dataclass
class Selected:
    entries: list[tuple[Mutant, oracle_mod.OracleResult]]
    tally: Tally


def select(
    *,
    candidates: Candidates,
    preflight_fn: Callable[[Mutant], Preflight],
    grade_fn: Callable[[Mutant], oracle_mod.OracleResult],
    max_mutants: int = MAX_MUTANTS,
    grade_cap: int = GRADE_CAP,
    preflight_cap: int = PREFLIGHT_CAP,
) -> Selected:
    """The spec's rule: ``(priority, sha256)`` order, preflight, grade, keep ≤3 bad.

    Both the preflight and the grader are injected so the whole rule is executable
    without a VM (a stubbed preflight shows a mutant that does not compile never
    reaches the grader, and that an equivalent mutant is dropped rather than kept).

    [VANTA-WHY] Only a **definite bad** mutant enters the pool (item B4). A
    mutant whose patch applied but whose grading was *unreadable* — the hidden run
    wrote no pytest report at all — used to be appended here as a known-bad patch,
    which would score every review row against a patch the oracle never actually
    judged. That state is ``inconclusive``: it is dropped and counted separately,
    exactly as an equivalent mutant is, because neither is a patch a reviewer can
    be credited for finding.

    [VANTA-WHY] C10 — two caps, not one. ``grade_cap`` bounds the *gradings*, and
    ``preflight_cap`` bounds the *preflights*, because the two are not the same
    count: a mutant that fails to apply or to ``py_compile`` is discarded before it
    can be graded, and that discard still booted a Shuru VM. Without the second cap
    a fix whose candidates mostly fail to apply preflights every candidate the
    operators produced. Either cap stopping the loop is recorded as ``cap_hit``, so
    a starved pool is visible rather than silent.
    """
    tally = Tally(per_operator=dict(candidates.per_operator))
    ordered = sorted(candidates.mutants, key=lambda mutant: (mutant.priority, mutant.sha256))
    kept: list[tuple[Mutant, oracle_mod.OracleResult]] = []
    preflights = 0
    for mutant in ordered:
        if tally.graded >= grade_cap or preflights >= preflight_cap:
            tally.cap_hit = True
            break
        preflights += 1
        outcome = preflight_fn(mutant)
        if not outcome.applied:
            tally.did_not_apply += 1
            continue
        if not outcome.compiled:
            tally.did_not_compile += 1
            continue
        tally.graded += 1
        result = grade_fn(mutant)
        if result.patch_good:
            # Still a real fix: dropped as equivalent, never assumed bad.
            tally.equivalent += 1
            continue
        if result.label_input == "inconclusive":
            # Item B4: applied but unreadable — not a definite bad mutant, so it
            # never enters the pool. Counted so the exclusion is not silent.
            tally.inconclusive += 1
            continue
        kept.append((mutant, result))
        if len(kept) >= max_mutants:
            break
    tally.selected = len(kept)
    return Selected(entries=kept, tally=tally)


# ---------------------------------------------------------------------------
# the preflight — Shuru only, apply then py_compile
# ---------------------------------------------------------------------------
def _changed_py_files(text: str) -> list[str]:
    return [path for path in tree_mod.parse_patch(text).paths.all if path.endswith(".py")]


def _interpreter(test_cmd: Sequence[str]) -> str:
    return str(test_cmd[0]) if test_cmd else "python"


def preflight(
    *,
    prep: BugPrep,
    batch_dir: Path,
    checkpoint: str,
    mutant: Mutant,
    patch_path: Path,
    disk_mb: int,
    apply_timeout_secs: int = oracle_mod.APPLY_TIMEOUT_SECS,
    compile_timeout_secs: int = POOL_COMPILE_TIMEOUT_SECS,
) -> Preflight:
    """Apply the mutant to a fresh ``oracle_tree`` copy, then compile its Python.

    The first two steps mirror ``oracle.grade``'s own opening (``copy_tree`` then
    ``git apply`` with ``drop_test_hunks=True``) so a mutant this accepts is one
    ``grade`` will apply too, and the third is the gate that keeps a syntax error
    out of the pool. It runs inside the checkpoint, so ``py_compile`` uses the
    project's own interpreter — the one whose version decides what parses.
    """
    work_rel = f"{oracle_mod.ORACLE_WORK_DIR}/{prep.slug}__pre-{mutant.sha256[:12]}"
    work = batch_dir / work_rel
    tree_mod.copy_tree(prep.root / "oracle_tree", work)
    # [VANTA-WHY] C3 — this tree is *scratch*: ``Preflight`` reports what happened to
    # a patch, never where, and the graded tree belongs to ``grade``'s own label
    # (``cand-…``), so nothing downstream reads this copy after the answer is
    # computed. A mutant that is refused must therefore leave no tree: without the
    # ``finally`` the ``applied=False`` early return kept one, so a mutant that
    # failed to apply — the discarded majority — cost a whole project tree each, and
    # ``oracle_work`` grew by that on every generation. The ``finally`` sheds it on
    # every path, the successful one and the raising one included, so the preflight's
    # footprint is bounded by one tree while it runs rather than by the bug's mutants.
    try:
        applied, detail, _ = suite_mod.git_apply_patch(
            checkpoint=checkpoint,
            batch_dir=batch_dir,
            tree_rel=work_rel,
            patch_path=patch_path,
            disk_mb=disk_mb,
            drop_test_hunks=True,
            memory_mb=prep.vm_memory_mb,
            timeout_secs=apply_timeout_secs,
        )
        if not applied:
            return Preflight(applied=False, compiled=False, detail=detail)

        files = _changed_py_files(mutant.text)
        if not files:
            return Preflight(applied=True, compiled=True, detail="no python file changed")
        result = shuru_mod.run(
            checkpoint=checkpoint,
            host_root=batch_dir,
            command=[_interpreter(prep.test_cmd), "-m", "py_compile", *files],
            disk_mb=disk_mb,
            memory_mb=prep.vm_memory_mb,
            guest_cwd=f"/workspace/{work_rel}",
            timeout_secs=compile_timeout_secs,
        )
        compiled = result.exit_code == 0
        detail = "" if compiled else (result.stdout + result.stderr).strip()[-2000:]
        return Preflight(applied=True, compiled=compiled, detail=detail)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# the artifact — header + JSON body, hash-locked
# ---------------------------------------------------------------------------
_REQUIRED_HEADER_KEYS = (
    "schema",
    "bug",
    "bug-patch-sha256",
    "operators",
    "sort-key",
    "max-mutants",
    "grade-cap",
    "generated",
    "did-not-apply",
    "did-not-compile",
    "graded",
    "equivalent",
    "selected",
    "cap-hit",
    "content-sha256",
)


@dataclass
class PoolEntry:
    """One mutant, as the locked file records it (kind, operator, hash, label)."""

    kind: str
    operator: str
    patch_sha256: str
    source: str
    patch_path: str
    oracle: str


@dataclass
class MutantPool:
    """A locked mutants file, read back and verified."""

    path: Path
    bug: str
    bug_patch_sha256: str
    operators: str
    max_mutants: int
    grade_cap: int
    tally: Tally
    entries: list[PoolEntry]
    content_sha256: str


def _canonical_body(body: str) -> str:
    """The one form the JSON body takes — in the file and in the hash of it.

    The same rule ``issue_source._canonical`` applies, for the same reason: the
    writer hashes the bytes it writes and the reader hashes the bytes it reads, so
    an artifact that was edited — even by one byte — cannot verify. The rule lives
    in :func:`locked.canonical`.
    """
    return locked.canonical(body)


def _header_block(text: str) -> tuple[dict[str, str], str]:
    """``(header, the rest)`` — the ``- key: value`` lines, up to the first blank line."""
    return locked.header_block(text)


def _section(text: str, marker: str) -> tuple[str, str]:
    """``(before, after)`` the first line equal to ``marker``, that line dropped."""
    return locked.section(text, marker, error=MutantError, what="pool file")


def render_pool(
    *,
    prep: BugPrep,
    source_sha256: str,
    tally: Tally,
    entries: Sequence[PoolEntry],
    max_mutants: int = MAX_MUTANTS,
    grade_cap: int = GRADE_CAP,
) -> str:
    """The artifact's exact bytes: the lock header, the marker, the JSON body."""
    body = json.dumps(
        {"entries": [entry.__dict__ for entry in entries]},
        indent=2,
        ensure_ascii=False,
    )
    canonical = _canonical_body(body)
    generated = " ".join(
        f"{name}={tally.per_operator.get(name, 0)}" for name in OPERATOR_NAMES
    )
    header = [
        f"# {prep.key} — the reference-fix mutant pool",
        "",
        f"- schema: {SCHEMA}",
        f"- bug: {prep.key}",
        f"- bug-patch-sha256: {source_sha256}",
        f"- operators: {OPERATOR_ORDER}",
        f"- sort-key: {SORT_KEY}",
        f"- max-mutants: {max_mutants}",
        f"- grade-cap: {grade_cap}",
        f"- generated: {generated}",
        f"- did-not-apply: {tally.did_not_apply}",
        f"- did-not-compile: {tally.did_not_compile}",
        f"- graded: {tally.graded}",
        f"- equivalent: {tally.equivalent}",
        f"- inconclusive: {tally.inconclusive}",
        f"- selected: {tally.selected}",
        f"- cap-hit: {'true' if tally.cap_hit else 'false'}",
        f"- content-sha256: {sha256_text(canonical)}",
    ]
    return "\n".join(header) + "\n\n" + MUTANTS_MARKER + "\n\n" + canonical + "\n"


def write_pool(
    path: Path,
    *,
    prep: BugPrep,
    source_sha256: str,
    tally: Tally,
    entries: Sequence[PoolEntry],
    max_mutants: int = MAX_MUTANTS,
    grade_cap: int = GRADE_CAP,
) -> Path:
    """Render, verify with the reader, then write — ``newline=""`` so the bytes hash.

    Verifying before writing mirrors ``issue_source.write_frozen``: an artifact its
    own reader refuses is impossible to create rather than merely detectable, and
    no file is left behind when the two disagree. The sequence is
    :func:`locked.write`.
    """
    target = Path(path)
    text = render_pool(
        prep=prep,
        source_sha256=source_sha256,
        tally=tally,
        entries=entries,
        max_mutants=max_mutants,
        grade_cap=grade_cap,
    )
    return locked.write(target, text, verify=parse_pool)


def load_pool(path: Path) -> MutantPool:
    """Read a locked pool file from disk, and verify it (:func:`parse_pool`)."""
    return locked.load(path, parse=parse_pool, error=MutantError, what="mutant pool")


def parse_pool(text: str, *, path: Path) -> MutantPool:
    """Verify and read the artifact's text — the one parser, for writer and reader."""
    target = Path(path)
    header, body_text = _header_block(text)
    locked.require_keys(
        header,
        _REQUIRED_HEADER_KEYS,
        path=target,
        error=MutantError,
        what="pool file",
        why="say what it came from",
    )
    _, after = _section(body_text, MUTANTS_MARKER)
    body = _canonical_body(after)
    actual = sha256_text(body)
    if actual != header["content-sha256"]:
        raise MutantError(
            f"{target}: content-sha256 in the header is {header['content-sha256']}, but "
            f"the body hashes to {actual} — the file was edited after it was frozen"
        )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise MutantError(f"{target}: the body is not JSON: {exc}") from exc
    entries = [
        PoolEntry(
            kind=str(raw.get("kind", "")),
            operator=str(raw.get("operator", "")),
            patch_sha256=str(raw.get("patch_sha256", "")),
            source=str(raw.get("source", "")),
            patch_path=str(raw.get("patch_path", "")),
            oracle=str(raw.get("oracle", "")),
        )
        for raw in payload.get("entries", [])
    ]
    tally = Tally(
        per_operator=_parse_generated(header["generated"]),
        did_not_apply=_parse_int(header, "did-not-apply", target),
        did_not_compile=_parse_int(header, "did-not-compile", target),
        graded=_parse_int(header, "graded", target),
        equivalent=_parse_int(header, "equivalent", target),
        inconclusive=_parse_int_default(header, "inconclusive", target, 0),
        selected=_parse_int(header, "selected", target),
        cap_hit=header["cap-hit"].strip().lower() in ("true", "1", "yes"),
    )
    return MutantPool(
        path=target,
        bug=header["bug"],
        bug_patch_sha256=header["bug-patch-sha256"],
        operators=header["operators"],
        max_mutants=_parse_int(header, "max-mutants", target),
        grade_cap=_parse_int(header, "grade-cap", target),
        tally=tally,
        entries=entries,
        content_sha256=header["content-sha256"],
    )


def _parse_int(header: dict[str, str], key: str, path: Path) -> int:
    try:
        return int(header[key])
    except ValueError as exc:
        raise MutantError(f"{path}: header field {key!r} is {header[key]!r}, not an integer") from exc


def _parse_int_default(header: dict[str, str], key: str, path: Path, default: int) -> int:
    """``_parse_int`` for a key that older files predate — absent reads as ``default``.

    [VANTA-WHY] ``inconclusive`` was added after the first pools were frozen, and a
    pool is written once and hash-locked. Making it a required header key would
    make every already-frozen pool unreadable, so it stays out of
    ``_REQUIRED_HEADER_KEYS`` and an absent (or blank) value reads as ``0`` — the
    third state simply did not exist when that pool was made. A present but
    non-integer value is still an error.
    """
    raw = header.get(key, "")
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise MutantError(f"{path}: header field {key!r} is {raw!r}, not an integer") from exc


def _parse_generated(value: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in value.split():
        key, _, count = part.partition("=")
        try:
            out[key] = int(count)
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# the whole of item 5, for one bug
# ---------------------------------------------------------------------------
def pool_path(batch_dir: Path, prep: BugPrep) -> Path:
    return batch_dir / MUTANTS_DIRNAME / f"{prep.slug}{MUTANTS_SUFFIX}"


@dataclass
class PoolResult:
    path: Path
    source_sha256: str
    tally: Tally
    entries: list[PoolEntry]


def _write_candidate(
    batch_dir: Path,
    prep: BugPrep,
    mutant: Mutant,
    cache: dict[str, runner_mod.Candidate],
) -> runner_mod.Candidate:
    """Write (once) the mutant's patch file, reusing ``runner``'s candidate shape."""
    existing = cache.get(mutant.sha256)
    if existing is not None:
        return existing
    candidate = runner_mod.write_candidate_patch(
        batch_dir,
        prep,
        MUTANT_KIND,
        mutant.text,
        source=f"{MUTANT_KIND}:{mutant.operator}",
    )
    cache[mutant.sha256] = candidate
    return candidate


def run_mutants(
    *,
    prep: BugPrep,
    batch_dir: Path,
    checkpoint: str,
    disk_mb: int,
    max_mutants: int = MAX_MUTANTS,
    grade_cap: int = GRADE_CAP,
    preflight_cap: int = PREFLIGHT_CAP,
    preflight_fn: Optional[Callable[[Mutant], Preflight]] = None,
    grade_fn: Optional[Callable[[Mutant], oracle_mod.OracleResult]] = None,
) -> PoolResult:
    """Generate, preflight, grade and lock ``pool/<slug>.mutants.json`` for one bug.

    The real preflight and grader boot Shuru; both are injectable so the selection
    rule can be exercised with no VM. ``grade_candidate`` caches a grading keyed by
    the mutant's own patch hash, so a re-run of this does not grade a patch the
    batch already graded.
    """
    source = runner_mod.test_hunk_free(
        prep.bug.bug_patch_txt, what=f"{prep.key} reference fix"
    )
    source_sha256 = sha256_text(source)
    candidates = generate(source)
    written: dict[str, runner_mod.Candidate] = {}

    def default_preflight(mutant: Mutant) -> Preflight:
        candidate = _write_candidate(batch_dir, prep, mutant, written)
        return preflight(
            prep=prep,
            batch_dir=batch_dir,
            checkpoint=checkpoint,
            mutant=mutant,
            patch_path=candidate.patch_path,
            disk_mb=disk_mb,
        )

    def default_grade(mutant: Mutant) -> oracle_mod.OracleResult:
        candidate = _write_candidate(batch_dir, prep, mutant, written)
        runner_mod.grade_candidate(
            batch_dir=batch_dir,
            prep=prep,
            candidate=candidate,
            checkpoint=checkpoint,
            disk_mb=disk_mb,
        )
        return candidate.oracle

    selected = select(
        candidates=candidates,
        preflight_fn=preflight_fn or default_preflight,
        grade_fn=grade_fn or default_grade,
        max_mutants=max_mutants,
        grade_cap=grade_cap,
        preflight_cap=preflight_cap,
    )
    entries: list[PoolEntry] = []
    for mutant, result in selected.entries:
        # The path `write_candidate_patch` chose, so the record points at the bytes
        # that were graded. Falls back to that same destination for an injected
        # preflight/grader, which never wrote a file.
        candidate = written.get(mutant.sha256)
        patch_path = (
            candidate.patch_path
            if candidate is not None
            else batch_dir / "candidates" / prep.slug / f"{mutant.name}.patch"
        )
        entries.append(
            PoolEntry(
                kind=MUTANT_KIND,
                operator=mutant.operator,
                patch_sha256=mutant.sha256,
                source=f"{MUTANT_KIND}:{mutant.operator}",
                patch_path=str(patch_path),
                oracle=result.label_input,
            )
        )
    path = write_pool(
        pool_path(batch_dir, prep),
        prep=prep,
        source_sha256=source_sha256,
        tally=selected.tally,
        entries=entries,
        max_mutants=max_mutants,
        grade_cap=grade_cap,
    )
    return PoolResult(
        path=path, source_sha256=source_sha256, tally=selected.tally, entries=entries
    )


def render_pool_line(prep: BugPrep, result: PoolResult) -> str:
    """One line per bug, for the batch's console log."""
    tally = result.tally
    generated = " ".join(
        f"{name}={tally.per_operator.get(name, 0)}" for name in OPERATOR_NAMES
    )
    return (
        f"{prep.key}: generated {generated} · did-not-apply {tally.did_not_apply} · "
        f"did-not-compile {tally.did_not_compile} · graded {tally.graded} · "
        f"equivalent {tally.equivalent} · inconclusive {tally.inconclusive} · "
        f"selected {tally.selected} · "
        f"cap-hit {'true' if tally.cap_hit else 'false'} · {result.path}"
    )


def plan_lines(prep: BugPrep, *, max_mutants: int = MAX_MUTANTS) -> list[str]:
    """What the pool would grade, from the reference fix alone (no VM, no checkpoint)."""
    try:
        source = runner_mod.test_hunk_free(
            prep.bug.bug_patch_txt, what=f"{prep.key} reference fix"
        )
    except runner_mod.RunnerError as exc:
        return [f"{prep.key}: no mutant source ({exc})"]
    candidates = generate(source)
    generated = " ".join(
        f"{name}={candidates.per_operator.get(name, 0)}" for name in OPERATOR_NAMES
    )
    ordered = sorted(candidates.mutants, key=lambda m: (m.priority, m.sha256))
    lines = [
        f"{prep.key}: {len(ordered)} candidates (deduplicated), {generated}, "
        f"up to {max_mutants} kept, grade-cap {GRADE_CAP}",
    ]
    for mutant in ordered[:max_mutants]:
        lines.append(f"  {mutant.priority} {mutant.operator} {mutant.name}")
    return lines
