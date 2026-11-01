"""Item 6 — the assembled review pool: every wrong patch a bug produced (§5.4).

A review row needs a *known-bad* patch to review, and item 5 supplies one kind of
them: broken versions of the developer fix, made mechanically. This is the other
kind — the patches the Creator itself wrote and the oracle graded bad. They are
worth more than a mutant because the reviewer's complaint can be about the edit's
*reasoning*, and they exist for free in a batch that has already run.

[VANTA-WHY] Five rules here are the kind that fail silently, and each has a cheap
failure mode that looks like a result rather than a bug:

* **Every attempt, not just the last.** ``final.patch`` is the last attempt's
  patch; a Creator that failed three times before is a bug whose first two wrong
  answers are the interesting ones. The per-attempt snapshots the driver already
  writes (``attempt_N_patched_tree``, ``compact.py``) are the source, diffed
  against the same sanitized base compaction diffs against.
* **A compacted attempt is still the attempt.** Most sessions are compacted
  (``compact.py``'s rule), so reading only directories would discard most of the
  corpus. A compacted tree is stored as a diff pinned by its own
  ``diff_sha256``, so it is *reconstructed* — read, hash-checked, used — and never
  guessed. Only a failed hash check or a missing pair is ``missing_snapshot``, and
  it is recorded per attempt with the reason.
* **Test paths go, exactly as the grader drops them.** ``runner.test_hunk_free``
  drops what ``suite.drop_test_file_hunks`` drops, which is the tree the oracle
  actually applies a patch to. A candidate that smuggled a change into the tests
  would be graded against a tree the oracle never builds.
* **A patch that is not text is not a candidate.** A binary file change (git's
  ``Binary files … differ`` or a ``GIT binary patch`` section) or bytes that are
  not UTF-8 cannot be read, diffed or applied as a reviewable diff, so it is
  counted ``not_reviewable`` with its reason and dropped *before* the test-hunk
  stripper rather than being silently corrupted by a lossy decode. The check lives
  here, on the assembled text, so ``runner.git_diff_no_index`` — which the grading
  path shares — keeps producing its bytes unchanged.
* **Normalisation is a pure function of the diff, and it preserves application.**
  Hunks are sorted by *position*, never by hash, because the normalised diff is
  the thing that gets applied. Grading the normalised text is what makes
  "deduplicate by normalised hash" and "grade each candidate" one experiment.
* **Selection is a pure function of the diff and its provenance.** Order by
  ``(source priority, sha256 of the normalised diff)``; no randomness, no
  filesystem order, no counter. A bug with no bad patch keeps what it found and
  the pool says so — **recording, never padding**.

The artifact is ``<out>/pool/<slug>.json``: the lock form ``locked.py`` owns (a
title, a ``- key: value`` header carrying the body's hash, a marker, then JSON),
referring to item 5's ``.mutants.json`` **by its own ``content-sha256``** rather
than copying its entries, so neither file edits the other and a reader can refuse
a mutants file that moved. The reader (``load_pool``) verifies both files and
returns the union — reference fix, no-op, mutants, failed Creator patches and
oracle-good Creator patches — each entry carrying ``patch_source``,
``source_model``, ``source_run_id`` and its oracle label. A changed pool or a
changed mutants file refuses (§Item 6: "reviews refuse a changed pool").

Sources are **pluggable and tagged**: ``--batch TAG=DIR``, where ``TAG`` is one of
:data:`SOURCE_TAGS` (``repair``, then ``weak``, and ``leaderboard`` when M4's
patches arrive). The tag is what the grading order's first key is and what every
harvested entry records as its ``patch_source``, so a source can never be
ambiguous and a new one is a new row in one table.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import arms as arms_mod
from . import compact as compact_mod
from . import locked
from . import manifest as manifest_mod
from . import mutants as mutants_mod
from . import oracle as oracle_mod
from . import runner as runner_mod
from .prepare import BugPrep
from .split import sha256_bytes, sha256_text

POOL_DIRNAME = "pool"
POOL_SUFFIX = ".json"
POOL_MARKER = "<!-- pool -->"

SCHEMA = 1

GRADE_CAP = 24
"""The most gradings one bug may spend looking for its two bad patches.

Item 5's 12 is the *mutants'* budget and is separate on purpose: the two are
different generator configs, frozen separately, and one number standing for both
would make a change to either look like a change to the other.
"""

MAX_BAD = 2
"""Failed Creator patches kept per bug (``failed_model``)."""
MAX_GOOD = 2
"""Oracle-good Creator patches kept per bug (``model_good``).

Kept because the good-patch side needs *model-written* fixes, not just the
developer's: a review of a good candidate that a model wrote is a different
measurement from a review of the reference fix.
"""

DEV_FIX = runner_mod.DEV_FIX
NOOP = runner_mod.NOOP
MUTANT = mutants_mod.MUTANT_KIND
FAILED_MODEL = "failed_model"
MODEL_GOOD = "model_good"

REFERENCE_SOURCE = "reference"
NOOP_SOURCE = "noop"
NOOP_MODEL = "harness"

SORT_KEY = "(source_priority, sha256)"

MISSING_SNAPSHOT = "missing_snapshot"
"""The reason bucket for an attempt whose patch could not be reconstructed."""

COMMENT_PREFIXES: tuple[str, ...] = ("#", "//", "/*", "*", "<!--")
"""Language-neutral comment openers, so the same rule survives M4's Java.

The set is deliberately openers only: a Java ``/**`` opens with ``/*`` and a
Python docstring's interior lines are not comments at all — a rule that tried to
recognise *closers* would need per-language knowledge this module does not have.
"""


class PoolError(RuntimeError):
    """The pool could not be assembled, or a written pool file failed its lock."""


# ---------------------------------------------------------------------------
# sources — a fixed vocabulary, defined once
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceTag:
    """One harvestable Creator source, and where it sorts in the grading order."""

    name: str
    priority: int


SOURCE_TAGS: tuple[SourceTag, ...] = (
    SourceTag("repair", 1),
    SourceTag("weak", 2),
    SourceTag("leaderboard", 3),
)
"""The registry: the tag vocabulary, the priority order, and the one place a tag lives.

M4's leaderboard patches are a new *source behind this table*, not a rewrite —
they arrive from outside the repair batches and sort last, so a bug that has
matrix attempts never spends its budget on a leaderboard patch first.
"""

TAG_ORDER = ">".join(tag.name for tag in SOURCE_TAGS)


def tag_for(name: str) -> SourceTag:
    """One tag by name; an unknown tag is refused rather than defaulted."""
    for tag in SOURCE_TAGS:
        if tag.name == name:
            return tag
    raise PoolError(
        f"unknown source tag {name!r}; known tags in priority order: {TAG_ORDER}"
    )


def tag_names() -> tuple[str, ...]:
    return tuple(tag.name for tag in SOURCE_TAGS)


def parse_batch_argument(value: str) -> tuple[str, Path]:
    """``TAG=DIR`` → ``(tag, path)``; a value with no tag is refused.

    Refused rather than defaulted: the tag is the grading order's first key and
    the entry's recorded ``patch_source``, so an untagged batch would be a source
    nobody could name, and "which source did this patch come from" would be
    answered by the order the flags happened to be typed in.
    """
    tag, separator, rest = value.partition("=")
    if not separator or not rest:
        raise PoolError(
            f"`--batch {value}` has no source tag. Write `--batch TAG=DIR` with TAG "
            f"one of {TAG_ORDER} (e.g. `--batch repair={value}`), so every patch "
            "records which source it came from."
        )
    return tag_for(tag).name, Path(rest)


def priority_of(tags: Sequence[str]) -> list[str]:
    """The distinct tags present, in the registry's priority order."""
    wanted = {tag.name: tag.priority for tag in SOURCE_TAGS}
    present = {tag for tag in tags if tag in wanted}
    return [tag.name for tag in SOURCE_TAGS if tag.name in present]


# ---------------------------------------------------------------------------
# normalisation — a pure function of the diff
# ---------------------------------------------------------------------------
_DIFF_GIT_RE = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+)$")
_WHITESPACE_RE = re.compile(r"\s+")


def _is_file_section(section: mutants_mod.FileSection) -> bool:
    return bool(section.header) and section.header[0].startswith("diff --git ")


def _section_path(section: mutants_mod.FileSection) -> str:
    """The ``b/`` path of a section, for the canonical file order."""
    if not section.header:
        return ""
    match = _DIFF_GIT_RE.match(section.header[0])
    return match.group("b") if match else section.header[0]


def _changed(body: Sequence[str], sign: str) -> list[str]:
    """The ``+``/``-`` content lines of a hunk body, without their sign.

    The three-character header lines are excluded the way item 5's operators
    exclude them: an added line whose own content is ``++`` renders as ``+++`` and
    would otherwise be read as a file header.
    """
    return [line[1:] for line in body if line.startswith(sign) and not line.startswith(sign * 3)]


def _is_whitespace_only(hunk: mutants_mod.Hunk) -> bool:
    """Does every removed line have an added counterpart that differs only in whitespace?"""
    removed = _changed(hunk.body, "-")
    added = _changed(hunk.body, "+")
    if not removed and not added:
        return False
    strip = lambda lines: sorted(_WHITESPACE_RE.sub("", line) for line in lines)  # noqa: E731
    return strip(removed) == strip(added)


def _is_comment_line(content: str) -> bool:
    return content.strip().startswith(COMMENT_PREFIXES)


def _is_comment_only(hunk: mutants_mod.Hunk) -> bool:
    """Is every non-blank changed line in the hunk a comment?"""
    changed = [line for line in _changed(hunk.body, "+") + _changed(hunk.body, "-") if line.strip()]
    if not changed:
        return False
    return all(_is_comment_line(line) for line in changed)


def normalize(text: str) -> str:
    """The canonical form of a diff: cosmetic hunks dropped, positions sorted.

    A pure function of the text, so the same patch normalises the same way in
    every run and on every machine. Sorting is **by position**, never by hash: the
    normalised diff still has to apply, and a hash order would scatter a file's
    hunks into an order ``git apply`` refuses.
    """
    sections = mutants_mod.parse_diff(text)
    preambles = [section for section in sections if not _is_file_section(section)]
    files: list[mutants_mod.FileSection] = []
    for section in sections:
        if not _is_file_section(section):
            continue
        kept = [
            hunk
            for hunk in section.hunks
            if not _is_whitespace_only(hunk) and not _is_comment_only(hunk)
        ]
        if not kept:
            # A section left with no hunks is not a patch; this is item 5's prune.
            continue
        files.append(mutants_mod.FileSection(list(section.header), kept))
    files.sort(key=_section_path)
    for section in files:
        section.hunks.sort(key=lambda hunk: (hunk.old_start, hunk.new_start, hunk.body))
    return mutants_mod.render_diff(preambles + files)


def normalized_sha256(text: str) -> str:
    return sha256_text(normalize(text))


def is_noop(text: str) -> bool:
    """Does the diff normalise to nothing at all — no file section left?"""
    return not normalize(text).strip()


# ---------------------------------------------------------------------------
# harvesting — every attempt of every finished repair run
# ---------------------------------------------------------------------------
SNAPSHOT = "snapshot"
COMPACTED = "compacted"
FINAL_PATCH = "final_patch"
STORED = "stored"

_ATTEMPT_RE = re.compile(
    r"^attempt_(?P<n>\d+)_(?P<kind>patched|graded)_tree"
    r"(?P<suffix>\.diff|\.hashes\.json)?$"
)


@dataclass(frozen=True)
class HarvestedAttempt:
    """One Creator attempt, already test-hunk-free, with where it came from."""

    slug: str
    attempt: int
    patch_text: str
    origin: str
    tag: str
    source_model: str
    source_run_id: str


@dataclass(frozen=True)
class MissingAttempt:
    """An attempt whose patch could not be reconstructed — recorded, never guessed."""

    slug: str
    attempt: int
    reason: str


@dataclass(frozen=True)
class NotReviewable:
    """An attempt that WAS reconstructed but is not a reviewable candidate.

    Two shapes so far: a patch that is nothing but test hunks (there is no code to
    review) and a patch that is binary or not valid UTF-8 (it cannot be read as a
    diff at all). Both carry the reason, so the count is not a bare number.
    """

    slug: str
    attempt: int
    reason: str


@dataclass
class Harvest:
    """What one tagged batch contributed, and what it cost in missing attempts."""

    tag: str
    priority: int
    batch_dir: Path
    attempts: list[HarvestedAttempt] = field(default_factory=list)
    missing: list[MissingAttempt] = field(default_factory=list)
    not_reviewable: list[NotReviewable] = field(default_factory=list)


def _highest_attempt(debug_dir: Path) -> int:
    """The highest attempt number the session has *any* artifact for."""
    if not debug_dir.is_dir():
        return 0
    highest = 0
    try:
        names = [path.name for path in debug_dir.iterdir()]
    except OSError:
        return 0
    for name in names:
        match = _ATTEMPT_RE.match(name)
        if match:
            highest = max(highest, int(match.group("n")))
    return highest


def _attempt_total(record: Mapping[str, Any], debug_dir: Path) -> int:
    """How many attempts the run *had*, not how many left an artifact.

    ``attempts_run`` is the run's own count and is the authority: an attempt whose
    snapshot is gone is still an attempt, and counting only the artifacts would
    silently shrink the set the pool reports on.
    """
    try:
        run = int(record.get("attempts_run"))
    except (TypeError, ValueError):
        run = 0
    return max(run, _highest_attempt(debug_dir))


def _read_compacted(diff_path: Path, hashes_path: Path) -> tuple[Optional[str], Optional[str]]:
    """``(diff text, reason)`` — the compacted diff, used only if its own hash verifies.

    Read and verified the way :func:`compact.compact_session` wrote it: UTF-8 with
    ``errors="surrogateescape"`` and a ``sha256`` over those same bytes. A tracked
    non-UTF-8 file (a PDF, a ``.ppm``) survives the round trip as surrogate code
    points; a strict decode here would reject a diff the writer legitimately stored.
    """
    try:
        payload = json.loads(hashes_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"the compacted .hashes.json could not be read: {exc}"
    pinned = str(payload.get("diff_sha256") or "")
    try:
        with diff_path.open("r", encoding="utf-8", errors="surrogateescape", newline="") as handle:
            text = handle.read()
    except OSError as exc:
        return None, f"the compacted .diff could not be read: {exc}"
    actual = sha256_bytes(compact_mod.diff_bytes(text))
    if actual != pinned:
        return None, (
            f"the compacted .diff hashes to {actual}, not the recorded "
            f"diff_sha256 {pinned or '(absent)'}"
        )
    return text, None


DELETED_FILE_MARKER = "deleted file mode"
"""The one line that marks a section as a deletion — text and binary alike.

``git diff`` prints it for a deleted text file *and* for a deleted binary one,
whose body is ``Binary files … differ`` with no ``+++ /dev/null`` line for a path
test to key on. Matching this line is what makes the filter total.
"""


def _covered_paths(debug_dir: Path, tree_name: str) -> Optional[set[str]]:
    """The files a snapshot tree *covers* — the manifest Vanta wrote beside it.

    ``harness::write_tree_snapshot`` is handed the driver's ``# FILE:``
    concatenation, so the tree holds **only the tracked source files**, and it
    records ``{paths, file_count}`` next to them. Diffing that partial tree against
    the full sanitized base reports every *other* base file — ``.git/``, ``LICENSE``,
    ``README.md`` — as deleted, and applying that diff to the grading tree stops at
    the first one (``error: invalid path '.git/HEAD'``): a real attempt read as an
    unobservable grade. ``None`` when the manifest is absent (a session from before
    it was recorded), which leaves :func:`drop_base_only_deletions` with no scope and
    so keeps every section — the old behaviour, since without a scope the spurious
    deletions cannot be told from the real ones.
    """
    manifest = debug_dir / f"{tree_name}_manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    paths = payload.get("paths")
    if not isinstance(paths, list):
        return None
    return {str(path) for path in paths}


def _section_path_of(line: str) -> Optional[str]:
    match = _DIFF_GIT_RE.match(line)
    return match.group("b") if match else None


def drop_base_only_deletions(diff_text: str, covered: Optional[set[str]]) -> str:
    """Drop the sections that delete a file the snapshot never covered.

    [VANTA-WHY] The test is on the **path**, not on "this is a deletion". A snapshot
    genuinely omits a tracked file when the attempt deleted it, and that deletion is
    a real part of the patch; what is never real is a deletion of a path the
    snapshot's manifest never listed, because those files were absent from the tree
    from the start rather than removed by the patch. ``covered is None`` keeps every
    section — the old behaviour — because without a scope nothing can be told apart.
    """
    if covered is None:
        return diff_text
    kept: list[str] = []
    path: Optional[str] = None
    skipping = False
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            path = _section_path_of(line)
            skipping = False
            kept.append(line)
            continue
        if not skipping and line.startswith(DELETED_FILE_MARKER):
            if path is not None and path not in covered:
                # The section's only kept line so far is its `diff --git` header.
                kept.pop()
                skipping = True
                continue
        if not skipping:
            kept.append(line)
    return "".join(kept)


def attempt_patch(
    *, debug_dir: Path, base_tree: Path, attempt: int
) -> tuple[Optional[str], Optional[str], str]:
    """``(patch text, reason, origin)`` for one attempt, or ``(None, reason, "")``.

    Three cases, in the order the design fixes them: the tree snapshot is still on
    disk (diff it against the sanitized base — exact); the compaction replaced it
    with a diff pinned by its own hash (read it, verify it, use it as a full
    snapshot); or it is gone. The third case is **never** guessed here — the
    caller may fall back to ``final.patch``, and only for the last attempt.

    Both surviving cases pass through :func:`drop_base_only_deletions` (#F bug 2):
    the tree is partial, so the raw diff carries a deletion for every base file the
    snapshot never held, and `git apply` refuses the whole patch at the first of
    them.
    """
    tree = debug_dir / f"attempt_{attempt}_patched_tree"
    if tree.is_dir():
        try:
            text = compact_mod.tree_diff(base_tree, tree)
        except compact_mod.CompactError as exc:
            return None, f"the snapshot tree could not be diffed: {exc}", ""
        return drop_base_only_deletions(text, _covered_paths(debug_dir, tree.name)), None, SNAPSHOT
    diff_path = debug_dir / f"{tree.name}.diff"
    hashes_path = debug_dir / f"{tree.name}.hashes.json"
    if diff_path.is_file() and hashes_path.is_file():
        text, reason = _read_compacted(diff_path, hashes_path)
        if text is None:
            return None, reason, ""
        # The manifest outlives the tree it describes (compaction removes only the
        # tree), so the covered set is read the same way in both cases.
        return drop_base_only_deletions(text, _covered_paths(debug_dir, tree.name)), None, COMPACTED
    if diff_path.is_file() or hashes_path.is_file():
        return None, "the compacted .diff and .hashes.json are not both present", ""
    return None, "no snapshot tree and no compacted .diff/.hashes.json", ""


def _final_patch_text(record: Mapping[str, Any]) -> tuple[Optional[str], str]:
    """``(text, reason)`` for the row's ``final.patch``, if it is readable."""
    raw = (record.get("artifacts") or {}).get("final_patch")
    if not raw:
        return None, "the row records no final.patch"
    path = Path(str(raw))
    if not path.is_file():
        return None, f"final.patch {path} is not on disk"
    try:
        return path.read_text(encoding="utf-8", errors="replace"), ""
    except OSError as exc:
        return None, f"final.patch could not be read: {exc}"


BINARY_SECTION_MARKER = "Binary files "
GIT_BINARY_MARKER = "GIT binary patch"
"""git's two ways of saying a section is not text.

``Binary files a/x and b/x differ`` is what ``--no-index``/``--binary``-off prints
for a tracked binary file; ``GIT binary patch`` is the literal/delta form printed
under ``--binary``. Either one means the section carries no reviewable text.
"""


def binary_patch_reason(text: str) -> Optional[str]:
    """Why ``text`` is not a reviewable patch, or ``None`` when it is.

    [VANTA-WHY] A patch that changes a binary file, or that arrived as bytes which
    are not UTF-8, cannot be read, diffed or applied as text. It is caught in the
    pool — on the *assembled* text — rather than in ``runner.git_diff_no_index``,
    which the grading path also calls and which must keep producing its bytes
    untouched. The two non-UTF-8 tells are the decoder's own: ``U+FFFD`` from a
    lossy ``errors="replace"`` read, and lone surrogates from the
    ``errors="surrogateescape"`` read the compacted diff uses.
    """
    for line in text.splitlines():
        if line.startswith(BINARY_SECTION_MARKER) and line.rstrip().endswith(" differ"):
            return "the patch changes a binary file"
        if line.startswith(GIT_BINARY_MARKER):
            return "the patch carries a git binary-patch section"
    if "\ufffd" in text or any(0xDC80 <= ord(character) <= 0xDCFF for character in text):
        return "the patch is not valid UTF-8"
    return None


def source_model_for(record: Mapping[str, Any], tag: str) -> str:
    """What the entry records as the model that wrote the patch.

    A named matrix arm (``b0``–``b3``) is the *experimental condition*, and that is
    what a repair row's arm column knows; every other source — the weak-Creator
    pass, and M4's leaderboard — is identified by the model the run recorded. It is
    never inferred from an endpoint URL.
    """
    if tag == "repair":
        return arms_mod.row_arm(record) or str(record.get("config") or "")
    model = record.get("model")
    return str(model) if model else (arms_mod.row_arm(record) or str(record.get("config") or ""))


def failed_keys_for(state: Mapping[str, str], bug: str) -> int:
    """How many of **this bug's** jobs failed in a batch's cursor (``state.json``).

    Per bug, not per batch: the header's ``failed-keys`` sits beside that bug's own
    ``missing-snapshot`` and ``not-reviewable`` counts and has to mean the same kind
    of thing — a fact about *this* bug. A batch-wide total written into every bug's
    pool would say the same number for a bug whose jobs all succeeded and one whose
    jobs all broke, and the number is only interesting when it is not zero.

    The bug is read from its position in the job key
    (:func:`manifest.job_key`: ``batch#bug#config#mode#seed[#candidate]``), because
    that is the field the key defines rather than something a prefix match would
    guess at — a batch id that merely *started* with the bug's key would match a
    prefix test and not this one.
    """
    count = 0
    for key, value in state.items():
        if value != manifest_mod.FAILED:
            continue
        parts = key.split("#")
        if len(parts) >= 2 and parts[1] == bug:
            count += 1
    return count


def harvest_batch(
    *,
    tag: str,
    batch_dir: Path,
    sessions_dir: Path,
    preps: Mapping[str, BugPrep],
    manifest: Optional[manifest_mod.Manifest] = None,
) -> Harvest:
    """Every Creator attempt of every repair row in one tagged batch.

    Only ``mode == "repair"`` rows: a review row's candidate is not a Creator
    attempt, and the review's own decisions are the thing being measured, not a
    source of patches.

    [VANTA-WHY] F3 — the rows are read through ``manifest.effective_records``, so a
    re-run's superseded session is not harvested alongside the row that replaced it.
    Harvesting both would offer the same bug's stale attempts as review candidates
    twice and inflate every count downstream of the pool.
    """
    entry = tag_for(tag)
    reader = manifest if manifest is not None else manifest_mod.Manifest(batch_dir)
    harvest = Harvest(tag=entry.name, priority=entry.priority, batch_dir=batch_dir)
    for record in manifest_mod.effective_records(reader.records()):
        if str(record.get("mode") or "") != arms_mod.REPAIR:
            continue
        session_id = record.get("session_id")
        if not session_id:
            continue
        key = str(record.get("bug") or "")
        prep = preps.get(key)
        if prep is None:
            # A row for a bug this batch carries no prep for cannot be given a
            # slug or a base to diff against; skipping it silently would lose an
            # attempt, so it is skipped *loudly* by the caller's plan listing.
            continue
        slug = prep.slug
        base_tree = prep.root / "vanta_clean"
        debug_dir = sessions_dir / str(session_id) / "debug"
        model = source_model_for(record, entry.name)
        total = _attempt_total(record, debug_dir)
        for attempt in range(1, total + 1):
            text, reason, origin = attempt_patch(
                debug_dir=debug_dir, base_tree=base_tree, attempt=attempt
            )
            if text is None and attempt == total:
                # Only the LAST attempt may fall back to final.patch, because
                # that is the attempt final.patch *is*; any other number would be
                # a patch attributed to a round it did not come from.
                text, final_reason = _final_patch_text(record)
                if text is not None:
                    origin = FINAL_PATCH
                else:
                    reason = f"{reason}; {final_reason}"
            if text is None:
                harvest.missing.append(
                    MissingAttempt(slug=slug, attempt=attempt, reason=reason or "no snapshot")
                )
                continue
            # A binary or non-UTF-8 patch is not a diff to review, and must not be
            # handed to the test-hunk stripper (which would only see mojibake). It is
            # counted with its reason, in front of everything else.
            binary_reason = binary_patch_reason(text)
            if binary_reason is not None:
                harvest.not_reviewable.append(
                    NotReviewable(slug=slug, attempt=attempt, reason=binary_reason)
                )
                continue
            try:
                cleaned = runner_mod.test_hunk_free(
                    text, what=f"{slug} attempt {attempt}"
                )
            except runner_mod.RunnerError:
                # Nothing but test hunks: there is no code to review.
                harvest.not_reviewable.append(
                    NotReviewable(
                        slug=slug, attempt=attempt, reason="the patch is nothing but test hunks"
                    )
                )
                continue
            harvest.attempts.append(
                HarvestedAttempt(
                    slug=slug,
                    attempt=attempt,
                    patch_text=cleaned,
                    origin=origin,
                    tag=entry.name,
                    source_model=model,
                    source_run_id=str(session_id),
                )
            )
    return harvest


# ---------------------------------------------------------------------------
# candidates, grading and selection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchCandidate:
    """One deduplicated patch, ready to be ordered and graded."""

    text: str
    normalized: str
    normalized_sha256: str
    tag: str
    priority: int
    source_model: str
    source_run_id: Optional[str]
    attempt: Optional[int]

    @property
    def order_key(self) -> tuple[int, str]:
        return (self.priority, self.normalized_sha256)


@dataclass
class Tally:
    """Every count the artifact's header keeps, per bug."""

    harvested: int = 0
    missing_snapshot: int = 0
    not_reviewable: int = 0
    duplicates: int = 0
    candidates: int = 0
    graded: int = 0
    bad: int = 0
    good: int = 0
    equivalent: int = 0
    cap_reached: bool = False
    failed_keys: int = 0
    unobservable: int = 0
    """Graded candidates whose patch would not apply (``label_input`` =
    ``unobservable``). Censored, not failed (§5.2 step 2): neither ``bad`` nor
    ``good``, so they are counted here and excluded from selection entirely."""
    unobservable_reasons: list[dict[str, Any]] = field(default_factory=list)
    """One record per unobservable grade — its hash, provenance and why it did not
    apply — so the count can be read back with the reason that produced it."""

    inconclusive: int = 0
    """Graded candidates whose patch applied but whose hidden run wrote no report
    (``label_input`` = ``inconclusive``, item B4). The third state: neither ``bad``
    nor ``good``, so it is counted here and excluded from selection entirely — the
    same "never counted as good or bad anywhere" rule that governs ``unobservable``,
    applied to the pool as item B4's follow-up.
    """
    inconclusive_reasons: list[dict[str, Any]] = field(default_factory=list)
    """One record per inconclusive grade — its hash, provenance and what the grading
    could not read — so the exclusion is citable, not a silent drop."""


def dedupe(attempts: Sequence[HarvestedAttempt]) -> tuple[list[PatchCandidate], int]:
    """``(candidates, duplicates)`` — one candidate per normalised diff.

    The duplicate count is kept rather than dropped: "12 attempts, 4 distinct
    patches" is the number that shows how repetitive a Creator was, and a tally
    that only counted the survivors could not say it.
    """
    seen: dict[str, PatchCandidate] = {}
    duplicates = 0
    for item in attempts:
        normalized = normalize(item.patch_text)
        if not normalized.strip():
            # A patch that normalises to nothing changes no code; it is the no-op
            # floor's shape, not a candidate.
            duplicates += 1
            continue
        digest = sha256_text(normalized)
        if digest in seen:
            duplicates += 1
            continue
        seen[digest] = PatchCandidate(
            text=item.patch_text,
            normalized=normalized,
            normalized_sha256=digest,
            tag=item.tag,
            priority=tag_for(item.tag).priority,
            source_model=item.source_model,
            source_run_id=item.source_run_id,
            attempt=item.attempt,
        )
    return sorted(seen.values(), key=lambda candidate: candidate.order_key), duplicates


@dataclass
class Selected:
    bad: list[tuple[PatchCandidate, oracle_mod.OracleResult]]
    good: list[tuple[PatchCandidate, oracle_mod.OracleResult]]
    tally: Tally


def select(
    *,
    candidates: Sequence[PatchCandidate],
    grade_fn: Callable[[PatchCandidate], oracle_mod.OracleResult],
    reference_sha256: str,
    tally: Tally,
    max_bad: int = MAX_BAD,
    max_good: int = MAX_GOOD,
    grade_cap: int = GRADE_CAP,
) -> Selected:
    """The spec's rule: ``(tag priority, sha256)`` order, grade, keep ≤2 bad and ≤2 good.

    The grader is injected so the whole rule is executable without a VM. A
    candidate that reproduces the reference fix is counted ``equivalent`` and does
    not spend budget — it is neither bad nor model-written-good.

    An ``unobservable`` grade (#F bug 1) is counted and **dropped**, exactly as the
    mutants pool drops a candidate that will not apply: it is a fact about the patch
    artifact, not about the patch's behaviour, so it may be neither ``bad`` nor
    ``good``, and keeping it would put a patch no reviewer can be scored on into a
    pool every review row is measured against. The search keeps spending its budget
    (``graded`` counts it) until the cap or a full pair of sides stops it.

    [VANTA-WHY] Item B4 follow-up — an ``inconclusive`` grade (the patch applied but
    the hidden run wrote no report) is dropped and counted the same way. It is the
    third state, and the general rule is that it is never counted as good or bad
    anywhere: keeping it here would write a ``failed_model`` entry whose ``oracle``
    reads ``inconclusive``, i.e. a bad patch to reject, on no evidence either way.

    [VANTA-WHY] "Keep at most two of each side" is a **cap on the kept lists**, not
    only a stop condition. The two sides fill at different speeds: a bug whose
    Creator usually works produces good patches long after its two bad ones, and
    keeping every later good one would put a pool of a dozen ``model_good`` entries
    behind a header that says ``bad: 2 good: 2`` — the artifact and the spec would
    disagree about the size of the good-patch side, and every extra entry is a
    review session the plan never budgeted. Grading continues either way (a full
    good side is not a reason to stop looking for the second bad patch), so
    ``graded`` can exceed ``bad + good``; that difference is the budget the search
    spent, and it is the honest number to record.
    """
    ordered = sorted(candidates, key=lambda candidate: candidate.order_key)
    bad: list[tuple[PatchCandidate, oracle_mod.OracleResult]] = []
    good: list[tuple[PatchCandidate, oracle_mod.OracleResult]] = []
    for candidate in ordered:
        if tally.graded >= grade_cap:
            tally.cap_reached = True
            break
        if candidate.normalized_sha256 == reference_sha256:
            tally.equivalent += 1
            continue
        tally.graded += 1
        result = grade_fn(candidate)
        if result.label_input == "unobservable":
            tally.unobservable += 1
            tally.unobservable_reasons.append(
                {
                    "patch_sha256": sha256_text(candidate.text),
                    "patch_source": candidate.tag,
                    "attempt": candidate.attempt,
                    "reason": result.apply_detail
                    or result.error
                    or "the patch did not apply to the grading tree",
                }
            )
            continue
        if result.label_input == "inconclusive":
            # [VANTA-WHY] Item B4 follow-up. The patch applied but the hidden run wrote
            # no report, so there is no evidence that it fails *or* that it fixes
            # anything. Item B4 already dropped such a grade from the *mutants* pool;
            # the assembled pool was still keeping it as ``bad`` and
            # ``entry_for_model`` would write ``kind=failed_model, oracle="inconclusive"``
            # — a patch every review row is scored against, labelled a bad patch to
            # reject on the strength of an unreadable run. It is dropped and counted
            # here too, and the search keeps spending its budget.
            tally.inconclusive += 1
            tally.inconclusive_reasons.append(
                {
                    "patch_sha256": sha256_text(candidate.text),
                    "patch_source": candidate.tag,
                    "attempt": candidate.attempt,
                    "reason": result.error
                    or "the hidden run produced no pytest report at all",
                }
            )
            continue
        if result.patch_good:
            if len(good) < max_good:
                good.append((candidate, result))
        elif len(bad) < max_bad:
            bad.append((candidate, result))
        if len(bad) >= max_bad and len(good) >= max_good:
            break
    tally.bad = len(bad)
    tally.good = len(good)
    return Selected(bad=bad, good=good, tally=tally)


# ---------------------------------------------------------------------------
# the artifact — header + JSON body, hash-locked
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PoolEntry:
    """One patch in the pool, with its provenance and its oracle label."""

    kind: str
    patch_source: str
    source_model: str
    source_run_id: Optional[str]
    patch_sha256: str
    patch_path: str
    oracle: str
    attempt: Optional[int] = None


@dataclass(frozen=True)
class MutantsRef:
    """Item 5's file, referenced rather than copied."""

    path: str
    content_sha256: str
    count: int


@dataclass
class PoolFile:
    """A locked assembled pool, read back and verified against its mutants file."""

    path: Path
    bug: str
    mutants: MutantsRef
    reference_sha256: str
    noop_sha256: str
    grade_cap: int
    sources: list[str]
    tally: Tally
    previous_sha256: Optional[str]
    entries: list[PoolEntry]
    mutants_entries: list[PoolEntry]
    content_sha256: str

    def union(self) -> list[PoolEntry]:
        """Everything a review may be shown: the mutants, then this file's own entries.

        The mutants come first because they are the reference side of the pool (the
        known-bad patches derived from the developer fix), and the file's own
        entries follow in their recorded order: reference, no-op, then the model
        patches.
        """
        return [*self.mutants_entries, *self.entries]


_REQUIRED_HEADER_KEYS = (
    "schema",
    "bug",
    "mutants",
    "mutants-sha256",
    "mutants-count",
    "reference-sha256",
    "noop-sha256",
    "sort-key",
    "grade-cap",
    "sources",
    "graded",
    "bad",
    "good",
    "equivalent",
    "unobservable",
    "inconclusive",
    "cap-reached",
    "missing-snapshot",
    "not-reviewable",
    "failed-keys",
    "previous-sha256",
    "content-sha256",
)


def _canonical_body(body: str) -> str:
    """The one form the JSON body takes — in the file and in the hash of it."""
    return locked.canonical(body)


def _header_block(text: str) -> tuple[dict[str, str], str]:
    return locked.header_block(text)


def _section(text: str, marker: str) -> tuple[str, str]:
    return locked.section(text, marker, error=PoolError, what="pool file")


def _optional_sha(value: Optional[str]) -> str:
    return value if value else "null"


def _verified_body(text: str, target: Path) -> tuple[dict[str, str], str]:
    """``(header, canonical body)`` — after the body is checked against its header hash.

    The one place the lock's content check lives: :func:`parse_pool` runs it and
    keeps going, :func:`previous_pool_sha256` runs it and stops at the hash.
    """
    header, body_text = _header_block(text)
    _, after = _section(body_text, POOL_MARKER)
    body = _canonical_body(after)
    recorded = header.get("content-sha256")
    if not recorded:
        raise PoolError(
            f"{target}: the header has no 'content-sha256' line, so the file's bytes "
            "cannot be checked against what it says it is"
        )
    actual = sha256_text(body)
    if actual != recorded:
        raise PoolError(
            f"{target}: content-sha256 in the header is {recorded}, but the body "
            f"hashes to {actual} — the file was edited after it was frozen"
        )
    return header, body


def render_pool(
    *,
    prep: BugPrep,
    mutants: MutantsRef,
    reference_sha256: str,
    noop_sha256: str,
    tally: Tally,
    entries: Sequence[PoolEntry],
    sources: Sequence[str],
    previous_sha256: Optional[str] = None,
    grade_cap: int = GRADE_CAP,
) -> str:
    """The artifact's exact bytes: the lock header, the marker, the JSON body."""
    body = json.dumps(
        {
            "mutants": {
                "path": mutants.path,
                "content_sha256": mutants.content_sha256,
                "count": mutants.count,
            },
            "entries": [_entry_dict(entry) for entry in entries],
            "unobservable": list(tally.unobservable_reasons),
            "inconclusive": list(tally.inconclusive_reasons),
        },
        indent=2,
        ensure_ascii=False,
    )
    canonical = _canonical_body(body)
    header = [
        f"# {prep.key} — the assembled review pool",
        "",
        f"- schema: {SCHEMA}",
        f"- bug: {prep.key}",
        f"- mutants: {mutants.path}",
        f"- mutants-sha256: {mutants.content_sha256}",
        f"- mutants-count: {mutants.count}",
        f"- reference-sha256: {reference_sha256}",
        f"- noop-sha256: {noop_sha256}",
        f"- sort-key: {SORT_KEY}",
        f"- grade-cap: {grade_cap}",
        f"- sources: {' '.join(sources)}",
        f"- graded: {tally.graded}",
        f"- bad: {tally.bad}",
        f"- good: {tally.good}",
        f"- equivalent: {tally.equivalent}",
        f"- unobservable: {tally.unobservable}",
        f"- inconclusive: {tally.inconclusive}",
        f"- cap-reached: {'true' if tally.cap_reached else 'false'}",
        f"- missing-snapshot: {tally.missing_snapshot}",
        f"- not-reviewable: {tally.not_reviewable}",
        f"- failed-keys: {tally.failed_keys}",
        f"- previous-sha256: {_optional_sha(previous_sha256)}",
        f"- content-sha256: {sha256_text(canonical)}",
    ]
    return "\n".join(header) + "\n\n" + POOL_MARKER + "\n\n" + canonical + "\n"


def _entry_dict(entry: PoolEntry) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": entry.kind,
        "patch_source": entry.patch_source,
        "source_model": entry.source_model,
        "source_run_id": entry.source_run_id,
    }
    if entry.attempt is not None:
        out["attempt"] = entry.attempt
    out["patch_sha256"] = entry.patch_sha256
    out["patch_path"] = entry.patch_path
    out["oracle"] = entry.oracle
    return out


def pool_path(out_dir: Path, prep: BugPrep) -> Path:
    """``<out>/pool/<slug>.json`` — the file reviews read."""
    return Path(out_dir) / POOL_DIRNAME / f"{prep.slug}{POOL_SUFFIX}"


def resolve_path(recorded: str, base: Path) -> Path:
    """Resolve a path the pool file recorded, relative to the pool file's directory.

    The join is **lexical** (``os.path.normpath``, no filesystem lookup). The pool
    lives outside every batch it harvests, so a recorded mutants path legitimately
    contains ``..`` segments that climb out of ``<out>/pool``; the writer verifies
    its own bytes *before* it creates that directory (:func:`locked.write`), so a
    resolver that needed each intermediate component to exist would fail every
    first build with an error about a file that is right there. Collapsing ``..``
    by text keeps the reader independent of whether the directory exists yet.
    """
    path = Path(recorded)
    if path.is_absolute():
        return path
    return Path(os.path.normpath(Path(base) / path))


def record_path(path: Path, base: Path) -> str:
    """A path as the pool file records it: relative to the pool file's directory.

    Relative — and possibly containing ``..`` — because the pool spans several
    batches and arms and so does not live inside any of them: an absolute path
    would bind the frozen file to one machine's checkout layout, and a plain
    basename would not say which batch the file is in.
    """
    try:
        return os.path.relpath(Path(path).resolve(), Path(base).resolve())
    except ValueError:  # pragma: no cover - different drives, POSIX cannot happen
        return str(Path(path).resolve())


def write_pool(
    path: Path,
    *,
    prep: BugPrep,
    mutants: MutantsRef,
    reference_sha256: str,
    noop_sha256: str,
    tally: Tally,
    entries: Sequence[PoolEntry],
    sources: Sequence[str],
    previous_sha256: Optional[str] = None,
    grade_cap: int = GRADE_CAP,
) -> Path:
    """Render, verify with the reader, then write — ``newline=""`` so the bytes hash."""
    text = render_pool(
        prep=prep,
        mutants=mutants,
        reference_sha256=reference_sha256,
        noop_sha256=noop_sha256,
        tally=tally,
        entries=entries,
        sources=sources,
        previous_sha256=previous_sha256,
        grade_cap=grade_cap,
    )
    return locked.write(Path(path), text, verify=parse_pool)


def load_pool(path: Path) -> PoolFile:
    """Read a locked pool file from disk, and verify it **and** its mutants file."""
    return locked.load(path, parse=parse_pool, error=PoolError, what="assembled pool")


def previous_pool_sha256(path: Path) -> str:
    """The ``content-sha256`` of the pool file a ``--rebuild`` is about to replace.

    [VANTA-WHY] Read with the **body check alone**, not through :func:`parse_pool`:
    a rebuild exists to replace a file an *older build* wrote, and that file may be
    missing a header key this version added (``unobservable``). Imposing the current
    key set on the file being replaced would make ``--rebuild`` refuse exactly the
    files it is for. The hash it returns is still the hash of the bytes on disk, and
    the body is still verified against it, so a file edited after it was frozen
    cannot be silently replaced.
    """
    target = Path(path)
    try:
        with target.open("r", encoding="utf-8", newline="") as handle:
            text = handle.read()
    except OSError as exc:
        raise PoolError(f"cannot read the pool file {target} to rebuild it: {exc}") from exc
    header, _ = _verified_body(text, target)
    return header["content-sha256"]


def parse_pool(text: str, *, path: Path) -> PoolFile:
    """Verify and read the artifact's text — the one parser, for writer and reader."""
    target = Path(path)
    header, body = _verified_body(text, target)
    locked.require_keys(
        header,
        _REQUIRED_HEADER_KEYS,
        path=target,
        error=PoolError,
        what="pool file",
        why="say what it came from and what it holds",
    )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise PoolError(f"{target}: the body is not JSON: {exc}") from exc
    raw_mutants = payload.get("mutants") or {}
    mutants = MutantsRef(
        path=str(raw_mutants.get("path", "")),
        content_sha256=str(raw_mutants.get("content_sha256", "")),
        count=_int_of(raw_mutants.get("count")),
    )
    # The mutants reference is in the header and in the body; a file whose two
    # halves disagree about it is not one the reader may use, because the half it
    # believed would decide which file got verified.
    for key, value in (
        ("mutants", mutants.path),
        ("mutants-sha256", mutants.content_sha256),
        ("mutants-count", str(mutants.count)),
    ):
        if header[key] != value:
            raise PoolError(
                f"{target}: the header's {key} is {header[key]!r} but the body says "
                f"{value!r} — the two halves of the mutants reference disagree"
            )
    entries = [_entry_from(raw) for raw in payload.get("entries", [])]
    raw_unobservable = payload.get("unobservable")
    unobservable_reasons = (
        [dict(item) for item in raw_unobservable if isinstance(item, dict)]
        if isinstance(raw_unobservable, list)
        else []
    )
    raw_inconclusive = payload.get("inconclusive")
    inconclusive_reasons = (
        [dict(item) for item in raw_inconclusive if isinstance(item, dict)]
        if isinstance(raw_inconclusive, list)
        else []
    )
    mutants_entries = _verify_mutants(target, mutants)
    previous = header["previous-sha256"]
    return PoolFile(
        path=target,
        bug=header["bug"],
        mutants=mutants,
        reference_sha256=header["reference-sha256"],
        noop_sha256=header["noop-sha256"],
        grade_cap=_int_of(header["grade-cap"]),
        sources=header["sources"].split(),
        tally=Tally(
            graded=_int_of(header["graded"]),
            bad=_int_of(header["bad"]),
            good=_int_of(header["good"]),
            equivalent=_int_of(header["equivalent"]),
            unobservable=_int_of(header["unobservable"]),
            unobservable_reasons=unobservable_reasons,
            inconclusive=_int_of(header["inconclusive"]),
            inconclusive_reasons=inconclusive_reasons,
            cap_reached=header["cap-reached"].strip().lower() in ("true", "1", "yes"),
            missing_snapshot=_int_of(header["missing-snapshot"]),
            not_reviewable=_int_of(header["not-reviewable"]),
            failed_keys=_int_of(header["failed-keys"]),
        ),
        previous_sha256=None if previous in ("", "null") else previous,
        entries=entries,
        mutants_entries=mutants_entries,
        content_sha256=header["content-sha256"],
    )


def _verify_mutants(target: Path, mutants: MutantsRef) -> list[PoolEntry]:
    """Load the referenced mutants file, verify it, and expand it into entries.

    Verifying the *referent* is what makes the reference worth writing: a mutants
    file that was regenerated after this pool was frozen is a different set of
    patches, and a review that read it would be reviewing a pool the pool file does
    not describe.
    """
    mutants_path = resolve_path(mutants.path, target.parent)
    frozen = mutants_mod.load_pool(mutants_path)
    if frozen.content_sha256 != mutants.content_sha256:
        raise PoolError(
            f"{target}: the mutants file {mutants_path} is {frozen.content_sha256}, not the "
            f"frozen {mutants.content_sha256} — it changed after this pool was built"
        )
    if len(frozen.entries) != mutants.count:
        raise PoolError(
            f"{target}: the mutants file {mutants_path} holds {len(frozen.entries)} entries, "
            f"not the frozen {mutants.count}"
        )
    return [
        PoolEntry(
            kind=MUTANT,
            patch_source=entry.source,
            source_model=REFERENCE_SOURCE,
            source_run_id=None,
            patch_sha256=entry.patch_sha256,
            patch_path=record_path(resolve_path(entry.patch_path, mutants_path.parent), target.parent)
            if not Path(entry.patch_path).is_absolute()
            else entry.patch_path,
            oracle=entry.oracle,
            attempt=None,
        )
        for entry in frozen.entries
    ]


def _int_of(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _entry_from(raw: Mapping[str, Any]) -> PoolEntry:
    attempt = raw.get("attempt")
    return PoolEntry(
        kind=str(raw.get("kind", "")),
        patch_source=str(raw.get("patch_source", "")),
        source_model=str(raw.get("source_model", "")),
        source_run_id=(
            None if raw.get("source_run_id") in (None, "") else str(raw.get("source_run_id"))
        ),
        patch_sha256=str(raw.get("patch_sha256", "")),
        patch_path=str(raw.get("patch_path", "")),
        oracle=str(raw.get("oracle", "")),
        attempt=None if attempt is None else _int_of(attempt),
    )


# ---------------------------------------------------------------------------
# the whole of item 6, for one bug
# ---------------------------------------------------------------------------
@dataclass
class PoolResult:
    path: Path
    reference_sha256: str
    noop_sha256: str
    tally: Tally
    entries: list[PoolEntry]
    sources: list[str]


def entry_for_model(
    candidate: PatchCandidate, result: oracle_mod.OracleResult, *, patch_path: Path, base: Path
) -> PoolEntry:
    return PoolEntry(
        kind=MODEL_GOOD if result.patch_good else FAILED_MODEL,
        patch_source=candidate.tag,
        source_model=candidate.source_model,
        source_run_id=candidate.source_run_id,
        patch_sha256=sha256_text(candidate.text),
        patch_path=record_path(patch_path, base),
        oracle=result.label_input,
        attempt=candidate.attempt,
    )


def _kind_for(candidate: PatchCandidate) -> str:
    """The patch's ``kind`` column: the fix, the floor, or a model patch."""
    if candidate.tag == REFERENCE_SOURCE:
        return DEV_FIX
    if candidate.tag == NOOP_SOURCE:
        return NOOP
    return FAILED_MODEL


def _source_label(candidate: PatchCandidate, rel: str) -> str:
    """The ``source`` string ``write_candidate_patch`` records for the patch file."""
    if candidate.tag == REFERENCE_SOURCE:
        return REFERENCE_SOURCE
    if candidate.tag == NOOP_SOURCE:
        return f"{NOOP_SOURCE}:{rel}"
    return f"{candidate.tag}:{candidate.source_model}"


def _candidate_entries(
    *,
    prep: BugPrep,
    out_dir: Path,
    base: Path,
    reference_candidate: PatchCandidate,
    reference_oracle: oracle_mod.OracleResult,
    reference_sha256: str,
    noop_candidate: PatchCandidate,
    noop_oracle: oracle_mod.OracleResult,
    noop_sha256: str,
    selected: Selected,
    written: Mapping[str, runner_mod.Candidate],
) -> list[PoolEntry]:
    """The reference fix, the no-op floor, then the selected model patches."""

    def path_for(candidate: PatchCandidate) -> Path:
        record = written.get(candidate.normalized_sha256)
        if record is not None:
            return record.patch_path
        # An injected grader wrote no file; record the destination the default one
        # would have chosen, so the pool still points at a path-shaped location.
        kind = _kind_for(candidate)
        return out_dir / "candidates" / prep.slug / f"{kind}-{sha256_text(candidate.text)[:12]}.patch"

    entries = [
        PoolEntry(
            kind=DEV_FIX,
            patch_source=REFERENCE_SOURCE,
            source_model=REFERENCE_SOURCE,
            source_run_id=None,
            patch_sha256=reference_sha256,
            patch_path=record_path(path_for(reference_candidate), base),
            oracle=reference_oracle.label_input,
        ),
        PoolEntry(
            kind=NOOP,
            patch_source=NOOP_SOURCE,
            source_model=NOOP_MODEL,
            source_run_id=None,
            patch_sha256=noop_sha256,
            patch_path=record_path(path_for(noop_candidate), base),
            oracle=noop_oracle.label_input,
        ),
    ]
    for candidate, result in [*selected.bad, *selected.good]:
        entries.append(
            entry_for_model(candidate, result, patch_path=path_for(candidate), base=base)
        )
    return entries


def run_pool(
    *,
    prep: BugPrep,
    out_dir: Path,
    harvests: Sequence[Harvest],
    mutants_file: Path,
    checkpoint: str,
    grade_batch_dir: Path,
    disk_mb: int,
    grade_fn: Optional[Callable[[PatchCandidate], oracle_mod.OracleResult]] = None,
    rebuild: bool = False,
    failed_keys: int = 0,
    grade_cap: int = GRADE_CAP,
    max_bad: int = MAX_BAD,
    max_good: int = MAX_GOOD,
) -> PoolResult:
    """Assemble, grade and lock ``<out>/pool/<slug>.json`` for one bug.

    The reference fix, the no-op floor and every harvested candidate all travel the
    **same** grading seam (:func:`_kind_for` decides the patch's ``kind``), so one
    injected grader covers the whole of item 6 and the selection rule is executable
    with no VM. ``grade_batch_dir`` is the batch that holds this bug's oracle trees
    and its grading cache — a rebuild must not re-grade a patch that batch already
    graded, which is what keying the cache by the patch's own hash buys.
    """
    target = pool_path(out_dir, prep)
    previous_sha256: Optional[str] = None
    if target.exists():
        if not rebuild:
            raise PoolError(
                f"{target} already exists. The pool is frozen once and read by every "
                "review; rebuilding it re-labels candidates that reviews already scored. "
                "Pass --rebuild to replace it (the replaced file's hash is recorded)."
            )
        # Read the header hash *without* the full key check: an older build could
        # have written this file before ``unobservable`` existed, and a rebuild that
        # refuses to replace the file it was asked to replace is useless (#F).
        previous_sha256 = previous_pool_sha256(target)

    # Built exactly as `candidate_set` builds them (`runner.py`), so a review reads
    # one floor and one fix, not two spellings.
    reference_text = runner_mod.test_hunk_free(
        prep.bug.bug_patch_txt, what=f"{prep.key} reference fix"
    )
    rel, noop_text = runner_mod.noop_patch_text(prep.root / "vanta_tree")
    noop_text = runner_mod.test_hunk_free(noop_text, what=f"{prep.key} no-op")
    reference_sha256 = sha256_text(reference_text)
    noop_sha256 = sha256_text(noop_text)
    reference_normalized = normalized_sha256(reference_text)
    reference_candidate = PatchCandidate(
        text=reference_text,
        normalized=normalize(reference_text),
        normalized_sha256=reference_normalized,
        tag=REFERENCE_SOURCE,
        priority=0,
        source_model=REFERENCE_SOURCE,
        source_run_id=None,
        attempt=None,
    )
    noop_candidate = PatchCandidate(
        text=noop_text,
        normalized=normalize(noop_text),
        normalized_sha256=normalized_sha256(noop_text),
        tag=NOOP_SOURCE,
        priority=0,
        source_model=NOOP_MODEL,
        source_run_id=None,
        attempt=None,
    )

    written: dict[str, runner_mod.Candidate] = {}

    def default_grade(candidate: PatchCandidate) -> oracle_mod.OracleResult:
        record = written.get(candidate.normalized_sha256)
        if record is None:
            record = runner_mod.write_candidate_patch(
                out_dir,
                prep,
                _kind_for(candidate),
                candidate.text,
                source=_source_label(candidate, rel),
            )
            written[candidate.normalized_sha256] = record
        runner_mod.grade_candidate(
            batch_dir=grade_batch_dir,
            prep=prep,
            candidate=record,
            checkpoint=checkpoint,
            disk_mb=disk_mb,
        )
        return record.oracle

    grader = grade_fn or default_grade

    tally = Tally(failed_keys=failed_keys)
    attempts: list[HarvestedAttempt] = []
    sources: list[str] = []
    for harvest in harvests:
        attempts.extend(harvest.attempts)
        tally.missing_snapshot += len(harvest.missing)
        tally.not_reviewable += len(harvest.not_reviewable)
        sources.append(harvest.tag)
    tally.harvested = len(attempts)
    candidates, duplicates = dedupe(attempts)
    tally.duplicates = duplicates
    tally.candidates = len(candidates)

    reference_oracle = grader(reference_candidate)
    noop_oracle = grader(noop_candidate)
    selected = select(
        candidates=candidates,
        grade_fn=grader,
        reference_sha256=reference_normalized,
        tally=tally,
        max_bad=max_bad,
        max_good=max_good,
        grade_cap=grade_cap,
    )

    entries = _candidate_entries(
        prep=prep,
        out_dir=out_dir,
        base=target.parent,
        reference_candidate=reference_candidate,
        reference_oracle=reference_oracle,
        reference_sha256=reference_sha256,
        noop_candidate=noop_candidate,
        noop_oracle=noop_oracle,
        noop_sha256=noop_sha256,
        selected=selected,
        written=written,
    )
    frozen_mutants = mutants_mod.load_pool(mutants_file)
    mutants_ref = MutantsRef(
        path=record_path(mutants_file, target.parent),
        content_sha256=frozen_mutants.content_sha256,
        count=len(frozen_mutants.entries),
    )
    sources_in_order = priority_of(sources)
    path = write_pool(
        target,
        prep=prep,
        mutants=mutants_ref,
        reference_sha256=reference_sha256,
        noop_sha256=noop_sha256,
        tally=tally,
        entries=entries,
        sources=sources_in_order,
        previous_sha256=previous_sha256,
        grade_cap=grade_cap,
    )
    return PoolResult(
        path=path,
        reference_sha256=reference_sha256,
        noop_sha256=noop_sha256,
        tally=tally,
        entries=entries,
        sources=sources_in_order,
    )


def render_pool_line(prep: BugPrep, result: PoolResult) -> str:
    """One line per bug, for the batch's console log."""
    tally = result.tally
    return (
        f"{prep.key}: attempts {tally.harvested} · missing-snapshot {tally.missing_snapshot} · "
        f"not-reviewable {tally.not_reviewable} · duplicates {tally.duplicates} · "
        f"candidates {tally.candidates} · graded {tally.graded} · bad {tally.bad} · "
        f"good {tally.good} · equivalent {tally.equivalent} · "
        f"unobservable {tally.unobservable} · "
        f"inconclusive {tally.inconclusive} · "
        f"cap-reached {'true' if tally.cap_reached else 'false'} · {result.path}"
    )


def plan_lines(prep: BugPrep, *, harvests: Sequence[Harvest], limit: int = 8) -> list[str]:
    """What the pool would grade, from the batches' evidence alone (no VM)."""
    attempts: list[HarvestedAttempt] = []
    missing = 0
    not_reviewable = 0
    for harvest in harvests:
        attempts.extend(harvest.attempts)
        missing += len(harvest.missing)
        not_reviewable += len(harvest.not_reviewable)
    candidates, duplicates = dedupe(attempts)
    tags = priority_of([harvest.tag for harvest in harvests])
    lines = [
        f"{prep.key}: {len(attempts)} attempts from {' '.join(tags) or '(no sources)'} "
        f"→ {len(candidates)} candidates (deduplicated), duplicates {duplicates}, "
        f"missing-snapshot {missing}, not-reviewable {not_reviewable}, "
        f"up to {MAX_BAD} bad + {MAX_GOOD} good, grade-cap {GRADE_CAP}"
    ]
    for index, candidate in enumerate(candidates[:limit], start=1):
        lines.append(
            f"  {index} {candidate.priority} {candidate.tag} "
            f"{candidate.normalized_sha256[:12]} attempt {candidate.attempt}"
        )
    if len(candidates) > limit:
        lines.append(f"  … {len(candidates) - limit} more")
    for harvest in harvests:
        for gone in harvest.missing:
            lines.append(f"  missing {gone.slug} attempt {gone.attempt}: {gone.reason}")
    return lines
