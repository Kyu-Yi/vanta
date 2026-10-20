"""Harness-side compaction of a session's attempt snapshots (design §6.3, D6).

Vanta's driver writes a **full copy** of every tree it snapshots to
``vanta_sessions/<sid>/debug/attempt_N_{patched,graded}_tree/`` — plus a
``{paths, file_count}`` manifest — once per attempt. On a repo the size of
``black`` that dominates a pilot-scale disk budget, and §6.3's decision is to let
the **harness** do the compacting: ``write_tree_snapshot`` and ``rerun.rs`` stay
untouched (C2), because the pipeline's own snapshotting is not what is expensive
here — the *pile of pilot runs* is.

[VANTA-WHY] The one thing compaction must not break is ``--rerun-signals``. The
rerun reads the bad side back as a **directory**
(``BadTreeSource::GradedSnapshot``, falling back to ``_patched_tree``), so
deleting the directory would turn "the signal did not reproduce" into "there was
no session to re-run" — a fabricated ``spurious_reject``. §6.3's rule is
therefore *not* "compact everything". It is:

* a session with **no rejecting signal** has nothing to re-run, so its snapshots
  may go at once;
* a session that **was** rejected keeps its snapshots until its rerun has been
  recorded (``<sid>_signal_reruns.json``), because that rerun is the only thing
  that can tell a real catch from a spurious one (§5.4).

That rule lives here in one place — :func:`compaction_plan` decides,
:func:`compact_session` acts — so there is no second copy of it to drift.

[VANTA-WHY] What is deleted is replaced by a **diff against ``vanta_clean/``**
plus a ``{path: sha256}`` manifest **and the sha256 of the tree's concatenation**,
and :func:`reconstruct` turns the first two back into a tree. That is what §6.3's
test proves (a compacted session replays byte-identically) and what a future
"rebuild from base + diff" rerun would need. The concat hash is not implied by the
per-file hashes for free: it is the unit §6.3's test compares, so recording it is
what lets the *replay* be checked against an artifact on disk rather than against a
tree the test still happened to hold in memory.

Both the ``patched`` and the ``graded`` snapshot are diffed before deletion: the
graded one is what the rerun reads, the patched one is its documented fallback,
and deleting a tree without a diff would destroy evidence the rule never
described.

[VANTA-WHY] The hashes are ``sha256``, not the blake3 the design first sketched:
``sha256_text`` / ``sha256_file`` are already this harness's only hash, and a
second algorithm would be a second thing to get right for no gain. The artifact
is ``attempt_N_<kind>_tree.hashes.json`` rather than ``_manifest.json`` because
Vanta *already* writes ``attempt_N_<kind>_tree_manifest.json`` and the rerun's
tree reader **skips** ``*_manifest.json`` — reusing that name would either
overwrite Vanta's own forensics or be invisible to the reader that matters.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .split import sha256_bytes, sha256_text

FULL = "full"
"""Snapshots were found and left on disk as directories (nothing compacted)."""
COMPACTED = "compacted"
"""Every snapshot found was replaced by a diff + hashes manifest."""
NONE = "none"
"""The session wrote no attempt snapshot at all (an ``InfraFailure`` or a patch
that never applied). A third value rather than reusing ``full``: "the snapshots
are still here" and "there were never any" are different facts, and a metrics
reader that folded them together could not tell a cheap session from an
uncompacted one."""

NOISE_DIRS = frozenset({".git", "__pycache__", "node_modules"})
NOISE_SUFFIXES = (".pyc", ".pyo", "_manifest.json")
"""Exactly what ``rerun.rs::collect_tree_files`` skips: version control, bytecode
caches, dependency trees and snapshot manifests. The concat and the hashes use
the same filter, so both describe the file set the rerun would actually read — a
manifest that hashed a ``.pyc`` the reader ignores would be evidence about a file
nobody looks at."""


class CompactError(RuntimeError):
    """Compaction could not be trusted to be reversible."""


# ---------------------------------------------------------------------------
# reading a tree the way the rerun reads it
# ---------------------------------------------------------------------------
def tree_files(root: Path) -> list[tuple[str, Path]]:
    """``(relative path, path)`` for every file under ``root``, sorted by path.

    Sorted because the driver's concatenation is ``BTreeMap``-ordered
    (``read_tree_concat`` sorts for the same reason): a concat whose order came
    from the filesystem would differ between machines and the equivalence proof
    would be about nothing.
    """
    if not root.is_dir():
        raise CompactError(f"{root} is not a directory")
    found: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Pruned in place so `os.walk` never descends into them — the same trees
        # the reader skips, at any depth.
        dirnames[:] = sorted(name for name in dirnames if name not in NOISE_DIRS)
        for name in sorted(filenames):
            if name.endswith(NOISE_SUFFIXES):
                continue
            path = Path(dirpath) / name
            found.append((path.relative_to(root).as_posix(), path))
    found.sort(key=lambda pair: pair[0])
    return found


def tree_manifest(root: Path) -> dict[str, str]:
    """``{relative path: sha256}`` for every file the rerun's reader would see."""
    return {rel: sha256_bytes(path.read_bytes()) for rel, path in tree_files(root)}


def tree_concat(root: Path) -> str:
    """The driver's ``# FILE:`` concatenation of a tree.

    A faithful copy of ``rerun.rs::read_tree_concat``: sorted relative paths,
    ``# FILE: {path}\\n{contents}`` joined by a blank line, and non-UTF-8 files
    **skipped rather than lossily decoded** (a replacement-character version of a
    binary is a file the harness could not distinguish from real source). It is
    duplicated here on purpose — the Rust function is the definition the
    equivalence test has to match, and importing it would mean the test compared
    the implementation to itself.
    """
    parts: list[str] = []
    for rel, path in tree_files(root):
        try:
            contents = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        parts.append(f"# FILE: {rel}\n{contents}")
    return "\n\n".join(parts)


def tree_diff(base: Path, tree: Path) -> str:
    """A unified diff of ``tree`` against ``base``, in ``git apply -p1``'s shape.

    ``git diff --no-index`` names each side after the path it was given, so the
    two trees are addressed *relative to their common parent* and the resulting
    ``a/<base>/…`` / ``b/<tree>/…`` headers are rewritten down to ``a/…`` /
    ``b/…``. Hand-writing the hunk headers instead would risk a miscount that
    ``git apply --recount`` accepts silently.
    """
    base_abs = base.resolve()
    tree_abs = tree.resolve()
    if not base_abs.is_dir():
        raise CompactError(f"{base_abs} is not a directory — nothing to diff against")
    if not tree_abs.is_dir():
        raise CompactError(f"{tree_abs} is not a directory — nothing to diff")
    common = Path(os.path.commonpath([str(base_abs), str(tree_abs)]))
    base_rel = base_abs.relative_to(common).as_posix()
    tree_rel = tree_abs.relative_to(common).as_posix()
    proc = subprocess.run(
        ["git", "diff", "--no-index", "--no-color", "--", base_rel, tree_rel],
        cwd=str(common),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
        stdin=subprocess.DEVNULL,
    )
    # `--no-index` exits 1 whenever the trees differ — the normal case.
    if proc.returncode not in (0, 1):
        raise CompactError(f"git diff --no-index failed: {proc.stderr.strip()[:400]}")
    return _rewrite_diff_headers(proc.stdout, base_rel, tree_rel)


def _rewrite_diff_headers(diff_text: str, base_rel: str, tree_rel: str) -> str:
    """Strip the two scratch path prefixes from the three header line shapes.

    A *new* file is named ``a/<tree>/…`` on both sides of its ``diff --git`` line
    (git prints the post-image path for both), and a *deleted* one is named
    ``a/<base>/…`` on both — so each side of each header has to be tested against
    **both** prefixes rather than assuming ``a/`` means base.
    """
    rewrites = [
        (f"a/{base_rel}/", "a/"),
        (f"a/{tree_rel}/", "a/"),
        (f"b/{base_rel}/", "b/"),
        (f"b/{tree_rel}/", "b/"),
    ]
    out: list[str] = []
    for line in diff_text.splitlines(keepends=True):
        if line.startswith(("diff --git ", "--- ", "+++ ")):
            for old, new in rewrites:
                line = line.replace(old, new)
        out.append(line)
    return "".join(out)


def reconstruct(base: Path, diff_text: str, dest: Path) -> Path:
    """Rebuild a compacted tree: copy ``base`` to ``dest``, then apply ``diff_text``.

    The reverse of what :func:`compact_session` wrote, and the thing §6.3's test
    asserts on — not a helper for the harness's own flow. ``git apply`` refuses a
    patch it cannot apply cleanly (no fuzzy fallback here), so a diff that lost a
    hunk fails loudly instead of producing a plausible-looking tree.
    """
    if dest.exists():
        raise CompactError(f"{dest} already exists; reconstruct needs an empty destination")
    shutil.copytree(base, dest, symlinks=True)
    diff_path = dest.parent / f".{dest.name}.reconstruct.patch"
    diff_path.write_text(diff_text, encoding="utf-8")
    try:
        proc = subprocess.run(
            ["git", "apply", "-p1", "--verbose", str(diff_path)],
            cwd=str(dest),
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
            stdin=subprocess.DEVNULL,
        )
    finally:
        diff_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise CompactError(
            f"reconstructing {dest.name} from the diff failed: {proc.stderr.strip()[:400]}"
        )
    return dest


# ---------------------------------------------------------------------------
# the §6.3 rule
# ---------------------------------------------------------------------------
def has_rejecting_signal(verdict: str) -> bool:
    """``Rejected`` is the only verdict a signal rerun exists to explain (§5.4).

    A rerun answers "did the signal that made Vanta refuse *discriminate*", which
    only turns a ``Rejected`` into ``true_catch`` or ``spurious_reject``. An
    approval and a give-up have no rejection to explain, so holding their
    snapshots would cost disk and buy nothing.
    """
    return verdict == "Rejected"


def rerun_recorded(sessions_dir: Path, session_id: str) -> bool:
    """Has ``--rerun-signals`` already written this session's rerun artifact?

    Strict on purpose: a file that exists but has no ``signals`` list or no
    ``final_decision`` is *not* a recorded rerun. Treating a truncated write as
    "the rerun happened" would let compaction delete the only tree the rerun
    needed, which is the exact failure §6.3 exists to prevent.
    """
    path = sessions_dir / f"{session_id}_signal_reruns.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload.get("signals"), list) and bool(payload.get("final_decision"))


def compaction_plan(
    *, verdict: str, session_id: Optional[str], sessions_dir: Path
) -> tuple[str, str]:
    """``(state, reason)`` for one session, before anything is touched."""
    if not has_rejecting_signal(verdict):
        return COMPACTED, f"verdict {verdict!r} has no rejecting signal to re-run (§6.3)"
    if session_id and rerun_recorded(sessions_dir, session_id):
        return COMPACTED, f"rerun recorded in {session_id}_signal_reruns.json (§6.3)"
    return FULL, (
        "rejecting signal with no recorded rerun — the graded snapshot stays until "
        "`vanta_m3 --rerun-signals` has run, or a rerun rebuilds it from base + diff "
        "(§6.3, D6)"
    )


def _attempt_trees(debug_dir: Path) -> list[Path]:
    """The ``attempt_N_{patched,graded}_tree/`` directories in one session."""
    if not debug_dir.is_dir():
        return []
    return sorted(path for path in debug_dir.glob("attempt_*_tree") if path.is_dir())


@dataclass
class Compaction:
    """What compaction did to one session, for the manifest row (§7.3)."""

    state: str
    reason: str
    diff: Optional[str] = None
    """Session-relative path of the **graded** tree's diff, or ``None``. The
    graded tree is the one ``--rerun-signals`` reads, so its diff is the
    reconstruction that matters; the patched tree's diff is written too but is
    not what the row points at."""
    trees: list[str] = field(default_factory=list)
    """Names of the snapshots that were deleted (or, for ``full``, that stayed)."""


def compact_session(
    *,
    session_dir: Path,
    base_tree: Path,
    verdict: str,
    session_id: Optional[str],
    sessions_dir: Path,
    base_commit: Optional[str] = None,
) -> Compaction:
    """Apply §6.3 to one session: decide, then compact if the rule allows.

    Raises rather than degrading: a compaction that wrote a wrong diff and then
    deleted the tree would have destroyed the only copy, so every write is
    finished (and fsynced by ``close``) before the corresponding ``rmtree``.
    """
    state, reason = compaction_plan(
        verdict=verdict, session_id=session_id, sessions_dir=sessions_dir
    )
    trees = _attempt_trees(session_dir / "debug")
    if not trees:
        return Compaction(
            state=NONE, reason="the session wrote no attempt tree snapshot to compact"
        )
    if state == FULL:
        return Compaction(state=FULL, reason=reason, trees=[tree.name for tree in trees])
    if not base_tree.is_dir():
        raise CompactError(
            f"compaction needs the sanitized base to diff against, but {base_tree} "
            "is not a directory"
        )

    graded_diff: Optional[str] = None
    compacted: list[str] = []
    for tree in trees:
        diff_text = tree_diff(base_tree, tree)
        hashes = tree_manifest(tree)
        concat_sha = sha256_text(tree_concat(tree))
        diff_path = tree.parent / f"{tree.name}.diff"
        hashes_path = tree.parent / f"{tree.name}.hashes.json"
        diff_path.write_text(diff_text, encoding="utf-8")
        hashes_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "compacted_tree",
                    "tree": tree.name,
                    "base": base_tree.name,
                    "base_commit": base_commit,
                    "diff": diff_path.name,
                    "diff_sha256": sha256_text(diff_text),
                    "concat_sha256": concat_sha,
                    "file_count": len(hashes),
                    "files": hashes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(tree)
        compacted.append(tree.name)
        if tree.name.endswith("_graded_tree"):
            graded_diff = f"debug/{diff_path.name}"
    return Compaction(state=COMPACTED, reason=reason, diff=graded_diff, trees=compacted)
