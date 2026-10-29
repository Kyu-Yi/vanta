"""Checkouts and the SWE-bench tree split (design §1.2, §1.3).

Four trees per bug, three of them seen by something:

``src_buggy/``   plain buggy commit — only used to learn which failures are
                 pre-existing (the relaxed fixed-tree gate, §1.3 step 6).
``src_fixed/``   plain fixed commit — the gate and ``PASS_TO_PASS``.
``vanta_tree/``  buggy source + the FIXED commit's test files, minus the hidden
                 tests. This is what Vanta is given.
``oracle_tree/`` the same tree with the hidden tests intact — the oracle.

[VANTA-WHY] Why the fixed commit's test files at all: the fix usually *updates*
existing tests to the new behaviour. Grading a correct fix against the old
versions of those tests manufactures a native regression that is not the patch's
fault — a false reject invented by the harness (design §1.2). Taking the post-fix
test files is the standard SWE-bench split and is the fair reading.

[VANTA-WHY] Why the strip is AST-level: the hidden test is the oracle. A regex
that removed a function by name would either miss a decorator block or eat the
next function's body, and a *partially* removed test is worse than none — it
leaves the assertion (the answer) in the tree the Creator can read.

Nothing in this module executes the project's code. Clones and tree surgery are
git and filesystem work on the host; every suite run happens in Shuru (§suite.py).
"""

from __future__ import annotations

import ast
import hashlib
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .bugsinpy_meta import BugInfo, decode_metadata


class TreeError(RuntimeError):
    """A checkout or a tree operation failed."""


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------
def _git(args: list[str], *, cwd: Optional[Path]) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise TreeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def ensure_project_clone(cache_root: Path, project: str, url: str) -> Path:
    """Clone (once) the upstream project repo and return the cache path."""
    dest = cache_root / f"{project}.git"
    if not dest.is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--quiet", "--", url, str(dest)], cwd=None)
    return dest


def read_file_at(repo: Path, commit: str, path: str) -> Optional[str]:
    """One file out of a clone at ``commit``, or ``None`` when it is not there.

    [VANTA-WHY] Read through ``git`` rather than from a checked-out tree: a bake
    needs a project's *buggy-commit* ``setup.py`` and ``requirements.txt`` before any
    tree exists, and a checkout would be a second materialisation of the same commit
    that could disagree with the one the suites run. The file is read as text and
    never executed — C3 has no exception for "just asking setuptools what it wants".
    """
    proc = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, check=False
    )
    if proc.returncode != 0:
        return None
    # The same reader the BugsInPy metadata goes through. This path feeds the other
    # half of the dependency sources — the project's own ``setup.py``, the file its
    # ``install_requires`` names, and ``test_requirements.txt`` — so a UTF-16
    # requirements file in the *repo* would corrupt a requirement exactly the way
    # black/22's did in BugsInPy's copy of it.
    return decode_metadata(proc.stdout, path=Path(f"{repo.name}:{path}@{commit[:12]}"))


def commit_date(repo: Path, commit: str) -> str:
    """``YYYY-MM-DD`` of a commit's own date — the cutoff for date-pinned deps (§3.2).

    [VANTA-WHY] The date is read in the commit's own offset and not converted to
    UTC. Dependencies are pinned to the newest release on or before the day the tree
    was committed, and a release published while the author's working day was half
    over is not a release that tree could have used.
    """
    date = _git(["show", "-s", "--format=%cI", commit], cwd=repo).strip()[:10]
    if len(date) != 10 or date[4] != "-":
        raise TreeError(f"cannot read a commit date for {commit} in {repo}: {date!r}")
    return date


def list_files_at(repo: Path, commit: str) -> list[str]:
    """Every path in ``commit``'s tree, relative to the repository root.

    [VANTA-WHY] A list and not a checkout: the dependency closure (§3.2) has to ask
    which test-side files the tree under test carries, and it asks that question at
    bake time, before any tree is materialised. Reading a file is
    :func:`read_file_at`; this is only the index of what could be read.
    """
    out = _git(["ls-tree", "-r", "--name-only", commit], cwd=repo)
    return [line.strip() for line in out.splitlines() if line.strip()]


def export_tree(repo: Path, commit: str, dest: Path) -> None:
    """Materialise ``commit`` of ``repo`` at ``dest`` as a self-contained tree.

    A real clone rather than ``git archive`` or ``git worktree``: ``git worktree``
    leaves a ``.git`` *file* whose contents are an absolute host path, which is
    meaningless once the tree is mounted inside the microVM, and ``git archive``
    drops the ``.git`` directory entirely, which costs us ``git apply`` and
    ``git diff`` inside the tree for no benefit.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(["clone", "--quiet", "--no-checkout", "--", str(repo), str(dest)], cwd=None)
    try:
        _git(["checkout", "--quiet", "--detach", commit], cwd=dest)
    except TreeError:
        # A commit that lives on no branch (a PR merge, a tag-only commit) is not
        # in the clone; fetch it by sha and retry rather than failing the bug.
        _git(["fetch", "--quiet", "origin", commit], cwd=dest)
        _git(["checkout", "--quiet", "--detach", commit], cwd=dest)


# ---------------------------------------------------------------------------
# which paths are "test files"
# ---------------------------------------------------------------------------
def is_test_path(path: str) -> bool:
    """Mirror of ``harness::is_native_test_path`` (Rust) for the tree split.

    Kept deliberately identical in behaviour: the harness uses that function to
    decide what the native suite owns, and the tree split uses this one to decide
    which paths come from the fixed commit. If the two disagreed, a file could be
    "fixed" for Vanta but "original" for the oracle.
    """
    name = path.rsplit("/", 1)[-1]
    if name == "conftest.py":
        return True
    if not name.endswith(".py"):
        return False
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    parts = path.split("/")[:-1]
    return any(part in ("test", "tests") for part in parts)


def is_test_side(path: str) -> bool:
    """Everything the fix-side copy may legitimately bring into the tree.

    [VANTA-WHY] A superset of :func:`is_test_path`, for the same reason the third
    source of :func:`test_paths_from_fix` is a whole directory: a test package is
    not only its ``.py`` modules. ``tests/data/golden.json``, a fixture image, a
    ``tests/__init__.py`` — none of them is a test *module*, all of them are test
    material the fixed commit may have changed, and all of them live under a
    directory named ``test``/``tests``. So the question "may this path come from
    the fixed commit?" is asked of the whole path, not of the file name.
    """
    if is_test_path(path):
        return True
    return any(part in ("test", "tests") for part in path.split("/")[:-1])


@dataclass
class PatchPaths:
    """File paths a unified diff touches, split three ways."""

    changed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return sorted(set(self.changed) | set(self.added) | set(self.deleted))


@dataclass
class ParsedPatch:
    paths: PatchPaths
    added_lines: list[str]
    removed_lines: list[str]


_DIFF_HEADER = re.compile(r"^diff --git a/(?P<a>.+?) b/(?P<b>.+)$")


def parse_patch(patch_text: str) -> ParsedPatch:
    """Paths and added/removed content lines of a unified diff, by hand.

    ``git apply --numstat`` needs a work tree and a repository; the split's
    metadata is read before any tree exists, so the diff is parsed directly. Only
    the three markers git itself emits are trusted: ``diff --git a/… b/…``,
    ``+++ b/…`` / ``--- a/…`` with ``/dev/null`` for creations and deletions.
    """
    paths = PatchPaths()
    added: list[str] = []
    removed: list[str] = []
    current: Optional[str] = None
    current_is_new = False
    current_is_del = False

    for line in patch_text.splitlines():
        header = _DIFF_HEADER.match(line)
        if header:
            current = header.group("b")
            current_is_new = False
            current_is_del = False
            continue
        if line.startswith("new file mode"):
            current_is_new = True
            continue
        if line.startswith("deleted file mode"):
            current_is_del = True
            continue
        if line.startswith("rename to "):
            current = line[len("rename to ") :].strip()
            continue
        if line.startswith("--- "):
            target = line[4:].strip()
            if target == "/dev/null":
                current_is_new = True
            elif current is None:
                current = target[2:] if target.startswith(("a/", "b/")) else target
            continue
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target == "/dev/null":
                current_is_del = True
                if current is not None:
                    paths.deleted.append(current)
                current = None
                continue
            if target.startswith("b/"):
                current = target[2:]
            if current is not None:
                if current_is_new:
                    paths.added.append(current)
                else:
                    paths.changed.append(current)
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])

    paths.changed = sorted(set(paths.changed))
    paths.added = sorted(set(paths.added))
    paths.deleted = sorted(set(paths.deleted))
    return ParsedPatch(paths=paths, added_lines=added, removed_lines=removed)


def test_dirs_from_fix(test_paths: Iterable[str]) -> list[str]:
    """The directories the fix-side test paths are synced from, whole.

    [VANTA-WHY] The **outermost** path component named ``test``/``tests``, and not
    ``path.split("/")[0]``. The two agree only when the test package is the
    repository's first component, which is the common layout but not the rule: for
    ``pandas/tests/dtypes/test_dtypes.py`` the first component is ``pandas`` and for
    ``lib/matplotlib/tests/test_bbox_tight.py`` it is ``lib``, so the "test
    directory" that rule names is the project's **whole source root**, and
    :func:`compose_vanta_tree` copies the FIXED commit's source over the buggy one.
    The consequence is worse than a wrong directory: the fix itself lands in the
    tree Vanta reads (leakage — the answer is in the tree) and in ``oracle_tree``
    (the oracle's ``F_before`` is measured against already-fixed source, comes out
    empty, and the bug reads as "the declared test does not flip"). pandas/1 and
    matplotlib/1 were both prepared that way.

    The outermost and not the module's own directory: helpers live beside the test
    *package*, not beside the test module — ``tests/rules/test_x.py`` imports
    ``tests.utils``, so syncing ``tests/rules`` alone would leave that import
    broken. Syncing ``tests`` whole is also what keeps the test *data* a test
    package carries (``tests/data/golden.json``) out of the leak question: those
    files are not test modules by name, and only the directory rule brings them.

    A path with no ``test``/``tests`` component falls back to the module's own
    directory — the narrowest reading, since a test module usually sits beside the
    helpers it imports. That fallback can still name a source root (a tree that
    keeps ``pkg/test_x.py`` with no test directory at all), and the honest answer
    there is not a cleverer guess but :func:`leaked_source`: ``prepare`` refuses
    such a tree loudly instead of leaking the fix into it.
    """
    dirs = set()
    for path in test_paths:
        parts = path.split("/")
        if len(parts) < 2:
            # A test module at the repository root has no directory to sync; the
            # module itself is already named by the first two sources.
            continue
        ancestor = None
        for index, part in enumerate(parts[:-1]):
            if part in ("test", "tests"):
                ancestor = index
                break
        if ancestor is None:
            dirs.add("/".join(parts[:-1]))
        else:
            dirs.add("/".join(parts[: ancestor + 1]))
    return sorted(dirs)


def test_paths_from_fix(bug: BugInfo, fixed_dir: Path) -> list[str]:
    """The files Vanta must take from the FIXED commit.

    Three sources, unioned:

    * every path the fix patch touches that looks like a test (the fix usually
      updates existing tests, and that update is part of the new contract);
    * ``bug.info``'s ``test_file``, which is BugsInPy's own statement of where the
      bug-revealing test lives — it is authoritative even if the patch's diff
      header is unusual;
    * every file under the test directory each of those paths lives in
      (:func:`test_dirs_from_fix`), read from ``fixed_dir``.

    [VANTA-WHY] The third source is not tidiness, it is the difference between a
    tree that can be collected and one that cannot. The first two sources name test
    *modules*, and a module is not self-contained: the fixed test file imports
    helpers from beside it (``tests/utils.py``, a ``tests/mini_toolbox/`` package,
    ``tests/__init__.py``), and those helpers are regularly only present — or only
    current — in the fixed commit. Copying the module alone hands Vanta a tree that
    dies at collection with ImportError, which the harness reads as "0 tests
    collected", indistinguishable from a broken environment. The first real
    prepare run lost PySnooper/2 exactly that way: its test file does
    ``from . import mini_toolbox``, and ``tests/mini_toolbox/`` does not exist in
    the buggy tree at all.

    Whole directory rather than "the helpers it imports": which helper a test
    imports is a question the harness cannot answer without importing the test
    module, which is the thing it is not allowed to run outside Shuru. What keeps
    the hidden tests out of the tree is the strip below, not the copy — the copy
    brings everything, ``strip_hidden_tests`` then removes every hidden nodeid, so
    a newly-synced test module is present only as far as it has kept tests.

    Union, not intersection: a path is included if *any* source names it.
    """
    parsed = parse_patch(bug.bug_patch_txt)
    paths = {p for p in parsed.paths.all if is_test_path(p)}
    paths.add(bug.test_file)
    for rel_dir in test_dirs_from_fix(paths):
        src_dir = fixed_dir / rel_dir
        if not src_dir.is_dir():
            continue
        for file in src_dir.rglob("*"):
            if file.is_file():
                paths.add(file.relative_to(fixed_dir).as_posix())
    return sorted(paths)


# ---------------------------------------------------------------------------
# the leak guard
# ---------------------------------------------------------------------------
_NOISE_DIRS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".hypothesis"}
)
"""Directories whose contents are never compared.

Kept after the guard moved to git-tracked paths (item N), and it is now inert for
every real project: ``git ls-tree`` never lists a path under ``.git`` or
``__pycache__``, and a repository that tracks a file under a cache-named directory
is committing a cache — the wrong thing to treat as source either way. ``.hypothesis``
is hypothesis's example database, a runtime cache the suite writes into the tree it
runs in, and it is the shape that made this list necessary; it stays here so that a
project which *does* track one is still handled the same way it was before.

The generated files the guard exists to ignore — ``build/**``, ``*.o``, ``*.so``,
in-place extensions, ``*.egg-info`` — are not on this list, and must not be. They are
ignored because git does not track them (:func:`tracked_paths`), so the guard has no
noise list to grow as a project's build output changes."""


def tracked_paths(checkouts: Iterable[Path]) -> set[str]:
    """Every path git tracks in ``checkouts``, unioned, read from each one's ``HEAD``.

    [VANTA-WHY] This is the candidate set the leak guard compares, and it is derived
    from git rather than from a walk of the filesystem on purpose. A project with C
    extensions leaves build output in the tree in place — ``build/temp…/<name>.o``,
    ``build/lib…/<name>.so``, an in-place extension, ``*.egg-info``, a cache — and none
    of it is tracked, so a filesystem walk finds files that differ between the buggy
    commit and the tree for a reason that has nothing to do with the fix, and the guard
    names the project's own build as a source leak. pandas/1 (100 ``.o`` files) and
    matplotlib/1 (1338 ``.so`` files) were both refused that way.

    ``src_buggy`` is exported at the buggy commit and ``src_fixed`` at the fixed one, so
    their union is exactly "tracked in the buggy or fixed commit". The fixed commit is
    in the union and not only the buggy one because a source file the fix *added* is
    tracked in the fixed commit alone: leaving it out would let an added fixed source
    file sit in the tree unnamed.
    """
    tracked: set[str] = set()
    for checkout in checkouts:
        tracked.update(list_files_at(checkout, "HEAD"))
    return tracked


def _digest(path: Path) -> Optional[str]:
    """``sha256`` of ``path``'s bytes, or ``None`` when it is not a regular file."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leaked_source(
    buggy_dir: Path, tree_dir: Path, tracked: Iterable[str]
) -> list[str]:
    """Tracked source files where ``tree_dir`` is not the buggy commit's tree.

    [VANTA-WHY] Empty is the invariant, and it is the whole point of the tree
    split: the tree Vanta reads and the tree the oracle grades must carry the
    **buggy commit's source** and only the fixed commit's *tests* (§1.2). A tree
    that carries fixed source is two failures at once — the answer is in the tree
    the Creator can read, and the oracle's "failures before the fix" are measured
    against source that is already fixed, which is why the symptom shows up as an
    empty ``FAIL_TO_PASS`` rather than as an obvious corruption.

    Only the paths git tracks in the buggy or fixed commit (``tracked``, from
    :func:`tracked_paths`) are compared, so a generated file the build or the suite
    leaves in the tree — ``build/**``, ``*.o``, ``*.so``, an in-place extension,
    ``*.egg-info``, a cache — is ignored *by construction*: the guard cannot name a
    path git does not track.

    Compared by content rather than by mtime and both directions at once: a source
    file the fix *added* leaks exactly as much as one it rewrote, and neither
    ``git status`` inside one tree nor a diff against the other commit catches a
    file whose content the copy silently replaced. Test-side paths
    (:func:`is_test_side`) are exempt because the fixed commit's test material is
    the thing that is *supposed* to be there — ``tests/data/golden.json`` as much
    as ``tests/test_x.py``.
    """
    return sorted(
        rel
        for rel in set(tracked)
        if not is_test_side(rel)
        and not any(part in _NOISE_DIRS for part in rel.split("/"))
        and _digest(buggy_dir / rel) != _digest(tree_dir / rel)
    )


# ---------------------------------------------------------------------------
# composing the trees
# ---------------------------------------------------------------------------
@dataclass
class ComposeReport:
    copied_from_fixed: list[str] = field(default_factory=list)
    dropped_fix_deleted: list[str] = field(default_factory=list)
    test_paths: list[str] = field(default_factory=list)


def copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, symlinks=True)


def compose_vanta_tree(
    buggy_dir: Path, fixed_dir: Path, dest: Path, test_paths: Iterable[str]
) -> ComposeReport:
    """``dest`` = ``buggy_dir`` with ``test_paths`` replaced by the fixed versions."""
    copy_tree(buggy_dir, dest)
    report = ComposeReport(test_paths=sorted(test_paths))
    for rel in report.test_paths:
        src = fixed_dir / rel
        target = dest / rel
        if src.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            report.copied_from_fixed.append(rel)
        elif target.exists():
            # The fix deleted this test file. The fixed version's test set does not
            # contain it, so neither does Vanta's tree — otherwise Vanta would keep
            # running a suite the upstream project removed.
            target.unlink()
            report.dropped_fix_deleted.append(rel)
    return report


# ---------------------------------------------------------------------------
# AST-level removal of the hidden tests
# ---------------------------------------------------------------------------
class StripError(RuntimeError):
    """A hidden nodeid could not be located in the tree."""


@dataclass
class StripReport:
    removed_nodeids: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    partially_hidden_functions: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "removed_nodeids": sorted(set(self.removed_nodeids)),
            "deleted_files": sorted(self.deleted_files),
            "partially_hidden_functions": sorted(set(self.partially_hidden_functions)),
            "not_found": sorted(self.not_found),
        }


def nodeid_file(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def nodeid_chain(nodeid: str) -> list[str]:
    """``a/b.py::C::test_m[3]`` → ``['C', 'test_m']`` (parametrisation stripped)."""
    parts = nodeid.split("::")
    chain = parts[1:]
    if chain:
        chain[-1] = re.sub(r"\[.*\]$", "", chain[-1])
    return chain


def strip_hidden_tests(
    tree_dir: Path,
    hidden_nodeids: Iterable[str],
    collected_by_file: dict[str, set[str]],
    delete_files: Iterable[str] = (),
) -> StripReport:
    """Remove the hidden tests from ``tree_dir``, in place.

    A file is deleted outright when *every* test the suite collected in it is
    hidden; otherwise only the named functions go. A function is removed whole
    even when only one of its parametrisations is hidden — a parametrisation
    cannot be excised from source, and keeping the function would leave the
    assertion (the answer) in the tree. A definition removed although the oracle
    did not name *all* of its collected nodeids is recorded in
    ``partially_hidden_functions``, so the shrunken native suite is explainable
    from the split instead of quietly smaller than ``FAIL_TO_PASS`` implies.

    ``delete_files`` names files to remove whole whatever the collected set says.
    That is the collect-flip shape: the file raises at import on the buggy source,
    so the oracle's collection run never saw a single nodeid in it (that *is* the
    flip), and the "everything collected here is hidden" test cannot fire on an
    empty collected set. Every test such a file carries is oracle material by
    construction, and leaving it in place would leave a module that dies at import
    in the tree the Creator reads and in the native baseline's suite.
    """
    report = StripReport()
    by_file: dict[str, list[str]] = {}
    for nodeid in hidden_nodeids:
        by_file.setdefault(nodeid_file(nodeid), []).append(nodeid)

    for rel in sorted({str(rel) for rel in delete_files}):
        path = tree_dir / rel
        if path.is_file():
            path.unlink()
            report.deleted_files.append(rel)
            report.removed_nodeids.extend(by_file.pop(rel, []))

    for rel, nodeids in sorted(by_file.items()):
        path = tree_dir / rel
        if not path.is_file():
            report.not_found.extend(nodeids)
            continue

        collected = collected_by_file.get(rel, set())
        hidden_fns = {tuple(nodeid_chain(n)) for n in nodeids}
        collected_fns = {tuple(nodeid_chain(n)) for n in collected}
        if collected and collected_fns and collected_fns <= hidden_fns:
            path.unlink()
            report.deleted_files.append(rel)
            report.removed_nodeids.extend(nodeids)
            continue

        if (path.parent / "__pycache__").is_dir():
            shutil.rmtree(path.parent / "__pycache__", ignore_errors=True)

        source = path.read_text(encoding="utf-8")
        new_source, removed, missed = _strip_source(source, hidden_fns, rel)
        if missed:
            report.not_found.extend(missed)
        if removed:
            path.write_text(new_source, encoding="utf-8")
            report.removed_nodeids.extend(nodeids)
            # Counted per definition, not per file: a function whose collected
            # nodeids were not all hidden loses a parametrisation the oracle never
            # named, so the native suite shrinks by more than FAIL_TO_PASS explains.
            hidden_counts = Counter(tuple(nodeid_chain(n)) for n in nodeids)
            collected_counts = Counter(tuple(nodeid_chain(n)) for n in collected)
            for chain in removed:
                if hidden_counts[chain] < collected_counts[chain]:
                    report.partially_hidden_functions.append(f"{rel}::{'::'.join(chain)}")

    if report.not_found:
        raise StripError(
            "hidden nodeid(s) not found in the tree, so the oracle would still be "
            f"visible to the Creator: {sorted(set(report.not_found))}"
        )
    return report


def _strip_source(
    source: str, chains: set[tuple[str, ...]], rel: str
) -> tuple[str, list[tuple[str, ...]], list[str]]:
    """Delete the named definitions from ``source``; return the new text.

    Line-span deletion rather than ``ast.unparse``: unparsing rewrites the entire
    file (losing comments and formatting), which would show up as a huge diff
    against the original tree and make the "what did the harness change" question
    unanswerable.
    """
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise StripError(f"{rel}: cannot parse for stripping: {exc}") from exc

    lines = source.splitlines(keepends=True)
    drop: set[int] = set()  # 1-based line numbers
    removed: list[tuple[str, ...]] = []
    missed: list[str] = []
    insert_pass_after: dict[int, str] = {}

    for chain in sorted(chains):
        node = _resolve(module, chain)
        if node is None:
            missed.append(f"{rel}::{'::'.join(chain)}")
            continue
        start = min(
            [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
        )
        end = node.end_lineno or node.lineno
        for lineno in range(start, end + 1):
            drop.add(lineno)
        removed.append(chain)

    # A class left empty by the removal needs a `pass`, or the file stops parsing.
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef) or not node.body:
            continue
        body_lines = {stmt.lineno for stmt in node.body}
        if body_lines and body_lines <= drop:
            indent = " " * (node.body[0].col_offset)
            insert_pass_after[node.lineno] = f"{indent}pass\n"

    out: list[str] = []
    for index, line in enumerate(lines, start=1):
        if index in drop:
            if index in insert_pass_after:
                out.append(insert_pass_after[index])
            continue
        out.append(line)
        if index in insert_pass_after:
            out.append(insert_pass_after[index])

    new_source = "".join(out)
    try:
        ast.parse(new_source)
    except SyntaxError as exc:
        raise StripError(f"{rel}: stripping produced invalid Python: {exc}") from exc
    return new_source, removed, missed


def _resolve(module: ast.Module, chain: tuple[str, ...]) -> Optional[ast.AST]:
    """Walk ``chain`` down the module body and return the named node, if any."""
    body: Optional[list[ast.stmt]] = module.body
    node: Optional[ast.AST] = None
    for name in chain:
        node = None
        for stmt in body or []:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
                stmt.name == name
            ):
                node = stmt
                break
        if node is None:
            return None
        body = getattr(node, "body", None)
    return node
