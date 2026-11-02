"""Content-keyed cache for whole suite runs — the M3 speed package (items S1–S4).

[VANTA-WHY] A suite run is the most expensive thing this harness does and the
least variable. Measured through prepare, pandas/1's own suite takes ~487 s per
invocation and every native/oracle gate in a Phase-3 grid runs the SAME bytes
with the SAME command again: `native_original` on a bug's original tree is a
constant of the bug, so one arm-and-seed grid re-measured it once per arm, per
seed, per attempt and per review round — ~20 454 s per seed where ~10 714 s
carries the information.

The key is the whole input: the exact tree bytes, the test command, the
checkpoint, the harness build, the suite's effective environment and the
effective timeout cap. The stored value is the WHOLE pair (both runs, both
reports, the flaky/missing sets, and the in-VM wall times), so a flakiness
reading is always made on the recorded pair and never re-derived from one run.

Four rules make the reuse safe rather than merely fast:

* **A key is the input, so a hit is the same experiment.** Anything that can
  move an outcome is in the key — including the timeout cap, because a run's
  TIMEOUT is a function of the cap and would otherwise be inherited across arms
  as a fact about the tree.
* **A bad entry is never used.** An entry that is unreadable, carries another
  schema version, another key, other material or a payload that does not match
  its own digest is treated as absent: the caller recomputes and overwrites it.
* **Reuse is auditable.** Every entry records who wrote it, and every reader is
  handed the key plus the writer's identity, so a row can say which earlier row
  its native result came from.
* **An unmeasured run is never stored or served** (M3 item U2). A pair whose
  runs measured nothing — no report, a cap, a kill — is not a measurement, and
  caching one would serve an empty suite as if it were "no regressions". The
  write side refuses to store one and the read side refuses to serve one, so an
  entry written before the rule existed is repaired rather than believed. The
  Rust half of the same rule is item U1.

Where it lives is a deliberate choice (Nathan, speed-package mandate): under
`runs/` — gitignored — read host-side only as data, never inside a
`vanta_sessions` directory and never inside a sandbox.

This module is deliberately free of any import of :mod:`suite` (which imports
it): it deals in JSON-shaped material and payloads so the serialisation of the
harness's own result types stays in the harness, next to those types.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

SCHEMA_VERSION = 1
"""The entry format's version, and part of every key.

[VANTA-WHY] A key that did not carry the format version could be satisfied by an
entry written by an older shape of this module: the tree hash and the test
command would still agree, the payload would parse as *something*, and a field
added to the payload later would come back missing — read as "no flakiness"
rather than as "this entry predates the field". Putting the version in the key
material makes a format change unable to read old entries at all: the key
changes, the entries are never found, and nothing has to be migrated or purged.
"""

KIND_SUITE_PAIR = "suite_pair"
"""The kind discriminator for the Python harness's `SuitePair` entries.

[VANTA-WHY] The Rust in-session gate stores a single `NativeSuiteReport` under
its own kind. Both live in one directory and both hash a tree, so a shared key
space without a discriminator would let one kind's entry answer the other's
lookup if the two ever hashed the same bytes.
"""

CACHE_DIR_ENV = "VANTA_SUITE_CACHE_DIR"
"""Overrides the cache root. Set by the harness so a batch can point every
process of a run at one directory; unset uses the repo's own `runs/`."""

DISABLE_ENV = "VANTA_NO_CACHE"
"""Set by `--no-cache` on `run`, `queue` and `prepare`.

[VANTA-WHY] The flag forces a FRESH run; it does not make the cache invisible.
`read_entry` is still consulted, and the row records the lookup in its own
`cache` list — one entry per side, each with `state` (`off:no_cache`) and
`would_hit` — so a cached result can be checked against a fresh one on demand,
which is the only way to verify the cache without a second experiment design.
The fresh payload then REPLACES the entry (M3 item U2), so a bad entry cannot
outlive the one invocation that was asked to check it; a fresh run that measured
nothing replaces nothing, because `run_state` refuses it on both sides.
"""

# Directories whose contents are a run's own leavings or another tool's state,
# never part of the tree a suite reads. `.pytest_cache` and friends are here
# because a graded suite run does NOT pass `-p no:cacheprovider` (only
# `suite.collect_command` does), so the run writes them into the host tree and
# `suite._clean_tree_noise` deletes them afterwards; hashing them would make the
# key depend on whether an earlier run had been cleaned, which is not an input.
NOISE_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
    }
)

NOISE_SUFFIXES = (".pyc", ".pyo")
"""Byte-code, which the interpreter regenerates from the source it was built
from — a cache hit and a fresh run cannot differ because of it.

This is deliberately NOT `compact.NOISE_SUFFIXES`: that set also skips
`_manifest.json` because the *rerun reader* skips it, and a manifest is a file
the suite itself may legitimately read. A cache key has to describe every byte
the run can see, so it skips only what cannot be read as source.
"""

_HARNESS_FILES = ("suite.py", "shuru.py", "suite_cache.py", "compact.py")
"""The harness files that decide how a suite is run.

[VANTA-WHY] The stored value describes a run performed by THIS code, so a change
to the code that builds the shell script, resolves the checkpoint or reads the
report must invalidate every entry rather than be served from one. Hashing the
sources themselves is the honest version of a "harness build stamp" on a side
that has no compiled binary; the Rust side uses `VANTA_BUILD_COMMIT` instead.
"""


def repo_root() -> Path:
    """The repository root, from this file's own location."""
    return Path(__file__).resolve().parents[2]


def cache_dir() -> Path:
    """The cache root, `runs/bugsinpy/suite_cache` unless overridden.

    [VANTA-WHY] Under `runs/` because that is gitignored (Nathan's rule) and
    shared by every batch of every arm and seed — the whole point of the item is
    that the same bytes are not measured twice, and a per-batch directory would
    make "across batches" impossible.
    """
    override = os.environ.get(CACHE_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return repo_root() / "runs" / "bugsinpy" / "suite_cache"


def cache_disabled() -> bool:
    """Whether `--no-cache` asked for a fresh run."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def tree_fingerprint(root: Path) -> str:
    """A sha256 over every file a suite running in `root` can read.

    Per-file digests rather than one concatenation of the contents: the tree can
    hold multi-megabyte files, and the per-file form is what makes the walk's own
    ordering (sorted relative paths) the only ordering that matters.

    An unreadable file is hashed as unreadable rather than raising: the caller is
    inside a run that is about to fail for its own reason, and a cache key that
    could raise would turn a permission problem into a harness crash.
    """
    digest = hashlib.sha256()
    for rel, path in _tree_files(root):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        try:
            data = path.read_bytes()
        except OSError:
            digest.update(b"\0unreadable\0")
            continue
        digest.update(hashlib.sha256(data).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _tree_files(root: Path) -> list[tuple[str, Path]]:
    """`(relative path, path)` for every readable-file candidate, sorted."""
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    found: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in NOISE_DIRS)
        for name in sorted(filenames):
            if name.endswith(NOISE_SUFFIXES):
                continue
            path = Path(dirpath) / name
            found.append((path.relative_to(root).as_posix(), path))
    found.sort(key=lambda pair: pair[0])
    return found


def harness_build() -> str:
    """The Python harness's build stamp, its suite-running sources hashed."""
    override = os.environ.get("VANTA_HARNESS_BUILD", "").strip()
    if override:
        return override
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _HARNESS_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(hashlib.sha256((here / name).read_bytes()).digest())
        except OSError:
            digest.update(b"\0missing\0")
        digest.update(b"\n")
    return "py:" + digest.hexdigest()[:32]


def key_of(material: Mapping[str, object]) -> str:
    """The cache key: sha256 of the canonical rendering of `material`.

    Canonical because the key is compared as a string and the material is
    re-derived on every run: `sort_keys` plus the compact separators make the
    rendering of one mapping unique, so two processes that mean the same
    experiment cannot produce two keys.
    """
    blob = json.dumps(dict(material), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def payload_digest(payload: object) -> str:
    """The digest of a stored payload, recorded with it and re-checked on read."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def entry_path(key: str) -> Path:
    path = cache_dir() / key[:2] / f"{key}.json"
    return path


def read_entry(material: Mapping[str, object], key: str) -> Optional[dict]:
    """The validated entry for `material`, or `None`.

    [VANTA-WHY] `None` covers corruption and mismatch in one place, because both
    have the same consequence — the caller recomputes — and a caller that had to
    tell them apart would have one more branch to get wrong. Every check is a
    reason the entry cannot describe THIS experiment:

    * unreadable or unparseable — a truncated write, a hand-edited file;
    * another `schema` — an entry from a different format of this module;
    * another `key` — the file is not where its own content says it belongs;
    * other `material` — the key was recomputed from different inputs, so the
      payload describes a different experiment even though it was stored here;
    * a `payload` that does not match its own `payload_digest` — the file was
      changed after it was written.
    """
    try:
        data = json.loads(entry_path(key).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema") != SCHEMA_VERSION:
        return None
    if data.get("key") != key:
        return None
    if data.get("material") != dict(material):
        return None
    if data.get("payload_digest") != payload_digest(data.get("payload")):
        return None
    return data


def write_entry(entry: dict) -> Path:
    """Write `entry` atomically, so a reader never sees a torn file.

    [VANTA-WHY] `os.replace` on the same filesystem is the whole guarantee: the
    queue runs several bugs at once, and a crashed or killed writer must leave
    either the previous entry or none, never a half-written one that a later
    lookup would find and (correctly) refuse — losing a usable entry to a
    crash of an unrelated process.
    """
    path = entry_path(entry["key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(entry, handle)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_entry(
    material: Mapping[str, object], payload: object, *, origin: Optional[str]
) -> dict:
    """The entry document for one stored result."""
    return {
        "schema": SCHEMA_VERSION,
        "kind": material.get("kind"),
        "key": key_of(material),
        "material": dict(material),
        "created": utc_now(),
        "created_by": origin or "",
        "payload_digest": payload_digest(payload),
        "payload": payload,
    }
