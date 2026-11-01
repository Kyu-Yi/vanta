"""The batch manifest and its resume cursor (design §7.3, §7.4).

[VANTA-WHY] Two files, not one, because they answer different questions and need
different durability. ``manifest.jsonl`` is the **evidence**: one JSON object per
line, append-only, never rewritten. A batch that produced 40 rows has to read as
40 rows after a crash, and a row that a paper cites must not be able to change
under a reader. ``state.json`` is the **cursor**: a small
``{key: done|running|failed}`` map that is rewritten at every job boundary and
whose only job is to let a resumed run skip work that is already finished. Folding
the cursor into the evidence would mean rewriting the evidence.

[VANTA-WHY] ``running`` is written *before* the work and ``done`` only *after* the
oracle record is flushed. That ordering IS the resume guarantee: a process killed
mid-run leaves the key at ``running``, so the next start re-runs that job rather
than skipping a row whose artifacts may be half-written. A job is idempotent —
every Vanta run gets a fresh random session directory (``fresh_session_id``), so
re-running costs time and cannot corrupt an earlier record.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 1

MANIFEST_LOCK_NAME = ".manifest.lock"


class BatchLocked(RuntimeError):
    """A second invocation tried to own a batch directory that is already owned."""


class FileLock:
    """An ``flock`` on a file, held for the life of a ``with`` block (M3 item 13).

    [VANTA-WHY] The batch directory is shared by every job the queue runs at once,
    and three of its files are read-modify-write: ``state.json`` (read, change one
    key, replace), ``manifest.jsonl`` (append) and ``failures.jsonl`` (append). Two
    workers interleaving those writes lose updates — the cursor forgets a
    ``running`` mark and a resume re-runs the job, or two failure lines interleave
    into one line no reader can parse.

    ``flock`` rather than a ``threading.Lock`` because the unit of exclusion is the
    *directory*, not the process: a second ``queue`` invocation, or a ``run``
    started by hand against the same batch, is a different process and has to be
    excluded too. ``flock`` is per open file description on macOS and BSD, so two
    threads that each hold their own :class:`FileLock` on one path exclude each
    other exactly as two processes do — one object per caller, not one per path.

    The lock file is created on first :meth:`acquire`, never in ``__init__``, so a
    reader that only reads does not write a dotfile into the batch.
    """

    def __init__(self, path: Path, *, held_message: Optional[str] = None) -> None:
        self.path = Path(path)
        self.held_message = held_message
        self._fd: Optional[int] = None

    def acquire(self, *, blocking: bool = True) -> bool:
        """Take the lock. ``False`` means it was held and ``blocking`` was off."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "FileLock":
        # [VANTA-WHY] C2 — a lock that could not be taken must not be skipped. The
        # caller reads ``with self.lock:`` as "I hold the directory", and running the
        # covered read-modify-write unlocked is exactly the lost-update this class
        # exists to prevent; so a failed ``acquire`` is a refusal here, not a no-op.
        if not self.acquire():
            raise BatchLocked(
                self.held_message
                or f"the lock {self.path} could not be taken; refusing to proceed without it"
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


RUNNING = "running"
DONE = "done"
FAILED = "failed"
"""``failed`` is not ``done``: the job broke before it could produce a usable row
(no session id, no readable summary). It is re-run on resume, because a job that
left no evidence has not been done."""

MANIFEST_NAME = "manifest.jsonl"
STATE_NAME = "state.json"


def oracle_inconclusive(record: Optional[dict[str, Any]]) -> bool:
    """True when a row's own oracle recorded an INCONCLUSIVE grading (item E6).

    [VANTA-WHY] Item E6 — an inconclusive grading (the patch applied but a run
    wrote no readable report) established no ground truth. Such a row is not
    trustworthy evidence, so `Manifest.is_done` treats it as not-done and a resume
    re-runs the job; the re-run appends a new row naming the superseded session.
    A row without an oracle block (a `failed` job, a non-run row) is not
    inconclusive.
    """
    if not isinstance(record, dict):
        return False
    oracle = record.get("oracle")
    return bool(isinstance(oracle, dict) and oracle.get("inconclusive"))


def effective_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows a reader may count: the last per key, minus any superseded session.

    [VANTA-WHY] F3 — ``manifest.jsonl`` is append-only, so the same evidence appears
    two ways that a *report* must not double-count: a resumed or re-run job appends a
    second row under the same ``key``, and a re-run of an INCONCLUSIVE job names the
    session it replaces in ``supersedes``. ``analyse.load_manifest`` and
    ``metrics.oracle_by_session`` already dropped these, but every other reader (the
    batch digest, ``write_labels``, the pool harvest, the replay selector) read the
    raw rows and counted both, so a superseded session could appear in the digest and
    in the labels file. This is the ONE definition of "the rows that are current";
    every reader calls it rather than re-deriving the rule. It is idempotent, so a
    reader may apply it to rows that were already filtered.

    ``key`` is the identity of a job (:func:`job_key`), so the last row for a key is
    the current one — the same rule :meth:`Manifest.find` uses for the cursor. A row
    with no key has no identity to replace it, so it is kept under a synthetic key (as
    the analyse reader has always done); ``supersedes`` names a *session*, not a key,
    so a named session is dropped even when its own key is not the one the re-run was
    appended under (§7.3, item E6).
    """
    latest: dict[str, dict[str, Any]] = {}
    superseded: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        supersedes = record.get("supersedes")
        if isinstance(supersedes, dict) and supersedes.get("session_id"):
            superseded.add(str(supersedes["session_id"]))
        key = str(record.get("key") or "")
        latest[key if key else f"#{len(latest)}"] = record
    rows = [latest[key] for key in sorted(latest)]
    return [row for row in rows if str(row.get("session_id") or "") not in superseded]


def job_key(
    *,
    batch_id: str,
    bug: str,
    config: str,
    mode: str,
    seed: int,
    candidate: Optional[str] = None,
    show_hints: bool = False,
) -> str:
    """The stable identity of one job, and therefore of one resume decision.

    ``candidate`` is empty for repair jobs. It exists for review jobs because a
    review pass is one session *per candidate* (§7.1) — without it, the second
    candidate of a bug would shadow the first in ``state.json`` and be skipped on
    resume.

    ``show_hints`` is appended only when it is set, because a row that sends the
    report's comments is a run of a *different prompt* (§2.4.4): sharing a key with the
    default row would let a resume skip one as though it were the other, which is the
    conflation ``candidate`` is already here to prevent. A suffix rather than a seventh
    column so that every key a default run has already written stays exactly as it was —
    the with-hints ablation is additive, not a re-keying of the corpus.
    """
    key = "#".join([batch_id, bug, config, mode, str(seed), candidate or ""])
    return f"{key}#hints" if show_hints else key


class Manifest:
    """Append-only evidence plus an atomically replaced cursor."""

    def __init__(self, batch_dir: Path):
        self.batch_dir = Path(batch_dir)
        self.path = self.batch_dir / MANIFEST_NAME
        self.state_path = self.batch_dir / STATE_NAME
        # Item 13 — the batch directory's own exclusion, taken for the two
        # read-modify-write operations below and for nothing else: a job's work is
        # minutes long and must not hold the directory while it runs.
        self.lock = FileLock(self.batch_dir / MANIFEST_LOCK_NAME)

    # -- evidence ----------------------------------------------------------
    def append(self, record: dict[str, Any]) -> None:
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=False)
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                # Flushed *and* fsynced before the caller may mark the job done: a row
                # that exists only in the page cache is not evidence a reboot can read.
                handle.flush()
                os.fsync(handle.fileno())

    def records(self) -> list[dict[str, Any]]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn last line means the process died mid-write. Dropping it
                # (rather than raising) keeps the batch readable; the cursor still
                # says `running` for that job, so it is re-run rather than trusted.
                continue
        return out

    def effective_records(self) -> list[dict[str, Any]]:
        """``records()`` filtered to the rows that are current (F3).

        See :func:`effective_records`: the raw list is every row ever appended, so a
        report built from it counts a superseded or re-run job twice.
        """
        return effective_records(self.records())

    def find(self, key: str) -> Optional[dict[str, Any]]:
        for record in reversed(self.records()):
            if record.get("key") == key:
                return record
        return None

    # -- cursor ------------------------------------------------------------
    def state(self) -> dict[str, str]:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}

    def mark(self, key: str, value: str) -> None:
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            state = self.state()
            state[key] = value
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            # ``os.replace`` is atomic on the same filesystem, so a crash during the
            # rewrite leaves either the old cursor or the new one — never a truncated
            # file that would read as "no jobs done" and re-run the whole batch.
            #
            # The read above has to be inside the lock (item 13): otherwise two jobs
            # finishing at once each read the cursor without the other's key, and the
            # second replace drops the first job's ``done`` — which a resume then
            # re-runs, producing a duplicate row for work that is already on disk.
            os.replace(tmp, self.state_path)

    def is_done(self, key: str) -> bool:
        # Strictly `done`. `running` and `failed` both mean "no trustworthy row",
        # so both are re-run.
        if self.state().get(key) != DONE:
            return False
        # [VANTA-WHY] Item E6 — a `done` row whose oracle is INCONCLUSIVE (the patch
        # applied but a run wrote no readable report) established no ground truth,
        # so a resume must re-run it rather than skip it. The row stays on disk
        # (append-only); the re-run appends a new row under the same key naming the
        # superseded session, and only then is the key `done` again.
        return not oracle_inconclusive(self.find(key))

    def pending(self, jobs: Iterable[Any]) -> list[Any]:
        return [job for job in jobs if not self.is_done(job.key)]
