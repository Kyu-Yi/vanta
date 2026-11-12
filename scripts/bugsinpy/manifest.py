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
from typing import Any, Iterable, Mapping, Optional

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

INFRA_VERDICT = "InfraFailure"
"""The driver's verdict for "the environment, not the model" (§5.2, ``label_for``'s
``infra_failure``). Named here as well as in the runner because
:meth:`Manifest.is_done` decides from it under ``--rerun-infra``: a second literal
could not be kept in step with the table the label is read from."""

MANIFEST_NAME = "manifest.jsonl"
STATE_NAME = "state.json"

# ---------------------------------------------------------------------------
# the OUTCOME label, and its one fixed precedence
# ---------------------------------------------------------------------------
# [VANTA-WHY] A row's outcome label is what the run concluded about the patch it
# produced. Two later corrections may overrule it, and neither rewrites the row:
# `manifest.jsonl` is append-only evidence, and a row a paper cites must not be
# able to change under a reader. So the corrections are recorded BESIDE the row —
#   * `label_relabelled`          derived from the row's own artefacts (M3-E,
#                                 `rule m3_e_terminal_final_patch`), on the row.
#   * `label_relabelled_measured` the label was MEASURED by re-running the oracle
#                                 on the reconstructed final patch. It lives in a
#                                 sidecar, `<batch>/relabel/<stem>.json`, never in
#                                 the manifest, so the evidence file is untouched.
# Every reader of the outcome label resolves it through :func:`label_of`, in the
# ONE order below, and reports the SOURCE it used — two tools must never disagree
# about which value a row carries, and a reader must never have to guess whether
# it read the original or a correction. `metrics.py`'s `known_bad` axis is a
# different thing: it is the CANDIDATE's ground truth, which no relabel moves.
RELABEL_DIRNAME = "relabel"
LABEL_KEY = "label"
DERIVED_RELABEL_KEY = "label_relabelled"
MEASURED_RELABEL_KEY = "label_relabelled_measured"

LABEL_SOURCE_MEASURED = "measured_relabel"
LABEL_SOURCE_DERIVED = "derived_relabel"
LABEL_SOURCE_ORIGINAL = "original_label"

# Most specific first — this is a precedence, not a worst-case ranking: a measured
# label beats a derived one, which beats the original the run wrote.
LABEL_PRECEDENCE: tuple[tuple[str, str], ...] = (
    (MEASURED_RELABEL_KEY, LABEL_SOURCE_MEASURED),
    (DERIVED_RELABEL_KEY, LABEL_SOURCE_DERIVED),
    (LABEL_KEY, LABEL_SOURCE_ORIGINAL),
)


def label_of(record: Mapping[str, Any]) -> tuple[str, str]:
    """``(label, source)`` for one row, under the fixed precedence above.

    ``source`` is :data:`LABEL_SOURCE_MEASURED`, :data:`LABEL_SOURCE_DERIVED` or
    :data:`LABEL_SOURCE_ORIGINAL`, so a reader can report which one it used. A row
    that carries no label at all returns ``("", "")`` — an absence, never a value.
    """
    for key, source in LABEL_PRECEDENCE:
        raw = record.get(key)
        # The relabel keys hold a marker block with the label inside it; the plain
        # key holds the label itself. One read covers both shapes.
        value = raw.get(LABEL_KEY) if isinstance(raw, Mapping) else raw
        if isinstance(value, str) and value:
            return value, source
    return "", ""


def stem(key: str) -> str:
    """A filesystem-safe stem for one row's files, derived from its manifest key.

    The ONE definition. ``run_logs/<stem>.log``, ``oracle/<stem>.json``,
    ``issues/<stem>.txt`` and the measured-relabel sidecar
    ``relabel/<stem>.json`` are all named by it, so a reader holding a row's key
    can find every file that belongs to that row.
    """
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(key))


def relabel_dir(batch_dir: Path) -> Path:
    """Where a batch's measured relabels live — ``<batch>/relabel/``."""
    return Path(batch_dir) / RELABEL_DIRNAME


def relabel_path(batch_dir: Path, key: str) -> Path:
    """The sidecar one row's MEASURED relabel lives in."""
    return relabel_dir(batch_dir) / f"{stem(key)}.json"


def measured_relabels(batch_dir: Path) -> dict[str, dict[str, Any]]:
    """``stem -> label_relabelled_measured block``, read from a batch's sidecars.

    A batch with no ``relabel/`` directory is the NORMAL case, not an error: it is
    a batch nobody has re-measured, and it contributes an empty table. A sidecar
    that is unreadable or malformed is skipped rather than raised on, so one bad
    file cannot take a whole report down; every other file is still read.
    """
    table: dict[str, dict[str, Any]] = {}
    directory = relabel_dir(batch_dir)
    if not directory.is_dir():
        return table
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        block = payload.get(MEASURED_RELABEL_KEY)
        if isinstance(block, Mapping):
            table[path.stem] = dict(block)
    return table


def with_measured_relabels(
    records: Iterable[dict[str, Any]], batch_dir: Path
) -> list[dict[str, Any]]:
    """``records`` with each row's measured relabel attached from its sidecar.

    [VANTA-WHY] The measured block lives beside the batch rather than on the row, so
    a reader that only has the rows (the digest, ``write_labels``, the analysis)
    cannot see it. Attaching it here — under :data:`MEASURED_RELABEL_KEY`, the same
    key :func:`label_of` looks for — gives every reader the one view of a row, and
    :func:`label_of` then resolves the precedence in exactly one place.

    A batch with no sidecars returns the rows unchanged (the same objects, no
    copies), so nothing on disk is touched and the cost of the normal case is one
    ``stat``.
    """
    table = measured_relabels(batch_dir)
    if not table:
        return list(records)
    out: list[dict[str, Any]] = []
    for record in records:
        block = table.get(stem(str(record.get("key") or ""))) if isinstance(record, Mapping) else None
        if block is None:
            out.append(record)
        else:
            row = dict(record)
            row[MEASURED_RELABEL_KEY] = block
            out.append(row)
    return out


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

        A row a batch has MEASURED-relabelled carries that correction too
        (:func:`with_measured_relabels`), so a reader that goes through the manifest
        sees the same label every other reader resolves — the sidecar is read here,
        once, rather than by each of them.
        """
        return with_measured_relabels(effective_records(self.records()), self.batch_dir)

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

    def is_done(self, key: str, *, rerun_infra: bool = False) -> bool:
        # Strictly `done`. `running` and `failed` both mean "no trustworthy row",
        # so both are re-run.
        if self.state().get(key) != DONE:
            return False
        record = self.find(key)
        # [VANTA-WHY] Item E6 — a `done` row whose oracle is INCONCLUSIVE (the patch
        # applied but a run wrote no readable report) established no ground truth,
        # so a resume must re-run it rather than skip it. The row stays on disk
        # (append-only); the re-run appends a new row under the same key naming the
        # superseded session, and only then is the key `done` again.
        if oracle_inconclusive(record):
            return False
        # [VANTA-WHY] Item E — `--rerun-infra`: a `done` row whose verdict is
        # `InfraFailure` is a measurement the ENVIRONMENT failed, and the flag is
        # the caller's statement that it wants those re-run rather than skipped.
        # Off (the default), the row is done exactly as before, and no other done
        # row is ever re-run by this flag. Like the E6 case, the old row stays on
        # disk and the re-run names it in `supersedes`.
        if (
            rerun_infra
            and isinstance(record, dict)
            and record.get("verdict") == INFRA_VERDICT
        ):
            return False
        return True

    def pending(self, jobs: Iterable[Any]) -> list[Any]:
        return [job for job in jobs if not self.is_done(job.key)]
