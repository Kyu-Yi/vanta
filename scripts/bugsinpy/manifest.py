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

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 1

RUNNING = "running"
DONE = "done"
FAILED = "failed"
"""``failed`` is not ``done``: the job broke before it could produce a usable row
(no session id, no readable summary). It is re-run on resume, because a job that
left no evidence has not been done."""

MANIFEST_NAME = "manifest.jsonl"
STATE_NAME = "state.json"


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

    # -- evidence ----------------------------------------------------------
    def append(self, record: dict[str, Any]) -> None:
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=False)
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
        state = self.state()
        state[key] = value
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # ``os.replace`` is atomic on the same filesystem, so a crash during the
        # rewrite leaves either the old cursor or the new one — never a truncated
        # file that would read as "no jobs done" and re-run the whole batch.
        os.replace(tmp, self.state_path)

    def is_done(self, key: str) -> bool:
        # Strictly `done`. `running` and `failed` both mean "no trustworthy row",
        # so both are re-run.
        return self.state().get(key) == DONE

    def pending(self, jobs: Iterable[Any]) -> list[Any]:
        return [job for job in jobs if not self.is_done(job.key)]
