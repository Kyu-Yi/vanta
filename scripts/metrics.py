#!/usr/bin/env python3
"""Classify Vanta sessions as true_catch / spurious_reject / false_approval.

[WHY] A Review-mode `Rejected` verdict is not evidence about the model on its
own. The Attacker writes different tests every run, so a rejecting test usually
does not exist in the paired "good" run — pairing runs cannot answer the
question. What CAN answer it is re-running the exact rejecting signals against a
known-good tree, which is what `vanta_m3 --rerun-signals` does and what this
script reads back:

    <sessions_dir>/<sid>_session.jsonl          the run's verdict + signal records
    <sessions_dir>/<sid>_signal_reruns.json     per-signal, bad-tree vs good-tree

This script NEVER executes code. It only reads files and counts.

Buckets
-------
true_catch         Rejected, and at least one rejecting signal fails on the bad
                   tree and passes (or is absent) on the good tree.
spurious_reject    Rejected, every signal could be evaluated, and none of them
                   discriminates.
false_approval     Approved on a patch whose ground-truth label says it is bad.
undetermined       Rejected, but no signal could be evaluated (e.g. a legacy v0
                   session whose static findings were never recorded). Reported
                   separately so it is never silently folded into
                   spurious_reject.
approved_unlabeled Approved with no ground-truth label, so nothing is claimed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

SUPPORTED_SCHEMA_VERSION = 1

TRUE_CATCH = "true_catch"
SPURIOUS_REJECT = "spurious_reject"
FALSE_APPROVAL = "false_approval"
UNDETERMINED = "undetermined"
APPROVED_UNLABELED = "approved_unlabeled"

APPROVED_DECISIONS = {"AllTestsPassed", "MaxAttemptsReached"}


# ============================================================================
# Loading
# ============================================================================


@dataclass
class SessionData:
    """One session's on-disk facts, plus every warning raised while reading it."""

    session_id: str
    records: list[dict[str, Any]]
    schema_version: Optional[int]
    warnings: list[str] = field(default_factory=list)
    reruns: Optional[dict[str, Any]] = None

    @property
    def schema_label(self) -> str:
        if self.schema_version is None:
            return "v0 (legacy)"
        return f"v{self.schema_version}"

    def field(self, phase: str, key: str) -> Optional[str]:
        """Value of `key` from the `key=value` summary of the given phase.

        [WHY] Values may contain spaces (`test_cmd=env PYTHONPATH=src ... -m
        pytest -q`), so a value ends at the next ` key=` token rather than at the
        next space.
        """
        for record in self.records:
            if record.get("phase") != phase:
                continue
            value = summary_field(record.get("summary", ""), key)
            if value is not None:
                return value
        return None


def summary_field(summary: str, key: str) -> Optional[str]:
    """Extract `key=value` from a single-line space-separated summary."""
    needle = f"{key}="
    start = 0
    while True:
        index = summary.find(needle, start)
        if index == -1:
            return None
        # Only accept the key at the start of the string or after a space, so
        # `mode=` does not match inside `patch_mode=`.
        if index == 0 or summary[index - 1] == " ":
            value_start = index + len(needle)
            value_end = _next_key_start(summary, value_start)
            return summary[value_start:value_end]
        start = index + 1


def _next_key_start(summary: str, value_start: int) -> int:
    """Index where the value ends: the next ' ' that begins a `key=` token."""
    index = value_start
    while True:
        space = summary.find(" ", index)
        if space == -1:
            return len(summary)
        rest = summary[space + 1 :]
        name = ""
        for char in rest:
            if char.isalnum() or char == "_":
                name += char
            else:
                break
        if name and rest[len(name) : len(name) + 1] == "=":
            return space
        index = space + 1


def read_schema_version(records: list[dict[str, Any]]) -> Optional[int]:
    """The session's schema version, or None when the marker is absent (v0).

    [WHY] Absence is a DIFFERENT fact from "an older version of the current
    schema": v0 files were written before the marker existed and lack fields
    (failure output, static findings). Treating absence as "current" would make
    the script invent data it does not have.
    """
    for record in records:
        if "schema_version" in record and record["schema_version"] is not None:
            return int(record["schema_version"])
    return None


def load_session(sessions_dir: str, session_id: str) -> SessionData:
    """Load one session's JSONL + (optional) signal_reruns.json."""
    path = os.path.join(sessions_dir, f"{session_id}_session.jsonl")
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"{path}:{line_number}: not valid JSON ({error})"
                ) from error

    schema_version = read_schema_version(records)
    warnings: list[str] = []
    if schema_version is None:
        warnings.append(
            f"{session_id}: no schema_version marker — treating as legacy v0. "
            "Fields v0 never recorded (per-test failure output, new static "
            "findings) are reported as unavailable, not guessed."
        )
    elif schema_version != SUPPORTED_SCHEMA_VERSION:
        raise SystemExit(
            f"{session_id}: schema_version={schema_version} but this script "
            f"understands {SUPPORTED_SCHEMA_VERSION}. Refusing to score a "
            "session written by a schema it does not know."
        )

    data = SessionData(
        session_id=session_id,
        records=records,
        schema_version=schema_version,
        warnings=warnings,
    )

    reruns_path = os.path.join(sessions_dir, f"{session_id}_signal_reruns.json")
    if os.path.exists(reruns_path):
        with open(reruns_path, encoding="utf-8") as handle:
            data.reruns = json.load(handle)
    else:
        data.warnings.append(
            f"{session_id}: no {os.path.basename(reruns_path)} — no signal was "
            "re-run, so discrimination cannot be established."
        )
    return data


def discover_sessions(sessions_dir: str) -> list[str]:
    """Every session id that has a session JSONL in the directory."""
    ids = []
    for name in sorted(os.listdir(sessions_dir)):
        if name.endswith("_session.jsonl"):
            ids.append(name[: -len("_session.jsonl")])
    return ids


# ============================================================================
# Classification
# ============================================================================


@dataclass
class Verdict:
    session_id: str
    decision: str
    schema_label: str
    bucket: str
    signals_total: int
    signals_deciding: int
    signals_undecidable: int
    label: Optional[bool] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def explanation(self) -> str:
        if self.bucket == TRUE_CATCH:
            return f"{self.signals_deciding} discriminating signal(s)"
        if self.bucket == SPURIOUS_REJECT:
            return f"all {self.signals_total} signal(s) evaluated, none discriminates"
        if self.bucket == UNDETERMINED:
            if self.signals_total == 0:
                return "no rejecting signal could be reconstructed"
            return f"all {self.signals_undecidable} signal(s) undecidable"
        if self.bucket == FALSE_APPROVAL:
            return "approved on a patch labelled known-bad"
        if self.label is False:
            return "approved, and the label agrees it is good"
        return "approved, no ground-truth label"


def classify(
    data: SessionData, known_bad: Optional[bool]
) -> Verdict:
    """Bucket one session. Never guesses: an un-evaluated signal is not a pass."""
    decision = data.field("run.summary", "final_decision") or "unknown"
    reruns = data.reruns

    signals: list[dict[str, Any]] = []
    if isinstance(reruns, dict):
        raw = reruns.get("signals")
        if isinstance(raw, list):
            signals = [entry for entry in raw if isinstance(entry, dict)]

    deciding = [s for s in signals if s.get("discriminates") is True]
    undecidable = [s for s in signals if s.get("discriminates") is None]

    if decision == "Rejected":
        if deciding:
            bucket = TRUE_CATCH
        elif signals and not undecidable:
            bucket = SPURIOUS_REJECT
        else:
            bucket = UNDETERMINED
    elif decision in APPROVED_DECISIONS:
        if known_bad is True:
            bucket = FALSE_APPROVAL
        elif known_bad is False:
            bucket = APPROVED_UNLABELED  # approved and the label agrees
        else:
            bucket = APPROVED_UNLABELED
    else:
        # PatchApplicationFailed / HarnessFailure / unknown: not a verdict about
        # the candidate, so nothing is claimed either way.
        bucket = UNDETERMINED

    return Verdict(
        session_id=data.session_id,
        decision=decision,
        schema_label=data.schema_label,
        bucket=bucket,
        signals_total=len(signals),
        signals_deciding=len(deciding),
        signals_undecidable=len(undecidable),
        label=known_bad,
        warnings=list(data.warnings),
    )


# ============================================================================
# Reporting
# ============================================================================


def render_table(verdicts: list[Verdict]) -> str:
    header = (
        f"{'SESSION':<10} {'VERDICT':<12} {'SCHEMA':<12} {'SIGNALS':<13} "
        f"{'BUCKET':<18} NOTE"
    )
    lines = [header, "-" * len(header)]
    for verdict in verdicts:
        signals = f"{verdict.signals_deciding}/{verdict.signals_total}"
        lines.append(
            f"{verdict.session_id[:8]:<10} {verdict.decision:<12} "
            f"{verdict.schema_label:<12} {signals:<13} {verdict.bucket:<18} "
            f"{verdict.explanation}"
        )
    lines.append("")
    lines.append("TOTALS")
    for bucket in (
        TRUE_CATCH,
        SPURIOUS_REJECT,
        FALSE_APPROVAL,
        UNDETERMINED,
        APPROVED_UNLABELED,
    ):
        count = sum(1 for verdict in verdicts if verdict.bucket == bucket)
        lines.append(f"  {bucket:<18} {count}")
    lines.append("")
    scored = sum(1 for v in verdicts if v.bucket in (TRUE_CATCH, SPURIOUS_REJECT))
    seen = sum(1 for v in verdicts if v.decision == "Rejected")
    lines.append(f"  {'scored rejections':<18} {scored}")
    lines.append(f"  {'rejections seen':<18} {seen}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score Vanta sessions as true_catch / spurious_reject / false_approval."
    )
    parser.add_argument("--sessions-dir", default="vanta_sessions")
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Score only this session id (repeatable). Default: every session in the dir.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="JSON file mapping session_id -> {\"known_bad\": true|false} for approvals.",
    )
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Also write the per-session detail to this file.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.sessions_dir):
        print(f"error: {args.sessions_dir} is not a directory", file=sys.stderr)
        return 1

    labels: dict[str, Any] = {}
    if args.labels:
        with open(args.labels, encoding="utf-8") as handle:
            labels = json.load(handle)

    session_ids = args.session or discover_sessions(args.sessions_dir)
    if not session_ids:
        print(f"error: no *_session.jsonl under {args.sessions_dir}", file=sys.stderr)
        return 1

    verdicts: list[Verdict] = []
    for session_id in session_ids:
        data = load_session(args.sessions_dir, session_id)
        label = labels.get(session_id) or {}
        known_bad = label.get("known_bad")
        verdicts.append(classify(data, known_bad))

    print(render_table(verdicts))

    warnings = [w for verdict in verdicts for w in verdict.warnings]
    if warnings:
        print("\nWARNINGS")
        for warning in warnings:
            print(f"  ! {warning}")

    if args.json_out:
        detail = [
            {
                "session_id": verdict.session_id,
                "decision": verdict.decision,
                "schema_version": verdict.schema_label,
                "bucket": verdict.bucket,
                "signals_total": verdict.signals_total,
                "signals_deciding": verdict.signals_deciding,
                "signals_undecidable": verdict.signals_undecidable,
                "warning": verdict.warnings,
            }
            for verdict in verdicts
        ]
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(detail, handle, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
