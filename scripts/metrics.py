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
true_catch         Rejected on a patch labelled known-bad, and at least one
                   rejecting signal fails on the bad tree and passes (or is
                   absent) on the good tree.
spurious_reject    Rejected on a patch labelled known-bad, every signal could be
                   evaluated, and none of them discriminates.
false_reject       Rejected on a patch whose ground-truth label says it is GOOD.
                   [WHY] This is decided by the LABEL, not by the discrimination
                   table: when the candidate is itself the known-good code (B v2
                   applies the fixed patch), every signal fires on the good tree
                   too, so the table can only say "does not discriminate". A
                   rejection there is simply wrong, and reading it as
                   spurious_reject would hide a wrong verdict behind a
                   measurement artefact.
false_approval     Approved on a patch whose ground-truth label says it is bad.
true_approval      Approved on a patch whose ground-truth label says it is good.
unresolved         The run stopped at its attempt budget without an approved
                   patch (`MaxAttemptsReached`). The Governor returns that only
                   after a round that did not pass, and the Gatekeeper — the only
                   component that approves — runs solely on `AllTestsPassed`, so
                   nothing this run produced was ever accepted. It is "gave up",
                   never an approval: counted on its own so it can never inflate
                   an approval bucket, whatever ground-truth label it carries.
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
FALSE_REJECT = "false_reject"
FALSE_APPROVAL = "false_approval"
TRUE_APPROVAL = "true_approval"
UNRESOLVED = "unresolved"
UNDETERMINED = "undetermined"
APPROVED_UNLABELED = "approved_unlabeled"

# [WHY] `AllTestsPassed` alone. `MaxAttemptsReached` used to live here and was
# bucketed as an approval; it is the opposite — the Creator exhausted the budget
# without an approved patch (see the `unresolved` bucket).
APPROVED_DECISIONS = {"AllTestsPassed"}
UNRESOLVED_DECISIONS = {"MaxAttemptsReached"}


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


def final_decision(records: list[dict[str, Any]]) -> Optional[str]:
    """The run's `final_decision`, or None when the run summary is absent.

    [WHY] Both the bucketing and the "is there anything to re-run?" question
    read this one field, so they can never disagree about what the run decided.
    """
    for record in records:
        if record.get("phase") != "run.summary":
            continue
        value = summary_field(record.get("summary", ""), "final_decision")
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
    """Index where the value ends: the next ' ' that begins a `key=` token.

    [WHY] A key is `[a-z_][a-z0-9_]*` — LOWERCASE — because every key the
    producer writes is lowercase, while the variable a `test_cmd` assigns is
    uppercase by convention (`env PYTHONPATH=src <python> -m pytest -q`).
    Treating any `NAME=` as a key would cut that value at `env` and make the
    rerun re-execute the wrong command. This mirrors
    `rerun.rs::next_key_start` byte for byte.
    """
    index = value_start
    while True:
        space = summary.find(" ", index)
        if space == -1:
            return len(summary)
        rest = summary[space + 1 :]
        length = 0
        while length < len(rest) and _is_key_char(rest[length], first=(length == 0)):
            length += 1
        if length and rest[length : length + 1] == "=":
            return space
        index = space + 1


def _is_key_char(char: str, first: bool) -> bool:
    if ("a" <= char <= "z") or char == "_":
        return True
    return not first and char.isdigit()


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
    elif final_decision(records) == "Rejected":
        # [WHY] Only a rejection produces signals worth re-running. An approval
        # (or a run that stopped without a verdict) was never owed a probe, so
        # warning about the absent file would blame the session for a measurement
        # nobody needed.
        data.warnings.append(
            f"{session_id}: no {os.path.basename(reruns_path)} — the session was "
            "rejected but no signal was re-run, so discrimination cannot be "
            "established."
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
        if self.bucket == FALSE_REJECT:
            return "rejected on a patch labelled known-good"
        if self.bucket == UNDETERMINED:
            if self.signals_total == 0:
                return "no rejecting signal could be reconstructed"
            return f"all {self.signals_undecidable} signal(s) undecidable"
        if self.bucket == FALSE_APPROVAL:
            return "approved on a patch labelled known-bad"
        if self.bucket == TRUE_APPROVAL:
            return "approved on a patch labelled known-good"
        if self.bucket == UNRESOLVED:
            return "attempt budget exhausted with no approved patch (not an approval)"
        return "approved, no ground-truth label"


def classify(
    data: SessionData, known_bad: Optional[bool]
) -> Verdict:
    """Bucket one session. Never guesses: an un-evaluated signal is not a pass."""
    decision = final_decision(data.records) or "unknown"
    reruns = data.reruns

    signals: list[dict[str, Any]] = []
    if isinstance(reruns, dict):
        raw = reruns.get("signals")
        if isinstance(raw, list):
            signals = [entry for entry in raw if isinstance(entry, dict)]

    deciding = [s for s in signals if s.get("discriminates") is True]
    undecidable = [s for s in signals if s.get("discriminates") is None]

    if decision == "Rejected":
        if known_bad is False:
            # [WHY] The label decides, not the discrimination table. When the
            # candidate IS the known-good code (B v2 applies the fixed patch),
            # every signal fires on the good tree too, so no signal can ever
            # discriminate — the table would call a wrong rejection "spurious".
            bucket = FALSE_REJECT
        elif deciding:
            bucket = TRUE_CATCH
        elif signals and not undecidable:
            bucket = SPURIOUS_REJECT
        else:
            bucket = UNDETERMINED
    elif decision in UNRESOLVED_DECISIONS:
        # [WHY] The attempt budget ran out with no approved patch. The Governor
        # only returns this after a round whose verdict was not `Passed` (a
        # `Passed` round returns AllTestsPassed first), and the Gatekeeper — the
        # only component that emits `Approved` — runs solely on AllTestsPassed.
        # So nothing here was accepted, and no ground-truth label can make it an
        # approval: it is unresolved whatever the label says.
        bucket = UNRESOLVED
    elif decision in APPROVED_DECISIONS:
        if known_bad is True:
            bucket = FALSE_APPROVAL
        elif known_bad is False:
            bucket = TRUE_APPROVAL
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
        FALSE_REJECT,
        FALSE_APPROVAL,
        TRUE_APPROVAL,
        UNRESOLVED,
        UNDETERMINED,
        APPROVED_UNLABELED,
    ):
        count = sum(1 for verdict in verdicts if verdict.bucket == bucket)
        lines.append(f"  {bucket:<18} {count}")
    lines.append("")
    scored = sum(
        1
        for v in verdicts
        if v.bucket in (TRUE_CATCH, SPURIOUS_REJECT, FALSE_REJECT)
    )
    seen = sum(1 for v in verdicts if v.decision == "Rejected")
    lines.append(f"  {'scored rejections':<18} {scored}")
    lines.append(f"  {'rejections seen':<18} {seen}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score Vanta sessions as true_catch / spurious_reject / false_reject / "
            "false_approval / true_approval."
        )
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
        help=(
            "JSON file mapping session_id -> {\"known_bad\": true|false} ground "
            "truth. Drives false_reject / false_approval / true_approval; an "
            "unlabelled session claims nothing about approvals. When given, only "
            "the labelled sessions are reported unless --all is passed."
        ),
    )
    parser.add_argument(
        "--all",
        dest="include_all",
        action="store_true",
        help=(
            "With --labels, report every session in the directory, not just the "
            "labelled ones."
        ),
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

    if args.labels and not args.include_all:
        # [WHY] A label file is the study's declared evaluation set. Reporting
        # unlabelled sessions next to it invites a reader to add rows whose
        # ground truth nobody checked into a count that is supposed to be
        # label-driven. `--all` is the explicit opt-in for the wider view.
        session_ids = [sid for sid in session_ids if sid in labels]
        if not session_ids:
            print(
                f"error: none of the {len(labels)} session(s) in {args.labels} has "
                f"a *_session.jsonl under {args.sessions_dir} (pass --all to "
                "report unlabelled sessions)",
                file=sys.stderr,
            )
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
