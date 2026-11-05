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
undetermined       Rejected, but nothing supports a claim about it: no signal
                   could be evaluated (e.g. a legacy v0 session whose static
                   findings were never recorded), or there is no ground-truth
                   label at all (M3 item E8 — an inconclusive candidate, whose
                   patch applied but was never graded). `true_catch` and
                   `spurious_reject` both claim the rejected patch was known-bad,
                   so a rejection with no label cannot be either. Reported
                   separately so it is never silently folded into
                   spurious_reject, and excluded from every ground-truth-dependent
                   denominator.
approved_unlabeled Approved with no ground-truth label, so nothing is claimed.
needs_review_on_good
                   The run refused to decide (`NeedsReview`, §14) on a patch the
                   ground truth says is GOOD. [WHY] Its own bucket: it is neither
                   a false_reject (Vanta never rejected) nor a true_approval
                   (nothing was approved), and folding it into either would
                   report a decision the run did not make.
needs_review_on_bad
                   The run refused to decide on a patch the ground truth says is
                   BAD. Same reasoning: not a true_reject, not a false_approval.
needs_review_unlabeled
                   The run refused to decide and no ground truth was supplied, so
                   nothing is claimed about whether the refusal was right.
infra_failure      The run ended because an LLM call failed for every attempt
                   the retry loop was allowed (`final_decision=InfraFailure`).
                   No test ever judged the candidate, so this is neither an
                   approval, a rejection, nor an "unresolved" budget exhaustion —
                   it is a property of the harness, not of the model. Counted on
                   its own so it can never inflate or deflate any other bucket.
judge_failed       The judge arm's model call was made and answered, but the
                   answer could not be read as a verdict after the one retry
                   (`final_decision=JudgeFailed`). [WHY] Its own bucket, never
                   folded into `needs_review_unlabeled` (the run did not refuse —
                   it failed to produce a readable answer) nor into
                   `infra_failure` (the machine was fine and the call succeeded).
                   The judge's verdicts that ARE readable are scored in the
                   approval/rejection buckets, so this counts only the calls the
                   arm could not turn into a verdict.

Pooled, per config
------------------
[WHY] The buckets above score ONE session. `--pool-by-config` adds §9.1's pooled
metrics, computed inside each §4.4 provenance group and printed with that group's
axes beside them — never averaged across groups, because two runs whose config,
seed, model, sampling schedule or build commit differ are two experiments. That
section is additive: no bucket, and no existing line of output, changes.

M3 item 7 — the judge arms, and what this module is NOT for
-----------------------------------------------------------
`judge` and `judge+suite` state a verdict instead of running tests, so this module
scores them verdict-only: `JudgeApproved` joins the approval arm and `JudgeRejected`
the rejection arm, and a judge rejection of a patch labelled known-bad is a
`true_catch` — never `spurious_reject`, which claims every rejecting SIGNAL failed
to discriminate, a statement about signals a judge row does not have. Because `arm`
is one of the §4.4 axes, judge rows and Vanta rows always group apart and never
share a pooled bucket table.

[WHY] This module is the LEGACY pooled view, kept for the arms that produce signals.
The comparison the item-7 study actually asks for — judge against the B3 arm it is
run at the same model, effort, context budget and seed as — is a PAIRING, and the
labels it pairs on come from ``runner.label_for`` at analysis time (M3 item 12),
where a judge row and its B3 counterpart carry the same TA/FA/TR/FR label under the
same candidate key. Read this module for the wave's own counts; read the pairing for
the item-7 result. `judge_failed` is counted here — on its own line, and only for a
batch that has one — but is deliberately not part of ``BUCKET_ORDER``, so no batch
recorded before the judge existed prints a different table than it did before.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

# [VANTA-WHY] F3 — `manifest.effective_records` is the ONE definition of "the rows
# that are current" (the last row per key, minus any session a later row names in
# `supersedes`). This module shares it rather than re-deriving the supersede rule,
# so the oracle index and the batch digest can never disagree about which rows
# exist. The scripts directory is on `sys.path` for every way this file is run
# (`python3 scripts/metrics.py`, `analyse.py`'s import, the tests), but it is added
# explicitly so the dependency is not an accident of the working directory.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from bugsinpy import manifest as manifest_mod  # noqa: E402

SUPPORTED_SCHEMA_VERSION = 1

TRUE_CATCH = "true_catch"
SPURIOUS_REJECT = "spurious_reject"
FALSE_REJECT = "false_reject"
FALSE_APPROVAL = "false_approval"
TRUE_APPROVAL = "true_approval"
UNRESOLVED = "unresolved"
UNDETERMINED = "undetermined"
APPROVED_UNLABELED = "approved_unlabeled"
NEEDS_REVIEW_ON_GOOD = "needs_review_on_good"
NEEDS_REVIEW_ON_BAD = "needs_review_on_bad"
NEEDS_REVIEW_UNLABELED = "needs_review_unlabeled"
INFRA_FAILURE = "infra_failure"
JUDGE_FAILED = "judge_failed"

# [WHY] `AllTestsPassed` alone. `MaxAttemptsReached` used to live here and was
# bucketed as an approval; it is the opposite — the Creator exhausted the budget
# without an approved patch (see the `unresolved` bucket).
# [WHY] M3 item 7 — `JudgeApproved` joins it. The judge states a verdict instead of
# running tests, but it is the same verdict: an approval is scored against the
# ground-truth label exactly as `AllTestsPassed` is. `JudgeFailed` is deliberately
# NOT here (it approves nothing — see `JUDGE_FAILED`), and neither is
# `JudgeRejected` (the set below).
APPROVED_DECISIONS = {"AllTestsPassed", "JudgeApproved"}
# [WHY] M3 item 7 — the rejection arm's two tokens. `JudgeRejected` is the same
# verdict as `Rejected` for the ground-truth label, so it is bucketed by the same
# arm rather than a parallel one; the arms differ only in whether a rejecting
# SIGNAL exists to re-run (see `classify`).
REJECTED_DECISIONS = {"Rejected", "JudgeRejected"}
UNRESOLVED_DECISIONS = {"MaxAttemptsReached"}
# [WHY] §14's refusal to decide. Kept in its own set, never merged into
# APPROVED_DECISIONS, UNRESOLVED_DECISIONS or the `Rejected` arm: the run did not
# reject the candidate and did not give up mid-patch, it declined to make the
# claim at all because no rule corroborated a rejection. Its row's own label
# (`needs_review_on_good`/`needs_review_on_bad`, runner.py) already carries the
# ground truth, so this arm only has to keep it out of the other buckets.
NEEDS_REVIEW_DECISIONS = {"NeedsReview"}
# [WHY] An LLM call failed after its retries. Kept in its own set, never merged
# into APPROVED_DECISIONS or UNRESOLVED_DECISIONS: the run produced no verdict
# about the candidate at all.
INFRA_FAILURE_DECISIONS = {"InfraFailure"}
# [WHY] M3 item 7 — the judge's call succeeded but its answer could not be read as
# a verdict after the one retry. Kept in its own set, never merged into
# NEEDS_REVIEW_DECISIONS (the run did not refuse; it failed to answer) or
# INFRA_FAILURE_DECISIONS (the harness was fine). The judge's readable verdicts
# are `JudgeApproved`/`JudgeRejected`, which join the approval/rejection arms above.
JUDGE_FAILED_DECISIONS = {"JudgeFailed"}

# [WHY] The order every bucket is reported in, and the reason it is one tuple
# rather than a literal in each renderer: a bucket that exists in `classify` but
# is missing from a report is a count nobody sees. The two decisions that were
# RIGHT (`true_catch`, `true_approval`) sit next to their wrong counterparts,
# and the §14.6 refusals follow the approvals and rejections they are neither of,
# so a reader comparing a wave against the previous one sees a refusal count next
# to the two counts it could otherwise have been mistaken for.
BUCKET_ORDER = (
    TRUE_CATCH,
    SPURIOUS_REJECT,
    FALSE_REJECT,
    FALSE_APPROVAL,
    TRUE_APPROVAL,
    NEEDS_REVIEW_ON_GOOD,
    NEEDS_REVIEW_ON_BAD,
    NEEDS_REVIEW_UNLABELED,
    UNRESOLVED,
    UNDETERMINED,
    APPROVED_UNLABELED,
    INFRA_FAILURE,
)
# [WHY] M3 item 7 — `judge_failed` is deliberately NOT in this tuple. It is a bucket
# a verdict can land in, but a line for it in every report would change the output of
# every batch already on disk, and those rows have no judge. It is counted on its own
# line, and only for a batch that has one (see `render_table`).


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
    # [WHY] `<sid>_transitions.jsonl` is a SEPARATE artifact from the session log
    # and is where the ECR/EIR rows live. Absent for a run that produced no
    # transition at all (a first-attempt success), which is a fact, not an error.
    transitions: list[dict[str, Any]] = field(default_factory=list)

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
    data.transitions = load_transitions(sessions_dir, session_id)
    return data


def load_transitions(sessions_dir: str, session_id: str) -> list[dict[str, Any]]:
    """Read `<sid>_transitions.jsonl`, or `[]` when the run wrote none."""
    path = os.path.join(sessions_dir, f"{session_id}_transitions.jsonl")
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"{path}:{line_number}: not valid JSON ({error})"
                ) from error
    return rows


def discover_sessions(sessions_dir: str) -> list[str]:
    """Every session id that has a session JSONL in the directory."""
    ids = []
    for name in sorted(os.listdir(sessions_dir)):
        if name.endswith("_session.jsonl"):
            ids.append(name[: -len("_session.jsonl")])
    return ids


# ============================================================================
# ECR/EIR — the SAME rule as the Rust analyser, read from transition rows
# ============================================================================
#
# [WHY] One definition of the metric: the console's live print, the M5 analysis
# layer and this script all apply the counting rule to the same
# `transitions.jsonl`. `counts_toward_metrics` in `src/core/mod.rs` is the other
# half (the failure-class gate) and this is its mirror.
#
# [WHY] The rates are CONDITIONAL on the row's state at N, exactly as in Rust:
#   ECR = rows Failed at N that Pass at N+1 / rows Failed at N
#   EIR = rows Passed at N that Fail at N+1 / rows Passed at N
# Per row instead (divide by every countable row) the denominator becomes the
# size of the repo's own test suite, so 4d55123d's single native fix would read
# 1/5 and a BugsInPy repo with hundreds of native tests would score ~0.1% for a
# real repair. `complete` survives as the run's churn count, not as a rate
# denominator.

# [WHY] The three sources the breakdown is reported over. A source the file adds
# later shows up on its own rather than being folded into AttackerGenerated.
SOURCES = ("AttackerGenerated", "Native", "Static")


def _countable_side(result: Any, failure_class: Any, source: str) -> bool:
    """One side of a row may move the metric only if it is pass or Logic-class.

    [WHY] `source == "Static"` reopens the exclusion for its OWN rows: a static
    row is a blocking finding already differenced against the pre-patch
    baseline, so the finding that remains is the candidate's and one that
    disappears on the next attempt is a real correction. The allowance is NOT
    given to AttackerGenerated rows — an `import-not-found` there is the
    environment's fault.
    """
    if result == "Passed":
        return True
    if failure_class == "Logic":
        return True
    return source == "Static" and failure_class == "Static"


def is_countable_patch_transition(row: dict[str, Any]) -> bool:
    """A row counts only when both ends are observed AND different.

    [WHY] `from == to` is a review round (the candidate never changed) and a
    null end is a recorded hole. Both are excluded, never filled in — the same
    condition as `TransitionRow::is_countable_patch_transition`.
    """
    if row.get("censored"):
        return False
    frm = row.get("from_patch_attempt_id")
    to = row.get("to_patch_attempt_id")
    if frm is None or to is None:
        return False
    return frm != to


# [WHY] One counter set per scope (the totals and each source), so the totals can
# never disagree with the breakdown. `ecr_denominator` counts rows FAILING at N,
# `eir_denominator` counts rows PASSING at N, and `complete` counts every
# countable row (churn).
_EMPTY_COUNTERS = {
    "ecr": 0,
    "eir": 0,
    "ecr_denominator": 0,
    "eir_denominator": 0,
    "complete": 0,
}


def compute_ecr_eir(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Combined CONDITIONAL ECR/EIR plus the per-source breakdown.

    ECR = F->P / rows failing at N; EIR = P->F / rows passing at N. A row whose
    state at N is `Error` joins neither denominator: it was neither passing nor
    failing, so it could neither be fixed nor broken.
    """
    total = dict(_EMPTY_COUNTERS)
    per_source: dict[str, dict[str, int]] = {}
    for row in rows:
        if not is_countable_patch_transition(row):
            continue
        source = row.get("evidence_source") or "AttackerGenerated"
        if not _countable_side(
            row.get("from_result"), row.get("from_failure_class"), source
        ):
            continue
        if not _countable_side(row.get("to_result"), row.get("to_failure_class"), source):
            continue
        bucket = per_source.setdefault(source, dict(_EMPTY_COUNTERS))
        from_result = row.get("from_result")
        pair = (from_result, row.get("to_result"))
        for counters in (total, bucket):
            counters["complete"] += 1
            if from_result == "Failed":
                counters["ecr_denominator"] += 1
            elif from_result == "Passed":
                counters["eir_denominator"] += 1
            if pair == ("Failed", "Passed"):
                counters["ecr"] += 1
            elif pair == ("Passed", "Failed"):
                counters["eir"] += 1
    return {**total, "rows_total": len(rows), "per_source": per_source}


def rate_text(count: int, denominator: int) -> str:
    """`count/denominator (pct%)`, or `n/a` when the denominator is zero.

    [WHY] "No countable transition" and "every countable transition was
    uncorrected" are opposite claims about a run; `0.00%` states the second.
    `n/a` states the first. The denominator is the CONDITIONAL one
    (`ecr_denominator` / `eir_denominator`), never `complete`.
    """
    if denominator == 0:
        return "n/a"
    return f"{count}/{denominator} ({count / denominator * 100:.2f}%)"


def unshown_text(count: Optional[int]) -> str:
    """`n` for a measured count, `n/a` when the field was never recorded.

    [WHY] M3 item V — `None` is "not measured" (a review run, or a row written
    before the field existed), never a `0` that would claim the run measured and
    found none. The two must not render the same.
    """
    return "n/a" if count is None else str(count)


def source_text(per_source: dict[str, dict[str, int]]) -> str:
    """`AttackerGenerated[ECR=… EIR=…] Native[…] Static[…]`."""
    parts = []
    for name in SOURCES:
        bucket = per_source.get(name) or _EMPTY_COUNTERS
        parts.append(
            f"{name}[ECR={rate_text(bucket.get('ecr', 0), bucket.get('ecr_denominator', 0))} "
            f"EIR={rate_text(bucket.get('eir', 0), bucket.get('eir_denominator', 0))}]"
        )
    return " ".join(parts)


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
    metrics: dict[str, Any] = field(default_factory=dict)
    # [WHY] M3 item V — how many attempts edited a file the Creator was not shown
    # in full (an outline, a bare path, dropped for budget, or never selected).
    # Read from this session's own `run.summary`; `None` when the key is absent,
    # which is a review run (no Creator) or a row written before the field
    # existed. It is deliberately not defaulted to `0`: "did not measure" and
    # "measured zero" are different facts.
    attempts_edited_unshown_file: Optional[int] = None

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
        if self.bucket == NEEDS_REVIEW_ON_GOOD:
            return "refused to decide on a patch labelled known-good (not a rejection)"
        if self.bucket == NEEDS_REVIEW_ON_BAD:
            return "refused to decide on a patch labelled known-bad (not a rejection)"
        if self.bucket == NEEDS_REVIEW_UNLABELED:
            return "refused to decide; no ground-truth label, so nothing is claimed"
        if self.bucket == INFRA_FAILURE:
            return "LLM call failed after its retries (not a verdict about the candidate)"
        if self.bucket == JUDGE_FAILED:
            return (
                "judge answered but the verdict was unreadable after the one retry "
                "(not a verdict about the candidate)"
            )
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

    if decision in REJECTED_DECISIONS:
        if known_bad is False:
            # [WHY] The label decides, not the discrimination table. When the
            # candidate IS the known-good code (B v2 applies the fixed patch),
            # every signal fires on the good tree too, so no signal can ever
            # discriminate — the table would call a wrong rejection "spurious".
            bucket = FALSE_REJECT
        elif decision == "JudgeRejected" and known_bad is True:
            # [WHY] M3 item 7 — the judge produces no rejecting signals (no
            # Attacker), so the signal-evidence test can never fire for it. Its
            # verdict IS the decision, and the plan scores it as the test-driven
            # arm's is scored: a rejection of a patch labelled known-bad is a
            # catch. It is never `spurious_reject`, a word that claims the run
            # rejected while every signal failed to discriminate — a statement
            # about signals a judge row does not have. An unlabelled judge
            # rejection still claims nothing and falls through to `undetermined`.
            bucket = TRUE_CATCH
        elif known_bad is None:
            # [VANTA-WHY] M3 item E8 — with NO ground truth (an inconclusive
            # candidate: the patch applied but no run wrote a readable grade), a
            # rejection supports neither `true_catch` nor `spurious_reject`: both
            # are claims about a KNOWN-BAD patch, and `deciding` says only that a
            # signal discriminated — not that the patch was bad. It falls to
            # `undetermined`, which claims nothing, and is thereby excluded from
            # every ground-truth-dependent denominator (`spurious_reject_rate`
            # divides by `true_catch + spurious_reject`, and no labelled row is
            # affected). The bucket is still visible in the totals.
            bucket = UNDETERMINED
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
    elif decision in INFRA_FAILURE_DECISIONS:
        # [WHY] The LLM call failed for every attempt the retry loop was allowed,
        # so the pipeline never reached the Governor's Passed/Failed branch and
        # no test ever judged the candidate. Checked BEFORE the labelled-approval
        # and final fallthrough arms: an infrastructure failure is not a claim
        # about the model, so it must be neither approval, rejection, nor the
        # budget-exhausted `unresolved`. It goes in its own bucket.
        bucket = INFRA_FAILURE
    elif decision in JUDGE_FAILED_DECISIONS:
        # [WHY] M3 item 7 — the judge arm made its call and the answer could not be
        # read as a verdict after the one retry. Checked BEFORE the labelled-approval
        # and fallthrough arms for the same reason as `InfraFailure`: it is not a
        # claim about the model (it never produced a readable verdict) and not a
        # refusal (`NeedsReview`), so it must be neither an approval, a rejection,
        # nor `needs_review_*`. It goes in its own bucket.
        bucket = JUDGE_FAILED
    elif decision in NEEDS_REVIEW_DECISIONS:
        # [WHY] §14.6: the loop declined to decide because no rule corroborated a
        # rejection. Checked BEFORE the labelled-approval and fallthrough arms —
        # an unlabelled NeedsReview must not be filed as `approved_unlabeled`
        # ("nothing is claimed") nor as `undetermined` (a broken rejection),
        # because it states a fact of its own: the run refused. The ground-truth
        # label, when there is one, only says which side the refusal fell on.
        if known_bad is True:
            bucket = NEEDS_REVIEW_ON_BAD
        elif known_bad is False:
            bucket = NEEDS_REVIEW_ON_GOOD
        else:
            bucket = NEEDS_REVIEW_UNLABELED
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
        metrics=compute_ecr_eir(data.transitions),
        attempts_edited_unshown_file=unshown_edit_count(
            _run_summary_text(data.records)
        ),
    )


def _run_summary_text(records: list[dict[str, Any]]) -> str:
    """This session's `run.summary` line, or `""` when it wrote none."""
    record = run_summary_record(records)
    return str(record.get("summary") or "") if record else ""


def unshown_edit_count(summary: str) -> Optional[int]:
    """M3 item V — the attempts this run edited a file the Creator could not read.

    [WHY] Read from `run.summary` rather than the manifest, so the count is
    available from the session directory alone (the same reason `provenance_of`
    reads the row instead of an env var). An absent key and a malformed value are
    both `None` — "did not measure" — never a `0` that would be read as a run that
    measured and found none.
    """
    raw = summary_field(summary or "", ATTEMPTS_EDITED_UNSHOWN_FILE_KEY)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# ============================================================================
# Reporting
# ============================================================================


def render_table(verdicts: list[Verdict]) -> str:
    # [WHY] 22 is the width of the longest bucket name (`needs_review_unlabeled`).
    # The three §14.6 refusals are longer than every bucket before them, and a
    # narrower column would run the bucket into its own note.
    bucket_width = 22
    header = (
        f"{'SESSION':<10} {'VERDICT':<12} {'SCHEMA':<12} {'SIGNALS':<13} "
        f"{'BUCKET':<{bucket_width}} NOTE"
    )
    lines = [header, "-" * len(header)]
    for verdict in verdicts:
        signals = f"{verdict.signals_deciding}/{verdict.signals_total}"
        lines.append(
            f"{verdict.session_id[:8]:<10} {verdict.decision:<12} "
            f"{verdict.schema_label:<12} {signals:<13} "
            f"{verdict.bucket:<{bucket_width}} "
            f"{verdict.explanation}"
        )
    lines.append("")
    lines.append("TOTALS")
    for bucket in BUCKET_ORDER:
        count = sum(1 for verdict in verdicts if verdict.bucket == bucket)
        lines.append(f"  {bucket:<{bucket_width}} {count}")
    # [VANTA-WHY] M3 item 7 — `judge_failed` is not in BUCKET_ORDER (its line would
    # change every batch's output), so it is counted on its own line, and only when
    # the batch has one. A judge that answered unreadably is a fact about the run, so
    # it must be visible; a batch with no judge must not grow a line for it.
    judge_failed = sum(1 for v in verdicts if v.bucket == JUDGE_FAILED)
    if judge_failed:
        lines.append(f"  {JUDGE_FAILED:<{bucket_width}} {judge_failed}")
    lines.append("")
    scored = sum(
        1
        for v in verdicts
        if v.bucket in (TRUE_CATCH, SPURIOUS_REJECT, FALSE_REJECT)
    )
    seen = sum(1 for v in verdicts if v.decision in REJECTED_DECISIONS)
    lines.append(f"  {'scored rejections':<18} {scored}")
    lines.append(f"  {'rejections seen':<18} {seen}")
    return "\n".join(lines)


def render_metrics(verdicts: list[Verdict]) -> str:
    """ECR/EIR per session and pooled, combined and per evidence source.

    [WHY] Pooled alone hides WHICH evidence moved the number. Session 4d55123d's
    only correction was a repo test (`Native`), and reading it as an
    AttackerGenerated result would credit the model for something the
    repository's own suite proved. A zero denominator prints `n/a`, never
    `0.00%`.

    [WHY] M3 item V — the `UNSHOWN` column counts this run's attempts that edited
    a file the Creator was not shown in full (an outline, a bare path, dropped for
    budget, or never selected). It is a logging fact about the context, not a
    verdict; a run that never recorded it (a review run, or a row from before the
    field) prints `n/a`, and the pooled cell sums only the runs that did measure.
    """
    lines = [
        "",
        "ECR/EIR (conditional on the state at N; n/a = no such row)",
    ]
    header = f"  {'SESSION':<10} {'ECR':<20} {'EIR':<20} {'UNSHOWN':<8} BY SOURCE"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    pooled = dict(_EMPTY_COUNTERS)
    pooled_sources: dict[str, dict[str, int]] = {}
    measured_unshown: list[int] = []
    for verdict in verdicts:
        metrics_ = verdict.metrics or {}
        per_source = metrics_.get("per_source") or {}
        if verdict.attempts_edited_unshown_file is not None:
            measured_unshown.append(verdict.attempts_edited_unshown_file)
        lines.append(
            f"  {verdict.session_id[:8]:<10} "
            f"{rate_text(metrics_.get('ecr', 0), metrics_.get('ecr_denominator', 0)):<20} "
            f"{rate_text(metrics_.get('eir', 0), metrics_.get('eir_denominator', 0)):<20} "
            f"{unshown_text(verdict.attempts_edited_unshown_file):<8} "
            f"{source_text(per_source)}"
        )
        for key in pooled:
            pooled[key] += metrics_.get(key, 0)
        for name, bucket in per_source.items():
            acc = pooled_sources.setdefault(name, dict(_EMPTY_COUNTERS))
            for key in acc:
                acc[key] += bucket.get(key, 0)
    pooled_unshown = sum(measured_unshown) if measured_unshown else None
    lines.append(
        f"  {'POOLED':<10} "
        f"{rate_text(pooled['ecr'], pooled['ecr_denominator']):<20} "
        f"{rate_text(pooled['eir'], pooled['eir_denominator']):<20} "
        f"{unshown_text(pooled_unshown):<8} "
        f"{source_text(pooled_sources)}"
    )
    lines.append(
        "  UNSHOWN = attempts that edited a file the Creator was not shown in "
        "full (M3 item V); n/a = not recorded"
    )
    return "\n".join(lines)


# ============================================================================
# Pooled, per config (design §9.1–§9.4) — ADDITIVE
# ============================================================================
#
# [WHY] Nothing above changes. `classify()` still buckets one session from that
# session's own files, and the tables above are still per session. This section
# answers the other question the pilot asks — "this config, many runs, one
# number" — and it answers it by GROUPING, never by averaging across groups.
#
# [WHY] §4.4: a run's config, mode, seed, model, sampling schedule and the commit
# its binary was built from are PROVENANCE, not metadata. Two runs that differ in
# any of them are two experiments, and a pooled number that mixes them reports
# one study's result under another study's name. So every metric here is computed
# inside a group and printed with the group it belongs to.
#
# [WHY] The intervals are naive (§9.3): rows within one bug are not independent,
# so the Wilson interval understates uncertainty. It is printed with that caveat
# beside it rather than omitted, because "pooled 40 %" with no interval at all
# would read as a measurement it is not.

PROVENANCE_KEYS = ("config", "arm", "mode", "seed", "model", "top_p", "commit", "dirty")
"""The named §4.4 axes, in the order they are keyed and printed.

[WHY] ``arm`` was added by M3 item 8's follow-up. It sits beside ``config`` because
the two answer different questions: ``config`` is the split's own column (``baseline``
/ ``vanta``), which two arms can share — ``b1`` and ``suite`` are both
``config=baseline`` — while ``arm`` is the canonical condition
(``b0``/``b1``/``b2``/``b3``/``suite``). A session written before the field existed
reads ``unset`` here, exactly like any other unrecorded axis, so it groups apart from
one that recorded its arm rather than being pooled with it under a guessed name.
"""

PROVENANCE_UNKNOWN = "unset"
"""What an axis reads as when `run.summary` did not record it.

[WHY] `unset` is a provenance fact, not a hole to fill. A session written before
§4.4 existed did not run with "no config" — nobody recorded one — and pooling it
with a session that DID record its config would claim the two are the same
experiment. The grouping key uses this same word, so an unrecorded axis can never
collide with a recorded one.
"""

SAMPLING_PRESET_KEY = "sampling_preset"
"""The one named key of the sampling block; it names the schedule (§4.3)."""

ATTEMPTS_EDITED_UNSHOWN_FILE_KEY = "attempts_edited_unshown_file"
"""M3 item V — the `run.summary` key for attempts that edited an unshown file.

[WHY] A repair-only key (the Creator is the only role a "did the patch touch a
file it could not read?" question applies to). A review run and any row written
before the field existed simply do not carry it, which is why `unshown_edit_count`
(and `Verdict.attempts_edited_unshown_file`) read `None` for them rather than `0`.
"""

SAMPLING_PIN_PREFIXES = (
    "temperature_",
    "max_tokens_",
    "top_p_",
    "top_k_",
    "min_p_",
    "presence_penalty_",
)
"""The per-site pins `llm::sampling_pin_table` can emit, `_<site>` suffixed.

[WHY] Under the default preset only `temperature_<site>` / `max_tokens_<site>`
appear; a `coding` run additionally pins `top_p`/`top_k`/`min_p`/
`presence_penalty` on every site. All of them are provenance — a `top_k` one run
never sampled at is not a value another run may be pooled with.
[WHY] A bare `top_p` (no underscore) is deliberately NOT one of these: that key
is the server's own default, recorded separately under `PROVENANCE_KEYS`.
"""

WILSON_Z = 1.96


def run_summary_record(records: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The first `run.summary` record, or None when the session wrote none.

    [WHY] The row that states the run's own facts — its verdict AND its
    provenance — so a reader of either looks at the same row.
    """
    for record in records:
        if record.get("phase") == "run.summary":
            return record
    return None


def summary_pairs(summary: str) -> dict[str, str]:
    """Every lowercase `key=value` token of a summary line, as a dict.

    [WHY] `summary_field` answers "what is <key>" for a key the caller names in
    advance; the per-call sampling schedule (§4.3) has one key per call site, so
    a caller that wants the whole schedule cannot name them all. This scan
    applies the same rule `_next_key_start` already encodes — a key starts at the
    string's start or after a space and is `[a-z_][a-z0-9_]*` — over the whole
    line, so a value may contain spaces (`test_cmd=env PYTHONPATH=src <python> -m
    pytest -q`) without being cut short.
    """
    pairs: dict[str, str] = {}
    index = 0
    size = len(summary)
    while index < size:
        if index != 0 and summary[index - 1] != " ":
            index += 1
            continue
        key_end = index
        while key_end < size and _is_key_char(summary[key_end], first=(key_end == index)):
            key_end += 1
        if key_end == index or summary[key_end : key_end + 1] != "=":
            index += 1
            continue
        value_start = key_end + 1
        value_end = _next_key_start(summary, value_start)
        pairs[summary[index:key_end]] = summary[value_start:value_end]
        index = value_end
    return pairs


def is_sampling_key(key: str) -> bool:
    """Whether a summary key belongs to the §4.3 sampling schedule."""
    return key == SAMPLING_PRESET_KEY or key.startswith(SAMPLING_PIN_PREFIXES)


def provenance_of(summary: str) -> dict[str, str]:
    """The §4.4 provenance axes, read from one `run.summary` line.

    [WHY] Read from the line itself, not from an env var or the directory name a
    session happens to sit in: a session copied out of the batch has to stay
    self-describing (§4.4). An axis the line does not carry reads
    `PROVENANCE_UNKNOWN`.
    """
    pairs = summary_pairs(summary or "")
    axes = {key: pairs.get(key, PROVENANCE_UNKNOWN) for key in PROVENANCE_KEYS}
    for key, value in pairs.items():
        if is_sampling_key(key):
            axes[key] = value
    return axes


def session_provenance(data: SessionData) -> dict[str, str]:
    """The §4.4 provenance axes of one loaded session."""
    record = run_summary_record(data.records) or {}
    return provenance_of(str(record.get("summary") or ""))


def provenance_key(axes: dict[str, str]) -> tuple:
    """The identity of a provenance group: every axis, in a fixed order.

    [WHY] The sampling schedule is variable-length (one pair per call site), so
    the named axes come first in `PROVENANCE_KEYS` order and the sampling keys
    follow sorted. A fixed order is what makes two groups' keys comparable and a
    group's identity independent of the order the line happened to be written in.
    """
    named = tuple(
        (key, axes.get(key, PROVENANCE_UNKNOWN)) for key in PROVENANCE_KEYS
    )
    sampling = tuple(
        (key, axes[key]) for key in sorted(axes) if is_sampling_key(key)
    )
    return named + sampling


def wilson_interval(
    count: int, denominator: int, z: float = WILSON_Z
) -> Optional[tuple[float, float]]:
    """The Wilson score interval for `count / denominator` (§9.2).

    [WHY] `None` — never `(0.0, 0.0)` — when there is no denominator. "No row
    was in this denominator" and "every row failed" are opposite claims, and an
    interval printed for the first would state the second. `z` is a parameter so
    the arithmetic can be checked against a hand-computed interval.
    """
    if denominator <= 0:
        return None
    n = float(denominator)
    p = count / n
    z2 = z * z
    shrink = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / shrink
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / shrink
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_text(interval: Optional[tuple[float, float]]) -> str:
    """`[0.118, 0.769]`, or `n/a` when there is no interval to print."""
    if interval is None:
        return "n/a"
    low, high = interval
    return f"[{low:.3f}, {high:.3f}]"


def load_manifest(path: str) -> list[dict[str, Any]]:
    """Read a batch `manifest.jsonl` (§7.3).

    [WHY] A torn last line means the writer died mid-append, and the batch's own
    cursor still says `running` for that job, so `Manifest.records` drops it and
    the job is re-run. This reader drops it for the same reason: a report taken
    from a crashed batch must report the rows that exist rather than refuse to
    run.

    [WHY] F3 — the rows are then filtered to the current ones through
    `manifest.effective_records`, so a resumed or re-run job is not read twice and a
    re-run of an inconclusive job drops the session it supersedes.

    [WHY] M3-E follow-up — the rows carry any measured relabel the batch recorded
    (`manifest.with_measured_relabels`), so this module sees the same row every
    other reader resolves. What this module scores is NOT that label: its buckets
    come from the session's `final_decision` paired with the CANDIDATE's ground
    truth (`known_bad`, from the candidate's own oracle), which no relabel moves.
    The attach is what keeps a future reader of the outcome label here honest.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return manifest_mod.with_measured_relabels(
        manifest_mod.effective_records(records), os.path.dirname(path)
    )


@dataclass
class OracleFacts:
    """One manifest row's oracle verdict, the source of §9.1's oracle gauges."""

    bug: str
    seed: Any
    apply_ok: bool
    hidden_pass: bool
    patch_good: bool
    suite_regressions: int
    inconclusive: bool = False
    """The patch applied but the hidden run wrote no report (item B4's third state).

    [VANTA-WHY] Item B4's general rule — an inconclusive grading is never counted as
    good or bad anywhere. ``apply_ok`` is true, so without this field the row would
    land in ``observable`` and, because ``patch_good`` is false, count against
    ``patch_good_rate``: a model failure invented from a grading nobody could read.
    Read here so the pooled table can hold it apart, exactly as it holds the censor.
    """

    @property
    def reason(self) -> str:
        """Why a known-bad approval was bad (§9.1): hidden_fail | suite_regression."""
        if not self.hidden_pass:
            return "hidden_fail"
        return "suite_regression"


def oracle_by_session(records: list[dict[str, Any]]) -> dict[str, OracleFacts]:
    """Index the manifest's oracle readings by session id.

    [WHY] `hidden_pass` / `patch_good` / `apply_ok` are the oracle's facts, not
    Vanta's: they are what makes a false approval NAMEABLE ("the hidden test
    still failed" vs "the native suite regressed") instead of merely counted.
    Reading them from the manifest keeps one copy of the oracle in one place.

    [WHY] Item E6 + Q3 + F3 — a re-run of a job whose previous grade was
    INCONCLUSIVE appends a row naming the session it supersedes. That superseded
    session is dropped here, so the inconclusive reading can never be counted. The
    filter is `manifest.effective_records`, the one shared rule: it keeps the last
    row per key and drops a named superseded session, so this index is the same set
    of rows the batch digest and `write_labels` count. Rows that already passed
    through it are unaffected (the rule is idempotent), so a caller may hand this
    reader either the raw manifest rows or an effective subset.
    """
    out: dict[str, OracleFacts] = {}
    for record in manifest_mod.effective_records(records):
        session_id = record.get("session_id")
        oracle = record.get("oracle")
        if not session_id or not isinstance(oracle, dict):
            continue
        regressions = oracle.get("suite_regressions_count")
        if regressions is None:
            regressions = len(oracle.get("suite_regressions") or [])
        out[str(session_id)] = OracleFacts(
            bug=str(record.get("bug")),
            seed=record.get("seed"),
            apply_ok=bool(oracle.get("apply_ok")),
            hidden_pass=bool(oracle.get("hidden_pass")),
            patch_good=bool(oracle.get("patch_good")),
            suite_regressions=int(regressions),
            inconclusive=bool(oracle.get("inconclusive")),
        )
    return out


@dataclass
class ProvenanceGroup:
    """One §4.4 provenance group: its axes and the sessions that share them."""

    provenance: dict[str, str]
    verdicts: list[Verdict] = field(default_factory=list)


@dataclass
class PooledMetrics:
    """§9.1's numbers for one provenance group."""

    runs: int
    buckets: dict[str, int]
    oracle_runs: int
    observable: int
    hidden_pass: int
    patch_good: int
    unobservable: int
    inconclusive: int
    ecr: int
    ecr_denominator: int
    eir: int
    eir_denominator: int
    false_approvals: list[tuple[Any, Any, str]]
    false_rejects: list[tuple[Any, Any]]

    def rates(self) -> dict[str, tuple[int, int]]:
        """§9.1's ratios as `(count, denominator)`, named as §9.1 names them.

        [WHY] One definition of every denominator, so the printed rate and the
        tested rate cannot disagree. Each denominator is the CONDITIONAL one the
        metric's own definition asks for — `approval_precision` over the rows the
        run actually approved, `spurious_reject_rate` over the known-bad
        rejections — never `runs`, which would silently count rows the metric
        never asked about.
        """
        approvals = self.buckets.get(TRUE_APPROVAL, 0) + self.buckets.get(
            FALSE_APPROVAL, 0
        )
        rejected_known_bad = self.buckets.get(TRUE_CATCH, 0) + self.buckets.get(
            SPURIOUS_REJECT, 0
        )
        return {
            "resolve_rate": (self.hidden_pass, self.observable),
            "patch_good_rate": (self.patch_good, self.observable),
            "approval_precision": (self.buckets.get(TRUE_APPROVAL, 0), approvals),
            "spurious_reject_rate": (
                self.buckets.get(SPURIOUS_REJECT, 0),
                rejected_known_bad,
            ),
            "infra_failure_rate": (self.buckets.get(INFRA_FAILURE, 0), self.runs),
            "unobservable_rate": (self.unobservable, self.runs),
            "inconclusive_rate": (self.inconclusive, self.runs),
            "ECR": (self.ecr, self.ecr_denominator),
            "EIR": (self.eir, self.eir_denominator),
        }

    @property
    def ecr_interval(self) -> Optional[tuple[float, float]]:
        """Wilson interval on ECR's own conditional denominator (§9.2)."""
        return wilson_interval(self.ecr, self.ecr_denominator)

    @property
    def eir_interval(self) -> Optional[tuple[float, float]]:
        """Wilson interval on EIR's own conditional denominator (§9.2)."""
        return wilson_interval(self.eir, self.eir_denominator)


def pool_by_provenance(
    verdicts: list[Verdict], provenance: dict[str, dict[str, str]]
) -> list[ProvenanceGroup]:
    """Group verdicts by their session's §4.4 provenance, in first-seen order.

    [WHY] First-seen order, not sorted: a report of one batch reads in the order
    the batch was built, and the ORDER IS NOT THE GROUPING — two groups are
    distinct because their axes differ, never because of where they sit.
    """
    groups: dict[tuple, ProvenanceGroup] = {}
    for verdict in verdicts:
        axes = provenance.get(verdict.session_id)
        if axes is None:
            axes = provenance_of("")
        group = groups.setdefault(
            provenance_key(axes), ProvenanceGroup(provenance=axes)
        )
        group.verdicts.append(verdict)
    return list(groups.values())


def pooled_metrics(
    group: ProvenanceGroup, facts_by_session: dict[str, OracleFacts]
) -> PooledMetrics:
    """Compute §9.1's metrics for one group.

    [WHY] `oracle_runs` is counted separately from `runs` because a session with
    no manifest row is not a run whose patch failed the oracle — it is a run whose
    oracle fact is absent, and dividing `hidden_pass` by `runs` would report that
    absence as a failure.
    """
    buckets = {name: 0 for name in BUCKET_ORDER}
    counters = dict(_EMPTY_COUNTERS)
    oracle_runs = observable = hidden_pass = patch_good = unobservable = inconclusive = 0
    false_approvals: list[tuple[Any, Any, str]] = []
    false_rejects: list[tuple[Any, Any]] = []
    for verdict in group.verdicts:
        buckets[verdict.bucket] = buckets.get(verdict.bucket, 0) + 1
        session_metrics = verdict.metrics or {}
        for key in counters:
            counters[key] += session_metrics.get(key, 0)

        facts = facts_by_session.get(verdict.session_id)
        if facts is not None:
            oracle_runs += 1
            if not facts.apply_ok:
                unobservable += 1
            elif facts.inconclusive:
                # Item B4 — applied, but no readable grading: neither a resolve nor a
                # patch_good, so it is held apart rather than counted against the model.
                inconclusive += 1
            else:
                observable += 1
                hidden_pass += 1 if facts.hidden_pass else 0
                patch_good += 1 if facts.patch_good else 0
        if verdict.bucket == FALSE_APPROVAL:
            false_approvals.append(
                (
                    facts.bug if facts else None,
                    facts.seed if facts else None,
                    facts.reason if facts else "unknown",
                )
            )
        if verdict.bucket == FALSE_REJECT:
            false_rejects.append(
                (facts.bug if facts else None, facts.seed if facts else None)
            )
    return PooledMetrics(
        runs=len(group.verdicts),
        buckets=buckets,
        oracle_runs=oracle_runs,
        observable=observable,
        hidden_pass=hidden_pass,
        patch_good=patch_good,
        unobservable=unobservable,
        inconclusive=inconclusive,
        ecr=counters["ecr"],
        ecr_denominator=counters["ecr_denominator"],
        eir=counters["eir"],
        eir_denominator=counters["eir_denominator"],
        false_approvals=false_approvals,
        false_rejects=false_rejects,
    )


def _entry_text(values: tuple) -> str:
    """`(httpie/1, 0, hidden_fail)`; a value nobody recorded prints `?`."""
    return "(" + ", ".join("?" if v is None else str(v) for v in values) + ")"


def _entries_text(entries: list[tuple]) -> str:
    if not entries:
        return "(none)"
    return " ".join(_entry_text(entry) for entry in entries)


def render_axes(axes: dict[str, str]) -> str:
    """The named axes on one line, the sampling schedule on the next."""
    named = " ".join(
        f"{key}={axes.get(key, PROVENANCE_UNKNOWN)}" for key in PROVENANCE_KEYS
    )
    lines = [f"  {named}"]
    sampling = sorted(key for key in axes if is_sampling_key(key))
    if sampling:
        lines.append("  sampling: " + " ".join(f"{key}={axes[key]}" for key in sampling))
    else:
        lines.append(f"  sampling: {PROVENANCE_UNKNOWN} (no schedule was recorded)")
    return "\n".join(lines)


def render_pooled(
    groups: list[ProvenanceGroup], facts_by_session: dict[str, OracleFacts]
) -> str:
    """§9.1's pooled table, one block per provenance group.

    [WHY] Every line names its own group's axes, so a number can never be read
    without the experiment it came from. When no manifest was given the oracle
    gauges print `n/a` and the false-approval list prints `?` for the facts the
    manifest would have supplied — an absence, never a zero.
    """
    lines = [
        "",
        "POOLED BY CONFIG (design §9.1; intervals are naive — §9.3)",
    ]
    if not groups:
        lines.append("  (no sessions)")
        return "\n".join(lines)
    for group in groups:
        pooled = pooled_metrics(group, facts_by_session)
        rates = pooled.rates()
        lines.append("")
        lines.append(render_axes(group.provenance))
        counted = " ".join(
            f"{name}={count}" for name, count in pooled.buckets.items() if count
        )
        lines.append(f"  runs={pooled.runs}  {counted}")
        lines.append(
            "  "
            + "  ".join(
                f"{name} {rate_text(*rates[name])}"
                for name in ("resolve_rate", "patch_good_rate", "approval_precision")
            )
        )
        lines.append(
            "  "
            + "  ".join(
                f"{name} {rate_text(*rates[name])}"
                for name in (
                    "spurious_reject_rate",
                    "infra_failure_rate",
                    "unobservable_rate",
                    "inconclusive_rate",
                )
            )
        )
        lines.append(
            f"  ECR {rate_text(*rates['ECR'])} "
            f"wilson 95% {wilson_text(pooled.ecr_interval)}   "
            f"EIR {rate_text(*rates['EIR'])} "
            f"wilson 95% {wilson_text(pooled.eir_interval)}"
        )
        lines.append(f"  oracle rows {pooled.oracle_runs}/{pooled.runs}")
        lines.append(
            f"  false_approvals[{len(pooled.false_approvals)}]: "
            f"{_entries_text(pooled.false_approvals)}"
        )
        lines.append(
            f"  false_rejects[{len(pooled.false_rejects)}]: "
            f"{_entries_text(pooled.false_rejects)}"
        )
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
    parser.add_argument(
        "--pool-by-config",
        dest="pool_by_config",
        action="store_true",
        help=(
            "Also print §9.1's metrics pooled per §4.4 provenance group "
            "(config x mode x seed x model x sampling x commit). Never pools "
            "across groups; each block names its own axes."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "The batch's manifest.jsonl (§7.3). Supplies the oracle facts "
            "(hidden_pass / patch_good / apply_ok) that --pool-by-config's "
            "oracle gauges and false-approval reasons need; without it those "
            "print n/a rather than zero."
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

    facts_by_session: dict[str, OracleFacts] = {}
    if args.pool_by_config and args.manifest:
        if not os.path.isfile(args.manifest):
            print(f"error: {args.manifest} is not a file", file=sys.stderr)
            return 1
        facts_by_session = oracle_by_session(load_manifest(args.manifest))

    verdicts: list[Verdict] = []
    provenance: dict[str, dict[str, str]] = {}
    for session_id in session_ids:
        data = load_session(args.sessions_dir, session_id)
        label = labels.get(session_id) or {}
        known_bad = label.get("known_bad")
        verdicts.append(classify(data, known_bad))
        if args.pool_by_config:
            provenance[session_id] = session_provenance(data)

    print(render_table(verdicts))
    print(render_metrics(verdicts))

    if args.pool_by_config:
        print(render_pooled(pool_by_provenance(verdicts, provenance), facts_by_session))

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
