#!/usr/bin/env python3
"""Regression tests for scripts/metrics.py.

Run with:  python3 -m pytest scripts/test_metrics.py -q

[WHY] Every test here pins a rule that, if it silently changed, would let the
study report a NUMBER that is wrong rather than a number that is missing:
a v0 session scored as if it were current, a missing probe read as "no signal
fired", or a rejection on an unlabeLLed session counted as a catch.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402


# ============================================================================
# Fixture helpers
# ============================================================================


def write_session(
    tmp_path,
    session_id: str,
    decision: str = "Rejected",
    schema_version=None,
    summary: str = None,
) -> str:
    """Write a minimal session JSONL and return the sessions dir.

    `summary` overrides the whole `run.summary` line, for a test that needs a
    field the default does not carry (e.g. item V's unshown-edit count).
    """
    if summary is None:
        summary = f"attempts_run=1 fixed_at_attempt_1=false final_decision={decision}"
    record = {
        "session_id": session_id,
        "patch_attempt_id": None,
        "testing_iteration_id": None,
        "test_id": None,
        "phase": "run.summary",
        "summary": summary,
        "timestamp": "2026-01-01T00:00:00Z",
    }
    if schema_version is not None:
        record["schema_version"] = schema_version
    path = tmp_path / f"{session_id}_session.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return str(tmp_path)


def write_reruns(tmp_path, session_id: str, signals: list[dict]) -> None:
    payload = {
        "schema_version": 1,
        "session_id": session_id,
        "final_decision": "Rejected",
        "final_attempt": 1,
        "signals": signals,
    }
    (tmp_path / f"{session_id}_signal_reruns.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def signal(discriminates) -> dict:
    return {
        "kind": "generated_test",
        "id": "test_x",
        "bad_outcome": "Failed",
        "good_outcome": "Passed",
        "discriminates": discriminates,
        "note": None,
    }


def transition(
    *,
    from_result: str = "Passed",
    to_result: str = "Passed",
    from_class=None,
    to_class=None,
    source: str = "AttackerGenerated",
    from_patch: str = "patch-aaa",
    to_patch: str = "patch-bbb",
    censored: bool = False,
) -> dict:
    """One transition row, with the defaults a REPAIR transition needs."""
    return {
        "session_id": "s",
        "schema_version": 1,
        "from_patch_attempt_id": from_patch,
        "to_patch_attempt_id": to_patch,
        "from_attempt": 0,
        "to_attempt": 1,
        "test_id": "id",
        "test_name": "name",
        "shared_set_id": "native",
        "from_result": from_result,
        "to_result": to_result,
        "from_failure_class": from_class,
        "to_failure_class": to_class,
        "evidence_source": source,
        "test_intent": "BehaviorPreservation",
        "censored": censored,
        "apply_failed": False,
        "gap": False,
        "in_s0": True,
    }


def write_transitions(tmp_path, session_id: str, rows: list[dict]) -> None:
    path = tmp_path / f"{session_id}_transitions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# ============================================================================
# ECR/EIR from transition rows (native results must count)
# ============================================================================


def test_a_native_correction_counts_in_the_combined_ecr(tmp_path):
    """Session 4d55123d's shape: the only real correction was a repo test.

    [WHY] The native suite wrote no transition row at all, so ECR was 0/0 while
    the run had in fact fixed a regression. A `Native` `Failed -> Passed` row
    must move BOTH the combined ECR and the `Native` breakdown.

    [WHY] The rate is CONDITIONAL: one row was failing at N and it was fixed, so
    ECR is 1/1 — not 1/2. The Attacker row that passed twice is never in ECR's
    denominator; it belongs to EIR's (`0/1`), where it says nothing got broken.
    """
    write_session(tmp_path, "repair", decision="AllTestsPassed", schema_version=1)
    write_transitions(
        tmp_path,
        "repair",
        [
            transition(
                from_result="Failed",
                from_class="Logic",
                to_result="Passed",
                source="Native",
            ),
            transition(),  # one AttackerGenerated Passed->Passed
        ],
    )

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "repair"), known_bad=False)
    stats = verdict.metrics

    assert stats["ecr"] == 1
    assert stats["ecr_denominator"] == 1
    assert stats["eir"] == 0
    assert stats["eir_denominator"] == 1
    assert stats["complete"] == 2
    assert stats["per_source"]["Native"] == {
        "ecr": 1,
        "eir": 0,
        "ecr_denominator": 1,
        "eir_denominator": 0,
        "complete": 1,
    }
    assert stats["per_source"]["AttackerGenerated"] == {
        "ecr": 0,
        "eir": 0,
        "ecr_denominator": 0,
        "eir_denominator": 1,
        "complete": 1,
    }
    assert metrics.rate_text(stats["ecr"], stats["ecr_denominator"]) == "1/1 (100.00%)"


def test_a_broken_native_test_counts_as_an_introduced_error(tmp_path):
    """A native test that passed at N and fails at N+1 is EIR, not silence."""
    write_session(tmp_path, "broke-it", decision="MaxAttemptsReached", schema_version=1)
    write_transitions(
        tmp_path,
        "broke-it",
        [transition(from_result="Passed", to_result="Failed", to_class="Logic", source="Native")],
    )

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "broke-it"), known_bad=True)
    assert verdict.metrics["eir"] == 1
    assert verdict.metrics["per_source"]["Native"]["eir"] == 1


def test_a_fixed_static_error_counts_as_a_correction(tmp_path):
    """A blocking finding gone by N+1 is a correction under its OWN source.

    [WHY] A `Static`-class failure is normally excluded (it is the environment's
    fault on an Attacker row). But a row whose own source IS Static is a
    baseline-differenced blocking finding — the candidate's doing — so it must
    count.
    """
    write_session(tmp_path, "static-fixed", decision="AllTestsPassed", schema_version=1)
    write_transitions(
        tmp_path,
        "static-fixed",
        [
            transition(
                from_result="Failed",
                from_class="Static",
                to_result="Passed",
                source="Static",
            )
        ],
    )

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "static-fixed"), known_bad=False)
    assert verdict.metrics["ecr"] == 1
    assert verdict.metrics["per_source"]["Static"] == {
        "ecr": 1,
        "eir": 0,
        "ecr_denominator": 1,
        "eir_denominator": 0,
        "complete": 1,
    }


def test_a_static_class_attacker_row_stays_excluded(tmp_path):
    """The Static allowance is for Static rows only, never for Attacker tests."""
    write_session(tmp_path, "attacker-static", decision="Rejected", schema_version=1)
    write_transitions(
        tmp_path,
        "attacker-static",
        [transition(from_result="Failed", from_class="Static", to_result="Passed")],
    )

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "attacker-static"), known_bad=True)
    assert verdict.metrics["complete"] == 0
    assert verdict.metrics["ecr"] == 0


def test_a_review_round_row_is_excluded_from_ecr(tmp_path):
    """`from == to` is the same candidate — attack depth, not a repair."""
    write_session(tmp_path, "review", decision="Rejected", schema_version=1)
    write_transitions(
        tmp_path,
        "review",
        [
            transition(
                from_result="Failed",
                from_class="Logic",
                to_result="Passed",
                from_patch="patch-same",
                to_patch="patch-same",
            )
        ],
    )

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "review"), known_bad=True)
    assert verdict.metrics["complete"] == 0


def test_zero_denominator_prints_not_applicable_never_a_zero_percent(tmp_path):
    """No countable transition is `n/a`, NOT `0.00%`.

    [WHY] "Nothing countable" and "nothing corrected" are opposite claims about
    a run, and `0.00%` states the second. The distinction is the whole point of
    printing `n/a`.
    """
    write_session(tmp_path, "empty", decision="AllTestsPassed", schema_version=1)

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "empty"), known_bad=False)
    assert verdict.metrics["complete"] == 0

    text = metrics.render_metrics([verdict])
    assert "n/a" in text
    assert "0.00%" not in text, text
    assert "0/0" not in text, text


def test_render_metrics_reports_the_native_breakdown_separately(tmp_path):
    """A pooled number must not hide which source moved it."""
    write_session(tmp_path, "repair", decision="AllTestsPassed", schema_version=1)
    write_transitions(
        tmp_path,
        "repair",
        [
            transition(
                from_result="Failed",
                from_class="Logic",
                to_result="Passed",
                source="Native",
            )
        ],
    )

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "repair"), known_bad=False)
    text = metrics.render_metrics([verdict])
    assert "Native[ECR=1/1 (100.00%) EIR=n/a]" in text
    assert "AttackerGenerated[ECR=n/a EIR=n/a]" in text
    assert "Static[ECR=n/a EIR=n/a]" in text


# ============================================================================
# M3 item V — attempts that edited a file the Creator was not shown in full
# ============================================================================


def test_unshown_edit_count_reads_the_summary_key(tmp_path):
    """The count comes off `run.summary`; absent/malformed is None, never 0."""
    assert metrics.unshown_edit_count("attempts_edited_unshown_file=2") == 2
    assert metrics.unshown_edit_count("attempts_edited_unshown_file=0") == 0
    # [WHY] Absent (a review run, or a pre-field row) and malformed are both
    # "did not measure". A `0` here would be read as "measured, found none".
    assert metrics.unshown_edit_count("final_decision=AllTestsPassed") is None
    assert metrics.unshown_edit_count("attempts_edited_unshown_file=oops") is None
    assert metrics.unshown_edit_count("") is None


def test_classify_reads_the_unshown_count_from_the_session(tmp_path):
    """A repair run that recorded the count carries it; a run without it does not."""
    write_session(
        tmp_path,
        "blind",
        decision="MaxAttemptsReached",
        schema_version=1,
        summary=(
            "attempts_run=3 fixed_at_attempt_1=false "
            "attempts_edited_unshown_file=3 final_decision=MaxAttemptsReached"
        ),
    )
    write_session(tmp_path, "review", decision="Rejected", schema_version=1)

    blind = metrics.classify(metrics.load_session(str(tmp_path), "blind"), known_bad=True)
    review = metrics.classify(
        metrics.load_session(str(tmp_path), "review"), known_bad=True
    )
    assert blind.attempts_edited_unshown_file == 3
    assert review.attempts_edited_unshown_file is None


def test_render_metrics_shows_unshown_edits_and_not_measured(tmp_path):
    """A measured count prints as a number; an absent one prints `n/a`."""
    write_session(
        tmp_path,
        "blind",
        decision="MaxAttemptsReached",
        schema_version=1,
        summary=(
            "attempts_run=3 fixed_at_attempt_1=false "
            "attempts_edited_unshown_file=3 final_decision=MaxAttemptsReached"
        ),
    )
    write_session(tmp_path, "quiet", decision="AllTestsPassed", schema_version=1)

    verdicts = [
        metrics.classify(metrics.load_session(str(tmp_path), "blind"), known_bad=True),
        metrics.classify(metrics.load_session(str(tmp_path), "quiet"), known_bad=False),
    ]
    text = metrics.render_metrics(verdicts)
    assert "UNSHOWN" in text
    # One measured session (3) and one that did not record it (n/a); the pooled
    # cell sums only the measured one.
    assert "3" in text
    assert "n/a" in text
    assert "POOLED" in text


# ============================================================================
# Schema handling (item 11: v0 tolerated, an explicit mismatch is loud)
# ============================================================================


def test_v0_session_loads_with_a_warning_and_v1_loads_clean(tmp_path):
    """A file written before the marker existed is v0, not the current version.

    [WHY] Absence of `schema_version` is a DIFFERENT fact from "an older
    version of the current schema". Reading it as the current version would make
    the script claim fields the file never contained.
    """
    v0_id, v1_id = "v0-session", "v1-session"
    write_session(tmp_path, v0_id, schema_version=None)
    write_session(tmp_path, v1_id, schema_version=metrics.SUPPORTED_SCHEMA_VERSION)

    v0 = metrics.load_session(str(tmp_path), v0_id)
    v1 = metrics.load_session(str(tmp_path), v1_id)

    assert v0.schema_version is None
    assert v0.schema_label == "v0 (legacy)"
    assert any("legacy v0" in warning for warning in v0.warnings), v0.warnings

    assert v1.schema_version == metrics.SUPPORTED_SCHEMA_VERSION
    assert v1.schema_label == "v1"
    assert not any("schema_version" in warning for warning in v1.warnings), v1.warnings


def test_explicit_schema_version_mismatch_fails_loudly(tmp_path):
    """A version this script does not know is a hard error, never a best guess."""
    write_session(tmp_path, "future-session", schema_version=99)
    with pytest.raises(SystemExit) as excinfo:
        metrics.load_session(str(tmp_path), "future-session")
    assert "99" in str(excinfo.value)


def test_summary_field_reads_a_spaced_test_cmd(tmp_path):
    """The Python parser must agree with the Rust producer on where a value ends.

    [WHY] `test_cmd` is a shell command containing spaces and is followed by
    `mode=`. A parser that stopped at the first space would silently re-run the
    wrong command for every session.
    """
    summary = (
        "model=qwen test_cmd=env PYTHONPATH=src /usr/bin/python -m pytest -q "
        "mode=patch_review benchmark_instance=none"
    )
    assert metrics.summary_field(summary, "test_cmd") == (
        "env PYTHONPATH=src /usr/bin/python -m pytest -q"
    )
    assert metrics.summary_field(summary, "mode") == "patch_review"
    assert metrics.summary_field(summary, "absent") is None


# ============================================================================
# Bucketing
# ============================================================================


def test_true_catch_requires_a_discriminating_signal(tmp_path):
    write_session(tmp_path, "catch", decision="Rejected", schema_version=1)
    write_reruns(tmp_path, "catch", [signal(True), signal(False)])

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "catch"), known_bad=True)
    assert verdict.bucket == metrics.TRUE_CATCH
    assert verdict.signals_deciding == 1
    assert verdict.signals_total == 2


def test_spurious_reject_when_every_signal_was_evaluated_and_none_discriminates(tmp_path):
    write_session(tmp_path, "spurious", decision="Rejected", schema_version=1)
    write_reruns(tmp_path, "spurious", [signal(False), signal(False)])

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "spurious"), known_bad=True)
    assert verdict.bucket == metrics.SPURIOUS_REJECT


def test_undetermined_when_no_signal_could_be_evaluated(tmp_path):
    """A missing probe is not a spurious reject.

    [WHY] A v0 session has no static findings and no failure output, so its
    signals come back `null`. Folding that into spurious_reject would report a
    measurement failure as an empirical finding about the model.
    """
    write_session(tmp_path, "legacy", decision="Rejected", schema_version=None)
    write_reruns(tmp_path, "legacy", [signal(None)])

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "legacy"), known_bad=True)
    assert verdict.bucket == metrics.UNDETERMINED


def test_rejected_with_no_rerun_at_all_is_undetermined_not_spurious(tmp_path):
    write_session(tmp_path, "norerun", decision="Rejected", schema_version=1)

    data = metrics.load_session(str(tmp_path), "norerun")
    assert any("signal_reruns" in warning for warning in data.warnings)
    assert metrics.classify(data, known_bad=True).bucket == metrics.UNDETERMINED


def test_a_rejection_with_no_ground_truth_is_undetermined_not_a_catch(tmp_path):
    """Item E8 — `true_catch`/`spurious_reject` both claim the patch was known-bad.

    [WHY] An inconclusive candidate (the patch applied but no readable grade) has no
    ground truth, so `known_bad` is None. A discriminating signal is evidence about the
    SIGNAL, not that the rejected patch was bad; crediting a catch here would put a
    number into the detection rate that nobody's ground truth supports.
    """
    write_session(tmp_path, "unknown", decision="Rejected", schema_version=1)
    write_reruns(tmp_path, "unknown", [signal(True)])

    verdict = metrics.classify(metrics.load_session(str(tmp_path), "unknown"), known_bad=None)
    assert verdict.bucket == metrics.UNDETERMINED


def test_an_unknown_ground_truth_rejection_leaves_the_ground_truth_denominator_empty(tmp_path):
    """The unknown-ground-truth rejection is out of every ground-truth-dependent count.

    `spurious_reject_rate` divides by `true_catch + spurious_reject`, and both are
    claims about a known-bad patch. With no label the rejection is in neither, so the
    denominator stays empty — an honest "n/a" rather than a rate over a row nobody's
    ground truth supports. It is still visible in the bucket totals.
    """
    write_provenance_session(tmp_path, "unknown", provenance_summary())
    write_reruns(tmp_path, "unknown", [signal(True)])

    groups, facts = groups_and_facts(tmp_path)
    pooled = metrics.pooled_metrics(groups[0], facts)
    assert pooled.buckets[metrics.UNDETERMINED] == 1
    assert pooled.buckets[metrics.TRUE_CATCH] == 0
    assert pooled.buckets[metrics.SPURIOUS_REJECT] == 0
    assert pooled.rates()["spurious_reject_rate"] == (0, 0)


def test_false_approval_on_a_known_bad_patch(tmp_path):
    write_session(tmp_path, "approved-bad", decision="AllTestsPassed", schema_version=1)

    verdict = metrics.classify(
        metrics.load_session(str(tmp_path), "approved-bad"), known_bad=True
    )
    assert verdict.bucket == metrics.FALSE_APPROVAL


def test_false_reject_on_a_known_good_patch(tmp_path):
    """A rejection on a known-good patch is wrong even if a signal "discriminated".

    [WHY] B v2 applies the fixed patch, so its candidate IS the known-good tree
    and every signal fires on both sides — the discrimination table can never
    prove the catch. The ground-truth label is the only thing that can say the
    rejection was wrong, and it must win over the table.
    """
    write_session(tmp_path, "rejected-good", decision="Rejected", schema_version=1)
    write_reruns(tmp_path, "rejected-good", [signal(True), signal(False)])

    verdict = metrics.classify(
        metrics.load_session(str(tmp_path), "rejected-good"), known_bad=False
    )
    assert verdict.bucket == metrics.FALSE_REJECT


def test_true_approval_on_a_known_good_patch(tmp_path):
    """Approving a labelled-good patch is correct, not "unlabelled"."""
    write_session(tmp_path, "approved-good", decision="AllTestsPassed", schema_version=1)

    verdict = metrics.classify(
        metrics.load_session(str(tmp_path), "approved-good"), known_bad=False
    )
    assert verdict.bucket == metrics.TRUE_APPROVAL
    assert verdict.bucket != metrics.APPROVED_UNLABELED


def test_approval_without_a_ground_truth_label_claims_nothing(tmp_path):
    write_session(tmp_path, "approved-unknown", decision="AllTestsPassed", schema_version=1)

    verdict = metrics.classify(
        metrics.load_session(str(tmp_path), "approved-unknown"), known_bad=None
    )
    assert verdict.bucket == metrics.APPROVED_UNLABELED


# ============================================================================
# MaxAttemptsReached is a give-up, not an approval (`unresolved`)
# ============================================================================


def test_max_attempts_reached_is_unresolved_whatever_the_label(tmp_path):
    """Burning the attempt budget is "gave up", never an approval.

    [WHY] The Governor returns MaxAttemptsReached only after a round that did
    not pass (a passing round returns AllTestsPassed first), and the Gatekeeper —
    the only component that approves — runs solely on AllTestsPassed. So no
    ground-truth label can turn this into a false_approval or a true_approval.
    """
    write_session(tmp_path, "gave-up", decision="MaxAttemptsReached", schema_version=1)
    data = metrics.load_session(str(tmp_path), "gave-up")

    for label in (True, False, None):
        verdict = metrics.classify(data, label)
        assert verdict.bucket == metrics.UNRESOLVED, label
        assert verdict.bucket not in (
            metrics.FALSE_APPROVAL,
            metrics.TRUE_APPROVAL,
            metrics.APPROVED_UNLABELED,
        ), label


def test_unresolved_is_counted_on_its_own_line(tmp_path):
    """The give-up must not be summed into any approval bucket."""
    write_session(tmp_path, "gave-up", decision="MaxAttemptsReached", schema_version=1)
    verdicts = [
        metrics.classify(metrics.load_session(str(tmp_path), "gave-up"), known_bad=True)
    ]
    table = metrics.render_table(verdicts)

    assert re.search(r"unresolved\s+1", table), table
    assert re.search(r"false_approval\s+0", table), table
    assert re.search(r"true_approval\s+0", table), table
    assert re.search(r"scored rejections\s+0", table), table


# ============================================================================
# InfraFailure is a harness fault, never approval / rejection / unresolved
# ============================================================================


def test_infra_failure_gets_its_own_bucket_whatever_the_label(tmp_path):
    """An LLM call that failed after its retries is not a verdict about the model.

    [WHY] No Attacker test ever ran, so the run says nothing about the candidate.
    Bucketing it as approval, rejection, or the budget-exhausted `unresolved`
    would all be false claims; it must be counted on its own.
    """
    write_session(tmp_path, "infra", decision="InfraFailure", schema_version=1)
    data = metrics.load_session(str(tmp_path), "infra")

    for label in (True, False, None):
        verdict = metrics.classify(data, label)
        assert verdict.bucket == metrics.INFRA_FAILURE, label
        assert verdict.bucket not in (
            metrics.FALSE_APPROVAL,
            metrics.TRUE_APPROVAL,
            metrics.APPROVED_UNLABELED,
            metrics.UNRESOLVED,
            metrics.TRUE_CATCH,
            metrics.SPURIOUS_REJECT,
            metrics.FALSE_REJECT,
        ), label


def test_infra_failure_is_counted_on_its_own_line(tmp_path):
    """The harness fault must not be summed into any approval or reject bucket."""
    write_session(tmp_path, "infra", decision="InfraFailure", schema_version=1)
    verdicts = [
        metrics.classify(metrics.load_session(str(tmp_path), "infra"), known_bad=True)
    ]
    table = metrics.render_table(verdicts)

    assert re.search(r"infra_failure\s+1", table), table
    assert re.search(r"false_approval\s+0", table), table
    assert re.search(r"true_approval\s+0", table), table
    assert re.search(r"unresolved\s+0", table), table
    assert re.search(r"scored rejections\s+0", table), table


# ============================================================================
# JudgeFailed is its own bucket, never a refusal nor an infra fault
# ============================================================================


def test_judge_failed_gets_its_own_bucket_whatever_the_label(tmp_path):
    """M3 item 7 — the judge's call succeeded; only its answer was unreadable.

    [WHY] `infra_failure` would blame the harness for a call that returned, and
    `needs_review_*` would credit the run with a deliberate refusal it never
    made. The run reached no verdict about the candidate, so it is counted alone.
    """
    write_session(tmp_path, "judgefail", decision="JudgeFailed", schema_version=1)
    data = metrics.load_session(str(tmp_path), "judgefail")

    for label in (True, False, None):
        verdict = metrics.classify(data, label)
        assert verdict.bucket == metrics.JUDGE_FAILED, label
        assert verdict.bucket not in (
            metrics.INFRA_FAILURE,
            metrics.NEEDS_REVIEW_ON_BAD,
            metrics.NEEDS_REVIEW_ON_GOOD,
            metrics.NEEDS_REVIEW_UNLABELED,
            metrics.UNDETERMINED,
            metrics.FALSE_APPROVAL,
            metrics.TRUE_APPROVAL,
            metrics.APPROVED_UNLABELED,
        ), label


def test_judge_failed_is_counted_on_its_own_line(tmp_path):
    """The judge failure must not be summed into any approval, reject or infra count.

    [WHY] It is counted, but on its own line and OUTSIDE ``BUCKET_ORDER``: a line for
    it in every report would change the table of every batch already on disk, and
    those rows have no judge.
    """
    write_session(tmp_path, "judgefail", decision="JudgeFailed", schema_version=1)
    verdicts = [
        metrics.classify(metrics.load_session(str(tmp_path), "judgefail"), known_bad=True)
    ]
    table = metrics.render_table(verdicts)

    assert metrics.JUDGE_FAILED not in metrics.BUCKET_ORDER
    assert re.search(r"judge_failed\s+1", table), table
    assert re.search(r"infra_failure\s+0", table), table
    assert re.search(r"needs_review_on_bad\s+0", table), table
    assert re.search(r"false_approval\s+0", table), table
    assert re.search(r"scored rejections\s+0", table), table


def test_a_batch_with_no_judge_prints_no_judge_line(tmp_path):
    """[WHY] M3 item 7 — the judge's count line must not appear for a batch that
    predates it, or every recorded table would be a different table."""
    write_session(tmp_path, "catch", decision="Rejected", schema_version=1)
    write_reruns(tmp_path, "catch", [signal(True)])
    write_session(tmp_path, "approved", decision="AllTestsPassed", schema_version=1)
    verdicts = [
        metrics.classify(metrics.load_session(str(tmp_path), sid), None)
        for sid in ("catch", "approved")
    ]
    table = metrics.render_table(verdicts)

    assert metrics.JUDGE_FAILED not in table, table
    assert "judge_failed" not in table, table


def test_a_judge_failure_states_its_reason(tmp_path):
    write_session(tmp_path, "judgefail", decision="JudgeFailed", schema_version=1)
    verdict = metrics.classify(
        metrics.load_session(str(tmp_path), "judgefail"), None
    )

    assert verdict.bucket == metrics.JUDGE_FAILED
    assert "unreadable" in verdict.explanation


def test_a_judge_approval_is_bucketed_as_an_approval(tmp_path):
    """M3 item 7 — the judge's readable verdicts score as the test-driven arm's do."""
    write_session(tmp_path, "judge-approve", decision="JudgeApproved", schema_version=1)
    approve = metrics.load_session(str(tmp_path), "judge-approve")

    assert metrics.classify(approve, False).bucket == metrics.TRUE_APPROVAL
    assert metrics.classify(approve, True).bucket == metrics.FALSE_APPROVAL
    assert metrics.classify(approve, None).bucket == metrics.APPROVED_UNLABELED


def test_a_judge_rejection_of_a_bad_patch_is_a_true_catch_never_spurious(tmp_path):
    """M3 item 7 — the judge produces no rejecting signals, so the signal-evidence
    test can never fire for it. Its verdict IS the decision.

    [WHY] `spurious_reject` means "the run rejected while every signal failed to
    discriminate" — a claim about signals a judge row does not have. Bucketing a
    judge rejection there would report the judge as a failed measurement.
    """
    write_session(tmp_path, "judge-reject", decision="JudgeRejected", schema_version=1)
    data = metrics.load_session(str(tmp_path), "judge-reject")

    verdict = metrics.classify(data, True)
    assert verdict.bucket == metrics.TRUE_CATCH
    assert verdict.bucket != metrics.SPURIOUS_REJECT
    assert verdict.signals_total == 0, "a judge row has no signals to check"
    assert metrics.classify(data, False).bucket == metrics.FALSE_REJECT
    # An unlabelled judge rejection still claims nothing.
    assert metrics.classify(data, None).bucket == metrics.UNDETERMINED


def test_a_judge_error_is_not_a_spurious_reject(tmp_path):
    """The shared rejection arm must not leak the signal test onto judge tokens."""
    write_session(tmp_path, "judge-reject", decision="JudgeRejected", schema_version=1)
    write_session(tmp_path, "plain-reject", decision="Rejected", schema_version=1)
    write_reruns(tmp_path, "plain-reject", [signal(False)])
    judge = metrics.classify(metrics.load_session(str(tmp_path), "judge-reject"), True)
    plain = metrics.classify(metrics.load_session(str(tmp_path), "plain-reject"), True)

    assert judge.bucket == metrics.TRUE_CATCH
    assert plain.bucket == metrics.SPURIOUS_REJECT


def test_judge_and_vanta_rows_never_share_a_pooled_bucket_table(tmp_path):
    """[WHY] M3 item 7 — `arm` is a provenance axis, so the judge arm's rows can
    never be pooled with the evidence buckets of the arm they are compared with."""
    write_provenance_session(
        tmp_path,
        "judge-row",
        provenance_summary("JudgeRejected", config="judge", arm="judge"),
    )
    write_provenance_session(
        tmp_path,
        "vanta-row",
        provenance_summary("Rejected", config="vanta", arm="b3"),
    )
    write_reruns(tmp_path, "vanta-row", [signal(True)])

    groups, facts = groups_and_facts(
        tmp_path,
        labels={"judge-row": True, "vanta-row": True},
        rows=[manifest_row("judge-row"), manifest_row("vanta-row")],
    )
    tables = {
        group.provenance["arm"]: metrics.pooled_metrics(group, facts)
        for group in groups
    }

    assert set(tables) == {"judge", "b3"}
    assert tables["judge"].buckets[metrics.TRUE_CATCH] == 1
    assert tables["judge"].buckets[metrics.SPURIOUS_REJECT] == 0
    assert tables["b3"].buckets[metrics.TRUE_CATCH] == 1
    # One judge row, one Vanta row: neither table holds the other's rows.
    assert tables["judge"].runs == 1
    assert tables["b3"].runs == 1


# ============================================================================
# NeedsReview is a refusal to decide: its own buckets, never a decision
# ============================================================================


def test_needs_review_gets_its_own_bucket_on_both_sides_of_the_label(tmp_path):
    """§14.6: a refusal is scored as itself, never as a reject or an approval.

    [WHY] `false_reject` is a scored Vanta error and `true_reject` a scored
    catch; a NeedsReview is neither, so folding it into either would report a
    decision the run explicitly declined to make. The ground-truth label says
    only which side the refusal fell on.
    """
    write_session(tmp_path, "refused", decision="NeedsReview", schema_version=1)
    data = metrics.load_session(str(tmp_path), "refused")

    assert metrics.classify(data, True).bucket == metrics.NEEDS_REVIEW_ON_BAD
    assert metrics.classify(data, False).bucket == metrics.NEEDS_REVIEW_ON_GOOD
    assert metrics.classify(data, None).bucket == metrics.NEEDS_REVIEW_UNLABELED

    for label in (True, False, None):
        bucket = metrics.classify(data, label).bucket
        assert bucket not in (
            metrics.FALSE_REJECT,
            metrics.TRUE_CATCH,
            metrics.SPURIOUS_REJECT,
            metrics.UNDETERMINED,
            metrics.FALSE_APPROVAL,
            metrics.TRUE_APPROVAL,
            metrics.APPROVED_UNLABELED,
            metrics.UNRESOLVED,
        ), label


def test_an_unlabelled_refusal_is_not_an_approval_and_not_undetermined(tmp_path):
    """A refusal states a fact even with no label: it is `needs_review_unlabeled`.

    [WHY] The two tempting fallthroughs are both wrong. `approved_unlabeled` says
    "approved, nothing claimed about the patch"; `undetermined` is the bucket for
    a rejection whose signals could not be reconstructed. A run that refused to
    decide neither approved nor rejected anything, and it owes no signal rerun.
    """
    write_session(tmp_path, "refused", decision="NeedsReview", schema_version=1)
    verdict = metrics.classify(metrics.load_session(str(tmp_path), "refused"), None)

    assert verdict.bucket == metrics.NEEDS_REVIEW_UNLABELED
    assert verdict.bucket != metrics.APPROVED_UNLABELED
    assert verdict.bucket != metrics.UNDETERMINED
    assert "refused to decide" in verdict.explanation


def test_the_refusal_counts_sit_next_to_the_decision_counts(tmp_path):
    """The wave digest shows the refusals beside the approvals and rejections."""
    write_session(tmp_path, "refused-good", decision="NeedsReview", schema_version=1)
    write_session(tmp_path, "refused-bad", decision="NeedsReview", schema_version=1)
    verdicts = [
        metrics.classify(metrics.load_session(str(tmp_path), "refused-good"), False),
        metrics.classify(metrics.load_session(str(tmp_path), "refused-bad"), True),
    ]
    table = metrics.render_table(verdicts)

    assert re.search(r"needs_review_on_good\s+1", table), table
    assert re.search(r"needs_review_on_bad\s+1", table), table
    assert re.search(r"needs_review_unlabeled\s+0", table), table
    assert re.search(r"false_reject\s+0", table), table
    assert re.search(r"scored rejections\s+0", table), table
    assert re.search(r"rejections seen\s+0", table), table


def test_only_a_rejection_owes_a_signal_rerun(tmp_path):
    """An approval has nothing to re-run, so the absent probe file is not a warning.

    [WHY] Only a rejection produces rejecting signals worth re-running. Warning
    on an approval would blame the session for a measurement nobody needed. A
    refusal is in the same position as an approval: it issued no rejecting
    signal, so the absent file is not a missing measurement.
    """
    write_session(tmp_path, "approved", decision="AllTestsPassed", schema_version=1)
    write_session(tmp_path, "gave-up", decision="MaxAttemptsReached", schema_version=1)
    write_session(tmp_path, "refused", decision="NeedsReview", schema_version=1)

    approved = metrics.load_session(str(tmp_path), "approved")
    gave_up = metrics.load_session(str(tmp_path), "gave-up")
    refused = metrics.load_session(str(tmp_path), "refused")

    assert not any("signal_reruns" in w for w in approved.warnings), approved.warnings
    assert not any("signal_reruns" in w for w in gave_up.warnings), gave_up.warnings
    assert not any("signal_reruns" in w for w in refused.warnings), refused.warnings


def test_render_table_reports_the_buckets_separately(tmp_path):
    """The outcomes the study reports must never be summed together."""
    write_session(tmp_path, "catch", decision="Rejected", schema_version=1)
    write_reruns(tmp_path, "catch", [signal(True)])
    write_session(tmp_path, "spurious", decision="Rejected", schema_version=1)
    write_reruns(tmp_path, "spurious", [signal(False)])
    write_session(tmp_path, "bad-approval", decision="AllTestsPassed", schema_version=1)
    write_session(tmp_path, "false-reject", decision="Rejected", schema_version=1)
    write_reruns(tmp_path, "false-reject", [signal(False)])

    verdicts = [
        metrics.classify(metrics.load_session(str(tmp_path), "catch"), known_bad=True),
        metrics.classify(metrics.load_session(str(tmp_path), "spurious"), known_bad=True),
        metrics.classify(metrics.load_session(str(tmp_path), "bad-approval"), known_bad=True),
        metrics.classify(metrics.load_session(str(tmp_path), "false-reject"), known_bad=False),
    ]
    table = metrics.render_table(verdicts)

    assert metrics.TRUE_CATCH in table
    assert metrics.SPURIOUS_REJECT in table
    assert metrics.FALSE_APPROVAL in table
    assert metrics.FALSE_REJECT in table
    assert re.search(r"scored rejections\s+3", table), table
    assert re.search(r"rejections seen\s+3", table), table


# ============================================================================
# --labels scopes the report to the declared evaluation set; --all widens it
# ============================================================================


def write_labels(tmp_path, mapping: dict) -> str:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return str(path)


def test_labels_report_only_labelled_sessions_by_default(tmp_path, capsys):
    """A label file is the study's evaluation set, not a filter over everything.

    [WHY] Reporting unlabelled sessions beside the labelled ones invites a reader
    to add rows whose ground truth nobody checked to a count that is supposed to
    be label-driven.
    """
    write_session(tmp_path, "keep-session", decision="AllTestsPassed", schema_version=1)
    write_session(tmp_path, "drop-session", decision="AllTestsPassed", schema_version=1)
    labels = write_labels(tmp_path, {"keep-session": {"known_bad": False}})

    assert metrics.main(["--sessions-dir", str(tmp_path), "--labels", labels]) == 0
    out = capsys.readouterr().out
    assert "keep-ses" in out
    assert "drop-ses" not in out


def test_all_flag_reports_unlabelled_sessions_too(tmp_path, capsys):
    write_session(tmp_path, "keep-session", decision="AllTestsPassed", schema_version=1)
    write_session(tmp_path, "drop-session", decision="AllTestsPassed", schema_version=1)
    labels = write_labels(tmp_path, {"keep-session": {"known_bad": False}})

    assert (
        metrics.main(["--sessions-dir", str(tmp_path), "--labels", labels, "--all"]) == 0
    )
    out = capsys.readouterr().out
    assert "keep-ses" in out
    assert "drop-ses" in out


def test_labels_with_no_matching_session_is_an_error(tmp_path, capsys):
    """A label file that matches nothing is loud, not an empty table."""
    write_session(tmp_path, "drop-session", decision="AllTestsPassed", schema_version=1)
    labels = write_labels(tmp_path, {"other-session": {"known_bad": True}})

    assert metrics.main(["--sessions-dir", str(tmp_path), "--labels", labels]) == 1
    assert "none of the" in capsys.readouterr().err


# ============================================================================
# Pooled, per config (§9.1–§9.4) — provenance grouping, Wilson, the oracle block
# ============================================================================


def provenance_summary(
    decision: str = "Rejected",
    *,
    config: str = "vanta",
    arm: str = "b3",
    mode: str = "repair",
    seed: str = "0",
    model: str = "qwen3-4b",
    top_p: str = "0.9",
    commit: str = "abc1234",
    dirty: str = "false",
    sampling: str = (
        "sampling_preset=default temperature_creator=0.1 max_tokens_creator=32768"
    ),
) -> str:
    """A `run.summary` line carrying the §4.4 provenance keys, as driver.rs writes it.

    [WHY] `arm` sits between `config` and `mode` because that is where `driver.rs`
    writes it (M3 item 8's follow-up); a fixture in another order would still satisfy
    the reader and quietly stop representing the real line.
    """
    return (
        f"attempts_run=1 fixed_at_attempt_1=false final_decision={decision} "
        f"config={config} arm={arm} mode={mode} seed={seed} model={model} top_p={top_p} "
        f"{sampling} commit={commit} dirty={dirty}"
    )


def write_provenance_session(tmp_path, session_id: str, summary: str) -> None:
    record = {
        "session_id": session_id,
        "phase": "run.summary",
        "summary": summary,
        "timestamp": "2026-01-01T00:00:00Z",
        "schema_version": 1,
    }
    (tmp_path / f"{session_id}_session.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )


def manifest_row(
    session_id: str,
    *,
    bug: str = "pysnooper/1",
    seed=0,
    apply_ok: bool = True,
    hidden_pass: bool = True,
    patch_good: bool = True,
    regressions: int = 0,
    inconclusive: bool = False,
) -> dict:
    return {
        "session_id": session_id,
        "bug": bug,
        "seed": seed,
        "oracle": {
            "apply_ok": apply_ok,
            "hidden_pass": hidden_pass,
            "patch_good": patch_good,
            "suite_regressions_count": regressions,
            "suite_regressions": [],
            "inconclusive": inconclusive,
        },
    }


def write_manifest(tmp_path, rows: list[dict], name: str = "manifest.jsonl") -> str:
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def groups_and_facts(tmp_path, labels=None, rows=None):
    """Classify every session in tmp_path, then group it by provenance."""
    labels = labels or {}
    verdicts = []
    provenance = {}
    for session_id in sorted(metrics.discover_sessions(str(tmp_path))):
        data = metrics.load_session(str(tmp_path), session_id)
        verdicts.append(metrics.classify(data, labels.get(session_id)))
        provenance[session_id] = metrics.session_provenance(data)
    return (
        metrics.pool_by_provenance(verdicts, provenance),
        metrics.oracle_by_session(rows or []),
    )


def test_summary_pairs_keeps_a_spaced_value_together():
    """`test_cmd`'s value contains spaces and uppercase assignments.

    [WHY] The sampling schedule is not the only multi-token value on the line.
    A scan that cut every value at the first space would read `test_cmd=env` and
    then treat `PYTHONPATH=src` as a key — exactly the bug `_next_key_start`
    exists to prevent.
    """
    summary = (
        "attempts_run=2 test_cmd=env PYTHONPATH=src python -m pytest -q "
        "final_decision=Rejected"
    )
    pairs = metrics.summary_pairs(summary)

    assert pairs["attempts_run"] == "2"
    assert pairs["test_cmd"] == "env PYTHONPATH=src python -m pytest -q"
    assert pairs["final_decision"] == "Rejected"
    assert "PYTHONPATH" not in pairs


def test_provenance_reads_every_axis_and_the_whole_schedule(tmp_path):
    write_provenance_session(tmp_path, "s1", provenance_summary(seed="7"))
    axes = metrics.provenance_of(provenance_summary(seed="7"))

    for key in metrics.PROVENANCE_KEYS:
        assert axes[key] != metrics.PROVENANCE_UNKNOWN, key
    assert axes["seed"] == "7"
    # The per-site pairs are not in PROVENANCE_KEYS — they are found by prefix,
    # one per call site, and both of the default preset's pins must land here.
    assert axes["sampling_preset"] == "default"
    assert axes["temperature_creator"] == "0.1"
    assert axes["max_tokens_creator"] == "32768"


def test_a_coding_preset_pin_is_provenance_too():
    """Under `coding` the schedule pins `top_p_<site>`; it must be keyed, not lost.

    [WHY] A bare `top_p` is the SERVER's default and a separate axis. If
    `top_p_creator` were dropped, two runs sampling at different `top_p` would
    pool as one experiment.
    """
    assert metrics.is_sampling_key("top_p_creator")
    assert metrics.is_sampling_key("presence_penalty_auditor")
    assert metrics.is_sampling_key("sampling_preset")
    assert not metrics.is_sampling_key("top_p")


def test_an_unrecorded_axis_does_not_pool_with_a_recorded_one(tmp_path):
    """A legacy session has no provenance, and that is its own group.

    [WHY] `unset` is a fact, not a default. Pooling a pre-§4.4 session with one
    that recorded `config=vanta` would claim they are the same experiment.
    """
    write_provenance_session(tmp_path, "recorded", provenance_summary())
    write_session(tmp_path, "legacy", decision="AllTestsPassed")

    groups, _ = groups_and_facts(tmp_path)
    assert len(groups) == 2
    by_seed = {group.provenance["config"]: group for group in groups}
    assert set(by_seed) == {"vanta", metrics.PROVENANCE_UNKNOWN}


def test_a_different_seed_or_schedule_is_a_different_group(tmp_path):
    write_provenance_session(tmp_path, "s1", provenance_summary(seed="0"))
    write_provenance_session(tmp_path, "s2", provenance_summary(seed="1"))
    write_provenance_session(
        tmp_path,
        "s3",
        provenance_summary(seed="0", sampling="sampling_preset=default "
                         "temperature_creator=0.6 max_tokens_creator=32768"),
    )

    groups, _ = groups_and_facts(tmp_path)
    assert len(groups) == 3
    sizes = sorted(len(group.verdicts) for group in groups)
    assert sizes == [1, 1, 1]


def test_the_arm_is_a_provenance_axis(tmp_path):
    """[WHY] M3 item 8 follow-up — two arms under ONE config must not pool.

    `b1` and `suite` already share `baseline`; M3.5's `b3-v2-*` arms share `vanta`.
    Grouping by `config` alone would report one number for two experiments, which is
    the failure the extra axis exists to prevent.
    """
    assert "arm" in metrics.PROVENANCE_KEYS
    write_provenance_session(tmp_path, "a", provenance_summary(config="vanta", arm="b3"))
    write_provenance_session(tmp_path, "b", provenance_summary(config="vanta", arm="b3-v1"))

    groups, _ = groups_and_facts(tmp_path)
    assert len(groups) == 2
    assert {group.provenance["config"] for group in groups} == {"vanta"}
    assert {group.provenance["arm"] for group in groups} == {"b3", "b3-v1"}


def test_pooled_metrics_use_each_metrics_own_denominator(tmp_path):
    """§9.1's rates divide by what the metric asks for, never by `runs`.

    [WHY] `approval_precision` over four runs, only two of which were approvals,
    would read 25 % for a config that got both its approvals right; the
    denominator is the rows the run actually approved.
    """
    summary = provenance_summary()
    write_provenance_session(tmp_path, "catch", summary)
    write_reruns(tmp_path, "catch", [signal(True)])
    write_provenance_session(tmp_path, "spurious", summary)
    write_reruns(tmp_path, "spurious", [signal(False)])
    write_provenance_session(tmp_path, "good-approval", provenance_summary("AllTestsPassed"))
    write_provenance_session(tmp_path, "bad-reject", summary)

    groups, facts = groups_and_facts(
        tmp_path,
        labels={
            "catch": True,
            "spurious": True,
            "good-approval": False,
            "bad-reject": False,
        },
        rows=[
            manifest_row("catch"),
            manifest_row("spurious", hidden_pass=False, patch_good=False),
            manifest_row("good-approval"),
            manifest_row("bad-reject"),
        ],
    )
    assert len(groups) == 1
    pooled = metrics.pooled_metrics(groups[0], facts)
    rates = pooled.rates()

    assert pooled.runs == 4
    assert pooled.buckets[metrics.TRUE_CATCH] == 1
    assert pooled.buckets[metrics.SPURIOUS_REJECT] == 1
    assert pooled.buckets[metrics.TRUE_APPROVAL] == 1
    assert pooled.buckets[metrics.FALSE_REJECT] == 1
    assert rates["approval_precision"] == (1, 1)
    assert rates["spurious_reject_rate"] == (1, 2)
    assert rates["infra_failure_rate"] == (0, 4)
    assert rates["unobservable_rate"] == (0, 4)
    assert rates["resolve_rate"] == (3, 4)
    assert rates["patch_good_rate"] == (3, 4)


def test_the_ecr_wilson_interval_uses_ecrs_conditional_denominator(tmp_path):
    """`n` is the rows FAILING at N, not the rows in the run (C2)."""
    write_provenance_session(tmp_path, "s1", provenance_summary())
    write_transitions(
        tmp_path,
        "s1",
        [transition(from_result="Failed", from_class="Logic", to_result="Passed")],
    )

    groups, facts = groups_and_facts(tmp_path)
    pooled = metrics.pooled_metrics(groups[0], facts)
    rates = pooled.rates()

    assert rates["ECR"] == (1, 1)
    assert rates["EIR"] == (0, 0)
    assert pooled.eir_interval is None
    low, high = pooled.ecr_interval
    assert high == 1.0
    assert low == pytest.approx(0.2065, abs=0.001)


def test_wilson_interval_matches_the_closed_form_and_needs_a_denominator():
    assert metrics.wilson_interval(0, 0) is None
    assert metrics.wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=0.001)
    assert metrics.wilson_interval(5, 10) == pytest.approx(
        (0.2366, 0.7634), abs=0.001
    )
    # An all-success run's interval is clamped at 1.0, never printed above it.
    assert metrics.wilson_interval(10, 10)[1] == 1.0
    assert metrics.wilson_text(None) == "n/a"
    assert metrics.wilson_text((0.11803, 0.76931)) == "[0.118, 0.769]"


def test_an_unobservable_oracle_is_not_a_failed_patch(tmp_path):
    """`apply_ok=false` is the oracle's own gauge, not a resolve failure.

    [WHY] §9.1 keeps them apart: `unobservable_rate` says the harness could not
    grade the patch, and folding those rows into `resolve_rate`'s denominator
    would report a broken oracle as a model failure.
    """
    summary = provenance_summary()
    write_provenance_session(tmp_path, "visible", summary)
    write_provenance_session(tmp_path, "invisible", summary)

    groups, facts = groups_and_facts(
        tmp_path,
        rows=[
            manifest_row("visible"),
            manifest_row("invisible", apply_ok=False, hidden_pass=False, patch_good=False),
        ],
    )
    pooled = metrics.pooled_metrics(groups[0], facts)
    rates = pooled.rates()

    assert pooled.oracle_runs == 2
    assert pooled.observable == 1
    assert pooled.unobservable == 1
    assert rates["resolve_rate"] == (1, 1)
    assert rates["unobservable_rate"] == (1, 2)


def test_an_inconclusive_oracle_is_neither_a_resolve_nor_a_patch_good(tmp_path):
    """Item B4's general rule, review side: applied but unreadable is a third state.

    [WHY] `apply_ok=true` with a hidden run that wrote no report gives no evidence the
    patch resolves anything, so counting it in `resolve_rate`/`patch_good_rate` would
    blame the model for a run nobody could read. It is counted on its own instead.
    """
    summary = provenance_summary()
    write_provenance_session(tmp_path, "run", summary)
    write_provenance_session(tmp_path, "unreadable", summary)

    groups, facts = groups_and_facts(
        tmp_path,
        rows=[
            manifest_row("run"),
            manifest_row("unreadable", hidden_pass=False, patch_good=False, inconclusive=True),
        ],
    )
    pooled = metrics.pooled_metrics(groups[0], facts)
    rates = pooled.rates()

    assert pooled.oracle_runs == 2
    assert pooled.observable == 1
    assert pooled.inconclusive == 1
    assert pooled.unobservable == 0
    assert rates["resolve_rate"] == (1, 1)
    assert rates["patch_good_rate"] == (1, 1)
    assert rates["inconclusive_rate"] == (1, 2)


def test_a_superseded_session_is_dropped_from_the_oracle_index(tmp_path):
    """Item E6 + Q3 — a rerun's new session id fully replaces the old inconclusive one.

    The manifest stays append-only (both rows are on disk); this reader drops the
    session a later row names in `supersedes`, so the unreadable grade can never be
    counted even though its row is still there.
    """
    rows = [
        manifest_row("old", hidden_pass=False, patch_good=False, inconclusive=True),
        {**manifest_row("new"), "supersedes": {"session_id": "old", "key": "k1"}},
    ]
    write_manifest(tmp_path, rows)

    facts = metrics.oracle_by_session(rows)
    assert set(facts) == {"new"}
    assert facts["new"].inconclusive is False
    assert facts["new"].patch_good is True


def test_a_group_without_manifest_rows_reports_na_not_zero(tmp_path):
    write_provenance_session(tmp_path, "s1", provenance_summary())
    groups, facts = groups_and_facts(tmp_path)
    assert facts == {}

    text = metrics.render_pooled(groups, facts)
    assert "resolve_rate n/a" in text
    assert "oracle rows 0/1" in text
    assert "false_approvals[0]: (none)" in text


def test_the_false_approval_list_names_the_bug_and_the_oracle_reason(tmp_path):
    """§9.1: `(bug, seed, reason)` where reason is hidden_fail | suite_regression."""
    summary = provenance_summary("AllTestsPassed")
    write_provenance_session(tmp_path, "hidden", summary)
    write_provenance_session(tmp_path, "regressed", summary)

    groups, facts = groups_and_facts(
        tmp_path,
        labels={"hidden": True, "regressed": True},
        rows=[
            manifest_row("hidden", bug="httpie/1", seed=0, hidden_pass=False, patch_good=False),
            manifest_row("regressed", bug="luigi/1", seed=3, patch_good=False, regressions=2),
        ],
    )
    pooled = metrics.pooled_metrics(groups[0], facts)

    assert pooled.false_approvals == [
        ("httpie/1", 0, "hidden_fail"),
        ("luigi/1", 3, "suite_regression"),
    ]
    assert "false_approvals[2]" in metrics.render_pooled(groups, facts)
    assert "(httpie/1, 0, hidden_fail)" in metrics.render_pooled(groups, facts)


def test_a_false_approval_with_no_manifest_row_prints_question_marks(tmp_path):
    """An absent oracle fact is `?`, never a plausible-looking reason."""
    write_provenance_session(tmp_path, "s1", provenance_summary("AllTestsPassed"))
    groups, facts = groups_and_facts(tmp_path, labels={"s1": True})

    pooled = metrics.pooled_metrics(groups[0], facts)
    assert pooled.false_approvals == [(None, None, "unknown")]
    assert "(?, ?, unknown)" in metrics.render_pooled(groups, facts)


def test_render_pooled_names_every_group_with_its_own_axes(tmp_path):
    write_provenance_session(tmp_path, "a", provenance_summary(config="vanta", seed="0"))
    write_provenance_session(tmp_path, "b", provenance_summary(config="baseline", seed="0"))
    groups, facts = groups_and_facts(tmp_path)

    text = metrics.render_pooled(groups, facts)
    assert "POOLED BY CONFIG" in text
    assert "§9.3" in text
    assert "config=vanta" in text
    assert "config=baseline" in text
    # The schedule prints in the same sorted order `provenance_key` groups by,
    # so a reader can compare two blocks' keys by eye.
    assert (
        "sampling: max_tokens_creator=32768 sampling_preset=default "
        "temperature_creator=0.1" in text
    )
    assert text.count("config=") == 2
    assert "wilson 95% n/a" in text


def test_pool_by_config_reports_the_pooled_table_from_the_cli(tmp_path, capsys):
    write_provenance_session(tmp_path, "s1", provenance_summary())
    manifest = write_manifest(tmp_path, [manifest_row("s1")])

    assert metrics.main(["--sessions-dir", str(tmp_path), "--pool-by-config"]) == 0
    assert "POOLED BY CONFIG" in capsys.readouterr().out

    assert (
        metrics.main(
            [
                "--sessions-dir",
                str(tmp_path),
                "--pool-by-config",
                "--manifest",
                manifest,
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "oracle rows 1/1" in out
    assert "resolve_rate 1/1 (100.00%)" in out


def test_pool_by_config_is_off_unless_asked_for(tmp_path, capsys):
    """The additive section must not appear in the existing report."""
    write_provenance_session(tmp_path, "s1", provenance_summary())
    assert metrics.main(["--sessions-dir", str(tmp_path)]) == 0
    assert "POOLED BY CONFIG" not in capsys.readouterr().out


def test_a_missing_manifest_is_loud(tmp_path, capsys):
    write_provenance_session(tmp_path, "s1", provenance_summary())
    assert (
        metrics.main(
            [
                "--sessions-dir",
                str(tmp_path),
                "--pool-by-config",
                "--manifest",
                str(tmp_path / "nope.jsonl"),
            ]
        )
        == 1
    )
    assert "is not a file" in capsys.readouterr().err


def test_load_manifest_drops_a_torn_last_line(tmp_path):
    """A writer killed mid-append leaves a half row, not a corrupt batch."""
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(manifest_row("s1")) + "\n" + '{"session_id": "s2", "ora',
        encoding="utf-8",
    )
    records = metrics.load_manifest(str(path))
    assert [record["session_id"] for record in records] == ["s1"]

