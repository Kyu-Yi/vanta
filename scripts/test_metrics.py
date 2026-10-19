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
) -> str:
    """Write a minimal session JSONL and return the sessions dir."""
    record = {
        "session_id": session_id,
        "patch_attempt_id": None,
        "testing_iteration_id": None,
        "test_id": None,
        "phase": "run.summary",
        "summary": f"attempts_run=1 fixed_at_attempt_1=false final_decision={decision}",
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


def test_only_a_rejection_owes_a_signal_rerun(tmp_path):
    """An approval has nothing to re-run, so the absent probe file is not a warning.

    [WHY] Only a rejection produces rejecting signals worth re-running. Warning
    on an approval would blame the session for a measurement nobody needed.
    """
    write_session(tmp_path, "approved", decision="AllTestsPassed", schema_version=1)
    write_session(tmp_path, "gave-up", decision="MaxAttemptsReached", schema_version=1)

    approved = metrics.load_session(str(tmp_path), "approved")
    gave_up = metrics.load_session(str(tmp_path), "gave-up")

    assert not any("signal_reruns" in w for w in approved.warnings), approved.warnings
    assert not any("signal_reruns" in w for w in gave_up.warnings), gave_up.warnings


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
