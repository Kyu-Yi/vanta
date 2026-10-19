// ============================================================================
// RERUN — re-run every signal behind a review-mode REJECTION against a
//         known-good tree, so a script can call the catch true or spurious
// ============================================================================
//
// [VANTA-WHY] Item 11 of the heretic fix list. A review-mode `Rejected` is a
// CLAIM that the candidate is broken. To score the claim the study needs one
// fact the run never records: does the SAME signal that rejected the candidate
// also fire on a tree we already know is good? If it does not, the signal
// discriminates the bad patch from the good one and the rejection is a TRUE
// CATCH; if it fires on both, the signal is measuring something the patch did
// not cause and the rejection is SPURIOUS. This module reconstructs the exact
// rejecting signals from a finished session's own artifacts, re-runs each one
// against both trees, and writes the verdicts where a separate script can read
// them. It DECIDES NOTHING about the session's final bucket — it only produces
// the per-signal observations, and every undecidable case stays `null` rather
// than being guessed into a pass or a fail.
//
// The signals are the three review-mode rejections from the driver's Governor:
//   - generated tests the Attacker wrote that FAILED with failure_class=Logic;
//   - native (the repo's own) tests that regressed original-pass -> patched-fail;
//   - new static findings in a touched file that BLOCKED the round.
// The first two ride in the session JSONL; the third only exists in the
// evidence file, which is why sessions whose evidence file predates the
// `blocking` flag (identified by a MISSING `schema_version`) are reported as
// "static signals unavailable" instead of as "no static signals".

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::agents::attacker::GeneratedTest;
use crate::core::TestResult;
use crate::pipeline::candidate::TestCommand;
use crate::pipeline::harness::{self, NativeSuiteReport, StaticDiagnostic, StaticRun, TestExecutionDetails};
use crate::sandbox::{SandboxKind, backend_kind_label};
use crate::session::SessionRecord;

/// Everything the rerun needs to reconstruct a session's rejecting signals.
///
/// [VANTA-WHY] `good_tree` is a directory OR a patch file and `base_tree` is
/// only meaningful in the second case. They are two fields rather than one
/// enum so the CLI surface stays "pass a path with a flag" and the "a patch
/// needs a base" precondition is checked at exactly one place (the resolver),
/// not in the argument parser as well.
pub struct RerunConfig {
    /// The session whose `session.jsonl` / `evidence.json` describe the run.
    pub session_id: String,
    /// The known-good source: a directory, or a `.patch`/`.diff` file to apply
    /// onto `base_tree`.
    pub good_tree: PathBuf,
    /// Required when `good_tree` is a patch file: the tree the patch applies to.
    pub base_tree: Option<PathBuf>,
    /// Explicit opt-in to re-run a `subprocess` session under shuru.
    ///
    /// [VANTA-WHY] Default is to honour the RECORDED backend, because a rerun
    /// exists to explain the original rejection and a different isolation
    /// boundary is a different experiment. Upgrading to shuru is stricter and so
    /// is allowed — but only when asked for, and only when the test command can
    /// actually run inside the VM (see `test_cmd_points_at_host_path`).
    pub allow_backend_upgrade: bool,
    /// Directory holding the session artifacts (the driver hardcodes
    /// `vanta_sessions`, and this defaults to the same path at the call site).
    pub sessions_dir: PathBuf,
}

/// One blocking static finding, flattened out of the evidence file.
///
/// [VANTA-WHY] The line number is deliberately NOT carried: identity for a
/// static finding is `(file, code, message)` — `StaticDiagnostic`'s own
/// `PartialEq`/`Hash` ignore the line so a patch that shifts every line does
/// not read as "the finding changed". Carrying the line here would tempt a
/// later comparison into using it and reintroduce exactly that false diff.
struct StaticSignal {
    tool: String,
    file: String,
    code: String,
    message: String,
}

/// Re-run a session's review-mode rejection signals against a known-good tree.
///
/// Returns the path of the written `vanta_sessions/<sid>_signal_reruns.json`.
pub async fn rerun_signals(
    cfg: RerunConfig,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let sid = &cfg.session_id;
    let sessions = &cfg.sessions_dir;

    // ---- 1. The session log: the run's provenance + its attempt records ----
    let session_path = sessions.join(format!("{sid}_session.jsonl"));
    if !session_path.exists() {
        return Err(format!(
            "session log not found: {} — nothing to re-run",
            session_path.display()
        )
        .into());
    }
    let text = fs::read_to_string(&session_path)?;
    let mut records: Vec<SessionRecord> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        // [VANTA-WHY] Blank lines are skipped rather than treated as parse
        // errors: the sink flushes one record per line, and a torn final flush
        // (or a trailing newline) must not make an otherwise good log unreadable.
        if line.trim().is_empty() {
            continue;
        }
        let record: SessionRecord = serde_json::from_str(line).map_err(|e| {
            format!(
                "{}:{}: not a valid SessionRecord: {e}",
                session_path.display(),
                i + 1
            )
        })?;
        records.push(record);
    }

    // ---- 2. `session.start` — the backend and the test command ----
    let start = records
        .iter()
        .find(|r| r.phase == "session.start")
        .ok_or("session log has no `session.start` record — cannot read the run's backend/test_cmd")?;
    let recorded_backend = summary_field(&start.summary, "actual_backend")
        .unwrap_or("unknown")
        .to_string();
    let test_cmd_str = summary_field(&start.summary, "test_cmd").map(str::to_string);
    let mode = summary_field(&start.summary, "mode").unwrap_or("unknown");

    // ---- 3. Backend pinning — a rerun must not silently change isolation ----
    // [VANTA-WHY] The recorded backend is the SECURITY BOUNDARY the original
    // rejection was produced under, so the rerun runs under THAT boundary by
    // default. This used to consult `actual_backend_kind()` instead, which on a
    // machine where shuru boots resolved to shuru: session 23999eea recorded
    // `subprocess` and its `test_cmd` was the HOST venv python
    // (/Users/.../heretic/.venv/bin/python), which does not exist inside the
    // VM — so pytest never ran on either tree and the empty result was misread
    // as "the test is absent". Honouring the record keeps the rerun the SAME
    // experiment as the original.
    //   - shuru recorded → re-run under shuru; a machine that cannot boot shuru
    //     is a hard error (a silent downgrade would run untrusted generated code
    //     with no boundary, invalidating the comparison);
    //   - subprocess recorded → re-run under subprocess. Upgrading to shuru is
    //     stricter and is allowed, but ONLY as an explicit opt-in, and never when
    //     the test command points at a host path the VM cannot see.
    let need_shuru = recorded_backend == "shuru"
        || (recorded_backend == "subprocess" && cfg.allow_backend_upgrade);
    let shuru_usable = if need_shuru {
        crate::sandbox::shuru_usable().await
    } else {
        false
    };
    let rerun_kind = resolve_rerun_backend(
        &recorded_backend,
        cfg.allow_backend_upgrade,
        test_cmd_str.as_deref(),
        shuru_usable,
    )
    .map_err(|reason| format!("refusing to re-run session {sid}: {reason}"))?;
    let rerun_backend = backend_kind_label(rerun_kind).to_string();
    if recorded_backend != rerun_backend {
        eprintln!(
            "⚠ session {sid} recorded actual_backend={recorded_backend}; re-running under \
             {rerun_backend} (explicit --allow-backend-upgrade)"
        );
    }
    // [VANTA-WHY] `Backend::spawn()` resolves the backend from
    // `VANTA_SANDBOX_BACKEND`, so pinning the variable here is what makes every
    // harness call below run under the boundary just resolved — not under
    // whatever the invoking shell happened to prefer.
    // SAFETY: the rerun short-circuits before the driver; no other thread in
    // this process reads or writes the environment at this point.
    unsafe {
        std::env::set_var("VANTA_SANDBOX_BACKEND", rerun_backend.as_str());
    }

    // ---- 4. The FINAL attempt and where its records begin ----
    let (final_attempt, attempt_start_index) = final_attempt_start(&records)
        .ok_or("session log has no `attempt.start` record — nothing to re-run")?;
    let final_decision = records
        .iter()
        .find(|r| r.phase == "run.summary")
        .and_then(|r| summary_field(&r.summary, "final_decision"))
        .unwrap_or("unknown")
        .to_string();

    println!(
        "▌ Re-running signals for {sid} (mode={mode}, final_attempt={final_attempt}, decision={final_decision})"
    );

    // ---- 5. Collect the signals -----
    let failing_tests = collect_failing_generated_tests(&records, attempt_start_index);
    let native_regressions = collect_native_regressions(&records, attempt_start_index);
    let (static_signals, static_signals_available) =
        load_blocking_static_signals(sessions, sid, final_attempt)?;

    // ---- 6. The bad tree is the patched snapshot the session already kept ----
    let bad_tree = sessions
        .join(sid)
        .join("debug")
        .join(format!("attempt_{final_attempt}_patched_tree"));
    if !bad_tree.is_dir() {
        return Err(format!(
            "patched tree for attempt {final_attempt} not found at {} — cannot reconstruct the bad side",
            bad_tree.display()
        )
        .into());
    }

    // ---- 7. The good tree: a directory, or base + patch ----
    let (good_tree, good_tree_from_patch, _good_tmp_guard) = resolve_good_tree(&cfg)?;

    // ---- 8. The test command to re-run with ----
    let (test_cmd_text, test_cmd) = match test_cmd_str {
        Some(s) => (s.clone(), TestCommand::from_shell(&s)),
        None => {
            eprintln!(
                "⚠ session.start has no `test_cmd` key — falling back to `python3 -m pytest`"
            );
            let default = TestCommand::default();
            let rendered = format!("{} {}", default.program, default.args.join(" "));
            (rendered, default)
        }
    };

    // ---- 9. Both source trees, concatenated the driver's exact way ----
    let bad_concat = read_tree_concat(&bad_tree)?;
    let good_concat = read_tree_concat(&good_tree)?;

    // ---- 10. Measure both sides for every signal ----
    let shared_tests = load_shared_tests(sessions, sid);
    let timeout = harness::test_timeout_secs();

    // The native suite and the static runs are the two EXPENSIVE, tree-level
    // measurements: each spawns a sandbox. Run each tree's suite/static pass
    // ONCE and look every signal up in the cached report, rather than once per
    // signal (which would be dozens of sandbox boots for one session).
    let native_reports: Option<(NativeSuiteReport, NativeSuiteReport)> =
        if native_regressions.is_empty() {
            None
        } else {
            let bad = harness::run_native_suite(&bad_concat, &test_cmd).await;
            let good = harness::run_native_suite(&good_concat, &test_cmd).await;
            Some((bad, good))
        };
    // [VANTA-WHY] Both sides must be analysed with the SAME repo config, so the
    // set diff compares two runs of one toolchain. `base_tree` is the real repo
    // (it holds `pyproject.toml`); when the good tree is a directory rather than
    // a patch, that directory is the repo and supplies the config instead.
    let config_root: Option<&Path> = match cfg.base_tree.as_deref() {
        Some(base) => Some(base),
        None if !good_tree_from_patch => Some(good_tree.as_path()),
        None => None,
    };
    let static_runs: Option<(StaticRun, StaticRun)> = if static_signals.is_empty() {
        None
    } else {
        let bad = harness::run_static(&bad_concat, &test_cmd, config_root).await;
        let good = harness::run_static(&good_concat, &test_cmd, config_root).await;
        Some((bad, good))
    };

    // [VANTA-WHY] One row per signal, and every row is only ever built from an
    // OBSERVATION. `null` is a first-class value here — "we could not decide" —
    // because the consumer (scripts/metrics.py) buckets a decision on "at least
    // one discriminating signal" and treats an all-null row set as UNDETERMINED.
    // Turning an unobserved signal into `false` would silently manufacture a
    // spurious_reject for a session that simply could not be re-run.
    let mut results: Vec<SignalResult> = Vec::new();

    for name in &failing_tests {
        let found = shared_tests.iter().find(|t| &t.test_name == name);
        let Some(test) = found else {
            results.push(SignalResult {
                kind: "generated_test",
                id: name.clone(),
                bad: None,
                good: None,
                disc: None,
                note: Some("source unavailable in shared_tests.json".to_string()),
                entry: serde_json::json!({
                    "kind": "generated_test",
                    "id": name,
                    "bad_outcome": serde_json::Value::Null,
                    "good_outcome": serde_json::Value::Null,
                    "discriminates": serde_json::Value::Null,
                    "note": "source unavailable in shared_tests.json",
                }),
            });
            continue;
        };
        let bad_details = harness::run_single_test_with_timeout(
            &bad_concat,
            &test.test_source,
            &test_cmd,
            timeout,
        )
        .await;
        let good_details = harness::run_single_test_with_timeout(
            &good_concat,
            &test.test_source,
            &test_cmd,
            timeout,
        )
        .await;
        let (bad_label, good_label, disc, note) =
            generated_signal_observation(&bad_details, &good_details);
        results.push(SignalResult {
            kind: "generated_test",
            id: name.clone(),
            bad: Some(bad_label.clone()),
            good: Some(good_label.clone()),
            disc,
            note: note.clone(),
            entry: serde_json::json!({
                "kind": "generated_test",
                "id": name,
                "bad_outcome": bad_label,
                "good_outcome": good_label,
                "discriminates": option_bool(disc),
                "note": note.map(serde_json::Value::String).unwrap_or(serde_json::Value::Null),
            }),
        });
    }

    for nodeid in &native_regressions {
        let (bad, good, note) = match &native_reports {
            Some((bad_report, good_report)) => {
                native_signal_observation(bad_report, good_report, nodeid)
            }
            None => (None, None, Some("native suite not re-run".to_string())),
        };
        let disc = outcome_discriminates(bad.as_ref(), good.as_ref());
        let bad_name = bad.as_ref().map(test_result_name);
        let good_name = good.as_ref().map(test_result_name);
        results.push(SignalResult {
            kind: "native_test",
            id: nodeid.clone(),
            bad: bad_name.clone(),
            good: good_name.clone(),
            disc,
            note: note.clone(),
            entry: serde_json::json!({
                "kind": "native_test",
                "id": nodeid,
                "bad_outcome": bad_name.map(serde_json::Value::String).unwrap_or(serde_json::Value::Null),
                "good_outcome": good_name.map(serde_json::Value::String).unwrap_or(serde_json::Value::Null),
                "discriminates": option_bool(disc),
                "note": note.map(serde_json::Value::String).unwrap_or(serde_json::Value::Null),
            }),
        });
    }

    for signal in &static_signals {
        let (bad_present, good_present) = match &static_runs {
            Some((bad_run, good_run)) => (
                static_contains(bad_run, &signal.tool, &signal.file, &signal.code, &signal.message),
                static_contains(good_run, &signal.tool, &signal.file, &signal.code, &signal.message),
            ),
            None => (false, false),
        };
        let disc = static_discriminates(bad_present, good_present);
        // [VANTA-WHY] The id keeps the tool's own rendering shape
        // (`file: [code] message`, no line) so a reader can eyeball it against
        // the evidence file even though the line was dropped from identity.
        let id = format!("{}: [{}] {}", signal.file, signal.code, signal.message);
        results.push(SignalResult {
            kind: "static_finding",
            id,
            bad: Some(bad_present.to_string()),
            good: Some(good_present.to_string()),
            disc,
            note: None,
            entry: serde_json::json!({
                "kind": "static_finding",
                "tool": signal.tool,
                "id": format!("{}: [{}] {}", signal.file, signal.code, signal.message),
                "file": signal.file,
                "code": signal.code,
                "message": signal.message,
                "bad_present": bad_present,
                "good_present": good_present,
                "discriminates": option_bool(disc),
                "note": serde_json::Value::Null,
            }),
        });
    }

    // ---- 11. Console table + summary tail ----
    println!(
        "\n{:<16} {:<46} {:<8} {:<8} {}",
        "kind", "id", "bad", "good", "DISCRIMINATES?"
    );
    for r in &results {
        let verdict = match r.disc {
            Some(true) => "yes",
            Some(false) => "no",
            None => "?",
        };
        // [VANTA-WHY] Fix 2: a `tree_run_failed` note has to be VISIBLE, not only
        // in the JSON. Without it a failed run reads as `bad=- good=- ?` and the
        // reader cannot tell "the suite could not run" from "the signal did not
        // reproduce", which are different facts about the session.
        let note = match &r.note {
            Some(note) => format!("  {}", truncate(note, 90)),
            None => String::new(),
        };
        println!(
            "{:<16} {:<46} {:<8} {:<8} {}{}",
            r.kind,
            truncate(&r.id, 46),
            r.bad.clone().unwrap_or_else(|| "-".to_string()),
            r.good.clone().unwrap_or_else(|| "-".to_string()),
            verdict,
            note,
        );
    }
    let total = results.len();
    let discriminating = results.iter().filter(|r| r.disc == Some(true)).count();
    let undecidable = results.iter().filter(|r| r.disc.is_none()).count();
    println!("signals={total} discriminating={discriminating} undecidable={undecidable}");

    // ---- 12. Write the artifact the metrics script reads ----
    let entries: Vec<serde_json::Value> = results.into_iter().map(|r| r.entry).collect();
    let output = serde_json::json!({
        "schema_version": 1,
        "session_id": sid,
        "final_decision": final_decision,
        "final_attempt": final_attempt,
        "recorded_backend": recorded_backend,
        "rerun_backend": rerun_backend,
        "test_cmd": test_cmd_text,
        "bad_tree": bad_tree.display().to_string(),
        "good_tree": good_tree.display().to_string(),
        "good_tree_from_patch": good_tree_from_patch,
        "static_signals_available": static_signals_available,
        "signals": entries,
    });
    fs::create_dir_all(sessions)?;
    let out_path = sessions.join(format!("{sid}_signal_reruns.json"));
    fs::write(&out_path, serde_json::to_string_pretty(&output)?)?;
    println!("▌ Signal reruns → {}", out_path.display());
    Ok(out_path)
}

/// A measured signal row: the JSON entry plus the pieces the console needs.
struct SignalResult {
    kind: &'static str,
    id: String,
    bad: Option<String>,
    good: Option<String>,
    disc: Option<bool>,
    /// A human-readable caveat for the row — a run failure, an unavailable
    /// source. `None` when the row is a plain observation.
    note: Option<String>,
    entry: serde_json::Value,
}

/// Serialise an `Option<bool>` decision as a JSON bool or `null`.
fn option_bool(value: Option<bool>) -> serde_json::Value {
    match value {
        Some(v) => serde_json::Value::Bool(v),
        None => serde_json::Value::Null,
    }
}

/// The name serde would give a `TestResult` (`Passed`/`Failed`/`Error`).
fn test_result_name(result: &TestResult) -> String {
    match result {
        TestResult::Passed => "Passed",
        TestResult::Failed => "Failed",
        TestResult::Error => "Error",
    }
    .to_string()
}

/// Extract `key=value` from a single-line space-separated `key=value` summary.
///
/// [VANTA-WHY] The VALUE may itself contain spaces — `test_cmd=env PYTHONPATH=src
/// /usr/bin/python -m pytest -q mode=patch_review` — so a value does NOT end at
/// the next space, it ends at the next ` key=` token: a space followed by a
/// KEY-ish token and then `=`. A key is `[a-z_][a-z0-9_]*` — LOWERCASE — because
/// every key the producer writes is lowercase, while the variable a `test_cmd`
/// assigns is uppercase by convention (`env PYTHONPATH=src`). Accepting any
/// `NAME=` would cut that value at `env` and re-run the wrong command. The key
/// itself must sit at the start of the summary or right after a space, so
/// searching for `mode` never matches inside `patch_mode=`. This mirrors
/// `scripts/metrics.py::summary_field` byte for byte so the Rust producer and
/// the Python consumer can never disagree about where a value ends.
fn summary_field<'a>(summary: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("{key}=");
    let bytes = summary.as_bytes();
    let mut search = 0usize;
    let value_start = loop {
        let rel = summary.get(search..)?.find(&needle)?;
        let index = search + rel;
        if index == 0 || bytes[index - 1] == b' ' {
            break index + needle.len();
        }
        search = index + 1;
    };
    let end = next_key_start(summary, value_start);
    Some(&summary[value_start..end])
}

/// Byte index where a value ends: the space that begins the next `key=` token,
/// or the end of the string when there is none.
fn next_key_start(summary: &str, value_start: usize) -> usize {
    let bytes = summary.as_bytes();
    let mut i = value_start;
    while i < bytes.len() {
        if bytes[i] == b' ' {
            let mut j = i + 1;
            if j < bytes.len() && (bytes[j].is_ascii_lowercase() || bytes[j] == b'_') {
                while j < bytes.len()
                    && (bytes[j].is_ascii_lowercase()
                        || bytes[j].is_ascii_digit()
                        || bytes[j] == b'_')
                {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'=' {
                    return i;
                }
            }
        }
        i += 1;
    }
    bytes.len()
}

/// The id inside a `phase` of the form `name[<id>]` — the text between the
/// FIRST `[` and the LAST `]`.
///
/// [VANTA-WHY] First-bracket / last-bracket, not a naive split on `]`: a native
/// nodeid is itself full of `[`-free chars but a generated test NAME could
/// contain a `]`; taking the outermost pair keeps the whole id.
fn bracket_id(phase: &str) -> Option<String> {
    let start = phase.find('[')?;
    let end = phase.rfind(']')?;
    if end <= start {
        return None;
    }
    Some(phase[start + 1..end].to_string())
}

/// The last `attempt.start` record: its 1-based attempt number and its index.
///
/// [VANTA-WHY] Only the FINAL attempt's records describe the rejection that
/// ended the run; earlier attempts were superseded by design. Walking back from
/// the end (`rposition`) rather than taking the first match is what makes the
/// "everything AFTER this index" slice below the final attempt's records.
fn final_attempt_start(records: &[SessionRecord]) -> Option<(u32, usize)> {
    let index = records.iter().rposition(|r| r.phase == "attempt.start")?;
    let attempt = summary_field(&records[index].summary, "attempt")?.parse::<u32>().ok()?;
    Some((attempt, index))
}

/// Failing generated tests of the final attempt: `patched.result[<name>]`
/// records whose summary carries `failure_class=Logic`.
///
/// [VANTA-WHY] The other failure classes (TestDefect, Infrastructure, Static,
/// Unknown) do not reject a review round, so replaying them would reconstruct a
/// signal the Governor never acted on. Logic is the one that means "the
/// candidate's behaviour is wrong", which is exactly the claim under audit.
fn collect_failing_generated_tests(records: &[SessionRecord], start_index: usize) -> Vec<String> {
    let mut out = Vec::new();
    for record in records.iter().skip(start_index + 1) {
        if record.phase.starts_with("patched.result[")
            && record.summary.contains("failure_class=Logic")
        {
            if let Some(id) = bracket_id(&record.phase) {
                out.push(id);
            }
        }
    }
    out
}

/// Native regressions of the final attempt: `native_suite.test[<nodeid>]`
/// records whose summary carries `regression=true`.
fn collect_native_regressions(records: &[SessionRecord], start_index: usize) -> Vec<String> {
    let mut out = Vec::new();
    for record in records.iter().skip(start_index + 1) {
        if record.phase.starts_with("native_suite.test[") && record.summary.contains("regression=true")
        {
            if let Some(id) = bracket_id(&record.phase) {
                out.push(id);
            }
        }
    }
    out
}

/// The blocking static findings the final attempt recorded in the evidence file.
///
/// Returns `(signals, available)`. `available == false` is the LEGACY case: the
/// evidence file is missing, or it has NO top-level `schema_version`, which is
/// the marker of the pre-`blocking` format. In the legacy case the caller must
/// report static signals as UNAVAILABLE and never as "there were none" — a v0
/// run recorded its findings but not which of them rejected, so reading absence
/// as a clean bill of health would hide the very signal the 40e84874 post-mortem
/// was about.
fn load_blocking_static_signals(
    sessions_dir: &Path,
    sid: &str,
    final_attempt: u32,
) -> Result<(Vec<StaticSignal>, bool), Box<dyn std::error::Error + Send + Sync>> {
    let path = sessions_dir.join(format!("{sid}_evidence.json"));
    if !path.exists() {
        eprintln!(
            "⚠ {} not found — static signals are UNAVAILABLE (this is not the same as 'no static signals')",
            path.display()
        );
        return Ok((Vec::new(), false));
    }
    let text = fs::read_to_string(&path)?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("{}: not valid JSON: {e}", path.display()))?;
    // [VANTA-WHY] Fix 3. The ONLY thing that makes an evidence file legacy is the
    // ABSENCE of `schema_version`: every file the stamped writer produces carries
    // it, and a file without it predates the `blocking` flag. This used to infer
    // legacy from "no entry has a `blocking` key" instead, which misfires on the
    // most common current file — `schema_version: 1` with an EMPTY
    // `new_static_findings` array — because an empty array trivially has no
    // entry carrying any key. An empty finding list is a statement ("no new
    // findings"), never a missing measurement.
    if value.get("schema_version").is_none() {
        eprintln!(
            "⚠ {} has no `schema_version` (legacy format, pre-`blocking`) — static signals are UNAVAILABLE",
            path.display()
        );
        return Ok((Vec::new(), false));
    }
    let Some(array) = value.get("new_static_findings").and_then(|v| v.as_array()) else {
        eprintln!(
            "⚠ {} has a `schema_version` but no `new_static_findings` array — static signals are UNAVAILABLE",
            path.display()
        );
        return Ok((Vec::new(), false));
    };
    // [VANTA-WHY] The `blocking` key distinguishes "the finding that rejected"
    // from "the findings that were merely reported". With `schema_version`
    // present, the flag is part of the schema, so an empty array means "no
    // blocking findings" — a real, reportable zero.
    let mut signals = Vec::new();
    for entry in array {
        if entry.get("attempt").and_then(|v| v.as_u64()) != Some(u64::from(final_attempt)) {
            continue;
        }
        if entry.get("blocking").and_then(|v| v.as_bool()) != Some(true) {
            continue;
        }
        signals.push(StaticSignal {
            tool: entry.get("tool").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            file: entry.get("file").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            code: entry.get("code").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            message: entry.get("message").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        });
    }
    Ok((signals, true))
}

/// Load the persisted shared test set, if any, for generated-test source lookup.
fn load_shared_tests(sessions_dir: &Path, sid: &str) -> Vec<GeneratedTest> {
    let path = sessions_dir.join(sid).join("debug").join("shared_tests.json");
    if !path.exists() {
        eprintln!("⚠ {} not found — generated-test sources cannot be looked up", path.display());
        return Vec::new();
    }
    match fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str(&s).ok()) {
        Some(tests) => tests,
        None => {
            eprintln!("⚠ {} is not a valid shared-test list — treating as empty", path.display());
            Vec::new()
        }
    }
}

/// Does a static run contain a finding with this `(file, code, message)` identity?
///
/// [VANTA-WHY] The identity is exactly what `StaticDiagnostic`'s hand-written
/// `PartialEq`/`Hash` compare (the line is deliberately excluded), so both the
/// `HashSet` and the `contains` here agree with the driver's set diff — one
/// definition of "the same finding".
fn static_contains(run: &StaticRun, tool: &str, file: &str, code: &str, message: &str) -> bool {
    let probe = StaticDiagnostic {
        file: file.to_string(),
        line: None,
        code: code.to_string(),
        message: message.to_string(),
    };
    let set: HashSet<&StaticDiagnostic> = if tool == "ruff" {
        run.ruff.iter().collect()
    } else {
        run.mypy.iter().collect()
    };
    set.contains(&probe)
}

/// Look a nodeid up in a native-suite report, if present.
fn lookup_native(report: &NativeSuiteReport, nodeid: &str) -> Option<TestResult> {
    report
        .tests
        .iter()
        .find(|(name, _)| name == nodeid)
        .map(|(_, result)| result.clone())
}

/// Which backend a rerun must run under, given what its session recorded.
///
/// [VANTA-WHY] Fix 1: the rerun runs under the RECORDED backend by default. The
/// old code asked this machine's `actual_backend_kind()` instead, so on a Mac
/// where shuru boots a `subprocess` session was silently re-run inside the VM —
/// and since its `test_cmd` named a HOST venv python, pytest never started, and
/// the empty result was reported as the test being absent. Honouring the record
/// keeps the rerun the same experiment as the original run.
///
/// `shuru_usable` is passed in rather than probed here so the decision is a pure
/// function of the machine facts (and therefore unit-testable).
///
/// * `shuru` recorded → shuru, and a machine that cannot boot shuru is a hard
///   error: a silent downgrade would run untrusted generated code with no
///   boundary at all, invalidating the comparison.
/// * `subprocess` recorded → subprocess. Upgrading to shuru is STRICTER and
///   allowed, but only via `allow_upgrade`, and never when `test_cmd` names a
///   host path — the VM cannot see it, so the suite would fail to run and the
///   rerun would measure nothing.
/// * anything else → an error. An unrecognised record is not a licence to guess
///   which boundary to run untrusted code under.
fn resolve_rerun_backend(
    recorded_backend: &str,
    allow_upgrade: bool,
    test_cmd: Option<&str>,
    shuru_usable: bool,
) -> Result<SandboxKind, String> {
    match recorded_backend {
        "shuru" => {
            if !shuru_usable {
                return Err(
                    "it recorded actual_backend=shuru but shuru is not usable on this machine. \
                     Downgrading to subprocess would run untrusted generated code without the \
                     security boundary the original session had, invalidating the rerun."
                        .to_string(),
                );
            }
            Ok(SandboxKind::Shuru)
        }
        "subprocess" => {
            if !allow_upgrade {
                return Ok(SandboxKind::Subprocess);
            }
            if test_cmd_points_at_host_path(test_cmd) {
                return Err(format!(
                    "--allow-backend-upgrade was given, but test_cmd names a host path \
                     ({}). That path does not exist inside the shuru VM, so the suite would \
                     fail to run there and the rerun would measure nothing. Drop \
                     --allow-backend-upgrade to re-run under the recorded subprocess backend.",
                    test_cmd.unwrap_or("")
                ));
            }
            if !shuru_usable {
                return Err(
                    "--allow-backend-upgrade was given but shuru is not usable on this machine."
                        .to_string(),
                );
            }
            Ok(SandboxKind::Shuru)
        }
        other => Err(format!(
            "it recorded actual_backend={other}, which is neither `subprocess` nor `shuru`; \
             cannot decide which security boundary to re-run the untrusted code under"
        )),
    }
}

/// Does a test command name a path that only exists on the host?
///
/// [VANTA-WHY] A shuru run happens inside a Linux microVM that does not mount the
/// host's home directory. A command like
/// `/Users/nathan/heretic/.venv/bin/python -m pytest` therefore CANNOT run in the
/// VM — and that failure is silent: pytest simply never starts. The check is
/// deliberately a small, explicit prefix list rather than "any absolute path":
/// `/usr/bin/python` and `/usr/local/bin/...` are perfectly valid INSIDE the VM,
/// so treating every absolute path as host-only would refuse upgrades that would
/// have worked.
fn test_cmd_points_at_host_path(test_cmd: Option<&str>) -> bool {
    let Some(cmd) = test_cmd else {
        return false;
    };
    cmd.split_whitespace()
        .any(|token| looks_like_host_path(token.trim_matches(|c| c == '"' || c == '\'')))
}

/// Is this single token a macOS host path (user home, mounted volume, or the
/// Homebrew prefix)?
fn looks_like_host_path(token: &str) -> bool {
    token.starts_with("/Users/")
        || token.starts_with("/Volumes/")
        || token.starts_with("/opt/homebrew/")
}

/// The observation for one generated-test signal, from both trees' runs.
///
/// [VANTA-WHY] Fix 2: `report_path_exists == false` means the test command did
/// NOT produce a usable report on that tree — it never started, or pytest
/// collected nothing. That tree was never measured, so the signal is
/// UNDECIDABLE (`None`), not "the test is absent" and certainly not
/// discriminating. `report_path_exists` is exactly the "did this tree reach a
/// verdict at all" flag — a plain assertion failure sets it true.
///
/// Returns `(bad_label, good_label, discriminates, note)`.
fn generated_signal_observation(
    bad: &TestExecutionDetails,
    good: &TestExecutionDetails,
) -> (String, String, Option<bool>, Option<String>) {
    let bad_failed_to_run = !bad.report_path_exists;
    let good_failed_to_run = !good.report_path_exists;
    if !bad_failed_to_run && !good_failed_to_run {
        return (
            test_result_name(&bad.result),
            test_result_name(&good.result),
            outcome_discriminates(Some(&bad.result), Some(&good.result)),
            None,
        );
    }
    let (bad_label, note) = if bad_failed_to_run {
        (
            "run_failed".to_string(),
            tree_run_failed_note("bad", bad.exit_code, &harness::stderr_tail(&bad.stderr)),
        )
    } else {
        (test_result_name(&bad.result), String::new())
    };
    let (good_label, note) = if good_failed_to_run {
        (
            "run_failed".to_string(),
            tree_run_failed_note("good", good.exit_code, &harness::stderr_tail(&good.stderr)),
        )
    } else {
        (test_result_name(&good.result), note)
    };
    (bad_label, good_label, None, Some(note))
}

/// The observation for one native-test signal, from both trees' suite reports.
///
/// [VANTA-WHY] Fix 2: `lookup_native` returning `None` used to mean one thing —
/// "absent". It means two. Either the tree's suite RAN and simply does not
/// contain the nodeid (item 11's absence: a real difference, and it
/// discriminates), or the suite produced NO usable report at all (a command
/// that could not run, which measured nothing). Reading the second as the first
/// manufactured a discriminating signal out of a tree that was never tested —
/// exactly the 23999eea misreport. `collected_nothing` is the discriminator, and
/// a failed run is reported as `tree_run_failed` with its exit code and stderr.
///
/// Returns `(bad, good, note)`.
fn native_signal_observation(
    bad_report: &NativeSuiteReport,
    good_report: &NativeSuiteReport,
    nodeid: &str,
) -> (Option<TestResult>, Option<TestResult>, Option<String>) {
    if bad_report.collected_nothing() {
        return (
            None,
            None,
            Some(tree_run_failed_note(
                "bad",
                bad_report.exit_code,
                &bad_report.stderr_tail,
            )),
        );
    }
    if good_report.collected_nothing() {
        return (
            None,
            None,
            Some(tree_run_failed_note(
                "good",
                good_report.exit_code,
                &good_report.stderr_tail,
            )),
        );
    }
    let bad_result = lookup_native(bad_report, nodeid);
    let good_result = lookup_native(good_report, nodeid);
    let note = if good_result.is_none() {
        Some("absent from the good tree".to_string())
    } else {
        None
    };
    (bad_result, good_result, note)
}

/// The note for a tree whose test command produced no usable measurement.
///
/// [VANTA-WHY] Fix 2: the exit code and the last stderr lines are what turn "we
/// have no number" into something a reader can act on — a nonzero exit with
/// "No such file or directory" says the command never started, while exit 2 with
/// a collection error says the tree does not import. Both must be reported as
/// `tree_run_failed`, never as the test being absent.
fn tree_run_failed_note(tree: &str, exit_code: Option<i32>, stderr_tail: &str) -> String {
    let exit = exit_code
        .map(|code| code.to_string())
        .unwrap_or_else(|| "none".to_string());
    let tail = stderr_tail.trim();
    if tail.is_empty() {
        format!("tree_run_failed: {tree} tree produced no test report (exit={exit})")
    } else {
        format!("tree_run_failed: {tree} tree produced no test report (exit={exit}); stderr: {tail}")
    }
}

/// The discrimination rule for a test signal.
///
/// `None` = could not decide (missing observation, infrastructure error, or the
/// signal did not even reproduce on the bad tree).
///
/// [VANTA-WHY] The asymmetry is the whole point: a PASSING bad side is not
/// evidence of a spurious rejection, it means the signal never reproduced, so
/// it is UNDECIDABLE rather than `false`. `Error` is infrastructure, not a
/// verdict about the code, so it never decides in either direction. A `Failed`
/// bad side with an ABSENT good side decides the same as Failed/Passed — the
/// good tree not containing the test is a difference, and item 11 says absence
/// counts.
fn outcome_discriminates(bad: Option<&TestResult>, good: Option<&TestResult>) -> Option<bool> {
    match bad {
        None => None,
        Some(TestResult::Failed) => match good {
            Some(TestResult::Passed) => Some(true),
            None => Some(true),
            Some(TestResult::Failed) => Some(false),
            Some(TestResult::Error) => None,
        },
        Some(TestResult::Passed) | Some(TestResult::Error) => None,
    }
}

/// The discrimination rule for a static finding.
///
/// [VANTA-WHY] Symmetric to `outcome_discriminates`: absent on the bad tree
/// means the finding did not reproduce (undecidable), present on both means the
/// patch did not cause it (spurious), present on bad / absent on good means it
/// discriminates.
fn static_discriminates(bad_present: bool, good_present: bool) -> Option<bool> {
    match (bad_present, good_present) {
        (false, _) => None,
        (true, false) => Some(true),
        (true, true) => Some(false),
    }
}

/// Resolve the good tree: a directory used as-is, or a patch applied to a copy
/// of `base_tree` inside a `TempDir`.
///
/// Returns `(tree_path, from_patch, tempdir_guard)`. The guard MUST be kept
/// alive by the caller: dropping the `TempDir` deletes the applied tree, and we
/// still need it for every measurement below.
fn resolve_good_tree(
    cfg: &RerunConfig,
) -> Result<(PathBuf, bool, Option<tempfile::TempDir>), Box<dyn std::error::Error + Send + Sync>> {
    if cfg.good_tree.is_dir() {
        return Ok((cfg.good_tree.clone(), false, None));
    }
    if cfg.good_tree.is_file() {
        let base = cfg.base_tree.as_ref().ok_or(
            "--good-tree is a patch file; --base-tree PATH (the tree it applies to) is required",
        )?;
        let guard = tempfile::TempDir::new()?;
        let tree = guard.path().to_path_buf();
        copy_tree_recursive(base, &tree)?;
        let patch = fs::canonicalize(&cfg.good_tree)?;
        // [VANTA-WHY] `git apply` with cwd = the copied tree and an ABSOLUTE
        // patch path, so the patch is interpreted against the copy regardless of
        // where the process was launched from. `-p1` matches the `a/`/`b/`
        // prefixes git-produced diffs carry; `--whitespace=nowarn` stops a
        // harmless whitespace difference in the base from failing the apply.
        let out = Command::new("git")
            .arg("apply")
            .arg("-p1")
            .arg("--whitespace=nowarn")
            .arg(&patch)
            .current_dir(&tree)
            .output()?;
        if !out.status.success() {
            return Err(format!(
                "git apply failed ({}): patch={}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                out.status,
                patch.display(),
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            )
            .into());
        }
        return Ok((tree, true, Some(guard)));
    }
    Err(format!(
        "--good-tree {} is neither a directory nor a patch file",
        cfg.good_tree.display()
    )
    .into())
}

/// Copy a directory tree recursively, skipping `.git`.
///
/// [VANTA-WHY] `.git` is skipped only to avoid copying an entire object
/// database into the temp dir for a large clone; `git apply` does not need it
/// (it reads the patch and the working files, not the repo). A base given as a
/// git WORKTREE has a tiny `.git` file, so nothing of value is lost either way.
/// Symlinks are skipped deliberately: a symlink into the original tree would
/// let a `git apply` escape the copy.
fn copy_tree_recursive(src: &Path, dst: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_string_lossy() == ".git" {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            copy_tree_recursive(&from, &to)?;
        } else if file_type.is_file() {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Read a whole source tree as the driver's `# FILE:` concatenation.
///
/// [VANTA-WHY] This MUST stay byte-identical to the driver's
/// `build_source_concat`: `harness::parse_source_files` splits on `# FILE: `
/// headers and strips exactly the `\n\n` separator, so any change to the header
/// text or the join would make the harness write the wrong files (or refuse the
/// tree outright). It is a copy rather than a shared helper because the driver
/// builds its concat from an in-memory `HashMap`, not from a directory.
pub fn read_tree_concat(root: &Path) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut files: Vec<(String, String)> = Vec::new();
    collect_tree_files(root, root, &mut files)?;
    // Sort by relative path so the concat is deterministic and matches the
    // driver's `BTreeMap`-ordered output.
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let joined = files
        .into_iter()
        .map(|(path, contents)| format!("# FILE: {path}\n{contents}"))
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok(joined)
}

/// Walk `dir` under `root`, collecting UTF-8 text files with the noise filtered.
fn collect_tree_files(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, String)>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        // [VANTA-WHY] These three names are version control, bytecode caches and
        // dependency trees: none is source, all can be huge, and a `.pyc` or a
        // `_manifest.json` under the tree would otherwise be fed to the harness
        // as if it were the candidate's code.
        if name == ".git" || name == "__pycache__" || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_tree_files(root, &path, out)?;
        } else if file_type.is_file() {
            if name.ends_with(".pyc")
                || name.ends_with(".pyo")
                || name.ends_with("_manifest.json")
            {
                continue;
            }
            let bytes = fs::read(&path)?;
            // [VANTA-WHY] Non-UTF-8 files (binaries, images) are skipped, not
            // lossily decoded: a replacement-character version of a binary would
            // be written into the sandbox as a corrupt file the harness could not
            // distinguish from real source.
            let Ok(contents) = String::from_utf8(bytes) else {
                continue;
            };
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let relative = relative.to_string_lossy().replace('\\', "/");
            out.push((relative, contents));
        }
    }
    Ok(())
}

/// Truncate a table cell so a long nodeid does not wrap the console line.
fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let mut out: String = value.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{SCHEMA_VERSION, SessionId};

    /// Build a minimal session record for exercising the pure collectors.
    fn record(phase: &str, summary: &str) -> SessionRecord {
        SessionRecord {
            session_id: SessionId::new("s"),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: None,
            testing_iteration_id: None,
            test_id: None,
            phase: phase.to_string(),
            summary: summary.to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            benchmark_instance: None,
            benchmark_label: None,
        }
    }

    /// [VANTA-WHY] The value of `test_cmd` contains spaces, so a naive
    /// split-on-whitespace would truncate it at `python3`. This pins that the
    /// value runs to the next ` key=` token (here `mode=`), and that the key
    /// after it is still reachable — the exact shape scripts/metrics.py relies
    /// on to agree with this producer.
    #[test]
    fn summary_field_keeps_a_spaced_value_and_reads_the_next_key() {
        let summary = "model=qwen test_cmd=python3 -m pytest -q mode=patch_review";
        assert_eq!(
            summary_field(summary, "test_cmd"),
            Some("python3 -m pytest -q")
        );
        assert_eq!(summary_field(summary, "mode"), Some("patch_review"));
    }

    /// [VANTA-WHY] The REAL heretic runs record `test_cmd=env PYTHONPATH=src …`,
    /// and `PYTHONPATH=` is itself a `NAME=` token. If any `NAME=` counted as a
    /// key, the value would be cut at `env` and the rerun would re-execute the
    /// wrong command for every session whose test command starts with `env`.
    /// Uppercase names are therefore NOT keys (the producer's keys are all
    /// lowercase) and the value must survive to ` mode=`.
    #[test]
    fn summary_field_keeps_a_value_that_assigns_an_uppercase_variable() {
        let summary = "deps_freeze=none test_cmd=env PYTHONPATH=src /usr/bin/python -m pytest -q mode=patch_review benchmark_instance=none";
        assert_eq!(
            summary_field(summary, "test_cmd"),
            Some("env PYTHONPATH=src /usr/bin/python -m pytest -q")
        );
        assert_eq!(summary_field(summary, "mode"), Some("patch_review"));
        assert_eq!(summary_field(summary, "benchmark_instance"), Some("none"));
    }

    /// [VANTA-WHY] A trailing key has no following ` key=` token, so the value
    /// must run to the end of the summary rather than vanish.
    #[test]
    fn summary_field_reads_a_trailing_value_to_the_end() {
        let summary = "attempt=3 mode=patch_review test_cmd=run it all";
        assert_eq!(summary_field(summary, "test_cmd"), Some("run it all"));
    }

    /// [VANTA-WHY] An absent key is `None`, never an empty string — the caller
    /// uses absence to pick a fallback, and an empty value would look present.
    /// A key that only appears as a SUFFIX of another (`mode` inside
    /// `patch_mode=`) must not match either.
    #[test]
    fn summary_field_returns_none_for_absent_and_suffix_keys() {
        let summary = "model=qwen patch_mode=review";
        assert_eq!(summary_field(summary, "test_cmd"), None);
        assert_eq!(summary_field(summary, "mode"), None);
    }

    /// [VANTA-WHY] Pins the full decision table: the asymmetry between a `false`
    /// (reproduces but does not discriminate) and a `None` (cannot tell) is what
    /// keeps a spurious_reject from being manufactured out of a missing probe.
    #[test]
    fn outcome_discriminates_pins_every_case() {
        use TestResult::{Error, Failed, Passed};
        assert_eq!(outcome_discriminates(Some(&Failed), Some(&Passed)), Some(true));
        assert_eq!(outcome_discriminates(Some(&Failed), None), Some(true));
        assert_eq!(outcome_discriminates(Some(&Failed), Some(&Failed)), Some(false));
        assert_eq!(outcome_discriminates(Some(&Failed), Some(&Error)), None);
        assert_eq!(outcome_discriminates(None, Some(&Passed)), None);
        assert_eq!(outcome_discriminates(None, None), None);
        assert_eq!(outcome_discriminates(Some(&Passed), Some(&Failed)), None);
        assert_eq!(outcome_discriminates(Some(&Error), Some(&Passed)), None);
    }

    /// [VANTA-WHY] Same asymmetry for static findings: not-reproduced on the bad
    /// tree is undecidable, present-on-both is spurious, present-bad/absent-good
    /// discriminates.
    #[test]
    fn static_discriminates_pins_the_rule() {
        assert_eq!(static_discriminates(false, false), None);
        assert_eq!(static_discriminates(false, true), None);
        assert_eq!(static_discriminates(true, false), Some(true));
        assert_eq!(static_discriminates(true, true), Some(false));
    }

    /// [VANTA-WHY] Only the FINAL attempt's rejecting records may be replayed:
    /// earlier attempts were superseded, and re-running their Logic failures
    /// would reconstruct signals the Governor never rejected on. A record with
    /// `failure_class=None` did not reject either, so it must not be collected.
    #[test]
    fn signal_collection_reads_only_the_final_attempt() {
        let records = vec![
            record("session.start", "mode=patch_review"),
            record("attempt.start", "attempt=1 mode=patch_review"),
            record(
                "patched.result[old_logic]",
                "outcome=Failed intent=FeatureAcceptance failure_class=Logic",
            ),
            record(
                "native_suite.test[tests/test_a.py::A::test_old]",
                "original=Passed patched=Some(Failed) regression=true",
            ),
            record("attempt.start", "attempt=2 mode=patch_review"),
            record(
                "patched.result[x]",
                "outcome=VerifiedWithinBounds intent=FeatureAcceptance failure_class=Logic",
            ),
            record(
                "patched.result[y]",
                "outcome=VerifiedWithinBounds intent=FeatureAcceptance failure_class=None",
            ),
            record(
                "native_suite.test[tests/test_b.py::B::test_reg]",
                "original=Passed patched=Some(Failed) regression=true",
            ),
            record(
                "native_suite.test[tests/test_c.py::C::test_ok]",
                "original=Passed patched=Some(Passed) regression=false",
            ),
        ];

        let (attempt, index) = final_attempt_start(&records).unwrap();
        assert_eq!(attempt, 2, "the LAST attempt.start wins");

        assert_eq!(collect_failing_generated_tests(&records, index), vec!["x".to_string()]);
        assert_eq!(
            collect_native_regressions(&records, index),
            vec!["tests/test_b.py::B::test_reg".to_string()]
        );
    }

    /// [VANTA-WHY] The tree reader must reproduce the driver's `# FILE:`
    /// concatenation exactly, filter the noise (`__pycache__`, `*.pyc`,
    /// `*_manifest.json`) the driver never snapshots, and sort by relative path
    /// so the output is deterministic.
    #[test]
    fn read_tree_concat_skips_noise_sorts_and_emits_headers() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("pkg/__pycache__")).unwrap();
        fs::write(root.join("pkg/b.py"), "b = 1\n").unwrap();
        fs::write(root.join("pkg/a.py"), "a = 1\n").unwrap();
        fs::write(root.join("pkg/__pycache__/a.cpython-311.pyc"), "junk").unwrap();
        fs::write(root.join("attempt_1_patched_tree_manifest.json"), "{}").unwrap();

        let concat = read_tree_concat(root).unwrap();

        assert!(concat.contains("# FILE: pkg/a.py\n"), "got: {concat}");
        assert!(concat.contains("# FILE: pkg/b.py\n"), "got: {concat}");
        assert!(!concat.contains("__pycache__"), "bytecode caches are skipped");
        assert!(!concat.contains("_manifest.json"), "manifests are skipped");

        let a = concat.find("# FILE: pkg/a.py").unwrap();
        let b = concat.find("# FILE: pkg/b.py").unwrap();
        assert!(a < b, "files must be sorted by relative path, got: {concat}");
    }

    // ---- Fix 1: the rerun honours the RECORDED backend -----------------------

    /// [VANTA-WHY] Fix 1: a `subprocess` session is re-run under subprocess even
    /// on a machine that CAN boot shuru. The old code asked the machine instead
    /// of the record, silently moved the rerun into the VM, and the host-path
    /// test command then never ran — session 23999eea's misreport.
    #[test]
    fn a_subprocess_session_reruns_under_subprocess_without_an_explicit_upgrade() {
        let kind = resolve_rerun_backend(
            "subprocess",
            false,
            Some("env PYTHONPATH=src /Users/nathan/heretic/.venv/bin/python -m pytest -q"),
            true,
        )
        .expect("no upgrade was asked for, so the host path is irrelevant");
        assert_eq!(kind, SandboxKind::Subprocess);
    }

    #[test]
    fn a_shuru_session_reruns_under_shuru() {
        assert_eq!(
            resolve_rerun_backend("shuru", false, None, true).unwrap(),
            SandboxKind::Shuru
        );
    }

    /// [VANTA-WHY] A recorded shuru session on a machine that cannot boot shuru
    /// must be a hard error: a silent downgrade would run the untrusted
    /// generated tests on the host with no boundary at all.
    #[test]
    fn a_shuru_session_on_a_machine_without_shuru_is_refused() {
        let err = resolve_rerun_backend("shuru", false, None, false).unwrap_err();
        assert!(err.contains("shuru is not usable"), "got: {err}");
    }

    /// [VANTA-WHY] The explicit upgrade is the ONLY way a subprocess session may
    /// leave subprocess, and it is refused when the recorded test command names a
    /// host path — the VM cannot see it, so the suite would fail to run there and
    /// the rerun would measure nothing.
    #[test]
    fn a_backend_upgrade_is_refused_when_the_test_cmd_names_a_host_path() {
        let err = resolve_rerun_backend(
            "subprocess",
            true,
            Some("/Users/nathan/heretic/.venv/bin/python -m pytest -q"),
            true,
        )
        .unwrap_err();
        assert!(err.contains("host path"), "got: {err}");
    }

    #[test]
    fn an_explicit_backend_upgrade_needs_a_machine_that_can_run_shuru() {
        assert_eq!(
            resolve_rerun_backend("subprocess", true, Some("python3 -m pytest -q"), true).unwrap(),
            SandboxKind::Shuru
        );
        assert!(
            resolve_rerun_backend("subprocess", true, None, false).is_err(),
            "asking for an upgrade on a machine that cannot run shuru is an error"
        );
    }

    /// [VANTA-WHY] An unrecognised record is not a licence to guess which
    /// boundary to run untrusted code under.
    #[test]
    fn an_unknown_recorded_backend_is_refused() {
        let err = resolve_rerun_backend("unknown", false, None, true).unwrap_err();
        assert!(err.contains("neither `subprocess` nor `shuru`"), "got: {err}");
    }

    /// [VANTA-WHY] The host-path check must stay NARROW: `/usr/bin/python` is
    /// perfectly valid INSIDE the VM, so treating every absolute path as
    /// host-only would refuse upgrades that would have worked.
    #[test]
    fn host_path_detection_is_narrow() {
        assert!(test_cmd_points_at_host_path(Some(
            "/Users/nathan/heretic/.venv/bin/python -m pytest"
        )));
        assert!(test_cmd_points_at_host_path(Some(
            "env PYTHONPATH=src /Volumes/Data/heretic/.venv/bin/python -m pytest"
        )));
        assert!(test_cmd_points_at_host_path(Some(
            "\"/Users/nathan/venv/bin/python\" -m pytest"
        )));
        assert!(!test_cmd_points_at_host_path(Some("python3 -m pytest -q")));
        assert!(!test_cmd_points_at_host_path(Some("/usr/bin/python3 -m pytest")));
        assert!(!test_cmd_points_at_host_path(Some(
            "env PYTHONPATH=src /usr/local/bin/python -m pytest"
        )));
        assert!(!test_cmd_points_at_host_path(None));
    }

    // ---- Fix 2: a tree that could not run is never "absent" ------------------

    fn suite_report(tests: &[(&str, TestResult)], ran: bool) -> NativeSuiteReport {
        NativeSuiteReport {
            tests: tests
                .iter()
                .map(|(name, result)| ((*name).to_string(), result.clone()))
                .collect(),
            failure_outputs: std::collections::HashMap::new(),
            ran,
            exit_code: Some(0),
            stderr_tail: String::new(),
        }
    }

    fn details(
        result: TestResult,
        report_path_exists: bool,
        exit_code: Option<i32>,
        stderr: &str,
    ) -> TestExecutionDetails {
        TestExecutionDetails {
            result,
            stdout: String::new(),
            stderr: stderr.to_string(),
            exit_code,
            timed_out: false,
            report_path_exists,
            sandbox_root: None,
        }
    }

    /// [VANTA-WHY] Fix 2, the 23999eea case verbatim: the good tree's suite
    /// produced nothing at all, so the nodeid is not "absent" — the tree was
    /// never measured. The signal must be UNDECIDABLE and must carry the exit
    /// code and stderr, never read as a discriminating difference.
    #[test]
    fn a_good_tree_that_could_not_run_is_tree_run_failed_not_absent() {
        let bad = suite_report(&[("tests/test_k.py::K::test_x", TestResult::Failed)], true);
        let good = NativeSuiteReport {
            tests: Vec::new(),
            failure_outputs: std::collections::HashMap::new(),
            ran: false,
            exit_code: Some(127),
            stderr_tail:
                "env: /Users/nathan/heretic/.venv/bin/python: No such file or directory".to_string(),
        };

        let (bad_r, good_r, note) =
            native_signal_observation(&bad, &good, "tests/test_k.py::K::test_x");

        let note = note.expect("a failed run must carry a note");
        assert!(note.starts_with("tree_run_failed"), "got: {note}");
        assert!(note.contains("good tree"), "got: {note}");
        assert!(note.contains("exit=127"), "got: {note}");
        assert!(note.contains("No such file or directory"), "got: {note}");
        assert_eq!(
            outcome_discriminates(bad_r.as_ref(), good_r.as_ref()),
            None,
            "a tree that never ran cannot discriminate anything"
        );
    }

    /// [VANTA-WHY] Fix 2 must NOT break item 11's real absence: when BOTH suites
    /// ran and the good tree simply does not contain the regressed test, that is
    /// a genuine difference and still discriminates.
    #[test]
    fn a_genuine_absence_still_discriminates() {
        let bad = suite_report(&[("tests/test_k.py::K::test_x", TestResult::Failed)], true);
        let good = suite_report(
            &[("tests/test_k.py::K::test_other", TestResult::Passed)],
            true,
        );

        let (bad_r, good_r, note) =
            native_signal_observation(&bad, &good, "tests/test_k.py::K::test_x");

        assert_eq!(note.as_deref(), Some("absent from the good tree"));
        assert_eq!(bad_r, Some(TestResult::Failed));
        assert_eq!(good_r, None);
        assert_eq!(
            outcome_discriminates(bad_r.as_ref(), good_r.as_ref()),
            Some(true)
        );
    }

    #[test]
    fn a_generated_test_whose_command_did_not_run_is_tree_run_failed() {
        let bad = details(TestResult::Failed, true, Some(1), "");
        let good = details(
            TestResult::Error,
            false,
            Some(2),
            "ERROR: file or directory not found: test_bate.py",
        );

        let (bad_label, good_label, disc, note) = generated_signal_observation(&bad, &good);

        assert_eq!(bad_label, "Failed", "the tree that ran keeps its real outcome");
        assert_eq!(good_label, "run_failed");
        assert_eq!(disc, None);
        let note = note.expect("a failed run must carry a note");
        assert!(note.starts_with("tree_run_failed"), "got: {note}");
        assert!(note.contains("exit=2"), "got: {note}");
        assert!(note.contains("test_bate.py"), "got: {note}");
    }

    /// [VANTA-WHY] The untouched path must be unchanged: both trees ran, bad
    /// failed, good passed → discriminates, no note.
    #[test]
    fn a_clean_generated_test_comparison_still_discriminates() {
        let bad = details(TestResult::Failed, true, Some(1), "");
        let good = details(TestResult::Passed, true, Some(0), "");

        let (bad_label, good_label, disc, note) = generated_signal_observation(&bad, &good);

        assert_eq!((bad_label.as_str(), good_label.as_str()), ("Failed", "Passed"));
        assert_eq!(disc, Some(true));
        assert!(note.is_none());
    }

    // ---- Fix 3: `schema_version`, not the `blocking` key, marks legacy -------

    fn write_evidence(dir: &Path, sid: &str, json: &str) {
        fs::write(dir.join(format!("{sid}_evidence.json")), json).unwrap();
    }

    /// [VANTA-WHY] Fix 3: the common CURRENT file — `schema_version: 1` with an
    /// EMPTY `new_static_findings` array — must report static signals as
    /// AVAILABLE with zero signals. The old code inferred legacy from "no entry
    /// carries a `blocking` key", which an empty array satisfies vacuously.
    #[test]
    fn an_empty_finding_list_with_a_schema_version_is_available_not_legacy() {
        let dir = tempfile::tempdir().unwrap();
        write_evidence(
            dir.path(),
            "s1",
            r#"{"schema_version":1,"new_static_findings":[]}"#,
        );

        let (signals, available) = load_blocking_static_signals(dir.path(), "s1", 1).unwrap();

        assert!(available, "schema_version=1 means the file is current");
        assert!(
            signals.is_empty(),
            "an empty list is zero signals, not an unknown"
        );
    }

    /// [VANTA-WHY] Fix 3: legacy is exactly "no schema_version". A file that
    /// predates the stamp recorded findings but not which blocked, so the
    /// category is UNAVAILABLE — never "there were none".
    #[test]
    fn a_file_without_a_schema_version_is_legacy_and_unavailable() {
        let dir = tempfile::tempdir().unwrap();
        write_evidence(
            dir.path(),
            "s2",
            r#"{"new_static_findings":[{"attempt":1,"tool":"ruff","file":"a.py","code":"F821","message":"Undefined name"}]}"#,
        );

        let (signals, available) = load_blocking_static_signals(dir.path(), "s2", 1).unwrap();

        assert!(
            !available,
            "no schema_version means the pre-`blocking` format"
        );
        assert!(signals.is_empty());
    }

    /// [VANTA-WHY] Fix 3 must not stop a current file from reporting its real
    /// blocking findings — the whole reason the file is read.
    #[test]
    fn a_current_file_still_reports_its_blocking_findings() {
        let dir = tempfile::tempdir().unwrap();
        write_evidence(
            dir.path(),
            "s3",
            r#"{"schema_version":1,"new_static_findings":[
                {"attempt":1,"tool":"ruff","file":"a.py","code":"F821","message":"Undefined name","blocking":true},
                {"attempt":1,"tool":"mypy","file":"b.py","code":"return-value","message":"bad return","blocking":false},
                {"attempt":2,"tool":"ruff","file":"c.py","code":"E501","message":"line too long","blocking":true}
            ]}"#,
        );

        let (signals, available) = load_blocking_static_signals(dir.path(), "s3", 1).unwrap();

        assert!(available);
        assert_eq!(signals.len(), 1, "only attempt 1's BLOCKING finding counts");
        assert_eq!(signals[0].code, "F821");
        assert_eq!(signals[0].tool, "ruff");
    }
}
