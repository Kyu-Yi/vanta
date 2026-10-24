// ============================================================================
// DIAGNOSTICIAN — 4-way FailureClass triage (Spec §7 Phase 2)
// ============================================================================
//
// [VANTA-WHY] When a test FAILS on the patched repo, the Diagnostician must
// classify WHY it failed so the right repair strategy is used:
//
//   Infrastructure → sandbox/network/timeout (retry, not a code bug)
//   Static         → new lint/type errors introduced by the patch
//   Logic          → runs but wrong behavior (the patch has a logic bug)
//   Unknown        → can't classify (default before analysis)
//
// Diagnostic signal: ruff + mypy diff vs baseline. If pytest dies during
// import/collect without new static signals, treat that as Infrastructure so the
// harness does not blame the model for a broken tree. If tests run but assert
// fails → Logic.
//
// [VANTA-WHY] Order matters: a test that RAN and failed is classified by its own
// failure BEFORE the static short-circuit, and any new ruff/mypy finding is
// recorded separately in the payload rather than overwriting that class. With
// Static first, a real behavioural failure was labelled Static whenever the
// patch also carried a lint/type finding (session e8d393be).
// ============================================================================

use crate::agents::attacker::{
    GeneratedTest, attacker_simplify_test, simplification_keeps_the_asserted_contract,
};
use crate::config::LlmConfig;
use crate::core::{FailureClass, TestResult};
use crate::pipeline::bate::{BateOutcome, validate_test_against_baseline};
use crate::pipeline::candidate::TestCommand;
use crate::pipeline::harness::{self, TestExecutionDetails};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, info, warn};

// ============================================================================
// DIAGNOSTICIAN_PROMPT_VERSION — the run-identity tag for the Diagnostician
// ============================================================================
//
// [VANTA-WHY] The Diagnostician is rule-based today (no prompt text of its own),
// but its Unknown path calls the Attacker's simplify prompt. `session.start`
// lists every agent prompt so that a future Diagnostician-side prompt change is
// visible in the log rather than silently counted as "no change"; that change
// must bump this tag.
//
// [VANTA-WHY] Bumped for the context-localization change (§1.6.2): the
// Diagnostician is in the set of roles shown source, so what it reads changed
// even though its own text did not.
pub const DIAGNOSTICIAN_PROMPT_VERSION: &str = "2";

/// Cap on the pytest output saved with a failing test (the fix-list's ~4 KB).
/// A full traceback for one test is far below this; the cap only bounds a
/// pathological case (e.g. a dump of a huge fixture) so evidence.json stays
/// readable.
pub const FAILURE_OUTPUT_LIMIT: usize = 4096;

/// First `limit` bytes of `s`, cut on a char boundary, with a marker when cut.
pub fn truncate_output(s: &str, limit: usize) -> String {
    if s.len() <= limit {
        return s.to_string();
    }
    let mut end = limit;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n… [truncated: {} of {} bytes shown]",
        &s[..end],
        end,
        s.len()
    )
}

// ---------------------------------------------------------------------------
// Reading the failure's own traceback out of a pytest run
// ---------------------------------------------------------------------------
//
// [VANTA-WHY] A pytest run is a SEQUENCE of titled sections — `FAILURES`,
// `short test summary info`, `warnings summary`, `--- Captured stdout ---`
// blocks — and only the FAILURES/ERRORS sections carry the traceback of the
// exception that was actually raised. Scanning the raw output bottom-up (as
// this file first did) reads whichever section pytest printed LAST: in the
// py3.8 BugsInPy images `future` emits a `DeprecationWarning` for every imported
// module, so a failing test ends with
// `…/future/standard_library/__init__.py:65: DeprecationWarning` in its warnings
// summary, and that warning — never the `NameError` the test hit — was taken as
// the failure site (PySnooper/3, ff241462); every failing test in that image
// was affected.

/// The title between the runs of `ch` that bookend `line`, when `line` is a
/// pytest marker (`===== FAILURES =====`, `--- Captured stdout call ---`,
/// `______ test_x ______`). Runs shorter than 3 characters are rejected so that
/// ordinary prose (`- one - two -`) is not read as a marker.
fn marker_title(line: &str, ch: char) -> Option<&str> {
    let t = line.trim();
    if !t.starts_with(ch) || !t.ends_with(ch) {
        return None;
    }
    let start = t.find(|c| c != ch).unwrap_or(t.len());
    let end = t.rfind(|c| c != ch).map_or(start, |i| i + 1);
    if start < 3 || t.len() - end < 3 {
        return None;
    }
    Some(t[start..end].trim())
}

/// Is `line` a bare run of `ch` (a titleless pytest rule)?
fn is_rule(line: &str, ch: char) -> bool {
    let t = line.trim();
    t.len() >= 4 && t.chars().all(|c| c == ch)
}

/// Is `line` a pytest marker of kind `ch`, titled or bare?
fn is_marker(line: &str, ch: char) -> bool {
    is_rule(line, ch) || marker_title(line, ch).is_some()
}

/// Parse a bare `path.py:LINE: ExceptionType` traceback line.
///
/// The exception type is Capitalised, which also rejects the `test_x.py:14: `
/// lines pytest prints for a traceback's intermediate frames: those carry no
/// exception type, so only the last such line is the frame where the exception
/// was raised.
fn parse_frame_line(line: &str) -> Option<(String, u32, String)> {
    let mut parts = line.splitn(3, ':');
    let (file, lineno, rest) = match (parts.next(), parts.next(), parts.next()) {
        (Some(f), Some(l), Some(r)) => (f.trim(), l.trim(), r.trim()),
        _ => return None,
    };
    if !file.ends_with(".py") {
        return None;
    }
    let line_no = lineno.parse::<u32>().ok()?;
    let exc = rest
        .split(|c: char| c == ':' || c.is_whitespace())
        .next()
        .unwrap_or("");
    if !exc.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        return None;
    }
    Some((file.to_string(), line_no, exc.to_string()))
}

/// The innermost frame of the exception raised within `lines`.
///
/// Scans forward, remembering every frame line, and stops where the traceback
/// ends: a section boundary, a `--- Captured stdout ---` block, or the next
/// failing test's `______ test_name ______` header. The last frame seen before
/// that is the innermost one.
fn innermost_frame(lines: &[&str]) -> Option<(String, u32, String)> {
    let mut found = None;
    for raw in lines {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if is_marker(line, '=') || is_marker(line, '-') {
            break;
        }
        // A `______ test_name ______` header opens the NEXT failing test; a bare
        // `_ _ _ _` run only separates the frames of the current traceback.
        if is_marker(line, '_') && found.is_some() {
            break;
        }
        if let Some(frame) = parse_frame_line(line) {
            found = Some(frame);
        }
    }
    found
}

/// Every `FAILURES` / `ERRORS` section of a pytest run, and whether the run had
/// one at all (a run that died before testing prints none).
fn failure_sections(output: &str) -> (bool, Vec<Vec<&str>>) {
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    let mut open: Option<Vec<&str>> = None;
    for raw in output.lines() {
        if is_marker(raw, '=') {
            if let Some(block) = open.take() {
                blocks.push(block);
            }
            // A FAILURES/ERRORS boundary opens a section; every other boundary
            // (short test summary, warnings summary, the final "N failed" line)
            // closes the one that is open.
            let opens = marker_title(raw, '=')
                .is_some_and(|t| t.eq_ignore_ascii_case("failures") || t.eq_ignore_ascii_case("errors"));
            if opens {
                open = Some(Vec::new());
            }
            continue;
        }
        if let Some(block) = open.as_mut() {
            block.push(raw);
        }
    }
    if let Some(block) = open {
        blocks.push(block);
    }
    (!blocks.is_empty(), blocks)
}

/// The innermost frame in an output that has NO FAILURES/ERRORS section (a
/// fragment of a run, or a runner that prints only the traceback line), skipping
/// the sections that can never hold the failure's own frame.
///
/// [VANTA-WHY] Only the SUMMARY sections are skipped, and each of them is the
/// LAST thing a pytest run prints. The session-start banner is deliberately NOT
/// one of them: its lines (`rootdir:`, `plugins:`, `collected 1 item`) cannot be
/// mistaken for frames, and skipping it swallowed every fragment that begins
/// with the banner — `an_assertion_failure_saves_the_test_source_and_the_failure_site`
/// lost its exception type that way (e432d2c).
fn innermost_frame_anywhere(output: &str) -> Option<(String, u32, String)> {
    let mut found = None;
    let mut skipped = false;
    for raw in output.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if is_marker(line, '=') {
            skipped = marker_title(line, '=').is_some_and(|t| {
                let t = t.to_ascii_lowercase();
                t.contains("warnings summary") || t.contains("short test summary")
            });
            continue;
        }
        if is_marker(line, '-') {
            skipped = true;
        }
        if skipped {
            continue;
        }
        if let Some(frame) = parse_frame_line(line) {
            found = Some(frame);
        }
    }
    found
}

/// Locate a pytest failure's innermost frame: the frame of the exception that
/// was actually raised, never a warning or a captured-output line that merely
/// follows it in the log.
///
/// The triple is what lets a later step tell a failure in the CODE UNDER TEST
/// ("the patch has a bug") from one raised in the TEST's own file ("the test is
/// broken"), so what it points at has to be the failure itself.
///
/// Returns `(file, line, exception_type)`, each `None` when the output has no
/// recognisable frame (e.g. the test never got that far).
pub fn parse_failure_site(output: &str) -> (Option<String>, Option<u32>, Option<String>) {
    let (has_sections, blocks) = failure_sections(output);
    let frame = if has_sections {
        blocks.iter().find_map(|block| innermost_frame(block))
    } else {
        innermost_frame_anywhere(output)
    };
    match frame {
        Some((file, line, exc)) => (Some(file), Some(line), Some(exc)),
        None => (None, None, None),
    }
}

/// The signature of a failure: `(exception_type, file, line)` from its output.
///
/// [VANTA-WHY] Two runs are "the same failure" when they raise the same
/// exception at the same site. Comparing signatures — not merely "both trees
/// failed" — is what lets a patch-insensitive failure be told apart from a
/// coincidence: two DIFFERENT bugs that happen to both fail must stay Logic.
fn failure_signature(exec: &TestExecutionDetails) -> Option<(String, Option<String>, Option<u32>)> {
    let output = format!("{}\n{}", exec.stdout, exec.stderr);
    let (file, line, exception_type) = parse_failure_site(&output);
    exception_type.map(|e| (e, file, line))
}

/// Does the same test fail the SAME way on the unpatched tree?
///
/// [VANTA-WHY] Item 2 of the heretic fix list: a failure that reproduces
/// identically on the original tree carries no information about the patch, so
/// it must not reject. `false` whenever either side has no parseable signature
/// — an unreadable failure must not be silently reclassified as a test defect.
fn is_patch_insensitive(patched: &TestExecutionDetails, original: &TestExecutionDetails) -> bool {
    match (failure_signature(patched), failure_signature(original)) {
        (Some(p), Some(o)) => p == o,
        _ => false,
    }
}

/// Is this path one of the test runner's own generated test files?
///
/// [VANTA-WHY] The harness writes each generated test to a fixed name in the
/// sandbox (`test_run.py` for a review round, `test_bate.py` for validation), so
/// a frame in a `test_*` / `*_test` / `conftest` file is the generated test's own
/// code — not the code under test.
fn is_test_file(path: &str) -> bool {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    name.starts_with("test_") || name.ends_with("_test.py") || name == "conftest.py"
}

fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Does `source` declare `class <name>`?
fn declares_class(source: &str, name: &str) -> bool {
    let needle = format!("class {name}");
    source.match_indices(&needle).any(|(i, _)| {
        source[i + needle.len()..]
            .chars()
            .next()
            .map_or(true, |c| !(c.is_ascii_alphanumeric() || c == '_'))
    })
}

/// Does the failure name an object of a type the TEST itself declares?
///
/// [VANTA-WHY] `'FakeTensor' object has no attribute 'shape'` is the TEST's fake
/// (56eb4fd3 round 2), not the patch: the test stubbed the model incompletely,
/// so wherever the crash lands — in the test's file or inside the code under
/// test that was handed the fake — the crash is a property of the test. This is
/// the one signal that reaches a crash landing in the code under test, which the
/// test-file check below cannot see. Re-instated with its original signal after
/// the e432d2c rewrite dropped it.
fn names_test_defined_type(output: &str, test_source: &str) -> bool {
    for (idx, _) in output.match_indices("' object") {
        let before = &output[..idx];
        let Some(open) = before.rfind('\'') else {
            continue;
        };
        let name = &before[open + 1..];
        if is_identifier(name) && declares_class(test_source, name) {
            return true;
        }
    }
    false
}

/// Is an `AssertionError` the TEST's own `assert` / mock `assert_*`?
///
/// [VANTA-WHY] An assertion failure IS a verdict on the patch: the test checked
/// behaviour and the behaviour was wrong. Two shapes qualify — a plain `assert`
/// in the test file (pytest reports the test's own file as the innermost frame),
/// and a mock assertion, whose message names `to have been called` / `Calls:`.
/// An `AssertionError` raised anywhere else (a library, or production code's own
/// invariant) is not the test speaking, so it does not qualify.
fn assertion_is_the_tests_own(output: &str, frame_file: Option<&str>) -> bool {
    const MOCK_MARKERS: [&str; 4] = [
        "to have been called",
        "to not have been called",
        "Calls:",
        "assert_called",
    ];
    if MOCK_MARKERS.iter().any(|m| output.contains(m)) {
        return true;
    }
    match frame_file {
        Some(f) => is_test_file(f),
        // pytest reported an assertion but no parseable frame; the assertion is
        // still the test's own check.
        None => true,
    }
}

/// Locate the fault: is this failure about the PATCH or about the TEST?
///
/// [VANTA-WHY] `Logic` claims "the patch is wrong", so before blaming the patch
/// we must find WHERE the exception was raised. BATE has already proved how the
/// test behaves on the original tree, so a validated test that fails on the
/// patched tree is evidence about the patch: an exception raised inside the code
/// under test counts, whether or not the patch touched the file the traceback
/// ends in. A no-op patch leaves the bug exactly where it was, and "the patch
/// did not touch this file" is a reason to REJECT it, not to excuse it — the
/// earlier rule demoted exactly those failures to TestDefect and so threw away
/// the BugReproduction counterexamples that argued for rejection (PySnooper/3,
/// 3ef3ac33).
///
/// The one real test defect is the generated test's OWN code breaking: the
/// innermost frame of a non-assertion exception is inside the test file itself
/// — a test that instantiated a CLI-reading settings object and got `SystemExit`
/// — or the exception names an object of a class the test itself declared, so
/// the crash is on the test's incomplete fake wherever it landed (56eb4fd3's
/// `'FakeTensor' object has no attribute 'shape'`).
///
/// Returns the TestDefect reason + payload, or `None` when the patch is a fair
/// target. Also `None` when no fault can be located at all: an unreadable
/// failure must not be quietly turned into a test defect.
pub fn locate_fault(
    execution: &TestExecutionDetails,
    test_source: &str,
) -> Option<(String, serde_json::Value)> {
    let output = format!("{}\n{}", execution.stdout, execution.stderr);
    let (frame_file, frame_line, exception_type) = parse_failure_site(&output);
    let exception = exception_type.clone().unwrap_or_default();
    if exception.is_empty() {
        return None;
    }

    // An assertion the test itself made is a verdict on the patch, not a defect
    // in the test.
    if exception == "AssertionError" && assertion_is_the_tests_own(&output, frame_file.as_deref()) {
        return None;
    }
    let site = format!(
        "{}:{}",
        frame_file.clone().unwrap_or_else(|| "?".into()),
        frame_line.map(|l| l.to_string()).unwrap_or_else(|| "?".into()),
    );

    // A crash on an object of a type the TEST declared is the test's incomplete
    // fake, even when the frame is in the code under test.
    if names_test_defined_type(&output, test_source) {
        let reason = format!(
            "Failure located in the TEST's own object ({exception} at {site}) — the generated test \
             defined the type that broke, so the failure is a property of the test, not the patch. \
             Sent back to the Attacker for revision."
        );
        return Some((
            reason,
            json!({
                "signal": "test_defect_test_defined_type",
                "exception_type": exception_type,
                "frame_file": frame_file,
                "frame_line": frame_line,
            }),
        ));
    }
    // Anything raised outside the generated test's own file was raised by the
    // code under test, so it is the patch's business.
    if !frame_file.as_deref().is_some_and(is_test_file) {
        return None;
    }

    let reason = format!(
        "Failure located in the TEST's own code ({exception} at {site}) — the generated test raised \
         it before the code under test could run, so the failure is a property of the test, not the \
         patch. Sent back to the Attacker for revision."
    );
    Some((
        reason,
        json!({
            "signal": "test_defect_exception_in_test_file",
            "exception_type": exception_type,
            "frame_file": frame_file,
            "frame_line": frame_line,
        }),
    ))
}

// ============================================================================
// DiagnosticResult — output of Diagnostician
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    pub failure_class: Option<FailureClass>,
    pub reason: String,
    /// Optional structured payload (e.g. ruff diff, mypy errors).
    pub structured_payload: Option<serde_json::Value>,
}

// ============================================================================
// Diagnostician inputs — what we feed the 4-way classifier
// ============================================================================
#[derive(Debug, Clone, Default)]
pub struct DiagnosticInputs {
    /// Did the test runner exit with a non-zero code due to import/syntax errors?
    pub import_or_syntax_error: bool,
    /// ruff stderr diff (patched vs original) — does patched introduce new lint errors?
    pub ruff_new_errors: bool,
    /// mypy stderr diff — does patched introduce new type errors?
    pub mypy_new_errors: bool,
    /// Wall time hit the timeout?
    pub timed_out: bool,
    /// Test ran to completion but assertion failed?
    pub assertion_failed: bool,
    /// Network connectivity check failed (network killed mid-run)?
    /// [VANTA-WHY] M2 EXIT TEST: "Kill the network mid-run and it classifies as
    /// Infrastructure, not a model error." If network_alive() returns false,
    /// any failure is Infrastructure regardless of other signals.
    pub network_down: bool,
    /// Raw ruff stderr from patched run (for structured_payload).
    pub ruff_stderr: String,
    /// Raw mypy stderr from patched run (for structured_payload).
    pub mypy_stderr: String,
    /// Raw pytest stdout from the failing test run.
    pub pytest_stdout: String,
    /// Raw pytest stderr from the failing test run.
    pub pytest_stderr: String,
    /// Whether pytest failed before producing report.json.
    pub pytest_report_missing: bool,
    /// The failing test's own source, saved with the evidence so a reader can
    /// see WHAT failed without hunting for the GeneratedTest artifact.
    pub test_source: String,
}

// ============================================================================
// diagnose — the 4-way classifier (Spec §7 Phase 2)
// ============================================================================
pub fn diagnose(inputs: &DiagnosticInputs) -> DiagnosticResult {
    debug!(?inputs, "Diagnostician: classifying failure");

    // Priority order: Infrastructure > Static > Logic > Unknown
    // [VANTA-WHY] Infrastructure failures are environmental — retrying might
    // fix them. Static failures are the patch breaking the build. Logic is the
    // patch having wrong behavior. Unknown is the fallback.

    // Network-down takes highest priority — if the network is killed mid-run,
    // the failure is environmental, not a code bug. (M2 EXIT TEST requirement.)
    if inputs.network_down {
        info!("Diagnostician → Infrastructure (network down)");
        return DiagnosticResult {
            failure_class: Some(FailureClass::Infrastructure),
            reason: "Network connectivity failed mid-run — infrastructure issue".into(),
            structured_payload: Some(serde_json::json!({
                "signal": "network_down",
            })),
        };
    }

    if inputs.timed_out {
        info!("Diagnostician → Infrastructure (timeout)");
        return DiagnosticResult {
            failure_class: Some(FailureClass::Infrastructure),
            reason: "Test timed out — likely infrastructure/resource issue".into(),
            structured_payload: Some(serde_json::json!({
                "signal": "timeout",
            })),
        };
    }

    // [VANTA-WHY] Item 5 of the heretic fix list — order: a test that RAN and
    // failed is classified by its own failure, BEFORE the static short-circuit.
    // The old order put "new ruff/mypy errors" above "assertion failed", so a
    // real behavioural failure was labelled Static whenever the patch also
    // carried a lint/type finding (e8d393be). A static finding is evidence about
    // the patch, not a substitute for the failure that actually happened: it is
    // recorded SEPARATELY in this payload and never overwrites the test's class.
    if inputs.assertion_failed {
        info!("Diagnostician → Logic (assertion failed)");
        // [VANTA-WHY] This payload used to be `{"signal":"assertion_failed"}` and
        // nothing more: no test source, no pytest output, no traceback. A reader
        // of evidence.json then could not tell WHY the patch was blamed — the
        // exact defect the heretic runs exposed. Save the test's own source, the
        // failure output (capped), and the innermost frame so the claim "the
        // patch has a logic bug" is auditable from the record alone.
        let full_output = format!("{}\n{}", inputs.pytest_stdout, inputs.pytest_stderr);
        let (frame_file, frame_line, exception_type) = parse_failure_site(&full_output);
        let mut payload = serde_json::json!({
            "signal": "assertion_failed",
            "test_source": inputs.test_source,
            "failure_output": truncate_output(&full_output, FAILURE_OUTPUT_LIMIT),
            "exception_type": exception_type,
            "frame_file": frame_file,
            "frame_line": frame_line,
        });
        if inputs.ruff_new_errors || inputs.mypy_new_errors {
            let mut static_errors = serde_json::json!({
                "ruff_new_errors": inputs.ruff_new_errors,
                "mypy_new_errors": inputs.mypy_new_errors,
            });
            if inputs.ruff_new_errors && !inputs.ruff_stderr.is_empty() {
                static_errors["ruff_stderr"] = serde_json::Value::String(inputs.ruff_stderr.clone());
            }
            if inputs.mypy_new_errors && !inputs.mypy_stderr.is_empty() {
                static_errors["mypy_stderr"] = serde_json::Value::String(inputs.mypy_stderr.clone());
            }
            payload["new_static_errors"] = static_errors;
        }
        return DiagnosticResult {
            failure_class: Some(FailureClass::Logic),
            reason: "Test ran but assertion failed — patch has a logic bug".into(),
            structured_payload: Some(payload),
        };
    }

    if inputs.ruff_new_errors || inputs.mypy_new_errors {
        info!("Diagnostician → Static (new lint/type errors)");
        // [VANTA-WHY] Populate structured_payload with the actual ruff/mypy
        // stderr so M3's Creator can see exactly what static errors the patch
        // introduced. This is the "structured JSON diagnostic payload back to
        // Creator" requirement from Spec §7 Phase 2. Reached only when the test
        // did NOT run to a failure — a test that ran is classified by its
        // failure above.
        let mut payload = serde_json::json!({
            "signal": "new_static_errors",
            "ruff_new_errors": inputs.ruff_new_errors,
            "mypy_new_errors": inputs.mypy_new_errors,
        });
        if inputs.ruff_new_errors && !inputs.ruff_stderr.is_empty() {
            payload["ruff_stderr"] = serde_json::Value::String(inputs.ruff_stderr.clone());
        }
        if inputs.mypy_new_errors && !inputs.mypy_stderr.is_empty() {
            payload["mypy_stderr"] = serde_json::Value::String(inputs.mypy_stderr.clone());
        }
        return DiagnosticResult {
            failure_class: Some(FailureClass::Static),
            reason: "Patch introduces new ruff/mypy errors".into(),
            structured_payload: Some(payload),
        };
    }

    if inputs.import_or_syntax_error {
        info!(
            "Diagnostician → Infrastructure (pytest import/collect error without new static diagnostics)"
        );
        return DiagnosticResult {
            failure_class: Some(FailureClass::Infrastructure),
            reason: "Pytest failed before executing the test, but no new ruff/mypy diagnostics were introduced".into(),
            structured_payload: Some(serde_json::json!({
                "signal": "pytest_import_or_collect_error",
                "pytest_stdout": inputs.pytest_stdout,
                "pytest_stderr": inputs.pytest_stderr,
                "pytest_report_missing": inputs.pytest_report_missing,
            })),
        };
    }

    info!("Diagnostician → Unknown");
    DiagnosticResult {
        failure_class: Some(FailureClass::Unknown),
        reason: "Could not classify failure with available signals".into(),
        structured_payload: None,
    }
}

// ============================================================================
// diagnose_failure — classify ONE failing test WITH the Spec §7 Phase-2
// remediation workflow (reset-and-rerun for Infrastructure, simplify for Unknown)
// ============================================================================
//
// [VANTA-WHY] `diagnose` above is the pure classifier. This is the workflow that
// ACTS on the class, which the M3 loop had dropped:
//   Infrastructure → reset the sandbox and rerun once. A single transient fault
//     is discovered and cleared here rather than being recorded against the model.
//   Unknown        → rerun in a fresh sandbox, then ask the Attacker to simplify
//     the test, then rerun the simplified one. Only if it is STILL Unknown is the
//     Attacker-generated test QUARANTINED (excluded from EIR/ECR, reason logged)
//     rather than silently counted as a model error.
//   Static / Logic → no remediation here (M3 feeds them back to the Creator).
//
// `bate_validated` says whether BATE holds a baseline for this test (it ran it
// on the original tree and accepted it). It gates the patch-insensitivity probe;
// see the probe's comment for why a validated test must never be demoted by it.
pub async fn diagnose_failure(
    test: &GeneratedTest,
    execution: &TestExecutionDetails,
    ruff_new_errors: bool,
    mypy_new_errors: bool,
    ruff_stderr: &str,
    mypy_stderr: &str,
    network_down: bool,
    patched_source: &str,
    cfg: &LlmConfig,
    original_source: &str,
    test_cmd: &TestCommand,
    bate_validated: bool,
) -> DiagnosticResult {
    let force_unknown = std::env::var("VANTA_TEST_FORCE_UNKNOWN").ok().as_deref() == Some("1");
    // [VANTA-WHY] The FIRST classification sees the real execution: whether the
    // run timed out, whether pytest produced report.json, and the actual streams.
    // Throwing those away (as an earlier cut did) downgrades a timeout — a
    // genuine Infrastructure failure — to an unclassifiable Unknown, and an
    // Unknown triggers an Attacker call it never needed.
    let classify = |exec: &TestExecutionDetails| -> DiagnosticResult {
        if force_unknown {
            return DiagnosticResult {
                failure_class: Some(FailureClass::Unknown),
                reason: "Forced Unknown classification for regression testing".into(),
                structured_payload: Some(json!({
                    "signal": "forced_unknown",
                })),
            };
        }
        diagnose(&DiagnosticInputs {
            import_or_syntax_error: exec.result == TestResult::Error,
            ruff_new_errors,
            mypy_new_errors,
            timed_out: exec.timed_out,
            assertion_failed: exec.result == TestResult::Failed,
            network_down,
            ruff_stderr: ruff_stderr.to_string(),
            mypy_stderr: mypy_stderr.to_string(),
            pytest_stdout: exec.stdout.clone(),
            pytest_stderr: exec.stderr.clone(),
            pytest_report_missing: !exec.report_path_exists,
            test_source: test.test_source.clone(),
        })
    };

    let enrich_with_sandbox_roots =
        |mut diag: DiagnosticResult, initial_root: Option<String>, rerun_root: Option<String>| {
            let mut payload = diag.structured_payload.take().unwrap_or_else(|| json!({}));
            if let Some(obj) = payload.as_object_mut() {
                if let Some(root) = initial_root.clone() {
                    obj.insert(
                        "initial_sandbox_root".into(),
                        serde_json::Value::String(root),
                    );
                }
                if let Some(root) = rerun_root.clone() {
                    obj.insert("rerun_sandbox_root".into(), serde_json::Value::String(root));
                }
                obj.insert(
                    "fresh_sandbox_spawned".into(),
                    serde_json::Value::Bool(
                        initial_root.is_some()
                            && rerun_root.is_some()
                            && initial_root != rerun_root,
                    ),
                );
            }
            diag.structured_payload = Some(payload);
            diag
        };

    let diag = classify(execution);

    // [VANTA-WHY] Locate the fault BEFORE blaming the patch. A `Logic` verdict
    // asserts "the patch is wrong", so the failure has to be attributable to the
    // patch: an assertion the test made, or an exception raised in the code
    // under test. Only a crash inside the generated test's own code is a defect
    // of the test — a `SystemExit` from a test that built a CLI-reading settings
    // object, or a fake that lacks the attribute the test uses next. This runs
    // before the (expensive) patch-insensitivity probe below because it needs no
    // rerun.
    if diag.failure_class == Some(FailureClass::Logic) {
        if let Some((reason, payload)) = locate_fault(execution, &test.test_source) {
            tracing::info!(
                test = %test.test_name,
                "fault located in the TEST, not the patch → TestDefect"
            );
            return enrich_with_sandbox_roots(
                DiagnosticResult {
                    failure_class: Some(FailureClass::TestDefect),
                    reason,
                    structured_payload: Some(payload),
                },
                execution.sandbox_root.clone(),
                None,
            );
        }
    }

    // [VANTA-WHY] The patch-insensitivity probe (b7325c9) may NOT demote a test
    // that BATE has a baseline for, which is why it is gated on `bate_validated`.
    // BATE has already run the test on the ORIGINAL tree and accepted it, so
    // "fails identically on both trees" is not new information about the test:
    //   - BugReproduction: BATE accepted it because it FAILS on the original, so
    //     failing identically on the patched tree is exactly the evidence that
    //     the patch did not fix the bug → Logic, which rejects. A no-op patch is
    //     byte-identical to the original, so it would otherwise intercept EVERY
    //     no-op by construction (PySnooper/3, 5cc225fc).
    //   - FeatureAcceptance / BehaviorPreservation: BATE proved they PASS on the
    //     original, so a failure on the patched tree is patch-caused by
    //     construction and the probe has no job there either.
    // It is kept for a failure that carries no BATE baseline — a test the
    // Diagnostician is asked about without one, e.g. a native pre-existing
    // failure. No such failure reaches here today: every test in `tests_to_run`
    // is `shared ∪ valid`, i.e. BATE-validated at generation and replayed
    // thereafter. The gate is what makes that boundary explicit and testable.
    // A test's OWN assertion is never a test defect (§B3): the test checked
    // behaviour and the behaviour was wrong.
    if !bate_validated && diag.failure_class == Some(FailureClass::Logic) {
        let output = format!("{}\n{}", execution.stdout, execution.stderr);
        let (frame_file, _, exception_type) = parse_failure_site(&output);
        let exception = exception_type.clone().unwrap_or_default();
        let tests_own_assertion = exception == "AssertionError"
            && assertion_is_the_tests_own(&output, frame_file.as_deref());
        if !exception.is_empty() && !tests_own_assertion {
            if let Some(patched_sig) = failure_signature(execution) {
                let original = harness::run_single_test_with_timeout(
                    original_source,
                    &test.test_source,
                    test_cmd,
                    harness::patched_test_timeout_secs(),
                )
                .await;
                if is_patch_insensitive(execution, &original) {
                    let (exception, file, line) = &patched_sig;
                    let site = format!(
                        "{}:{}",
                        file.clone().unwrap_or_else(|| "?".into()),
                        line.map(|l| l.to_string()).unwrap_or_else(|| "?".into()),
                    );
                    tracing::info!(
                        test = %test.test_name,
                        "failure reproduces on the ORIGINAL tree → TestDefect (patch-insensitive)"
                    );
                    return enrich_with_sandbox_roots(
                        DiagnosticResult {
                            failure_class: Some(FailureClass::TestDefect),
                            reason: format!(
                                "Test fails identically on the unpatched tree ({exception} at \
                                 {site}) — patch-insensitive, so the failure is a property of the \
                                 test, not the patch. Sent back to the Attacker for revision."
                            ),
                            structured_payload: Some(json!({
                                "signal": "patch_insensitive_failure",
                                "exception_type": exception,
                                "frame_file": file,
                                "frame_line": line,
                                "original_sandbox_root": original.sandbox_root.clone(),
                            })),
                        },
                        execution.sandbox_root.clone(),
                        original.sandbox_root.clone(),
                    );
                }
            }
        }
    }

    match diag.failure_class {
        Some(FailureClass::Infrastructure) => {
            // Spec §7 Phase 2: "Infrastructure: reset backend and rerun."
            tracing::info!(test = %test.test_name, "Infrastructure → resetting sandbox and rerunning");
            let rerun = harness::run_single_test_with_timeout(
                patched_source,
                &test.test_source,
                test_cmd,
                harness::patched_test_timeout_secs(),
            )
            .await;
            match rerun.result {
                TestResult::Passed => {
                    // Flaky infra — the test actually passes on rerun.
                    enrich_with_sandbox_roots(
                        DiagnosticResult {
                            failure_class: None,
                            reason: "Infrastructure failure resolved on rerun".into(),
                            structured_payload: None,
                        },
                        execution.sandbox_root.clone(),
                        rerun.sandbox_root.clone(),
                    )
                }
                _ => {
                    let mut re = classify(&rerun);
                    re.reason = format!("{} (after sandbox reset + rerun)", re.reason);
                    enrich_with_sandbox_roots(
                        re,
                        execution.sandbox_root.clone(),
                        rerun.sandbox_root.clone(),
                    )
                }
            }
        }
        Some(FailureClass::Unknown) => {
            // Spec §7 Phase 2 Unknown handling.
            tracing::info!(test = %test.test_name, "Unknown → rerunning in fresh sandbox");

            // Step 1: fresh-sandbox rerun.
            let rerun = harness::run_single_test_with_timeout(
                patched_source,
                &test.test_source,
                test_cmd,
                harness::patched_test_timeout_secs(),
            )
            .await;
            if rerun.result == TestResult::Passed {
                return enrich_with_sandbox_roots(
                    DiagnosticResult {
                        failure_class: None,
                        reason: "Unknown resolved on fresh-sandbox rerun".into(),
                        structured_payload: None,
                    },
                    execution.sandbox_root.clone(),
                    rerun.sandbox_root.clone(),
                );
            }
            let rerun_diag = classify(&rerun);
            if !matches!(rerun_diag.failure_class, Some(FailureClass::Unknown)) {
                return enrich_with_sandbox_roots(
                    rerun_diag,
                    execution.sandbox_root.clone(),
                    rerun.sandbox_root.clone(),
                );
            }

            // Step 2: ask the Attacker to simplify the test.
            // `scrap_note` records WHY a passing simplification was refused, so
            // the quarantine below names the reason instead of reporting a
            // generic persistent Unknown.
            let mut scrap_note: Option<String> = None;
            tracing::info!(test = %test.test_name, "Unknown → asking Attacker to simplify");
            match attacker_simplify_test(cfg, original_source, test).await {
                Ok(simplified) => {
                    // Step 3: run the simplified test.
                    let simp_result = harness::run_single_test_with_timeout(
                        patched_source,
                        &simplified.test_source,
                        test_cmd,
                        harness::patched_test_timeout_secs(),
                    )
                    .await;
                    // [VANTA-WHY] A10 — a passing simplified test does NOT by
                    // itself retire the Unknown. Simplification is free to delete
                    // the assertion that was failing: `assert add(2, 3) == 5`
                    // becomes `add(2, 3)`, which passes and still mentions the
                    // same inputs, so the run used to record "Unknown resolved"
                    // over a probe that tests nothing. The pass is believed only
                    // when BOTH hold:
                    //   (a) the simplified body still asserts the SAME input and
                    //       expected value as the test it replaces, and
                    //   (b) BATE finds that test's baseline on the ORIGINAL tree
                    //       consistent with its declared intent — a reproduction
                    //       probe that no longer fails on the original tree
                    //       reproduced nothing.
                    // Otherwise the simplified test is SCRAPPED and the failure
                    // stands: it is the finding, not a fix.
                    if simp_result.result == TestResult::Passed {
                        if !simplification_keeps_the_asserted_contract(
                            &test.test_source,
                            &simplified.test_source,
                        ) {
                            tracing::info!(
                                test = %simplified.test_name,
                                "simplified test passes but no longer asserts the original's input \
                                 and expected value → SCRAPPED, Unknown stands"
                            );
                            scrap_note = Some(
                                "the simplified test dropped the assertion it should have kept \
                                 (the same input and expected value must still be asserted)"
                                    .to_string(),
                            );
                        } else {
                            let original_run = harness::run_single_test_with_timeout(
                                original_source,
                                &simplified.test_source,
                                test_cmd,
                                harness::patched_test_timeout_secs(),
                            )
                            .await;
                            match validate_test_against_baseline(
                                &simplified.test_name,
                                &simplified.intent,
                                original_run.result.clone(),
                            ) {
                                BateOutcome::Valid => {
                                    return enrich_with_sandbox_roots(
                                        DiagnosticResult {
                                            failure_class: None,
                                            reason: "Unknown resolved after Attacker simplification (BATE re-validated the simplified test against the original tree)".into(),
                                            structured_payload: Some(json!({
                                                "signal": "simplification_revalidated",
                                                "simplified_test": simplified.test_name,
                                                "original_sandbox_root": original_run.sandbox_root.clone(),
                                            })),
                                        },
                                        execution.sandbox_root.clone(),
                                        simp_result.sandbox_root.clone(),
                                    );
                                }
                                BateOutcome::Mismatch(reason) => {
                                    tracing::info!(
                                        test = %simplified.test_name,
                                        "simplified test passes on the patched tree but its ORIGINAL \
                                         baseline contradicts its intent → SCRAPPED, Unknown stands: {reason}"
                                    );
                                    scrap_note = Some(
                                        "the simplified test passes on the patched tree but its \
                                         baseline on the ORIGINAL tree contradicts its declared \
                                         intent, so it no longer asserts what the issue requires"
                                            .to_string(),
                                    );
                                }
                            }
                        }
                    }
                    let simp_diag = classify(&simp_result);
                    if !matches!(simp_diag.failure_class, Some(FailureClass::Unknown)) {
                        return enrich_with_sandbox_roots(
                            simp_diag,
                            execution.sandbox_root.clone(),
                            simp_result.sandbox_root.clone(),
                        );
                    }
                }
                Err(e) => {
                    warn!("attacker_simplify_test failed: {e}");
                }
            }

            // Step 4: still Unknown → quarantine (Attacker-generated tests).
            // Repository/benchmark tests would escalate to human review instead.
            // [VANTA-WHY] A10 — a simplification whose PASS was refused leaves the
            // Unknown exactly as it was; the scrap reason is named so the record
            // shows the failure was not erased by a test that stopped asserting.
            DiagnosticResult {
                failure_class: Some(FailureClass::Unknown),
                reason: match scrap_note {
                    Some(note) => format!(
                        "Unknown persisted after simplification: {note} → QUARANTINED \
                         (Attacker-generated test). For repo/benchmark tests this would escalate to \
                         human review."
                    ),
                    None => "Unknown persisted after rerun + simplification → QUARANTINED (Attacker-generated test). For repo/benchmark tests this would escalate to human review.".into(),
                },
                structured_payload: Some(serde_json::json!({
                    "action": "quarantine",
                    "evidence_source": "AttackerGenerated",
                    "initial_sandbox_root": execution.sandbox_root,
                    "note": "ExistingRepositoryTest and BenchmarkOracle tests escalate to human review instead of quarantine",
                })),
            }
        }
        // Static and Logic: no remediation in this phase (M3 feeds back to Creator).
        _ => diag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_is_infrastructure() {
        let inputs = DiagnosticInputs {
            timed_out: true,
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Infrastructure));
    }

    #[test]
    fn import_or_syntax_error_without_new_static_signals_is_infrastructure() {
        let inputs = DiagnosticInputs {
            import_or_syntax_error: true,
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Infrastructure));
    }

    #[test]
    fn ruff_new_errors_is_static() {
        let inputs = DiagnosticInputs {
            ruff_new_errors: true,
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Static));
    }

    #[test]
    fn assertion_failed_is_logic() {
        let inputs = DiagnosticInputs {
            assertion_failed: true,
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Logic));
    }

    /// Item 5 of the heretic fix list: a test that RAN and failed is classified
    /// by its failure, not by a static finding the patch also carries. With the
    /// static check first, e8d393be's real failures were labelled Static and the
    /// failure evidence was lost.
    #[test]
    fn a_test_that_ran_and_failed_is_logic_even_with_new_static_errors() {
        let inputs = DiagnosticInputs {
            assertion_failed: true,
            ruff_new_errors: true,
            mypy_new_errors: true,
            ruff_stderr: "src/pkg/mod.py:7:1: F821 Undefined name `x`".to_string(),
            pytest_stdout: "test_x.py:2: AssertionError\n".to_string(),
            ..Default::default()
        };
        let result = diagnose(&inputs);

        assert_eq!(
            result.failure_class,
            Some(FailureClass::Logic),
            "the failure that actually happened outranks a lint finding"
        );
        let payload = result.structured_payload.expect("payload");
        assert_eq!(payload["signal"], json!("assertion_failed"));
        assert_eq!(
            payload["new_static_errors"]["ruff_new_errors"],
            json!(true),
            "the static finding must still be RECORDED, just not as the class"
        );
        assert!(
            payload["new_static_errors"]["ruff_stderr"]
                .as_str()
                .unwrap()
                .contains("F821"),
            "the static evidence must ride with the payload"
        );
    }

    /// ...and Static still classifies a tree where the test never ran.
    #[test]
    fn new_static_errors_without_a_ran_test_are_static() {
        let inputs = DiagnosticInputs {
            mypy_new_errors: true,
            mypy_stderr: "src/pkg/mod.py:7: error: incompatible type".to_string(),
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Static));
    }

    /// Item 1 of the heretic fix list: the assertion payload used to be a bare
    /// `{"signal":"assertion_failed"}` — no test source, no output, no site. The
    /// evidence must now let a reader audit WHY the patch was blamed.
    #[test]
    fn an_assertion_failure_saves_the_test_source_and_the_failure_site() {
        let inputs = DiagnosticInputs {
            assertion_failed: true,
            test_source: "def test_x():\n    assert f() == 5\n".to_string(),
            pytest_stdout: concat!(
                "============================= test session starts ==============================\n",
                "collected 1 item\n\n",
                "test_x.py::test_x\n",
                "\n",
                "    def test_x():\n",
                ">       assert f() == 5\n",
                "E       assert 6 == 5\n",
                "\n",
                "test_x.py:2: AssertionError\n",
            )
            .to_string(),
            ..Default::default()
        };

        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Logic));
        let payload = result
            .structured_payload
            .expect("the failing test's evidence must be saved, not just its class");
        assert_eq!(payload["signal"], json!("assertion_failed"));
        assert_eq!(payload["exception_type"], json!("AssertionError"));
        assert_eq!(payload["frame_file"], json!("test_x.py"));
        assert_eq!(payload["frame_line"], json!(2));
        assert!(
            payload["test_source"]
                .as_str()
                .unwrap()
                .contains("def test_x"),
            "the failing test's own source must ride with the evidence"
        );
        assert!(
            payload["failure_output"]
                .as_str()
                .unwrap()
                .contains("assert 6 == 5"),
            "pytest's failure output must ride with the evidence"
        );
    }

    #[test]
    fn a_huge_failure_output_is_truncated_with_a_marker() {
        let big = "x".repeat(FAILURE_OUTPUT_LIMIT * 3);
        let out = truncate_output(&big, FAILURE_OUTPUT_LIMIT);
        assert!(out.len() < big.len());
        assert!(out.contains("truncated"), "a cut must be visible: {out}");
    }

    #[test]
    fn the_innermost_frame_of_an_error_in_the_test_object_is_located() {
        // The 56eb4fd3 shape: the fake the TEST defined lacks an attribute.
        let output = "test_x.py:9: AttributeError: 'FakeTensor' object has no attribute 'shape'\n";
        let (file, line, exc) = parse_failure_site(output);
        assert_eq!(file.as_deref(), Some("test_x.py"));
        assert_eq!(line, Some(9));
        assert_eq!(exc.as_deref(), Some("AttributeError"));
    }

    fn exec_at(result: TestResult, output: &str) -> TestExecutionDetails {
        TestExecutionDetails {
            result,
            stdout: output.to_string(),
            stderr: String::new(),
            exit_code: Some(1),
            timed_out: false,
            report_path_exists: true,
            sandbox_root: Some("sandbox".to_string()),
        }
    }

    /// Item 2 of the heretic fix list: a failure that reproduces identically on
    /// the unpatched tree says nothing about the patch (B v2's HereticSettings
    /// raised SystemExit on both trees). Same exception at the same site →
    /// patch-insensitive; anything else → not.
    #[test]
    fn a_failure_that_reproduces_on_the_original_tree_is_patch_insensitive() {
        let patched = exec_at(TestResult::Failed, "test_x.py:9: SystemExit\n");
        let identical = patched.clone();
        assert!(
            is_patch_insensitive(&patched, &identical),
            "same exception at the same site on both trees is the test's own defect"
        );

        let other_site = exec_at(TestResult::Failed, "test_x.py:42: SystemExit\n");
        assert!(
            !is_patch_insensitive(&patched, &other_site),
            "same exception at a DIFFERENT site is not the same failure"
        );

        let other_exception = exec_at(TestResult::Failed, "test_x.py:9: AssertionError\n");
        assert!(
            !is_patch_insensitive(&patched, &other_exception),
            "a different exception type is a different failure"
        );

        let original_passed = exec_at(TestResult::Passed, "");
        assert!(
            !is_patch_insensitive(&patched, &original_passed),
            "a test that passes on the original is evidence about the patch"
        );
    }

    /// A real py3.8 pytest run (the wave-1 PySnooper/3 shape): the FAILURES
    /// section ends with the `NameError` the test hit, and `future`'s
    /// `DeprecationWarning` — which the old bottom-up scan mistook for the
    /// failure site — follows in the warnings summary.
    const PYSNOOPER_NOOP_OUTPUT: &str = concat!(
        "============================= test session starts ==============================\n",
        "platform linux -- Python 3.8.20, pytest-8.0.2, pluggy-1.5.0\n",
        "rootdir: /tmp/sandbox\n",
        "plugins: json-report-1.5.0, metadata-3.1.1\n",
        "collected 1 item\n\n",
        "test_run.py::test_snoop_file_output_contains_starting_var                   [100%]\n\n",
        "=================================== FAILURES ===================================\n",
        "_____________ test_snoop_file_output_contains_starting_var _____________\n\n",
        "    def test_snoop_file_output_contains_starting_var():\n",
        "        with pysnooper.snoop(trace_file):\n",
        ">           assert 'Starting var' in trace_file.read()\n",
        "E           NameError: name 'output_path' is not defined\n\n",
        "pysnooper/pysnooper.py:26: NameError\n",
        "=========================== short test summary info ============================\n",
        "FAILED test_run.py::test_snoop_file_output_contains_starting_var - NameError\n",
        "=============================== warnings summary ===============================\n",
        "/opt/venv38/lib/python3.8/site-packages/future/standard_library/__init__.py:65\n",
        "  /opt/venv38/lib/python3.8/site-packages/future/standard_library/__init__.py:65: DeprecationWarning: the imp module is deprecated\n",
        "    __import__('future_builtins')\n\n",
        "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n",
        "============================== 1 failed in 0.31s ==============================\n",
    );

    /// The failure site is the frame of the exception that was actually raised —
    /// never the warning pytest prints after it (PySnooper/3, 3ef3ac33).
    #[test]
    fn the_failure_site_ignores_a_later_warning_summary() {
        let (file, line, exc) = parse_failure_site(PYSNOOPER_NOOP_OUTPUT);
        assert_eq!(file.as_deref(), Some("pysnooper/pysnooper.py"));
        assert_eq!(line, Some(26));
        assert_eq!(exc.as_deref(), Some("NameError"));
    }

    /// A `NameError` raised in the code under test is the patch's fault even
    /// though the no-op patch touched only another file — the PySnooper/3 no-op
    /// appends a newline to `pysnooper/__init__.py` and leaves `pysnooper.py` as
    /// buggy as it was. BATE proved the test's behaviour on the original tree,
    /// so a validated test that still hits the bug is evidence AGAINST the patch.
    #[test]
    fn a_failure_in_the_code_under_test_is_blamed_on_the_patch() {
        let execution = exec_at(TestResult::Failed, PYSNOOPER_NOOP_OUTPUT);
        assert!(
            locate_fault(&execution, "").is_none(),
            "the patched tree still raises the bug's NameError, so the patch must be rejected"
        );
    }

    /// Captured output inside the FAILURES section can mention any file and
    /// line; it is not the failure site.
    #[test]
    fn captured_output_is_not_read_as_a_failure_site() {
        let output = concat!(
            "=================================== FAILURES ===================================\n",
            "___________________________________ test_x ____________________________________\n\n",
            "    def test_x():\n",
            ">       raise RuntimeError('real')\n",
            "E       RuntimeError: real\n\n",
            "pkg/mod.py:12: RuntimeError\n",
            "----------------------------- Captured stdout call -----------------------------\n",
            "log line mentioning other/thing.py:99: ValueError: not the failure\n",
            "=========================== short test summary info ============================\n",
            "FAILED test_x.py::test_x - RuntimeError: real\n",
        );
        let (file, line, exc) = parse_failure_site(output);
        assert_eq!(file.as_deref(), Some("pkg/mod.py"));
        assert_eq!(line, Some(12));
        assert_eq!(exc.as_deref(), Some("RuntimeError"));
    }

    /// An exception raised by the generated test's own body is a test defect:
    /// the code under test never ran.
    #[test]
    fn a_failure_in_the_generated_tests_own_body_is_a_test_defect() {
        let output = concat!(
            "=================================== FAILURES ===================================\n",
            "___________________________________ test_run ___________________________________\n\n",
            "    def test_x():\n",
            ">       raise RuntimeError('the test itself is broken')\n",
            "E       RuntimeError: the test itself is broken\n\n",
            "test_run.py:5: RuntimeError\n",
        );
        let located = locate_fault(&exec_at(TestResult::Failed, output), "")
            .expect("a crash in the test's own file says nothing about the patch");
        assert_eq!(
            located.1["signal"],
            json!("test_defect_exception_in_test_file")
        );
        assert_eq!(located.1["frame_file"], json!("test_run.py"));
        assert_eq!(located.1["exception_type"], json!("RuntimeError"));
        assert!(located.0.contains("Attacker"), "the reason must say it goes back: {}", located.0);
    }

    /// The 56eb4fd3 shape, when the crash lands in the test's own file: the fake
    /// the TEST defined lacks an attribute, so the test is at fault.
    #[test]
    fn a_failure_inside_the_tests_own_file_is_a_test_defect() {
        let execution = exec_at(
            TestResult::Failed,
            "test_run.py:9: AttributeError: 'FakeTensor' object has no attribute 'shape'\n",
        );
        let located = locate_fault(&execution, "")
            .expect("a fault in the test's own file must not be blamed on the patch");
        assert_eq!(
            located.1["signal"],
            json!("test_defect_exception_in_test_file")
        );
        assert_eq!(located.1["exception_type"], json!("AttributeError"));
        assert!(located.0.contains("Attacker"), "the reason must say it goes back: {}", located.0);
    }

    /// The same 56eb4fd3 shape when the crash lands in the code under test: the
    /// frame file is the patch's, but the object it choked on is the fake the
    /// TEST declared — the test stubbed its dependency incompletely, so the
    /// failure is the test's. This is the signal the e432d2c rewrite dropped.
    #[test]
    fn a_crash_on_a_test_defined_type_is_a_test_defect() {
        let execution = exec_at(
            TestResult::Failed,
            "src/heretic/modifiers/abliteration.py:88: AttributeError: 'FakeTensor' object has no attribute 'shape'\n",
        );
        let test_source = "class FakeTensor:\n    def __init__(self, ids):\n        self.ids = ids\n";
        let located = locate_fault(&execution, test_source)
            .expect("the test's own fake lacks the attribute, so the test is at fault");
        assert_eq!(located.1["signal"], json!("test_defect_test_defined_type"));
        assert_eq!(located.1["exception_type"], json!("AttributeError"));
        assert_eq!(located.1["frame_file"], json!("src/heretic/modifiers/abliteration.py"));
    }

    /// ...and a crash in the patch's own file on an object the test did NOT
    /// define is the patch's fault.
    #[test]
    fn a_crash_in_the_patchs_own_file_is_the_patchs_fault() {
        let execution = exec_at(
            TestResult::Failed,
            "src/heretic/modifiers/abliteration.py:88: AttributeError: 'FakeTensor' object has no attribute 'shape'\n",
        );
        assert!(
            locate_fault(&execution, "def test_x():\n    pass\n").is_none(),
            "the test declared no such type, so the exception was raised by the patch's own file"
        );
    }

    /// An exception raised in the code under test on real input is the patch's
    /// fault.
    #[test]
    fn an_exception_in_the_code_under_test_is_blamed_on_the_patch() {
        let execution = exec_at(
            TestResult::Failed,
            "src/heretic/modifiers/abliteration.py:88: TypeError: unsupported operand type(s)\n",
        );
        assert!(
            locate_fault(&execution, "").is_none(),
            "the patch's own file throwing on real input is the patch's fault"
        );
    }

    /// A `SystemExit` from a test that instantiated a CLI-reading settings
    /// object (B v2's `HereticSettings`) is the test's setup, not the patch.
    #[test]
    fn a_cli_system_exit_in_test_setup_is_a_test_defect() {
        let execution = exec_at(TestResult::Failed, "test_run.py:5: SystemExit: 2\n");
        let located = locate_fault(&execution, "")
            .expect("a CLI/argparse SystemExit is a test defect");
        assert_eq!(located.1["exception_type"], json!("SystemExit"));
    }

    /// The test's own `assert` failing IS a verdict on the patch.
    #[test]
    fn the_tests_own_assertion_is_blamed_on_the_patch() {
        let execution = exec_at(TestResult::Failed, "test_x.py:2: AssertionError\n");
        assert!(
            locate_fault(&execution, "").is_none(),
            "an assertion the test made is a fair verdict on the patch"
        );
    }

    /// A mock `assert_*` failure is likewise the test's own check.
    #[test]
    fn a_mock_assertion_failure_is_blamed_on_the_patch() {
        let execution = exec_at(
            TestResult::Failed,
            "test_x.py:6: AssertionError: Expected 'get_responses' to have been called once. Called 0 times.\n",
        );
        assert!(
            locate_fault(&execution, "").is_none(),
            "a mock assertion failure is a fair verdict on the patch"
        );
    }

    /// An unreadable failure (no traceback at all) must not be quietly turned
    /// into a test defect.
    #[test]
    fn an_unlocatable_failure_is_not_reclassified() {
        let execution = exec_at(TestResult::Failed, "no frames here\n");
        assert!(
            locate_fault(&execution, "").is_none(),
            "with no fault located, the patch keeps its Logic class"
        );
    }

    fn generated_test(test_source: &str) -> GeneratedTest {
        GeneratedTest {
            test_name: "test_snoop_file_output_contains_starting_var".to_string(),
            test_source: test_source.to_string(),
            intent: crate::core::TestIntent::BugReproduction,
            reason: "reproduce the reported NameError".to_string(),
            issue_quote: None,
            relabeled: false,
        }
    }

    /// The BATE gate on the patch-insensitivity probe. BATE accepted this
    /// BugReproduction test because it FAILS on the original tree, so the
    /// Diagnostician must not demote the identical failure on the no-op tree:
    /// the no-op is byte-identical to the original, so the probe would otherwise
    /// intercept every no-op by construction and the bug would be approved
    /// (PySnooper/3, 5cc225fc). If the gate is removed, this test fails.
    #[tokio::test]
    async fn a_bate_validated_no_op_failure_stays_logic() {
        let test = generated_test("def test_snoop_file_output_contains_starting_var():\n    pass\n");
        let execution = exec_at(TestResult::Failed, PYSNOOPER_NOOP_OUTPUT);
        let diag = diagnose_failure(
            &test,
            &execution,
            false,
            false,
            "",
            "",
            false,
            "",
            &LlmConfig::default(),
            "",
            &TestCommand::default(),
            true,
        )
        .await;

        assert_eq!(
            diag.failure_class,
            Some(FailureClass::Logic),
            "a no-op patch must be rejected, not excused: {:?}",
            diag.reason
        );
        assert_ne!(diag.failure_class, Some(FailureClass::TestDefect));
    }

    /// ...and the dev fix's own bogus test (3ef3ac33 round 4 asserted
    /// `'Starting var'`, which `tracer.py:170` never emits for a zero-argument
    /// function) is still the test's assertion, so it stays Logic too. That
    /// false reject is recorded, not tuned away (plans/bugsinpy.md §13).
    #[tokio::test]
    async fn a_bate_validated_tests_own_assertion_stays_logic() {
        let test = generated_test(
            "def test_snoop_file_output_contains_starting_var():\n    assert 'Starting var' in content\n",
        );
        let output = concat!(
            "=================================== FAILURES ===================================\n",
            "_____________ test_snoop_file_output_contains_starting_var _____________\n\n",
            "    def test_snoop_file_output_contains_starting_var():\n",
            ">       assert 'Starting var' in content\n",
            "E       AssertionError: assert 'Starting var' in ''\n\n",
            "test_run.py:17: AssertionError\n",
            "=========================== short test summary info ============================\n",
            "FAILED test_run.py::test_snoop_file_output_contains_starting_var - AssertionError\n",
        );
        let execution = exec_at(TestResult::Failed, output);
        let diag = diagnose_failure(
            &test,
            &execution,
            false,
            false,
            "",
            "",
            false,
            "",
            &LlmConfig::default(),
            "",
            &TestCommand::default(),
            true,
        )
        .await;

        assert_eq!(
            diag.failure_class,
            Some(FailureClass::Logic),
            "the test's own assertion is a verdict on the patch: {:?}",
            diag.reason
        );
    }

    #[test]
    fn path_touched_matches_patch_headers_and_repo_relative_reports() {
        let touched = vec!["src/pkg/mod.py".to_string()];
        assert!(harness::path_touched("src/pkg/mod.py", &touched));
        assert!(harness::path_touched("a/src/pkg/mod.py", &touched));
        assert!(harness::path_touched("./src/pkg/mod.py", &touched));
        assert!(!harness::path_touched("src/other/mod.py", &touched));
    }

    #[test]
    fn no_signals_is_unknown() {
        let inputs = DiagnosticInputs::default();
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Unknown));
    }

    #[test]
    fn network_down_is_infrastructure() {
        // M2 EXIT TEST: "Kill the network mid-run and it classifies as
        // Infrastructure, not a model error."
        let inputs = DiagnosticInputs {
            network_down: true,
            assertion_failed: true, // would otherwise be Logic
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Infrastructure));
    }

    #[test]
    fn import_error_without_new_static_signals_is_not_static() {
        let inputs = DiagnosticInputs {
            import_or_syntax_error: true,
            pytest_stderr: "ModuleNotFoundError: No module named 'calc'".into(),
            ..Default::default()
        };
        let result = diagnose(&inputs);
        assert_eq!(result.failure_class, Some(FailureClass::Infrastructure));
        assert_ne!(result.failure_class, Some(FailureClass::Static));
    }

    #[test]
    fn infrastructure_payload_captures_pytest_stderr() {
        let inputs = DiagnosticInputs {
            import_or_syntax_error: true,
            pytest_stderr: "ImportError while importing test module".into(),
            pytest_stdout: "collected 0 items / 1 error".into(),
            ..Default::default()
        };
        let result = diagnose(&inputs);
        let payload = result
            .structured_payload
            .expect("payload should be present");
        assert_eq!(payload["signal"], "pytest_import_or_collect_error");
        assert_eq!(
            payload["pytest_stderr"],
            "ImportError while importing test module"
        );
        assert_eq!(payload["pytest_stdout"], "collected 0 items / 1 error");
    }
}
