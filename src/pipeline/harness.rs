// ============================================================================
// HARNESS — the sandbox test-running layer (Spec §7 Phase 1/2 execution)
// ============================================================================
//
// [VANTA-WHY] M2 and M3 each grew their own copy of "spawn a sandbox, write the
// repo + tests, run pytest, parse the report". The two copies drifted (different
// report parsing, different static-analysis streams). This module is the single
// definition of HOW Vanta executes tests; the Creator/Attacker/BATE/Governor
// logic is separate and calls in here. Every caller — the BATE baseline, the
// patched run, the Diagnostician's rerun/simplify probes — goes through these
// functions, so a fix to test execution lands once.
// ============================================================================

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::RwLock;

use futures::future::join_all;
use serde::{Deserialize, Serialize};

use crate::agents::attacker::GeneratedTest;
use crate::agents::creator::SourceFile;
use crate::core::TestResult;
use crate::pipeline::candidate::TestCommand;
use crate::pipeline::safe_path::confined_join;
use crate::pipeline::suite_cache;
use crate::sandbox::{Backend, ExecutionResult};

// ============================================================================
// TestExecutionDetails — a single test's outcome plus the raw streams
// ============================================================================
//
// [VANTA-WHY] The Diagnostician needs more than pass/fail: it classifies a
// failure from the timeout flag, whether a report was produced, and the captured
// stdout/stderr. Carrying the streams alongside the result is what lets a
// "pytest never wrote report.json" (Infrastructure) be told apart from a real
// assertion failure (Logic).
#[derive(Debug, Clone, PartialEq)]
pub struct TestExecutionDetails {
    pub result: TestResult,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub report_path_exists: bool,
    pub sandbox_root: Option<String>,
}

impl TestExecutionDetails {
    pub fn infrastructure_error() -> Self {
        Self {
            result: TestResult::Error,
            stdout: String::new(),
            stderr: String::new(),
            exit_code: None,
            timed_out: false,
            report_path_exists: false,
            sandbox_root: None,
        }
    }

    /// Wrap a bare `TestResult` from a rerun (`run_single_test`, which returns
    /// only pass/fail) as details the Diagnostician can classify.
    ///
    /// [VANTA-WHY] A rerun DID produce a verdict, so `report_path_exists` is
    /// true: the "pytest never wrote report.json" Infrastructure signal belongs
    /// to the original run, not to a probe that already returned a result.
    pub fn from_rerun(result: TestResult) -> Self {
        Self {
            result,
            stdout: String::new(),
            stderr: String::new(),
            exit_code: None,
            timed_out: false,
            report_path_exists: true,
            sandbox_root: None,
        }
    }
}

// ============================================================================
// run_tests_parallel_detailed — one sandbox per test, in parallel
// ============================================================================
pub async fn run_tests_parallel_detailed(
    source: &str,
    tests: &[&GeneratedTest],
    test_cmd: &TestCommand,
) -> Result<HashMap<String, TestExecutionDetails>, Box<dyn std::error::Error + Send + Sync>> {
    let mut handles = Vec::new();
    for test in tests {
        let source = source.to_string();
        let test = (*test).clone();
        let cmd_template = test_cmd.clone();
        handles.push(tokio::spawn(async move {
            let name = test.test_name.clone();
            let sb = match Backend::spawn().await {
                Ok(b) => b,
                Err(_) => return (name, TestExecutionDetails::infrastructure_error()),
            };
            if seed_source_and_conftest(sb.root(), &source).is_err() {
                sb.destroy();
                return (name, TestExecutionDetails::infrastructure_error());
            }
            // [VANTA-WHY] F1 — a test file that is never written makes pytest
            // grade NOTHING, which reads as a normal verdict rather than a setup
            // failure. Guard it the same way as the seed.
            if fs::write(sb.root().join("test_run.py"), &test.test_source).is_err() {
                sb.destroy();
                return (name, TestExecutionDetails::infrastructure_error());
            }
            let cmd = build_single_test_cmd(&cmd_template, "test_run.py");
            let cmd_refs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();
            let sandbox_root = sb.root().display().to_string();
            let exec = sb
                .execute(
                    &cmd_refs,
                    Path::new(""),
                    patched_test_timeout_secs(),
                    &cmd_template.env,
                    &cmd_template.precompile,
                )
                .await;
            let result = match exec {
                Ok(e) => parse_pytest_report_detailed(
                    sb.root().join("report.json"),
                    &e,
                    Some(sandbox_root),
                    "test_run.py",
                ),
                Err(_) => TestExecutionDetails::infrastructure_error(),
            };
            sb.destroy();
            (name, result)
        }));
    }

    let mut outcomes = HashMap::new();
    for res in join_all(handles).await {
        if let Ok((name, result)) = res {
            outcomes.insert(name, result);
        }
    }
    Ok(outcomes)
}

/// Run multiple tests in parallel and return only their pass/fail results.
pub async fn run_tests_parallel(
    source: &str,
    tests: &[&GeneratedTest],
    test_cmd: &TestCommand,
) -> Result<HashMap<String, TestResult>, Box<dyn std::error::Error + Send + Sync>> {
    let detailed = run_tests_parallel_detailed(source, tests, test_cmd).await?;
    Ok(detailed
        .into_iter()
        .map(|(name, details)| (name, details.result))
        .collect())
}

// ============================================================================
// run_single_test — run ONE test against a source tree, return its TestResult
// ============================================================================
//
// [VANTA-WHY] Used by the BATE baseline (and its flaky pre-check) and by the
// Diagnostician's rerun/simplify probes. `run_single_test` returns only a
// pass/fail, so a timeout here collapses to `TestResult::Error` — the detailed
// path (which keeps the `timed_out` flag) is `run_tests_parallel_detailed`.
pub async fn run_single_test(
    source_concat: &str,
    test_source: &str,
    test_cmd: &TestCommand,
) -> TestResult {
    run_single_test_with_timeout(source_concat, test_source, test_cmd, test_timeout_secs())
        .await
        .result
}

/// [VANTA-WHY] §14.19 — the one filename a BATE baseline run writes the
/// generated test to. Named rather than spelled inline twice because
/// `bate::harness_error` asks "does the failure output name the generated
/// test?", and that question and the write have to agree on the name: if they
/// drifted, a test whose own collection failed would be classified as a broken
/// environment and never revised.
pub const BATE_TEST_FILE: &str = "test_bate.py";

/// Run one test and keep the execution details for rerun diagnostics.
pub async fn run_single_test_with_timeout(
    source_concat: &str,
    test_source: &str,
    test_cmd: &TestCommand,
    timeout_secs: u64,
) -> TestExecutionDetails {
    let sb = match Backend::spawn().await {
        Ok(b) => b,
        Err(_) => return TestExecutionDetails::infrastructure_error(),
    };
    // [VANTA-WHY] F1 — the parallel path above already treats a failed
    // `seed_and_write` as infrastructure; this path discarded it, so a test could
    // run against the seeded tree instead of `source_concat` and return a
    // normal-looking verdict. Destroy the sandbox and report infrastructure, as
    // the parallel path does. The graded test file's own write is guarded the
    // same way: a test file that is never written makes pytest grade nothing.
    if seed_source_and_conftest(sb.root(), source_concat).is_err() {
        sb.destroy();
        return TestExecutionDetails::infrastructure_error();
    }
    if fs::write(sb.root().join(BATE_TEST_FILE), test_source).is_err() {
        sb.destroy();
        return TestExecutionDetails::infrastructure_error();
    }
    let cmd = build_single_test_cmd(test_cmd, BATE_TEST_FILE);
    let cmd_refs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();
    let sandbox_root = sb.root().display().to_string();
    let exec = sb
        .execute(
            &cmd_refs,
            Path::new(""),
            timeout_secs,
            &test_cmd.env,
            &test_cmd.precompile,
        )
        .await;
    let result = match exec {
        Ok(e) => parse_pytest_report_detailed(
            sb.root().join("report.json"),
            &e,
            Some(sandbox_root),
            BATE_TEST_FILE,
        ),
        Err(_) => TestExecutionDetails::infrastructure_error(),
    };
    sb.destroy();
    result
}

// ============================================================================
// Native suite — the repo's OWN tests, run as a gate
// ============================================================================
//
// [VANTA-WHY] Every other function here runs ONE generated test file. The
// repo's own suite is a different question: it does not depend on what the
// Attacker happened to sample, so a patch that breaks an existing repo test
// fails here even when no generated test covers that input. It is NOT part of
// the shared set and never contributes to EIR/ECR — it is a separate
// `native_suite` section — because it is not a transition measurement.
//
// [VANTA-WHY] Per-test IDs, not one aggregate verdict: "pass on original, fail
// on patch" is a regression; "already failing on original" is not (the patch
// did not break it). A single pass/fail bit cannot express that difference.
//
// [VANTA-WHY] M3 item S1 — the serde derives are what let this whole report be
// stored and served by `suite_cache`: a native run is a pure function of its
// key, so the report that came out of it IS the cached value. Hand-rolling the
// payload would be a second shape that could drift from the struct (a field
// added here and forgotten there would be silently absent from every cached
// report); deriving it means the compiler keeps the two the same.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeSuiteReport {
    /// nodeid → outcome. Empty when the suite could not be run at all.
    pub tests: Vec<(String, TestResult)>,
    /// nodeid → the test's own failure output, for a regression report.
    ///
    /// [VANTA-WHY] A regression the Creator is told about has to carry the
    /// output that shows WHY, exactly as a generated-test failure does. Without
    /// it the repair feedback for a native regression reduces to "a test broke"
    /// — the model cannot act on that. Only failing/erroring tests appear here.
    pub failure_outputs: HashMap<String, String>,
    /// False when no report was produced (missing pytest/plugin, timeout,
    /// crash). A gate that could not run must never be read as "no regressions".
    pub ran: bool,
    /// Did the suite hit its wall-clock cap and get killed?
    ///
    /// [VANTA-WHY] M3 item D1 — a TIMEOUT is not the same emptiness as a
    /// collection error. Both leave `tests` empty and both make the native gate
    /// vacuous, but the code to act on is the opposite: a collection error is the
    /// repo's own import failing on the ORIGINAL tree (a fact about the tree),
    /// while a timeout is the harness's cap being too small for a legitimate
    /// suite (a fact about the harness). Measured on pandas/1: the native suite
    /// takes ~487 s while the in-session cap was a fixed 120 s, so the gate could
    /// never finish for that bug — and reporting it as "collected nothing" sends
    /// the operator to the wrong file. Recorded here so `native_vacuous_reason`
    /// can name it as its own reason (`native_timeout`).
    pub timed_out: bool,
    /// The test command's exit status, when it ran at all.
    pub exit_code: Option<i32>,
    /// The last few non-empty stderr lines, for a run-failure report.
    pub stderr_tail: String,
    /// The report's own account of its COLLECTION failures, or `None` when the
    /// document carries no `collectors` array at all.
    ///
    /// [VANTA-WHY] M3 item W follow-up. `empty_report_note` rendered
    /// `errors=unset` for pandas/1's in-session run, because pytest's
    /// json-report document has no top-level `errors` key — the thing that
    /// actually failed is named by the `collectors` array, where a module that
    /// cannot be imported appears as `outcome: "failed"` with the
    /// `ImportError while importing test module …` text in `longrepr`. Without
    /// that count and that name the reason says "0 tests" and nothing else, and
    /// the offending module is only findable by re-running the sandbox.
    /// Carried on the report (not only inside `stderr_tail`) so
    /// `native_suite.summary` can record it for BOTH sides of the gate — a
    /// value with spaces in a summary is already the norm (`judge_reason=`).
    pub collection: Option<CollectionFailure>,
}

/// A report's census of its own collectors, and its FIRST collection failure.
///
/// [VANTA-WHY] M3 item W follow-up. pytest's terminal summary says
/// `Interrupted: N errors during collection` on STDOUT, which the harness never
/// reads — it parses only `report.json` and stderr. The same fact is in the
/// document, in the `collectors` array: a module pytest could not import is an
/// entry with `outcome: "failed"` whose `longrepr` is the
/// `ImportError while importing test module …` text. Naming it here is what
/// turns "0 tests" from a dead end into a pointer at the module that broke the
/// session, without a second sandbox boot.
///
/// [VANTA-WHY] M3 item S1 — derived for the same reason as `NativeSuiteReport`:
/// an in-session run's collection census rides the cached report, so the two
/// types have to be storable together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionFailure {
    /// Collectors whose `outcome` is neither `passed` nor `skipped`: a module
    /// that failed to import, or any future outcome this reader does not
    /// recognise. A module-level optional-import skip is `skipped` and is NOT
    /// a failure (pandas/1's 72 MB prepare report lists 2624 collectors, every
    /// one `passed` or `skipped`).
    failed: usize,
    /// The first failed collector's nodeid and its `one_line`-collapsed error.
    first: Option<(String, String)>,
}

impl CollectionFailure {
    /// The census of a report document, or `None` when it has no `collectors`
    /// array (the flat 0.1/0.2 schema, or a document of an unforeseen shape).
    fn of(report: &serde_json::Value) -> Option<Self> {
        let collectors = report.get("collectors")?.as_array()?;
        let mut failed = 0usize;
        let mut first: Option<(String, String)> = None;
        for collector in collectors {
            let outcome = collector
                .get("outcome")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            if outcome == "passed" || outcome == "skipped" {
                continue;
            }
            failed += 1;
            if first.is_none() {
                let nodeid = collector
                    .get("nodeid")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let error = collector
                    .get("longrepr")
                    .and_then(serde_json::Value::as_str)
                    .map(one_line)
                    .unwrap_or_default();
                first = Some((nodeid, error));
            }
        }
        Some(Self { failed, first })
    }

    /// `failed_collectors=N`, plus the first failure's nodeid and error when
    /// there is one.
    fn note(&self) -> String {
        match &self.first {
            Some((nodeid, error)) => format!(
                "failed_collectors={} first_failed_collector={} first_collector_error={}",
                self.failed, nodeid, error
            ),
            None => format!("failed_collectors={}", self.failed),
        }
    }
}

impl NativeSuiteReport {
    /// The suite never started: the backend could not spawn, or the source tree
    /// could not be written into the sandbox.
    fn not_run() -> Self {
        Self {
            tests: Vec::new(),
            failure_outputs: HashMap::new(),
            ran: false,
            timed_out: false,
            exit_code: None,
            stderr_tail: String::new(),
            collection: None,
        }
    }

    /// The suite started but produced no usable report (crash, bad JSON, or a
    /// collection error). Carries the exit code and stderr tail so a caller can
    /// REPORT why the tree could not be measured instead of reading the emptiness
    /// as "the tree has no tests".
    ///
    /// [VANTA-WHY] M3 item D1 — a TIMEOUT is split out into
    /// :func:`timed_out_run` rather than folded in here: it is the one failed run
    /// whose cause is the harness's own cap, so it gets its own reason.
    fn failed_run(exit_code: Option<i32>, stderr_tail: String) -> Self {
        Self {
            tests: Vec::new(),
            failure_outputs: HashMap::new(),
            ran: false,
            timed_out: false,
            exit_code,
            stderr_tail,
            collection: None,
        }
    }

    /// The suite hit its wall-clock cap and was killed, so it measured nothing.
    ///
    /// [VANTA-WHY] M3 item D1. Kept apart from :func:`failed_run` because the
    /// reason to report is different: a redundant "collected nothing" would send
    /// the operator looking at the tree, when the fix is the cap. Measured on
    /// pandas/1: both in-session runs (`shuru exec TIMEOUT after 120s`) died this
    /// way while the same suite ran to completion in ~487 s under prepare's own
    /// 1800 s cap.
    fn timed_out_run(exit_code: Option<i32>, stderr_tail: String) -> Self {
        Self {
            tests: Vec::new(),
            failure_outputs: HashMap::new(),
            ran: false,
            timed_out: true,
            exit_code,
            stderr_tail,
            collection: None,
        }
    }

    /// The recorded failure output for a nodeid, collapsed to one line.
    pub fn failure_output(&self, nodeid: &str) -> Option<&str> {
        self.failure_outputs.get(nodeid).map(String::as_str)
    }

    /// Did this tree produce no usable test set — because the suite never ran,
    /// or because it ran and collected ZERO tests?
    ///
    /// [VANTA-WHY] Fix 2 of the rerun fixes. `lookup_native` returns `None` both
    /// when a tree's suite ran and genuinely lacks a nodeid (item 11's absence,
    /// which discriminates) and when the suite produced nothing at all (a broken
    /// command, which measures nothing). The two are only distinguishable by
    /// asking whether the tree collected anything, so that question is a method.
    pub fn collected_nothing(&self) -> bool {
        !self.ran || self.tests.is_empty()
    }
}

/// The last few non-empty lines of a tool's stderr, for a run-failure note.
///
/// [VANTA-WHY] A failed test command explains itself at the END of stderr —
/// pytest prints the collection error and its traceback last. The whole stream
/// can be tens of KB, so the tail is what makes a `tree_run_failed` note
/// actionable without burying the reader.
pub fn stderr_tail(text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .collect();
    let tail = lines[lines.len().saturating_sub(3)..].join(" | ");
    if tail.chars().count() <= 300 {
        return tail;
    }
    let mut out: String = tail.chars().take(299).collect();
    out.push('…');
    out
}

/// Run the repository's own test suite against a source tree.
///
/// The source tree includes the repo's `tests/`, so pytest discovers them from
/// the sandbox root; no generated test file is written.
///
/// ``timeout_secs`` is the wall-clock cap for the pytest invocation. ``None``
/// keeps the historical behaviour — :func:`test_timeout_secs` (120 s, or
/// ``VANTA_TEST_TIMEOUT_SECS``) — which is every hand-run invocation and every
/// call site that records no measurement.
///
/// [VANTA-WHY] M3 item D1. A repo's own suite is not all fast: measured through
/// prepare, pandas/1's native suite takes ~487 s while ``test_timeout_secs()``
/// capped the in-session gate at 120 s, so that gate could never finish and rows
/// died as a false "collected nothing". The caller that HAS a per-bug measurement
/// (prepare's own native wall time) therefore passes a derived cap here, and every
/// suite that runs in-session for that row uses it.
pub async fn run_native_suite(
    source: &str,
    test_cmd: &TestCommand,
    timeout_secs: Option<u64>,
) -> NativeSuiteReport {
    let sb = match Backend::spawn().await {
        Ok(b) => b,
        Err(_) => return NativeSuiteReport::not_run(),
    };
    // [VANTA-WHY] F1 — the conftest write was discarded here too: a `src/`-layout
    // repo would then fail to import and the gate would report regressions from a
    // tree it could not set up. Same treatment as the seed.
    if seed_source_and_conftest(sb.root(), source).is_err() {
        sb.destroy();
        return NativeSuiteReport::not_run();
    }

    // No test-file argument: pytest collects the repo's own tests from rootdir.
    let mut cmd = test_cmd.as_cmd_vec();
    cmd.push("--json-report".to_string());
    cmd.push("--json-report-file=report.json".to_string());
    cmd.push("-q".to_string());
    let cmd_refs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();
    let exec = sb
        .execute(
            &cmd_refs,
            Path::new(""),
            timeout_secs.unwrap_or_else(test_timeout_secs),
            &test_cmd.env,
            &test_cmd.precompile,
        )
        .await;
    let report = match exec {
        Ok(e) => parse_native_report(&sb.root().join("report.json"), &e),
        Err(_) => NativeSuiteReport::not_run(),
    };
    sb.destroy();
    report
}

/// [`run_native_suite`] behind the content-keyed cache (M3 item S1).
///
/// [VANTA-WHY] The in-session native gate runs the SAME bytes twice per attempt —
/// `native_original` against `native_patched` — and `native_original` is a
/// function of the BUG alone, yet a Phase-3 arm-and-seed grid re-measured it once
/// per arm, per seed, per attempt and per review round. `origin` names who asked
/// (the session, attempt and side), so a served entry can be traced back to the
/// row that produced it, which is what makes the reuse auditable rather than
/// merely fast.
///
/// Four rules, in the order they matter:
///
///   * **A key is the input, so a hit is the same experiment.** The key carries
///     the tree, the command, the env, the checkpoint and the WALL-CLOCK CAP
///     (a run's timeout is a function of the cap, so a cap left out of the key
///     would let one arm's kill be inherited as a fact about the tree), and the
///     sandbox kind that ACTUALLY ran (M3 item U4): `auto` resolves through the
///     shuru boot test, so the same request is two experiments on two machines,
///     and a subprocess suite must never answer a shuru run.
///   * **A bad entry is never used.** `plan_native_suite` already refuses a
///     mismatched file; a payload that THIS build cannot decode is refused here
///     too, and the fresh run overwrites it.
///   * **An unmeasured run is never cached.** A report with `ran: false` or
///     `timed_out: true` measured nothing, so it is neither written here nor
///     served by `plan_native_suite` (M3 item U1) — a cached emptiness would
///     read as "no regressions" forever.
///   * **A dirty build is not cached at all.** See `suite_cache`'s module note:
///     `VANTA_BUILD_COMMIT` cannot separate two development builds of one commit,
///     so reuse is off and the reason lands on the row. An UNIDENTIFIED layer
///     (M3 item U3, no `VANTA_SHURU_CHECKPOINT_ID`) is off for the same shape of
///     reason: a name a re-bake reuses is not an identity, so `plan.write` is
///     false and the row says `off:unknown_checkpoint`.
pub async fn run_native_suite_cached(
    source: &str,
    test_cmd: &TestCommand,
    timeout_secs: Option<u64>,
    backend: &str,
    origin: &str,
) -> (NativeSuiteReport, suite_cache::CacheNote) {
    let cap = timeout_secs.unwrap_or_else(test_timeout_secs);
    let key = suite_cache::native_key(source, test_cmd, cap, backend);
    let plan = suite_cache::plan_native_suite(&key);
    let mut note = plan.note.clone();
    if let Some(payload) = &plan.reuse {
        match serde_json::from_value::<NativeSuiteReport>(payload.clone()) {
            Ok(report) => return (report, note),
            // The entry is intact by `suite_cache`'s rules but this build cannot
            // read it: a corrupt-for-us entry. Run fresh and overwrite it.
            Err(_) => {
                note.state = "unreadable".to_string();
                note.would_hit = true;
            }
        }
    }
    let report = run_native_suite(source, test_cmd, Some(cap)).await;
    // [VANTA-WHY] M3 item U1 — ONLY A MEASUREMENT IS CACHED. `not_run`,
    // `failed_run` and `timed_out_run` all carry `ran: false`, and one of them
    // inside the cache is served forever after as an empty suite, which
    // `native_gate_vacuous` then reads as "the patch broke nothing". The write
    // is therefore gated on the run having actually happened, not merely on the
    // entry's absence. A plain miss writes the entry and a corrupt-for-us one is
    // repaired; a `--no-cache` call replaces the entry it could not use, which is
    // what `DISABLE_ENV` and the flag's own help promise (M3 item U2). A dirty
    // build still writes nothing — `plan.write` stays false for it, because its
    // key is not trusted to describe the binary that ran.
    let measured = report_is_measured(&report);
    if measured && (plan.write || note.state == "unreadable") {
        if let Ok(payload) = serde_json::to_value(&report) {
            let _ = suite_cache::write_entry(&key.hex, &key.material, &payload, origin);
        }
    }
    (report, note)
}

/// Is this report a MEASUREMENT, i.e. may it be written to the suite cache?
///
/// [VANTA-WHY] M3 item U1. Pure over the report so the rule can be tested
/// without a sandbox: the write gate is the only thing standing between a
/// transient failure (no backend, a killed run) and an entry that would answer
/// every future lookup with an empty suite.
fn report_is_measured(report: &NativeSuiteReport) -> bool {
    report.ran && !report.timed_out
}

/// Parse a pytest JSON report into per-test outcomes.
fn parse_native_report(report_path: &Path, exec: &ExecutionResult) -> NativeSuiteReport {
    let tail = stderr_tail(&exec.stderr);
    if exec.timed_out {
        // [VANTA-WHY] M3 item D1 — a killed run is `timed_out_run`, not
        // `failed_run`: its emptiness is the cap's fault, and `native_vacuous_reason`
        // reports it as `native_timeout` with the cap named instead of a bare
        // "collected nothing".
        return NativeSuiteReport::timed_out_run(exec.exit_code, tail);
    }
    let Ok(text) = fs::read_to_string(report_path) else {
        return NativeSuiteReport::failed_run(exec.exit_code, tail);
    };
    let Ok(report) = serde_json::from_str::<serde_json::Value>(&text) else {
        return NativeSuiteReport::failed_run(exec.exit_code, tail);
    };
    let mut tests = Vec::new();
    let mut failure_outputs: HashMap<String, String> = HashMap::new();
    // The shape is the plugin version's business, not this reader's (§3.2), so
    // the entries come from the one helper that knows both schemas. A report of
    // neither shape is a FAILED run: `ran: true` with zero tests is the reading
    // that made a 0.2-pinned bug look like a tree with no tests.
    let Ok(entries) = report_tests(&report) else {
        return NativeSuiteReport::failed_run(exec.exit_code, tail);
    };
    for t in entries {
        let nodeid = t["nodeid"].as_str().unwrap_or_default().to_string();
        let Some(result) = native_outcome(t["outcome"].as_str().unwrap_or_default()) else {
            continue;
        };
        if nodeid.is_empty() {
            continue;
        }
        if result != TestResult::Passed {
            // [VANTA-WHY] pytest attaches the traceback to whichever phase
            // produced it: `call` for an assertion, `setup`/`teardown` for a
            // fixture or import error. Reading all three is what makes a
            // collection-time failure explain itself instead of arriving as
            // a bare nodeid.
            for phase in ["call", "setup", "teardown"] {
                let repr = t[phase]["longrepr"]
                    .as_str()
                    .or_else(|| t[phase]["crash"]["message"].as_str());
                if let Some(repr) = repr {
                    failure_outputs.insert(nodeid.clone(), one_line(repr));
                    break;
                }
            }
        }
        tests.push((nodeid, result));
    }
    // [VANTA-WHY] M3 item W — an EMPTY test list is not a clean suite. pytest
    // writes a structurally valid `--json-report` document for a collection
    // error (exit 2), for a session that collected nothing (exit 5) and for a
    // run that died after collection; in every one of those the `tests` array
    // is empty. Read as `ran: true, tests: []` it makes
    // `native_regressions(original = [], ..)` VACUOUSLY empty, so the native
    // gate reports "0 regressions" and the attempt is approved on no evidence
    // at all. Measured on pandas/1's in-session run (session c9289597:
    // `Native: original ran=true (0 tests), patched ran=true (0 tests), 0
    // regression(s)`) while the SAME tree collected 88643 tests through the
    // prepare path. A recognised shape with zero usable outcomes is therefore a
    // FAILED run, and the report's own diagnostics travel with it so the reason
    // is a fact instead of a guess.
    if tests.is_empty() {
        let mut failed = NativeSuiteReport::failed_run(
            exec.exit_code,
            empty_report_note(&report, &tail),
        );
        // [VANTA-WHY] M3 item W follow-up — the note above is free text rendered
        // into `stderr_tail`; the structured census rides alongside it so
        // `native_suite.summary` can name the offending module for BOTH sides of
        // the gate, not only for the side whose note made it into the reason.
        failed.collection = CollectionFailure::of(&report);
        return failed;
    }
    NativeSuiteReport {
        tests,
        failure_outputs,
        ran: true,
        timed_out: false,
        exit_code: exec.exit_code,
        stderr_tail: tail,
        collection: CollectionFailure::of(&report),
    }
}

/// The report's own account of a zero-test result, appended to the stderr tail.
///
/// [VANTA-WHY] M3 item W — `exitcode`, `summary.collected` and `collectors` are
/// the fields pytest's json-report plugin writes that say WHY nothing was
/// collected, and a reader that turns a report into a regression verdict must
/// not discard them: without them "0 tests" is indistinguishable from "the tree
/// has no tests", which is the exact reading that approved a patch on nothing.
/// Each is rendered `unset` when the document does not carry it, so a report of
/// a shape this reader did not expect cannot look like a clean exit either.
///
/// [VANTA-WHY] M3 item W follow-up — the fourfold note used to end in
/// `errors=…`, and that field is ALWAYS `unset`: `errors` is not in
/// pytest-json-report's top-level schema (the 72 MB prepare report carries
/// `collectors`, `created`, `duration`, `environment`, `exitcode`, `root`,
/// `summary`, `tests`, `warnings` and nothing else), so it could never say
/// anything and printed a reassuring `unset` over a collection that had in fact
/// failed. The real fact is the collector census, so it replaces the dead
/// field: `failed_collectors=N` and, when N>0, the first failing module's
/// nodeid and error. Measured on pandas/1's in-session run, whose reason was
/// `tests=0 exit_code=2 … errors=unset` against a tree that prepare collected
/// 88642 tests from.
fn empty_report_note(report: &serde_json::Value, stderr_tail: &str) -> String {
    let field = |path: &[&str]| -> String {
        let mut value = report;
        for key in path {
            match value.get(*key) {
                Some(inner) => value = inner,
                None => return "unset".to_string(),
            }
        }
        match value {
            serde_json::Value::Array(items) => items.len().to_string(),
            serde_json::Value::Number(number) => number.to_string(),
            _ => "unset".to_string(),
        }
    };
    let census = CollectionFailure::of(report)
        .map(|failure| failure.note())
        .unwrap_or_else(|| "failed_collectors=unset".to_string());
    let note = format!(
        "native report collected 0 tests (report exitcode={} summary.collected={} \
         collectors={} {})",
        field(&["exitcode"]),
        field(&["summary", "collected"]),
        field(&["collectors"]),
        census,
    );
    if stderr_tail.is_empty() {
        note
    } else {
        format!("{stderr_tail} | {note}")
    }
}

/// Collapse a multi-line failure report to its last meaningful line, capped.
///
/// [VANTA-WHY] A pytest `longrepr` is the whole assertion block; the actionable
/// line (the `E   TypeError: ...`) is the last one. The feedback prompt and the
/// console both want that line, not ten lines of file context.
fn one_line(text: &str) -> String {
    let last = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .next_back()
        .unwrap_or("")
        .to_string();
    if last.chars().count() <= 300 {
        return last;
    }
    let mut out: String = last.chars().take(299).collect();
    out.push('…');
    out
}

/// Map one pytest-json-report `outcome` string to a `TestResult`.
///
/// [VANTA-WHY] Item 9 — pytest reports a `unittest.TestCase.subTest` item with
/// the outcome `"subtests passed"`, NOT `"passed"`. This parser used to match
/// only `"passed"/"failed"/"error"` and drop everything else, so every subTest
/// item silently vanished: the heretic suite has three subTest tests, and Vanta
/// counted 4 native tests on master where pytest ran 5, and 12 on the C patch
/// where pytest ran 15. The undercount is not merely cosmetic — a subTest item
/// that PASSED on the original was absent from the original side, and
/// `native_regressions` walks the ORIGINAL side, so a patch that broke such a
/// test could never be reported as a regression. A failing subtest already
/// arrives as `"failed"`; only the passing case needed mapping.
fn native_outcome(outcome: &str) -> Option<TestResult> {
    match outcome {
        "passed" | "subtests passed" => Some(TestResult::Passed),
        // "failed" = assertion failure (a failing subtest included); "error" =
        // exception/setup error.
        "failed" | "error" => Some(TestResult::Failed),
        // A skipped test is neither a pass nor a failure signal, and an
        // unrecognised outcome is not a licence to guess.
        _ => None,
    }
}

/// Nodeids that PASS on the original and did not pass on the patch — the repo's
/// own suite reporting a regression. Tests already failing on the original are
/// excluded: the patch did not break them.
///
/// [VANTA-WHY] This used to iterate the PATCHED side only and look for
/// `patched == Failed && original == Passed`. A patch that introduces a syntax
/// error stops pytest at collection, so the patched side collected ZERO tests —
/// and a filter over an empty list reports zero regressions. The heretic run
/// went 4 tests → 0 tests and printed "0 regression(s)": the scariest possible
/// clean bill of health for a tree that does not import.
///
/// So the walk is over the ORIGINAL side now, and "gone" is a regression too. A
/// test that passed on the original and is MISSING from the patched report
/// (collection error, crashed interpreter, deleted test file) is not evidence
/// that the behaviour survived.
///
/// [VANTA-WHY] "Gone" is also how pass -> SKIPPED / XFAIL / DESELECTED is
/// caught, and that is deliberate rather than incidental. `native_outcome` maps
/// only `passed`/`subtests passed` and `failed`/`error`; a `skipped` (or an
/// unknown future) outcome becomes `None` and is dropped from the patched
/// report entirely, so the test lands in the `None` branch below and is counted
/// as a regression. The same holds for a test pytest never ran at all because a
/// config change deselected it (`-k`, `--deselect`, `--ignore`, a narrowed
/// `python_files`): it is absent, so it is a regression.
///
/// This is the ONLY thing that catches those cases. `modified_test_config` is
/// warn-only by design — a patch that rewrites `[tool.pytest.ini_options]` is
/// reported, never itself rejected — so a patch CAN silence a test through
/// `addopts` or a plugin, and the only defence is that a silenced test is a
/// missing test and a missing test is a regression. Adding an explicit
/// `Some(Skipped)` arm here would be redundant: the outcome is discarded before
/// this function is ever called.
pub fn native_regressions(
    original: &[(String, TestResult)],
    patched: &[(String, TestResult)],
) -> Vec<String> {
    let patched_map = native_results_index(patched);

    let mut regressions: Vec<String> = Vec::new();
    for (name, result) in original {
        if *result != TestResult::Passed {
            continue;
        }
        match patched_map.get(name.as_str()) {
            Some(TestResult::Passed) => {}
            // Still present and no longer passing: a plain regression.
            Some(_) => regressions.push(name.clone()),
            // Absent from the patched report: the suite collapsed before (or
            // during) this test. Also a regression.
            None => regressions.push(name.clone()),
        }
    }
    regressions
}

/// nodeid → outcome for one native suite's test list, built ONCE.
///
/// [VANTA-WHY] M3 item E2 — the callers that ask "what did this nodeid do on
/// the other tree?" used to re-scan the list for every nodeid, so comparing two
/// reports was O(n_original × n_patched). On pandas that is ~88k tests per side
/// and the scan ran per attempt. One index per report makes each lookup O(1);
/// callers that also need set membership build a `HashSet` from its keys.
///
/// A duplicated nodeid keeps the FIRST occurrence — that is what the
/// `iter().find(...)` scans this replaces returned, and a report that lists the
/// same nodeid twice is pathological either way.
pub fn native_results_index(tests: &[(String, TestResult)]) -> HashMap<&str, &TestResult> {
    let mut index: HashMap<&str, &TestResult> = HashMap::with_capacity(tests.len());
    for (name, result) in tests {
        index.entry(name.as_str()).or_insert(result);
    }
    index
}

/// Did the patched side collect FEWER tests than the original ran?
///
/// [VANTA-WHY] Kept separate from `native_regressions` so the console can say
/// "the suite collapsed" rather than just listing names: 4 → 0 is a different
/// failure from 4 → 3, even though both are regressions.
pub fn native_suite_collapsed(
    original: &NativeSuiteReport,
    patched: &NativeSuiteReport,
) -> bool {
    original.ran && !original.tests.is_empty() && patched.tests.len() < original.tests.len()
}

/// Is this attempt's native gate VACUOUS — did either side measure nothing,
/// for a bug whose own preparation recorded that it has tests?
///
/// [VANTA-WHY] M3 item W. `native_regressions` over an empty original list is
/// VACUOUSLY empty, and `native_suite_collapsed` requires a non-empty original,
/// so a native pair of `(0 tests, 0 tests)` clears BOTH holes and reads as "0
/// regressions" — a clean gate on no evidence. `expected_tests` is the count the
/// row's own `prepare` recorded for this bug (``pass_to_pass_count`` plus the
/// native-failure baseline, the same figure `prepare_summary.md` prints); when
/// it is zero the bug genuinely ships no native tests and the gate is
/// legitimately silent, so the guard does not fire. When it is positive and the
/// in-session suite collected nothing, the check COULD NOT BE DONE — an
/// infrastructure fact, never a pass.
pub fn native_gate_vacuous(
    expected_tests: usize,
    original: &NativeSuiteReport,
    patched: &NativeSuiteReport,
) -> bool {
    expected_tests > 0 && (original.collected_nothing() || patched.collected_nothing())
}

/// The recorded reason a native gate is vacuous, naming the side and the cause.
///
/// [VANTA-WHY] M3 item W. The reader acts on the reason, not the boolean: "the
/// original side never ran", "the patch side collected nothing" and "neither
/// side produced a usable report" need different responses, and the exit code
/// and the stderr tail of the empty side are what tell them apart.
///
/// [VANTA-WHY] M3 item D1 — a side killed by the wall-clock cap gets its OWN
/// reason, ``native_timeout``, rather than the generic "collected nothing". The
/// two emptinesses point at opposite fixes: a collection error is the tree's own
/// import failing (pandas/1's xarray `TypeError`, fixed by the version pin),
/// while a timeout is the harness's cap being too small for a legitimate suite
/// (pandas/1's ~487 s native run under a 120 s in-session cap). `cap_secs` is the
/// cap the run actually used, so the reason states the number the operator must
/// raise. The guard is unchanged in force: either reason still ends the run as
/// `InfraFailure`, and neither can ever count as zero regressions.
pub fn native_vacuous_reason(
    expected_tests: usize,
    cap_secs: u64,
    original: &NativeSuiteReport,
    patched: &NativeSuiteReport,
) -> String {
    let side = |label: &str, report: &NativeSuiteReport| -> String {
        format!(
            "{label}={}{} (tests={} exit_code={} stderr={:?})",
            if report.ran { "ran" } else { "not_run" },
            if report.timed_out { " timed_out" } else { "" },
            report.tests.len(),
            report
                .exit_code
                .map(|code| code.to_string())
                .unwrap_or_else(|| "unset".to_string()),
            report.stderr_tail,
        )
    };
    let sides = format!(
        "{} {}",
        side("original", original),
        side("patched", patched),
    );
    if original.timed_out || patched.timed_out {
        let which = match (original.timed_out, patched.timed_out) {
            (true, true) => "both sides",
            (true, false) => "the original side",
            _ => "the patch side",
        };
        return format!(
            "native_timeout=true expected_tests={expected_tests} native_timeout_secs={cap_secs} — \
             {which} of the in-session native suite hit the run's {cap_secs}s wall-clock cap and was \
             killed, so the native gate could not run and is NOT evidence of zero regressions. This is \
             the cap being too small for this bug's suite, not the suite collecting nothing: raise the \
             derived cap (prepare's measured native time × 3) rather than reading the emptiness as a \
             tree with no tests. {sides}",
        );
    }
    format!(
        "native_vacuous=true expected_tests={expected_tests} — the row's own prepare recorded \
         {expected_tests} native test(s) for this bug, but the in-session suite collected nothing, \
         so the native gate could not run and is NOT evidence of zero regressions: {sides}",
    )
}

/// Collapse whitespace runs to `_`, so a text field can live inside the
/// whitespace-separated `k=v` form of a `native_suite.summary` record.
///
/// [VANTA-WHY] M3 item W follow-up. `pr/mod.rs::parse_summary_fields` splits a
/// summary on whitespace and each token on `=`. A raw `ImportError while
/// importing test module …` would therefore be truncated at `ImportError` AND
/// inject bogus `key=value` pairs from its own interior tokens (a path like
/// `module=foo` becomes a spurious field). The free-text copies of the same
/// census — the `empty_report_note` in `stderr_tail` and the
/// `native_vacuous_reason` — keep their spaces; only the summary is tokenised.
fn summary_token(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join("_")
}

/// The collection census of BOTH sides of the native gate, as the
/// whitespace-free `k=v` tail of a `native_suite.summary` record.
///
/// [VANTA-WHY] M3 item W follow-up. When the guard fires, the session record
/// said only `original_tests=0 patched_tests=0` — the count of what was
/// collected, never WHY. pytest's own pointer to the module that broke the
/// session is the `collectors` census (`outcome: "failed"` plus the
/// `ImportError while importing test module …` text), so the count, the first
/// failing module's nodeid and its error are recorded for BOTH sides. `0` and
/// `none none` describe a side that collected cleanly (or has no census at
/// all): the point is that a side which DID fail names itself here.
pub fn native_collection_summary(
    original: &NativeSuiteReport,
    patched: &NativeSuiteReport,
) -> String {
    let side = |label: &str, report: &NativeSuiteReport| -> String {
        let failed = report.collection.as_ref().map(|c| c.failed).unwrap_or(0);
        let (nodeid, error) = match report.collection.as_ref().and_then(|c| c.first.clone()) {
            Some((nodeid, error)) => (summary_token(&nodeid), summary_token(&error)),
            None => ("none".to_string(), "none".to_string()),
        };
        format!(
            "{label}_failed_collectors={failed} {label}_first_failed_collector={nodeid} \
             {label}_first_collector_error={error}"
        )
    };
    format!("{} {}", side("original", original), side("patched", patched))
}

// ============================================================================
// The native tree — the patched source graded by the ORIGINAL tests
// ============================================================================
//
// [VANTA-WHY] A patch must never control the tests that grade it. The vulnerable
// shape, seen in repair session 888b6931: the Creator's fix made
// `keyword_rate.py` always pass `max_response_length` to `get_responses`, which
// the issue forbids when the setting is unset — and the SAME patch edited
// `tests/test_keyword_rate.py`'s `FakeContext.get_responses` to swallow the new
// kwarg. The native suite ran the patched tree, so it collected the Creator's
// edited test, the weakened test passed, Vanta reported `regressions=0`, and the
// Gatekeeper approved a patch the repo's own test rejects with a `TypeError`.
//
// The native suite therefore runs the ORIGINAL test files. Every file pytest
// would collect or load — `test_*.py`, `*_test.py`, anything under a collected
// test directory, and every `conftest.py` — is pinned back to the contents the
// original repo had, on top of the patched source. A patch-added test file is
// not part of the baseline either, so it is left out of the native run
// entirely; the baseline is the repo's ORIGINAL suite and nothing else.

/// Is this a file the repo's own test suite owns — collected by pytest, or
/// loaded by it as a conftest?
///
/// [VANTA-WHY] `conftest.py` is its own case because pytest loads it even though
/// the name matches no collection pattern, and a conftest is the cheapest place
/// to neutralize a test (`collect_ignore`, a monkeypatched fixture, a
/// `pytest_collection_modifyitems` that deselects). Anything under a `test`/
/// `tests` directory counts too: a helper module there changes what the tests do
/// without being a test file itself.
pub fn is_native_test_path(path: &str) -> bool {
    let path_ref = Path::new(path);
    let file_name = path_ref
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if file_name == "conftest.py" {
        return true;
    }
    if !file_name.ends_with(".py") {
        return false;
    }
    if file_name.starts_with("test_") || file_name.ends_with("_test.py") {
        return true;
    }
    path_ref.parent().is_some_and(|dir| {
        dir.components().any(|component| {
            matches!(component, std::path::Component::Normal(name)
                if name == "test" || name == "tests")
        })
    })
}

/// The native-suite view of one patch attempt.
pub struct NativeTree {
    /// The tree the native (patched-side) run is given.
    pub files: HashMap<String, String>,
    /// Existing test files the patch rewrote; their edits were discarded.
    pub modified_existing_tests: Vec<String>,
    /// Existing test files the patch deleted; they were restored.
    pub removed_existing_tests: Vec<String>,
    /// Test files the patch added. Not part of the baseline; reported as info.
    pub added_test_files: Vec<String>,
}

/// Build the patched tree the native suite runs: patched source, ORIGINAL tests.
///
/// `patch_deleted_tests` comes from the patch TEXT, not from `patched_files`:
/// the apply cascade leaves a deleted file at its original contents (the strict
/// path skips a `/dev/null` destination outright), so a deletion is invisible in
/// the file map and can only be read off the diff itself.
pub fn native_tree_with_original_tests(
    original_files: &HashMap<String, String>,
    patched_files: &HashMap<String, String>,
    patch_deleted_tests: &[String],
) -> NativeTree {
    let mut files: HashMap<String, String> = HashMap::new();
    let mut modified_existing_tests: Vec<String> = Vec::new();
    let mut added_test_files: Vec<String> = Vec::new();

    let mut paths: Vec<&String> = patched_files.keys().collect();
    paths.sort();
    for path in paths {
        let contents = &patched_files[path];
        if !is_native_test_path(path) {
            files.insert(path.clone(), contents.clone());
            continue;
        }
        match original_files.get(path) {
            Some(original) => {
                if original != contents {
                    modified_existing_tests.push(path.clone());
                }
                files.insert(path.clone(), original.clone());
            }
            // A test file the patch ADDED — including a new `conftest.py`. It is
            // not part of the original baseline, so it is excluded rather than
            // restored, and it can never be the reason an original test "passes"
            // or disappears.
            None => added_test_files.push(path.clone()),
        }
    }

    // A test file the patch deleted: restore the original so the test still
    // runs. Without this a patch could silence a test simply by removing it.
    let mut removed_existing_tests: Vec<String> = Vec::new();
    for path in patch_deleted_tests {
        if !is_native_test_path(path) {
            continue;
        }
        let Some(original) = original_files.get(path) else {
            continue;
        };
        removed_existing_tests.push(path.clone());
        files.insert(path.clone(), original.clone());
    }
    removed_existing_tests.sort();
    removed_existing_tests.dedup();

    NativeTree {
        files,
        modified_existing_tests,
        removed_existing_tests,
        added_test_files,
    }
}

// ============================================================================
// Source-tree helpers — write the concatenated source into a sandbox
// ============================================================================

/// Parse a source string concatenated with `# FILE: <path>` headers into
/// (path, contents) pairs.
pub fn parse_source_files(
    source_concat: &str,
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error + Send + Sync>> {
    let mut files = Vec::new();
    let mut current_path: Option<String> = None;
    let mut current_content_start = 0usize;
    let mut offset = 0usize;

    for segment in source_concat.split_inclusive('\n') {
        let line = segment.strip_suffix('\n').unwrap_or(segment);
        if let Some(rest) = line.strip_prefix("# FILE: ") {
            if let Some(path) = current_path.take() {
                let content_end = offset.saturating_sub(2);
                let content = source_concat
                    .get(current_content_start..content_end)
                    .ok_or("invalid concatenated source boundaries")?;
                files.push((path, content.to_string()));
            }
            current_path = Some(rest.trim().to_string());
            current_content_start = offset + segment.len();
        }
        offset += segment.len();
    }

    if let Some(path) = current_path.take() {
        let content = source_concat
            .get(current_content_start..)
            .ok_or("invalid final source boundaries")?;
        files.push((path, content.to_string()));
    }

    if files.is_empty() {
        return Err("no # FILE: entries found in concatenated source".into());
    }

    Ok(files)
}

/// Write source files (concatenated with `# FILE:` headers) into a sandbox root.
pub fn write_source_files(
    root: &Path,
    source_concat: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    for (path, content) in parse_source_files(source_concat)? {
        // [VANTA-WHY] T1 — the `# FILE:` header carries a path the patch chose
        // (via the apply's own map), and `root.join(path)` is a WRITE. Confine
        // it: an absolute or escaping path cannot reach here through the apply,
        // but the concatenation is also read back from a snapshot on the rerun
        // path, so this is the lock on the tree the tests actually run in.
        let out = confined_join(root, &path)?;
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(out, content)?;
    }
    Ok(())
}

/// Write a forensics snapshot of the source tree plus a manifest.
pub fn write_tree_snapshot(
    snapshot_root: &Path,
    source_concat: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if snapshot_root.exists() {
        fs::remove_dir_all(snapshot_root)?;
    }
    fs::create_dir_all(snapshot_root)?;

    let files = parse_source_files(source_concat)?;
    let mut paths: Vec<String> = files.iter().map(|(path, _)| path.clone()).collect();
    paths.sort();

    let file_count = files.len();
    for (path, content) in &files {
        // [VANTA-WHY] T1 — CodeRabbit's flagged writer: `snapshot_root.join(path)`
        // took the path from the concatenation verbatim and wrote it on the host.
        // Confined like every other write; the root is created just above, so the
        // containment check can resolve it.
        let out = confined_join(snapshot_root, path)?;
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(out, content)?;
    }

    let manifest = serde_json::json!({
        "paths": paths,
        "file_count": file_count,
    });
    let manifest_name = format!(
        "{}_manifest.json",
        snapshot_root
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("snapshot_root missing file name")?
    );
    let manifest_path = snapshot_root
        .parent()
        .ok_or("snapshot_root missing parent")?
        .join(manifest_name);
    fs::write(manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

/// Write a conftest.py at the sandbox root that puts `src/` and `.` on sys.path.
/// [VANTA-WHY] Repos with a `src/` layout need `src/` on sys.path for
/// `import <pkg>` to work. pytest auto-loads conftest.py from the rootdir.
pub fn write_conftest(root: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // [VANTA-WHY] Use a raw string literal (r#"..."#) so the Python indentation
    // is preserved exactly. Line-continuation style stripped leading whitespace
    // and caused IndentationError → every test returned Error (pytest exit 4,
    // no report.json).
    let conftest = r#"import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, 'src'), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
"#;
    fs::write(root.join("conftest.py"), conftest)?;
    Ok(())
}

/// Lay down the tree a sandbox will be graded on: the source overlay
/// (`seed_and_write`) followed by the conftest that puts `src/` on `sys.path`.
///
/// [VANTA-WHY] F1 — every one of these writes used to be discarded (`let _ =`)
/// on a path that ends in a verdict about the candidate. `seed_and_write` can
/// fail inside `prune_source_files_not_in`, AFTER the seed copy but BEFORE the
/// overlay, so the sandbox is left holding the SEEDED tree rather than
/// `source_concat`; a discarded `write_conftest` failure leaves `src/` off
/// `sys.path` (a `src/`-layout repo then fails to import). Either way the test
/// runs against a tree it is not being asked about and returns a normal-looking
/// Passed/Failed. `Err` here means the caller MUST destroy the sandbox and
/// return its own failure value (`infrastructure_error()` / `not_run()`), never
/// grade the tree.
fn seed_source_and_conftest(
    root: &Path,
    source_concat: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    seed_and_write(root, source_concat)?;
    write_conftest(root)
}

// ============================================================================
// The per-run sandbox SEED (item P) — the built tree, cloned per sandbox
// ============================================================================
//
// [VANTA-WHY] Item P. Until this, every sandbox was seeded from the UTF-8
// SOURCE CONCATENATION alone. `driver::build_source_concat` is built from
// `context::walk_candidates`, which keeps `SOURCE_EXTENSIONS` files only, so a
// compiled project (pandas/1, matplotlib/1) reached the VM without its `.so`
// extensions — every in-session import died as a partially-initialised module
// — and without its root `pytest.ini`/`setup.cfg`, so the in-session run was
// NOT the run the harness's own oracle and native suites perform. The
// in-session suite must run in the tree the prepare step built in place; the
// concatenation is only the SOURCE overlay on top of it.
//
// The stage is built ONCE per run (`prepare_seed_tree`), from the on-disk
// execution tree, and every sandbox gets its own writable tree from it
// (`seed_sandbox`): a copy-on-write clonefile where the volume allows one, a
// plain recursive copy otherwise. Which of the two the first seed used is
// recorded, so the row can state it (a CoW row and a copy row are not the same
// run). `set_seed_root(None)` — the default, and what the in-file unit tests
// use — leaves every function byte-identical to the concat-only behaviour.

/// Which mechanism a sandbox seed used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedMethod {
    /// `cp -c` / clonefile — the sandbox shares blocks with the run's seed.
    Clonefile,
    /// A full recursive copy; the fallback when clonefile is unavailable.
    CopyTree,
}

impl SeedMethod {
    /// The token the row records this mechanism under.
    pub fn as_str(self) -> &'static str {
        match self {
            SeedMethod::Clonefile => "clonefile",
            SeedMethod::CopyTree => "copy_tree",
        }
    }
}

/// The on-disk tree every sandbox of this run is seeded from.
static SEED_ROOT: RwLock<Option<PathBuf>> = RwLock::new(None);
/// Which mechanism the FIRST sandbox seed used.
static SEED_METHOD: RwLock<Option<SeedMethod>> = RwLock::new(None);

/// Point the harness at the run's seed tree (item P). `None` restores the
/// concat-only behaviour, which is what a caller that never built a seed gets.
pub fn set_seed_root(root: Option<PathBuf>) {
    let mut guard = SEED_ROOT.write().unwrap_or_else(|e| e.into_inner());
    *guard = root;
    let mut method = SEED_METHOD.write().unwrap_or_else(|e| e.into_inner());
    *method = None;
}

/// The run's seed tree, if one was set.
pub fn seed_root() -> Option<PathBuf> {
    SEED_ROOT.read().ok().and_then(|guard| guard.clone())
}

/// Which mechanism the run's first sandbox seed used, for the row's own record.
pub fn seed_method() -> Option<SeedMethod> {
    SEED_METHOD.read().ok().and_then(|guard| *guard)
}

/// Build the run's seed tree once: a staging copy of the on-disk execution tree
/// with `SKIP_DIRS` and symlinks left out, so a sandbox can be seeded from it by
/// a copy-on-write clone.
///
/// [VANTA-WHY] The staging dir must live under the process's CWD: the Shuru
/// backend mounts a sandbox host dir into the VM with `--mount`, and shuru
/// 0.7.0 restricts `--mount` to paths within CWD (`sandbox/shuru.rs`). The
/// caller holds the returned `TempDir` for the whole run; dropping it removes
/// the stage.
pub fn prepare_seed_tree(
    execution_root: &Path,
) -> Result<tempfile::TempDir, Box<dyn std::error::Error + Send + Sync>> {
    let staging = tempfile::Builder::new()
        .prefix("vanta-seed-")
        .tempdir_in(std::env::current_dir()?)?;
    copy_tree_excluding(staging.path(), execution_root)?;
    Ok(staging)
}

/// Recursive copy that leaves out `SKIP_DIRS` names and symlinks.
///
/// [VANTA-WHY] `.git` and the build output are excluded for the reason
/// `rerun::copy_tree_recursive` gives (an entire object database copied per
/// sandbox), and symlinks for the same reason: a symlink pointing back into the
/// original tree would let a `git apply` escape the copy.
fn copy_tree_excluding(dst: &Path, src: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if crate::context::SKIP_DIRS.contains(&name_str.as_ref()) {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            copy_tree_excluding(&to, &from)?;
        } else if file_type.is_file() {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Seed one sandbox root from the run's seed tree. `None` when no seed was set
/// (concat-only), so the caller can tell "seeded" from "nothing to do".
///
/// [VANTA-WHY] `cp -cR` clones rather than copies, so a 200 MB built tree costs
/// metadata rather than 200 MB of I/O per sandbox — and
/// `run_tests_parallel_detailed` spawns one sandbox PER TEST. If the clone fails
/// (a non-APFS volume, or a file clonefile refuses) the tree is copied instead,
/// and the fallback is what the row records.
pub fn seed_sandbox(root: &Path) -> Option<SeedMethod> {
    let seed = seed_root()?;
    let method = seed_sandbox_from(&seed, root)?;
    let mut recorded = SEED_METHOD.write().unwrap_or_else(|e| e.into_inner());
    if recorded.is_none() {
        *recorded = Some(method);
    }
    Some(method)
}

/// The mechanism-only half of `seed_sandbox`, with the seed named explicitly:
/// no global is read and none is written, so a unit test can exercise the clone
/// and its fallback without disturbing a concurrent run's harness.
fn seed_sandbox_from(seed: &Path, root: &Path) -> Option<SeedMethod> {
    let cloned = std::process::Command::new("cp")
        .arg("-cR")
        .arg(format!("{}/.", seed.display()))
        .arg(format!("{}/", root.display()))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if cloned {
        return Some(SeedMethod::Clonefile);
    }
    copy_tree_excluding(root, seed).ok()?;
    Some(SeedMethod::CopyTree)
}

/// Seed a sandbox root (when a seed is set) and overlay the concatenated source
/// on top — the one entry point every test-running function uses.
///
/// [VANTA-WHY] The overlay is the whole point: the seed carries what the walk
/// cannot (`pyproject.toml`, `pytest.ini`, `setup.cfg`, and every compiled
/// extension), and the concatenation carries the SOURCE UNDER TEST, which for a
/// patched attempt is not what is on disk. A source file the concatenation no
/// longer lists — a path the patch deleted — is pruned from the seeded tree
/// first, so the executed tree's source is exactly the concatenation's.
pub fn seed_and_write(
    root: &Path,
    source_concat: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // [VANTA-WHY] Item P follow-up — the production path goes through
    // `seed_sandbox`, which is the one place that RECORDS the run's mechanism.
    // `seed_sandbox_from` is deliberately global-free (the unit tests use it), so
    // seeding through it, as this did, seeded every tree correctly and still left
    // every row saying `seed_method=unset`.
    let seeded = seed_sandbox(root).is_some();
    seed_and_write_after(seeded, root, source_concat)
}

/// The seed-plus-overlay step with the seed named explicitly (`None` = no seed),
/// so the exact behaviour both with and without a seed is unit-testable. It is
/// the tests' seam and nothing else now that `seed_and_write` seeds through
/// `seed_sandbox` — the variant that records the mechanism — so it is compiled
/// only for them.
#[cfg(test)]
fn seed_and_write_from(
    seed: Option<&Path>,
    root: &Path,
    source_concat: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let seeded = seed
        .map(|seed| seed_sandbox_from(seed, root).is_some())
        .unwrap_or(false);
    seed_and_write_after(seeded, root, source_concat)
}

/// Overlay the concatenation on a root that `seeded` says was seeded: the source
/// the concatenation dropped is pruned first. Only a seed can carry source the
/// walk did not, so an unseeded root has nothing to prune.
fn seed_and_write_after(
    seeded: bool,
    root: &Path,
    source_concat: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if seeded {
        prune_source_files_not_in(root, source_concat)?;
    }
    write_source_files(root, source_concat)
}

/// Delete every `SOURCE_EXTENSIONS` file under `root` the concatenation does not
/// list. Bounded to the source extensions the walk reads, so a `.so` or a config
/// file the seed brought in is never touched.
fn prune_source_files_not_in(
    root: &Path,
    source_concat: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let keep: std::collections::HashSet<String> = parse_source_files(source_concat)?
        .into_iter()
        .map(|(path, _)| path)
        .collect();
    prune_walk(root, root, &keep)?;
    Ok(())
}

/// Decide whether the prune keeps the source file at `relative`.
///
/// [VANTA-WHY] C7 — `keep` is the set of path keys the concatenation lists, and
/// every key is a UTF-8 `String` (the concatenation is text). A relative path that
/// is not valid UTF-8 therefore cannot be one of them, and the old
/// `to_string_lossy()` decode turned it into a string that matched nothing —
/// deleting a source file the concatenation never told us to drop. Membership is
/// undecidable for such a path, and the prune must not answer an undecidable case
/// by deleting: it keeps the file. A valid-UTF-8 path is compared exactly as
/// before, so the drop of a file the concatenation omits is unchanged.
fn prune_keeps(relative: &Path, keep: &std::collections::HashSet<String>) -> bool {
    match relative.to_str() {
        Some(relative) => keep.contains(relative),
        None => true,
    }
}

fn prune_walk(
    root: &Path,
    dir: &Path,
    keep: &std::collections::HashSet<String>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        // [VANTA-WHY] C7 — `SKIP_DIRS` holds ASCII names, so a name that is not valid
        // UTF-8 cannot be one of them; reading it lossily could only swap its invalid
        // bytes for U+FFFD, which matches nothing here. Comparing only when the name
        // decodes states that in the type instead of relying on the substitution.
        if let Some(name_str) = name.to_str() {
            if crate::context::SKIP_DIRS.contains(&name_str) {
                continue;
            }
        }
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            prune_walk(root, &path, keep)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !crate::context::SOURCE_EXTENSIONS.contains(&extension) {
            continue;
        }
        let Ok(relative) = path.strip_prefix(root) else {
            continue;
        };
        if !prune_keeps(relative, keep) {
            fs::remove_file(&path)?;
        }
    }
    Ok(())
}

// ============================================================================
// The in-session IMPORT GUARD (item P, point 2)
// ============================================================================

/// The project's own top-level package: the first `__init__.py` directory
/// directly under a `sys.path` root (the workspace itself, then each
/// `PYTHONPATH` entry the run records).
///
/// [VANTA-WHY] `detect_import_path` reads a `src/` layout and misses
/// matplotlib's, whose package sits under the `lib/` its own `PYTHONPATH=lib`
/// names — which is exactly the layout the guard has to cover. Reading the
/// `PYTHONPATH` the command actually carries is what makes the same rule work
/// for both. Names that are the repo's own tooling rather than its package are
/// skipped, so a stray `tests/__init__.py` cannot become "the project".
pub fn project_package(tree_root: &Path, test_cmd: &TestCommand) -> Option<String> {
    let mut roots: Vec<PathBuf> = vec![PathBuf::new()];
    for (key, value) in &test_cmd.env {
        if key != "PYTHONPATH" {
            continue;
        }
        for part in value.split(':') {
            let part = part.trim().trim_start_matches("./").trim_end_matches('/');
            if !part.is_empty() && part != "." {
                roots.push(PathBuf::from(part));
            }
        }
    }
    for relative in roots {
        let Ok(entries) = fs::read_dir(tree_root.join(&relative)) else {
            continue;
        };
        let mut names: Vec<String> = entries
            .flatten()
            .filter(|entry| {
                let path = entry.path();
                path.is_dir() && path.join("__init__.py").is_file()
            })
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| !is_tooling_package(name))
            .collect();
        names.sort();
        if let Some(name) = names.into_iter().next() {
            return Some(name);
        }
    }
    None
}

/// A directory that is the repo's own tooling, not the package under test.
fn is_tooling_package(name: &str) -> bool {
    matches!(
        name,
        "test" | "tests" | "doc" | "docs" | "example" | "examples" | "benchmark"
            | "benchmarks" | "asv_bench" | "scripts" | "tools" | "build" | "dist"
    )
}

/// The in-session twin of the prepare-time gate: the project's package must
/// import FROM THE TREE UNDER TEST, or the run is infrastructure, not a
/// measurement.
///
/// [VANTA-WHY] `scripts/bugsinpy/prepare.py` refuses a bug whose package
/// resolves outside the tree — but that gate runs in the PREPARE layer and
/// cannot see a sandbox whose tree was seeded from the source concatenation
/// alone: there the package was simply absent and every import resolved to
/// whatever site-packages held, which is precisely the silent, unmeasurable
/// state this guard converts into a stated `InfraFailure`. `Ok(..)` carries the
/// line the run records (`import_guard=<module> file=<path>`); `Err(..)` is the
/// reason the run must stop.
pub async fn guard_project_import(
    source_concat: &str,
    test_cmd: &TestCommand,
    tree_root: &Path,
) -> Result<String, String> {
    let module = match project_package(tree_root, test_cmd) {
        Some(module) => module,
        None => {
            return Ok(
                "import_guard=none — no top-level package under a sys.path root (a single-module \
                 project); the guard has nothing to check"
                    .to_string(),
            )
        }
    };
    let sandbox = Backend::spawn()
        .await
        .map_err(|e| format!("import_guard: sandbox could not spawn: {e}"))?;
    let host_root = sandbox.root().display().to_string();
    if seed_and_write(sandbox.root(), source_concat).is_err() {
        sandbox.destroy();
        return Err(
            "import_guard: the tree under test could not be written into the sandbox".to_string(),
        );
    }
    let _ = write_conftest(sandbox.root());
    // [VANTA-WHY] No `.pyc` may be left in the tree under test: the probe
    // imports the package exactly as a suite run would, and a bytecode cache
    // written by the guard would be a run's leavings inside the graded tree.
    let mut env = test_cmd.env.clone();
    env.push(("PYTHONDONTWRITEBYTECODE".to_string(), "1".to_string()));
    let probe = format!("import {module} as _vanta_import_guard; print(_vanta_import_guard.__file__)");
    let program = test_cmd.program.clone();
    let exec = sandbox
        .execute(
            &[program.as_str(), "-c", probe.as_str()],
            Path::new(""),
            120,
            &env,
            &test_cmd.precompile,
        )
        .await;
    sandbox.destroy();
    let exec = exec.map_err(|e| format!("import_guard: the probe could not run: {e}"))?;
    let Some(where_from) = exec
        .stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .next_back()
    else {
        return Err(format!(
            "import_guard: `import {module}` printed nothing (exit {:?}, stderr: {})",
            exec.exit_code,
            stderr_tail(&exec.stderr)
        ));
    };
    // [VANTA-WHY] The sandbox host dir is mounted into the VM at `/workspace`
    // (`sandbox/shuru.rs`), so a package that resolved from the tree under test
    // reports a `/workspace/...` path in the guest — while the same probe under
    // the subprocess backend reports the host root. Both are the tree under
    // test; anything else (site-packages, a frozen `setup.py install`) is not.
    if !where_from.starts_with("/workspace/") && !where_from.starts_with(&format!("{host_root}/")) {
        return Err(format!(
            "import_guard: the package does not import from the tree under test — \
             `import {module}` loaded {where_from}, which is neither under /workspace/ nor under \
             {host_root}/"
        ));
    }
    Ok(format!("import_guard={module} file={where_from}"))
}

// ============================================================================
// pytest report parsing
// ============================================================================

/// The `tests` array of a pytest-json-report file, whichever schema wrote it.
///
/// [VANTA-WHY] Which schema the plugin writes is a function of its *version*,
/// and its version is a function of the bug's own pytest pin
/// (plans/bugsinpy.md §3.2): `pytest-json-report` 0.1/0.2 write a bare JSON
/// list, 0.3 and later write `{"tests": [...]}`. `httpie/1` pins
/// `pytest==3.2.1`, which caps the plugin at 0.2, so that bug's layer writes the
/// flat list. Every reader here looked only under `["tests"]`, so on that layer
/// the native suite reported `ran: true` with ZERO tests and every generated
/// test graded `Error` — an InfraFailure that has nothing to do with the model.
///
/// `Err` is "neither schema", and a caller must not read it as an empty report:
/// a report of the wrong top-level type is a reason to fail closed (no verdict
/// was reached), never a reason to conclude the tree has no tests.
pub fn report_tests(report: &serde_json::Value) -> Result<&[serde_json::Value], &'static str> {
    if let Some(entries) = report.as_array() {
        return Ok(entries);
    }
    if let Some(entries) = report.get("tests").and_then(serde_json::Value::as_array) {
        return Ok(entries);
    }
    Err(
        "the report is neither the flat list pytest-json-report 0.1/0.2 write nor the \
         {\"tests\": [...]} mapping 0.3 and later write",
    )
}

/// Parse pytest's JSON report into the outcome of a specific generated test
/// file.
///
/// [VANTA-WHY] An "error" outcome (ImportError/AttributeError during the test
/// body) means the test did NOT pass — equivalent to "failed". `TestResult::Error`
/// is reserved for INFRASTRUCTURE failures (timeout, missing report, crash),
/// which now also covers "the report does not contain the generated test".
pub fn parse_pytest_report(
    report_path: PathBuf,
    exec: &ExecutionResult,
    generated_test_file: &str,
) -> TestResult {
    parse_pytest_report_detailed(report_path, exec, None, generated_test_file).result
}

pub fn parse_pytest_report_detailed(
    report_path: PathBuf,
    exec: &ExecutionResult,
    sandbox_root: Option<String>,
    generated_test_file: &str,
) -> TestExecutionDetails {
    if exec.timed_out {
        return TestExecutionDetails {
            result: TestResult::Error,
            stdout: exec.stdout.clone(),
            stderr: exec.stderr.clone(),
            exit_code: exec.exit_code,
            timed_out: true,
            report_path_exists: report_path.exists(),
            sandbox_root,
        };
    }
    if report_path.exists() {
        if let Ok(text) = fs::read_to_string(&report_path) {
            if let Ok(report) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(result) = generated_test_outcome(&report, generated_test_file) {
                    return TestExecutionDetails {
                        result,
                        stdout: exec.stdout.clone(),
                        stderr: exec.stderr.clone(),
                        exit_code: exec.exit_code,
                        timed_out: false,
                        report_path_exists: true,
                        sandbox_root,
                    };
                }
            }
        }
    }
    // [VANTA-WHY] Fail closed. The report exists but the GENERATED test's nodeid
    // is not in it (or there is no usable report at all): the generated test was
    // never graded, so `report_path_exists == false` — "this tree did not reach a
    // verdict" — and the result is `Error`, never `Passed`. `tests.first()` used
    // to stand here, and the first entry was a passing repository test.
    TestExecutionDetails {
        result: TestResult::Error,
        stdout: exec.stdout.clone(),
        stderr: exec.stderr.clone(),
        exit_code: exec.exit_code,
        timed_out: false,
        report_path_exists: false,
        sandbox_root,
    }
}

/// The outcome of the GENERATED test file, selected by the file part of its
/// nodeid.
///
/// [VANTA-WHY] The grading bug. A `--test-cmd` that named a test path (e.g.
/// `python -m pytest tests`) made pytest collect the repository's own suite
/// before the generated file, and this reader took `report["tests"].first()` as
/// the generated test's result. That first entry was always one of the repo's
/// tests — which passes — so every BugReproduction test was recorded as "passed
/// on the buggy original", scrapped by BATE, and no candidate could ever be
/// rejected for failing to reproduce the bug. The generated file is written into
/// the sandbox as one fixed name (`test_run.py` / `test_bate.py`), so its nodeid
/// is identified by that FILE. A nodeid that is absent from the report yields
/// `None` — the caller fails closed rather than reading someone else's result.
///
/// The entries come from :func:`report_tests`, so the plugin's schema (0.1/0.2
/// flat list or 0.3+ mapping) is not this function's business; `None` covers an
/// unreadable shape as well as an absent nodeid, and both fail closed.
fn generated_test_outcome(
    report: &serde_json::Value,
    generated_test_file: &str,
) -> Option<TestResult> {
    let wanted = nodeid_file(generated_test_file);
    for t in report_tests(report).ok()? {
        let nodeid = t["nodeid"].as_str().unwrap_or_default();
        if nodeid_file(nodeid) == wanted {
            let result = match t["outcome"].as_str().unwrap_or_default() {
                "passed" | "subtests passed" => TestResult::Passed,
                // "failed" = assertion failure; "error" = exception during the
                // test body. Both mean the generated test did not pass.
                "failed" | "error" => TestResult::Failed,
                // An outcome we do not recognise is not a licence to pass.
                _ => TestResult::Error,
            };
            return Some(result);
        }
    }
    None
}

/// The file part of a pytest nodeid, normalised so `./a.py` and `a.py` match.
fn nodeid_file(nodeid: &str) -> String {
    nodeid
        .split("::")
        .next()
        .unwrap_or(nodeid)
        .trim()
        .trim_start_matches("./")
        .replace('\\', "/")
}

// ============================================================================
// Static analysis (ruff + mypy) — SET differences, not COUNT differences
// ============================================================================
//
// [VANTA-WHY] The old check compared COUNTS: `patch_ruff > base_ruff`. A count
// is not a set. If the patch fixed two pre-existing ruff errors and introduced
// two new ones, the count is identical and the run reports "no new static
// errors" for a patch that just added two. Worse, ruff and mypy both number
// their diagnostics by LINE, so a patch that shifts every line down by one
// changes every diagnostic's position without changing a single finding, and the
// raw text comparison would call all of them new.
//
// The fix is to compare the findings as a SET keyed on (file, code, message) —
// deliberately WITHOUT line numbers — and to hand the Diagnostician only the
// findings the patch actually introduced.

/// One static-analysis finding.
///
/// [VANTA-WHY] The `(file, code, message)` triple is the IDENTITY used by the
/// set diff, so the line number is excluded from `PartialEq`/`Hash`: the same
/// diagnostic moving down the file after a patch is not a NEW error, and
/// counting it as one would reject a candidate that merely changed its own line
/// count. The line is still CARRIED whenever the tool reported one — the
/// evidence record and the console need the exact site, the set diff must not.
#[derive(Debug, Clone)]
pub struct StaticDiagnostic {
    /// Repo-relative path the tool reported.
    pub file: String,
    /// 1-based line the tool reported, when it reported one.
    pub line: Option<u32>,
    /// ruff rule (e.g. `F821`) or mypy error code (e.g. `return-value`).
    pub code: String,
    /// The human-readable finding text, without the location.
    pub message: String,
}

// [VANTA-WHY] Identity is (file, code, message) — see the struct doc. Deriving
// PartialEq/Hash would fold the line in and break the set diff, so they are
// written by hand and deliberately ignore `line`.
impl PartialEq for StaticDiagnostic {
    fn eq(&self, other: &Self) -> bool {
        self.file == other.file && self.code == other.code && self.message == other.message
    }
}
impl Eq for StaticDiagnostic {}
impl std::hash::Hash for StaticDiagnostic {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.file.hash(state);
        self.code.hash(state);
        self.message.hash(state);
    }
}

impl StaticDiagnostic {
    /// Is this finding a fact about the ANALYSIS ENVIRONMENT rather than the
    /// code?
    ///
    /// [VANTA-WHY] Item 10. mypy's `import-not-found` / `import-untyped` mean
    /// exactly "I cannot see this module from here" — an environment fact, not a
    /// defect in the candidate. Session 40e84874 rejected C because the generated
    /// `tests/test_auto_rank.py` imports `torch` and the sandbox had no
    /// dependencies installed, so `torch` was reported as `import-not-found`.
    /// These findings are still RECORDED (a reader must be able to see them) but
    /// must never block a review round: blocking on one claims the candidate is
    /// broken when it merely imports a library the analysis environment lacks.
    pub fn is_environment_finding(&self) -> bool {
        matches!(self.code.as_str(), "import-not-found" | "import-untyped")
    }

    /// One line for the Diagnostician's payload and the session log, with the
    /// reported line number when the tool gave one.
    pub fn render(&self) -> String {
        let location = match self.line {
            Some(line) => format!("{}:{}", self.file, line),
            None => self.file.clone(),
        };
        if self.code.is_empty() {
            format!("{}: {}", location, self.message)
        } else {
            format!("{}: [{}] {}", location, self.code, self.message)
        }
    }
}

/// What one static run found, per tool, plus the raw streams.
#[derive(Debug, Clone, Default)]
pub struct StaticRun {
    pub ruff: Vec<StaticDiagnostic>,
    pub mypy: Vec<StaticDiagnostic>,
    pub ruff_raw: String,
    pub mypy_raw: String,
}

/// Repo config files that change what ruff / mypy / pytest report.
///
/// [VANTA-WHY] Item 10: the sandbox only carried `src/` and `tests/`, so ruff
/// ran with its DEFAULT ruleset and mypy with no config at all. A repo's own
/// config is part of how that repo defines "correct" — without it the static
/// step reports the wrong findings and says nothing about the candidate. These
/// are the standard discovery names of the three tools, plus pytest's, so a
/// root `conftest.py` that adjusts `sys.path` is honoured too.
pub const STATIC_CONFIG_FILES: &[&str] = &[
    "pyproject.toml",
    "setup.cfg",
    "mypy.ini",
    ".mypy.ini",
    "ruff.toml",
    ".ruff.toml",
    "pytest.ini",
    "tox.ini",
    "conftest.py",
];

/// The ruff invocation, minus the interpreter.
///
/// [VANTA-WHY] Item 10: ruff's DEFAULT output is the multi-line "full" report
/// (`F821 Undefined name ...` then ` --> path.py:1:5` on its own line), which
/// `parse_ruff_line` cannot read — so every ruff run so far silently parsed to
/// ZERO findings, whether the tree had one finding or forty. `concise` is the
/// one-line `path.py:1:5: F821 message` form the parser expects, so the flag is
/// part of the contract, not a preference.
pub const RUFF_ARGS: &[&str] = &["-m", "ruff", "check", "--output-format=concise", "."];

/// Copy the repo's config files from `repo_root` into the sandbox `dest`.
///
/// [VANTA-WHY] Only files that actually exist are copied: a repo without a
/// `ruff.toml` must not gain an empty one, because an empty config is not the
/// same as "no config" — it would change ruff's behaviour rather than reproduce
/// it. Returns the names copied, so a caller can log them.
pub fn copy_static_configs(repo_root: &Path, dest: &Path) -> Vec<String> {
    let mut copied = Vec::new();
    for name in STATIC_CONFIG_FILES {
        let src = repo_root.join(name);
        if src.is_file() && fs::copy(&src, dest.join(name)).is_ok() {
            copied.push((*name).to_string());
        }
    }
    copied
}

/// The Python interpreter a test command actually runs.
///
/// [VANTA-WHY] Item 10: the static tools must run under the SAME interpreter as
/// the tests — only that interpreter has the repo's dependencies installed, so
/// only it can resolve `torch`/`pydantic` instead of reporting them as
/// `import-not-found`, and only under it does mypy see real types rather than
/// treating every module as `Any`. The real heretic command is
/// `env PYTHONPATH=src /path/.venv/bin/python -m pytest -q`, where the
/// interpreter hides behind an `env` wrapper and an env assignment; a plain
/// `python3 -m pytest` names it directly. Both must resolve to the venv python,
/// never to the host's bare `python3`. Falls back to `VANTA_PYTHON` (then
/// `python3`) when the command names no interpreter at all.
pub fn interpreter_for(test_cmd: &TestCommand) -> String {
    let mut tokens = vec![test_cmd.program.as_str()];
    tokens.extend(test_cmd.args.iter().map(String::as_str));
    for token in tokens {
        // `PYTHONPATH=src` is an env assignment and `-m` is a flag; neither is
        // the interpreter.
        if token.contains('=') || token.starts_with('-') {
            continue;
        }
        let name = Path::new(token)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if name.starts_with("python") {
            return token.to_string();
        }
    }
    python_exe()
}

/// Can `python` import `module`?
///
/// [VANTA-WHY] A cheap HOST-side probe (`import <module>` only resolves an
/// already-installed package — no untrusted code runs). It is what lets the
/// static step pick a runner that actually has the tool installed, instead of
/// assuming the project interpreter and the tool interpreter are the same
/// process.
fn has_module(python: &str, module: &str) -> bool {
    std::process::Command::new(python)
        .args(["-c", &format!("import {module}")])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// The interpreter to RUN a static tool with.
///
/// [VANTA-WHY] A tool must run where the tool is installed, and mypy must still
/// SEE the project's dependencies — and those are not always one interpreter.
/// heretic's venv has `ruff` but no `mypy` (its dev type checker is `ty`), while
/// Vanta's own `python3` has mypy and ruff but none of heretic's dependencies.
/// So the runner is the PROJECT interpreter when it can import the tool, and
/// Vanta's own (`VANTA_PYTHON` / `python3`) otherwise; mypy is separately
/// pointed at the project interpreter with `--python-executable`. Hard-coding
/// either one makes the step fail: the bare host python reports every dependency
/// as `import-not-found` (the 40e84874 false reject), and the bare venv cannot
/// run mypy at all and so reports NOTHING.
fn static_tool_runner(project_python: &str, module: &str) -> String {
    if has_module(project_python, module) {
        project_python.to_string()
    } else {
        python_exe()
    }
}

/// Run ruff and mypy over a source tree; parse each tool's findings into a set.
///
/// [VANTA-WHY] `test_cmd` and `config_root` exist so the static step is run the
/// way the repo is actually checked (item 10): the tools run under an
/// interpreter that has them, mypy inspects the interpreter the TESTS run under
/// (the one with the dependencies), and the repo's own config files are copied
/// in beside `src/` and `tests/`. Before this, the step ran the host's `python3`
/// against a tree with no config and no dependencies — noisy (one
/// `import-not-found` per third-party import) and blind (a missing module types
/// as `Any`, hiding real errors).
pub async fn run_static(
    source: &str,
    test_cmd: &TestCommand,
    config_root: Option<&Path>,
) -> StaticRun {
    let sb = match Backend::spawn().await {
        Ok(b) => b,
        Err(_) => return StaticRun::default(),
    };
    // [VANTA-WHY] F1 — a discarded `seed_and_write` here let ruff/mypy run over
    // the SEEDED tree rather than `source`, and their findings feed the verdict
    // (a blocking static error is a failure signal). Same treatment as the
    // test-running paths: destroy the sandbox and return the "could not run"
    // value.
    if seed_and_write(sb.root(), source).is_err() {
        sb.destroy();
        return StaticRun::default();
    }
    if let Some(root) = config_root {
        copy_static_configs(root, sb.root());
    }

    // [VANTA-WHY] The interpreter the TESTS run under — the only one with the
    // repo's dependencies. mypy is told to inspect it (`--python-executable`) so
    // a third-party import resolves to the installed package rather than
    // collapsing to `Any` (item 10).
    let project_py = interpreter_for(test_cmd);

    let ruff_py = static_tool_runner(&project_py, "ruff");
    let mut ruff_argv = vec![ruff_py.as_str()];
    ruff_argv.extend_from_slice(RUFF_ARGS);
    // [VANTA-WHY] The recorded pre-compile rule is deliberately NOT passed to the
    // static tools: it is a step for the runs of the test command whose report is
    // read as evidence, and ruff/mypy are neither. An empty slice is what a run
    // without the rule already sent.
    let (ruff, ruff_raw) = match sb
        .execute(&ruff_argv, Path::new(""), 30, &test_cmd.env, &[])
        .await
    {
        Ok(r) => {
            let raw = format!("{}\n{}", r.stdout, r.stderr);
            let findings = raw
                .lines()
                .filter_map(parse_ruff_line)
                .collect::<Vec<_>>();
            (findings, raw)
        }
        Err(_) => (Vec::new(), String::new()),
    };

    let mypy_py = static_tool_runner(&project_py, "mypy");
    let (mypy, mypy_raw) = match sb
        .execute(
            &[
                mypy_py.as_str(),
                "-m",
                "mypy",
                "--python-executable",
                project_py.as_str(),
                ".",
            ],
            Path::new(""),
            60,
            &test_cmd.env,
            &[],
        )
        .await
    {
        Ok(r) => {
            let raw = format!("{}\n{}", r.stdout, r.stderr);
            let findings = raw
                .lines()
                .filter_map(parse_mypy_line)
                .collect::<Vec<_>>();
            (findings, raw)
        }
        Err(_) => (Vec::new(), String::new()),
    };

    sb.destroy();
    StaticRun {
        ruff,
        mypy,
        ruff_raw,
        mypy_raw,
    }
}

/// Findings present in `patched` and absent from `baseline`.
///
/// [VANTA-WHY] Set difference, not a count comparison — a patch that fixes two
/// findings and adds two must report two NEW findings, not zero.
pub fn new_diagnostics<'a>(
    baseline: &[StaticDiagnostic],
    patched: &'a [StaticDiagnostic],
) -> Vec<&'a StaticDiagnostic> {
    let known: std::collections::HashSet<&StaticDiagnostic> = baseline.iter().collect();
    patched.iter().filter(|d| !known.contains(d)).collect()
}

/// Is this ruff finding an ERROR — code that cannot run or cannot resolve — or a
/// style preference?
///
/// [VANTA-WHY] A new finding in a file the patch touches may reject a review
/// round, but ruff's default ruleset is mostly STYLE (B, BLE, UP, SIM, TRY, ...).
/// Rejecting on those reports a candidate as broken when it merely has a nit,
/// which is a false finding about the model and burns an attack round. Only rules
/// that mean "this code is wrong" may block:
///   - `F`  — Pyflakes: undefined names, unreachable code, broken redefinitions;
///   - `E9` — ruff's syntax errors (`E999` unparsable).
/// Everything else is still REPORTED (the Creator sees it, the session log keeps
/// it); it just does not reject.
///
/// [VANTA-WHY] mypy needs no equivalent: its parser only accepts `: error:`
/// lines, so every finding it produces is already error-class.
pub fn ruff_is_error_class(code: &str) -> bool {
    code.starts_with('F') || code.starts_with("E9")
}

/// Is `file` a path the patch touches?
///
/// [VANTA-WHY] ruff reports paths relative to the repo root and a patch names
/// them as its headers do (`a/src/pkg/mod.py`), so neither plain equality nor a
/// suffix match alone is right. Both are tried. One definition, shared by the
/// static gate and the Diagnostician's "was the fault in a file the patch
/// touched?" probe — two copies would drift exactly as M2/M3's harnesses did.
pub fn path_touched(file: &str, touched: &[String]) -> bool {
    let norm = file.trim_start_matches("./");
    touched.iter().any(|p| {
        let p = p.trim_start_matches("./");
        p == norm || norm.ends_with(p) || p.ends_with(norm)
    })
}

/// Render findings as one newline-separated block (the Diagnostician payload).
pub fn render_diagnostics(findings: &[&StaticDiagnostic]) -> String {
    findings
        .iter()
        .map(|d| d.render())
        .collect::<Vec<_>>()
        .join("\n")
}

/// `path.py:12:5: F821 Undefined name ...` — the code is the leading token of
/// the text after the location.
fn parse_ruff_line(line: &str) -> Option<StaticDiagnostic> {
    let parts: Vec<&str> = line.splitn(4, ':').collect();
    if parts.len() < 4 {
        return None;
    }
    let file = parts[0].trim();
    if !file.ends_with(".py") || !is_line_number(parts[1]) || !is_line_number(parts[2]) {
        return None;
    }
    let rest = parts[3].trim();
    let (code, message) = match rest.split_once(char::is_whitespace) {
        Some((c, m)) => (c.to_string(), m.trim().to_string()),
        None => (rest.to_string(), String::new()),
    };
    Some(StaticDiagnostic {
        file: file.to_string(),
        line: parts[1].trim().parse::<u32>().ok(),
        code,
        message,
    })
}

/// `path.py:12: error: message  [code]` (mypy reports the line, sometimes a
/// column, then `error:`).
fn parse_mypy_line(line: &str) -> Option<StaticDiagnostic> {
    let (head, tail) = line.split_once(": error:")?;
    let mut head_parts = head.split(':');
    let file = head_parts.next()?.trim();
    if !file.ends_with(".py") {
        return None;
    }
    // mypy prints `path.py:LINE: error: …` (a column only under
    // --show-column-numbers, in which case the first number is still LINE).
    let line_no = head_parts.next().and_then(|n| n.trim().parse::<u32>().ok());
    let tail = tail.trim();
    // The bracketed error code, when present, is the final `[...]` group.
    let (message, code) = match tail.rfind('[') {
        Some(i) if tail.ends_with(']') => (
            tail[..i].trim().to_string(),
            tail[i + 1..tail.len() - 1].trim().to_string(),
        ),
        _ => (tail.to_string(), String::new()),
    };
    Some(StaticDiagnostic {
        file: file.to_string(),
        line: line_no,
        code,
        message,
    })
}

fn is_line_number(token: &str) -> bool {
    !token.trim().is_empty() && token.trim().parse::<u32>().is_ok()
}

// ============================================================================
// Command construction
// ============================================================================

/// Options that consume the NEXT argument as their value.
///
/// [VANTA-WHY] Needed to tell a positional TEST PATH from the value of a flag.
/// `python -m pytest tests` and `pytest -m slow` look identical to a reader that
/// does not know the option set: in the first the trailing token is a path to be
/// REPLACED, in the second it is a marker expression to KEEP. Getting it wrong
/// in either direction is expensive — dropping an option's value breaks the
/// command, keeping a test path re-collects the repository's own suite (the bug
/// `build_single_test_cmd` exists to fix). Only this table can separate them.
const VALUE_TAKING_OPTIONS: &[&str] = &[
    "-k", "-m", "-p", "-c", "-n", "-o", "-W", "-r",
    "--rootdir", "--ignore", "--ignore-glob", "--deselect", "--maxfail",
    "--tb", "--timeout", "--junitxml", "--junit-xml", "--basetemp", "--dist",
    "--durations", "--capture", "--override-ini", "--import-mode",
    "--log-level", "--log-file", "--log-cli-level", "--json-report-file",
];

/// Build the command that runs **only** the generated test file.
///
/// [VANTA-WHY] The user's test command still supplies the interpreter, the flags
/// and (through the backend) the cwd — but its test SELECTION is replaced, not
/// extended. Appending the generated file to a command that already named a path
/// (`python -m pytest tests`) made pytest collect the repository's own suite
/// first, and the report reader then took the first result in the report as the
/// generated test's. Every BugReproduction test was therefore graded against a
/// passing repository test: it "passed on the buggy original", so BATE scrapped
/// it, so no candidate could ever be rejected for failing to reproduce the bug.
/// The two halves — this and `generated_test_outcome` — are fixed together.
///
/// [VANTA-WHY] Every non-flag argument is dropped, because in a pytest
/// invocation a positional argument has exactly one meaning: a test selection.
/// The tempting alternative — drop the positionals that name a path *in the
/// tree* — needs the tree, and the only tree available here is the source
/// concat, which is capped at 64 KB per file and 256 KB in total. The
/// repository's own test file is regularly the thing that falls outside those
/// caps (black's `tests/test_black.py` is ~200 KB), so a membership test would
/// miss, `tests` would be kept, and the repository's suite would be
/// re-collected: the bug this function exists to fix comes straight back, and
/// this time only for the largest repositories. The rule's cost is the opposite
/// direction — an option missing from `VALUE_TAKING_OPTIONS` loses its value —
/// and that is the benign failure: a reporting flag falls back to its default,
/// instead of a test being graded against the wrong result with nobody noticing.
///
/// [VANTA-WHY] Item U — the generated test file lands at the REPO ROOT, so
/// pytest reads the repository's own ini (`setup.cfg` / `pytest.ini`). That ini
/// can set `addopts` naming an option the repository registers only from its own
/// `conftest.py` — pandas' `setup.cfg` sets `addopts = --strict-data-files`, and
/// `pandas/conftest.py` is what calls `pytest_addoption` for it. A root-level
/// `test_bate.py` does NOT collect `pandas/conftest.py`, so pytest exits with
/// `unrecognized arguments: --strict-data-files` before it can run anything: the
/// generated test never reaches a verdict and every baseline run is an
/// environment failure (§14.19). `-o addopts=` clears the repository's `addopts`
/// for this single generated test, so those repo-conftest options never reach a
/// command that does not load their conftest. It is a no-op for a repo whose
/// `addopts` is empty, and the harness's own flags (`--json-report`, `-q`) are on
/// the command line, not in `addopts`, so they survive. The repository's NATIVE
/// suite is unaffected: it runs at the repo's own paths, collects the repo's
/// conftest, and does not go through this function.
pub fn build_single_test_cmd(test_cmd: &TestCommand, test_file: &str) -> Vec<String> {
    let mut cmd = vec![test_cmd.program.clone()];
    let args = &test_cmd.args;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--" {
            // Everything after `--` is positional by definition, and positional
            // arguments to pytest ARE test selections. The marker itself only
            // exists to introduce them, so it goes too.
            break;
        }
        if VALUE_TAKING_OPTIONS.contains(&arg.as_str()) {
            cmd.push(arg.clone());
            if let Some(value) = args.get(i + 1) {
                cmd.push(value.clone());
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if arg.starts_with('-') {
            cmd.push(arg.clone());
        }
        // Anything else is a positional test selection: replaced, not extended,
        // by the generated file pushed below.
        i += 1;
    }

    cmd.push("-o".to_string());
    cmd.push("addopts=".to_string());
    cmd.push(test_file.to_string());
    cmd.push("--json-report".to_string());
    cmd.push("--json-report-file=report.json".to_string());
    cmd.push("-q".to_string());
    cmd
}

/// Python interpreter to use for test execution.
/// [VANTA-WHY] Some repos ship their own venv with heavy deps. Set
/// VANTA_PYTHON to that venv's python. Defaults to `python3` from PATH.
pub fn python_exe() -> String {
    std::env::var("VANTA_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

/// Wall-clock cap for a single test run, in seconds.
///
/// [VANTA-WHY] 120s by default: importing torch/transformers can take 10-20s, so
/// a tight cap would misreport a slow import as a timeout. `VANTA_TEST_TIMEOUT_SECS`
/// lowers it so a test can drive a REAL timeout — a hanging test actually killed
/// by the backend — without waiting two minutes. The timeout signal path
/// (`ExecutionResult.timed_out` → `TestExecutionDetails.timed_out` →
/// Infrastructure → reset-and-rerun) can only be exercised end-to-end when the
/// timeout really fires.
pub fn test_timeout_secs() -> u64 {
    std::env::var("VANTA_TEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(120)
}

/// Timeout for PATCHED-side execution. Tests may lower only this cap so the
/// real timeout path is exercised without making BATE's baseline load-sensitive.
pub fn patched_test_timeout_secs() -> u64 {
    std::env::var("VANTA_PATCHED_TEST_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(test_timeout_secs)
}

/// Detect the Python import path for a repo (Spec §7 "Import paths" row).
///
/// Rules:
///   - `src/<pkg>/...` → `<pkg>` (PEP 517 src-layout; conftest adds `src/`).
///   - `<pkg>/…` with `__init__.py` → `<pkg>` (flat layout with a package dir).
///   - files at repo root → None (import modules directly).
pub fn detect_import_path(source_files: &[SourceFile], _repo_root: &Path) -> Option<String> {
    for sf in source_files {
        if let Some(rest) = sf.path.strip_prefix("src/") {
            if let Some(pkg) = rest.split('/').next() {
                if !pkg.is_empty() {
                    return Some(pkg.to_string());
                }
            }
        }
    }
    for sf in source_files {
        let parts: Vec<&str> = sf.path.split('/').collect();
        if parts.len() >= 2 && parts[1] == "__init__.py" {
            return Some(parts[0].to_string());
        }
    }
    None
}

// ============================================================================
// Dependency self-healing
// ============================================================================

/// Probe the sandbox for Python modules pytest cannot import.
///
/// [VANTA-WHY] Manifest detection tells us what the repo DECLARED; tests can
/// import more. We ask pytest to COLLECT the generated tests inside the VM —
/// collection imports them, so a missing third-party import surfaces as
/// "No module named 'X'".
pub async fn probe_missing_modules(
    original_source: &str,
    tests: &[GeneratedTest],
    test_cmd: &TestCommand,
) -> Vec<String> {
    let sb = match Backend::spawn().await {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    if seed_and_write(sb.root(), original_source).is_err() {
        sb.destroy();
        return Vec::new();
    }
    let _ = write_conftest(sb.root());
    for (i, t) in tests.iter().enumerate() {
        let _ = fs::write(sb.root().join(format!("test_probe_{i}.py")), &t.test_source);
    }
    let mut cmd = test_cmd.as_cmd_vec();
    cmd.push("--collect-only".to_string());
    cmd.push("-q".to_string());
    let refs: Vec<&str> = cmd.iter().map(|s| s.as_str()).collect();
    let output = sb
        .execute(&refs, Path::new(""), 120, &test_cmd.env, &test_cmd.precompile)
        .await;
    sb.destroy();
    match output {
        Ok(e) => {
            let mut combined = e.stdout;
            combined.push('\n');
            combined.push_str(&e.stderr);
            crate::sandbox::deps::parse_missing_modules(&combined)
        }
        Err(_) => Vec::new(),
    }
}

/// Self-healing dependency resolution.
///
/// [VANTA-WHY] Probes the tests for undeclared imports, installs them into a
/// checkpoint DERIVED from the current one, and switches the sandbox to it.
/// Bounded to 3 rounds. Skipped when auto-deps is disabled or shuru is absent.
pub async fn heal_dependencies(
    original_source: &str,
    tests: &[GeneratedTest],
    test_cmd: &TestCommand,
    import_hint: Option<&str>,
) {
    if std::env::var("VANTA_SHURU_AUTO_DEPS").as_deref() == Ok("0") {
        return;
    }
    if !crate::sandbox::shuru_available() {
        return;
    }

    let mut parent =
        std::env::var("VANTA_SHURU_CHECKPOINT").unwrap_or_else(|_| "vanta-python".into());
    let mut parent_disk: u64 = std::env::var("VANTA_SHURU_DISK_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(crate::sandbox::deps::BASE_DISK_MB);

    for _ in 0..3 {
        let missing: Vec<String> = probe_missing_modules(original_source, tests, test_cmd)
            .await
            .into_iter()
            // A missing *repo* submodule (e.g. "heretic.foo") is a bad test,
            // not a missing dependency — never try to pip install the repo.
            .filter(|m| import_hint != Some(m.as_str()))
            .collect();
        if missing.is_empty() {
            return;
        }

        println!(
            "  ⟳ VM deps: undeclared imports detected: {}",
            missing.join(", ")
        );

        match crate::sandbox::shuru::ShuruBackend::ensure_derived_checkpoint(
            &parent,
            parent_disk,
            &missing,
        ) {
            Ok(Some((name, disk))) => {
                println!("  ⟳ VM deps: derived checkpoint {name} (disk {disk} MB)");
                parent = name;
                parent_disk = disk;
                // SAFETY: single-threaded CLI; no other thread reads env concurrently.
                unsafe {
                    std::env::set_var("VANTA_SHURU_CHECKPOINT", &parent);
                    std::env::set_var("VANTA_SHURU_DISK_MB", parent_disk.to_string());
                }
            }
            Ok(None) => return,
            Err(e) => {
                eprintln!("  ⚠ VM deps self-heal failed: {e}");
                return;
            }
        }
    }
    eprintln!("  ⚠ VM deps self-heal hit the round limit — some imports may stay missing");
}

// ============================================================================
// UNIT TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::{
        CollectionFailure, NativeSuiteReport, RUFF_ARGS, StaticDiagnostic, build_single_test_cmd,
        copy_static_configs, copy_tree_excluding, empty_report_note, interpreter_for,
        is_native_test_path, is_tooling_package, native_collection_summary, native_gate_vacuous,
        native_outcome, native_regressions, native_results_index, native_suite_collapsed,
        native_tree_with_original_tests,
        native_vacuous_reason, new_diagnostics, parse_mypy_line, parse_native_report,
        parse_pytest_report_detailed, parse_ruff_line, project_package, prune_keeps,
        report_is_measured, report_tests,
        ruff_is_error_class, run_single_test, run_tests_parallel, seed_and_write,
        seed_and_write_from, seed_method, seed_root, seed_sandbox_from, seed_source_and_conftest,
        set_seed_root,
        static_tool_runner, write_source_files, write_tree_snapshot,
    };
    use crate::agents::attacker::GeneratedTest;
    use crate::core::{TestIntent, TestResult};
    use std::collections::{HashMap, HashSet};
    use crate::pipeline::candidate::TestCommand;
    use crate::sandbox::ExecutionResult;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard};

    fn generated_test(name: &str, assertion: &str) -> GeneratedTest {
        GeneratedTest {
            test_name: name.to_string(),
            test_source: format!("from calc import add\n\ndef {name}():\n    {assertion}\n"),
            intent: TestIntent::BehaviorPreservation,
            reason: "exercise the same repo code through both harness paths".to_string(),
            issue_quote: None,
            relabeled: false,
        }
    }

    #[test]
    fn write_source_files_preserves_nested_tree_structure_and_contents() {
        let source = concat!(
            "# FILE: pkg/__init__.py\n",
            "from .core import add\n",
            "\n\n",
            "# FILE: pkg/core.py\n",
            "def add(a, b):\n",
            "    return a + b\n",
            "\n",
        );
        let dir = tempfile::tempdir().unwrap();

        write_source_files(dir.path(), source).unwrap();

        let mut paths: Vec<String> = fs::read_dir(dir.path().join("pkg"))
            .unwrap()
            .map(|entry| format!("pkg/{}", entry.unwrap().file_name().to_string_lossy()))
            .collect();
        paths.sort();

        assert_eq!(paths, vec!["pkg/__init__.py", "pkg/core.py"]);
        assert_eq!(
            fs::read_to_string(dir.path().join("pkg/__init__.py")).unwrap(),
            "from .core import add\n"
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("pkg/core.py")).unwrap(),
            "def add(a, b):\n    return a + b\n\n"
        );
    }

    // ---- M3 item T1: a write site may not leave the tree root ----------------

    /// [VANTA-WHY] M3 item T1. The `# FILE:` header carries a path the patch
    /// chose, and `write_source_files` writes it on the host. A `..` component
    /// must be refused as `unsafe_path`, and NOTHING may appear next to the tree.
    #[test]
    fn write_source_files_refuses_a_parent_escape_and_writes_nothing() {
        let base = tempfile::tempdir().unwrap();
        let root = base.path().join("root");
        fs::create_dir_all(&root).unwrap();

        let err = write_source_files(&root, "# FILE: ../escape.py\nevil = True\n")
            .expect_err("a parent escape must be refused");

        assert!(err.to_string().starts_with("unsafe_path"), "got: {err}");
        assert!(
            !base.path().join("escape.py").exists(),
            "nothing may be written outside the tree root"
        );
        assert!(
            root.read_dir().unwrap().next().is_none(),
            "nothing at all is written when the patch is refused"
        );
    }

    /// [VANTA-WHY] M3 item T1 — the same rule for an absolute path.
    #[test]
    fn write_source_files_refuses_an_absolute_path_and_writes_nothing() {
        let base = tempfile::tempdir().unwrap();
        let root = base.path().join("root");
        fs::create_dir_all(&root).unwrap();
        let outside = base.path().join("outside.py");
        let source = format!("# FILE: {}\nevil = True\n", outside.display());

        let err = write_source_files(&root, &source).expect_err("an absolute path must be refused");

        assert!(err.to_string().starts_with("unsafe_path"), "got: {err}");
        assert!(
            !outside.exists(),
            "nothing may be written outside the tree root"
        );
    }

    /// [VANTA-WHY] M3 item T1. A path that stays lexically inside the root but
    /// RESOLVES — through a symlink that already exists on disk — to somewhere
    /// outside must be refused, and the file it would have written must not
    /// exist. Only a resolve-then-contains check can catch this: every lexical
    /// check passes, because `link/evil.py` has no `..` and is not absolute.
    #[test]
    fn write_source_files_refuses_a_symlink_escape_and_writes_nothing() {
        let base = tempfile::tempdir().unwrap();
        let root = base.path().join("root");
        let outside = base.path().join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();

        let err = write_source_files(&root, "# FILE: link/evil.py\nevil = True\n")
            .expect_err("a symlink escape must be refused");

        assert!(err.to_string().starts_with("unsafe_path"), "got: {err}");
        assert!(
            !outside.join("evil.py").exists(),
            "nothing may be written through an escaping symlink"
        );
    }

    /// [VANTA-WHY] M3 item T1 — the writer CodeRabbit flagged. The snapshot
    /// writer must refuse an absolute path exactly as `write_source_files` does,
    /// and write no manifest or file for it.
    #[test]
    fn write_tree_snapshot_refuses_an_absolute_path_and_writes_nothing() {
        let base = tempfile::tempdir().unwrap();
        let root = base.path().join("snap");
        let outside = base.path().join("outside.py");
        let source = format!("# FILE: {}\nevil = True\n", outside.display());

        let err =
            write_tree_snapshot(&root, &source).expect_err("an absolute path must be refused");

        assert!(err.to_string().starts_with("unsafe_path"), "got: {err}");
        assert!(
            !outside.exists(),
            "nothing may be written outside the tree root"
        );
    }

    #[tokio::test]
    async fn baseline_and_patched_paths_match_on_identical_source() {
        let source = "# FILE: calc.py\ndef add(a, b):\n    return a + b\n";
        let test_cmd = TestCommand::default();
        let tests = vec![
            (
                "test_add_passes",
                generated_test("test_add_passes", "assert add(2, 3) == 5"),
                TestResult::Passed,
            ),
            (
                "test_add_fails",
                generated_test("test_add_fails", "assert add(2, 3) == 6"),
                TestResult::Failed,
            ),
        ];

        for (name, test, expected) in tests {
            let baseline = run_single_test(source, &test.test_source, &test_cmd).await;
            let refs = vec![&test];
            let patched = run_tests_parallel(source, &refs, &test_cmd).await.unwrap();
            let patched_result = patched.get(&test.test_name).cloned();

            assert_eq!(
                baseline, expected,
                "{name} should produce the expected baseline result"
            );
            assert_eq!(
                patched_result,
                Some(expected.clone()),
                "{name} should match between BATE and patched-run paths when the source is identical"
            );
        }
    }

    #[tokio::test]
    async fn semantic_patch_reports_assertion_failures_not_static_errors() {
        let patched_source = "# FILE: calc.py\ndef add(a, b):\n    return a - b\n";
        let test_cmd = TestCommand::default();
        let test = generated_test("test_behavior_preservation", "assert add(2, 3) == 5");
        let refs = vec![&test];
        let patched = run_tests_parallel(patched_source, &refs, &test_cmd)
            .await
            .unwrap();

        assert_eq!(
            patched.get(&test.test_name),
            Some(&TestResult::Failed),
            "a semantic behavior change should stay an assertion failure, not become Error/Static"
        );
    }

    // [VANTA-WHY] Item U — the pandas BATE baseline went `actual=Error` in ~1.2 s
    // because the generated test lands at the repo ROOT and pytest therefore reads
    // the repository's own ini: pandas' `setup.cfg` sets
    // `addopts = --strict-data-files`, and only `pandas/conftest.py` registers that
    // option. A root-level `test_bate.py` does not collect `pandas/conftest.py`, so
    // pytest exits with `unrecognized arguments: --strict-data-files` and the test
    // never reaches a verdict. This builds that exact shape — a root `setup.cfg`
    // whose `addopts` names an option that only a sub-package conftest registers —
    // and pins that the generated-test command neutralizes it so the test is graded.
    #[tokio::test]
    async fn a_repo_conftest_only_addopts_does_not_stop_the_generated_test() {
        let source = concat!(
            "# FILE: setup.cfg\n",
            "[tool:pytest]\n",
            "addopts = --strict-data-files\n",
            "\n",
            "# FILE: pkg/__init__.py\n",
            "\n",
            "\n",
            "# FILE: pkg/conftest.py\n",
            "def pytest_addoption(parser):\n",
            "    parser.addoption(\"--strict-data-files\", action=\"store_true\", default=False)\n",
            "\n",
        );
        let test = GeneratedTest {
            test_name: "test_repo_addopts_is_neutralized".to_string(),
            test_source: "def test_repo_addopts_is_neutralized():\n    assert True\n".to_string(),
            intent: TestIntent::BehaviorPreservation,
            reason: "pin that a repo-conftest-only addopts cannot stop the generated test"
                .to_string(),
            issue_quote: None,
            relabeled: false,
        };

        let result = run_single_test(source, &test.test_source, &TestCommand::default()).await;

        assert_eq!(
            result,
            TestResult::Passed,
            "a repo whose setup.cfg addopts names an option its own sub-package conftest \
             registers must still grade the generated test; an Error here is the pandas \
             `unrecognized arguments` environment failure, not a result"
        );
    }

    // ---- native-suite gate ----------------------------------------------------
    //
    // [VANTA-WHY] The heretic run went 4 tests → 0 tests and printed
    // "0 regression(s)": a tree that does not import collected no tests, and a
    // filter over an empty list finds nothing wrong with it. The walk is over the
    // ORIGINAL side, and a test that passed there and is ABSENT from the patched
    // report is a regression — that is the case these pin.

    fn native_report(tests: &[(&str, TestResult)]) -> NativeSuiteReport {
        NativeSuiteReport {
            tests: tests
                .iter()
                .map(|(name, result)| ((*name).to_string(), result.clone()))
                .collect(),
            failure_outputs: HashMap::new(),
            ran: true,
            timed_out: false,
            exit_code: Some(0),
            stderr_tail: String::new(),
            collection: None,
        }
    }

    #[test]
    fn a_suite_that_collapsed_to_zero_tests_is_four_regressions() {
        let original = native_report(&[
            ("test_a", TestResult::Passed),
            ("test_b", TestResult::Passed),
            ("test_c", TestResult::Passed),
            ("test_d", TestResult::Passed),
        ]);
        let patched = native_report(&[]);

        assert_eq!(
            native_regressions(&original.tests, &patched.tests).len(),
            4,
            "4 tests → 0 tests is four regressions, not a clean bill of health"
        );
        assert!(
            native_suite_collapsed(&original, &patched),
            "the console must be able to say the suite COLLAPSED, not just list names"
        );
    }

    #[test]
    fn a_suite_that_kept_its_tests_has_not_collapsed() {
        let original = native_report(&[("test_a", TestResult::Passed)]);
        let patched = native_report(&[("test_a", TestResult::Passed)]);

        assert!(native_regressions(&original.tests, &patched.tests).is_empty());
        assert!(!native_suite_collapsed(&original, &patched));
    }

    /// M3 item E2 — the one index must answer exactly what the per-nodeid linear
    /// scan it replaces answered, so swapping the scan for the index cannot change
    /// a verdict. Covers a present, a failing, a duplicated and an absent nodeid.
    #[test]
    fn the_native_results_index_matches_a_linear_scan() {
        let report = native_report(&[
            ("a::t", TestResult::Passed),
            ("b::t", TestResult::Failed),
            ("b::t", TestResult::Error),
            ("c::t", TestResult::Passed),
        ]);
        let index = native_results_index(&report.tests);

        for (name, _) in &report.tests {
            let linear = report.tests.iter().find(|(n, _)| n == name).map(|(_, r)| r);
            assert_eq!(
                index.get(name.as_str()).copied(),
                linear,
                "the index must agree with the scan it replaces for {name}"
            );
        }
        assert!(
            index.get("absent::t").is_none(),
            "a nodeid the report does not list must not be invented"
        );
    }

    /// M3 item U1 — only a MEASUREMENT may be cached. `not_run`, `failed_run`
    /// and `timed_out_run` all describe a suite that said nothing, so writing one
    /// would leave an entry that every later lookup serves as "no regressions".
    #[test]
    fn only_a_measurement_is_written_to_the_cache() {
        assert!(!report_is_measured(&NativeSuiteReport::not_run()));
        assert!(!report_is_measured(&NativeSuiteReport::failed_run(
            Some(1),
            "boom".to_string()
        )));
        assert!(!report_is_measured(&NativeSuiteReport::timed_out_run(
            None,
            "killed".to_string()
        )));
        assert!(
            report_is_measured(&native_report(&[("a::t", TestResult::Passed)])),
            "a report that parsed is a measurement"
        );
    }

    #[test]
    fn a_suite_that_never_ran_is_not_read_as_collapsed() {
        let original = NativeSuiteReport::not_run();
        let patched = native_report(&[]);

        assert!(
            !native_suite_collapsed(&original, &patched),
            "a gate that could not run must not be reported as a collapse — that would \
             invent a finding out of a missing measurement"
        );
    }

    // ---- the native tree is graded by the ORIGINAL tests ---------------------
    //
    // [VANTA-WHY] The 888b6931 hole: the native run collected the patch's OWN
    // edited test file, so the weakened test passed and the Gatekeeper approved a
    // patch the original test rejects. These pin that the patch's edits to
    // existing tests never reach the native tree.

    #[test]
    fn a_collected_test_file_is_recognised_by_every_name_pytest_uses() {
        for path in [
            "test_core.py",
            "tests/test_core.py",
            "pkg/tests/test_deep.py",
            "core_test.py",
            "tests/helpers.py",
            "conftest.py",
            "tests/conftest.py",
        ] {
            assert!(is_native_test_path(path), "{path} is owned by the test suite");
        }
        for path in ["calc.py", "src/heretic/plugin.py", "latest.py"] {
            assert!(
                !is_native_test_path(path),
                "{path} is not a test file and must not be pinned back"
            );
        }
    }

    #[test]
    fn a_patch_that_weakens_an_existing_test_is_graded_by_the_original_test() {
        let original = HashMap::from([
            ("calc.py".to_string(), "def scale(x):\n    return x * 2\n".to_string()),
            (
                "tests/test_core.py".to_string(),
                "def test_scale():\n    assert scale(3) == 6\n".to_string(),
            ),
        ]);
        let mut patched = original.clone();
        patched.insert(
            "calc.py".to_string(),
            "def scale(x):\n    return x * 2 + 1\n".to_string(),
        );
        patched.insert(
            "tests/test_core.py".to_string(),
            "def test_scale():\n    assert True\n".to_string(),
        );

        let tree = native_tree_with_original_tests(&original, &patched, &[]);

        assert_eq!(
            tree.files["tests/test_core.py"],
            original["tests/test_core.py"],
            "the native run must grade the patch with the ORIGINAL test, not the weakened one"
        );
        assert_eq!(
            tree.files["calc.py"], patched["calc.py"],
            "the patched source must be what the native run executes"
        );
        assert_eq!(tree.modified_existing_tests, vec!["tests/test_core.py"]);
        assert!(tree.added_test_files.is_empty());
        assert!(tree.removed_existing_tests.is_empty());
    }

    #[test]
    fn a_patch_added_conftest_and_test_file_never_reach_the_native_tree() {
        let original = HashMap::from([(
            "calc.py".to_string(),
            "def scale(x):\n    return x * 2\n".to_string(),
        )]);
        let mut patched = original.clone();
        patched.insert(
            "conftest.py".to_string(),
            "collect_ignore = ['tests/test_core.py']\n".to_string(),
        );
        patched.insert(
            "tests/test_new.py".to_string(),
            "def test_new():\n    assert True\n".to_string(),
        );

        let tree = native_tree_with_original_tests(&original, &patched, &[]);

        assert!(
            !tree.files.contains_key("conftest.py"),
            "a patch-added conftest.py can deselect the very tests that grade the patch"
        );
        assert!(
            !tree.files.contains_key("tests/test_new.py"),
            "a test file the patch added is not part of the original baseline"
        );
        assert_eq!(
            tree.added_test_files,
            vec!["conftest.py".to_string(), "tests/test_new.py".to_string()],
            "added test files are reported as info"
        );
    }

    #[test]
    fn a_deleted_test_file_is_restored_for_the_native_run() {
        let original = HashMap::from([
            ("calc.py".to_string(), "def scale(x):\n    return x * 2\n".to_string()),
            (
                "tests/test_core.py".to_string(),
                "def test_scale():\n    assert scale(3) == 6\n".to_string(),
            ),
        ]);
        // The cascade leaves a deleted file at its original contents, so the
        // deletion is only visible in the diff.
        let patched = original.clone();
        let deleted = vec!["tests/test_core.py".to_string()];

        let tree = native_tree_with_original_tests(&original, &patched, &deleted);

        assert_eq!(
            tree.files["tests/test_core.py"], original["tests/test_core.py"],
            "a deleted test must still run against the patched source"
        );
        assert_eq!(tree.removed_existing_tests, vec!["tests/test_core.py"]);
    }

    // ---- subTest parsing (item 9) --------------------------------------------
    //
    // [VANTA-WHY] pytest-json-report reports a `unittest.TestCase.subTest` item
    // with outcome `"subtests passed"`. The old parser matched only
    // `"passed"/"failed"/"error"` and silently dropped everything else, so the
    // heretic suite's three subTest tests vanished: Vanta counted 4 where pytest
    // ran 5 on master, and 12 where pytest ran 15 on the C patch. The undercount
    // was not cosmetic — the dropped item was missing from the ORIGINAL side
    // `native_regressions` walks, so a patch that broke it could never be
    // reported. These pin that the item is counted.

    #[test]
    fn a_passing_subtest_item_is_counted_as_passed() {
        assert_eq!(
            native_outcome("subtests passed"),
            Some(TestResult::Passed),
            "pytest reports a passing subTest item as \"subtests passed\", not \"passed\""
        );
        assert_eq!(native_outcome("failed"), Some(TestResult::Failed));
        assert_eq!(native_outcome("error"), Some(TestResult::Failed));
        assert_eq!(
            native_outcome("skipped"),
            None,
            "a skip is neither a pass nor a failure and must not be counted as either"
        );
    }

    #[test]
    fn parse_native_report_no_longer_drops_subtest_items() {
        let dir = tempfile::tempdir().unwrap();
        let report_path = dir.path().join("report.json");
        fs::write(
            &report_path,
            r#"{"tests":[
                {"nodeid":"tests/test_config.py::C::test_plain","outcome":"passed"},
                {"nodeid":"tests/test_config.py::C::test_subtests","outcome":"subtests passed"},
                {"nodeid":"tests/test_config.py::C::test_broken","outcome":"failed"},
                {"nodeid":"tests/test_config.py::C::test_skipped","outcome":"skipped"}
            ]}"#,
        )
        .unwrap();
        let exec = ExecutionResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(1),
            wall_time_ms: 0,
            timed_out: false,
        };

        let report = parse_native_report(&report_path, &exec);

        assert!(report.ran, "a readable report means the suite ran");
        assert_eq!(
            report.tests.len(),
            3,
            "the subTest item must be counted; a skipped test must not"
        );
        assert_eq!(
            report
                .tests
                .iter()
                .find(|(name, _)| name.ends_with("test_subtests"))
                .map(|(_, result)| result.clone()),
            Some(TestResult::Passed),
            "the passing subTest item must survive parsing as a Passed test"
        );
    }

    // ---- M3 item W: an empty native report is not a clean suite --------------

    /// A report of a recognised shape whose `tests` array is empty must read as
    /// a FAILED run, never as `ran: true` with zero tests, and the report's own
    /// diagnostics must travel with it.
    ///
    /// [VANTA-WHY] This is the false accept that approved a patch on nothing
    /// (pandas/1 session c9289597). pytest writes a structurally valid
    /// `--json-report` for a collection error (exit 2), a session that collected
    /// nothing (exit 5) and a run that died after collection; in all three the
    /// `tests` array is empty, so emptiness must not read as a clean bill of
    /// health.
    ///
    /// [VANTA-WHY] M3 item W follow-up — the fixture is a REAL collection error
    /// (`collectors` with one `outcome: "failed"` entry), not a synthetic
    /// top-level `errors` array: `errors` is not in pytest-json-report's schema,
    /// so asserting on it proved the note rendered a field that can never carry
    /// information. pytest's pointer at the offending module is the failed
    /// collector, and that is what the note must name.
    #[test]
    fn an_empty_native_report_is_a_failed_run_never_a_clean_suite() {
        let dir = tempfile::tempdir().unwrap();
        let report_path = dir.path().join("report.json");
        // The real shape: the session collected 88592 items, then a module
        // failed to import and pytest refused to run any test (exit 2).
        fs::write(
            &report_path,
            r#"{"exitcode":2,"summary":{"collected":88592},
                "collectors":[
                    {"nodeid":"pandas/tests/io/sas","outcome":"passed","result":[]},
                    {"nodeid":"pandas/tests/io/test_x.py","outcome":"failed",
                     "result":[],
                     "longrepr":"ImportError while importing test module 'pandas/tests/io/test_x.py'.\nHint: make sure your test modules have valid Python names.\nE   ImportError: cannot import name 'X' from 'pandas.io'"},
                    {"nodeid":"pandas/tests/io/test_y.py","outcome":"skipped",
                     "result":[],"longrepr":"Skipped: could not import 'pyarrow'"}
                ],
                "tests":[]}"#,
        )
        .unwrap();
        let exec = ExecutionResult {
            stdout: String::new(),
            stderr: "ERROR collecting pandas/tests".to_string(),
            exit_code: Some(2),
            wall_time_ms: 0,
            timed_out: false,
        };

        let report = parse_native_report(&report_path, &exec);

        assert!(!report.ran, "a report with no tests is a FAILED run, not a clean one");
        assert!(report.tests.is_empty());
        assert_eq!(
            report.exit_code,
            Some(2),
            "the exit code must survive so a caller can report WHY the suite failed"
        );
        for needle in [
            "exitcode=2",
            "summary.collected=88592",
            "collectors=3",
            "failed_collectors=1",
            "first_failed_collector=pandas/tests/io/test_x.py",
            "first_collector_error=E   ImportError: cannot import name 'X' from 'pandas.io'",
        ] {
            assert!(
                report.stderr_tail.contains(needle),
                "the note must carry the report's own diagnostics; missing {needle:?} in {:?}",
                report.stderr_tail
            );
        }
        let census = report
            .collection
            .as_ref()
            .expect("the failed collector must reach the structured census");
        assert_eq!(census.failed, 1, "a skipped optional import is not a failure");
        assert_eq!(
            census.first.as_ref().map(|(nodeid, _)| nodeid.as_str()),
            Some("pandas/tests/io/test_x.py"),
            "the FIRST failed collector in document order is the one named"
        );
    }

    /// A collection failure is only named when a collector actually failed: a
    /// clean census reports `failed_collectors=0` and a document with no
    /// `collectors` array at all reports `failed_collectors=unset`, so neither
    /// can be mistaken for "there was a failed module".
    #[test]
    fn a_failed_collector_is_named_only_when_one_exists() {
        let clean = serde_json::json!({
            "exitcode": 5,
            "summary": {"collected": 0},
            "collectors": [{"nodeid": "a", "outcome": "passed"}],
            "tests": []
        });
        let note = empty_report_note(&clean, "");
        assert!(note.contains("collectors=1"), "{note}");
        assert!(note.contains("failed_collectors=0"), "{note}");
        assert!(
            !note.contains("first_failed_collector"),
            "nothing failed, so no module may be named: {note}"
        );

        let unreadable = serde_json::json!({"tests": []});
        let note = empty_report_note(&unreadable, "");
        assert!(
            note.contains("collectors=unset") && note.contains("failed_collectors=unset"),
            "a document with no census must render unset, never an invented zero: {note}"
        );
    }

    /// `native_collection_summary` is the machine-readable half: both sides'
    /// counts plus the first failing module's nodeid and error, whitespace-free
    /// so `parse_summary_fields`' whitespace split cannot truncate a field or
    /// read an interior token of the error as a bogus `key=value`.
    #[test]
    fn native_collection_summary_names_the_failing_module_for_both_sides() {
        let mut original = vacuity_report(&[], false, Some(2));
        original.collection = Some(CollectionFailure {
            failed: 2,
            first: Some((
                "pandas/tests/io/test_x.py".to_string(),
                "ImportError while importing test module 'pandas/tests/io/test_x.py'.".to_string(),
            )),
        });
        let patched = vacuity_report(&["a"], true, Some(1));

        let summary = native_collection_summary(&original, &patched);
        assert!(
            summary.contains("original_failed_collectors=2"),
            "{summary}"
        );
        assert!(
            summary.contains("original_first_failed_collector=pandas/tests/io/test_x.py"),
            "{summary}"
        );
        assert!(
            summary.contains(
                "original_first_collector_error=ImportError_while_importing_test_module_\
                 'pandas/tests/io/test_x.py'."
            ),
            "the error must be one token: {summary}"
        );
        assert!(
            summary.contains("patched_failed_collectors=0"),
            "a clean side reports 0 and names no module: {summary}"
        );
        assert!(
            summary.contains("patched_first_failed_collector=none")
                && summary.contains("patched_first_collector_error=none"),
            "{summary}"
        );
        let tokens: Vec<&str> = summary.split_whitespace().collect();
        assert_eq!(
            tokens.len(),
            6,
            "six `k=v` fields, no token split off by an interior space: {summary}"
        );
        for token in tokens {
            assert_eq!(
                token.matches('=').count(),
                1,
                "every token is exactly one `key=value`, so no interior token can be \
                 misread as a bogus field: {token:?} in {summary}"
            );
        }
    }

    /// The plugin's other schema — a top-level array — must be refused exactly
    /// the same way when it carries no tests.
    #[test]
    fn an_empty_top_level_array_report_is_a_failed_run_too() {
        let dir = tempfile::tempdir().unwrap();
        let report_path = dir.path().join("report.json");
        fs::write(&report_path, "[]").unwrap();
        let exec = ExecutionResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(5),
            wall_time_ms: 0,
            timed_out: false,
        };

        let report = parse_native_report(&report_path, &exec);

        assert!(!report.ran, "an empty array is not a populated suite");
        assert_eq!(report.exit_code, Some(5));
        assert!(
            report.stderr_tail.contains("collected 0 tests"),
            "the note must state the emptiness: {:?}",
            report.stderr_tail
        );
    }

    /// `empty_report_note` renders `unset` for every field the document does not
    /// carry, so a shape this reader did not anticipate cannot look clean either.
    #[test]
    fn empty_report_note_marks_missing_fields_unset() {
        let note = empty_report_note(&serde_json::json!({}), "");
        for needle in [
            "exitcode=unset",
            "summary.collected=unset",
            "collectors=unset",
            "failed_collectors=unset",
        ] {
            assert!(note.contains(needle), "missing {needle:?} in {note:?}");
        }
        assert!(!note.contains(" | "), "no stderr tail means no separator: {note:?}");
    }

    /// A report with the given nodeids and run state, for the gate tests.
    fn vacuity_report(tests: &[&str], ran: bool, exit_code: Option<i32>) -> NativeSuiteReport {
        NativeSuiteReport {
            tests: tests
                .iter()
                .map(|name| (name.to_string(), TestResult::Passed))
                .collect(),
            failure_outputs: HashMap::new(),
            ran,
            timed_out: false,
            exit_code,
            stderr_tail: String::new(),
            collection: None,
        }
    }

    /// The guard: with a POSITIVE expected count, either side measuring nothing
    /// makes the gate vacuous; with zero the bug genuinely has no native tests
    /// and the gate is legitimately silent.
    ///
    /// [VANTA-WHY] `native_regressions` over an empty original is vacuously empty
    /// and `native_suite_collapsed` requires a non-empty original, so `(0, 0)`
    /// cleared both holes and read as "0 regressions" (pandas/1 c9289597). This
    /// is the only thing that tells "no tests to check" apart from "the suite
    /// could not run".
    #[test]
    fn native_gate_vacuous_fires_only_when_prepare_measured_tests() {
        let full = vacuity_report(&["a", "b"], true, Some(1));
        let empty = vacuity_report(&[], true, Some(5));
        let not_run = vacuity_report(&[], false, None);

        assert!(
            native_gate_vacuous(88642, &full, &empty),
            "a patch side that collected nothing must not read as 0 regressions"
        );
        assert!(
            native_gate_vacuous(88642, &empty, &full),
            "an original side that collected nothing is just as vacuous"
        );
        assert!(
            native_gate_vacuous(88642, &not_run, &not_run),
            "neither side running is the worst case, not a clean gate"
        );
        assert!(
            native_gate_vacuous(88642, &empty, &empty),
            "0 vs 0 is exactly the false accept the guard exists for"
        );
        assert!(
            !native_gate_vacuous(88642, &full, &full),
            "both sides measured: this is a real gate, not a vacuous one"
        );
        assert!(
            !native_gate_vacuous(0, &empty, &empty),
            "a bug with no native tests must not be forced into infra_failure"
        );
        assert!(
            !native_gate_vacuous(0, &not_run, &not_run),
            "a hand-run binary records no count and keeps the old reading"
        );
    }

    /// The reason names the failing side and its exit code, so the reader can
    /// tell "the original never ran" from "the patch collected nothing".
    ///
    /// [VANTA-WHY] M3 item W follow-up — the reason also has to carry WHICH
    /// module broke collection. `empty_report_note` renders the collector census
    /// into the report's `stderr_tail`, and the reason prints that tail, so the
    /// module name reaches the operator here (and, tokenised, in
    /// `native_suite.summary`) instead of forcing a second sandbox boot.
    #[test]
    fn native_vacuous_reason_names_the_side_and_its_exit_code() {
        let full = vacuity_report(&["a", "b"], true, Some(0));
        let mut empty = vacuity_report(&[], true, Some(5));
        empty.stderr_tail = "shuru: booting VM | native report collected 0 tests (report \
             exitcode=2 summary.collected=88592 collectors=2620 failed_collectors=1 \
             first_failed_collector=pandas/tests/io/test_x.py \
             first_collector_error=ImportError while importing test module)"
            .to_string();

        let reason = native_vacuous_reason(88642, 120, &full, &empty);

        assert!(reason.contains("native_vacuous=true"), "{reason}");
        assert!(reason.contains("expected_tests=88642"), "{reason}");
        assert!(
            reason.contains("original=ran (tests=2 exit_code=0"),
            "the measuring side must be reported too: {reason}"
        );
        assert!(
            reason.contains("patched=ran (tests=0 exit_code=5"),
            "the empty side and its exit code must be named: {reason}"
        );
        assert!(
            reason.contains("failed_collectors=1")
                && reason.contains("first_failed_collector=pandas/tests/io/test_x.py")
                && reason.contains("first_collector_error=ImportError while importing test module"),
            "the reason must name the module that broke collection: {reason}"
        );
    }

    /// [VANTA-WHY] M3 item D1 — a run the backend KILLED is `timed_out`, not a
    /// plain failed run. pandas/1's in-session suite died at the 120 s cap on BOTH
    /// sides (`shuru exec TIMEOUT after 120s`, exit code unset), which the old
    /// reader recorded as "collected nothing" — the same shape as a collection
    /// error, and the wrong file for the operator to open.
    #[test]
    fn a_killed_native_run_is_recorded_as_timed_out_not_failed() {
        let killed = ExecutionResult {
            stdout: String::new(),
            stderr: "shuru exec TIMEOUT after 120s\n".to_string(),
            exit_code: None,
            wall_time_ms: 120_000,
            timed_out: true,
        };
        let report = parse_native_report(std::path::Path::new("/nonexistent/report.json"), &killed);
        assert!(
            report.timed_out,
            "the kill must be recorded, not read as an empty suite"
        );
        assert!(!report.ran, "a killed run produced no report");
        assert!(report.collected_nothing(), "a killed run is still vacuous");

        // An unreadable report that was NOT killed must not acquire the flag.
        let plain = ExecutionResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(2),
            wall_time_ms: 1,
            timed_out: false,
        };
        let report = parse_native_report(std::path::Path::new("/nonexistent/report.json"), &plain);
        assert!(
            !report.timed_out,
            "an unreadable report is a failed run, not a timeout"
        );
    }

    /// [VANTA-WHY] M3 item D1 — a vacuous gate caused by a KILLED side reports
    /// `native_timeout` with the cap named, not `native_vacuous`. The two send the
    /// operator to opposite places — the cap (a harness fact) vs the tree (an
    /// import error) — and the guard's force is unchanged either way: a killed
    /// side is still vacuous and still ends the run as `InfraFailure`.
    #[test]
    fn native_vacuous_reason_names_a_timeout_as_its_own_reason() {
        let full = vacuity_report(&["a", "b"], true, Some(1));
        let mut killed = vacuity_report(&[], false, None);
        killed.timed_out = true;
        killed.stderr_tail = "shuru exec TIMEOUT after 120s".to_string();

        let reason = native_vacuous_reason(88642, 1461, &full, &killed);

        assert!(reason.contains("native_timeout=true"), "{reason}");
        assert!(
            !reason.contains("native_vacuous=true"),
            "a timeout is its own reason, not the generic one: {reason}"
        );
        assert!(
            reason.contains("native_timeout_secs=1461"),
            "the cap the run used must be named: {reason}"
        );
        assert!(reason.contains("the patch side"), "{reason}");
        assert!(
            reason.contains("patched=not_run timed_out (tests=0 exit_code=unset"),
            "the killed side and its state must be reported: {reason}"
        );
        assert!(
            native_gate_vacuous(88642, &full, &killed),
            "the guard's force is unchanged: a killed side is still vacuous"
        );
    }

    // ---- static findings: error-class only ------------------------------------

    #[test]
    fn only_error_class_ruff_codes_may_block_a_review_round() {
        assert!(ruff_is_error_class("F821"), "undefined name is code that cannot run");
        assert!(ruff_is_error_class("E999"), "ruff's own syntax error");
        assert!(!ruff_is_error_class("E401"), "import hygiene is a style nit");
        assert!(!ruff_is_error_class("B008"), "flake8-bugbear style");
        assert!(!ruff_is_error_class("BLE001"), "broad-except style");
        assert!(!ruff_is_error_class("UP032"), "pyupgrade style");
        assert!(!ruff_is_error_class("SIM108"), "simplify style");
        assert!(!ruff_is_error_class("TRY003"), "tryceratops style");
    }

    /// A new static finding must carry its exact site for the evidence record,
    /// yet that site must NOT be part of the finding's identity — otherwise a
    /// patch that merely shifts a line would read as introducing a fresh error.
    #[test]
    fn a_new_static_finding_keeps_its_site_but_not_as_its_identity() {
        let ruff = parse_ruff_line("pkg/core.py:12:5: F821 Undefined name `x`").unwrap();
        assert_eq!(ruff.line, Some(12), "the reported line must be retained");
        assert_eq!(ruff.file, "pkg/core.py");
        assert_eq!(ruff.code, "F821");

        let mypy = parse_mypy_line("pkg/core.py:88: error: bad return  [return-value]").unwrap();
        assert_eq!(mypy.line, Some(88), "mypy's line must be retained");
        assert_eq!(mypy.code, "return-value");

        let baseline = vec![ruff.clone()];
        let moved = StaticDiagnostic {
            line: Some(40),
            ..ruff.clone()
        };
        assert!(
            new_diagnostics(&baseline, &[moved]).is_empty(),
            "the same finding moved down the file is not a NEW finding"
        );

        let different = StaticDiagnostic {
            code: "F841".to_string(),
            ..ruff.clone()
        };
        assert_eq!(
            new_diagnostics(&baseline, &[different]).len(),
            1,
            "a genuinely different finding in the same file is still new"
        );
    }

    /// The ruff invocation must ask for the concise report.
    ///
    /// [VANTA-WHY] Item 10 follow-up: ruff's default "full" report is
    /// multi-line, so `parse_ruff_line` read ZERO findings from every run no
    /// matter what the tree contained. The flag is what makes the parser's
    /// contract hold, so its absence is a silent blindness bug, not a style.
    #[test]
    fn ruff_is_asked_for_the_report_the_parser_can_read() {
        assert!(
            RUFF_ARGS.contains(&"--output-format=concise"),
            "without concise output every ruff finding parses to nothing"
        );

        let concise = "pkg/core.py:3:5: F821 Undefined name `x`";
        assert_eq!(
            parse_ruff_line(concise).map(|d| d.code),
            Some("F821".to_string()),
            "the concise line is exactly the parser's input"
        );

        let full = "F821 Undefined name `x`\n  --> pkg/core.py:3:5\n   |\n 3 | x = x\n";
        assert!(
            full.lines().filter_map(parse_ruff_line).next().is_none(),
            "the default full report carries no parsable one-line finding"
        );
    }

    // ---- item 10: the static step must run like the repo does ---------------

    /// The static tools must run under the interpreter the TESTS run under —
    /// that is the only one with the repo's dependencies. Session 40e84874 was
    /// rejected because mypy ran under the host's `python3` and reported
    /// `import torch` as `import-not-found`.
    #[test]
    fn the_static_interpreter_follows_the_test_command() {
        let heretic = TestCommand::from_shell(
            "env PYTHONPATH=src /Users/kyu-yi/Documents/Trae/default/heretic/.venv/bin/python -m pytest -q",
        );
        assert_eq!(
            interpreter_for(&heretic),
            "/Users/kyu-yi/Documents/Trae/default/heretic/.venv/bin/python",
            "the venv python hides behind `env` and a `PYTHONPATH=src` assignment, \
             not behind the host's bare python3"
        );

        assert_eq!(
            interpreter_for(&TestCommand::from_shell("python3 -m pytest")),
            "python3"
        );

        // A command that names no interpreter at all must fall back rather than
        // pick a token out of thin air (`pytest` is not an interpreter).
        assert_eq!(
            interpreter_for(&TestCommand::from_shell("pytest -q")),
            super::python_exe()
        );
    }

    /// The tool must RUN where it is installed, which is not always where the
    /// project lives: heretic's venv has `ruff` but no `mypy`.
    #[test]
    fn a_static_tool_runs_where_it_is_installed() {
        let host = super::python_exe();
        // Every CPython can import `sys`, so a real interpreter is used as-is.
        assert_eq!(static_tool_runner(&host, "sys"), host.clone());
        // An interpreter that cannot import the tool must not be used to run it,
        // or mypy would fail with `No module named mypy` and report NOTHING.
        assert_eq!(
            static_tool_runner("/nonexistent/venv/bin/python", "mypy"),
            host
        );
    }

    /// Only the config files the repo actually has may be copied — inventing an
    /// empty `ruff.toml` would CHANGE ruff's behaviour instead of reproducing it.
    #[test]
    fn only_the_configs_that_exist_reach_the_sandbox() {
        let repo = tempfile::tempdir().unwrap();
        let sandbox = tempfile::tempdir().unwrap();
        fs::write(
            repo.path().join("pyproject.toml"),
            "[tool.ruff.lint]\nignore = [\"B023\"]\n",
        )
        .unwrap();
        fs::write(repo.path().join("ruff.toml"), "line-length = 100\n").unwrap();

        let mut copied = copy_static_configs(repo.path(), sandbox.path());
        copied.sort();

        assert_eq!(
            copied,
            vec!["pyproject.toml".to_string(), "ruff.toml".to_string()]
        );
        assert_eq!(
            fs::read_to_string(sandbox.path().join("pyproject.toml")).unwrap(),
            "[tool.ruff.lint]\nignore = [\"B023\"]\n",
            "the repo's own ruff config must reach the sandbox verbatim"
        );
        assert!(
            !sandbox.path().join("mypy.ini").exists(),
            "a config the repo does not have must not be invented"
        );
    }

    /// `import-not-found` / `import-untyped` are facts about the analysis
    /// environment, not defects in the candidate.
    #[test]
    fn a_missing_dependency_is_an_environment_finding() {
        let line = "tests/test_auto_rank.py:5: error: Cannot find implementation or library \
                    stub for module named \"torch\"  [import-not-found]";
        let missing = parse_mypy_line(line).expect("a mypy error line parses");
        assert_eq!(missing.code, "import-not-found");
        assert_eq!(missing.file, "tests/test_auto_rank.py");
        assert_eq!(missing.line, Some(5));
        assert!(missing.is_environment_finding());

        let untyped = StaticDiagnostic {
            file: "src/heretic/data.py".to_string(),
            line: Some(1),
            code: "import-untyped".to_string(),
            message: "Library stubs not installed for \"requests\"".to_string(),
        };
        assert!(untyped.is_environment_finding());

        let real = StaticDiagnostic {
            file: "src/heretic/modifiers/abliteration.py".to_string(),
            line: Some(440),
            code: "assignment".to_string(),
            message: "Incompatible types in assignment".to_string(),
        };
        assert!(
            !real.is_environment_finding(),
            "a genuine type error is a finding about the code"
        );
    }

    // ---- the generated test is graded by ITS OWN nodeid -----------------------
    //
    // [VANTA-WHY] The grading bug. A `--test-cmd` that named a test path
    // (`python -m pytest tests`) made the one-test run collect the repository's
    // own suite FIRST, and the reader took `report["tests"].first()` as the
    // generated test's result. That first entry was a repo test — which passes —
    // so every BugReproduction test was recorded as "passed on the buggy
    // original", BATE scrapped it, and no candidate could ever be rejected for
    // failing to reproduce the bug. These pin both halves of the fix: the run
    // REPLACES the selection with the generated file, and the reader selects only
    // that file's result and fails closed when it is absent.

    fn buggy_exec() -> ExecutionResult {
        ExecutionResult {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(1),
            wall_time_ms: 0,
            timed_out: false,
        }
    }

    fn written_report(tests_json: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");
        fs::write(&path, format!("{{\"tests\":[{tests_json}]}}")).unwrap();
        (dir, path)
    }

    #[test]
    fn a_report_where_repo_tests_come_first_grades_the_generated_test() {
        let (_dir, path) = written_report(concat!(
            r#"{"nodeid":"tests/test_core.py::test_a","outcome":"passed"},"#,
            r#"{"nodeid":"tests/test_core.py::test_b","outcome":"passed"},"#,
            r#"{"nodeid":"test_bate.py::test_reproduces_bug","outcome":"failed"}"#
        ));

        let details = parse_pytest_report_detailed(path, &buggy_exec(), None, "test_bate.py");

        assert_eq!(
            details.result,
            TestResult::Failed,
            "the generated test FAILED on the buggy original; the passing repo test that \
             happened to come first must not be read as its result"
        );
        assert!(
            details.report_path_exists,
            "a graded test reached a verdict, even though it failed"
        );
    }

    #[test]
    fn a_report_missing_the_generated_nodeid_fails_closed() {
        let (_dir, path) = written_report(concat!(
            r#"{"nodeid":"tests/test_core.py::test_a","outcome":"passed"},"#,
            r#"{"nodeid":"tests/test_core.py::test_b","outcome":"passed"}"#
        ));

        let details = parse_pytest_report_detailed(path, &buggy_exec(), None, "test_bate.py");

        assert_eq!(
            details.result,
            TestResult::Error,
            "a report without the generated nodeid is a harness error, never a Passed"
        );
        assert!(
            !details.report_path_exists,
            "the generated test was never graded, so this tree did not reach a verdict"
        );
    }

    #[test]
    fn a_generated_file_named_with_a_leading_dot_slash_still_matches() {
        let (_dir, path) =
            written_report(r#"{"nodeid":"./test_run.py::test_x","outcome":"passed"}"#);

        let details = parse_pytest_report_detailed(path, &buggy_exec(), None, "test_run.py");

        assert_eq!(
            details.result,
            TestResult::Passed,
            "rootdir-relative nodeids may carry a `./` prefix and must still match"
        );
    }

    // ---- both report schemas, in both readers --------------------------------
    //
    // [VANTA-WHY] Which schema pytest-json-report writes is a function of its
    // version, and the version follows the bug's own pytest pin
    // (plans/bugsinpy.md §3.2): 0.1/0.2 write a bare JSON list, 0.3 and later
    // write {"tests": [...]}. `httpie/1` pins `pytest==3.2.1`, which caps the
    // plugin at 0.2 — so the BugSinPy layer for that one bug writes the flat
    // list, and both readers here (the native suite and the generated test)
    // would have seen "no tests" rather than a verdict: an InfraFailure at
    // evaluation time for a reason that has nothing to do with the model.
    // Three cases are pinned: each schema, and a top level that is neither.

    fn written_flat_report(tests_json: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");
        fs::write(&path, format!("[{tests_json}]")).unwrap();
        (dir, path)
    }

    #[test]
    fn a_flat_list_report_is_read_by_both_readers() {
        let entries = concat!(
            r#"{"nodeid":"tests/test_core.py::test_a","outcome":"passed"},"#,
            r#"{"nodeid":"test_bate.py::test_reproduces_bug","outcome":"failed"}"#
        );
        let (_dir, path) = written_flat_report(entries);

        let native = parse_native_report(&path, &buggy_exec());
        assert!(native.ran, "a flat-list report is a report the suite produced");
        assert_eq!(
            native.tests.len(),
            2,
            "the 0.2 plugin's flat list carries the same per-test entries"
        );
        assert_eq!(
            native
                .tests
                .iter()
                .find(|(name, _)| name.ends_with("test_a"))
                .map(|(_, result)| result.clone()),
            Some(TestResult::Passed)
        );

        let details = parse_pytest_report_detailed(path, &buggy_exec(), None, "test_bate.py");
        assert_eq!(
            details.result,
            TestResult::Failed,
            "the generated test still fails on the buggy original under the 0.2 schema"
        );
        assert!(
            details.report_path_exists,
            "a graded test reached a verdict, so the shape was read"
        );
    }

    #[test]
    fn a_mapping_report_is_read_by_both_readers() {
        let entries = concat!(
            r#"{"nodeid":"tests/test_core.py::test_a","outcome":"passed"},"#,
            r#"{"nodeid":"test_bate.py::test_reproduces_bug","outcome":"failed"}"#
        );
        let (_dir, path) = written_report(entries);

        let native = parse_native_report(&path, &buggy_exec());
        assert!(native.ran);
        assert_eq!(native.tests.len(), 2, "0.3+ puts the same entries under `tests`");

        let details = parse_pytest_report_detailed(path, &buggy_exec(), None, "test_bate.py");
        assert_eq!(details.result, TestResult::Failed);
        assert!(details.report_path_exists);
    }

    #[test]
    fn a_report_of_an_unexpected_shape_is_not_an_empty_report() {
        let dir = tempfile::tempdir().unwrap();
        let string_report = dir.path().join("string.json");
        fs::write(&string_report, r#""passed""#).unwrap();
        let keyless_mapping = dir.path().join("keyless.json");
        fs::write(&keyless_mapping, r#"{"created":0.0}"#).unwrap();

        for path in [&string_report, &keyless_mapping] {
            let native = parse_native_report(path, &buggy_exec());
            assert!(
                !native.ran,
                "a report of neither schema is a run that produced no usable \
                 report, not a tree with zero tests: {path:?}"
            );

            let details = parse_pytest_report_detailed(
                path.clone(),
                &buggy_exec(),
                None,
                "test_bate.py",
            );
            assert_eq!(
                details.result,
                TestResult::Error,
                "an unreadable report can never grade a generated test: {path:?}"
            );
            assert!(!details.report_path_exists);
        }
    }

    #[test]
    fn report_tests_rejects_a_shape_neither_plugin_writes() {
        assert!(report_tests(&serde_json::json!([])).unwrap().is_empty());
        assert_eq!(
            report_tests(&serde_json::json!({"tests": [{"nodeid": "a"}]}))
                .unwrap()
                .len(),
            1
        );
        assert!(report_tests(&serde_json::json!("passed")).is_err());
        assert!(report_tests(&serde_json::json!({"created": 0.0})).is_err());
    }

    #[test]
    fn a_test_cmd_that_names_a_path_has_it_replaced_not_extended() {
        let cmd = build_single_test_cmd(
            &TestCommand::from_shell("python3 -m pytest tests -q"),
            "test_bate.py",
        );

        assert!(
            !cmd.iter().any(|arg| arg == "tests"),
            "the repository's own test path must be REPLACED, not extended: {cmd:?}"
        );
        assert_eq!(
            cmd,
            vec![
                "python3",
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "test_bate.py",
                "--json-report",
                "--json-report-file=report.json",
                "-q",
            ],
            "the interpreter and flags are kept; only the test selection is replaced"
        );
    }

    #[test]
    fn a_value_taking_flag_keeps_its_value_and_is_not_read_as_a_path() {
        let cmd = build_single_test_cmd(
            &TestCommand::from_shell("python3 -m pytest -k test_add -m slow tests"),
            "test_bate.py",
        );

        assert!(
            cmd.windows(2).any(|w| w == ["-k", "test_add"]),
            "`-k`'s value is a marker expression, not a path to drop: {cmd:?}"
        );
        assert!(
            cmd.windows(2).any(|w| w == ["-m", "slow"]),
            "`-m`'s value is a marker expression, not a path to drop: {cmd:?}"
        );
        assert!(
            !cmd.iter().any(|arg| arg == "tests"),
            "the trailing path is still a selection and must be replaced: {cmd:?}"
        );
        assert!(cmd.iter().any(|arg| arg == "test_bate.py"), "{cmd:?}");
    }

    #[test]
    fn a_test_cmd_without_a_path_keeps_its_flags_and_gains_the_generated_file() {
        let cmd = build_single_test_cmd(&TestCommand::default(), "test_run.py");

        assert_eq!(
            cmd,
            vec![
                "python3",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "test_run.py",
                "--json-report",
                "--json-report-file=report.json",
                "-q",
            ],
            "with no selection in the command the behaviour is unchanged, minus the append"
        );
    }

    // ------------------------------------------------------------------
    // Item P — the per-run sandbox SEED and the in-session import guard
    // ------------------------------------------------------------------

    /// A small "built tree" on disk: the sources, the compiled extension and the
    /// pytest config a compiled project has after `build_ext --inplace`, plus the
    /// paths the seed stage must drop.
    fn built_tree(dir: &std::path::Path) {
        fs::create_dir_all(dir.join("pkg")).unwrap();
        fs::write(dir.join("pkg/__init__.py"), "VALUE = 1\n").unwrap();
        fs::write(dir.join("pkg/_ext.so"), [0u8, 1, 2, 3]).unwrap();
        fs::write(dir.join("pytest.ini"), "[pytest]\nminversion=3.6\n").unwrap();
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(dir.join(".git/config"), "[core]\n").unwrap();
        fs::create_dir_all(dir.join("build")).unwrap();
        fs::write(dir.join("build/out.o"), "obj").unwrap();
        std::os::unix::fs::symlink(dir.join("pkg"), dir.join("link")).unwrap();
    }

    #[test]
    fn the_seed_stage_drops_git_build_output_and_symlinks() {
        let exec = tempfile::tempdir().unwrap();
        built_tree(exec.path());
        let seed = tempfile::tempdir().unwrap();

        copy_tree_excluding(seed.path(), exec.path()).unwrap();

        assert!(seed.path().join("pkg/__init__.py").is_file());
        assert!(
            seed.path().join("pkg/_ext.so").is_file(),
            "a compiled extension is exactly what the source walk cannot carry"
        );
        assert!(
            seed.path().join("pytest.ini").is_file(),
            "the project's config must survive staging"
        );
        assert!(!seed.path().join(".git").exists(), ".git is a SKIP_DIR");
        assert!(!seed.path().join("build").exists(), "build output is a SKIP_DIR");
        assert!(
            !seed.path().join("link").exists(),
            "a symlink is not copied: one pointing back into the tree would let a \
             `git apply` escape the copy"
        );
    }

    #[test]
    fn seed_and_write_from_overlays_the_source_on_the_built_tree() {
        let exec = tempfile::tempdir().unwrap();
        built_tree(exec.path());
        let seed = tempfile::tempdir().unwrap();
        copy_tree_excluding(seed.path(), exec.path()).unwrap();
        let root = tempfile::tempdir().unwrap();

        let concat = "# FILE: pkg/__init__.py\nVALUE = 2\n\n\n# FILE: pkg/core.py\ndef add(a, b):\n    return a + b\n";
        seed_and_write_from(Some(seed.path()), root.path(), concat).unwrap();

        assert_eq!(
            fs::read_to_string(root.path().join("pkg/__init__.py")).unwrap(),
            "VALUE = 2\n",
            "the concatenation is the SOURCE UNDER TEST and must win over the seed"
        );
        assert!(
            root.path().join("pkg/core.py").is_file(),
            "a file only the concatenation names is written"
        );
        assert_eq!(
            fs::read(root.path().join("pkg/_ext.so")).unwrap(),
            vec![0u8, 1, 2, 3],
            "the compiled extension reaches the sandbox, so `import pkg` is not a \
             partially-initialised module"
        );
        assert_eq!(
            fs::read_to_string(root.path().join("pytest.ini")).unwrap(),
            "[pytest]\nminversion=3.6\n",
            "the project's pytest config reaches the sandbox, so pytest resolves the \
             same rootdir/ini as the harness's own oracle and native runs"
        );
    }

    #[test]
    fn seed_and_write_from_prunes_the_source_the_concat_dropped() {
        let exec = tempfile::tempdir().unwrap();
        built_tree(exec.path());
        fs::write(exec.path().join("pkg/old.py"), "OLD = 1\n").unwrap();
        fs::write(exec.path().join("pkg/keep.py"), "KEEP = 1\n").unwrap();
        let seed = tempfile::tempdir().unwrap();
        copy_tree_excluding(seed.path(), exec.path()).unwrap();
        let root = tempfile::tempdir().unwrap();

        let concat = "# FILE: pkg/keep.py\nKEEP = 2\n";
        seed_and_write_from(Some(seed.path()), root.path(), concat).unwrap();

        assert_eq!(
            fs::read_to_string(root.path().join("pkg/keep.py")).unwrap(),
            "KEEP = 2\n"
        );
        assert!(
            !root.path().join("pkg/old.py").exists(),
            "a source file the concatenation no longer lists — a path the patch deleted — \
             must not survive the overlay"
        );
        assert!(
            !root.path().join("pkg/__init__.py").exists(),
            "the executed tree's SOURCE is exactly the concatenation's, no more"
        );
    }

    // [VANTA-WHY] C7 — `keep` is keyed by the concatenation's UTF-8 paths, so a
    // source file whose relative path is not valid UTF-8 can never be one of them.
    // The prune must keep it rather than delete it: the old `to_string_lossy()`
    // decode made the path match nothing and the file was removed. The non-UTF-8
    // path is built in memory (unix `OsStrExt`) because macOS, where this suite
    // runs, refuses to create such a file at all (`EILSEQ`); the branch it covers
    // is reachable on the Linux VMs the tree is actually seeded into.
    #[cfg(unix)]
    #[test]
    fn a_source_file_whose_relative_path_is_not_valid_utf8_is_kept_by_the_prune() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let keep: HashSet<String> = ["pkg/keep.py".to_string()].into_iter().collect();

        let weird = Path::new(OsStr::from_bytes(b"pkg/\xff\xfe.py"));
        assert!(
            weird.to_str().is_none(),
            "the fixture must actually be invalid UTF-8, or the test proves nothing"
        );
        assert!(
            prune_keeps(weird, &keep),
            "a non-UTF-8 relative path cannot be a `keep` key, so the prune keeps it"
        );

        assert!(
            prune_keeps(Path::new("pkg/keep.py"), &keep),
            "a valid-UTF-8 path the concatenation lists is kept"
        );
        assert!(
            !prune_keeps(Path::new("pkg/old.py"), &keep),
            "a valid-UTF-8 path the concatenation omits is still pruned"
        );
    }

    /// One lock plus a drop guard for the tests that touch the process-wide
    /// `SEED_ROOT`/`SEED_METHOD` globals (C6).
    ///
    /// [VANTA-WHY] C6 — `SEED_ROOT` and `SEED_METHOD` are process-wide statics, and
    /// the production-seed test below sets them and restores them by hand. Under
    /// the default parallel runner another test that reads `seed_root()` — or calls
    /// `seed_and_write`, which seeds through it — can observe a seed set mid-test,
    /// which is why the suite had to run `--test-threads=1`. This lock makes the two
    /// tests that touch the globals exclusive, and the guard clears the global on
    /// `take` and restores it on `drop` (on every exit path, a panicking assert
    /// included), so one test's seed can never leak into another's and the suite no
    /// longer needs the flag.
    static SEED_GLOBAL_LOCK: Mutex<()> = Mutex::new(());

    struct SeedGlobalGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl SeedGlobalGuard {
        /// Take the lock and start from the harness default (`None`), so a seed left
        /// behind by an earlier panic cannot reach the test that now holds the lock.
        fn take() -> Self {
            let lock = SEED_GLOBAL_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            set_seed_root(None);
            Self { _lock: lock }
        }

        /// Point the harness at a seed tree for the rest of the guard's life.
        fn set(&self, root: Option<PathBuf>) {
            set_seed_root(root);
        }
    }

    impl Drop for SeedGlobalGuard {
        fn drop(&mut self) {
            set_seed_root(None);
        }
    }

    #[test]
    fn seed_and_write_without_a_seed_writes_only_the_concat() {
        // [VANTA-WHY] C6 — hold the seed lock so the production-seed test cannot set
        // the global between this test's read and its `seed_and_write` call.
        let _seed = SeedGlobalGuard::take();
        assert!(seed_root().is_none(), "no seed is the harness's default");
        let root = tempfile::tempdir().unwrap();
        let concat = "# FILE: pkg/__init__.py\nVALUE = 2\n";

        seed_and_write(root.path(), concat).unwrap();

        assert_eq!(
            fs::read_to_string(root.path().join("pkg/__init__.py")).unwrap(),
            "VALUE = 2\n"
        );
        assert_eq!(
            fs::read_dir(root.path()).unwrap().count(),
            1,
            "without a seed nothing but the concatenation's own paths is written"
        );
    }

    // [VANTA-WHY] F1 — `seed_source_and_conftest` is the seam every graded path
    // now goes through, and the bug it pins is that its two writes used to be
    // DISCARDED (`let _ =`) on a path that ends in a verdict about the
    // candidate. A concat that names no file makes `seed_and_write` fail
    // (production hits the same condition one step later, inside
    // `prune_source_files_not_in`, after the seed copy but before the overlay);
    // the helper must REPORT that failure and must NOT then leave a conftest
    // behind, which would make a tree that was never overlaid look set up. The
    // success path must land both writes.
    #[test]
    fn seed_source_and_conftest_reports_a_failed_seed_and_lands_both_writes_on_success() {
        // Hold the seed lock: the helper seeds through the process-wide
        // `SEED_ROOT`, which the production-seed test also writes.
        let _seed = SeedGlobalGuard::take();
        assert!(seed_root().is_none(), "no seed is the harness's default");

        let bad_root = tempfile::tempdir().unwrap();
        let err = seed_source_and_conftest(bad_root.path(), "no headers here\n")
            .expect_err("a seed that names no file must be reported, not swallowed");
        assert!(
            err.to_string().contains("no # FILE: entries"),
            "the helper propagates the seed's own error, got {err}"
        );
        assert!(
            !bad_root.path().join("conftest.py").exists(),
            "the failed seed must not be followed by the conftest write"
        );

        let root = tempfile::tempdir().unwrap();
        let concat = "# FILE: pkg/__init__.py\nVALUE = 2\n";
        seed_source_and_conftest(root.path(), concat).unwrap();

        assert_eq!(
            fs::read_to_string(root.path().join("pkg/__init__.py")).unwrap(),
            "VALUE = 2\n",
            "the overlay is written"
        );
        let conftest = fs::read_to_string(root.path().join("conftest.py")).unwrap();
        assert!(
            conftest.contains("'src'"),
            "the conftest puts src/ on sys.path so a src-layout repo imports, got {conftest:?}"
        );
    }

    // [VANTA-WHY] Item P follow-up — the bug this pins: `seed_and_write` seeded
    // through `seed_sandbox_from` (the global-free variant the tests above use),
    // so the mechanism was never recorded and every row read `seed_method=unset`
    // however the tree had actually been seeded. It drives the production pair —
    // `set_seed_root`, then `seed_and_write` — and restores the global after.
    #[test]
    fn the_production_seed_records_which_mechanism_it_used() {
        // [VANTA-WHY] C6 — take the seed lock and let the guard restore the global,
        // so the parallel runner cannot observe this test's seed.
        let seed_guard = SeedGlobalGuard::take();
        let exec = tempfile::tempdir().unwrap();
        built_tree(exec.path());
        let seed = tempfile::tempdir().unwrap();
        copy_tree_excluding(seed.path(), exec.path()).unwrap();
        let root = tempfile::tempdir().unwrap();
        let concat = "# FILE: pkg/__init__.py\nVALUE = 2\n";

        seed_guard.set(Some(seed.path().to_path_buf()));
        assert!(seed_method().is_none(), "a run has recorded nothing before it seeds");
        seed_and_write(root.path(), concat).unwrap();
        let first = seed_method();
        // The record is the FIRST seed's mechanism (`seed_method`'s own doc), so a
        // later sandbox of the same run must not rewrite it.
        seed_and_write(root.path(), concat).unwrap();
        assert_eq!(seed_method(), first, "the record belongs to the run's first seed");
        // Release the lock (and restore the global) here; the asserts below never
        // read the globals, so they cannot observe a seed either way.
        drop(seed_guard);
        assert!(
            seed_root().is_none(),
            "the guard must restore the global on drop, so no later test can observe this seed"
        );

        assert!(
            matches!(first.map(|m| m.as_str()), Some("clonefile") | Some("copy_tree")),
            "the row must be able to state which mechanism seeded the run, got {first:?}"
        );
        assert!(
            root.path().join("pkg/_ext.so").is_file(),
            "and the production path really did seed"
        );
    }

    #[test]
    fn seed_sandbox_from_clones_the_seed_tree() {
        let exec = tempfile::tempdir().unwrap();
        built_tree(exec.path());
        let seed = tempfile::tempdir().unwrap();
        copy_tree_excluding(seed.path(), exec.path()).unwrap();
        let root = tempfile::tempdir().unwrap();

        let method = seed_sandbox_from(seed.path(), root.path()).expect("the seed is readable");

        assert!(
            matches!(method.as_str(), "clonefile" | "copy_tree"),
            "the mechanism is one of the two the row names, got {:?}",
            method.as_str()
        );
        assert!(root.path().join("pkg/_ext.so").is_file());
        assert!(root.path().join("pytest.ini").is_file());
        // A clone (or a copy) is the sandbox's OWN tree: removing a file from it
        // must not touch the seed every other sandbox is cloned from.
        fs::remove_file(root.path().join("pkg/__init__.py")).unwrap();
        assert!(
            seed.path().join("pkg/__init__.py").is_file(),
            "each sandbox must get its own writable tree"
        );
    }

    #[test]
    fn project_package_reads_the_pythonpath_root() {
        let tree = tempfile::tempdir().unwrap();
        fs::create_dir_all(tree.path().join("lib/matplotlib")).unwrap();
        fs::write(tree.path().join("lib/matplotlib/__init__.py"), "").unwrap();
        fs::create_dir_all(tree.path().join("tests")).unwrap();
        fs::write(tree.path().join("tests/__init__.py"), "").unwrap();
        let cmd = TestCommand::default().with_env(vec![("PYTHONPATH".into(), "./lib".into())]);

        assert_eq!(
            project_package(tree.path(), &cmd),
            Some("matplotlib".to_string()),
            "matplotlib's package sits under the lib/ its own PYTHONPATH names — the layout \
             `detect_import_path` reads as nothing"
        );
    }

    #[test]
    fn project_package_is_none_when_only_tooling_looks_like_a_package() {
        let tree = tempfile::tempdir().unwrap();
        fs::create_dir_all(tree.path().join("tests")).unwrap();
        fs::write(tree.path().join("tests/__init__.py"), "").unwrap();

        assert_eq!(
            project_package(tree.path(), &TestCommand::default()),
            None,
            "a stray tests/__init__.py must not become `the project`"
        );
    }

    #[test]
    fn is_tooling_package_names_the_repos_tooling_directories() {
        for name in ["tests", "test", "docs", "examples", "benchmarks", "scripts", "build"] {
            assert!(is_tooling_package(name), "{name} is repo tooling, not the package");
        }
        for name in ["matplotlib", "sanic", "pandas", "pysnooper", "fastapi"] {
            assert!(!is_tooling_package(name), "{name} is a package under test");
        }
    }

    #[test]
    fn the_in_session_invocation_differs_from_the_harnesss_only_in_the_report_file() {
        // [VANTA-WHY] Item P. The harness's own oracle/native runs execute the
        // split's `test_cmd` with `{report}` substituted. Vanta is handed that
        // command minus its report flags (`runner.py::vanta_test_cmd`) and appends
        // its own (`build_single_test_cmd`). With the seed now carrying the
        // project's config files, the two invocations must agree on everything the
        // report file's path does not name — otherwise the in-session run measures
        // a different suite from the one the harness grades.
        let harness_argv: Vec<String> = [
            "/venv/bin/python",
            "-m",
            "pytest",
            "lib/matplotlib/tests",
            "-p",
            "no:cacheprovider",
            "--json-report",
            "--json-report-file=/workspace/report.json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let handed_over =
            TestCommand::from_shell("/venv/bin/python -m pytest lib/matplotlib/tests -p no:cacheprovider");
        let in_session = build_single_test_cmd(&handed_over, "test_run.py");

        let flags = |argv: &[String]| -> Vec<String> {
            argv.iter()
                .filter(|a| a.as_str().starts_with('-'))
                .cloned()
                .collect()
        };
        let without_report = |argv: &[String]| -> Vec<String> {
            argv.iter()
                .filter(|a| {
                    !a.as_str().starts_with("--json-report")
                        && a.as_str() != "-q"
                        // [VANTA-WHY] Item U — the in-session run also adds the
                        // harness-owned `-o addopts=`, which the native harness
                        // invocation does not carry (it runs the repo's own suite
                        // and loads the repo's conftest, so the repo's addopts is
                        // valid there). It is not part of the handoff.
                        && a.as_str() != "-o"
                })
                .cloned()
                .collect()
        };

        assert_eq!(
            without_report(&flags(&in_session)),
            without_report(&flags(&harness_argv)),
            "the interpreter, the module invocation and every pinned flag must survive the \
             handoff unchanged; the in-session run adds only its own report flags, `-q` and \
             the `-o addopts=` that neutralizes a repo-conftest-only addopts"
        );
        assert!(
            in_session.iter().any(|a| a.starts_with("--json-report-file=")),
            "the in-session run still writes a JSON report for the driver to read: {in_session:?}"
        );
    }
}
