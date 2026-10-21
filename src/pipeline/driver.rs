// ============================================================================
// DRIVER — the single repair-loop / run driver (Spec §7, §8)
// ============================================================================
//
// [VANTA-WHY] Before this module existed, M2 and M3 each carried their own copy
// of the attempt cycle, and the copies drifted: M3's BATE ran the baseline twice
// for flakiness but never sent a mismatched test back to the Attacker for
// revision; M3's Diagnostician classified a failure but never acted on the class
// (no sandbox reset for Infrastructure, no simplify for Unknown). Those two
// regressions were invisible because the two loops were separate code.
//
// There is now ONE loop. M3 drives it at N attempts; M2 drives the SAME code at
// N = 1. A fix to the loop lands in both, and "M2 and M3 behave differently" can
// only mean the config differs, never the logic.
//
// Layers (each is a single definition the driver calls into):
//   harness      — execute tests / static analysis inside the sandbox
//   bate         — validate + revise Attacker tests against the baseline
//   diagnostician— classify a patched failure and run Phase-2 remediation
//   governor     — record shared-set transitions, decide continue/stop
//   gatekeeper   — deterministic approve/reject
// ============================================================================

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agents::attacker::{
    AttackerRequest, GeneratedTest, IntentCheckRecord, SharedProbe, attacker_generate_tests,
    audit_tests_intent,
};
use crate::agents::creator::{
    CreatorRequest, SourceFile, creator_generate_patch, split_multi_file_patch,
    split_patch_into_files,
};
use crate::config::LlmConfig;
use crate::core::{
    AttemptTransition, EvidenceSource, FailureClass, GovernorDecision, TestEvidence, TestIntent,
    TestResult, VerificationOutcome, VerificationPolicy, static_finding_identity,
};
use crate::llm::{build_reqwest_client, drain_call_usages, reset_call_usages, server_top_p_default};
use crate::pipeline::bate::run_bate;
use crate::pipeline::candidate::{CandidatePair, TestCommand};
use crate::pipeline::diagnostician::diagnose_failure;
use crate::pipeline::gatekeeper::{Gatekeeper, GatekeeperVerdict};
use crate::pipeline::governor::{
    AttemptVerdict, Governor, GovernorMode, SharedObservation,
};
use crate::pipeline::harness::{self, TestExecutionDetails};
use crate::pipeline::patch::{
    ApplyFailure, apply_patch_detailed, apply_patch_strict, patch_deleted_paths,
};
use crate::repo::Repo;
use crate::session::{
    PatchAttemptId, SCHEMA_VERSION, SessionId, SessionLog, SessionRecord, TransitionRow,
    TransitionWriter, ecr_eir_from_rows, fresh_session_id, fresh_test_id,
    fresh_testing_iteration_id, vanta_revision,
};

// ============================================================================
// RunConfig — everything that distinguishes one driver invocation from another
// ============================================================================
//
// [VANTA-WHY] The ONLY thing M2 and M3 are allowed to differ in. M2 passes
// `max_attempts: 1`; everything else is the same code path.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub repo_path: PathBuf,
    pub issue_path: Option<PathBuf>,
    pub max_attempts: u32,
    pub model_override: Option<String>,
    /// If true, open a GitHub PR after Gatekeeper approval.
    pub open_pr: bool,
    /// If Some(path), skip the Creator and use this patch file directly.
    /// Used for verifying an EXISTING patch (e.g. a developer's working tree diff).
    pub patch_path: Option<PathBuf>,
    /// Explicit test command (Spec §7: "Require explicit --test-cmd, pytest as
    /// fallback. Do not infer."). e.g. "pytest tests/ -q".
    pub test_cmd: Option<String>,
    /// Path to a clean original tree (git worktree of the base commit) for
    /// two-tree mode. If set, BATE runs against this tree instead of the
    /// in-memory original files. Spec §7: "both trees present simultaneously".
    pub original_tree: Option<PathBuf>,
}

// ============================================================================
// run_with_spinner — print dots while a future runs (LLM calls take minutes)
// ============================================================================
//
// [VANTA-WHY] Creator and Attacker phases make LLM calls that can take 1-5
// minutes. Without any output, the user thinks the process is frozen. This
// helper spawns a background task that prints a dot every 2 seconds while
// the future runs, then clears the line when done.
// ============================================================================
async fn run_with_spinner<F, T>(label: &str, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    // [VANTA-WHY] The dots exist so a human watching a 1-5 minute LLM call does
    // not think the process froze. When stderr is NOT a terminal — piped to a
    // file, run under CI, or captured by `run_matrix.sh` — they are noise that
    // only corrupt the log, so the spinner is skipped entirely and the future
    // is simply awaited.
    if !io::stderr().is_terminal() {
        return fut.await;
    }
    eprint!("  ⏳ {label} ");
    let _ = io::Write::flush(&mut io::stderr());
    let label_owned = label.to_string();
    let spinner = tokio::spawn(async move {
        let mut count = 0;
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            count += 1;
            if count % 15 == 0 {
                // Every ~30s, print elapsed time
                let elapsed = count * 2;
                eprint!("\n  ⏳ {label_owned} ...({elapsed}s) ");
            } else {
                eprint!(".");
            }
            let _ = io::Write::flush(&mut io::stderr());
        }
    });
    let result = fut.await;
    spinner.abort();
    eprintln!(" ✓");
    let _ = io::Write::flush(&mut io::stderr());
    result
}

// ============================================================================
// env_flag — read a boolean harness switch
// ============================================================================
//
// [VANTA-WHY] Strict on purpose. `VANTA_ATTACKER=offf` quietly meaning "on"
// would produce a `vanta` run that the batch manifest records as `baseline` —
// a mislabelled datum, which is worse than a failed run, because nothing
// downstream can detect it. An unrecognised value is an error, and an absent
// var is the documented default (plans/bugsinpy.md §4.2).
//
// `on`/`off` are the documented spellings; the usual aliases are accepted
// because a shell that exports `VANTA_ATTACKER=0` means the same thing.
fn env_flag(name: &str, default: bool) -> Result<bool, String> {
    match std::env::var(name) {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "on" | "1" | "true" | "yes" => Ok(true),
            "off" | "0" | "false" | "no" => Ok(false),
            other => Err(format!(
                "CONFIG_ERROR: {name}='{other}' is not a switch value (expected on|off)."
            )),
        },
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "CONFIG_ERROR: {name} env var contained non-UTF-8 bytes."
        )),
    }
}

/// The `baseline` / `vanta` / `vanta_planner` name for a pair of switches.
///
/// [VANTA-WHY] Deriving the name from the switches instead of taking a fourth
/// env var means the config a session records CANNOT disagree with the config
/// it ran: §4.2 defines the three configurations as exactly these switch
/// settings, so this is the single definition of the mapping.
fn benchmark_config_name(attacker_enabled: bool, planner_enabled: bool) -> &'static str {
    if planner_enabled {
        "vanta_planner"
    } else if attacker_enabled {
        "vanta"
    } else {
        "baseline"
    }
}

/// Render one effective sampling value for the `run.summary` provenance keys.
///
/// [VANTA-WHY] `unset` (not `null`, not a plausible default) is what an absent
/// `/props` key becomes. The published numbers are the SERVER's, and the record
/// must say so by showing nothing when the server showed nothing.
fn sampling_value(value: Option<f64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "unset".to_string())
}

/// Render a token count for a record line, `unset` when the server published
/// nothing — the same word the sampling provenance uses for the same fact, never
/// a zero, which would be a claim that the call used no tokens.
fn token_value(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "unset".to_string())
}

/// Sum the server-measured token counts over one run's chat calls.
///
/// [VANTA-WHY] Returns the call counts beside the sums, because a total over
/// every call and a total over the calls the server actually measured are
/// different numbers and the record must not conflate them. A call whose
/// `source` is `estimate` (see `llm::CallUsage`) carries a prompt figure derived
/// from CHARACTERS — deliberately high — and no completion figure at all, so
/// folding it into the sum would mix an over-count into a measurement.
/// `(calls, measured, prompt_total, completion_total)`.
fn token_totals(calls: &[crate::llm::CallUsage]) -> (usize, usize, u64, Option<u64>) {
    let measured: Vec<&crate::llm::CallUsage> =
        calls.iter().filter(|c| c.source != "estimate").collect();
    let prompt: u64 = measured.iter().filter_map(|c| c.prompt_tokens).sum();
    let completions: Vec<u64> = measured.iter().filter_map(|c| c.completion_tokens).collect();
    let completion = if completions.is_empty() {
        None
    } else {
        Some(completions.iter().sum())
    };
    (calls.len(), measured.len(), prompt, completion)
}

/// Render the pinned per-call-site sampling schedule as `run.summary` keys.
///
/// [VANTA-WHY] Underscores, not dots. `rerun.rs::summary_field` and
/// `scripts/metrics.py::summary_field` both accept a key only if it matches
/// `[a-z_][a-z0-9_]*`, so `temperature.creator=0.1` would not terminate at the
/// next key — the dot is not a key character, the value would run on through the
/// rest of the line, and BOTH consumers would silently read the wrong thing.
/// `{v:?}` for the float so `0.0` stays `0.0` and the record matches the
/// request body byte for byte.
fn sampling_schedule_keys() -> String {
    crate::llm::SAMPLING_SCHEDULE
        .iter()
        .map(|(site, temperature, max_tokens)| {
            format!("temperature_{site}={temperature:?} max_tokens_{site}={max_tokens}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Read the server's own `top_p` default for the provenance block.
///
/// [VANTA-WHY] Bounded, unlike every other HTTP call in this file.
/// [`build_reqwest_client`] deliberately carries NO total timeout — a 27B model
/// legitimately streams for 20 minutes and a deadline there killed it mid-
/// generation (see the note in `llm::build_reqwest_client`). That is right for a
/// generation and wrong for this probe: it runs AFTER the run has already
/// reached its verdict, so an unreachable or wedged `/props` must not be able to
/// stop the session from ever writing its own `run.summary` — the row that
/// records the verdict. On timeout the key is recorded as unset, which is the
/// truth: the server did not answer.
async fn server_top_p_for_provenance(cfg: &LlmConfig) -> Option<f64> {
    let Ok(client) = build_reqwest_client(cfg) else {
        return None;
    };
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        server_top_p_default(&client, cfg),
    )
    .await
    .unwrap_or(None)
}

// ============================================================================
// run — drive one full Vanta run and return the Governor's final decision
// ============================================================================
pub async fn run(
    config: RunConfig,
) -> Result<GovernorDecision, Box<dyn std::error::Error + Send + Sync>> {
    let _terminal_restore = crate::terminal::TerminalRestoreGuard::new();

    // ---- Config ----
    let mut cfg = LlmConfig::from_env()?;
    if let Some(m) = &config.model_override {
        cfg.model_id = m.clone();
    }
    println!("{cfg}\n");

    // ---- Harness switches (plans/bugsinpy.md §4.2) ----
    // [VANTA-WHY] Read ONCE, here, so no later code can disagree about which
    // configuration this run is. Both default to today's behaviour, so an
    // ordinary invocation is unchanged; the same binary then serves every
    // configuration the harness runs, which is what keeps the comparison
    // honest — two binaries could drift and the difference would be attributed
    // to the configuration.
    let attacker_enabled = env_flag("VANTA_ATTACKER", true)?;
    let planner_enabled = env_flag("VANTA_PLANNER", false)?;
    if !attacker_enabled {
        println!(
            "▌ Config: {} — Attacker off (VANTA_ATTACKER=off): the repo's own suite and static \
             analysis are the whole question.",
            benchmark_config_name(attacker_enabled, planner_enabled)
        );
    }

    // ---- Load repo ----
    println!("▌ Loading repo: {}", config.repo_path.display());
    let repo = Repo::load(&config.repo_path, config.issue_path.as_deref())?;
    println!(
        "  Found {} source files ({} bytes), issue text {} bytes",
        repo.source_files.len(),
        repo.source_files
            .iter()
            .map(|f| f.contents.len())
            .sum::<usize>(),
        repo.issue_text.len()
    );

    // ---- Test command (Spec §7: explicit --test-cmd, pytest fallback) ----
    let test_cmd: TestCommand = match &config.test_cmd {
        Some(s) => TestCommand::from_shell(s),
        None => TestCommand::default(),
    };
    println!(
        "▌ Test command: {} {}",
        test_cmd.program,
        test_cmd.args.join(" ")
    );

    // ---- Import path detection (Spec §7: "Give the Attacker the real import
    //      path, or tests fail as Static and burn attempts") ----
    let import_hint = harness::detect_import_path(&repo.source_files, &config.repo_path);
    println!(
        "▌ Import hint: {}",
        import_hint.as_deref().unwrap_or("(flat layout)")
    );

    // ---- Selected backend (fixed for the process: env preference + boot test) ----
    let requested_backend = crate::sandbox::requested_backend_label()?;
    let backend_kind = crate::sandbox::actual_backend_kind().await?;
    let actual_backend = crate::sandbox::backend_kind_label(backend_kind);

    // ---- Auto-detect repo dependencies and prepare the VM checkpoint ----
    // [VANTA-WHY] The base shuru checkpoint only ships python3 + pytest + ruff +
    // mypy. A real repo declares its own deps which are absent inside the VM, so
    // tests die at import and BATE sees `Error` for everything. Build/cache a
    // checkpoint up front so test runs use the warmed environment.
    //
    // Only for shuru runs: a subprocess run tests against the host environment,
    // so a VM checkpoint would download dependencies the run never uses.
    let mut deps_checkpoint =
        std::env::var("VANTA_SHURU_CHECKPOINT").unwrap_or_else(|_| "none".to_string());
    let mut deps_freeze: Option<String> = None;
    if crate::sandbox::deps::auto_build_enabled(backend_kind)
        && std::env::var("VANTA_SHURU_CHECKPOINT").is_err()
        && std::env::var("VANTA_SHURU_AUTO_DEPS").as_deref() != Ok("0")
    {
        println!("▌ VM deps: detecting declared dependencies...");
        match crate::sandbox::shuru::ShuruBackend::ensure_repo_checkpoint(&config.repo_path) {
            Ok(Some(checkpoint)) => {
                println!(
                    "▌ VM deps checkpoint: {} (disk {} MB)",
                    checkpoint.name, checkpoint.disk_mb
                );
                deps_checkpoint = checkpoint.name.clone();
                deps_freeze = Some(checkpoint.pip_freeze.clone());
                // SAFETY: single-threaded CLI; no other thread reads env concurrently.
                unsafe {
                    std::env::set_var("VANTA_SHURU_CHECKPOINT", &checkpoint.name);
                    std::env::set_var("VANTA_SHURU_DISK_MB", checkpoint.disk_mb.to_string());
                }
            }
            Ok(None) => println!("▌ VM deps: none declared — using base checkpoint"),
            Err(e) => eprintln!("  ⚠ VM deps auto-install failed: {e}"),
        }
    }

    // ---- Session setup ----
    // [VANTA-WHY] The session log is opened as a durable append-and-flush sink
    // BEFORE attempt 1, and its first record is `session.start`. That record is
    // the run's attribution: model, provider, prompts, Vanta build, backend,
    // benchmark instance. The transition rows deliberately do NOT carry these
    // facts (Spec §8) — they live on the session records — so if the log were
    // written once at the end, a crash would leave rows saying WHAT changed with
    // no record of WHICH model produced them, and an unattributed run is useless
    // to a study whose central comparison is across model tiers.
    let session_id = fresh_session_id();
    let mut session_log = SessionLog::open(Path::new("vanta_sessions"), session_id.clone())?;
    // [VANTA-WHY] Token accounting is a process-global accumulator (see
    // `llm::CallUsage`), so a second run in the same process — the batch runner,
    // or a test process — would otherwise attribute the previous run's calls to
    // this one. Cleared before the first agent call, not at the end, so a run
    // that dies still leaves only its own calls behind.
    reset_call_usages();
    let debug_dir = session_log.ensure_debug_dir(Path::new("vanta_sessions"))?;
    // SAFETY: single-threaded CLI; no other thread reads env concurrently.
    unsafe {
        std::env::set_var("VANTA_DEBUG_DIR", &debug_dir);
    }
    println!("▌ Session: {}", session_id.as_str());
    println!("▌ Forensics → {}", debug_dir.display());
    println!("▌ Max attempts: {}", config.max_attempts);
    println!("▌ Backend requested: {requested_backend}");
    println!("▌ Backend actual: {actual_backend}");

    // The run-identifying facts, written before any attempt exists so that
    // transition rows surviving a crash can always be traced to their run.
    // [VANTA-WHY] `prompt_versions()` is the single definition (see
    // `pipeline::prompt_versions`) shared by this record and the PR body, so an
    // edit to any one prompt surfaces in both places rather than drifting.
    let prompt_versions = crate::pipeline::prompt_versions();
    let tested_commit = tested_commit_sha(&config.repo_path);
    let target_repo_dirty = repo_working_tree_dirty(&config.repo_path).ok();
    let run_mode = if config.patch_path.is_some() {
        "patch_review"
    } else {
        "creator"
    };
    let baseline_test_timeout_secs = crate::pipeline::harness::test_timeout_secs();
    let patched_test_timeout_secs = crate::pipeline::harness::patched_test_timeout_secs();
    if baseline_test_timeout_secs != patched_test_timeout_secs {
        tracing::warn!(
            baseline_test_timeout_secs,
            patched_test_timeout_secs,
            "patched-side test timeout differs from baseline timeout; this seam exists for tests and can bias a real run"
        );
        eprintln!(
            "  ⚠ patched-side test timeout differs from baseline timeout: baseline={}s patched={}s",
            baseline_test_timeout_secs, patched_test_timeout_secs
        );
    }
    // [VANTA-WHY] `dirty` accompanies the commit so the reproducibility claim is
    // checkable: a run on a dirty tree includes edits that are not in HEAD, so it
    // must never claim that commit reproduces it. The paper's rule is that any
    // run feeding the paper has `dirty = false`.
    let (vanta_commit, vanta_dirty) = vanta_revision();
    session_log.append(SessionRecord {
        session_id: session_id.clone(),
        schema_version: Some(SCHEMA_VERSION),
        // [VANTA-WHY] A run-level record: no patch attempt exists yet, so this is
        // null rather than an invented id.
        patch_attempt_id: None,
        testing_iteration_id: None,
        test_id: None,
        phase: "session.start".to_string(),
        summary: format!(
            "model={} model_version={} prompt_version={} provider={} api_key={} \
             vanta_version={} vanta_commit={} vanta_dirty={} tested_commit={} target_repo_dirty={} requested_backend={} actual_backend={} max_attempts={} \
             baseline_test_timeout_secs={} patched_test_timeout_secs={} deps_checkpoint={} deps_freeze={} test_cmd={} mode={} benchmark_instance={}",
            cfg.model_id,
            std::env::var("VANTA_MODEL_VERSION").unwrap_or_else(|_| "unknown".into()),
            prompt_versions,
            cfg.base_url,
            if cfg.api_key.is_some() { "set" } else { "none" },
            env!("CARGO_PKG_VERSION"),
            vanta_commit,
            vanta_dirty,
            tested_commit,
            target_repo_dirty
                .map(|dirty| if dirty { "true" } else { "false" })
                .unwrap_or("unknown"),
            requested_backend,
            actual_backend,
            config.max_attempts,
            baseline_test_timeout_secs,
            patched_test_timeout_secs,
            deps_checkpoint,
            dependency_freeze_summary_value(deps_freeze.as_deref()),
            format!("{} {}", test_cmd.program, test_cmd.args.join(" ")),
            run_mode,
            "none",
        ),
        timestamp: now_iso(),
        benchmark_instance: None,
        benchmark_label: None,
    })?;
    if config.open_pr {
        require_clean_target_repo_for_open_pr(target_repo_dirty)
            .map_err(|msg| -> Box<dyn std::error::Error + Send + Sync> { msg.into() })?;
        crate::pr::preflight_open_pr(
            &config.repo_path,
            session_log
                .sink_path()
                .ok_or("session log sink path missing for PR preflight")?,
        )
        .await?;
    }

    // ---- Build original file map (path -> contents) ----
    // [VANTA-WHY] Spec §7: "The original is a second checkout via git worktree
    // of the base commit. Not the same directory stashed and reverted." If
    // --original-tree is given, read original files from that clean worktree
    // instead of the (possibly edited) working tree.
    let mut original_files: HashMap<String, String> = HashMap::new();
    if let Some(orig_tree) = &config.original_tree {
        println!("▌ Two-tree mode: original tree = {}", orig_tree.display());
        let pair = crate::pipeline::candidate::GitWorkingTreePair::new(
            orig_tree.clone(),
            config.repo_path.clone(),
            repo.issue_text.clone(),
            test_cmd.clone(),
        );
        original_files = pair.original_files();
        println!(
            "  Loaded {} original files from worktree",
            original_files.len()
        );
    } else {
        for sf in &repo.source_files {
            original_files.insert(sf.path.clone(), sf.contents.clone());
        }
    }

    // ---- Governor + Gatekeeper ----
    let mut governor = Governor::new(config.max_attempts);
    let gatekeeper = Gatekeeper::new(VerificationPolicy::BestEffort);

    // ---- Repair loop ----
    let mut prior_feedback: Option<String> = None;
    let mut final_evidence: Vec<TestEvidence> = Vec::new();
    let mut final_patched_files: Option<HashMap<String, String>> = None;
    // Accepted Attacker tests (concatenated source) for the PR, set on the
    // final approved attempt.
    let mut final_accepted_tests: Option<String> = None;

    // [VANTA-WHY] The FIXED SHARED TEST SET (Spec §8 Measurement protocol).
    // EIR/ECR describe transitions between patch attempts. Comparing attempt N
    // vs N+1 requires running BOTH on the SAME saved tests. Newly generated
    // tests run immediately and can reject a patch on the spot, but they are
    // folded into the shared set only for the FOLLOWING comparison.
    // `shared_tests` persists across loop iterations; `new_valid_tests` is the
    // fresh batch generated this iteration.
    //
    // [PERSISTENCE] The shared set is also written to disk as
    // vanta_sessions/<id>/shared_tests.json so that a resumed run (or M5
    // re-analysis) replays the SAME tests. Spec §10 item 2: "Persist and
    // replay the shared test set across attempts."
    let shared_tests_path = debug_dir.join("shared_tests.json");
    let mut shared_tests: Vec<GeneratedTest> = if shared_tests_path.exists() {
        match fs::read_to_string(&shared_tests_path)
            .and_then(|s| Ok(serde_json::from_str::<Vec<GeneratedTest>>(&s)?))
        {
            Ok(st) => {
                println!("▌ Replayed {} shared test(s) from disk", st.len());
                st
            }
            Err(e) => {
                eprintln!("  ⚠ Failed to load shared_tests.json: {e} — starting fresh");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    // [VANTA-WHY] Consecutive all-Infrastructure attempts. Reset by any attempt
    // that produced a usable classification; drives the retry-once ladder.
    let mut harness_collapse_streak: u32 = 0;
    // [VANTA-WHY] patch_attempt_id per attempt (index == governor attempt
    // number). A transition row records BOTH ends of the comparison, so it needs
    // the id of each side: the candidate the older attempt ran and the candidate
    // the newer one ran. In --patch mode both are the same id (the candidate is
    // fixed), which is what marks those rows as attack depth rather than a patch
    // transition. The id is a deterministic fingerprint of the patch text.
    let mut attempt_patch_ids: Vec<PatchAttemptId> = Vec::new();
    // [VANTA-WHY] Attempt indices whose patch never reached the tree. In repair
    // mode such an attempt is NOT fatal: the Creator gets the reason as feedback
    // and the loop tries again, so the attempt has to appear in the dataset as a
    // recorded hole (see `Governor::record_unobserved_attempt`) rather than
    // vanish. The index list is kept, not a bare count, because each censored row
    // must carry `apply_failed` for ITS OWN attempt — and the same list's length
    // is the run-level `apply_failed_attempts` counter.
    let mut apply_failed_attempt_indices: Vec<u32> = Vec::new();
    // [VANTA-WHY] Rows are flushed to vanta_sessions/{id}_transitions.jsonl as
    // each transition is computed, so a run that aborts mid-way (HarnessFailure
    // is terminal by design) still leaves every row it produced on disk. The
    // cursor says how many transitions have already been persisted.
    let mut transition_writer = TransitionWriter::open(Path::new("vanta_sessions"), &session_id)?;
    let mut persisted_transitions: usize = 0;
    // [VANTA-WHY] The console's live EIR/ECR is computed by the SAME function the
    // M5 analysis layer will run over the file — `ecr_eir_from_rows` — so there is
    // one definition of the metric. That needs the rows in memory, not the
    // Governor's in-memory transitions, because only the rows carry the patch ids
    // that separate a repair transition from a review round.
    let mut metric_rows: Vec<TransitionRow> = Vec::new();
    // [VANTA-WHY] The native suite and the static gate observe the SAME candidate
    // the Governor does, but they do not go through it, so until now they wrote
    // no transition row at all. Repair session 4d55123d's only real correction
    // was a native test the previous attempt had broken, and it was invisible to
    // ECR/EIR for exactly that reason. These two carry the PREVIOUS attempt's
    // effective outcome so the current attempt can be differenced against it and
    // emit ordinary rows.
    let mut previous_native: Option<(u32, PatchAttemptId, AttemptNativeOutcome)> = None;
    let mut previous_static: Option<(u32, PatchAttemptId, AttemptStaticOutcome)> = None;
    // [VANTA-WHY] Per-attempt native regressions for evidence.json. The console
    // printed them and the Creator was fed them, but no artifact kept them, so
    // "which repo test did attempt 1 break?" could not be answered from the
    // run's files alone.
    let mut native_regression_records: Vec<serde_json::Value> = Vec::new();
    // [VANTA-WHY] Every NEW static finding this run's patch introduced, per
    // attempt, so evidence.json can name the finding that rejected a review
    // round. Session 40e84874 was rejected for a mypy finding that appeared
    // NOWHERE in the session files: the console printed `mypy_new=true` and the
    // detail was dropped. The record below is that missing detail.
    let mut static_findings_records: Vec<serde_json::Value> = Vec::new();
    // [VANTA-WHY] Test-integrity records for the evidence file. A patch must
    // never control the tests that grade it, so the native suite runs the
    // ORIGINAL test files and these four lists are what make that visible after
    // the run: which existing tests the patch rewrote (their edits discarded),
    // which it deleted (restored), which it added (left out of the baseline),
    // and which pytest config section it changed. They are refreshed per attempt
    // — the LAST attempt is the one the run's verdict is about.
    let mut modified_existing_tests: Vec<String> = Vec::new();
    let mut removed_existing_tests: Vec<String> = Vec::new();
    let mut added_test_files: Vec<String> = Vec::new();
    let mut modified_test_config: Vec<serde_json::Value> = Vec::new();

    // [VANTA-WHY] `max_attempts` means two different things and the Governor now
    // has to be told which. With `--patch` the candidate is FIXED, so an attempt
    // is another batch of attacks and the cap is ATTACK DEPTH; without it the
    // Creator writes a new candidate each attempt and the cap is a PATCH BUDGET.
    // This is the ONLY place the mode is derived, so repair and review cannot
    // drift apart again.
    let governor_mode = if config.patch_path.is_some() {
        GovernorMode::Review
    } else {
        GovernorMode::Repair
    };
    // [VANTA-WHY] The message from an LLM call that failed every attempt it was
    // allowed. It is carried OUT of the loop so the run can end as a recorded
    // verdict (`final_decision=InfraFailure`) with the session summary, evidence
    // and transitions written exactly as they are for any other outcome. Before
    // this, a dead LLM call propagated with `?` and the process exited with a
    // stack-style error and NO final decision on disk — the run was unreadable.
    let mut infra_failure_error: Option<String> = None;
    let decision = loop {
        let attempt = governor.current_attempt();

        // [VANTA-WHY] `VANTA_PLANNER=on` is the third configuration in §4.2 and
        // it is a PLACEHOLDER: no planner half of the pipeline exists. It ends
        // as an explicit `InfraFailure` before any work is attempted, and the
        // message says why. A config that "succeeds" because it does nothing is
        // exactly the failure mode this project exists to catch, and a
        // fabricated `vanta_planner` row in the manifest would be worse than a
        // missing one — so the switch is addressable and cannot silently pass.
        // It reuses the existing InfraFailure machinery (run.infra_failure row,
        // run.summary with final_decision=InfraFailure, exit code 2) rather
        // than inventing a fourth exit path the harness would have to learn.
        if planner_enabled {
            infra_failure_error = Some(
                "VANTA_PLANNER=on: the planner configuration is not implemented \
                 (plans/bugsinpy.md §4.2) — no planner stage exists, so this run \
                 cannot produce a result about a candidate."
                    .to_string(),
            );
            break GovernorDecision::InfraFailure;
        }
        // [VANTA-WHY] Forensics filenames carry the ROUND, not just the
        // structured-call retry index. Without this, every round wrote over the
        // previous one and the debug folder of a 3-round run held round 3 three
        // times over.
        crate::llm::set_forensics_round(attempt + 1);
        println!("\n══════════════════════════════════════════════════════════════");
        println!("  ATTEMPT {} / {}", attempt + 1, config.max_attempts);
        println!("══════════════════════════════════════════════════════════════");

        // ---- 1. Patch (Creator or --patch file) ----
        let patch_files: HashMap<String, String>;
        let patch_text: String;
        if let Some(patch_file) = &config.patch_path {
            // [VANTA-WHY] --patch mode verifies an EXISTING patch. Skip
            // Creator, but still allow multiple verification attempts so the
            // Attacker/BATE loop can try alternate tests against the same diff.
            println!("\n▌ PATCH mode: using {}", patch_file.display());
            patch_text = fs::read_to_string(patch_file)?;
            println!("  Patch loaded: {} bytes", patch_text.len());
            patch_files = split_patch_into_files(&patch_text);
        } else {
            println!("\n▌ CREATOR phase...");
            let creator_request = CreatorRequest {
                issue_text: repo.issue_text.clone(),
                source_files: repo.source_files.clone(),
                prior_feedback: prior_feedback.clone(),
            };
            let creator_response = match run_with_spinner("Creator generating patch", async {
                creator_generate_patch(&cfg, &creator_request).await
            })
            .await
            {
                Ok(response) => response,
                Err(error) => {
                    infra_failure_error = Some(format!("Creator LLM call failed: {error}"));
                    break GovernorDecision::InfraFailure;
                }
            };
            println!(
                "  ✔ Creator: patch_text={} chars, files_touched={:?}, confidence={}",
                creator_response.patch_text.len(),
                creator_response.files_touched,
                creator_response.confidence
            );
            patch_text = creator_response.patch_text.clone();
            patch_files = split_multi_file_patch(
                &creator_response.patch_text,
                &creator_response.files_touched,
            );
        }
        println!("  Patch spans {} file(s)", patch_files.len());
        let patch_attempt_id = patch_attempt_id_for_patch(&patch_text);
        attempt_patch_ids.push(patch_attempt_id.clone());
        let testing_iteration_id = fresh_testing_iteration_id();

        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: Some(testing_iteration_id.clone()),
            test_id: None,
            phase: "attempt.start".to_string(),
            summary: format!(
                "attempt={} mode={} patch_files={} patch_bytes={}",
                attempt + 1,
                if config.patch_path.is_some() {
                    "patch_review"
                } else {
                    "creator"
                },
                patch_files.len(),
                patch_text.len()
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;

        // ---- 2. Apply patch to each touched file ----
        println!("\n▌ Applying patch...");
        let mut patched_files = original_files.clone();
        let mut any_changed = false;
        let mut all_patch_failed = true;
        // [VANTA-WHY] The reason a failed apply must reach the forensics file and
        // the console in plain language. In review mode `git apply` tells us
        // exactly which hunk it could not place; hiding that behind "no file was
        // changed" is how three rounds were burnt re-testing a file Vanta itself
        // had corrupted.
        let mut patch_failure_reason: Option<String> = None;
        // [VANTA-WHY] WHY the cascade gave up, not just THAT it did. "I could not
        // place your hunks" is a formatting problem the Creator can fix by
        // re-emitting the diff; "your patch applied but the result is not valid
        // Python" is a content problem it can only fix by changing the code.
        // Telling it the wrong one wastes the retry the failure just bought.
        let mut apply_failure: Option<ApplyFailure> = None;

        if config.patch_path.is_some() {
            // [VANTA-WHY] REVIEW MODE IS STRICT. The candidate was written
            // outside this run, so only `git apply` (then `git apply --recount`
            // for a miscounted header) may touch the tree. A fuzzy strategy that
            // "recovers" it would put a tree nobody wrote under test, and the
            // resulting verdict would be a statement about Vanta's repair engine
            // rather than about the candidate. Anything else stops the run with
            // PatchApplicationFailed.
            let touched: Vec<String> = patch_files.keys().cloned().collect();
            match apply_patch_strict(&original_files, &patch_text, &touched) {
                Ok(changed) => {
                    for (path, content) in changed {
                        println!("  ✔ {path}: patched (git apply)");
                        patched_files.insert(path, content);
                        any_changed = true;
                        all_patch_failed = false;
                    }
                }
                Err(reason) => {
                    println!("  ✗ PATCH APPLICATION FAILED — review mode applies `git apply` only");
                    for line in reason.lines() {
                        println!("      {line}");
                    }
                    patch_failure_reason = Some(reason);
                }
            }
        } else {
            // [VANTA-WHY] REPAIR MODE keeps the cascade: the Creator can emit a
            // diff with wrong headers, and the loop has a budget to recover. The
            // per-file path is passed through so `git apply`/`patch(1)` see the
            // file at the name the diff header uses, and so the syntax guard
            // knows whether the result must parse as Python.
            for (path, patch_content) in &patch_files {
                print!(
                    "  ⏳ {path}: applying ({:.1}KB patch)... ",
                    patch_content.len() as f64 / 1024.0
                );
                let _ = io::Write::flush(&mut io::stdout());
                if let Some(original) = original_files.get(path) {
                    let (patched, changed, why) =
                        apply_patch_detailed(original, patch_content, path);
                    if changed {
                        patched_files.insert(path.clone(), patched);
                        any_changed = true;
                        all_patch_failed = false;
                        println!("✔ patched (changed)");
                    } else {
                        println!("✗ patch did not change file");
                        // Keep the FIRST reason: it is the strategy the patch was
                        // most likely written for, and a later syntax-guard
                        // refusal of a different strategy would otherwise hide it.
                        if apply_failure.is_none() {
                            apply_failure = why;
                        }
                    }
                } else {
                    // New file not in original
                    if !patch_content.trim().is_empty() {
                        patched_files.insert(path.clone(), patch_content.clone());
                        any_changed = true;
                        all_patch_failed = false;
                        println!("  ✔ {path}: new file");
                    }
                }
            }
        }

        if all_patch_failed {
            let reason = patch_failure_reason
                .clone()
                .unwrap_or_else(|| "no file was changed by any strategy".to_string());
            // Write forensics
            let faillog = debug_dir.join("patch_application_failed.json");
            let _ = fs::write(
                &faillog,
                serde_json::to_string_pretty(&serde_json::json!({
                    "patch_text": patch_text,
                    "files_touched": patch_files.keys().cloned().collect::<Vec<_>>(),
                    "reason": reason
                }))
                .unwrap(),
            );
            println!("  ✗ PATCH APPLICATION FAILED — see {}", faillog.display());
            session_log.append(SessionRecord {
                session_id: session_id.clone(),
                schema_version: Some(SCHEMA_VERSION),
                patch_attempt_id: Some(patch_attempt_id.clone()),
                testing_iteration_id: Some(testing_iteration_id.clone()),
                test_id: None,
                phase: "patch.apply_failed".to_string(),
                summary: reason.clone(),
                timestamp: now_iso(),
                benchmark_instance: None,
                benchmark_label: None,
            })?;

            if governor_mode == GovernorMode::Repair {
                // [VANTA-WHY] REPAIR MODE DOES NOT STOP. A Creator patch that
                // will not apply is "the diff is malformed" or "the diff applies
                // but the result is not valid Python" — both are things the
                // Creator can fix when it is TOLD which one happened. Ending the
                // run here would make every bad diff cost a whole benchmark
                // instance instead of one attempt, which is exactly how the
                // repair rate gets dragged down. The attempt is spent (the
                // Governor advances the budget), the failed apply is recorded as
                // a censored transition row, and the reason goes back as the
                // Creator's feedback. Review mode, whose candidate was written
                // outside the run, still stops — see `Governor::decide`.
                let feedback = match &apply_failure {
                    // Content problem: the diff applied, the code does not parse.
                    Some(ApplyFailure::InvalidPython(err)) => {
                        format!("patch result is not valid Python: {err}")
                    }
                    // Formatting problem: nothing could place the hunks at all.
                    _ => format!("patch did not apply: {reason}"),
                };
                println!("  ↻ Returning to Creator: {feedback}");
                apply_failed_attempt_indices.push(attempt);
                // [VANTA-WHY] Record the hole BEFORE deciding, so the rows belong
                // to this attempt whatever the budget does next.
                // `previous_shared` is deliberately untouched, so the attempt
                // that finally applies is still compared against the last
                // candidate anyone OBSERVED — the comparison that matters ("did
                // the patch that finally applied fix the defect the last observed
                // patch had?") is not lost.
                governor.record_unobserved_attempt();
                persisted_transitions = flush_transition_rows(
                    &governor,
                    &mut transition_writer,
                    persisted_transitions,
                    &session_id,
                    &attempt_patch_ids,
                    &apply_failed_attempt_indices,
                    &mut metric_rows,
                )?;
                prior_feedback = Some(feedback);
            }
            let decision = governor.decide(governor_mode, AttemptVerdict::Inconclusive, false);
            if decision == GovernorDecision::Continue {
                continue;
            }
            break decision;
        }
        let _ = any_changed; // used implicitly via all_patch_failed

        // ---- Build source strings ----
        let original_concat = build_source_concat(&original_files);
        let review_prompt_concat = if config.patch_path.is_some() {
            let narrowed = review_prompt_source_concat(&repo.source_files, &original_files, &patch_files);
            if narrowed.trim().is_empty() {
                original_concat.clone()
            } else {
                narrowed
            }
        } else {
            original_concat.clone()
        };

        // ---- 3. Attacker ----
        // [VANTA-WHY] `VANTA_ATTACKER=off` (the `baseline` configuration,
        // plans/bugsinpy.md §4.2) asks a different question: what does this
        // candidate look like with NO adversarial tests at all? The repo's own
        // suite and static analysis are then the whole question. The phase is
        // SKIPPED rather than run-and-emptied: an `attacker.summary` row saying
        // `generated_tests=0` would assert "the Attacker ran and found nothing",
        // which is a different fact from "the Attacker did not run", and no
        // reader could tell the two apart in the artifact afterwards.
        //
        // Nothing about the verdict machinery changes here: with no generated
        // tests, `all_passed` is vacuously true and `patched_classes` is empty,
        // so `attempt_verdict` can only reject on a native regression or a
        // blocking static finding — which is exactly the native+static-only
        // verdict this configuration is defined by. `attempt_verdict` itself is
        // untouched (C2).
        let attacker_tests: Vec<GeneratedTest> = if !attacker_enabled {
            println!(
                "\n▌ ATTACKER phase skipped (VANTA_ATTACKER=off): the repo's own suite and static \
                 analysis are the whole question."
            );
            Vec::new()
        } else {
            println!("\n▌ ATTACKER phase...");
            // [VANTA-WHY] Spec §7 "Import paths" row: append the detected import
            // path to the issue text so the Attacker writes tests with the correct
            // `from heretic.foo import bar` instead of `from foo import bar`.
            let attacker_issue = match &import_hint {
                Some(hint) => format!(
                    "{}\n\n[IMPORT PATH HINT] This repo uses import path: `{}`. Generated tests MUST import using this path. If the repo has a src/ layout, the package is under src/.",
                    repo.issue_text, hint
                ),
                None => repo.issue_text.clone(),
            };
            let attacker_request = AttackerRequest {
                patch_text: patch_text.clone(),
                original_source: review_prompt_concat.clone(),
                issue_text: attacker_issue,
                // [VANTA-WHY] Round >= 2: the Attacker is shown the probes already in
                // the shared set so it attacks somewhere new instead of re-deriving
                // the same boundary test. Empty on round 1.
                existing_tests: shared_tests
                    .iter()
                    .map(|t| SharedProbe {
                        test_name: t.test_name.clone(),
                        test_source: t.test_source.clone(),
                    })
                    .collect(),
            };
            let attacker_response = match run_with_spinner("Attacker generating tests", async {
                attacker_generate_tests(&cfg, &attacker_request).await
            })
            .await
            {
                Ok(response) => response,
                Err(error) => {
                    infra_failure_error = Some(format!("Attacker LLM call failed: {error}"));
                    break GovernorDecision::InfraFailure;
                }
            };
            // [VANTA-WHY] A probe whose body is already in the shared set adds no
            // coverage: it will be BATE-validated and patched-run again, and the
            // comparison it produces is a self-transition. Identity is the
            // content-based `normalized_test_body_hash`, so a rename is still a
            // duplicate. The count of drops is logged per round — silently keeping
            // them would make a stalled Attacker look like a productive one.
            let (generated_tests, dropped_duplicates) =
                drop_shared_duplicates(&attacker_response.tests, &shared_tests);
            if !dropped_duplicates.is_empty() {
                println!(
                    "  ⚠ Attacker repeated {} probe(s) already in the shared set: {:?}",
                    dropped_duplicates.len(),
                    dropped_duplicates
                );
            }
            session_log.append(SessionRecord {
                session_id: session_id.clone(),
                schema_version: Some(SCHEMA_VERSION),
                patch_attempt_id: Some(patch_attempt_id.clone()),
                testing_iteration_id: Some(testing_iteration_id.clone()),
                test_id: None,
                phase: "attacker.dedup".to_string(),
                summary: format!(
                    "shared_set_size={} generated={} dropped_duplicates={} remaining={} dropped=[{}]",
                    shared_tests.len(),
                    attacker_response.tests.len(),
                    dropped_duplicates.len(),
                    generated_tests.len(),
                    dropped_duplicates.join(",")
                ),
                timestamp: now_iso(),
                benchmark_instance: None,
                benchmark_label: None,
            })?;
            // [VANTA-WHY] The patch-blind intent audit runs here, before BATE: only
            // an audit that cannot see the patch can tell whether the ISSUE asks for
            // the behaviour a non-preservation test asserts. An Attacker that read
            // the patch could always label the patch's own side effect
            // "FeatureAcceptance" and then pass its own acceptance test. The
            // quote-substring evidence is recorded, but the audit decides.
            let (attacker_tests, initial_relabels, initial_intent_checks) = audit_tests_intent(
                &cfg,
                &repo.issue_text,
                &review_prompt_concat,
                &generated_tests,
            )
            .await;
            let intent_scrapped = initial_intent_checks.iter().filter(|c| c.scrapped).count();
            println!(
                "  ✔ Attacker: {} tests ({} generated, {} duplicate(s) dropped, {} scrapped by the intent check)",
                attacker_tests.len(),
                attacker_response.tests.len(),
                dropped_duplicates.len(),
                intent_scrapped
            );
            for t in &attacker_tests {
                println!("    - {} (intent={:?})", t.test_name, t.intent);
            }
            session_log.append(SessionRecord {
                session_id: session_id.clone(),
                schema_version: Some(SCHEMA_VERSION),
                patch_attempt_id: Some(patch_attempt_id.clone()),
                testing_iteration_id: Some(testing_iteration_id.clone()),
                test_id: None,
                phase: "attacker.summary".to_string(),
                summary: format!(
                    "generated_tests={} dropped_duplicates={} intent_scrapped={} relabels={} valid_tests={}",
                    attacker_response.tests.len(),
                    dropped_duplicates.len(),
                    intent_scrapped,
                    initial_relabels.len(),
                    attacker_tests.len()
                ),
                timestamp: now_iso(),
                benchmark_instance: None,
                benchmark_label: None,
            })?;
            log_intent_checks(&mut session_log, &session_id, &patch_attempt_id, &testing_iteration_id, "attacker.generate", &initial_intent_checks)?;
            for relabel in &initial_relabels {
                session_log.append(SessionRecord {
                    session_id: session_id.clone(),
                    schema_version: Some(SCHEMA_VERSION),
                    patch_attempt_id: Some(patch_attempt_id.clone()),
                    testing_iteration_id: Some(testing_iteration_id.clone()),
                    test_id: Some(fresh_test_id()),
                    phase: format!("attacker.relabel[{}]", relabel.test_name),
                    summary: format!(
                        "source=attacker.generate from={:?} to={:?} reason={}",
                        relabel.from_intent, relabel.to_intent, relabel.reason
                    ),
                    timestamp: now_iso(),
                    benchmark_instance: None,
                    benchmark_label: None,
                })?;
            }
            attacker_tests
        };

        // ---- 3b. Self-heal dependencies the manifest did not declare ----
        // [VANTA-WHY] The manifest says what the repo DECLARED; the tests may
        // import more. Probe inside the VM and install anything missing into a
        // derived checkpoint BEFORE BATE, otherwise every test dies at import
        // and BATE sees `Error` for all of them.
        harness::heal_dependencies(
            &original_concat,
            &attacker_tests,
            &test_cmd,
            import_hint.as_deref(),
        )
        .await;

        // ---- 4. BATE: baseline validation (validate + Attacker revise) ----
        // [VANTA-WHY] The shared BATE phase owns BOTH retry layers: the flaky
        // pre-check + 3 same-test baseline reruns (Layer 1), and up to 3 Attacker
        // revisions when the baseline still disagrees with the declared intent
        // (Layer 2). M3's local copy had dropped Layer 2, so a test that merely
        // declared the wrong intent was scrapped instead of being fixed.
        //
        // [VANTA-WHY] BATE runs on the patch-FREE ORIGINAL tree
        // (`bate_baseline_concat`), unlike the patched-side Attacker run, which
        // runs on the graded tree. BATE's question is "does this test fail on the
        // code as it was?", so the fix must not be present — a tree containing the
        // patch would make a correct probe look wrong. This is also why BATE
        // needs no sanitizing of its own: it never receives a patch-added
        // `conftest.py` or an edited test in the first place.
        println!("\n▌ BATE: validating tests against baseline...");
        let bate_run = run_with_spinner("BATE validating tests against baseline", async {
            run_bate(
                &cfg,
                &repo.issue_text,
                &bate_baseline_concat(&original_files),
                &review_prompt_concat,
                &attacker_tests,
                &test_cmd,
            )
            .await
        })
        .await;
        let bate_report = bate_run.report;
        let valid_tests: Vec<GeneratedTest> = bate_run.valid_tests;
        let bate_relabels = bate_run.relabel_events;
        let bate_intent_checks = bate_run.intent_checks;
        let discard_rate = if attacker_tests.is_empty() {
            0.0
        } else {
            bate_report.scrapped_tests.len() as f64 / attacker_tests.len() as f64
        };
        println!(
            "  BATE: {} valid, {} scrapped (discard rate {:.1}%)",
            bate_report.valid_tests.len(),
            bate_report.scrapped_tests.len(),
            discard_rate * 100.0
        );
        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: Some(testing_iteration_id.clone()),
            test_id: None,
            phase: "bate.summary".to_string(),
            summary: format!(
                "valid={} scrapped={} relabels={} discard_rate_pct={:.1}",
                bate_report.valid_tests.len(),
                bate_report.scrapped_tests.len(),
                bate_relabels.len(),
                discard_rate * 100.0
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
        log_intent_checks(&mut session_log, &session_id, &patch_attempt_id, &testing_iteration_id, "attacker.revise", &bate_intent_checks)?;
        for relabel in &bate_relabels {
            session_log.append(SessionRecord {
                session_id: session_id.clone(),
                schema_version: Some(SCHEMA_VERSION),
                patch_attempt_id: Some(patch_attempt_id.clone()),
                testing_iteration_id: Some(testing_iteration_id.clone()),
                test_id: Some(fresh_test_id()),
                phase: format!("attacker.relabel[{}]", relabel.test_name),
                summary: format!(
                    "source=attacker.revise from={:?} to={:?} reason={}",
                    relabel.from_intent, relabel.to_intent, relabel.reason
                ),
                timestamp: now_iso(),
                benchmark_instance: None,
                benchmark_label: None,
            })?;
        }

        let scrapped_names: Vec<String> = bate_report
            .scrapped_tests
            .iter()
            .map(|(n, _)| n.clone())
            .collect();
        for (name, reason) in &bate_report.scrapped_tests {
            session_log.append(SessionRecord {
                session_id: session_id.clone(),
                schema_version: Some(SCHEMA_VERSION),
                patch_attempt_id: Some(patch_attempt_id.clone()),
                testing_iteration_id: Some(testing_iteration_id.clone()),
                test_id: Some(fresh_test_id()),
                phase: format!("bate.scrap[{name}]"),
                summary: reason.clone(),
                timestamp: now_iso(),
                benchmark_instance: None,
                benchmark_label: None,
            })?;
        }

        // [VANTA-WHY] With the Attacker off there is never a generated test for
        // BATE to validate, so `valid_tests` and `shared_tests` are permanently
        // empty. Entering this branch would turn that permanent emptiness into
        // `Inconclusive` → `MaxAttemptsReached` — a verdict about the harness,
        // not about the candidate — when the entire point of `baseline` is the
        // native-and-static verdict. Skipping it lets the attempt fall through
        // to the graded run, where the native suite and static analysis are the
        // evidence (§4.2). `valid_tests` stays empty, so the patched run runs no
        // generated tests and `all_passed` is vacuously true — which is exactly
        // what `attempt_verdict` needs to decide on native+static alone.
        if attacker_enabled && valid_tests.is_empty() && shared_tests.is_empty() {
            // No valid new tests AND no shared set from previous attempts — nothing to run.
            println!("  ⚠ No valid tests after BATE and no shared set — cannot verify patch");
            if config.patch_path.is_none() {
                prior_feedback = Some(format!(
                    "All generated tests failed baseline validation (BATE). The scrapped tests were: {:?}. Write tests that actually reproduce the bug on the original code.",
                    scrapped_names
                ));
            }
            let decision = governor.decide(
                governor_mode,
                AttemptVerdict::Inconclusive,
                true,
            );
            if decision != GovernorDecision::Continue {
                break decision;
            }
            if config.patch_path.is_some() {
                println!("  → Retrying verification with the provided patch");
            }
            continue;
        }
        if attacker_enabled && valid_tests.is_empty() {
            // New tests failed BATE, but we have a shared set from previous attempts.
            // Run only the shared set (no new tests to fold in).
            println!("  ⚠ No new valid tests after BATE — running shared set only");
        }

        // ---- 5. Patched run (shared set + new tests) ----
        // [VANTA-WHY] Spec §8: run BOTH the fixed shared set (for EIR/ECR
        // transition measurement) AND the newly generated tests (for immediate rejection)
        // against the patched code. The shared set is what gets compared
        // attempt-to-attempt; new tests can reject on the spot but only join
        // the shared set for the NEXT comparison.
        println!(
            "\n▌ PATCHED run: shared set ({} tests) + new tests ({} tests)...",
            shared_tests.len(),
            valid_tests.len()
        );

        let patched_concat = build_source_concat(&patched_files);
        let snapshot_root = debug_dir.join(format!("attempt_{}_patched_tree", attempt + 1));
        let _ = harness::write_tree_snapshot(&snapshot_root, &patched_concat);

        // [VANTA-WHY] The Attacker's generated tests must be graded on the SAME
        // sanitized tree the native suite uses, so a patch cannot silence them
        // either. `patched_concat` above is the patch's own tree, tests and all:
        // a patch-added root `conftest.py` carrying a `pytest_runtest_makereport`
        // hook can rewrite every outcome to "passed", and a patch-added
        // `pytest_collection_modifyitems` (or a narrowed `addopts`) can deselect
        // tests outright. Running the generated tests there let the candidate
        // grade itself — the same hole the native gate closed for the repo's own
        // suite. The graded tree pins every collected test and every
        // `conftest.py` back to its original contents and drops patch-added test
        // files, so what the patch claims about its own tests cannot reach the
        // run that judges it.
        let native_tree = harness::native_tree_with_original_tests(
            &original_files,
            &patched_files,
            &patch_deleted_paths(&patch_text),
        );
        let graded_concat = build_source_concat(&native_tree.files);
        // [VANTA-WHY] Snapshot the graded tree per attempt. `--rerun-signals`
        // reads it back for a session that predates this key, and it is the ONLY
        // artifact that reproduces the exact tree the Attacker ran on (the
        // `_patched_tree` snapshot above is the patch's own, tests included).
        let graded_snapshot_root =
            debug_dir.join(format!("attempt_{}_graded_tree", attempt + 1));
        let _ = harness::write_tree_snapshot(&graded_snapshot_root, &graded_concat);

        // Build the combined test list: shared first, then new.
        let mut tests_to_run: Vec<&GeneratedTest> = Vec::new();
        for t in &shared_tests {
            tests_to_run.push(t);
        }
        for t in &valid_tests {
            tests_to_run.push(t);
        }

        let patched_results = run_with_spinner("Running tests on patched code", async {
            harness::run_tests_parallel_detailed(&graded_concat, &tests_to_run, &test_cmd).await
        })
        .await?;

        // Static analysis diff.
        //
        // [VANTA-WHY] This used to compare COUNTS. Patching one line of a file
        // that already had N findings could leave the count at N while swapping
        // which findings exist, so a fresh error would slip through unnoticed.
        // The set diff below keys on (file, code, message) with line numbers
        // stripped: the same diagnostic moving down the file is not a new
        // error, and a genuinely new one is always caught. Only the NEW
        // diagnostics reach the Diagnostician, so its Static payload names what
        // the patch introduced — not the repo's pre-existing backlog.
        let base_static =
            harness::run_static(&original_concat, &test_cmd, Some(&config.repo_path)).await;
        let patch_static =
            harness::run_static(&patched_concat, &test_cmd, Some(&config.repo_path)).await;
        let new_ruff = harness::new_diagnostics(&base_static.ruff, &patch_static.ruff);
        let new_mypy = harness::new_diagnostics(&base_static.mypy, &patch_static.mypy);
        let ruff_new = !new_ruff.is_empty();
        let mypy_new = !new_mypy.is_empty();
        let ruff_new_text = harness::render_diagnostics(&new_ruff);
        let mypy_new_text = harness::render_diagnostics(&new_mypy);
        // [VANTA-WHY] Fix 3 — a new finding in a file the patch TOUCHES is a
        // finding about the candidate, so review mode can reject on it. Scoped
        // to the touched files so a pre-existing error elsewhere in the tree
        // (the repo's backlog, which the patch did not create) cannot be
        // mistaken for the patch's doing.
        //
        // [VANTA-WHY] But "a new finding" is not the same as "the candidate is
        // broken": ruff's default ruleset is mostly style (E4 import hygiene,
        // E7 statement style, and the B/BLE/UP/SIM/TRY families when selected by
        // a repo config). Rejecting on those reports a candidate as broken when
        // it merely has a nit, which is a false finding about the model and burns
        // an attack round. Only ERROR-class findings block: ruff `F`/`E9` and
        // every mypy error (its parser accepts `: error:` lines only). The style
        // findings are still REPORTED — the Creator sees them and the session log
        // keeps them — they just do not reject.
        let touched_paths: Vec<String> = patch_files.keys().cloned().collect();
        // [VANTA-WHY] Item 4 of the heretic fix list: keep the NAME of every
        // finding that blocks, not just a bool. A rejection must be able to say
        // WHICH finding rejected the candidate — 40e84874 printed "fails 0
        // test(s)" while the real signal was a mypy error no artifact recorded.
        //
        // [VANTA-WHY] The diagnostics themselves are kept (not just their
        // rendered text) so the transition-row writer can key a Static row on
        // the finding's IDENTITY rather than on a string that carries the line
        // number the finding happened to sit on.
        let blocking_diags = blocking_static_diagnostics(&new_ruff, &new_mypy, &touched_paths);
        let blocking_static: Vec<String> =
            blocking_diags.iter().map(|d| d.render()).collect();
        let new_static_in_touched_files = !blocking_static.is_empty();
        println!("  Static: ruff_new={ruff_new}, mypy_new={mypy_new}");
        // [VANTA-WHY] The two booleans above are not enough to audit a rejection.
        // Print each NEW finding — tool, file, line, code, message — and keep it
        // for evidence.json, so "which finding rejected this round" is answerable
        // from the artifacts alone (the 40e84874 gap).
        for finding in record_new_static_findings(
            &mut static_findings_records,
            attempt,
            &new_ruff,
            &new_mypy,
            &blocking_static,
        ) {
            println!("    {finding}");
        }
        // [VANTA-WHY] A blocking finding is evidence about the candidate, so it
        // produces a transition row like any other: present at N and gone at
        // N+1 is a CORRECTION (Failed -> Passed under `evidence_source: Static`),
        // which is what makes a static error fixed in the next attempt visible
        // instead of structurally excluded. Only blocking (error-class, touched)
        // findings participate — a style nit never moved the metric.
        let current_static = AttemptStaticOutcome::from_findings(&blocking_diags);
        if let Some((from_attempt, from_patch, previous)) = &previous_static {
            if from_patch != &patch_attempt_id {
                let rows = static_transition_rows(
                    &session_id,
                    from_patch,
                    &patch_attempt_id,
                    *from_attempt,
                    attempt,
                    previous,
                    &current_static,
                );
                append_derived_rows(&mut transition_writer, rows, &mut metric_rows)?;
            }
        }
        previous_static = Some((attempt, patch_attempt_id.clone(), current_static));

        // ---- 5b. Native suite gate — the repo's OWN tests ----
        // [VANTA-WHY] Generated tests depend on what the Attacker sampled; the
        // repo's own suite does not. A test that passes on the original and
        // fails on the patch is a regression even when no generated test covers
        // that input. This is its own `native_suite` section — not part of the
        // shared set, never counted in EIR/ECR — because it is not a transition
        // measurement.
        println!("\n▌ NATIVE suite: repo's own tests (original vs patch)...");
        // [VANTA-WHY] The native suite grades the patched SOURCE with the
        // ORIGINAL tests. Session 888b6931 approved a patch that broke
        // `get_responses` because the Creator also edited the very test that
        // catches it, and the native run collected the edited file. The patch's
        // test edits are discarded here — existing tests restored to their
        // original contents, a deleted test restored, an added test file left
        // out of the baseline — so the patch cannot grade itself.
        //
        // `native_tree` / `graded_concat` were built before the Attacker run (see
        // step 5) so the generated tests and the repo's own suite are graded on
        // the SAME sanitized tree; the two can never disagree about which tests
        // were allowed to run.
        let native_patched_concat = graded_concat.clone();
        modified_existing_tests = native_tree.modified_existing_tests.clone();
        removed_existing_tests = native_tree.removed_existing_tests.clone();
        added_test_files = native_tree.added_test_files.clone();
        modified_test_config = modified_test_config_records(&patch_files);
        if !modified_existing_tests.is_empty()
            || !removed_existing_tests.is_empty()
            || !added_test_files.is_empty()
            || !modified_test_config.is_empty()
        {
            println!(
                "  Test integrity: native_test_source=original modified_existing_tests=[{}] \
                 removed_existing_tests=[{}] added_test_files=[{}] modified_test_config=[{}]",
                modified_existing_tests.join(", "),
                removed_existing_tests.join(", "),
                added_test_files.join(", "),
                modified_test_config
                    .iter()
                    .filter_map(|v| v["path"].as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let native_original = harness::run_native_suite(&original_concat, &test_cmd).await;
        let native_patched = harness::run_native_suite(&native_patched_concat, &test_cmd).await;
        let mut native_regressions =
            harness::native_regressions(&native_original.tests, &native_patched.tests);
        // [VANTA-WHY] A suite that ran N>0 tests on the original and collects
        // fewer on the patch is a regression even if every collected test
        // passes: the missing ones never ran. A syntax error stops pytest at
        // collection, so a patch that breaks imports reads as "4 tests → 0
        // tests, 0 regressions" — the single most dangerous false-clean in the
        // pipeline. `native_regressions` already names the original tests that
        // vanished; `native_suite_collapsed` catches the edge case where the
        // originals that vanished had not been passing either.
        let native_collapsed = harness::native_suite_collapsed(&native_original, &native_patched);
        if native_collapsed {
            for (name, _) in &native_original.tests {
                if !native_patched.tests.iter().any(|(n, _)| n == name)
                    && !native_regressions.iter().any(|n| n == name)
                {
                    native_regressions.push(name.clone());
                }
            }
        }
        let native_regression = !native_regressions.is_empty();
        println!(
            "  Native: original ran={} ({} tests), patched ran={} ({} tests), {} regression(s){}",
            native_original.ran,
            native_original.tests.len(),
            native_patched.ran,
            native_patched.tests.len(),
            native_regressions.len(),
            if native_collapsed { " [SUITE COLLAPSED]" } else { "" }
        );
        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: Some(testing_iteration_id.clone()),
            test_id: None,
            phase: "native_suite.summary".to_string(),
            summary: format!(
                "ran_original={} ran_patched={} original_tests={} patched_tests={} regressions={}",
                native_original.ran,
                native_patched.ran,
                native_original.tests.len(),
                native_patched.tests.len(),
                native_regressions.len()
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
        for (nodeid, original_result) in &native_original.tests {
            let patched_result = native_patched
                .tests
                .iter()
                .find(|(n, _)| n == nodeid)
                .map(|(_, r)| r);
            // [VANTA-WHY] A native test already failing on the original is
            // RECORDED, not blocking: the patch did not break it.
            session_log.append(SessionRecord {
                session_id: session_id.clone(),
                schema_version: Some(SCHEMA_VERSION),
                patch_attempt_id: Some(patch_attempt_id.clone()),
                testing_iteration_id: Some(testing_iteration_id.clone()),
                test_id: Some(fresh_test_id()),
                phase: format!("native_suite.test[{nodeid}]"),
                summary: format!(
                    "original={original_result:?} patched={:?} regression={}",
                    patched_result,
                    native_regressions.iter().any(|n| n == nodeid)
                ),
                timestamp: now_iso(),
                benchmark_instance: None,
                benchmark_label: None,
            })?;
        }

        // [VANTA-WHY] A native regression is evidence about the candidate, so it
        // produces a transition row like any other: one row per repo test that
        // ran on BOTH attempts, `Failed -> Passed` when the next attempt fixed a
        // regression the previous one introduced (the correction 4d55123d hid),
        // `Passed -> Failed` when it introduced one. Emitted only between two
        // DIFFERENT patches, so a review round (same candidate) stays excluded,
        // exactly as before.
        let current_native =
            AttemptNativeOutcome::from_suites(&native_original, &native_patched);
        if let Some((from_attempt, from_patch, previous)) = &previous_native {
            if from_patch != &patch_attempt_id {
                let rows = native_transition_rows(
                    &session_id,
                    from_patch,
                    &patch_attempt_id,
                    *from_attempt,
                    attempt,
                    previous,
                    &current_native,
                );
                append_derived_rows(&mut transition_writer, rows, &mut metric_rows)?;
            }
        }
        previous_native = Some((attempt, patch_attempt_id.clone(), current_native));
        // [VANTA-WHY] Keep this attempt's regressions in evidence.json, keyed by
        // attempt, so the artifact answers "which repo test did this attempt
        // break?" without replaying the console.
        native_regression_records.push(serde_json::json!({
            "attempt": attempt + 1,
            "patch_attempt_id": patch_attempt_id.as_str(),
            "regressions": &native_regressions,
        }));

        // ---- 6. Diagnostician + build evidence ----
        println!("\n▌ DIAGNOSTICIAN: classifying failures...");

        // [VANTA-WHY] Split results into shared (for EIR/ECR) and new (for
        // immediate rejection). On the first attempt shared_tests is empty, so
        // ALL results become the shared set.
        let shared_names: std::collections::HashSet<String> =
            shared_tests.iter().map(|t| t.test_name.clone()).collect();

        let mut shared_observations: Vec<SharedObservation> = Vec::new();
        let mut new_observations: Vec<SharedObservation> = Vec::new();
        let mut attempt_evidence: Vec<TestEvidence> = Vec::new();

        for test in &tests_to_run {
            let execution = patched_results
                .get(&test.test_name)
                .cloned()
                .unwrap_or_else(TestExecutionDetails::infrastructure_error);
            let result = execution.result.clone();

            let failure_class = if result == TestResult::Passed {
                None
            } else {
                let net_alive = crate::sandbox::network_alive().await;
                // [VANTA-WHY] The shared Diagnostician owns the Spec §7 Phase-2
                // remediation (reset+rerun for Infrastructure, simplify for
                // Unknown) that M3's inline call had dropped.
                let diag = diagnose_failure(
                    test,
                    &execution,
                    ruff_new,
                    mypy_new,
                    &ruff_new_text,
                    &mypy_new_text,
                    !net_alive,
                    &graded_concat,
                    &cfg,
                    &review_prompt_concat,
                    &test_cmd,
                )
                .await;
                println!(
                    "    {} → {:?} ({:?})",
                    test.test_name, result, diag.failure_class
                );
                let safe_name = test
                    .test_name
                    .replace(|c: char| !c.is_alphanumeric() && c != '_', "_");
                let diag_path = debug_dir.join(format!(
                    "attempt_{}_diagnostic_{}.json",
                    attempt + 1,
                    safe_name
                ));
                let _ = fs::write(
                    &diag_path,
                    serde_json::to_string_pretty(&serde_json::json!({
                        "test_name": test.test_name,
                        "result": format!("{:?}", result),
                        "failure_class": format!("{:?}", diag.failure_class),
                        "reason": diag.reason,
                        "structured_payload": diag.structured_payload,
                    }))
                    .unwrap(),
                );
                Some(diag.failure_class.unwrap_or(FailureClass::Unknown))
            };

            // [VANTA-WHY] Split this attempt's results into the carried-over
            // shared set (compared against the previous attempt) and the new
            // tests (folded in for the NEXT comparison). Both carry the failure
            // class, because the Governor excludes non-Logic failures from
            // EIR/ECR at calculation time — a single transient Infrastructure
            // blip must never be recorded as an ordinary fail.
            let observation = SharedObservation {
                test_name: test.test_name.clone(),
                test_source: test.test_source.clone(),
                result: result.clone(),
                failure_class: failure_class.clone(),
                // Provenance travels with the observation so each persisted
                // transition row can be re-selected by source later (the
                // oracle-only robustness check) without another lookup.
                evidence_source: EvidenceSource::AttackerGenerated,
                test_intent: test.intent.clone(),
            };
            if shared_names.contains(&test.test_name) || shared_tests.is_empty() {
                shared_observations.push(observation);
            } else {
                new_observations.push(observation);
            }

            attempt_evidence.push(TestEvidence {
                test_name: test.test_name.clone(),
                outcome: match result {
                    TestResult::Passed => VerificationOutcome::VerifiedWithinBounds,
                    _ => VerificationOutcome::Counterexample,
                },
                failure_class,
                structured_payload: if result == TestResult::Passed {
                    None
                } else {
                    let safe_name = test
                        .test_name
                        .replace(|c: char| !c.is_alphanumeric() && c != '_', "_");
                    let diag_path = debug_dir.join(format!(
                        "attempt_{}_diagnostic_{}.json",
                        attempt + 1,
                        safe_name
                    ));
                    read_json_value_from_debug(&diag_path)
                        .and_then(|v| v.get("structured_payload").cloned())
                },
                intent: test.intent.clone(),
                evidence_source: EvidenceSource::AttackerGenerated,
            });
        }

        // ---- 7. Harness-collapse check (MUST precede governor bookkeeping) ----
        // [VANTA-WHY] A censored attempt must never reach EIR/ECR. The Governor
        // now excludes non-Logic transitions at calculation time, but a collapse
        // is a claim about the whole attempt (the harness never produced a
        // runnable tree), so it is still detected first and its bookkeeping
        // skipped entirely rather than recorded as evidence.
        let patched_classes: Vec<Option<FailureClass>> = attempt_evidence
            .iter()
            .map(|e| e.failure_class.clone())
            .collect();
        let harness_collapsed = harness_infrastructure_collapse(&patched_classes);
        if harness_collapsed {
            harness_collapse_streak += 1;
        } else {
            harness_collapse_streak = 0;
        }

        // ---- 8. Governor: record shared results, fold new tests, decide ----
        // [VANTA-WHY] record_shared_results computes transitions by comparing
        // this attempt's shared set against the previous attempt's shared set,
        // keyed by normalised body hash so a renamed probe is the same probe.
        // Then fold_new_tests adds this attempt's new tests into the shared set
        // so they participate in the NEXT comparison.
        //
        // [VANTA-WHY] This runs in --patch mode too. In that mode the candidate
        // is fixed, so the comparison measures ATTACK DEPTH against one candidate
        // (a review round) rather than a patch transition — real evidence about
        // how a growing question set behaves, whose rows are identifiable by
        // from_patch_attempt_id == to_patch_attempt_id.
        if !harness_collapsed {
            governor.record_shared_results(shared_observations);
            governor.fold_new_tests(&new_observations);
        }
        // [VANTA-WHY] Flush the rows this attempt produced, immediately, before
        // anything else can fail. A later HarnessFailure — or a full disk — then
        // costs only the work not yet done, never the rows already computed.
        persisted_transitions = flush_transition_rows(
            &governor,
            &mut transition_writer,
            persisted_transitions,
            &session_id,
            &attempt_patch_ids,
            &apply_failed_attempt_indices,
            &mut metric_rows,
        )?;
        // Persist the actual GeneratedTest objects for re-running next attempt.
        // A censored attempt contributes no usable evidence, so it does not grow
        // the shared corpus either — the retry then behaves like a fresh attempt.
        if !harness_collapsed {
            for t in &valid_tests {
                if !shared_names.contains(&t.test_name) {
                    shared_tests.push(t.clone());
                }
            }
        }
        // [VANTA-WHY] Persist the shared set to disk so a resumed run (or M5
        // re-analysis) replays the IDENTICAL tests. Spec §10 item 2.
        if let Ok(json) = serde_json::to_string_pretty(&shared_tests) {
            let _ = fs::write(&shared_tests_path, json);
        }

        let all_passed = patched_results
            .values()
            .all(|details| details.result == TestResult::Passed);
        // [VANTA-WHY] The Governor needs three-way evidence, not a bool. In review
        // mode a Logic failure is a REJECTION; an Infrastructure-only round is
        // not a finding at all and must never be reported as one.
        let verdict = attempt_verdict(
            all_passed,
            &patched_classes,
            native_regression,
            new_static_in_touched_files,
            governor_mode,
        );
        let governor_decision = governor.decide(governor_mode, verdict, true);
        let decision = if harness_collapsed {
            let next = collapse_decision(
                harness_collapse_streak,
                governor_decision == GovernorDecision::Continue,
            );
            if next == GovernorDecision::HarnessFailure {
                eprintln!(
                    "  ✖ HARNESS FAILURE: attempt {} collapsed again — all {} patched test(s) \
                     classified Infrastructure. Reproducible, so this is a broken harness, not a \
                     transient blip. Aborting.",
                    attempt + 1,
                    patched_classes.len()
                );
            } else {
                eprintln!(
                    "  ⚠ HARNESS WARNING: attempt {} had all {} patched test(s) classified \
                     Infrastructure. Retrying once to tell a transient blip from a broken harness.",
                    attempt + 1,
                    patched_classes.len()
                );
            }
            next
        } else {
            governor_decision
        };
        let mut failing_names: Vec<String> = attempt_evidence
            .iter()
            .filter(|e| e.outcome != VerificationOutcome::VerifiedWithinBounds)
            .map(|e| e.test_name.clone())
            .collect();
        // [VANTA-WHY] A native-suite regression is a failing test too: when the
        // generated tests all pass, the rejection is otherwise empty and the
        // reader cannot tell which repo test the patch broke.
        failing_names.extend(native_regressions.iter().cloned());
        if config.patch_path.is_some() {
            // [VANTA-WHY] These rows measure attack depth against an UNCHANGED
            // candidate, not a patch transition, so they are excluded from
            // EIR/ECR by `from_patch_attempt_id == to_patch_attempt_id`. But on a
            // round-1 stop there ARE no rows at all: a comparison needs two
            // attempts, so a terminal first round records nothing. Printing
            // "rows recorded" there sent the reader looking for a file the run
            // never wrote.
            if persisted_transitions == 0 {
                println!(
                    "\n▌ GOVERNOR: decision={:?}, review_round={} (no transition rows written — this round ended the review, and a transition needs two attempts)",
                    decision,
                    attempt + 1
                );
            } else {
                println!(
                    "\n▌ GOVERNOR: decision={:?}, review_round={} ({} transition row(s) recorded as attack depth, excluded from EIR/ECR by from_patch_attempt_id == to_patch_attempt_id)",
                    decision,
                    attempt + 1,
                    persisted_transitions
                );
            }
            if decision == GovernorDecision::Rejected {
                println!(
                    "  ✖ REJECTED — {}",
                    rejection_signal_message(
                        &native_regressions,
                        &blocking_static,
                        &failing_names,
                        &modified_existing_tests
                    )
                );
            }
        } else {
            // The SAME function the M5 analysis layer runs over the on-disk rows,
            // so the live number and the paper's number cannot drift apart.
            // [VANTA-WHY] CONDITIONAL ECR/EIR: ECR = rows Failed at N that Pass
            // at N+1 ÷ rows Failed at N; EIR = rows Passed at N that Fail at N+1
            // ÷ rows Passed at N. The old per-row divisor (every countable row)
            // made ECR shrink with the size of the repo's suite — session
            // 4d55123d read ECR(Native)=1/5 for one fix in five native tests,
            // and an 800-test BugsInPy repo would read ~0.1% for a real fix.
            // [VANTA-WHY] Combined AND per source: a pooled ECR hides WHICH
            // evidence moved it, and the whole point of counting native rows is
            // that session 4d55123d's correction was a repo test, not an
            // AttackerGenerated one. A zero denominator prints `n/a`, never
            // `0.00%` — "nothing countable" and "nothing corrected" are opposite
            // claims about the run.
            let snap = ecr_eir_from_rows(&metric_rows);
            println!(
                "\n▌ GOVERNOR: decision={:?}, ECR={} EIR={}",
                decision,
                snap.ecr_text(),
                snap.eir_text()
            );
            println!("            per source: {}", snap.per_source_text());
        }

        final_evidence = attempt_evidence;
        for e in &final_evidence {
            let class = e
                .failure_class
                .as_ref()
                .map(|c| format!("{c:?}"))
                .unwrap_or_else(|| "None".to_string());
            session_log.append(SessionRecord {
                session_id: session_id.clone(),
                schema_version: Some(SCHEMA_VERSION),
                patch_attempt_id: Some(patch_attempt_id.clone()),
                testing_iteration_id: Some(testing_iteration_id.clone()),
                test_id: Some(fresh_test_id()),
                phase: format!("patched.result[{}]", e.test_name),
                summary: format!(
                    "outcome={:?} intent={:?} failure_class={}",
                    e.outcome, e.intent, class
                ),
                timestamp: now_iso(),
                benchmark_instance: None,
                benchmark_label: None,
            })?;
        }
        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: Some(testing_iteration_id.clone()),
            test_id: None,
            phase: "attempt.summary".to_string(),
            summary: format!(
                "attempt={} decision={:?} valid_tests={} shared_tests={} all_passed={}{}",
                attempt + 1,
                decision,
                valid_tests.len(),
                shared_tests.len(),
                all_passed,
                // [VANTA-WHY] Mark the attempt as censored in the durable log, not
                // just on stderr. A run that only prints a warning today leaves M5
                // with a run that stops partway through and no recorded reason.
                if harness_collapsed {
                    " censored_from_eir_ecr=true"
                } else {
                    ""
                }
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;

        match decision {
            GovernorDecision::AllTestsPassed => {
                final_patched_files = Some(patched_files);
                // Collect accepted tests (test sources) for the PR.
                final_accepted_tests = Some(render_committed_test_file(&valid_tests));
                break decision;
            }
            GovernorDecision::MaxAttemptsReached => {
                final_patched_files = Some(patched_files);
                break decision;
            }
            GovernorDecision::Rejected => {
                // [VANTA-WHY] A review-mode rejection is a finding, not a budget
                // outcome: the candidate demonstrably fails. It is recorded with
                // the candidate attached so the failing tests can be read back
                // from the evidence, and it never reaches the Gatekeeper — there
                // is nothing to approve.
                final_patched_files = Some(patched_files);
                break decision;
            }
            GovernorDecision::PatchApplicationFailed => {
                break decision;
            }
            GovernorDecision::HarnessFailure => {
                // [VANTA-WHY] No retry: the failure is the setup, not the patch.
                // Rerunning would burn a VM boot and re-produce the same all-
                // Infrastructure attempt.
                break decision;
            }
            GovernorDecision::InfraFailure => {
                // [VANTA-WHY] Reached only when an LLM call fails, which the two
                // call sites above handle by breaking out with the error text
                // already captured. Kept as an arm so the match stays exhaustive
                // and so a future LLM call site cannot silently fall through to
                // the Creator-feedback path with no patch to feed back.
                break decision;
            }
            GovernorDecision::Continue => {
                // [VANTA-WHY] With `VANTA_ATTACKER=off` this list is empty by
                // construction, not by filtering: `final_evidence` is built from
                // the graded run of the generated tests, and the Attacker-off
                // path never produces one. So `baseline`'s Creator is fed the
                // native regressions and the blocking static findings and nothing
                // else — the configuration §4.2 defines — and there is no
                // generated-test signal that could be dropped from it silently.
                let failed: Vec<&TestEvidence> = final_evidence
                    .iter()
                    .filter(|e| e.outcome != VerificationOutcome::VerifiedWithinBounds)
                    .collect();
                if config.patch_path.is_some() {
                    println!(
                        "  → Retrying verification with the provided patch after {} failure(s)",
                        failed.len()
                    );
                } else {
                    // Build structured feedback for Creator.
                    // [VANTA-WHY] The count is every failure the Creator is being
                    // sent, not just the generated-test ones: a run whose ONLY
                    // signal was a native regression (or a blocking static error)
                    // used to print "Feeding 0 failure(s)" while handing the
                    // Creator a non-empty message — session 4d55123d.
                    let native_failure_count = native_regressions.len();
                    let static_failure_count = blocking_static.len();
                    prior_feedback = Some(build_creator_feedback(
                        &failed,
                        ruff_new,
                        mypy_new,
                        &native_regressions,
                        &native_patched,
                        &native_tree.modified_existing_tests,
                        &native_tree.removed_existing_tests,
                        &blocking_static,
                    ));
                    println!(
                        "  → Feeding {} failure(s) back to Creator for retry \
                         ({} generated, {} native, {} static)",
                        failed.len() + native_failure_count + static_failure_count,
                        failed.len(),
                        native_failure_count,
                        static_failure_count
                    );
                }
            }
        }
    };

    // ---- Final outcome ----
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  FINAL: {:?}", decision);
    println!("══════════════════════════════════════════════════════════════");

    // [VANTA-WHY] The error text travels WITH the verdict. "InfraFailure" on its
    // own says the run died but not why, and the reason is what the reader acts
    // on: an idle timeout, a prompt refused by the pre-flight guard and an
    // unreachable server all need different responses. Recorded as its own
    // phase so it cannot be mistaken for a statement about the candidate.
    if let Some(error) = &infra_failure_error {
        println!("  ✗ INFRASTRUCTURE FAILURE: {error}");
        let one_line = error
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" | ");
        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: attempt_patch_ids.last().cloned(),
            testing_iteration_id: None,
            test_id: None,
            phase: "run.infra_failure".to_string(),
            summary: format!(
                "final_decision=InfraFailure error={}",
                one_line.chars().take(1000).collect::<String>()
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
    }

    // [VANTA-WHY] "Fixed at attempt 1" is its own number, not a rate. A repair
    // that passes on the first candidate has no N→N+1 transition, so ECR/EIR
    // cannot see it by construction. Reporting it separately tells a reader what
    // share of defects the rates actually cover.
    //
    // [VANTA-WHY] The rates printed here are CONDITIONAL: ECR = rows Failed at N
    // that Pass at N+1 ÷ rows Failed at N, EIR = rows Passed at N that Fail at
    // N+1 ÷ rows Passed at N. `metric_rows.len()` is the churn count (countable
    // rows observed), NOT either denominator.
    //
    // [VANTA-WHY] The "outside ECR/EIR by construction" claim belongs ONLY to a
    // run that produced NO transition rows. Session 4d55123d wrote three
    // AttackerGenerated rows and the line still claimed there was no transition,
    // which is false about the file on disk. When rows exist, print the metric
    // they carry instead.
    let attempts_run = attempt_patch_ids.len();
    let fixed_at_attempt_1 = decision == GovernorDecision::AllTestsPassed && attempts_run == 1;

    // ---- Per-call token accounting (plans/bugsinpy.md §4.3) ----
    // [VANTA-WHY] The counts are measured by the server (see `llm::CallUsage`);
    // this is where they become a durable record. One row per chat call, so a
    // cost figure can be cut by call site, plus run totals in `run.summary` for
    // the batch tables. The `source` travels on every row rather than being
    // assumed, because a cost table has to be able to drop `estimate` rows
    // instead of averaging a measurement with a guess.
    let call_usages = drain_call_usages();
    for usage in &call_usages {
        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: attempt_patch_ids.last().cloned(),
            testing_iteration_id: None,
            test_id: None,
            phase: "call.usage".to_string(),
            summary: format!(
                "seq={} site={} prompt_tokens={} completion_tokens={} total_tokens={} source={}",
                usage.seq,
                usage.site,
                token_value(usage.prompt_tokens),
                token_value(usage.completion_tokens),
                token_value(usage.total_tokens),
                usage.source,
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
    }
    let (llm_calls, llm_calls_measured, llm_prompt_tokens, llm_completion_tokens) =
        token_totals(&call_usages);

    // ---- Provenance (plans/bugsinpy.md §4.4) ----
    // [VANTA-WHY] A session directory copied out of a batch has to be
    // self-describing: an env var is not provenance, because the shell that set
    // it is gone by the time anyone reads the record. The same fields also go
    // into the batch manifest (§7.3), and `metrics.py` groups by them (§9.1), so
    // any table can be cut by config × seed × model without re-running.
    //
    // [VANTA-WHY] `config` is DERIVED from the two switches rather than read
    // from a fourth env var, so the config the record claims cannot disagree
    // with the config the run executed (§4.2 defines the three configurations as
    // exactly these switch settings — one definition, `benchmark_config_name`).
    //
    // [VANTA-WHY] The sampling record is the REAL per-call schedule, not a
    // run-level scalar. The harness pins `temperature` and `max_tokens` per call
    // site — each agent's `extra_fields`, merged in `llm::post_chat` AFTER the
    // body defaults so the agent's value WINS — which is why one review run
    // samples at 0.1 (Creator), 0.2 (Attacker generate), 0.3 (Attacker
    // revise/simplify) and 0.0 (Intent Auditor). Those pins are part of v0.2's
    // behaviour; the harness must never override them, only record them.
    // `SAMPLING_SCHEDULE` names them (underscore keys — see its `[VANTA-WHY]`),
    // and `m3_session_log.rs` asserts the table equals the `temperature` /
    // `max_tokens` actually present in the request bodies. `top_p` is the ONE
    // parameter no call site pins, so its value is the SERVER's own default read
    // from `/props`. `seed` is what the HARNESS asked for. `unset` means the
    // server did not publish the key — never a plausible-looking default, which
    // would be a fabricated claim about a sampler nobody pinned.
    let config_name = benchmark_config_name(attacker_enabled, planner_enabled);
    let top_p_value = sampling_value(server_top_p_for_provenance(&cfg).await);
    let seed_value = cfg
        .seed
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unset".to_string());
    // [VANTA-WHY] `benchmark_label` is the ORACLE label (§5.3), and the oracle
    // runs from the harness AFTER this process exits (§5.2) — it needs
    // `final.patch` and a fresh tree. So a run can label itself only where its
    // own verdict already settles the question: every other row of §5.3's table
    // needs `patch_good`, which is not knowable here. Those stay `None` and are
    // filled in by the manifest; the alternative — guessing from the verdict —
    // would write `true_approval` for a patch the oracle may well call a
    // `false_approval`.
    let benchmark_label = match decision {
        GovernorDecision::InfraFailure => Some("infra_failure".to_string()),
        _ => None,
    };
    session_log.append(SessionRecord {
        session_id: session_id.clone(),
        schema_version: Some(SCHEMA_VERSION),
        patch_attempt_id: attempt_patch_ids.last().cloned(),
        testing_iteration_id: None,
        test_id: None,
        phase: "run.summary".to_string(),
        summary: format!(
            "attempts_run={} fixed_at_attempt_1={} final_decision={:?} \
             config={} mode={} seed={} model={} top_p={} {} \
             llm_calls={} llm_calls_measured={} llm_prompt_tokens={} llm_completion_tokens={} \
             commit={} dirty={}",
            attempts_run,
            fixed_at_attempt_1,
            decision,
            config_name,
            run_mode,
            seed_value,
            cfg.model_id,
            top_p_value,
            sampling_schedule_keys(),
            llm_calls,
            llm_calls_measured,
            llm_prompt_tokens,
            token_value(llm_completion_tokens),
            vanta_commit,
            vanta_dirty,
        ),
        timestamp: now_iso(),
        benchmark_instance: Some(config_name.to_string()),
        benchmark_label,
    })?;
    let final_snapshot = ecr_eir_from_rows(&metric_rows);
    if metric_rows.is_empty() {
        println!(
            "  • attempts_run={} fixed_at_attempt_1={} — a repair with no N→N+1 transition is outside ECR/EIR by construction",
            attempts_run, fixed_at_attempt_1
        );
    } else {
        println!(
            "  • attempts_run={} fixed_at_attempt_1={} — ECR={} EIR={} over {} transition row(s), per source: {}",
            attempts_run,
            fixed_at_attempt_1,
            final_snapshot.ecr_text(),
            final_snapshot.eir_text(),
            metric_rows.len(),
            final_snapshot.per_source_text()
        );
    }

    // ---- Final patch (written for EVERY terminal verdict) ----
    // [VANTA-WHY] The external oracle (BugsInPy harness, plans/bugsinpy.md §5.1)
    // consumes `final.patch` on every terminal outcome, not just approvals. A
    // `Rejected` or `MaxAttemptsReached` run is exactly the run whose ground-truth
    // label matters most, and writing the patch only inside the Approved branch
    // left those runs with no patch on disk at all. `final_patched_files` is the
    // same map in all three arms (AllTestsPassed / MaxAttemptsReached / Rejected),
    // so one write here covers them; the Approved branch below only reports it.
    //
    // The diff keeps BOTH the non-test and the test hunks: it is the faithful
    // record of what the Creator (or the reviewed candidate) changed. A consumer
    // that must not apply test changes filters them out by path — the oracle does.
    let final_patch_path = debug_dir.join("final.patch");
    let final_patch_written = match &final_patched_files {
        Some(patched) => match write_final_patch(&original_files, patched, &final_patch_path) {
            Ok(()) => true,
            Err(e) => {
                println!("  ⚠ Could not write {}: {e}", final_patch_path.display());
                false
            }
        },
        // PatchApplicationFailed / HarnessFailure / InfraFailure produce no patch
        // of their own; there is nothing to diff.
        None => false,
    };

    // ---- Gatekeeper (only if all tests passed) ----
    if decision == GovernorDecision::AllTestsPassed {
        let all_passed = final_evidence
            .iter()
            .all(|e| e.outcome == VerificationOutcome::VerifiedWithinBounds);
        // Static errors were checked in the loop; recompute for gatekeeper
        let new_static = final_evidence
            .iter()
            .any(|e| e.failure_class == Some(FailureClass::Static));

        let verdict = gatekeeper.evaluate(all_passed, new_static);
        println!("\n▌ GATEKEEPER: {:?}", verdict);

        if verdict == GatekeeperVerdict::Approved {
            // Write patched files to a staging dir for PR / inspection
            if let Some(patched) = &final_patched_files {
                let stage_dir = debug_dir.join("patched_repo");
                let _ = fs::create_dir_all(&stage_dir);
                for (path, content) in patched {
                    let out_path = stage_dir.join(path);
                    if let Some(parent) = out_path.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    let _ = fs::write(&out_path, content);
                }
                println!("  ✔ Patched files written to {}", stage_dir.display());
                // [VANTA-WHY] The review artifact is a DIFF against the original
                // repo, not the staged copy. `patched_repo` is not a git repo, so
                // the old `cd patched_repo && git diff` hint produced an error
                // instead of a patch; final.patch is a real unified diff the
                // reader can apply. It is written once, above, for every terminal
                // verdict — this branch only reports it.
                if final_patch_written {
                    println!(
                        "  ✔ Ready for PR. Diff against the original repo: {}\n    \
                         (apply inside the original repo with: git apply {})",
                        final_patch_path.display(),
                        final_patch_path.display()
                    );
                }

                // ---- Open PR if requested ----
                if config.open_pr {
                    let accepted_tests = final_accepted_tests.clone().unwrap_or_default();
                    match crate::pr::open_pr(
                        &config.repo_path,
                        patched,
                        &accepted_tests,
                        &final_evidence,
                        session_log
                            .sink_path()
                            .ok_or("session log sink path missing for PR creation")?,
                        &session_id,
                        &repo.issue_text,
                    )
                    .await {
                        Ok(pr) => {
                            println!("  ✔ PR branch: {}", pr.branch);
                            if let Some(url) = &pr.pr_url {
                                println!("  ✔ PR URL: {}", url);
                            } else {
                                println!(
                                    "  ⚠ Branch pushed but PR creation failed (check gh auth)"
                                );
                            }
                        }
                        Err(e) => {
                            println!("  ✗ PR creation failed: {e}");
                        }
                    }
                }
            }
        }
    }

    // ---- Session log + evidence ----
    // [VANTA-WHY] The session log is ALREADY on disk: every record above was
    // appended and flushed as it was produced, `session.start` first. There is no
    // end-of-run write, so an abort costs only the records not yet made — never
    // the attribution.
    session_log.append(SessionRecord {
        session_id: session_id.clone(),
        schema_version: Some(SCHEMA_VERSION),
        // The candidate the run ended on, or null if no attempt ever produced
        // one (e.g. the run died before its first patch).
        patch_attempt_id: attempt_patch_ids.last().cloned(),
        testing_iteration_id: None,
        test_id: None,
        phase: "M3_repair_loop".to_string(),
        summary: format!("final_decision={:?}", decision),
        timestamp: now_iso(),
        benchmark_instance: None,
        benchmark_label: None,
    })?;

    // ---- The raw transition rows (the M5 dataset) are already on disk ----
    // [VANTA-WHY] Row-level and immutable, appended and flushed as each attempt
    // produced them (see the TransitionWriter above), NOT written here. The
    // paper's analyses (S^(0)-restricted rates, defect-level cluster bootstrap,
    // oracle-only robustness check) all re-select rows AFTER the run, so metrics
    // are computed FROM that file rather than only alongside it. Because the sink
    // is append-and-flush, reaching this line is not what makes the dataset
    // exist — a run that aborted at attempt 3 kept every row up to attempt 3.
    let transitions_path = transition_writer.path().to_path_buf();

    let evidence_path =
        Path::new("vanta_sessions").join(format!("{}_evidence.json", session_id.as_str()));
    // [VANTA-WHY] Run-level counter rides ALONGSIDE the records rather than inside
    // them: "how many attempts spent their budget on a patch that never reached
    // the tree" is a property of the RUN, not of any single test. It is reported
    // separately because those attempts are censored out of EIR/ECR by
    // construction, so without this number a repair rate cannot be told apart
    // from a candidate that was never evaluated.
    let _ = fs::write(
        &evidence_path,
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": crate::session::SCHEMA_VERSION,
            "apply_failed_attempts": apply_failed_attempt_indices.len(),
            "new_static_findings": static_findings_records,
            // [VANTA-WHY] The repo's own suite, per attempt. `native_test_source`
            // says the suite ran the ORIGINAL tests; this key names WHICH of them
            // each attempt regressed, so the correction a later attempt made is
            // recorded in the run's artifacts and not only in the transitions
            // file the metric counts from.
            "native_regressions": native_regression_records,
            // [VANTA-WHY] Test-integrity evidence. `native_test_source` names the
            // rule the native suite ran under — the ORIGINAL tests — and the four
            // lists name every way the patch tried to influence them, so an audit
            // can see that the patch's control over its own tests was removed
            // rather than trusted. All four are empty on a patch that leaves the
            // test files and pytest config alone.
            "native_test_source": "original",
            "modified_existing_tests": modified_existing_tests,
            "removed_existing_tests": removed_existing_tests,
            "added_test_files": added_test_files,
            "modified_test_config": modified_test_config,
            "evidence": final_evidence,
        }))
        .unwrap(),
    );
    println!("\n▌ Evidence → {}", evidence_path.display());
    println!(
        "▌ Transitions → {} ({} row(s))",
        transitions_path.display(),
        metric_rows.len()
    );
    let session_path = session_log
        .sink_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| format!("vanta_sessions/{}.json", session_id.as_str()));
    println!("▌ Session  → {session_path}");

    Ok(decision)
}

// ============================================================================
// write_final_patch — the run's diff against the ORIGINAL repo
// ============================================================================
//
// [VANTA-WHY] The Gatekeeper used to point the reader at `cd patched_repo && git
// diff`. That command cannot work: `patched_repo` is a flat copy of the written
// files, not a git repository, so `git diff` there fails with "not a git
// repository" and the reader is left with no diff at all. `git diff --no-index`
// compares two DIRECTORIES and needs no repository; with `--no-prefix` its
// headers come out as the canonical `a/<path>` / `b/<path>`, so the file applies
// with a plain `git apply` inside the original repo.
fn write_final_patch(
    original_files: &HashMap<String, String>,
    patched_files: &HashMap<String, String>,
    destination: &Path,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let scratch = tempfile::tempdir()?;
    let before = scratch.path().join("a");
    let after = scratch.path().join("b");
    write_diff_tree(&before, original_files)?;
    write_diff_tree(&after, patched_files)?;

    let output = Command::new("git")
        .arg("diff")
        .arg("--no-index")
        .arg("--no-color")
        .arg("--no-prefix")
        .arg("--")
        .arg("a")
        .arg("b")
        .current_dir(scratch.path())
        .output()?;
    // [VANTA-WHY] `git diff --no-index` exits 1 whenever the two trees differ,
    // which is the NORMAL case here. Treating that as failure would mean the
    // patch is only ever written when there is nothing to write.
    if !output.status.success() && output.status.code() != Some(1) {
        return Err(format!(
            "git diff --no-index failed (status {:?}): {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    fs::write(destination, &output.stdout)?;
    Ok(())
}

fn write_diff_tree(root: &Path, files: &HashMap<String, String>) -> io::Result<()> {
    for (path, contents) in files {
        let out_path = root.join(path);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out_path, contents)?;
    }
    Ok(())
}

// ============================================================================
// build_creator_feedback — structured feedback for the Creator from failures
// ============================================================================
fn tested_commit_sha(repo_path: &Path) -> String {
    match Command::new("git")
        .arg("-C")
        .arg(repo_path)
        .args(["rev-parse", "HEAD"])
        .output()
    {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "none".to_string(),
    }
}

fn repo_working_tree_dirty(
    repo_path: &Path,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo_path)
        .args(["status", "--porcelain"])
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "failed to inspect target repo dirty state with `git -C {} status --porcelain`: {}",
            repo_path.display(),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(!String::from_utf8_lossy(&out.stdout).trim().is_empty())
}

fn build_source_concat(files: &HashMap<String, String>) -> String {
    let mut ordered: Vec<_> = files.iter().collect();
    ordered.sort_by(|(left, _), (right, _)| left.cmp(right));
    ordered
        .into_iter()
        .map(|(path, contents)| format!("# FILE: {path}\n{contents}"))
        .collect::<Vec<_>>()
        .join("\n\n")
}

// [VANTA-WHY] BATE validates the Attacker's declared intent against the
// BASELINE, so it must see the patch-FREE original tree. It is deliberately not
// given the patched (let alone the graded) tree: a candidate must never be
// allowed to validate its own tests, and BATE's job is to check that a test
// actually fails on the ORIGINAL code — a tree that already contains the fix
// would make every correct probe look wrong. This named seam is what the
// regression test asserts against, so "BATE gets the original tree" is enforced
// by a test rather than by a comment.
fn bate_baseline_concat(original_files: &HashMap<String, String>) -> String {
    build_source_concat(original_files)
}

fn review_prompt_source_concat(
    repo_files: &[SourceFile],
    original_files: &HashMap<String, String>,
    patch_files: &HashMap<String, String>,
) -> String {
    let selected = select_review_prompt_paths(repo_files, patch_files);
    let mut prompt_files = HashMap::new();
    for path in selected {
        if let Some(contents) = original_files.get(&path) {
            prompt_files.insert(path, contents.clone());
        }
    }
    build_source_concat(&prompt_files)
}

fn select_review_prompt_paths(
    repo_files: &[SourceFile],
    patch_files: &HashMap<String, String>,
) -> BTreeSet<String> {
    let touched_paths: BTreeSet<String> = patch_files.keys().cloned().collect();
    if touched_paths.is_empty() {
        return touched_paths;
    }
    let mut selected = touched_paths.clone();

    let module_map = python_module_to_path(repo_files);
    let source_by_path: BTreeMap<&str, &SourceFile> =
        repo_files.iter().map(|file| (file.path.as_str(), file)).collect();

    let touched_modules: BTreeSet<String> = selected
        .iter()
        .filter_map(|path| python_module_path(path))
        .collect();

    let mut related_paths = BTreeSet::new();
    for path in selected.clone() {
        if let Some(file) = source_by_path.get(path.as_str()) {
            related_paths.extend(imported_repo_paths(&file.path, &file.contents, &module_map));
        }
    }
    selected.extend(related_paths);

    for file in repo_files.iter().filter(|file| is_test_path(&file.path)) {
        let imports = imported_repo_paths(&file.path, &file.contents, &module_map);
        if !imports.is_disjoint(&touched_paths)
            || test_covers_modules(&file.contents, &touched_modules)
        {
            selected.insert(file.path.clone());
        }
    }

    selected
}

fn python_module_to_path(repo_files: &[SourceFile]) -> BTreeMap<String, String> {
    let mut module_map = BTreeMap::new();
    for file in repo_files {
        if let Some(module) = python_module_path(&file.path) {
            module_map.insert(module, file.path.clone());
        }
    }
    module_map
}

fn python_module_path(path: &str) -> Option<String> {
    if !path.ends_with(".py") {
        return None;
    }
    let trimmed = path.trim_end_matches(".py");
    if let Some(pkg) = trimmed.strip_suffix("/__init__") {
        return Some(pkg.replace('/', "."));
    }
    Some(trimmed.replace('/', "."))
}

fn is_test_path(path: &str) -> bool {
    let file_name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    path.starts_with("tests/")
        || path.contains("/tests/")
        || file_name.starts_with("test_")
        || file_name.ends_with("_test.py")
}

fn imported_repo_paths(
    current_path: &str,
    contents: &str,
    module_map: &BTreeMap<String, String>,
) -> BTreeSet<String> {
    let mut imported = BTreeSet::new();
    let current_package = current_package_parts(current_path);

    for raw_line in contents.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if let Some(rest) = line.strip_prefix("import ") {
            for target in rest.split(',') {
                let module = target
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim();
                if let Some(path) = module_map.get(module) {
                    imported.insert(path.clone());
                }
            }
            continue;
        }

        let Some(rest) = line.strip_prefix("from ") else {
            continue;
        };
        let Some((module_part, names_part)) = rest.split_once(" import ") else {
            continue;
        };
        let Some(base_module) = resolve_import_base(&current_package, module_part.trim()) else {
            continue;
        };
        if let Some(path) = module_map.get(&base_module) {
            imported.insert(path.clone());
        }
        for name in parse_imported_names(names_part) {
            let candidate = if base_module.is_empty() {
                name
            } else {
                format!("{base_module}.{name}")
            };
            if let Some(path) = module_map.get(&candidate) {
                imported.insert(path.clone());
            }
        }
    }

    imported
}

fn current_package_parts(path: &str) -> Vec<String> {
    let Some(module) = python_module_path(path) else {
        return Vec::new();
    };
    let mut parts: Vec<String> = module.split('.').map(|part| part.to_string()).collect();
    if !path.ends_with("__init__.py") {
        let _ = parts.pop();
    }
    parts
}

fn resolve_import_base(current_package: &[String], module_part: &str) -> Option<String> {
    let trimmed = module_part.trim();
    if trimmed.is_empty() {
        return None;
    }
    if !trimmed.starts_with('.') {
        return Some(trimmed.to_string());
    }

    let dot_count = trimmed.chars().take_while(|c| *c == '.').count();
    let remainder = trimmed[dot_count..].trim();
    let levels_up = dot_count.saturating_sub(1);
    if levels_up > current_package.len() {
        return None;
    }
    let kept = &current_package[..current_package.len().saturating_sub(levels_up)];
    let mut base = kept.join(".");
    if !remainder.is_empty() {
        if !base.is_empty() {
            base.push('.');
        }
        base.push_str(remainder);
    }
    Some(base)
}

fn parse_imported_names(names_part: &str) -> Vec<String> {
    names_part
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .filter_map(|name| {
            let token = name
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim()
                .trim_matches(',');
            if token.is_empty() || token == "*" {
                None
            } else {
                Some(token.to_string())
            }
        })
        .collect()
}

fn test_covers_modules(test_source: &str, touched_modules: &BTreeSet<String>) -> bool {
    touched_modules.iter().any(|module| test_source.contains(module))
}

fn dependency_freeze_summary_value(freeze: Option<&str>) -> String {
    let rendered = freeze
        .unwrap_or("")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("|");
    if rendered.is_empty() {
        "none".to_string()
    } else {
        rendered
    }
}

fn render_committed_test_file(tests: &[GeneratedTest]) -> String {
    let mut imports: Vec<String> = Vec::new();
    let mut bodies: Vec<String> = Vec::new();

    for test in tests {
        let mut body_lines = Vec::new();
        let mut in_body = false;
        for line in test.test_source.lines() {
            let trimmed = line.trim();
            if !in_body && (trimmed.starts_with("def ") || trimmed.starts_with("@")) {
                in_body = true;
            }
            if in_body {
                body_lines.push(line);
            } else if !trimmed.is_empty()
                && (trimmed.starts_with("import ") || trimmed.starts_with("from "))
                && !imports.iter().any(|existing| existing == trimmed)
            {
                imports.push(trimmed.to_string());
            }
        }
        let body = body_lines.join("\n").trim().to_string();
        if !body.is_empty() {
            bodies.push(body);
        }
    }

    let mut rendered = String::new();
    if !imports.is_empty() {
        rendered.push_str(&imports.join("\n"));
        rendered.push_str("\n\n");
    }
    rendered.push_str(&bodies.join("\n\n\n"));
    if !rendered.ends_with('\n') {
        rendered.push('\n');
    }
    rendered
}

fn require_clean_target_repo_for_open_pr(target_repo_dirty: Option<bool>) -> Result<(), String> {
    match target_repo_dirty {
        Some(false) => Ok(()),
        Some(true) => Err(
            "--open-pr requires a clean target repository working tree; commit or stash the target repo changes first"
                .to_string(),
        ),
        None => Err(
            "--open-pr requires a git-backed target repository so Vanta can record and validate the tested commit"
                .to_string(),
        ),
    }
}

fn build_creator_feedback(
    failed: &[&TestEvidence],
    ruff_new: bool,
    mypy_new: bool,
    native_regressions: &[String],
    native_patched: &harness::NativeSuiteReport,
    modified_existing_tests: &[String],
    removed_existing_tests: &[String],
    blocking_static: &[String],
) -> String {
    let mut fb = String::new();
    fb.push_str("The previous patch failed verification:\n\n");

    for e in failed {
        let class = e
            .failure_class
            .as_ref()
            .map(|c| format!("{c:?}"))
            .unwrap_or_else(|| "Unknown".to_string());
        fb.push_str(&format!(
            "- Test '{}' (intent={:?}) failed with failure_class={}\n",
            e.test_name, e.intent, class
        ));
        if let Some(payload) = &e.structured_payload {
            fb.push_str(&format!("  structured_payload={payload}\n"));
        }
    }

    if ruff_new {
        fb.push_str("- New ruff lint errors were introduced.\n");
    }
    if mypy_new {
        fb.push_str("- New mypy type errors were introduced.\n");
    }

    // [VANTA-WHY] The repo's OWN suite is a failure signal the generated tests
    // cannot express, and the Creator has to be told BOTH that it regressed and
    // that any edit it made to an existing test was thrown away — otherwise it
    // will "fix" the same test again and burn the next attempt the same way.
    if !modified_existing_tests.is_empty() || !removed_existing_tests.is_empty() {
        fb.push_str(
            "- You modified or deleted existing test file(s). Those edits were DISCARDED: the \
             repo's own tests are run from their ORIGINAL contents, so a patch cannot weaken the \
             test that grades it.\n",
        );
        for path in modified_existing_tests {
            fb.push_str(&format!("  modified (discarded): {path}\n"));
        }
        for path in removed_existing_tests {
            fb.push_str(&format!("  deleted (restored): {path}\n"));
        }
        fb.push_str(
            "  Do not modify or delete existing tests. If you need a new test, put it in a NEW \
             test file.\n",
        );
    }
    if !native_regressions.is_empty() {
        fb.push_str("- The repository's own tests regressed (run with the ORIGINAL test files):\n");
        for nodeid in native_regressions {
            fb.push_str(&format!("  - {nodeid}\n"));
            if let Some(output) = native_patched.failure_output(nodeid) {
                fb.push_str(&format!("    failure: {output}\n"));
            }
        }
    }
    // [VANTA-WHY] A blocking static error is a failure the Creator has to fix,
    // and it reaches it through the SAME feedback the generated-test failures
    // do. `ruff_new`/`mypy_new` above only say a category fired; naming the
    // finding is what lets the Creator fix the right line.
    if !blocking_static.is_empty() {
        fb.push_str("- Blocking static errors were introduced in the file(s) you touched:\n");
        for finding in blocking_static {
            fb.push_str(&format!("  - {finding}\n"));
        }
    }

    fb.push_str(
        "\nPlease revise the patch to fix these failures while preserving existing behavior.",
    );
    fb
}

fn read_json_value_from_debug(path: &Path) -> Option<serde_json::Value> {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
}

// ============================================================================
// Candidate identity — a deterministic fingerprint of the patch text
// ============================================================================
//
// [VANTA-WHY] The patch_attempt_id is derived from the CONTENT of the patch, not
// from an attempt counter. A counter would make the same candidate carry a
// different id on a re-run, so two rows that both compared "patch A" would look
// like a transition. A content hash makes `from == to` mean "the same candidate"
// across runs as well as within one.
fn stable_patch_fingerprint(patch_text: &str) -> String {
    // FNV-1a keeps the patch-mode candidate ID deterministic without needing
    // extra crate features.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in patch_text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn patch_attempt_id_for_patch(patch_text: &str) -> PatchAttemptId {
    PatchAttemptId::new(format!("patch-{}", stable_patch_fingerprint(patch_text)))
}

// [VANTA-WHY] Stamp a Governor transition with the two candidate identities it
// compares. Both ends are needed: in --patch mode the candidate never changes, so
// `from == to` on every row and the patch-transition analysis can exclude them
// with a single condition. An end with NO real id (a skipped attempt left a gap
// in the index) is stamped `null`, never filled in from a neighbouring attempt —
// filling it would write a value nobody observed to disk as a fact, and it is
// wrong both ways (a differing guess looks like a repair transition, a matching
// one looks like a review round). A null is filterable: `from_transition` then
// censors the row, so the analysis counts a row only when both ids are non-null
// and differ.
// ============================================================================
// Native + static transition rows — candidate evidence the Governor never saw
// ============================================================================
//
// [VANTA-WHY] EIR/ECR is defined over TRANSITION ROWS, and until this pass only
// the Governor wrote them. The native suite and the static gate observe the same
// candidate without going through the Governor, so a repair that fixed ONLY a
// native regression (session 4d55123d) wrote no row and the metric could not see
// it. These helpers difference one attempt's observation against the previous
// attempt's and emit ordinary rows, so the metric keeps ONE definition and one
// source of truth.

/// One attempt's effective native-suite outcome, per repo test.
///
/// [VANTA-WHY] "Effective" uses the regression walk's own semantics: a test that
/// PASSED on the original tree and fails (or is absent) on the patched tree is a
/// FAILURE at that attempt; a test already failing on the original is not the
/// patch's doing and is left out entirely, exactly as `native_regressions` skips
/// it.
#[derive(Debug, Default, Clone)]
struct AttemptNativeOutcome {
    results: HashMap<String, TestResult>,
}

impl AttemptNativeOutcome {
    fn from_suites(
        original: &harness::NativeSuiteReport,
        patched: &harness::NativeSuiteReport,
    ) -> Self {
        let mut results = HashMap::new();
        for (nodeid, original_result) in &original.tests {
            if *original_result != TestResult::Passed {
                continue;
            }
            // A vanished test is a regression (`native_suite_collapsed` also
            // names it), so "absent on the patched tree" is a failure here too.
            let outcome = match patched.tests.iter().find(|(n, _)| n == nodeid) {
                Some((_, TestResult::Passed)) => TestResult::Passed,
                _ => TestResult::Failed,
            };
            results.insert(nodeid.clone(), outcome);
        }
        Self { results }
    }
}

/// One attempt's BLOCKING static findings, keyed by normalised identity.
#[derive(Debug, Default, Clone)]
struct AttemptStaticOutcome {
    /// identity -> rendered finding (kept for the row's `test_name`).
    findings: BTreeMap<String, String>,
}

impl AttemptStaticOutcome {
    fn from_findings(findings: &[&harness::StaticDiagnostic]) -> Self {
        let mut map = BTreeMap::new();
        for d in findings {
            map.insert(
                static_finding_identity(&d.code, &d.file, &d.message),
                d.render(),
            );
        }
        Self { findings: map }
    }
}

/// One row per native test that ran on BOTH attempts.
///
/// [VANTA-WHY] `test_id` is the nodeid itself, not a fresh id and not the
/// normalised body hash: a repo test's identity IS its nodeid, and it is what
/// makes the same test match across attempts.
fn native_transition_rows(
    session_id: &SessionId,
    from_patch: &PatchAttemptId,
    to_patch: &PatchAttemptId,
    from_attempt: u32,
    to_attempt: u32,
    previous: &AttemptNativeOutcome,
    current: &AttemptNativeOutcome,
) -> Vec<TransitionRow> {
    let mut rows: Vec<TransitionRow> = current
        .results
        .iter()
        .filter_map(|(nodeid, to_result)| {
            let from_result = previous.results.get(nodeid)?;
            Some(TransitionRow {
                session_id: session_id.clone(),
                schema_version: Some(SCHEMA_VERSION),
                from_patch_attempt_id: Some(from_patch.clone()),
                to_patch_attempt_id: Some(to_patch.clone()),
                from_attempt,
                to_attempt,
                test_id: nodeid.clone(),
                test_name: nodeid.clone(),
                shared_set_id: "native".to_string(),
                from_result: from_result.clone(),
                to_result: to_result.clone(),
                from_failure_class: native_failure_class(from_result),
                to_failure_class: native_failure_class(to_result),
                evidence_source: EvidenceSource::Native,
                test_intent: TestIntent::BehaviorPreservation,
                censored: false,
                apply_failed: false,
                gap: to_attempt.saturating_sub(from_attempt) > 1,
                in_s0: true,
            })
        })
        .collect();
    // Deterministic order so two runs of the same comparison write the same
    // file — HashMap iteration order is not stable.
    rows.sort_by(|a, b| a.test_id.cmp(&b.test_id));
    rows
}

/// The class of a native-suite failure.
///
/// [VANTA-WHY] A repo test that passed on the original and fails on the
/// candidate is a behaviour regression — the same Logic class the Diagnostician
/// gives an assertion failure. That is what makes the row countable under the
/// ordinary Logic gate with `evidence_source = Native`, without weakening the
/// gate for anyone else.
fn native_failure_class(result: &TestResult) -> Option<FailureClass> {
    match result {
        TestResult::Passed => None,
        _ => Some(FailureClass::Logic),
    }
}

/// One row per blocking static finding identity seen on either attempt, so a
/// static error introduced at N and fixed at N+1 counts as a correction.
fn static_transition_rows(
    session_id: &SessionId,
    from_patch: &PatchAttemptId,
    to_patch: &PatchAttemptId,
    from_attempt: u32,
    to_attempt: u32,
    previous: &AttemptStaticOutcome,
    current: &AttemptStaticOutcome,
) -> Vec<TransitionRow> {
    let mut identities: Vec<&String> = previous
        .findings
        .keys()
        .chain(current.findings.keys())
        .collect();
    identities.sort();
    identities.dedup();
    identities
        .into_iter()
        .map(|identity| {
            let from_present = previous.findings.get(identity);
            let to_present = current.findings.get(identity);
            TransitionRow {
                session_id: session_id.clone(),
                schema_version: Some(SCHEMA_VERSION),
                from_patch_attempt_id: Some(from_patch.clone()),
                to_patch_attempt_id: Some(to_patch.clone()),
                from_attempt,
                to_attempt,
                test_id: identity.clone(),
                test_name: to_present
                    .or(from_present)
                    .cloned()
                    .unwrap_or_else(|| identity.clone()),
                shared_set_id: "static".to_string(),
                from_result: presence_result(from_present),
                to_result: presence_result(to_present),
                from_failure_class: presence_class(from_present),
                to_failure_class: presence_class(to_present),
                evidence_source: EvidenceSource::Static,
                test_intent: TestIntent::BugReproduction,
                censored: false,
                apply_failed: false,
                gap: to_attempt.saturating_sub(from_attempt) > 1,
                in_s0: true,
            }
        })
        .collect()
}

/// A static finding PRESENT at an attempt is a failure there; absent is a pass.
fn presence_result(present: Option<&String>) -> TestResult {
    if present.is_some() {
        TestResult::Failed
    } else {
        TestResult::Passed
    }
}

fn presence_class(present: Option<&String>) -> Option<FailureClass> {
    if present.is_some() {
        Some(FailureClass::Static)
    } else {
        None
    }
}

/// Append rows the Governor did not compute to BOTH the on-disk file and the
/// in-memory set the live metric is computed from.
///
/// [VANTA-WHY] The two must be updated together: a row written only to disk
/// would make the console print a different number from `metrics.py`, and a row
/// kept only in memory would vanish from the dataset the paper reads.
fn append_derived_rows(
    writer: &mut TransitionWriter,
    rows: Vec<TransitionRow>,
    metric_rows: &mut Vec<TransitionRow>,
) -> Result<(), io::Error> {
    if rows.is_empty() {
        return Ok(());
    }
    writer.append(&rows)?;
    metric_rows.extend(rows);
    Ok(())
}

/// Append every transition not yet persisted and return the new cursor.
///
/// [VANTA-WHY] One definition, called from two places: the ordinary post-attempt
/// flush, and the repair-mode failed-apply path (which must persist its censored
/// rows BEFORE the loop continues, because if the budget is then exhausted the
/// ordinary flush never runs again and the hole would be lost). The rows are
/// also kept in memory so the live EIR/ECR print derives from the same rows with
/// the one shared function rather than a second rule.
fn flush_transition_rows(
    governor: &Governor,
    writer: &mut TransitionWriter,
    persisted: usize,
    session_id: &SessionId,
    attempt_patch_ids: &[PatchAttemptId],
    apply_failed_attempts: &[u32],
    metric_rows: &mut Vec<TransitionRow>,
) -> Result<usize, io::Error> {
    let transitions = governor.transitions();
    if transitions.len() <= persisted {
        return Ok(persisted);
    }
    let rows: Vec<TransitionRow> = transitions[persisted..]
        .iter()
        .map(|t| {
            stamp_transition_row(t, session_id, attempt_patch_ids, apply_failed_attempts)
        })
        .collect();
    writer.append(&rows)?;
    metric_rows.extend(rows);
    Ok(transitions.len())
}

fn stamp_transition_row(
    transition: &AttemptTransition,
    session_id: &SessionId,
    attempt_patch_ids: &[PatchAttemptId],
    apply_failed_attempts: &[u32],
) -> TransitionRow {
    let from_id = attempt_patch_ids
        .get(transition.from_attempt as usize)
        .cloned();
    let to_id = attempt_patch_ids
        .get(transition.to_attempt as usize)
        .cloned();
    // [VANTA-WHY] The flag belongs to the NEWER side: the row exists because that
    // attempt's patch never reached the tree. Taken from the attempt index rather
    // than a run-level bool so a later successful attempt's rows are not tainted
    // by an earlier hole.
    let apply_failed = apply_failed_attempts.contains(&transition.to_attempt);
    TransitionRow::from_transition(transition, session_id, from_id, to_id, apply_failed)
}

// [VANTA-WHY] Infrastructure is the class that gets reset, rerun, and excluded
// from EIR/ECR — it never counts against the model. That is correct for a single
// backend fault, but it is the perfect disguise for a broken harness: if every
// patched test fails to import, all of them classify Infrastructure, all get
// excluded, and the run looks clean while the patch was never evaluated. A real
// environment blip hits one or two tests; a 100% rate is always the setup.
// Detecting that turns a silent wrong answer back into a loud one.
fn harness_infrastructure_collapse(classes: &[Option<FailureClass>]) -> bool {
    !classes.is_empty()
        && classes
            .iter()
            .all(|c| c.as_ref() == Some(&FailureClass::Infrastructure))
}

// [VANTA-WHY] A 100% Infrastructure rate can come from correlated parallel
// spawn: an attempt launches all its tests at once, so a momentary outage at
// that instant takes every one of them out together. That is bad luck, not a
// broken harness. The discriminator is REPRODUCIBILITY — a broken harness fails
// the same way every time, a transient blip does not. So retry once, and only
// treat a repeat collapse as a real defect. Retrying costs one attempt only in
// the failure case; aborting on the first collapse costs a valid run on bad luck.
fn collapse_decision(streak: u32, governor_allows_retry: bool) -> GovernorDecision {
    if streak >= 2 || !governor_allows_retry {
        GovernorDecision::HarnessFailure
    } else {
        GovernorDecision::Continue
    }
}

// [VANTA-WHY] The Governor's three-way verdict is computed here, once, from the
// attempt's evidence. A Logic-class failure is a finding about the candidate: it
// is the one class the Diagnostician can attribute to the code (Spec §7).
// Infrastructure / Unknown mean we could not get clean evidence — reporting
// either as a rejection would claim the model was wrong when the environment
// was. A repo-native regression (`native_suite`) is a real finding even though
// it is not part of the shared set, so it is passed in separately.
//
// [VANTA-WHY] Static used to be left entirely to the Gatekeeper, and that was a
// dead zone. The Governor said "Static is the Gatekeeper's job", but the
// Gatekeeper only runs after `AllTestsPassed`, so a round whose ONLY failure was
// Static reached neither a rejection nor the gate that would have produced one.
// In review mode there is also no Creator to repair it, so it just kept
// attacking a candidate it had already found broken. A new static finding in a
// file the patch touches is terminal in REVIEW mode only; in repair mode the
// class stays non-decisive (the Creator gets it as feedback, and the Gatekeeper
// still blocks approval if it survives).
fn attempt_verdict(
    all_passed: bool,
    failure_classes: &[Option<FailureClass>],
    native_regression: bool,
    new_static_in_touched_files: bool,
    mode: GovernorMode,
) -> AttemptVerdict {
    let static_rejects = new_static_in_touched_files && mode == GovernorMode::Review;
    if native_regression
        || static_rejects
        || failure_classes
            .iter()
            .any(|c| c.as_ref() == Some(&FailureClass::Logic))
    {
        AttemptVerdict::Rejected
    } else if all_passed {
        AttemptVerdict::Passed
    } else {
        AttemptVerdict::Inconclusive
    }
}

/// Name the REAL signal behind a review-mode rejection.
///
/// [VANTA-WHY] Item 4 of the heretic fix list. The old message counted only the
/// generated tests and, when they all passed, printed "the candidate fails 0
/// test(s): (classified by the repo's own suite)" — a sentence that names no
/// signal at all, which is what 40e84874 did while the real reason was a mypy
/// error no artifact recorded. This names the actual reason: the regressed
/// native tests, the new static findings, and/or the generated tests that
/// failed. Exactly one of those is present whenever the verdict is Rejected.
///
/// [VANTA-WHY] A regressing test that the patch ALSO rewrote stays a regression
/// (the native run was graded with the original file), but it is annotated. A
/// human patch may legitimately change a test's expected behaviour, and the
/// reviewer must be able to see that the rejection rests on a file the candidate
/// itself touched rather than discovering it later.
fn rejection_signal_message(
    native_regressions: &[String],
    blocking_static: &[String],
    failing_names: &[String],
    modified_existing_tests: &[String],
) -> String {
    let mut signals: Vec<String> = Vec::new();
    if !native_regressions.is_empty() {
        let rendered: Vec<String> = native_regressions
            .iter()
            .map(|nodeid| match modified_test_file_for(nodeid, modified_existing_tests) {
                Some(path) => format!("{nodeid} (the patch also modified {path})"),
                None => nodeid.clone(),
            })
            .collect();
        signals.push(format!("native regression: {}", rendered.join(", ")));
    }
    if !blocking_static.is_empty() {
        signals.push(format!(
            "new static finding in a touched file: {}",
            blocking_static.join("; ")
        ));
    }
    if !failing_names.is_empty() {
        signals.push(format!("failing test(s): {}", failing_names.join(", ")));
    }
    if signals.is_empty() {
        "verdict=Rejected but no signal could be named — inspect the attempt's diagnostic \
         payloads"
            .to_string()
    } else {
        signals.join(" | ")
    }
}

/// The modified test file a pytest nodeid belongs to, if the patch rewrote one.
///
/// A pytest nodeid is `<path>` for a module-level item or `<path>::<name>` for a
/// function/class — both start with the file path, which is how a regression can
/// be matched back to the `modified_existing_tests` entry that caused it.
fn modified_test_file_for<'a>(
    nodeid: &str,
    modified_existing_tests: &'a [String],
) -> Option<&'a String> {
    modified_existing_tests
        .iter()
        .find(|path| nodeid == path.as_str() || nodeid.starts_with(&format!("{path}::")))
}

/// The pytest config files a patch must not be able to grade itself with.
///
/// [VANTA-WHY] `pyproject.toml` cannot be reverted wholesale: it also carries
/// dependencies, build config and tool settings that have nothing to do with
/// tests, and restoring the whole file would discard a legitimate change. So a
/// change is detected SECTION-scoped and recorded with its diff. This is the one
/// case the native run CANNOT neutralise by restoration — a patched `addopts`
/// or `collect_ignore` would reach pytest — so it is reported loudly instead,
/// and the native verdict is annotated as resting on a config the patch touched.
const PYTEST_CONFIG_SECTIONS: &[(&str, &str)] = &[
    ("pytest.ini", ""),
    ("pyproject.toml", "tool.pytest.ini_options"),
    ("setup.cfg", "tool:pytest"),
    ("tox.ini", "[pytest]"),
];

/// Record every pytest-config change a patch carries, with its diff.
fn modified_test_config_records(
    patch_files: &HashMap<String, String>,
) -> Vec<serde_json::Value> {
    let mut records = Vec::new();
    let mut paths: Vec<&String> = patch_files.keys().collect();
    paths.sort();
    for path in paths {
        let Some((config_path, marker)) = PYTEST_CONFIG_SECTIONS
            .iter()
            .find(|(name, _)| *name == path.as_str())
        else {
            continue;
        };
        let diff = &patch_files[path];
        // An empty marker means the whole file is pytest config (`pytest.ini`).
        if !marker.is_empty() && !diff.contains(marker) {
            continue;
        }
        println!(
            "  ⚠ pytest config '{config_path}' changed by the patch — recorded as \
             modified_test_config with its diff. This section is the one the native run cannot \
             restore (pyproject also carries non-test content), so the patched config is used \
             and the native verdict must be read with that in mind"
        );
        records.push(serde_json::json!({
            "path": config_path,
            "diff": diff,
        }));
    }
    records
}

/// The new static findings that BLOCK a review round, rendered as evidence.
///
/// [VANTA-WHY] Item 4 of the heretic fix list: the rejection message must name
/// the finding that rejected the candidate, so the boolean gate and the message
/// come from ONE list. A finding blocks only when it sits in a file the patch
/// touches AND is of an ERROR class — ruff `F`/`E9`, or any mypy error (its
/// parser accepts `: error:` lines only). A style rule (B, BLE, UP, SIM, TRY, …)
/// in a touched file is REPORTED to the Creator and kept in the session log, but
/// must not reject: rejecting on a nit claims the candidate is broken when it is
/// not, and burns an attack round.
#[cfg(test)]
fn blocking_static_findings(
    new_ruff: &[&harness::StaticDiagnostic],
    new_mypy: &[&harness::StaticDiagnostic],
    touched: &[String],
) -> Vec<String> {
    blocking_static_diagnostics(new_ruff, new_mypy, touched)
        .iter()
        .map(|d| d.render())
        .collect()
}

/// The blocking findings THEMSELVES, so a Static transition row can key on a
/// finding's identity instead of its rendered text.
///
/// [VANTA-WHY] `blocking_static_findings` is this list rendered, so the gate,
/// the rejection message and the rows all read ONE filter. Two copies of the
/// error-class/environment/touched conditions would drift, and a row keyed on a
/// different set of findings than the one that rejected the candidate would be
/// an invisible inconsistency in the metric.
fn blocking_static_diagnostics<'a>(
    new_ruff: &[&'a harness::StaticDiagnostic],
    new_mypy: &[&'a harness::StaticDiagnostic],
    touched: &[String],
) -> Vec<&'a harness::StaticDiagnostic> {
    let mut out = Vec::new();
    for d in new_ruff {
        if harness::ruff_is_error_class(&d.code) && harness::path_touched(&d.file, touched) {
            out.push(*d);
        }
    }
    for d in new_mypy {
        // [VANTA-WHY] Item 10: `import-not-found` / `import-untyped` say the
        // ANALYSIS ENVIRONMENT cannot see a module (no dependencies installed),
        // not that the candidate is wrong. Session 40e84874 rejected C on
        // `test_auto_rank.py [import-not-found] Cannot find module "torch"`.
        // They are recorded (see `record_new_static_findings`) but never block.
        if !d.is_environment_finding() && harness::path_touched(&d.file, touched) {
            out.push(*d);
        }
    }
    out
}

/// Record each NEW static finding for the evidence file; return console lines.
///
/// [VANTA-WHY] The set diff keeps only the findings the patch introduced. This
/// turns each into `{attempt, tool, file, line, code, message}` so a run's
/// artifacts name the exact finding behind a review-mode rejection instead of a
/// bare boolean. Session 40e84874 rejected on a mypy finding that the session
/// files never recorded; this is where it is now kept.
///
/// [VANTA-WHY] `blocking` is the rendered list of the findings that actually
/// REJECTED the round (from `blocking_static_findings`). The `"blocking"`
/// boolean it adds to each record is what lets an audit tell the finding that
/// rejected the candidate apart from the findings that were merely REPORTED to
/// the Creator: both land in the same list, and without this flag a style nit
/// and a fatal error are indistinguishable after the run. It is a look-up of the
/// finding's own `render()` in the SAME list the gate and the message read, so
/// the recorded flag cannot disagree with the verdict.
fn record_new_static_findings(
    sink: &mut Vec<serde_json::Value>,
    attempt: u32,
    new_ruff: &[&harness::StaticDiagnostic],
    new_mypy: &[&harness::StaticDiagnostic],
    blocking: &[String],
) -> Vec<String> {
    let mut lines = Vec::new();
    for (tool, findings) in [("ruff", new_ruff), ("mypy", new_mypy)] {
        for d in findings {
            sink.push(serde_json::json!({
                "attempt": attempt + 1,
                "tool": tool,
                "file": d.file,
                "line": d.line,
                "code": d.code,
                "message": d.message,
                "blocking": blocking.contains(&d.render()),
            }));
            lines.push(format!("[{tool}] {}", d.render()));
        }
    }
    lines
}

/// Render a tri-state audit answer for the session log.
///
/// [VANTA-WHY] `issue_requires_change` is `Option<bool>` precisely because an
/// omitted field must mean UNKNOWN rather than the `false` ("preserve") that
/// serde's default used to fill in. Logging it as `Some(false)`/`None` would hide
/// which of the three states the model actually returned, and the audit record is
/// the only place a reader can check that the gate failed closed.
fn tri_state_text(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "unknown",
    }
}

/// Append one `intent.check[…]` record per audited test.
///
/// [VANTA-WHY] The verdict is logged for EVERY test — kept, demoted or scrapped,
/// preservation included — so a run's log proves the patch-blind check ran and
/// what it decided. The quote-substring outcome rides along as evidence a reader
/// can re-analyse; it is deliberately not what decided the label.
fn log_intent_checks(
    session_log: &mut SessionLog,
    session_id: &SessionId,
    patch_attempt_id: &PatchAttemptId,
    testing_iteration_id: &crate::session::TestingIterationId,
    source: &str,
    checks: &[IntentCheckRecord],
) -> Result<(), io::Error> {
    for check in checks {
        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: Some(testing_iteration_id.clone()),
            test_id: Some(fresh_test_id()),
            phase: format!("intent.check[{}]", check.test_name),
            summary: format!(
                "source={source} declared={:?} issue_asks_for_behaviour={} issue_requires_change={} scrapped={} quote_present={} quote_in_issue={} quote_names_asserted_input={} asserted_inputs=[{}] reason={}",
                check.declared_intent,
                check.issue_asks_for_behaviour,
                tri_state_text(check.issue_requires_change),
                check.scrapped,
                check.evidence.quote_present,
                check.evidence.quote_in_issue,
                check.evidence.quote_names_asserted_input,
                check.evidence.asserted_inputs.join(","),
                check.reason
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
    }
    Ok(())
}

/// Drop Attacker tests whose body is already in the shared set.
///
/// [VANTA-WHY] Identity is the content-based `normalized_test_body_hash`, so a
/// renamed duplicate is still a duplicate. Keeping them would let the Attacker
/// spend its attack depth re-running one probe it already wrote: every round
/// would produce a self-transition, the shared set would never grow, and a
/// stalled Attacker would look productive. Returns the kept tests plus the names
/// of the ones dropped so the count is logged per round.
fn drop_shared_duplicates(
    generated: &[GeneratedTest],
    shared_tests: &[GeneratedTest],
) -> (Vec<GeneratedTest>, Vec<String>) {
    let mut seen: Vec<String> = shared_tests
        .iter()
        .map(|t| crate::core::normalized_test_body_hash(&t.test_source))
        .collect();
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for test in generated {
        let hash = crate::core::normalized_test_body_hash(&test.test_source);
        if seen.iter().any(|h| h == &hash) {
            dropped.push(test.test_name.clone());
        } else {
            seen.push(hash);
            kept.push(test.clone());
        }
    }
    (kept, dropped)
}

fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{secs}")
}

#[cfg(test)]
mod tests {
    use super::{
        attempt_verdict, bate_baseline_concat, blocking_static_findings, build_source_concat,
        collapse_decision, dependency_freeze_summary_value, harness_infrastructure_collapse,
        modified_test_config_records, patch_attempt_id_for_patch, record_new_static_findings,
        rejection_signal_message, render_committed_test_file,
        require_clean_target_repo_for_open_pr, review_prompt_source_concat, stamp_transition_row,
        stable_patch_fingerprint,
    };
    use crate::agents::creator::SourceFile;
    use crate::agents::attacker::GeneratedTest;
    use crate::core::{AttemptTransition, EvidenceSource, FailureClass, GovernorDecision, TestIntent, TestResult};
    use crate::pipeline::governor::{AttemptVerdict, GovernorMode};
    use crate::pipeline::harness::StaticDiagnostic;
    use crate::session::{PatchAttemptId, SessionId};
    use std::collections::HashMap;

    #[test]
    fn first_harness_collapse_retries_once() {
        assert_eq!(collapse_decision(1, true), GovernorDecision::Continue);
    }

    #[test]
    fn repeated_harness_collapse_is_a_harness_failure() {
        assert_eq!(
            collapse_decision(2, true),
            GovernorDecision::HarnessFailure,
            "a collapse that reproduces is a broken harness, not a transient blip"
        );
    }

    #[test]
    fn harness_collapse_without_retry_budget_aborts() {
        assert_eq!(
            collapse_decision(1, false),
            GovernorDecision::HarnessFailure,
            "with no attempt left to retry, censoring must be reported rather than hidden"
        );
    }

    #[test]
    fn all_infrastructure_is_a_harness_collapse() {
        let classes = vec![
            Some(FailureClass::Infrastructure),
            Some(FailureClass::Infrastructure),
            Some(FailureClass::Infrastructure),
        ];
        assert!(harness_infrastructure_collapse(&classes));
    }

    #[test]
    fn one_healthy_class_is_not_a_harness_collapse() {
        let classes = vec![
            Some(FailureClass::Infrastructure),
            Some(FailureClass::Logic),
        ];
        assert!(!harness_infrastructure_collapse(&classes));
    }

    #[test]
    fn empty_attempt_is_not_a_harness_collapse() {
        assert!(!harness_infrastructure_collapse(&[]));
    }

    #[test]
    fn patch_attempt_id_is_deterministic_for_same_patch() {
        let patch = "diff --git a/foo.py b/foo.py\n+print('hi')\n";
        let first = patch_attempt_id_for_patch(patch);
        let second = patch_attempt_id_for_patch(patch);

        assert_eq!(first.as_str(), second.as_str());
        assert_eq!(
            stable_patch_fingerprint(patch),
            stable_patch_fingerprint(patch)
        );
    }

    #[test]
    fn patch_attempt_id_changes_when_patch_changes() {
        let first = patch_attempt_id_for_patch("diff --git a/foo.py b/foo.py\n+print('hi')\n");
        let second = patch_attempt_id_for_patch("diff --git a/foo.py b/foo.py\n+print('bye')\n");

        assert_ne!(first.as_str(), second.as_str());
    }

    // ---- the three-way verdict -------------------------------------------------
    //
    // [VANTA-WHY] These pin that an environment fault is never reported as a
    // finding about the candidate. "Found a regression" and "could not get clean
    // evidence" must not collapse into the same decision.

    #[test]
    fn a_logic_failure_is_a_rejection() {
        assert_eq!(
            attempt_verdict(
                false,
                &[Some(FailureClass::Logic)],
                false,
                false,
                GovernorMode::Repair
            ),
            AttemptVerdict::Rejected
        );
    }

    #[test]
    fn an_infrastructure_only_round_is_inconclusive_not_a_rejection() {
        assert_eq!(
            attempt_verdict(
                false,
                &[Some(FailureClass::Infrastructure), Some(FailureClass::Unknown)],
                false,
                false,
                GovernorMode::Review
            ),
            AttemptVerdict::Inconclusive,
            "a timeout says nothing about the candidate, so it must not reject it"
        );
    }

    #[test]
    fn a_static_only_round_is_inconclusive_in_repair_mode() {
        assert_eq!(
            attempt_verdict(
                false,
                &[Some(FailureClass::Static)],
                false,
                false,
                GovernorMode::Repair
            ),
            AttemptVerdict::Inconclusive,
            "in repair mode a static finding is the Creator's feedback and the \
             Gatekeeper's gate, not a Logic-class finding"
        );
    }

    // [VANTA-WHY] Fix 3 — the review-mode dead zone. A new static finding in a
    // file the patch touches reaches neither the Gatekeeper (only runs on
    // AllTestsPassed) nor a rejection under the old rule, so review kept
    // attacking a candidate it had already found broken.
    #[test]
    fn a_static_failure_in_a_touched_file_rejects_in_review_mode() {
        assert_eq!(
            attempt_verdict(
                false,
                &[Some(FailureClass::Static)],
                false,
                true,
                GovernorMode::Review
            ),
            AttemptVerdict::Rejected,
            "review mode must be able to reject on a static regression the patch introduced"
        );
    }

    #[test]
    fn a_static_finding_elsewhere_does_not_reject_in_review_mode() {
        assert_eq!(
            attempt_verdict(
                false,
                &[Some(FailureClass::Static)],
                false,
                false,
                GovernorMode::Review
            ),
            AttemptVerdict::Inconclusive,
            "a finding the patch did not introduce is not a finding about the candidate"
        );
    }

    #[test]
    fn a_clean_round_is_a_pass() {
        assert_eq!(
            attempt_verdict(true, &[], false, false, GovernorMode::Repair),
            AttemptVerdict::Passed
        );
    }

    #[test]
    fn a_native_suite_regression_rejects_even_when_the_generated_tests_pass() {
        assert_eq!(
            attempt_verdict(true, &[], true, false, GovernorMode::Review),
            AttemptVerdict::Rejected,
            "the repo's own suite failing on the patch is a regression the generated \
             tests did not happen to cover"
        );
    }

    fn audit_finding(file: &str, code: &str) -> StaticDiagnostic {
        StaticDiagnostic {
            file: file.to_string(),
            line: Some(1),
            code: code.to_string(),
            message: "stub finding".to_string(),
        }
    }

    /// The relaxation the user asked for, at the exact seam that decides it: a
    /// patch that adds ONE style finding in a file it touches must not block a
    /// review round.
    #[test]
    fn a_patch_that_adds_one_style_finding_does_not_block_a_review_round() {
        let touched = vec!["calc.py".to_string()];
        let style = audit_finding("calc.py", "UP032");
        assert!(
            blocking_static_findings(&[&style], &[], &touched).is_empty(),
            "a style rule is reported, not a reason to reject the candidate"
        );
    }

    /// ...but the error classes still do, and only inside the touched files.
    #[test]
    fn only_error_class_findings_in_touched_files_block_a_review_round() {
        let touched = vec!["calc.py".to_string()];
        let f821 = audit_finding("calc.py", "F821");
        let e999 = audit_finding("calc.py", "E999");
        let mypy_error = audit_finding("calc.py", "return-value");
        let elsewhere = audit_finding("vendor/other.py", "F821");
        assert!(!blocking_static_findings(&[&f821], &[], &touched).is_empty());
        assert!(!blocking_static_findings(&[&e999], &[], &touched).is_empty());
        assert!(!blocking_static_findings(&[], &[&mypy_error], &touched).is_empty());
        assert!(
            blocking_static_findings(&[&elsewhere], &[], &touched).is_empty(),
            "an error the patch did not introduce is not a finding about the candidate"
        );
    }

    /// Item 1 of the heretic fix list: a review-mode rejection must name the
    /// finding that caused it. The console used to print only `mypy_new=true`
    /// and the finding itself was dropped (session 40e84874). Every new finding
    /// is now recorded with its site and echoed to the console.
    #[test]
    fn every_new_static_finding_is_recorded_with_its_site() {
        let mut sink: Vec<serde_json::Value> = Vec::new();
        let ruff = StaticDiagnostic {
            file: "src/pkg/mod.py".to_string(),
            line: Some(7),
            code: "F821".to_string(),
            message: "Undefined name `x`".to_string(),
        };
        let mypy = StaticDiagnostic {
            file: "src/heretic/modifiers/abliteration.py".to_string(),
            line: Some(88),
            code: "index".to_string(),
            message: "Item of \"list\" has no attribute ...".to_string(),
        };

        let lines = record_new_static_findings(&mut sink, 0, &[&ruff], &[&mypy], &[]);

        assert_eq!(sink.len(), 2, "both new findings must reach evidence.json");
        assert_eq!(sink[0]["tool"], serde_json::json!("ruff"));
        assert_eq!(sink[0]["attempt"], serde_json::json!(1));
        assert_eq!(sink[0]["line"], serde_json::json!(7));
        assert_eq!(sink[1]["tool"], serde_json::json!("mypy"));
        assert_eq!(
            sink[1]["file"],
            serde_json::json!("src/heretic/modifiers/abliteration.py")
        );
        assert_eq!(sink[1]["line"], serde_json::json!(88));
        assert_eq!(sink[1]["code"], serde_json::json!("index"));
        assert!(
            lines[1].contains("abliteration.py:88"),
            "the console must name the exact site, got: {}",
            lines[1]
        );
    }

    /// Item 11 of the heretic fix list: the evidence file must mark WHICH new
    /// finding actually rejected the round. A style rule (ruff E501) in a touched
    /// file is reported but does not block; an error-class finding (ruff F821) in
    /// the same touched file does. Only the latter may carry `"blocking": true`,
    /// or an audit cannot tell the finding that rejected the candidate apart from
    /// the nit reported beside it.
    #[test]
    fn record_new_static_findings_marks_only_the_blocking_finding() {
        let touched = vec!["calc.py".to_string()];
        let f821 = audit_finding("calc.py", "F821");
        let style = audit_finding("calc.py", "E501");
        let blocking = blocking_static_findings(&[&f821, &style], &[], &touched);
        assert_eq!(blocking.len(), 1, "only the error-class finding blocks");

        let mut sink: Vec<serde_json::Value> = Vec::new();
        record_new_static_findings(&mut sink, 0, &[&f821, &style], &[], &blocking);

        assert_eq!(sink.len(), 2);
        assert_eq!(sink[0]["code"], serde_json::json!("F821"));
        assert_eq!(sink[0]["blocking"], serde_json::json!(true));
        assert_eq!(sink[1]["code"], serde_json::json!("E501"));
        assert_eq!(sink[1]["blocking"], serde_json::json!(false));
    }

    /// Item 10 of the heretic fix list: a finding that says the ANALYSIS
    /// ENVIRONMENT cannot see a module must be recorded but must never reject.
    /// 40e84874 rejected C on `test_auto_rank.py [import-not-found] Cannot find
    /// module "torch"` — a missing dependency, not a defect in the candidate.
    #[test]
    fn an_environment_finding_is_recorded_but_never_blocks() {
        let touched = vec!["tests/test_auto_rank.py".to_string()];
        let missing = audit_finding("tests/test_auto_rank.py", "import-not-found");
        let untyped = audit_finding("tests/test_auto_rank.py", "import-untyped");

        assert!(
            blocking_static_findings(&[], &[&missing], &touched).is_empty(),
            "a missing dependency must not reject the candidate"
        );
        assert!(
            blocking_static_findings(&[], &[&untyped], &touched).is_empty(),
            "a package without stubs must not reject the candidate"
        );

        // ...but the finding still reaches evidence.json, marked non-blocking.
        let mut sink: Vec<serde_json::Value> = Vec::new();
        record_new_static_findings(&mut sink, 0, &[], &[&missing], &[]);
        assert_eq!(sink.len(), 1, "the environment finding must be recorded");
        assert_eq!(sink[0]["code"], serde_json::json!("import-not-found"));
        assert_eq!(sink[0]["blocking"], serde_json::json!(false));

        // A genuine error in the same touched file still blocks.
        let real = audit_finding("tests/test_auto_rank.py", "assignment");
        assert!(
            !blocking_static_findings(&[], &[&real], &touched).is_empty(),
            "a real type error is still a finding about the candidate"
        );
    }

    /// Item 4 of the heretic fix list: the rejection message must name the REAL
    /// signal. 40e84874 printed "the candidate fails 0 test(s): (classified by
    /// the repo's own suite)" while the actual reason was a mypy finding.
    #[test]
    fn a_rejection_names_the_regressed_native_test_not_a_zero_count() {
        let msg = rejection_signal_message(
            &["tests/test_batch.py::test_pads".to_string()],
            &[],
            &[],
            &[],
        );
        assert!(
            msg.contains("native regression: tests/test_batch.py::test_pads"),
            "the regressed native test must be named, got: {msg}"
        );
        assert!(!msg.contains("fails 0 test(s)"), "the old dead-end message is gone");
        assert!(
            !msg.contains("classified by the repo's own suite"),
            "the old dead-end message is gone"
        );
    }

    #[test]
    fn a_rejection_names_the_new_static_finding() {
        let msg = rejection_signal_message(
            &[],
            &["src/heretic/modifiers/abliteration.py:88: error: index".to_string()],
            &[],
            &[],
        );
        assert!(
            msg.contains("new static finding in a touched file: src/heretic/modifiers/abliteration.py:88"),
            "the finding that rejected the round must be named, got: {msg}"
        );
    }

    #[test]
    fn a_rejection_names_the_failing_generated_tests() {
        let msg = rejection_signal_message(
            &[],
            &[],
            &["test_keyword_rate_default".to_string()],
            &[],
        );
        assert!(
            msg.contains("failing test(s): test_keyword_rate_default"),
            "the failing generated test must be named, got: {msg}"
        );
    }

    /// A regressing test the patch also rewrote stays a regression, and the
    /// rejection says so — a human patch may legitimately change a test's
    /// expected behaviour, and the reviewer must see that the rejection rests on
    /// a file the candidate itself touched.
    #[test]
    fn a_regression_the_patch_itself_modified_is_still_a_regression_and_is_marked() {
        let msg = rejection_signal_message(
            &["tests/test_keyword_rate.py::test_print_responses_shows_brackets_literally".to_string()],
            &[],
            &[],
            &["tests/test_keyword_rate.py".to_string()],
        );

        assert!(
            msg.contains("native regression: tests/test_keyword_rate.py::test_print_responses_shows_brackets_literally"),
            "the weakened test's regression must survive the annotation, got: {msg}"
        );
        assert!(
            msg.contains("the patch also modified tests/test_keyword_rate.py"),
            "the rejection must say the patch modified this test, got: {msg}"
        );
    }

    #[test]
    fn a_regression_in_an_untouched_test_carries_no_modification_note() {
        let msg = rejection_signal_message(
            &["tests/test_other.py::test_x".to_string()],
            &[],
            &[],
            &["tests/test_keyword_rate.py".to_string()],
        );

        assert!(msg.contains("native regression: tests/test_other.py::test_x"));
        assert!(
            !msg.contains("the patch also modified"),
            "a test the patch never touched must not be annotated, got: {msg}"
        );
    }

    /// A pytest-config change is recorded with the diff that carries it, and a
    /// pyproject.toml edit that has nothing to do with pytest is not.
    #[test]
    fn a_pytest_config_change_is_recorded_with_its_diff() {
        let patch_files = HashMap::from([
            (
                "pyproject.toml".to_string(),
                "@@ -0,0 +1 @@\n+[tool.pytest.ini_options]\n+addopts = \"-p no:cacheprovider\"\n"
                    .to_string(),
            ),
            (
                "tox.ini".to_string(),
                "@@ -1 +1 @@\n-[pytest]\n+[pytest]\n+addopts = -x\n".to_string(),
            ),
        ]);

        let records = modified_test_config_records(&patch_files);

        let paths: Vec<&str> = records
            .iter()
            .filter_map(|r| r["path"].as_str())
            .collect();
        assert_eq!(paths, vec!["pyproject.toml", "tox.ini"]);
        assert!(
            records
                .iter()
                .all(|r| r["diff"].as_str().is_some_and(|d| !d.is_empty())),
            "each config change must carry its diff"
        );
    }

    #[test]
    fn a_non_pytest_config_edit_is_not_recorded() {
        let patch_files = HashMap::from([(
            "pyproject.toml".to_string(),
            "@@ -1 +1 @@\n-dependencies = [\"a\"]\n+dependencies = [\"a\", \"b\"]\n".to_string(),
        )]);

        assert!(
            modified_test_config_records(&patch_files).is_empty(),
            "pyproject.toml carries non-test content; an unrelated edit is not a test-config change"
        );
    }

    /// The message and the gate read the SAME list, so the finding a rejection
    /// names is exactly the finding that rejected it.
    #[test]
    fn the_blocking_list_is_what_a_rejection_names() {
        let touched = vec!["calc.py".to_string()];
        let f821 = audit_finding("calc.py", "F821");
        let style = audit_finding("calc.py", "UP032");
        let elsewhere = audit_finding("vendor/other.py", "F821");

        let blocking = blocking_static_findings(&[&f821, &style], &[], &touched);
        assert_eq!(blocking.len(), 1);
        assert!(blocking[0].contains("F821"));
        assert!(blocking_static_findings(&[&elsewhere], &[], &touched).is_empty());
    }

    /// Item 2 of the heretic fix list: a patch-insensitive failure (TestDefect)
    /// must NOT reject a review round — it is a property of the test, not the
    /// candidate. The round is Inconclusive, so the Attacker is asked to revise.
    #[test]
    fn a_test_defect_does_not_reject_a_review_round() {
        assert_eq!(
            attempt_verdict(
                false,
                &[Some(FailureClass::TestDefect)],
                false,
                false,
                GovernorMode::Review
            ),
            AttemptVerdict::Inconclusive,
            "a test defect is neither a rejection nor a pass"
        );
    }

    /// The inspectable promise of the repair-mode apply-failure path.
    ///
    /// [VANTA-WHY] This is the direct answer to "how does a failed apply count
    /// for ECR/EIR": it does NOT. The attempt never produced a tree, so the row
    /// is `complete: false` and therefore CENSORED, and
    /// `is_countable_patch_transition` refuses every censored row — no failed
    /// apply can reach the denominator. `apply_failed` rides alongside (not
    /// instead of) so the analysis can report how many holes were Vanta's own
    /// apply path rather than an unattributed gap.
    #[test]
    fn a_failed_apply_is_recorded_as_a_censored_apply_failed_row() {
        let transition = AttemptTransition {
            from_attempt: 0,
            to_attempt: 1,
            shared_set_id: "set-a".into(),
            test_id: "hash-add".into(),
            test_name: "test_behavior_preservation_only".into(),
            from_result: TestResult::Passed,
            to_result: TestResult::Error,
            from_failure_class: None,
            to_failure_class: None,
            complete: false,
            in_s0: true,
            evidence_source: EvidenceSource::AttackerGenerated,
            test_intent: TestIntent::BehaviorPreservation,
        };
        let session_id = SessionId::new("apply-failed-test");
        let attempt_ids = vec![PatchAttemptId::new("p0"), PatchAttemptId::new("p1")];

        let row = stamp_transition_row(&transition, &session_id, &attempt_ids, &[1]);
        assert!(row.apply_failed, "the failed-apply attempt must be marked");
        assert!(
            row.censored,
            "an attempt whose patch never applied is a hole, not an observation"
        );
        assert!(
            !row.is_countable_patch_transition(),
            "a hole must never reach the EIR/ECR denominator"
        );
        assert!(
            row.from_patch_attempt_id.is_some() && row.to_patch_attempt_id.is_some(),
            "both candidate ids exist even when the newer one never ran"
        );
    }

    #[test]
    fn open_pr_requires_a_clean_target_repo() {
        let err = require_clean_target_repo_for_open_pr(Some(true)).unwrap_err();
        assert!(
            err.contains("clean target repository working tree"),
            "dirty target repos must refuse --open-pr, got {err}"
        );
    }

    #[test]
    fn open_pr_accepts_a_clean_target_repo() {
        assert!(require_clean_target_repo_for_open_pr(Some(false)).is_ok());
    }

    #[test]
    fn committed_test_file_uses_test_prefix_and_hoists_shared_imports() {
        let tests = vec![
            GeneratedTest {
                test_name: "test_one".into(),
                test_source: "from toy_calc import shipping_band\n\ndef test_one():\n    assert shipping_band(500) == \"standard\"\n".into(),
                intent: crate::core::TestIntent::BehaviorPreservation,
                reason: "keep 500 unchanged".into(),
                issue_quote: None,
                relabeled: false,
            },
            GeneratedTest {
                test_name: "test_two".into(),
                test_source: "from toy_calc import shipping_band\n\ndef test_two():\n    assert shipping_band(1000) == \"heavy\"\n".into(),
                intent: crate::core::TestIntent::BugReproduction,
                reason: "fix 1000 boundary".into(),
                issue_quote: None,
                relabeled: false,
            },
        ];

        let rendered = render_committed_test_file(&tests);
        assert_eq!(
            rendered.matches("from toy_calc import shipping_band").count(),
            1,
            "shared imports should be hoisted once to the top of the committed test file"
        );
        assert!(rendered.contains("def test_one():"));
        assert!(rendered.contains("def test_two():"));
    }

    #[test]
    fn dependency_freeze_summary_value_is_single_line() {
        let rendered = dependency_freeze_summary_value(Some(
            "flake8==7.4.1\nisort==6.1.0\npytest==8.4.2\n",
        ));
        assert_eq!(rendered, "flake8==7.4.1|isort==6.1.0|pytest==8.4.2");
        assert!(
            !rendered.contains('\n'),
            "session.start summary must stay on one line"
        );
    }

    #[test]
    fn dependency_freeze_summary_value_defaults_to_none() {
        assert_eq!(dependency_freeze_summary_value(None), "none");
    }

    #[test]
    fn review_prompt_context_includes_touched_imports_and_covering_tests() {
        let repo_files = vec![
            SourceFile {
                path: "heretic/keyword_rate.py".into(),
                contents: "from .plugin import Context\n\nclass KeywordRate:\n    pass\n".into(),
            },
            SourceFile {
                path: "heretic/plugin.py".into(),
                contents: "class Context:\n    pass\n".into(),
            },
            SourceFile {
                path: "heretic/other.py".into(),
                contents: "class Other:\n    pass\n".into(),
            },
            SourceFile {
                path: "tests/test_keyword_rate.py".into(),
                contents: "from heretic.keyword_rate import KeywordRate\nfrom heretic.plugin import Context\n\ndef test_keyword_rate():\n    assert KeywordRate is not None\n".into(),
            },
            SourceFile {
                path: "tests/test_other.py".into(),
                contents: "from heretic.other import Other\n\ndef test_other():\n    assert Other is not None\n".into(),
            },
        ];
        let mut original_files = HashMap::new();
        for file in &repo_files {
            original_files.insert(file.path.clone(), file.contents.clone());
        }

        let rendered = review_prompt_source_concat(
            &repo_files,
            &original_files,
            &HashMap::from([("heretic/keyword_rate.py".to_string(), "patched".to_string())]),
        );

        assert!(rendered.contains("# FILE: heretic/keyword_rate.py"));
        assert!(rendered.contains("# FILE: heretic/plugin.py"));
        assert!(rendered.contains("# FILE: tests/test_keyword_rate.py"));
        assert!(
            !rendered.contains("# FILE: heretic/other.py"),
            "unrelated imports should stay out of the review prompt"
        );
        assert!(
            !rendered.contains("# FILE: tests/test_other.py"),
            "unrelated tests should stay out of the review prompt"
        );
    }

    // [VANTA-WHY] BATE's tree is the patch-free ORIGINAL tree (see
    // `bate_baseline_concat`). "Document only" is not enough here: a candidate
    // that could reach BATE with its own `conftest.py` or an edited test could
    // validate itself. This locks the invariant in place so a future edit that
    // hands BATE the patched tree fails here.
    #[test]
    fn bate_validates_against_a_tree_with_no_patch_added_or_modified_test_files() {
        let original_files = HashMap::from([
            (
                "calc.py".to_string(),
                "def double(x):\n    return x\n".to_string(),
            ),
            (
                "tests/test_core.py".to_string(),
                "from calc import double\n\n\ndef test_double():\n    assert double(2) == 4\n"
                    .to_string(),
            ),
        ]);

        // The candidate the run would otherwise be tempted to hand BATE: it
        // breaks the source, adds a root conftest that forces every outcome to
        // "passed", and rewrites the covering test to assert nothing.
        let mut patched_files = original_files.clone();
        patched_files.insert(
            "calc.py".to_string(),
            "def double(x):\n    return x + 1\n".to_string(),
        );
        patched_files.insert(
            "conftest.py".to_string(),
            "import pytest\n\n\ndef pytest_runtest_makereport(item, call):\n    pass\n".to_string(),
        );
        patched_files.insert(
            "tests/test_core.py".to_string(),
            "from calc import double\n\n\ndef test_double():\n    assert True\n".to_string(),
        );

        let bate_tree = bate_baseline_concat(&original_files);
        let patched_tree = build_source_concat(&patched_files);

        assert_ne!(
            bate_tree, patched_tree,
            "the two trees must genuinely differ, or this test proves nothing"
        );
        assert_eq!(
            bate_tree,
            build_source_concat(&original_files),
            "BATE must receive exactly the patch-free original tree"
        );
        assert!(
            !bate_tree.contains("pytest_runtest_makereport"),
            "a patch-added conftest must never reach BATE's tree"
        );
        assert!(
            !bate_tree.contains("assert True"),
            "a patch-modified test must never reach BATE's tree"
        );
        assert!(bate_tree.contains("assert double(2) == 4"));
    }
}
