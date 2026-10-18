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

use std::collections::HashMap;
use std::fs;
use std::io;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agents::attacker::{
    AttackerRequest, GeneratedTest, IntentCheckRecord, SharedProbe, attacker_generate_tests,
    audit_tests_intent,
};
use crate::agents::creator::{
    CreatorRequest, creator_generate_patch, split_multi_file_patch, split_patch_into_files,
};
use crate::config::LlmConfig;
use crate::core::{
    AttemptTransition, EvidenceSource, FailureClass, GovernorDecision, TestEvidence, TestResult,
    VerificationOutcome, VerificationPolicy,
};
use crate::pipeline::bate::run_bate;
use crate::pipeline::candidate::{CandidatePair, TestCommand};
use crate::pipeline::diagnostician::diagnose_failure;
use crate::pipeline::gatekeeper::{Gatekeeper, GatekeeperVerdict};
use crate::pipeline::governor::{
    AttemptVerdict, Governor, GovernorMode, SharedObservation,
};
use crate::pipeline::harness::{self, TestExecutionDetails};
use crate::pipeline::patch::apply_patch;
use crate::repo::Repo;
use crate::session::{
    PatchAttemptId, SessionId, SessionLog, SessionRecord, TransitionRow, TransitionWriter,
    ecr_eir_from_rows, fresh_session_id, fresh_test_id, fresh_testing_iteration_id, vanta_revision,
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
// run — drive one full Vanta run and return the Governor's final decision
// ============================================================================
pub async fn run(
    config: RunConfig,
) -> Result<GovernorDecision, Box<dyn std::error::Error + Send + Sync>> {
    // ---- Config ----
    let mut cfg = LlmConfig::from_env()?;
    if let Some(m) = &config.model_override {
        cfg.model_id = m.clone();
    }
    println!("{cfg}\n");

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

    // ---- Auto-detect repo dependencies and prepare the VM checkpoint ----
    // [VANTA-WHY] The base shuru checkpoint only ships python3 + pytest + ruff +
    // mypy. A real repo declares its own deps which are absent inside the VM, so
    // tests die at import and BATE sees `Error` for everything. Build/cache a
    // checkpoint up front so test runs use the warmed environment.
    if std::env::var("VANTA_SHURU_CHECKPOINT").is_err()
        && std::env::var("VANTA_SHURU_AUTO_DEPS").as_deref() != Ok("0")
        && crate::sandbox::shuru_available()
    {
        println!("▌ VM deps: detecting declared dependencies...");
        match crate::sandbox::shuru::ShuruBackend::ensure_repo_checkpoint(&config.repo_path) {
            Ok(Some((name, disk_mb))) => {
                println!("▌ VM deps checkpoint: {name} (disk {disk_mb} MB)");
                // SAFETY: single-threaded CLI; no other thread reads env concurrently.
                unsafe {
                    std::env::set_var("VANTA_SHURU_CHECKPOINT", &name);
                    std::env::set_var("VANTA_SHURU_DISK_MB", disk_mb.to_string());
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
    let requested_backend = crate::sandbox::requested_backend_label()?;
    let actual_backend = crate::sandbox::actual_backend_label().await?;
    let session_id = fresh_session_id();
    let mut session_log = SessionLog::open(Path::new("vanta_sessions"), session_id.clone())?;
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
    // [VANTA-WHY] `prompt_version` names EVERY agent prompt, not just the
    // Attacker's: the Attacker, the patch-blind intent auditor, the Creator and
    // the Diagnostician each change what a run can find or accept. If only the
    // Attacker were tagged, a later edit to the intent-check or Creator prompt
    // would be logged as "no change" and two different experiments would look
    // identical. Keys are the prompt names, `;`-separated inside the value.
    let prompt_versions = format!(
        "attacker={};intent_check={};creator={};diagnostician={}",
        crate::agents::attacker::ATTACKER_PROMPT_VERSION,
        crate::agents::attacker::INTENT_AUDITOR_PROMPT_VERSION,
        crate::agents::creator::CREATOR_PROMPT_VERSION,
        crate::pipeline::diagnostician::DIAGNOSTICIAN_PROMPT_VERSION,
    );
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
        // [VANTA-WHY] A run-level record: no patch attempt exists yet, so this is
        // null rather than an invented id.
        patch_attempt_id: None,
        testing_iteration_id: None,
        test_id: None,
        phase: "session.start".to_string(),
        summary: format!(
            "model={} model_version={} prompt_version={} provider={} api_key={} \
             vanta_version={} vanta_commit={} vanta_dirty={} requested_backend={} actual_backend={} max_attempts={} \
             baseline_test_timeout_secs={} patched_test_timeout_secs={} test_cmd={} mode={} benchmark_instance={}",
            cfg.model_id,
            std::env::var("VANTA_MODEL_VERSION").unwrap_or_else(|_| "unknown".into()),
            prompt_versions,
            cfg.base_url,
            if cfg.api_key.is_some() { "set" } else { "none" },
            env!("CARGO_PKG_VERSION"),
            vanta_commit,
            vanta_dirty,
            requested_backend,
            actual_backend,
            config.max_attempts,
            baseline_test_timeout_secs,
            patched_test_timeout_secs,
            format!("{} {}", test_cmd.program, test_cmd.args.join(" ")),
            run_mode,
            "none",
        ),
        timestamp: now_iso(),
        benchmark_instance: None,
        benchmark_label: None,
    })?;

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
    let decision = loop {
        let attempt = governor.current_attempt();
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
            let creator_response = run_with_spinner("Creator generating patch", async {
                creator_generate_patch(&cfg, &creator_request).await
            })
            .await?;
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

        for (path, patch_content) in &patch_files {
            print!(
                "  ⏳ {path}: applying ({:.1}KB patch)... ",
                patch_content.len() as f64 / 1024.0
            );
            let _ = io::Write::flush(&mut io::stdout());
            if let Some(original) = original_files.get(path) {
                let (patched, changed) = apply_patch(original, patch_content);
                if changed {
                    patched_files.insert(path.clone(), patched);
                    any_changed = true;
                    all_patch_failed = false;
                    println!("✔ patched (changed)");
                } else {
                    println!("✗ patch did not change file");
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

        if all_patch_failed {
            // Write forensics
            let faillog = debug_dir.join("patch_application_failed.json");
            let _ = fs::write(
                &faillog,
                serde_json::to_string_pretty(&serde_json::json!({
                    "patch_text": patch_text,
                    "files_touched": patch_files.keys().cloned().collect::<Vec<_>>(),
                    "reason": "no file was changed by any strategy"
                }))
                .unwrap(),
            );
            println!("  ✗ PATCH APPLICATION FAILED — see {}", faillog.display());
            session_log.append(SessionRecord {
                session_id: session_id.clone(),
                patch_attempt_id: Some(patch_attempt_id.clone()),
                testing_iteration_id: Some(testing_iteration_id.clone()),
                test_id: None,
                phase: "patch.apply_failed".to_string(),
                summary: "no file was changed by any patch strategy".to_string(),
                timestamp: now_iso(),
                benchmark_instance: None,
                benchmark_label: None,
            })?;

            let decision = governor.decide(
                governor_mode,
                AttemptVerdict::Inconclusive,
                false,
            );
            break decision;
        }
        let _ = any_changed; // used implicitly via all_patch_failed

        // ---- Build original source string (for Attacker + BATE) ----
        let original_concat = original_files
            .iter()
            .map(|(p, c)| format!("# FILE: {p}\n{c}"))
            .collect::<Vec<_>>()
            .join("\n\n");

        // ---- 3. Attacker ----
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
            original_source: original_concat.clone(),
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
        let attacker_response = run_with_spinner("Attacker generating tests", async {
            attacker_generate_tests(&cfg, &attacker_request).await
        })
        .await?;
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
            &original_concat,
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
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: Some(testing_iteration_id.clone()),
            test_id: None,
            phase: "attacker.summary".to_string(),
            summary: format!(
                "generated_tests={} dropped_duplicates={} intent_scrapped={} valid_tests={}",
                attacker_response.tests.len(),
                dropped_duplicates.len(),
                intent_scrapped,
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
        println!("\n▌ BATE: validating tests against baseline...");
        let bate_run = run_with_spinner("BATE validating tests against baseline", async {
            run_bate(&cfg, &repo.issue_text, &original_concat, &attacker_tests, &test_cmd).await
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
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: Some(testing_iteration_id.clone()),
            test_id: None,
            phase: "bate.summary".to_string(),
            summary: format!(
                "valid={} scrapped={} discard_rate_pct={:.1}",
                bate_report.valid_tests.len(),
                bate_report.scrapped_tests.len(),
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

        if valid_tests.is_empty() && shared_tests.is_empty() {
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
        if valid_tests.is_empty() {
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

        let patched_concat = patched_files
            .iter()
            .map(|(p, c)| format!("# FILE: {p}\n{c}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let snapshot_root = debug_dir.join(format!("attempt_{}_patched_tree", attempt + 1));
        let _ = harness::write_tree_snapshot(&snapshot_root, &patched_concat);

        // Build the combined test list: shared first, then new.
        let mut tests_to_run: Vec<&GeneratedTest> = Vec::new();
        for t in &shared_tests {
            tests_to_run.push(t);
        }
        for t in &valid_tests {
            tests_to_run.push(t);
        }

        let patched_results = run_with_spinner("Running tests on patched code", async {
            harness::run_tests_parallel_detailed(&patched_concat, &tests_to_run, &test_cmd).await
        })
        .await?;

        // Static analysis diff — keep the raw streams so the Static payload the
        // Creator receives names the exact diagnostics the patch introduced.
        let (base_ruff, _base_ruff_err, base_mypy, _base_mypy_err) =
            harness::run_static_raw(&original_concat).await;
        let (patch_ruff, patch_ruff_err, patch_mypy, patch_mypy_err) =
            harness::run_static_raw(&patched_concat).await;
        let ruff_new = patch_ruff > base_ruff;
        let mypy_new = patch_mypy > base_mypy;
        println!("  Static: ruff_new={ruff_new}, mypy_new={mypy_new}");

        // ---- 5b. Native suite gate — the repo's OWN tests ----
        // [VANTA-WHY] Generated tests depend on what the Attacker sampled; the
        // repo's own suite does not. A test that passes on the original and
        // fails on the patch is a regression even when no generated test covers
        // that input. This is its own `native_suite` section — not part of the
        // shared set, never counted in EIR/ECR — because it is not a transition
        // measurement.
        println!("\n▌ NATIVE suite: repo's own tests (original vs patch)...");
        let native_original = harness::run_native_suite(&original_concat, &test_cmd).await;
        let native_patched = harness::run_native_suite(&patched_concat, &test_cmd).await;
        let native_regressions =
            harness::native_regressions(&native_original.tests, &native_patched.tests);
        let native_regression = !native_regressions.is_empty();
        println!(
            "  Native: original ran={} ({} tests), patched ran={} ({} tests), {} regression(s)",
            native_original.ran,
            native_original.tests.len(),
            native_patched.ran,
            native_patched.tests.len(),
            native_regressions.len()
        );
        session_log.append(SessionRecord {
            session_id: session_id.clone(),
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
                    &patch_ruff_err,
                    &patch_mypy_err,
                    !net_alive,
                    &patched_concat,
                    &cfg,
                    &original_concat,
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
        let transitions = governor.transitions();
        if transitions.len() > persisted_transitions {
            let rows: Vec<TransitionRow> = transitions[persisted_transitions..]
                .iter()
                .map(|t| stamp_transition_row(t, &session_id, &attempt_patch_ids))
                .collect();
            transition_writer.append(&rows)?;
            persisted_transitions = transitions.len();
            // Keep the same rows in memory so the live print below derives
            // EIR/ECR from them with the one shared function, not a second rule.
            metric_rows.extend(rows);
        }
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
        let verdict = attempt_verdict(all_passed, &patched_classes, native_regression);
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
                    "  ✖ REJECTED — the candidate fails {} test(s): {}",
                    failing_names.len(),
                    if failing_names.is_empty() {
                        "(classified by the repo's own suite)".to_string()
                    } else {
                        failing_names.join(", ")
                    }
                );
            }
        } else {
            // The SAME function the M5 analysis layer runs over the on-disk rows,
            // so the live number and the paper's number cannot drift apart.
            let snap = ecr_eir_from_rows(&metric_rows);
            println!(
                "\n▌ GOVERNOR: decision={:?}, ECR={} ({:.2}%), EIR={} ({:.2}%)",
                decision,
                snap.ecr_count,
                snap.ecr_rate() * 100.0,
                snap.eir_count,
                snap.eir_rate() * 100.0
            );
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
                final_accepted_tests = Some(
                    valid_tests
                        .iter()
                        .map(|t| t.test_source.as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                );
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
            GovernorDecision::Continue => {
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
                    prior_feedback = Some(build_creator_feedback(&failed, ruff_new, mypy_new));
                    println!(
                        "  → Feeding {} failure(s) back to Creator for retry",
                        failed.len()
                    );
                }
            }
        }
    };

    // ---- Final outcome ----
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  FINAL: {:?}", decision);
    println!("══════════════════════════════════════════════════════════════");

    // [VANTA-WHY] "Fixed at attempt 1" is its own number, not a rate. A repair
    // that passes on the first candidate has no N→N+1 transition, so ECR/EIR
    // cannot see it by construction. Reporting it separately tells a reader what
    // share of defects the rates actually cover.
    let attempts_run = attempt_patch_ids.len();
    let fixed_at_attempt_1 = decision == GovernorDecision::AllTestsPassed && attempts_run == 1;
    session_log.append(SessionRecord {
        session_id: session_id.clone(),
        patch_attempt_id: attempt_patch_ids.last().cloned(),
        testing_iteration_id: None,
        test_id: None,
        phase: "run.summary".to_string(),
        summary: format!(
            "attempts_run={} fixed_at_attempt_1={} final_decision={:?}",
            attempts_run, fixed_at_attempt_1, decision
        ),
        timestamp: now_iso(),
        benchmark_instance: None,
        benchmark_label: None,
    })?;
    println!(
        "  • attempts_run={} fixed_at_attempt_1={} — a repair with no N→N+1 transition is outside ECR/EIR by construction",
        attempts_run, fixed_at_attempt_1
    );

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
                println!(
                    "  ✔ Ready for PR. Use: cd {} && git diff",
                    stage_dir.display()
                );

                // ---- Open PR if requested ----
                if config.open_pr {
                    let accepted_tests = final_accepted_tests.clone().unwrap_or_default();
                    match crate::pr::open_pr(
                        &config.repo_path,
                        patched,
                        &accepted_tests,
                        &final_evidence,
                        &session_id,
                        &repo.issue_text,
                    ) {
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
    let _ = fs::write(
        &evidence_path,
        serde_json::to_string_pretty(&final_evidence).unwrap(),
    );
    println!("\n▌ Evidence → {}", evidence_path.display());
    println!(
        "▌ Transitions → {} ({} row(s))",
        transitions_path.display(),
        persisted_transitions
    );
    let session_path = session_log
        .sink_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| format!("vanta_sessions/{}.json", session_id.as_str()));
    println!("▌ Session  → {session_path}");

    Ok(decision)
}

// ============================================================================
// build_creator_feedback — structured feedback for the Creator from failures
// ============================================================================
fn build_creator_feedback(failed: &[&TestEvidence], ruff_new: bool, mypy_new: bool) -> String {
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
fn stamp_transition_row(
    transition: &AttemptTransition,
    session_id: &SessionId,
    attempt_patch_ids: &[PatchAttemptId],
) -> TransitionRow {
    let from_id = attempt_patch_ids
        .get(transition.from_attempt as usize)
        .cloned();
    let to_id = attempt_patch_ids
        .get(transition.to_attempt as usize)
        .cloned();
    TransitionRow::from_transition(transition, session_id, from_id, to_id)
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
// attempt's evidence. Only a Logic-class failure is a finding about the
// candidate: it is the one class the Diagnostician can attribute to the code
// (Spec §7). Static failures are the Gatekeeper's own concern and already block
// approval on its path, and Infrastructure / Unknown mean we could not get clean
// evidence — reporting either as a rejection would claim the model was wrong
// when the environment was. A repo-native regression (`native_suite`) is a real
// finding even though it is not part of the shared set, so it is passed in
// separately.
fn attempt_verdict(
    all_passed: bool,
    failure_classes: &[Option<FailureClass>],
    native_regression: bool,
) -> AttemptVerdict {
    if native_regression
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
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: Some(testing_iteration_id.clone()),
            test_id: Some(fresh_test_id()),
            phase: format!("intent.check[{}]", check.test_name),
            summary: format!(
                "source={source} declared={:?} issue_asks_for_behaviour={} issue_requires_change={} scrapped={} quote_present={} quote_in_issue={} quote_names_asserted_input={} asserted_inputs=[{}] reason={}",
                check.declared_intent,
                check.issue_asks_for_behaviour,
                check.issue_requires_change,
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
        attempt_verdict, collapse_decision, harness_infrastructure_collapse,
        patch_attempt_id_for_patch, stable_patch_fingerprint,
    };
    use crate::core::{FailureClass, GovernorDecision};
    use crate::pipeline::governor::AttemptVerdict;

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
            attempt_verdict(false, &[Some(FailureClass::Logic)], false),
            AttemptVerdict::Rejected
        );
    }

    #[test]
    fn an_infrastructure_only_round_is_inconclusive_not_a_rejection() {
        assert_eq!(
            attempt_verdict(
                false,
                &[Some(FailureClass::Infrastructure), Some(FailureClass::Unknown)],
                false
            ),
            AttemptVerdict::Inconclusive,
            "a timeout says nothing about the candidate, so it must not reject it"
        );
    }

    #[test]
    fn a_static_only_round_is_inconclusive() {
        assert_eq!(
            attempt_verdict(false, &[Some(FailureClass::Static)], false),
            AttemptVerdict::Inconclusive,
            "Static is the Gatekeeper's own gate; it is not a Logic-class finding here"
        );
    }

    #[test]
    fn a_clean_round_is_a_pass() {
        assert_eq!(attempt_verdict(true, &[], false), AttemptVerdict::Passed);
    }

    #[test]
    fn a_native_suite_regression_rejects_even_when_the_generated_tests_pass() {
        assert_eq!(
            attempt_verdict(true, &[], true),
            AttemptVerdict::Rejected,
            "the repo's own suite failing on the patch is a regression the generated \
             tests did not happen to cover"
        );
    }
}
