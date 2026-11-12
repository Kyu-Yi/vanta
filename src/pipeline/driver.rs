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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agents::attacker::{
    AttackerRequest, GeneratedTest, IntentCheckRecord, QuarantinedProbe, SharedProbe,
    attacker_generate_tests, audit_tests_intent,
};
use crate::agents::creator::{
    CreatorRequest, SourceFile, creator_generate_patch, split_multi_file_patch,
    split_patch_into_files,
};
use crate::agents::judge::{JudgeError, JudgeRequest, JudgeVerdict, judge_candidate};
use crate::config::LlmConfig;
use crate::core::{
    AttemptTransition, EvidenceSource, FailureClass, GovernorDecision, TestEvidence, TestIntent,
    TestResult, VerificationOutcome, VerificationPolicy, static_finding_identity,
};
use crate::llm::{build_reqwest_client, drain_call_usages, reset_call_usages, server_top_p_default};
use crate::pipeline::bate::run_bate;
use crate::pipeline::candidate::TestCommand;
use crate::pipeline::corroborate::{self, CorroborationInput, FailureEvidence};
use crate::pipeline::diagnostician::{diagnose_failure, is_quarantined};
use crate::pipeline::gatekeeper::{Gatekeeper, GatekeeperVerdict};
use crate::pipeline::governor::{
    AttemptVerdict, Governor, GovernorMode, SharedObservation,
};
use crate::pipeline::harness::{self, TestExecutionDetails};
use crate::pipeline::patch::{
    ApplyFailure, apply_patch_detailed, apply_patch_strict, patch_deleted_paths,
};
use crate::pipeline::safe_path::{confined_join, unsafe_relative_path_reason};
use crate::pipeline::summary_escape::{encode_summary_path_list, encode_summary_value};
use crate::pipeline::{
    BUDGET_EXHAUSTED, MODEL_OUTPUT_INVALID, REVISE_BUDGET_PER_ATTEMPT, ReviseBudget,
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
    /// Environment injected into every sandbox command of this run, from the
    /// harness's recorded per-project rule (e.g. `PYTHONPATH=./docs_src`).
    /// Empty in the normal case.
    pub sandbox_env: Vec<(String, String)>,
    /// The harness's recorded per-project pre-compile rule: tree-relative `.py`
    /// files every suite VM byte-compiles before the test command runs (item L).
    /// Empty in the normal case, and an empty rule leaves every run exactly as
    /// it was.
    pub precompile: Vec<String>,
    /// [VANTA-WHY] M3 item W — how many native tests the row's own `prepare`
    /// collected for this bug (`pass_to_pass_count` plus the native-failure
    /// baseline, the figure `prepare_summary.md` prints). `None` means the
    /// caller did not record one, which is every manual invocation; `Some(0)`
    /// means the bug genuinely ships no native tests.
    ///
    /// The native gate is `run_native_suite(original) vs run_native_suite(patch)`
    /// and `native_regressions` over an empty original is VACUOUSLY empty, so a
    /// suite that collected nothing reads as "0 regressions" and the attempt is
    /// approved on no evidence (measured: pandas/1 session c9289597). With a
    /// positive count recorded, an in-session suite that collected nothing is an
    /// infrastructure failure instead of a pass. Zero and `None` leave every
    /// pre-item-W run byte-for-byte as it was.
    pub native_expected_tests: Option<usize>,
    /// [VANTA-WHY] M3 item D1 / v1-m3.1 — `--native-timeout-secs N`: the wall-clock cap
    /// for the in-session native suite, derived per bug from the run's own prepare
    /// measurement (`max(1200, 10 × measured)`, floor `max(1200, test_timeout_secs())`).
    /// `None` keeps the session floor (`harness::native_session_timeout_floor`, 1200 s),
    /// which is every hand-run invocation and every row whose prepare recorded no time.
    ///
    /// The native gate runs the repo's own suite twice in-session — original tree vs
    /// patched tree, both under THIS one cap — and that suite is not always fast:
    /// pandas/1's takes ~487 s through prepare while the fixed 120 s cap killed BOTH
    /// in-session runs, so the row died as a false "collected nothing" (a dev run).
    /// The patched side is the suite PLUS the patch, so the cap has
    /// to absorb the patch's own cost as well: a one-file `luigi/scheduler.py`
    /// patch cost ~3.6× the unpatched suite on an idle machine (119/116 s vs 33/31 s),
    /// above D1's 3× (a dev probe, 2026-11-12). A per-bug cap
    /// derived from a measured run is the only value that is both large enough for the
    /// bug and still bounded; the two sides share it because the cap is part of the
    /// suite-cache key (`suite_cache::native_key`, M3 item S1).
    pub native_timeout_secs: Option<u64>,
    /// Path to a clean original tree (git worktree of the base commit) for
    /// two-tree mode. If set, BATE runs against this tree instead of the
    /// in-memory original files. Spec §7: "both trees present simultaneously".
    pub original_tree: Option<PathBuf>,
    /// [VANTA-WHY] M3 item 9 — fixed-round repair (`--fixed-rounds K`).
    ///
    /// `None` is every run before this option and every ordinary run: the loop
    /// stops the moment an attempt passes. `Some(k)` makes the loop run EXACTLY
    /// `k` attempts — a passing attempt is evidence, not the end, because the
    /// attempt-to-attempt transitions M3.5's EIR/ECR measurement needs only
    /// exist between attempts (Liu & Meng continue after a correct answer to
    /// see whether the model breaks it). The final verdict is the one attempt
    /// `k` produces, and every attempt's patch, test set and feedback is
    /// stored. Repair only: a `--patch` review has no Creator to re-ask.
    pub fixed_rounds: Option<u32>,
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

/// Read a whole number of SECONDS from the environment, or `None` when unset.
///
/// [VANTA-WHY] §14.15 — a malformed value is an error rather than a silent
/// default: `VANTA_DEADLINE_SECS=1h` read as "no deadline" would give the
/// operator the opposite of what they asked for, and the failure would show up
/// hours later as a harness kill that looks like a machine fault.
fn env_secs(name: &str) -> Result<Option<u64>, String> {
    match std::env::var(name) {
        Ok(raw) => raw.trim().parse::<u64>().map(Some).map_err(|_| {
            format!(
                "CONFIG_ERROR: {name}='{}' is not a number of seconds.",
                raw.trim()
            )
        }),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "CONFIG_ERROR: {name} env var contained non-UTF-8 bytes."
        )),
    }
}

/// Read a floating-point number from the environment, or `None` when unset.
///
/// [VANTA-WHY] §14.16 — the same argument as `env_secs`: `VANTA_PRESENCE_PENALTY=high`
/// read as "no override" would run an arm the operator did not ask for and
/// silently compare it against the wrong baseline. An empty value is `None`,
/// which is what a shell that expands an unset variable (`VANTA_X=`) hands over
/// and must not become a parse error.
fn env_float(name: &str) -> Result<Option<f64>, String> {
    match std::env::var(name) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            trimmed
                .parse::<f64>()
                .map(Some)
                .map_err(|_| format!("CONFIG_ERROR: {name}='{trimmed}' is not a number."))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "CONFIG_ERROR: {name} env var contained non-UTF-8 bytes."
        )),
    }
}

/// Read the per-call-site output caps from `VANTA_MAX_TOKENS_<SITE>` (§14.20).
///
/// [VANTA-WHY] §14.20 — the switch is derived from the schedule `site` names, so
/// the set of caps a run can ask for and the set of sites that exist cannot
/// drift: `attacker_revise` is `VANTA_MAX_TOKENS_ATTACKER_REVISE` and there is no
/// spelling for a site the pin table would panic on. A malformed value is an
/// error for `env_secs`'s reason — `VANTA_MAX_TOKENS_ATTACKER_REVISE=12k` read as
/// "no override" would run an arm the operator did not ask for and compare it
/// against the wrong baseline. Zero is rejected rather than honoured: a cap of
/// zero is a site that can emit nothing, which is a misconfiguration and never a
/// measurement, and the operator who wants the schedule's value unset the var.
///
/// [VANTA-WHY] An empty value is skipped, not an error, for `env_float`'s reason:
/// a shell expanding an unset variable (`VANTA_X=`) must not fail the run.
fn env_max_tokens_overrides() -> Result<Vec<(&'static str, u32)>, String> {
    let mut overrides = Vec::new();
    for &(site, _, _) in crate::llm::SAMPLING_SCHEDULE {
        let name = crate::llm::max_tokens_env_var(site);
        match std::env::var(&name) {
            Ok(raw) => {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let cap = trimmed.parse::<u32>().map_err(|_| {
                    format!("CONFIG_ERROR: {name}='{trimmed}' is not a number of tokens.")
                })?;
                if cap == 0 {
                    return Err(format!(
                        "CONFIG_ERROR: {name}=0 would let `{site}` emit nothing; unset it instead."
                    ));
                }
                overrides.push((site, cap));
            }
            Err(std::env::VarError::NotPresent) => {}
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(format!(
                    "CONFIG_ERROR: {name} env var contained non-UTF-8 bytes."
                ));
            }
        }
    }
    Ok(overrides)
}

/// Read the reasoning budget from `VANTA_REASONING_BUDGET` (§14.21).
///
/// [VANTA-WHY] §14.21 — one variable, not one per site: the budget is a property
/// of the arm ("reason for at most N tokens per thinking block"), and the set of
/// sites it reaches is the provider-neutral `REASONING_BUDGET_SITES` allow-list,
/// not a spelling the operator can get wrong. Zero is rejected for §14.20's
/// reason squared: a budget of zero is not "off" (unset is off), it is a model
/// told to reason for no tokens at all, which measures nothing and silently
/// changes what the arm means. A malformed value is an error, never "no switch".
fn env_reasoning_budget() -> Result<Option<u32>, String> {
    const NAME: &str = "VANTA_REASONING_BUDGET";
    match std::env::var(NAME) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            let tokens = trimmed.parse::<u32>().map_err(|_| {
                format!("CONFIG_ERROR: {NAME}='{trimmed}' is not a number of tokens.")
            })?;
            if tokens == 0 {
                return Err(format!(
                    "CONFIG_ERROR: {NAME}=0 is not a budget; unset it to run without one."
                ));
            }
            Ok(Some(tokens))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "CONFIG_ERROR: {NAME} env var contained non-UTF-8 bytes."
        )),
    }
}

/// Refuse `VANTA_REASONING_BUDGET` on a judge arm (C5).
///
/// [VANTA-WHY] C5 — the budget reaches only the Attacker's three sites
/// (`REASONING_BUDGET_SITES`), and a judge arm makes exactly ONE call — to the
/// `judge` site, which is deliberately outside that list. A budget set on a judge
/// arm would therefore reach nothing and the row would silently describe an arm
/// other than the one the operator asked for. The switch is a property of the arm
/// (§14.21), so an arm it cannot act on is a misconfiguration, not a no-op: the
/// run is refused by name before the repo is loaded and before any request is
/// built. A non-judge arm — including one whose Attacker is off — is returned
/// `Ok`: there the budget reaches the Attacker's sites whenever the Attacker runs,
/// and this guard is about the arm that HAS no such site, not about every arm that
/// happens not to call it.
fn refuse_reasoning_budget_on_judge_arm(
    reasoning_budget: Option<u32>,
    arm: &ResolvedArm,
) -> Result<(), String> {
    if reasoning_budget.is_some() && arm.judge.is_some() {
        return Err(format!(
            "CONFIG_ERROR: VANTA_REASONING_BUDGET is set but arm '{}' is a judge arm: \
             the budget reaches only the Attacker sites ({}), and a judge arm makes no \
             Attacker call. Unset it.",
            arm.name,
            crate::llm::REASONING_BUDGET_SITES.join(", ")
        ));
    }
    Ok(())
}

/// Read the reasoning effort from `VANTA_REASONING_EFFORT` (§14.21).
///
/// [VANTA-WHY] §14.21 — the levels and the rejection of anything else live with
/// the enum (`ReasoningEffort::parse`), so the string this accepts and the string
/// a body carries are one definition. An unparsable level is an error for the
/// same reason a bad budget is: an ablation arm must run at the level it named.
fn env_reasoning_effort() -> Result<Option<crate::llm::ReasoningEffort>, String> {
    const NAME: &str = "VANTA_REASONING_EFFORT";
    match std::env::var(NAME) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            crate::llm::ReasoningEffort::parse(trimmed).map(Some)
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!(
            "CONFIG_ERROR: {NAME} env var contained non-UTF-8 bytes."
        )),
    }
}

/// The `baseline` / `vanta` / `vanta_planner` / `b2` name for a set of switches.
///
/// [VANTA-WHY] Deriving the name from the switches instead of taking a fourth
/// env var means the config a session records CANNOT disagree with the config
/// it ran: §4.2 defines the three configurations as exactly these switch
/// settings, so this is the single definition of the mapping.
///
/// [VANTA-WHY] M3 item 8 — `b2` is the paper's second adversarial arm: the
/// Attacker runs but BATE is bypassed (`VANTA_BATE=off`), so every generated
/// test is admitted regardless of declared intent or baseline behaviour. It is
/// derived here, not from a fourth env var, for the same reason as the rest:
/// the recorded arm name is then a function of the switches that actually ran.
/// The check is nested inside the Attacker-on branch and AFTER the planner, so
/// `baseline` (no Attacker — no adversarial intake to withhold) and
/// `vanta_planner` are untouched by the new switch and stay byte-identical.
fn benchmark_config_name(
    attacker_enabled: bool,
    planner_enabled: bool,
    bate_enabled: bool,
) -> &'static str {
    if planner_enabled {
        "vanta_planner"
    } else if attacker_enabled {
        if bate_enabled {
            "vanta"
        } else {
            "b2"
        }
    } else {
        "baseline"
    }
}

/// One arm's definition — the Rust mirror of `scripts/bugsinpy/arms.py::ARMS`.
///
/// [VANTA-WHY] M3 item 8 follow-up — the harness STATES the arm rather than letting
/// the switches imply it, because two arms can share a switch pair: `b0` and
/// `baseline` are both "attacker off, planner off, BATE on" and differ ONLY in the
/// attempt budget, which `benchmark_config_name` cannot see. The stated name is what
/// `run.summary` records as `arm`, and this table is what checks the statement.
///
/// It is duplicated here rather than only in Python because the binary is also run by
/// hand (no harness, no `VANTA_ARM`), and because the refusal has to happen in the
/// process that is about to spend the budget rather than only in its caller. The
/// Python table is the source of truth; `scripts/test_bugsinpy.py::TestArms` parses
/// this array and fails if the two disagree, so a hand-edit on one side cannot
/// silently make the arm a run recorded different from the arm it executed.
struct ArmSpec {
    name: &'static str,
    /// The `config` column value the arm's rows carry. The legacy arms keep the name
    /// every row already on disk carries, so their sessions read unchanged and their
    /// job keys stay byte-identical (the key is built from `config`, never `arm`).
    config: &'static str,
    attacker: bool,
    planner: bool,
    bate: bool,
    /// The attempt budget this arm pins, or `None` for the invocation's own.
    max_attempts: Option<u32>,
    /// [VANTA-WHY] M3 item 7 — the judge mode this arm runs under, or `None` for a
    /// non-judge arm. `Some("plain")`/`Some("suite")` is the `VANTA_JUDGE` value the
    /// arm IS, so `resolve_arm` can refuse a stated name whose judge setting
    /// disagrees — exactly as it refuses a contradicting `--max-attempts`. The
    /// setting is a switch, not a mode, so it lives here rather than in `arms.py`'s
    /// `modes` (which is the harness's repair/review split).
    judge: Option<&'static str>,
    /// [VANTA-WHY] M3 item 9/A5 — the K a fixed-round arm runs the repair loop for
    /// (`--fixed-rounds K`), or `None` for every ordinary arm. The value mirrors the
    /// Python registry's `Arm.fixed_rounds`, which itself mirrors
    /// `configs/v1-m3.toml`'s `[fixed_rounds]` table (the declaration of record); a
    /// Python test binds the file, and the table-mirror test binds this array to the
    /// registry. `resolve_arm` refuses a stated K arm whose `--fixed-rounds` does not
    /// state its K — a fixed-round arm IS K attempts, exactly as B0 IS one.
    fixed_rounds: Option<u32>,
}

const ARMS: &[ArmSpec] = &[
    // M3 item 11 — B0: one Creator call, no feedback, no Attacker.
    ArmSpec {
        name: "b0",
        config: "b0",
        attacker: false,
        planner: false,
        bate: true,
        max_attempts: Some(1),
        judge: None,
        fixed_rounds: None,
    },
    // B1 — today's `baseline` repair.
    ArmSpec {
        name: "b1",
        config: "baseline",
        attacker: false,
        planner: false,
        bate: true,
        max_attempts: None,
        judge: None,
        fixed_rounds: None,
    },
    // M3 item 8 — B2: the Attacker runs, BATE does not admit-test.
    ArmSpec {
        name: "b2",
        config: "b2",
        attacker: true,
        planner: false,
        bate: false,
        max_attempts: None,
        judge: None,
        fixed_rounds: None,
    },
    // B3 — today's `vanta`, both modes.
    ArmSpec {
        name: "b3",
        config: "vanta",
        attacker: true,
        planner: false,
        bate: true,
        max_attempts: None,
        judge: None,
        fixed_rounds: None,
    },
    // The review half of today's `baseline`: attacker off, reviewing a candidate.
    ArmSpec {
        name: "suite",
        config: "baseline",
        attacker: false,
        planner: false,
        bate: true,
        max_attempts: None,
        judge: None,
        fixed_rounds: None,
    },
    // M3 item 7 — the judge: ONE model call decides the candidate. Review-only and
    // attacker-off (nothing is generated), and it pins `--max-attempts 1` for the
    // same reason B0 does: "one call per verdict" is what the arm IS.
    ArmSpec {
        name: "judge",
        config: "judge",
        attacker: false,
        planner: false,
        bate: true,
        max_attempts: Some(1),
        judge: Some("plain"),
        fixed_rounds: None,
    },
    // M3 item 7 — the judge WITH the native suite's result on the candidate folded
    // into its prompt. Same one call; the only difference is what the call is shown.
    ArmSpec {
        name: "judge+suite",
        config: "judge+suite",
        attacker: false,
        planner: false,
        bate: true,
        max_attempts: Some(1),
        judge: Some("suite"),
        fixed_rounds: None,
    },
    // M3 item 9/A5 — the fixed-round arms: B1/B3 with the repair loop forced to run
    // exactly K attempts (`--fixed-rounds K`), taking the verdict from attempt K so
    // a patch that passes early is re-examined rather than accepted (EIR). The K
    // below mirrors the Python registry's `Arm.fixed_rounds`, which mirrors
    // `configs/v1-m3.toml`'s `[fixed_rounds]` table.
    ArmSpec {
        name: "b1-k4",
        config: "b1-k4",
        attacker: false,
        planner: false,
        bate: true,
        max_attempts: None,
        judge: None,
        fixed_rounds: Some(4),
    },
    ArmSpec {
        name: "b3-k4",
        config: "b3-k4",
        attacker: true,
        planner: false,
        bate: true,
        max_attempts: None,
        judge: None,
        fixed_rounds: Some(4),
    },
];

/// The arm a run is executing: the name it records and the `config` it belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedArm {
    /// What `run.summary` records as `arm`, and what the harness passed as `VANTA_ARM`.
    name: String,
    /// What `run.summary` records as `config` — the split's own column.
    config: String,
    /// [VANTA-WHY] M3 item 7 — `Some("plain"|"suite")` when the arm is a judge arm.
    /// Drives the judge branch and the arm-gating of the `judge` sampling keys, both
    /// of which read the RESOLVED arm rather than the raw environment, so the record
    /// cannot describe a different judge setting from the one that ran.
    judge: Option<&'static str>,
}

impl ResolvedArm {
    /// The pre-`VANTA_ARM` answer: name and config both derived from the switches.
    fn from_switches(attacker: bool, planner: bool, bate: bool) -> Self {
        let name = benchmark_config_name(attacker, planner, bate).to_string();
        Self {
            config: name.clone(),
            name,
            judge: None,
        }
    }
}

/// `on`/`off` for a switch, so a refusal sentence reads like the vars it names.
fn on_off(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}

/// Read `VANTA_JUDGE` (`off|plain|suite`), or `None` when the judge is off.
///
/// [VANTA-WHY] M3 item 7 — a THREE-valued switch, so it is read here rather than
/// through `env_flag` (which knows only on/off). `off` and an absent/empty var are
/// both "no judge", matching `env_arm`'s reading of an empty value as "no
/// statement" so a shell that expands an unset variable does not fail the run.
fn env_judge() -> Result<Option<&'static str>, String> {
    match std::env::var("VANTA_JUDGE") {
        Ok(raw) => match raw.trim() {
            "" | "off" => Ok(None),
            "plain" => Ok(Some("plain")),
            "suite" => Ok(Some("suite")),
            other => Err(format!(
                "CONFIG_ERROR: VANTA_JUDGE='{other}' is not a judge mode (expected off|plain|suite)."
            )),
        },
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(
            "CONFIG_ERROR: VANTA_JUDGE env var contained non-UTF-8 bytes.".to_string(),
        ),
    }
}

/// Read `VANTA_ARM`, or `None` when it is absent or empty.
///
/// [VANTA-WHY] An empty value is "no statement", not a name: a shell that expands an
/// unset variable (`VANTA_ARM=`) hands over an empty string, and treating it as a
/// stated-but-unknown arm would fail a run the operator configured correctly. This is
/// the same reading `env_float` gives an empty value.
fn env_arm() -> Result<Option<String>, String> {
    match std::env::var("VANTA_ARM") {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(
            "CONFIG_ERROR: VANTA_ARM env var contained non-UTF-8 bytes.".to_string(),
        ),
    }
}

/// Resolve the run's arm, refusing a stated name the switches or budget contradict.
///
/// [VANTA-WHY] M3 item 8 follow-up — the driver's half of "the harness refuses before
/// launching". The harness builds every switch, the arm name and the budget from ONE
/// `arms.py` entry, so it cannot produce an inconsistent combination; this check is
/// what catches a hand-run binary, a wrapper that exported the switches itself, or a
/// `--max-attempts` that would silently redefine an arm (`b0` is one Creator call, not
/// one attempt of four). A refusal here happens before the repo is loaded, so a
/// contradiction costs nothing.
fn resolve_arm(
    stated: Option<&str>,
    attacker: bool,
    planner: bool,
    bate: bool,
    max_attempts: u32,
    judge: Option<&str>,
    fixed_rounds: Option<u32>,
) -> Result<ResolvedArm, String> {
    let Some(name) = stated else {
        // [VANTA-WHY] M3 item 7 — a judge run MUST state its arm. The
        // switch-derived name has no judge in it (`judge` off is `baseline`), so
        // inferring one would record a judge verdict under the arm it is being
        // compared against — the one thing the experiment must not do.
        if let Some(mode) = judge {
            return Err(format!(
                "CONFIG_ERROR: VANTA_JUDGE='{mode}' needs VANTA_ARM to name the arm \
                 (judge or judge+suite); a judge verdict must not be recorded under a \
                 switch-derived name."
            ));
        }
        return Ok(ResolvedArm::from_switches(attacker, planner, bate));
    };
    let Some(spec) = ARMS.iter().find(|spec| spec.name == name) else {
        let known: Vec<&str> = ARMS.iter().map(|spec| spec.name).collect();
        return Err(format!(
            "CONFIG_ERROR: VANTA_ARM='{name}' is not a known arm (expected one of {}).",
            known.join(", ")
        ));
    };
    if spec.attacker != attacker || spec.planner != planner || spec.bate != bate {
        return Err(format!(
            "CONFIG_ERROR: VANTA_ARM='{name}' disagrees with the switches: the arm is \
             attacker={} planner={} bate={}, but VANTA_ATTACKER={} VANTA_PLANNER={} \
             VANTA_BATE={}.",
            on_off(spec.attacker),
            on_off(spec.planner),
            on_off(spec.bate),
            on_off(attacker),
            on_off(planner),
            on_off(bate),
        ));
    }
    if let Some(pinned) = spec.max_attempts {
        if max_attempts != pinned {
            return Err(format!(
                "CONFIG_ERROR: VANTA_ARM='{name}' pins --max-attempts {pinned} (that is \
                 what the arm IS), but the invocation asked for {max_attempts}."
            ));
        }
    }
    // [VANTA-WHY] M3 item E9 — a fixed-round count is a claim about how many attempts
    // the run makes, so it must not exceed an arm that PINS its budget. `b0` (and the
    // judge arms) ARE one attempt, and `--fixed-rounds 4` under one of those names would
    // run four: the check above only guards `--max-attempts`, and `--fixed-rounds` never
    // touches it, so before E9 this slipped through. A fixed-round arm is checked by its
    // own `fixed_rounds` pin below; an ordinary arm with no pinned budget keeps item 9's
    // legitimate use. Only GREATER is refused — a count equal to the pinned budget
    // contradicts nothing.
    if let (Some(pinned), Some(rounds)) = (spec.max_attempts, fixed_rounds) {
        if rounds > pinned {
            return Err(format!(
                "CONFIG_ERROR: VANTA_ARM='{name}' pins --max-attempts {pinned} (that is \
                 what the arm IS), but --fixed-rounds {rounds} asks for {rounds} rounds."
            ));
        }
    }
    // [VANTA-WHY] M3 item 7 — the judge switch and the stated arm must agree, in
    // BOTH directions: a judge arm with `VANTA_JUDGE=off` would run the whole
    // adversarial pipeline under a name that claims a one-call verdict, and a
    // non-judge arm with `VANTA_JUDGE=plain` would record judge semantics for a
    // run that never called the judge. `judge` arrives already normalised by
    // `env_judge` (`None` for off/absent), so the comparison needs no second
    // reading of the variable — one reader, one meaning.
    if spec.judge != judge {
        return Err(format!(
            "CONFIG_ERROR: VANTA_ARM='{name}' disagrees with VANTA_JUDGE: the arm is \
             judge={}, but VANTA_JUDGE={}.",
            spec.judge.unwrap_or("off"),
            judge.unwrap_or("off")
        ));
    }
    // [VANTA-WHY] M3 item 9/A5 — a fixed-round arm IS K attempts, so the invocation
    // must state the same K, exactly as B0 must be handed the `--max-attempts 1` it
    // pins. The harness always states it (`runner.invoke_vanta` builds the flag from
    // the arm), so only a hand-run binary or a wrapper that named the arm but not its
    // K reaches this refusal. The check is one-directional on purpose: `--fixed-rounds`
    // on an ORDINARY arm is a legitimate item-9 run (the option is available to any
    // repair run), and only the arm whose NAME claims K is contradicted by a mismatch.
    if let Some(pinned) = spec.fixed_rounds {
        if fixed_rounds != Some(pinned) {
            return Err(format!(
                "CONFIG_ERROR: VANTA_ARM='{name}' IS a {pinned}-attempt run \
                 (--fixed-rounds {pinned}), but the invocation {}.",
                match fixed_rounds {
                    Some(k) => format!("asked for --fixed-rounds {k}"),
                    None => "did not state --fixed-rounds".to_string(),
                }
            ));
        }
    }
    Ok(ResolvedArm {
        name: spec.name.to_string(),
        config: spec.config.to_string(),
        judge: spec.judge,
    })
}

/// The in-session native wall-clock cap, from `--native-timeout-secs` or the default.
///
/// [VANTA-WHY] M3 item E13 — the flag RAISES the cap for a slow bug; it must never
/// LOWER it. [`harness::native_session_timeout_floor`] (1200 s, or the environment's own
/// `VANTA_TEST_TIMEOUT_SECS` when that is larger) is the cap this run would otherwise
/// use, and the Python side already passes its own session rule, so a value below that
/// floor asks every suite to time out sooner than the default — the gate could no longer
/// measure the very bug the cap was raised for. Refused here, where the cap is resolved,
/// naming the flag and the floor. `None` keeps the floor; the CLI still refuses `0` at
/// parse time.
///
/// [VANTA-WHY] v1-m3.1 — the floor moved from `test_timeout_secs()` (120 s) to the
/// session floor, because the in-session gate now gives BOTH sides one raised cap: a
/// patched side can legitimately cost several times the suite prepare measured, so a cap
/// that could sit at 120 would kill it and report the patch as an infrastructure fault.
pub fn resolve_native_timeout_secs(requested: Option<u64>) -> Result<u64, String> {
    let floor = harness::native_session_timeout_floor();
    match requested {
        None => Ok(floor),
        Some(secs) if secs < floor => Err(format!(
            "CONFIG_ERROR: --native-timeout-secs {secs} is below the effective cap of \
             {floor} s — the flag may only RAISE the cap, never lower it."
        )),
        Some(secs) => Ok(secs),
    }
}

/// The native gate's per-attempt cap, clamped to what the run's clock has left
/// (v1-m3.1 §8.4).
///
/// [VANTA-WHY] The gate runs AFTER the deadline check at the top of the attempt, so a
/// row with a slow suite (a 725 s suite: 7250 s per side) can start the gate with less than
/// 7250 s left and outlive `VANTA_DEADLINE_SECS` on its own. The harness then
/// hard-kills the whole row at `RUN_TIMEOUT_SECS` before the next attempt top can
/// honour the deadline, and a hard-killed row leaves no manifest entry — it is
/// invisible to the wave-1 gate and re-runs identically on resume. BOTH sides of the
/// `tokio::join!` therefore get `min(native_cap, remaining)`, so the gate can never
/// outlive the deadline by more than one side's own duration. The clamp can only LOWER
/// the cap, never raise it.
///
/// `remaining` is the value from [`crate::llm::remaining_secs`]: `None` when no deadline
/// is armed (the caller keeps the full cap), `Some(0)` when nothing is left (the caller
/// SKIPS the gate and takes the existing deadline exit), `Some(n)` otherwise.
fn native_gate_cap(native_cap: u64, remaining: Option<u64>) -> Option<u64> {
    match remaining {
        None => Some(native_cap),
        Some(0) => None,
        Some(left) => Some(native_cap.min(left)),
    }
}

/// Which exit a VACUOUS native gate takes (v1-m3.1 §8.5).
///
/// [VANTA-WHY] A gate is vacuous when the ORIGINAL side collected nothing (§8.1 row 1),
/// which is normally an infrastructure fault (`InfraFailure`: the base tree could not
/// be graded). But when the gate was CLAMPED to the deadline (`gate_clamped`) AND the
/// original side was KILLED by that clamp (`original_timed_out`), its emptiness is the
/// run's CLOCK and not the machine: the estimate of what the suite needs was never
/// falsified, the run simply ran out of time. That case must take the SAME deadline
/// exit the top of the attempt loop takes — `NeedsReview` with
/// `stop_reason=budget_exhausted` — never `InfraFailure`, so a `budget_exhausted` row
/// is not misreported as a machine fault. A clamp that did NOT kill the side (a real
/// collection error inside the shortened window) and a run under the full cap both
/// stay `InfraFailure`, exactly as before.
fn native_vacuous_exit(gate_clamped: bool, original_timed_out: bool) -> GovernorDecision {
    if gate_clamped && original_timed_out {
        GovernorDecision::NeedsReview
    } else {
        GovernorDecision::InfraFailure
    }
}

/// Does a VACUOUS native gate set the run-level INFRASTRUCTURE-vacuity flag
/// (`run.summary`'s `native_vacuous`)?
///
/// [VANTA-WHY] v1-m3.1 F3 — that flag documents one thing: the base tree could not
/// be graded, so the native check could not be performed (`InfraFailure`). A gate
/// made vacuous by the DEADLINE CLAMP takes the deadline exit
/// (`budget_exhausted`/`NeedsReview`) instead — a fact about the run's clock, not
/// the machine — so it must NOT set the flag; it records `native_vacuous=false`
/// and `native_gate_clamped=true`. This is the exact complement of
/// `native_vacuous_exit`: the flag is set iff that function picks `InfraFailure`.
fn native_vacuous_is_infrastructure(gate_clamped: bool, original_timed_out: bool) -> bool {
    native_vacuous_exit(gate_clamped, original_timed_out) != GovernorDecision::NeedsReview
}

/// Does a VACUOUS native gate set the run-level TIMEOUT flag (`run.summary`'s
/// `native_timeout`)?
///
/// [VANTA-WHY] Finding C (PR #4 review) — the flag names one thing: a side was
/// KILLED by the CONFIGURED cap (`native_cap`), so the vacuity's cause is the
/// cap and not a tree that collected nothing. A kill by the DEADLINE CLAMP
/// lowered the cap below the configured one, so it is the run's CLOCK killing
/// the side, not the cap the run was configured with; that case must NOT set
/// this flag (it is recorded by `native_gate_clamped` instead). It is still set
/// only for a VACUOUS gate, so `native_timeout=true` continues to imply
/// `native_vacuous=true` — the contract every consumer of the field relies on.
fn native_timeout_flag(
    native_vacuous: bool,
    gate_clamped: bool,
    original_timed_out: bool,
    patched_timed_out: bool,
) -> bool {
    native_vacuous && !gate_clamped && (original_timed_out || patched_timed_out)
}

/// Did the native gate end as an INFRASTRUCTURE fault because a side could not be
/// SET UP or transported (Finding A, PR #4 review)?
///
/// [VANTA-WHY] An empty side is patch-attributable ONLY when the suite command ran
/// and exited (`exit_code` is `Some` — a collection or import error, pytest exit
/// 2/4/5) or hit the cap (`timed_out`); `NativeSuiteReport::setup_failed` is false
/// for exactly those two. When it is set the harness never obtained a measurement,
/// so `native_regressions` over it would list every ORIGINAL test as a regression
/// against a suite Vanta never ran — a machine fault blamed on the patch.
///
/// An ORIGINAL side that could not be set up is this fault UNCONDITIONALLY: its
/// emptiness is not the deadline clamp's doing, so it must not take the clamp's
/// `budget_exhausted` exit. A PATCHED side's is this fault whenever the ORIGINAL
/// side is not itself vacuous — an original vacuity already ends the run as
/// `InfraFailure` through its own branch, with its own (more specific) reason.
fn native_setup_failure(
    original_setup_failed: bool,
    patched_setup_failed: bool,
    original_vacuous: bool,
) -> bool {
    original_setup_failed || (patched_setup_failed && !original_vacuous)
}

/// The run-level per-side NATIVE-gate facts that ride `run.summary` and
/// `evidence.json` (v1-m3.1 §4, F2/F3 of the PR #4 review).
///
/// [VANTA-WHY] F3 (PR #4 review) — `original_secs`/`patched_secs`, the
/// `*_timed_out_last` pair and `effective_cap_secs` all describe the SAME (last)
/// gate the run reached, so a reader comparing them never pairs one attempt's
/// wall time or cap with another attempt's kill. The `secs` are exactly the LAST
/// gate's: a side's measured seconds when it completed, or `None` when it did not
/// (timed out, failed to set up, or never ran) — never an earlier attempt's value.
/// `*_timed_out_any` keeps the WIDER meaning the
/// `native_original_timed_out`/`native_patched_timed_out` fields always had — a
/// kill on ANY attempt — so a kill is not lost when a later attempt happens to
/// finish.
#[derive(Default)]
struct NativeGateFacts {
    original_secs: Option<f64>,
    patched_secs: Option<f64>,
    original_timed_out_last: bool,
    patched_timed_out_last: bool,
    original_timed_out_any: bool,
    patched_timed_out_any: bool,
    effective_cap_secs: Option<u64>,
}

impl NativeGateFacts {
    /// Fold ONE native gate's per-side results in.
    ///
    /// [VANTA-WHY] F2 (PR #4 review) — called immediately after the two sides
    /// `tokio::join!` and BEFORE the `native_setup_failure` guard. That ordering is
    /// the point: a gate whose ORIGINAL side completed while the PATCHED side
    /// failed to set up still obtained an original-side measurement, and recording
    /// it here keeps that measurement on the row even though the guard ends the run
    /// as `InfraFailure`.
    ///
    /// Each call REPLACES both sides' seconds with THIS gate's own value, so the
    /// last-gate fields always describe one gate: a side that completed contributes
    /// its measured seconds, a side that did not (timed out, failed to set up, never
    /// ran) contributes `None`. A previous attempt's measurement is never carried
    /// forward, because a reader must never pair the last gate's `*_timed_out_last`
    /// or `effective_cap_secs` with an earlier attempt's wall time.
    fn record(
        &mut self,
        cap: u64,
        original: &harness::NativeSuiteReport,
        patched: &harness::NativeSuiteReport,
    ) {
        self.original_secs = original.secs;
        self.patched_secs = patched.secs;
        self.original_timed_out_last = original.timed_out;
        self.patched_timed_out_last = patched.timed_out;
        self.original_timed_out_any |= original.timed_out;
        self.patched_timed_out_any |= patched.timed_out;
        self.effective_cap_secs = Some(cap);
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

/// Render a USD amount for a record line (§14.29).
///
/// [VANTA-WHY] Six decimal places, always, and `unset` when there is no number.
/// Six is what makes a single cheap Luna call (`$0.000012`) legible without a
/// scientific-notation exponent, which `{}` on an `f64` would print and which
/// `scripts/metrics.py::summary_field` would read as a value containing `e`.
/// Fixed-point also means two runs' rows are comparable as text.
///
/// [VANTA-WHY] `unset` rather than `0.000000`: "this run cost nothing" and "no
/// price applied to this run" are different claims — the local backend is the
/// first, an unpriced model is the second — and a cost column that renders both
/// as zero cannot be summed honestly.
fn cost_value(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.6}")).unwrap_or_else(|| "unset".to_string())
}

/// Render the run's cache hit rate (§14.32).
///
/// [VANTA-WHY] The rate is `cached / prompt` over the calls the server MEASURED,
/// computed from the same two totals `run.summary` already carries
/// (`llm_cached_tokens`, `llm_prompt_tokens`) so the three numbers on one line
/// cannot disagree — a reader dividing them by hand gets this value back. It is
/// token-WEIGHTED, not the mean of the per-call rates: a run whose first call is
/// a 10k-token prompt and whose last is a 100-token one has a cache hit rate
/// dominated by the call that actually spent the tokens.
///
/// [VANTA-WHY] `unset` in three cases, and they are all "no rate exists": no
/// call was measured, no measured call reported cached tokens (the local server
/// said nothing about caching), or the measured prompt total is zero (a
/// division that would print `inf`/`NaN` into a `key=value` row). `0.0000` is
/// the different claim that caching was measured and nothing was served from it.
///
/// [VANTA-WHY] A call that reported a prompt count but no cache detail
/// contributes its prompt to the denominator, so the rate is a LOWER bound in
/// that mixed case rather than an overstatement. That is the safe direction for
/// a number a reader uses to decide whether prefix caching is working.
fn cache_hit_rate_value(cached: Option<u64>, prompt: u64) -> String {
    match cached {
        Some(cached) if prompt > 0 => format!("{:.4}", cached as f64 / prompt as f64),
        _ => "unset".to_string(),
    }
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

/// The run's spend, its two nested token details, and the bases that priced it
/// (§14.29).
///
/// Returns `(cost_usd, cached_tokens, reasoning_tokens, cost_basis)`:
///
/// * `cost_usd` — the sum over the calls that WERE priced. `None` when none was:
///   the local backend is `not_billed` and a hosted model with no price row is
///   `unpriced_family`, and both are holes a batch table has to be able to see
///   rather than a `0.0` it would silently average.
/// * `cached_tokens` / `reasoning_tokens` — summed over the calls that REPORTED
///   them, and `None` when no call in the run did. Same rule and same reason as
///   `token_totals`'s completion sum: a `0` would claim the server said zero,
///   and on a local run the server says nothing at all.
/// * `cost_basis` — the DISTINCT bases the run's calls recorded, in first-seen
///   order, joined by `,`; `none` when the run made no calls. A run whose rows
///   disagree (one priced, one fallen back to the estimate) names both, because
///   "which prices produced this total?" is exactly the question the versioned
///   table exists to answer and a single basis would answer it wrongly.
fn cost_and_detail_totals(
    calls: &[crate::llm::CallUsage],
) -> (Option<f64>, Option<u64>, Option<u64>, String) {
    let measured: Vec<&crate::llm::CallUsage> =
        calls.iter().filter(|c| c.source != "estimate").collect();
    let costs: Vec<f64> = calls.iter().filter_map(|c| c.cost_usd).collect();
    let cost = if costs.is_empty() {
        None
    } else {
        Some(costs.iter().sum())
    };
    let sum_of = |pick: fn(&crate::llm::CallUsage) -> Option<u64>| -> Option<u64> {
        let values: Vec<u64> = measured.iter().filter_map(|c| pick(c)).collect();
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum())
        }
    };
    let mut bases: Vec<&str> = Vec::new();
    for basis in calls.iter().map(|c| c.cost_basis.as_str()) {
        if !bases.contains(&basis) {
            bases.push(basis);
        }
    }
    let basis_text = if bases.is_empty() {
        "none".to_string()
    } else {
        bases.join(",")
    };
    (
        cost,
        sum_of(|c| c.cached_tokens),
        sum_of(|c| c.reasoning_tokens),
        basis_text,
    )
}

/// The run's served model version, collapsed from the per-call `served_model`
/// facts (M3 item B).
///
/// * every call that reported one agrees → that snapshot;
/// * the calls disagree → the sorted distinct snapshots as the `[a;b]` list
///   `run.summary` writes for a set;
/// * no call named one (including a run that made no call) → `unknown`.
///
/// [VANTA-WHY] M3 item B — `unknown` rather than the requested id: the field's
/// job is to record what the SERVER said, and a run that never got an answer, or
/// a local server that names nothing, did not state one. A run whose calls
/// disagree is NOT collapsed to the first: that is exactly the
/// alias-rolled-mid-run case the per-call rows exist to surface, and picking one
/// would hide it. Values are encoded space-free so the `key=value` reader cannot
/// be truncated by a model id that carries a space.
fn served_model_version_value(calls: &[crate::llm::CallUsage]) -> String {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for model in calls.iter().filter_map(|c| c.served_model.as_deref()) {
        if !model.is_empty() {
            seen.insert(model);
        }
    }
    match seen.len() {
        0 => "unknown".to_string(),
        1 => encode_summary_value(seen.iter().next().expect("len() == 1")),
        _ => {
            let owned: Vec<String> = seen.iter().map(|s| (*s).to_string()).collect();
            encode_summary_path_list(owned.iter())
        }
    }
}

/// Count the calls the server cut off at `max_tokens`, per role (§14.5).
///
/// [VANTA-WHY] A truncated call is one whose generation hit the output cap —
/// the shape a model "runs away" in. The role comes from the call SITE via
/// `llm::truncation_role`, because the Attacker has four call sites and a
/// run-away on any of them is the same fact about the same agent.
/// `(creator, attacker, intent_audit)`.
fn truncation_counts(calls: &[crate::llm::CallUsage]) -> (u64, u64, u64) {
    let mut counts = (0_u64, 0_u64, 0_u64);
    for site in calls.iter().filter(|c| c.truncated).map(|c| c.site.as_str()) {
        match crate::llm::truncation_role(site) {
            "creator" => counts.0 += 1,
            "attacker" => counts.1 += 1,
            "intent_audit" => counts.2 += 1,
            // [VANTA-WHY] A site outside the schedule (the M0 spike names its
            // own) has no role to report under. Said out loud rather than
            // silently dropped, because a run-away that vanishes from every
            // count would make the split's rate a lower bound nobody labelled.
            other => tracing::warn!(
                site,
                role = other,
                "a truncated call's site maps to no reported role, so it is not counted"
            ),
        }
    }
    counts
}

/// Per-call-SITE totals: calls, truncated calls and completion tokens (§14.12).
///
/// [VANTA-WHY] Roles (`llm::truncation_role`) answer "which AGENT ran away";
/// sites answer "which of that agent's calls did". The Attacker alone has three
/// sites, and a round whose revisions all run away is a different problem from
/// one whose first generation did — a distinction the role split collapses by
/// construction. Sites are reported in `SAMPLING_SCHEDULE` order and include the
/// ones this run never reached, as a zero rather than a missing key: a breakdown
/// that silently omits a site makes "it never ran" indistinguishable from "the
/// record forgot it". A site outside the schedule (the M0 spike names its own)
/// is reported too, for the same reason `truncation_counts` warns about one.
///
/// The completion sum excludes rows whose `source` is `estimate`, exactly as
/// `token_totals` does: a per-site number that mixes a character-derived guess
/// into a measurement is worse than an absent one.
///
/// [VANTA-WHY] M3 item 7 — `judge_arm` gates ONE site: the `judge` site's
/// zero-row placeholder is emitted only on a judge ARM's row. Every pre-existing
/// site is still emitted on every row (b0/b1 included), so no non-judge
/// `run.summary` changes. A judge call actually MADE by some other arm would
/// still be reported by the loop below — the gate suppresses the placeholder for
/// a site the arm never reaches, never a call that really happened.
fn per_site_totals(
    calls: &[crate::llm::CallUsage],
    judge_arm: bool,
) -> Vec<(String, u64, u64, Option<u64>)> {
    let mut sites: Vec<&str> = crate::llm::SAMPLING_SCHEDULE
        .iter()
        .map(|(site, _, _)| *site)
        .filter(|site| judge_arm || !is_judge_site(site))
        .collect();
    for call in calls {
        if !sites.contains(&call.site.as_str()) {
            sites.push(call.site.as_str());
        }
    }
    sites
        .into_iter()
        .map(|site| {
            let rows: Vec<&crate::llm::CallUsage> =
                calls.iter().filter(|c| c.site == site).collect();
            let truncated = rows.iter().filter(|c| c.truncated).count() as u64;
            let completions: Vec<u64> = rows
                .iter()
                .filter(|c| c.source != "estimate")
                .filter_map(|c| c.completion_tokens)
                .collect();
            let completion = if completions.is_empty() {
                None
            } else {
                Some(completions.iter().sum())
            };
            (site.to_string(), rows.len() as u64, truncated, completion)
        })
        .collect()
}

/// Render the run's EFFECTIVE per-call-site sampling as `run.summary` keys.
///
/// [VANTA-WHY] Underscores, not dots. `rerun.rs::summary_field` and
/// `scripts/metrics.py::summary_field` both accept a key only if it matches
/// `[a-z_][a-z0-9_]*`, so `temperature.creator=0.1` would not terminate at the
/// next key — the dot is not a key character, the value would run on through the
/// rest of the line, and BOTH consumers would silently read the wrong thing.
/// `{v:?}` for the float so `0.0` stays `0.0` and the record matches the
/// request body byte for byte.
///
/// [VANTA-WHY] §14.11 — the record is the EFFECTIVE schedule, not the static
/// `pinned` table: under `VANTA_SAMPLING=coding` every site, the intent auditor
/// included, samples at 0.6 with `top_p`/`top_k`/`min_p`/`presence_penalty`
/// pinned (the auditor's 8192-token cap is not part of the preset), and a summary
/// that still said 0.1 would describe a run nobody made. The preset names itself
/// first, and the per-role keys come from `llm::sampling_pins_text`, the same
/// table the request bodies are built from, so the record cannot drift from the
/// sampling that actually happened.
///
/// [VANTA-WHY] M3 item 7 — the `judge` site's `temperature_judge` /
/// `max_tokens_judge` keys are emitted only on a judge ARM's row, because only a
/// judge arm has a judge call to pin. Every other site keeps being emitted on
/// every row (b0/b1 included), so a non-judge `run.summary` is byte-identical to
/// what it was before the judge existed — the same reason `per_site_totals`
/// takes the same flag.
fn sampling_schedule_keys(judge_arm: bool) -> String {
    let preset = crate::llm::sampling_preset();
    let mut keys = vec![format!("sampling_preset={}", preset.name())];
    for (site, _, _) in crate::llm::SAMPLING_SCHEDULE {
        if is_judge_site(site) && !judge_arm {
            continue;
        }
        for (key, value) in crate::llm::sampling_pins_text(preset, site) {
            keys.push(format!("{key}_{site}={value}"));
        }
    }
    keys.join(" ")
}

/// Is this call SITE the judge's own (M3 item 7)?
///
/// [VANTA-WHY] The judge is the only site an ARM adds or removes, so the two
/// places that describe a run's sampling and spend key their arm-gating on this
/// one predicate rather than on a string literal apiece — one spelling of "the
/// judge site", so the schedule entry, the pins and the row cannot disagree.
fn is_judge_site(site: &str) -> bool {
    site == "judge"
}

/// Render the JUDGE prompt's provenance as `run.summary` keys (M3 item 7).
///
/// [VANTA-WHY] M3 item 7 — the plan requires the judge's prompt text to be
/// versioned and hashed like every other prompt, and the decision is that those
/// keys appear ONLY on a judge arm's row. The judge prompt is not one of the
/// prompts every run sends, so it cannot go in `prompt_versions()` /
/// `prompt_hashes()`, which every row — `session.start` included — carries: that
/// would claim every run sent a judge prompt. This returns the EMPTY string for a
/// non-judge row and a LEADING-SPACE-prefixed string for a judge row, so it can be
/// spliced in right after `prompt_hash=` and a non-judge `run.summary` stays
/// byte-identical to what it was before the judge existed.
///
/// [VANTA-WHY] C4 — the hash is `judge::judge_prompt_hash()`, the same value
/// `configs/v1-m3.toml [prompts] judge` pins: it covers the judge's USER templates
/// as well as its system one, so rewording a user block now moves both the row's
/// recorded provenance and `config_hash`. Taking it from that one accessor is what
/// keeps the row and the frozen config from describing different judge prompts.
fn judge_prompt_keys(judge_arm: bool) -> String {
    if !judge_arm {
        return String::new();
    }
    let hash = crate::agents::judge::judge_prompt_hash();
    format!(
        " judge_prompt_version={} judge_prompt_hash={}",
        crate::agents::judge::JUDGE_PROMPT_VERSION,
        hash
    )
}

/// Render the repo's own suite's result on the candidate for the `judge+suite`
/// prompt (M3 item 7).
///
/// [VANTA-WHY] `judge+suite`'s ONLY difference from `judge` is that the model is
/// shown what the repo's tests said about the candidate, so what it is shown has
/// to be the facts the test-driven arm acts on: the pass/fail summary AND the
/// failing tests with their own output, not just a bit. A suite that did not run
/// says so — `ran=false` with the stderr tail — because "no failing tests" and
/// "nothing ran" are opposite readings of an empty report, and a judge shown the
/// second as the first would approve on evidence that does not exist.
fn render_judge_suite_result(report: &harness::NativeSuiteReport) -> String {
    if !report.ran {
        let tail = report.stderr_tail.trim();
        return format!(
            "ran=false exit_code={:?} stderr_tail={}",
            report.exit_code,
            if tail.is_empty() { "(empty)" } else { tail }
        );
    }
    let total = report.tests.len();
    let failures: Vec<&(String, TestResult)> = report
        .tests
        .iter()
        .filter(|(_, result)| *result != TestResult::Passed)
        .collect();
    let mut out = format!(
        "ran=true total={total} passed={} failed={} exit_code={:?}",
        total - failures.len(),
        failures.len(),
        report.exit_code
    );
    for (nodeid, result) in failures {
        out.push_str(&format!("\n- {nodeid} [{result:?}]"));
        if let Some(output) = report.failure_output(nodeid) {
            out.push_str("\n    ");
            out.push_str(&output.replace('\n', "\n    "));
        }
    }
    out
}

/// The request-shaping rule this run's backend follows, in the same one-line
/// `key=value` shape as [`sampling_schedule_keys`].
///
/// [VANTA-WHY] §14.26 — item 3 asks for what was REQUESTED and what was actually
/// SENT to be recorded in `call.usage` and `run.summary`. The per-call rows carry
/// the per-body detail (a body's own `requested_fields`/`sent_fields`); this is the
/// run-level RULE that explains them, so a reader who sees `temperature` absent from
/// a hosted body reads the reason off the run instead of guessing the pipeline lost
/// it. The local backend has no such rule — it sends the body it built — and this
/// line says so with `not_sent=none renamed=none` rather than staying silent, which
/// is what keeps the local run's summary self-describing without changing its bodies.
fn request_shape_rule(cfg: &LlmConfig) -> String {
    let backend = cfg.backend.as_str();
    if !cfg.is_openai() {
        return format!(
            "request_shape_backend={backend} request_shape_family=native \
             request_shape_not_sent=none request_shape_renamed=none"
        );
    }
    let family = crate::llm::capabilities::OpenAiFamily::from_model_id(&cfg.model_id);
    // `unsent_keys` is a const table walked in a fixed order, so this list is stable
    // across runs — the same reason the per-call `requested`/`sent` lists are.
    let mut unsent = crate::llm::capabilities::unsent_keys(family);
    // [VANTA-WHY] §14.28 — the effort is not a property of the family the way the
    // rows above are: it is dropped when THIS run asked for no level, so a default
    // hosted run must name it here and a run that set `VANTA_REASONING_EFFORT` must
    // not. The predicate is the shaper's own (`keeps_effort`), so the run-level
    // rule and the bodies it describes cannot disagree about what went out.
    if !crate::llm::capabilities::keeps_effort(family, crate::llm::requested_effort()) {
        unsent.push((
            crate::llm::capabilities::REASONING_EFFORT_KEY,
            "no level requested by this run; the call sites' `low` is a llama.cpp fact (§14.28)",
        ));
    }
    let not_sent: Vec<&str> = unsent.into_iter().map(|(key, _why)| key).collect();
    let not_sent = if not_sent.is_empty() {
        "none".to_string()
    } else {
        not_sent.join(",")
    };
    format!(
        "request_shape_backend={backend} request_shape_family={} \
         request_shape_not_sent={not_sent} request_shape_renamed={}->{}",
        family.name(),
        crate::llm::capabilities::LLAMACPP_OUTPUT_CAP_KEY,
        crate::llm::capabilities::OPENAI_OUTPUT_CAP_KEY,
    )
}

/// The sampling knobs the preset pins that this backend does NOT put in the
/// body, and the value the server applies in their place (Item S).
///
/// [VANTA-WHY] Item S — `sampling_schedule_keys` records the PRESET's own
/// `temperature_<site>` / `top_p_<site>` pins, and it records them for every
/// backend because the preset is defined independently of one. On the local
/// backend that is honest: the body carries 0.6/0.95 and the server samples at
/// them. On a hosted reasoning model it is not — §14.26's shaper drops both
/// before the body leaves, so the model samples at the provider's OWN default.
/// A row that shows `temperature_creator=0.6` without saying so reads as a run
/// that sampled at 0.6. This names the keys that did not reach the server and the
/// value it used instead, so the preset and the sent body are both on the row and
/// neither can be mistaken for the other. It is the sampling-specific twin of
/// `request_shape_not_sent`, which states the same drop as a bare field list;
/// this states the EFFECT (a server default), which a field list cannot.
fn sampling_not_sent_rule(cfg: &LlmConfig) -> String {
    // The two sampling knobs `SamplingPreset::Coding` pins on the generative
    // sites and no call site pins under the default preset (§14.11). Named here
    // in a fixed order so the list is stable across runs.
    const SAMPLED_KNOBS: [&str; 2] = ["temperature", "top_p"];
    if !cfg.is_openai() {
        return "sampling_not_sent=none".to_string();
    }
    let family = crate::llm::capabilities::OpenAiFamily::from_model_id(&cfg.model_id);
    let unsent: Vec<&str> = crate::llm::capabilities::unsent_keys(family)
        .into_iter()
        .map(|(key, _why)| key)
        .collect();
    let replaced: Vec<String> = SAMPLED_KNOBS
        .iter()
        .filter(|key| unsent.contains(key))
        .map(|key| format!("{key}:provider_default"))
        .collect();
    if replaced.is_empty() {
        // Either the local backend or a hosted family this table does not know:
        // every preset pin reached (or was kept for) the server.
        "sampling_not_sent=none".to_string()
    } else {
        format!("sampling_not_sent={}", replaced.join(","))
    }
}

/// The effort the CREATOR's request body actually carried, and its mechanism
/// (Item S).
///
/// [VANTA-WHY] Item S — the run-level `reasoning_effort` field names the level
/// this RUN asked for (`VANTA_REASONING_EFFORT`, effective). It cannot answer
/// the narrower question a reader of a hosted row has: what did the CREATOR's
/// own body carry? On the local backend the call site's pin is sent verbatim and
/// the server ignores it (`llamacpp:noop`); on a hosted reasoning model the pin
/// is dropped unless the run asked for a level the family accepts (§14.28), and
/// then the body carries nothing. The two facts are recorded separately so a run
/// whose Creator was never asked for an effort cannot be read as one where it
/// was. The decision reuses the shaper's own predicate (`keeps_effort`), so the
/// field and the body it describes cannot disagree.
fn creator_effort_text(cfg: &LlmConfig) -> (String, &'static str) {
    let requested = crate::llm::requested_effort();
    // The value `apply_reasoning_control` leaves in the body: the call site's pin,
    // replaced by the run switch when one is set.
    let carried = requested.unwrap_or(crate::llm::CREATOR_PINNED_EFFORT);
    if !cfg.is_openai() {
        return (
            carried.to_string(),
            crate::llm::reasoning_effort_mechanism(false),
        );
    }
    let family = crate::llm::capabilities::OpenAiFamily::from_model_id(&cfg.model_id);
    if crate::llm::capabilities::keeps_effort(family, requested) {
        (carried.to_string(), crate::llm::OPENAI_EFFORT_MECHANISM)
    } else {
        // No level was requested, so `capabilities` removed the call site's `low`
        // before the body left. `none` (a level nobody sent) is the fact; the
        // mechanism is `unset` for the same reason the run-level field uses it.
        ("none".to_string(), "unset")
    }
}

/// The real assembled prompt tokens for each Creator call, in call order
/// (Item S).
///
/// [VANTA-WHY] Item S — the batch table carries the Creator prompt ESTIMATE
/// (`est tok`, priced by the harness before the run) and the model's own count
/// (`llm_prompt_tokens`, a run total). Neither answers "how big was the Creator's
/// assembled prompt, and how close did it come to the window?" — the estimate is
/// a pre-run guess and the run total mixes every site. Recorded per Creator call,
/// beside `n_ctx`, so the two can be read against each other. `unmeasured` is a
/// call the server published no `usage` block for (the `estimate` source), kept
/// as a word rather than a `0`, which would read as an empty prompt.
fn creator_prompt_tokens_text(calls: &[crate::llm::CallUsage]) -> String {
    let values: Vec<String> = calls
        .iter()
        .filter(|call| call.site == "creator")
        .map(|call| {
            call.prompt_tokens
                .map(|tokens| tokens.to_string())
                .unwrap_or_else(|| "unmeasured".to_string())
        })
        .collect();
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

/// The run-log account of a BATE harness error (Item (a)).
///
/// [VANTA-WHY] Item (a) — the evidence that says WHY BATE stopped used to live
/// only in the `bate.harness_error[…]` session row, so the run's own console
/// printed "the failure is not the test's own" without ever saying what the
/// failure WAS: a reader of the run log could not see the `ModuleNotFoundError`
/// (or the pytest option that was not recognized) that ended the attempt. The
/// evidence is bounded (`bate::REVISE_EVIDENCE_MAX_LINES`), so quoting it here
/// restores the lost information without risking a runaway test flooding the
/// log. Indented so the multi-line block reads as one report under the header.
fn harness_error_log_line(test_name: &str, evidence: &str) -> String {
    let mut out = format!(
        "  ✖ BATE HARNESS ERROR: the baseline never reached a verdict for '{test_name}' and the \
         failure is not the test's own. Stopping this attempt — the Attacker cannot fix \
         the environment."
    );
    for line in evidence.lines() {
        out.push('\n');
        out.push_str("      ");
        out.push_str(line);
    }
    out
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
    // [VANTA-WHY] M3 item 8 — the B2 arm's switch, read here with the other two
    // so the arm name and the code path cannot disagree. `VANTA_BATE=on` (the
    // default) is the run every existing measurement was taken under: BATE
    // validates each Attacker test before it may reject. `off` is the paper's
    // B2 arm — adversarial tests admitted with NO baseline validation and NO
    // intent audit — and is the only setting that changes behaviour, which is
    // why the default is `on` and an absent var means "BATE ran" (a default-ON
    // switch is the shape that keeps every existing arm byte-identical).
    let bate_enabled = env_flag("VANTA_BATE", true)?;
    // [VANTA-WHY] M3 item 8 follow-up — the arm, STATED by the harness (`VANTA_ARM`)
    // rather than inferred from the switches. `b0` and `baseline` are the same switch
    // pair and differ only in the attempt budget, so `benchmark_config_name` cannot
    // name them apart; the stated name is what `run.summary` records as `arm`. Read
    // here, with the switches, so a name that contradicts them — or a `--max-attempts`
    // that would redefine the arm — fails before the repo is loaded and before any
    // call is made. An absent `VANTA_ARM` is a hand-run binary and resolves to the
    // switch-derived name exactly as every run before this change did.
    let stated_arm = env_arm()?;
    // [VANTA-WHY] M3 item 7 — the judge setting (`VANTA_JUDGE=off|plain|suite`),
    // read here with the arm so a stated arm and a stated judge mode cannot
    // disagree: `resolve_arm` refuses the pair rather than recording a verdict
    // under a name the judge setting contradicts. Absent/`off` is `None`, which
    // is every non-judge arm and every run before this change.
    let judge = env_judge()?;
    let resolved_arm = resolve_arm(
        stated_arm.as_deref(),
        attacker_enabled,
        planner_enabled,
        bate_enabled,
        config.max_attempts,
        judge,
        config.fixed_rounds,
    )?;
    // [VANTA-WHY] §14.2/§14.4/§14.5 — the three robustness switches. All
    // default OFF so v0.2 behaves exactly as before until they are flipped, and
    // the values are written into `run.summary` so a session's artifacts state
    // which rules produced its verdict. Read once here for the same reason the
    // two above are: a later reader must not be able to disagree with the code
    // path that actually ran.
    let corroborate_enabled = env_flag("VANTA_CORROBORATE", false)?;
    let grounded_audit_enabled = env_flag("VANTA_GROUNDED_AUDIT", false)?;
    let batch_audit_enabled = env_flag("VANTA_BATCH_AUDIT", false)?;
    // [VANTA-WHY] §14.22 — the one PROMPT switch in this list: it appends a
    // sentence to the Attacker's first request telling the model that one check is
    // enough. Default OFF, so the prompt every existing measurement was taken
    // under is the prompt a run sends unless the arm is asked for explicitly, and
    // it is recorded in `run.summary` so an ablation row states which prompt it
    // sent.
    let attacker_concise_enabled = env_flag("VANTA_ATTACKER_CONCISE", false)?;
    crate::agents::attacker::set_attacker_concise(attacker_concise_enabled);
    // [VANTA-WHY] §14.15 — Vanta's own wall-clock deadline, in seconds from here.
    // Unset means no deadline, which is what every run before this did; the
    // harness sets it ten minutes short of its own `RUN_TIMEOUT_SECS` so a run
    // that would have been killed ends itself instead, with a verdict and a
    // `run.summary` on disk. Armed in the LLM layer so it is checked before every
    // call, including the ones inside BATE and the Diagnostician.
    let deadline_secs = env_secs("VANTA_DEADLINE_SECS")?;
    crate::llm::set_deadline_secs(deadline_secs);
    if let Some(secs) = deadline_secs {
        println!(
            "  Deadline: the run will stop making LLM calls after {secs} s and end as NeedsReview \
             ({BUDGET_EXHAUSTED})"
        );
    }
    // [VANTA-WHY] §14.16 — the presence-penalty arm. Set here so a malformed value
    // fails the run before it spends an hour generating, and so the value reaches
    // the pin table (and therefore every request body) before the first call. A
    // zero pins nothing but is still RECORDED, because "asked for zero" and "asked
    // for nothing" send identical bodies and a batch table has to tell them apart.
    let presence_penalty = env_float("VANTA_PRESENCE_PENALTY")?;
    crate::llm::set_presence_penalty(presence_penalty);
    if let Some(penalty) = presence_penalty {
        println!(
            "  Presence penalty: every call site will pin presence_penalty={penalty:?} \
             (VANTA_PRESENCE_PENALTY)"
        );
    }
    // [VANTA-WHY] §14.20 — the output-cap arm. Set here, before the repo is even
    // loaded, for §14.16's reason: a malformed value must fail the run before it
    // spends an hour generating, and the caps must reach the pin table (and
    // therefore every request body) before the first call. An empty list is every
    // run without the switch, and `set_max_tokens_overrides` with an empty list
    // leaves `sampling_pin_table` exactly as it was, so those bodies stay
    // byte-identical.
    let max_tokens_overrides = env_max_tokens_overrides()?;
    crate::llm::set_max_tokens_overrides(max_tokens_overrides.clone());
    if !max_tokens_overrides.is_empty() {
        let pairs: Vec<String> = max_tokens_overrides
            .iter()
            .map(|(site, cap)| format!("{site}={cap}"))
            .collect();
        println!(
            "  Output caps: {} (VANTA_MAX_TOKENS_<SITE>)",
            pairs.join(", ")
        );
    }
    // [VANTA-WHY] §14.21 — the reasoning arm. Read and applied here, beside the
    // other sampling switches and before the repo loads, so a malformed value
    // stops the run before it spends an hour generating and so the controls are
    // in place before the first request body is built. `None`/`None` — the
    // default, and every run before this change — leaves every body untouched.
    let reasoning_budget = env_reasoning_budget()?;
    // [VANTA-WHY] C5 — refuse the switch on a judge arm before it is applied, so a
    // budget that could never reach the judge's own (non-allow-listed) site stops
    // the run instead of silently producing an arm the operator did not ask for.
    refuse_reasoning_budget_on_judge_arm(reasoning_budget, &resolved_arm)?;
    crate::llm::set_reasoning_budget(reasoning_budget);
    let reasoning_effort = env_reasoning_effort()?;
    // [VANTA-WHY] §14.28 — the effort is validated against the MODEL that will
    // receive it, here, before the repo loads and before the first request is
    // built. The two hosted families do not accept the same ladder (Sol rejects
    // `none`), so a level that is legal in the abstract can be a 400 on this
    // model; failing the run at config time is what keeps that from being
    // discovered twenty minutes into a batch. The local backend has no ladder to
    // check against — its knob is a no-op — so the switch is only validated when
    // a hosted model is going to read it.
    if let Some(effort) = reasoning_effort {
        if cfg.is_openai() {
            let family = crate::llm::capabilities::OpenAiFamily::from_model_id(&cfg.model_id);
            if !family.accepts_effort(effort.name()) {
                return Err(format!(
                    "CONFIG_ERROR: VANTA_REASONING_EFFORT='{}' is not a level {} accepts \
                     (expected one of {}).",
                    effort.name(),
                    family.name(),
                    family.accepted_efforts().join("|"),
                )
                .into());
            }
        }
    }
    crate::llm::set_reasoning_effort(reasoning_effort);
    if reasoning_budget.is_some() || reasoning_effort.is_some() {
        println!(
            "  Reasoning: budget={} effort={} (mechanism {}, {}) \
             — sites: {}",
            crate::llm::reasoning_budget_text(),
            crate::llm::reasoning_effort_text(cfg.is_openai()),
            crate::llm::reasoning_budget_mechanism(cfg.is_openai()),
            crate::llm::reasoning_effort_mechanism(cfg.is_openai()),
            crate::llm::REASONING_BUDGET_SITES.join(", ")
        );
    }
    if !attacker_enabled {
        println!(
            "▌ Arm: {} — Attacker off (VANTA_ATTACKER=off): the repo's own suite and static \
             analysis are the whole question.",
            resolved_arm.name
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

    // ---- Context budget (design §1.6.2; feature item 1/2) ----
    // [VANTA-WHY] The context selection fills a TOKEN budget, not a byte cap, so
    // it has to know the window. This resolves it exactly as the run-time guard
    // does (`llm::context_window_tokens`: `VANTA_LLM_CONTEXT_TOKENS`, else the
    // server's `/props`, else 65536) and subtracts the same output reserve and
    // chat overhead the guard subtracts, plus the harness margin. The probe is
    // the same cached call the first LLM request would make, so this moves that
    // cost earlier rather than adding one — and it lets a run state its budget
    // before spending a single token on a prompt that could never fit.
    let n_ctx = match build_reqwest_client(&cfg) {
        Ok(client) => crate::llm::context_window_tokens(&client, &cfg).await,
        Err(_) => crate::context::FALLBACK_CONTEXT_TOKENS,
    };
    let (prompt_budget_tokens, _prompt_margin_tokens) = crate::context::budget_tokens(n_ctx);
    println!(
        "▌ File-selection budget: {prompt_budget_tokens} tok (n_ctx {n_ctx} − {} output − {} overhead − 25 % margin)",
        crate::context::OUTPUT_RESERVE_TOKENS,
        crate::context::CHAT_OVERHEAD_TOKENS,
    );

    // ---- Test command (Spec §7: explicit --test-cmd, pytest fallback) ----
    let test_cmd: TestCommand = match &config.test_cmd {
        Some(s) => TestCommand::from_shell(s),
        None => TestCommand::default(),
    }
    .with_env(config.sandbox_env.clone())
    .with_precompile(config.precompile.clone());
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
    // [VANTA-WHY] M3 item D1 — the two sides' GENERATED-TEST timeouts may differ only
    // through the test-only seam. `VANTA_PATCHED_TEST_TIMEOUT_SECS` exists so the timeout
    // signal path can be exercised end to end without making BATE's baseline
    // load-sensitive, and a real run that carries it is not a real measurement — hence
    // the warning.
    //
    // [VANTA-WHY] v1-m3.1 — the warning is about that seam ONLY, and now says so. It can
    // no longer describe the in-session native gate: the gate hands the original and the
    // patched tree ONE shared cap (`native_cap`, recorded as `native_session_cap_secs` on
    // `run.summary`), so "the patched side's native cap differs from the baseline's" is
    // impossible by construction. Read as written, the message invited the opposite
    // reading — that the native gate ran its two sides under two different caps — which
    // §8.5 of the v1-m3.1 design note removes.
    let baseline_test_timeout_secs = crate::pipeline::harness::test_timeout_secs();
    let patched_test_timeout_secs = crate::pipeline::harness::patched_test_timeout_secs();
    if baseline_test_timeout_secs != patched_test_timeout_secs {
        tracing::warn!(
            baseline_test_timeout_secs,
            patched_test_timeout_secs,
            "patched-side test timeout differs from baseline timeout; this seam exists for tests and can bias a real run (generated tests only — the native gate runs both sides under one shared cap)"
        );
        eprintln!(
            "  ⚠ patched-side test timeout differs from baseline timeout: baseline={}s patched={}s \
             (generated tests only — the in-session native gate runs both sides under one shared cap)",
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
            "model={} llm_backend={} model_version={} prompt_version={} prompt_hash={} provider={} api_key={} \
             vanta_version={} vanta_commit={} vanta_dirty={} tested_commit={} target_repo_dirty={} requested_backend={} actual_backend={} max_attempts={} \
             baseline_test_timeout_secs={} patched_test_timeout_secs={} deps_checkpoint={} deps_freeze={} test_cmd={} mode={} benchmark_instance={} sandbox_env={:?}",
            cfg.model_id,
            // [VANTA-WHY] The backend is part of the run's identity, not a
            // detail of its transport: a hosted row and a local row that shared
            // a model string would otherwise be indistinguishable in the log,
            // even though they reject different sampling fields, bill different
            // amounts, and never probe the same endpoints.
            cfg.backend.as_str(),
            std::env::var("VANTA_MODEL_VERSION").unwrap_or_else(|_| "unknown".into()),
            prompt_versions,
            crate::pipeline::prompt_hashes(),
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
            config.sandbox_env,
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
    //
    // [VANTA-WHY] §14.18 — this map is the FILESYSTEM every test run is executed
    // against (BATE's baseline and every revision, the native suite, the graded
    // patched tree, the dependency probe, the static pass), so it is the whole
    // repo, not a budgeted excerpt of it. It used to be `repo.source_files` /
    // `GitWorkingTreePair::original_files()` — the deterministic walk stopped at
    // 64 KB per file and 256 KB in total and blind to non-source extensions.
    // What that cost, and why both failures were silent, is written up in
    // `repo::complete_tree_files`.
    let execution_root: PathBuf = config
        .original_tree
        .clone()
        .unwrap_or_else(|| config.repo_path.clone());
    if let Some(orig_tree) = &config.original_tree {
        println!("▌ Two-tree mode: original tree = {}", orig_tree.display());
    }
    let original_files = crate::repo::complete_tree_files(&execution_root);
    println!(
        "  Execution tree: {} files from {}",
        original_files.len(),
        execution_root.display()
    );

    // [VANTA-WHY] Item P — the run's SEED: the on-disk execution tree, staged once
    // here so every in-session sandbox can be cloned from it. Before this, a
    // sandbox was seeded from the source concatenation alone, so a compiled
    // project (`pandas/1`, `matplotlib/1`) reached the VM without its `.so`
    // extensions and without its root `pytest.ini`/`setup.cfg` — the in-session
    // suite was then NOT the suite the harness's own oracle and native gates run.
    // The concatenation is still the SOURCE overlay (`harness::seed_and_write`);
    // the seed carries everything the walk cannot (compiled extensions, config
    // files). A failure to stage is deliberately not fatal: the in-session import
    // guard below turns a tree that did not arrive into a stated InfraFailure
    // rather than a silent zero-test run.
    let seed_tree = if resolved_arm.judge == Some("plain") {
        // A plain judge arm spawns no sandbox at all, so there is no tree to stage.
        None
    } else {
        match harness::prepare_seed_tree(&execution_root) {
            Ok(tree) => Some(tree),
            Err(e) => {
                println!(
                    "  ⚠ Could not stage the seed tree ({e}); sandboxes fall back to \
                     source-concat seeding"
                );
                None
            }
        }
    };
    harness::set_seed_root(seed_tree.as_ref().map(|tree| tree.path().to_path_buf()));

    // ---- Governor + Gatekeeper ----
    // [VANTA-WHY] M3 item 9 — `--fixed-rounds K` makes the loop run EXACTLY K
    // attempts, so the Governor's cap IS K when the option is set. `K` wins over
    // `--max-attempts`: the option's whole meaning is "always run K attempts",
    // and a smaller `--max-attempts` would silently truncate the run the option
    // asked for. `None` leaves the cap exactly what it always was, so a default
    // run is byte-identical.
    //
    // The second half of the reading — that `--patch` review cannot honour the
    // option — is in `resolve_fixed_rounds`, which is where the decision is
    // testable. All it leaves here is the loud sentence: a run that asked for
    // something it cannot have must say so rather than quietly do something else.
    let (fixed_rounds, fixed_rounds_refused) =
        resolve_fixed_rounds(config.fixed_rounds, config.patch_path.is_some());
    if fixed_rounds_refused {
        println!(
            "  ⚠ --fixed-rounds is a repair option and this run reviews a FIXED patch (--patch): \
             the option is IGNORED and the attempt cap stays --max-attempts"
        );
    }
    let loop_attempts = fixed_rounds.unwrap_or(config.max_attempts);
    let mut governor = Governor::new(loop_attempts);
    let gatekeeper = Gatekeeper::new(VerificationPolicy::BestEffort);

    // [VANTA-WHY] The tree the context selection reads. In two-tree mode the
    // patch applies to the original worktree, so that is the tree whose contents
    // belong in the prompt; otherwise the repo path is both. It is the same tree
    // the tests execute in — the prompt and the execution differ in what they
    // SELECT from it, never in which tree it is.
    let context_root: PathBuf = execution_root.clone();
    // [VANTA-WHY] The context selection for the roles that are shown source. In
    // review mode it is selected from the patch under review; in repair mode it is
    // ranked from the issue text. Both are functions of (tree, issue text,
    // patch-if-any) — all fixed for the whole run — so each is selected ONCE and
    // reused across attempts. Re-walking per attempt would re-read the repo every
    // round and, worse, could select differently if the tree changed under us,
    // which would make two attempts of one run incomparable.
    let repair_selection: Option<crate::context::ContextSelection> =
        if config.patch_path.is_none() {
            Some(select_repair_prompt(
                &context_root,
                &repo.issue_text,
                prompt_budget_tokens,
            ))
        } else {
            None
        };
    // Review mode selects lazily (only once the patch's touched files are known);
    // repair mode is already fixed above.
    let mut review_selection: Option<crate::context::ContextSelection> = None;

    // ---- Repair loop ----
    let mut prior_feedback: Option<String> = None;
    let mut final_evidence: Vec<TestEvidence> = Vec::new();
    let mut final_patched_files: Option<HashMap<String, String>> = None;
    // [VANTA-WHY] M3 item E — the last tree the run actually PRODUCED, i.e. the
    // `patched_files` of the most recent attempt whose patch applied. The loop
    // re-applies each attempt from `original_files` and writes nothing when an
    // apply fails, so a run that ends on a malformed patch would otherwise leave
    // `final_patched_files` at `None` and write no `final.patch` — censoring a
    // `MaxAttemptsReached` row to `unobservable` even though the run produced a
    // perfectly gradeable tree one attempt earlier. Holding the last applied tree
    // lets the terminal arms below write THAT tree, and only a run where nothing
    // ever applied falls back to the original (an empty patch, a measured
    // failure). `None` until the first successful apply.
    let mut last_applied_patched_files: Option<HashMap<String, String>> = None;
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
    // [VANTA-WHY] M3 item W — did any attempt's native gate collect nothing
    // while the row's own prepare recorded tests for this bug? Run-level, like
    // `apply_failed_attempts`: it says the native check could not be done at all,
    // which is a property of the run and not of any one test.
    let mut native_vacuous_seen = false;
    // [VANTA-WHY] M3 item D1 — was any attempt's native gate vacuous BECAUSE a
    // side was killed by the wall-clock cap (rather than a tree that collected
    // nothing)? Run-level like `native_vacuous_seen`, and kept separate because
    // the two emptinesses have opposite remedies: raise the derived cap, or fix
    // the tree's import. A `true` here never changes the label — the run is
    // `InfraFailure` either way — it only records WHICH infra fault it was.
    //
    // [VANTA-WHY] Finding C (PR #4 review) — `native_timeout` means an in-session
    // native run hit the CONFIGURED cap `native_cap` (either side). A kill by the
    // DEADLINE CLAMP is NOT this flag: the clamp lowered the cap below the one the
    // run was configured with, so the kill is the deadline's doing and is recorded
    // by `native_gate_clamped` instead. Without the exclusion a clamped kill would
    // set `native_timeout=true` and report a cap that the suite never actually ran
    // under. The flag still implies `native_vacuous_seen` (it is set only when the
    // gate is vacuous), which is what every consumer below relies on.
    let mut native_timeout_seen = false;
    // [VANTA-WHY] v1-m3.1 (§4) and F3 (PR #4 review) — the per-side facts of the
    // last native gate the run reached (its seconds, its kill flags, the cap it ran
    // under) plus the any-attempt kill flags, carried out of the loop onto
    // `run.summary` and `evidence.json` so the §8.1 reading is auditable from the
    // artifact. `secs` are the side's own seconds (a cache hit carries the served
    // report's measurement, which is still a measurement of identical bytes —
    // `native_cache` names the source row). See `NativeGateFacts` for why the
    // last-gate trio and the any-attempt pair are kept apart.
    let mut native_gate = NativeGateFacts::default();
    // [VANTA-WHY] v1-m3.1 F3 — was the LAST native gate run under a cap that the
    // deadline clamp LOWERED below the resolved `native_cap`? Run-level, like the
    // per-side fields above, and separate from `native_vacuous_seen` on purpose:
    // an original side cut by the clamp takes the deadline exit
    // (`budget_exhausted`/`NeedsReview`), NOT the infrastructure exit, so it must
    // record `native_vacuous=false` and this flag instead. Without it the
    // distinction is invisible: a clamped `budget_exhausted` row would look like a
    // machine fault that could not grade the base tree.
    let mut native_gate_clamped = false;
    // [VANTA-WHY] M3 item S1 — every native gate call's cache note, per attempt
    // and per side, so the reuse is auditable from the run's own artifacts: which
    // results were served from an earlier row, under which key, and from whose
    // row. `native_cache_hits` is the count that rides `run.summary` so a digest
    // can cut by "how much of this run was measured vs. served" without a reader
    // having to walk `evidence.json`.
    let mut native_cache_records: Vec<serde_json::Value> = Vec::new();
    let mut native_cache_hits: u32 = 0;
    // [VANTA-WHY] M3 item D1 / v1-m3.1 — the wall-clock cap every in-session native
    // suite of this run uses. Declared once, here, so the gate, the judge's run and
    // the terminal `run.summary` all name the SAME number: `None` on the config keeps
    // the session floor (`harness::native_session_timeout_floor`, 1200 s), and a row
    // whose prepare measured its native suite passes `max(1200, 10 × measured)`.
    let native_cap = resolve_native_timeout_secs(config.native_timeout_secs)?;
    // [VANTA-WHY] Every NEW static finding this run's patch introduced, per
    // attempt, so evidence.json can name the finding that rejected a review
    // round. Session 40e84874 was rejected for a mypy finding that appeared
    // NOWHERE in the session files: the console printed `mypy_new=true` and the
    // detail was dropped. The record below is that missing detail.
    let mut static_findings_records: Vec<serde_json::Value> = Vec::new();
    // [VANTA-WHY] M3 item V — the file-visibility audit, per attempt. matplotlib/1
    // repaired `lib/matplotlib/image.py` on three attempts while that file sat in
    // the selection's `dropped_for_budget` set, so the Creator was editing the
    // exact source lines it had never been shown. Nothing recorded that. Each
    // attempt now contributes one record naming, for every file the patch touches,
    // how that file stood in the Creator's own context (`full` / `outline` /
    // `path_only` / `dropped_for_budget` / `not_selected`), and the run carries —
    // out of the loop, onto `run.summary` and into the digest — the number of
    // attempts that edited at least one file it could not read in full.
    //
    // [VANTA-WHY] LOGGING ONLY. This reads the selection and the patch that already
    // exist and writes down what they were; it never changes the prompt, never
    // injects a file the Creator was not shown, and never touches the selection.
    let mut context_visibility_records: Vec<serde_json::Value> = Vec::new();
    let mut attempts_edited_unshown_file: u32 = 0;
    // [VANTA-WHY] Q1 — how many `===FILE:` marker lines across this run's Creator
    // attempts carried leading whitespace and had to be trimmed to be recognised.
    // The Creator prompt shows the markers indented, so a run whose count is above
    // zero is a Creator that copied the prompt's shape and a parser that would
    // once have mis-read it. LOGGING ONLY: the number is counted and reported, and
    // nothing — no metric, label or verdict — reads it. Present only where a
    // Creator ran; a review run has no markers of its own.
    let mut markers_trimmed: u32 = 0;
    // [VANTA-WHY] Q2 — the paths a Creator patch INTRODUCED: blocks whose path the
    // original tree did not have and whose content was not empty. A mistyped marker
    // path (`===FILE: cartt.py===` for a tree that has `cart.py`) is exactly one of
    // these, and the apply already accepts it verbatim rather than refusing it — it
    // is a new module the fix adds as far as the tree can tell. Nothing used to
    // record it, so a patch that quietly created a file next to the one it meant to
    // edit looked like an ordinary apply. LOGGING ONLY: no metric, label or verdict
    // reads the list; it names what happened, it does not judge it. Union across
    // attempts, sorted; present only where a Creator ran.
    let mut new_files_created: BTreeSet<String> = BTreeSet::new();
    // [VANTA-WHY] M3 item 9 — one record per attempt of a fixed-rounds run: the
    // patch it produced, the test set it ran, the feedback it handed to the next
    // attempt, the hash of the Creator prompt that produced it, and whether the
    // attempt passed. The artifacts themselves live in the session's debug dir
    // (`attempt_<n>_creator_patch.txt`, `attempt_<n>_tests.json`,
    // `attempt_<n>_feedback.txt`); this is the index the evidence file carries, so
    // "with K = 4 the run stores 4 patches and the per-attempt test sets" is
    // answerable from the run's own files without a reader globbing a directory
    // whose naming is otherwise a convention. Always populated (a default run
    // simply has one entry), because the fold-ins exist to make M3 data usable by
    // M3.5 later and a field that only exists on an optional flag is a field that
    // cannot be recovered.
    let mut fixed_round_records: Vec<serde_json::Value> = Vec::new();
    // [VANTA-WHY] M3 item 9 — the patch the PREVIOUS attempt returned, so this
    // attempt's record can say whether the Creator returned it byte-identical
    // ("unchanged") or wrote a new one. `None` on attempt 1, where there is no
    // previous attempt to have repeated.
    let mut previous_patch_text: Option<String> = None;
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

    // [VANTA-WHY] §14.2/§14.3 — the corroboration hand-over out of the loop.
    // `deciding_rule` / `deciding_evidence` name the rule (a–e) that decided the
    // rejection, so a finished run's artifacts can say WHY it was a rejection and
    // not just that one happened. `needs_review_weak` lists the uncorroborated
    // failures behind a `NeedsReview`, because "needs a human" is useless without
    // naming what the human should look at. All three stay empty when the flag is
    // off, so v0.2's `run.summary` is unchanged.
    let mut deciding_rule: Option<String> = None;
    let mut deciding_evidence: Vec<String> = Vec::new();
    let mut needs_review_weak: Vec<String> = Vec::new();
    // [VANTA-WHY] The intent auditor's answer to "does the issue require the
    // behaviour at this test's input to change?", keyed by test name. Rule (d)
    // needs it to downgrade a preservation/acceptance failure the issue overrides.
    // The audit runs per round and re-runs after a BATE revision, so the LAST
    // answer for a name wins — it is the one that describes the test version that
    // actually ran. Absent means UNKNOWN, and rule (d) only downgrades on an
    // explicit `true`.
    let mut issue_requires_change_by_test: HashMap<String, Option<bool>> = HashMap::new();
    // [VANTA-WHY] §14.5 — the largest output cap the batched audit was given in
    // this run, over every round. The cap scales with the round (`8192` per test,
    // ceilinged), so it is a property of the run's rounds, not a constant, and a
    // reader of `run.summary` cannot recover it from the schedule: the schedule
    // records the SINGLE-call pin (`max_tokens_intent_audit`), which the batch
    // deliberately does not use. `None` means no batched call was made — the flag
    // was off, or no round reached the audit — and is recorded as `none`, the
    // same word the sampling provenance uses for "the server showed nothing".
    let mut batch_audit_max_tokens: Option<u64> = None;

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
    // [VANTA-WHY] M3 item 7 — the judge's own failure text, carried out of the
    // loop so a `JudgeFailed` run writes it to the session log. Deliberately NOT
    // `infra_failure_error`: the machine was fine and the call was made — what
    // came back could not be read as a verdict after the one retry — so this must
    // not be reported, retried or counted as infrastructure. The raw answer
    // itself is already on disk in the forensics file.
    let mut judge_failed_error: Option<String> = None;
    // [VANTA-WHY] v1-m3.2 item B — the failed model call's own text, carried out
    // of the loop so a run ended by an unreadable answer (`stop_reason =
    // model_output_invalid`) records WHY on its own phase row. Deliberately NOT
    // `infra_failure_error`: the machine was fine and the call was made, and a
    // reader retries an infrastructure fault but not a model that cannot answer
    // this schema. `None` on every run that reached a verdict about the candidate.
    let mut model_output_error: Option<String> = None;
    // [VANTA-WHY] M3 item 7 — the judge's verdict and reasoning, carried out of
    // the loop for the `run.judge` provenance row. `None` on every non-judge arm,
    // so that row exists only where a judge actually ran.
    let mut judge_outcome: Option<(bool, f64, String)> = None;
    // [VANTA-WHY] §14.14 — the run's total Attacker remediation calls, summed
    // over attempts from the per-attempt budget below. Reported in `run.summary`
    // (and thus the manifest) beside the cap, so a reader can tell a run that
    // never needed a revision from one that spent its budget.
    let mut revise_calls_used_total: u32 = 0;
    // [VANTA-WHY] §14.15 — why the run stopped, when it stopped itself rather
    // than reaching a verdict about the candidate. `None` is every ordinary
    // outcome; `budget_exhausted` is the deadline above. Recorded in
    // `run.summary` (and thus the manifest) under its own name, because the
    // harness's `timed_out` means the OPPOSITE thing — that Vanta was killed
    // before it could decide anything.
    let mut stop_reason: Option<&'static str> = None;
    // [VANTA-WHY] §14.19 — the test whose BATE baseline run died before pytest
    // graded it. Carried out of the loop so the terminal row can say WHY the run
    // ended as `HarnessFailure` without the reader opening the session log, and so
    // this class of bug is one legible manifest row rather than a long trail of
    // revisions. `None` is every run in which the environment held.
    let mut harness_error: Option<String> = None;
    // [VANTA-WHY] M3 item 1 — the grounding rule's ledger for the run. The plan
    // asks for the number of reproduction tests the rule scrapped per review;
    // reporting the pair (scrapped, kept because the issue names what the test
    // asserts) makes it readable as a before/after, which is the whole point of
    // the loosening. `None` means NO grounded audit ran in this run
    // (`VANTA_GROUNDED_AUDIT=off`), and it is written as `null` rather than `0`:
    // `0` is a measurement ("the audit ran and scrapped nothing"), `null` is the
    // absence of one, and a reader must not confuse the two. The slot becomes
    // `Some(0)` at the first grounded-audited round.
    let mut grounded_scrapped_total: Option<usize> = None;
    let mut grounded_kept_by_issue_total: Option<usize> = None;
    // [VANTA-WHY] M3 item 17 — every test the Diagnostician quarantined this run,
    // in the order it was quarantined. A quarantined test is REMOVED from the
    // shared set so it is not re-run next round, and its name + reason are handed
    // to the Attacker (`AttackerRequest::quarantined_tests`) so it does not write
    // it again. The accumulator is run-scoped, not attempt-scoped, because a
    // quarantine is a durable fact about the probe: once removed, it stays out.
    let mut quarantined: Vec<QuarantinedProbe> = Vec::new();
    // [VANTA-WHY] Item P, point 2 — the in-session import guard's own line, carried
    // out of the loop for the `run.summary` row and a `run.import_guard` session
    // row. `None` means the guard has not run yet (and, for a plain judge arm,
    // never will). It runs ONCE, before the run's first sandbox.
    let mut import_guard_line: Option<String> = None;
    let decision = 'attempts: loop {
        // [VANTA-WHY] §14.15 — the deadline is checked here as well as in the LLM
        // layer, and this is the check that turns "the calls stopped" into "the
        // run stopped": a phase whose calls were refused would otherwise start
        // another attempt — another Attacker round, another BATE pass, another
        // native suite run — before anything noticed the clock. `NeedsReview`
        // rather than a rejection, because the deadline is a fact about the run
        // and not about the candidate: nothing has been shown to be wrong.
        if crate::llm::deadline_past() {
            stop_reason = Some(BUDGET_EXHAUSTED);
            println!(
                "\n  ⏱ Wall-clock deadline reached — stopping here and ending the run as \
                 NeedsReview ({BUDGET_EXHAUSTED}); the candidate has been neither approved nor \
                 rejected."
            );
            // [VANTA-WHY] v1-m3.1 item 1 — a deadline exit is a terminal verdict, so
            // it leaves a `final.patch` like every other: the run's last PRODUCED
            // tree, picked by EXACTLY the rule `MaxAttemptsReached` uses
            // (`final_patched_files = Some(patched_files)`, the attempt's own
            // applied tree). Here the attempt top is reached BEFORE this round's
            // apply, so the most recent produced tree is `last_applied_patched_files`
            // — `None` until the first apply, which is what keeps a run that never
            // produced one (`unobservable`) censored and halting, as intended.
            // Without this the runner censors the oracle and a single deadline halts
            // the whole wave.
            final_patched_files = last_applied_patched_files.clone();
            break 'attempts GovernorDecision::NeedsReview;
        }
        let attempt = governor.current_attempt();

        // [VANTA-WHY] §14.14 — this ATTEMPT's shared revise+simplify budget. Fresh
        // per attempt (a new round may revise again), shared by every test in it
        // and by BATE and the Diagnostician, so the cap cannot be escaped by a
        // revision that returns a renamed test.
        let mut revise_budget = ReviseBudget::new();

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

        // [VANTA-WHY] Item P, point 2 — the in-session IMPORT GUARD. Runs once,
        // before the run's first sandbox, and only for an arm that actually runs a
        // suite (a plain judge arm spawns no sandbox by design, so there is nothing
        // to guard). It imports the project's package in a sandbox seeded exactly
        // as the suite will be, and refuses a run whose package resolves OUTSIDE
        // the tree under test — the silent, unmeasurable state a
        // concatenation-only seed produced for a compiled project. The refusal is
        // infrastructure, not a statement about the candidate, so it reuses the
        // existing InfraFailure machinery (run.infra_failure row, exit code 2).
        if import_guard_line.is_none() && resolved_arm.judge != Some("plain") {
            let guard_concat = build_source_concat(&original_files);
            match harness::guard_project_import(&guard_concat, &test_cmd, &execution_root).await {
                Ok(line) => {
                    println!("  {line}");
                    import_guard_line = Some(line);
                }
                Err(reason) => {
                    println!("  ✗ IMPORT GUARD: {reason}");
                    infra_failure_error = Some(reason);
                    break GovernorDecision::InfraFailure;
                }
            }
        }
        // [VANTA-WHY] Forensics filenames carry the ROUND, not just the
        // structured-call retry index. Without this, every round wrote over the
        // previous one and the debug folder of a 3-round run held round 3 three
        // times over.
        crate::llm::set_forensics_round(attempt + 1);
        println!("\n══════════════════════════════════════════════════════════════");
        if fixed_rounds.is_some() {
            // [VANTA-WHY] M3 item 9 — say WHICH run this is. "ATTEMPT 2 / 4" reads
            // the same whether the cap is a patch budget or a fixed round count,
            // and the two mean different things to a reader of the console.
            println!(
                "  ATTEMPT {} / {} (fixed rounds)",
                attempt + 1,
                loop_attempts
            );
        } else {
            println!("  ATTEMPT {} / {}", attempt + 1, loop_attempts);
        }
        println!("══════════════════════════════════════════════════════════════");

        // ---- 1. Patch (Creator or --patch file) ----
        let patch_files: HashMap<String, String>;
        let patch_text: String;
        // [VANTA-WHY] M3 item 9 fold-in — the hash of the Creator prompt THIS
        // attempt sent (system + the exact user prompt, which differs per attempt
        // because the feedback does). `None` on a `--patch` review: no Creator
        // ran, so there is no Creator prompt. The prompt itself is NOT stored —
        // the selection it was built from is already on disk (`context.select`)
        // and the hash is what makes "two attempts whose Creator read the same
        // thing" checkable without a second copy of the repository in the log.
        let mut creator_prompt_hash: Option<String> = None;
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
                // [VANTA-WHY] The Creator is shown the SELECTED context (the files
                // the issue names, their one-hop imports, and the repo's file tree),
                // not the capped dump `repo.source_files` holds. `repair_selection`
                // is `Some` in exactly this branch (no `--patch`), so the fallback
                // never fires in practice; it keeps the type honest.
                source_files: repair_selection
                    .as_ref()
                    .map(creator_source_files)
                    .unwrap_or_else(|| repo.source_files.clone()),
                prior_feedback: prior_feedback.clone(),
            };
            // [VANTA-WHY] M3 item 9 fold-in — hash the prompt the call is about
            // to send. `creator_user_prompt` is the same function the call site
            // uses, so the hash cannot describe a prompt the model never saw.
            let creator_user_prompt = crate::agents::creator::creator_user_prompt(&creator_request);
            creator_prompt_hash = Some(crate::core::prompt_content_hash(&[
                crate::agents::creator::creator_system_prompt(),
                creator_user_prompt.as_str(),
            ]));
            // [VANTA-WHY] M3 item F5 — price the prompt BEFORE spending the call, so
            // a prompt that cannot fit this window is named `prompt_over_budget`
            // rather than arriving later as a generic "the Creator call failed".
            let creator_prompt_tokens = crate::context::estimate_tokens(
                crate::agents::creator::creator_system_prompt().len()
                    + creator_user_prompt.len(),
            );
            if let Some(reason) = prompt_over_budget_reason(
                creator_prompt_tokens,
                crate::context::OUTPUT_RESERVE_TOKENS,
                n_ctx,
            ) {
                println!("\n  ✗ {reason}");
                infra_failure_error = Some(reason);
                break GovernorDecision::InfraFailure;
            }
            let creator_response = match run_with_spinner("Creator generating patch", async {
                creator_generate_patch(&cfg, &creator_request).await
            })
            .await
            {
                Ok(response) => response,
                // [VANTA-WHY] §14.15 — a call the deadline refused is NOT an
                // infrastructure failure. "The machine broke" and "the run ran out
                // of clock" are opposite claims about the run, and the exit codes
                // say so (2 versus 3), so the marker is matched before the arm
                // that would otherwise blame the server.
                Err(error) if error.to_string().contains(crate::llm::DEADLINE_EXCEEDED_MARKER) => {
                    stop_reason = Some(BUDGET_EXHAUSTED);
                    println!(
                        "\n  ⏱ The wall-clock deadline passed before the Creator call — ending \
                         the run as NeedsReview ({BUDGET_EXHAUSTED}), not as an infrastructure \
                         failure."
                    );
                    // [VANTA-WHY] v1-m3.1 item 1 — same rule as the attempt-top deadline
                    // exit above: attach the run's last PRODUCED tree (the same one
                    // `MaxAttemptsReached` attaches). This attempt's Creator was
                    // refused, so no tree was produced THIS round; `last_applied_patched_files`
                    // is the previous attempt's tree, and stays `None` on attempt 1 —
                    // there the row is genuinely unobservable and halts, as intended.
                    final_patched_files = last_applied_patched_files.clone();
                    break 'attempts GovernorDecision::NeedsReview;
                }
                Err(error) => {
                    let message = error.to_string();
                    // [VANTA-WHY] v1-m3.2 item B — an exhausted Creator call that
                    // failed on the MODEL'S OWN OUTPUT (a schema parse error, or an
                    // answer cut off twice) is a FAILED ATTEMPT, never an
                    // infrastructure failure: the machine was fine and the call was
                    // made, so this attempt simply produced no patch. The loop spends
                    // the attempt (the Governor advances the budget) and asks again
                    // with the next round's seed; if it was the LAST attempt the run
                    // ends `MaxAttemptsReached`, whose terminal arm attaches the run's
                    // last PRODUCED tree, so the row is graded (`unresolved`) rather
                    // than censored to `unobservable`. A transport error or a
                    // `prompt_over_budget` refusal is a different claim about the run
                    // and keeps `InfraFailure` below.
                    if is_model_output_failure(&message) {
                        println!(
                            "  ↻ Creator produced no patch (the model's own output could not be \
                             read) — spending this attempt and asking again"
                        );
                        // The attempt produced no patch, so it is a recorded hole in
                        // the dataset exactly as a failed apply is. `previous_shared`
                        // is deliberately untouched (the next attempt is still compared
                        // against the last candidate anyone OBSERVED), and the rows are
                        // persisted BEFORE the loop moves on so a budget that ends here
                        // cannot lose them.
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
                        prior_feedback = Some(format!(
                            "The previous attempt produced no usable patch — the answer could not \
                             be read as a patch:\n{message}\nPlease return ONLY the corrected \
                             unified diff."
                        ));
                        let decision =
                            governor.decide(governor_mode, AttemptVerdict::Inconclusive, false);
                        if decision == GovernorDecision::Continue {
                            continue 'attempts;
                        }
                        // [VANTA-WHY] The LAST attempt's Creator produced nothing, so
                        // the run ends `MaxAttemptsReached`. It is terminal, so it must
                        // still leave a `final.patch`: the run's last PRODUCED tree, or
                        // the original tree when nothing ever applied — the same rule
                        // the failed-apply exit takes, which is what makes this row
                        // `unresolved` rather than `unobservable`.
                        final_patched_files = Some(final_patch_tree(
                            last_applied_patched_files.as_ref(),
                            &original_files,
                        ));
                        break 'attempts decision;
                    }
                    // [VANTA-WHY] M3 item F5 — the pre-call check is a lower bound
                    // (it prices the two prompt strings; the guard also prices its
                    // own role wrappers), so the guard can still be the one that
                    // refuses. Give that case the same name instead of folding it
                    // into "the Creator call failed".
                    infra_failure_error = Some(if message
                        .contains(crate::llm::DETERMINISTIC_CONTEXT_PREFIX)
                    {
                        format!("prompt_over_budget: {message}")
                    } else {
                        format!("Creator LLM call failed: {message}")
                    });
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
            let (split_files, attempt_markers_trimmed) = split_multi_file_patch(
                &creator_response.patch_text,
                &creator_response.files_touched,
            );
            patch_files = split_files;
            // [VANTA-WHY] Q1 — how many `===FILE:` markers this attempt emitted
            // with leading whitespace and so needed trimming. The prompt shows the
            // markers indented, so a positive count is the Creator copying the
            // prompt's shape; it rides the run's record because a count that stops
            // being zero would name a prompt/parser mismatch before it becomes a
            // "patch changed nothing".
            markers_trimmed += attempt_markers_trimmed;
        }

        // [VANTA-WHY] F2 — open this attempt's record and advance the comparison
        // basis the moment the patch text exists, BEFORE any apply check can
        // `continue` past it (see `open_fixed_round_record`). Doing it later is
        // exactly the defect: a repair attempt whose patch fails to apply
        // `continue`s to the next round, so the attempt owned no record AND
        // `previous_patch_text` never advanced — the attempt after it was
        // compared with a patch two attempts old.
        let fixed_round_record_index = open_fixed_round_record(
            &mut fixed_round_records,
            &mut previous_patch_text,
            attempt,
            &patch_text,
            prior_feedback.clone(),
            creator_prompt_hash.clone(),
        );
        println!("  Patch spans {} file(s)", patch_files.len());
        let patch_attempt_id = patch_attempt_id_for_patch(&patch_text);
        attempt_patch_ids.push(patch_attempt_id.clone());
        let testing_iteration_id = fresh_testing_iteration_id();

        // [VANTA-WHY] M3 item 9 — store the patch THIS attempt produced, so
        // "every attempt's patch is stored" holds for the run's own files and not
        // only for the tree snapshot the pool already derives from. This is the
        // Creator's OWN text (full source or search/replace), exactly as emitted
        // — `attempt_<n>_patched_tree` is the applied result and `final.patch` is
        // the diff of the LAST attempt; neither is the text of an earlier one.
        // A `--patch` review writes the same bytes every round; stored anyway,
        // because the file is what makes an attempt's input recoverable.
        let creator_patch_path =
            debug_dir.join(format!("attempt_{}_creator_patch.txt", attempt + 1));
        let _ = fs::write(&creator_patch_path, &patch_text);

        // [VANTA-WHY] M3 item V — how the files THIS patch touches stood in the
        // Creator's own context. Only a repair attempt has a Creator, so only a
        // repair attempt has this question: `repair_selection` is `Some` exactly
        // when `--patch` is absent, and a review run's patch was written outside
        // the run against a different role's selection, which is not "the
        // Creator's context" and must not be recorded as if it were. Logging
        // only: the selection, the prompt and the patch are all read, never
        // changed.
        if let Some(selection) = &repair_selection {
            let touched: Vec<String> = patch_files.keys().cloned().collect();
            let visibility = selection.touched_visibility(&touched);
            let unshown: Vec<&str> = visibility
                .iter()
                .filter(|file| !file.shown_in_full())
                .map(|file| file.path.as_str())
                .collect();
            if !unshown.is_empty() {
                attempts_edited_unshown_file += 1;
                // [VANTA-WHY] the one line worth printing: a patch touched a file
                // the Creator could not read in full (matplotlib/1's `image.py`
                // was dropped for budget while three attempts patched it). An
                // attempt whose files were all shown in full prints nothing — the
                // record in `evidence.json` still carries every attempt.
                println!(
                    "  context visibility: {} edited file(s) not shown in full: {}",
                    unshown.len(),
                    visibility
                        .iter()
                        .filter(|file| !file.shown_in_full())
                        .map(|file| format!("{}={}", file.path, file.visibility))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            context_visibility_records.push(serde_json::json!({
                "attempt": attempt + 1,
                "files": visibility
                    .iter()
                    .map(|file| serde_json::json!({
                        "path": file.path,
                        "visibility": file.visibility,
                    }))
                    .collect::<Vec<_>>(),
                "unshown_files": unshown,
                "edited_unshown": !unshown.is_empty(),
            }));
        }

        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: Some(testing_iteration_id.clone()),
            test_id: None,
            phase: "attempt.start".to_string(),
            summary: format!(
                "attempt={} mode={} patch_files={} patch_bytes={} creator_prompt_hash={}",
                attempt + 1,
                if config.patch_path.is_some() {
                    "patch_review"
                } else {
                    "creator"
                },
                patch_files.len(),
                patch_text.len(),
                // [VANTA-WHY] M3 item 9 fold-in — the 12-hex hash of the Creator
                // prompt this attempt sent, and `none` where no Creator ran. The
                // hash is space-free by construction, so it survives the session
                // log's reader.
                creator_prompt_hash.as_deref().unwrap_or("none")
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
            // [VANTA-WHY] REPAIR MODE keeps the cascade — the Creator can emit a
            // diff with wrong headers, and the loop has a budget to recover — but
            // the cascade now runs per file INSIDE one all-or-nothing apply (M3
            // item A1, Contract B): if any block fails to anchor, no block is
            // written. Letting a first block land while a second fails would test
            // a tree no patch describes, and `final.patch` (built from the same
            // map) would silently disagree with the tree the tests ran on.
            print!(
                "  ⏳ applying {} file block(s), all-or-nothing... ",
                patch_files.len()
            );
            let _ = io::Write::flush(&mut io::stdout());
            match apply_patch_files_all_or_nothing(&original_files, &patch_files) {
                Ok(changed) if changed != original_files => {
                    println!("✔ all blocks applied");
                    for (path, content) in &changed {
                        if original_files.get(path) != Some(content) {
                            println!("  ✔ {path}: patched");
                        }
                    }
                    // [VANTA-WHY] Assigning the WHOLE map (not merging per file) is
                    // what makes Contract B structural: on success the tested tree is
                    // exactly what the helper returned, and on failure
                    // `patched_files` is never touched, so nothing is written.
                    patched_files = changed;
                    any_changed = true;
                    all_patch_failed = false;
                    // [VANTA-WHY] Q2 — the blocks that introduced a file the
                    // original tree did not have. Recorded only on a success: a
                    // Contract-B failure writes nothing, so it creates nothing. The
                    // helper is pure, so the rule is tested directly.
                    new_files_created
                        .extend(new_files_from_patch(&original_files, &patch_files));
                }
                Ok(_) => {
                    // Every block was a no-op (e.g. an all-empty new-file block):
                    // nothing changed, so this is the cascade's historical "patch
                    // produced no change" failure. `all_patch_failed` stays true.
                    println!("✗ patch changed nothing");
                }
                Err(failure) => {
                    // `apply_failure` carries item R's account (the file and every
                    // strategy tried) to the Creator; `patch_failure_reason` makes
                    // the forensics file and the `patch.apply_failed` row name the
                    // same thing instead of the generic "nothing changed".
                    println!("✗ a block did not apply — Contract B writes no file");
                    patch_failure_reason = Some(failure.feedback());
                    apply_failure = Some(failure);
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
                // [VANTA-WHY] Item R — `ApplyFailure::feedback()` carries the
                // cascade's own account (the file, every strategy tried with its
                // reason, and why reconstruction was skipped) instead of the
                // fixed sentence this used to send. Every arm that reaches
                // repair mode (b0/b1/b2/vanta) is `GovernorMode::Repair` and so
                // goes through this one path; review mode never enters here.
                let feedback = match &apply_failure {
                    Some(failure) => failure.feedback(),
                    None => format!("patch did not apply: {reason}"),
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
            // [VANTA-WHY] M3 item E — a run whose LAST attempt is the one that
            // failed to apply terminates here (review mode stops on it, repair
            // mode stops when that attempt exhausts the budget). It is a terminal
            // verdict, so it must still leave a `final.patch`: the run's last
            // PRODUCED tree, or the original tree when nothing ever applied. Only
            // `HarnessFailure`/`InfraFailure` below stay patchless.
            final_patched_files =
                Some(final_patch_tree(last_applied_patched_files.as_ref(), &original_files));
            break decision;
        }
        let _ = any_changed; // used implicitly via all_patch_failed
        // [VANTA-WHY] F2 — reaching here means the patch APPLIED (a failure
        // broke or `continue`d above), so mark the attempt's own record, which
        // was opened before the apply checks and so exists either way.
        if let Some(record) = fixed_round_records.get_mut(fixed_round_record_index) {
            record["applied"] = serde_json::json!(true);
        }
        // [VANTA-WHY] M3 item E — remember this applied tree as the run's last
        // produced one, so a LATER attempt that fails to apply cannot erase it
        // from `final.patch`. Cheap (a clone of the map the run already holds);
        // the terminal arms read it only on the failed-apply path.
        last_applied_patched_files = Some(patched_files.clone());

        // ---- Build source strings ----
        // [VANTA-WHY] §14.18 — the concat the tests run in is now the whole
        // repo, which is NOT what the prompt may contain. The prompt stays a
        // selection; when a selection legitimately renders empty the fallback is
        // still the capped dump it has always been, built from the same tree, so
        // no prompt grows because the execution tree did.
        let original_concat = build_source_concat(&original_files);
        // [VANTA-WHY] The source the Attacker, the auditor, BATE and the
        // Diagnostician are shown. This used to be the capped dump (the first
        // ~262 KB of the repo) in BOTH modes — which dropped the patch's own file
        // on any repo whose cap had already passed it, and dropped the bug's file
        // on matplotlib entirely. Now:
        //   * review mode: the patch's touched files, their one-hop imports and
        //     the visible tests that exercise them, so "empty" means the patch
        //     touches nothing in this tree (only then does the old concat apply);
        //   * repair mode: the files the issue names (path / traceback /
        //     identifier) and their one-hop imports, with the repo's file tree.
        let prompt_source_concat = if config.patch_path.is_some() {
            let selection = review_selection.get_or_insert_with(|| {
                let touched: BTreeSet<String> = patch_files.keys().cloned().collect();
                select_review_prompt(&context_root, &touched, prompt_budget_tokens, &repo.issue_text)
            });
            let rendered = selection.render();
            if rendered.trim().is_empty() {
                capped_dump_concat(&execution_root)
            } else {
                rendered
            }
        } else if let Some(selection) = &repair_selection {
            selection.render()
        } else {
            capped_dump_concat(&execution_root)
        };

        // ---- 2b. JUDGE (M3 item 7) ----
        // [VANTA-WHY] A judge arm answers the review question with ONE model call
        // instead of the adversarial loop: the Attacker, BATE, the Diagnostician,
        // the graded run, the native gate and static analysis are all SKIPPED, and
        // skipped rather than run-and-emptied for the same reason item 8 skips the
        // Attacker — an `attacker.summary` row saying `generated_tests=0` would
        // assert the Attacker ran and found nothing. This is also why the branch is
        // HERE rather than beside the decision machinery: a plain judge must not
        // run the sandbox at all, so nothing that spawns one may come first.
        //
        // The judge is shown exactly what the test-driven review arm's roles are
        // shown (`prompt_source_concat`, the `select_review_prompt` selection under
        // the same budget), so item 7's fairness rule holds by construction rather
        // than by two selections that happen to agree.
        if let Some(judge_mode) = resolved_arm.judge {
            let suite_result = if judge_mode == "suite" {
                // [VANTA-WHY] The candidate graded with the ORIGINAL tests, as the
                // test-driven arm's native gate does: a patch must not grade itself.
                let native_tree = harness::native_tree_with_original_tests(
                    &original_files,
                    &patched_files,
                    &patch_deleted_paths(&patch_text),
                );
                let judge_patched_concat = build_source_concat(&native_tree.files);
                // [VANTA-WHY] v1-m3.1 F2 — the judge's native run is CLAMPED to the
                // run's clock exactly as the gate's two sides are (§8.4), because it
                // is the same sized step and, unclamped, the same failure follows:
                // the attempt top honours the deadline only once the deadline has
                // ALREADY passed, so a run taken with the whole resolved cap can
                // start at `VANTA_DEADLINE_SECS` (this attempt's own in-session
                // import guard is allowed to start first, and it has its own cap)
                // and then outlive `RUN_TIMEOUT_SECS` on its own. The harness
                // SIGKILLs the row mid-run, and a killed row writes no manifest
                // entry at all — invisible to the wave-1 gate. The margin between
                // the deadline and that kill is ONE model call's worth, and this
                // arm's single model call (the judge's, below and AFTER this run)
                // already needs it: it cannot cover the suite as well. With nothing
                // left there is no run to make, and the attempt takes the SAME
                // deadline exit the top of the loop takes (`stop_reason =
                // BUDGET_EXHAUSTED`, `NeedsReview`), adding no new outcome. The
                // clamp can only LOWER the cap, so a run with time to spare keeps
                // the resolved `native_cap`.
                let remaining = crate::llm::remaining_secs();
                let judge_suite_cap = match native_gate_cap(native_cap, remaining) {
                    Some(cap) => cap,
                    None => {
                        stop_reason = Some(BUDGET_EXHAUSTED);
                        println!(
                            "\n  ⏱ Wall-clock deadline reached before the judge's native suite — \
                             skipping it and ending the run as NeedsReview ({BUDGET_EXHAUSTED}); \
                             the candidate has been neither approved nor rejected."
                        );
                        // [VANTA-WHY] v1-m3.1 item 1 — a deadline exit leaves a
                        // `final.patch`, exactly as `MaxAttemptsReached` does
                        // (`final_patched_files = Some(patched_files)`). This is the
                        // judge arm, so the most recent produced tree IS the candidate
                        // patch; attaching it lets the oracle grade the row instead of
                        // censoring it to `unobservable` and halting the wave.
                        final_patched_files = Some(patched_files);
                        break 'attempts GovernorDecision::NeedsReview;
                    }
                };
                println!("\n▌ JUDGE: running the repo's own suite on the candidate...");
                let report = run_with_spinner("Running the native suite for the judge", async {
                    // [VANTA-WHY] M3 item D1 — the judge's native run is the same
                    // full repo suite as the gate's, so it takes the same per-bug
                    // cap. v1-m3.1 — the RESOLVED `native_cap`, not the raw
                    // config option: `None` would otherwise fall through to the
                    // 120 s default inside `run_native_suite` and grade the
                    // candidate under a cap the gate never used.
                    harness::run_native_suite(
                        &judge_patched_concat,
                        &test_cmd,
                        Some(judge_suite_cap),
                    )
                    .await
                })
                .await;
                // [VANTA-WHY] Finding A (PR #4 review) — a judge suite the harness
                // could not SET UP or transport measured nothing about the candidate,
                // so the judge must not be handed an empty result as if the tree had
                // no tests, and the row must not read as a REJECTED candidate. Take
                // the same infrastructure exit the native gate takes.
                if report.setup_failed {
                    let reason = format!(
                        "Judge's native suite could not be SET UP or transported (no exit status \
                         and no cap timeout) — the candidate was neither graded nor rejected; \
                         patched=not_run (tests={} exit_code={} stderr={:?})",
                        report.tests.len(),
                        report
                            .exit_code
                            .map(|code| code.to_string())
                            .unwrap_or_else(|| "unset".to_string()),
                        report.stderr_tail,
                    );
                    println!("  ✗ NATIVE SUITE SETUP FAILED: {reason}");
                    infra_failure_error = Some(reason);
                    break 'attempts GovernorDecision::InfraFailure;
                }
                // [VANTA-WHY] Finding B (PR #4 review) — the CLAMPED suite above may
                // have consumed the last of the run's clock (the cap is derived from
                // what is left, §8.4), so the deadline is honoured HERE, before the
                // judge's model call: a run with nothing left takes the deadline exit
                // instead of a refused call being read as an infrastructure failure.
                // This is the same exit the top of the attempt loop takes.
                if crate::llm::deadline_past() {
                    stop_reason = Some(BUDGET_EXHAUSTED);
                    println!(
                        "\n  ⏱ The wall-clock deadline passed while the judge's native suite ran \
                         — ending the run as NeedsReview ({BUDGET_EXHAUSTED}); the candidate has \
                         been neither approved nor rejected."
                    );
                    // [VANTA-WHY] v1-m3.1 item 1 — deadline exit; attach the candidate
                    // patch under the same rule `MaxAttemptsReached` uses
                    // (`final_patched_files = Some(patched_files)`), so the oracle
                    // grades the row instead of the runner censoring it.
                    final_patched_files = Some(patched_files);
                    break 'attempts GovernorDecision::NeedsReview;
                }
                Some(render_judge_suite_result(&report))
            } else {
                None
            };
            let request = JudgeRequest {
                issue_text: repo.issue_text.clone(),
                context: prompt_source_concat.clone(),
                candidate_diff: patch_text.clone(),
                suite_result,
            };
            println!("\n▌ JUDGE phase ({judge_mode}): one call per verdict, no Attacker...");
            let outcome = run_with_spinner("Judge deciding the candidate", async {
                judge_candidate(&cfg, &request).await
            })
            .await;
            let decision = match outcome {
                Ok(response) => {
                    let approved = response.verdict == JudgeVerdict::Approve;
                    println!(
                        "  ⚖ JUDGE: {} (confidence={:.2}) — {}",
                        if approved { "approve" } else { "reject" },
                        response.confidence,
                        response.reason
                    );
                    judge_outcome = Some((approved, response.confidence, response.reason));
                    if approved {
                        GovernorDecision::JudgeApproved
                    } else {
                        GovernorDecision::JudgeRejected
                    }
                }
                Err(JudgeError::Malformed(message)) => {
                    // [VANTA-WHY] The machine was fine and the call was made; what
                    // came back could not be read as a verdict after the one retry.
                    // That is this arm's own outcome — its own exit code (4) and its
                    // own bucket — and deliberately NOT `InfraFailure`, which would
                    // report a broken candidate as a broken server (and be retried
                    // as one). The raw answer is in the forensics file.
                    println!(
                        "  ✖ JUDGE FAILED — the judge did not return a readable verdict after \
                         the one retry (the raw answer is in the forensics file); this is a \
                         recorded judge failure, not an infrastructure fault"
                    );
                    judge_failed_error = Some(message);
                    GovernorDecision::JudgeFailed
                }
                Err(JudgeError::Call(message)) => {
                    // [VANTA-WHY] §14.15 / Finding B (PR #4 review) — a call the
                    // deadline refused is NOT an infrastructure failure. "The machine
                    // broke" and "the run ran out of clock" are opposite claims, and
                    // the judge's native suite (just above) is a sized step that can
                    // consume the remaining clock, so the judge call can be refused
                    // with the same `deadline_exceeded` marker the Creator call
                    // carries. Matched BEFORE the arm that would blame the server, and
                    // it takes the identical exit: `stop_reason=budget_exhausted` with
                    // `NeedsReview` (see the `creator_generate_patch` arm above).
                    // Every OTHER judge call error stays `InfraFailure`.
                    if message.contains(crate::llm::DEADLINE_EXCEEDED_MARKER) {
                        stop_reason = Some(BUDGET_EXHAUSTED);
                        println!(
                            "\n  ⏱ The wall-clock deadline passed before the Judge call — ending \
                             the run as NeedsReview ({BUDGET_EXHAUSTED}), not as an infrastructure \
                             failure."
                        );
                        // [VANTA-WHY] v1-m3.1 item 1 — deadline exit; attach the
                        // candidate patch under the same rule `MaxAttemptsReached`
                        // uses (`final_patched_files = Some(patched_files)`), so the
                        // row is graded instead of censored to `unobservable` (which
                        // would halt the wave).
                        final_patched_files = Some(patched_files);
                        break 'attempts GovernorDecision::NeedsReview;
                    }
                    // [VANTA-WHY] v1-m3.2 item B — an exhausted Judge call that
                    // failed on the MODEL'S OWN OUTPUT is not an infrastructure
                    // failure: the machine was fine and the call was made, what came
                    // back could not be read. The candidate is attached exactly as the
                    // deadline exit above attaches it (`Some(patched_files)`) and the
                    // run ends `NeedsReview` with `stop_reason=model_output_invalid`,
                    // so the row is GRADED and the wave gate passes it — before this,
                    // one unreadable judge answer was an `InfraFailure` that halted the
                    // wave (§8.2). Every OTHER judge call error stays `InfraFailure`.
                    if is_model_output_failure(&message) {
                        stop_reason = Some(MODEL_OUTPUT_INVALID);
                        model_output_error = Some(message.clone());
                        println!(
                            "\n  ⚠ The Judge did not return a readable verdict after its retries \
                             — ending the run as NeedsReview ({MODEL_OUTPUT_INVALID}), not as an \
                             infrastructure failure."
                        );
                        final_patched_files = Some(patched_files);
                        break 'attempts GovernorDecision::NeedsReview;
                    }
                    infra_failure_error = Some(format!("Judge LLM call failed: {message}"));
                    GovernorDecision::InfraFailure
                }
            };
            // [VANTA-WHY] A verdict about the candidate attaches it, exactly as
            // `Rejected` does: `final.patch` is written for every terminal verdict
            // and the oracle that labels the row needs the diff. `JudgeFailed` had
            // no verdict, so it attaches nothing.
            if matches!(
                decision,
                GovernorDecision::JudgeApproved | GovernorDecision::JudgeRejected
            ) {
                final_patched_files = Some(patched_files);
            }
            break 'attempts decision;
        }

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
                original_source: prompt_source_concat.clone(),
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
                // [VANTA-WHY] M3 item 17 — tell the Attacker which probes were
                // quarantined in an earlier round so it attacks somewhere new
                // instead of re-deriving a test the Diagnostician already threw
                // away. Empty on round 1 and whenever nothing was quarantined.
                quarantined_tests: quarantined.clone(),
            };
            let attacker_response = match run_with_spinner("Attacker generating tests", async {
                attacker_generate_tests(&cfg, &attacker_request).await
            })
            .await
            {
                Ok(response) => Some(response),
                // [VANTA-WHY] §14.5 item 2 — a generation the server cut off at
                // `max_tokens` BOTH times it was asked (the request and the brief
                // `answer directly` re-ask) is NOT an infrastructure failure. The
                // run-away is the model spending its whole output cap on reasoning
                // and emitting nothing, so it produces NO TEST from this call —
                // which is a fact about the attack, not about the machine. Treating
                // it as `InfraFailure` (exit 2) turned one run-away into a dead run
                // AND blocked the wave gate (§14.7). With no test from this round
                // `valid_tests` stays empty, so the attempt contributes no generated
                // evidence: it either retries with an empty round (the Governor
                // bounds the attempts, and repetition ends the run without an
                // approval) or, when a shared set survives, runs the shared set
                // alone. Either way this is never a stronger approval than the
                // native suite alone would give (C2 untouched).
                Err(error) if error.to_string().contains(crate::llm::TRUNCATED_OUTPUT_PREFIX) => {
                    tracing::warn!(
                        error = %error,
                        "Attacker generation was CUT OFF at max_tokens twice — this round \
                         contributes NO test"
                    );
                    println!(
                        "  ⚠ Attacker generation was cut off at max_tokens twice ({}): this round \
                         contributes no test",
                        crate::llm::TRUNCATED_TWICE_MARKER
                    );
                    session_log.append(SessionRecord {
                        session_id: session_id.clone(),
                        schema_version: Some(SCHEMA_VERSION),
                        patch_attempt_id: Some(patch_attempt_id.clone()),
                        testing_iteration_id: Some(testing_iteration_id.clone()),
                        test_id: None,
                        phase: "attacker.truncated".to_string(),
                        summary: format!(
                            "site=attacker_generate {} generated_tests=0 — the call was cut off at \
                             max_tokens on both the request and the brief re-ask, so this round \
                             contributes no test",
                            crate::llm::TRUNCATED_TWICE_MARKER
                        ),
                        timestamp: now_iso(),
                        benchmark_instance: None,
                        benchmark_label: None,
                    })?;
                    None
                }
                // [VANTA-WHY] §14.15 — the deadline's refusal, not the server's
                // failure; see the Creator arm above for why the two must not be
                // the same exit code. Placed AFTER the cut-off arm, so a genuinely
                // truncated generation keeps its §14.5 treatment.
                Err(error) if error.to_string().contains(crate::llm::DEADLINE_EXCEEDED_MARKER) => {
                    stop_reason = Some(BUDGET_EXHAUSTED);
                    println!(
                        "\n  ⏱ The wall-clock deadline passed before the Attacker call — ending \
                         the run as NeedsReview ({BUDGET_EXHAUSTED}); this round contributes no \
                         test."
                    );
                    // [VANTA-WHY] v1-m3.1 item 1 — deadline exit; attach this
                    // attempt's applied tree under the same rule `MaxAttemptsReached`
                    // uses (`final_patched_files = Some(patched_files)`). The patch
                    // already applied (the Attacker runs after it), so this IS the
                    // run's most recent produced tree, and the row is graded rather
                    // than censored.
                    final_patched_files = Some(patched_files);
                    break 'attempts GovernorDecision::NeedsReview;
                }
                // [VANTA-WHY] v1-m3.2 item B — an exhausted Attacker call that
                // failed on the MODEL'S OWN OUTPUT (a schema parse error, or an
                // answer cut off twice) is not an infrastructure failure: the machine
                // was fine and the call was made, what came back could not be read.
                // The candidate patch has already applied (the Attacker runs after
                // it), so `patched_files` IS this run's produced tree; attaching it
                // and ending `NeedsReview` with `stop_reason=model_output_invalid`
                // grades the row, where `InfraFailure` censored it to
                // `unobservable` and halted the wave (§8.2). Placed AFTER the
                // cut-off arm above, so a plain truncation still contributes no test.
                // A transport error stays `InfraFailure` below.
                Err(error) if is_model_output_failure(&error.to_string()) => {
                    stop_reason = Some(MODEL_OUTPUT_INVALID);
                    model_output_error = Some(error.to_string());
                    println!(
                        "\n  ⚠ The Attacker's answer could not be read after its retries — ending \
                         the run as NeedsReview ({MODEL_OUTPUT_INVALID}); this round contributes \
                         no test."
                    );
                    final_patched_files = Some(patched_files);
                    break 'attempts GovernorDecision::NeedsReview;
                }
                Err(error) => {
                    infra_failure_error = Some(format!("Attacker LLM call failed: {error}"));
                    break GovernorDecision::InfraFailure;
                }
            };
            // How many tests the CALL returned, which is 0 when it was cut off
            // twice and produced nothing.
            let generated_from_call = attacker_response.as_ref().map_or(0, |r| r.tests.len());
            // [VANTA-WHY] A probe whose body is already in the shared set adds no
            // coverage: it will be BATE-validated and patched-run again, and the
            // comparison it produces is a self-transition. Identity is the
            // content-based `normalized_test_body_hash`, so a rename is still a
            // duplicate. The count of drops is logged per round — silently keeping
            // them would make a stalled Attacker look like a productive one.
            let (generated_tests, dropped_duplicates) = match &attacker_response {
                Some(response) => drop_shared_duplicates(&response.tests, &shared_tests),
                None => (Vec::new(), Vec::new()),
            };
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
                    generated_from_call,
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
            //
            // [VANTA-WHY] M3 item 8 — under B2 (`VANTA_BATE=off`) this audit is
            // SKIPPED entirely, not merely weakened: the arm is defined as
            // adversarial tests admitted "regardless of declared intent", and this
            // is the patch-blind intent-admission gate (`route_intent`'s scrap/
            // demote) that a B2 run must not apply. The Attacker still generated
            // `generated_tests`; with no audit they enter BATE unmodified, so the
            // only filter between the model and the patched run is the mechanical
            // duplicate drop above. Skipping (rather than auditing and ignoring the
            // answer) is what makes "no intent-check records" true and keeps the
            // flag-off log identical to the pre-item-8 one.
            //
            // [VANTA-WHY] §14.5 — the cap the batched call is about to be given,
            // recorded here rather than read back from the audit, because the
            // audit's answer is exactly what a truncated batch fails to produce.
            // The value is the one `attacker_check_intent_batch` will send. Gated
            // on `bate_enabled` too: with no intent audit there is no batched call
            // to cap, and writing the cap anyway would claim a call the run never
            // made.
            if bate_enabled && batch_audit_enabled {
                let cap = crate::agents::attacker::intent_audit_batch_max_tokens(
                    generated_tests.len(),
                );
                batch_audit_max_tokens = Some(batch_audit_max_tokens.map_or(cap, |m| m.max(cap)));
            }
            let (attacker_tests, initial_relabels, initial_intent_checks, batch_audit_transport) =
                if bate_enabled {
                    audit_tests_intent(
                        &cfg,
                        &repo.issue_text,
                        &prompt_source_concat,
                        &generated_tests,
                        grounded_audit_enabled,
                        batch_audit_enabled,
                    )
                    .await
                } else {
                    (generated_tests.clone(), Vec::new(), Vec::new(), None)
                };
            // [VANTA-WHY] §14.5 — what the batched call DID, recorded once per
            // round and only when such a call was made: `Some` here is exactly
            // "a batched call happened", so the flag-off log is unchanged. A
            // cut-off answer stops after one call and falls back to per-test
            // audits; a malformed one spends the whole retry budget and then
            // gives every test the single-call failure shape. The two used to be
            // told apart by counting calls, which the fail-fast makes impossible,
            // so the record says it outright.
            if let Some(transport) = batch_audit_transport {
                session_log.append(SessionRecord {
                    session_id: session_id.clone(),
                    schema_version: Some(SCHEMA_VERSION),
                    patch_attempt_id: Some(patch_attempt_id.clone()),
                    testing_iteration_id: Some(testing_iteration_id.clone()),
                    test_id: None,
                    phase: "intent.audit_batch".to_string(),
                    summary: format!(
                        "source=attacker.generate transport={} tests={}",
                        transport.code(),
                        generated_tests.len()
                    ),
                    timestamp: now_iso(),
                    benchmark_instance: None,
                    benchmark_label: None,
                })?;
            }
            let intent_scrapped = initial_intent_checks.iter().filter(|c| c.scrapped).count();
            println!(
                "  ✔ Attacker: {} tests ({} generated, {} duplicate(s) dropped, {} scrapped by the intent check)",
                attacker_tests.len(),
                generated_from_call,
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
                    generated_from_call,
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
            announce_ungrounded_scraps(&initial_intent_checks);
            let (scrapped, kept_by_issue) = grounding_counts(&initial_intent_checks);
            // [VANTA-WHY] M3 item 1 — from here the ledger is a MEASUREMENT, not an
            // absence: a grounded-audited round ran. Only with `VANTA_GROUNDED_AUDIT=on`
            // is there a field for the auditor to answer, and `grounding_counts` reads
            // nothing else, so the flag is the honest gate for "`Some` rather than
            // `null`".
            //
            // [VANTA-WHY] M3 item 8 — `bate_enabled` is the second half of that
            // gate: under B2 the intent audit (which is the only thing that
            // populates `initial_intent_checks`) never ran, so there is no
            // measurement to record and the ledger must stay `null` even if
            // `VANTA_GROUNDED_AUDIT=on`. Writing `0` would claim an audit that found
            // nothing, which is exactly the absence-vs-zero confusion `ledger_text`
            // exists to prevent.
            if bate_enabled && grounded_audit_enabled {
                *grounded_scrapped_total.get_or_insert(0) += scrapped;
                *grounded_kept_by_issue_total.get_or_insert(0) += kept_by_issue;
            }
            for check in &initial_intent_checks {
                issue_requires_change_by_test
                    .insert(check.test_name.clone(), check.issue_requires_change);
            }
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
        //
        // [VANTA-WHY] M3 item 8 — `admit_all` is `VANTA_BATE=off` (the B2 arm).
        // The bypass is expressed as an argument to `run_bate` rather than a
        // second path around it, so BATE stays the ONE place that decides
        // admission: a pass-through there still produces the `bate.summary` row
        // every existing arm writes (valid=N, scrapped=0), and the flag-off call
        // is byte-identical because `admit_all` is `false` whenever the switch is
        // unset.
        println!("\n▌ BATE: validating tests against baseline...");
        let bate_run = run_with_spinner("BATE validating tests against baseline", async {
            run_bate(
                &cfg,
                &repo.issue_text,
                &bate_baseline_concat(&original_files),
                &prompt_source_concat,
                &attacker_tests,
                &test_cmd,
                grounded_audit_enabled,
                !bate_enabled,
                &mut revise_budget,
            )
            .await
        })
        .await;
        let bate_report = bate_run.report;
        let valid_tests: Vec<GeneratedTest> = bate_run.valid_tests;
        let bate_relabels = bate_run.relabel_events;
        let bate_intent_checks = bate_run.intent_checks;
        let bate_harness_error = bate_run.harness_error;
        let bate_generated_tests = bate_run.generated_tests;
        // [VANTA-WHY] M3 item 10 — one entry per Attacker test this round
        // produced, admitted and discarded alike. Written HERE rather than beside
        // `attempt_<n>_tests.json` further down, because a round whose tests were
        // ALL discarded (or the round a `HarnessFailure` ended) is exactly the
        // round whose sources exist nowhere else, and both of those paths leave
        // before the `attempt_<n>_tests.json` write. Additive logging only: no
        // verdict, label, score, key or existing row changes, and the name does
        // not collide with the `attempt_<n>_{patched,graded}_tree` names
        // `compact.py` / `pool.py` match on.
        let attempt_generated_tests_path =
            debug_dir.join(format!("attempt_{}_generated_tests.json", attempt + 1));
        if let Ok(json) = serde_json::to_string_pretty(&bate_generated_tests) {
            let _ = fs::write(&attempt_generated_tests_path, json);
        }
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
                "valid={} scrapped={} relabels={} discard_rate_pct={:.1} \
                 revise_budget={} revise_calls_used={}",
                bate_report.valid_tests.len(),
                bate_report.scrapped_tests.len(),
                bate_relabels.len(),
                discard_rate * 100.0,
                // [VANTA-WHY] §14.14 — the attempt's Attacker remediation cap and
                // how much of it BATE has spent. A reader can tell a round that
                // never revised (`0`) from one whose tests were scrapped because
                // the budget ran out, without opening the scrap reasons.
                REVISE_BUDGET_PER_ATTEMPT,
                revise_budget.used(),
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
        // [VANTA-WHY] §14.19 — the environment, not the candidate. BATE stopped
        // because no baseline run reached a verdict for the generated test at all
        // (a plugin that fails to load, a `conftest.py` that fails to import, a
        // collection error that names no generated test). Revising cannot help —
        // the Attacker's file is not what broke — and continuing would run the
        // patched suite against the same broken environment and read its silence
        // as evidence. So the attempt ends here as `HarnessFailure`: the existing
        // verdict for "the setup is broken, not the patch", which is terminal and
        // therefore one row rather than a retry.
        if let Some(broken) = bate_harness_error {
            // [VANTA-WHY] Item (a) — the evidence is quoted into the run log, not
            // left only in the session row: otherwise the console says BATE stopped
            // without saying what broke it.
            eprintln!("{}", harness_error_log_line(&broken.test_name, &broken.evidence));
            session_log.append(SessionRecord {
                session_id: session_id.clone(),
                schema_version: Some(SCHEMA_VERSION),
                patch_attempt_id: Some(patch_attempt_id.clone()),
                testing_iteration_id: Some(testing_iteration_id.clone()),
                test_id: Some(fresh_test_id()),
                phase: format!("bate.harness_error[{}]", broken.test_name),
                summary: broken.evidence.clone(),
                timestamp: now_iso(),
                benchmark_instance: None,
                benchmark_label: None,
            })?;
            // [VANTA-WHY] `run.summary` is parsed as space-free `key=value` pairs
            // (`SUMMARY_PAIR = (\S+)`), so a raw test name carrying a space would
            // be silently cut at the first one. The name is sanitized into a
            // single token; the untruncated evidence is the session row above.
            harness_error = Some(broken.test_name.replace(char::is_whitespace, "_"));
            break GovernorDecision::HarnessFailure;
        }
        log_intent_checks(&mut session_log, &session_id, &patch_attempt_id, &testing_iteration_id, "attacker.revise", &bate_intent_checks)?;
        announce_ungrounded_scraps(&bate_intent_checks);
        let (scrapped, kept_by_issue) = grounding_counts(&bate_intent_checks);
        if grounded_audit_enabled {
            *grounded_scrapped_total.get_or_insert(0) += scrapped;
            *grounded_kept_by_issue_total.get_or_insert(0) += kept_by_issue;
        }
        // [VANTA-WHY] BATE's answers come AFTER a possible revision, so they
        // describe the test version that actually runs. They overwrite the
        // generation-time answers for the same name — the newer answer is the
        // one about the test on the tree.
        for check in &bate_intent_checks {
            issue_requires_change_by_test
                .insert(check.test_name.clone(), check.issue_requires_change);
        }
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
        // [VANTA-WHY] M3 item 9 — store the test set THIS attempt ran (the fixed
        // shared set plus the round's new tests), so "the test sets per attempt"
        // is a file per attempt rather than something a reader has to
        // reconstruct from `shared_tests.json` and the fold order.
        let attempt_tests_path =
            debug_dir.join(format!("attempt_{}_tests.json", attempt + 1));
        if let Ok(json) = serde_json::to_string_pretty(&tests_to_run) {
            let _ = fs::write(&attempt_tests_path, json);
        }
        // [VANTA-WHY] M3 item 9 — the attempt's test names, OWNED, taken here while
        // `tests_to_run` is still the live reading. `tests_to_run` is a
        // `Vec<&GeneratedTest>` that borrows `shared_tests`, and the fold-in below
        // (item 17) mutates `shared_tests`; the per-attempt record is pushed long
        // after that mutation, so it must carry owned names rather than re-reading
        // the borrowed list — which would hold the immutable borrow open across the
        // mutation and not compile.
        let attempt_test_names: Vec<String> =
            tests_to_run.iter().map(|t| t.test_name.clone()).collect();

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
        // [VANTA-WHY] M3 item D1 / v1-m3.1 — the suite runs under `native_cap`
        // (declared once for the run, above): the row's own prepare measurement
        // decides it (`max(1200, 10 × measured)`), and a row with no measurement
        // keeps the session floor (`harness::native_session_timeout_floor`).
        // [VANTA-WHY] M3 item S1 — the two sides go through the content-keyed
        // cache. `native_original` is a function of the BUG (same original tree,
        // same command, same cap), so across a Phase-3 arm-and-seed grid it is
        // measured once per bug and served afterwards; `native_patched` is
        // measured once per candidate patch. `origin` names this session, attempt
        // and side so a served result is traceable to the row that produced it.
        // [VANTA-WHY] M3 item AA3 — the two sides run at the same time. They are
        // independent measurements (different trees, different cache keys) whose
        // only shared resource is the sandbox counter, so overlapping them halves
        // the wall time of a cold gate without changing what is measured. A side
        // that is a cache hit returns at its own `plan_native_suite` lookup and
        // spawns nothing, so a warm gate still boots no VM; the cold double-miss
        // is the case that boots two at once and `peak_sandboxes` records 2 (a
        // descriptive count, never a verdict). Everything the two sides produce —
        // the reports, the cache keys and the entries written — is byte-identical
        // to running them in sequence: each is the same call with the same
        // arguments, joined rather than awaited one after the other.
        //
        // [VANTA-WHY] v1-m3.1 — ONE cap for BOTH sides, and it is the RESOLVED
        // `native_cap` rather than the raw `config.native_timeout_secs`: passing
        // the raw option would let `None` fall through to
        // `harness::test_timeout_secs()` (120 s) inside `run_native_suite_cached`
        // while the row recorded 1200 — a cap the suite did not run under. The two
        // sides must also share the value, not merely both be raised: the cap is
        // part of the suite-cache key (`suite_cache::native_key`), so a per-side
        // cap would give the original side a new key on every row and destroy the
        // cross-arm reuse (item S1) it is measured for.
        // [VANTA-WHY] v1-m3.1 (§8.4/§8.5) — the gate is CLAMPED to the run's clock.
        // The attempt top checked the deadline, but this attempt has already spent
        // time (Attacker, BATE, the patched run), so a slow row's gate (a 725 s suite:
        // 7250 s per side) can start with less than 7250 s left and outlive `VANTA_DEADLINE_SECS`
        // on its own. The harness would then hard-kill the whole row at
        // `RUN_TIMEOUT_SECS` before the next attempt top could honour the deadline, and a
        // hard-killed row writes no manifest entry — invisible to the wave-1 gate. BOTH
        // sides are therefore given `min(native_cap, remaining)` from the ONE value
        // below, so the two sides stay the same experiment even when clamped; the clamp
        // can only LOWER the cap.
        //
        // [VANTA-WHY] v1-m3.1 (§8.5) — with nothing left there is no run to make: a
        // suite under a zero-second cap would only manufacture a kill. The attempt takes
        // the SAME deadline exit the top of the loop takes (`stop_reason =
        // BUDGET_EXHAUSTED`, `NeedsReview`), adding no new outcome. `native_gate_cap`
        // returns `None` for that case, and the full cap when no deadline is armed (the
        // behaviour of every run before §14.15).
        //
        // [VANTA-WHY] v1-m3.1 — the clamp is per-call, and the cap is part of the
        // suite-cache key (`suite_cache::native_key`), so a clamped ORIGINAL side misses
        // the cross-arm entry (item S1) and re-measures. That is confined to the ONE
        // attempt that runs near the deadline — a warm, unclamped gate still hits — and a
        // clamped side that is killed writes nothing (U1), so no tree is mislabelled by a
        // shorter cap. `gate_cap` is what the suite actually ran under, so the row
        // records it rather than the resolved `native_cap`.
        let gate_cap = match native_gate_cap(native_cap, crate::llm::remaining_secs()) {
            Some(cap) => cap,
            None => {
                stop_reason = Some(BUDGET_EXHAUSTED);
                println!(
                    "\n  ⏱ Wall-clock deadline reached before the native gate — skipping it and \
                     ending the run as NeedsReview ({BUDGET_EXHAUSTED}); the candidate has been \
                     neither approved nor rejected."
                );
                // [VANTA-WHY] v1-m3.1 item 1 — deadline exit; attach this attempt's
                // applied tree under the same rule `MaxAttemptsReached` uses
                // (`final_patched_files = Some(patched_files)`). The gate runs after
                // the apply, so this is the run's most recent produced tree and the
                // row is graded rather than censored to `unobservable`.
                final_patched_files = Some(patched_files);
                break 'attempts GovernorDecision::NeedsReview;
            }
        };
        let gate_clamped = gate_cap < native_cap;
        // [VANTA-WHY] v1-m3.1 F3 — the last gate's clamp state rides out to
        // `run.summary` (`native_gate_clamped`), so a reader can tell a clamped
        // gate from an unclamped one without comparing caps across rows.
        native_gate_clamped = gate_clamped;
        // [VANTA-WHY] v1-m3.1 F5 — and the cap the gate ACTUALLY ran under
        // (`native_gate_effective_cap_secs`), which the clamp may have lowered
        // below the resolved `native_cap`. Last gate wins, like the per-side secs;
        // it is recorded with them by `native_gate.record` right after the two
        // sides join, below.
        let original_origin = format!("{}/a{}/original", session_id.as_str(), attempt + 1);
        let patched_origin = format!("{}/a{}/patched", session_id.as_str(), attempt + 1);
        let ((native_original, original_cache), (native_patched, patched_cache)) = tokio::join!(
            harness::run_native_suite_cached(
                &original_concat,
                &test_cmd,
                Some(gate_cap),
                actual_backend,
                &original_origin,
            ),
            harness::run_native_suite_cached(
                &native_patched_concat,
                &test_cmd,
                Some(gate_cap),
                actual_backend,
                &patched_origin,
            ),
        );
        // [VANTA-WHY] F2/F3 (PR #4 review) — fold this gate's per-side facts in
        // NOW, before the `native_setup_failure` guard below can exit as
        // `InfraFailure`: a gate whose ORIGINAL side completed but whose PATCHED
        // side never set up still obtained an original-side measurement, and the
        // guard must not discard it. Recording here (rather than after the guard)
        // is also what keeps the last-gate seconds, kill flags and effective cap on
        // the SAME gate.
        native_gate.record(gate_cap, &native_original, &native_patched);
        for (side, note) in [("original", &original_cache), ("patched", &patched_cache)] {
            if note.state == "hit" {
                native_cache_hits += 1;
            }
            native_cache_records.push(serde_json::json!({
                "attempt": attempt + 1,
                "side": side,
                "state": note.state.as_str(),
                "key": note.key.as_str(),
                "source": note.source.as_str(),
                "would_hit": note.would_hit,
            }));
        }
        // [VANTA-WHY] M3 item W — a native gate that measured NOTHING is not a
        // clean gate. `native_regressions` over an empty original list is
        // vacuously empty and `native_suite_collapsed` needs a non-empty
        // original, so a suite that collected nothing on BOTH sides reads as "0
        // regressions" and the attempt is approved on no evidence at all.
        // Measured on pandas/1's in-session run (session c9289597: `Native:
        // original ran=true (0 tests), patched ran=true (0 tests), 0
        // regression(s)`) while the SAME tree collected 88643 tests through the
        // prepare path. When the row's own prepare recorded tests for this bug,
        // a native run in which the ORIGINAL side collected nothing means the
        // base tree could not be graded at all: an infrastructure fact, never a
        // pass, and it ends the run as `InfraFailure` (exit 2) so the row is
        // `infra_failure`.
        //
        // [VANTA-WHY] v1-m3.1 (§8.1) — the guard is keyed on the ORIGINAL side
        // ALONE. The original side running is what makes the patched side's
        // emptiness attributable to the PATCH: a patched side that collected
        // nothing while the original ran (a collection error, OR a cap kill) is a
        // FAILED ATTEMPT, never `InfraFailure`. It reaches the Governor on its
        // own, because every original test is then absent from the patched report
        // (`native_regressions`, and `native_suite_collapsed`). Reading it as
        // vacuous is what labelled an in-batch session `infra_failure`
        // and dropped the bug out of the attempt loop, though its patch alone
        // broke collection while the original ran its full suite — the pre-registration
        // counts a patch-caused import error as a `fail`, not censored.
        let native_expected = config.native_expected_tests.unwrap_or(0);
        let native_vacuous = harness::native_gate_vacuous(native_expected, &native_original);
        // [VANTA-WHY] Finding A (PR #4 review) — a side the harness could not even
        // SET UP or transport measured nothing about the tree, so its emptiness must
        // not be read as regressions against the candidate (nor, for the ORIGINAL
        // side, as the deadline clamp's exit). Checked BEFORE `native_regressions`
        // (below) and before the timeout flag, so a setup failure never counts as a
        // candidate regression and never sets `native_timeout`; the run ends as
        // `InfraFailure` in BOTH repair and review modes. `native_vacuous_seen` stays
        // FALSE — that flag documents "the BASE tree could not be graded", which this
        // is not.
        if native_setup_failure(
            native_original.setup_failed,
            native_patched.setup_failed,
            native_vacuous,
        ) {
            let reason = harness::native_setup_failure_reason(&native_original, &native_patched);
            println!("  ✗ NATIVE SUITE SETUP FAILED: {reason}");
            infra_failure_error = Some(reason);
            break 'attempts GovernorDecision::InfraFailure;
        }
        // [VANTA-WHY] v1-m3.1 F3 — `native_vacuous_seen` documents an
        // INFRASTRUCTURE fault (the base tree could not be graded), so it is set
        // where that exit is actually taken, below — NOT here. A gate made vacuous
        // by the DEADLINE clamp takes the `budget_exhausted`/`NeedsReview` exit a
        // few lines down, and setting the flag here first would make a clamped row
        // read as `native_vacuous=true`, i.e. a machine fault it is not. The clamp
        // is recorded by `native_gate_clamped` instead.
        // [VANTA-WHY] M3 item D1 — record that the cause was a KILL, not a tree
        // that collected nothing. Only meaningful when the gate is actually
        // vacuous; a timed-out side on a non-vacuous gate would mean the suite
        // still measured something, which the flag does not claim.
        //
        // [VANTA-WHY] Finding C (PR #4 review) — a kill by the DEADLINE CLAMP is
        // excluded (`!gate_clamped`): the clamp cut the cap below the CONFIGURED
        // `native_cap`, so the kill is the clock's doing and must not be reported
        // as a configured-cap timeout. The clamp is recorded by
        // `native_gate_clamped` instead, and the original side it killed takes the
        // `budget_exhausted`/`NeedsReview` exit below — not the timeout label.
        native_timeout_seen |= native_timeout_flag(
            native_vacuous,
            gate_clamped,
            native_original.timed_out,
            native_patched.timed_out,
        );
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
            // [VANTA-WHY] M3 item E2 — one nodeid index, not a `.any()` scan per
            // original test, and a `HashSet` of the names already recorded rather
            // than a `.any()` over `native_regressions` per test. The pair ran
            // per attempt over ~88k nodeids on pandas.
            let patched_index = harness::native_results_index(&native_patched.tests);
            let mut named: HashSet<String> = native_regressions.iter().cloned().collect();
            for (name, _) in &native_original.tests {
                if !patched_index.contains_key(name.as_str()) && named.insert(name.clone()) {
                    native_regressions.push(name.clone());
                }
            }
        }
        let native_regression = !native_regressions.is_empty();
        println!(
            "  Native: original ran={} ({} tests), patched ran={} ({} tests), {} regression(s){}{}",
            native_original.ran,
            native_original.tests.len(),
            native_patched.ran,
            native_patched.tests.len(),
            native_regressions.len(),
            if native_collapsed { " [SUITE COLLAPSED]" } else { "" },
            if native_vacuous { " [NATIVE VACUOUS]" } else { "" }
        );
        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: Some(patch_attempt_id.clone()),
            testing_iteration_id: Some(testing_iteration_id.clone()),
            test_id: None,
            phase: "native_suite.summary".to_string(),
            // [VANTA-WHY] M3 item W follow-up — the collection census is
            // appended so a vacuous gate's session record names the module that
            // broke collection, for BOTH sides, instead of only reporting that
            // 0 tests were collected. See `harness::native_collection_summary`.
            //
            // [VANTA-WHY] M3 item D1 — `native_timeout_secs` is the cap this
            // suite actually ran under, and `original_timed_out`/`patched_timed_out`
            // say whether a side was killed by it. Without them a session record of
            // a killed run is indistinguishable from one whose tree failed to
            // import — the exact confusion that made pandas/1's night3 row read as
            // "collected nothing" when its suite simply needed ~487 s.
            //
            // [VANTA-WHY] v1-m3.1 (§8.4) — the value is `gate_cap`, which the
            // deadline clamp may have LOWERED below the resolved `native_cap`. The
            // field claims the cap the suite ran under, so a clamped attempt records
            // the shorter number and a reader sees the clamp by comparing it with
            // `run.summary`'s resolved cap.
            summary: format!(
                "ran_original={} ran_patched={} original_tests={} patched_tests={} regressions={} \
                 native_vacuous={} native_expected_tests={} native_timeout_secs={} \
                 original_timed_out={} patched_timed_out={} \
                 native_original_cache={} native_patched_cache={} {}",
                native_original.ran,
                native_patched.ran,
                native_original.tests.len(),
                native_patched.tests.len(),
                native_regressions.len(),
                native_vacuous,
                native_expected,
                gate_cap,
                native_original.timed_out,
                native_patched.timed_out,
                // [VANTA-WHY] M3 item S1 — each side's cache note
                // (`state|key|source|would_hit`). Appended after the D1 fields so
                // an older row's reader sees the same tokens in the same order.
                original_cache.token(),
                patched_cache.token(),
                harness::native_collection_summary(&native_original, &native_patched)
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
        // [VANTA-WHY] M3 item W — the vacuous gate ends the RUN, not just the
        // attempt: the native check is the one signal that does not depend on
        // what the Attacker sampled, so a run that could not perform it has no
        // evidence that survives this attempt's own tests. Reusing the existing
        // `InfraFailure` machinery (run.infra_failure row, exit code 2, row
        // verdict `InfraFailure` → label `infra_failure`) is what makes the row
        // unapprovable on this evidence rather than a silent false accept.
        //
        // [VANTA-WHY] v1-m3.1 (§8.1/§8.5) — the branch is reachable only on §8.1
        // row 1 (`native_vacuous` is now "the ORIGINAL side collected nothing"),
        // so it fires when the BASE tree could not be graded, whatever the patched
        // side did. A patched side that collected nothing while the original ran
        // does NOT reach here: it stays in the attempt loop as a failed attempt,
        // which in review mode REJECTS the candidate (`attempt_verdict` sees the
        // native regression) instead of ending the run as an infrastructure fault.
        //
        // [VANTA-WHY] v1-m3.1 (§8.5) — an ORIGINAL side cut by the DEADLINE clamp
        // collected nothing because the clock ran out, not because the tree could
        // not be graded: that is `budget_exhausted` (the same exit the attempt top
        // takes), NEVER `InfraFailure`. `native_vacuous_exit` draws the line — a
        // clamp that did not kill the original side, and a run under the full cap,
        // both stay `InfraFailure`.
        if native_vacuous {
            if !native_vacuous_is_infrastructure(gate_clamped, native_original.timed_out) {
                stop_reason = Some(BUDGET_EXHAUSTED);
                println!(
                    "\n  ⏱ the native gate's original side was cut by the deadline clamp \
                     (resolved cap {native_cap}s, clamped to {gate_cap}s) — ending the run as \
                     NeedsReview ({BUDGET_EXHAUSTED}), not an infrastructure fault; the candidate \
                     has been neither approved nor rejected."
                );
                // [VANTA-WHY] v1-m3.1 item 1 — deadline exit; attach this attempt's
                // applied tree under the same rule `MaxAttemptsReached` uses
                // (`final_patched_files = Some(patched_files)`). The gate ran after
                // the apply, so the candidate tree is the run's most recent produced
                // one — attaching it makes the row gradeable instead of censored.
                final_patched_files = Some(patched_files);
                break 'attempts GovernorDecision::NeedsReview;
            }
            let reason = harness::native_vacuous_reason(
                native_expected,
                gate_cap,
                &native_original,
                &native_patched,
            );
            // [VANTA-WHY] v1-m3.1 F3 — this is the INFRASTRUCTURE exit (the base
            // tree could not be graded), so it is here, and only here, that the
            // run-level vacuity flag is set. The clamp-caused exit above returns
            // before this line, so it records `native_vacuous=false` and
            // `native_gate_clamped=true`.
            native_vacuous_seen = true;
            println!("  ✗ NATIVE VACUOUS (original side collected nothing): {reason}");
            infra_failure_error = Some(reason);
            break 'attempts GovernorDecision::InfraFailure;
        }
        // [VANTA-WHY] M3 item E2 — one nodeid index and one regression-membership
        // set, both built once, instead of a pair of linear scans per original
        // test (~88k of them on pandas, per attempt).
        let patched_index = harness::native_results_index(&native_patched.tests);
        let regression_set: HashSet<&str> = native_regressions.iter().map(|n| n.as_str()).collect();
        for (nodeid, original_result) in &native_original.tests {
            let patched_result = patched_index.get(nodeid.as_str()).copied();
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
                    regression_set.contains(nodeid.as_str())
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

            // [VANTA-WHY] M3 item 17 — the quarantine request for THIS test, if
            // any. It is captured in its own slot because `diag` is consumed below
            // (its payload moves into the per-test debug file) and the decision has
            // to outlive it.
            let mut quarantine: Option<QuarantinedProbe> = None;
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
                    &prompt_source_concat,
                    &test_cmd,
                    // [VANTA-WHY] Every test in `tests_to_run` has a BATE baseline:
                    // `valid_tests` were validated this attempt and `shared_tests`
                    // were validated when they were first generated (replayed from
                    // disk). The Diagnostician must not demote a validated test's
                    // failure to a test defect — for a BugReproduction test,
                    // failing identically on the original and the patched tree is
                    // the evidence that the patch did not fix the bug.
                    true,
                    &mut revise_budget,
                )
                .await;
                println!(
                    "    {} → {:?} ({:?})",
                    test.test_name, result, diag.failure_class
                );
                // [VANTA-WHY] M3 item 17 — the Diagnostician asks for a quarantine
                // (a `TestDefect`, or a persistent `Unknown`) through the SAME
                // payload `action` the Unknown arm has always written, and the one
                // detector `is_quarantined` is what both classes pass through — so
                // there is a single removal path rather than a second one for
                // TestDefect. Read here, before the two fields below are consumed,
                // because it borrows the whole `diag`.
                let quarantined_now = is_quarantined(&diag);
                let quarantine_reason = diag.reason.clone();
                let quarantine_class = diag.failure_class.clone().unwrap_or(FailureClass::Unknown);
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
                if quarantined_now {
                    quarantine = Some(QuarantinedProbe {
                        test_name: test.test_name.clone(),
                        test_source: test.test_source.clone(),
                        reason: quarantine_reason,
                    });
                }
                Some(quarantine_class)
            };

            // [VANTA-WHY] M3 item 17 — a quarantined test leaves the shared set and
            // this attempt's evidence. It is not a finding about the candidate:
            // folding it in would make every later round re-run and re-diagnose the
            // same broken probe until the budget ran out (the 16 stalled
            // `MaxAttemptsReached` rows), and leaving it in the evidence would let a
            // probe the Diagnostician judged DEFECTIVE drive the Governor or the
            // approval invariant. The name + reason go to the Attacker and onto the
            // session log; the `continue` skips the observation and evidence pushes
            // below.
            if let Some(probe) = quarantine {
                println!(
                    "    ⛔ QUARANTINE: {} removed from the shared set — {}",
                    probe.test_name, probe.reason
                );
                session_log.append(SessionRecord {
                    session_id: session_id.clone(),
                    schema_version: Some(SCHEMA_VERSION),
                    patch_attempt_id: Some(patch_attempt_id.clone()),
                    testing_iteration_id: Some(testing_iteration_id.clone()),
                    test_id: None,
                    phase: format!("quarantine[{}]", probe.test_name),
                    summary: format!("reason={}", probe.reason),
                    timestamp: now_iso(),
                    benchmark_instance: None,
                    benchmark_label: None,
                })?;
                if !quarantined.iter().any(|q| q.test_name == probe.test_name) {
                    quarantined.push(probe);
                }
                continue;
            }

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
            // [VANTA-WHY] M3 item 17 — a quarantined test is never folded into the
            // shared set, so it is not re-counted next round. The `retain` below
            // then removes any test quarantined THIS attempt that was already
            // carried in the shared set (the `Unknown` class can quarantine a test
            // that was admitted in an earlier round), so the removal holds for both
            // classes and both directions.
            for t in &valid_tests {
                if !shared_names.contains(&t.test_name)
                    && !quarantined.iter().any(|q| q.test_name == t.test_name)
                {
                    shared_tests.push(t.clone());
                }
            }
            shared_tests.retain(|t| !quarantined.iter().any(|q| q.test_name == t.test_name));
        }
        // [VANTA-WHY] Persist the shared set to disk so a resumed run (or M5
        // re-analysis) replays the IDENTICAL tests. Spec §10 item 2.
        if let Ok(json) = serde_json::to_string_pretty(&shared_tests) {
            let _ = fs::write(&shared_tests_path, json);
        }

        // [VANTA-WHY] M3 item 17 — a quarantined test's failure is a fact about
        // the TEST, not the candidate, so it must not make the round read as "not
        // all passed". Left in, it would spend another attack round on a probe the
        // Diagnostician already discarded and stall the run at
        // `MaxAttemptsReached` — the exact stall item 17 exists to remove. No test
        // is dropped from `patched_results` itself (the run is already over); the
        // filter is applied to the all-passed reading only.
        let all_passed = patched_results
            .iter()
            .filter(|(name, _)| !quarantined.iter().any(|q| &q.test_name == *name))
            .all(|(_, details)| details.result == TestResult::Passed);
        // [VANTA-WHY] The Governor needs three-way evidence, not a bool. In review
        // mode a Logic failure is a REJECTION; an Infrastructure-only round is
        // not a finding at all and must never be reported as one.
        let mut verdict = attempt_verdict(
            all_passed,
            &patched_classes,
            native_regression,
            new_static_in_touched_files,
            governor_mode,
        );

        // ---- 9. Corroboration overlay (§14.2, `VANTA_CORROBORATE=on`) ----
        // [VANTA-WHY] This is the whole point of §14: a rejection is a claim
        // about the PATCH, so it must survive the possibility that the failing
        // test is wrong. `attempt_verdict` above is v0.2's rule and is left
        // untouched; with the flag off this block does not run and the verdict is
        // byte-for-byte what it always was. With it on, only a review-mode
        // rejection is re-examined, and it is downgraded to `WeaklyRejected`
        // (never a stop, and `NeedsReview` at budget exhaustion) when no rule
        // covers the failure. Rules (a) and (e) are deterministic — a native
        // regression or a static error cannot be an LLM's bad test — so they
        // always stand.
        if corroborate_enabled
            && governor_mode == GovernorMode::Review
            && verdict == AttemptVerdict::Rejected
        {
            let failures = failure_evidence_for(
                &attempt_evidence,
                attempt + 1,
                &issue_requires_change_by_test,
            );
            let assessment = corroborate::assess(&CorroborationInput {
                failures: &failures,
                native_regressions: &native_regressions,
                new_static_in_touched_files: &blocking_static,
                issue_text: &repo.issue_text,
            });
            match assessment.rule {
                Some(rule) => {
                    println!(
                        "  ⚖ CORROBORATED by rule ({}) — {}",
                        rule.code(),
                        assessment.evidence.join(", ")
                    );
                    deciding_rule = Some(rule.code().to_string());
                    deciding_evidence = assessment.evidence.clone();
                    // The rejection is corroborated, so no weak-failure list
                    // belongs in the summary: the reason is the rule, not a
                    // "human should look".
                    needs_review_weak.clear();
                }
                // No rule fired, and there IS at least one uncorroborated Logic
                // failure: the only evidence against the candidate might be a
                // wrong test, so this is not a rejection.
                None if !assessment.weak.is_empty() => {
                    println!(
                        "  ⚖ UNCORROBORATED failure(s) — not a rejection on this evidence: {}",
                        assessment.weak.join(", ")
                    );
                    verdict = AttemptVerdict::WeaklyRejected;
                    // Accumulate across rounds: a run that ends `NeedsReview`
                    // should name EVERY weak failure it saw, not just the last
                    // round's, so the human knows the full set to look at.
                    for weak in &assessment.weak {
                        if !needs_review_weak.contains(weak) {
                            needs_review_weak.push(weak.clone());
                        }
                    }
                }
                // Fail CLOSED: a rejection whose evidence this module could not
                // enumerate stays a rejection rather than becoming an approval.
                None => {}
            }
        }

        let governor_decision = governor.decide(governor_mode, verdict, true);
        // [VANTA-WHY] M3 item 9 — the Governor's own reading of THIS attempt, taken
        // here because the very next binding MOVES `governor_decision` into
        // `decision`, which the fixed-round interception then rewrites to
        // `Continue`. The per-attempt record has to say what the attempt did, not
        // what the loop did next, so the reading is captured before either happens.
        let attempt_passed = governor_decision == GovernorDecision::AllTestsPassed;
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
        // [VANTA-WHY] M3 item 9 — fixed-round repair. In REPAIR mode the ONLY
        // decision that ends the loop early is `AllTestsPassed`; a failing
        // attempt already spends its budget and continues. A fixed-rounds run
        // must therefore run all K attempts, so a pass before the last one is
        // recorded as evidence and the loop continues. `advance_attempt` moves the
        // cursor exactly as `decide`'s continue path would — `AllTestsPassed`
        // deliberately does not advance it — so the next round's transition keys
        // stay contiguous. Nothing is suppressed on the last attempt, so the final
        // verdict is the one attempt K produced.
        let fixed_round_pass =
            fixed_round_continues(fixed_rounds, &decision, attempt, loop_attempts);
        let decision = if fixed_round_pass {
            println!(
                "\n  ↻ FIXED ROUNDS: attempt {}/{} passed — running attempt {} anyway; the \
                 verdict is taken from attempt {}.",
                attempt + 1,
                loop_attempts,
                attempt + 2,
                loop_attempts
            );
            governor.advance_attempt();
            GovernorDecision::Continue
        } else {
            decision
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
            // [VANTA-WHY] §14.3. A `NeedsReview` must name the evidence, or a
            // human has nothing to look at. The failing tests are printed for
            // context; `needs_review_weak` (recorded in `run.summary`) says which
            // of them no §14.2 rule covered.
            if decision == GovernorDecision::NeedsReview {
                println!(
                    "  ⚠ NEEDS REVIEW — no corroboration rule covers the failure(s) after {} \
                     round(s): {}",
                    attempt + 1,
                    if needs_review_weak.is_empty() {
                        "inspect the attempt's diagnostic payloads".to_string()
                    } else {
                        needs_review_weak.join(", ")
                    }
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
                "attempt={} decision={:?} valid_tests={} shared_tests={} all_passed={} \
                 revise_budget={} revise_calls_used={}{}",
                attempt + 1,
                decision,
                valid_tests.len(),
                shared_tests.len(),
                all_passed,
                // [VANTA-WHY] §14.14 — the attempt's whole Attacker remediation
                // spend (BATE's revisions + the Diagnostician's simplifications)
                // against the cap. `bate.summary` reports the same counter at the
                // point BATE finished; this is the end-of-attempt reading the run
                // total is summed from.
                REVISE_BUDGET_PER_ATTEMPT,
                revise_budget.used(),
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
        // [VANTA-WHY] §14.14 — fold this attempt's spend into the run's total while
        // the reading is in hand; the budget is reset at the top of the next
        // attempt, so this is the only place it is observable.
        revise_calls_used_total += revise_budget.used();

        // [VANTA-WHY] M3 item 9 — the per-attempt index of this run, and the two
        // fold-ins the M3.5 plan asked for (the attempt's feedback text and the
        // hash of the Creator prompt). Recorded HERE, before the decision match,
        // because every terminal arm below `break`s out of the loop: a record
        // pushed after it would be missing for exactly the LAST attempt, which is
        // the one the final verdict is taken from — so "with K = 4 the run stores
        // 4 patches and the per-attempt test sets" would be false for every run
        // that ended when it was supposed to.
        //
        // `passed` is the Governor's OWN reading of this attempt, taken from
        // `governor_decision` and therefore BEFORE the fixed-round interception
        // above rewrote an early pass into a `Continue`: the record has to say
        // what the attempt did, not what the loop did next. `feedback` is the text
        // this attempt RECEIVED (the text that drove this patch), not the one it
        // produced: `prior_feedback` is only reassigned in the `Continue` arm
        // below, so here it still holds the previous attempt's hand-over, which is
        // what pairs each patch with the thing that produced it. It is `null` on
        // attempt 1 and on every `--patch` review, where no Creator ran.
        //
        // The artifacts themselves also live in the session's debug dir, under
        // NEW names (`attempt_<n>_creator_patch.txt`, `attempt_<n>_tests.json`,
        // `attempt_<n>_feedback.txt`) that deliberately do not collide with the
        // pinned `attempt_<n>_{patched,graded}_tree` names `compact.py` / `pool.py`
        // match on; this list is what makes them reachable without a reader
        // globbing a directory whose naming is otherwise a convention.
        {
            let received_feedback = prior_feedback.clone();
            if let Some(fb) = &received_feedback {
                let feedback_path =
                    debug_dir.join(format!("attempt_{}_feedback.txt", attempt + 1));
                let _ = fs::write(&feedback_path, fb);
            }
            // [VANTA-WHY] F2 — the entry was opened right after the Creator
            // answered (so an attempt whose patch never applied still owns one);
            // here it is completed with what only the graded attempt can know.
            // `patch`, `feedback`, `creator_prompt_hash` and `unchanged` were
            // captured then and are deliberately NOT rewritten: `prior_feedback`
            // is only reassigned on paths that `break`/`continue` before reaching
            // here, and the comparison basis (`previous_patch_text`) was already
            // advanced there too. `unchanged` is `fixed_round_patch_is_unchanged`
            // of the PREVIOUS Creator response, applied or not.
            if let Some(record) = fixed_round_records.get_mut(fixed_round_record_index) {
                record["passed"] = serde_json::json!(attempt_passed);
                record["tests"] = serde_json::json!(attempt_test_names);
            }
        }

        match decision {
            GovernorDecision::AllTestsPassed => {
                // [VANTA-WHY] M3 item 1 — the approval invariant. §1.4's defect
                // was 12/12 reviews approving a no-op: the grounded audit had
                // scrapped every reproduction probe, the preservation probes all
                // passed, and `AllTestsPassed` was read as "the candidate is
                // good" when all it said was "nothing that survived disagrees".
                // An approval must therefore rest on at least one admitted test
                // that EXPECTS THE ORIGINAL TO FAIL — a `BugReproduction` or
                // `FeatureAcceptance` probe that actually ran on the candidate —
                // and the answer is `NeedsReview`, never `Approve`, when none
                // did. `approval_invariant_status` is the one place that decides
                // this, and `run.summary` reads the same function, so the verdict
                // and the row cannot disagree. The gate is deliberately NOT
                // conditioned on any §14 flag: turning the audits off must not be
                // a way to buy back an approval.
                let (applied, reason) = approval_invariant_status(
                    governor_mode,
                    attacker_enabled,
                    bate_enabled,
                    resolved_arm.judge.is_some(),
                    &final_evidence,
                );
                if applied && reason != INVARIANT_SATISFIED {
                    // [VANTA-WHY] M3 item 1 — the two withheld-approval reasons are
                    // two different diagnoses, so they get two different sentences
                    // and the recorded token is printed either way.
                    let why = match reason {
                        NO_ADMITTED_TESTS => {
                            "no test was admitted, so an approval would rest on nothing at all"
                        }
                        _ => {
                            "no admitted test that ran against the candidate expects the \
                             original tree to fail (no BugReproduction or FeatureAcceptance probe \
                             survived and executed), so every surviving test passing only says \
                             the candidate broke nothing"
                        }
                    };
                    println!("  ⚠ NEEDS REVIEW — {why}: {reason}");
                    // The candidate is attached exactly as for the corroboration
                    // `NeedsReview` above: the human needs the patch, and
                    // `final.patch` is written for every terminal verdict.
                    final_patched_files = Some(patched_files);
                    break GovernorDecision::NeedsReview;
                }
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
            GovernorDecision::NeedsReview => {
                // [VANTA-WHY] §14.3. Not an approval and not a rejection: the
                // candidate spent the whole attack depth on failures no §14.2
                // rule corroborates, so a human must judge it. The candidate is
                // attached exactly as for `Rejected` — the reader needs the
                // patch, and `final.patch` is written for every terminal verdict
                // — but it never reaches the Gatekeeper, because "needs review"
                // is not the Gatekeeper's `Approved`.
                final_patched_files = Some(patched_files);
                break decision;
            }
            GovernorDecision::JudgeApproved | GovernorDecision::JudgeRejected => {
                // [VANTA-WHY] M3 item 7. A judge arm breaks out of the loop at
                // step 2b, before the machinery this match belongs to, so this arm
                // is the record of what a judge verdict MEANS rather than a path
                // today's arms take. It is written to be correct on its own — a
                // verdict about the candidate attaches it, exactly as `Rejected`
                // and `NeedsReview` do, because `final.patch` is what the oracle
                // labels the row from — so a future path that routes a judge
                // verdict through here cannot silently lose the candidate.
                final_patched_files = Some(patched_files);
                break decision;
            }
            GovernorDecision::JudgeFailed => {
                // [VANTA-WHY] M3 item 7. The judge answered and its answer could
                // not be read as a verdict after the one retry: this arm's own
                // outcome (exit code 4, its own `judge_failed` bucket), never
                // folded into `NeedsReview`, `Rejected` or `InfraFailure`. No
                // verdict was reached, so nothing is attached; the raw answer is in
                // the forensics file.
                break decision;
            }
            GovernorDecision::PatchApplicationFailed => {
                // [VANTA-WHY] M3 item E — a terminal verdict, so it leaves a
                // `final.patch` like every other: the run's last PRODUCED tree, or
                // the original tree when nothing ever applied (an empty patch, a
                // measured failure). This arm is the review-mode path — the
                // candidate was written outside the run and would not apply — and
                // an empty patch graded against the tests says "this candidate did
                // not fix the bug", which is the honest label, not `unobservable`.
                final_patched_files =
                    Some(final_patch_tree(last_applied_patched_files.as_ref(), &original_files));
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
                } else if fixed_round_pass {
                    // [VANTA-WHY] M3 item 9 — a PASSING attempt under
                    // `--fixed-rounds`. The loop continues by design here, so the
                    // Creator has to be given the suite result it just produced;
                    // the ordinary branch below would tell it "the previous patch
                    // failed verification" and then name no failure at all, which
                    // is both untrue and useless as feedback. The spec's own
                    // instruction is different in kind — "review and improve the
                    // patch if needed, or return it unchanged" — so it is said in
                    // its own words, by `build_fixed_round_pass_feedback`.
                    let passed_names: Vec<String> =
                        final_evidence.iter().map(|e| e.test_name.clone()).collect();
                    prior_feedback = Some(build_fixed_round_pass_feedback(
                        &passed_names,
                        &patch_text,
                        attempt + 1,
                        loop_attempts,
                    ));
                    println!(
                        "  ↻ FIXED ROUNDS: attempt {}/{} passed — asking the Creator to review \
                         and improve, or return the patch unchanged ({} test(s))",
                        attempt + 1,
                        loop_attempts,
                        passed_names.len()
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

    // [VANTA-WHY] Item P — the seed's lifetime ends with the loop: no sandbox is
    // spawned after this point, so the stage can go and the harness can stop
    // pointing at it. Clearing the root first keeps a later run in the same
    // process from cloning a stage this one owns (and is about to remove). The
    // mechanism the first seed used is read BEFORE the clear, because
    // `set_seed_root` deliberately resets it alongside the root.
    let seed_method_record = harness::seed_method().map(|m| m.as_str()).unwrap_or("unset");
    harness::set_seed_root(None);
    drop(seed_tree);

    // [VANTA-WHY] Item P, point 2 — the import guard's outcome as its own phase,
    // the way `run.infra_failure` and `run.judge` record their own. The full line
    // names the package AND the path it imported from, which is the evidence that
    // an in-session suite ran against the tree under test rather than
    // site-packages; `run.summary` can carry only the space-free token.
    if let Some(line) = &import_guard_line {
        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: attempt_patch_ids.last().cloned(),
            testing_iteration_id: None,
            test_id: None,
            phase: "run.import_guard".to_string(),
            summary: line.chars().take(1000).collect::<String>(),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
    }

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

    // [VANTA-WHY] §14.15 — the run's own deadline gets the same treatment as an
    // infrastructure failure: its own phase row, so the reason is a fact in the
    // session log rather than something inferred from `final_decision=NeedsReview`
    // plus an empty `needs_review_weak`. An ordinary `NeedsReview` names the weak
    // failures that triggered it; this one has none to name, because nothing about
    // the candidate was decided — which is exactly why it must say so itself.
    if stop_reason == Some(BUDGET_EXHAUSTED) {
        session_log.append(SessionRecord {
            session_id: session_id.clone(),
            schema_version: Some(SCHEMA_VERSION),
            patch_attempt_id: attempt_patch_ids.last().cloned(),
            testing_iteration_id: None,
            test_id: None,
            phase: "run.budget_exhausted".to_string(),
            summary: format!(
                "final_decision=NeedsReview stop_reason={BUDGET_EXHAUSTED} \
                 deadline_secs={} — the run's own wall clock ended it before it could decide \
                 anything about the candidate; this is NOT the harness killing the process, which \
                 is reported as `timed_out` and usually leaves no run.summary at all",
                deadline_secs
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unset".to_string())
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
    }

    // [VANTA-WHY] v1-m3.2 item B — an unreadable model answer gets the same
    // treatment as the deadline: its own phase row, so `stop_reason =
    // model_output_invalid` comes with the text that earned it and is a fact in
    // the log rather than something inferred from `final_decision=NeedsReview`
    // plus an empty `needs_review_weak`. It is deliberately its OWN phase, not
    // `run.infra_failure`: the machine was fine and the call was made — a reader
    // retries an infrastructure fault, and this one is the model's answer, so
    // reporting it as infrastructure is what turned one unreadable answer into a
    // halted wave (§8.2). The raw answer itself is in the forensics file.
    if let Some(error) = &model_output_error {
        println!("  ✖ MODEL OUTPUT INVALID: {error}");
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
            phase: "run.model_output_invalid".to_string(),
            summary: format!(
                "final_decision=NeedsReview stop_reason={MODEL_OUTPUT_INVALID} error={}",
                one_line.chars().take(1000).collect::<String>()
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
    }

    // [VANTA-WHY] M3 item 7 — the judge arm's verdict on its own phase row. The
    // verdict and its reasoning are the ONLY things a judge run concludes, and the
    // `JudgeApproved`/`JudgeRejected` decision token says which mechanism reached
    // an answer but not WHICH answer the model gave or why: `runner.label_for`
    // folds both tokens into the same TA/FA/TR/FR labels the test-driven arm uses
    // (so the per-candidate pairing the plan asks for works), which would
    // otherwise leave the judge's OWN words nowhere. `confidence` is the model's
    // number, recorded verbatim. This row exists only where a judge actually ran
    // (`judge_outcome` is `None` on every other arm), so no non-judge session
    // gains a row.
    if let Some((approved, confidence, reason)) = &judge_outcome {
        let one_line = reason
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
            phase: "run.judge".to_string(),
            summary: format!(
                "judge_verdict={} judge_confidence={:.2} judge_reason={}",
                if *approved { "approve" } else { "reject" },
                confidence,
                one_line.chars().take(1000).collect::<String>()
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
    }

    // [VANTA-WHY] M3 item 7 — a `JudgeFailed` run records WHY the judge's answer
    // could not be read after the one retry, exactly as `run.infra_failure` does
    // for a dead call: the reason is a fact in the log rather than something a
    // reader infers from `final_decision=JudgeFailed` alone. It is deliberately
    // its OWN phase, not `run.infra_failure`: the machine was fine and the call
    // was made, so this must not be read, retried or counted as infrastructure,
    // and its exit code (4) is the arm's own. The raw answer is in the forensics
    // file; this carries the parse/truncation diagnosis.
    if let Some(error) = &judge_failed_error {
        println!("  ✖ JUDGE FAILED: {error}");
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
            phase: "run.judge_failed".to_string(),
            summary: format!(
                "final_decision=JudgeFailed error={}",
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
                "seq={} site={} round={} attempt={} prompt_tokens={} completion_tokens={} \
                 total_tokens={} source={} seed={} seed_mechanism={} finish_reason={} wall_ms={} \
                 truncated={} {} cached_tokens={} reasoning_tokens={} cost_usd={} cost_basis={} \
                 provider_retries={} service_tier={} served_model={}",
                usage.seq,
                usage.site,
                // [VANTA-WHY] §14.12 — the round and the retry index travel with
                // every call, because the runaway diagnosis could not tell "four
                // attempts of one generation" from "four separate generations"
                // without them: the log had the calls but not their round.
                usage.round,
                usage.attempt,
                token_value(usage.prompt_tokens),
                token_value(usage.completion_tokens),
                token_value(usage.total_tokens),
                usage.source,
                // [VANTA-WHY] §14.5 — the seed THIS attempt sent, which is not
                // `VANTA_LLM_SEED` when a retry advanced it. Written per call so
                // a truncated-and-retried call is reproducible after the fact.
                token_value(usage.seed),
                // [VANTA-WHY] §14.23 — how that seed reached the body, so a call
                // made by a backend that cannot send one is distinguishable from
                // a call that was simply never asked for a seed.
                usage.seed_mechanism,
                // [VANTA-WHY] §14.12 — the server's own word, verbatim. `unset`
                // rather than a plausible-looking `stop`, which would be a claim
                // the server never made.
                usage.finish_reason.as_deref().unwrap_or("unset"),
                usage.wall_ms,
                usage.truncated,
                // [VANTA-WHY] §14.26 — what the body asked for, what it carried,
                // and what was removed or renamed on the way. Last of the
                // pre-§14.29 keys, so every key this row carried before keeps its
                // position.
                usage.request_shape.render(),
                // [VANTA-WHY] §14.29 — the usage the server reported that is not
                // a token count: how much of the prompt was served from cache,
                // how much of the completion was reasoning, what those cost, and
                // on what price basis. `unset` rather than `0` wherever the
                // server did not publish the detail, so "not measured" cannot be
                // pooled with "measured zero". Appended after the pre-existing
                // keys so a reader of an older row is unaffected.
                token_value(usage.cached_tokens),
                token_value(usage.reasoning_tokens),
                cost_value(usage.cost_usd),
                usage.cost_basis,
                // [VANTA-WHY] §14.30 — how many times the provider was asked
                // again before it served THIS call, beside `attempt` and not
                // inside it. A row with `attempt=1 provider_retries=2` is a call
                // the service throttled twice and the model answered once; a row
                // with `attempt=2 provider_retries=0` is the opposite. Appended
                // after the pre-existing keys so an older row's reader is
                // unaffected.
                usage.provider_retries,
                // [VANTA-WHY] §14.31 — the tier the provider served this call
                // on, beside the cost it produced and not far from it: the tier
                // IS the price row, so a reader matching a cost against the
                // table needs the two on one line. `unset` on the local backend
                // (nothing was sold), `standard`/`flex` on the hosted one.
                // Appended last so every key an older row carried keeps its
                // position.
                usage.service_tier,
                // [VANTA-WHY] M3 item B — the model the SERVER said it served
                // this call with, beside the tier and the cost it produced. The
                // per-call fact; `run.summary` collapses these to one
                // `model_version`. `unset` when the server named none — never the
                // requested id, which would hide a provider serving a different
                // snapshot than the one asked for. Appended last so every key an
                // older row carried keeps its position.
                //
                // [VANTA-WHY] T2 — encoded space-free like every other free-text
                // value on this row (`served_model_version_value` already encodes
                // the collapsed form). A served id carrying a space would
                // otherwise end the value early and could inject a `key=value`
                // the row never wrote; the encoding leaves an ordinary id and the
                // `unset` sentinel byte-for-byte, so every row already on disk
                // keeps its meaning.
                encode_summary_value(usage.served_model.as_deref().unwrap_or("unset")),
            ),
            timestamp: now_iso(),
            benchmark_instance: None,
            benchmark_label: None,
        })?;
    }
    let (llm_calls, llm_calls_measured, llm_prompt_tokens, llm_completion_tokens) =
        token_totals(&call_usages);
    // [VANTA-WHY] §14.29 — the run's cost and the two usage details that are not
    // token counts. Summed from the rows the run already recorded, for the same
    // reason `token_totals` is: an aggregate that re-derives its inputs can
    // disagree with the rows it summarises. `cost_basis` names the price rows
    // that produced the figure (`none` when the run made no call), so a total
    // built from two different price dates is visible rather than silently
    // pooled.
    let (cost_usd, llm_cached_tokens, llm_reasoning_tokens, cost_basis) =
        cost_and_detail_totals(&call_usages);

    // ---- Run-away (truncated) calls, per role (plans/bugsinpy.md §14.5) ----
    // [VANTA-WHY] A model that spends its whole output cap on reasoning and
    // returns nothing parseable is a fact about the MODEL, not about the bug it
    // was pointed at, and the paper has to be able to say how often it happens
    // (Ornith-1.0 does this roughly 1 run in 9 on the Attacker). Counting from
    // the calls the run already recorded means the count cannot disagree with
    // the run it describes; counting per ROLE means the number is reportable
    // beside the split's results rather than buried per call site.
    let (truncated_creator, truncated_attacker, truncated_intent_audit) =
        truncation_counts(&call_usages);
    // [VANTA-WHY] §14.12 — the run-level total and the per-SITE breakdown the
    // role split cannot give. `truncated_total` is counted off the rows rather
    // than summed from the three roles, so an off-schedule site cannot make the
    // total disagree with the calls it summarises.
    let truncated_total = call_usages.iter().filter(|c| c.truncated).count() as u64;
    // [VANTA-WHY] §14.30 — how often the PROVIDER, not the model, was asked again.
    // Read from the run-level counters rather than summed over the rows, and that
    // is the point: a call the provider kept refusing past `PROVIDER_RETRY_MAX`
    // never answered, so it writes no `call.usage` row at all. Counting only the
    // rows would under-report exactly the runs that were throttled hardest, which
    // is the opposite of what a throttling count is for. The two classes are kept
    // apart because they mean different things operationally (`rate_limit` is a
    // window rolling over; `resource_unavailable` is `flex` capacity).
    let (provider_retries, provider_retries_rate_limit, provider_retries_unavailable) =
        crate::llm::provider_retry_counts();
    let per_site_keys = per_site_totals(&call_usages, resolved_arm.judge.is_some())
        .iter()
        .map(|(site, calls, truncated, completion)| {
            format!(
                "llm_calls_site_{site}={calls} truncated_calls_site_{site}={truncated} \
                 completion_tokens_site_{site}={}",
                token_value(*completion)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");

    // ---- Provenance (plans/bugsinpy.md §4.4) ----
    // [VANTA-WHY] A session directory copied out of a batch has to be
    // self-describing: an env var is not provenance, because the shell that set
    // it is gone by the time anyone reads the record. The same fields also go
    // into the batch manifest (§7.3), and `metrics.py` groups by them (§9.1), so
    // any table can be cut by config × seed × model without re-running.
    //
    // [VANTA-WHY] `config` is the resolved ARM's config, and `arm` is the resolved
    // arm's name. Before item 8's follow-up `config` was derived from the switches
    // alone; it is now taken from the arm the harness STATED (`VANTA_ARM`), because
    // `b0` and `baseline` are the SAME switch pair and differ only in the attempt
    // budget — `benchmark_config_name` cannot tell them apart. The two can never
    // disagree with the run: `resolve_arm` refuses a stated name whose switches or
    // budget contradict it, and an absent `VANTA_ARM` (a hand-run binary) resolves to
    // exactly the switch-derived name, so those runs record what they always did.
    //
    // [VANTA-WHY] The sampling record is the REAL per-call schedule, not a
    // run-level scalar. The harness pins `temperature` and `max_tokens` per call
    // site — each agent's `extra_fields`, merged in `llm::post_chat` AFTER the
    // body defaults so the agent's value WINS — which is why one review run
    // samples at 0.1 (Creator), 0.2 (Attacker generate), 0.3 (Attacker
    // revise/simplify) and 0.0 (Intent Auditor). Those pins are part of v0.2's
    // behaviour; the harness must never override them, only record them.
    // `SAMPLING_SCHEDULE` names them (underscore keys — see its `[VANTA-WHY]`),
    // and `m3_session_log.rs` asserts the schedule equals the `temperature` /
    // `max_tokens` actually present in the request bodies. `top_p` is not pinned
    // by any call site under the default preset, so its value is the SERVER's own
    // default read from `/props` — unless `VANTA_SAMPLING=coding` (§14.11) pins
    // it on the generative sites, in which case those pins appear per role as
    // `top_p_<site>` and the `/props` value states only what the auditor
    // inherits. `seed` is what the HARNESS asked for. `unset` means the server
    // did not publish the key — never a plausible-looking default, which would be
    // a fabricated claim about a sampler nobody pinned.
    let config_name = resolved_arm.config.as_str();
    let arm_name = resolved_arm.name.as_str();
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
    // [VANTA-WHY] §1.6.2 item 4 — the source selection's provenance, on its own
    // run-level row AND in `run.summary`. "What was this run shown, and why" has
    // to be answerable from the artifacts alone: the prompt that carried the
    // selection is only in the debug forensics, and a report that has to re-run
    // the selector to learn which files a run saw is measuring a different
    // thing from the run. One selection serves every source-seeing role, so
    // there is at most one: `repair_selection` in repair mode, `review_selection`
    // in review mode (selected on first use, above); "none" is written rather
    // than nothing so a run that showed no source is distinguishable from a run
    // whose row is missing.
    //
    // `patch_attempt_id` is null for the same reason `session.start`'s is: the
    // selection happens before attempt 1 and is fixed for the whole run, so
    // stamping an attempt id would write an unobserved value to disk as a fact.
    let context_selection_line = repair_selection
        .as_ref()
        .or(review_selection.as_ref())
        .map(crate::context::ContextSelection::provenance_line)
        .unwrap_or_else(|| "none".to_string());
    session_log.append(SessionRecord {
        session_id: session_id.clone(),
        schema_version: Some(SCHEMA_VERSION),
        patch_attempt_id: None,
        testing_iteration_id: None,
        test_id: None,
        phase: "context.select".to_string(),
        summary: context_selection_line.clone(),
        timestamp: now_iso(),
        benchmark_instance: Some(config_name.to_string()),
        benchmark_label: None,
    })?;
    // [VANTA-WHY] M3 item 1 — the approval invariant's own answer, carried on the
    // row so the exemption is a recorded fact and not an inference from the
    // flags. Computed from the same function the `AllTestsPassed` arm used, so
    // the row and the verdict cannot disagree. A run that ended without an
    // approval (`Rejected`/`MaxAttemptsReached`/`InfraFailure`/`HarnessFailure`)
    // gave the invariant nothing to constrain — `invariant_applied=false` with
    // `no_approval` says exactly that, rather than reporting a satisfied/not
    // reading about a decision the run never made.
    let (invariant_applied_here, invariant_reason_here) =
        approval_invariant_status(
            governor_mode,
            attacker_enabled,
            bate_enabled,
            resolved_arm.judge.is_some(),
            &final_evidence,
        );
    let invariant_applied = invariant_applied_here
        && matches!(
            decision,
            GovernorDecision::AllTestsPassed | GovernorDecision::NeedsReview
        );
    let invariant_reason = if invariant_applied_here && !invariant_applied {
        "no_approval"
    } else {
        invariant_reason_here
    };
    // [VANTA-WHY] Item S — the Creator's own request facts, read from the same
    // values the body was built from and the same `call.usage` rows the run
    // already records. Purely additive: nothing here feeds a verdict, a selection
    // or a guard, and the fields are appended after every pre-existing key so an
    // older row's reader is unaffected.
    let (creator_effort, creator_effort_mechanism) = creator_effort_text(&cfg);
    let creator_prompt_tokens = creator_prompt_tokens_text(&call_usages);
    session_log.append(SessionRecord {
        session_id: session_id.clone(),
        schema_version: Some(SCHEMA_VERSION),
        patch_attempt_id: attempt_patch_ids.last().cloned(),
        testing_iteration_id: None,
        test_id: None,
        phase: "run.summary".to_string(),
        summary: format!(
            "attempts_run={} fixed_at_attempt_1={} final_decision={:?} \
             config={} arm={} mode={} seed={} seed_policy={} model={} top_p={} {} {} {} \
             llm_calls={} llm_calls_measured={} llm_prompt_tokens={} llm_completion_tokens={} \
             llm_cached_tokens={} llm_reasoning_tokens={} llm_cache_hit_rate={} \
             cost_usd={} cost_basis={} \
             truncated_calls_creator={} truncated_calls_attacker={} \
             truncated_calls_intent_audit={} \
             truncated_calls={} truncated_twice_calls={} \
             provider_retries={} provider_retries_rate_limit={} provider_retries_unavailable={} \
             completion_tokens_total={} {} \
             corroborate={} grounded_audit={} batch_audit={} attacker_concise={} \
             batch_audit_tokens_per_test={} batch_audit_max_tokens={} \
             revise_budget_per_attempt={} revise_calls_used={} \
             deadline_secs={} stop_reason={} presence_penalty={} max_tokens_overrides={} \
             reasoning_budget={} reasoning_budget_mechanism={} reasoning_effort={} \
             reasoning_effort_mechanism={} \
             service_tier={} \
             deciding_rule={} deciding_evidence=[{}] needs_review_weak=[{}] \
             commit={} dirty={} prompt_version={} prompt_hash={}{} context_selection={} \
             harness_error={} \
             invariant_applied={} invariant_reason={} \
             grounded_scrapped={} grounded_kept_by_issue={} \
             quarantine_count={} quarantined=[{}] \
             seed_method={} import_guard={} \
             n_ctx={} creator_effort={} creator_effort_mechanism={} \
             creator_prompt_tokens={}{}{}{} fixed_rounds={} native_vacuous={} \
             native_timeout_secs={} native_timeout={} native_cache_hits={} \
             native_original_secs={} native_patched_secs={} \
             native_original_timed_out={} native_patched_timed_out={} \
             native_original_timed_out_last={} native_patched_timed_out_last={} \
             native_session_cap_secs={} native_gate_effective_cap_secs={} native_gate_clamped={} \
             peak_sandboxes={} sandbox_backend={} model_version={}",
            attempts_run,
            fixed_at_attempt_1,
            decision,
            config_name,
            arm_name,
            run_mode,
            seed_value,
            // [VANTA-WHY] §14.23 — the seed POLICY, not the seed. The offset is
            // default-on, so a row's own `seed` cannot say whether its later rounds
            // were independent draws; this field is what makes a pre-§14.23 row
            // (which has no `seed_policy` at all) impossible to pool by accident.
            crate::llm::SEED_POLICY,
            cfg.model_id,
            top_p_value,
            sampling_schedule_keys(resolved_arm.judge.is_some()),
            request_shape_rule(&cfg),
            sampling_not_sent_rule(&cfg),
            llm_calls,
            llm_calls_measured,
            llm_prompt_tokens,
            token_value(llm_completion_tokens),
            // [VANTA-WHY] §14.29 — the run's cached/reasoning tokens and its
            // cost, with the price basis that produced them. `unset`/`none`
            // wherever nothing was measured (the local backend is
            // `not_billed`, an unpriced family names itself), never `0`.
            token_value(llm_cached_tokens),
            token_value(llm_reasoning_tokens),
            // [VANTA-WHY] §14.32 — the cache hit rate, derived from the two
            // totals on this same line so the three cannot disagree. `unset`
            // wherever no rate exists (nothing measured, nothing reported, or a
            // zero prompt total), never `0.0000`.
            cache_hit_rate_value(llm_cached_tokens, llm_prompt_tokens),
            cost_value(cost_usd),
            cost_basis,
            // [VANTA-WHY] §14.5 — how often a model ran away on each role in THIS
            // run. A count, not a rate: the split's report pools it over runs.
            truncated_creator,
            truncated_attacker,
            truncated_intent_audit,
            // [VANTA-WHY] §14.12 — the same three facts the role split reports,
            // at the two granularities a runaway report needs: the run total, and
            // every call site (including the Attacker's three, which the single
            // `attacker` role deliberately merges).
            truncated_total,
            // [VANTA-WHY] §14.5 item 2 — how many calls gave up because BOTH the
            // request and the brief re-ask were cut off. A single truncation is a
            // run-away; two in a row is a call this site cannot answer inside its
            // cap, and the fix to look for is a smaller question, not a retry.
            crate::llm::truncated_twice_calls(),
            // [VANTA-WHY] §14.30 — the provider-retry totals, deliberately beside
            // the model's truncation counts: "the model could not finish" and "the
            // service would not take the request" are the two reasons a call is
            // repeated, and a run that mixed them should say so rather than leave
            // a reader to guess which one a repeated call was.
            provider_retries,
            provider_retries_rate_limit,
            provider_retries_unavailable,
            token_value(llm_completion_tokens),
            per_site_keys,
            // [VANTA-WHY] §14.2/§14.4/§14.5 — the switches this run actually ran
            // under, and the corroboration that produced its verdict. A reader
            // must be able to tell a `Rejected` decided by rule (a) from one
            // decided by rule (c), and a `NeedsReview` must name the weak
            // failures that triggered it — otherwise "needs a human" names
            // nothing to look at.
            flag_text(corroborate_enabled),
            flag_text(grounded_audit_enabled),
            flag_text(batch_audit_enabled),
            // [VANTA-WHY] §14.22 — the prompt arm. Recorded beside the rule
            // switches because it is the same kind of fact: which of two prompts
            // this run sent, so an ablation row can be put in the right arm.
            // `off` is the default and every run before this change.
            flag_text(attacker_concise_enabled),
            crate::agents::attacker::INTENT_AUDIT_BATCH_TOKENS_PER_TEST,
            token_value(batch_audit_max_tokens),
            // [VANTA-WHY] §14.14 — the Attacker remediation cap and the run's
            // total spend against it. Reported beside the §14.5 truncation counts
            // because they are the two answers to "is the Attacker stuck?": a
            // truncation says a call could not finish, this says the round kept
            // asking.
            REVISE_BUDGET_PER_ATTEMPT,
            revise_calls_used_total,
            // [VANTA-WHY] §14.15 — the run's own deadline and why it stopped, when
            // it stopped itself. `unset`/`none` is every run before this one and
            // every run without `VANTA_DEADLINE_SECS`; `budget_exhausted` is the
            // deadline firing, which is the one stop the harness's `timed_out`
            // must not be confused with.
            deadline_secs
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unset".to_string()),
            stop_reason.unwrap_or("none"),
            // [VANTA-WHY] §14.16 — the presence penalty that was ASKED FOR, not
            // the one that was sent: zero and unset send identical bodies, so a
            // comparison of the two arms can only be made from this field.
            // `unset` is the default and every run before this one.
            presence_penalty
                .map(|v| format!("{v:?}"))
                .unwrap_or_else(|| "unset".to_string()),
            // [VANTA-WHY] §14.20 — the output caps that were ASKED FOR, in the
            // schedule's own order, `site=cap` pairs. The per-site
            // `max_tokens_<site>` keys earlier on this line carry the EFFECTIVE
            // cap each body was sent under; this field carries the switch, which
            // is what identifies an ablation arm. `unset` is the default and
            // every run before this change. The value holds no spaces — the
            // parser in `rerun.rs`/`metrics.py` reads a value only up to the next
            // space that begins a lowercase `key=`, and `attacker_revise=12288,…`
            // therefore survives intact.
            crate::llm::max_tokens_overrides_text(),
            // [VANTA-WHY] §14.21 — the reasoning arm, recorded at the two levels
            // §14.20 established: the budget/effort that were ASKED FOR (the
            // switch, which identifies the arm) and the mechanism each backend
            // used to implement them, so an arm run here can be compared with the
            // same arm run against a hosted API. `reasoning_effort` is the
            // EFFECTIVE level and defaults to `low` — the six call sites have sent
            // `low` since v0.2, so `unset` would describe a run that sent nothing.
            //
            // [VANTA-WHY] §14.28 — both mechanisms are read per backend, and the
            // hosted default is NOT `low`: with the switch unset the call sites'
            // `low` never reaches a hosted model, so the honest value there is
            // `provider_default` with an `unset` mechanism. On the local backend
            // the same switch records `low` + `llamacpp:noop`, because the key is
            // in the body and the server ignores it. Neither value contains a
            // space, so both survive the `rerun.rs`/`metrics.py` value-terminator
            // rule intact.
            crate::llm::reasoning_budget_text(),
            crate::llm::reasoning_budget_mechanism(cfg.is_openai()),
            crate::llm::reasoning_effort_text(cfg.is_openai()),
            crate::llm::reasoning_effort_mechanism(cfg.is_openai()),
            // [VANTA-WHY] §14.31 — the run-level tier, read once from the same
            // config field that shaped every request body and priced every row,
            // so the summary cannot disagree with either. `unset` on the local
            // backend, where no tier exists to name. Appended after the
            // pre-existing keys so an older row's reader is unaffected.
            cfg.service_tier_text(),
            deciding_rule.as_deref().unwrap_or("none"),
            // [VANTA-WHY] M3 item T2 — the evidence and the weak-failure list are
            // TEST NAMES, and a pytest node id can carry a space (a parametrize id
            // is free text). Each element is percent-encoded so one such name can
            // neither truncate the value nor inject a `key=value`; a name without
            // whitespace is written unchanged.
            deciding_evidence
                .iter()
                .map(|evidence| encode_summary_value(evidence))
                .collect::<Vec<_>>()
                .join(","),
            needs_review_weak
                .iter()
                .map(|weak| encode_summary_value(weak))
                .collect::<Vec<_>>()
                .join(","),
            vanta_commit,
            vanta_dirty,
            // [VANTA-WHY] §14.10 — the prompt identity of the run, the same
            // string `session.start` carries (`pipeline::prompt_versions`). A
            // session's debug artifacts outlive the shell that produced them, so
            // "which prompts wrote this row" must be readable from the row: a run
            // made before the Attacker prompt's preservation fix is otherwise
            // indistinguishable from one made after it.
            crate::pipeline::prompt_versions(),
            // [VANTA-WHY] §14.10 f — the content hash beside the version label.
            // A version tag is only as good as the discipline of bumping it, and
            // that discipline slipped twice already (the A3 and A10 follow-ups
            // reworded prompts without a bump). The hash is computed over the
            // exact text each role sends, so a wording change shows up in the log
            // whether or not anyone remembered the label.
            crate::pipeline::prompt_hashes(),
            // [VANTA-WHY] M3 item 7 — the JUDGE prompt's own version and content
            // hash, spliced in immediately after the pipeline prompt provenance.
            // Only a judge arm has a judge prompt to name, so this renders the
            // empty string on every other row (the leading space lives inside the
            // helper) and every non-judge `run.summary` stays byte-identical.
            judge_prompt_keys(resolved_arm.judge.is_some()),
            // [VANTA-WHY] §1.6.2 item 4 — the same provenance the `context.select`
            // row carries, folded into the terminal row so a reader of
            // `run.summary` alone can tell whether the bug's own file was in the
            // prompt without opening the session log.
            context_selection_line,
            // [VANTA-WHY] §14.19 — which test's baseline died before pytest graded
            // it, when the environment broke. `unset` is every run in which BATE
            // reached verdicts, including every run before this change, so an
            // unset value keeps the field byte-identical in meaning to its
            // absence. The full evidence is the `bate.harness_error[…]` session
            // row; this holds the space-free token the parser needs.
            harness_error.as_deref().unwrap_or("unset"),
            // [VANTA-WHY] M3 item 1 — the approval invariant's answer and the
            // grounding rule's ledger, appended after the pre-existing keys so an
            // older row's reader is unaffected. `invariant_applied=false` with
            // its reason names every configuration the gate does not constrain
            // (a repair run, the Attacker-off `baseline` review, a review that
            // ended without an approval); `grounded_scrapped` /
            // `grounded_kept_by_issue` are the plan's per-review grounding
            // report, split into what the rule removed and what item 1's
            // loosening put back — and `null` when no grounded audit ran.
            invariant_applied,
            invariant_reason,
            ledger_text(grounded_scrapped_total),
            ledger_text(grounded_kept_by_issue_total),
            // [VANTA-WHY] M3 item 17 — the quarantine ledger: how many probes the
            // Diagnostician removed from the shared set this run, and which ones.
            // [VANTA-WHY] M3 item T2 — each name is percent-encoded rather than
            // assumed space-free: a pytest node id is free text (a parametrize id
            // can carry a space), and an unencoded one could truncate this value.
            // The reason each was removed is the per-test `quarantine[<name>]`
            // session row and the `attempt_*_diagnostic_<name>.json` debug
            // artifact, both of which can hold free text the row cannot.
            quarantined.len(),
            quarantined
                .iter()
                .map(|q| encode_summary_value(&q.test_name))
                .collect::<Vec<_>>()
                .join(","),
            // [VANTA-WHY] Item P — which mechanism seeded this run's sandboxes
            // (`clonefile` for a copy-on-write clone, `copy_tree` for the
            // fallback), and the in-session import guard's space-free token
            // (`import_guard=<module>`, or `import_guard=none` for a
            // single-module project with no top-level package to check). Both are
            // space-free, so they survive the `key=value` reader. `unset` is every
            // run before this change, and also a plain judge arm, which spawns no
            // sandbox to seed or guard. Appended after the pre-existing keys so an
            // older row's reader is unaffected; the guard's full line (with the
            // path the package imported from) is the `run.import_guard` session
            // row.
            seed_method_record,
            import_guard_line
                .as_deref()
                .and_then(|line| line.split_whitespace().next())
                .unwrap_or("unset"),
            // [VANTA-WHY] Item S — appended after every pre-existing key so an
            // older row's reader is unaffected. `n_ctx` is the window this run
            // sized its file selection against; `creator_*` are the Creator's own
            // request facts (see `creator_effort_text` /
            // `creator_prompt_tokens_text`). All three values are space-free, so
            // they survive the `key=value` reader.
            n_ctx,
            creator_effort,
            creator_effort_mechanism,
            creator_prompt_tokens,
            // [VANTA-WHY] M3 item V — appended after every pre-existing key so an
            // older row's reader is unaffected. The count is present only where a
            // Creator ran (a repair run): a review run's patch was written outside
            // the run, so "how many attempts edited a file the Creator could not
            // read in full" is a question that run cannot be asked, and an absent
            // key says so more honestly than a `0` would. The leading space lives
            // in this value, the way `judge_prompt_keys` carries its own.
            if config.patch_path.is_none() {
                format!(" attempts_edited_unshown_file={attempts_edited_unshown_file}")
            } else {
                String::new()
            },
            // [VANTA-WHY] Q1 — appended after every pre-existing key so an older
            // row's reader is unaffected, and present only where a Creator ran:
            // the count of `===FILE:` markers that needed trimming is a statement
            // about the Creator's own output, which a review run (whose patch was
            // written outside the run) cannot make. LOGGING ONLY — no metric, label
            // or verdict reads it. The leading space lives in the value, the way
            // `attempts_edited_unshown_file` carries its own.
            if config.patch_path.is_none() {
                format!(" markers_trimmed={markers_trimmed}")
            } else {
                String::new()
            },
            // [VANTA-WHY] Q2 — appended after every pre-existing key so an older
            // row's reader is unaffected, and present only where a Creator ran: a
            // review run's patch was written outside the run, so "which files did
            // this patch introduce" is answered there by the patch the caller
            // supplied, not by Vanta. Space-free, `;`-joined paths (the shape
            // `selected=[…]` already uses) with the literal `none` for a genuine
            // empty list — so an ABSENT key (an older row, or a review row) reads as
            // UNKNOWN, never as "created no files". RECORD ONLY: no metric, label or
            // verdict reads it.
            //
            // [VANTA-WHY] M3 item T2 — each path is percent-encoded
            // (`summary_escape::encode_summary_value`) before the join, because a
            // path is the Creator's own output: one shaped like
            // `x final_decision=AllTestsPassed y.py` would otherwise TRUNCATE this
            // value at its space and INJECT a `final_decision` the row never wrote.
            // The runner's `_list` decodes each entry, so the row still names the
            // path the patch really created.
            if config.patch_path.is_none() {
                format!(
                    " new_files_created={}",
                    encode_summary_path_list(new_files_created.iter())
                )
            } else {
                String::new()
            },
            // [VANTA-WHY] M3 item 9 — appended after every pre-existing key so an
            // older row's reader is unaffected, and space-free so it survives the
            // `key=value` reader. `off` is the default and means the loop stopped
            // at its first pass (or at `--max-attempts`); a number means the run was
            // asked to spend exactly that many attempts. The row needs it because
            // the SAME verdict means different things on the two settings: an
            // `AllTestsPassed` row from a K = 4 run is attempt 4's verdict, and a
            // reader who pooled it with a default run's first-attempt approval
            // would be comparing a patch that was re-examined three times with one
            // that was never re-examined at all. `attempts` in `evidence.json`
            // holds the per-attempt detail this number indexes.
            if fixed_rounds.is_some() {
                loop_attempts.to_string()
            } else {
                "off".to_string()
            },
            // [VANTA-WHY] M3 item W — appended after every pre-existing key so an
            // older row's reader is unaffected. `true` means the run's native gate
            // collected nothing while the row's own prepare had recorded tests for
            // this bug, so the run ended as `InfraFailure`: the row carries
            // `verdict=InfraFailure` and the label `infra_failure`, and a reader
            // must never pool it with a run whose native suite actually ran.
            native_vacuous_seen,
            // [VANTA-WHY] M3 item D1 — the cap this run's native suite ran under,
            // and whether its vacuity (if any) was caused by that cap KILLING a
            // side rather than a tree that collected nothing. Appended after
            // `native_vacuous` so an older row's reader sees the same fields in
            // the same order; `native_timeout=true` always implies
            // `native_vacuous=true` and never changes the label.
            //
            // [VANTA-WHY] Finding C (PR #4 review) — the kill must be by the
            // CONFIGURED cap. A side cut by the DEADLINE CLAMP is excluded here
            // and recorded by `native_gate_clamped` instead, so `native_timeout`
            // never names a cap the suite did not actually run under.
            native_cap,
            native_timeout_seen,
            // [VANTA-WHY] M3 item S1 — how many of this run's native gate calls
            // were served from the cache instead of measured. Purely descriptive
            // (nothing reads it), appended after `native_timeout` so an older
            // row's reader sees the same fields in the same order.
            native_cache_hits,
            // [VANTA-WHY] v1-m3.1 (§4) and F3 (PR #4 review) — the per-side
            // in-session seconds, the per-side kill flags and the cap those runs
            // actually used. `unset` (never a bare `None`) for a side that produced
            // no wall time, so a `key=value` reader sees a token, not a Rust Debug
            // string. The `*_secs`, the `*_timed_out_last` pair and
            // `native_gate_effective_cap_secs` describe the SAME (last) gate; the
            // `*_timed_out` pair keeps the wider ANY-attempt meaning. Appended after
            // `native_cache_hits` so an older row's reader is unaffected.
            native_gate
                .original_secs
                .map(|s| format!("{:.3}", s))
                .unwrap_or_else(|| "unset".to_string()),
            native_gate
                .patched_secs
                .map(|s| format!("{:.3}", s))
                .unwrap_or_else(|| "unset".to_string()),
            native_gate.original_timed_out_any,
            native_gate.patched_timed_out_any,
            native_gate.original_timed_out_last,
            native_gate.patched_timed_out_last,
            native_cap,
            // [VANTA-WHY] v1-m3.1 F5 — the cap the LAST native gate actually ran
            // under (after the deadline clamp), which is what a clamped timeout
            // must be attributed to. `native_session_cap_secs` above keeps its own
            // meaning (the resolved/configured cap, the rule). `unset` when no gate
            // ran, so a reader sees a token, never a bare `None`.
            native_gate
                .effective_cap_secs
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unset".to_string()),
            // [VANTA-WHY] v1-m3.1 F3 — whether the LAST native gate ran under a
            // clamp-lowered cap. `true` records the deadline clamp, which is how a
            // `budget_exhausted`/`NeedsReview` row whose ORIGINAL side was cut
            // differs from a real infrastructure fault: its `native_vacuous` stays
            // false and this flag carries the clamp instead. Appended after every
            // pre-existing key so an older row's reader is unaffected.
            native_gate_clamped,
            // [VANTA-WHY] M3 item S6 — the exact peak of simultaneous live
            // microVMs this run booted (`sandbox::peak_sandboxes`), space-free so
            // it survives the `key=value` reader. The scheduler charges a row
            // `peak_vms × vm_memory_mb`; a reader compares this observed peak with
            // the conservative `--peak-vms` charge to calibrate it from real runs.
            // Purely descriptive — nothing here feeds a verdict, a selection or a
            // guard. Appended after every pre-existing key so an older row's
            // reader is unaffected.
            crate::sandbox::peak_sandboxes(),
            // [VANTA-WHY] W1 — which security boundary ACTUALLY ran this row.
            // `sandbox_backend=shuru` is a microVM; `sandbox_backend=subprocess`
            // is untrusted code on the host with no boundary, reachable only by
            // an explicit `VANTA_SANDBOX_BACKEND=subprocess` (an `auto` run that
            // cannot boot shuru now refuses before any row is written). Recording
            // it per row is what makes that downgrade auditable: the value is the
            // same one `session.start` already carries as `actual_backend`, read
            // through to the row (`runner.build_record`). Space-free, so it
            // survives the `key=value` reader; appended after every pre-existing
            // key so an older row's reader is unaffected.
            actual_backend,
            // [VANTA-WHY] M3 item B — the run's served model version, collapsed
            // from the per-call `served_model` rows (see
            // `served_model_version_value`): the snapshot when every call that
            // named one agrees, the sorted `[a;b]` list when they differ, and
            // `unknown` when no call named one. This is ADDITIVE to
            // `session.start`'s `model_version`, not a replacement: that row is
            // written before the first call and so cannot know what the server
            // served; it keeps its own `VANTA_MODEL_VERSION` env read. The
            // measured, post-hoc fact lives here. Appended after every
            // pre-existing key so an older row's reader is unaffected.
            served_model_version_value(&call_usages),
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
        // [VANTA-WHY] M3 item E / v1-m3.1 item 1 — `None` here means the run never
        // had a tree to diff, and that is exactly two histories: a setup / LLM
        // fault stopped the run before anything applied (`HarnessFailure`,
        // `InfraFailure`), OR a `budget_exhausted` deadline exit fired while no
        // attempt had yet produced a tree (attempt 1's Creator was refused by the
        // clock, say). Both are genuinely unmeasurable and stay `unobservable`.
        // Note the deadline exits are the deadline exits: they attach
        // `last_applied_patched_files` (or this attempt's `patched_files`), NOT the
        // fabricated original. Every OTHER terminal verdict — including
        // `PatchApplicationFailed`, and a repair run that ended on a malformed
        // patch — attaches a tree (the last one produced, or the original when
        // nothing ever applied), so the oracle grades it instead of censoring it.
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
                    // [VANTA-WHY] T1 — the last patch-derived write site. The
                    // apply already refused an unsafe path, so this is the second
                    // lock on the same door: a path that could not be written
                    // under Contract B cannot be written into the debug copy of
                    // the tree either.
                    let Ok(out_path) = confined_join(&stage_dir, path) else {
                        continue;
                    };
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
            // [VANTA-WHY] M3 item W — did the native gate measure anything? A
            // `true` here means the in-session suite collected no tests while the
            // row's prepare had recorded some, so the run ended as `InfraFailure`
            // and the native check was NOT performed. Recorded beside the
            // regressions because "0 regressions" and "0 regressions measured over
            // 0 tests" are opposite claims that the old row could not tell apart.
            "native_vacuous": native_vacuous_seen,
            "native_expected_tests": config.native_expected_tests,
            // [VANTA-WHY] M3 item D1 — the cap this run's in-session native suite
            // actually ran under (`None` = the historical `test_timeout_secs()`), so
            // the artifact states the number that produced the vacuity. A `true`
            // `native_timeout` beside it means a side was KILLED by that cap rather
            // than a tree that failed to import — opposite fixes, so the row must
            // not have to guess which one it got.
            //
            // [VANTA-WHY] Finding C (PR #4 review) — the kill must be by the
            // CONFIGURED cap. A side cut by the DEADLINE CLAMP is excluded and
            // recorded by `native_gate_clamped` on the row instead.
            "native_timeout_secs": config.native_timeout_secs,
            "native_timeout": native_timeout_seen,
            // [VANTA-WHY] v1-m3.1 (§4) and F3 (PR #4 review) — the per-side
            // in-session wall time, the per-side kill flags and the cap those runs
            // actually ran under, so the §8.1 reading and the number that produced
            // it are both in the artifact. `None` = a side that produced no wall
            // time. The `*_secs` and the `*_timed_out_last` pair describe the SAME
            // (last) gate; `native_original_timed_out`/`native_patched_timed_out`
            // (the any-attempt pair) keep the wider meaning. Appended after
            // `native_timeout` so an older row's reader is unaffected.
            "native_original_secs": native_gate.original_secs,
            "native_patched_secs": native_gate.patched_secs,
            "native_original_timed_out": native_gate.original_timed_out_any,
            "native_patched_timed_out": native_gate.patched_timed_out_any,
            "native_original_timed_out_last": native_gate.original_timed_out_last,
            "native_patched_timed_out_last": native_gate.patched_timed_out_last,
            "native_session_cap_secs": native_cap,
            // [VANTA-WHY] M3 item S1 — one entry per native gate call (attempt ×
            // side) recording whether the result was measured or served from the
            // suite cache, under which key, and from whose row. The reuse is
            // deliberately auditable from the artifact: a row whose
            // `native_original` came from an earlier session is a legitimate
            // measurement of the same bytes, and this list is what lets a reader
            // confirm that rather than assume it.
            "native_cache": native_cache_records,
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
            // [VANTA-WHY] M3 item V — per attempt, for every file the patch
            // touches, how that file stood in the Creator's own context, and the
            // run-level count of attempts that edited a file it could not read in
            // full. The count also rides `run.summary` so the digest and
            // `metrics.py` can cut by it; the per-file detail lives here, where a
            // reader can see WHICH file an attempt overreached on. Empty on a
            // review run, which has no Creator.
            "context_visibility": context_visibility_records,
            "attempts_edited_unshown_file": attempts_edited_unshown_file,
            // [VANTA-WHY] Q2 — the paths a Creator patch INTRODUCED (absent from the
            // original tree, non-empty content), unioned across attempts and sorted.
            // RECORD ONLY: a mistyped marker path lands here, so "the patch quietly
            // created a file next to the one it meant to edit" is answerable from the
            // run's own artifacts. Nothing reads this but a reader. The same list
            // rides `run.summary` as `new_files_created=[…]`.
            "new_files_created": new_files_created.iter().cloned().collect::<Vec<_>>(),
            // [VANTA-WHY] M3 item 9 — every attempt's patch, the test set it ran,
            // the feedback it was given, the hash of the Creator prompt that
            // produced it, and whether the Governor read it as passing. The
            // acceptance for this item is "with K = 4 the run stores 4 patches and
            // the test sets per attempt", and a fixed-rounds run's whole purpose is
            // to be replayed attempt by attempt (item 10), so the index has to be a
            // fact in the run's own artifacts rather than a directory convention.
            // Always written, for every mode: a default run simply has one entry,
            // and a review run's entry names the provided patch.
            "attempts": fixed_round_records,
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

/// The tree `final.patch` is written from when a terminal verdict carried no
/// tree of its own (M3 item E): the run's last PRODUCED tree — the most recent
/// attempt whose patch applied — or the original tree when nothing ever applied.
///
/// [VANTA-WHY] M3 item E — the loop re-applies each attempt from the original and
/// writes nothing on a failed apply, so a run whose LAST attempt was malformed
/// (a review candidate that would not apply, or a repair run that spent its
/// budget on one) has no `patched_files` of its own. Falling back to the original
/// rather than leaving the terminal arms patchless keeps the label a measurement:
/// the empty patch does not fix the bug, so the oracle grades the row a failure
/// and it counts against the arm — which is the honest reading — instead of
/// censoring it to `unobservable`.
fn final_patch_tree(
    last_applied: Option<&HashMap<String, String>>,
    original: &HashMap<String, String>,
) -> HashMap<String, String> {
    last_applied
        .cloned()
        .unwrap_or_else(|| original.clone())
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
    // [VANTA-WHY] T1 — this stages the original/patched trees for `git diff
    // --no-index`, so `root.join(path)` is a WRITE named by the patch. Confine
    // it: the caller only ever passes the apply's own (already-vetted) map, and
    // this is the second lock on the same door. `create_dir_all(root)` first, so
    // the containment check can resolve the root.
    fs::create_dir_all(root)?;
    for (path, contents) in files {
        let out_path = confined_join(root, path).map_err(|why| {
            io::Error::new(io::ErrorKind::PermissionDenied, why)
        })?;
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out_path, contents)?;
    }
    Ok(())
}

// ============================================================================
// apply_patch_files_all_or_nothing — Contract B (M3 item A1)
// ============================================================================
//
// [VANTA-WHY] A Creator patch is ALL-OR-NOTHING across its `===FILE===` blocks.
// The repair loop used to apply each block on its own and keep whichever landed,
// so a two-file patch whose second block failed to anchor left the TESTED tree
// holding the first file's change — a program the Creator never proposed — while
// `final.patch`, written from the same map, was the only honest record of it. The
// verdict was then a statement about a tree no patch describes, and the oracle
// graded something other than what repair tested.
//
// [VANTA-WHY] One block failing now fails the whole apply, so NOTHING is written:
// the tested tree, the source strings the roles read and `final.patch` are all
// derived from the map this returns, so they can never disagree. The reason names
// the failing file, which is the whole point of item R's feedback.
//
// [VANTA-WHY] PURE on purpose — `(original tree, patch blocks) -> all patched
// files | one error`, with no I/O and no logging — so the rule is tested directly
// instead of through a sandbox. The caller does the writing, and on `Err` it
// writes nothing.
fn apply_patch_files_all_or_nothing(
    original_files: &HashMap<String, String>,
    patch_files: &HashMap<String, String>,
) -> Result<HashMap<String, String>, ApplyFailure> {
    // [VANTA-WHY] Sorted, so WHICH file is reported as the failure does not
    // depend on HashMap iteration order: "the second block did not apply" has to
    // be the same statement on every run, or the feedback the Creator gets is
    // nondeterministic.
    let mut paths: Vec<&String> = patch_files.keys().collect();
    paths.sort();
    // [VANTA-WHY] T1 — the path in a `===FILE:` marker is MODEL text, taken
    // verbatim at any indentation (item Q1). Every block is checked before any
    // is applied: a block naming an absolute path, a `..` component, a drive or
    // UNC prefix, an empty or NUL-bearing path — or a padded one the
    // concatenation would re-read as a DIFFERENT path — fails the WHOLE patch
    // under Contract B with the reason `unsafe_path`, and nothing is written
    // anywhere. PURE, like the rest of this function: the check is a string
    // predicate, so the rule is tested directly instead of through a sandbox.
    // Sorted first, so WHICH unsafe block is reported does not depend on
    // HashMap iteration order.
    for path in &paths {
        if let Some(why) = unsafe_relative_path_reason(path) {
            return Err(ApplyFailure::UnsafePath {
                path: (*path).clone(),
                why,
            });
        }
    }
    let mut patched = original_files.clone();
    for path in paths {
        let patch_content = &patch_files[path];
        match original_files.get(path) {
            Some(original) => {
                let (text, _changed, why) = apply_patch_detailed(original, patch_content, path);
                // [VANTA-WHY] Item E5 — `why`, not `changed`, is the decision. A
                // block that anchored but produced text identical to the original
                // returns `why = None` (the file simply did not change); it is not
                // a per-block failure, and the caller's whole-patch check is what
                // turns "nothing changed" into "patch changed nothing". A block
                // that could not anchor returns `Some(...)`, and Contract B fails
                // the whole apply. Before E5 an anchored no-change block was
                // rejected here with the misleading reason "search/replace:
                // produced text identical to the original".
                match why {
                    None => {
                        patched.insert(path.clone(), text);
                    }
                    Some(failure) => {
                        // Contract B: one block that does not anchor fails them all.
                        return Err(match failure {
                            // The syntax refusal's own message does not carry the path;
                            // add it so the Creator is told WHICH block.
                            ApplyFailure::InvalidPython(err) => {
                                ApplyFailure::InvalidPython(format!("{path}: {err}"))
                            }
                            other => other,
                        });
                    }
                }
            }
            // A block that introduces a file the tree does not have (a new module
            // the fix adds) is taken verbatim; an empty block adds nothing, and the
            // caller's "did anything change" check catches a patch that changes
            // nothing at all.
            None if !patch_content.trim().is_empty() => {
                patched.insert(path.clone(), patch_content.clone());
            }
            None => {}
        }
    }
    Ok(patched)
}

/// [VANTA-WHY] Q2 — the paths a repair patch INTRODUCES: blocks whose path the
/// original tree does not have and whose content is not empty. An empty new-file
/// block adds nothing (the apply's `None =>` arm), so it is not a created file.
///
/// [VANTA-WHY] PURE on purpose — `(original tree, patch blocks) -> sorted paths`,
/// with no I/O and no logging — so the rule is tested directly. This decides
/// NOTHING: `apply_patch_files_all_or_nothing` already accepts a new path, and
/// this only names the ones it wrote so the run can RECORD them. Nothing reads
/// the result but the record itself.
fn new_files_from_patch(
    original_files: &HashMap<String, String>,
    patch_files: &HashMap<String, String>,
) -> Vec<String> {
    let mut created: Vec<String> = patch_files
        .iter()
        .filter(|(path, content)| {
            !original_files.contains_key(*path) && !content.trim().is_empty()
        })
        .map(|(path, _)| path.clone())
        .collect();
    created.sort();
    created
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

/// The capped dump the PROMPT falls back to when a selection renders empty.
///
/// [VANTA-WHY] §14.18 — separated from the execution tree on purpose. The
/// prompt budget and the tree tests run in are different questions: the tree
/// has to be the whole repo (`repo::complete_tree_files`) or an import fails in
/// a plugin, while a prompt that is handed every file of a large repo is a
/// prompt that overflows. This fallback is byte-for-byte the dump the fallback
/// has always been — `repo::capped_source_files` is the loader's original walk,
/// applied to the same tree the prompt is about.
fn capped_dump_concat(root: &Path) -> String {
    let files: HashMap<String, String> = crate::repo::capped_source_files(root)
        .into_iter()
        .map(|sf| (sf.path, sf.contents))
        .collect();
    build_source_concat(&files)
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

/// The roles that are shown source in REVIEW mode, in the order they are called.
///
/// [VANTA-WHY] One selection serves every role that sees source, so recording a
/// single role's name in the provenance would be a lie in the other roles' rows.
/// The label names the consumers instead: "which roles got this" is the question
/// the artifact is read to answer.
const REVIEW_SOURCE_ROLES: &str = "attacker+auditor+bate+diagnostician";
/// The roles that are shown source in REPAIR mode: the same set, plus the
/// Creator, which is the role that has to decide where to patch.
const REPAIR_SOURCE_ROLES: &str = "creator+attacker+auditor+bate+diagnostician";

/// Select and render the review prompt's source context from the FULL tree.
///
/// [VANTA-WHY] The selection itself lives in `crate::context` — the ONE
/// implementation of the walk, the ranking and the budget, which the harness's
/// context gate reaches through `vanta_m3 --select-context --json`. This helper
/// is only the seam: walk the tree the patch applies to, select for review, and
/// log what was chosen so a reader can audit the prompt without re-running it.
fn select_review_prompt(
    root: &Path,
    touched: &BTreeSet<String>,
    budget_tokens: u32,
    issue_text: &str,
) -> crate::context::ContextSelection {
    let (candidates, skipped) = crate::context::walk_candidates(root);
    let mut selection = crate::context::select_review_context(
        &candidates,
        touched,
        REVIEW_SOURCE_ROLES,
        budget_tokens,
        crate::context::estimate_tokens(issue_text.len()),
    );
    selection.skipped = skipped;

    println!(
        "\n▌ REVIEW CONTEXT: {} of {} candidate files, {} tok of a {} tok budget",
        selection.files.len(),
        candidates.len(),
        selection.total_tokens,
        selection.budget_tokens,
    );
    for line in selection.summary_lines() {
        println!("{line}");
    }
    if !selection.skipped.is_empty() {
        println!("  (pruned {} path(s); see the session log)", selection.skipped.len());
    }
    selection
}

/// Select and render the repair prompt's source context from the FULL tree.
///
/// [VANTA-WHY] Same seam as `select_review_prompt`, but ranked from the issue
/// text: nothing else is an input, so this cannot read the fix commit, the fix
/// diff, `FAIL_TO_PASS` or `bug.info` even by accident — none of them is a
/// parameter of `crate::context::select_repair_context` and none can be reached
/// from here. Before this, repair mode showed the Creator `repo.source_files`:
/// the first ~262 KB of the repo in `read_dir` order, which on matplotlib
/// (11.8 MB) usually did not contain the bug's own file.
fn select_repair_prompt(
    root: &Path,
    issue_text: &str,
    budget_tokens: u32,
) -> crate::context::ContextSelection {
    let (candidates, skipped) = crate::context::walk_candidates(root);
    let mut selection = crate::context::select_repair_context(
        &candidates,
        issue_text,
        REPAIR_SOURCE_ROLES,
        budget_tokens,
    );
    selection.skipped = skipped;

    println!(
        "\n▌ REPAIR CONTEXT: {} of {} candidate files, {} tok of a {} tok budget",
        selection.files.len(),
        candidates.len(),
        selection.total_tokens,
        selection.budget_tokens,
    );
    for line in selection.summary_lines() {
        println!("{line}");
    }
    if !selection.tree.is_empty() {
        println!(
            "  (plus the repo's file tree, {} paths, {} tok)",
            selection.tree.len(),
            selection.tree_tokens,
        );
    }
    if !selection.skipped.is_empty() {
        println!("  (pruned {} path(s); see the session log)", selection.skipped.len());
    }
    selection
}

/// The Creator's `source_files`, from a selection: the selected files in score
/// order, then — repair mode only — the repo's file tree as one listing entry.
///
/// [VANTA-WHY] `CreatorRequest` carries `Vec<SourceFile>` rather than a rendered
/// prompt, so the tree has to travel as an entry. It is labelled as a listing,
/// not as a path, so the Creator cannot mistake it for a file it can patch. The
/// Attacker, the auditor, BATE and the Diagnostician get the same tree through
/// `ContextSelection::render`, so every role that sees source sees it.
fn creator_source_files(selection: &crate::context::ContextSelection) -> Vec<SourceFile> {
    let mut files: Vec<SourceFile> = selection
        .files
        .iter()
        .map(|file| SourceFile {
            path: file.path.clone(),
            contents: file.body.clone(),
        })
        .collect();
    if !selection.tree.is_empty() {
        files.push(SourceFile {
            path: "REPOSITORY FILE TREE (paths only)".to_string(),
            contents: selection.tree.join("\n"),
        });
    }
    files
}

// [VANTA-WHY] The python import resolution that used to live here
// (`python_module_to_path`, `python_module_path`, `is_test_path`,
// `imported_repo_paths`, `resolve_import_base`, `parse_imported_names`,
// `test_covers_modules`) now lives in `crate::context`, next to the walk that
// feeds it. Two copies of "what does this file import" is exactly the drift the
// context gate exists to catch, so there is one.

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

/// Up to this many regressed nodeids are named in full in the Creator feedback.
///
/// [VANTA-WHY] A patched tree that collects NOTHING makes every nodeid the base
/// tree passed a regression, so on a very large suite (pandas/1's ~88,642 tests)
/// the regression list is the whole suite. Naming every one of them puts tens of
/// thousands of nodeids into `prior_feedback`, the next Creator prompt then fails
/// `prompt_over_budget_reason`, and the run ends as an `InfraFailure` instead of
/// letting the Creator retry — the feedback meant to enable the retry is what
/// kills it. At or below this many the message is byte-identical to the one the
/// unbounded code produced; above it the list is bounded (see the sample size).
const REGRESSION_FEEDBACK_LIMIT: usize = 50;
/// How many nodeids the bounded (`> REGRESSION_FEEDBACK_LIMIT`) message names.
const REGRESSION_FEEDBACK_SAMPLE: usize = 20;

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
        let total = native_regressions.len();
        fb.push_str("- The repository's own tests regressed (run with the ORIGINAL test files):\n");
        if total <= REGRESSION_FEEDBACK_LIMIT {
            // Byte-identical to the unbounded message: at or below the limit every
            // nodeid is named, in the suite's own order, with its failure output.
            for nodeid in native_regressions {
                fb.push_str(&format!("  - {nodeid}\n"));
                if let Some(output) = native_patched.failure_output(nodeid) {
                    fb.push_str(&format!("    failure: {output}\n"));
                }
            }
        } else {
            // [VANTA-WHY] Above the limit the list is bounded: a sorted sample of
            // `REGRESSION_FEEDBACK_SAMPLE` nodeids, the true total, and how many are
            // left to the evidence artifact (which keeps the COMPLETE list). Sorting
            // makes the sample a stable prefix of the suite rather than whatever
            // order the report parser happened to emit, so two runs that regress the
            // same suite name the same 20.
            //
            // [VANTA-WHY] Finding D (PR #4 review) — the sample is taken in TWO
            // groups, each sorted: nodeids the patched report actually recorded a
            // `failure_output` for come FIRST, then the absent ones fill the
            // remaining slots. A partially running patched suite (say 3 tests fail on
            // assertion while 4,000 others are simply absent) would otherwise spend
            // the whole sample on absent nodeids that sort before the failures,
            // handing the Creator neither the one real assertion failure nor a
            // collapsed suite's stderr. `with_output` preserves the suite order of the
            // first group, `without_output` of the second.
            let mut sorted: Vec<&String> = native_regressions.iter().collect();
            sorted.sort();
            let (with_output, without_output): (Vec<&String>, Vec<&String>) = sorted
                .into_iter()
                .partition(|nodeid| native_patched.failure_output(nodeid).is_some());
            for nodeid in with_output
                .iter()
                .chain(without_output.iter())
                .take(REGRESSION_FEEDBACK_SAMPLE)
            {
                fb.push_str(&format!("  - {nodeid}\n"));
                if let Some(output) = native_patched.failure_output(nodeid) {
                    fb.push_str(&format!("    failure: {output}\n"));
                }
            }
            fb.push_str(&format!(
                "  - {total} tests regressed; showing {REGRESSION_FEEDBACK_SAMPLE} sampled \
                 (failures first, then sorted), {} more in the evidence artifact.\n",
                total - REGRESSION_FEEDBACK_SAMPLE
            ));
        }
        // [VANTA-WHY] A patched side that collected nothing has no `failure_output`
        // for any nodeid — the regressed nodeids are simply ABSENT from its report —
        // so the regression list alone cannot say WHY the suite collapsed. The
        // patched run's stderr tail is that diagnostic (import error, collection
        // error, crash), so it is added whenever the patched side collected nothing,
        // bounded to the same tail length the harness already uses.
        if native_patched.collected_nothing() {
            let tail = harness::stderr_tail(&native_patched.stderr_tail);
            if !tail.is_empty() {
                fb.push_str(&format!(
                    "  The patched suite collected no tests; its stderr tail: {tail}\n"
                ));
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

/// [VANTA-WHY] v1-m3.2 item B — whether a failed structured model call failed on
/// the MODEL'S OWN OUTPUT rather than on the machine.
///
/// A schema parse error (`llm::PARSE_FAILURE_PREFIX`) or an answer the server cut
/// off twice (`llm::TRUNCATED_TWICE_MARKER`) is a fact about what the model
/// returned: the call was made, the server answered, and the answer could not be
/// read. Every OTHER reason a structured call fails — an unreachable server, a
/// refused connection, a 5xx after the client's own retries, a `prompt_over_budget`
/// refusal — is a fact about the machine or the run's configuration, and stays
/// `InfraFailure`. This is the ONE predicate the three arms (Creator, Attacker,
/// Judge) share, so the class cannot drift between them.
fn is_model_output_failure(message: &str) -> bool {
    message.contains(crate::llm::PARSE_FAILURE_PREFIX)
        || message.contains(crate::llm::TRUNCATED_TWICE_MARKER)
}

/// [VANTA-WHY] M3 item F5 — whether a Creator prompt can fit the run's context
/// window, priced exactly as the run-time guard prices it
/// (`llm::context_budget_exceeded`: prompt + completion + chat overhead > window).
/// `Some(reason)` when it cannot, `None` when it can.
///
/// [VANTA-WHY] Why this exists when the guard already refuses: the guard's refusal
/// reaches the driver as an ordinary `Err` from the Creator call and was reported
/// as `Creator LLM call failed: …` — the same row an unreachable server or a
/// refused connection produces. A prompt that cannot fit is a fact about the RUN'S
/// CONFIGURATION (a window too small for the file-selection budget plus one
/// attempt's patch), not about the candidate, and the reader acts on it
/// differently. Naming it `prompt_over_budget` — before the call spends a token —
/// is what makes it its own reason.
///
/// [VANTA-WHY] For the window the batch runs against this can never fire: see
/// `context::tests::the_selection_budget_leaves_room_for_the_largest_prompt_the_guard_allows`,
/// where the budget, one whole patch, the frame and this call's completion still
/// fit with 29,768 tokens to spare. The selection is deliberately NOT shrunk to
/// make room — the K arms must see the same context as the normal arms — so this
/// check exists to NAME a window that does break that invariant rather than to
/// change what the Creator is shown.
///
/// `prompt_tokens` is the caller's conservative estimate of everything the call
/// sends (system + user). The guard's own estimate also prices its role wrappers,
/// so this is a lower bound by a handful of tokens; the caller therefore names the
/// guard's refusal too (see the `Err` arm of the Creator call).
fn prompt_over_budget_reason(
    prompt_tokens: u32,
    output_tokens: u32,
    context_window_tokens: u32,
) -> Option<String> {
    let cost = prompt_tokens
        .saturating_add(output_tokens)
        .saturating_add(crate::context::CHAT_OVERHEAD_TOKENS);
    (cost > context_window_tokens).then(|| {
        format!(
            "prompt_over_budget: prompt_tokens≈{prompt_tokens} + max_tokens={output_tokens} \
             + overhead={} > context_window={context_window_tokens} — the file-selection budget \
             and the previous attempt's patch do not fit this window; the selection is not shrunk \
             to make room, because the K arms must see the same context as the normal arms",
            crate::context::CHAT_OVERHEAD_TOKENS,
        )
    })
}

// [VANTA-WHY] M3 item 9 — the feedback a PASSING attempt hands to the next attempt
// under `--fixed-rounds`. The ordinary `Continue` is driven by failures, and
// `build_creator_feedback` builds its message from a failure list; on a passing
// attempt that list is empty, so reusing it would tell the Creator "the previous
// patch failed verification" and then name nothing. The option's own instruction
// is different in kind, so it gets its own words: the suite passed, the run will
// spend another attempt regardless, and the Creator should improve the patch if
// it can and return it UNCHANGED if it cannot. Saying "return it unchanged" is
// the point — without it a model asked for another attempt will invent a change,
// and the EIR this arm exists to measure would be an artefact of the prompt.
fn build_fixed_round_pass_feedback(
    passed_names: &[String],
    patch_text: &str,
    attempt: u32,
    loop_attempts: u32,
) -> String {
    let mut fb = String::new();
    fb.push_str(&format!(
        "Your previous patch PASSED the test suite (attempt {attempt}/{loop_attempts}):\n\n"
    ));
    for name in passed_names {
        fb.push_str(&format!("- Test '{name}' passed\n"));
    }
    // [VANTA-WHY] M3 item 9 — the Creator is shown the patch it just returned.
    // Without it "review your patch and improve it if needed, or return it
    // unchanged" is undecidable: the model cannot compare a patch it cannot see,
    // so the b1-k4/b3-k4 arms would measure prompt confusion rather than
    // willingness to return a working patch unchanged.
    fb.push_str(&format!("\nYour patch, as applied:\n\n{patch_text}\n\n"));
    fb.push_str(&format!(
        "This run always runs {loop_attempts} attempt(s), so you get another one regardless.\n\
         Review your patch above against the issue: if it can be improved — a more robust fix, an edge \
         case it misses, a clearer or smaller change — return an improved patch. If it is already \
         correct and complete, return the SAME patch unchanged. Returning it unchanged is a valid \
         and expected answer; do not change working code only to have something to say."
    ));
    fb
}

/// Whether THIS attempt's patch is byte-identical to the one the previous
/// attempt returned — M3 item 9's "return it unchanged", recorded rather than
/// left to a reader re-hashing the attempt records. `false` on the first
/// attempt, which by definition repeated nothing.
fn fixed_round_patch_is_unchanged(previous_patch: Option<&str>, current_patch: &str) -> bool {
    previous_patch
        .map(|previous| {
            stable_patch_fingerprint(previous) == stable_patch_fingerprint(current_patch)
        })
        .unwrap_or(false)
}

/// Open this attempt's `fixed_round_records` entry the moment the Creator has
/// answered, and advance the comparison basis in the same step.
///
/// [VANTA-WHY] F2 — the two acts are deliberately fused, and both happen BEFORE
/// any apply check. `previous_patch_text` must name the previous CREATOR
/// RESPONSE whether or not that response applied: with A, then B (which fails to
/// apply), then A, the third attempt repeats the FIRST patch, but the basis is
/// B, so its `unchanged` is `false` — comparing against a two-attempt-old patch
/// would have read `true`. And the entry is created here rather than after the
/// graded run so an attempt whose patch never applies still owns a record: a
/// K=4 run always shows 4, with `applied: false` on the ones that never landed.
/// The returned index is how the caller fills in `applied`/`passed`/`tests`
/// once the attempt has actually been graded.
fn open_fixed_round_record(
    records: &mut Vec<serde_json::Value>,
    previous_patch_text: &mut Option<String>,
    attempt_index: u32,
    patch_text: &str,
    feedback: Option<String>,
    creator_prompt_hash: Option<String>,
) -> usize {
    let unchanged = fixed_round_patch_is_unchanged(previous_patch_text.as_deref(), patch_text);
    *previous_patch_text = Some(patch_text.to_string());
    records.push(serde_json::json!({
        "attempt": attempt_index + 1,
        "applied": false,
        "passed": false,
        "patch": patch_text,
        "tests": Vec::<String>::new(),
        "feedback": feedback,
        "creator_prompt_hash": creator_prompt_hash,
        "unchanged": unchanged,
    }));
    records.len() - 1
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
        // [VANTA-WHY] M3 item E2 — index the patched report once; the previous
        // `patched.tests.iter().find(...)` ran a full scan per original test.
        let patched_index = harness::native_results_index(&patched.tests);
        for (nodeid, original_result) in &original.tests {
            if *original_result != TestResult::Passed {
                continue;
            }
            // A vanished test is a regression (`native_suite_collapsed` also
            // names it), so "absent on the patched tree" is a failure here too.
            let outcome = match patched_index.get(nodeid.as_str()).copied() {
                Some(TestResult::Passed) => TestResult::Passed,
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

// [VANTA-WHY] M3 item 9 — does this decision have to be turned back into a
// `Continue` because the run was asked for a FIXED number of rounds? In REPAIR
// mode `AllTestsPassed` is the loop's only early exit (a failing attempt already
// spends its budget and continues), so a fixed-rounds run that passes at attempt
// 1 would stop there and produce no transition at all — which is exactly why
// there is no EIR data today. True only for a passing attempt that is NOT the
// last: at attempt K the verdict is kept, so what the run reports is attempt K's
// own outcome and nothing is suppressed. A `None` (the default) makes this
// unconditionally false, so a run without the option is byte-identical.
fn fixed_round_continues(
    fixed_rounds: Option<u32>,
    decision: &GovernorDecision,
    attempt: u32,
    loop_attempts: u32,
) -> bool {
    fixed_rounds.is_some()
        && *decision == GovernorDecision::AllTestsPassed
        && attempt + 1 < loop_attempts
}

// [VANTA-WHY] M3 item 9 — read `--fixed-rounds` against the KIND of run it is
// attached to, returning the K the loop should actually use and whether a
// request had to be refused.
//
// Two readings live here. A K of zero (or no option at all) is not a
// fixed-rounds run: `--fixed-rounds 0` would ask for "no attempts", which is not
// a run, so it is read as absent rather than honoured. And `--fixed-rounds` is a
// REPAIR option — "the loop always runs K attempts". In `--patch` review mode
// `max_attempts` is ATTACK DEPTH against a candidate that cannot change, so
// "always run K rounds" is a statement about a different loop and honouring it
// there would silently change how deep a review attacks. It is REFUSED, not
// reinterpreted: the caller gets `refused = true` and says so out loud.
fn resolve_fixed_rounds(
    fixed_rounds: Option<u32>,
    is_patch_review: bool,
) -> (Option<u32>, bool) {
    let fixed_rounds = fixed_rounds.filter(|k| *k > 0);
    if fixed_rounds.is_some() && is_patch_review {
        return (None, true);
    }
    (fixed_rounds, false)
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

/// Build the §14.2 corroboration evidence for one round (§14.2, item 1).
///
/// [VANTA-WHY] The two facts a `FailureEvidence` needs that are NOT on
/// `TestEvidence` are supplied here and nowhere else: the exception the failure
/// actually raised (nested inside the Diagnostician's `structured_payload`, at
/// `structured_payload["exception_type"]`) and the auditor's answer about the
/// issue (`issue_requires_change_by_test`). `bate_validated` is `true` for every
/// failure by construction — `attempt_evidence` is built from `tests_to_run`,
/// and every entry there has a BATE baseline (validated this attempt, or
/// replayed from disk having been validated when first generated).
fn failure_evidence_for(
    evidence: &[TestEvidence],
    attempt: u32,
    issue_requires_change_by_test: &HashMap<String, Option<bool>>,
) -> Vec<FailureEvidence> {
    evidence
        .iter()
        .filter(|e| e.failure_class.is_some())
        .map(|e| FailureEvidence {
            test_name: e.test_name.clone(),
            attempt,
            intent: e.intent.clone(),
            failure_class: e.failure_class.clone(),
            exception_type: e
                .structured_payload
                .as_ref()
                .and_then(|payload| payload.get("exception_type"))
                .and_then(|value| value.as_str())
                .map(|s| s.to_string()),
            bate_validated: true,
            issue_requires_change: issue_requires_change_by_test
                .get(&e.test_name)
                .copied()
                .flatten(),
        })
        .collect()
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

/// Render one of §14.3's three robustness switches the way the operator set it.
///
/// [VANTA-WHY] §14.3 records them as `on|off`, the spelling
/// `VANTA_CORROBORATE=on` uses, so `run.summary` and the command that produced
/// the run can be read against each other without translating `true` back into a
/// flag name.
fn flag_text(enabled: bool) -> &'static str {
    if enabled { "on" } else { "off" }
}

/// M3 item 1 — the `NeedsReview` reason a withheld approval carries.
///
/// [VANTA-WHY] `run.summary` is parsed as space-free `key=value` pairs, so the
/// reason has to be one token; these constants are the one place they are
/// spelled, and both the console line and the row read them, so the printed
/// reason and the recorded one cannot drift. There are two withheld-approval
/// reasons because there are two distinct failures of evidence, and a reader
/// triaging a row must be able to tell them apart:
///   * `no_admitted_tests` — nothing was admitted at all, so the approval would
///     rest on literally no test. Nothing ran, so there is no signal to reason
///     about; the Attacker or the admission gate, not a weak probe, is what to
///     look at.
///   * `no_reproduction_evidence` — tests WERE admitted and ran, but none of
///     them expects the original tree to fail. The candidate is unpinned: it
///     broke nothing the surviving probes check, which a no-op also achieves.
const NO_REPRODUCTION_EVIDENCE: &str = "no_reproduction_evidence";
const NO_ADMITTED_TESTS: &str = "no_admitted_tests";
/// M3 item 7 — the invariant's reason on a judge arm: no test runs, so the
/// approval-evidence requirement is inapplicable rather than unmet.
const JUDGE_ARM: &str = "judge_arm";

/// The invariant's reason when an approval IS backed by the required evidence.
const INVARIANT_SATISFIED: &str = "satisfied";

/// Does any admitted test that RAN on the candidate expect the ORIGINAL tree to
/// fail?
///
/// [VANTA-WHY] M3 item 1 — `BugReproduction` and `FeatureAcceptance` are the two
/// intents whose BATE contract demands a failure on the unpatched tree
/// (`bate::expected_baseline`), i.e. the two labels under which a PASSING
/// candidate is evidence that the issue is fixed. `BehaviorPreservation` asserts
/// the opposite — that nothing changed — so a set of preservation-only tests
/// that all pass is exactly the shape a no-op produces, and it must not be
/// enough to approve on. `evidence` is the attempt's graded run, so every entry
/// here was executed against the candidate.
fn expects_baseline_failure(evidence: &[TestEvidence]) -> bool {
    evidence.iter().any(|e| {
        matches!(
            e.intent,
            TestIntent::BugReproduction | TestIntent::FeatureAcceptance
        )
    })
}

/// M3 item 1 — is the approval invariant in force on this configuration, and if
/// so does the attempt's evidence satisfy it?
///
/// [VANTA-WHY] Returns `(applied, reason)`. The invariant constrains only an
/// APPROVAL, and only an approval Vanta could have backed with its own probes,
/// so two configurations are recorded as exempt rather than silently skipped —
/// a reader of the row must be able to see the gate was inapplicable, not infer
/// it:
///   * `not_review`  — a repair run. There is no review approval to constrain.
///   * `attacker_off` — a REVIEW with no Attacker (the §4.2 `baseline` arm). No
///     test was ever admitted, so the requirement could never be met and every
///     baseline review would be withheld. `baseline` is the study's comparison
///     arm and must stay byte-identical, so it is exempt by construction rather
///     than by a value that could drift.
///   * `b2_arm` — a REVIEW under the B2 arm (`VANTA_BATE=off`). B2 admits every
///     Attacker test regardless of declared intent, so "at least one admitted
///     test expects the original to fail" is not a property of its intake: a
///     B2 review whose probes are all mislabelled would be withheld for a reason
///     that is the arm's whole design, not a defect in the candidate. Exempt by
///     construction for the same reason `attacker_off` is — see M3 item 8.
///   * `judge_arm` — the M3 item 7 judge arms. They run no test at all (the whole
///     point of the arm is a verdict from reading, not from evidence), so the
///     requirement is inapplicable rather than unmet, and reporting it as
///     `attacker_off` would name the wrong cause: the judge does not lack an
///     Attacker, it IS the alternative to one.
/// With the invariant in force there are three answers: `satisfied` when at
/// least one admitted test expects the original to fail, `no_reproduction_evidence`
/// when tests ran but none of them does, and `no_admitted_tests` when nothing was
/// admitted at all. The last two are different diagnoses and are recorded
/// differently on purpose — see `NO_ADMITTED_TESTS`.
///
/// [VANTA-WHY] M3 item 8 — the B2 exemption is checked AFTER `not_review` and
/// `attacker_off`, so an existing arm's reason is untouched: a repair run is
/// still `not_review` (it has no approval for the invariant to constrain,
/// whatever the BATE switch says) and a baseline review is still `attacker_off`.
///
/// [VANTA-WHY] M3 item 7 — `judge_arm` is checked SECOND, right after
/// `not_review` and BEFORE `attacker_off`. A judge review has the Attacker off,
/// so without this ordering it would be recorded as `attacker_off` like
/// `baseline`; the ordering is what keeps the two distinguishable, and it cannot
/// change an existing arm's reason because no arm before item 7 sets it.
fn approval_invariant_status(
    mode: GovernorMode,
    attacker_enabled: bool,
    bate_enabled: bool,
    judge_arm: bool,
    evidence: &[TestEvidence],
) -> (bool, &'static str) {
    if mode != GovernorMode::Review {
        return (false, "not_review");
    }
    if judge_arm {
        return (false, JUDGE_ARM);
    }
    if !attacker_enabled {
        return (false, "attacker_off");
    }
    if !bate_enabled {
        return (false, "b2_arm");
    }
    if evidence.is_empty() {
        return (true, NO_ADMITTED_TESTS);
    }
    (
        true,
        if expects_baseline_failure(evidence) {
            INVARIANT_SATISFIED
        } else {
            NO_REPRODUCTION_EVIDENCE
        },
    )
}

/// M3 item 1 — renders a grounding-ledger slot for `run.summary`.
///
/// [VANTA-WHY] `null` is not `0`. The ledger is `None` exactly when no grounded
/// audit ran (`VANTA_GROUNDED_AUDIT=off`), and writing `0` there would claim a
/// measurement the run never made — indistinguishable from an audit that ran and
/// found nothing to scrap. `run.summary` is space-free `key=value`, so the two
/// cases have to be two tokens.
fn ledger_text(value: Option<usize>) -> String {
    match value {
        Some(count) => count.to_string(),
        None => "null".to_string(),
    }
}

/// M3 item 1 — how the grounding rule treated this round's audits.
///
/// [VANTA-WHY] Returns `(scrapped, kept_by_issue)`. Every audit record whose
/// `assertions_grounded` is an EXPLICIT `Some(false)` is exactly one of the two:
/// `route_intent` sends a `BugReproduction` test with that answer to
/// `ScrapUngrounded` unconditionally, and item 1's loosening either leaves it
/// scrapped or turns it into `Keep` when the issue itself names what the test
/// asserts. So the counts partition the set and neither can double-count the
/// other. `None` records — the flag off, or a field the model omitted — are
/// counted by neither, which is why a run with `VANTA_GROUNDED_AUDIT=off`
/// reports `null` for both rather than `0`.
fn grounding_counts(checks: &[IntentCheckRecord]) -> (usize, usize) {
    let ungrounded = checks
        .iter()
        .filter(|c| c.assertions_grounded == Some(false))
        .count();
    let scrapped = checks
        .iter()
        .filter(|c| c.assertions_grounded == Some(false) && c.scrapped)
        .count();
    (scrapped, ungrounded - scrapped)
}

/// Print one line per §14.4 ungrounded scrap, with the auditor's reason.
///
/// [VANTA-WHY] A scrap removes a rejection signal, so the one thing that must be
/// checkable on a real run is whether it removed the RIGHT one: a legitimate
/// `BugReproduction` test that happens to name a specific the issue states in
/// prose would be scrapped here and its bug would then have no probe at all. The
/// session log proves it happened (`intent.check[…]` carries `scrapped=true` and
/// the reason), but a batch run's log is thousands of lines; the console line is
/// what makes over-scrapping visible while the dev split is still in front of
/// you. A record whose `assertions_grounded` is `Some(false)` is the only shape
/// this routing can produce, so the filter cannot drift from `route_intent`.
fn announce_ungrounded_scraps(checks: &[IntentCheckRecord]) {
    for check in checks
        .iter()
        .filter(|c| c.scrapped && c.assertions_grounded == Some(false))
    {
        println!(
            "      ⚖ UNGROUNDED: {} scrapped — {}",
            check.test_name, check.reason
        );
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
                "source={source} declared={:?} issue_asks_for_behaviour={} issue_requires_change={} assertions_grounded={} scrapped={} quote_present={} quote_in_issue={} quote_names_asserted_input={} asserted_inputs=[{}] reason={}",
                check.declared_intent,
                check.issue_asks_for_behaviour,
                tri_state_text(check.issue_requires_change),
                tri_state_text(check.assertions_grounded),
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
        apply_patch_files_all_or_nothing,
        ApplyFailure,
        approval_invariant_status, attempt_verdict, bate_baseline_concat, benchmark_config_name,
        blocking_static_findings,
        build_creator_feedback,
        build_fixed_round_pass_feedback,
        build_source_concat, cache_hit_rate_value, collapse_decision, creator_effort_text,
        creator_prompt_tokens_text,
        dependency_freeze_summary_value, env_arm, env_flag, fixed_round_continues,
        fixed_round_patch_is_unchanged,
        harness_error_log_line,
        harness_infrastructure_collapse,
        is_judge_site,
        is_model_output_failure,
        ledger_text,
        modified_test_config_records, native_gate_cap, NativeGateFacts, native_setup_failure,
        native_vacuous_exit,
        native_vacuous_is_infrastructure, native_timeout_flag, new_files_from_patch,
        open_fixed_round_record,
        patch_attempt_id_for_patch,
        prompt_over_budget_reason,
        record_new_static_findings,
        refuse_reasoning_budget_on_judge_arm,
        rejection_signal_message, render_committed_test_file, require_clean_target_repo_for_open_pr,
        resolve_arm, resolve_fixed_rounds, resolve_native_timeout_secs, sampling_not_sent_rule,
        served_model_version_value,
        final_patch_tree,
        stamp_transition_row,
        stable_patch_fingerprint, ARMS,
        INVARIANT_SATISFIED,
        JUDGE_ARM, NO_ADMITTED_TESTS, NO_REPRODUCTION_EVIDENCE,
        REGRESSION_FEEDBACK_SAMPLE,
    };
    use crate::agents::attacker::GeneratedTest;
    use crate::config::{LlmBackend, LlmConfig};
    use crate::llm::{CallUsage, ShapedBody};
    use crate::core::{
        AttemptTransition, EvidenceSource, FailureClass, GovernorDecision, TestEvidence,
        TestIntent, TestResult, VerificationOutcome,
    };
    use crate::pipeline::governor::{AttemptVerdict, GovernorMode};
    use crate::pipeline::harness::StaticDiagnostic;
    use crate::session::{PatchAttemptId, SessionId};
    use std::collections::HashMap;

    /// §14.32 — the rate, and the three "no rate exists" cases kept apart from a
    /// measured zero.
    #[test]
    fn the_cache_hit_rate_is_unset_when_no_rate_exists_and_never_a_guessed_zero() {
        // 314 of 777 is what the stub publishes; four decimals, fixed point, so
        // two runs' rows compare as text and no `e` reaches a `key=value` row.
        assert_eq!(cache_hit_rate_value(Some(314), 777), "0.4041");
        // A measured zero IS a rate: the server said it served nothing from cache.
        assert_eq!(cache_hit_rate_value(Some(0), 777), "0.0000");
        // Nothing reported cached tokens at all (a server that says nothing about
        // its cache) is NOT that claim.
        assert_eq!(cache_hit_rate_value(None, 777), "unset");
        // A zero prompt total would divide to `NaN`, which must never be printed.
        assert_eq!(cache_hit_rate_value(Some(0), 0), "unset");
        // A whole prompt served from cache is a rate of one, not a special case.
        assert_eq!(cache_hit_rate_value(Some(500), 500), "1.0000");
    }

    /// M3 item B — the run-level `model_version` is DERIVED, not guessed: one
    /// snapshot when every call that named one agrees, the sorted `[a;b]` list
    /// when they disagree (an alias rolled mid-run), and `unknown` only when no
    /// call named one. A value that collapsed a disagreement to the first would
    /// hide exactly the drift the per-call rows exist to show.
    #[test]
    fn the_run_model_version_is_collapsed_from_the_per_call_facts_never_invented() {
        // A call with no served model leaves nothing behind — it must not read as
        // agreement with a named one.
        assert_eq!(served_model_version_value(&[]), "unknown");

        assert_eq!(
            served_model_version_value(&[call_with_served(Some("gpt-5-2025-10-06"))]),
            "gpt-5-2025-10-06"
        );
        // Two calls naming the SAME snapshot collapse to it, once.
        assert_eq!(
            served_model_version_value(&[
                call_with_served(Some("gpt-5-2025-10-06")),
                call_with_served(Some("gpt-5-2025-10-06")),
            ]),
            "gpt-5-2025-10-06"
        );
        // Two DIFFERENT snapshots are the `[a;b]` list, sorted, not the first.
        assert_eq!(
            served_model_version_value(&[
                call_with_served(Some("gpt-5-2025-11-01")),
                call_with_served(Some("gpt-5-2025-10-06")),
            ]),
            "[gpt-5-2025-10-06;gpt-5-2025-11-01]"
        );
        // A silent call beside a named one does not erase the named one, and does
        // not turn it into a list.
        assert_eq!(
            served_model_version_value(&[
                call_with_served(None),
                call_with_served(Some("gpt-5-2025-10-06")),
            ]),
            "gpt-5-2025-10-06"
        );
        // A model id with a space is encoded space-free so the `key=value` reader
        // cannot truncate it.
        assert_eq!(
            served_model_version_value(&[call_with_served(Some("local model v2"))]),
            "local%20model%20v2"
        );
    }

    /// A `CallUsage` carrying only what `served_model_version_value` reads — the
    /// rest are the type's own defaults for this purpose.
    fn call_with_served(model: Option<&str>) -> CallUsage {
        CallUsage {
            seq: 0,
            site: "test".to_string(),
            round: 1,
            attempt: 1,
            provider_retries: 0,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            cached_tokens: None,
            reasoning_tokens: None,
            cost_usd: None,
            cost_basis: "no_usage".to_string(),
            source: "estimate".to_string(),
            seed: None,
            seed_mechanism: "unset".to_string(),
            finish_reason: None,
            wall_ms: 0,
            truncated: false,
            request_shape: ShapedBody { requested: Vec::new(), sent: Vec::new(), dropped: Vec::new(), renamed: Vec::new() },
            service_tier: "unset".to_string(),
            served_model: model.map(str::to_string),
        }
    }

    /// M3 item E — the tree a terminal verdict with no tree of its own writes
    /// `final.patch` from: the last tree the run PRODUCED when there was one, and
    /// the original tree when nothing ever applied. The second half is the point:
    /// a run whose only attempt was malformed must still be graded (an empty patch
    /// that does not fix the bug is a measured failure) rather than censored to
    /// `unobservable`.
    #[test]
    fn the_final_patch_falls_back_to_the_last_produced_tree_then_the_original() {
        let original: HashMap<String, String> = HashMap::from([
            ("pkg/mod.py".to_string(), "value = 1\n".to_string()),
            ("pkg/tests/test_mod.py".to_string(), "def test_x():\n    pass\n".to_string()),
        ]);
        // Nothing ever applied: the fallback is the original tree, i.e. the empty
        // diff — never a panic and never a missing patch.
        let none = final_patch_tree(None, &original);
        assert_eq!(none, original, "with no produced tree the original is the tree");

        // One attempt applied, then the run ended on a later failed apply: the
        // LAST produced tree is kept, not the original and not the first.
        let applied_last: HashMap<String, String> = HashMap::from([
            ("pkg/mod.py".to_string(), "value = 2\n".to_string()),
            ("pkg/tests/test_mod.py".to_string(), "def test_x():\n    pass\n".to_string()),
        ]);
        let kept = final_patch_tree(Some(&applied_last), &original);
        assert_eq!(kept, applied_last, "the last produced tree is what is written");
        assert_ne!(kept, original, "it is not silently reset to the original");
    }

    // ---- M3 item F5: the prompt-over-budget reason ----------------------------

    /// F5 — the check prices a Creator prompt exactly as the run-time guard does,
    /// so the two cannot disagree about whether a prompt fits; when it does not
    /// fit, the reason names itself instead of reading as a failed call.
    #[test]
    fn a_prompt_that_cannot_fit_is_reported_as_prompt_over_budget() {
        // The configured window's worst case — the budget spent to the last token
        // plus one whole previous patch plus the frame (the F5 invariant): no reason.
        assert!(prompt_over_budget_reason(163_584 + 32_768 + 3_000, 32_768, 262_144).is_none());

        // Exactly ON the guard's boundary is NOT over it: the guard refuses `>`.
        // boundary = n_ctx − output reserve − overhead = 262144 − 32768 − 256.
        assert!(prompt_over_budget_reason(262_144 - 32_768 - 256, 32_768, 262_144).is_none());

        // One token past it: its own reason, carrying the numbers the reader acts on.
        let reason = prompt_over_budget_reason(262_144 - 32_768 - 256 + 1, 32_768, 262_144)
            .expect("one token past the guard's boundary must be named");
        assert!(reason.starts_with("prompt_over_budget: "), "{reason}");
        assert!(reason.contains("prompt_tokens≈229121"), "{reason}");
        assert!(reason.contains("max_tokens=32768"), "{reason}");
        assert!(reason.contains("context_window=262144"), "{reason}");
        assert!(
            !reason.contains("Creator LLM call failed"),
            "it is its own reason, not the generic call failure: {reason}"
        );

        // A window too small for the reserve is named whatever the prompt holds —
        // the check is a comparison, not a threshold on the prompt alone.
        let small = prompt_over_budget_reason(0, 32_768, 32_768)
            .expect("a window smaller than the reserve cannot fit any prompt");
        assert!(small.contains("context_window=32768"), "{small}");
    }

    // ---- M3 item 9: fixed-round repair (`--fixed-rounds K`) -------------------
    //
    // [VANTA-WHY] The option exists so the attempt-to-attempt transitions M3.5
    // measures exist at all: today a repair run stops the moment an attempt
    // passes, so there is no transition into a passing attempt to look at. These
    // tests pin the three pure readings the loop is built from — when a pass is
    // turned back into a `Continue`, what the Creator is told when that happens,
    // and what the option means under `--patch`.

    /// The turn-back rule. A pass continues ONLY when the option is on AND the
    /// attempt is not the last, so nothing is suppressed where the run reports
    /// its verdict.
    #[test]
    fn a_fixed_rounds_run_runs_past_a_pass_but_never_past_the_last_attempt() {
        // Off (the default): a pass is always the end of the loop.
        assert!(!fixed_round_continues(
            None,
            &GovernorDecision::AllTestsPassed,
            0,
            4
        ));
        // K = 4: a pass at attempt 1 or 3 continues...
        assert!(fixed_round_continues(
            Some(4),
            &GovernorDecision::AllTestsPassed,
            0,
            4
        ));
        assert!(fixed_round_continues(
            Some(4),
            &GovernorDecision::AllTestsPassed,
            2,
            4
        ));
        // ...but a pass at attempt 4 (index 3) IS the verdict, so it is kept.
        assert!(!fixed_round_continues(
            Some(4),
            &GovernorDecision::AllTestsPassed,
            3,
            4
        ));
        // Only a pass is turned back. Every other decision is the loop's own
        // business and must keep the reading it made.
        for other in [
            GovernorDecision::Rejected,
            GovernorDecision::NeedsReview,
            GovernorDecision::MaxAttemptsReached,
            GovernorDecision::Continue,
            GovernorDecision::InfraFailure,
        ] {
            assert!(
                !fixed_round_continues(Some(4), &other, 0, 4),
                "{other:?} must not be rewritten as a fixed-round continuation"
            );
        }
    }

    /// The refusal rule: `--fixed-rounds` is a repair option and is refused under
    /// `--patch`, where `max_attempts` is attack depth against a fixed candidate.
    #[test]
    fn fixed_rounds_is_read_against_the_kind_of_run_it_is_attached_to() {
        // Off, and an explicit zero, are both "not a fixed-rounds run".
        assert_eq!(resolve_fixed_rounds(None, false), (None, false));
        assert_eq!(resolve_fixed_rounds(Some(0), false), (None, false));
        // A repair run honours K.
        assert_eq!(resolve_fixed_rounds(Some(4), false), (Some(4), false));
        // A `--patch` review cannot honour it: refused, and the caller is told so
        // it can say it out loud rather than silently attacking to a different
        // depth than the one the option named.
        assert_eq!(resolve_fixed_rounds(Some(4), true), (None, true));
        // Nothing to refuse where the option was not set (or was zero).
        assert_eq!(resolve_fixed_rounds(None, true), (None, false));
        assert_eq!(resolve_fixed_rounds(Some(0), true), (None, false));
    }

    /// The feedback a passing attempt hands on. It has to say the suite PASSED
    /// and that returning the patch unchanged is a valid answer — without the
    /// second half, a model asked for another attempt invents a change and the
    /// EIR this arm measures would be an artefact of the prompt.
    #[test]
    fn the_fixed_round_pass_feedback_says_it_passed_and_permits_an_unchanged_patch() {
        let names = vec!["t_band".to_string(), "t_discount".to_string()];
        let patch = "===FILE=== foo.py\n===SEARCH===\nx = 1\n===REPLACE===\nx = 2\n===END===";
        let fb = build_fixed_round_pass_feedback(&names, patch, 1, 4);
        assert!(
            fb.contains("PASSED the test suite"),
            "the attempt that passed must be told it passed: {fb}"
        );
        assert!(
            fb.contains("attempt 1/4"),
            "which attempt, of how many, so the hand-over is unambiguous: {fb}"
        );
        assert!(
            fb.contains("t_band") && fb.contains("t_discount"),
            "the tests that passed must be named: {fb}"
        );
        assert!(
            fb.contains(patch),
            "M3 item 9: the Creator must be shown the patch it is asked to review, or \
             'return it unchanged' is undecidable: {fb}"
        );
        assert!(
            fb.contains("return the SAME patch unchanged"),
            "returning it unchanged has to be named as valid, or the next patch is invented: {fb}"
        );
        assert!(
            !fb.contains("failed"),
            "a passing attempt must never be told its patch failed: {fb}"
        );
    }

    /// M3 item 9 — the `unchanged` flag on a per-attempt record is decided by
    /// the candidate fingerprint, not by the attempt counter: a Creator that
    /// returns the same patch makes attempt N `unchanged`, and any edit makes it
    /// new. `None` (attempt 1) repeated nothing and must never read as unchanged.
    #[test]
    fn a_repeated_patch_is_recorded_as_unchanged() {
        assert!(
            !fixed_round_patch_is_unchanged(None, "patch A"),
            "the first attempt has no previous patch to repeat"
        );
        assert!(
            fixed_round_patch_is_unchanged(Some("patch A"), "patch A"),
            "a byte-identical re-return is the unchanged answer item 9 asks for"
        );
        assert!(
            !fixed_round_patch_is_unchanged(Some("patch A"), "patch B"),
            "any edit is a new patch, not an unchanged one"
        );
    }

    /// [VANTA-WHY] F2 — the failed-apply sequence that made the old code lie.
    /// The record is opened, and the comparison basis advanced, right after each
    /// Creator response — BEFORE the apply checks — so:
    ///
    /// * an attempt whose patch never applies still owns a record (`applied`
    ///   stays `false`), so a K=4 run always shows 4; and
    /// * attempt 3, which returns patch A again, is compared against attempt 2's
    ///   patch B (the previous REsponse), so it is NOT `unchanged`. Comparing
    ///   against the last patch that APPLIED (A, from attempt 1) is the bug: it
    ///   would have read `true`.
    #[test]
    fn every_creator_response_opens_its_own_record_and_advances_the_basis() {
        let mut records: Vec<serde_json::Value> = Vec::new();
        let mut previous: Option<String> = None;

        // A K=4 run: attempt 1 returns A; attempt 2 returns B and (in production)
        // fails to apply, so it is never completed; attempt 3 returns A; attempt
        // 4 returns A again.
        open_fixed_round_record(&mut records, &mut previous, 0, "patch A", None, Some("h1".into()));
        open_fixed_round_record(
            &mut records,
            &mut previous,
            1,
            "patch B",
            Some("fb-1".into()),
            Some("h2".into()),
        );
        open_fixed_round_record(
            &mut records,
            &mut previous,
            2,
            "patch A",
            Some("fb-2".into()),
            Some("h3".into()),
        );
        open_fixed_round_record(&mut records, &mut previous, 3, "patch A", None, Some("h4".into()));

        assert_eq!(
            records.len(),
            4,
            "a K=4 run records all four Creator attempts, the one that never applied included"
        );
        assert_eq!(
            records[1]["applied"],
            serde_json::json!(false),
            "attempt 2's patch never applied and its record must say so, not vanish"
        );
        assert_eq!(records[0]["unchanged"], serde_json::json!(false));
        assert_eq!(records[1]["unchanged"], serde_json::json!(false));
        assert_eq!(
            records[2]["unchanged"],
            serde_json::json!(false),
            "attempt 3 repeats attempt 1, but attempt 2 was the previous response and the basis, \
             so it is a NEW patch — the whole point of advancing the basis on a failed apply"
        );
        assert_eq!(
            records[3]["unchanged"],
            serde_json::json!(true),
            "attempt 4 repeats attempt 3, which is genuinely the previous response"
        );
        assert_eq!(records[2]["feedback"], serde_json::json!("fb-2"));
        assert_eq!(records[2]["creator_prompt_hash"], serde_json::json!("h3"));
        assert_eq!(records[0]["attempt"], serde_json::json!(1));
        assert_eq!(records[3]["attempt"], serde_json::json!(4));
    }

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

    /// [VANTA-WHY] v1-m3.1 (§8.1 row 3) — an EMPTY patched side is a failed
    /// attempt in BOTH modes, and the mode only decides what that costs.
    ///
    /// `native_regression` is the path row 3 takes to the Governor (every
    /// original test is absent from an empty patched report, so every one is a
    /// regression), and `attempt_verdict` rejects on it in any mode. In review
    /// mode that rejection IS the candidate's verdict — a recorded reject, not a
    /// censored `infra_failure`; in repair mode it is a round the Creator gets
    /// back and retries. An in-batch session used to leave this path
    /// entirely and end the row `infra_failure`, dropping the bug out of the
    /// attempt loop, though its patch alone broke collection while the original
    /// ran its full suite.
    #[test]
    fn an_empty_patched_side_fails_the_attempt_in_both_modes() {
        for mode in [GovernorMode::Repair, GovernorMode::Review] {
            assert_eq!(
                attempt_verdict(true, &[], true, false, mode),
                AttemptVerdict::Rejected,
                "row 3 in {mode:?}: a patch that wiped the native suite must reject, \
                 never read as a clean gate nor as an infrastructure fault"
            );
        }
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

    // ---- M3 item A1: Contract B — a multi-file patch is all-or-nothing --------

    /// [VANTA-WHY] M3 item A1 (Contract B). Two `===FILE===` blocks where the
    /// SECOND does not anchor must write NOTHING: the tested tree must never keep
    /// the first block's change while `final.patch` — built from the same map —
    /// records a patch the tree no longer matches. The old per-file loop applied
    /// the first block and continued on the partial tree.
    #[test]
    fn a_multi_file_patch_whose_second_block_does_not_anchor_writes_nothing() {
        let original = HashMap::from([
            ("calc.py".to_string(), "def scale(x):\n    return x\n".to_string()),
            ("util.py".to_string(), "def helper():\n    return 1\n".to_string()),
        ]);
        let patch_text = concat!(
            "===FILE: calc.py===\n",
            "def scale(x):\n    return x * 2\n",
            "===FILE: util.py===\n",
            "===SEARCH===\n    return 999\n===REPLACE===\n    return 1\n===END===\n",
        );
        let (blocks, _trimmed) = crate::agents::creator::split_multi_file_patch(&patch_text, &[]);
        assert_eq!(blocks.len(), 2, "the patch must name two files");

        // The driver's own decision, exactly: on `Err` it writes nothing.
        let (patched, failure) = match apply_patch_files_all_or_nothing(&original, &blocks) {
            Ok(changed) => (changed, None),
            Err(failure) => (original.clone(), Some(failure)),
        };

        let failure = failure.expect("the second block cannot anchor, so the apply must fail");
        assert!(
            failure.feedback().contains("util.py"),
            "the failing block must be named; got: {}",
            failure.feedback()
        );
        assert_eq!(
            patched, original,
            "Contract B: no block may be written when any block fails"
        );
    }

    /// The other half of Contract B: when EVERY block anchors, the returned map is
    /// the whole patched tree — the same map the tests run on and the same map
    /// `final.patch` is diffed from, so the two cannot disagree.
    #[test]
    fn a_multi_file_patch_whose_blocks_all_anchor_applies_every_block() {
        let original = HashMap::from([
            ("calc.py".to_string(), "def scale(x):\n    return x\n".to_string()),
            ("util.py".to_string(), "def helper():\n    return 1\n".to_string()),
        ]);
        let patch_text = concat!(
            "===FILE: calc.py===\n",
            "def scale(x):\n    return x * 2\n",
            "===FILE: util.py===\n",
            "def helper():\n    return 2\n",
        );
        let (blocks, _trimmed) = crate::agents::creator::split_multi_file_patch(&patch_text, &[]);

        let patched =
            apply_patch_files_all_or_nothing(&original, &blocks).expect("both blocks anchor");

        assert!(patched["calc.py"].contains("return x * 2"), "{patched:?}");
        assert!(patched["util.py"].contains("return 2"), "{patched:?}");
        assert_ne!(patched, original);
    }

    // ---- M3 item T1: a block naming an unsafe path fails the whole patch -----

    /// [VANTA-WHY] M3 item T1. The path in a `===FILE:` marker is MODEL text,
    /// taken verbatim at any indentation. An ABSOLUTE path must fail the whole
    /// patch under Contract B with the reason `unsafe_path` — and the innocent
    /// block beside it must not be written either, or the tree under test is a
    /// tree nobody wrote.
    #[test]
    fn a_patch_naming_an_absolute_path_fails_with_unsafe_path_and_writes_nothing() {
        let original = HashMap::from([(
            "calc.py".to_string(),
            "def scale(x):\n    return x\n".to_string(),
        )]);
        let patch_text = concat!(
            "===FILE: calc.py===\n",
            "def scale(x):\n    return x * 2\n",
            "===FILE: /home/user/.ssh/authorized_keys===\n",
            "ssh-rsa AAAA\n",
        );
        let (blocks, _trimmed) = crate::agents::creator::split_multi_file_patch(&patch_text, &[]);
        assert_eq!(blocks.len(), 2, "both blocks must be recognised");

        let (patched, failure) = match apply_patch_files_all_or_nothing(&original, &blocks) {
            Ok(changed) => (changed, None),
            Err(failure) => (original.clone(), Some(failure)),
        };

        let failure = failure.expect("an absolute path must fail the whole patch");
        assert!(
            matches!(failure, ApplyFailure::UnsafePath { .. }),
            "expected UnsafePath; got {failure:?}"
        );
        assert!(
            failure.feedback().starts_with("unsafe_path"),
            "the reason must reach the `patch.apply_failed` row and the Creator's \
             feedback verbatim; got: {}",
            failure.feedback()
        );
        assert!(
            failure.feedback().contains("/home/user/.ssh/authorized_keys"),
            "the refused path must be named; got: {}",
            failure.feedback()
        );
        assert_eq!(
            patched, original,
            "Contract B: no block may be written when any block names an unsafe path"
        );
    }

    /// [VANTA-WHY] M3 item T1. The same rule for a `..` component: a patch that
    /// climbs out of the tree is refused as `unsafe_path`, and the block that
    /// would have written a legitimate file is refused with it.
    #[test]
    fn a_patch_naming_a_parent_escape_fails_with_unsafe_path_and_writes_nothing() {
        let original = HashMap::from([(
            "calc.py".to_string(),
            "def scale(x):\n    return x\n".to_string(),
        )]);
        let patch_text = concat!(
            "===FILE: calc.py===\n",
            "def scale(x):\n    return x * 2\n",
            "===FILE: ../escape.py===\n",
            "evil = True\n",
        );
        let (blocks, _trimmed) = crate::agents::creator::split_multi_file_patch(&patch_text, &[]);
        assert_eq!(blocks.len(), 2, "both blocks must be recognised");

        let (patched, failure) = match apply_patch_files_all_or_nothing(&original, &blocks) {
            Ok(changed) => (changed, None),
            Err(failure) => (original.clone(), Some(failure)),
        };

        let failure = failure.expect("a parent escape must fail the whole patch");
        assert!(
            matches!(failure, ApplyFailure::UnsafePath { .. }),
            "expected UnsafePath; got {failure:?}"
        );
        assert!(
            failure.feedback().starts_with("unsafe_path"),
            "got: {}",
            failure.feedback()
        );
        assert_eq!(
            patched, original,
            "Contract B: the escape fails every block, not just the offending one"
        );
    }

    // ---- Q2: the paths a Creator patch introduces (record only) ----------------

    /// [VANTA-WHY] Q2 — a mistyped marker path (`cartt.py` for a tree that has
    /// `cart.py`) is accepted verbatim by the apply (it looks like a new module the
    /// fix adds), and the pure helper names it so the run can RECORD it. Nothing
    /// refuses it: the point of Q2 is that this is recorded, not policed.
    #[test]
    fn a_mistyped_marker_path_is_recorded_as_a_new_file() {
        let original = HashMap::from([(
            "cart.py".to_string(),
            "def total():\n    return 0\n".to_string(),
        )]);
        // The Creator meant `cart.py` and typed one letter too many.
        let patch_text = "===FILE: cartt.py===\ndef total():\n    return 1\n";
        let (blocks, _trimmed) = crate::agents::creator::split_multi_file_patch(&patch_text, &[]);
        assert_eq!(blocks.len(), 1);
        assert!(blocks.contains_key("cartt.py"));

        // The apply accepts it (the new-file arm) and it does change the tree.
        let patched =
            apply_patch_files_all_or_nothing(&original, &blocks).expect("a new-file block applies");
        assert_ne!(patched, original);

        // The record names the file the patch actually wrote, not the one it meant.
        assert_eq!(
            new_files_from_patch(&original, &blocks),
            vec!["cartt.py".to_string()]
        );
    }

    /// An edit to a file the tree already has is not a created file, and an empty
    /// new-file block adds nothing, so neither is recorded.
    #[test]
    fn an_edited_file_and_an_empty_new_block_are_not_new_files() {
        let original = HashMap::from([(
            "cart.py".to_string(),
            "def total():\n    return 0\n".to_string(),
        )]);
        let blocks = HashMap::from([
            (
                "cart.py".to_string(),
                "def total():\n    return 1\n".to_string(),
            ),
            ("empty.py".to_string(), "   \n".to_string()),
        ]);
        assert!(
            new_files_from_patch(&original, &blocks).is_empty(),
            "an existing file is an edit and a blank block writes nothing"
        );
    }

    /// [VANTA-WHY] A syntax-broken block is refused, and the refusal names the
    /// file — `ApplyFailure::InvalidPython`'s own message is the interpreter's
    /// stderr and carries no path, so Contract B's "name the failing file" would
    /// be lost without the wrapper the helper adds.
    #[test]
    fn a_syntax_breaking_block_names_the_file_in_the_refusal() {
        let original = HashMap::from([(
            "util.py".to_string(),
            "def helper():\n    return 1\n".to_string(),
        )]);
        let blocks = HashMap::from([(
            "util.py".to_string(),
            "def helper(\n    return 1\n".to_string(),
        )]);

        let failure = apply_patch_files_all_or_nothing(&original, &blocks)
            .expect_err("the replacement does not parse as Python");

        assert!(
            failure.feedback().contains("util.py"),
            "the refusal must name the file; got: {}",
            failure.feedback()
        );
        assert!(
            failure
                .feedback()
                .starts_with("patch result is not valid Python:"),
            "and it must still read as a syntax refusal; got: {}",
            failure.feedback()
        );
    }

    /// A block that adds a file the tree does not have is applied verbatim; an
    /// empty new-file block changes nothing (and the driver's `all_patch_failed`
    /// check is what turns "changed nothing" into a failure).
    #[test]
    fn a_new_file_block_is_added_and_an_empty_one_changes_nothing() {
        let original = HashMap::from([(
            "calc.py".to_string(),
            "def scale(x):\n    return x\n".to_string(),
        )]);

        let added = apply_patch_files_all_or_nothing(
            &original,
            &HashMap::from([(
                "brand_new.py".to_string(),
                "def f():\n    return 3\n".to_string(),
            )]),
        )
        .expect("adding a new file is not an anchor failure");
        assert_eq!(added["brand_new.py"], "def f():\n    return 3\n");
        assert!(added.contains_key("calc.py"));

        let empty = apply_patch_files_all_or_nothing(
            &original,
            &HashMap::from([("empty.py".to_string(), String::new())]),
        )
        .expect("an empty new-file block is not an anchor failure");
        assert_eq!(
            empty, original,
            "an empty block changes nothing; the driver reports that as the apply failure"
        );
    }

    /// [VANTA-WHY] Item E4 + Contract B (M3 item A1). A block with one good
    /// search/replace and one malformed block must write NOTHING. Before E4 the
    /// good block applied and the malformed one was skipped, so the write
    /// succeeded — a program the Creator never proposed.
    #[test]
    fn a_block_with_a_good_and_a_malformed_search_replace_writes_nothing() {
        let original = HashMap::from([(
            "calc.py".to_string(),
            "def scale(x):\n    return x\n".to_string(),
        )]);
        let blocks = HashMap::from([(
            "calc.py".to_string(),
            concat!(
                "===SEARCH===\n    return x\n===REPLACE===\n    return x * 2\n===END===\n",
                "===SEARCH===\n    return x\n",
            )
            .to_string(),
        )]);

        let (patched, failure) = match apply_patch_files_all_or_nothing(&original, &blocks) {
            Ok(changed) => (changed, None),
            Err(failure) => (original.clone(), Some(failure)),
        };

        let failure = failure.expect("the malformed block must fail the apply, not be skipped");
        assert!(
            failure.feedback().contains("calc.py"),
            "the failing file must be named; got: {}",
            failure.feedback()
        );
        assert!(
            failure.feedback().contains("malformed"),
            "the malformed block must be named; got: {}",
            failure.feedback()
        );
        assert_eq!(patched, original, "Contract B: nothing may be written");
    }

    /// [VANTA-WHY] Item E5. A block that anchors but produces identical text is
    /// NOT an anchor failure: the file just did not change. The helper returns
    /// `Ok` with the unchanged tree, and the caller's whole-patch check reports
    /// "patch changed nothing". Before E5 this block was rejected here with the
    /// misleading reason "search/replace: produced text identical to the
    /// original" — naming a strategy failure for a block that DID anchor.
    #[test]
    fn a_search_replace_block_that_changes_nothing_is_not_an_anchor_failure() {
        let original = HashMap::from([(
            "calc.py".to_string(),
            "def scale(x):\n    return x\n".to_string(),
        )]);
        let blocks = HashMap::from([(
            "calc.py".to_string(),
            "===SEARCH===\n    return x\n===REPLACE===\n    return x\n===END===\n".to_string(),
        )]);

        let patched = apply_patch_files_all_or_nothing(&original, &blocks)
            .expect("SEARCH == REPLACE anchors; it just changes nothing");
        assert_eq!(
            patched, original,
            "nothing was written; the whole-patch check reports 'patch changed nothing'"
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

    // [VANTA-WHY] The review-prompt selection test moved to `crate::context`
    // together with the selection code it exercises. A test that pokes a private
    // helper here would have kept passing while the prompt the run sends was
    // built somewhere else.

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

    /// One admitted, executed test with the intent under test.
    fn evidence(name: &str, intent: TestIntent) -> TestEvidence {
        TestEvidence {
            test_name: name.to_string(),
            outcome: VerificationOutcome::VerifiedWithinBounds,
            failure_class: None,
            structured_payload: None,
            intent,
            evidence_source: EvidenceSource::AttackerGenerated,
        }
    }

    // [VANTA-WHY] M3 item 1 — "nothing was admitted" is its own withheld-approval
    // answer, not a shade of "nothing pinned the candidate".
    //
    // [VANTA-WHY] This is the one answer the DRIVER cannot reach through an
    // approval today: with the Attacker on, an attempt that admitted nothing
    // (and has no shared set) is routed to `Inconclusive` by the pre-run guard
    // before the graded run, so the `AllTestsPassed` arm is never reached with an
    // empty evidence list; `run.summary` then records `no_approval`, because
    // `MaxAttemptsReached` is not an approval for the invariant to constrain. The
    // guard protects the VERDICT, not the ANSWER. The rule still has to answer for
    // the empty set — it is what `run.summary` reads and what the quarantine path
    // (item 17) will read when quarantine empties the shared set — so the answer
    // is pinned here where it can be checked without a model.
    #[test]
    fn no_admitted_tests_is_a_distinct_withheld_approval_reason() {
        assert_eq!(
            approval_invariant_status(GovernorMode::Review, true, true, false, &[]),
            (true, NO_ADMITTED_TESTS),
            "zero admitted tests must withhold the approval with its own reason"
        );
        assert_ne!(
            NO_ADMITTED_TESTS, NO_REPRODUCTION_EVIDENCE,
            "the two withheld-approval reasons are different diagnoses and a reader \
             triaging a row must be able to tell them apart"
        );
    }

    // [VANTA-WHY] M3 item 1 — every answer the invariant can give, including the
    // configurations where it is INAPPLICABLE. An exemption has to be
    // recorded with its reason rather than silently skipped, or a baseline row
    // would be indistinguishable from a Vanta row that happened to be satisfied.
    //
    // [VANTA-WHY] M3 item 8 — `bate_enabled` is threaded through so a B2 row can
    // carry its own exemption, and the existing reasons are pinned here at
    // `bate_enabled = true` (the only value an existing arm ever has, since the
    // switch defaults on) to prove the new one did not shift them.
    #[test]
    fn the_approval_invariant_names_each_exemption_and_each_answer() {
        // Exemptions: a repair run, and the Attacker-off `baseline` review.
        assert_eq!(
            approval_invariant_status(GovernorMode::Repair, true, true, false, &[]),
            (false, "not_review")
        );
        assert_eq!(
            approval_invariant_status(GovernorMode::Review, false, true, false, &[]),
            (false, "attacker_off")
        );
        // In force and satisfied by a probe that expects the original to fail —
        // under either label, because a feature-request issue is approvable too.
        for intent in [TestIntent::BugReproduction, TestIntent::FeatureAcceptance] {
            let label = format!("{intent:?}");
            assert_eq!(
                approval_invariant_status(
                    GovernorMode::Review,
                    true,
                    true,
                    false,
                    &[evidence("t_pins", intent)],
                ),
                (true, INVARIANT_SATISFIED),
                "{label} expects the original to fail, so it pins the candidate"
            );
        }
        // In force and withheld: preservation-only evidence ran, so a passing set
        // says only that the candidate broke nothing — which a no-op also achieves.
        assert_eq!(
            approval_invariant_status(
                GovernorMode::Review,
                true,
                true,
                false,
                &[evidence("t_pres", TestIntent::BehaviorPreservation)],
            ),
            (true, NO_REPRODUCTION_EVIDENCE)
        );
    }

    // [VANTA-WHY] M3 item 8 — the B2 arm is its OWN exemption with its own
    // reason, not `attacker_off`: the Attacker DID run there, it is the
    // admission gate that was removed. A reader must be able to tell a B2 row
    // from a `baseline` review, and `b2_arm` is what says so. The `not_review`
    // and `attacker_off` exemptions still win where they apply, so a B2 repair
    // run is `not_review` and a B2-shaped `baseline` is `attacker_off`.
    #[test]
    fn the_b2_arm_is_exempt_from_the_invariant_with_its_own_reason() {
        assert_eq!(
            approval_invariant_status(
                GovernorMode::Review,
                true,
                false,
                false,
                &[evidence("t_pins", TestIntent::BugReproduction)],
            ),
            (false, "b2_arm"),
            "a B2 review admits every probe regardless of intent, so the invariant is \
             inapplicable — and it must say so, not read as satisfied"
        );
        assert_eq!(
            approval_invariant_status(GovernorMode::Review, true, false, false, &[]),
            (false, "b2_arm"),
            "the exemption is a property of the arm, not of the evidence it happens to hold"
        );
        // The two pre-existing exemptions keep precedence over the new one.
        assert_eq!(
            approval_invariant_status(GovernorMode::Repair, true, false, false, &[]),
            (false, "not_review"),
            "B2 does not make a repair run into a review"
        );
        assert_eq!(
            approval_invariant_status(GovernorMode::Review, false, false, false, &[]),
            (false, "attacker_off"),
            "with no Attacker there is no B2 intake either; the arm is `baseline`"
        );
    }

    // [VANTA-WHY] M3 item 7 — a judge review reaches its verdict with NO
    // adversarial test at all, so the approval invariant has nothing to
    // constrain: it is exempt under its OWN reason, `judge_arm`, not `attacker_off`
    // (an arm that never runs the Attacker at all) and not a bare `no_approval`
    // (a run that ended without an approval). A reader must be able to tell a
    // judge row from those, and the pre-existing exemptions must still win where
    // they apply, so a judge-shaped REPAIR run is `not_review`.
    #[test]
    fn the_judge_arm_is_exempt_from_the_invariant_with_its_own_reason() {
        assert_eq!(
            approval_invariant_status(GovernorMode::Review, false, true, true, &[]),
            (false, JUDGE_ARM),
            "a judge review runs no adversarial test, so the invariant is inapplicable — \
             and it must say so under its own reason, not read as satisfied"
        );
        assert_ne!(
            JUDGE_ARM, "attacker_off",
            "a judge arm is not the Attacker-off `baseline`; the two must be distinguishable"
        );
        assert_eq!(
            approval_invariant_status(GovernorMode::Repair, false, true, true, &[]),
            (false, "not_review"),
            "a repair run is exempt as a repair run, judge arm or not"
        );
    }

    // [VANTA-WHY] M3 item 8 — the arm-name mapping. `b2` exists so a B2 row can
    // never be mistaken for a `vanta` row, and every other name is unchanged:
    // the new switch only ever fires inside the Attacker-on, non-planner branch,
    // so `vanta_planner` (planner wins first) and `baseline` (no Attacker) are
    // byte-identical to their pre-item-8 spellings.
    #[test]
    fn the_b2_arm_name_fires_only_for_attacker_on_with_bate_off() {
        // Attacker on, BATE off, planner off → the new arm.
        assert_eq!(benchmark_config_name(true, false, false), "b2");
        // Every other combination keeps the pre-item-8 name.
        assert_eq!(benchmark_config_name(true, false, true), "vanta");
        assert_eq!(benchmark_config_name(true, true, true), "vanta_planner");
        assert_eq!(benchmark_config_name(false, false, true), "baseline");
        // With BATE off, the planner and the no-Attacker arm are still themselves:
        // the switch cannot rename a configuration the Attacker does not touch,
        // and it cannot outrank the planner.
        assert_eq!(benchmark_config_name(true, true, false), "vanta_planner");
        assert_eq!(benchmark_config_name(false, false, false), "baseline");
        assert_eq!(benchmark_config_name(false, true, false), "vanta_planner");
    }

    // [VANTA-WHY] M3 item 8 — "new behaviour is OFF by default" has to be a
    // property of the code, not a hope. The pre-item-8 mapping is re-stated here
    // as a fixture and compared against the live one at the value an unset
    // switch produces, so a later edit that renamed an existing arm (or made
    // `b2` win over the planner) fails this test rather than the next batch.
    #[test]
    fn an_unset_bate_switch_reproduces_the_pre_item_8_arm_name_and_invariant() {
        fn pre_item_8_config_name(attacker: bool, planner: bool) -> &'static str {
            if planner {
                "vanta_planner"
            } else if attacker {
                "vanta"
            } else {
                "baseline"
            }
        }

        // M3 item X4 — the crate-wide environment lock: two of the three switch
        // tests below write `VANTA_BATE`, so they are exclusive with each other and
        // with every other test that reads the environment.
        let _env = crate::test_env::lock();
        // An absent switch is the documented default ON, not an error.
        let previous = std::env::var("VANTA_BATE").ok();
        unsafe { std::env::remove_var("VANTA_BATE") };
        let bate_enabled =
            env_flag("VANTA_BATE", true).expect("an absent VANTA_BATE is the default, not an error");

        for attacker in [false, true] {
            for planner in [false, true] {
                assert_eq!(
                    benchmark_config_name(attacker, planner, bate_enabled),
                    pre_item_8_config_name(attacker, planner),
                    "attacker={attacker} planner={planner}: an unset VANTA_BATE must \
                     not rename any existing arm"
                );
            }
        }

        // The invariant fields are unchanged at the default switch too: every
        // pre-change answer still answers the same way.
        assert_eq!(
            approval_invariant_status(GovernorMode::Review, true, bate_enabled, false, &[]),
            (true, NO_ADMITTED_TESTS)
        );
        assert_eq!(
            approval_invariant_status(GovernorMode::Repair, true, bate_enabled, false, &[]),
            (false, "not_review")
        );
        assert_eq!(
            approval_invariant_status(GovernorMode::Review, false, bate_enabled, false, &[]),
            (false, "attacker_off")
        );
        assert_eq!(
            approval_invariant_status(
                GovernorMode::Review,
                true,
                bate_enabled,
                false,
                &[evidence("t_pins", TestIntent::BugReproduction)],
            ),
            (true, INVARIANT_SATISFIED)
        );
        assert_eq!(
            approval_invariant_status(
                GovernorMode::Review,
                true,
                bate_enabled,
                false,
                &[evidence("t_pres", TestIntent::BehaviorPreservation)],
            ),
            (true, NO_REPRODUCTION_EVIDENCE)
        );

        match previous {
            Some(value) => unsafe { std::env::set_var("VANTA_BATE", value) },
            None => unsafe { std::env::remove_var("VANTA_BATE") },
        }
    }

    // [VANTA-WHY] M3 item 8 — `VANTA_BATE` is read with `env_flag` like every
    // other switch, so an unrecognised value is a `CONFIG_ERROR` and never a
    // silent default: `VANTA_BATE=maybe` quietly meaning "on" would record a
    // `vanta` row for a run the operator asked to be `b2`.
    #[test]
    fn the_bate_switch_is_strict_and_defaults_on() {
        // M3 item X4 — the crate-wide environment lock (see the note above).
        let _env = crate::test_env::lock();
        let previous = std::env::var("VANTA_BATE").ok();

        unsafe { std::env::remove_var("VANTA_BATE") };
        assert_eq!(
            env_flag("VANTA_BATE", true).unwrap(),
            true,
            "an absent VANTA_BATE is the default (BATE on), which keeps every existing arm"
        );

        unsafe { std::env::set_var("VANTA_BATE", "off") };
        assert_eq!(env_flag("VANTA_BATE", true).unwrap(), false);

        unsafe { std::env::set_var("VANTA_BATE", "maybe") };
        let error = env_flag("VANTA_BATE", true)
            .expect_err("an unrecognised switch value must never be a silent default");
        assert!(
            error.contains("CONFIG_ERROR") && error.contains("VANTA_BATE"),
            "the error must name the switch and the class, got: {error}"
        );

        match previous {
            Some(value) => unsafe { std::env::set_var("VANTA_BATE", value) },
            None => unsafe { std::env::remove_var("VANTA_BATE") },
        }
    }

    // [VANTA-WHY] M3 item 8 follow-up — `VANTA_ARM` is the arm the harness STATED,
    // and an ABSENT one is a hand-run binary: it must resolve to exactly the
    // switch-derived name and config every run before the var recorded, or an
    // ordinary `vanta_m3` invocation would start writing a different `arm` than
    // before.
    #[test]
    fn an_absent_arm_var_resolves_to_the_switch_derived_name() {
        // M3 item X4 — the crate-wide environment lock (see the note above).
        let _env = crate::test_env::lock();
        let previous = std::env::var("VANTA_ARM").ok();

        unsafe { std::env::remove_var("VANTA_ARM") };
        assert_eq!(
            env_arm().unwrap(),
            None,
            "an absent VANTA_ARM is 'no statement', not an unknown arm"
        );
        // An empty value is the same statement: a shell expanding an unset var
        // (`VANTA_ARM=`) must not turn a correct invocation into a refusal.
        unsafe { std::env::set_var("VANTA_ARM", "") };
        assert_eq!(env_arm().unwrap(), None, "an empty VANTA_ARM is no statement");
        unsafe { std::env::set_var("VANTA_ARM", "  ") };
        assert_eq!(env_arm().unwrap(), None, "whitespace is no statement either");

        for attacker in [false, true] {
            for planner in [false, true] {
                for bate in [false, true] {
                    let resolved =
                        resolve_arm(None, attacker, planner, bate, 4, None, None).unwrap();
                    let pre = benchmark_config_name(attacker, planner, bate);
                    assert_eq!(
                        resolved.name, pre,
                        "attacker={attacker} planner={planner} bate={bate}: a hand-run \
                         binary must record the arm it always recorded"
                    );
                    assert_eq!(
                        resolved.config, pre,
                        "and the same value in the config column"
                    );
                }
            }
        }

        match previous {
            Some(value) => unsafe { std::env::set_var("VANTA_ARM", value) },
            None => unsafe { std::env::remove_var("VANTA_ARM") },
        }
    }

    // [VANTA-WHY] C5 — `VANTA_REASONING_BUDGET` reaches only the Attacker's three
    // sites, and a judge arm makes exactly one call, to the `judge` site. A budget
    // asked for on a judge arm therefore reaches nothing and the row would describe
    // a different arm than the operator asked for, so it is refused by name.
    #[test]
    fn a_reasoning_budget_is_refused_on_a_judge_arm_and_allowed_elsewhere() {
        let judge = resolve_arm(Some("judge"), false, false, true, 1, Some("plain"), None).unwrap();
        let judge_suite =
            resolve_arm(Some("judge+suite"), false, false, true, 1, Some("suite"), None).unwrap();
        // Arms whose Attacker is off are NOT refused: their budget reaches the
        // Attacker's sites whenever the Attacker runs, which is a real arm.
        let baseline = resolve_arm(Some("b1"), false, false, true, 4, None, None).unwrap();
        let attacker = resolve_arm(Some("b3"), true, false, true, 4, None, None).unwrap();

        for arm in [&judge, &judge_suite] {
            let error = refuse_reasoning_budget_on_judge_arm(Some(12000), arm)
                .expect_err("a judge arm has no allow-listed call site for a budget");
            assert!(
                error.contains("CONFIG_ERROR")
                    && error.contains("VANTA_REASONING_BUDGET")
                    && error.contains(&arm.name),
                "the refusal must name the switch and the arm, got: {error}"
            );
            // No budget asked for: a judge arm runs exactly as before.
            assert!(
                refuse_reasoning_budget_on_judge_arm(None, arm).is_ok(),
                "an unset budget must never refuse a judge arm"
            );
        }
        for arm in [&baseline, &attacker] {
            assert!(
                refuse_reasoning_budget_on_judge_arm(Some(12000), arm).is_ok(),
                "a non-judge arm's budget reaches the Attacker's sites and is allowed"
            );
        }
    }

    // [VANTA-WHY] C5 — the refusal above is only CORRECT because the judge's own
    // site is outside the allow-list. Were `judge` ever added to it, the guard
    // would reject a switch that could act, so the two facts are pinned together:
    // the site is the judge's, and it is not budgetable.
    #[test]
    fn the_judge_site_is_outside_the_reasoning_budget_allow_list() {
        assert!(is_judge_site("judge"), "the judge site is spelled `judge`");
        assert!(
            !crate::llm::REASONING_BUDGET_SITES.contains(&"judge"),
            "the judge site must stay outside the budget allow-list, or C5's refusal \
             would reject a switch that could act on it"
        );
    }

    // [VANTA-WHY] M3 item 8 follow-up — a stated arm carries its own config, which
    // is what lets `b0` and `baseline` (the SAME switch pair) be told apart on disk.
    #[test]
    fn every_stated_arm_resolves_to_its_own_config() {
        for spec in ARMS {
            let budget = spec.max_attempts.unwrap_or(4);
            let resolved = resolve_arm(
                Some(spec.name),
                spec.attacker,
                spec.planner,
                spec.bate,
                budget,
                spec.judge,
                spec.fixed_rounds,
            )
            .unwrap_or_else(|e| panic!("arm '{}' must resolve: {e}", spec.name));
            assert_eq!(resolved.name, spec.name);
            assert_eq!(
                resolved.config, spec.config,
                "arm '{}' must carry its own config value",
                spec.name
            );
        }
        // The two legacy arms keep the config name every row on disk already has.
        assert_eq!(
            resolve_arm(Some("b1"), false, false, true, 4, None, None)
                .unwrap()
                .config,
            "baseline"
        );
        assert_eq!(
            resolve_arm(Some("b3"), true, false, true, 4, None, None)
                .unwrap()
                .config,
            "vanta"
        );
        assert_eq!(
            resolve_arm(Some("b0"), false, false, true, 1, None, None)
                .unwrap()
                .config,
            "b0"
        );
    }

    // [VANTA-WHY] M3 item 8 follow-up — the driver's half of "the harness refuses
    // before launching". A name that disagrees with the switches, or a budget that
    // would silently redefine an arm, is a `CONFIG_ERROR` before the repo is loaded.
    #[test]
    fn a_stated_arm_that_contradicts_the_switches_or_budget_is_refused() {
        // Unknown name: refused, and the sentence lists the known arms. The list
        // holds ARM names, so `baseline` (a config, not an arm) must not appear in it.
        let error = resolve_arm(Some("nope"), false, false, true, 4, None, None)
            .expect_err("an unknown arm name must not fall back to the switches");
        assert!(
            error.contains("CONFIG_ERROR")
                && error.contains("VANTA_ARM='nope'")
                && error.contains("b0")
                && !error.contains("baseline"),
            "the refusal must name the class, the stated name and the known arms, got: {error}"
        );
        assert!(
            error.contains("b3") && error.contains("suite"),
            "every known arm must be listed, got: {error}"
        );

        // Switch disagreement: `b2` is attacker=on, bate=off; the switches say otherwise.
        let error = resolve_arm(Some("b2"), true, false, true, 4, None, None)
            .expect_err("VANTA_ARM must not disagree with the switches");
        assert!(
            error.contains("CONFIG_ERROR")
                && error.contains("VANTA_ARM='b2'")
                && error.contains("disagrees with the switches"),
            "got: {error}"
        );
        // `b1` is attacker=off; asking for it with the Attacker on is the same fault.
        assert!(resolve_arm(Some("b1"), true, false, true, 4, None, None).is_err());

        // Budget: `b0` IS one Creator call, so a four-attempt invocation is not b0.
        let error = resolve_arm(Some("b0"), false, false, true, 4, None, None)
            .expect_err("b0 pins its attempt budget and the invocation must not override it");
        assert!(
            error.contains("CONFIG_ERROR")
                && error.contains("VANTA_ARM='b0'")
                && error.contains("--max-attempts 1"),
            "got: {error}"
        );
        // The budget the arm pins is accepted.
        assert!(resolve_arm(Some("b0"), false, false, true, 1, None, None).is_ok());
        // An arm with no pinned budget takes the invocation's own.
        assert!(resolve_arm(Some("b1"), false, false, true, 4, None, None).is_ok());

        // [VANTA-WHY] M3 item 7 — a judge setting is a STATEMENT like the arm name
        // is: it needs an arm to hang off, and the arm it names must agree with it.
        assert!(resolve_arm(Some("judge"), false, false, true, 1, Some("plain"), None).is_ok());
        assert!(resolve_arm(Some("judge+suite"), false, false, true, 1, Some("suite"), None).is_ok());
        // A judge mode with no arm would be recorded under a switch-derived name.
        let error = resolve_arm(None, false, false, true, 4, Some("plain"), None)
            .expect_err("a judge mode must not be recorded under a switch-derived arm");
        assert!(
            error.contains("CONFIG_ERROR") && error.contains("VANTA_JUDGE='plain'"),
            "got: {error}"
        );
        // The arm and the setting must agree in both directions.
        let error = resolve_arm(Some("judge"), false, false, true, 1, Some("suite"), None)
            .expect_err("VANTA_ARM/VANTA_JUDGE must agree");
        assert!(
            error.contains("CONFIG_ERROR") && error.contains("disagrees"),
            "got: {error}"
        );
        // A non-judge arm with a judge mode set is the same fault.
        assert!(resolve_arm(Some("b1"), false, false, true, 4, Some("plain"), None).is_err());
    }

    // [VANTA-WHY] M3 item 9/A5 — a fixed-round arm IS K attempts, so the stated arm
    // and the stated `--fixed-rounds` must agree, exactly as B0's name and its pinned
    // `--max-attempts` must. The harness builds both from one registry entry, so this
    // catches a hand-run binary that named the arm but not its K. An ordinary arm with
    // `--fixed-rounds` is deliberately NOT refused: the option is available to any
    // repair run (item 9), and only the name that claims K is contradicted by a miss.
    #[test]
    fn a_fixed_round_arm_must_be_handed_the_k_its_name_claims() {
        // The K the arm claims is accepted.
        assert!(resolve_arm(Some("b1-k4"), false, false, true, 4, None, Some(4)).is_ok());
        assert!(resolve_arm(Some("b3-k4"), true, false, true, 4, None, Some(4)).is_ok());
        // No `--fixed-rounds`: the arm's K is unstated, so the name is not earned.
        let error = resolve_arm(Some("b1-k4"), false, false, true, 4, None, None)
            .expect_err("a fixed-round arm without its K must be refused");
        assert!(
            error.contains("CONFIG_ERROR")
                && error.contains("VANTA_ARM='b1-k4'")
                && error.contains("--fixed-rounds 4"),
            "got: {error}"
        );
        // A different K contradicts the name.
        let error = resolve_arm(Some("b3-k4"), true, false, true, 4, None, Some(3))
            .expect_err("a fixed-round arm handed the wrong K must be refused");
        assert!(
            error.contains("CONFIG_ERROR")
                && error.contains("VANTA_ARM='b3-k4'")
                && error.contains("--fixed-rounds 3"),
            "got: {error}"
        );
        // An ordinary arm is free to state a K — that is item 9's own option.
        assert!(resolve_arm(Some("b3"), true, false, true, 4, None, Some(4)).is_ok());
        assert!(resolve_arm(Some("b1"), false, false, true, 4, None, Some(2)).is_ok());
    }

    // [VANTA-WHY] M3 item E9 — a fixed-round count must not exceed an arm that PINS its
    // attempt budget. `b0` IS one Creator call, and before E9 `b0` + `--fixed-rounds 4`
    // ran four attempts under a name that pins one: `b0`'s `fixed_rounds` is `None`, so
    // the arm-vs-K check above was skipped, and `--fixed-rounds` never touches
    // `--max-attempts`, so the budget check never saw it either.
    #[test]
    fn a_fixed_round_count_must_not_exceed_a_pinned_attempt_budget() {
        // Four rounds contradict b0's one attempt; the refusal names the arm and both
        // numbers.
        let error = resolve_arm(Some("b0"), false, false, true, 1, None, Some(4))
            .expect_err("a fixed-round count above b0's pinned budget must be refused");
        assert!(
            error.contains("CONFIG_ERROR")
                && error.contains("VANTA_ARM='b0'")
                && error.contains("--max-attempts 1")
                && error.contains("--fixed-rounds 4"),
            "got: {error}"
        );
        // A count EQUAL to the pinned budget contradicts nothing.
        assert!(resolve_arm(Some("b0"), false, false, true, 1, None, Some(1)).is_ok());
        // An arm with no pinned budget keeps item 9's own option, however large.
        assert!(resolve_arm(Some("b1"), false, false, true, 4, None, Some(4)).is_ok());
        assert!(resolve_arm(Some("b3"), true, false, true, 4, None, Some(9)).is_ok());
    }

    // [VANTA-WHY] M3 item E13 / v1-m3.1 — `--native-timeout-secs` may only RAISE the
    // in-session native cap. Below the effective cap (the session floor,
    // `harness::native_session_timeout_floor()`, 1200 s or `VANTA_TEST_TIMEOUT_SECS` when
    // that is larger) the value would make every suite time out sooner than the default,
    // which is the opposite of what the flag exists for; refused where the cap is
    // resolved, naming the flag and the floor.
    //
    // [VANTA-WHY] M3 item F6 — every expectation is DERIVED from the floor the run
    // actually resolves, so this passes whatever `VANTA_TEST_TIMEOUT_SECS` is set to.
    // It used to assert the floor was 120 and hard-code 120/119, which would fail —
    // for no reason about the code — on any machine that raised the variable.
    #[test]
    fn a_native_timeout_below_the_effective_cap_is_refused() {
        let floor = crate::pipeline::harness::native_session_timeout_floor();
        assert!(floor > 0, "the effective cap is a positive number of seconds");
        assert!(
            floor >= crate::pipeline::harness::NATIVE_SESSION_TIMEOUT_FLOOR_SECS,
            "the session floor is the floor of the effective cap"
        );
        // Absent keeps the floor; the floor itself and anything above it are accepted.
        assert_eq!(resolve_native_timeout_secs(None).unwrap(), floor);
        assert_eq!(resolve_native_timeout_secs(Some(floor)).unwrap(), floor);
        assert_eq!(
            resolve_native_timeout_secs(Some(floor + 1)).unwrap(),
            floor + 1
        );
        assert_eq!(
            resolve_native_timeout_secs(Some(floor * 4 + 1)).unwrap(),
            floor * 4 + 1
        );
        // Below the floor is refused, naming the flag and both numbers.
        let mut below = vec![floor - 1, 1];
        if floor > 30 {
            below.push(30);
        }
        for bad in below {
            let error = resolve_native_timeout_secs(Some(bad))
                .expect_err("a cap below the effective cap must be refused");
            assert!(
                error.contains("CONFIG_ERROR")
                    && error.contains("--native-timeout-secs")
                    && error.contains(&bad.to_string())
                    && error.contains(&floor.to_string()),
                "got: {error}"
            );
        }
    }

    // [VANTA-WHY] v1-m3.1 (§8.4) — the native gate's cap is CLAMPED to the time the
    // run's clock has left, so a slow row's gate (a 725 s suite: 7250 s per side) cannot
    // start with less than its cap on the clock and outlive `VANTA_DEADLINE_SECS` on
    // its own. `native_gate_cap` is the pure rule; the call site passes the ONE value
    // it returns to BOTH sides of the `tokio::join!`, so the two sides stay the same
    // experiment even when the clamp bites.
    #[test]
    fn the_native_gate_cap_is_clamped_to_the_time_left_and_can_only_lower() {
        // Less time on the clock than the cap: the gate gets the time left —
        // a 725 s suite's 7250 s resolved cap, with 3000 s left.
        assert_eq!(native_gate_cap(7250, Some(3000)), Some(3000));
        assert_eq!(native_gate_cap(1200, Some(1199)), Some(1199));
        // At least the cap left: the FULL cap, unchanged. The clamp can only LOWER
        // the cap, never raise it past the resolved value.
        assert_eq!(native_gate_cap(7250, Some(7250)), Some(7250));
        assert_eq!(native_gate_cap(1200, Some(99999)), Some(1200));
        // No deadline armed: the full cap, exactly as every run before §14.15.
        assert_eq!(native_gate_cap(7250, None), Some(7250));
    }

    // [VANTA-WHY] v1-m3.1 (§8.5) — with NOTHING left on the clock there is no run to
    // make: a suite under a zero-second cap would only manufacture a kill.
    // `native_gate_cap` therefore returns `None`, which the call site maps to the
    // SAME deadline exit the attempt top takes (`stop_reason = BUDGET_EXHAUSTED`,
    // `NeedsReview`) — adding no new outcome. The `Some(0)` (clock due) and `None`
    // (no clock) cases must not collapse: `None` keeps the full cap.
    #[test]
    fn nothing_left_on_the_clock_skips_the_gate_rather_than_running_it_empty() {
        assert_eq!(native_gate_cap(7250, Some(0)), None);
        assert_eq!(native_gate_cap(1200, Some(0)), None);
        // The distinguishing case: no deadline is NOT zero seconds left.
        assert_eq!(native_gate_cap(1200, None), Some(1200));
    }

    // [VANTA-WHY] v1-m3.1 F2 — the judge's own native run (`judge+suite`) is the SAME
    // sized step as the gate's, and unclamped it fails the same way: the attempt top
    // honours the deadline only once it has passed, so a run taken with the whole
    // resolved cap can start AT the deadline — the in-session import guard may start
    // first and has its own cap — and reach the harness's `RUN_TIMEOUT_SECS` on its
    // own. The source once claimed a `max_call_secs + native_cap` bound for it, and
    // that is not how the clock works: the margin between the deadline and the kill is
    // ONE in-flight model call's worth, and this arm's model call comes AFTER the run,
    // so the margin is already spent by then. The run is therefore clamped like the
    // gate's two sides. Both halves are pinned here because they are two claims: the
    // arithmetic, and the call site that takes the rule.
    #[test]
    fn the_judge_suite_run_takes_the_gate_s_clamped_cap_and_the_deadline_exit() {
        // The two clocks, as the harness hands them out: the deadline Vanta is given,
        // and the backstop it SIGKILLs the row at.
        let deadline: u64 = 10_200;
        let run_timeout: u64 = 13_800;
        // The margin is one model call's backstop — exactly what the judge's own call
        // (after this run) may spend, so it cannot also cover the suite.
        assert_eq!(run_timeout - deadline, 3_600);
        // A slow suite's per-side cap, larger than that margin: this is the case the
        // clamp exists for.
        let cap: u64 = 7_200;
        assert!(cap > run_timeout - deadline);
        // The import guard's own cap, which may run before the suite in the same attempt.
        let guard_cap: u64 = 120;
        assert!(
            deadline + guard_cap + cap >= run_timeout,
            "an unclamped run started at the deadline reaches the harness's kill \
             ({} s >= {} s), and a killed row writes no manifest entry",
            deadline + guard_cap + cap,
            run_timeout
        );
        // The clamp itself: the full cap while the run has at least that much left...
        assert_eq!(native_gate_cap(cap, Some(deadline)), Some(cap));
        // ...the time left when it has less, so the run ends by the deadline...
        assert_eq!(native_gate_cap(cap, Some(3_000)), Some(3_000));
        // ...and NO run at all when the clock is out, which is the value the call site
        // maps to the existing deadline exit (`budget_exhausted`/`NeedsReview`).
        assert_eq!(native_gate_cap(cap, Some(0)), None);
        // The rule is worth nothing if the judge's native run is not what takes it, so
        // the call site is pinned too. `native_gate_cap` is called from two places and
        // the gate's own line is a different one: this is the judge's.
        let source = include_str!("driver.rs");
        assert!(
            source.contains("let judge_suite_cap = match native_gate_cap(native_cap, remaining) {"),
            "the judge's native run must take `native_gate_cap`, exactly as the gate \
             does; this file has no clamped judge-suite call site"
        );
    }

    // [VANTA-WHY] Finding B (PR #4 review) — the judge's native suite is a sized
    // step that can consume the remaining clock, so the judge's model call can be
    // refused by the deadline. A refusal is "the run ran out of clock", not "the
    // machine broke", and it must take the SAME exit the Creator path takes
    // (`stop_reason=budget_exhausted` with `NeedsReview`) — never `InfraFailure`.
    // Pin the arm (it is evaluated before the server-blaming line) and the deadline
    // check that runs right after the clamped suite.
    #[test]
    fn the_judge_call_honours_the_deadline_as_budget_exhausted_not_infra_failure() {
        let source = include_str!("driver.rs");
        let marker = source
            .find("if message.contains(crate::llm::DEADLINE_EXCEEDED_MARKER) {")
            .expect("the judge's `JudgeError::Call` arm must recognise the deadline refusal");
        let blames_server = source
            .find("infra_failure_error = Some(format!(\"Judge LLM call failed: {message}\"));")
            .expect("the judge's call arm must still record a real call failure");
        assert!(
            marker < blames_server,
            "the deadline marker must be matched BEFORE the judge call is reported as an \
             infrastructure failure, or a `budget_exhausted` row reads as a machine fault"
        );
        assert!(
            source.contains("The wall-clock deadline passed while the judge's native suite ran"),
            "the deadline must be checked right after the judge's native suite, before the call"
        );
    }

    // [VANTA-WHY] v1-m3.1 item 1 — every deadline exit is a TERMINAL verdict, so it
    // must leave a `final.patch` exactly as `MaxAttemptsReached` does
    // (`final_patched_files = Some(patched_files)`). Before this fix all eight
    // `break 'attempts GovernorDecision::NeedsReview` sites broke BEFORE the
    // terminal `match decision` — the one place a plain `NeedsReview` attaches the
    // patch — so no `final.patch` was written, the runner censored the oracle, the
    // row read `unobservable`, and the wave gate halted on that ONE row (§8.2). Pin
    // all eight: each must assign `final_patched_files` between its
    // `stop_reason = Some(BUDGET_EXHAUSTED)` and its break, so a graded deadline row
    // can never regress to an ungraded one.
    //
    // [VANTA-WHY] v1-m3.2 item B — the Attacker's and the Judge's exhausted
    // model-output arms take the SAME `NeedsReview` shape on the SAME rule
    // (attach the produced tree, or the row is censored and the wave halts), so
    // they are pinned here too. They are counted separately from the eight: the
    // `budget_exhausted` count must stay eight, exactly the deadline exits, or a
    // model-output exit could be mistaken for "the run ran out of clock".
    #[test]
    fn every_deadline_exit_attaches_the_last_produced_tree() {
        let source = include_str!("driver.rs");
        // Built with `concat!` so this test's OWN source does not contain the
        // literal it counts (`include_str!` reads the whole file, tests included).
        let brk = concat!("break 'attempts GovernorDecision::NeedsReview", ";");
        let budget = concat!("stop_reason = Some(BUDGET_EXHAUSTED)", ";");
        let model_output = concat!("stop_reason = Some(MODEL_OUTPUT_INVALID)", ";");
        let attach = concat!("final_patched_files = ", "");
        let breaks: Vec<usize> = source.match_indices(brk).map(|(i, _)| i).collect();
        assert_eq!(
            breaks.len(),
            10,
            "this file must carry exactly the eight deadline `NeedsReview` breaks \
             (attempt top, Creator refusal, Attacker refusal, two clamp skips, \
             clamp-cut original, judge-suite deadline, judge-call refusal) plus the \
             two v1-m3.2 model-output breaks (judge, Attacker)"
        );
        assert_eq!(
            source.match_indices(budget).count(),
            8,
            "the `budget_exhausted` exits are the eight deadline ones; an item-B \
             model-output exit must not be counted as a deadline exit"
        );
        let mut prev = 0usize;
        for (n, &b) in breaks.iter().enumerate() {
            let segment = &source[prev..b];
            assert!(
                segment.contains(budget) || segment.contains(model_output),
                "break #{n} does not follow a `budget_exhausted` or a \
                 `model_output_invalid` stop_reason"
            );
            assert!(
                segment.contains(attach),
                "break #{n} skips the terminal patch attach, so the runner \
                 would censor the oracle and the wave gate would halt on this row"
            );
            prev = b + brk.len();
        }
    }

    // [VANTA-WHY] v1-m3.1 item 1 — the attached tree is the run's most recent
    // PRODUCED one, never a fabricated original. The six post-apply deadline exits
    // (Attacker refusal, both clamp skips, clamp-cut original, both judge deadline
    // exits) attach this attempt's `patched_files`, the same value
    // `MaxAttemptsReached` attaches; the two PRE-apply exits (attempt top, Creator
    // refusal) attach `last_applied_patched_files`, which is `None` until the first
    // apply. That `None` is what keeps a run which never produced a tree
    // `unobservable` (it halts, as intended) — a `final_patch_tree` fallback to the
    // original here would fabricate a gradeable tree for an experiment that did not
    // run.
    #[test]
    fn the_deadline_exit_attaches_the_produced_tree_never_a_fabricated_original() {
        let source = include_str!("driver.rs");
        let brk = concat!("break 'attempts GovernorDecision::NeedsReview", ";");
        // `concat!` again: this literal must not appear verbatim in the test source.
        let pre_apply = concat!(
            "final_patched_files = last_applied_patched_files.clone()",
            ";"
        );
        let breaks: Vec<usize> = source.match_indices(brk).map(|(i, _)| i).collect();
        let mut prev = 0usize;
        for (n, &b) in breaks.iter().enumerate() {
            let segment = &source[prev..b];
            if n < 2 {
                // The two pre-apply sites, in file order: attempt top, Creator refusal.
                assert!(
                    segment.contains(pre_apply),
                    "deadline break #{n} (pre-apply) must attach \
                     `last_applied_patched_files`, the run's last produced tree"
                );
            } else {
                assert!(
                    !segment.contains(pre_apply),
                    "deadline break #{n} (post-apply) must attach THIS attempt's tree, \
                     not `last_applied_patched_files`"
                );
            }
            prev = b + brk.len();
        }
        assert_eq!(
            source.matches(pre_apply).count(),
            2,
            "exactly the two pre-apply deadline exits read `last_applied_patched_files`"
        );
    }

    // [VANTA-WHY] v1-m3.2 item B — the ONE predicate that decides whether an
    // exhausted structured call is the MODEL's mistake (`NeedsReview`, graded) or
    // the MACHINE's (`InfraFailure`, gated). Only the two markers the retry loop
    // puts on a reply it READ and could not use count. Everything else — an
    // unreachable server, a 5xx after the client's own retries, a prompt the
    // pre-flight guard refused — stays infrastructure, because ending those
    // `NeedsReview` would grade a run in which the call never happened.
    #[test]
    fn only_the_model_output_markers_classify_as_a_model_output_failure() {
        use crate::llm::{PARSE_FAILURE_PREFIX, TRUNCATED_TWICE_MARKER};

        // The schema-parse arm: the server answered, the answer did not satisfy
        // the type. This is the arm the wave-1 rows died on.
        assert!(is_model_output_failure(&format!(
            "structured_call_with_retry 5/5 attempts FAILED. Last:\nattempt 5 \
             {PARSE_FAILURE_PREFIX}: unknown variant `not_a_label`, expected one of \
             `Reproduction`, `EdgeCase`"
        )));
        // The cut-off-twice arm: two answers the server stopped at `max_tokens`.
        assert!(is_model_output_failure(&format!(
            "structured_call_with_retry 3/3 attempts FAILED. Last:\nattempt 3 TRUNCATED: … \
             ({TRUNCATED_TWICE_MARKER})"
        )));

        // The machine. A transport failure is NOT the model's output, even though
        // it exhausts the same budget and reaches the same error string.
        assert!(
            !is_model_output_failure(
                "Creator LLM call failed: attempt 3 HTTP err: HTTP ERROR 503 from \
                 http://127.0.0.1:8080/v1/chat/completions. Body snippet: overloaded"
            ),
            "an unreachable or 5xx server is the machine, not the model, and must stay \
             InfraFailure"
        );
        assert!(
            !is_model_output_failure(
                "prompt_over_budget: the Creator prompt does not fit — a prompt that cannot \
                 fit is arithmetic decided before the call, not the model's answer"
            ),
            "a `prompt_over_budget` refusal never reached the model, so it must stay \
             InfraFailure"
        );
        // The deadline refusal is neither: it is the run's own clock.
        assert!(
            !is_model_output_failure(&format!(
                "the run's wall-clock deadline passed before this `attacker_generate` call \
                 could start ({})",
                crate::llm::DEADLINE_EXCEEDED_MARKER
            )),
            "a deadline refusal is `budget_exhausted`, which is its own exit and keeps its \
             own stop_reason"
        );
    }

    // [VANTA-WHY] v1-m3.2 item B — the Creator is the ONE model-output arm that
    // must NOT take the `NeedsReview` exit the other two take. The Attacker and the
    // Judge both run AFTER the candidate applied, so `patched_files` is a tree they
    // can attach and `NeedsReview` grades the row. The Creator runs FIRST, so an
    // exhausted call produced no patch at all — this attempt decided nothing, and
    // the only honest reading is a FAILED ATTEMPT: record the hole, ask again with
    // the next round's seed, and let the Governor advance the budget. On the LAST
    // attempt that verdict is `MaxAttemptsReached`, whose terminal arm attaches the
    // run's last PRODUCED tree — the original tree when nothing ever applied — so
    // `final.patch` is written and the row is graded (`unresolved`) rather than
    // censored to `unobservable` (which would halt the wave, §8.2). A transport
    // error or a `prompt_over_budget` refusal is a different claim about the run and
    // keeps `InfraFailure` in the fallback BELOW this arm.
    #[test]
    fn the_creators_exhausted_model_output_is_a_failed_attempt_not_an_infrastructure_failure() {
        let source = include_str!("driver.rs");
        // The Creator's arm is the FIRST `is_model_output_failure(&message)` in the
        // file (the Judge's is the second). Bound it by the transport / over-budget
        // fallback that must stay AFTER it — a machine fault never reaches this arm.
        let marker = source
            .find("if is_model_output_failure(&message) {")
            .expect("the Creator call arm must classify an exhausted model-output error");
        let transport = source
            .find("infra_failure_error = Some(if message")
            .expect("the Creator's transport / prompt-over-budget fallback must still exist");
        assert!(
            marker < transport,
            "the model-output arm must be matched BEFORE the transport fallback, or an \
             unreachable server would be read as the model's own bad answer"
        );
        let branch = &source[marker..transport];
        assert!(
            branch.contains("governor.record_unobserved_attempt();"),
            "an exhausted Creator produced no patch, so the attempt must be recorded as a \
             hole in the dataset exactly as a failed apply is"
        );
        assert!(
            branch.contains("AttemptVerdict::Inconclusive"),
            "this attempt produced nothing to observe, so it is `Inconclusive` to the Governor"
        );
        assert!(
            branch.contains("continue 'attempts;"),
            "a non-last attempt must go round again, with the seed advanced"
        );
        assert!(
            branch.contains("final_patch_tree("),
            "the LAST attempt must attach the run's last produced tree (the original tree \
             when nothing ever applied), so the `MaxAttemptsReached` row is graded, not censored"
        );
        assert!(
            branch.contains("break 'attempts decision;"),
            "the last attempt must break with the Governor's OWN decision \
             (`MaxAttemptsReached`), not a hard-coded verdict"
        );
        assert!(
            !branch.contains("infra_failure_error"),
            "the model-output arm must never set an infrastructure failure"
        );
        assert!(
            !branch.contains("GovernorDecision::InfraFailure"),
            "the model-output arm must never end the run `InfraFailure`"
        );
    }

    /// A `NativeSuiteReport` for the gate-clamp tests: the nodeids and run state
    /// spelled out, `timed_out` set explicitly (the harness's own test helper
    /// hard-codes `false`, so a killed side cannot be built with it).
    ///
    /// [VANTA-WHY] Finding A — `setup_failed` defaults to `false` (an attributable
    /// report); a test that needs a SETUP/TRANSPORT failure sets the flag on the
    /// value it builds.
    fn clamped_report(
        tests: &[&str],
        ran: bool,
        timed_out: bool,
    ) -> crate::pipeline::harness::NativeSuiteReport {
        crate::pipeline::harness::NativeSuiteReport {
            tests: tests
                .iter()
                .map(|name| (name.to_string(), TestResult::Passed))
                .collect(),
            failure_outputs: HashMap::new(),
            ran,
            timed_out,
            secs: None,
            exit_code: None,
            stderr_tail: String::new(),
            collection: None,
            setup_failed: false,
        }
    }

    /// A `NativeSuiteReport` for the `NativeGateFacts` tests: only the fields the
    /// fold reads (`secs`, `timed_out`, `setup_failed`) are spelled out, so a test
    /// can build a completed side, a capped side, and a side that never set up.
    fn gate_report(
        secs: Option<f64>,
        timed_out: bool,
        setup_failed: bool,
    ) -> crate::pipeline::harness::NativeSuiteReport {
        crate::pipeline::harness::NativeSuiteReport {
            tests: Vec::new(),
            failure_outputs: HashMap::new(),
            ran: secs.is_some(),
            timed_out,
            secs,
            exit_code: None,
            stderr_tail: String::new(),
            collection: None,
            setup_failed,
        }
    }

    // [VANTA-WHY] F2 (PR #4 review) — a gate whose ORIGINAL side completed while
    // the PATCHED side never set up still obtained an original-side measurement.
    // `NativeGateFacts::record` is called BEFORE the `native_setup_failure` guard,
    // so that measurement reaches the row instead of being discarded by the
    // `InfraFailure` exit. And because F3 makes the `*_secs` fields the LAST gate's
    // own values, a side that did not complete in a later gate records `None` — the
    // earlier measurement is not carried forward.
    #[test]
    fn a_completed_side_is_recorded_even_when_the_other_side_never_completed() {
        let mut facts = NativeGateFacts::default();
        // Original completed with wall time; patched failed SETUP (no secs, not a
        // timeout — the `failed_run(None, …)` shape).
        facts.record(
            1200,
            &gate_report(Some(33.0), false, false),
            &gate_report(None, false, true),
        );
        assert_eq!(
            facts.original_secs,
            Some(33.0),
            "the completed side's secs is kept despite the setup-failure exit"
        );
        assert_eq!(
            facts.patched_secs, None,
            "a side that never completed records no secs"
        );
        assert!(
            !facts.patched_timed_out_last && !facts.patched_timed_out_any,
            "a setup failure is not a timeout"
        );

        // A later gate whose ORIGINAL side fails setup and whose PATCHED side
        // completes: each side takes THIS gate's own value, so the earlier original
        // measurement is gone and only the patched one is recorded.
        facts.record(
            1200,
            &gate_report(None, false, true),
            &gate_report(Some(119.0), false, false),
        );
        assert_eq!(
            facts.original_secs, None,
            "the last gate's original side did not complete, so no secs carries over"
        );
        assert_eq!(facts.patched_secs, Some(119.0));
    }

    // [VANTA-WHY] F2/F3 (PR #4 review) — a side killed in the LAST gate must not
    // keep its earlier attempt's wall time: the row would then pair the last gate's
    // kill and effective cap with a previous attempt's measurement. The any-attempt
    // kill flag still records the earlier completion's opposite, and the last-gate
    // kill flag is the last gate's.
    #[test]
    fn a_side_killed_in_the_last_gate_does_not_keep_an_earlier_attempts_secs() {
        let mut facts = NativeGateFacts::default();
        // Attempt 1: both sides complete; the patched side measured 500 s under the
        // full 1200 s cap, so the row would otherwise carry 500 s forward.
        facts.record(
            1200,
            &gate_report(Some(33.0), false, false),
            &gate_report(Some(500.0), false, false),
        );
        assert_eq!(facts.patched_secs, Some(500.0));
        assert!(!facts.patched_timed_out_last);
        assert!(!facts.patched_timed_out_any);

        // Attempt 2 (the LAST gate): the patched side is killed by the clamp to
        // 240 s — no wall time, timed_out.
        facts.record(
            240,
            &gate_report(Some(31.0), false, false),
            &gate_report(None, true, false),
        );
        assert_eq!(
            facts.patched_secs, None,
            "the last gate's patched side produced no secs, so the earlier 500 s is gone"
        );
        assert!(
            facts.patched_timed_out_last,
            "the LAST gate's patched side hit the cap"
        );
        assert!(
            facts.patched_timed_out_any,
            "the any-attempt flag records the kill"
        );
        assert_eq!(facts.effective_cap_secs, Some(240));
        assert_eq!(facts.original_secs, Some(31.0));
    }

    // [VANTA-WHY] F3 (PR #4 review) — the last-gate trio (`*_secs`,
    // `*_timed_out_last`, `effective_cap_secs`) describes the LAST gate only, so a
    // reader comparing them never pairs one attempt's seconds with another
    // attempt's kill; `*_timed_out_any` keeps the wider "any attempt" meaning so a
    // kill is not lost when a later attempt finishes.
    #[test]
    fn the_last_gate_fields_describe_the_last_gate_not_an_earlier_attempt() {
        let mut facts = NativeGateFacts::default();
        // Attempt 1: the PATCHED side hits the cap (no secs, timed_out), the
        // original side completes.
        facts.record(
            600,
            &gate_report(Some(33.0), false, false),
            &gate_report(None, true, false),
        );
        assert!(facts.patched_timed_out_last);
        assert!(facts.patched_timed_out_any);
        assert_eq!(facts.effective_cap_secs, Some(600));

        // Attempt 2 (the LAST gate): both sides complete and the patched side now
        // measures a time.
        facts.record(
            1200,
            &gate_report(Some(31.0), false, false),
            &gate_report(Some(119.0), false, false),
        );
        assert!(
            !facts.patched_timed_out_last,
            "the LAST gate did not time out"
        );
        assert!(
            facts.patched_timed_out_any,
            "but an earlier attempt did, and that is not lost"
        );
        assert_eq!(
            facts.patched_secs,
            Some(119.0),
            "the seconds are the LAST gate's, not the capped one's"
        );
        assert_eq!(facts.original_secs, Some(31.0));
        assert_eq!(facts.effective_cap_secs, Some(1200));
    }

    // [VANTA-WHY] v1-m3.1 (§8.5, case b) — an ORIGINAL side KILLED by the clamp
    // collected nothing because the CLOCK ran out, not because the base tree could
    // not be graded. That emptiness is normally an infrastructure fault (§8.1 row 1),
    // but here it must take the SAME deadline exit the attempt top takes
    // (`NeedsReview` / `budget_exhausted`), never `InfraFailure` — so a
    // `budget_exhausted` row is not misreported as a machine fault. The line is drawn
    // on BOTH conditions: the gate was clamped AND the original side was killed.
    // Anything else stays `InfraFailure`.
    #[test]
    fn an_original_side_killed_by_the_clamp_exits_as_budget_exhausted_not_infra_failure() {
        // The kill makes the gate vacuous (the original side collected nothing) ...
        let original = clamped_report(&[], false, true);
        assert!(
            crate::pipeline::harness::native_gate_vacuous(88_642, &original),
            "an original side that collected nothing makes the gate vacuous"
        );
        // ... but the emptiness is the clock, so the exit is the deadline exit.
        assert_eq!(
            native_vacuous_exit(true, original.timed_out),
            GovernorDecision::NeedsReview
        );
        // A clamp that did NOT kill the side (a real collection error inside the
        // shortened window) and a run under the full cap both stay InfraFailure.
        assert_eq!(
            native_vacuous_exit(true, false),
            GovernorDecision::InfraFailure
        );
        assert_eq!(
            native_vacuous_exit(false, true),
            GovernorDecision::InfraFailure
        );
        assert_eq!(
            native_vacuous_exit(false, false),
            GovernorDecision::InfraFailure
        );
    }

    // [VANTA-WHY] v1-m3.1 F3 — `native_vacuous_seen` (run.summary's
    // `native_vacuous`) documents an INFRASTRUCTURE fault, so it is set ONLY on the
    // `InfraFailure` exit. A gate made vacuous by the deadline clamp takes the
    // deadline exit (`budget_exhausted`/`NeedsReview`) and records
    // `native_vacuous=false` plus `native_gate_clamped=true`. Test both.
    #[test]
    fn a_vacuous_gate_sets_the_infra_flag_only_on_the_infrastructure_exit() {
        // The clamp case: 3000 s left under a 7250 s cap IS a clamp, and an
        // original side killed by it is a DEADLINE event.
        let gate_cap = native_gate_cap(7250, Some(3000)).expect("time remains, so the gate runs");
        let gate_clamped = gate_cap < 7250;
        assert!(gate_clamped, "3000 s left under a 7250 s cap is a clamp");
        assert_eq!(
            native_vacuous_exit(gate_clamped, true),
            GovernorDecision::NeedsReview
        );
        assert!(
            !native_vacuous_is_infrastructure(gate_clamped, true),
            "a clamp-caused vacuity must NOT set native_vacuous_seen"
        );

        // The infrastructure cases: a real collection error inside the shortened
        // window (clamped but the side was not killed) and a run under the full cap
        // both set the flag.
        assert!(native_vacuous_is_infrastructure(gate_clamped, false));
        assert!(native_vacuous_is_infrastructure(false, false));
        assert!(native_vacuous_is_infrastructure(false, true));

        // And the flag is the exact complement of the exit the gate takes.
        for gate_clamped in [false, true] {
            for original_timed_out in [false, true] {
                assert_eq!(
                    native_vacuous_is_infrastructure(gate_clamped, original_timed_out),
                    native_vacuous_exit(gate_clamped, original_timed_out)
                        != GovernorDecision::NeedsReview,
                );
            }
        }
    }

    // [VANTA-WHY] Finding C (PR #4 review) — `native_timeout` (run.summary's
    // `native_timeout`) must name a kill by the CONFIGURED cap, never a kill by the
    // deadline clamp. The clamped kill is recorded by `native_gate_clamped` instead,
    // and the row reads `native_timeout=false, native_vacuous=false,
    // native_gate_clamped=true` — the exact triple the review asked for.
    #[test]
    fn a_clamped_kill_records_the_clamp_not_a_native_timeout() {
        // The clamp case: 3000 s left under a 7250 s cap IS a clamp, and a side
        // killed by it was killed by the CLOCK, not the configured cap.
        let gate_cap = native_gate_cap(7250, Some(3000)).expect("time remains, so the gate runs");
        let gate_clamped = gate_cap < 7250;
        assert!(gate_clamped, "3000 s left under a 7250 s cap is a clamp");
        assert!(
            !native_timeout_flag(true, gate_clamped, true, false),
            "a clamped kill of the original side must NOT set native_timeout"
        );
        assert!(
            !native_timeout_flag(true, gate_clamped, false, true),
            "a clamped kill of the patched side must NOT set native_timeout either"
        );

        // An UNCLAMPED vacuous gate killed by the configured cap DOES set it, on
        // EITHER side — the flag's one documented meaning.
        assert!(native_timeout_flag(true, false, true, false));
        assert!(native_timeout_flag(true, false, false, true));
        // A vacuous gate that was not killed (a tree that collected nothing) and a
        // non-vacuous gate never set it; and `native_timeout=true` keeps implying
        // `native_vacuous=true` for every combination.
        assert!(!native_timeout_flag(true, false, false, false));
        assert!(!native_timeout_flag(false, false, true, true));
        for native_vacuous in [false, true] {
            for gate_clamped in [false, true] {
                for original_timed_out in [false, true] {
                    for patched_timed_out in [false, true] {
                        let flag = native_timeout_flag(
                            native_vacuous,
                            gate_clamped,
                            original_timed_out,
                            patched_timed_out,
                        );
                        assert!(
                            !flag || native_vacuous,
                            "native_timeout must always imply native_vacuous"
                        );
                        assert!(
                            !flag || !gate_clamped,
                            "a clamped kill must never set native_timeout"
                        );
                    }
                }
            }
        }
    }

    // [VANTA-WHY] v1-m3.1 (§8.1/§8.5, case a) — a PATCHED side KILLED by the clamp
    // is NOT §8.1 row 1: the ORIGINAL side ran, so the emptiness is the patch's own
    // cost. It stays in the attempt loop as a FAILED ATTEMPT (the Governor sees every
    // original nodeid vanish from the patched report — a native regression), never the
    // vacuous `InfraFailure` branch. The run then continues to the next attempt top,
    // which sees the passed deadline and takes the existing `NeedsReview`
    // (`budget_exhausted`) exit — the same path every other deadline expiry takes.
    #[test]
    fn a_patched_side_killed_by_the_clamp_is_a_failed_attempt_not_the_vacuous_exit() {
        let original = clamped_report(&["t/test_x.py::test_a"], true, false);
        let patched = clamped_report(&[], false, true);
        // The guard is keyed on the ORIGINAL side ALONE: the original ran, so the gate
        // is NOT vacuous and the infra branch is never reached.
        assert!(
            !crate::pipeline::harness::native_gate_vacuous(1, &original),
            "a patched-side kill over a running original is not a vacuous gate"
        );
        // The patch's emptiness reaches the Governor as a regression — a failed
        // attempt — because every original nodeid is absent from the patched report.
        let regressions =
            crate::pipeline::harness::native_regressions(&original.tests, &patched.tests);
        assert_eq!(regressions, vec!["t/test_x.py::test_a".to_string()]);
    }

    // [VANTA-WHY] Finding A (PR #4 review) — a PATCHED side the harness could not
    // SET UP or transport measured nothing, so `native_regressions` over it would
    // list every ORIGINAL test as a regression and fail the attempt on a suite Vanta
    // never ran. The marker (`NativeSuiteReport::setup_failed`) is read BEFORE the
    // regression list, and the run ends as `InfraFailure` — in BOTH repair and review
    // modes, since the guard takes no `GovernorMode`.
    #[test]
    fn a_setup_failed_patched_side_is_an_infrastructure_fault_not_a_regression() {
        let original = clamped_report(&["t/test_x.py::test_a"], true, false);
        let mut patched = clamped_report(&[], false, false);
        patched.setup_failed = true;
        let vacuous = crate::pipeline::harness::native_gate_vacuous(1, &original);
        assert!(!vacuous, "the original ran, so the gate is not vacuous");
        // The list the driver would otherwise build blames the patch ...
        assert_eq!(
            crate::pipeline::harness::native_regressions(&original.tests, &patched.tests),
            vec!["t/test_x.py::test_a".to_string()],
            "an unattributed empty patched report names every original test as a regression"
        );
        // ... and this is the marker that stops it.
        assert!(
            native_setup_failure(original.setup_failed, patched.setup_failed, vacuous),
            "a patched side the harness never set up is an infrastructure fault, not a failed attempt"
        );
        // An ATTRIBUTABLE empty patched side over a running original is unchanged:
        // the patch's own cost, a failed attempt (row 3 of §8.1).
        assert!(!native_setup_failure(false, false, false));
        // An ORIGINAL side that could not be set up is the fault UNCONDITIONALLY —
        // even under a clamp, its emptiness is not the clock's doing and must not
        // take the clamp's `budget_exhausted` exit.
        assert!(native_setup_failure(true, false, true));
        assert!(native_setup_failure(true, false, false));
        // But an original that is vacuous WITHOUT a setup failure keeps its own,
        // more specific reason: an original vacuity already ends the run as
        // `InfraFailure` through the vacuity branch.
        assert!(!native_setup_failure(false, true, true));
    }

    // [VANTA-WHY] Finding A (PR #4 review) — the guard is worth nothing if it does
    // not run in the shared attempt loop, and it must run BEFORE the regression list
    // is built (and before the timeout flag), so a setup failure never counts as a
    // candidate regression. The judge's own suite takes the same exit. Pin both call
    // sites.
    #[test]
    fn the_setup_failure_guard_runs_before_native_regressions_in_the_shared_loop() {
        let source = include_str!("driver.rs");
        let guard = source
            .find("if native_setup_failure(")
            .expect("the shared loop must take the setup-failure guard");
        let regressions = source
            .find("let mut native_regressions =")
            .expect("the native gate must build its regression list");
        assert!(
            guard < regressions,
            "the setup-failure guard must run BEFORE `native_regressions`, or a machine \
             fault is counted as a candidate regression"
        );
        assert!(
            source.contains("if report.setup_failed {"),
            "the judge's native suite must take the same setup-failure exit"
        );
    }

    // [VANTA-WHY] v1-m3.1 (§8.5, case a) — an empty patched report over a large
    // suite must not blow the next Creator prompt. Every nodeid the original passed
    // is a regression when the patched side collected NOTHING, so on a suite this
    // size the unbounded list could not fit any window and the run would end as an
    // `InfraFailure` instead of retrying. Bounded, the feedback stays tiny, carries
    // the patched-side stderr tail (the ONLY diagnostic an empty report has, since
    // absent nodeids carry no failure output), and names the true total.
    #[test]
    fn an_empty_patched_report_over_a_large_suite_stays_under_the_prompt_budget_and_carries_the_stderr_tail(
    ) {
        // The regression list a caller builds: the whole suite, in report order.
        let mut regressions: Vec<String> = (0..88_642)
            .map(|i| format!("pandas/tests/io/test_{:04}.py::test_case_{}", i % 2_000, i))
            .collect();
        // Deliberately shuffled, so the test proves the >limit branch SORTS rather
        // than reprinting whatever order the report parser emitted.
        regressions.reverse();

        let mut patched = clamped_report(&[], false, false);
        patched.stderr_tail =
            "collector: ImportError: cannot import name 'numpy_helper' | native report collected 0 tests"
                .to_string();

        let fb = build_creator_feedback(&[], false, false, &regressions, &patched, &[], &[], &[]);

        // (1) Under the prompt budget. Priced exactly as the call site prices it —
        // the file-selection budget, the frame, and the fixed output/overhead
        // reserves — the whole prompt still fits the configured 262144-token window.
        let prompt = 163_584 + crate::context::estimate_tokens(fb.len()) + 3_000;
        assert!(
            prompt_over_budget_reason(prompt, crate::context::OUTPUT_RESERVE_TOKENS, 262_144)
                .is_none(),
            "bounded feedback must fit the window; feedback was {} bytes ({prompt} est. tokens)",
            fb.len()
        );
        // (2) Carries the patched-side stderr tail, the only cause an empty report has.
        assert!(
            fb.contains("cannot import name 'numpy_helper'"),
            "the patched suite's stderr tail must be in the feedback:\n{fb}"
        );
        // (3) Bounded to the sample and naming the true total plus the remainder.
        let shown = fb.matches("\n  - pandas/tests/io/").count();
        assert_eq!(
            shown, REGRESSION_FEEDBACK_SAMPLE,
            "only the bounded sample is named; got {shown}"
        );
        assert!(fb.contains("88642 tests regressed"), "{fb}");
        assert!(fb.contains("more in the evidence artifact"), "{fb}");
        // The sample is the FIRST 20 nodeids, sorted — not the reversed report order.
        let mut sorted: Vec<&String> = regressions.iter().collect();
        sorted.sort();
        assert!(
            fb.contains(&format!("  - {}\n", sorted[0])),
            "the sample must be sorted; expected {} first:\n{fb}",
            sorted[0]
        );
        assert!(
            !fb.contains(&format!("  - {}\n", regressions[0])),
            "the unsorted report order must not decide the sample"
        );
    }

    // [VANTA-WHY] Finding D (PR #4 review) — in the bounded sample a nodeid the
    // patched report RECORDED a failure for must be named before the absent ones. A
    // partially running patched suite can have thousands of absent nodeids that sort
    // before its one real assertion failure; taking the plain first 20 gives the
    // Creator neither the failure nor (if it did collect) the collapsed-suite stderr.
    #[test]
    fn a_lone_failing_nodeid_sorted_late_is_kept_in_the_bounded_sample() {
        // 60 regressions — above REGRESSION_FEEDBACK_LIMIT — of which all but one are
        // simply ABSENT from the patched report.
        let regressions: Vec<String> = (0..60)
            .map(|i| format!("pkg/test_{:03}.py::test_{:03}", i, i))
            .collect();
        let mut sorted: Vec<&String> = regressions.iter().collect();
        sorted.sort();
        let failing: String = sorted.last().expect("60 regressions").to_string();
        // The one nodeid that actually failed sorts LAST, so the plain first-20 sample
        // would drop it entirely.
        assert!(
            !sorted
                .iter()
                .take(REGRESSION_FEEDBACK_SAMPLE)
                .any(|nodeid| **nodeid == failing),
            "sanity: the failing nodeid is outside the naive first-20 sample"
        );

        let mut patched = clamped_report(&[], true, false);
        patched
            .failure_outputs
            .insert(failing.clone(), "AssertionError: boom".to_string());

        let fb = build_creator_feedback(&[], false, false, &regressions, &patched, &[], &[], &[]);

        // Still bounded to the sample size.
        let shown = fb.matches("\n  - pkg/test_").count();
        assert_eq!(
            shown, REGRESSION_FEEDBACK_SAMPLE,
            "the sample stays bounded; got {shown}"
        );
        // The recorded failure is kept, output and all, even though it sorted last.
        assert!(
            fb.contains(&format!("  - {failing}\n")),
            "the lone failing nodeid must survive the bounded sample:\n{fb}"
        );
        assert!(
            fb.contains("    failure: AssertionError: boom\n"),
            "its failure output must ride along:\n{fb}"
        );
        // The remaining slots are filled with the absent nodeids, in sorted order.
        assert!(
            fb.contains(&format!("  - {}\n", sorted[0])),
            "the absent nodeids still fill the rest of the sample:\n{fb}"
        );
    }

    // [VANTA-WHY] v1-m3.1 (§8.5) — at or below the limit the feedback is
    // BYTE-IDENTICAL to the unbounded message the code produced before the bound
    // existed: every nodeid in the report's own order, with its failure output, and
    // NO stderr-tail line (the patched side ran, so it is not the empty case). The
    // bound changes only what a huge list looks like; it must change nothing else.
    #[test]
    fn a_regression_list_at_the_limit_is_byte_identical_to_the_unbounded_feedback() {
        let regressions: Vec<String> = (0..10).map(|i| format!("t/test_{i}.py::test_{i}")).collect();
        let names: Vec<&str> = regressions.iter().map(String::as_str).collect();
        let mut patched = clamped_report(&names, true, false);
        patched.failure_outputs.insert(
            "t/test_3.py::test_3".to_string(),
            "AssertionError: boom".to_string(),
        );

        let fb = build_creator_feedback(&[], false, false, &regressions, &patched, &[], &[], &[]);

        let mut expected = String::from("The previous patch failed verification:\n\n");
        expected
            .push_str("- The repository's own tests regressed (run with the ORIGINAL test files):\n");
        for i in 0..10 {
            expected.push_str(&format!("  - t/test_{i}.py::test_{i}\n"));
            if i == 3 {
                expected.push_str("    failure: AssertionError: boom\n");
            }
        }
        expected.push_str(
            "\nPlease revise the patch to fix these failures while preserving existing behavior.",
        );
        assert_eq!(fb, expected, "the small-list message must be unchanged");
    }

    // [VANTA-WHY] M3 item 1 — the grounding ledger separates "the audit ran and
    // found nothing" from "no audit ran". `run.summary` is space-free
    // `key=value`, so the two cases have to be two tokens and `0` cannot stand in
    // for the absence of a measurement.
    #[test]
    fn the_grounding_ledger_writes_null_when_no_audit_ran_and_a_count_when_one_did() {
        assert_eq!(ledger_text(None), "null");
        assert_eq!(ledger_text(Some(0)), "0");
        assert_eq!(ledger_text(Some(7)), "7");
        assert_ne!(
            ledger_text(None),
            ledger_text(Some(0)),
            "an audit that ran and scrapped nothing is not the same row as one that never ran"
        );
    }

    /// A `CallUsage` with only the two fields `creator_prompt_tokens_text` reads
    /// set; built through `serde` so a new field on the struct cannot make this
    /// helper lie about the ones it does set.
    fn a_usage(site: &str, prompt_tokens: Option<u64>) -> crate::llm::CallUsage {
        serde_json::from_value(serde_json::json!({
            "seq": 0,
            "site": site,
            "round": 1,
            "attempt": 1,
            "provider_retries": 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": null,
            "total_tokens": null,
            "cached_tokens": null,
            "reasoning_tokens": null,
            "cost_usd": null,
            "cost_basis": "not_billed",
            "source": "server-usage",
            "seed": null,
            "seed_mechanism": "unset",
            "finish_reason": "stop",
            "wall_ms": 0,
            "truncated": false,
            "request_shape": {"requested": [], "sent": [], "dropped": [], "renamed": []},
            "service_tier": "unset"
        }))
        .expect("a CallUsage with only the fields the reader needs")
    }

    /// Item (a) — the run log must carry WHY BATE stopped, not just that it did.
    #[test]
    fn a_harness_error_quotes_its_evidence_into_the_run_log() {
        let line = harness_error_log_line(
            "test_middleware_order",
            "verdict: ERROR (the test was never graded: it failed to import or collect)\n\
             exception: ModuleNotFoundError: No module named 'sanic.websocket'",
        );
        assert!(
            line.contains("test_middleware_order"),
            "the failing test must be named: {line}"
        );
        assert!(
            line.contains("ModuleNotFoundError: No module named 'sanic.websocket'"),
            "the run log must quote the error that broke the environment: {line}"
        );
        assert!(
            line.contains("never reached a verdict"),
            "and still state what happened in BATE's own words: {line}"
        );
    }

    /// Item S — the Creator's OWN effort and its mechanism, apart from the
    /// run-level `reasoning_effort` switch.
    #[test]
    fn the_creator_effort_field_names_what_the_creator_body_carried() {
        // The local backend: the call site's pin is sent verbatim and the
        // server ignores it, whatever the switch says.
        assert_eq!(
            creator_effort_text(&LlmConfig::default()),
            ("low".to_string(), "llamacpp:noop")
        );
        // A hosted run that asked for no level: the shaper drops the call site's
        // `low`, so the body carries none of our making and the level is the
        // provider's. `none` is the honest value — nobody sent a level.
        let hosted = LlmConfig {
            backend: LlmBackend::OpenAi,
            model_id: "gpt-6-luna".to_string(),
            ..LlmConfig::default()
        };
        assert_eq!(
            creator_effort_text(&hosted),
            ("none".to_string(), "unset")
        );
    }

    /// Item S — the sampling knobs the preset PINS that this backend did NOT put
    /// in the body. On a hosted reasoning model the model samples at the
    /// provider's own default and the row must say so.
    #[test]
    fn the_sampling_not_sent_field_names_the_preset_keys_the_backend_dropped() {
        // Local: every preset pin reaches the server.
        assert_eq!(
            sampling_not_sent_rule(&LlmConfig::default()),
            "sampling_not_sent=none"
        );
        // A hosted known family: §14.26's shaper drops `temperature` and `top_p`,
        // so their effective value is the provider's own.
        let hosted = LlmConfig {
            backend: LlmBackend::OpenAi,
            model_id: "gpt-6-luna".to_string(),
            ..LlmConfig::default()
        };
        assert_eq!(
            sampling_not_sent_rule(&hosted),
            "sampling_not_sent=temperature:provider_default,top_p:provider_default"
        );
        // A hosted model this table does not know: the per-family rejection is
        // not applied, so the shaper keeps both and nothing is unsent.
        let unknown = LlmConfig {
            backend: LlmBackend::OpenAi,
            model_id: "qwen3.8-27b".to_string(),
            ..LlmConfig::default()
        };
        assert_eq!(sampling_not_sent_rule(&unknown), "sampling_not_sent=none");
    }

    /// Item S — the real assembled prompt tokens per Creator call, in call
    /// order, never scored as a zero.
    #[test]
    fn the_creator_prompt_tokens_are_recorded_per_call_and_never_scored_as_zero() {
        let calls = vec![
            a_usage("creator", Some(1234)),
            a_usage("attacker_generate", Some(999)),
            a_usage("creator", None),
        ];
        assert_eq!(creator_prompt_tokens_text(&calls), "1234,unmeasured");
        // A run whose Creator call published no `usage` block records the word,
        // not a `0` that would read as an empty prompt. No Creator call at all
        // is `none`, not an empty string.
        assert_eq!(creator_prompt_tokens_text(&[]), "none");
    }
}
