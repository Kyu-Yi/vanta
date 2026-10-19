// ============================================================================
// vanta_m3 — thin CLI wrapper over the shared driver (Spec §7, §8)
// ============================================================================
//
// [VANTA-WHY] This binary used to carry its own copy of the attempt cycle.
// M2 carried another, and the two drifted (M3's BATE lost the Attacker-revise
// layer; M3's Diagnostician lost Phase-2 remediation). The loop now lives in
// exactly one place — `vanta::pipeline::driver` — and this file is only the
// argument surface plus an exit-code mapping. M3 is the driver at N attempts
// (default 4); M2 is the SAME driver at N = 1.
//
// Flow (owned by the driver, listed here for orientation):
//   Creator → Attacker → BATE → patched run → Diagnostician → Governor →
//   Gatekeeper → (PR)
// ============================================================================

use std::path::PathBuf;

use vanta::core::GovernorDecision;
use vanta::logging::init_logging;
use vanta::pipeline::driver::{self, RunConfig};

// ============================================================================
// CLI — manual arg parsing (no clap dep; keep Cargo.toml minimal)
// ============================================================================
struct CliArgs {
    repo_path: Option<PathBuf>,
    issue_path: Option<PathBuf>,
    max_attempts: u32,
    model_override: Option<String>,
    /// If true, open a GitHub PR after Gatekeeper approval.
    open_pr: bool,
    /// If Some(path), skip the Creator and use this patch file directly.
    /// Used for verifying an EXISTING patch (e.g. a developer's working tree diff).
    patch_path: Option<PathBuf>,
    /// Explicit test command (Spec §7: "Require explicit --test-cmd, pytest as
    /// fallback. Do not infer."). e.g. "pytest tests/ -q".
    test_cmd: Option<String>,
    /// Path to a clean original tree (git worktree of the base commit) for
    /// Mode B. If set, BATE runs against this tree instead of the in-memory
    /// original files. Spec §7: "both trees present simultaneously".
    original_tree: Option<PathBuf>,
    /// If Some(session_id), do NOT run the pipeline: re-run the signals behind
    /// that session's review-mode rejection against a known-good tree and write
    /// the per-signal verdicts. Item 11 of the heretic fix list.
    rerun_signals: Option<String>,
    /// Known-good source tree for `--rerun-signals` (a directory, or a patch
    /// file that `--base-tree` is applied to).
    good_tree: Option<PathBuf>,
    /// Base tree the `--good-tree` patch applies to; required only when
    /// `--good-tree` is a patch file.
    base_tree: Option<PathBuf>,
    /// Explicit opt-in to re-run a `subprocess`-recorded session under shuru.
    /// Refused when the recorded `test_cmd` names a host path the VM cannot see.
    allow_backend_upgrade: bool,
}

fn parse_args() -> Result<CliArgs, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut repo_path: Option<PathBuf> = None;
    let mut issue_path: Option<PathBuf> = None;
    let mut max_attempts: u32 = 4;
    let mut model_override: Option<String> = None;
    let mut open_pr = false;
    let mut patch_path: Option<PathBuf> = None;
    let mut test_cmd: Option<String> = None;
    let mut original_tree: Option<PathBuf> = None;
    let mut rerun_signals: Option<String> = None;
    let mut good_tree: Option<PathBuf> = None;
    let mut base_tree: Option<PathBuf> = None;
    let mut allow_backend_upgrade = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--repo" | "-r" => {
                i += 1;
                if i >= args.len() {
                    return Err("--repo requires a path".to_string());
                }
                repo_path = Some(PathBuf::from(&args[i]));
            }
            "--issue" | "-i" => {
                i += 1;
                if i >= args.len() {
                    return Err("--issue requires a path".to_string());
                }
                issue_path = Some(PathBuf::from(&args[i]));
            }
            "--max-attempts" | "-n" => {
                i += 1;
                if i >= args.len() {
                    return Err("--max-attempts requires a number".to_string());
                }
                max_attempts = args[i]
                    .parse()
                    .map_err(|e| format!("--max-attempts: {e}"))?;
            }
            "--model" | "-m" => {
                i += 1;
                if i >= args.len() {
                    return Err("--model requires a model name".to_string());
                }
                model_override = Some(args[i].clone());
            }
            "--open-pr" | "--pr" => {
                open_pr = true;
            }
            "--patch" | "-p" => {
                i += 1;
                if i >= args.len() {
                    return Err("--patch requires a path".to_string());
                }
                patch_path = Some(PathBuf::from(&args[i]));
            }
            "--test-cmd" => {
                i += 1;
                if i >= args.len() {
                    return Err("--test-cmd requires a command".to_string());
                }
                test_cmd = Some(args[i].clone());
            }
            "--original-tree" => {
                i += 1;
                if i >= args.len() {
                    return Err("--original-tree requires a path".to_string());
                }
                original_tree = Some(PathBuf::from(&args[i]));
            }
            "--rerun-signals" => {
                i += 1;
                if i >= args.len() {
                    return Err("--rerun-signals requires a session id".to_string());
                }
                rerun_signals = Some(args[i].clone());
            }
            "--good-tree" => {
                i += 1;
                if i >= args.len() {
                    return Err("--good-tree requires a path".to_string());
                }
                good_tree = Some(PathBuf::from(&args[i]));
            }
            "--base-tree" => {
                i += 1;
                if i >= args.len() {
                    return Err("--base-tree requires a path".to_string());
                }
                base_tree = Some(PathBuf::from(&args[i]));
            }
            "--allow-backend-upgrade" => {
                allow_backend_upgrade = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                // Allow positional: first positional = repo
                if repo_path.is_none() {
                    repo_path = Some(PathBuf::from(other));
                } else {
                    return Err(format!("unknown argument: {other}"));
                }
            }
        }
        i += 1;
    }

    // [VANTA-WHY] `--repo` is only required for the run path. In `--rerun-signals`
    // mode the session artifacts and the good tree ARE the inputs, and demanding
    // a repo would force the caller to pass a path the rerun never reads.
    if rerun_signals.is_none() && repo_path.is_none() {
        return Err("missing --repo PATH".to_string());
    }

    Ok(CliArgs {
        repo_path,
        issue_path,
        max_attempts,
        model_override,
        open_pr,
        patch_path,
        test_cmd,
        original_tree,
        rerun_signals,
        good_tree,
        base_tree,
        allow_backend_upgrade,
    })
}

fn print_help() {
    println!(
        "Vanta M3 — Adversarial verification repair loop\n\
         \n\
         USAGE:\n    vanta --repo PATH [--issue PATH] [--max-attempts N] [--model NAME] [--open-pr] [--patch PATH] [--test-cmd CMD] [--original-tree PATH]\n    \
         vanta --rerun-signals SESSION_ID --good-tree PATH [--base-tree PATH] [--allow-backend-upgrade]\n\
         \n\
         OPTIONS:\n    --repo, -r PATH        Path to the repository to repair\n    \
         --issue, -i PATH       Path to the issue/bug description (default: auto-detect)\n    \
         --max-attempts, -n N   Max repair attempts (default: 4)\n    \
         --model, -m NAME       Override LLM model (default: from VANTA_LLM_MODEL)\n    \
         --open-pr              Open a GitHub PR after Gatekeeper approval\n    \
         --patch, -p PATH       Verify an EXISTING patch (skip Creator, run Attacker+BATE on this diff)\n    \
         --test-cmd CMD         Test command (default: 'python3 -m pytest'). e.g. 'pytest tests/ -q'\n    \
         --original-tree PATH   Clean original tree (git worktree of base commit) for Mode B two-tree mode\n    \
         --rerun-signals ID     Re-run session ID's review-mode rejection signals against --good-tree (writes vanta_sessions/ID_signal_reruns.json; no --repo needed)\n    \
         --good-tree PATH       Known-good source: a directory, or a patch file applied onto --base-tree\n    \
         --base-tree PATH       Base tree the --good-tree patch applies to (required only for a patch file)\n    \
         --allow-backend-upgrade  Re-run a subprocess-recorded session under shuru (refused if test_cmd names a host path)\n    \
         --help, -h             Show this help\n"
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_logging();

    let cli = parse_args().map_err(|e| format!("CLI error: {e}\n\nUse --help for usage."))?;

    // [VANTA-WHY] The rerun is a DIFFERENT program shape: it reads a finished
    // session's artifacts instead of driving a new run, so it short-circuits
    // before the banner and the driver. It never needs --repo, and it must not
    // fall through to the repair loop (which would need one).
    if let Some(session_id) = cli.rerun_signals {
        let good_tree = cli
            .good_tree
            .ok_or("--rerun-signals requires --good-tree PATH")?;
        let config = vanta::pipeline::rerun::RerunConfig {
            session_id,
            good_tree,
            base_tree: cli.base_tree,
            allow_backend_upgrade: cli.allow_backend_upgrade,
            sessions_dir: PathBuf::from("vanta_sessions"),
        };
        match vanta::pipeline::rerun::rerun_signals(config).await {
            Ok(path) => {
                println!("▌ Signal reruns written to {}", path.display());
                return Ok(());
            }
            Err(error) => {
                eprintln!("rerun-signals failed: {error}");
                std::process::exit(1);
            }
        }
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VANTA M3 — Repair Loop (Creator → Attacker → BATE → Governor)║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let repo_path = cli
        .repo_path
        .ok_or("missing --repo PATH")?;

    let decision = driver::run(RunConfig {
        repo_path,
        issue_path: cli.issue_path,
        max_attempts: cli.max_attempts,
        model_override: cli.model_override,
        open_pr: cli.open_pr,
        patch_path: cli.patch_path,
        test_cmd: cli.test_cmd,
        original_tree: cli.original_tree,
    })
    .await?;

    // Exit code: 0 if approved, 1 otherwise
    if decision == GovernorDecision::AllTestsPassed {
        Ok(())
    } else {
        std::process::exit(1);
    }
}
