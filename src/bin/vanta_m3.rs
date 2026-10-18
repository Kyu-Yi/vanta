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
    repo_path: PathBuf,
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

    let repo_path = repo_path.ok_or("missing --repo PATH")?;

    Ok(CliArgs {
        repo_path,
        issue_path,
        max_attempts,
        model_override,
        open_pr,
        patch_path,
        test_cmd,
        original_tree,
    })
}

fn print_help() {
    println!(
        "Vanta M3 — Adversarial verification repair loop\n\
         \n\
         USAGE:\n    vanta --repo PATH [--issue PATH] [--max-attempts N] [--model NAME] [--open-pr] [--patch PATH] [--test-cmd CMD] [--original-tree PATH]\n\
         \n\
         OPTIONS:\n    --repo, -r PATH        Path to the repository to repair\n    \
         --issue, -i PATH       Path to the issue/bug description (default: auto-detect)\n    \
         --max-attempts, -n N   Max repair attempts (default: 4)\n    \
         --model, -m NAME       Override LLM model (default: from VANTA_LLM_MODEL)\n    \
         --open-pr              Open a GitHub PR after Gatekeeper approval\n    \
         --patch, -p PATH       Verify an EXISTING patch (skip Creator, run Attacker+BATE on this diff)\n    \
         --test-cmd CMD         Test command (default: 'python3 -m pytest'). e.g. 'pytest tests/ -q'\n    \
         --original-tree PATH   Clean original tree (git worktree of base commit) for Mode B two-tree mode\n    \
         --help, -h             Show this help\n"
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_logging();

    let cli = parse_args().map_err(|e| format!("CLI error: {e}\n\nUse --help for usage."))?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VANTA M3 — Repair Loop (Creator → Attacker → BATE → Governor)║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let decision = driver::run(RunConfig {
        repo_path: cli.repo_path,
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
