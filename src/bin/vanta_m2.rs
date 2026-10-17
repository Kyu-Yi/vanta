// ============================================================================
// vanta_m2 — thin CLI wrapper: the SAME driver as M3, pinned to N = 1
// ============================================================================
//
// M2 SCOPE (per Spec): sandbox + BATE + Diagnostician. NO repair loop, NO PR.
//
// [VANTA-WHY] M2 is not a separate pipeline. It is the shared driver
// (`vanta::pipeline::driver`) with `max_attempts = 1`, so the two runs execute
// byte-identical code and a fix to the loop lands in both. This wrapper only
// exposes the repo/issue/patch/test-command surface needed to point that same
// driver at different targets.
// ============================================================================

use std::path::PathBuf;

use vanta::logging::init_logging;
use vanta::pipeline::driver::{self, RunConfig};

const BUGGY_DIR: &str = "examples/buggy_python";

struct CliArgs {
    repo_path: PathBuf,
    issue_path: Option<PathBuf>,
    model_override: Option<String>,
    patch_path: Option<PathBuf>,
    test_cmd: Option<String>,
    original_tree: Option<PathBuf>,
}

fn parse_args() -> Result<CliArgs, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut repo_path: Option<PathBuf> = None;
    let mut issue_path: Option<PathBuf> = None;
    let mut model_override: Option<String> = None;
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
            "--model" | "-m" => {
                i += 1;
                if i >= args.len() {
                    return Err("--model requires a model name".to_string());
                }
                model_override = Some(args[i].clone());
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
                if repo_path.is_none() {
                    repo_path = Some(PathBuf::from(other));
                } else {
                    return Err(format!("unknown argument: {other}"));
                }
            }
        }
        i += 1;
    }

    let repo_path = repo_path.unwrap_or_else(|| PathBuf::from(BUGGY_DIR));
    let issue_path = issue_path.or_else(|| Some(repo_path.join("issue.md")));

    Ok(CliArgs {
        repo_path,
        issue_path,
        model_override,
        patch_path,
        test_cmd,
        original_tree,
    })
}

fn print_help() {
    println!(
        "Vanta M2 — single-attempt verification over the shared driver\n\
         \n\
         USAGE:\n    vanta_m2 [--repo PATH] [--issue PATH] [--model NAME] [--patch PATH] [--test-cmd CMD] [--original-tree PATH]\n\
         \n\
         OPTIONS:\n    --repo, -r PATH        Path to the repository to verify (default: examples/buggy_python)\n    \
         --issue, -i PATH       Path to the issue/bug description (default: <repo>/issue.md)\n    \
         --model, -m NAME       Override LLM model (default: from VANTA_LLM_MODEL)\n    \
         --patch, -p PATH       Verify an EXISTING patch (skip Creator, run Attacker+BATE on this diff)\n    \
         --test-cmd CMD         Test command (default: 'python3 -m pytest'). e.g. 'pytest tests/ -q'\n    \
         --original-tree PATH   Clean original tree (git worktree of base commit) for two-tree mode\n    \
         --help, -h             Show this help\n"
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_logging();

    let cli = parse_args().map_err(|e| format!("CLI error: {e}\n\nUse --help for usage."))?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VANTA M2 — Creator → Attacker → BATE → Diagnostician       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    driver::run(RunConfig {
        repo_path: cli.repo_path,
        issue_path: cli.issue_path,
        max_attempts: 1,
        model_override: cli.model_override,
        open_pr: false,
        patch_path: cli.patch_path,
        test_cmd: cli.test_cmd,
        original_tree: cli.original_tree,
    })
    .await?;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  M2 PIPELINE COMPLETE                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    Ok(())
}
