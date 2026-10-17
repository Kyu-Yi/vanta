// ============================================================================
// vanta_m2 — thin CLI wrapper: the SAME driver as M3, pinned to N = 1
// ============================================================================
//
// M2 SCOPE (per Spec): sandbox + BATE + Diagnostician. NO repair loop, NO
// Governor, NO PR.
//
// [VANTA-WHY] M2 is not a separate pipeline. It is the shared driver
// (`vanta::pipeline::driver`) with `max_attempts = 1`, so the two runs execute
// byte-identical code and a fix to the loop lands in both. Before this, M2
// carried its own copy of the attempt cycle and its own patch-application
// cascade, and the copies drifted.
//
// Flow (owned by the driver):
//   Creator → Attacker → BATE (validate + revise) → patched run (with static
//   diff) → Diagnostician (classify + Phase-2 remediation) → Session log
//
// The sample target is the bundled `examples/buggy_python` repo.
// ============================================================================

use std::path::PathBuf;

use vanta::logging::init_logging;
use vanta::pipeline::driver::{self, RunConfig};

const BUGGY_DIR: &str = "examples/buggy_python";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_logging();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VANTA M2 — Creator → Attacker → BATE → Diagnostician       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // [VANTA-WHY] M2 is the shared driver at N = 1: one attempt, no repair.
    // Everything else (which repo, which issue) is a config value, never a
    // different code path.
    let repo_path = PathBuf::from(BUGGY_DIR);
    let issue_path = Some(repo_path.join("issue.md"));

    driver::run(RunConfig {
        repo_path,
        issue_path,
        max_attempts: 1,
        model_override: None,
        open_pr: false,
        patch_path: None,
        test_cmd: None,
        original_tree: None,
    })
    .await?;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  M2 PIPELINE COMPLETE                                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    Ok(())
}
