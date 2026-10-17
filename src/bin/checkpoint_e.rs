// ============================================================================
// CHECKPOINT E — Sandbox Subprocess Backend + pytest --json-report parsing
// ============================================================================
//
// WHAT YOU'LL LEARN FROM THIS BINARY
//
//   # | Rust concept              → Where Vanta reuses it later
//  ---+---------------------------+-------------------------------------------
//   1 | SandboxBackend trait      → Spec §7 single interface for spawn/execute/
//     |                           →   destroy. M2 calls this without knowing
//     |                           →   if it's subprocess or shuru microVM.
//  ---+---------------------------+-------------------------------------------
//   2 | SubprocessBackend         → Fast path: copy repo to tempdir, run pytest,
//     |                           →   capture stdout/stderr, enforce timeout.
//  ---+---------------------------+-------------------------------------------
//   3 | tokio::time::timeout      → Hard execution deadline. M2 uses 30s for
//     |                           →   test batches; 120s for full suites.
//  ---+---------------------------+-------------------------------------------
//   4 | serde_json::Value parsing → pytest --json-report produces a JSON
//     |                           →   document; we extract per-test outcomes.
//  ---+---------------------------+-------------------------------------------
//   5 | TempDir RAII              → Dropping the backend deletes the temp dir
//     |                           →   automatically. No manual cleanup needed.
// ============================================================================

use std::path::{Path, PathBuf};

use vanta::sandbox::subprocess::SubprocessBackend;
use vanta::sandbox::SandboxBackend;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CHECKPOINT E — Subprocess sandbox + pytest --json-report    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let src = PathBuf::from("examples/buggy_python");
    if !src.exists() {
        return Err(format!("Sample repo not found at {}", src.display()).into());
    }

    // ---- Spawn sandbox (temp dir) ----
    let sb = SubprocessBackend::spawn().await?;
    println!("▌ Sandbox spawned at {}", sb.root.display());

    // ---- Copy repo into sandbox ----
    sb.copy_repo_in(&src)?;
    println!("▌ Copied {} → sandbox", src.display());

    // ---- Write a tiny test file inside the sandbox ----
    // [VANTA-WHY] In M2 the Attacker writes these tests. Here we hardcode one
    // to prove the execution + JSON parsing pipeline works end-to-end.
    let test_code = r#"
from cart import cart_total

def test_small_cart():
    assert cart_total([(100, 2), (50, 1)]) == 250

def test_empty_cart():
    assert cart_total([]) == 0
"#;
    let test_path = sb.root.join("test_cart.py");
    std::fs::write(&test_path, test_code)?;
    println!("▌ Wrote test_cart.py (2 tests) into sandbox");

    // ---- Run pytest with --json-report ----
    // [SYNTAX] cmd slice: [program, arg1, arg2, ...]. cwd is sandbox root.
    let cmd = ["python3", "-m", "pytest", "--json-report", "--json-report-file=report.json", "-q"];
    let result = sb.execute(&cmd, Path::new(""), 30).await?;

    println!("\n▌ pytest exit_code={:?}, wall_time={}ms, timed_out={}",
        result.exit_code, result.wall_time_ms, result.timed_out);
    if !result.stderr.is_empty() {
        eprintln!("--- stderr ---\n{}", result.stderr);
    }

    // ---- Parse pytest JSON report ----
    let report_path = sb.root.join("report.json");
    if report_path.exists() {
        let report_text = std::fs::read_to_string(&report_path)?;
        let report: serde_json::Value = serde_json::from_str(&report_text)?;

        let tests = report["tests"].as_array().ok_or("no 'tests' array in report")?;
        println!("\n▌ Parsed {} test results from report.json:", tests.len());
        for t in tests {
            let name = t["nodeid"].as_str().unwrap_or("?");
            let outcome = t["outcome"].as_str().unwrap_or("?");
            println!("    {:<40} → {}", name, outcome);
        }

        let summary = &report["summary"];
        println!("\n  Summary: total={}, passed={}, failed={}",
            summary["total"], summary["passed"], summary["failed"]);
    } else {
        println!("  (pytest --json-report not available — report.json not produced)");
        println!("  stdout preview:\n{}", result.stdout.lines().take(20).collect::<Vec<_>>().join("\n"));
    }

    // ---- Destroy sandbox (temp dir auto-deleted) ----
    println!("\n▌ Destroying sandbox...");
    sb.destroy();
    println!("  ✔ Sandbox destroyed (temp dir removed)");

    println!("\n✔ CHECKPOINT E COMPLETE — sandbox spawn → copy → execute → parse → destroy works.");
    Ok(())
}

// ============================================================================
// RECIPES — copy-paste templates for M2
// ============================================================================
//
// ----------------------------------------------------------------------------
// RECIPE E1 — "Spawn a subprocess sandbox and copy a repo in"
//
//   use vanta::sandbox::{subprocess::SubprocessBackend, SandboxBackend};
//   let sb = SubprocessBackend::spawn().await?;
//   sb.copy_repo_in(Path::new("examples/buggy_python"))?;
//
//   → USE IN: M2 before running any tests. Always copy into an isolated tempdir.
//
// ----------------------------------------------------------------------------
// RECIPE E2 — "Run a command with a timeout"
//
//   let result = sb.execute(&["pytest", "-q"], Path::new(""), 30).await?;
//   if result.timed_out { /* Infrastructure failure */ }
//
//   → USE IN: M2 every sandbox call. 30s per test batch; never infinite.
//
// ----------------------------------------------------------------------------
// RECIPE E3 — "Parse pytest --json-report per-test outcomes"
//
//   let report: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("report.json")?)?;
//   for t in report["tests"].as_array().unwrap() {
//       let name = t["nodeid"].as_str().unwrap();
//       let outcome = t["outcome"].as_str().unwrap();  // "passed" | "failed" | "error"
//   }
//
//   → USE IN: M2 mapping pytest outcomes → TestResult (Passed/Failed/Error).
//
// ----------------------------------------------------------------------------
// RECIPE E4 — "Destroy sandbox (RAII cleanup)"
//
//   sb.destroy();  // temp dir auto-deleted; no manual rm -rf needed
//
//   → USE IN: M2 end of every patch attempt. Never leak temp dirs.
//
// ----------------------------------------------------------------------------
// RECIPE E5 — "Check for Infrastructure vs Static vs Logic from exit code"
//
//   match (result.exit_code, result.timed_out) {
//       (_, true)        => FailureClass::Infrastructure,  // timeout
//       (Some(2), _)     => FailureClass::Static,          // pytest collection error
//       (Some(1), _)     => FailureClass::Logic,           // assertion failed
//       _                => FailureClass::Unknown,
//   }
//
//   → USE IN: M2 Diagnostician first-pass classification before deep signals.
//
// ============================================================================
