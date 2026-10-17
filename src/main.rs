// ============================================================================
// VANTA ENTRY POINT — Async binary, never sync again after this file
// ============================================================================
//
// [SYNTAX vs src/lib.rs DISTINCTION, CRITICAL FOR UNDERSTANDING]:
//
// This file (src/main.rs) compiles into the EXECUTABLE `/target/debug/vanta`.
// It is a BINARY TARGET. Binaries have exactly one main() function.
//
// src/lib.rs (added in M0 step 4) compiles into a RUST LIBRARY CRATE named
// `vanta` (a `.rlib` file in /target/debug/deps/). Libraries have no main().
//
// The relationship: ANY binary target in this project (including src/main.rs,
// src/bin/checkpoint_a.rs, src/bin/checkpoint_b.rs, and any tests/*.rs) can
// all `use vanta::core::*` and pull from the SAME shared library types. No
// duplication. This is the reason src/lib.rs exists.
//
// WHY WE CHANGED main.rs: Previously main.rs declared `mod core;` etc itself,
// which created a DUPLICATE copy of the core module attached only to the main
// binary. Checkpoint_a.rs was trying to use vanta::core::* but couldn't find
// it because there was no library target. Now: the LIBRARY (lib.rs) declares
// all modules ONCE, and EVERY binary (main + checkpoints) imports from that
// single library. This is the idiomatic Rust project structure for any code
// larger than a single-file script.
//
// [VANTA-WHY] Keeping main.rs THIN is a strict rule from the plan.
// Zero logic here. No if-statements, no loops, no parsing. Just library use
// declarations and one call to bootstrap::run(). Why? Because when we add CLI
// parsing with clap in M3, we replace the single bootstrap::run() call with
// clap's subcommand dispatch and touch exactly one function in main. All real
// logic lives in bootstrap/ or further down — never here.
// ============================================================================

// ----------------------------------------------------------------------------
// [EXAMPLE OF HOW BINARIES SHARE LIBRARY TYPES
//
// In M1 this file will import many names from the shared library:
//
//   use vanta::config::LlmConfig;
//   use vanta::core::{TestEvidence, VerificationOutcome};
//   use vanta::bootstrap;
//
// Currently main() uses full path syntax: vanta::bootstrap::run().
// Full-path avoids any chance of "unused imports" warnings when refactoring names.
// ----------------------------------------------------------------------------

// [SYNTAX] #[tokio::main] is an ATTRIBUTE MACRO. Before the compiler even
// sees your main function, this macro rewrites the source code below.
// What it expands into (conceptually):
//   fn main() {
//       let rt = tokio::runtime::Runtime::new().unwrap();
//       rt.block_on(real_main_async_shim());
//   }
// It creates a Tokio multi-threaded runtime, calls block_on with your
// async body, and translates the async Result return code into a process
// exit code for the OS.
//
// [VANTA-WHY] Vanta orchestrates MANY concurrent things in M2+:
//   - Creator LLM call
//   - Attacker LLM call
//   - N concurrent test executions in subprocesses
//   - Verifier backend calls (CrossHair, Kani)
// All of these use async APIs (reqwest for HTTP, tokio::process for
// subprocesses). If we deferred adding async until M2, we'd have to
// rewrite every fn signature and function body. Async is a one-time
// structural decision. Pay it once now and never think about it again.
//
// [PITFALL] Forgetting features = ["full"] on tokio in Cargo.toml.
// tokio is feature-gated aggressively. Without "full" or at minimum
// ["rt-multi-thread", "macros", "time"] the macro below, tokio::time::sleep,
// and tokio::process::Command all produce errors like:
//   "error[E0433]: failed to resolve: could not find `time` in `tokio`"
// Always enable "full" during development. Trim for release later if you want.
// [SYNTAX] main() now returns Result (Rust edition 2018+ allows returning Result
// from main). If the last expression in main is Ok(()) → exit code 0. If any ?
// in main encounters an Err, it propagates up and runtime prints the Err value
// (Display) to stderr AND exits with 1. No backtrace, clean error.
//
// [VANTA-WHY]: bootstrap::run() returns Result with HTTP failures, JSON parse
// failures, config errors, etc. By returning Result from main, every error in
// the entire 3-milestone chain surfaces exactly the same way as the checkpoint
// A/B/C binaries. Consistent UX = less to learn for the M1-M5 refactors.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // [SYNTAX] vanta::bootstrap::run().await?
    //
    // vanta:: = the library crate (src/lib.rs). We cannot use crate::bootstrap
    // here because src/main.rs is a BINARY target, not PART of the library.
    // The only way for the main binary to access library code is via `vanta::`.
    //
    // [SYNTAX] ? propagates Err up to the main() return value. If
    // bootstrap::run() returns Err, main() returns that Err. Runtime prints it
    // to stderr + exits 1. If Ok, we continue.
    vanta::bootstrap::run().await
}
