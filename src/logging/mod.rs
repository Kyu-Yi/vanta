// ============================================================================
// LOGGING MODULE — structured logging infrastructure (tracing + tracing-subscriber)
// ============================================================================
//
// WHAT YOU'LL LEARN HERE
//
//   # | Rust concept                  → Where Vanta reuses it later
//  ---+-------------------------------+---------------------------------------
//   1 | tracing::info!/debug!/trace! → Every phase in M1+ emits structured logs
//     |   (structured logging macros)        →   (Creator, Attacker, BATE, Diagnostician)
//  ---+-------------------------------+---------------------------------------
//   2 | tracing_subscriber::fmt    → Global logger installed ONCE per binary
//     |   (global subscriber)      →   at main() of vanta_m1, vanta_m2, checkpoints
//  ---+-------------------------------+---------------------------------------
//   3 | EnvFilter + std::env::var     → VANTA_LOG_LEVEL env var controls verbosity
//     |   (runtime log level)          →   without recompiling
// ============================================================================

use tracing_subscriber::{fmt, EnvFilter};

// ============================================================================
// init_logging() — install the global tracing subscriber ONCE per binary
// ============================================================================
//
// [SYNTAX] pub fn init_logging() — no args, no return. Called at the very top of
// every binary's main(). Installs a fmt subscriber that prints structured logs to
// stderr. Idempotent: calling it twice is harmless (the second call's set_global_default
// returns Err which we ignore).
//
// [VANTA-WHY] Every M1+ modules (agents, sandbox, pipeline) call tracing::info!/debug!/
// trace! macros. Without a subscriber installed, those calls are NO-OP (silently
// discarded). So every binary MUST call init_logging() first or you get zero output
// logs but no actual log lines.
//
// [PITFALL] If you forget to call init_logging(), tracing macros compile fine but
// produce NO output. This is the #1 "why are my logs missing" mistake. Fix: call
// init_logging() as the FIRST line of main().
//
// [RECIPE] Copy this into every binary main():
//   fn main() {
//       vanta::logging::init_logging();
//       // ... rest of main
//   }
pub fn init_logging() {
    // [SYNTAX] EnvFilter::try_from_default_env() reads RUST_LOG OR VANTA_LOG_LEVEL.
    // We explicitly check VANTA_LOG_LEVEL first (Vanta-specific env var),
    // then fall back to RUST_LOG (tracing default), then default to "info".
    let filter = EnvFilter::try_from_env("VANTA_LOG_LEVEL")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // [SYNTAX] fmt() builds a subscriber that writes to stderr (not stdout!).
    // Why stderr? So log lines don't pollute stdout JSON program output (e.g. session
    // log file contents) — critical when you pipe vanta_m2 | jq).
    // .with_target(true) shows which module emitted the log (vanta::agents::creator).
    // .with_thread_ids(false) — single-threaded orchestration, no need for thread IDs.
    let subscriber = fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false);

    // [SYNTAX] .try_init() returns Result. If a subscriber is already set
    // (e.g. called twice), it returns Err. We .ok() discards the error —
    // idempotent init is fine.
    let _ = subscriber.try_init();
}
