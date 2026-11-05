// ============================================================================
// VANTA LIBRARY ROOT (src/lib.rs) — why this file exists
// ============================================================================
//
// [SYNTAX] Before this file existed, Vanta was a BINARY-ONLY crate:
//   src/main.rs compiled to one executable called `vanta`.
//   src/bin/checkpoint_a.rs compiled to a SEPARATE executable.
//   These two are COMPLETELY SEPARATE compilation units. The checkpoint
//   binary cannot access types declared in main.rs's `mod core;` because
//   Cargo builds them as independent crates. There is no crate called `vanta`
//   that checkpoint_a can `use vanta::core::*` from.
//
// [SOLUTION] Adding src/lib.rs turns Vanta into a MIXED lib+bin crate:
//   * src/lib.rs → compiles to a LIBRARY crate named `vanta`
//     → all `pub mod` here are reusable from ANY binary in the same crate,
//       from integration tests in tests/, from examples/, from benchmarks/
//   * src/main.rs → compiles to the main BINARY, which uses `use vanta::core::*`
//     just like checkpoint_a.rs does (or can declare its own mods too)
//   * src/bin/*.rs → each compiles to a separate binary, ALL able to share
//     the library's pub types by writing `use vanta::core::*;`
//
// [VANTA-WHY] This is the ONE structural change you make at most once per Rust
// project. Without a lib target you cannot share code across binaries, across
// test harnesses, or across example programs. With it, every checkpoint
// binary (A/B/C), every integration test, and the main executable all use
// the same FailureClass/TestIntent/VerificationOutcome types — guaranteeing
// they're all talking about the same thing (no copy-paste type drift between
// checkpoint_a.rs and M2's real Diagnostician code later).
//
// [PITFALL] Beginners put everything in src/main.rs. Works for small programs.
// Breaks catastrophically once you have src/bin/*.rs, tests/, examples/, or
// multiple crates in a workspace. Add src/lib.rs on DAY ONE for any project
// with ambitions larger than a single file. This is the standard convention.
//
// Rule of thumb:
//   DOMAIN TYPES → put them in src/lib.rs's public modules (core, config, ...)
//   ORCHESTRATION (main, CLI parsing) → put in src/main.rs
// ============================================================================

// [SYNTAX] pub mod bootstrap; = "declare the bootstrap submodule as public;
// load its contents from bootstrap/mod.rs next to this file".
//
// [VANTA-WHY] All three core modules are re-exported from the library root so
// that `use vanta::core::*;` works in every binary and test. Without `pub`
// here the modules would exist but nobody outside the lib root could access
// them (they'd have default private visibility).
pub mod bootstrap;

// [VANTA-WHY] config::LlmConfig is needed by bootstrap for the spike AND by
// every future binary that needs to talk to an LLM. Publishing it here makes
// it reusable.
pub mod config;

// [VANTA-WHY] The ABSOLUTE most important module. Every piece of Vanta uses
// the domain types from core::. Without this as pub, checkpoint binaries
// cannot import the enums they exercise. Without this as lib-level pub,
// integration tests (tests/*.rs in M1) could not test the types either.
pub mod core;

// [VANTA-WHY] M1+ modules: agents (Creator/Attacker), llm (shared HTTP+cleaning
// helpers), logging (tracing subscriber), pipeline (BATE + Diagnostician +
// patch + Governor + Gatekeeper), sandbox (subprocess + Shuru backends),
// session (provenance IDs + disk log), repo (load arbitrary repos for M3).
pub mod agents;
// [VANTA-WHY] context localizes the source a role is shown: the deterministic
// walk, the prompt budget, and the ranking. It is the ONE implementation of
// both, shared by the run and by the harness's context gate (which shells out to
// `vanta_m3 --select-context --json`), so the gate measures the prompt the run
// actually sends.
pub mod context;
pub mod llm;
pub mod logging;
pub mod pipeline;
pub mod pr;
pub mod sandbox;
pub mod session;
pub mod terminal;
pub mod repo;

// [VANTA-WHY] M3 item X4 — the ONE lock for every test in this crate that writes
// the process environment. `std::env` is process-global and `cargo test` runs the
// lib's tests on parallel threads by default, so a test that sets a variable (or
// takes one away) can be observed mid-test by any other test that reads the
// environment — including the harness tests below that spawn real pytest
// subprocesses, which inherit it. Each site used to guard only its own module
// (`pr`'s `TEST_ENV_LOCK`, `sandbox`'s `ENV_LOCK`) and `harness`'s
// `ExplicitHostBackend` had no lock at all and relied on `--test-threads=1`; a
// per-module lock cannot see a sibling module's mutation, so the whole suite had
// to stay serial. One shared lock makes every such test exclusive with every
// other one, which is what lets the flag go — the same shape as C6's
// `SEED_GLOBAL_LOCK` for the process-wide seed globals.
//
// It lives at the crate root because the sites are spread over `pr`, `sandbox`,
// `config`, `pipeline::driver` and `pipeline::harness`, and none of those may
// depend on another just to borrow a test-only lock.
#[cfg(test)]
pub(crate) mod test_env {
    use std::sync::{Mutex, MutexGuard};

    /// Held for the whole of a test that writes the environment. Poisoning is
    /// ignored on purpose: the guard is a serialisation token, not data, so a
    /// panic in one test must not make the lock unusable for the next.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Take the process-environment lock for the rest of the caller's scope.
    pub(crate) fn lock() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
