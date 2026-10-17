// ============================================================================
// SANDBOX TYPES — SandboxBackend trait + ExecutionResult + SandboxKind
// ============================================================================
//
// Spec §7: "A single Rust trait exposes spawn, execute, and destroy. The
// orchestrator never knows which backend it is on."
//
// [VANTA-WHY] The subprocess-is-NOT-sandbox distinction (Spec §7):
//   - subprocess.rs = controlled executor for TRUSTED repos only (fast path)
//   - shuru.rs = real security boundary via Apple Virtualization microVM
// We NEVER call subprocess a "sandbox" in docs/logs. It's a "subprocess backend".
// ============================================================================

use serde::{Deserialize, Serialize};
use std::path::Path;

// ============================================================================
// ExecutionResult — what every sandbox execute() returns
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub wall_time_ms: u128,
    pub timed_out: bool,
}

// ============================================================================
// SandboxKind — which backend is in use
// ============================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxKind {
    Subprocess,
    Shuru,
}

// ============================================================================
// SandboxBackend trait — the single interface both backends implement
// ============================================================================
//
// [SYNTAX] trait with async methods requires the async_trait crate OR boxing.
// For M2 we use `async fn` directly in the trait (Rust 1.75+ native async fn in traits).
// Edition 2024 supports this.
//
// [VANTA-WHY] Three methods: spawn (create isolated env), execute (run command),
// destroy (cleanup). This is Spec §7's contract. The orchestrator calls these
// without knowing if it's subprocess or shuru — same API, different isolation.
// [VANTA-WHY] async_fn_in_trait: Rust warns that async fns in traits can't
// guarantee Send bounds on the returned Future. For Vanta's internal use this
// is acceptable; we #[allow] it rather than boxing every future.
#[allow(async_fn_in_trait)]
pub trait SandboxBackend: Send {
    fn kind(&self) -> SandboxKind;

    /// Spawn a fresh sandbox. For subprocess: create temp dir + copy repo.
    /// For shuru: boot microVM + mount workspace.
    async fn spawn() -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    where
        Self: Sized;

    /// Execute a command inside the sandbox. cwd is relative to sandbox root.
    async fn execute(
        &self,
        cmd: &[&str],
        cwd: &Path,
        timeout_secs: u64,
    ) -> Result<ExecutionResult, Box<dyn std::error::Error + Send + Sync>>;

    /// Destroy the sandbox. For subprocess: delete temp dir. For shuru: stop VM.
    fn destroy(self);
}
