// ============================================================================
// SANDBOX MODULE — execution backends (Spec §7)
// ============================================================================
// Spec §7: "A single Rust trait exposes spawn, execute, and destroy."
// Two backends: subprocess (fast path, trusted repos) + shuru (microVM security boundary).
// ============================================================================

pub mod deps;
pub mod subprocess;
pub mod shuru;

mod types;

pub use types::{ExecutionResult, SandboxBackend, SandboxKind};

use std::error::Error;
use std::sync::OnceLock;
use std::time::Duration;

/// Cached result of the shuru boot test. We run it ONCE per process because
/// booting a VM takes ~1s and doing it on every Backend::spawn() (one per
/// test in run_tests_parallel) would make the pipeline crawl.
static SHURU_BOOT_OK: OnceLock<bool> = OnceLock::new();

/// Run the shuru boot test once and cache the result.
/// Returns true if shuru VMs boot successfully, false if shuru is
/// available but VMs don't boot (e.g. missing virtualization entitlement).
async fn shuru_boot_ok() -> bool {
    if let Some(&ok) = SHURU_BOOT_OK.get() {
        return ok;
    }
    let boot_test = tokio::process::Command::new("shuru")
        .args(["run", "--", "true"])
        .output()
        .await;
    let ok = boot_test
        .as_ref()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        let stderr = boot_test
            .ok()
            .and_then(|o| String::from_utf8(o.stderr).ok())
            .unwrap_or_default();
        tracing::warn!(
            "shuru VM boot test failed — will use subprocess for this run. stderr: {}",
            stderr.lines().next().unwrap_or("")
        );
    }
    let _ = SHURU_BOOT_OK.set(ok);
    ok
}

// ============================================================================
// network_alive() — quick connectivity check for Infrastructure classification
// ============================================================================
//
// [VANTA-WHY] M2 EXIT TEST: "Kill the network mid-run and it classifies as
// Infrastructure, not a model error." We need a way to detect that the network
// is down. This function attempts a TCP connection to a well-known resolver
// (Google DNS 8.8.8.8:53) with a 2-second timeout. If it fails, the network
// is considered dead → any test failure in that window is Infrastructure.
//
// We use a raw TCP connect (not DNS) because DNS itself needs the network and
// we want to test L3/L4 reachability, not name resolution.
// ============================================================================
pub async fn network_alive() -> bool {
    use tokio::net::TcpStream;
    match tokio::time::timeout(
        Duration::from_secs(2),
        TcpStream::connect("8.8.8.8:53"),
    )
    .await
    {
        Ok(Ok(_)) => true,
        _ => false,
    }
}

// ============================================================================
// spawn_preferred_backend() — pick shuru if available, else subprocess
// ============================================================================
//
// [VANTA-WHY] Shuru is the real security boundary; subprocess is the fast
// path for trusted repos. We prefer shuru when the CLI is installed, and
// gracefully fall back to subprocess with a warning. This keeps the pipeline
// working on machines without shuru while using the stronger isolation when
// available.
//
// [PITFALL] We can't return `dyn SandboxBackend` easily because the trait
// uses async fn (returns unnameable futures). For M2 we use an enum dispatch
// pattern via a helper that returns one of the two concrete types. Callers
// that need a single type should use `spawn_subprocess()` or `spawn_shuru()`
// directly.
// ============================================================================

/// Returns true if the shuru CLI is available in PATH.
pub fn shuru_available() -> bool {
    shuru::ShuruBackend::shuru_available()
}

/// Spawn the preferred backend: shuru if available, else subprocess.
/// Returns a tagged enum so callers can match on the concrete type.
pub enum Backend {
    Subprocess(subprocess::SubprocessBackend),
    Shuru(shuru::ShuruBackend),
}

impl Backend {
    pub async fn spawn() -> Result<Self, Box<dyn Error + Send + Sync>> {
        if shuru_available() && shuru_boot_ok().await {
            match shuru::ShuruBackend::spawn().await {
                Ok(b) => {
                    tracing::info!("using SHURU microVM backend (preferred)");
                    return Ok(Backend::Shuru(b));
                }
                Err(e) => {
                    tracing::warn!("shuru spawn failed ({e}), falling back to subprocess");
                }
            }
        }
        let b = subprocess::SubprocessBackend::spawn().await?;
        tracing::warn!("using SUBPROCESS backend (shuru not available — no security boundary)");
        Ok(Backend::Subprocess(b))
    }

    pub fn kind(&self) -> SandboxKind {
        match self {
            Backend::Subprocess(b) => b.kind(),
            Backend::Shuru(b) => b.kind(),
        }
    }

    pub fn root(&self) -> &std::path::Path {
        match self {
            Backend::Subprocess(b) => &b.root,
            Backend::Shuru(b) => &b.root,
        }
    }

    pub async fn execute(
        &self,
        cmd: &[&str],
        cwd: &std::path::Path,
        timeout_secs: u64,
    ) -> Result<ExecutionResult, Box<dyn Error + Send + Sync>> {
        match self {
            Backend::Subprocess(b) => b.execute(cmd, cwd, timeout_secs).await,
            Backend::Shuru(b) => b.execute(cmd, cwd, timeout_secs).await,
        }
    }

    pub fn destroy(self) {
        match self {
            Backend::Subprocess(b) => b.destroy(),
            Backend::Shuru(b) => b.destroy(),
        }
    }
}
