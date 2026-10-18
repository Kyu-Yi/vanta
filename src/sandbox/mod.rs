// ============================================================================
// SANDBOX MODULE — execution backends (Spec §7)
// ============================================================================
// Spec §7: "A single Rust trait exposes spawn, execute, and destroy."
// Two backends: subprocess (fast path, trusted repos) + shuru (microVM security boundary).
// ============================================================================

pub mod deps;
pub mod shuru;
pub mod subprocess;

mod types;

pub use types::{ExecutionResult, SandboxBackend, SandboxKind};

use std::error::Error;
use std::sync::{Once, OnceLock};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendPreference {
    Auto,
    Shuru,
    Subprocess,
}

fn backend_preference() -> Result<BackendPreference, Box<dyn Error + Send + Sync>> {
    match std::env::var("VANTA_SANDBOX_BACKEND")
        .ok()
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        None | Some("auto") => Ok(BackendPreference::Auto),
        Some("shuru") => Ok(BackendPreference::Shuru),
        Some("subprocess") => Ok(BackendPreference::Subprocess),
        Some(other) => Err(format!(
            "invalid VANTA_SANDBOX_BACKEND={other:?}; expected auto, shuru, or subprocess"
        )
        .into()),
    }
}

pub fn requested_backend_label() -> Result<&'static str, Box<dyn Error + Send + Sync>> {
    Ok(match backend_preference()? {
        BackendPreference::Auto => "auto",
        BackendPreference::Shuru => "shuru",
        BackendPreference::Subprocess => "subprocess",
    })
}

pub fn backend_kind_label(kind: SandboxKind) -> &'static str {
    match kind {
        SandboxKind::Subprocess => "subprocess",
        SandboxKind::Shuru => "shuru",
    }
}

pub async fn actual_backend_label() -> Result<&'static str, Box<dyn Error + Send + Sync>> {
    let backend = Backend::spawn().await?;
    let label = backend_kind_label(backend.kind());
    backend.destroy();
    Ok(label)
}

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
        let first_lines = stderr.lines().take(3).collect::<Vec<_>>().join(" | ");
        tracing::warn!(
            "shuru VM boot test failed — will use subprocess for this run. stderr: {}",
            first_lines
        );
        eprintln!("⚠ shuru VM boot test failed: {first_lines}");
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
    if std::env::var("VANTA_TEST_FORCE_NETWORK_DOWN")
        .ok()
        .as_deref()
        == Some("1")
    {
        return false;
    }
    use tokio::net::TcpStream;
    match tokio::time::timeout(Duration::from_secs(2), TcpStream::connect("8.8.8.8:53")).await {
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

/// Guards the one-per-run announcement of which backend actually ran.
///
/// [VANTA-WHY] `Backend::spawn()` is called on every harness phase — dozens of
/// times in a single run (61 times in one observed Shuru run) — so announcing
/// the decision at each spawn floods the log with identical lines. The decision
/// cannot change within a process (the env var is fixed and the VM boot test is
/// memoized), so it is announced exactly once and later spawns stay silent.
static ANNOUNCE_BACKEND: Once = Once::new();

/// Announce the backend decision once per process (stdout).
fn announce_backend(line: &str) {
    ANNOUNCE_BACKEND.call_once(|| println!("{line}"));
}

/// Announce a subprocess fallback once per process (stderr — it is a warning).
fn announce_fallback(line: &str) {
    ANNOUNCE_BACKEND.call_once(|| eprintln!("{line}"));
}

impl Backend {
    pub async fn spawn() -> Result<Self, Box<dyn Error + Send + Sync>> {
        match backend_preference()? {
            BackendPreference::Subprocess => {
                let b = subprocess::SubprocessBackend::spawn().await?;
                tracing::warn!("using SUBPROCESS backend (forced by VANTA_SANDBOX_BACKEND)");
                announce_backend("▌ Backend: subprocess (forced, NO security boundary)");
                Ok(Backend::Subprocess(b))
            }
            BackendPreference::Shuru => {
                if !shuru_available() {
                    return Err(
                        "VANTA_SANDBOX_BACKEND=shuru but the shuru CLI is not available".into(),
                    );
                }
                if !shuru_boot_ok().await {
                    return Err(
                        "VANTA_SANDBOX_BACKEND=shuru but the shuru VM boot test failed".into(),
                    );
                }
                let b = shuru::ShuruBackend::spawn().await?;
                tracing::info!("using SHURU microVM backend (forced by VANTA_SANDBOX_BACKEND)");
                announce_backend("▌ Backend: SHURU microVM active (forced)");
                Ok(Backend::Shuru(b))
            }
            BackendPreference::Auto => {
                // [VANTA-WHY] `auto` degrades to subprocess, and a SILENT degrade
                // is indistinguishable from a real shuru run — which is exactly
                // how a missing CLI or a failed VM boot gets misread as "shuru
                // broke". Record the precise reason and print one greppable line
                // so a run can be audited for which boundary actually ran.
                let fallback_reason: String = if !shuru_available() {
                    "shuru CLI not found in PATH".to_string()
                } else if !shuru_boot_ok().await {
                    "shuru VM boot test failed (see warning above for stderr)".to_string()
                } else {
                    match shuru::ShuruBackend::spawn().await {
                        Ok(b) => {
                            tracing::info!("using SHURU microVM backend (preferred)");
                            announce_backend("▌ Backend: SHURU microVM active (auto-selected)");
                            return Ok(Backend::Shuru(b));
                        }
                        Err(e) => format!("shuru spawn failed: {e}"),
                    }
                };
                let b = subprocess::SubprocessBackend::spawn().await?;
                tracing::warn!(
                    reason = %fallback_reason,
                    "using SUBPROCESS backend — FALLBACK (no security boundary)"
                );
                announce_fallback(&format!(
                    "⚠ FALLBACK ACTIVE: subprocess backend (NO security boundary) — reason: {fallback_reason}"
                ));
                Ok(Backend::Subprocess(b))
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn backend_preference_defaults_to_auto() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            std::env::remove_var("VANTA_SANDBOX_BACKEND");
        }
        assert_eq!(backend_preference().unwrap(), BackendPreference::Auto);
    }

    #[test]
    fn backend_preference_accepts_forced_values() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            std::env::set_var("VANTA_SANDBOX_BACKEND", "shuru");
        }
        assert_eq!(backend_preference().unwrap(), BackendPreference::Shuru);
        unsafe {
            std::env::set_var("VANTA_SANDBOX_BACKEND", "subprocess");
        }
        assert_eq!(backend_preference().unwrap(), BackendPreference::Subprocess);
        unsafe {
            std::env::remove_var("VANTA_SANDBOX_BACKEND");
        }
    }

    #[test]
    fn backend_preference_rejects_unknown_value() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe {
            std::env::set_var("VANTA_SANDBOX_BACKEND", "mystery");
        }
        let err = backend_preference().unwrap_err().to_string();
        assert!(err.contains("invalid VANTA_SANDBOX_BACKEND"));
        unsafe {
            std::env::remove_var("VANTA_SANDBOX_BACKEND");
        }
    }
}
