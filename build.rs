use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", manifest_dir.join(".git/HEAD").display());
    println!("cargo:rerun-if-changed={}", manifest_dir.join(".git/index").display());

    let commit = git_stdout(&manifest_dir, &["rev-parse", "--short", "HEAD"])
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = git_stdout(&manifest_dir, &["status", "--porcelain"])
        .map(|s| if s.trim().is_empty() { "false" } else { "true" })
        .unwrap_or("true");

    println!("cargo:rustc-env=VANTA_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=VANTA_BUILD_DIRTY={dirty}");
}

fn git_stdout(repo_root: &PathBuf, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
