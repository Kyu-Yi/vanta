// ============================================================================
// DEPS MODULE — auto-detect a repo's dependencies for the shuru microVM
// ============================================================================
//
// [VANTA-WHY] The base shuru checkpoint ships only python3 + pytest + ruff +
// mypy. A repo under test (heretic, SWE-bench instances, ...) declares its own
// dependencies (torch, transformers, ...) which are NOT present. Without this
// module the user must hand-build a checkpoint per repo. Instead we read the
// repo's manifest, derive the pip install set, estimate a disk size, and let
// the shuru backend build+cache a checkpoint keyed by the dep-set fingerprint.
//
// This module is PURE (reads manifest files, does no process spawning) so it is
// unit-testable without shuru.
//
// Supported manifests:
//   - requirements.txt            (one spec per line)
//   - pyproject.toml  [project]   (PEP 621 `dependencies = [...]`)
//   - pyproject.toml  [tool.poetry.dependencies]  (table keys)
// ============================================================================

use std::collections::BTreeSet;
use std::path::Path;

/// Base checkpoint disk size in MB. Must match the base checkpoint that
/// shuru.rs creates ("vanta-python"), otherwise `shuru run --disk-size` rejects
/// the VM with "smaller than the base image".
pub const BASE_DISK_MB: u64 = 8192;

/// A repo's dependency install plan for the shuru microVM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyPlan {
    /// pip package specs to install (e.g. "torch", "transformers~=5.18").
    pub packages: Vec<String>,
    /// True if any package is a PyTorch distribution — then we add the CPU
    /// wheel index (no NVIDIA GPU inside the VM, and the CUDA wheels are ~4x
    /// larger and blow the disk budget).
    pub needs_pytorch_cpu_index: bool,
    /// Disk size to request when creating the checkpoint.
    pub disk_mb: u64,
    /// Stable hash of the (sorted, deduped) package set — used to name and
    /// cache the checkpoint so identical dep sets rebuild only once.
    pub fingerprint: String,
}

impl DependencyPlan {
    pub fn is_empty(&self) -> bool {
        self.packages.is_empty()
    }

    /// Checkpoint name derived from the dependency fingerprint.
    pub fn checkpoint_name(&self) -> String {
        format!("vanta-deps-{}", self.fingerprint)
    }

    /// Build the pip install argument string for the checkpoint build script,
    /// with each spec single-quoted so shell globbing (e.g. `lm-eval[hf]`) and
    /// metacharacters cannot corrupt the command.
    pub fn pip_args(&self) -> String {
        let mut s = String::from("--break-system-packages --root-user-action=ignore");
        if self.needs_pytorch_cpu_index {
            s.push_str(" --extra-index-url https://download.pytorch.org/whl/cpu");
        }
        for p in &self.packages {
            s.push_str(" '");
            s.push_str(&p.replace('\'', ""));
            s.push('\'');
        }
        s
    }
}

// ----------------------------------------------------------------------------
// detect — read the repo's manifests and build a plan
// ----------------------------------------------------------------------------
pub fn detect(repo_root: &Path) -> DependencyPlan {
    let mut specs: Vec<String> = Vec::new();

    if let Ok(text) = std::fs::read_to_string(repo_root.join("requirements.txt")) {
        specs.extend(parse_requirements(&text));
    }
    if let Ok(text) = std::fs::read_to_string(repo_root.join("pyproject.toml")) {
        specs.extend(parse_pyproject(&text));
    }

    // Dedupe by normalized package name (keep the first spec seen).
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut packages: Vec<String> = Vec::new();
    let mut extra_mb: u64 = 0;
    let mut needs_pytorch_cpu_index = false;

    for spec in specs {
        let name = normalize_name(&spec);
        if name.is_empty() || !seen.insert(name.clone()) {
            continue;
        }
        if is_pytorch_dist(&name) {
            needs_pytorch_cpu_index = true;
        }
        extra_mb += estimate_size_mb(&name);
        packages.push(spec);
    }

    let disk_mb = if packages.is_empty() {
        BASE_DISK_MB
    } else {
        // base + packages + 1.5 GB headroom (pip cache, build artifacts)
        round_up(BASE_DISK_MB + extra_mb + 1536, 1024)
    };

    let fingerprint = fnv1a_hex(&packages.join("\n"));

    DependencyPlan {
        packages,
        needs_pytorch_cpu_index,
        disk_mb,
        fingerprint,
    }
}

// ----------------------------------------------------------------------------
// requirements.txt parsing
// ----------------------------------------------------------------------------
fn parse_requirements(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.lines() {
        // strip inline comments
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        // skip pip flags / includes / editables (-r, -e, --index-url, ...)
        if line.starts_with('-') || line.starts_with("--") {
            continue;
        }
        out.push(line.to_string());
    }
    out
}

// ----------------------------------------------------------------------------
// pyproject.toml parsing (targeted, no TOML dependency)
// ----------------------------------------------------------------------------
fn parse_pyproject(text: &str) -> Vec<String> {
    let mut out = Vec::new();

    // PEP 621: `dependencies = [ "a", "b" ]`
    if let Some(arr) = extract_array(text, "dependencies") {
        out.extend(extract_quoted_strings(&arr));
    }
    // Poetry: `[tool.poetry.dependencies]` table of `name = "version"`
    out.extend(parse_poetry_table(text));

    out
}

/// Extract the bracketed array value of a top-level `KEY = [ ... ]` assignment.
/// Matches `dependencies = [` (and NOT `optional-dependencies`, which starts
/// with `optional-`). Returns the raw text between the brackets.
fn extract_array(text: &str, key: &str) -> Option<String> {
    let mut capturing = false;
    let mut buf = String::new();
    for line in text.lines() {
        let t = line.trim();
        if !capturing {
            let Some(rest) = t.strip_prefix(key) else {
                continue;
            };
            let Some(rest) = rest.trim_start().strip_prefix('=') else {
                continue;
            };
            let Some(rest) = rest.trim_start().strip_prefix('[') else {
                continue;
            };
            capturing = true;
            buf.push_str(rest);
            buf.push('\n');
            if rest.contains(']') {
                break;
            }
        } else {
            buf.push_str(t);
            buf.push('\n');
            if t.contains(']') {
                break;
            }
        }
    }
    if capturing { Some(buf) } else { None }
}

/// Extract all double-quoted strings from a fragment.
fn extract_quoted_strings(frag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = frag.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '"' {
            continue;
        }
        let mut s = String::new();
        for c in chars.by_ref() {
            if c == '"' {
                break;
            }
            s.push(c);
        }
        let s = s.trim().to_string();
        if !s.is_empty() {
            out.push(s);
        }
    }
    out
}

/// Parse the keys of the `[tool.poetry.dependencies]` table (ignoring the
/// `python` constraint key).
fn parse_poetry_table(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_section = false;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_section = t == "[tool.poetry.dependencies]";
            continue;
        }
        if !in_section {
            continue;
        }
        let Some((k, _)) = t.split_once('=') else {
            continue;
        };
        let k = k.trim().trim_matches('"').trim();
        if !k.is_empty() && k != "python" {
            out.push(k.to_string());
        }
    }
    out
}

// ----------------------------------------------------------------------------
// Name normalization + size estimation
// ----------------------------------------------------------------------------

/// Reduce a pip spec to a normalized distribution name:
/// `transformers[kernels]~=5.18` -> `transformers`, `py_cpuinfo` -> `py-cpuinfo`.
fn normalize_name(spec: &str) -> String {
    let base = spec.split('[').next().unwrap_or(spec);
    let base = base
        .split(|c| matches!(c, '=' | '>' | '<' | '~' | '!' | ' ' | ';' | '@'))
        .next()
        .unwrap_or(base);
    base.trim().to_ascii_lowercase().replace('_', "-")
}

fn is_pytorch_dist(name: &str) -> bool {
    matches!(name, "torch" | "pytorch" | "torchvision" | "torchaudio")
}

/// Rough installed-size estimate in MB, used only to size the VM disk.
/// Unknown packages get a conservative small default; the 1.5 GB headroom in
/// `detect` absorbs per-package error.
fn estimate_size_mb(name: &str) -> u64 {
    match name {
        "torch" | "pytorch" => 3200,
        "torchvision" | "torchaudio" => 700,
        "tensorflow" | "tensorflow-cpu" | "tensorflow-gpu" | "keras" => 4200,
        "jax" | "jaxlib" => 2200,
        "vllm" => 5000,
        "transformers" => 900,
        "sentence-transformers" => 600,
        "lm-eval" => 600,
        "bitsandbytes" => 500,
        "datasets" => 500,
        "numpy" | "scipy" | "pandas" => 350,
        "accelerate" => 250,
        "peft" | "optuna" => 200,
        _ => 140,
    }
}

fn round_up(v: u64, step: u64) -> u64 {
    v.div_ceil(step) * step
}

/// FNV-1a 64-bit hash, hex-encoded. Deterministic across runs and platforms
/// (unlike DefaultHasher), so checkpoint names are stable.
fn fnv1a_hex(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

// ============================================================================
// SELF-HEALING — resolve imports pytest reported as missing
// ============================================================================
//
// [VANTA-WHY] The manifest tells you what the AUTHOR intended, which is not
// always what the TEST actually imports (transitive, test-only, or simply
// undeclared deps). Rather than guess, we run the tests, read pytest's
// "No module named 'X'" errors, convert X to a PyPI package, install it into a
// checkpoint DERIVED from the current one, and re-run. This converges in one
// or two rounds and needs no maintainable list of "known bad" repos.
// ============================================================================

/// Extract the unique top-level module names from pytest output that reports
/// `ModuleNotFoundError: No module named 'X'` (also matches the attribute form
/// without the "Error" suffix, i.e. `No module named 'X'`).
pub fn parse_missing_modules(output: &str) -> Vec<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let needle = "No module named ";
    for line in output.lines() {
        let Some(idx) = line.find(needle) else {
            continue;
        };
        let rest = &line[idx + needle.len()..];
        let Some(open) = rest.chars().next() else {
            continue;
        };
        if open != '\'' && open != '"' {
            continue;
        }
        let rest = &rest[1..]; // safe: `open` is 1 ASCII byte
        let Some(end) = rest.find(open) else {
            continue;
        };
        // Take the ROOT package: "heretic.foo" -> "heretic".
        let root = rest[..end].split('.').next().unwrap_or("").trim();
        if !root.is_empty() {
            seen.insert(root.to_string());
        }
    }
    seen.into_iter().collect()
}

/// Map a Python import name to its PyPI distribution name. Most packages share
/// the name; this covers the common mismatches. Unknown names pass through
/// unchanged (correct for the large majority: numpy, pandas, requests, ...).
pub fn module_to_package(module: &str) -> String {
    let root = module.split('.').next().unwrap_or(module);
    match root {
        "cpuinfo" => "py-cpuinfo",
        "cv2" => "opencv-python",
        "PIL" | "Image" => "pillow",
        "yaml" => "pyyaml",
        "sklearn" => "scikit-learn",
        "bs4" => "beautifulsoup4",
        "dateutil" => "python-dateutil",
        "dotenv" => "python-dotenv",
        "attr" => "attrs",
        "serial" => "pyserial",
        "jwt" => "pyjwt",
        "docx" => "python-docx",
        "pptx" => "python-pptx",
        "fitz" => "pymupdf",
        "pkg_resources" => "setuptools",
        "pytest_jsonreport" | "pytest_json_report" => "pytest-json-report",
        "OpenSSL" => "pyopenssl",
        "lxml" => "lxml",
        other => other,
    }
    .to_string()
}

/// Build a plan for packages to ADD on top of an existing checkpoint. The
/// fingerprint is namespaced with "extra" so it can never collide with a
/// manifest-derived checkpoint name.
pub fn derived_plan(parent_disk_mb: u64, missing_modules: &[String]) -> DependencyPlan {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut packages: Vec<String> = Vec::new();
    let mut extra_mb: u64 = 0;
    let mut needs_pytorch_cpu_index = false;

    for module in missing_modules {
        let pkg = module_to_package(module);
        let name = normalize_name(&pkg);
        if name.is_empty() || !seen.insert(name.clone()) {
            continue;
        }
        if is_pytorch_dist(&name) {
            needs_pytorch_cpu_index = true;
        }
        extra_mb += estimate_size_mb(&name);
        packages.push(pkg);
    }

    // Grow the parent disk: the parent was already sized for its own deps, so
    // we only need room for the additions plus a little slack.
    let disk_mb = round_up(parent_disk_mb + extra_mb + 1024, 1024);

    let fingerprint = fnv1a_hex(&format!("extra\n{}", packages.join("\n")));

    DependencyPlan {
        packages,
        needs_pytorch_cpu_index,
        disk_mb,
        fingerprint,
    }
}


// ============================================================================
// UNIT TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_fnv_stable() {
        // Same input -> same hash; different input -> different hash.
        assert_eq!(fnv1a_hex("abc"), fnv1a_hex("abc"));
        assert_ne!(fnv1a_hex("abc"), fnv1a_hex("abd"));
    }

    #[test]
    fn test_parse_pep621_dependencies() {
        let toml = r#"
[project]
name = "heretic-llm"
dependencies = [
    "accelerate~=1.15",
    "torch",  # version deliberately unspecified
    "lm-eval[hf]~=0.4",
]

[project.optional-dependencies]
dev = ["pytest"]
"#;
        let deps = parse_pyproject(toml);
        assert!(deps.contains(&"accelerate~=1.15".to_string()));
        assert!(deps.contains(&"torch".to_string()));
        assert!(deps.contains(&"lm-eval[hf]~=0.4".to_string()));
        // optional-dependencies must NOT be picked up as the main array
        assert!(!deps.contains(&"pytest".to_string()));
    }

    #[test]
    fn test_parse_requirements_skips_flags_and_comments() {
        let txt = "# a comment\nnumpy~=2.2\n\n-r other.txt\n--index-url http://x\npandas  # inline\n";
        let reqs = parse_requirements(txt);
        assert_eq!(reqs, vec!["numpy~=2.2", "pandas"]);
    }

    #[test]
    fn test_parse_poetry_table() {
        let toml = r#"
[tool.poetry.dependencies]
python = "^3.11"
torch = ">=2.0"
numpy = { version = "^2.0" }

[tool.poetry.group.dev.dependencies]
pytest = "*"
"#;
        let deps = parse_poetry_table(toml);
        assert!(deps.contains(&"torch".to_string()));
        assert!(deps.contains(&"numpy".to_string()));
        assert!(!deps.contains(&"python".to_string()));
        assert!(!deps.contains(&"pytest".to_string()));
    }

    #[test]
    fn test_normalize_name() {
        assert_eq!(normalize_name("transformers[kernels]~=5.18"), "transformers");
        assert_eq!(normalize_name("py_cpuinfo~=9.0"), "py-cpuinfo");
        assert_eq!(normalize_name("Python"), "python");
    }

    #[test]
    fn test_detect_sizes_and_flags_pytorch() {
        let tmp = TempDir::new().unwrap();
        fs::write(
            tmp.path().join("pyproject.toml"),
            "[project]\nname='x'\ndependencies = [\n  \"torch\",\n  \"transformers\",\n  \"numpy\",\n]\n",
        )
        .unwrap();

        let plan = detect(tmp.path());
        assert_eq!(plan.packages.len(), 3);
        assert!(plan.needs_pytorch_cpu_index);
        // base 8192 + (torch 3200 + transformers 900 + numpy 350) + 1536 headroom
        assert!(plan.disk_mb > BASE_DISK_MB);
        assert_eq!(plan.disk_mb % 1024, 0);
        assert!(!plan.checkpoint_name().is_empty());
        // pip args quote each spec and add the CPU index
        assert!(plan.pip_args().contains("--extra-index-url https://download.pytorch.org/whl/cpu"));
        assert!(plan.pip_args().contains("'torch'"));
    }

    #[test]
    fn test_detect_empty_repo() {
        let tmp = TempDir::new().unwrap();
        let plan = detect(tmp.path());
        assert!(plan.is_empty());
        assert_eq!(plan.disk_mb, BASE_DISK_MB);
        assert!(!plan.needs_pytorch_cpu_index);
    }

    #[test]
    fn test_detect_dedupes_by_name() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("requirements.txt"), "numpy~=2.2\nnumpy>=1.0\n").unwrap();
        let plan = detect(tmp.path());
        assert_eq!(plan.packages, vec!["numpy~=2.2"]);
    }

    #[test]
    fn test_parse_missing_modules() {
        let out = "E   ModuleNotFoundError: No module named 'hypothesis'\n\
                   E   ModuleNotFoundError: No module named 'heretic.sub'\n\
                   also No module named \"yaml\"\n\
                   no match here\n";
        let mods = parse_missing_modules(out);
        // roots only, deduped, sorted
        assert_eq!(mods, vec!["heretic", "hypothesis", "yaml"]);
    }

    #[test]
    fn test_parse_missing_modules_ignores_unquoted() {
        assert!(parse_missing_modules("No module named foo").is_empty());
    }

    #[test]
    fn test_module_to_package_mapping() {
        assert_eq!(module_to_package("cv2"), "opencv-python");
        assert_eq!(module_to_package("PIL.Image"), "pillow");
        assert_eq!(module_to_package("sklearn.linear_model"), "scikit-learn");
        // unknown names pass through unchanged
        assert_eq!(module_to_package("requests"), "requests");
    }

    #[test]
    fn test_derived_plan_namespaced_and_grows_disk() {
        let plan = derived_plan(8192, &["hypothesis".to_string(), "cv2".to_string()]);
        assert_eq!(plan.packages, vec!["hypothesis", "opencv-python"]);
        assert!(plan.disk_mb > 8192);
        assert_eq!(plan.disk_mb % 1024, 0);
        // must never collide with a manifest fingerprint for the same packages
        let manifest = fnv1a_hex(&plan.packages.join("\n"));
        assert_ne!(plan.fingerprint, manifest);
    }
}
