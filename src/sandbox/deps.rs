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
//   - pyproject.toml  [project.optional-dependencies] (recorded, NOT installed)
//   - pyproject.toml  [tool.poetry.dependencies]  (table keys)
// ============================================================================

use crate::sandbox::SandboxKind;
use std::collections::BTreeSet;
use std::path::Path;

/// Base checkpoint disk size in MB. Must match the base checkpoint that
/// shuru.rs creates ("vanta-python"), otherwise `shuru run --disk-size` rejects
/// the VM with "smaller than the base image".
pub const BASE_DISK_MB: u64 = 8192;
/// Hard cap for auto-built dependency checkpoints. Above this we refuse to
/// build and let the run surface an Infrastructure result instead of filling
/// the disk with a giant environment.
pub const MAX_CHECKPOINT_DISK_MB: u64 = 16_384;

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

/// Whether an auto-dependency checkpoint should be built for a run whose
/// selected backend is `kind`.
///
/// [VANTA-WHY] The checkpoint exists to put a repo's declared dependencies
/// INSIDE the shuru VM, whose base image ships only python3 + pytest + ruff +
/// mypy. A subprocess run executes against the HOST environment, where those
/// dependencies are already installed (or deliberately absent), so building a
/// VM checkpoint for it downloads hundreds of MB the run never uses — and can
/// fill the VM disk, as heretic's torch + CUDA stack did. Gate on the backend
/// the run will actually use, not on whether the shuru CLI merely exists.
pub fn auto_build_enabled(kind: SandboxKind) -> bool {
    kind == SandboxKind::Shuru
}

pub fn policy_refusal_reason(plan: &DependencyPlan) -> Option<String> {
    let blocked: Vec<String> = plan
        .packages
        .iter()
        .filter(|pkg| is_blocked_package(pkg))
        .cloned()
        .collect();
    if !blocked.is_empty() {
        return Some(format!(
            "auto-dependency checkpoint refused huge package(s): {}. \
             Mark this run Infrastructure or pass an explicit benchmark environment instead",
            blocked.join(", ")
        ));
    }
    if plan.disk_mb > MAX_CHECKPOINT_DISK_MB {
        return Some(format!(
            "auto-dependency checkpoint refused disk size {} MB above cap {} MB. \
             Mark this run Infrastructure or pass an explicit benchmark environment instead",
            plan.disk_mb, MAX_CHECKPOINT_DISK_MB
        ));
    }
    None
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
        let pyproject = parse_pyproject(&text);
        // Extras (`pyproject.optional`) are recorded but deliberately NOT
        // installed: they are opt-in groups that pull in tooling and, often,
        // heavyweight wheels this run does not need.
        specs.extend(pyproject.required);
        specs.extend(pyproject.poetry);
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
// pyproject.toml parsing (via the `toml` crate)
// ----------------------------------------------------------------------------
//
// [VANTA-WHY] This used to be hand-rolled line scanning, and it failed exactly
// the way hand-rolled TOML parsing does: the array scan stopped at the first
// line containing `]`, so `"lm-eval[hf]~=0.4"` — whose EXTRAS bracket holds a
// `]` — ended the list. Every dependency after it was silently dropped,
// including heretic's explicitly declared `torch`, so the huge-package guard
// never saw torch and the auto-install downloaded torch + CUDA until the VM
// disk filled. The `toml` crate handles extras, inline comments, environment
// markers and multiline arrays correctly, so this is now a few lookups.
// ----------------------------------------------------------------------------

/// What a repo's `pyproject.toml` declares.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PyprojectDeps {
    /// PEP 621 `[project].dependencies` — installed into the checkpoint.
    pub required: Vec<String>,
    /// PEP 621 `[project.optional-dependencies]` extras — RECORDED, never
    /// installed. Extras are opt-in groups (`dev`, `gpu`, ...) that routinely
    /// pull in heavyweight wheels the run does not need.
    pub optional: Vec<String>,
    /// Poetry `[tool.poetry.dependencies]` table keys — installed.
    pub poetry: Vec<String>,
}

fn parse_pyproject(text: &str) -> PyprojectDeps {
    let mut out = PyprojectDeps::default();
    let Ok(doc) = toml::from_str::<toml::Value>(text) else {
        return out;
    };
    let Some(root) = doc.as_table() else {
        return out;
    };

    if let Some(project) = root.get("project").and_then(toml::Value::as_table) {
        if let Some(deps) = project.get("dependencies").and_then(toml::Value::as_array) {
            out.required = string_array(deps);
        }
        if let Some(extras) = project
            .get("optional-dependencies")
            .and_then(toml::Value::as_table)
        {
            for list in extras.values() {
                if let Some(list) = list.as_array() {
                    out.optional.extend(string_array(list));
                }
            }
        }
    }

    if let Some(poetry) = root
        .get("tool")
        .and_then(toml::Value::as_table)
        .and_then(|tool| tool.get("poetry"))
        .and_then(toml::Value::as_table)
        .and_then(|poetry| poetry.get("dependencies"))
        .and_then(toml::Value::as_table)
    {
        for key in poetry.keys() {
            // `python` is the interpreter constraint, not a package.
            if key.as_str() != "python" {
                out.poetry.push(key.clone());
            }
        }
    }

    out
}

/// Collect the string values of a TOML array.
fn string_array(values: &[toml::Value]) -> Vec<String> {
    values
        .iter()
        .filter_map(toml::Value::as_str)
        .map(str::to_string)
        .collect()
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

fn is_blocked_package(spec: &str) -> bool {
    let normalized = normalize_name(spec);
    matches!(
        normalized.as_str(),
        "torch"
            | "pytorch"
            | "torchvision"
            | "torchaudio"
            | "tensorflow"
            | "tensorflow-cpu"
            | "tensorflow-gpu"
            | "keras"
            | "jax"
            | "jaxlib"
            | "vllm"
    ) || spec.to_ascii_lowercase().contains("cuda")
        || spec.to_ascii_lowercase().contains("+cu")
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

    /// heretic's real `[project].dependencies`, verbatim. The extras entry
    /// (`lm-eval[hf]`) sits above `torch`/`torchvision`, which is what exposed
    /// the truncation bug.
    const HERETIC_PYPROJECT: &str = r#"
[project]
name = "heretic-llm"
version = "2.0.0.dev0"
dependencies = [
    "accelerate~=1.15",
    "bitsandbytes~=0.50",
    "datasets~=5.0",
    "huggingface-hub~=1.33",
    "immutabledict~=4.3",
    "langdetect~=1.0",
    "lm-eval[hf]~=0.4",
    "numpy~=2.2",
    "optuna~=5.0",
    "peft~=0.21",
    "psutil~=7.2",
    "py-cpuinfo~=9.0",
    "pydantic-settings~=2.15",
    "questionary~=2.1",
    "rich~=15.0",
    "tomli-w~=1.2",
    "torch",  # version deliberately unspecified
    "torchvision",  # version deliberately unspecified
    "tqdm~=4.70",
    "transformers[kernels]~=5.18",
]

[dependency-groups]
dev = ["ruff>=0.16.9"]
"#;

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
        assert_eq!(
            deps.required,
            vec!["accelerate~=1.15", "torch", "lm-eval[hf]~=0.4"]
        );
        // extras are recorded separately, never merged into the install set
        assert_eq!(deps.optional, vec!["pytest"]);
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
        let deps = parse_pyproject(toml);
        assert!(deps.poetry.contains(&"torch".to_string()));
        assert!(deps.poetry.contains(&"numpy".to_string()));
        assert!(!deps.poetry.contains(&"python".to_string()));
        assert!(!deps.poetry.contains(&"pytest".to_string()));
    }

    #[test]
    fn test_heretic_extras_do_not_truncate_the_dependency_array() {
        // [REGRESSION] The hand-written scanner stopped at the first line
        // containing `]`. `"lm-eval[hf]~=0.4"` carries a `]` inside its extras
        // bracket, so every dependency below it — including the explicitly
        // declared `torch` and `torchvision` — was silently dropped, and the
        // huge-package guard never saw torch.
        let deps = parse_pyproject(HERETIC_PYPROJECT);
        assert_eq!(
            deps.required.len(),
            20,
            "all 20 declared dependencies must survive parsing: {:?}",
            deps.required
        );
        assert!(deps.required.contains(&"lm-eval[hf]~=0.4".to_string()));
        assert!(
            deps
                .required
                .contains(&"transformers[kernels]~=5.18".to_string())
        );
        assert!(deps.required.contains(&"torch".to_string()));
        assert!(deps.required.contains(&"torchvision".to_string()));
    }

    #[test]
    fn test_heretic_declared_torch_is_refused_up_front() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("pyproject.toml"), HERETIC_PYPROJECT).unwrap();

        let plan = detect(tmp.path());
        assert_eq!(plan.packages.len(), 20, "plan: {:?}", plan.packages);
        let reason = policy_refusal_reason(&plan)
            .expect("heretic declares torch, so the checkpoint must be refused up front");
        assert!(
            reason.contains("torch"),
            "refusal must name the huge package, got {reason}"
        );
    }

    #[test]
    fn test_parse_pyproject_closing_bracket_comment_and_single_line_array() {
        // A `]` followed by a comment ends the array — it must not be swallowed
        // and must not make the parser read into the next key.
        let commented =
            "[project]\ndependencies = [\n  \"torch\",\n]  # deps\nname = \"x\"\n";
        assert_eq!(parse_pyproject(commented).required, vec!["torch"]);

        // Single-line array.
        let single_line = "[project]\ndependencies = [\"numpy\", \"torch\"]\n";
        assert_eq!(parse_pyproject(single_line).required, vec!["numpy", "torch"]);
    }

    #[test]
    fn test_parse_pyproject_environment_marker_normalizes_for_the_guard() {
        let toml = "[project]\ndependencies = [\"torch ; platform_system == 'Linux'\", \"lm-eval[hf]~=0.4\"]\n";
        let deps = parse_pyproject(toml);
        assert!(
            is_blocked_package(&deps.required[0]),
            "an environment marker must not hide torch from the guard"
        );
        assert_eq!(normalize_name(&deps.required[1]), "lm-eval");
    }

    #[test]
    fn test_optional_dependencies_are_recorded_but_not_installed() {
        let toml = r#"
[project]
dependencies = ["numpy"]
[project.optional-dependencies]
dev = ["pytest"]
gpu = ["torch"]
"#;
        let deps = parse_pyproject(toml);
        assert_eq!(deps.required, vec!["numpy"]);
        assert!(deps.optional.contains(&"pytest".to_string()));
        assert!(deps.optional.contains(&"torch".to_string()));

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("pyproject.toml"), toml).unwrap();
        let plan = detect(tmp.path());
        assert_eq!(plan.packages, vec!["numpy"], "extras must not be installed");
        assert!(
            policy_refusal_reason(&plan).is_none(),
            "a torch extra alone must not refuse the run"
        );
    }

    #[test]
    fn test_auto_build_only_for_shuru_backend() {
        // [REGRESSION] Auto-deps keyed off `shuru_available()`, so a subprocess
        // run still built a VM checkpoint — heretic's torch + CUDA install
        // filled the VM disk for a run that never used the VM.
        assert!(!auto_build_enabled(SandboxKind::Subprocess));
        assert!(auto_build_enabled(SandboxKind::Shuru));
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

    #[test]
    fn test_policy_rejects_known_huge_packages() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("requirements.txt"), "torch\npytest\n").unwrap();

        let plan = detect(tmp.path());
        let reason = policy_refusal_reason(&plan).expect("torch should be refused");
        assert!(
            reason.contains("torch"),
            "refusal should name the huge package, got {reason}"
        );
    }

    #[test]
    fn test_policy_rejects_checkpoints_over_disk_cap() {
        let plan = DependencyPlan {
            packages: vec!["small-a".into(), "small-b".into()],
            needs_pytorch_cpu_index: false,
            disk_mb: MAX_CHECKPOINT_DISK_MB + 1024,
            fingerprint: "abc".into(),
        };

        let reason =
            policy_refusal_reason(&plan).expect("oversized checkpoints should be refused");
        assert!(
            reason.contains(&MAX_CHECKPOINT_DISK_MB.to_string()),
            "refusal should mention the disk cap, got {reason}"
        );
    }
}
