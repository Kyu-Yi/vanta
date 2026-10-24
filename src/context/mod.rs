// ============================================================================
// CONTEXT — localize the source a role is shown (design §1.6, feature items 1-5)
// ============================================================================
//
// [VANTA-WHY] Before this module, the Creator was shown "the first ~262 KB of
// the repo": `repo::collect_source_files` walked in `read_dir` order and applied
// 256 KB total / 64 KB per file DURING the walk, `continue`-ing on any file that
// would exceed the cap. The surviving set was therefore "the files the
// filesystem happened to hand us first", and on matplotlib (11.8 MB) the bug's
// own file was usually not in it. `prepare --measure-only` rejected 10 of 11
// dev/reserve bugs for exactly that reason (`source_truncated_by_cap`).
//
// The fix is to select rather than to truncate:
//
//   1. Walk the WHOLE tree deterministically, applying no cap at all. Record
//      every directory and file we deliberately skipped and why.
//   2. Rank the candidates (review mode: the patch's own touched files; repair
//      mode: the issue text).
//   3. Fill the role's token budget greedily BY SCORE — a file that fits goes in
//      full, a file over 64 KB goes in as an outline (signatures + docstrings),
//      so a big touched file is never silently dropped.
//
// The caps now apply only to the SELECTED set, never before selection, and the
// selection uses only what a real run has (the tree, the issue text, and — in
// review mode — the patch under review). No hidden information is ever a
// selection input: see `select_review_context`'s contract and the module tests.
//
// This is the single implementation of both the walk and the budget. The
// harness's context gate shells out to `vanta_m3 --select-context --json`
// (feature item 5) rather than re-deriving either, because a gate that measures
// a different prompt from the one the run sends is worse than no gate.
// ============================================================================

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

// ----------------------------------------------------------------------------
// Walk constants — the extension set and prune list, one definition
// ----------------------------------------------------------------------------
pub const SOURCE_EXTENSIONS: &[&str] = &[
    "py", "rs", "go", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h", "hpp",
];

/// A selected file larger than this is shown as an outline rather than in full.
///
/// [VANTA-WHY] This was a *skip* threshold in the old walk: a 70 KB file was
/// dropped whole. A patch that touches a 70 KB file would then be reviewed
/// without it ever appearing in the prompt. As a *display* threshold it is the
/// right shape — the file still appears, its signatures and docstrings intact.
pub const MAX_FILE_BYTES: usize = 64 * 1024;

/// Directories never descended into: VCS metadata, build output, and vendored or
/// generated trees. Name-based, so it is cheap and stable.
pub const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "build", "dist", "target", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "site-packages", ".idea", ".vscode",
    "vendor", "third_party", ".tox", ".nox", "eggs", ".eggs", "htmlcov", ".cache",
];

// ----------------------------------------------------------------------------
// Budget constants — mirrors of the harness gate (§1.6.2) and of llm::mod
// ----------------------------------------------------------------------------
/// What the Creator requests as completion (`SAMPLING_SCHEDULE`'s `"creator"`
/// row). Prompt and completion share `n_ctx`, so it is subtracted first.
pub const OUTPUT_RESERVE_TOKENS: u32 = 32_768;
/// `llm::CHAT_OVERHEAD_TOKENS` — inside the guard's comparison, so outside it here.
pub const CHAT_OVERHEAD_TOKENS: u32 = 256;
/// Fraction of `n_ctx` held back for feedback growth across attempts.
pub const CONTEXT_MARGIN_FRACTION: f64 = 0.25;
/// `llm::TOKEN_ESTIMATE_BYTES_PER_TOKEN`, ceil division included.
pub const BYTES_PER_TOKEN_ESTIMATE: usize = 4;
/// System prompt, framing and per-attempt scaffolding.
pub const PROMPT_FRAME_TOKENS: u32 = 3_000;
/// What the run-time guard falls back to when the server cannot be reached.
pub const FALLBACK_CONTEXT_TOKENS: u32 = 65_536;

/// `ceil(bytes / 4)` — the same estimate `llm::conservative_token_estimate` uses.
pub fn estimate_tokens(bytes: usize) -> u32 {
    let whole = bytes / BYTES_PER_TOKEN_ESTIMATE;
    let over = usize::from(bytes % BYTES_PER_TOKEN_ESTIMATE != 0);
    (whole + over) as u32
}

/// `(budget, margin)` for a window: `n_ctx − output reserve − overhead − margin`,
/// with `margin = 25 % of n_ctx` (§1.6.2).
///
/// The first two terms are what `llm::context_budget_exceeded` subtracts, so this
/// budget's boundary sits on the run-time guard's boundary; the margin is the
/// harness's addition. Both are subtracted identically in
/// `scripts/bugsinpy/context.py`, which is what lets the gate's verdict mean
/// anything about the run.
pub fn budget_tokens(n_ctx: u32) -> (u32, u32) {
    let margin = (n_ctx as f64 * CONTEXT_MARGIN_FRACTION) as u32;
    let budget = n_ctx
        .saturating_sub(OUTPUT_RESERVE_TOKENS)
        .saturating_sub(CHAT_OVERHEAD_TOKENS)
        .saturating_sub(margin);
    (budget, margin)
}

// ----------------------------------------------------------------------------
// The deterministic walk
// ----------------------------------------------------------------------------
/// A source file found by the walk, with its FULL contents (no cap applied).
#[derive(Debug, Clone)]
pub struct Candidate {
    pub path: String,
    pub contents: String,
}

/// Something the walk deliberately did not descend into or read, and why.
///
/// [VANTA-WHY] "We skipped 3000 files" is not a fact anyone can check. "We
/// pruned `build/`, `node_modules/` and `.git/`, and these 4 symlinks" is. The
/// ledger is what makes the walk auditable instead of merely plausible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SkipRecord {
    pub path: String,
    pub reason: String,
}

/// Walk `root` deterministically and return every candidate plus the skip ledger.
///
/// Deterministic means: directory entries are sorted at every level and the
/// final candidate list is sorted by path, so two runs over the same tree —
/// on any filesystem — produce the same order. The old walk used `read_dir`
/// order, which is why the same bug could be in the prompt on one machine and
/// absent on another.
///
/// No cap is applied here. Caps belong to the selected set.
pub fn walk_candidates(root: &Path) -> (Vec<Candidate>, Vec<SkipRecord>) {
    let mut files: Vec<PathBuf> = Vec::new();
    let mut skipped: Vec<SkipRecord> = Vec::new();
    walk(root, root, &mut files, &mut skipped);
    files.sort();

    let mut candidates: Vec<Candidate> = Vec::with_capacity(files.len());
    for path in files {
        let rel = relative_path(root, &path);
        match fs::read_to_string(&path) {
            Ok(contents) => candidates.push(Candidate { path: rel, contents }),
            Err(_) => skipped.push(SkipRecord {
                path: rel,
                reason: "unreadable".to_string(),
            }),
        }
    }
    (candidates, skipped)
}

fn walk(root: &Path, current: &Path, files: &mut Vec<PathBuf>, skipped: &mut Vec<SkipRecord>) {
    let entries = match fs::read_dir(current) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
    paths.sort();

    for path in paths {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("")
            .to_string();

        // Symlinks are never followed: a link cycle would make the walk
        // non-terminating, and a link out of the tree makes "the repo" depend on
        // where it is checked out.
        if path.is_symlink() {
            skipped.push(SkipRecord {
                path: relative_path(root, &path),
                reason: "symlink".to_string(),
            });
            continue;
        }

        if path.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) {
                skipped.push(SkipRecord {
                    path: relative_path(root, &path),
                    reason: format!("pruned_dir:{name}"),
                });
                continue;
            }
            walk(root, &path, files, skipped);
            continue;
        }

        let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        if !SOURCE_EXTENSIONS.contains(&ext) {
            continue;
        }
        files.push(path);
    }
}

fn relative_path(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

// ----------------------------------------------------------------------------
// Selection result — what was shown, why, and at what cost (provenance, item 4)
// ----------------------------------------------------------------------------
/// Why a file was selected. The string form is what the session log records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionReason {
    /// A file the patch under review touches (review mode).
    PatchTouched,
    /// A module-level import neighbour of a selected file, one hop out.
    ImportNeighbour,
    /// A visible test file that exercises a selected module (review mode).
    Test,
    /// A frame named in a traceback inside the issue text.
    Traceback,
    /// A path or module named explicitly in the issue text.
    Path,
    /// A `def`/`class` defined in this file whose name the issue text mentions.
    Identifier,
}

impl SelectionReason {
    pub fn weight(self) -> i64 {
        match self {
            SelectionReason::PatchTouched => 1000,
            SelectionReason::ImportNeighbour => 100,
            SelectionReason::Test => 50,
            SelectionReason::Traceback => 60,
            SelectionReason::Path => 40,
            SelectionReason::Identifier => 30,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SelectionReason::PatchTouched => "patch-touched",
            SelectionReason::ImportNeighbour => "import-neighbour",
            SelectionReason::Test => "test",
            SelectionReason::Traceback => "traceback",
            SelectionReason::Path => "path",
            SelectionReason::Identifier => "identifier",
        }
    }
}

/// Whether a selected file is shown whole or as an outline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShownMode {
    Full,
    Outline,
}

/// One file in the selection, with the provenance a reader needs to explain it.
#[derive(Debug, Clone, Serialize)]
pub struct SelectedFile {
    pub path: String,
    pub score: i64,
    pub reasons: Vec<SelectionReason>,
    /// Bytes actually shown (the outline's bytes when `shown == Outline`).
    pub bytes: usize,
    pub tokens: u32,
    pub shown: ShownMode,
    /// The text that goes into the prompt. Not serialized: the JSON artifact is
    /// provenance, and the prompt text is already in the session's forensics.
    #[serde(skip)]
    pub text: String,
}

/// A complete context selection for one role.
#[derive(Debug, Clone, Serialize)]
pub struct ContextSelection {
    pub role: String,
    pub mode: String,
    pub budget_tokens: u32,
    pub total_bytes: usize,
    pub total_tokens: u32,
    /// The repo's file tree, paths only (repair mode; empty in review mode).
    pub tree: Vec<String>,
    pub tree_bytes: usize,
    pub tree_tokens: u32,
    pub files: Vec<SelectedFile>,
    /// Candidates dropped because even their outline did not fit the budget.
    pub dropped_for_budget: Vec<String>,
    pub skipped: Vec<SkipRecord>,
}

impl ContextSelection {
    /// The prompt text: the file tree (when present), then each file in full or
    /// as an outline, in score order.
    pub fn render(&self) -> String {
        let mut out = String::new();
        if !self.tree.is_empty() {
            out.push_str("# REPOSITORY FILES (paths only)\n");
            out.push_str(&self.tree.join("\n"));
            out.push_str("\n\n");
        }
        for file in &self.files {
            out.push_str(&file.text);
            out.push_str("\n\n");
        }
        out.trim_end().to_string()
    }

    /// One-line-per-file summary for the console, in score order.
    pub fn summary_lines(&self) -> Vec<String> {
        self.files
            .iter()
            .map(|file| {
                let reasons = file
                    .reasons
                    .iter()
                    .map(|reason| reason.as_str())
                    .collect::<Vec<_>>()
                    .join("+");
                format!(
                    "  {:<52} score={:<5} {:<14} {:>7} B {:>7} tok",
                    file.path,
                    file.score,
                    reasons,
                    file.bytes,
                    file.tokens,
                )
            })
            .collect()
    }
}

// ----------------------------------------------------------------------------
// Budget fill — greedy by score, outline instead of dropping
// ----------------------------------------------------------------------------
/// The prompt text for one file: full contents, or an outline when the file is
/// over `MAX_FILE_BYTES`.
fn full_text(path: &str, contents: &str) -> String {
    format!("# FILE: {path}\n{contents}")
}

fn outline_text(path: &str, contents: &str) -> String {
    let body = outline(path, contents);
    format!(
        "# FILE: {path}\n# [OUTLINE — this file is over {} KB; signatures and docstrings only]\n{body}",
        MAX_FILE_BYTES / 1024
    )
}

/// A structural outline. Nothing is executed; this is line-based parsing.
pub fn outline(path: &str, contents: &str) -> String {
    if path.ends_with(".py") || path.ends_with(".pyi") {
        python_outline(contents)
    } else {
        generic_outline(contents)
    }
}

/// `def`/`class` signatures plus docstrings, plus the module docstring and the
/// module-level imports (an outline without them cannot be reasoned about).
pub fn python_outline(contents: &str) -> String {
    let lines: Vec<&str> = contents.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut index = 0usize;
    let mut seen_symbol = false;

    while index < lines.len() {
        let raw = lines[index];
        let trimmed = raw.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            index += 1;
            continue;
        }

        // Decorators stay attached to the definition below them.
        if trimmed.starts_with('@') {
            out.push(raw.trim_end().to_string());
            index += 1;
            continue;
        }

        // The module docstring: a string statement before any def/class/import.
        if !seen_symbol && string_delimiter(trimmed).is_some() {
            let (text, next) = read_string_statement(&lines, index);
            out.push(text);
            index = next;
            continue;
        }

        if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
            out.push(raw.trim_end().to_string());
            index += 1;
            continue;
        }

        let is_definition = trimmed.starts_with("def ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("async def ");
        if is_definition {
            let (signature, next) = read_signature(&lines, index);
            out.push(signature);
            seen_symbol = true;
            index = next;
            let (doc, after) = read_docstring(&lines, index);
            if let Some(doc) = doc {
                out.push(doc);
                index = after;
            }
            continue;
        }

        index += 1;
    }

    out.join("\n")
}

fn generic_outline(contents: &str) -> String {
    const KEYWORDS: &[&str] = &[
        "fn ", "pub fn ", "def ", "class ", "struct ", "enum ", "interface ", "function ",
        "export ", "impl ", "trait ", "type ",
    ];
    contents
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            KEYWORDS.iter().any(|keyword| trimmed.starts_with(keyword))
        })
        .map(|line| line.trim_end().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

fn string_delimiter(trimmed: &str) -> Option<&'static str> {
    for delimiter in ["\"\"\"", "'''", "\"", "'"] {
        if trimmed.starts_with(delimiter) {
            return Some(delimiter);
        }
    }
    None
}

/// Read a string statement starting at `start`; returns its text and the index
/// of the first line after it.
fn read_string_statement(lines: &[&str], start: usize) -> (String, usize) {
    let trimmed = lines[start].trim_start();
    let Some(delimiter) = string_delimiter(trimmed) else {
        return (lines[start].trim_end().to_string(), start + 1);
    };

    let mut text: Vec<String> = vec![trimmed.trim_end().to_string()];
    let openers = &trimmed[delimiter.len()..];
    if openers.contains(delimiter) {
        // Closes on the same line.
        return (text.join("\n"), start + 1);
    }
    let mut index = start + 1;
    while index < lines.len() {
        text.push(lines[index].trim_end().to_string());
        if lines[index].contains(delimiter) {
            return (text.join("\n"), index + 1);
        }
        index += 1;
    }
    (text.join("\n"), lines.len())
}

/// Read a `def`/`class` signature: the logical line up to (and including) the
/// colon that ends it. Bracket depth is tracked so a multi-line signature with
/// annotated defaults is not cut at the first `:`.
fn read_signature(lines: &[&str], start: usize) -> (String, usize) {
    let mut text = String::new();
    let mut depth: i32 = 0;
    let mut index = start;

    while index < lines.len() {
        let line = lines[index].trim_end();
        if index > start {
            text.push('\n');
        }
        text.push_str(line);

        let code = line.split('#').next().unwrap_or("");
        for character in code.chars() {
            match character {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            }
        }
        if depth <= 0 && code.contains(':') {
            return (text, index + 1);
        }
        index += 1;
        if index - start > 40 {
            break;
        }
    }
    (text, index)
}

/// A docstring is the first statement of a body when it is a bare string.
fn read_docstring(lines: &[&str], start: usize) -> (Option<String>, usize) {
    let mut index = start;
    while index < lines.len() {
        let trimmed = lines[index].trim();
        if trimmed.is_empty() {
            index += 1;
            continue;
        }
        if let Some(_) = string_delimiter(trimmed) {
            let (text, next) = read_string_statement(lines, index);
            return (Some(text), next);
        }
        return (None, index);
    }
    (None, lines.len())
}

// ----------------------------------------------------------------------------
// Review-mode selection (item 1)
// ----------------------------------------------------------------------------
/// Select the context for a REVIEW run (a patch is given).
///
/// The selection is, in full:
///
/// * the patch's own touched files,
/// * the modules those files import at module level, one hop, and
/// * visible test files that import a touched module or name one of its modules.
///
/// [VANTA-WHY] Nothing else is an input, and nothing hidden can become one: the
/// function sees the tree and the patch under review, both of which the run
/// already has. `FAIL_TO_PASS`, the fix commit, the fix diff and `bug.info` are
/// not parameters and cannot be reached from here. The patch is "the thing under
/// review", so reading it is not a peek at the answer — it is the subject.
pub fn select_review_context(
    candidates: &[Candidate],
    touched: &BTreeSet<String>,
    role: &str,
    budget_tokens: u32,
) -> ContextSelection {
    let by_path: BTreeMap<&str, &Candidate> = candidates
        .iter()
        .map(|candidate| (candidate.path.as_str(), candidate))
        .collect();
    let module_map = python_module_to_path(candidates);

    let mut reasons: BTreeMap<String, BTreeSet<SelectionReason>> = BTreeMap::new();
    let mut mark = |path: &str, reason: SelectionReason| {
        reasons.entry(path.to_string()).or_default().insert(reason);
    };

    for path in touched {
        if by_path.contains_key(path.as_str()) {
            mark(path, SelectionReason::PatchTouched);
        }
    }

    let mut neighbours: BTreeSet<String> = BTreeSet::new();
    for path in touched {
        if let Some(candidate) = by_path.get(path.as_str()) {
            neighbours.extend(imported_repo_paths(
                &candidate.path,
                &candidate.contents,
                &module_map,
            ));
        }
    }
    for path in neighbours {
        if !touched.contains(&path) && by_path.contains_key(path.as_str()) {
            mark(&path, SelectionReason::ImportNeighbour);
        }
    }

    let touched_modules: BTreeSet<String> = touched
        .iter()
        .filter_map(|path| python_module_path(path))
        .collect();
    for candidate in candidates.iter().filter(|c| is_test_path(&c.path)) {
        let imports = imported_repo_paths(&candidate.path, &candidate.contents, &module_map);
        if !imports.is_disjoint(touched) || test_covers_modules(&candidate.contents, &touched_modules)
        {
            mark(&candidate.path, SelectionReason::Test);
        }
    }

    fill(
        "review",
        role,
        candidates,
        reasons,
        budget_tokens,
        &[],
    )
}

// ----------------------------------------------------------------------------
// The shared fill: order by score, then spend the budget
// ----------------------------------------------------------------------------
fn fill(
    mode: &str,
    role: &str,
    candidates: &[Candidate],
    reasons: BTreeMap<String, BTreeSet<SelectionReason>>,
    budget_tokens: u32,
    tree: &[String],
) -> ContextSelection {
    let by_path: BTreeMap<&str, &Candidate> = candidates
        .iter()
        .map(|candidate| (candidate.path.as_str(), candidate))
        .collect();

    // Score = sum of the weights of the reasons a file was picked for; ties are
    // broken by path, so the order is total and stable.
    let mut ranked: Vec<(String, i64, Vec<SelectionReason>)> = reasons
        .into_iter()
        .filter(|(path, _)| by_path.contains_key(path.as_str()))
        .map(|(path, reasons)| {
            let list: Vec<SelectionReason> = reasons.iter().copied().collect();
            let score: i64 = list.iter().map(|reason| reason.weight()).sum();
            (path, score, list)
        })
        .collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    let mut tree_lines: Vec<String> = tree.to_vec();
    tree_lines.sort();
    let mut tree_bytes: usize = 0;
    if !tree_lines.is_empty() {
        tree_bytes = "# REPOSITORY FILES (paths only)\n".len()
            + tree_lines.iter().map(|line| line.len() + 1).sum::<usize>();
    }
    let tree_tokens = estimate_tokens(tree_bytes);

    let mut remaining = budget_tokens.saturating_sub(tree_tokens);
    let mut files: Vec<SelectedFile> = Vec::new();
    let mut dropped_for_budget: Vec<String> = Vec::new();
    let mut total_bytes: usize = 0;
    let mut total_tokens: u32 = 0;

    for (path, score, reasons) in ranked {
        let candidate = match by_path.get(path.as_str()) {
            Some(candidate) => *candidate,
            None => continue,
        };

        // A file over the per-file cap is shown as an outline, never dropped.
        let fits_whole = candidate.contents.len() <= MAX_FILE_BYTES;
        let mut text = if fits_whole {
            full_text(&path, &candidate.contents)
        } else {
            outline_text(&path, &candidate.contents)
        };
        let mut shown = if fits_whole {
            ShownMode::Full
        } else {
            ShownMode::Outline
        };
        let mut cost = estimate_tokens(text.len());

        if cost > remaining && fits_whole {
            // Even a file that would fit whole may not fit now; fall back to the
            // outline rather than dropping it.
            text = outline_text(&path, &candidate.contents);
            shown = ShownMode::Outline;
            cost = estimate_tokens(text.len());
        }

        if cost > remaining {
            dropped_for_budget.push(path);
            continue;
        }

        remaining -= cost;
        total_bytes += text.len();
        total_tokens += cost;
        files.push(SelectedFile {
            path,
            score,
            reasons,
            bytes: text.len(),
            tokens: cost,
            shown,
            text,
        });
    }

    ContextSelection {
        role: role.to_string(),
        mode: mode.to_string(),
        budget_tokens,
        total_bytes: total_bytes + tree_bytes,
        total_tokens: total_tokens + tree_tokens,
        tree: tree_lines,
        tree_bytes,
        tree_tokens,
        files,
        dropped_for_budget,
        skipped: Vec::new(),
    }
}

// ----------------------------------------------------------------------------
// Python import resolution — one hop of module-level neighbours
// ----------------------------------------------------------------------------
pub fn python_module_to_path(candidates: &[Candidate]) -> BTreeMap<String, String> {
    let mut module_map = BTreeMap::new();
    for candidate in candidates {
        if let Some(module) = python_module_path(&candidate.path) {
            module_map.insert(module, candidate.path.clone());
        }
    }
    module_map
}

pub fn python_module_path(path: &str) -> Option<String> {
    if !path.ends_with(".py") {
        return None;
    }
    let trimmed = path.trim_end_matches(".py");
    if let Some(pkg) = trimmed.strip_suffix("/__init__") {
        return Some(pkg.replace('/', "."));
    }
    Some(trimmed.replace('/', "."))
}

pub fn is_test_path(path: &str) -> bool {
    let file_name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    path.starts_with("tests/")
        || path.contains("/tests/")
        || file_name.starts_with("test_")
        || file_name.ends_with("_test.py")
}

pub fn imported_repo_paths(
    current_path: &str,
    contents: &str,
    module_map: &BTreeMap<String, String>,
) -> BTreeSet<String> {
    let mut imported = BTreeSet::new();
    let current_package = current_package_parts(current_path);

    for raw_line in contents.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if let Some(rest) = line.strip_prefix("import ") {
            for target in rest.split(',') {
                let module = target
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim();
                if let Some(path) = module_map.get(module) {
                    imported.insert(path.clone());
                }
            }
            continue;
        }

        let Some(rest) = line.strip_prefix("from ") else {
            continue;
        };
        let Some((module_part, names_part)) = rest.split_once(" import ") else {
            continue;
        };
        let Some(base_module) = resolve_import_base(&current_package, module_part.trim()) else {
            continue;
        };
        if let Some(path) = module_map.get(&base_module) {
            imported.insert(path.clone());
        }
        for name in parse_imported_names(names_part) {
            let candidate = if base_module.is_empty() {
                name
            } else {
                format!("{base_module}.{name}")
            };
            if let Some(path) = module_map.get(&candidate) {
                imported.insert(path.clone());
            }
        }
    }

    imported
}

fn current_package_parts(path: &str) -> Vec<String> {
    let Some(module) = python_module_path(path) else {
        return Vec::new();
    };
    let mut parts: Vec<String> = module.split('.').map(|part| part.to_string()).collect();
    if !path.ends_with("__init__.py") {
        let _ = parts.pop();
    }
    parts
}

fn resolve_import_base(current_package: &[String], module_part: &str) -> Option<String> {
    let trimmed = module_part.trim();
    if trimmed.is_empty() {
        return None;
    }
    if !trimmed.starts_with('.') {
        return Some(trimmed.to_string());
    }

    let dot_count = trimmed.chars().take_while(|c| *c == '.').count();
    let remainder = trimmed[dot_count..].trim();
    let levels_up = dot_count.saturating_sub(1);
    if levels_up > current_package.len() {
        return None;
    }
    let kept = &current_package[..current_package.len().saturating_sub(levels_up)];
    let mut base = kept.join(".");
    if !remainder.is_empty() {
        if !base.is_empty() {
            base.push('.');
        }
        base.push_str(remainder);
    }
    Some(base)
}

fn parse_imported_names(names_part: &str) -> Vec<String> {
    names_part
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .filter_map(|name| {
            let token = name
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim()
                .trim_matches(',');
            if token.is_empty() || token == "*" {
                None
            } else {
                Some(token.to_string())
            }
        })
        .collect()
}

fn test_covers_modules(test_source: &str, touched_modules: &BTreeSet<String>) -> bool {
    touched_modules
        .iter()
        .any(|module| test_source.contains(module))
}

// ============================================================================
// UNIT TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(path: &str, contents: &str) -> Candidate {
        Candidate {
            path: path.to_string(),
            contents: contents.to_string(),
        }
    }

    fn paths(selection: &ContextSelection) -> Vec<&str> {
        selection
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect()
    }

    #[test]
    fn python_outline_keeps_signatures_and_docstrings_and_drops_bodies() {
        // Nothing here executes Python; the outline is line-based parsing.
        let source = "\"\"\"Module docs.\"\"\"\n\nimport os\nfrom . import other\n\n\nclass Context:\n    \"\"\"A context.\"\"\"\n\n    def method(self, a: int = 0) -> None:\n        \"\"\"Do a thing.\"\"\"\n        return None\n\n\ndef free_function(\n    first,\n    second,\n):\n    \"\"\"Free.\"\"\"\n    return first\n\n\ndef tail():\n    return 1\n";
        let outline = python_outline(source);
        assert!(outline.contains("\"\"\"Module docs.\"\"\""));
        assert!(outline.contains("import os"));
        assert!(outline.contains("from . import other"));
        assert!(outline.contains("class Context:"));
        assert!(outline.contains("\"\"\"A context.\"\"\""));
        assert!(outline.contains("def method(self, a: int = 0) -> None:"));
        assert!(outline.contains("def free_function("));
        assert!(outline.contains("def tail():"));
        // Bodies are gone: no bare statement survives.
        assert!(!outline.contains("return first"));
        assert!(!outline.contains("return None"));
    }

    #[test]
    fn the_walk_is_deterministic_and_records_what_it_pruned() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Written out of order on purpose: the walk must not depend on it.
        fs::write(tmp.path().join("zeta.py"), "def z():\n    pass\n").unwrap();
        fs::write(tmp.path().join("alpha.py"), "def a():\n    pass\n").unwrap();
        fs::create_dir_all(tmp.path().join("build")).unwrap();
        fs::write(tmp.path().join("build/generated.py"), "x = 1\n").unwrap();
        fs::create_dir_all(tmp.path().join("sub")).unwrap();
        fs::write(tmp.path().join("sub/mid.py"), "def m():\n    pass\n").unwrap();
        fs::write(tmp.path().join("README.md"), "not source\n").unwrap();

        let (candidates, skipped) = walk_candidates(tmp.path());

        assert_eq!(
            candidates.iter().map(|c| c.path.as_str()).collect::<Vec<_>>(),
            vec!["alpha.py", "sub/mid.py", "zeta.py"],
            "the candidate list is sorted by path and free of build output"
        );
        assert!(skipped.iter().any(|record| record.path == "build"
            && record.reason == "pruned_dir:build"));
        assert!(
            !skipped.iter().any(|record| record.path == "README.md"),
            "an unsupported extension is not worth a ledger row"
        );
    }

    #[test]
    fn review_selection_follows_the_patch_to_imports_and_covering_tests() {
        let candidates = vec![
            candidate(
                "heretic/keyword_rate.py",
                "from .plugin import Context\n\nclass KeywordRate:\n    pass\n",
            ),
            candidate("heretic/plugin.py", "class Context:\n    pass\n"),
            candidate("heretic/other.py", "class Other:\n    pass\n"),
            candidate(
                "tests/test_keyword_rate.py",
                "from heretic.keyword_rate import KeywordRate\nfrom heretic.plugin import Context\n\ndef test_keyword_rate():\n    assert KeywordRate is not None\n",
            ),
            candidate(
                "tests/test_other.py",
                "from heretic.other import Other\n\ndef test_other():\n    assert Other is not None\n",
            ),
        ];
        let touched = BTreeSet::from(["heretic/keyword_rate.py".to_string()]);

        let selection = select_review_context(&candidates, &touched, "review", 60_000);
        let rendered = selection.render();

        assert!(rendered.contains("# FILE: heretic/keyword_rate.py"));
        assert!(rendered.contains("# FILE: heretic/plugin.py"));
        assert!(rendered.contains("# FILE: tests/test_keyword_rate.py"));
        assert!(
            !rendered.contains("# FILE: heretic/other.py"),
            "unrelated imports stay out of the review prompt"
        );
        assert!(
            !rendered.contains("# FILE: tests/test_other.py"),
            "unrelated tests stay out of the review prompt"
        );

        let touched_file = selection
            .files
            .iter()
            .find(|file| file.path == "heretic/keyword_rate.py")
            .expect("the touched file is always shown");
        assert_eq!(touched_file.reasons, vec![SelectionReason::PatchTouched]);
        assert_eq!(touched_file.shown, ShownMode::Full);
        assert_eq!(
            paths(&selection)[0],
            "heretic/keyword_rate.py",
            "the patch's own file outranks its neighbours and the tests that cover it"
        );
    }
}
