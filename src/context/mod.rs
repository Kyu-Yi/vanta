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
///
/// [VANTA-WHY] The weights are ordered so that a file the *issue* names always
/// outranks a file that is merely an import neighbour of one: path 400,
/// traceback 300, identifier 200, neighbour 100. The earlier review-only ordering
/// had `ImportNeighbour` (100) above the issue signals, which would have put a
/// neighbour of a named file into the prompt *ahead of* the named file itself.
/// `PatchTouched` stays far above everything: in review mode the patch is the
/// subject, so nothing outranks the files it touches. `Tree` (10) sits below all
/// of them: it says only "this file is in the repository", so it is used only
/// when no other reason reached any file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionReason {
    /// A file the patch under review touches (review mode).
    PatchTouched,
    /// A path or module named explicitly in the issue text (repair mode).
    Path,
    /// A frame named in a traceback inside the issue text (repair mode).
    Traceback,
    /// A `def`/`class` defined in this file whose name the issue text mentions.
    Identifier,
    /// A module-level import neighbour of a selected file, one hop out.
    ImportNeighbour,
    /// A visible test file that exercises a selected module (review mode).
    Test,
    /// A file of the tree itself, in walk order: the fallback a ranking with no
    /// seed at all falls back to.
    ///
    /// [VANTA-WHY] A selection with no seed used to be *empty*, and the prompt
    /// was then the repository file tree and nothing else — a listing of paths,
    /// not source. A role shown only a listing has been told where the code is
    /// and nothing about what it does, which is not a context; the harness's gate
    /// passed 7 of 11 dev bugs on exactly that (a selection whose bytes were
    /// byte-for-byte the tree listing's, because `total_bytes = 0 + tree_bytes`).
    /// §1.6.2's prediction that such a bug "will still fall back to the tree plus
    /// whatever its one-hop neighbours reach" had no seed to hop from, so the
    /// fallback is made explicit here instead of silently empty: the tree's own
    /// files, in walk order, to the budget. It carries the weakest weight, it is
    /// used only when nothing else scored, and it is named in the provenance so a
    /// reader can tell a file the *issue* named from a file that is merely in the
    /// repository.
    Tree,
}

impl SelectionReason {
    pub fn weight(self) -> i64 {
        match self {
            SelectionReason::PatchTouched => 1000,
            SelectionReason::Path => 400,
            SelectionReason::Traceback => 300,
            SelectionReason::Identifier => 200,
            SelectionReason::ImportNeighbour => 100,
            SelectionReason::Test => 80,
            SelectionReason::Tree => 10,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SelectionReason::PatchTouched => "patch-touched",
            SelectionReason::Path => "path",
            SelectionReason::Traceback => "traceback",
            SelectionReason::Identifier => "identifier",
            SelectionReason::ImportNeighbour => "import-neighbour",
            SelectionReason::Test => "test",
            SelectionReason::Tree => "tree",
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

impl ShownMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ShownMode::Full => "full",
            ShownMode::Outline => "outline",
        }
    }
}

/// One file in the selection, with the provenance a reader needs to explain it.
#[derive(Debug, Clone, Serialize)]
pub struct SelectedFile {
    pub path: String,
    pub score: i64,
    pub reasons: Vec<SelectionReason>,
    /// Bytes actually shown (the framed text's bytes when `shown == Outline`).
    pub bytes: usize,
    pub tokens: u32,
    pub shown: ShownMode,
    /// The content shown for this file, WITHOUT the `# FILE: <path>` header: the
    /// full source, or the outline body for a file that does not fit whole.
    ///
    /// [VANTA-WHY] Stored headerless so the two consumers can frame it their own
    /// way — `render()` writes the `# FILE:` header the prompt uses, and the
    /// Creator's `CreatorRequest` gets a plain `SourceFile` it frames itself — and
    /// not serialized, since the JSON artifact is provenance and the prompt text
    /// is already in the session's forensics.
    #[serde(skip)]
    pub body: String,
}

impl SelectedFile {
    /// The prompt text for this file: the header plus its body.
    pub fn framed(&self) -> String {
        format!("# FILE: {}\n{}", self.path, self.body)
    }
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
            out.push_str(&file.framed());
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

    /// The provenance of this selection as ONE line, for the session log and
    /// `run.summary`.
    ///
    /// [VANTA-WHY] §1.6.2 item 4 — "was the bug's file in the prompt, and what
    /// did it cost" is a question a report must answer from the artifacts alone,
    /// without re-running the selector, so every selected file travels with its
    /// score, the reason(s) that chose it, whether it went in whole or as an
    /// outline, and the bytes and tokens it occupied. The walk's prune ledger
    /// rides along for the same reason: the console can only afford a count, and
    /// the count is read against the list that says WHICH paths were dropped —
    /// a repo that skipped a vendored directory and a repo that skipped the
    /// package under test must not look alike.
    ///
    /// The line is single and space-free because it is embedded in rows that are
    /// parsed as space-separated `key=value` pairs, where a space would silently
    /// split a key and a newline would split a row.
    pub fn provenance_line(&self) -> String {
        let files = if self.files.is_empty() {
            "none".to_string()
        } else {
            self.files
                .iter()
                .map(|file| {
                    format!(
                        "{}@{}[{}]{}:{}B/{}tok",
                        file.path,
                        file.score,
                        file.reasons
                            .iter()
                            .map(|reason| reason.as_str())
                            .collect::<Vec<_>>()
                            .join("+"),
                        file.shown.as_str(),
                        file.bytes,
                        file.tokens,
                    )
                })
                .collect::<Vec<_>>()
                .join(";")
        };
        let pruned = if self.skipped.is_empty() {
            "none".to_string()
        } else {
            self.skipped
                .iter()
                .map(|skip| format!("{}:{}", skip.path, skip.reason))
                .collect::<Vec<_>>()
                .join(";")
        };
        format!(
            "mode={}:role={}:budget={}:files={}:tree={}:dropped={}:total={}B/{}tok:selected=[{}]:pruned=[{}]",
            self.mode,
            self.role,
            self.budget_tokens,
            self.files.len(),
            self.tree.len(),
            self.dropped_for_budget.len(),
            self.total_bytes,
            self.total_tokens,
            files,
            pruned,
        )
    }
}

// ----------------------------------------------------------------------------
// Budget fill — greedy by score, outline instead of dropping
// ----------------------------------------------------------------------------
/// The framed bytes of one file: what `SelectedFile::framed` will write.
fn framed_len(path: &str, body: &str) -> usize {
    "# FILE: ".len() + path.len() + 1 + body.len()
}

/// The outline body for a file over `MAX_FILE_BYTES`: the outline plus a marker
/// saying why the file is not shown in full.
fn outline_body(path: &str, contents: &str) -> String {
    format!(
        "# [OUTLINE — this file is over {} KB; signatures and docstrings only]\n{}",
        MAX_FILE_BYTES / 1024,
        outline_for(path, contents),
    )
}

/// The outline for `path`, choosing the Python grammar for `.py`/`.pyi`.
/// Nothing is executed; this is line-based parsing.
pub fn outline_for(path: &str, contents: &str) -> String {
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
// Repair-mode selection (item 2) — rank from the issue text only
// ----------------------------------------------------------------------------
/// Select the context for a REPAIR run (no patch).
///
/// The ranking reads the issue text and nothing else that a real run does not
/// already have: the tree (for the file tree the model is shown) and the issue
/// text. No hidden test, no fix commit, no fix diff, no `bug.info` — none is a
/// parameter and none can be reached from here.
///
/// A file scores by the sum of the reasons it was picked for:
///
/// * `Path`       — the issue names its path, its basename or its module;
/// * `Traceback`  — the issue contains a traceback whose frames name it;
/// * `Identifier` — the issue names a symbol the file defines at module level
///                  (a `def`/`class`, found by parsing, never by running code);
/// * `ImportNeighbour` — a module-level import of one of the above, one hop out.
///
/// The seed files are the first three; import neighbours are added for them and
/// score below every seed, so a file the issue names is never displaced by a
/// neighbour of one. The repo's file tree (paths only) is included so the model
/// can see what exists beyond the files that fit.
///
/// [VANTA-WHY] An issue text that names nothing — BugsInPy's is usually the fix
/// commit message, one sentence of prose, e.g. `fix calling method on wrong
/// object` — reaches no file with any of the three reasons, has no seed for the
/// neighbour hop, and used to leave the selection empty: the prompt was then the
/// file tree alone. It now falls back to the tree's own files in walk order
/// (`SelectionReason::Tree`), so "no file the issue named" is not "no context".
pub fn select_repair_context(
    candidates: &[Candidate],
    issue_text: &str,
    role: &str,
    budget_tokens: u32,
) -> ContextSelection {
    let module_map = python_module_to_path(candidates);
    let path_tokens = issue_tokens(issue_text);
    let frame_tokens = issue_traceback_files(issue_text);
    let identifiers = issue_identifiers(issue_text);

    let mut reasons: BTreeMap<String, BTreeSet<SelectionReason>> = BTreeMap::new();

    for candidate in candidates {
        let module = python_module_path(&candidate.path);
        if matches_path(&path_tokens, &candidate.path, module.as_deref()) {
            mark_reason(&mut reasons, &candidate.path, SelectionReason::Path);
        }
        if matches_path(&frame_tokens, &candidate.path, module.as_deref()) {
            mark_reason(&mut reasons, &candidate.path, SelectionReason::Traceback);
        }
        if !identifiers.is_empty() {
            let symbols = module_level_symbols(&candidate.path, &candidate.contents);
            if symbols.iter().any(|symbol| identifiers.contains(symbol)) {
                mark_reason(&mut reasons, &candidate.path, SelectionReason::Identifier);
            }
        }
    }

    // One hop of import neighbours for the files the issue names. They score
    // below every seed (see `SelectionReason::weight`), so they fill the budget
    // only after the named files and their neighbours are in.
    let seeds: Vec<String> = reasons.keys().cloned().collect();
    let mut neighbours: BTreeSet<String> = BTreeSet::new();
    for path in &seeds {
        if let Some(candidate) = candidates.iter().find(|c| &c.path == path) {
            neighbours.extend(imported_repo_paths(
                &candidate.path,
                &candidate.contents,
                &module_map,
            ));
        }
    }
    for path in neighbours {
        if !reasons.contains_key(&path) {
            mark_reason(&mut reasons, &path, SelectionReason::ImportNeighbour);
        }
    }

    let tree: Vec<String> = candidates.iter().map(|c| c.path.clone()).collect();
    fill("repair", role, candidates, reasons, budget_tokens, &tree)
}

/// Select the context a role is shown from the inputs a REAL run has, and only
/// those: the tree, the issue text, the patch under review (when there is one),
/// the role, and the budget.
///
/// [VANTA-WHY] This is the entry point `vanta_m3 --select-context --json` calls,
/// so the harness's context gate measures the very selection the run will send
/// rather than a re-derivation of it. It has no parameter for `FAIL_TO_PASS`, a
/// hidden test, the fix commit, the fix diff or `bug.info`, and none of those can
/// be reached from here — a gate that could see the answer would size the prompt
/// around it.
pub fn select_for_role(
    tree: &Path,
    issue_text: &str,
    patch_text: Option<&str>,
    role: &str,
    budget_tokens: u32,
) -> ContextSelection {
    let (candidates, skipped) = walk_candidates(tree);
    let mut selection = match patch_text {
        Some(patch) => {
            let touched: BTreeSet<String> = crate::agents::creator::split_patch_into_files(patch)
                .keys()
                .cloned()
                .collect();
            select_review_context(&candidates, &touched, role, budget_tokens)
        }
        None => select_repair_context(&candidates, issue_text, role, budget_tokens),
    };
    selection.skipped = skipped;
    selection
}

/// Record one reason for one path.
fn mark_reason(
    reasons: &mut BTreeMap<String, BTreeSet<SelectionReason>>,
    path: &str,
    reason: SelectionReason,
) {
    reasons.entry(path.to_string()).or_default().insert(reason);
}

/// The whitespace/punctuation-delimited tokens of the issue, keeping `.` and `/`
/// so that `heretic/keyword_rate.py` and `heretic.keyword_rate` survive whole.
pub fn issue_tokens(issue_text: &str) -> BTreeSet<String> {
    issue_text
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '/' || c == '-'))
        .map(|token| token.trim_matches(|c: char| c == '.' || c == '/' || c == '-' || c == '_'))
        .filter(|token| token.len() >= 3 && token.chars().any(|c| c.is_alphanumeric()))
        .map(str::to_string)
        .collect()
}

/// The file paths named in a traceback inside the issue: `File "x/y.py", line N`
/// and the bare `x/y.py:NN` form.
pub fn issue_traceback_files(issue_text: &str) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    for line in issue_text.lines() {
        if let Some(start) = line.find("File \"") {
            let rest = &line[start + "File \"".len()..];
            if let Some(end) = rest.find('"') {
                let path = rest[..end].trim();
                if looks_like_source(path) {
                    out.insert(path.to_string());
                }
            }
        }
        for raw in line.split_whitespace() {
            let token = raw.trim_matches(|c: char| "(),;'\"[]".contains(c));
            if let Some((path, number)) = token.rsplit_once(':') {
                if !number.is_empty()
                    && number.chars().all(|c| c.is_ascii_digit())
                    && looks_like_source(path)
                {
                    out.insert(path.to_string());
                }
            }
        }
    }
    out
}

/// The identifiers the issue names: everything in backticks, plus bare CamelCase
/// or snake_case words. Dotted spans contribute each segment.
pub fn issue_identifiers(issue_text: &str) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();

    let segments: Vec<&str> = issue_text.split('`').collect();
    for chunk in segments.iter().skip(1).step_by(2) {
        for piece in chunk.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')) {
            for segment in piece.split('.') {
                if is_identifier(segment) && segment.len() >= 2 {
                    out.insert(segment.to_string());
                }
            }
        }
    }

    for piece in issue_text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')) {
        for segment in piece.split('.') {
            if looks_symbolish(segment) {
                out.insert(segment.to_string());
            }
        }
    }
    out
}

/// The module-level `def`/`class` names a file defines, by line-based parsing.
/// Nothing is imported or executed.
pub fn module_level_symbols(path: &str, contents: &str) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    for line in contents.lines() {
        if line.starts_with(' ') || line.starts_with('\t') {
            continue; // not module level
        }
        let trimmed = line.trim_end();
        let name = if path.ends_with(".py") || path.ends_with(".pyi") {
            python_def_name(trimmed)
        } else {
            generic_def_name(trimmed)
        };
        if let Some(name) = name {
            out.insert(name);
        }
    }
    out
}

fn python_def_name(line: &str) -> Option<String> {
    let rest = line
        .strip_prefix("async def ")
        .or_else(|| line.strip_prefix("def "))
        .or_else(|| line.strip_prefix("class "))?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    identifier_or_none(&name)
}

fn generic_def_name(line: &str) -> Option<String> {
    for keyword in ["pub fn ", "fn ", "pub struct ", "struct ", "pub enum ", "enum ", "interface ", "function ", "class "] {
        if let Some(rest) = line.strip_prefix(keyword) {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if let Some(name) = identifier_or_none(&name) {
                return Some(name);
            }
        }
    }
    None
}

fn identifier_or_none(name: &str) -> Option<String> {
    if is_identifier(name) {
        Some(name.to_string())
    } else {
        None
    }
}

fn is_identifier(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .next()
            .is_some_and(|c| c.is_alphabetic() || c == '_')
        && token.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// CamelCase (starts upper, has a lower) or snake_case (has `_`, all lower-ish).
fn looks_symbolish(token: &str) -> bool {
    if !is_identifier(token) || token.len() < 3 {
        return false;
    }
    let first_upper = token.chars().next().is_some_and(|c| c.is_uppercase());
    let has_lower = token.chars().any(|c| c.is_lowercase());
    let camel = first_upper && has_lower;
    let snake = token.contains('_')
        && token
            .chars()
            .all(|c| c.is_lowercase() || c.is_ascii_digit() || c == '_');
    camel || snake
}

fn looks_like_source(path: &str) -> bool {
    SOURCE_EXTENSIONS
        .iter()
        .any(|ext| path.ends_with(&format!(".{ext}")))
}

/// Does any token name `candidate_path` — exactly, by basename, by module name,
/// or as the tail of a longer (e.g. absolute) path?
fn matches_path(tokens: &BTreeSet<String>, candidate_path: &str, module: Option<&str>) -> bool {
    if tokens.is_empty() {
        return false;
    }
    let base = Path::new(candidate_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(candidate_path);
    if tokens.contains(candidate_path) || tokens.contains(base) {
        return true;
    }
    if let Some(module) = module {
        if tokens.contains(module) {
            return true;
        }
    }
    let full_tail = format!("/{candidate_path}");
    let base_tail = format!("/{base}");
    tokens
        .iter()
        .any(|token| token.ends_with(&full_tail) || token.ends_with(&base_tail))
}

// ----------------------------------------------------------------------------
// The shared fill: order by score, then spend the budget
// ----------------------------------------------------------------------------
/// Order the ranked files by score, then spend the budget on them.
///
/// A file that fits goes in `Full`; one over `MAX_FILE_BYTES` goes in as an
/// `Outline`; one whose outline still does not fit is dropped and named in
/// `dropped_for_budget`. When the ranking is empty the tree's own files are
/// ranked by walk order under `SelectionReason::Tree` (see the comment at the
/// fallback below) instead of an empty selection being returned as a "context".
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

    // A ranking that reached no file at all is not a context: the repository file
    // tree is a list of paths, and §1.6.2's fallback ("the tree plus whatever its
    // one-hop neighbours reach") has no seed to hop from when the issue names
    // nothing. Rather than send a listing and call it a prompt, the fallback is
    // made explicit — the tree's own files, in walk order (the walk sorts the
    // candidate list by path), which is as deterministic as any other ranking and
    // costs the same budget.
    //
    // [VANTA-WHY] This was the gate's wrong reason for passing: with `reasons`
    // empty the loop below selected nothing, `total_bytes` fell back to the tree
    // listing alone, and every such bug "fit" trivially because a listing always
    // fits. `SelectionReason::Tree` names what the fallback is so the report can
    // still tell a file the issue named from a file that merely exists.
    let fallback = ranked.is_empty();
    if fallback {
        ranked = candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.path.clone(),
                    SelectionReason::Tree.weight(),
                    vec![SelectionReason::Tree],
                )
            })
            .collect();
    }

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
        let mut body = if fits_whole {
            candidate.contents.clone()
        } else {
            outline_body(&path, &candidate.contents)
        };
        let mut shown = if fits_whole {
            ShownMode::Full
        } else {
            ShownMode::Outline
        };
        let mut cost = estimate_tokens(framed_len(&path, &body));

        if cost > remaining && fits_whole {
            // Even a file that would fit whole may not fit now; fall back to the
            // outline rather than dropping it.
            body = outline_body(&path, &candidate.contents);
            shown = ShownMode::Outline;
            cost = estimate_tokens(framed_len(&path, &body));
        }

        if cost > remaining {
            // A file the RANKING chose and the budget cannot take is a drop, and is
            // recorded: the run would be shown less than the ranking decided, which
            // is what the gate's `selection_dropped_for_budget` reads. In fallback
            // mode the ranking did not choose anything — the tree did — so leaving a
            // file out is not the ranking being overruled: it is the tail of a walk
            // whose paths the prompt already carries as the tree listing. Recording
            // those would turn "the fallback ran out of budget", which is the normal
            // case on a repo larger than the window, into a rejection.
            if !fallback {
                dropped_for_budget.push(path);
            }
            continue;
        }

        remaining -= cost;
        let bytes = framed_len(&path, &body);
        total_bytes += bytes;
        total_tokens += cost;
        files.push(SelectedFile {
            path,
            score,
            reasons,
            bytes,
            tokens: cost,
            shown,
            body,
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

    /// [VANTA-WHY] §1.6.2 — the localizer must not be moved by what the harness
    /// hides. The harness's half of that is tree surgery: `prepare` /
    /// `trees::strip_hidden_tests` removes the F2P tests from `vanta_tree` and
    /// from `vanta_clean` before any run sees them, and the fix commit, the fix
    /// diff and `bug.info` are never written into the tree at all. This test
    /// pins the localizer's half: walk a real tree, ADD a hidden test (a
    /// F2P-shaped file that imports the touched module, which is the strongest
    /// reason a test could ever be shown) and REMOVE it again, and assert that
    /// the files shown for the visible tree do not move — same paths, scores,
    /// reasons, bytes, tokens, order and bodies. A hidden test that could
    /// reorder, displace or re-cut a visible file would make the selection a
    /// channel for the answer.
    #[test]
    fn the_selection_is_unchanged_when_a_hidden_test_is_added_or_removed() {
        const HIDDEN: &str = "tests/test_hidden_bug.py";
        let tmp = tempfile::TempDir::new().unwrap();
        let write = |rel: &str, contents: &str| {
            let path = tmp.path().join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        };
        write(
            "pkg/core.py",
            "from .helper import helper\n\n\ndef frobnicate(value):\n    return helper(value)\n",
        );
        write("pkg/helper.py", "def helper(value):\n    return value\n");
        write(
            "tests/test_core.py",
            "from pkg.core import frobnicate\n\n\ndef test_frobnicate():\n    assert frobnicate(1) == 1\n",
        );

        let issue = "`frobnicate` raises on a str.\n\nTraceback (most recent call last):\n  File \"pkg/core.py\", line 5, in frobnicate\n    return helper(value)\nValueError: bad value\n";
        let touched = BTreeSet::from(["pkg/core.py".to_string()]);

        // The whole entry, body included, so a hidden test cannot move a visible
        // file even by changing what its outline would be.
        type Entry = (String, i64, Vec<SelectionReason>, usize, u32, ShownMode, String);
        let entries = |selection: &ContextSelection| -> Vec<Entry> {
            selection
                .files
                .iter()
                .filter(|file| file.path != HIDDEN)
                .map(|file| {
                    (
                        file.path.clone(),
                        file.score,
                        file.reasons.clone(),
                        file.bytes,
                        file.tokens,
                        file.shown,
                        file.body.clone(),
                    )
                })
                .collect()
        };
        let select = || {
            let (candidates, _skipped) = walk_candidates(tmp.path());
            (
                select_repair_context(&candidates, issue, "repair", 60_000),
                select_review_context(&candidates, &touched, "review", 60_000),
            )
        };

        let (repair_alone, review_alone) = select();
        assert!(
            entries(&repair_alone)
                .iter()
                .any(|entry| entry.0 == "pkg/core.py"),
            "the bug's own file is found from the issue before anything is added"
        );

        // ADDED: the hidden test now imports the touched module, so in review
        // mode it IS a candidate — which is exactly why the harness has to strip
        // it before the run sees the tree.
        write(
            HIDDEN,
            "from pkg.core import frobnicate\n\n\ndef test_hidden_bug():\n    assert frobnicate(2) == 2\n",
        );
        let (repair_with, review_with) = select();
        assert!(
            review_with.files.iter().any(|file| file.path == HIDDEN),
            "the scenario is only meaningful if the hidden test WOULD be shown"
        );

        // REMOVED again — the tree the run actually gets.
        fs::remove_file(tmp.path().join(HIDDEN)).unwrap();
        let (repair_after, review_after) = select();

        assert_eq!(entries(&repair_alone), entries(&repair_with));
        assert_eq!(entries(&repair_alone), entries(&repair_after));
        assert_eq!(entries(&review_alone), entries(&review_with));
        assert_eq!(entries(&review_alone), entries(&review_after));
        assert!(entries(&review_alone)
            .iter()
            .any(|entry| entry.0 == "tests/test_core.py"));
    }

    /// [VANTA-WHY] §1.6.2 item 4 — the provenance line is the artifact a report
    /// reads back, and it is embedded in `run.summary` and in a session record,
    /// both of which are parsed as space-separated `key=value` pairs. A space in
    /// a value would split a key and a newline would split a row, so both are
    /// pinned here; so is the content the item asks for — every selected file,
    /// with its score, its reason(s) and the bytes and tokens it was shown at.
    #[test]
    fn the_provenance_line_is_one_space_free_line_of_every_selected_file() {
        let candidates = vec![
            candidate(
                "pkg/core.py",
                "from .helper import helper\n\n\ndef frobnicate(value):\n    return helper(value)\n",
            ),
            candidate("pkg/helper.py", "def helper(value):\n    return value\n"),
        ];
        let issue = "`frobnicate` raises on a str.\n\nTraceback (most recent call last):\n  File \"pkg/core.py\", line 5, in frobnicate\n    return helper(value)\nValueError: bad value\n";
        let selection = select_repair_context(&candidates, issue, "creator+attacker", 60_000);
        let line = selection.provenance_line();

        assert!(!line.contains('\n'), "the line is embedded in a one-line row");
        assert!(
            !line.contains(' '),
            "no space: the row is space-separated key=value pairs; got {line}"
        );
        assert!(
            line.starts_with("mode=repair:role=creator+attacker:budget=60000:files=2:"),
            "the header names what the selection was for; got {line}"
        );

        for (path, score, reasons) in [
            ("pkg/core.py", 900, "path+traceback+identifier"),
            ("pkg/helper.py", 100, "import-neighbour"),
        ] {
            let file = selection
                .files
                .iter()
                .find(|file| file.path == path)
                .expect("both files are selected");
            let entry = format!(
                "{}@{}[{}]full:{}B/{}tok",
                path, score, reasons, file.bytes, file.tokens
            );
            assert!(
                line.contains(&entry),
                "each selected file travels with score, reason, bytes and tokens; missing {entry} in {line}"
            );
        }
    }

    /// [VANTA-WHY] §1.6.2 item 6 — the ranking is the whole point of the change,
    /// so it is pinned as a unit, on hand-built candidates: a file scores the SUM
    /// of the reasons it was picked for, a stronger reason is never displaced by a
    /// weaker one, and a tie is broken by path so the order is total and stable.
    /// The integration tests below exercise the same code through `select_for_role`
    /// on a real tree.
    #[test]
    fn the_ranking_orders_by_reason_weight_and_breaks_ties_by_path() {
        let candidates = vec![
            candidate(
                "pkg/core.py",
                "from .helper import helper\n\n\ndef frobnicate(value):\n    return helper(value)\n",
            ),
            candidate("pkg/helper.py", "def helper(value):\n    return value\n"),
            candidate("pkg/alpha.py", "def render():\n    return 1\n"),
            candidate("pkg/beta.py", "def render():\n    return 2\n"),
            candidate("pkg/unrelated.py", "def nothing():\n    return 3\n"),
        ];
        // `pkg/core.py` is named by path, named again by the traceback frame, and
        // defines the symbol the issue names — the strongest possible file.
        let issue = "`frobnicate` in pkg/core.py raises, and so does `render`.\n\n\
                     Traceback (most recent call last):\n  \
                     File \"pkg/core.py\", line 4, in frobnicate\n    \
                     return helper(value)\nValueError: bad value\n";

        let selection = select_repair_context(&candidates, issue, "creator", 60_000);

        assert_eq!(
            paths(&selection),
            vec!["pkg/core.py", "pkg/alpha.py", "pkg/beta.py", "pkg/helper.py"],
            "path+traceback+identifier (900) > identifier (200) > import-neighbour (100); \
             the identifier tie is broken by path"
        );
        let scores: Vec<i64> = selection.files.iter().map(|file| file.score).collect();
        assert_eq!(scores, vec![900, 200, 200, 100]);

        let core = &selection.files[0];
        assert_eq!(
            core.reasons,
            vec![
                SelectionReason::Path,
                SelectionReason::Traceback,
                SelectionReason::Identifier,
            ]
        );
        assert_eq!(
            selection
                .files
                .iter()
                .find(|file| file.path == "pkg/helper.py")
                .unwrap()
                .reasons,
            vec![SelectionReason::ImportNeighbour],
            "the neighbour of a named file scores below every file the issue names"
        );
        assert!(
            !paths(&selection).contains(&"pkg/unrelated.py"),
            "a file no reason reaches is not selected at all"
        );
    }

    /// [VANTA-WHY] The gate passed 7 of 11 dev bugs for the wrong reason: their
    /// issue text names no path, no traceback frame and no module-level symbol, so
    /// the ranking reached no file, the selection was empty, and the prompt was the
    /// repository file tree alone — `selected_bytes == tree_bytes` exactly. A
    /// listing always fits, so every one of them "passed" a check about a context
    /// none of them had. An empty ranking must fall back to the tree's own source,
    /// in walk order, under its own reason, and the fallback must not report the
    /// files it could not fit as *drops* (the ranking did not choose them; the walk
    /// did) — otherwise a repo larger than the window would be rejected for having
    /// a big tree rather than for having no context.
    #[test]
    fn a_ranking_that_reaches_no_file_falls_back_to_the_trees_own_source() {
        let candidates = vec![
            // In walk order: `walk_candidates` hands the list over sorted by path.
            candidate("pkg/alpha.py", "def a():\n    return 2\n"),
            candidate("pkg/mid.py", &"x = 1\n".repeat(4_000)),
            candidate("pkg/zeta.py", "def z():\n    return 1\n"),
        ];
        // One sentence of prose naming nothing the tree contains: no path, no
        // traceback frame, no `def`/`class` name.
        let issue = "Fix calling method on the wrong object.\n";

        // A budget that fits the two small files and not the big one whole.
        let selection = select_repair_context(&candidates, issue, "repair", 300);

        assert_eq!(
            paths(&selection),
            vec!["pkg/alpha.py", "pkg/mid.py", "pkg/zeta.py"],
            "the fallback shows the tree's own files in walk (path) order"
        );
        let scores: Vec<i64> = selection.files.iter().map(|file| file.score).collect();
        assert_eq!(scores, vec![10, 10, 10], "the fallback carries its own weight");
        assert!(selection
            .files
            .iter()
            .all(|file| file.reasons == vec![SelectionReason::Tree]));
        let big = selection
            .files
            .iter()
            .find(|file| file.path == "pkg/mid.py")
            .unwrap();
        assert_eq!(
            big.shown,
            ShownMode::Outline,
            "a fallback file too big to fit whole goes in as an outline, not out"
        );
        assert!(big.bytes < 24_000);
        assert!(
            selection.total_bytes > selection.tree_bytes,
            "the selection is not the tree listing: it carries source bytes"
        );

        // A budget the tree listing has already spent: the fallback shows nothing,
        // and still records nothing as a *drop* — the walk did not choose those
        // files, so a big repository must not be rejected for being big.
        let starved = select_repair_context(&candidates, issue, "repair", 5);
        assert!(starved.files.is_empty());
        assert!(
            starved.dropped_for_budget.is_empty(),
            "a file the fallback could not fit is not a file the ranking dropped"
        );

        // The same tree with an issue that DOES name a file is unaffected: the
        // fallback is only what an empty ranking falls back to.
        let named = select_repair_context(
            &candidates,
            "`a` fails in pkg/alpha.py\n",
            "repair",
            60_000,
        );
        assert_eq!(paths(&named), vec!["pkg/alpha.py"]);
        assert_eq!(named.files[0].reasons, vec![SelectionReason::Path]);
    }

    /// [VANTA-WHY] §1.6.2 item 6 — the budget fill is what makes the cap safe. A
    /// small file goes in whole; a file over `MAX_FILE_BYTES` goes in as an
    /// outline rather than being skipped, so a big touched file is never silently
    /// dropped; and a file whose outline still does not fit is *reported*, not
    /// forgotten. Nothing here executes Python — the outline is line-based.
    #[test]
    fn the_budget_fill_shows_small_files_whole_and_big_files_as_an_outline() {
        let big = format!(
            "def entry():\n    \"\"\"Doc.\"\"\"\n    return 1\n# {}\n",
            "x".repeat(MAX_FILE_BYTES + 1)
        );
        let candidates = vec![
            candidate("pkg/small.py", "def small():\n    return 2\n"),
            candidate("pkg/big.py", &big),
        ];
        // Both are named by path, so both are ranked; the budget is generous.
        let issue = "pkg/small.py and pkg/big.py both matter.\n";

        let selection = select_repair_context(&candidates, issue, "creator", 200_000);

        let small = selection
            .files
            .iter()
            .find(|file| file.path == "pkg/small.py")
            .expect("a file that fits whole is selected");
        assert_eq!(small.shown, ShownMode::Full);
        assert_eq!(small.body, "def small():\n    return 2\n");

        let big_file = selection
            .files
            .iter()
            .find(|file| file.path == "pkg/big.py")
            .expect("a file over the per-file cap is shown as an outline, never dropped");
        assert_eq!(big_file.shown, ShownMode::Outline);
        assert!(
            big_file.bytes < big.len(),
            "the outline is smaller than the source it stands in for ({} vs {})",
            big_file.bytes,
            big.len()
        );
        assert!(big_file.body.contains("def entry():"));
        assert!(selection.dropped_for_budget.is_empty());
    }

    /// [VANTA-WHY] §1.6.2 item 6 — a budget too small even for an outline must not
    /// silently lose a ranked file: the caller has to be able to say "the run was
    /// shown less than the ranking chose", which is what the gate's
    /// `selection_dropped_for_budget` verdict reads.
    #[test]
    fn the_budget_fill_reports_a_file_it_had_to_drop_instead_of_hiding_it() {
        let candidates = vec![candidate(
            "pkg/core.py",
            "def frobnicate(value):\n    return value\n",
        )];
        let issue = "pkg/core.py is broken.\n";

        // The file tree alone consumes more than this, so nothing else can fit.
        let selection = select_repair_context(&candidates, issue, "creator", 8);

        assert!(
            selection.files.is_empty(),
            "nothing fits a budget the file tree has already spent"
        );
        assert_eq!(
            selection.dropped_for_budget,
            vec!["pkg/core.py"],
            "the dropped file is named, so the gate can refuse the bug for the right reason"
        );
    }

    /// [VANTA-WHY] §1.6.2 item 6 — the end-to-end repair case, through the entry
    /// point the `--select-context` subcommand calls. A real tree is walked and the
    /// bug's own file is found from a traceback in the issue text, with the repo
    /// file tree carried alongside so the model sees what exists.
    #[test]
    fn the_buggy_file_is_found_from_a_traceback_in_the_issue() {
        let tmp = tempfile::TempDir::new().unwrap();
        let write = |rel: &str, contents: &str| {
            let path = tmp.path().join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        };
        write("pkg/parser.py", "def parse(value):\n    return value\n");
        write("pkg/other.py", "def other():\n    return 1\n");
        write("tests/test_parser.py", "def test_parse():\n    assert True\n");

        let issue = "Parsing blows up.\n\nTraceback (most recent call last):\n  \
                     File \"pkg/parser.py\", line 2, in parse\n    \
                     return value\nTypeError: bad\n";

        let selection = select_for_role(tmp.path(), issue, None, "creator", 60_000);

        assert_eq!(selection.mode, "repair");
        assert_eq!(paths(&selection), vec!["pkg/parser.py"]);
        let file = &selection.files[0];
        assert!(file.reasons.contains(&SelectionReason::Path));
        assert!(
            file.reasons.contains(&SelectionReason::Traceback),
            "the traceback frame is what makes this a localization and not a guess"
        );
        assert!(
            selection.tree.contains(&"pkg/parser.py".to_string()),
            "the repo file tree travels with the selection"
        );
        let rendered = selection.render();
        assert!(rendered.contains("# REPOSITORY FILES (paths only)"));
        assert!(rendered.contains("# FILE: pkg/parser.py"));
    }

    /// [VANTA-WHY] §1.6.2 item 6 — the end-to-end review case: a real patch is
    /// parsed into its touched files, and the selection is those files plus their
    /// one-hop imports and the visible tests that cover them. This goes through
    /// `select_for_role` (the `--select-context` entry point) rather than
    /// `select_review_context`, so the patch-splitting seam is exercised too.
    #[test]
    fn review_selection_follows_a_real_patch_to_its_imports_and_tests() {
        let tmp = tempfile::TempDir::new().unwrap();
        let write = |rel: &str, contents: &str| {
            let path = tmp.path().join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, contents).unwrap();
        };
        write(
            "pkg/core.py",
            "from .helper import helper\n\n\ndef frobnicate(value):\n    return helper(value)\n",
        );
        write("pkg/helper.py", "def helper(value):\n    return value\n");
        write("pkg/unrelated.py", "def unrelated():\n    return 1\n");
        write(
            "tests/test_core.py",
            "from pkg.core import frobnicate\n\n\ndef test_frobnicate():\n    assert frobnicate(1) == 1\n",
        );
        write(
            "tests/test_unrelated.py",
            "from pkg.unrelated import unrelated\n\n\ndef test_unrelated():\n    assert unrelated() == 1\n",
        );

        let patch = "diff --git a/pkg/core.py b/pkg/core.py\n\
                     --- a/pkg/core.py\n\
                     +++ b/pkg/core.py\n\
                     @@ -1,4 +1,4 @@\n\
                     -def frobnicate(value):\n\
                     +def frobnicate(value, *, strict=False):\n";

        let selection = select_for_role(tmp.path(), "", Some(patch), "creator+attacker", 60_000);

        assert_eq!(selection.mode, "review");
        assert!(
            selection.tree.is_empty(),
            "review mode has no file tree: the patch already localizes the repo"
        );
        assert_eq!(
            paths(&selection)[0],
            "pkg/core.py",
            "the patch's own file outranks its neighbours and the tests that cover it"
        );
        let touched = &selection.files[0];
        assert_eq!(touched.reasons, vec![SelectionReason::PatchTouched]);
        assert_eq!(touched.shown, ShownMode::Full);

        let shown = paths(&selection);
        assert!(
            shown.contains(&"pkg/helper.py"),
            "the touched file's module-level import comes along, one hop out"
        );
        assert!(
            shown.contains(&"tests/test_core.py"),
            "the visible test that covers the touched module is shown"
        );
        assert!(
            !shown.contains(&"pkg/unrelated.py") && !shown.contains(&"tests/test_unrelated.py"),
            "files the patch does not reach stay out of the review prompt"
        );
    }
}
