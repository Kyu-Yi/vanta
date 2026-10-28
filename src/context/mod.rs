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
//      mode: the issue text), and carry a BODY only for the project's own source
//      or a file the run itself named — everything else a signal reached travels
//      as a path (§1.6.8).
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
/// What the walk can *see*: a file whose extension is not here is not a
/// candidate at all, whichever signal would otherwise have reached it.
///
/// [VANTA-WHY] This answers "which paths exist", not "what may be shown as code"
/// — the second question is `is_project_source` (§1.6.8), and keeping the two
/// apart is what lets a `.pyx` be walked without a repo's own `docs/*.js` being
/// shown in full.
pub const SOURCE_EXTENSIONS: &[&str] = &[
    "py", "pyx", "pxd", "pyi", "rs", "go", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h", "hpp",
];

/// The extensions whose files are the project's own source, and therefore carry a
/// body without needing a signal to name them (§1.6.8).
///
/// [VANTA-WHY] `pyx`/`pxd` are here because pandas and spacy have real bugs in
/// Cython files — `pandas/_libs/algos.pyx` among them — and a `.pxd` is the
/// declaration half of the same code. `pyi` is Python the project wrote, and
/// `defined_names` already parses it by the Python rules; before this it
/// was not walked, so that branch was dead on every real tree.
pub const PROJECT_SOURCE_EXTENSIONS: &[&str] = &["py", "pyx", "pxd", "pyi"];

/// Directory names whose files are the project's *paratext* — its examples, its
/// documentation and its benchmarks — rather than the code a bug is fixed in
/// (§1.6.8 item 4).
///
/// [VANTA-WHY] A `.py` example is Python, so the extension rule alone showed
/// `sanic/1`'s `examples/add_task_sanic.py` in full at rank 4: the issue's own
/// symbol was declared in it — or, before the fence fix, merely appeared in it —
/// and an identifier is a name, not a place. §1.6.8 already refuses a body to a
/// `.js` class the issue names, for the same reason: being *reachable* from a
/// name is not being *named*. A demonstrator, a doc build and a `asv` harness
/// cannot hold the bug either, so they get the path the listing already gives
/// them and nothing more. A file here is still shown in full when the run names
/// its *path* — the same escape every non-source file has.
pub const PARATEXT_DIRECTORIES: &[&str] = &[
    "example",
    "examples",
    "doc",
    "docs",
    "benchmark",
    "benchmarks",
    "asv_bench",
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

/// What the whole prompt costs: the selection, the issue text, and the frame
/// around them.
///
/// [VANTA-WHY] One definition, because the gate's `prompt_exceeds_budget` and the
/// run's own guard must sit on the same number, and `--select-context` is what the
/// gate reads. The listing is **not** added here: `ContextSelection::total_tokens`
/// is already `files + tree` (§1.6.4), so adding `tree_tokens` back — as this
/// arithmetic used to — counted one listing twice. On the corpus's larger trees
/// that is on the order of ten thousand tokens, which is not a rounding
/// difference: it is a large part of the margin that decides an over-budget
/// verdict, so it was rejecting selections that fit.
pub fn prompt_tokens_estimate(selection: &ContextSelection, issue_tokens: u32) -> u32 {
    selection
        .total_tokens
        .saturating_add(issue_tokens)
        .saturating_add(PROMPT_FRAME_TOKENS)
}

/// The prompt the selection does *not* cover: the issue text and the frame
/// around it.
///
/// [VANTA-WHY] `fill` subtracts this — and the file-tree listing — from the
/// budget before it prices the first file, so `prompt_tokens_estimate`, which
/// adds exactly these two back, can never land above `budget_tokens`.
/// Reserving the listing alone (as the fill used to) left the issue text and the
/// frame unbudgeted, and a selection that spent everything it was handed then
/// overshot the whole-prompt budget by `issue_tokens + PROMPT_FRAME_TOKENS`
/// (§1.6.9). The same `estimate_tokens` the gate uses prices it, so the gate's
/// `prompt_exceeds_budget` and the fill's own arithmetic sit on one number.
pub fn reserved_prompt_tokens(issue_tokens: u32) -> u32 {
    issue_tokens.saturating_add(PROMPT_FRAME_TOKENS)
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
/// subject, so nothing outranks the files it touches.
///
/// [VANTA-WHY] There is deliberately no "it is in the tree" member. Selection
/// *content* is only ever a scored signal — a path or frame the issue names, a
/// symbol the issue names, a neighbour of one of those, a file the patch touches,
/// a test that exercises one of those. The repository file tree is still sent, as
/// listing of paths and nothing more, so a file that no signal reached is present
/// in the prompt as a line and absent from it as source. A variant meaning
/// "walked to it" was tried and removed: it made a prompt out of the listing while
/// looking exactly like a grounded selection in the report (§1.6.6).
///
/// [VANTA-WHY] There is deliberately no "this file mentions the name" member
/// either. A definition is evidence that the issue's own symbol lives here; a
/// mention is the string appearing in a file whether or not it is the symbol's
/// home, and on a real tree a name like `True` or `Series` is mentioned by most
/// of the repo — scoring mentions as a reason selects hundreds of files at the
/// bottom of the ranking and spends the budget on the noise (item 4's own
/// complaint, at repo scale). Mentions are not free, but they are *graded*: what
/// orders files by their contents is the BM25 fallback, and it is priced below
/// every signal here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionReason {
    /// A file the patch under review touches (review mode).
    PatchTouched,
    /// A path or module named explicitly in the issue text (repair mode).
    Path,
    /// A frame named in a traceback inside the issue text (repair mode).
    Traceback,
    /// A name this file *defines* — a `def`/`class` at any depth, or a
    /// module-level binding — that the issue text writes as code (`defined_names`,
    /// §1.6.2 item 2). Its weight is scaled by how rare the name is across the
    /// repo (`identifier_worth`), and never falls to a neighbour's.
    Identifier,
    /// A module-level import neighbour of a selected file, one hop out.
    ImportNeighbour,
    /// A visible test file that exercises a selected module (review mode).
    Test,
    /// The BM25 fallback: a file no stronger signal reached, ordered by how well
    /// its contents match the code-shaped tokens of the issue text (§1.6.2 item
    /// 3). The lowest-weight signal there is, and the only one that reaches an
    /// issue that names no *definition* — `fastapi/1` names
    /// `response_model_exclude_defaults`, a parameter, which no `def`/`class`/
    /// binding declares, so the names-based signals all come up empty and the
    /// ranking has to fall back to the text itself.
    Bm25,
}

/// What an `Identifier` match is worth at its utmost: a name no other candidate
/// in the repo defines or mentions.
pub const IDENTIFIER_WEIGHT: i64 = 200;
/// What an `Identifier` match is worth at its least: a name nearly every
/// candidate mentions, which is a word the repo shares rather than evidence.
///
/// [VANTA-WHY] The floor sits above `ImportNeighbour` (100) on purpose. §1.6.2
/// item 2 orders a file the *issue* names above a file merely reachable from one,
/// and that guarantee is a property of the weights, not of the corpus: a scaling
/// that could price a common name's definition below 100 would let a neighbour of
/// a named file displace the file the issue named — which is exactly the ordering
/// the weights were re-cut to fix. The band is [120, 200]; the rare end keeps the
/// historical 200.
pub const IDENTIFIER_WEIGHT_FLOOR: i64 = 120;

/// What a `Bm25` match is worth at its most (§1.6.2 item 3).
///
/// [VANTA-WHY] The band's top is below `Test` (80), the weakest signal that
/// reaches a file for something the *run named* rather than for the words it
/// happens to share with the issue. A file the text-only signal reached must
/// therefore never outrank one a real signal reached, whatever the corpus — the
/// same guarantee `IDENTIFIER_WEIGHT_FLOOR` states one rung up. The signal is
/// also recorded only for a file no stronger signal reached, so the two rules
/// agree and neither has to be relied on alone: BM25 orders the files nothing
/// else reached, and it cannot displace a file something did.
pub const BM25_WEIGHT: i64 = 40;
/// BM25's term-frequency saturation, the standard `k1`.
const BM25_K1: f64 = 1.2;
/// BM25's document-length normalization, the standard `b`.
const BM25_B: f64 = 0.75;

impl SelectionReason {
    pub fn weight(self) -> i64 {
        match self {
            SelectionReason::PatchTouched => 1000,
            SelectionReason::Path => 400,
            SelectionReason::Traceback => 300,
            SelectionReason::Identifier => IDENTIFIER_WEIGHT,
            SelectionReason::ImportNeighbour => 100,
            SelectionReason::Test => 80,
            SelectionReason::Bm25 => BM25_WEIGHT,
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
            SelectionReason::Bm25 => "bm25",
        }
    }

    /// Whether this reason says the run's inputs named *this file* — its path,
    /// its module, or a frame that points at it — which can be true of a file that
    /// is not the project's own source.
    ///
    /// [VANTA-WHY] §1.6.8 lets a body into the prompt for the project's source
    /// alone, plus any file the run itself named. "Named" is narrower than
    /// `grounds_content` below, and deliberately: being *reachable* from a named
    /// file — an import neighbour, a covering test — is not being named, so a
    /// neighbour of a `.js` the issue mentions arrives as its path and not as its
    /// code. `PatchTouched` is in the list because in review mode the patch is the
    /// subject: a hunk in a file the walk lists is content whatever its
    /// extension.
    pub fn names_the_file(self) -> bool {
        matches!(
            self,
            SelectionReason::PatchTouched | SelectionReason::Path | SelectionReason::Traceback
        )
    }

    /// Whether this reason grounds the *content* of a selection in something the
    /// run was given — the issue text, or the patch under review.
    ///
    /// [VANTA-WHY] The harness's gate refuses a role whose selection has no
    /// grounded file ("no issue-grounded selection", §1.6.2 item 5). That check
    /// lives here rather than in the harness so the two cannot disagree about
    /// what grounded means: the gate reads the reason vocabulary this enum
    /// serializes, and this is the predicate that classifies it. Every variant is
    /// grounded today; the predicate is the seam a future reason has to pass, and
    /// it is what makes the refusal a property of the selector rather than of a
    /// string the harness happens to recognize.
    ///
    /// `Bm25` is grounded: its score is computed from the issue text's own
    /// code-shaped tokens against the file's contents, so the file is reached by
    /// what the *report* said, not by its place in the walk. It is the one
    /// grounded reason a file may carry without the run having named it, which is
    /// why it is not in `names_the_file` — the issue's words are evidence of the
    /// subject, not of the file.
    pub fn grounds_content(self) -> bool {
        matches!(
            self,
            SelectionReason::PatchTouched
                | SelectionReason::Path
                | SelectionReason::Traceback
                | SelectionReason::Identifier
                | SelectionReason::ImportNeighbour
                | SelectionReason::Test
                | SelectionReason::Bm25
        )
    }
}

/// HOW each reason matched (§1.6.4) — the rule beside the signal.
///
/// [VANTA-WHY] A reason name alone says which input named a file, not which of
/// the rules inside that signal did. `path` covers five of them here, and the
/// difference is the whole question a reader of the ranking asks: `pkg/app.py`
/// named outright is a strong signal, `app.py` matched as a bare basename could
/// be a coincidence in another directory, and a plain English word that reached
/// a file only because it was written in backticks is weaker than a CamelCase
/// name that is code on its face. Recording the rule is what lets
/// `localization_table` show why a file ranked instead of asserting it did
/// (§1.6.5) — and it is display only: no weight, and no gate, reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SelectionRule {
    /// A hunk of the patch under review names the file.
    Hunk,
    /// The token IS the path.
    Exact,
    /// The token is the path's basename.
    Basename,
    /// The token is the file's dotted module name.
    Module,
    /// The token is a GitHub blob URL naming the path.
    BlobUrl,
    /// The token ends with the path or its basename (an absolute or longer path).
    Tail,
    /// A `def`/`class` name, or a module-level binding, with an underscore.
    Snake,
    /// The same, containing a digit.
    Digit,
    /// The same, CamelCase: a leading capital and a lower after it.
    Camel,
    /// A name written between backticks, whatever its shape.
    Backtick,
    /// A name written as a call: `name(`.
    Call,
    /// A name written as an attribute: `obj.name` — the name after the dot.
    Attribute,
    /// A module the file imports.
    Import,
    /// A visible test file that imports or covers a selected module.
    CoveringTest,
    /// The file's contents match the code-shaped tokens of the issue (BM25).
    Bm25,
}

impl SelectionRule {
    pub fn as_str(self) -> &'static str {
        match self {
            SelectionRule::Hunk => "hunk",
            SelectionRule::Exact => "exact",
            SelectionRule::Basename => "basename",
            SelectionRule::Module => "module",
            SelectionRule::BlobUrl => "blob-url",
            SelectionRule::Tail => "tail",
            SelectionRule::Snake => "snake",
            SelectionRule::Digit => "digit",
            SelectionRule::Camel => "camel",
            SelectionRule::Backtick => "backtick",
            SelectionRule::Call => "call",
            SelectionRule::Attribute => "attribute",
            SelectionRule::Import => "import",
            SelectionRule::CoveringTest => "covering-test",
            SelectionRule::Bm25 => "bm25",
        }
    }
}

/// One signal that matched one file, with what it was worth *for that file*.
///
/// [VANTA-WHY] The rule is display only (§1.6.4), but the weight is not: a name
/// an issue writes as code is worth less when the whole repo mentions it, so an
/// `Identifier` match is priced by how rare the name is across the candidates
/// (§1.6.2 item 2). The fixed signals keep `SelectionReason::weight` exactly;
/// only the identifier match scales, which is why the score lives beside the
/// match rather than being re-derived from the reason alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionMatch {
    pub rule: SelectionRule,
    /// This signal's contribution to the file's score.
    pub weight: i64,
}

impl SelectionMatch {
    /// A signal priced by its reason's own constant — every reason but
    /// `Identifier` and `Bm25`, whose worth is derived from the corpus.
    pub fn of(reason: SelectionReason, rule: SelectionRule) -> SelectionMatch {
        SelectionMatch {
            rule,
            weight: reason.weight(),
        }
    }
}

/// The form a selected file is shown in: its source, its outline, or its path
/// alone (§1.6.9).
///
/// The three are a ladder, and they are ordered: `Full` is the whole file,
/// `Outline` its signatures and docstrings, `Path` its `# FILE:` header and
/// nothing else. `Ord` is derived from that order so the fill can say "no file
/// below this one gets a richer form than this" as one comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShownMode {
    Path,
    Outline,
    Full,
}

impl ShownMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ShownMode::Full => "full",
            ShownMode::Outline => "outline",
            ShownMode::Path => "path",
        }
    }

    /// Whether this form carries the file's text at all — false only for `Path`,
    /// which is the header and nothing else (§1.6.9).
    ///
    /// The gate's question is whether a role has any of the project's *content*
    /// grounded in the run's inputs, and a file that travels as a path has none
    /// (§1.6.10). A path-only entry is a selected file all the same — it is what
    /// the ranking decided, and it is recorded — but it is not a body.
    pub fn carries_content(self) -> bool {
        self != ShownMode::Path
    }
}

/// One file in the selection, with the provenance a reader needs to explain it.
#[derive(Debug, Clone, Serialize)]
pub struct SelectedFile {
    pub path: String,
    /// The file's 1-based rank in the selection — its place in score order
    /// (§1.6.9).
    ///
    /// The array is already in that order, so this is a second statement of the
    /// same fact; it is recorded because the report is read on its own, by
    /// analyses that sort the list their own way, and a rank that had to be
    /// re-derived from the array would be re-derived differently by each of them.
    /// `rank == index + 1` is an invariant of the record, not a convention.
    pub rank: usize,
    pub score: i64,
    pub reasons: Vec<SelectionReason>,
    /// Per signal, the rule that matched it, keyed by the reason's own name —
    /// `{"path": "exact", "identifier": "camel"}`. Same signals as `reasons`, and
    /// a key for each of them; `SelectionRule` says why the two travel apart.
    pub rules: BTreeMap<String, String>,
    /// Bytes actually shown: the framed text's bytes, which for `Outline` and
    /// `Path` is the outline or the header rather than the source.
    pub bytes: usize,
    pub tokens: u32,
    pub shown: ShownMode,
    /// The content shown for this file, WITHOUT the `# FILE: <path>` header: the
    /// full source, the outline, or — for `Path` — the one-line marker that says
    /// the file is here and its body is not (§1.6.9).
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

/// One candidate the ranking chose and the budget could not fit — not even as a
/// bare path (§1.6.2 item 6, M3 item 2).
///
/// [VANTA-WHY] The size travels with the path because "how many files were
/// dropped" is not the question a row has to answer. A selection that had to drop
/// its *tail* is sanctioned, provided the file the ranking put first went in full
/// (M3 item 2); what a reader then needs to know is how much *source* the budget
/// left out, and this is the only place that knows it: the candidate was read to
/// be ranked, and the harness never walks the tree (§1.6.2 item 5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DroppedFile {
    pub path: String,
    /// The file's own source bytes — what its body would have cost had it fitted.
    /// It is the source's size and not a framed length, because nothing was framed:
    /// the file was dropped before any form fitted it.
    pub bytes: usize,
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
    /// Candidates the ranking chose and the budget could not fit even as a path,
    /// each with the source size it would have cost (M3 item 2).
    pub dropped_for_budget: Vec<DroppedFile>,
    pub skipped: Vec<SkipRecord>,
}

impl ContextSelection {
    /// Whether the selection's content is grounded in the run's own inputs — the
    /// question the gate's `no_issue_grounded_selection` asks (§1.6.10, §1.6.9).
    ///
    /// [VANTA-WHY] The answer is computed here, from the reasons, and travels in
    /// the report, so the harness refuses a role without holding its own copy of
    /// what "grounded" means. A reason that named a file without grounding its
    /// body would be refused by a gate that never learned the reason's name — the
    /// same reason `context.py` shells out to this selector instead of walking a
    /// tree itself (§1.6.2 item 5).
    ///
    /// A file shown as a path alone is not content, so it does not ground the
    /// selection: the question is whether the role has any of the project's *text*,
    /// and a selection that spent its whole budget on the listing and the tail's
    /// headers has none of it. That is the same refusal as a listing with no files
    /// at all, which is what it is.
    pub fn grounded(&self) -> bool {
        self.files.iter().any(|file| {
            file.shown.carries_content()
                && file.reasons.iter().any(|reason| reason.grounds_content())
        })
    }

    /// The prompt text: the file tree (when present), then each file in the form
    /// it got — full, outline or path only — in score order.
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
// Budget fill — greedy by score, three forms, never over budget
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

/// The body of a file shown as its path alone (§1.6.9): one marker line, no
/// source at all.
///
/// [VANTA-WHY] The marker is not decoration. A `# FILE:` header with nothing
/// under it reads as a file that is *empty*, and a model that believes a module
/// is empty will reason about the bug as if its code did not exist. The line says
/// what happened — the budget ran out, not the file — and it is the whole reason
/// the form costs anything at all.
fn path_only_body() -> String {
    "# [PATH ONLY — the budget ran out before this file's source could be shown]\n".to_string()
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
/// `issue_tokens` is the token cost of the issue text this prompt carries. It is
/// not a *ranking* input — nothing here reads the issue's words — it is the part
/// of the whole prompt the selection must leave room for (§1.6.9), exactly as the
/// gate counts it.
pub fn select_review_context(
    candidates: &[Candidate],
    touched: &BTreeSet<String>,
    role: &str,
    budget_tokens: u32,
    issue_tokens: u32,
) -> ContextSelection {
    let by_path: BTreeMap<&str, &Candidate> = candidates
        .iter()
        .map(|candidate| (candidate.path.as_str(), candidate))
        .collect();
    let module_map = python_module_to_path(candidates);

    let mut reasons: BTreeMap<String, BTreeMap<SelectionReason, SelectionMatch>> = BTreeMap::new();
    let mut mark = |path: &str, reason: SelectionReason, rule: SelectionRule| {
        reasons
            .entry(path.to_string())
            .or_default()
            .entry(reason)
            .or_insert(SelectionMatch::of(reason, rule));
    };

    for path in touched {
        if by_path.contains_key(path.as_str()) {
            mark(path, SelectionReason::PatchTouched, SelectionRule::Hunk);
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
            mark(&path, SelectionReason::ImportNeighbour, SelectionRule::Import);
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
            mark(&candidate.path, SelectionReason::Test, SelectionRule::CoveringTest);
        }
    }

    fill(
        "review",
        role,
        candidates,
        reasons,
        budget_tokens,
        &[],
        reserved_prompt_tokens(issue_tokens),
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
/// object` — reaches no file with any of the three reasons and has no seed for the
/// neighbour hop, so the selection is empty and `files` is empty with it. That is
/// the honest answer and the whole point of §1.6.6: the prompt is then the file
/// tree (paths only) and the issue text, nothing else, and the harness's gate
/// refuses the role as having no issue-grounded selection rather than a run
/// starting from a listing and guessing. BugsInPy/3 shows how thin these messages
/// are: its built text is `Fix #2` plus the kind template, and the selector
/// extracts no path, no frame and no symbol from it (§1.6.6's evidence table).
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

    let mut reasons: BTreeMap<String, BTreeMap<SelectionReason, SelectionMatch>> = BTreeMap::new();

    // What every candidate defines, read once. Two things need it: the per-file
    // `Identifier` match, and the document frequency that prices each name the
    // issue writes as code (§1.6.2 item 2).
    let defined: Vec<BTreeSet<String>> = candidates
        .iter()
        .map(|candidate| defined_names(&candidate.path, &candidate.contents))
        .collect();

    // What each name the issue writes as code is worth: a name one file of the
    // tree defines is the rarest there is, a name most of the tree defines is a
    // word the repo shares rather than evidence (§1.6.2 item 2). Computed once
    // per name, not once per (file, name): it is a property of the corpus, and
    // the corpus does not change while its files are being ranked.
    let rarity: BTreeMap<&str, i64> = identifiers
        .keys()
        .map(|symbol| {
            (
                symbol.as_str(),
                identifier_worth(identifier_df(symbol, &defined), defined.len()),
            )
        })
        .collect();

    for (candidate, names) in candidates.iter().zip(&defined) {
        let module = python_module_path(&candidate.path);
        if let Some(rule) = path_match_rule(&path_tokens, &candidate.path, module.as_deref()) {
            mark_reason(&mut reasons, &candidate.path, SelectionReason::Path, rule);
        }
        if let Some(rule) = path_match_rule(&frame_tokens, &candidate.path, module.as_deref()) {
            mark_reason(&mut reasons, &candidate.path, SelectionReason::Traceback, rule);
        }
        if !identifiers.is_empty() {
            // The first symbol in path order that the issue names, so the rule
            // recorded for the file is deterministic when it defines several.
            if let Some(symbol) = names.iter().find(|symbol| identifiers.contains_key(*symbol)) {
                let rule = identifiers[symbol];
                let weight = rarity[symbol.as_str()];
                mark_identifier(&mut reasons, &candidate.path, rule, weight);
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
            mark_reason(&mut reasons, &path, SelectionReason::ImportNeighbour, SelectionRule::Import);
        }
    }

    // The BM25 fallback (§1.6.2 item 3), for the issue that names no definition.
    // It is marked ONLY on a file no signal above reached: that is what makes it
    // a fallback rather than another signal — it orders the files the issue names
    // nothing in (`fastapi/1`'s `response_model_exclude_defaults` is a parameter,
    // so every names-based signal is empty), and it never re-prices or displaces a
    // file the report's own symbols already reached. `BM25_WEIGHT` (40) is below
    // every other reason's, so the guarantee holds even if that ever changed.
    //
    // [VANTA-WHY] This is not §1.6.3's deleted fallback. That one filled the tree
    // in walk order — a listing with the alphabet on top, ungrounded — and the
    // gate rightly refuses it. This one is reached from the issue text's own
    // tokens against the tree's contents, so the file it finds is grounded in what
    // the reporter wrote, which is what `grounds_content` asks.
    for (path, weight) in bm25_weights(candidates, &issue_code_tokens(issue_text)) {
        if !reasons.contains_key(&path) {
            mark_signal(&mut reasons, &path, SelectionReason::Bm25, SelectionRule::Bm25, weight);
        }
    }

    let tree: Vec<String> = candidates.iter().map(|c| c.path.clone()).collect();
    // The issue text is part of the prompt the model is sent, so its cost is
    // reserved out of the budget before the first file is priced (§1.6.9).
    let reserved = reserved_prompt_tokens(estimate_tokens(issue_text.len()));
    fill("repair", role, candidates, reasons, budget_tokens, &tree, reserved)
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
    // The issue text travels in both prompts, so both selections reserve it out
    // of the budget — same estimate `--select-context` reports to the gate.
    let issue_tokens = estimate_tokens(issue_text.len());
    let mut selection = match patch_text {
        Some(patch) => {
            let touched: BTreeSet<String> = crate::agents::creator::split_patch_into_files(patch)
                .keys()
                .cloned()
                .collect();
            select_review_context(&candidates, &touched, role, budget_tokens, issue_tokens)
        }
        None => select_repair_context(&candidates, issue_text, role, budget_tokens),
    };
    selection.skipped = skipped;
    selection
}

/// Record one reason for one path, with the rule that matched it and what the
/// match is worth for that file.
///
/// The FIRST match recorded for a (path, reason) pair is kept: each pair is
/// marked once per pass today, and where a signal could match by more than one
/// rule, the matcher itself returns the strongest (see `path_match_rule`), so
/// the record does not depend on the order candidates are visited.
fn mark_reason(
    reasons: &mut BTreeMap<String, BTreeMap<SelectionReason, SelectionMatch>>,
    path: &str,
    reason: SelectionReason,
    rule: SelectionRule,
) {
    mark_signal(reasons, path, reason, rule, reason.weight());
}

/// Record one signal for one path at an explicit weight.
///
/// Kept apart from `mark_reason` because two signals are priced by something
/// other than their reason's own constant: the identifier match, by how rare the
/// name is across the candidates (`identifier_worth`), and the BM25 fallback, by
/// how well the file's contents match the issue (`bm25_weights`). Both are
/// *derived*, so they are passed, not looked up, and every other signal goes
/// through `mark_reason` and cannot be priced by a number a caller invented.
fn mark_signal(
    reasons: &mut BTreeMap<String, BTreeMap<SelectionReason, SelectionMatch>>,
    path: &str,
    reason: SelectionReason,
    rule: SelectionRule,
    weight: i64,
) {
    reasons
        .entry(path.to_string())
        .or_default()
        .entry(reason)
        .or_insert(SelectionMatch { rule, weight });
}

/// Record an `Identifier` match with the weight that name is worth (§1.6.2 item
/// 2), which is the one signal whose weight is not the reason's own.
fn mark_identifier(
    reasons: &mut BTreeMap<String, BTreeMap<SelectionReason, SelectionMatch>>,
    path: &str,
    rule: SelectionRule,
    weight: i64,
) {
    mark_signal(reasons, path, SelectionReason::Identifier, rule, weight);
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

/// The names the issue writes as code, each with the rule that made it one — the
/// Rust half of the reader `scripts/bugsinpy/issue.py` implements for the paper
/// flag (§1.6.2 item 2, §2.4.6).
///
/// [VANTA-WHY] The rule is the whole point of this function, not a decoration.
/// The issue texts are prose: `sanic/1` says "the registry is never reset, and
/// the release drops the last row", and a reader that takes every word of that
/// as a name of the tree hands a file that happens to bind `release` a signal it
/// never earned. So a word counts only when the text writes it as code — shaped
/// like code (`snake`, `digit`, `camel`) or written as code (backticks, a call,
/// an attribute) — and `SelectionRule` records which of those it was, so the
/// file's rank can be read back as "matched because the issue backticked
/// `registry`" rather than as an unattributed score.
///
/// A dotted span contributes each segment on its own, and only the segment after
/// the dot counts as an attribute: in `fig.savefig(` it is `savefig` that the
/// reporter named, and `fig` is their own local variable.
///
/// A fenced code block is deliberately NOT one of the writing-as-code forms. The
/// reproduction snippet in `matplotlib/1`'s report is a ```python block whose
/// `fig`, `ax` and `data` are the reporter's own locals; treating the fence as
/// permission puts them back in the list this rule exists to keep honest. The
/// mechanism that makes this true is adjacency: a token is backticked when a
/// backtick touches it, and a fence's backticks touch only its first and last
/// lines. Counting backticks as a toggle instead made a three-backtick fence flip
/// the state and turned a whole block into one long `` ` `` span, which is how
/// `pandas/1`'s `dtype` — never backticked, never code-shaped — came to be one of
/// the names the issue was read as naming (§1.6.2 item 2).
pub fn issue_identifiers(issue_text: &str) -> BTreeMap<String, SelectionRule> {
    let mut out: BTreeMap<String, SelectionRule> = BTreeMap::new();
    let chars: Vec<char> = issue_text.chars().collect();
    let mut index = 0usize;
    // True for the token that starts right after a `.` whose own predecessor is
    // an identifier character — `obj.name`, never the `.` of a sentence.
    let mut after_dot = false;
    while index < chars.len() {
        let ch = chars[index];
        if !is_identifier_char(ch) {
            after_dot = ch == '.' && index > 0 && is_identifier_char(chars[index - 1]);
            index += 1;
            continue;
        }
        let start = index;
        while index < chars.len() && is_identifier_char(chars[index]) {
            index += 1;
        }
        let token: String = chars[start..index].iter().collect();
        let next = chars.get(index).copied().unwrap_or(' ');
        // The written forms are read by ADJACENCY, not by toggling a state
        // machine, because a fence is three backticks and a toggle reads them as
        // three flips. `sanic/1`'s reproduction is a ```python block: with a
        // toggle, everything inside it became `backtick` and the example files
        // that merely *mention* a name the block happens to contain were ranked
        // by `identifier` (§1.6.2 item 2, item 4). Adjacency is the reader the
        // flag uses (`issue.py`'s `_written_as_code`), so the two cannot disagree.
        let after_backtick = start > 0 && chars[start - 1] == '`';
        let rule = identifier_rule(&token, after_backtick, after_dot, next);
        after_dot = false;
        if let Some(rule) = rule {
            out.entry(token).or_insert(rule);
        }
    }
    out
}

fn is_identifier_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// The rule that makes `token` a name the issue wrote as code, or `None`.
///
/// Shape is tested before the written forms, the order the flag's reader tests
/// them in (``issue.py``'s ``is_code_shaped`` then ``_written_as_code``), so the
/// two record the same rule for the same word. The written forms are the flag's
/// three: a token touching a backtick (`` `name` ``), a token followed by an open
/// paren (``name(``), and the name side of an attribute access (``obj.name``,
/// read from `after_dot`). A token inside a fenced code block touches no backtick
/// — the fence's own backticks are on its first and last lines, not beside the
/// token — so a fence is not one of the forms, exactly as §1.6.2 says.
fn identifier_rule(
    token: &str,
    after_backtick: bool,
    after_dot: bool,
    next: char,
) -> Option<SelectionRule> {
    if !is_identifier(token) || token.len() < 2 {
        return None;
    }
    if let Some(rule) = code_shape_rule(token) {
        return Some(rule);
    }
    if after_backtick || next == '`' {
        return Some(SelectionRule::Backtick);
    }
    if after_dot {
        return Some(SelectionRule::Attribute);
    }
    if next == '(' {
        return Some(SelectionRule::Call);
    }
    None
}

/// Code-shaped: an underscore, a digit, or CamelCase — `register_named_middleware`,
/// `parse2`, `BigQueryTarget`. A plain lowercase English word is none of these.
fn code_shape_rule(token: &str) -> Option<SelectionRule> {
    if token.contains('_') {
        return Some(SelectionRule::Snake);
    }
    if token.chars().any(|c| c.is_ascii_digit()) {
        return Some(SelectionRule::Digit);
    }
    let first_upper = token.chars().next().is_some_and(|c| c.is_uppercase());
    let has_lower = token.chars().any(|c| c.is_lowercase());
    if first_upper && has_lower {
        return Some(SelectionRule::Camel);
    }
    None
}

/// The code-shaped tokens of the issue text — the query BM25 reads (§1.6.2 item
/// 3).
///
/// The query is `issue_identifiers`' keys: the names the issue writes as code,
/// whether because of their shape (`response_model_exclude_defaults` is
/// underscored) or because the text writes them as code (`` `registry` ``). Those
/// are exactly the tokens SWE-bench's own retrieval baseline indexes the report
/// for, and taking them from the same reader every other signal uses is what
/// keeps the fallback from seeing a *different* issue than the signals above it.
pub fn issue_code_tokens(issue_text: &str) -> BTreeSet<String> {
    issue_identifiers(issue_text).into_keys().collect()
}

/// The issue's **path-shaped** tokens — the half of `issue_tokens` a reader
/// means when they say "the report named a path" (§1.6.7).
///
/// [VANTA-WHY] `issue_tokens` is every word the `path` and `traceback` signals
/// match against, and it is mostly prose: `matplotlib/1`'s report contains
/// `figure`, `the` and `version`, and matching those against a path is what the
/// rules are for, not what the *extraction* is. This reader is the narrower set —
/// a token carrying a `/` or a source extension — so the §1.6.7 table can print
/// what the issue actually named as a file rather than hundreds of words that
/// happened to reach a basename. It is **reporting only**: nothing ranks with it,
/// and the signals still match against the whole of `issue_tokens`.
pub fn issue_path_tokens(issue_text: &str) -> BTreeSet<String> {
    issue_tokens(issue_text)
        .into_iter()
        .filter(|token| token.contains('/') || looks_like_source(token))
        .collect()
}

/// How often each token appears in one file's contents: the corpus the BM25
/// fallback ranks over (§1.6.2 item 3).
///
/// Split on every character an identifier cannot contain, so `register_named_
/// middleware(` yields `register_named_middleware` and `def`, and case is kept:
/// `Series` and `series` are different tokens, as they are in Python.
fn code_token_counts(contents: &str) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for token in contents.split(|c: char| !is_identifier_char(c)) {
        if !token.is_empty() {
            *counts.entry(token.to_string()).or_insert(0) += 1;
        }
    }
    counts
}

/// Score every candidate by BM25 against `query` and return the ones that
/// matched, priced onto `[1, BM25_WEIGHT]` (§1.6.2 item 3).
///
/// [VANTA-WHY] This is SWE-bench's own retrieval baseline — BM25 over the issue's
/// code-shaped tokens against the file contents — and it exists for the issue that
/// names no *definition*: `fastapi/1`'s `response_model_exclude_defaults` is a
/// parameter, so no candidate declares it and every names-based signal comes up
/// empty. The text still says which file the reporter was in.
///
/// The corpus is the project's own source with the tests left out. A test is the
/// harness's to strip — the F2P test *is* the answer — and a document that the
/// harness removes must not re-price the documents that stay, which is the same
/// rule the identifier's document frequency follows one paragraph up. The
/// frequencies are read from the tree the run holds, so the fallback never learns
/// anything the run was not given.
///
/// The raw score is quantized onto the band by the run's own maximum, so the best
/// match sits at `BM25_WEIGHT` and the rest are ordered below it; a file that
/// matched nothing gets no entry at all. A tie needs the whole score, so a run
/// with one strong match and several weak ones still orders the weak ones by ratio,
/// not by the accident of a fixed bucket width.
fn bm25_weights(candidates: &[Candidate], query: &BTreeSet<String>) -> BTreeMap<String, i64> {
    let mut out: BTreeMap<String, i64> = BTreeMap::new();
    if query.is_empty() {
        return out;
    }
    // (path, term counts, length in tokens) for each document of the corpus.
    let corpus: Vec<(&str, BTreeMap<String, usize>, usize)> = candidates
        .iter()
        .filter(|candidate| !is_test_path(&candidate.path))
        .map(|candidate| {
            let counts = code_token_counts(&candidate.contents);
            let length = counts.values().sum::<usize>();
            (candidate.path.as_str(), counts, length)
        })
        .collect();
    let total = corpus.len();
    if total == 0 {
        return out;
    }
    let average = corpus.iter().map(|(_, _, length)| *length).sum::<usize>() as f64 / total as f64;
    // ln(1 + (N - df + 0.5)/(df + 0.5)), the standard BM25 IDF. Only a term a
    // document actually contains is ever summed, so df >= 1 wherever it is used.
    let idf: BTreeMap<&str, f64> = query
        .iter()
        .map(|term| {
            let df = corpus
                .iter()
                .filter(|(_, counts, _)| counts.contains_key(term.as_str()))
                .count();
            let ratio = (total - df) as f64 + 0.5;
            (term.as_str(), (1.0 + ratio / (df as f64 + 0.5)).ln())
        })
        .collect();

    let raw: Vec<(&str, f64)> = corpus
        .iter()
        .map(|(path, counts, length)| {
            let score: f64 = query
                .iter()
                .filter_map(|term| {
                    let frequency = *counts.get(term.as_str())? as f64;
                    let normalizer =
                        BM25_K1 * (1.0 - BM25_B + BM25_B * (*length as f64) / average);
                    Some(idf[term.as_str()] * (frequency * (BM25_K1 + 1.0))
                        / (frequency + normalizer))
                })
                .sum();
            (*path, score)
        })
        .collect();
    let best = raw.iter().map(|(_, score)| *score).fold(0.0f64, f64::max);
    if best <= 0.0 {
        return out;
    }
    for (path, score) in raw {
        if score <= 0.0 {
            continue;
        }
        let weight = ((score / best) * BM25_WEIGHT as f64)
            .ceil()
            .clamp(1.0, BM25_WEIGHT as f64) as i64;
        out.insert(path.to_string(), weight);
    }
    out
}

/// How many of the candidates *define* `symbol` — the document frequency the
/// identifier weight is scaled by (§1.6.2 item 2).
///
/// [VANTA-WHY] The population is definitions, not mentions, and that choice is
/// what keeps a hidden test from re-pricing a visible file. The `Identifier`
/// reason means "this file defines a name the issue names", so a name's rarity is
/// measured on the same population the signal draws from. Counting mentions
/// instead would make the weight a function of every file in the corpus that
/// writes the name — including the F2P test the harness strips before the run —
/// and a file added or removed anywhere would then re-cut the score of a file the
/// issue named. `the_selection_is_unchanged_when_a_hidden_test_is_added_or_removed`
/// is exactly that invariant, and a hidden test defines no symbol the report
/// names. Within the definition population the count is still a real IDF: `dtype`
/// is defined by a dozen files of pandas and `is_string_dtype` by one, so the name
/// the issue actually named is the one that scores highest.
fn identifier_df(symbol: &str, defined: &[BTreeSet<String>]) -> usize {
    defined
        .iter()
        .filter(|names| names.contains(symbol))
        .count()
}

/// What an `Identifier` match is worth when the name is defined by `df` of
/// `total` candidates: a name one file defines is worth `IDENTIFIER_WEIGHT`, one
/// the whole tree defines `IDENTIFIER_WEIGHT_FLOOR` (§1.6.2 item 2).
///
/// [VANTA-WHY] The scaling is IDF — `ln(1 + N/df)` — mapped onto the band, not a
/// scale composed from scratch. `ln(1 + N/1)` (a name one file of the tree
/// defines) and `ln(1 + N/N) = ln 2` (a name every file defines) are the two ends,
/// and where a name sits between them is its worth. Monotone in `df`, so a rarer
/// name is never worth less than a commoner one; bounded, so no name ever beats a
/// `Path` (400) or falls below an `ImportNeighbour` (100) — which is what keeps
/// §1.6.2 item 2's guarantee that a file the issue names a definition in still
/// outranks a file merely reachable from one, whatever the corpus.
fn identifier_worth(df: usize, total: usize) -> i64 {
    if total <= 1 {
        return IDENTIFIER_WEIGHT;
    }
    let df = df.clamp(1, total) as f64;
    let total = total as f64;
    let idf = (1.0 + total / df).ln();
    let rarest = (1.0 + total).ln(); // df == 1
    let commonest = 2.0f64.ln(); // df == total
    let ratio = ((idf - commonest) / (rarest - commonest)).clamp(0.0, 1.0);
    let span = (IDENTIFIER_WEIGHT - IDENTIFIER_WEIGHT_FLOOR) as f64;
    IDENTIFIER_WEIGHT_FLOOR + (span * ratio).round() as i64
}

/// Every name a file DEFINES: a `def`/`class` at ANY depth — a method counts,
/// and so does a closure — and a module-level binding at column 0. By line-based
/// parsing; nothing is imported or executed.
///
/// [VANTA-WHY] This was `module_level_symbols`, and it skipped every indented
/// line. `sanic/1`'s report names `register_named_middleware`, which
/// `sanic/app.py` declares on line 642 as an indented `def` inside a class, so
/// the old reader answered "the tree does not define the one thing the report
/// names" — a false statement, made by the component whose whole job is to
/// answer that question. A `def`/`class` is a definition wherever it sits; the
/// depth that matters is depth in the *file*, and Python puts a method's name in
/// the same namespace a caller writes.
pub fn defined_names(path: &str, contents: &str) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    for line in contents.lines() {
        let trimmed = line.trim_end();
        // The project's own source is Python in every one of its forms — `.pyx`
        // is Python with Cython's declaration keywords, `.pxd` is the declaration
        // half of it, `.pyi` is the project's own stubs — so all four parse by the
        // Python rules (§1.6.8). Only the other extensions fall through to the
        // generic keywords.
        let name = if is_project_source(path) {
            python_def_name(trimmed.trim_start()).or_else(|| {
                // A binding is module level or it is a local: only column 0,
                // which is the flag's rule too (issue.py's MODULE_BINDING_RE).
                if line.starts_with(' ') || line.starts_with('\t') {
                    None
                } else {
                    python_binding_name(trimmed)
                }
            })
        } else {
            generic_def_name(trimmed.trim_start())
        };
        if let Some(name) = name {
            out.insert(name);
        }
    }
    out
}

/// `x =` / `x: T =` at column 0 — the name a module binds, not `==` and not a
/// tuple unpacking.
fn python_binding_name(line: &str) -> Option<String> {
    let (head, rest) = line.split_once('=')?;
    if rest.starts_with('=') {
        return None; // a comparison, not a binding
    }
    let head = head.split(':').next().unwrap_or(head).trim_end();
    identifier_or_none(head)
}

fn python_def_name(line: &str) -> Option<String> {
    if let Some(name) = cython_def_name(line) {
        return Some(name);
    }
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

/// The name of a Cython declaration: `cpdef int frobnicate(int x)` yields
/// `frobnicate`, not the return type, and `cdef class Context` yields `Context`.
///
/// [VANTA-WHY] `cpdef`/`cdef` are the declarations a Cython module's API consists
/// of, so an issue that names one has to reach its file by `Identifier` — that is
/// what makes the `.pyx` extension mean anything for localization (§1.6.8). The
/// name is taken as the last identifier before the declaration's `(`, which is
/// what makes the return type (`int`, `object *`) drop out. A declaration with
/// neither, a module-level `cdef int x = 5`, yields the name it declares or
/// nothing at all if that is a literal.
fn cython_def_name(line: &str) -> Option<String> {
    let rest = line
        .strip_prefix("cpdef ")
        .or_else(|| line.strip_prefix("cdef "))?;
    let rest = rest.strip_prefix("class ").unwrap_or(rest);
    let head = rest.split(['(', ':']).next().unwrap_or(rest);
    let name = head.split_whitespace().last().unwrap_or("");
    identifier_or_none(name.trim_matches('*'))
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

fn looks_like_source(path: &str) -> bool {
    SOURCE_EXTENSIONS
        .iter()
        .any(|ext| path.ends_with(&format!(".{ext}")))
}

/// Is this path the project's own source, and so entitled to a body on its own?
///
/// The question is the *extension* alone: an `examples/*.py` file is Python, and
/// `defined_names` must parse it as Python. Whether that extension entitles the
/// file to a body is `carries_body`'s question, and it also weighs the directory
/// (`is_paratext`).
fn is_project_source(path: &str) -> bool {
    PROJECT_SOURCE_EXTENSIONS
        .iter()
        .any(|ext| path.ends_with(&format!(".{ext}")))
}

/// Whether `path` sits in one of `PARATEXT_DIRECTORIES` — an example, the
/// documentation, or a benchmark harness (§1.6.8 item 4).
///
/// Only *directories* are tested: the file's own name is dropped first, so a
/// `pkg/docs.py` module is source and a `docs/conf.py` is not.
fn is_paratext(path: &str) -> bool {
    let mut components: Vec<&str> = path.split('/').collect();
    components.pop();
    components
        .iter()
        .any(|component| PARATEXT_DIRECTORIES.contains(component))
}

/// May this file's *body* be carried, or only its path (§1.6.8)?
///
/// A body is the expensive half of the prompt, so it is carried for the project's
/// own source — which is source by extension *and* not paratext by directory —
/// or for a file the run's inputs named: the issue's path or module, a traceback
/// frame, the patch under review. Everything else a signal reached (an import
/// neighbour, a covering test of a file that is not source, an example the
/// issue's symbol happens to appear in) is a path.
fn carries_body(path: &str, reasons: &[SelectionReason]) -> bool {
    if reasons.iter().any(|reason| reason.names_the_file()) {
        return true;
    }
    is_project_source(path) && !is_paratext(path)
}

/// Which rule names `candidate_path` — exactly, by basename, by module name, as a
/// GitHub blob URL, or as the tail of a longer (e.g. absolute) path — or `None`.
///
/// The rules are tried strongest first and the FIRST match is returned, so a
/// token that would match two ways is recorded by the stronger: `pkg/app.py`
/// written out is `exact` even though it also ends with `/app.py`. The blob URL
/// is tested before the tail rule for the same reason — `github.com/o/r/blob/
/// <sha>/pkg/app.py` does end with `/pkg/app.py`, and telling the two apart is
/// what the rule is for (§1.6.2 item 2).
fn path_match_rule(
    tokens: &BTreeSet<String>,
    candidate_path: &str,
    module: Option<&str>,
) -> Option<SelectionRule> {
    if tokens.is_empty() {
        return None;
    }
    let base = Path::new(candidate_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(candidate_path);
    if tokens.contains(candidate_path) {
        return Some(SelectionRule::Exact);
    }
    if tokens.contains(base) {
        return Some(SelectionRule::Basename);
    }
    if let Some(module) = module {
        // [VANTA-WHY] A module name with no `.` in it is the top-level package —
        // `python_module_path("pandas/__init__.py")` is `pandas` — and a bare
        // package name is the one word every one of that project's reports writes
        // in ordinary prose ("I have confirmed this bug exists on the latest
        // version of pandas"). Matching it made `pandas/__init__.py` the top-ranked
        // file for issues that never named it, at `path` weight, above the file
        // that defines the symbol the issue did name (§1.6.2 item 2). A dotted
        // module is a real path — `sanic.app` is something the reporter had to
        // spell out — and still counts.
        if module.contains('.') && tokens.contains(module) {
            return Some(SelectionRule::Module);
        }
    }
    if tokens
        .iter()
        .any(|token| blob_url_path(token).is_some_and(|path| path == candidate_path))
    {
        return Some(SelectionRule::BlobUrl);
    }
    let full_tail = format!("/{candidate_path}");
    let base_tail = format!("/{base}");
    if tokens
        .iter()
        .any(|token| token.ends_with(&full_tail) || token.ends_with(&base_tail))
    {
        return Some(SelectionRule::Tail);
    }
    None
}

/// The repo-relative path a GitHub blob URL names:
/// `github.com/sanic-org/sanic/blob/e7001b0/sanic/app.py` → `sanic/app.py`.
///
/// [VANTA-WHY] A reporter who links the file they are talking about has named
/// that file, which is what the `path` signal means — so the URL resolves to the
/// path and nothing else. The ref (a commit sha) is skipped, never read: nothing
/// here fetches, and nothing here lets a sha decide which file is talked about.
/// The `#L656` anchor is not part of the path either — `issue_tokens` has already
/// split it off, and it is trimmed here so this holds for any caller.
fn blob_url_path(token: &str) -> Option<String> {
    let rest = token.split_once("github.com/")?.1;
    let rest = rest.split(['#', '?']).next().unwrap_or(rest);
    let parts: Vec<&str> = rest.split('/').collect();
    // `owner/repo/blob/<ref>/<path…>`: five parts at the fewest, a file directly
    // under the repo root.
    if parts.len() < 5 || parts[2] != "blob" {
        return None;
    }
    Some(parts[4..].join("/"))
}

// ----------------------------------------------------------------------------
// The shared fill: order by score, then spend the budget
// ----------------------------------------------------------------------------
/// Order the ranked files by score, then spend the budget on them in three forms
/// (§1.6.9).
///
/// The forms are a ladder: `Full` while the file's source fits, then `Outline`
/// (signatures and docstrings) for the files below the point where it stopped
/// fitting, then `Path` — the header and a marker, no source — for the tail. The
/// budget is never exceeded: each form is priced before it is written, and the
/// richest form that fits is the one taken.
///
/// Two distinct things can deny a file its full body, and they are not the same
/// event. The **budget** denying it (§1.6.9's cascade) lowers the bar for every
/// file below, so the prompt reads top to bottom as full, then outlines, then
/// paths. The **per-file cap** (`MAX_FILE_BYTES`) denying it is that file's own
/// property: a file over the cap is an outline, and the ranking continues as if
/// nothing had happened, so a 1 KB helper below a 100 KB monolith still goes in
/// whole while the budget lasts.
///
/// A file the run did not name — not the project's own source, and reached only as
/// an import neighbour or a covering test — carries no body at all (§1.6.8) and so
/// never spends the budget. A file whose very header does not fit is dropped and
/// named in `dropped_for_budget`. A ranking that reached no file selects no file:
/// the tree listing alone is a list of paths, not a context, and the gate refuses
/// it by name ("no issue-grounded selection", §1.6.2 item 5) instead of the fill
/// inventing files to fill the prompt with (§1.6.6).
fn fill(
    mode: &str,
    role: &str,
    candidates: &[Candidate],
    reasons: BTreeMap<String, BTreeMap<SelectionReason, SelectionMatch>>,
    budget_tokens: u32,
    tree: &[String],
    reserved_tokens: u32,
) -> ContextSelection {
    let by_path: BTreeMap<&str, &Candidate> = candidates
        .iter()
        .map(|candidate| (candidate.path.as_str(), candidate))
        .collect();

    // Score = sum of what each signal a file was picked for is worth *for that
    // file* — the identifier's worth is scaled by the name's rarity, every other
    // signal's is its reason's constant (§1.6.2 item 2). Ties are broken by path,
    // so the order is total and stable. The keys of the inner map iterate in
    // `SelectionReason`'s own order, which is what makes `reasons` and `rules`
    // line up signal for signal.
    let mut ranked: Vec<(
        String,
        i64,
        Vec<SelectionReason>,
        BTreeMap<String, String>,
    )> = reasons
        .into_iter()
        .filter(|(path, _)| by_path.contains_key(path.as_str()))
        .map(|(path, signals)| {
            let score: i64 = signals.values().map(|matched| matched.weight).sum();
            let list: Vec<SelectionReason> = signals.keys().copied().collect();
            let rules: BTreeMap<String, String> = signals
                .iter()
                .map(|(reason, matched)| {
                    (reason.as_str().to_string(), matched.rule.as_str().to_string())
                })
                .collect();
            (path, score, list, rules)
        })
        .collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    // [VANTA-WHY] There is no fallback here, deliberately. §1.6.3 added one — when
    // the ranking reached no file, the tree's own files were filled in walk order
    // under a `SelectionReason::Tree` — and it was the wrong answer to a real
    // problem: the prompt was then the file tree with the top of the walk pasted
    // in, which is the listing plus alphabetical noise, not a context. The
    // measure-only run that followed made that legible: 11 of 11 dev bugs showed
    // every file at score 10 with reason `["tree"]`, in path order (`docs/en/docs/
    // js/chat.js` first for fastapi, `doc/conf.py` first for matplotlib), and 9
    // exceeded the budget because the fallback fills until the budget is gone.
    // A selection that reaches no file stays empty; the tree listing still travels
    // and the gate refuses the role for having no grounded file (§1.6.6).

    let mut tree_lines: Vec<String> = tree.to_vec();
    tree_lines.sort();
    let mut tree_bytes: usize = 0;
    if !tree_lines.is_empty() {
        tree_bytes = "# REPOSITORY FILES (paths only)\n".len()
            + tree_lines.iter().map(|line| line.len() + 1).sum::<usize>();
    }
    let tree_tokens = estimate_tokens(tree_bytes);

    // [VANTA-WHY] The budget is for the WHOLE prompt, so everything that is not a
    // selected file is reserved before the first file is priced: the file-tree
    // listing, and — through `reserved_tokens` — the issue text and the frame
    // around it. `prompt_tokens_estimate` adds exactly these back, so a selection
    // that spends to the last token of `remaining` still lands at or below
    // `budget_tokens` (§1.6.9). Reserving the listing alone let a full selection
    // overshoot by `issue_tokens + PROMPT_FRAME_TOKENS`.
    let mut remaining = budget_tokens
        .saturating_sub(tree_tokens)
        .saturating_sub(reserved_tokens);
    let mut files: Vec<SelectedFile> = Vec::new();
    let mut dropped_for_budget: Vec<DroppedFile> = Vec::new();
    let mut total_bytes: usize = 0;
    let mut total_tokens: u32 = 0;

    // [VANTA-WHY] §1.6.6 established which files may be *reached* — a scored
    // signal, never "it is in the tree". §1.6.8 closes the other half: a file a
    // signal reached carries a body only if it is the project's own source or the
    // run named it, so a `.js` the issue mentions does not spend the budget on
    // itself. A file refused here is not a budget drop: the budget never saw it,
    // and `dropped_for_budget` saying otherwise would report the wrong cause.
    //
    // `form` is the richest form still available to the files below the current
    // one (§1.6.9). It only ever descends, and only the budget moves it: a file the
    // per-file cap denies a full body leaves it untouched.
    let mut form = ShownMode::Full;
    for (path, score, reasons, rules) in ranked {
        let candidate = match by_path.get(path.as_str()) {
            Some(candidate) => *candidate,
            None => continue,
        };
        if !carries_body(&path, &reasons) {
            continue;
        }

        let over_cap = candidate.contents.len() > MAX_FILE_BYTES;
        // The richest form the *per-file cap* allows this file. A file over
        // `MAX_FILE_BYTES` is denied its source, but that denial is the file's own
        // property: it never moves `form`, so a small helper ranked below a
        // monolith still goes in whole (§1.6.9).
        let ceiling = if over_cap {
            ShownMode::Outline
        } else {
            ShownMode::Full
        };

        // Full: the whole source, while the cap allows it and the budget has room.
        if form.min(ceiling) == ShownMode::Full {
            let body = candidate.contents.clone();
            let cost = estimate_tokens(framed_len(&path, &body));
            if cost <= remaining {
                remaining -= cost;
                let bytes = framed_len(&path, &body);
                total_bytes += bytes;
                total_tokens += cost;
                files.push(SelectedFile {
                    rank: files.len() + 1,
                    path,
                    score,
                    reasons,
                    rules,
                    bytes,
                    tokens: cost,
                    shown: ShownMode::Full,
                    body,
                });
                continue;
            }
            // The budget, not the cap, denied this file its source: from here down
            // the ranking carries outlines at best. The file itself is not dropped
            // — it is priced again below, in the next form that fits.
            form = ShownMode::Outline;
        }

        // Outline: signatures and docstrings. It runs for the files the budget
        // stopped showing whole — and, on its own, for a file the per-file cap
        // denied a full body, in which case `form` was never moved (§1.6.9).
        if form.min(ceiling) >= ShownMode::Outline {
            let body = outline_body(&path, &candidate.contents);
            let cost = estimate_tokens(framed_len(&path, &body));
            if cost <= remaining {
                remaining -= cost;
                let bytes = framed_len(&path, &body);
                total_bytes += bytes;
                total_tokens += cost;
                files.push(SelectedFile {
                    rank: files.len() + 1,
                    path,
                    score,
                    reasons,
                    rules,
                    bytes,
                    tokens: cost,
                    shown: ShownMode::Outline,
                    body,
                });
                continue;
            }
            // Even the outline did not fit: the tail is paths only.
            form = ShownMode::Path;
        }

        // Path only: the header and the marker, which is all the budget leaves.
        let body = path_only_body();
        let cost = estimate_tokens(framed_len(&path, &body));
        if cost > remaining {
            // A file the ranking chose and the budget cannot take *at all* is a drop,
            // and is recorded: the run would be shown less than the ranking decided,
            // which is what the gate's `selection_dropped_for_budget` reads. It takes
            // a budget already spent to the last few tokens to reach this — the tree
            // listing plus every richer form above it — and everything named here
            // travels to the model as a path in that listing regardless, so a drop
            // costs the file's *source*, never its existence (§1.6.2 item 4). The
            // source's own size travels with the path so a row can say how much was
            // left out, not only how many files (M3 item 2).
            dropped_for_budget.push(DroppedFile {
                path,
                bytes: candidate.contents.len(),
            });
            continue;
        }

        remaining -= cost;
        let bytes = framed_len(&path, &body);
        total_bytes += bytes;
        total_tokens += cost;
        files.push(SelectedFile {
            rank: files.len() + 1,
            path,
            score,
            reasons,
            rules,
            bytes,
            tokens: cost,
            shown: ShownMode::Path,
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

        let selection = select_review_context(&candidates, &touched, "review", 60_000, 0);
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
                select_review_context(
                    &candidates,
                    &touched,
                    "review",
                    60_000,
                    estimate_tokens(issue.len()),
                ),
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
            // `helper` is a NAME of the tree: the traceback's own source line
            // writes `helper(`, and a call is one of the forms that count as code,
            // so the file that defines it is identified (200) — more than the 100
            // the import hop alone would have scored it.
            ("pkg/helper.py", 200, "identifier"),
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
                "from .helper import helper\nfrom .util import util_fn\n\n\ndef frobnicate(value):\n    return helper(value)\n",
            ),
            candidate("pkg/helper.py", "def helper(value):\n    return value\n"),
            candidate("pkg/util.py", "def util_fn():\n    return 0\n"),
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
            vec![
                "pkg/core.py",
                "pkg/helper.py",
                "pkg/alpha.py",
                "pkg/beta.py",
                "pkg/util.py",
            ],
            "path+traceback+identifier (900) > identifier > import-neighbour (100); \
             `helper` is defined by one file and `render` by two, so the rarer name \
             ranks higher, and the `render` tie is broken by path"
        );
        let scores: Vec<i64> = selection.files.iter().map(|file| file.score).collect();
        assert_eq!(
            scores,
            vec![900, 200, 164, 164, 100],
            "the identifier's worth is scaled by the name's rarity across the tree \
             (§1.6.2 item 2): `frobnicate` and `helper` are defined once (200), \
             `render` twice (164), and the import hop is the constant 100"
        );

        let core = &selection.files[0];
        assert_eq!(
            core.reasons,
            vec![
                SelectionReason::Path,
                SelectionReason::Traceback,
                SelectionReason::Identifier,
            ]
        );
        // `helper` is a NAME of the tree: the traceback's own source line writes
        // `helper(`, so the file that defines it is identified (200) — and a file
        // already reached is *not* also credited as a neighbour (the neighbour pass
        // skips it), so the two reasons do not add up to 300.
        let helper = selection
            .files
            .iter()
            .find(|file| file.path == "pkg/helper.py")
            .expect("the file that defines the called name is selected");
        assert_eq!(helper.reasons, vec![SelectionReason::Identifier]);
        assert_eq!(
            helper.rules.get("identifier").map(String::as_str),
            Some("call"),
            "the issue wrote `helper(`, and the recorded rule says so"
        );
        // `util_fn` is imported but named nowhere: the neighbour hop is the only
        // reason that reaches it, which is why the ladder has a rung below the
        // issue's own signals.
        let util = selection
            .files
            .iter()
            .find(|file| file.path == "pkg/util.py")
            .expect("an import neighbour of a named file is selected");
        assert_eq!(util.reasons, vec![SelectionReason::ImportNeighbour]);
        assert_eq!(
            util.rules.get("import-neighbour").map(String::as_str),
            Some("import")
        );
        assert!(
            !paths(&selection).contains(&"pkg/unrelated.py"),
            "a file no reason reaches is not selected at all"
        );
    }

    /// [VANTA-WHY] §1.6.2 item 2 — an identifier is a name, not a word. A bug
    /// report is prose, and a project binds ordinary English too: `matplotlib/1`
    /// says "the 3.2.0 release" and the tree binds `release`, so a reader that
    /// took every token a file defines as a match ranked a file for the sentence
    /// it appeared in. Only a name written as code reaches a file.
    #[test]
    fn a_plain_english_word_in_the_issue_never_reaches_the_file_that_binds_it() {
        let candidates = vec![
            candidate(
                "pkg/notes.py",
                "release = '3.2.0'\nversion = 1\nreport = {}\n",
            ),
            candidate(
                "pkg/middleware.py",
                "def register_named_middleware(app):\n    return app\n",
            ),
        ];
        let issue = "The 3.2.0 release is wrong, and version 4 of the report too.\n\
                     `register_named_middleware` is what breaks.\n";

        let selection = select_repair_context(&candidates, issue, "creator", 60_000);

        assert_eq!(
            paths(&selection),
            vec!["pkg/middleware.py"],
            "`release`, `version` and `report` are English: a word a file happens to \
             bind is not a name the issue named"
        );
        let file = &selection.files[0];
        assert_eq!(file.reasons, vec![SelectionReason::Identifier]);
        assert_eq!(
            file.rules.get("identifier").map(String::as_str),
            Some("snake"),
            "the underscore is what makes `register_named_middleware` a name"
        );
    }

    /// [VANTA-WHY] §1.6.2 item 2 — the other half of the shape rule: a word the
    /// text writes *as code* counts even when it is plain English, because
    /// backticks are the reporter saying "this is a name". `release` in "the 3.2.0
    /// release" is prose; `` `release` `` is the name of something.
    #[test]
    fn a_plain_word_the_issue_backticks_reaches_the_file_that_defines_it() {
        let candidates = vec![
            candidate("pkg/notes.py", "release = '3.2.0'\n"),
            candidate("pkg/other.py", "python = 1\n"),
        ];
        let selection = select_repair_context(
            &candidates,
            "`release` is the wrong value.\n",
            "creator",
            60_000,
        );

        assert_eq!(paths(&selection), vec!["pkg/notes.py"]);
        assert_eq!(
            selection.files[0].rules.get("identifier").map(String::as_str),
            Some("backtick")
        );
    }

    /// [VANTA-WHY] §1.6.2 item 2 — the top-level package name is not a path
    /// signal. `pandas/__init__.py`'s module is `pandas`, and every pandas report
    /// says "the latest version of pandas" in ordinary prose, so matching a
    /// dotless module made the package's own `__init__.py` the top-ranked file for
    /// an issue that never named it — above the file that defines the symbol the
    /// issue did name. A dotted module (`sanic.app`) is a path the reporter had to
    /// spell and still counts.
    #[test]
    fn a_bare_package_name_in_prose_is_not_a_path_signal() {
        let candidates = vec![
            candidate(
                "pandas/__init__.py",
                "from pandas.core.dtypes.common import is_string_dtype\n",
            ),
            candidate(
                "pandas/core/dtypes/common.py",
                "def is_string_dtype(value):\n    return True\n",
            ),
            candidate("pandas/io/other.py", "def other():\n    return 1\n"),
        ];
        let issue = "I have confirmed this bug exists on the latest version of pandas.\n\n\
                     The current implementation of `is_string_dtype` incorrectly \
                     evaluates to True for categorical series.\n";

        let selection = select_repair_context(&candidates, issue, "creator", 60_000);

        assert_eq!(
            selection.files[0].path, "pandas/core/dtypes/common.py",
            "the report named the symbol, not the package: the file that DEFINES \
             `is_string_dtype` is the ranking"
        );
        assert_eq!(
            selection.files[0].rules.get("identifier").map(String::as_str),
            Some("snake")
        );
        // The package's `__init__.py` re-exports the symbol, so it may be reached
        // by the BM25 fallback — it *mentions* the name — but never by a `path`
        // signal (the word `pandas` in prose is the project's name, not a path the
        // report gave) and never above the file that defines it (§1.6.2 items 2, 3).
        if let Some(init) = selection
            .files
            .iter()
            .find(|file| file.path == "pandas/__init__.py")
        {
            assert_eq!(
                init.reasons,
                vec![SelectionReason::Bm25],
                "a mention reaches a file only through the text fallback"
            );
            assert!(
                init.rank > selection.files[0].rank,
                "the definer outranks the mentioner"
            );
        }

        // The dotted form is still a path signal: a module the reporter spelled out.
        assert_eq!(
            path_match_rule(
                &BTreeSet::from(["pandas.core.dtypes.common".to_string()]),
                "pandas/core/dtypes/common.py",
                python_module_path("pandas/core/dtypes/common.py").as_deref(),
            ),
            Some(SelectionRule::Module)
        );
        assert_eq!(
            path_match_rule(
                &BTreeSet::from(["pandas".to_string()]),
                "pandas/__init__.py",
                python_module_path("pandas/__init__.py").as_deref(),
            ),
            None,
            "a dotless module is the top-level package, and its name is a prose word"
        );
    }

    /// [VANTA-WHY] §1.6.2 item 2(b) — a file that *defines* a name the issue named
    /// is evidence that the name lives here; a file that merely mentions it is the
    /// string passing through. The definer carries the `Identifier` reason and the
    /// mentioner carries none at all, so no amount of mentioning can outscore a
    /// definition.
    #[test]
    fn the_file_that_defines_a_name_the_issue_names_beats_one_that_mentions_it() {
        let candidates = vec![
            candidate(
                "pkg/caller.py",
                "from .impl import is_string_dtype\n\n\ndef use(value):\n    return is_string_dtype(value)\n",
            ),
            candidate("pkg/impl.py", "def is_string_dtype(value):\n    return value\n"),
        ];
        let issue = "`is_string_dtype` returns True for a categorical series.\n";

        let selection = select_repair_context(&candidates, issue, "creator", 60_000);

        let definer = selection
            .files
            .iter()
            .find(|file| file.path == "pkg/impl.py")
            .expect("the file that defines the named symbol is selected");
        assert_eq!(definer.reasons, vec![SelectionReason::Identifier]);
        assert_eq!(
            definer.rules.get("identifier").map(String::as_str),
            Some("snake")
        );

        let caller = selection
            .files
            .iter()
            .find(|file| file.path == "pkg/caller.py");
        assert!(
            caller.map_or(true, |file| !file.reasons.contains(&SelectionReason::Identifier)),
            "`pkg/caller.py` mentions `is_string_dtype` and defines nothing the issue \
             named, so it earns no identifier match"
        );
    }

    /// [VANTA-WHY] §1.6.2 item 2(c) — the identifier's worth is weighted by how
    /// rare the name is across the tree. A name one file defines is the strongest
    /// evidence there is; a name three files define is a word the project shares,
    /// and is worth less to the ranking — though never less than an import
    /// neighbour, so a definition can never be displaced by reachability.
    #[test]
    fn a_name_few_files_define_outranks_one_many_files_define() {
        let candidates = vec![
            candidate("pkg/rare.py", "def is_string_dtype(value):\n    return value\n"),
            candidate("pkg/common_a.py", "def dtype(value):\n    return value\n"),
            candidate("pkg/common_b.py", "def dtype(value):\n    return value\n"),
            candidate("pkg/common_c.py", "def dtype(value):\n    return value\n"),
        ];
        let issue = "`is_string_dtype` and `dtype` are both wrong.\n";

        let selection = select_repair_context(&candidates, issue, "creator", 60_000);

        let rare = &selection.files[0];
        assert_eq!(rare.path, "pkg/rare.py", "the rarer name is the stronger evidence");
        assert_eq!(
            rare.score, IDENTIFIER_WEIGHT,
            "a name exactly one file defines is priced at the top of the band"
        );
        let common = selection
            .files
            .iter()
            .find(|file| file.path == "pkg/common_a.py")
            .expect("a file that defines a widely defined name still matches");
        assert!(
            common.score < rare.score,
            "`dtype` is defined by three files of four and must score below the name \
             only one file defines; got {} against {}",
            common.score,
            rare.score
        );
        assert!(
            common.score >= IDENTIFIER_WEIGHT_FLOOR,
            "the floor sits above `ImportNeighbour` (100), so a definition always \
             beats reachability; got {}",
            common.score
        );
    }

    /// [VANTA-WHY] §1.6.2 item 2 — a fenced code block is not a form that writes a
    /// name as code. Reading backticks as a toggle made the three backticks of a
    /// ```python fence flip the state, so everything in the block became a
    /// backticked name: `pandas/1`'s reproduction binds `dtype`, and a dozen
    /// `pandas/core/arrays/*.py` files define a `dtype` method, so a fence the
    /// reporter pasted for a different reason ranked them ahead of the file that
    /// defines `is_string_dtype`. Only the forms the rule names survive: a name the
    /// block writes as `pd.Series` is still the name side of an attribute access.
    #[test]
    fn a_fenced_code_block_is_not_a_form_that_writes_a_name_as_code() {
        let issue = "Bug:\n\n\
                     ```python\n\
                     s = pd.Series(['a', 'b', 'c'], dtype='category')\n\
                     pdt.is_string_dtype(s)\n\
                     ```\n\n\
                     The result is wrong.\n";

        let identifiers = issue_identifiers(issue);

        assert!(
            !identifiers.contains_key("dtype"),
            "`dtype` is plain English inside the block: not code-shaped, and a fence \
             is not one of the writing-as-code forms"
        );
        assert!(
            !identifiers.contains_key("pd"),
            "`pd` is the object side of an access, and only the name side is a name"
        );
        assert!(
            !identifiers.contains_key("s"),
            "a one-character local is not a name: got {identifiers:?}"
        );
        assert_eq!(
            identifiers.get("is_string_dtype"),
            Some(&SelectionRule::Snake),
            "the underscore makes it a name of its own shape, fence or no fence"
        );
        assert_eq!(
            identifiers.get("Series"),
            Some(&SelectionRule::Camel),
            "`Series` is a name the block wrote as code — its own shape names it, \
             which is tested before the written forms, so the rule recorded is the \
             shape and not `attribute`"
        );
    }

    /// [VANTA-WHY] §1.6.7 — the extraction the report prints must be what the issue
    /// *named as a file*, not every word the path signal matches against. The whole
    /// of `issue_tokens` reaches a basename in some directory; the §1.6.7 table is
    /// read for the paths a reporter actually wrote.
    #[test]
    fn the_path_extraction_keeps_only_tokens_shaped_like_a_file() {
        let issue = "On the latest version of pandas, pdt.is_string_dtype(s) raises.\n\
                     See pandas/core/dtypes/common.py and app.py for the details.\n";

        let paths = issue_path_tokens(issue);

        assert_eq!(
            paths,
            ["app.py", "pandas/core/dtypes/common.py"]
                .iter()
                .map(|s| s.to_string())
                .collect::<BTreeSet<_>>(),
            "only the tokens carrying a `/` or a source extension are paths: got {paths:?}"
        );
        assert!(
            !paths.contains("pandas"),
            "the bare package name is a word of the prose, not a path the report named"
        );
    }

    /// [VANTA-WHY] §1.6.2 item 3 — the fallback for an issue that names no
    /// definition. `fastapi/1`'s report names `response_model_exclude_defaults`,
    /// which is a *parameter*: no candidate declares it, so the path, frame and
    /// identifier signals all come up empty and the selection used to be nothing at
    /// all. BM25 over the issue's code-shaped tokens against the file contents —
    /// SWE-bench's own retrieval baseline — still says which file the reporter was
    /// in, priced below every other signal so it orders only what nothing stronger
    /// reached.
    #[test]
    fn an_issue_that_names_no_definition_falls_back_to_the_text() {
        let candidates = vec![
            candidate(
                "fastapi/routing.py",
                "def get(path, response_model_exclude_defaults: bool = False):\n    return path\n",
            ),
            candidate("fastapi/utils.py", "def helper():\n    return 1\n"),
            candidate(
                "tests/test_routing.py",
                "def test_response_model_exclude_defaults():\n    assert True\n",
            ),
        ];
        let issue = "Setting `response_model_exclude_defaults` on a route drops the \
                     fields from the response.\n";

        let selection = select_repair_context(&candidates, issue, "creator", 60_000);

        assert_eq!(
            paths(&selection),
            vec!["fastapi/routing.py"],
            "the parameter is declared nowhere, so only the file whose text contains \
             it is reached — and that is `fastapi/routing.py`"
        );
        let file = &selection.files[0];
        assert_eq!(file.reasons, vec![SelectionReason::Bm25]);
        assert_eq!(file.rules.get("bm25").map(String::as_str), Some("bm25"));
        // The score is recorded, not hidden: it is the file's own score, and it sits
        // at the band's top because this file is the only match in the run.
        assert_eq!(file.score, BM25_WEIGHT, "the best match sits at the band's top");
        assert!(
            !paths(&selection).contains(&"tests/test_routing.py"),
            "a test is not a document of the corpus: the harness strips the F2P test, \
             so it must not re-price the files that stay"
        );
    }

    /// [VANTA-WHY] §1.6.2 item 3 — the fallback is a *fallback*: it never re-prices
    /// or displaces a file a stronger signal reached. `pkg/core.py` defines the name
    /// the issue writes as code, `pkg/noise.py` only mentions it; the definer keeps
    /// its identifier score exactly (no `bm25` folded in), the mentioner is reached
    /// by the text alone, and the definer still ranks first.
    #[test]
    fn the_text_fallback_never_displaces_a_file_a_signal_reached() {
        let candidates = vec![
            candidate("pkg/core.py", "def frobnicate(value):\n    return value\n"),
            candidate("pkg/noise.py", "# callers use frobnicate here\nx = 1\n"),
        ];
        let issue = "`frobnicate` returns the wrong type.\n";

        let selection = select_repair_context(&candidates, issue, "creator", 60_000);

        let core = selection
            .files
            .iter()
            .find(|file| file.path == "pkg/core.py")
            .expect("the file that defines the name is selected");
        assert_eq!(core.reasons, vec![SelectionReason::Identifier]);
        assert_eq!(
            core.score, IDENTIFIER_WEIGHT,
            "a file a signal reached is priced by that signal, and the text fallback \
             is not folded into it"
        );
        let noise = selection
            .files
            .iter()
            .find(|file| file.path == "pkg/noise.py")
            .expect("a file only the text reached is selected");
        assert_eq!(noise.reasons, vec![SelectionReason::Bm25]);
        assert!(noise.score >= 1 && noise.score <= BM25_WEIGHT);
        assert!(core.rank < noise.rank, "the definer outranks the mentioner");
    }

    /// [VANTA-WHY] §1.6.2 item 3 — the recorded weight ORDERS the files the
    /// fallback reached: the better match is at the band's top and the weaker one
    /// below it, so the two are told apart by their own scores instead of both
    /// reading the band's constant. Both files only *mention* the name, so neither
    /// has an identifier match to be priced by.
    #[test]
    fn the_text_fallback_records_a_score_that_orders_its_files() {
        let candidates = vec![
            candidate(
                "pkg/strong.py",
                "def a():\n    zzz_rare_token = 1\n    return zzz_rare_token\n",
            ),
            candidate("pkg/weak.py", "def b():\n    return zzz_rare_token\n"),
        ];
        let issue = "`zzz_rare_token` is wrong.\n";

        let selection = select_repair_context(&candidates, issue, "creator", 60_000);

        assert_eq!(paths(&selection), vec!["pkg/strong.py", "pkg/weak.py"]);
        let strong = &selection.files[0];
        let weak = &selection.files[1];
        assert_eq!(strong.score, BM25_WEIGHT, "the best match tops the band");
        assert!(
            weak.score < strong.score && weak.score >= 1,
            "the weaker match is ordered below it and still recorded: {} vs {}",
            weak.score,
            strong.score
        );
    }

    /// [VANTA-WHY] §1.6.2 item 2 — a method is a name of the tree, and
    /// `sanic/1`'s report names one: `register_named_middleware` is declared in
    /// `sanic/app.py` as an indented `def` inside a class. A module-level-only scan
    /// answered "the tree does not define it" for the file that defines it.
    #[test]
    fn an_indented_method_is_a_name_the_tree_defines() {
        let source = "class Sanic:\n    def register_named_middleware(self, app):\n        return app\n";
        assert!(
            defined_names("sanic/app.py", source).contains("register_named_middleware"),
            "depth is not a filter on `def`: a method is a name the tree defines"
        );

        let candidates = vec![
            candidate("sanic/app.py", source),
            candidate("sanic/other.py", "class Other:\n    pass\n"),
        ];
        let selection = select_repair_context(
            &candidates,
            "`register_named_middleware` never runs.\n",
            "creator",
            60_000,
        );

        assert_eq!(paths(&selection), vec!["sanic/app.py"]);
        assert_eq!(selection.files[0].reasons, vec![SelectionReason::Identifier]);
    }

    /// [VANTA-WHY] §1.6.2 item 2 — a reporter who pastes
    /// `github.com/sanic-org/sanic/blob/e7001b0/sanic/app.py#L656` has named
    /// `sanic/app.py` as much as one who typed the path. The `e7001b0` is a commit
    /// sha and is *skipped*, never resolved or fetched: nothing here reads it, and
    /// the `#L656` anchor is not part of the path either.
    #[test]
    fn a_github_blob_url_names_the_path_it_points_at() {
        assert_eq!(
            blob_url_path("github.com/sanic-org/sanic/blob/e7001b0/sanic/app.py#L656").as_deref(),
            Some("sanic/app.py"),
            "the ref and the line anchor are both stripped"
        );

        let candidates = vec![
            candidate("sanic/app.py", "def handle():\n    return 1\n"),
            candidate("sanic/other.py", "def other():\n    return 2\n"),
        ];
        let selection = select_repair_context(
            &candidates,
            "See https://github.com/sanic-org/sanic/blob/e7001b0/sanic/app.py#L656\n",
            "creator",
            60_000,
        );

        assert_eq!(paths(&selection), vec!["sanic/app.py"]);
        assert_eq!(selection.files[0].reasons, vec![SelectionReason::Path]);
        assert_eq!(
            selection.files[0].rules.get("path").map(String::as_str),
            Some("blob-url")
        );
    }

    /// [VANTA-WHY] §1.6.2 item 2 — in `fig.savefig(` the reporter named
    /// `savefig`, a name of the tree; `fig` is their own local variable and is one
    /// of the words the write-up must not report as a localisation. So only the
    /// name side of a dotted access is a name, and only the name side is recorded.
    #[test]
    fn only_the_name_side_of_a_dotted_access_is_a_name() {
        let candidates = vec![
            candidate("pkg/figure.py", "fig = None\n"),
            candidate("pkg/objects.py", "myfig = None\n"),
            candidate("pkg/saving.py", "def savefig(path):\n    return path\n"),
            candidate("pkg/plotting.py", "def dataframe(name):\n    return name\n"),
        ];
        let issue = "fig.savefig() writes nothing, and myfig.dataframe is empty.\n";

        let selection = select_repair_context(&candidates, issue, "creator", 60_000);

        assert_eq!(
            paths(&selection),
            vec!["pkg/plotting.py", "pkg/saving.py"],
            "`savefig` and `dataframe` are the names; `fig` and `myfig` are the \
             reporter's own objects and name nothing (the tie is broken by path)"
        );
        let rules: Vec<(&str, Option<&str>)> = selection
            .files
            .iter()
            .map(|file| {
                (
                    file.path.as_str(),
                    file.rules.get("identifier").map(String::as_str),
                )
            })
            .collect();
        assert_eq!(
            rules,
            vec![("pkg/plotting.py", Some("attribute")), ("pkg/saving.py", Some("attribute"))],
            "both are the name side of a dotted access, and that is what is recorded"
        );
    }

    /// [VANTA-WHY] §1.6.4 — a reason says which input named a file, not which rule
    /// inside that input did, and `path` alone covers five of them. The pair is
    /// what a reader of the report needs: `identifier` says the issue named a name
    /// this file defines, `camel` says the name was code on its face.
    #[test]
    fn the_rule_that_matched_each_signal_is_recorded_beside_it() {
        let candidates = vec![candidate("pkg/target.py", "class BigQueryTarget:\n    pass\n")];
        let issue = "`BigQueryTarget` in pkg/target.py returns nothing.\n";

        let selection = select_repair_context(&candidates, issue, "creator", 60_000);

        let file = &selection.files[0];
        assert_eq!(
            file.reasons,
            vec![SelectionReason::Path, SelectionReason::Identifier]
        );
        assert_eq!(
            file.rules
                .iter()
                .map(|(name, rule)| (name.as_str(), rule.as_str()))
                .collect::<Vec<_>>(),
            vec![("identifier", "camel"), ("path", "exact")],
            "one rule per signal, keyed by the signal's own name and in its order"
        );
    }

    /// [VANTA-WHY] §1.6.4 — review mode's three signals are recorded the same way,
    /// so a report of a review selection says why each file is in the prompt too.
    #[test]
    fn a_review_selection_records_the_rule_of_every_signal() {
        let candidates = vec![
            candidate(
                "heretic/keyword_rate.py",
                "from .plugin import Context\n\nclass KeywordRate:\n    pass\n",
            ),
            candidate("heretic/plugin.py", "class Context:\n    pass\n"),
            candidate(
                "tests/test_keyword_rate.py",
                "from heretic.keyword_rate import KeywordRate\n\ndef test_keyword_rate():\n    assert KeywordRate is not None\n",
            ),
        ];
        let touched = BTreeSet::from(["heretic/keyword_rate.py".to_string()]);

        let selection = select_review_context(&candidates, &touched, "review", 60_000, 0);

        let rules: Vec<(&str, Vec<(&str, &str)>)> = selection
            .files
            .iter()
            .map(|file| {
                (
                    file.path.as_str(),
                    file.rules
                        .iter()
                        .map(|(name, rule)| (name.as_str(), rule.as_str()))
                        .collect(),
                )
            })
            .collect();
        assert_eq!(
            rules,
            vec![
                ("heretic/keyword_rate.py", vec![("patch-touched", "hunk")]),
                ("heretic/plugin.py", vec![("import-neighbour", "import")]),
                (
                    "tests/test_keyword_rate.py",
                    vec![("test", "covering-test")]
                ),
            ]
        );
    }

    /// [VANTA-WHY] §1.6.3 made an empty ranking fall back to the tree's own source
    /// in walk order, so that a listing alone would not be sent as "the context".
    /// The measure-only run that followed showed what it actually produced: 11 of
    /// 11 dev bugs with every file at one score under the same `tree` reason, in
    /// path order (`docs/en/docs/js/chat.js` first for fastapi, `doc/conf.py` first
    /// for matplotlib), 9 of them over budget — the old whole-repo dump with a
    /// bigger cap. The fallback is gone: a ranking that reaches no file selects no
    /// file, and the gate refuses that as having no grounded file.
    ///
    /// What this pins is the signature the report reads: `total_bytes ==
    /// tree_bytes` exactly, because nothing but the listing is in the selection.
    #[test]
    fn a_ranking_that_reaches_no_file_selects_no_file() {
        let candidates = vec![
            // In walk order: `walk_candidates` hands the list over sorted by path.
            candidate("pkg/alpha.py", "def a():\n    return 2\n"),
            candidate("pkg/mid.py", &"x = 1\n".repeat(4_000)),
            candidate("pkg/zeta.py", "def z():\n    return 1\n"),
        ];
        // One sentence of prose naming nothing the tree contains: no path, no
        // traceback frame, no `def`/`class` name. This is what BugsInPy's own
        // messages look like once stripped (`Fix #2`, one behavioural sentence).
        let issue = "Fix calling method on the wrong object.\n";

        // A budget that would comfortably fit all three: the selection is empty
        // because nothing was *grounded*, not because nothing fit.
        let selection = select_repair_context(&candidates, issue, "repair", 60_000);

        assert!(
            selection.files.is_empty(),
            "a listing of paths is not a context, so an empty ranking selects nothing"
        );
        assert!(
            selection.dropped_for_budget.is_empty(),
            "nothing was dropped for budget: nothing was ever chosen"
        );
        assert_eq!(
            selection.total_bytes, selection.tree_bytes,
            "the selection is the listing and nothing else — the `source_bytes == \
             tree_bytes` signature the gate reads (§1.6.6)"
        );
        assert_eq!(
            selection.tree,
            vec!["pkg/alpha.py", "pkg/mid.py", "pkg/zeta.py"],
            "the tree listing still travels, paths only"
        );

        // The same tree with an issue that DOES name a file is unaffected: the walk
        // supplies paths, the ranking supplies content.
        let named = select_repair_context(
            &candidates,
            "`a` fails in pkg/alpha.py\n",
            "repair",
            60_000,
        );
        assert_eq!(paths(&named), vec!["pkg/alpha.py"]);
        assert_eq!(named.files[0].reasons, vec![SelectionReason::Path]);
    }

    /// [VANTA-WHY] The gate's `prompt_exceeds_budget` reads this arithmetic and
    /// nothing else, so the listing has to enter it exactly once: it travels inside
    /// `total_tokens` (which is `files + tree`, §1.6.4), and adding `tree_tokens`
    /// again inflated every estimate by one whole listing — enough to turn a
    /// selection that fits into an over-budget refusal.
    #[test]
    fn the_prompt_estimate_counts_the_listing_exactly_once() {
        let candidates = vec![
            candidate("pkg/alpha.py", "def one():\n    return 1\n"),
            candidate("pkg/beta.py", "def two():\n    return 2\n"),
        ];
        let selection = select_repair_context(
            &candidates,
            "`one` fails in pkg/alpha.py\n",
            "creator",
            60_000,
        );

        assert!(
            selection.tree_tokens > 0,
            "the fixture must have a listing for this test to mean anything"
        );
        assert_eq!(
            prompt_tokens_estimate(&selection, 100),
            selection.total_tokens + 100 + PROMPT_FRAME_TOKENS,
            "the estimate is the selection, the issue and the frame — the listing is \
             already in `total_tokens` and must not be counted twice"
        );
    }

    /// [VANTA-WHY] §1.6.8 — the walk's extension set says which paths exist; this
    /// says which of them may spend the budget on a body. Reaching a `.js` file is
    /// not the same as the run naming it: an issue that names its *path* buys it a
    /// body, an issue that names only a *symbol it defines* does not.
    #[test]
    fn a_body_goes_to_the_projects_source_and_to_what_the_run_named() {
        let candidates = vec![
            candidate("docs/chat.js", "class Chat {\n  render() {}\n}\n"),
            candidate("pkg/core.py", "def frobnicate():\n    return 1\n"),
        ];

        // The issue names the `.js` file by path: it is named, so it is content
        // whatever its extension.
        let by_path = select_repair_context(
            &candidates,
            "The bundle at docs/chat.js is stale.\n",
            "creator",
            60_000,
        );
        assert_eq!(paths(&by_path), vec!["docs/chat.js"]);
        assert!(by_path.files[0].reasons.contains(&SelectionReason::Path));
        assert_eq!(by_path.files[0].shown, ShownMode::Full);

        // The issue names a symbol the `.js` file *defines*, and nothing else. A
        // symbol is a name, not a file: the file is reachable, not named, so it
        // travels as a path in the listing and never as code.
        let by_symbol = select_repair_context(
            &candidates,
            "The `Chat` widget misrenders.\n",
            "creator",
            60_000,
        );
        assert!(
            by_symbol.files.is_empty(),
            "an `identifier` hit on a file that is not the project's source is a \
             path, not a body (§1.6.8); got {:?}",
            paths(&by_symbol)
        );
        assert!(
            by_symbol.tree.contains(&"docs/chat.js".to_string()),
            "its path still travels — the listing is where a reachable file lives"
        );
        assert!(
            by_symbol.dropped_for_budget.is_empty(),
            "the budget did not refuse it, so it must not be reported as a drop"
        );
    }

    /// [VANTA-WHY] §1.6.8 item 4 — an example, the documentation and a benchmark
    /// harness are the project's *paratext*, not the code a bug is fixed in. A
    /// `.py` example is Python, so the extension rule alone showed
    /// `sanic/1`'s `examples/add_task_sanic.py` in full at rank 4 for a symbol the
    /// issue named — reachable, not named. Being `Identifier`-reached is not
    /// enough: only a *path* signal buys these files a body.
    #[test]
    fn an_example_benchmark_or_doc_is_a_path_only_however_it_is_reached() {
        let candidates = vec![
            candidate("sanic/app.py", "def add_task(app):\n    return app\n"),
            candidate("examples/add_task_sanic.py", "def add_task(app):\n    return app\n"),
            candidate("docs/conf.py", "def add_task(app):\n    return app\n"),
            candidate("benchmarks/add_task.py", "def add_task(app):\n    return app\n"),
        ];
        let issue = "`add_task` never runs.\n";

        // A symbol the issue names, defining in all four. Only the project's own
        // source carries a body; the paratext is left to the listing.
        let by_symbol = select_repair_context(&candidates, issue, "creator", 60_000);
        assert_eq!(
            paths(&by_symbol),
            vec!["sanic/app.py"],
            "an `identifier` hit reaches a file; it does not name it (§1.6.8 item 4)"
        );
        for path in [
            "examples/add_task_sanic.py",
            "docs/conf.py",
            "benchmarks/add_task.py",
        ] {
            assert!(
                by_symbol.tree.contains(&path.to_string()),
                "the path still travels in the listing: {path}"
            );
        }

        // A path the issue does give buys the example its body, exactly as it does
        // for a `.js` file: the rule is about being named, not about the directory.
        let by_path = select_repair_context(
            &candidates,
            "`add_task` in examples/add_task_sanic.py never runs.\n",
            "creator",
            60_000,
        );
        let example = by_path
            .files
            .iter()
            .find(|file| file.path == "examples/add_task_sanic.py")
            .expect("the file the issue named by path is selected");
        assert!(
            example.reasons.contains(&SelectionReason::Path),
            "the path signal is what names it"
        );
        assert_eq!(example.shown, ShownMode::Full);
    }

    /// [VANTA-WHY] §1.6.8 — the other half of the rule, and the half that makes
    /// review mode meaningful: the patch under review is the subject, so a hunk in
    /// a file the walk lists is content even when the extension is not the
    /// project's own source. Without this the reviewer would be told about a file
    /// it cannot read.
    #[test]
    fn the_patch_gives_a_body_even_outside_the_projects_own_source() {
        let candidates = vec![
            candidate("pkg/core.py", "def frobnicate():\n    return 1\n"),
            candidate("web/toggle.js", "function toggle() {\n  return 1\n}\n"),
        ];
        let touched = BTreeSet::from(["web/toggle.js".to_string()]);

        let selection = select_review_context(&candidates, &touched, "review", 60_000, 0);

        assert_eq!(paths(&selection), vec!["web/toggle.js"]);
        assert_eq!(selection.files[0].reasons, vec![SelectionReason::PatchTouched]);
        assert_eq!(
            selection.files[0].shown,
            ShownMode::Full,
            "a patch-touched file always gets content, whatever its extension"
        );
        assert!(
            !selection.tree.contains(&"web/toggle.js".to_string()),
            "review mode has no listing: the patch is what localizes the repo"
        );
    }

    /// [VANTA-WHY] §1.6.8 — `.pyx`, `.pxd` and `.pyi` are the project's own
    /// source, so the walk has to *see* them and their declarations have to be
    /// parsed as the symbols they are. pandas and spacy have real bugs in Cython
    /// files, and an issue that names a `cpdef` has to reach its file.
    #[test]
    fn cython_and_stub_files_are_walked_and_their_declarations_are_symbols() {
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("pkg/_libs")).unwrap();
        fs::write(
            tmp.path().join("pkg/_libs/algos.pyx"),
            "cpdef int is_string_dtype(object arr):\n    return 1\n\n\ncdef class Engine:\n    pass\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("pkg/_libs/algos.pxd"),
            "cdef class Engine:\n    cdef int is_string_dtype(self, object arr)\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("pkg/_libs/algos.pyi"),
            "def is_string_dtype(arr: object) -> bool: ...\n",
        )
        .unwrap();

        let (candidates, _) = walk_candidates(tmp.path());
        assert_eq!(
            candidates.iter().map(|c| c.path.as_str()).collect::<Vec<_>>(),
            vec![
                "pkg/_libs/algos.pxd",
                "pkg/_libs/algos.pyi",
                "pkg/_libs/algos.pyx"
            ],
            "the walk lists Cython and stub source, not only `.py`"
        );

        // The declaration's own name, never its return type.
        let symbols = defined_names("pkg/_libs/algos.pyx", &candidates[2].contents);
        assert!(symbols.contains("is_string_dtype"), "got {symbols:?}");
        assert!(symbols.contains("Engine"), "got {symbols:?}");
        assert!(!symbols.contains("int"), "a return type is not a symbol");

        // End to end: an issue naming the Cython function reaches its file.
        let selection = select_repair_context(
            &[candidates[2].clone()],
            "`is_string_dtype` returns the wrong answer.\n",
            "creator",
            60_000,
        );
        assert_eq!(paths(&selection), vec!["pkg/_libs/algos.pyx"]);
        assert!(selection.files[0]
            .reasons
            .contains(&SelectionReason::Identifier));
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

    /// [VANTA-WHY] §1.6.9 — the fill descends a three-form ladder as the budget
    /// runs out: a file is shown whole while it fits, then as its outline, then as
    /// its path alone. Once the budget has denied a file a richer form, no file
    /// below it gets one — the run must be shown a *prefix* of the ranking's forms,
    /// not a patchwork — and every file records which form it got, so an analysis
    /// can tell "the model saw this file" from "the model saw its path".
    #[test]
    fn the_budget_fill_descends_the_form_ladder_and_the_tail_never_climbs_back() {
        let a = "def a():\n    return 1\n";
        // Big bodies under few definitions: the outline is far smaller than the
        // source, which is what lets a whole body be denied while its outline fits.
        let b = format!("def b():\n    \"\"\"Doc.\"\"\"\n{}", "    return 1\n".repeat(20));
        let c = format!("def c():\n    \"\"\"Doc.\"\"\"\n{}", "    return 1\n".repeat(20));
        let candidates = vec![
            candidate("pkg/a.py", a),
            candidate("pkg/b.py", &b),
            candidate("pkg/c.py", &c),
        ];
        let issue = "pkg/a.py and pkg/b.py and pkg/c.py all matter.\n";

        // The budget is exactly what the WHOLE prompt costs — the issue text, the
        // frame, the listing, a whole `a`, an outline `b` and a path-only `c` — so
        // the ladder is forced to descend and the test depends on no magic number.
        let tree = "# REPOSITORY FILES (paths only)\n".len()
            + "pkg/a.py\n".len()
            + "pkg/b.py\n".len()
            + "pkg/c.py\n".len();
        let budget = reserved_prompt_tokens(estimate_tokens(issue.len()))
            + estimate_tokens(tree)
            + estimate_tokens(framed_len("pkg/a.py", a))
            + estimate_tokens(framed_len("pkg/b.py", &outline_body("pkg/b.py", &b)))
            + estimate_tokens(framed_len("pkg/c.py", &path_only_body()));

        let selection = select_repair_context(&candidates, issue, "creator", budget);

        assert_eq!(paths(&selection), vec!["pkg/a.py", "pkg/b.py", "pkg/c.py"]);
        assert_eq!(selection.files[0].shown, ShownMode::Full);
        assert_eq!(selection.files[1].shown, ShownMode::Outline);
        assert_eq!(selection.files[2].shown, ShownMode::Path);

        // Each file records its place in score order and what it cost.
        for (index, file) in selection.files.iter().enumerate() {
            assert_eq!(file.rank, index + 1, "rank states the position in score order");
            assert!(file.tokens > 0);
        }

        // The path form travels as its header and the marker, and nothing else.
        assert_eq!(selection.files[2].body, path_only_body());
        assert!(
            !selection.files[2].body.contains("def c():"),
            "a path-only entry carries no source at all"
        );

        assert_eq!(
            selection.total_bytes,
            tree + framed_len("pkg/a.py", a)
                + framed_len("pkg/b.py", &outline_body("pkg/b.py", &b))
                + framed_len("pkg/c.py", &path_only_body()),
            "the listing plus the three framed files, exactly"
        );
        assert_eq!(
            prompt_tokens_estimate(&selection, estimate_tokens(issue.len())),
            budget,
            "the whole prompt — issue, frame, listing and the three framed files — \
             lands exactly on the budget, and so can never land above it"
        );
        assert!(selection.dropped_for_budget.is_empty());
    }

    /// [VANTA-WHY] §1.6.9 — the listing is part of the prompt, so a large listing
    /// has to be paid for out of the budget *before* any file is. This walks 2000
    /// paths, a listing that on its own is most of the budget, and shows the whole
    /// prompt — issue, frame, listing and the one file the issue names — still
    /// landing at or below the budget: one token less and the file is dropped by
    /// name, never silently overrun. Reserving the listing alone (the old fill)
    /// would have overshot here by the issue text plus the frame.
    #[test]
    fn a_large_tree_listing_is_spent_from_the_budget_before_any_file() {
        let mut candidates: Vec<Candidate> = (0..2000)
            .map(|n| candidate(&format!("pkg/module_{n:04}.py"), "def helper():\n    return 1\n"))
            .collect();
        // One definition over a long body, so the forms are strictly ordered by
        // cost — full > outline > path — and the path form is the only one that can
        // fit the token left over.
        let target = format!(
            "def frobnicate():\n    \"\"\"Doc.\"\"\"\n{}",
            "    return 1\n".repeat(40)
        );
        candidates.push(candidate("pkg/target.py", &target));
        let issue = "pkg/target.py raises on every call.\n";

        let tree_bytes = "# REPOSITORY FILES (paths only)\n".len()
            + candidates.iter().map(|c| c.path.len() + 1).sum::<usize>();
        let exact = reserved_prompt_tokens(estimate_tokens(issue.len()))
            + estimate_tokens(tree_bytes)
            + estimate_tokens(framed_len("pkg/target.py", &path_only_body()));

        // One token short of the path form: the listing has spent the rest, so the
        // file the issue names cannot be shown at all — and the prompt the
        // selection describes still does not exceed the budget.
        let short = select_repair_context(&candidates, issue, "creator", exact - 1);
        assert!(
            short.tree_tokens * 4 > short.budget_tokens * 3,
            "the fixture must have a listing large enough to matter ({} of {} tok)",
            short.tree_tokens,
            short.budget_tokens
        );
        assert!(
            short.files.is_empty(),
            "the listing left no room for a body, got {:?}",
            paths(&short)
        );
        assert_eq!(
            short.dropped_for_budget,
            vec![DroppedFile {
                path: "pkg/target.py".to_string(),
                bytes: target.len(),
            }],
            "the dropped file is named with the source bytes the budget left out"
        );
        assert!(
            prompt_tokens_estimate(&short, estimate_tokens(issue.len())) <= short.budget_tokens,
            "the whole prompt never exceeds the budget, however large the listing"
        );

        // Exactly enough: the path form fits, and the whole prompt lands on the
        // budget and no higher.
        let fitting = select_repair_context(&candidates, issue, "creator", exact);
        assert_eq!(paths(&fitting), vec!["pkg/target.py"]);
        assert_eq!(fitting.files[0].shown, ShownMode::Path);
        assert_eq!(
            prompt_tokens_estimate(&fitting, estimate_tokens(issue.len())),
            exact
        );
        assert!(fitting.dropped_for_budget.is_empty());
    }

    /// [VANTA-WHY] §1.6.10 — the gate refuses a role whose selection has no *body*
    /// grounded in the run's inputs. A file that travels as its path alone is a
    /// selected file — its reason still grounds the selection — but it carries no
    /// content, so a role shown only paths is refused exactly as a role shown only
    /// a listing is.
    #[test]
    fn a_selection_of_paths_only_is_not_grounded() {
        let d = format!("def d():\n    \"\"\"Doc.\"\"\"\n{}", "    return 1\n".repeat(20));
        let candidates = vec![candidate("pkg/d.py", &d)];
        let issue = "pkg/d.py is broken.\n";

        // Room for the issue, the frame, the tree and the path form, and no more:
        // the whole body and the outline are both denied.
        let tree = "# REPOSITORY FILES (paths only)\n".len() + "pkg/d.py\n".len();
        let budget = reserved_prompt_tokens(estimate_tokens(issue.len()))
            + estimate_tokens(tree)
            + estimate_tokens(framed_len("pkg/d.py", &path_only_body()));

        let selection = select_repair_context(&candidates, issue, "creator", budget);

        assert_eq!(selection.files.len(), 1);
        assert_eq!(selection.files[0].shown, ShownMode::Path);
        assert!(
            selection.files[0]
                .reasons
                .iter()
                .any(|reason| reason.grounds_content()),
            "the reason still grounds the selection; it is the form that carries no content"
        );
        assert!(
            !selection.grounded(),
            "a role shown only paths has no content grounded in the issue"
        );
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
            vec![DroppedFile {
                path: "pkg/core.py".to_string(),
                bytes: "def frobnicate(value):\n    return value\n".len(),
            }],
            "the dropped file is named, so the gate can refuse the bug for the right reason"
        );
    }

    /// [VANTA-WHY] M3 item 2 — a selection that spends its budget on the file the
    /// issue names and drops the *tail* records both halves: the top-ranked file
    /// went in full, and the candidate the budget could not take is named with the
    /// source bytes it would have cost. That pair is what lets the gate sanction a
    /// drop — the bug's own file is in the prompt — without the row hiding how much
    /// source was left out.
    ///
    /// It also pins the invariant the gate's rule rests on: a drop cannot take the
    /// ranking's *first* file while leaving a later one in, because the fill walks
    /// in score order and its remaining budget only ever falls. When any file is
    /// included, `files[0]` is therefore the one the ranking put first.
    #[test]
    fn a_drop_takes_the_tail_and_names_it_with_its_bytes() {
        let a = "def a():\n    return 1\n";
        // Many definitions, so the outline is large and a path is the only form the
        // budget could have afforded it — which is what makes the drop a drop.
        let b: String = (0..40)
            .map(|n| format!("def beta_{n}():\n    \"\"\"Doc.\"\"\"\n    return 2\n"))
            .collect();
        let candidates = vec![candidate("pkg/a.py", a), candidate("pkg/b.py", &b)];
        // Both are named by path, so both rank; the tie is broken by path, so `a` is
        // first — the file the ranking put first.
        let issue = "pkg/a.py and pkg/b.py both matter.\n";

        let tree = "# REPOSITORY FILES (paths only)\n".len()
            + "pkg/a.py\n".len()
            + "pkg/b.py\n".len();
        // Room for the whole prompt up to `a` in full and `b`'s path, one token
        // short of that path: `b` is the tail the budget cannot take at all.
        let budget = reserved_prompt_tokens(estimate_tokens(issue.len()))
            + estimate_tokens(tree)
            + estimate_tokens(framed_len("pkg/a.py", a))
            + estimate_tokens(framed_len("pkg/b.py", &path_only_body()))
            - 1;

        let selection = select_repair_context(&candidates, issue, "creator", budget);

        assert_eq!(paths(&selection), vec!["pkg/a.py"]);
        assert_eq!(
            selection.files[0].shown,
            ShownMode::Full,
            "the file the ranking put first goes in whole"
        );
        assert_eq!(
            selection.dropped_for_budget,
            vec![DroppedFile {
                path: "pkg/b.py".to_string(),
                bytes: b.len(),
            }],
            "the drop takes the tail and is named with the source bytes it cost"
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
