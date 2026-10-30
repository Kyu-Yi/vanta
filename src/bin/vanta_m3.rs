// ============================================================================
// vanta_m3 — thin CLI wrapper over the shared driver (Spec §7, §8)
// ============================================================================
//
// [VANTA-WHY] This binary used to carry its own copy of the attempt cycle.
// M2 carried another, and the two drifted (M3's BATE lost the Attacker-revise
// layer; M3's Diagnostician lost Phase-2 remediation). The loop now lives in
// exactly one place — `vanta::pipeline::driver` — and this file is only the
// argument surface plus an exit-code mapping. M3 is the driver at N attempts
// (default 4); M2 is the SAME driver at N = 1.
//
// Flow (owned by the driver, listed here for orientation):
//   Creator → Attacker → BATE → patched run → Diagnostician → Governor →
//   Gatekeeper → (PR)
// ============================================================================

use std::path::PathBuf;

use vanta::core::GovernorDecision;
use vanta::logging::init_logging;
use vanta::pipeline::driver::{self, RunConfig};

// ============================================================================
// `--select-context` — the inputs are exactly a real run's, and only those
// ============================================================================
//
// [VANTA-WHY] The harness's context gate used to re-derive the source walk and
// the budget in Python (`scripts/bugsinpy/context.py`), which meant the gate
// measured a *different* prompt from the one the run sends — the exact failure
// the gate exists to prevent. It now shells out to this subcommand, so the walk
// and the ranking have ONE implementation (`vanta::context`).
//
// [VANTA-WHY] The flag surface is a leak guard, not a convenience. The selector
// is given the tree, the issue text, the patch under review (when there is one),
// the role and the budget — the same things a run has — and NOTHING else. There
// is deliberately no flag for `FAIL_TO_PASS`, a hidden test, the fix commit, the
// fix diff or `bug.info`: a selector that could read the answer would size the
// prompt around it. The set is asserted in this file's tests, so adding such a
// flag fails the suite rather than quietly widening what the selector can see.
#[cfg(test)]
const SELECT_CONTEXT_FLAGS: &[&str] = &[
    "--select-context",
    "--json",
    "--repo",
    "--issue",
    "--patch",
    "--select-role",
    "--select-budget",
];

// ============================================================================
// CLI — manual arg parsing (no clap dep; keep Cargo.toml minimal)
// ============================================================================
struct CliArgs {
    repo_path: Option<PathBuf>,
    issue_path: Option<PathBuf>,
    max_attempts: u32,
    /// M3 item 9 — `--fixed-rounds K`: run EXACTLY K repair attempts, taking the
    /// verdict from attempt K, so a patch that passes early is still re-examined
    /// (EIR). Off by default. A repair option only; refused with `--patch`.
    fixed_rounds: Option<u32>,
    model_override: Option<String>,
    /// If true, open a GitHub PR after Gatekeeper approval.
    open_pr: bool,
    /// If Some(path), skip the Creator and use this patch file directly.
    /// Used for verifying an EXISTING patch (e.g. a developer's working tree diff).
    patch_path: Option<PathBuf>,
    /// Explicit test command (Spec §7: "Require explicit --test-cmd, pytest as
    /// fallback. Do not infer."). e.g. "pytest tests/ -q".
    test_cmd: Option<String>,
    /// Environment injected into every sandbox command of the run, from
    /// `--sandbox-env KEY=VALUE` (repeatable). Only the keys in the recorded
    /// allow-list (`PYTHONPATH`, `MPLBACKEND`) are accepted; any other key is
    /// refused rather than quietly injected.
    sandbox_env: Vec<(String, String)>,
    /// Tree-relative `.py` files byte-compiled before every suite run, from
    /// `--precompile PATH` (repeatable). This is the harness's recorded
    /// per-project rule (item L), handed to the run so Vanta's OWN suite runs
    /// meet the same compile ordering the preparation gates did — matplotlib's
    /// pytest configuration escalates the invalid-escape warning that file
    /// raises at compile time into a `SyntaxError`, which kills collection. A
    /// path outside the tree, a backslash, a `..` step or a non-`.py` name is
    /// refused rather than compiled.
    precompile: Vec<String>,
    /// Path to a clean original tree (git worktree of the base commit) for
    /// Mode B. If set, BATE runs against this tree instead of the in-memory
    /// original files. Spec §7: "both trees present simultaneously".
    original_tree: Option<PathBuf>,
    /// If Some(session_id), do NOT run the pipeline: re-run the signals behind
    /// that session's review-mode rejection against a known-good tree and write
    /// the per-signal verdicts. Item 11 of the heretic fix list.
    rerun_signals: Option<String>,
    /// Known-good source tree for `--rerun-signals` (a directory, or a patch
    /// file that `--base-tree` is applied to).
    good_tree: Option<PathBuf>,
    /// Base tree the `--good-tree` patch applies to; required only when
    /// `--good-tree` is a patch file.
    base_tree: Option<PathBuf>,
    /// Explicit opt-in to re-run a `subprocess`-recorded session under shuru.
    /// Refused when the recorded `test_cmd` names a host path the VM cannot see.
    allow_backend_upgrade: bool,
    /// If true, do NOT run the pipeline: select the context a role would be shown
    /// and print it. Inputs: `--repo` (the tree), `--issue`, `--patch` (optional),
    /// `--select-role`, `--select-budget`.
    select_context: bool,
    /// With `--select-context`: print the selection as JSON on stdout.
    json: bool,
    /// With `--select-context`: the role label the selection is made for.
    select_role: Option<String>,
    /// With `--select-context`: the prompt budget in tokens.
    select_budget: Option<u32>,
}


fn parse_args() -> Result<CliArgs, String> {
    let args: Vec<String> = std::env::args().collect();
    parse_args_from(&args)
}

/// [VANTA-WHY] Split from `parse_args` so the flag surface is testable without
/// an argv: the test that pins `--select-context`'s inputs (`SELECT_CONTEXT_FLAGS`)
/// has to name them itself, and `std::env::args` cannot be written to in a test.
fn parse_args_from(args: &[String]) -> Result<CliArgs, String> {
    let mut repo_path: Option<PathBuf> = None;
    let mut issue_path: Option<PathBuf> = None;
    let mut max_attempts: u32 = 4;
    let mut fixed_rounds: Option<u32> = None;
    let mut model_override: Option<String> = None;
    let mut open_pr = false;
    let mut patch_path: Option<PathBuf> = None;
    let mut test_cmd: Option<String> = None;
    let mut sandbox_env: Vec<(String, String)> = Vec::new();
    let mut precompile: Vec<String> = Vec::new();
    let mut original_tree: Option<PathBuf> = None;
    let mut rerun_signals: Option<String> = None;
    let mut good_tree: Option<PathBuf> = None;
    let mut base_tree: Option<PathBuf> = None;
    let mut allow_backend_upgrade = false;
    let mut select_context = false;
    let mut json = false;
    let mut select_role: Option<String> = None;
    let mut select_budget: Option<u32> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--repo" | "-r" => {
                i += 1;
                if i >= args.len() {
                    return Err("--repo requires a path".to_string());
                }
                repo_path = Some(PathBuf::from(&args[i]));
            }
            "--issue" | "-i" => {
                i += 1;
                if i >= args.len() {
                    return Err("--issue requires a path".to_string());
                }
                issue_path = Some(PathBuf::from(&args[i]));
            }
            "--max-attempts" | "-n" => {
                i += 1;
                if i >= args.len() {
                    return Err("--max-attempts requires a number".to_string());
                }
                max_attempts = args[i]
                    .parse()
                    .map_err(|e| format!("--max-attempts: {e}"))?;
            }
            "--fixed-rounds" | "-k" => {
                i += 1;
                if i >= args.len() {
                    return Err("--fixed-rounds requires a number".to_string());
                }
                fixed_rounds = Some(
                    args[i]
                        .parse()
                        .map_err(|e| format!("--fixed-rounds: {e}"))?,
                );
            }
            "--model" | "-m" => {
                i += 1;
                if i >= args.len() {
                    return Err("--model requires a model name".to_string());
                }
                model_override = Some(args[i].clone());
            }
            "--open-pr" | "--pr" => {
                open_pr = true;
            }
            "--patch" | "-p" => {
                i += 1;
                if i >= args.len() {
                    return Err("--patch requires a path".to_string());
                }
                patch_path = Some(PathBuf::from(&args[i]));
            }
            "--test-cmd" => {
                i += 1;
                if i >= args.len() {
                    return Err("--test-cmd requires a command".to_string());
                }
                test_cmd = Some(args[i].clone());
            }
            "--sandbox-env" => {
                i += 1;
                if i >= args.len() {
                    return Err("--sandbox-env requires KEY=VALUE".to_string());
                }
                let pair = vanta::sandbox::parse_sandbox_env(&args[i])
                    .map_err(|e| format!("--sandbox-env: {e}"))?;
                sandbox_env.push(pair);
            }
            "--precompile" => {
                i += 1;
                if i >= args.len() {
                    return Err("--precompile requires a tree-relative .py path".to_string());
                }
                let path = vanta::sandbox::parse_precompile_path(&args[i])
                    .map_err(|e| format!("--precompile: {e}"))?;
                precompile.push(path);
            }
            "--original-tree" => {
                i += 1;
                if i >= args.len() {
                    return Err("--original-tree requires a path".to_string());
                }
                original_tree = Some(PathBuf::from(&args[i]));
            }
            "--rerun-signals" => {
                i += 1;
                if i >= args.len() {
                    return Err("--rerun-signals requires a session id".to_string());
                }
                rerun_signals = Some(args[i].clone());
            }
            "--good-tree" => {
                i += 1;
                if i >= args.len() {
                    return Err("--good-tree requires a path".to_string());
                }
                good_tree = Some(PathBuf::from(&args[i]));
            }
            "--base-tree" => {
                i += 1;
                if i >= args.len() {
                    return Err("--base-tree requires a path".to_string());
                }
                base_tree = Some(PathBuf::from(&args[i]));
            }
            "--allow-backend-upgrade" => {
                allow_backend_upgrade = true;
            }
            "--select-context" => {
                select_context = true;
            }
            "--json" => {
                json = true;
            }
            "--select-role" => {
                i += 1;
                if i >= args.len() {
                    return Err("--select-role requires a role".to_string());
                }
                select_role = Some(args[i].clone());
            }
            "--select-budget" => {
                i += 1;
                if i >= args.len() {
                    return Err("--select-budget requires a number".to_string());
                }
                select_budget = Some(
                    args[i]
                        .parse()
                        .map_err(|e| format!("--select-budget: {e}"))?,
                );
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                // Allow positional: first positional = repo
                if repo_path.is_none() {
                    repo_path = Some(PathBuf::from(other));
                } else {
                    return Err(format!("unknown argument: {other}"));
                }
            }
        }
        i += 1;
    }

    // [VANTA-WHY] `--repo` is only required for the run path. In `--rerun-signals`
    // mode the session artifacts and the good tree ARE the inputs, and demanding
    // a repo would force the caller to pass a path the rerun never reads.
    if rerun_signals.is_none() && repo_path.is_none() {
        return Err("missing --repo PATH".to_string());
    }
    // `--select-context` is also a different program shape (see the module note):
    // it needs the tree and nothing else the run path requires.
    if select_context && select_budget.is_none() {
        return Err("--select-context requires --select-budget N".to_string());
    }
    if select_context && select_role.is_none() {
        return Err("--select-context requires --select-role ROLE".to_string());
    }

    Ok(CliArgs {
        repo_path,
        issue_path,
        max_attempts,
        fixed_rounds,
        model_override,
        open_pr,
        patch_path,
        test_cmd,
        sandbox_env,
        precompile,
        original_tree,
        rerun_signals,
        good_tree,
        base_tree,
        allow_backend_upgrade,
        select_context,
        json,
        select_role,
        select_budget,
    })
}

// ============================================================================
// `--select-context` — select, then print; never a pipeline
// ============================================================================
/// What the selector read OUT of the issue text, per signal (§1.6.7).
///
/// [VANTA-WHY] The report says which files the ranking *reached* and never a word
/// about what the reading found. That is the difference between "the ranking put
/// `pandas/core/dtypes/common.py` first" and "the issue named `is_string_dtype` and
/// `common.py` defines it" — and a reader auditing a ranking needs the second. Each
/// field is one signal's extraction, taken from the same readers the selection
/// itself uses, so the record cannot describe a reading the selection did not make.
///
/// The `bm25` fallback's query is `identifiers`' names: `issue_code_tokens` is
/// `issue_identifiers(..).into_keys()`, so the code-shaped tokens are recorded once,
/// beside the rule that made each one code.
#[derive(serde::Serialize)]
struct IssueSignals {
    /// The path-shaped tokens of the issue (§1.6.7) — what it named as a file.
    paths: Vec<String>,
    /// The file paths a traceback inside the issue names.
    traceback_files: Vec<String>,
    /// The names the issue writes as code, each with the rule that made it one.
    identifiers: Vec<IdentifierSignal>,
}

#[derive(serde::Serialize)]
struct IdentifierSignal {
    name: String,
    rule: &'static str,
}

/// The JSON artifact the harness's context gate reads.
///
/// [VANTA-WHY] `fits` is computed HERE, from the same constants the run's guard
/// uses, rather than in Python: the whole point of the subcommand is that the
/// gate stops carrying its own copy of the walk AND of the estimate. The gate
/// still owns the *window* it passes in as `--select-budget`.
///
/// `grounded` is here for the same reason (§1.6.10): whether the selection carries
/// content the run's own inputs grounded is a claim about the selector's reason
/// vocabulary, so the gate asks the selector instead of inferring it from
/// `file_count`.
#[derive(serde::Serialize)]
struct SelectContextReport<'a> {
    role: &'a str,
    mode: &'a str,
    budget_tokens: u32,
    files: &'a [vanta::context::SelectedFile],
    file_count: usize,
    grounded: bool,
    selected_bytes: usize,
    selected_tokens: u32,
    tree_count: usize,
    tree_bytes: usize,
    tree_tokens: u32,
    dropped_for_budget: &'a [vanta::context::DroppedFile],
    skipped: &'a [vanta::context::SkipRecord],
    /// What the reading found, per signal (§1.6.7) — not what it reached.
    signals: IssueSignals,
    issue_tokens: u32,
    prompt_frame_tokens: u32,
    prompt_tokens_estimate: u32,
    fits: bool,
}

fn run_select_context(cli: &CliArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use vanta::context;

    let tree = cli
        .repo_path
        .as_ref()
        .ok_or("--select-context requires --repo PATH (the tree to select from)")?;
    let role = cli
        .select_role
        .as_deref()
        .ok_or("--select-context requires --select-role ROLE")?;
    let budget_tokens = cli
        .select_budget
        .ok_or("--select-context requires --select-budget N")?;

    // The issue text is optional only because a tree with no issue still has a
    // selection (everything scores 0 and the tree listing fills the budget). An
    // unreadable `--issue` is an error, not an empty string: silently treating a
    // typo'd path as "no issue" would report a selection from the wrong prompt.
    let issue_text = match &cli.issue_path {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| format!("--issue {}: {e}", path.display()))?,
        None => String::new(),
    };
    let patch_text = match &cli.patch_path {
        Some(path) => Some(
            std::fs::read_to_string(path)
                .map_err(|e| format!("--patch {}: {e}", path.display()))?,
        ),
        None => None,
    };

    let selection = context::select_for_role(
        tree,
        &issue_text,
        patch_text.as_deref(),
        role,
        budget_tokens,
    );

    let issue_tokens = context::estimate_tokens(issue_text.len());
    let prompt_tokens_estimate = context::prompt_tokens_estimate(&selection, issue_tokens);
    // The extraction itself, from the selector's own readers (§1.6.7): the same
    // functions the ranking matches with, so the record cannot describe a reading
    // the selection did not make.
    let signals = IssueSignals {
        paths: context::issue_path_tokens(&issue_text).into_iter().collect(),
        traceback_files: context::issue_traceback_files(&issue_text)
            .into_iter()
            .collect(),
        identifiers: context::issue_identifiers(&issue_text)
            .into_iter()
            .map(|(name, rule)| IdentifierSignal {
                name,
                rule: rule.as_str(),
            })
            .collect(),
    };
    let report = SelectContextReport {
        role: &selection.role,
        mode: &selection.mode,
        budget_tokens: selection.budget_tokens,
        files: &selection.files,
        file_count: selection.files.len(),
        grounded: selection.grounded(),
        selected_bytes: selection.total_bytes,
        selected_tokens: selection.total_tokens,
        tree_count: selection.tree.len(),
        tree_bytes: selection.tree_bytes,
        tree_tokens: selection.tree_tokens,
        dropped_for_budget: &selection.dropped_for_budget,
        skipped: &selection.skipped,
        signals,
        issue_tokens,
        prompt_frame_tokens: context::PROMPT_FRAME_TOKENS,
        prompt_tokens_estimate,
        fits: prompt_tokens_estimate <= budget_tokens,
    };

    if cli.json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        println!(
            "selection: mode={} role={} budget={} tok, {} file(s), {} tok, grounded={}, fits={}",
            report.mode,
            report.role,
            report.budget_tokens,
            report.file_count,
            report.prompt_tokens_estimate,
            report.grounded,
            report.fits,
        );
        for line in selection.summary_lines() {
            println!("{line}");
        }
        if !report.dropped_for_budget.is_empty() {
            println!(
                "  (dropped {} selected file(s) for budget)",
                report.dropped_for_budget.len()
            );
        }
        if !report.skipped.is_empty() {
            println!("  (pruned {} path(s))", report.skipped.len());
        }
    }
    Ok(())
}

fn print_help() {
    println!(
        "Vanta M3 — Adversarial verification repair loop\n\
         \n\
         USAGE:\n    vanta --repo PATH [--issue PATH] [--max-attempts N] [--fixed-rounds K] [--model NAME] [--open-pr] [--patch PATH] [--test-cmd CMD] [--original-tree PATH]\n    \
         vanta --rerun-signals SESSION_ID --good-tree PATH [--base-tree PATH] [--allow-backend-upgrade]\n    \
         vanta --select-context --json --repo TREE --select-role ROLE --select-budget N [--issue PATH] [--patch PATH]\n\
         \n\
         OPTIONS:\n    --repo, -r PATH        Path to the repository to repair\n    \
         --issue, -i PATH       Path to the issue/bug description (default: auto-detect)\n    \
         --max-attempts, -n N   Max repair attempts (default: 4)\n    \
         --fixed-rounds, -k K   Run EXACTLY K repair attempts and take the verdict from attempt K, so a patch that passes early is re-examined (default: off). Repair only; ignored with --patch\n    \
         --model, -m NAME       Override LLM model (default: from VANTA_LLM_MODEL)\n    \
         --open-pr              Open a GitHub PR after Gatekeeper approval\n    \
         --patch, -p PATH       Verify an EXISTING patch (skip Creator, run Attacker+BATE on this diff)\n    \
         --test-cmd CMD         Test command (default: 'python3 -m pytest'). e.g. 'pytest tests/ -q'\n    \
         --sandbox-env K=V     Inject K=V into every sandbox command of the run (repeatable). PYTHONPATH or MPLBACKEND only; refused for any other key\n    \
         --precompile PATH      Byte-compile PATH (tree-relative .py) in every suite VM before the test command runs (repeatable). A path outside the tree, a .py-less name or a non-.py file is refused\n    \
         --original-tree PATH   Clean original tree (git worktree of base commit) for Mode B two-tree mode\n    \
         --rerun-signals ID     Re-run session ID's review-mode rejection signals against --good-tree (writes vanta_sessions/ID_signal_reruns.json; no --repo needed)\n    \
         --good-tree PATH       Known-good source: a directory, or a patch file applied onto --base-tree\n    \
         --base-tree PATH       Base tree the --good-tree patch applies to (required only for a patch file)\n    \
         --allow-backend-upgrade  Re-run a subprocess-recorded session under shuru (refused if test_cmd names a host path)\n    \
         --select-context       Print the source selection a role would be shown, and exit (no pipeline). The harness's context gate runs this so it measures the run's own walk\n    \
         --select-role ROLE     With --select-context: the role label to select for\n    \
         --select-budget N      With --select-context: the prompt budget in tokens\n    \
         --json                 With --select-context: print the selection as JSON\n    \
         --help, -h             Show this help\n"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        std::iter::once("vanta".to_string())
            .chain(items.iter().map(|item| item.to_string()))
            .collect()
    }

    /// [VANTA-WHY] `--select-context`'s inputs are the ones a real run has, and
    /// the set is pinned here so ADDING one is a deliberate, reviewed act. This is
    /// the structural half of "the selector cannot see the answer": the other half
    /// is `vanta::context::select_for_role`'s signature, which has no parameter for
    /// a hidden test, the fix commit, the fix diff or `bug.info` either.
    #[test]
    fn the_select_context_surface_is_exactly_a_real_runs_inputs() {
        let mut expected = vec![
            "--select-context",
            "--json",
            "--repo",
            "--issue",
            "--patch",
            "--select-role",
            "--select-budget",
        ];
        let mut actual = SELECT_CONTEXT_FLAGS.to_vec();
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(
            actual, expected,
            "a flag added to `--select-context` is a way to feed the selector \
             information a run does not have"
        );
    }

    /// The names of the hidden information the selector must not be able to read.
    /// Each must be rejected as an unknown argument — a flag that parsed would be
    /// a channel from the harness's answer key into the prompt.
    #[test]
    fn the_select_context_parser_has_no_flag_for_the_answer() {
        for flag in ["--f2p", "--hidden-tests", "--fix-commit", "--fix-diff", "--bug-info"] {
            let error = match parse_args_from(&args(&[
                "--select-context",
                "--repo",
                "tree",
                "--select-role",
                "creator",
                "--select-budget",
                "1000",
                flag,
                "value",
            ])) {
                Ok(_) => panic!("{flag} must not be a --select-context input"),
                Err(error) => error,
            };
            assert!(error.contains("unknown argument"),
                "{flag} must be rejected as unknown; got: {error}"
            );
        }
    }

    #[test]
    fn the_select_context_parser_reads_the_five_inputs() {
        let cli = parse_args_from(&args(&[
            "--select-context",
            "--json",
            "--repo",
            "/tmp/tree",
            "--issue",
            "/tmp/issue.md",
            "--patch",
            "/tmp/fix.patch",
            "--select-role",
            "creator+attacker",
            "--select-budget",
            "163584",
        ]))
        .unwrap();
        assert!(cli.select_context && cli.json);
        assert_eq!(cli.repo_path, Some(PathBuf::from("/tmp/tree")));
        assert_eq!(cli.issue_path, Some(PathBuf::from("/tmp/issue.md")));
        assert_eq!(cli.patch_path, Some(PathBuf::from("/tmp/fix.patch")));
        assert_eq!(cli.select_role.as_deref(), Some("creator+attacker"));
        assert_eq!(cli.select_budget, Some(163_584));
    }

    /// Without `--patch` the selection is repair mode; with it, review mode. The
    /// mode is not a flag — it is a consequence of what the run was given.
    #[test]
    fn the_mode_follows_the_patch_and_not_a_flag() {
        let repair = parse_args_from(&args(&[
            "--select-context", "--repo", "tree", "--select-role", "creator",
            "--select-budget", "1000",
        ]))
        .unwrap();
        assert!(repair.patch_path.is_none());
        let review = parse_args_from(&args(&[
            "--select-context", "--repo", "tree", "--patch", "fix.patch",
            "--select-role", "creator", "--select-budget", "1000",
        ]))
        .unwrap();
        assert_eq!(review.patch_path, Some(PathBuf::from("fix.patch")));
    }

    #[test]
    fn select_context_refuses_a_missing_budget_or_role() {
        assert!(parse_args_from(&args(&["--select-context", "--repo", "tree"])).is_err());
        assert!(parse_args_from(&args(&[
            "--select-context", "--repo", "tree", "--select-role", "creator",
        ]))
        .is_err());
    }

    /// [VANTA-WHY] The recorded pre-compile rule has to reach the run the same
    /// way `--sandbox-env` does: one flag per file, the list arriving in the
    /// order the split recorded it (so a run's argv is a readable record of the
    /// rule it was given).
    #[test]
    fn the_precompile_flag_is_repeatable_and_keeps_its_order() {
        let cli = parse_args_from(&args(&[
            "--repo", "tree",
            "--precompile", "lib/matplotlib/widgets.py",
            "--precompile", "lib/matplotlib/other.py",
        ]))
        .unwrap();
        assert_eq!(
            cli.precompile,
            vec![
                "lib/matplotlib/widgets.py".to_string(),
                "lib/matplotlib/other.py".to_string(),
            ]
        );
    }

    /// The rule is a path the compile step has to be able to honour. Every one of
    /// these would either compile something the benchmark does not ship or fail
    /// silently inside the VM, so each is refused at the flag rather than passed
    /// through — the same shape of refusal `--sandbox-env` makes about keys.
    #[test]
    fn the_precompile_flag_refuses_a_path_it_could_not_compile() {
        for path in [
            "/etc/passwd.py",
            "../outside.py",
            "lib/../../outside.py",
            "lib\\matplotlib\\widgets.py",
            "lib/matplotlib/widgets.txt",
            "lib/matplotlib",
            "",
        ] {
            let error = match parse_args_from(&args(&["--repo", "tree", "--precompile", path])) {
                Ok(_) => panic!("--precompile {path:?} must be refused"),
                Err(error) => error,
            };
            assert!(
                error.starts_with("--precompile:"),
                "{path:?} must be refused with the flag named; got: {error}"
            );
        }
        // A run that records nothing is still valid and carries no rule, so the
        // invocation is exactly the one that predates the flag.
        let cli = parse_args_from(&args(&["--repo", "tree"])).unwrap();
        assert!(cli.precompile.is_empty());
    }

    /// [VANTA-WHY] M3 item 9 — `--fixed-rounds K`. Off unless asked for, read
    /// from either spelling, and refused where no number follows. The flag is
    /// parsed here and INTERPRETED by the driver (`resolve_fixed_rounds`), which
    /// is where the `--patch` refusal lives; the parser's only job is to carry
    /// the number through as written.
    #[test]
    fn the_fixed_rounds_flag_is_off_by_default_and_reads_a_number() {
        // Off by default: every run before this option is unchanged.
        let cli = parse_args_from(&args(&["--repo", "tree"])).unwrap();
        assert_eq!(cli.fixed_rounds, None);

        let cli = parse_args_from(&args(&["--repo", "tree", "--fixed-rounds", "4"])).unwrap();
        assert_eq!(cli.fixed_rounds, Some(4));

        // The short spelling is the same flag.
        let cli = parse_args_from(&args(&["--repo", "tree", "-k", "2"])).unwrap();
        assert_eq!(cli.fixed_rounds, Some(2));

        // Both spellings, with nothing to read.
        for flag in ["--fixed-rounds", "-k"] {
            let error = match parse_args_from(&args(&["--repo", "tree", flag])) {
                Ok(_) => panic!("{flag} with no number must be refused"),
                Err(error) => error,
            };
            assert!(
                error.contains("requires a number"),
                "{flag} must name what is missing; got: {error}"
            );
        }

        // A value that is not a number is refused with the flag named.
        let error = match parse_args_from(&args(&["--repo", "tree", "--fixed-rounds", "four"])) {
            Ok(_) => panic!("--fixed-rounds four must be refused"),
            Err(error) => error,
        };
        assert!(
            error.starts_with("--fixed-rounds:"),
            "the refusal must name the flag; got: {error}"
        );
    }
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_logging();

    let cli = parse_args().map_err(|e| format!("CLI error: {e}\n\nUse --help for usage."))?;

    // [VANTA-WHY] `--select-context` is a third program shape: it prints the
    // selection a role would be shown and exits, so the harness's context gate
    // measures the run's own walk instead of re-deriving one. It short-circuits
    // before the banner — stdout must be the JSON alone, or the gate would have
    // to strip a banner it does not know the shape of.
    if cli.select_context {
        return run_select_context(&cli);
    }

    // [VANTA-WHY] The rerun is a DIFFERENT program shape: it reads a finished
    // session's artifacts instead of driving a new run, so it short-circuits
    // before the banner and the driver. It never needs --repo, and it must not
    // fall through to the repair loop (which would need one).
    if let Some(session_id) = cli.rerun_signals {
        let good_tree = cli
            .good_tree
            .ok_or("--rerun-signals requires --good-tree PATH")?;
        let config = vanta::pipeline::rerun::RerunConfig {
            session_id,
            good_tree,
            base_tree: cli.base_tree,
            allow_backend_upgrade: cli.allow_backend_upgrade,
            sessions_dir: PathBuf::from("vanta_sessions"),
        };
        match vanta::pipeline::rerun::rerun_signals(config).await {
            Ok(path) => {
                println!("▌ Signal reruns written to {}", path.display());
                return Ok(());
            }
            Err(error) => {
                eprintln!("rerun-signals failed: {error}");
                std::process::exit(1);
            }
        }
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VANTA M3 — Repair Loop (Creator → Attacker → BATE → Governor)║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let repo_path = cli
        .repo_path
        .ok_or("missing --repo PATH")?;

    let decision = driver::run(RunConfig {
        repo_path,
        issue_path: cli.issue_path,
        max_attempts: cli.max_attempts,
        fixed_rounds: cli.fixed_rounds,
        model_override: cli.model_override,
        open_pr: cli.open_pr,
        patch_path: cli.patch_path,
        test_cmd: cli.test_cmd,
        sandbox_env: cli.sandbox_env,
        precompile: cli.precompile,
        original_tree: cli.original_tree,
    })
    .await?;

    // Exit codes: 0 = approved, 2 = infrastructure failure, 3 = needs review,
    // 4 = judge failure, 1 = any other negative verdict. [VANTA-WHY] An
    // InfraFailure is not a verdict about the candidate at all — no Attacker
    // test ever judged the patch — so it must be distinguishable from a plain
    // rejection by whatever script drives this binary. It also gets its own exit
    // code so a harness can retry the target without mistaking it for a clean
    // "no".
    //
    // [VANTA-WHY] §14.3. `NeedsReview` gets code 3 for the same reason: it is
    // neither the approval (0) nor a rejection (1). A harness that treated it as
    // a rejection would count an uncorroborated failure as a false reject, which
    // is exactly the mislabel this change exists to remove — so it must be its
    // own outcome on the process boundary too, not folded into `_`.
    //
    // [VANTA-WHY] M3 item 7. The judge arms reuse the verdict conventions the
    // harness already reads: `JudgeApproved` is the approval (0), exactly as the
    // `AllTestsPassed` it is labelled onto, and `JudgeRejected` is the rejection
    // (1), exactly as the `Rejected` it is labelled onto — so no caller's
    // approve/reject reading changes just because the verdict came from the judge
    // rather than from a test. `JudgeFailed` gets its OWN code (4): the model
    // answered with something nobody can read as a verdict, which is neither an
    // approval, a rejection, nor an infrastructure fault (the machine worked
    // fine), so a harness must be able to tell it apart — the decision doc
    // records that it must not be retried as infrastructure nor counted as a
    // harness error.
    match decision {
        GovernorDecision::AllTestsPassed => Ok(()),
        GovernorDecision::JudgeApproved => Ok(()),
        GovernorDecision::InfraFailure => std::process::exit(2),
        GovernorDecision::NeedsReview => std::process::exit(3),
        GovernorDecision::JudgeFailed => std::process::exit(4),
        _ => std::process::exit(1),
    }
}
