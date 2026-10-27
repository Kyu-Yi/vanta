// ============================================================================
// ATTACKER AGENT — patch → adversarial tests with declared intent (Spec §4)
// ============================================================================
//
// Spec §4: "Attacker agent: Given a patch, writes adversarial tests with
// declared intent targeting its weak points."
//
// Spec §5 Test Intents: each test declares WHY it exists:
//   - BugReproduction: proves the reported defect exists (FAIL on original)
//   - FeatureAcceptance: checks a new capability works
//   - BehaviorPreservation: checks unrelated working behaviour survived
//
// M1 scope (per Master Plan M1 BUILD):
//   - AttackerRequest: patch, original_source, issue_text
//   - GeneratedTest: test_name, test_source, intent, reason
//   - TestIntent enum (reused from core::TestIntent)
// ============================================================================

use crate::config::LlmConfig;
use crate::core::TestIntent;
use crate::llm::{
    build_reqwest_client, extract_json_candidate, structured_call_with_retry,
    PARSE_FAILURE_PREFIX, TRUNCATED_OUTPUT_PREFIX,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, info, trace, warn};

// ============================================================================
// ATTACKER_PROMPT_VERSION — the run-identity tag for the Attacker's prompt
// ============================================================================
//
// [VANTA-WHY] The Attacker prompt is embedded in the binary with no other
// version tag, so two runs on the same commit that differ only in the prompt
// are indistinguishable in the session log. That matters here: a run BEFORE the
// boundary-probing rule and a run AFTER it are different experiments, and the
// boundary rule is exactly what the catch-rate comparison is about. Bump this
// string whenever the prompt text changes, so `session.start` names which
// prompt produced the rows.
//
// [VANTA-WHY] The tag is a bare version ("2.boundary-probe"), not
// "attacker.2.boundary-probe": `session.start` writes
// `prompt_version=attacker=<tag>;intent_check=<tag>;...`, so the prompt NAME is
// the key and repeating it inside the value would make the field ambiguous.
//
// [VANTA-WHY] Bumped for the prompt audit (plans/bugsinpy.md §14.10, A2–A18):
// every Attacker prompt — generate, revise, simplify — now states that a result
// on the PATCHED tree is an expectation about a correct patch, not a rule to be
// satisfied, and that a failure there is the finding. A row logged before this
// tag and a row after it come from different prompts.
//
// [VANTA-WHY] Bumped again for §14.10 f, which is a bump that is LATE rather
// than new: `3588728` (the A3 follow-up — the three priority items overlap, and a
// leftover slot defaults to the preserved behaviour) and `d4ea314` (the A10
// follow-up — the simplify prompt stopped licensing "keep the label, drop the
// assertion") both reworded Attacker prompts after the `a2617b3` bump and neither
// bumped. Rows written between those two commits therefore carry the
// `6.expected-on-a-correct-patch` tag for two different prompt texts. This tag
// covers the current text; the content hash recorded beside it is what makes the
// miss recoverable in the log rather than invisible.
//
// [VANTA-WHY] Bumped for §14.10 A21: the generate prompt gained a TERMINAL
// FALLBACK after the leftover-slot default. That default ("an input where the
// ORIGINAL source returns a DIFFERENT result than at the inputs you have already
// tested") has no answer once every meaningful input is in the shared set, and a
// round-3 reply on the toy looped 45 times on "Let me pick the three most valuable
// tests" / "Both are shared" / "-1 is shared". The reworded line is part of the
// prompt text, so the tag moves with it.
//
// [VANTA-WHY] Bumped again for the rest of A21, whose prompt edits land together
// under this one tag. `745cafb` reworded the round >= 2 user-prompt section so
// "DO NOT REPEAT" and the terminal fallback read as ONE rule (a bare prohibition
// pulled against the fallback in exactly the late-round state the fallback exists
// for), and the ONE PASS line bounds the plan to a single pass over the
// candidates so the 45x re-listing loop has no work to do.
//
// [VANTA-WHY] Bumped for the context-localization change (§1.6.2): the Attacker's
// system text is unchanged, but the source it is shown is not — the run now
// ranks the tree and shows the selected files plus the repo's file tree instead
// of the first ~262 KB in `read_dir` order. Which code a probe was written
// against is part of the experiment's identity.
pub const ATTACKER_PROMPT_VERSION: &str = "10.localized-source-context";

// ============================================================================
// INTENT_AUDITOR_PROMPT_VERSION — the run-identity tag for the intent check
// ============================================================================
//
// [VANTA-WHY] The intent auditor (audit_test_intent) is a SEPARATE prompt from
// the Attacker's and it is the one that decides whether a test is honest:
// demoting non-preservation labels and scrapping preservation probes the issue
// requires to change. A change to THAT wording changes what a run accepts, so
// it needs its own tag — otherwise editing the auditor would be logged as "no
// change" because the Attacker's tag is untouched.
//
// [VANTA-WHY] Bumped for the prompt audit (plans/bugsinpy.md §14.10, A7–A9):
// the auditor's headings no longer promise a question it may not ask, its
// grounding rule names the question and the test it closes, and the batch rule
// points at the numbered test instead of an ambiguous "this".
//
// [VANTA-WHY] Bumped for the auditor-truncation batch (§14.10 a–c): the
// preservation case of question 1 now has a decided answer instead of a debate,
// question 3 is asked only for a BugReproduction test, and "without changing X"
// is scoped to every input the issue does not ask to change. The wording changed
// in all four auditor prompts, so the auditor's own tag has to move — the run's
// budget behaviour changed with it (21 truncations in the 12-run Ornith-1.5 toy
// run that motivated the batch).
//
// [VANTA-WHY] Bumped for the context-localization change (§1.6.2), same reason as
// the Attacker's: the auditor's system text is unchanged, but it now reads the
// selected files and the repo file tree instead of the capped dump, and what the
// auditor is shown is what it judges against.
pub const INTENT_AUDITOR_PROMPT_VERSION: &str = "6.localized-source-context";

// ============================================================================
// AttackerRequest — input to the Attacker agent
// ============================================================================
//
// [VANTA-WHY] `existing_tests` is the shared set accumulated by PREVIOUS
// attempts (Spec §8). From round 2 on the Attacker is shown the probes it has
// already written so it can attack somewhere new: without this, every round
// re-derives the same boundary test, the fresh batch adds no coverage, and the
// run burns its attack depth re-running one probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackerRequest {
    pub patch_text: String,
    pub original_source: String,
    pub issue_text: String,
    /// Probes already in the fixed shared set. Empty on round 1.
    #[serde(default)]
    pub existing_tests: Vec<SharedProbe>,
}

/// One probe already in the shared set, as shown to the Attacker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SharedProbe {
    pub test_name: String,
    pub test_source: String,
}

// ============================================================================
// GeneratedTest — one adversarial test with DECLARED INTENT
// ============================================================================
//
// [VANTA-WHY] The `intent` field is what makes BATE (Phase 1) possible.
// Without it, you cannot judge whether a test's baseline result is correct.
// A BugReproduction test SHOULD fail on original code; a BehaviorPreservation
// test SHOULD pass. Mismatch → test is invalid → scrap after 3 retries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GeneratedTest {
    pub test_name: String,
    pub test_source: String,
    pub intent: TestIntent,
    pub reason: String,
    pub issue_quote: Option<String>,
    /// [VANTA-WHY] Item 6 — the at-most-one-relabel ceiling. A relabel is a
    /// verdict about the test's LABEL, and each intent has exactly one other
    /// label an audit can move it to (`BehaviorPreservation` ↔
    /// `FeatureAcceptance`, any non-preservation → `BehaviorPreservation`), so a
    /// SECOND relabel of the same test is necessarily a flip BACK to where it
    /// came from. Two verdicts that disagree about the same test would otherwise
    /// let it oscillate until it lands on the label BATE happens to accept; this
    /// flag makes the second one a scrap instead. `#[serde(default)]` because the
    /// Attacker never emits it — it is set by the audit, not by the model.
    #[serde(default)]
    pub relabeled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntentRelabel {
    pub test_name: String,
    pub from_intent: TestIntent,
    pub to_intent: TestIntent,
    pub reason: String,
}

// ============================================================================
// AttackerResponse — typed output: array of GeneratedTest
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttackerResponse {
    pub tests: Vec<GeneratedTest>,
}

// ============================================================================
// AttackerTestCase — M0 spike compatibility (simpler shape)
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttackerTestCase {
    pub test_name: String,
    pub test_source: String,
    pub expect_pass: bool,
}

// ============================================================================
// ATTACKER_SYSTEM_PROMPT — the generation prompt, kept as a const
// ============================================================================
//
// [VANTA-WHY] Hoisted out of `attacker_generate_tests` so its rules are a
// directly testable artifact (see the `prompt_rules` unit tests), exactly as
// INTENT_AUDITOR_SYSTEM_PROMPT already is. A rule that lives only in a local
// `let` can be dropped in a refactor with no test to notice; the version tag at
// the top of this file is only meaningful if the text it names is pinned too.
pub const ATTACKER_SYSTEM_PROMPT: &str = r#"You are Vanta Attacker. Given a patch, write adversarial tests designed to BREAK it. Each test MUST declare an intent. Save output budget for the JSON answer.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.

Required JSON shape:
{
  "tests": [
    {
      "test_name":   string,
      "test_source": string — MUST be a complete pytest function: `def test_name():` with assertions inside the function body. NO module-level statements. Import at top, then `def test_...():` with body. Example:
                       import pytest
                       from cart import cart_total

                       def test_example():
                           assert cart_total([(100, 2)]) == 200

                       IMPORTANT: Do NOT import constants or symbols that only exist in the PATCHED code (e.g. MAX_TOTAL). Use literal values instead so the test can run on BOTH original and patched code.
      "intent":      "BugReproduction" | "FeatureAcceptance" | "BehaviorPreservation",
      "reason":      string — why this test targets the patch,
      "issue_quote": string|null — for BugReproduction or FeatureAcceptance, quote the issue sentence you believe asks for this behaviour at the input you assert (a test asserting `shipping_band(500)` needs a quote about 500). The quote is RECORDED as evidence for the reader; it does not authorise the label — a patch-blind auditor, not your quote, decides whether the issue asks for that behaviour. For BehaviorPreservation use null
    }
  ]
}

Intent rules — every rule below describes what a CORRECT patch produces; it is an EXPECTATION, never a requirement the patch must satisfy, and never a reason to change the test. Each test is expected to produce a DIFFERENT result on original vs patched (except BehaviorPreservation): if the PATCHED code does not give a test its expected result, that is a FINDING about the patch, not a mistake in the test.
- BugReproduction: reproduces a bug that the patch fixes. On a correct patch it is FAIL on the ORIGINAL code and PASS on the PATCHED code — if it FAILS on the PATCHED code, that failure IS the finding. It exercises the exact code path the patch changes.
- FeatureAcceptance: checks a NEW capability added by the patch. On a correct patch: FAIL on the ORIGINAL code (feature absent) and PASS on the PATCHED code (feature present) — if it FAILS on the PATCHED code, that failure IS the finding.
- BehaviorPreservation: checks EXISTING behavior the issue does NOT ask to change. The test MUST PASS on the ORIGINAL code (BATE checks this). It is EXPECTED to pass on a correct patch — but that is not a requirement you may test the patch against: if it FAILS on the PATCHED code, that failure IS the finding, because a bad patch breaking existing behavior is exactly what this test exists to catch. NEVER drop a boundary probe because you expect the patch to break it. Test only functionality that existed before the patch.

CRITICAL:
- Every test must actually import and call the patched module's functions/classes. Do NOT mock away the code under test.
- Do NOT import symbols that only exist in the PATCHED code. Use hasattr/getattr or try/except to detect new features safely on original code.
- Use assertions (assert X == Y) so failures are clean, not crashes — EXCEPT when the ISSUE names the symptom the bug produces (a `NameError`, an exact message, a specific exception type): then the test must actually trigger THAT symptom, because the reported failure is what ties the test to the issue. An assertion on a neighbouring value still raises a bare AssertionError and never reproduces the symptom the issue describes.
- Tests run from the REPOSITORY ROOT, with the repo root and its `src/` directory already on `sys.path` (the harness writes a root `conftest.py` that inserts both). Import the package by its own top-level name; the `[IMPORT PATH HINT]` line in the ISSUE, when present, IS this repo's import path. Never assume an environment variable such as PYTHONPATH is set — it is not.
- Expected values come from the ISSUE and the ORIGINAL code's existing behaviour, never from the patch alone.
- BATE may run your test more than once against the ORIGINAL code. A test whose result changes between those runs — time.sleep, a timeout, randomness, network, iteration order, shared state — is flaky and will be scrapped. Keep every test deterministic.
- Label a test BugReproduction or FeatureAcceptance only when you believe the issue asks for that behaviour AT THAT INPUT; if the issue does not ask for it, label it BehaviorPreservation. A patch-blind auditor checks that belief, not your quote — the quote is evidence for the reader.
- A patch that changes behaviour is NOT evidence the issue wanted the change — "the patch does X" is never a reason to label a test FeatureAcceptance.

IMPORT RULES (violating these makes the test ERROR on collection — worse than failing):
- Only import a name you can LITERALLY SEE defined in the ORIGINAL SOURCE above, and import it from the file where it is defined. Do NOT guess that a symbol lives in a module.
- The ORIGINAL SOURCE may be a NARROWED excerpt of the repository (review mode shows the code the fix touches, not the whole tree). A name the excerpt does not show is UNKNOWN to you, not absent from the runtime: fetch it from the module object with getattr instead of concluding it does not exist or inventing the module it lives in.
- When unsure, import the MODULE and access the attribute defensively, e.g. `import pkg.module as mod; fn = getattr(mod, "name", None); assert fn is not None`. Never write `from m import name` for a name you have not seen defined.
- A test whose import raises is treated as an infra ERROR, not a clean failure — it is scrapped.

NO-REIMPLEMENTATION RULE:
- Never re-implement the patch's logic inside the test body (e.g. re-setting an env var yourself and asserting it is set). Such a test always passes on BOTH versions and proves nothing. Call the repo's real code instead.

INTENT-vs-API RULE:
- BehaviorPreservation may ONLY exercise APIs that already exist in the ORIGINAL source. That source may be an excerpt (see IMPORT RULES): a name it does not show may still exist at runtime, so reach it with getattr rather than assuming the patch adds it. If the behaviour you want to check uses a symbol the patch ADDS, it is a FeatureAcceptance test, not BehaviorPreservation.
- If the issue does not ask for that new behaviour either, DROP the probe. A test that cannot be labelled honestly is not a test: any label you could give it asserts something the issue never asked for, and the patch-blind audit scraps it anyway.

ONE INTENT PER TEST:
- Each test declares exactly ONE intent. Do NOT mix a new-feature assertion with a preservation assertion in the same test. A test that both checks a NEW value (e.g. a new field defaults to None) AND asserts that some OTHER call is "unchanged as before" has no honest single label: split it into two tests — one FeatureAcceptance/BugReproduction for the new value, one BehaviorPreservation for the unchanged behaviour — or drop the part you cannot label.

PRESERVATION SPECIFICITY:
- A BehaviorPreservation test targets the EXACT behaviour the issue's "must stay the same" clause names — the result the ORIGINAL code returns for the asserted input — and reads that result through the observable behaviour, not through internal call arguments. It must NOT pin the INTERNAL calls of a function the issue says to change: asserting that an inner function is called with the OLD arguments locks in internals the issue explicitly rewires, and such a test fails on a CORRECT patch. If the only probe you can write for a "must stay the same" clause pins an internal call the issue rewires, there is no preservation probe to write: cover that comparison with a BugReproduction test instead, or drop it.

TEST FIXTURES:
- Do NOT instantiate application settings/config objects that read CLI arguments, environment variables or files (a Settings class built from the command line, for example). Under pytest they parse pytest's OWN argv and exit, so the test fails on ANY code, original or patched. Use unittest.mock.MagicMock, or build the model with model_construct()/copy-update and set only the attributes the code under test reads.
- Every fake/stub/mock object MUST implement EVERY attribute the code under test touches. A fake missing an attribute (e.g. no `.shape` on a fake tensor) raises inside the fake BEFORE the assertion, so the test proves nothing about the patch.

BOUNDARY-PROBING RULE:
- A patch that widens, narrows or shifts a comparison (e.g. `<` → `<=`, or a changed threshold constant) changes behaviour ONLY at the exact boundary value. A test at a nearby interior value (e.g. 499 when the boundary is 500) returns the SAME result on original and patched code and proves nothing.
- For every comparison the patch changes, include a BehaviorPreservation test that asserts the ORIGINAL result at the EXACT boundary constant — the value named in the comparison itself, not its neighbour. Do this for every changed comparison the ISSUE does not ask to change.
- When the issue explicitly asks to change behaviour at a given boundary (e.g. "1000 should be standard"), do NOT write a preservation test for that boundary; cover it with a BugReproduction test instead.
- Example: a patch changing `if w < 500` to `if w <= 500` moves the ORIGINAL result at 500 from `"standard"` (because `< 500` is false at 500) to `"light"`. The preservation probe must therefore assert `shipping_band(500) == "standard"` — the original result AT 500 — and never assert `shipping_band(499)`, which returns the same band on both versions and proves nothing.
- Read the ORIGINAL result at the boundary out of the ORIGINAL SOURCE you were given. Never copy the patched value back as the expectation.

Write exactly 3 tests, spending them in THIS order of priority: (1) the exact comparison the patch changes, at its boundary value; (2) the behaviour the issue's "must stay the same" clause names; (3) the code path the issue asks you to change. The three items are NOT exclusive: one test may satisfy several of them at once, and you count it once, against the highest item it satisfies — when a single boundary is both the comparison the patch changes and the code path the issue asks you to change, items (1) and (3) are the SAME test, not two, and there is no third thing to find for it. Label each one honestly: use BugReproduction or FeatureAcceptance only when the issue asks for that behaviour at the input you assert, otherwise use BehaviorPreservation.
Once every comparison the patch changes and every behaviour the issue's "must stay the same" clause names are covered, spend each remaining slot on one more BehaviorPreservation probe: an input where the ORIGINAL source returns a DIFFERENT result than at the inputs you have already tested — the other side of an unchanged boundary, or the far end of a band. Take the FIRST such input you find and do not compare alternatives; when the issue names only one boundary, three BehaviorPreservation probes at three such inputs are a correct answer and you do not need a third kind of test.
TERMINAL FALLBACK: when EVERY input the defaults above point to is already covered by an existing test, so nothing new is left to find, do NOT keep searching or comparing candidates — spend each remaining slot on one more input inside a band that is already covered (a midpoint of the band, or the value on the other side of a boundary you already probed), chosen on the spot without weighing alternatives. A repeat of an existing test is an ACCEPTABLE answer here: it is discarded automatically, so the round still has its three tests, while a round that keeps hunting for a novel input loses all of them.
ONE PASS: list candidate tests at most once. Do not re-list, re-rank or re-check a candidate you have already considered; once you have three, write them.
This order and these defaults are FINAL. Choose the three tests once, write them, and do not re-derive the plan or reconsider the priority while drafting: if you identified a case and did not spend a slot on it, name what you left uncovered and why at the end of its `reason`."#;

// ============================================================================
// already_probed_section() — the round >= 2 addendum to the Attacker's user prompt
// ============================================================================
//
// [VANTA-WHY] A21 follow-up — the "do not repeat" rule and the system prompt's
// TERMINAL FALLBACK are ONE rule and must not pull apart. An unqualified "DO NOT
// REPEAT ANY OF THESE" is exactly wrong in the state the fallback exists for: a
// late round whose shared set already covers every useful input, where the model
// is told to find something new that does not exist. So the section asks for a
// probe outside the set and then says what to do when there is none left — use
// the TERMINAL FALLBACK, where a repeat is acceptable because the driver
// discards duplicates automatically. The duplicate-is-not-an-attack sentence is
// kept: it is what makes the repeat harmless rather than misleading.
fn already_probed_section(existing_tests: &[SharedProbe]) -> String {
    if existing_tests.is_empty() {
        return String::new();
    }
    let mut section = String::from(
        "\n\nALREADY IN THE SHARED TEST SET — AIM FOR A PROBE THAT IS NOT ONE OF THESE:\n",
    );
    for probe in existing_tests {
        let inputs = asserted_input_literals(&probe.test_source);
        section.push_str(&format!(
            "- {} (asserted inputs: [{}]):\n```\n{}\n```\n",
            probe.test_name,
            inputs.join(", "),
            probe.test_source
        ));
    }
    section.push_str(
        "Each test above already runs on every attempt. A new test that asserts the SAME input \
         with the SAME expected value is a duplicate, is discarded before it runs, and does not \
         count as an attack — so spend your slots on a probe that is NOT already in this set: a \
         DIFFERENT boundary, threshold, branch or input the patch touches. When every useful input \
         is already covered and nothing new is left, do NOT keep hunting for a novel input: use \
         the TERMINAL FALLBACK and spend the slot on another input inside a band that is already \
         covered. A repeat is acceptable there, because duplicates are discarded automatically \
         and the round still carries its three tests.\n",
    );
    section
}

// ============================================================================
// VANTA_ATTACKER_CONCISE (§14.22) — one extra sentence, on the first request only
// ============================================================================
//
// [VANTA-WHY] §14.13's measured run-away is a model that worked the answer out
// and then spent the cap re-checking it. §14.20 caps the answer and §14.21 caps
// the thinking; this is the cheapest possible attempt at the third cause — the
// prompt never told the model that one check is enough. It is deliberately the
// ATTACKER'S FIRST REQUEST and not the revise/simplify prompts: those are already
// answering a specific BATE result, and a bare "do not re-check" there would tell
// a revision not to look at the evidence §14.17 just gave it.
//
// [VANTA-WHY] The sentence is appended to the USER message, not to
// `ATTACKER_SYSTEM_PROMPT`, so the §14.10-f prompt-content hash is unchanged by an
// arm that only adds this sentence. That is deliberate and honest: the hash names
// the SYSTEM prompts the run sent, and this arm sends the same ones. The arm is
// still identifiable — `run.summary` records the switch and the manifest carries
// it — so an ablation row can be assigned to the right arm, which is what the hash
// is for.
const ATTACKER_CONCISE_SUFFIX: &str = "\n\nOnce each test is written and checked once, output the JSON immediately; do not re-verify formatting.";

/// Whether `VANTA_ATTACKER_CONCISE` asked for the shorter first prompt (§14.22).
static ATTACKER_CONCISE: AtomicBool = AtomicBool::new(false);

/// Set the §14.22 arm; `false` is the default and every run before it existed.
pub fn set_attacker_concise(enabled: bool) {
    ATTACKER_CONCISE.store(enabled, Ordering::SeqCst);
}

/// Whether the §14.22 sentence is appended to the first request's user message.
pub fn attacker_concise() -> bool {
    ATTACKER_CONCISE.load(Ordering::SeqCst)
}

/// The sentence the arm appends, exposed so a test can pin the text that is sent
/// rather than a copy of it.
pub fn attacker_concise_suffix() -> &'static str {
    ATTACKER_CONCISE_SUFFIX
}

// ============================================================================
// attacker_generate_tests() — call LLM, get typed AttackerResponse
// ============================================================================
pub async fn attacker_generate_tests(
    cfg: &LlmConfig,
    request: &AttackerRequest,
) -> Result<AttackerResponse, Box<dyn Error + Send + Sync>> {
    info!(patch_len = request.patch_text.len(), "attacker start");

    let client = build_reqwest_client(cfg)?;

    let system_prompt = ATTACKER_SYSTEM_PROMPT;

    // [VANTA-WHY] Round >= 2: show the probes already in the shared set, with
    // the input literals each one pins, and aim the Attacker somewhere new. A
    // duplicate probe adds no coverage but does consume a BATE slot and a
    // patched run, so the driver also drops exact duplicates before BATE; this
    // section is what makes the Attacker aim somewhere new instead of being
    // silently discarded. `already_probed_section` is where the "do not repeat"
    // rule is qualified against the system prompt's TERMINAL FALLBACK.
    let already_probed = already_probed_section(&request.existing_tests);

    let user_prompt = format!(
        "PATCH:\n```diff\n{}\n```\n\nORIGINAL SOURCE:\n```\n{}\n```\n\nISSUE:\n{}{}\n\nWrite AttackerResponse JSON.{}",
        request.patch_text,
        request.original_source,
        request.issue_text,
        already_probed,
        // [VANTA-WHY] §14.22 — empty unless the arm is on, so every default run's
        // first request body is byte-for-byte what it was before this existed.
        if attacker_concise() { ATTACKER_CONCISE_SUFFIX } else { "" }
    );

    debug!(system_len = system_prompt.len(), user_len = user_prompt.len(), "attacker prompts built");

    let sys_owned = system_prompt.to_string();
    let user_owned = user_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            let tail = format!(
                "REPROMPT (attempt #{}): Write ONLY JSON object {{\"tests\":[...]}}. Each test has test_name, test_source, intent, reason.",
                attempt
            );
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };

    let extras = crate::llm::with_sampling(
        json!({
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
        }),
        "attacker_generate",
    );

    let parse_closure = |raw: &str| -> Result<AttackerResponse, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        trace!(json = %cleaned.chars().take(300).collect::<String>(), "attacker extracted json");
        serde_json::from_str::<AttackerResponse>(cleaned).map_err(|e| {
            format!("ATTACKER PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e)
        })
    };

    let typed: AttackerResponse = structured_call_with_retry(
        &client, cfg, cfg.retry_attempts, "", "attacker_generate", build_messages, extras, parse_closure
    ).await?;

    info!(test_count = typed.tests.len(), "attacker done");
    for t in &typed.tests {
        debug!(test_name = %t.test_name, intent = ?t.intent, src_len = t.test_source.len(), "attacker test");
    }

    Ok(typed)
}

// ============================================================================
// ATTACKER_REVISE_SYSTEM_PROMPT — the BATE-revision prompt
// ============================================================================
//
// [VANTA-WHY] Hoisted out of `attacker_revise_test` for the same reason
// ATTACKER_SYSTEM_PROMPT was: a rule that lives only in a local `let` can be
// dropped in a refactor with nothing to notice, and the audit fixed one of these
// rules (plans/bugsinpy.md §14.10, A4), so the corrected text has to be pinned by
// a test rather than by a comment.
pub const ATTACKER_REVISE_SYSTEM_PROMPT: &str = r#"You are Vanta Attacker. A test you wrote failed BATE (Baseline-Aware Test Evaluation). The test's actual result on the ORIGINAL (unpatched) code did not match its declared intent. Revise the test so its baseline result matches its intent.

The user message carries WHAT THE BASELINE ACTUALLY DID: the verdict in BATE's own terms, the exception type and message, and the last traceback frames from the real run on the original code. Read it before you change anything — it says WHERE and WHY the test failed. Do not guess at the cause and do not rewrite the test from scratch on a hunch: a revision that repeats an earlier one has learned nothing from the output in front of you.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.

Required JSON shape:
{
  "test_name":   string,
  "test_source": string — MUST be a complete pytest function: `def test_name():` with assertions inside the function body. NO module-level statements. Import at top, then `def test_...():` with body.
                   IMPORTANT: Do NOT import constants or symbols that only exist in the PATCHED code. Use literal values.
  "intent":      "BugReproduction" | "FeatureAcceptance" | "BehaviorPreservation",
  "reason":      string — why this revised test now matches its intent,
  "issue_quote": string|null — for BugReproduction or FeatureAcceptance, quote the exact issue sentence that asks for this behaviour; for BehaviorPreservation use null
}

Intent rules (these predict the baseline result BATE will measure on the ORIGINAL code):
- BugReproduction:      must FAIL on original code (the bug is present)
- FeatureAcceptance:    must FAIL on original code (feature doesn't exist yet)
- BehaviorPreservation: must PASS on original code (existing behavior)

If the baseline still does not match the declared intent, fix the TEST first — the input it asserts, the expected value, or the label — and make the label the one the ISSUE supports for the input you assert. Relabel only when the LABEL was the mistake: the new intent must be honest for what the test actually asserts, and BugReproduction / FeatureAcceptance need an issue sentence that asks for that behaviour AT THAT INPUT. Relabelling is never a way to make a mismatch disappear; a test whose input or expected value is wrong stays wrong under any label."#;

// ============================================================================
// ATTACKER_SIMPLIFY_SYSTEM_PROMPT — the Unknown-simplification prompt
// ============================================================================
//
// [VANTA-WHY] Hoisted out of `attacker_simplify_test` for the same reason the
// revise prompt was. The audit fixed this rule too (plans/bugsinpy.md §14.10,
// A10): the prompt told the Attacker to simplify a test that produced an UNKNOWN
// but never said the ASSERTION had to survive, and the Diagnostician then read a
// passing simplified test as "Unknown resolved" — including a body whose
// assertion had been deleted. The corrected text is pinned by a test.
//
// [VANTA-WHY] This constant is the ONLY copy of the simplify prompt and the call
// site sends exactly it (`let system_prompt = ATTACKER_SIMPLIFY_SYSTEM_PROMPT;`),
// so the test below pins what the call really sends. The A10 follow-up found the
// hoisted text still carrying the sentence that let the assertion be reduced
// away — "Keep the same intent" — which is why the test refuses that phrase
// rather than only requiring the new one: the prompt no longer licenses a
// simplification that keeps the label and drops the claim.
pub const ATTACKER_SIMPLIFY_SYSTEM_PROMPT: &str = r#"You are Vanta Attacker. A test you wrote produced an UNKNOWN result (could not be classified as Pass/Fail/Infrastructure/Static). The test may be too complex, non-deterministic, or have side effects. Simplify it so it produces a clear PASS or FAIL.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.

Required JSON shape:
{
  "test_name":   string,
  "test_source": string — MUST be a complete pytest function: `def test_name():` with assertions inside the function body.
                   SIMPLIFY: remove network calls, time.sleep, randomness, complex setup. Use literal values.
                   KEEP THE ASSERTION: the simplified test MUST still assert the SAME input and the SAME expected value as the test you were given. Dropping the assertion — or replacing it with `assert True`, an `is not None` check, or a bare call — makes the test pass without testing anything, and it is then SCRAPPED and the failure stands. Simplify the SETUP, never the claim.
  "intent":      "BugReproduction" | "FeatureAcceptance" | "BehaviorPreservation",
  "reason":      string — what you simplified and why,
  "issue_quote": string|null — preserve the prior issue quote for BugReproduction/FeatureAcceptance; use null for BehaviorPreservation
}

Do NOT import constants or symbols that only exist in the PATCHED code."#;

/// The simplify prompt the call site sends.
///
/// Routed through an accessor for the same reason `intent_auditor_system_prompt`
/// is: "the prompt the call sends" and "the prompt the test pins" are then one
/// value by construction. The A10 follow-up is what makes that worth pinning —
/// the simplify text existed twice while it was being rewritten (a `pub const`
/// and the function-local literal), so a test could pin the copy the call does
/// not send.
pub fn attacker_simplify_system_prompt() -> &'static str {
    ATTACKER_SIMPLIFY_SYSTEM_PROMPT
}

/// Every system prompt the Attacker's call sites can send, in a fixed order.
///
/// [VANTA-WHY] §14.10 f — the ingredient of the `attacker=` prompt-content hash.
/// It is an accessor over the SAME literals the call sites assign to
/// `system_prompt` (`ATTACKER_SYSTEM_PROMPT` for generate,
/// `ATTACKER_REVISE_SYSTEM_PROMPT` for the BATE revision,
/// `attacker_simplify_system_prompt()` for the Unknown path), so the hash is
/// taken over the text that is sent and not over a copy of it. Order is part of
/// the hash's meaning and must not be reshuffled: `attacker_prompt_hash_follows_the_call_sites`
/// pins each slot against the accessor the call site actually uses.
pub fn attacker_system_prompts() -> [&'static str; 3] {
    [
        ATTACKER_SYSTEM_PROMPT,
        ATTACKER_REVISE_SYSTEM_PROMPT,
        attacker_simplify_system_prompt(),
    ]
}

/// Every system prompt the intent auditor can send, in a fixed order: the
/// one-test prompt, its grounded variant, the batched prompt, and the batched
/// grounded variant.
///
/// [VANTA-WHY] §14.10 f — all FOUR, because which one is sent depends on two
/// independent flags (`VANTA_GROUNDED_AUDIT`, `VANTA_BATCH_AUDIT`), and the
/// wording of each is part of the run's identity: a change to the grounded
/// variant alone must show in the log of a grounded run. Taken through
/// `intent_auditor_system_prompt`/`intent_auditor_batch_system_prompt`, the same
/// selectors the call sites use, so a literal that is not the one sent cannot
/// contribute to the hash.
pub fn intent_auditor_system_prompts() -> [&'static str; 4] {
    [
        intent_auditor_system_prompt(false),
        intent_auditor_system_prompt(true),
        intent_auditor_batch_system_prompt(false),
        intent_auditor_batch_system_prompt(true),
    ]
}

// ============================================================================
// attacker_revise_test() — BATE feedback loop: revise a mismatched test
// ============================================================================
//
// [VANTA-WHY] Spec §7 Phase 1: "Mismatch returns to the Attacker for revision,
// max 3 retries, then scrap." When a test's baseline result doesn't match its
// declared intent, we don't just scrap it — we ask the Attacker to fix it.
// We pass the original test, the mismatch reason, and the original source so
// the Attacker knows what went wrong and can produce a corrected test.
//
// This is the "repair" half of BATE. The retry loop (validate_with_retry)
// handles flakiness; this handles genuinely wrong tests.
// ============================================================================
pub async fn attacker_revise_test(
    cfg: &LlmConfig,
    issue_text: &str,
    original_source: &str,
    prior_test: &GeneratedTest,
    mismatch_reason: &str,
    baseline_evidence: &str,
) -> Result<GeneratedTest, Box<dyn Error + Send + Sync>> {
    info!(test_name = %prior_test.test_name, "attacker revise_test start");

    let client = build_reqwest_client(cfg)?;

    let system_prompt = ATTACKER_REVISE_SYSTEM_PROMPT;

    let user_prompt = format!(
        "ISSUE:\n{issue_text}\n\n\
         ORIGINAL SOURCE:\n```\n{original_source}\n```\n\n\
         PRIOR TEST (failed BATE):\n- name: {name}\n- intent: {intent:?}\n- source:\n```python\n{source}\n```\n\n\
         BATE MISMATCH REASON:\n{reason}\n\n\
         WHAT THE BASELINE ACTUALLY DID (the real run on the ORIGINAL code — read this, \
         do not guess at it):\n{evidence}\n\n\
         Revise the test so its baseline result on the original source matches its declared intent. Expected values come from the issue and original code, never from the patch alone. Output JSON.",
        issue_text = issue_text,
        original_source = original_source,
        name = prior_test.test_name,
        intent = prior_test.intent,
        source = prior_test.test_source,
        reason = mismatch_reason,
        evidence = baseline_evidence,
    );

    let sys_owned = system_prompt.to_string();
    let user_owned = user_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            let tail = format!(
                "REPROMPT (attempt #{}): Write ONLY JSON object with test_name, test_source, intent, reason, issue_quote.",
                attempt
            );
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };

    let extras = crate::llm::with_sampling(
        json!({
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
        }),
        "attacker_revise",
    );

    let parse_closure = |raw: &str| -> Result<GeneratedTest, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        serde_json::from_str::<GeneratedTest>(cleaned).map_err(|e| {
            format!("ATTACKER REVISE PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e)
        })
    };

    let revised: GeneratedTest = structured_call_with_retry(
        &client, cfg, cfg.retry_attempts, &prior_test.test_name, "attacker_revise", build_messages, extras, parse_closure
    ).await?;

    info!(test_name = %revised.test_name, intent = ?revised.intent, "attacker revise_test done");
    Ok(revised)
}

// ============================================================================
// attacker_simplify_test() — Unknown remediation: simplify a flaky/complex test
// ============================================================================
//
// [VANTA-WHY] Spec §7 Phase 2 Unknown handling: "rerun in a fresh sandbox,
// capture logs, return to the Attacker to simplify, quarantine if it persists."
// When a test produces an Unknown result (can't classify), we ask the Attacker
// to simplify it — reduce complexity, remove dependencies, make it more
// deterministic. If the simplified test still produces Unknown, we quarantine.
// ============================================================================
pub async fn attacker_simplify_test(
    cfg: &LlmConfig,
    original_source: &str,
    prior_test: &GeneratedTest,
) -> Result<GeneratedTest, Box<dyn Error + Send + Sync>> {
    info!(test_name = %prior_test.test_name, "attacker simplify_test start");

    let client = build_reqwest_client(cfg)?;

    let system_prompt = attacker_simplify_system_prompt();

    let user_prompt = format!(
        "ORIGINAL SOURCE:\n```\n{original_source}\n```\n\n\
         PRIOR TEST (Unknown result):\n- name: {name}\n- intent: {intent:?}\n- source:\n```python\n{source}\n```\n\n\
         Simplify this test so it produces a deterministic PASS or FAIL. Output JSON.",
        original_source = original_source,
        name = prior_test.test_name,
        intent = prior_test.intent,
        source = prior_test.test_source,
    );

    let sys_owned = system_prompt.to_string();
    let user_owned = user_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            let tail = format!(
                "REPROMPT (attempt #{}): Write ONLY JSON object with test_name, test_source, intent, reason, issue_quote.",
                attempt
            );
            messages.push(json!({"role": "user", "content": tail}));
        }
        messages
    };

    let extras = crate::llm::with_sampling(
        json!({
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
        }),
        "attacker_simplify",
    );

    let parse_closure = |raw: &str| -> Result<GeneratedTest, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        serde_json::from_str::<GeneratedTest>(cleaned).map_err(|e| {
            format!("ATTACKER SIMPLIFY PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e)
        })
    };

    let simplified: GeneratedTest = structured_call_with_retry(
        &client, cfg, cfg.retry_attempts, &prior_test.test_name, "attacker_simplify", build_messages, extras, parse_closure
    ).await?;

    info!(test_name = %simplified.test_name, "attacker simplify_test done");
    Ok(simplified)
}

// ============================================================================
// INTENT AUDITOR — a separate call that never sees the patch
// ============================================================================
//
// [VANTA-WHY] The Attacker declares WHY a test exists, and BATE trusts that
// declaration: a BugReproduction/FeatureAcceptance test is ALLOWED to fail on
// the original code, a BehaviorPreservation test is not. That asymmetry is what
// let a bad patch write its own acceptance test — the Attacker read the patch,
// saw a changed behaviour, and labelled the change "FeatureAcceptance", so the
// change became the requirement it satisfied.
//
// The Attacker prompt forbids that. A prompt is not a guarantee. This call is
// the independent check: it is given the ISSUE, the ORIGINAL source and the
// test — never the patch, never the patched source. A non-preservation label
// survives only if this audit says the issue asks for that behaviour at the
// input the test asserts.
//
// [VANTA-WHY] The audit relabels in BOTH directions, but only ever in the
// direction that makes a test harder to admit, and always from the verdict about
// the ISSUE rather than from a BATE result (see `route_intent`, which owns the
// invariant). Its directions are:
//   - a non-preservation label survives only if the audit says the issue asks
//     for that behaviour at the asserted input; otherwise it is demoted to
//     BehaviorPreservation. An auditor that cannot answer demotes rather than
//     letting the label through, because an unverified non-preservation label
//     must not stand;
//   - a BehaviorPreservation test that asserts a behaviour the issue ASKS for and
//     REQUIRES to change is a mislabelled acceptance test, and is relabelled to
//     FeatureAcceptance — item 6. Scrapping it threw away the probe that would
//     have failed on the broken patch (B v2);
//   - a BehaviorPreservation test the issue requires to change but does NOT ask
//     for is scrapped: the label is wrong and there is no honest one left.
//     This is what makes the "the issue only asks to change this one boundary"
//     exception a checked fact rather than a sentence the Attacker may or may
//     not have honoured.
// EVERY test is audited and gets a record, preservation included.

/// Input to the intent audit.
///
/// [VANTA-WHY] Deliberately has NO patch field and no patched-source field, so
/// there is nothing for the call to leak. A regression test asserts the request
/// body the stub server receives contains neither.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentCheckRequest {
    pub issue_text: String,
    pub original_source: String,
    pub test_name: String,
    pub test_source: String,
    pub declared_intent: TestIntent,
}

/// The auditor's answer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntentVerdict {
    /// Does the issue ask for the behaviour this test asserts, at the input the
    /// test asserts?
    pub issue_asks_for_behaviour: bool,
    /// [VANTA-WHY] The question a preservation test needs. A BehaviorPreservation
    /// test asserts the ORIGINAL result, so `issue_asks_for_behaviour` is the
    /// wrong question for it — an issue that asks for a DIFFERENT result there
    /// answers false, which would wrongly keep a probe that contradicts the
    /// issue. This field asks the complementary question: does the issue require
    /// the behaviour at the asserted input to CHANGE from what the original code
    /// does? `true` means the preservation label is wrong: either the issue asks
    /// for that new behaviour (`issue_asks_for_behaviour` is then true, and the
    /// probe is relabelled to `FeatureAcceptance` — item 6), or the issue asks for
    /// neither and no honest label remains (scrapped).
    ///
    /// [VANTA-WHY] `Option<bool>`, NOT `bool`. As a plain bool, `#[serde(default)]`
    /// turned a field the model simply forgot to emit into `false`, and `false`
    /// is the answer that means "the issue requires this behaviour to STAY" — so
    /// a dropped field silently promoted a test to BehaviorPreservation with
    /// nobody having decided that. A safety gate has to fail closed: absent is
    /// UNKNOWN, and unknown never relabels. Every consumer therefore keys on an
    /// EXPLICIT value — `route_intent` relabels only on `Some(true)` or, for the
    /// BATE relabel question, `Some(false)`.
    #[serde(default)]
    pub issue_requires_change: Option<bool>,
    /// [VANTA-WHY] §14.4 — the grounded-assertions question, asked only with
    /// `VANTA_GROUNDED_AUDIT=on`. `Option<bool>` for the same reason
    /// `issue_requires_change` is: with the flag off the field is absent and
    /// `None` says so, which must never be read as "ungrounded" — a field nobody
    /// asked for must not be able to scrap a test that earlier waves accepted.
    #[serde(default)]
    pub assertions_grounded: Option<bool>,
    pub reason: String,
}

/// [VANTA-WHY] The deterministic quote evidence is RECORDED, never enforced. It
/// tells a reader whether the Attacker's quote was real and whether it named the
/// asserted input; the verdict is what decides the label.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuoteEvidence {
    pub quote_present: bool,
    pub quote_in_issue: bool,
    pub quote_names_asserted_input: bool,
    pub asserted_inputs: Vec<String>,
}

/// One audit result. Emitted for EVERY test — kept, demoted or scrapped — so a
/// run's log shows the check ran and what it decided.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntentCheckRecord {
    pub test_name: String,
    pub declared_intent: TestIntent,
    pub issue_asks_for_behaviour: bool,
    /// [VANTA-WHY] The complementary question a preservation probe needs: does
    /// the issue require the behaviour at the asserted input to CHANGE? Logged
    /// alongside the verdict so a reader can see why a preservation test was
    /// kept or scrapped, not just that it was. `None` records that the auditor
    /// never answered — distinct from an explicit `Some(false)`, because the two
    /// lead to different decisions and a log that conflates them cannot explain
    /// the decision afterwards.
    pub issue_requires_change: Option<bool>,
    /// [VANTA-WHY] §14.4 — the grounded-assertions answer, logged for the same
    /// reason as the other two: a reader must be able to see WHY a test was
    /// scrapped as ungrounded, and `None` (flag off, or the model omitted the
    /// field) must be distinguishable from an explicit `false`.
    pub assertions_grounded: Option<bool>,
    /// [VANTA-WHY] True when a BehaviorPreservation test asserts behaviour the
    /// issue requires to CHANGE. It contradicts the issue, so it is scrapped
    /// rather than demoted: there is no honest label left for it.
    pub scrapped: bool,
    pub reason: String,
    pub evidence: QuoteEvidence,
}

/// [VANTA-WHY] §14.5 — one entry of the batched auditor's answer. The field set
/// is the single-test contract plus the number the prompt printed for the test,
/// so the per-test decision is computed from the same three answers either way.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntentAuditEntry {
    /// The 1-based number the prompt printed above the test.
    pub index: usize,
    pub issue_asks_for_behaviour: bool,
    #[serde(default)]
    pub issue_requires_change: Option<bool>,
    #[serde(default)]
    pub assertions_grounded: Option<bool>,
    pub reason: String,
}

impl IntentAuditEntry {
    fn verdict(&self) -> IntentVerdict {
        IntentVerdict {
            issue_asks_for_behaviour: self.issue_asks_for_behaviour,
            issue_requires_change: self.issue_requires_change,
            assertions_grounded: self.assertions_grounded,
            reason: self.reason.clone(),
        }
    }
}

/// [VANTA-WHY] §14.5 — the batched auditor's whole answer: one entry per test
/// the prompt numbered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntentBatchVerdict {
    pub tests: Vec<IntentAuditEntry>,
}

impl IntentBatchVerdict {
    /// Map the batch back onto the round, in the order the tests went in.
    ///
    /// [VANTA-WHY] Every index must appear exactly once: an unknown index, a
    /// duplicate or a missing one is a parse failure of the WHOLE response, not
    /// a shrug about one test. The alternative — filling the gap with "no
    /// answer" — would silently demote whichever test the model forgot, and a
    /// demotion that nobody decided is the failure mode this project exists to
    /// prevent. A count that does not match the round is a failure for the same
    /// reason.
    fn by_index(&self, count: usize) -> Result<Vec<IntentVerdict>, String> {
        if self.tests.len() != count {
            return Err(format!(
                "batch answered {} test(s) but {count} were sent",
                self.tests.len()
            ));
        }
        let mut slots: Vec<Option<IntentVerdict>> = vec![None; count];
        for entry in &self.tests {
            if entry.index == 0 || entry.index > count {
                return Err(format!(
                    "batch entry index {} is not one of 1..={count}",
                    entry.index
                ));
            }
            let slot = &mut slots[entry.index - 1];
            if slot.is_some() {
                return Err(format!("batch entry index {} appears twice", entry.index));
            }
            *slot = Some(entry.verdict());
        }
        slots
            .into_iter()
            .enumerate()
            .map(|(i, slot)| slot.ok_or_else(|| format!("batch entry index {} is missing", i + 1)))
            .collect()
    }
}

/// [VANTA-WHY] The first line is the stub server's dispatch marker for this call
/// kind; an integration test uses it to find the audit requests and prove their
/// bodies carry no patch.
const INTENT_AUDITOR_SYSTEM_PROMPT: &str = r#"You are Vanta Intent Auditor. You are given an ISSUE, the ORIGINAL source of a repository, and ONE test. You are NOT given the patch and must not speculate about it.

Answer these questions about the input the test asserts:
1. "issue_asks_for_behaviour": does the ISSUE ask for the result this test asserts, AT THE INPUT THE TEST ASSERTS?
2. "issue_requires_change": does the ISSUE require the behaviour at that input to CHANGE from what the ORIGINAL source does there — that is, does the issue ask for a DIFFERENT result than the original returns at that input?

Rules:
- The issue is the only source of requirements. The original source shows what the behaviour WAS; it is evidence of existing behaviour, never of a new requirement.
- A test asserting `shipping_band(500) == "light"` is authorised only by an issue sentence that asks for that result at that input. A sentence about a DIFFERENT input does not authorise it.
- A sentence that forbids changing behaviour ("without changing the existing behaviour at the other thresholds") FORBIDS the change; it never authorises it and never requires one.
- A "must not change" / "without changing X" clause covers EVERY input whose original result the issue does not ask to change. Do not enumerate those inputs and do not debate which of them count as "thresholds": the only input the clause does NOT cover is one the issue asks you to change.
- An issue that asks for a change at ONE input does not require a change at any other input: answer "issue_requires_change" for the input THIS test asserts, not for the issue in general.
- A preservation test that asserts a function's INTERNAL downstream call is unchanged (for example, that an inner helper is still called with the OLD arguments) contradicts an issue that says a new value is THREADED THROUGH that function: the internal call is exactly what the issue asks to change, so answer "issue_requires_change": true for it. Do not accept "the function is called exactly as before" as a preservation of behaviour the issue rewires.
- If the issue does not ask for the asserted behaviour at the asserted input, answer "issue_asks_for_behaviour": false. Preserving existing behaviour is always a valid answer.
- For a BehaviorPreservation test, when the asserted input is one the issue does NOT ask to change, both answers are already fixed: "issue_requires_change": false (the issue requires no change there) and "issue_asks_for_behaviour": false (a clause that forbids a change never ASKS for the old result). Do not deliberate over whether a "must stay the same" clause counts as asking: it forbids, so both answers are false. Decide once and move on.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.
{"issue_asks_for_behaviour": true|false, "issue_requires_change": true|false, "reason": "one short sentence"}"#;

/// [VANTA-WHY] §14.4 — the auditor's THIRD question, used only with
/// `VANTA_GROUNDED_AUDIT=on`.
///
/// A SEPARATE literal rather than a string built up from the one above, because
/// the flag-off prompt must stay byte-identical to the prompt earlier waves ran
/// (`intent_audit` is one of §4.3's call sites, so its text is part of a run's
/// identity — see `INTENT_AUDITOR_PROMPT_VERSION`). The shared rules are repeated
/// verbatim; `the_grounded_prompt_keeps_the_shared_rules` pins that both prompts
/// carry them, so an edit to one that is not mirrored in the other fails the
/// tests rather than silently making the two asks disagree.
///
/// The question exists because PySnooper/3's false reject was a test that
/// asserted output text — `'Starting var' in content` — that the issue never
/// mentions. That is not a wrong *label*, which is all the other questions can
/// see; it is an assertion invented on top of the issue. Asking about grounding
/// lets the audit scrap it before it can reject a correct patch. Scrapping only
/// ever REMOVES a rejection signal, so a model that answers `false` too eagerly
/// can lose a reject but cannot manufacture one (and only a
/// `BugReproduction` test is routed on the answer).
const INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED: &str = r#"You are Vanta Intent Auditor. You are given an ISSUE, the ORIGINAL source of a repository, and ONE test. You are NOT given the patch and must not speculate about it.

Answer these questions about the input the test asserts:
1. "issue_asks_for_behaviour": does the ISSUE ask for the result this test asserts, AT THE INPUT THE TEST ASSERTS?
2. "issue_requires_change": does the ISSUE require the behaviour at that input to CHANGE from what the ORIGINAL source does there — that is, does the issue ask for a DIFFERENT result than the original returns at that input?

Rules:
- The issue is the only source of requirements. The original source shows what the behaviour WAS; it is evidence of existing behaviour, never of a new requirement.
- A test asserting `shipping_band(500) == "light"` is authorised only by an issue sentence that asks for that result at that input. A sentence about a DIFFERENT input does not authorise it.
- A sentence that forbids changing behaviour ("without changing the existing behaviour at the other thresholds") FORBIDS the change; it never authorises it and never requires one.
- A "must not change" / "without changing X" clause covers EVERY input whose original result the issue does not ask to change. Do not enumerate those inputs and do not debate which of them count as "thresholds": the only input the clause does NOT cover is one the issue asks you to change.
- An issue that asks for a change at ONE input does not require a change at any other input: answer "issue_requires_change" for the input THIS test asserts, not for the issue in general.
- A preservation test that asserts a function's INTERNAL downstream call is unchanged (for example, that an inner helper is still called with the OLD arguments) contradicts an issue that says a new value is THREADED THROUGH that function: the internal call is exactly what the issue asks to change, so answer "issue_requires_change": true for it. Do not accept "the function is called exactly as before" as a preservation of behaviour the issue rewires.
- If the issue does not ask for the asserted behaviour at the asserted input, answer "issue_asks_for_behaviour": false. Preserving existing behaviour is always a valid answer.
- For a BehaviorPreservation test, when the asserted input is one the issue does NOT ask to change, both answers are already fixed: "issue_requires_change": false (the issue requires no change there) and "issue_asks_for_behaviour": false (a clause that forbids a change never ASKS for the old result). Do not deliberate over whether a "must stay the same" clause counts as asking: it forbids, so both answers are false. Decide once and move on.

3. "assertions_grounded": does this test assert ONLY behaviour the ISSUE describes, at the input the ISSUE names — or does it also pin a SPECIFIC VALUE the issue never states? Ask this question ONLY when the test's declared intent is BugReproduction.

Rules for question 3:
- Grounded: an assertion that follows from the issue's own words about the input it names ("it should not raise", "the file is written", "the boundary value returns the standard band").
- Ungrounded: an assertion that pins wording, an exact output string, an exact message, a format or a number the issue never mentions ("the output CONTAINS 'Starting var'", "the repr is exactly ...", "the log line begins with a timestamp").
- The issue is the only source of requirements. "It should not raise NameError" is grounded; "it should print exactly this string" is grounded only if the issue prints that string.
- A test may assert more than the issue asks about ONLY when the extra assertion is a direct consequence of what the issue requires. Invented specifics are ungrounded.
- Answer question 3 for the test you are given only when its declared intent is BugReproduction; for a FeatureAcceptance or BehaviorPreservation test write "assertions_grounded": null and do not reason about it — nothing acts on the answer for those intents, so deciding it spends budget that changes no outcome.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.
{"issue_asks_for_behaviour": true|false, "issue_requires_change": true|false, "assertions_grounded": true|false|null, "reason": "one short sentence"}"#;

/// [VANTA-WHY] §14.5 — the BATCHED form of the auditor prompt, used only with
/// `VANTA_BATCH_AUDIT=on`.
///
/// The same ask with the frame changed from "ONE test" to "SEVERAL tests, each
/// numbered": the rules are repeated verbatim from
/// [`INTENT_AUDITOR_SYSTEM_PROMPT`] and the output contract gains one entry per
/// test. `the_batch_prompt_keeps_the_shared_rules` pins that repetition, so an
/// edit to one prompt that is not mirrored in the other fails the tests rather
/// than silently making the two asks disagree. The first line is unchanged
/// because the stub server dispatches this call kind on it.
const INTENT_AUDITOR_BATCH_SYSTEM_PROMPT: &str = r#"You are Vanta Intent Auditor. You are given an ISSUE, the ORIGINAL source of a repository, and SEVERAL tests, each numbered from 1. You are NOT given the patch and must not speculate about it.

Answer these questions about the input EACH test asserts:
1. "issue_asks_for_behaviour": does the ISSUE ask for the result that test asserts, AT THE INPUT THE TEST ASSERTS?
2. "issue_requires_change": does the ISSUE require the behaviour at that input to CHANGE from what the ORIGINAL source does there — that is, does the issue ask for a DIFFERENT result than the original returns at that input?

Rules:
- The issue is the only source of requirements. The original source shows what the behaviour WAS; it is evidence of existing behaviour, never of a new requirement.
- A test asserting `shipping_band(500) == "light"` is authorised only by an issue sentence that asks for that result at that input. A sentence about a DIFFERENT input does not authorise it.
- A sentence that forbids changing behaviour ("without changing the existing behaviour at the other thresholds") FORBIDS the change; it never authorises it and never requires one.
- A "must not change" / "without changing X" clause covers EVERY input whose original result the issue does not ask to change. Do not enumerate those inputs and do not debate which of them count as "thresholds": the only input the clause does NOT cover is one the issue asks you to change.
- An issue that asks for a change at ONE input does not require a change at any other input: answer "issue_requires_change" for the input THAT numbered test asserts, not for the issue in general.
- A preservation test that asserts a function's INTERNAL downstream call is unchanged (for example, that an inner helper is still called with the OLD arguments) contradicts an issue that says a new value is THREADED THROUGH that function: the internal call is exactly what the issue asks to change, so answer "issue_requires_change": true for it. Do not accept "the function is called exactly as before" as a preservation of behaviour the issue rewires.
- If the issue does not ask for the asserted behaviour at the asserted input, answer "issue_asks_for_behaviour": false. Preserving existing behaviour is always a valid answer.
- For a BehaviorPreservation test, when the asserted input is one the issue does NOT ask to change, both answers are already fixed: "issue_requires_change": false (the issue requires no change there) and "issue_asks_for_behaviour": false (a clause that forbids a change never ASKS for the old result). Do not deliberate over whether a "must stay the same" clause counts as asking: it forbids, so both answers are false. Decide once and move on.
- Answer EVERY numbered test exactly once and no test that was not given to you. "index" is the number printed above that test.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.
{"tests": [{"index": 1, "issue_asks_for_behaviour": true|false, "issue_requires_change": true|false, "reason": "one short sentence"}]}"#;

/// [VANTA-WHY] §14.5 — the batched form of the grounded auditor prompt, used
/// when `VANTA_BATCH_AUDIT=on` and `VANTA_GROUNDED_AUDIT=on`.
///
/// A SEPARATE literal for the same reason §14.4's variant is one: the two flags
/// are independent, and a batch run without the grounded question must not be
/// able to scrap anything on an answer nobody asked for. The two questions and
/// the shared rules are repeated verbatim again; the batch rule and question 3
/// are added.
const INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED: &str = r#"You are Vanta Intent Auditor. You are given an ISSUE, the ORIGINAL source of a repository, and SEVERAL tests, each numbered from 1. You are NOT given the patch and must not speculate about it.

Answer these questions about the input EACH test asserts:
1. "issue_asks_for_behaviour": does the ISSUE ask for the result that test asserts, AT THE INPUT THE TEST ASSERTS?
2. "issue_requires_change": does the ISSUE require the behaviour at that input to CHANGE from what the ORIGINAL source does there — that is, does the issue ask for a DIFFERENT result than the original returns at that input?

Rules:
- The issue is the only source of requirements. The original source shows what the behaviour WAS; it is evidence of existing behaviour, never of a new requirement.
- A test asserting `shipping_band(500) == "light"` is authorised only by an issue sentence that asks for that result at that input. A sentence about a DIFFERENT input does not authorise it.
- A sentence that forbids changing behaviour ("without changing the existing behaviour at the other thresholds") FORBIDS the change; it never authorises it and never requires one.
- A "must not change" / "without changing X" clause covers EVERY input whose original result the issue does not ask to change. Do not enumerate those inputs and do not debate which of them count as "thresholds": the only input the clause does NOT cover is one the issue asks you to change.
- An issue that asks for a change at ONE input does not require a change at any other input: answer "issue_requires_change" for the input THAT numbered test asserts, not for the issue in general.
- A preservation test that asserts a function's INTERNAL downstream call is unchanged (for example, that an inner helper is still called with the OLD arguments) contradicts an issue that says a new value is THREADED THROUGH that function: the internal call is exactly what the issue asks to change, so answer "issue_requires_change": true for it. Do not accept "the function is called exactly as before" as a preservation of behaviour the issue rewires.
- If the issue does not ask for the asserted behaviour at the asserted input, answer "issue_asks_for_behaviour": false. Preserving existing behaviour is always a valid answer.
- For a BehaviorPreservation test, when the asserted input is one the issue does NOT ask to change, both answers are already fixed: "issue_requires_change": false (the issue requires no change there) and "issue_asks_for_behaviour": false (a clause that forbids a change never ASKS for the old result). Do not deliberate over whether a "must stay the same" clause counts as asking: it forbids, so both answers are false. Decide once and move on.
- Answer EVERY numbered test exactly once and no test that was not given to you. "index" is the number printed above that test.

3. "assertions_grounded": does that test assert ONLY behaviour the ISSUE describes, at the input the ISSUE names — or does it also pin a SPECIFIC VALUE the issue never states? Ask this question ONLY for a numbered test whose declared intent is BugReproduction.

Rules for question 3:
- Grounded: an assertion that follows from the issue's own words about the input it names ("it should not raise", "the file is written", "the boundary value returns the standard band").
- Ungrounded: an assertion that pins wording, an exact output string, an exact message, a format or a number the issue never mentions ("the output CONTAINS 'Starting var'", "the repr is exactly ...", "the log line begins with a timestamp").
- The issue is the only source of requirements. "It should not raise NameError" is grounded; "it should print exactly this string" is grounded only if the issue prints that string.
- A test may assert more than the issue asks about ONLY when the extra assertion is a direct consequence of what the issue requires. Invented specifics are ungrounded.
- Answer question 3 for EACH numbered test only when its declared intent is BugReproduction; for a FeatureAcceptance or BehaviorPreservation test write "assertions_grounded": null and do not reason about it — nothing acts on the answer for those intents, so deciding it spends budget that changes no outcome.

YOU MUST output valid JSON. First char '{', last char '}'. No prose, no fences.
{"tests": [{"index": 1, "issue_asks_for_behaviour": true|false, "issue_requires_change": true|false, "assertions_grounded": true|false|null, "reason": "one short sentence"}]}"#;

/// The system prompt for the requested audit variant.
///
/// [VANTA-WHY] The only place the two prompts are chosen between. Flag off
/// returns the original literal unchanged — not a re-derivation of it — so
/// "prompt is byte-identical when the flag is off" is a property of the code
/// rather than of a test.
fn intent_auditor_system_prompt(grounded: bool) -> &'static str {
    if grounded {
        INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED
    } else {
        INTENT_AUDITOR_SYSTEM_PROMPT
    }
}

/// §14.5 — the batched method's system prompt, selected the same way.
fn intent_auditor_batch_system_prompt(grounded: bool) -> &'static str {
    if grounded {
        INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED
    } else {
        INTENT_AUDITOR_BATCH_SYSTEM_PROMPT
    }
}

/// Audit one test's declared intent against the issue.
///
/// Never sees the patch: `IntentCheckRequest` has no field for it.
///
/// [VANTA-WHY] `grounded` selects the §14.4 prompt VARIANT. With it `false` both
/// messages are built exactly as before the variant existed — the closing
/// question, the re-prompt line and the system prompt are the original strings,
/// not re-derivations of them — so a flag-off run's request body is byte-identical
/// to a v0.2 one. Run identity for the variant is the base
/// `INTENT_AUDITOR_PROMPT_VERSION` plus the `grounded_audit=` field `run.summary`
/// records, so a flag-off session's prompt tag is untouched.
pub async fn attacker_check_intent(
    cfg: &LlmConfig,
    request: &IntentCheckRequest,
    grounded: bool,
) -> Result<IntentVerdict, Box<dyn Error + Send + Sync>> {
    info!(
        test_name = %request.test_name,
        intent = ?request.declared_intent,
        grounded,
        "intent auditor start"
    );

    let client = build_reqwest_client(cfg)?;

    let closing_question = if grounded {
        "Does the issue ask for this behaviour at the input this test asserts, does it require the behaviour at that input to change, and does the test assert only behaviour the issue describes? Answer with JSON."
    } else {
        "Does the issue ask for this behaviour at the input this test asserts, and does it require the behaviour at that input to change? Answer with JSON."
    };

    let user_prompt = format!(
        "ISSUE:\n{}\n\nORIGINAL SOURCE:\n```\n{}\n```\n\nTEST UNDER AUDIT ({}):\n```\n{}\n```\n\nDeclared intent: {:?}\n\n{}",
        request.issue_text,
        request.original_source,
        request.test_name,
        request.test_source,
        request.declared_intent,
        closing_question
    );

    let sys_owned = intent_auditor_system_prompt(grounded).to_string();
    let user_owned = user_prompt.clone();
    let reprompt = if grounded {
        "REPROMPT: output ONLY {\"issue_asks_for_behaviour\": true|false, \"issue_requires_change\": true|false, \"assertions_grounded\": true|false|null, \"reason\": \"...\"}."
    } else {
        "REPROMPT: output ONLY {\"issue_asks_for_behaviour\": true|false, \"issue_requires_change\": true|false, \"reason\": \"...\"}."
    };
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            messages.push(json!({"role": "user", "content": reprompt}));
        }
        messages
    };

    // [VANTA-WHY] The audit's sampler pins come from the SAME table the
    // generative sites use (§14.11), so `pinned` still samples it at temperature
    // 0.0 — the audit must not be a coin flip, and it never sees the patch, so
    // there is no patch-shaped variance to average over — while `coding` samples
    // it at that preset's temperature, alongside the Attacker whose output it
    // judges. `max_tokens` comes from the schedule too, so the 8192 cap is the
    // same under both presets. Because `structured_call_with_retry` reads the
    // temperature off the request itself, a truncated audit at 0.6 retries with
    // the seed advanced instead of taking the temperature-0.0 fail-fast.
    let extras = crate::llm::with_sampling(
        json!({
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
        }),
        "intent_audit",
    );

    let parse_closure = |raw: &str| -> Result<IntentVerdict, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        serde_json::from_str::<IntentVerdict>(cleaned).map_err(|e| {
            format!(
                "INTENT AUDIT PARSE FAIL (line {} col {}): {}",
                e.line(),
                e.column(),
                e
            )
        })
    };

    let verdict: IntentVerdict =
        structured_call_with_retry(&client, cfg, cfg.retry_attempts, &request.test_name, "intent_audit", build_messages, extras, parse_closure).await?;

    info!(
        test_name = %request.test_name,
        asks = verdict.issue_asks_for_behaviour,
        "intent auditor done"
    );
    Ok(verdict)
}

/// Deterministic quote evidence for the session log. RECORDED ONLY.
pub fn observe_quote_evidence(test: &GeneratedTest, issue_text: &str) -> QuoteEvidence {
    let normalized_issue = normalize_issue_text(issue_text);
    let normalized_quote = test
        .issue_quote
        .as_deref()
        .map(normalize_issue_text)
        .filter(|quote| !quote.is_empty());

    let quote_present = normalized_quote.is_some();
    let quote_in_issue = normalized_quote
        .as_deref()
        .is_some_and(|quote| normalized_issue.contains(quote));

    let asserted_inputs = asserted_input_literals(&test.test_source);
    let quote_names_asserted_input = normalized_quote.as_deref().is_some_and(|quote| {
        asserted_inputs.is_empty()
            || asserted_inputs
                .iter()
                .any(|literal| quote_names_literal(quote, literal))
    });

    QuoteEvidence {
        quote_present,
        quote_in_issue,
        quote_names_asserted_input,
        asserted_inputs,
    }
}

/// M3 item 1 — does the ISSUE itself name what this test asserts?
///
/// [VANTA-WHY] This is the loosened grounding question, and it is answered from
/// the issue and the test's OWN source, never from a candidate patch. Two
/// deterministic facts, either of which grounds a reproduction test whose
/// asserted values are not literally quoted:
///   * (a) the verified `issue_quote` sits in the issue AND names a literal
///     input the test asserts (`quote_in_issue && quote_names_asserted_input`,
///     with at least one asserted input to name), or
///   * (b) a literal the test's own `assert` statements pin appears in the issue
///     — the issue states the symptom the test checks.
/// Neither fact reads the patch, so the override in `apply_intent_verdict`
/// cannot be steered by a candidate: the worst it can do is keep a probe the
/// auditor asked to drop, and item 1's approval invariant bounds that.
fn issue_names_what_the_test_asserts(
    test: &GeneratedTest,
    issue_text: &str,
    evidence: &QuoteEvidence,
) -> bool {
    // (a) — a verified quote naming an asserted input. Guarded on a non-empty
    // input set: with no asserted input there is nothing for the quote to name,
    // and `quote_names_asserted_input` is vacuously true in that case.
    if evidence.quote_in_issue
        && evidence.quote_names_asserted_input
        && !evidence.asserted_inputs.is_empty()
    {
        return true;
    }

    // (b) — the issue states a literal the test's asserts pin. Read off the
    // assert lines only, so an input mentioned in the issue that the test merely
    // passes (without asserting anything about it) does not ground it.
    let normalized_issue = normalize_issue_text(issue_text);
    asserted_contract_literals(&test.test_source)
        .iter()
        .any(|literal| quote_names_literal(&normalized_issue, literal))
}

/// How the intent audit routes one test, given the auditor's verdict.
///
/// [VANTA-WHY] Item 6 — the whole routing is a pure function of the VERDICT, so
/// it can be unit-tested without an LLM, and so the one invariant the relabel
/// machinery must never break lives in one place: a relabel may only ever make a
/// test HARDER to admit, never easier, and it may never let a test skip a BATE
/// check. Both directions add a BATE requirement:
///   - `BehaviorPreservation`→`FeatureAcceptance` (promotion) is safe because
///     BATE then demands the test FAIL on the original, i.e. the test must prove
///     the new behaviour is ABSENT before the patch is even run;
///   - `FeatureAcceptance`→`BehaviorPreservation` (demotion) is safe because
///     BATE then demands the test PASS on the original.
/// A relabel is never driven by a BATE result — only by what the ISSUE asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntentRouting {
    /// The declared label survives.
    Keep,
    /// BehaviorPreservation → FeatureAcceptance. The issue ASKS for the asserted
    /// behaviour and REQUIRES it to change, so the probe is a mislabelled
    /// acceptance test for the new behaviour rather than a preservation probe.
    PromoteToFeature,
    /// Non-preservation → BehaviorPreservation. The issue does not ask for the
    /// asserted behaviour, so the only honest label left is "this is behaviour
    /// the original already has". An auditor that cannot answer takes this path:
    /// an unverified non-preservation label must not stand.
    DemoteToPreservation,
    /// BehaviorPreservation + the issue REQUIRES the asserted behaviour to
    /// change. The test contradicts the issue and no honest label remains.
    ScrapRequiresChange,
    /// [VANTA-WHY] §14.4 — a `BugReproduction` test whose assertions pin values
    /// the issue never states. Only reachable with `VANTA_GROUNDED_AUDIT=on` and
    /// only on an EXPLICIT `Some(false)`: with the flag off the field is absent,
    /// so `None` falls through to the ordinary arms and a test earlier waves
    /// accepted cannot be scrapped by a question nobody asked. Scrapping removes
    /// a rejection signal and can never create one, which is why an over-eager
    /// model here loses a reject rather than manufacturing one.
    ScrapUngrounded,
    /// The test already carries one relabel, so this verdict would flip it back.
    ScrapRelabelCeiling,
}

/// Decide the routing for one test. Pure: verdict in, routing out.
///
/// [VANTA-WHY] `already_relabeled` is checked FIRST and only against the two
/// relabelling routings: keeping a label is not a relabel, so a test that was
/// relabelled once can still be kept for the rest of a run. What it cannot do is
/// be relabelled a second time — see `GeneratedTest::relabeled`.
pub(crate) fn route_intent(
    intent: &TestIntent,
    verdict: &IntentVerdict,
    already_relabeled: bool,
) -> IntentRouting {
    // [VANTA-WHY] §14.4 — an ungrounded BugReproduction test is scrapped before
    // any other arm is consulted. A test that asserts an invented specific cannot
    // be trusted to reject a patch whatever else the verdict says, so the answer
    // is "no honest test here" (a scrap, i.e. back to the Attacker) rather than a
    // relabel. Every other intent keeps its routing: an ungrounded
    // BehaviorPreservation probe can only ever ASSERT a failure, and the ordinary
    // arms already decide what an unasked-for presupposition means.
    if *intent == TestIntent::BugReproduction && verdict.assertions_grounded == Some(false) {
        return IntentRouting::ScrapUngrounded;
    }

    let wanted = match intent {
        TestIntent::BehaviorPreservation => {
            if verdict.issue_requires_change == Some(true) {
                if verdict.issue_asks_for_behaviour {
                    IntentRouting::PromoteToFeature
                } else {
                    IntentRouting::ScrapRequiresChange
                }
            } else {
                IntentRouting::Keep
            }
        }
        _ => {
            if verdict.issue_asks_for_behaviour {
                IntentRouting::Keep
            } else {
                IntentRouting::DemoteToPreservation
            }
        }
    };

    if already_relabeled
        && matches!(
            wanted,
            IntentRouting::PromoteToFeature | IntentRouting::DemoteToPreservation
        )
    {
        IntentRouting::ScrapRelabelCeiling
    } else {
        wanted
    }
}

/// The verdict an unavailable auditor produces, in BOTH transports.
///
/// [VANTA-WHY] §14.5 — one definition, because the batched path's failure
/// behaviour is defined as "what the single-call path does when its call
/// fails": `issue_asks_for_behaviour = false` (so a non-preservation label is
/// demoted rather than admitted), `issue_requires_change = None` (UNKNOWN, so
/// nothing is relabelled) and `assertions_grounded = None` (so nothing is
/// scrapped as ungrounded). A broken auditor therefore cannot manufacture a
/// rejection, and it cannot manufacture a scrap either.
fn audit_unavailable_verdict(error: &dyn std::fmt::Display) -> IntentVerdict {
    IntentVerdict {
        issue_asks_for_behaviour: false,
        issue_requires_change: None,
        assertions_grounded: None,
        reason: format!("intent audit unavailable ({error})"),
    }
}

/// Apply one auditor verdict to one test: route it, relabel or scrap it, and
/// build its record.
///
/// [VANTA-WHY] §14.5 — the single-call and batched paths differ ONLY in how the
/// verdict was obtained, so the decision itself lives here, shared by both. That
/// is what makes "the batched path decides the same way" a property of the code
/// rather than of a test: the same `route_intent` call, the same record
/// construction, the same `[VANTA-WHY]` arms. The evidence is computed from this
/// test and the issue, so it does not depend on the transport either.
fn apply_intent_verdict(
    test: &GeneratedTest,
    issue_text: &str,
    verdict: IntentVerdict,
) -> (Option<GeneratedTest>, Option<IntentRelabel>, Option<IntentCheckRecord>) {
    let evidence = observe_quote_evidence(test, issue_text);

    // [VANTA-WHY] M3 item 1 — the grounding rule is LOOSENED here, in the one
    // direction the plan allows. `ScrapUngrounded` was §14.4's fix for a test
    // that reproduced a symptom AND laid an invented assertion on top; what the
    // rule on Luna did instead was scrap every reproduction test whose asserted
    // values were not literally quoted in the issue, and the review then
    // approved a no-op on preservation tests alone (M3-plan §1.4, 12/12). The
    // plan's loosening: a reproduction test is grounded when its asserted
    // behaviour follows from the issue — the issue NAMES the input or the
    // symptom — even when the exact values are not written out.
    //
    // [VANTA-WHY] Two deterministic facts settle "the issue names it", and
    // neither is the model's opinion again:
    //   * the verified `issue_quote` names a literal input the test asserts
    //     (`quote_in_issue && quote_names_asserted_input`), or
    //   * a literal the test's own `assert` statements pin is stated by the
    //     issue (the issue names the symptom the test checks).
    // Both are computed from the ISSUE, never from the patch, so the override
    // cannot be steered by a candidate.
    //
    // [VANTA-WHY] The override runs in ONE direction only: it can KEEP a test
    // the auditor asked to scrap, never scrap one it wanted kept. So a model
    // that answers `assertions_grounded: false` too eagerly still cannot
    // manufacture a rejection — the worst case is a probe kept that the audit
    // would have dropped, and item 1's approval invariant bounds what keeping it
    // can do. The records stay honest: `assertions_grounded` still carries the
    // auditor's own answer, with `scrapped=false` and a reason naming the
    // override.
    let grounded_by_issue = issue_names_what_the_test_asserts(test, issue_text, &evidence);

    let record = |scrapped: bool, reason: String| IntentCheckRecord {
        test_name: test.test_name.clone(),
        declared_intent: test.intent.clone(),
        issue_asks_for_behaviour: verdict.issue_asks_for_behaviour,
        issue_requires_change: verdict.issue_requires_change,
        assertions_grounded: verdict.assertions_grounded,
        scrapped,
        reason,
        evidence: evidence.clone(),
    };

    // [VANTA-WHY] The routing is `route_intent`'s — unchanged — with item 1's
    // one deterministic loosening applied to its single scrap arm. Nothing else
    // is touched, so every other routing decision remains a pure function of the
    // verdict and is tested as one.
    let routing = route_intent(&test.intent, &verdict, test.relabeled);
    let routing = if routing == IntentRouting::ScrapUngrounded && grounded_by_issue {
        IntentRouting::Keep
    } else {
        routing
    };

    match routing {
        IntentRouting::Keep => {
            // [VANTA-WHY] When the override fired, the record must say so: the
            // auditor's own `assertions_grounded=false` is kept on the record,
            // and this reason is what tells a reader why the test was NOT
            // scrapped despite it.
            let reason = if grounded_by_issue && verdict.assertions_grounded == Some(false) {
                format!(
                    "assertions_grounded=false overridden: the issue itself names what this test \
                     asserts ({}), so the assertion is grounded even though the exact value is not \
                     quoted: {}",
                    if evidence.asserted_inputs.is_empty() {
                        "the asserted literal appears in the issue".to_string()
                    } else {
                        format!("asserted input(s) {}", evidence.asserted_inputs.join(", "))
                    },
                    verdict.reason
                )
            } else {
                verdict.reason.clone()
            };
            (Some(test.clone()), None, Some(record(false, reason)))
        }

        IntentRouting::PromoteToFeature => {
            // [VANTA-WHY] Item 6 — B v2's shape. The test declares preservation
            // but asserts the NEW behaviour (the new field's default), and the
            // issue asks for that behaviour and requires it to change. Scrapping
            // it discarded the one probe that would have caught the broken patch,
            // because the broken patch is exactly a patch that FAILS this test —
            // which is what FeatureAcceptance's baseline rule measures. The label
            // was wrong, not the test.
            let mut promoted = test.clone();
            promoted.intent = TestIntent::FeatureAcceptance;
            promoted.relabeled = true;
            let reason = format!(
                "the issue asks for the asserted behaviour and requires it to change at the \
                 asserted input, so the probe is an acceptance test for the new behaviour: {}",
                verdict.reason
            );
            let event = IntentRelabel {
                test_name: test.test_name.clone(),
                from_intent: TestIntent::BehaviorPreservation,
                to_intent: TestIntent::FeatureAcceptance,
                reason: reason.clone(),
            };
            (Some(promoted), Some(event), Some(record(false, reason)))
        }

        IntentRouting::DemoteToPreservation => {
            let mut demoted = test.clone();
            demoted.intent = TestIntent::BehaviorPreservation;
            demoted.issue_quote = None;
            demoted.relabeled = true;
            let event = IntentRelabel {
                test_name: test.test_name.clone(),
                from_intent: test.intent.clone(),
                to_intent: TestIntent::BehaviorPreservation,
                reason: format!(
                    "the issue does not ask for {:?} at the asserted input: {}",
                    test.intent, verdict.reason
                ),
            };
            (Some(demoted), Some(event), Some(record(false, verdict.reason.clone())))
        }

        IntentRouting::ScrapRequiresChange => {
            let reason = format!(
                "the issue requires the asserted behaviour to change at the asserted input, and \
                 the issue does not ask for it either, so no honest label remains: {}",
                verdict.reason
            );
            (None, None, Some(record(true, reason)))
        }

        IntentRouting::ScrapUngrounded => {
            // [VANTA-WHY] §14.4 — PySnooper/3's shape. The test reproduces the
            // reported symptom, so every other arm is happy with it; what is
            // wrong is an assertion laid on top that the issue never states
            // (`'Starting var' in content`). Kept, it fails on a CORRECT patch and
            // rejects it in round 4. Scrapped, it is dropped like a BATE mismatch,
            // so the Attacker is asked for a test that asserts only what the issue
            // describes — and the rejection signal it would have supplied is gone,
            // which is the safe direction.
            let reason = format!(
                "the test's assertions are not grounded in the issue: it pins a specific value \
                 the issue never states, so it cannot be trusted to judge a patch: {}",
                verdict.reason
            );
            warn!(
                test = %test.test_name,
                auditor_reason = %verdict.reason,
                "intent audit: ungrounded BugReproduction assertions — scrapping the test"
            );
            (None, None, Some(record(true, reason)))
        }

        IntentRouting::ScrapRelabelCeiling => {
            let reason = format!(
                "the test already carries one relabel; a second relabel would flip it back, so it \
                 is scrapped instead: {}",
                verdict.reason
            );
            (None, None, Some(record(true, reason)))
        }
    }
}

/// Audit one test against the issue.
///
/// Returns the surviving test (`None` = scrapped), the relabel event if the test
/// was relabelled, and the audit record. A record is returned for EVERY test,
/// preservation included, so the session log shows the check ran and what it
/// decided — the demote-only asymmetry is what made a preservation test the one
/// label nobody re-checked.
///
/// Four outcomes, one per way the declared intent can relate to the issue (see
/// `route_intent`): keep; promote a preservation probe to FeatureAcceptance when
/// the issue ASKS for the asserted behaviour and requires it to change (the B v2
/// shape: a probe asserting the NEW field's default, labelled as preservation);
/// demote an unasked-for non-preservation label; scrap when no honest label
/// remains.
///
/// [VANTA-WHY] §14.5 — this is the ONE-test transport. The decision is
/// [`apply_intent_verdict`]'s, shared with the batched transport, so the two
/// differ only in the number of calls they make.
pub async fn audit_test_intent(
    cfg: &LlmConfig,
    issue_text: &str,
    original_source: &str,
    test: &GeneratedTest,
    grounded: bool,
) -> (Option<GeneratedTest>, Option<IntentRelabel>, Option<IntentCheckRecord>) {
    let request = IntentCheckRequest {
        issue_text: issue_text.to_string(),
        original_source: original_source.to_string(),
        test_name: test.test_name.clone(),
        test_source: test.test_source.clone(),
        declared_intent: test.intent.clone(),
    };

    let verdict = match attacker_check_intent(cfg, &request, grounded).await {
        Ok(verdict) => verdict,
        Err(e) => {
            warn!(
                test = %test.test_name,
                error = %e,
                "intent audit unavailable — demoting to BehaviorPreservation"
            );
            audit_unavailable_verdict(&e)
        }
    };

    apply_intent_verdict(test, issue_text, verdict)
}

/// Ask ONLY whether the issue requires the asserted behaviour to change.
///
/// [VANTA-WHY] Fix 6 — BATE's relabel-before-revise question. A BugReproduction
/// test whose baseline PASSED did not reproduce anything: it asserts behaviour
/// the original already has. Whether that is a legitimate probe depends on one
/// thing — does the ISSUE require the behaviour at the asserted input to STAY as
/// the original has it? This asks that single question and returns the verdict
/// with an audit record, deliberately WITHOUT the demote/scrap routing
/// `audit_test_intent` applies: BATE owns the decision here.
///
/// An unavailable auditor answers UNKNOWN (`None`), exactly as a present auditor
/// that omitted the field does. Both mean "nobody decided", and BATE answers
/// that the only safe way: take the ordinary revision path. Relabelling on an
/// unverified answer would admit a probe the issue may want changed, which would
/// then fail the patched side and reject a correct patch.
pub async fn audit_issue_requires_change(
    cfg: &LlmConfig,
    issue_text: &str,
    original_source: &str,
    test: &GeneratedTest,
    grounded: bool,
) -> (Option<bool>, IntentCheckRecord) {
    let evidence = observe_quote_evidence(test, issue_text);

    let request = IntentCheckRequest {
        issue_text: issue_text.to_string(),
        original_source: original_source.to_string(),
        test_name: test.test_name.clone(),
        test_source: test.test_source.clone(),
        declared_intent: test.intent.clone(),
    };

    let verdict = match attacker_check_intent(cfg, &request, grounded).await {
        Ok(verdict) => verdict,
        Err(e) => {
            warn!(
                test = %test.test_name,
                error = %e,
                "relabel question unavailable — answer is unknown, so the test takes the \
                 ordinary revision path"
            );
            audit_unavailable_verdict(&e)
        }
    };

    let record = IntentCheckRecord {
        test_name: test.test_name.clone(),
        declared_intent: test.intent.clone(),
        issue_asks_for_behaviour: verdict.issue_asks_for_behaviour,
        issue_requires_change: verdict.issue_requires_change,
        assertions_grounded: verdict.assertions_grounded,
        scrapped: false,
        reason: verdict.reason,
        evidence,
    };
    (verdict.issue_requires_change, record)
}

/// What happened to a round's single batched intent-audit call (§14.5).
///
/// [VANTA-WHY] `truncated_fallback` and `malformed_retried` are the two ways a
/// batched round can fail to be answered by its batch, and they are NOT the same
/// event: the first is a fact about the request's output cap and is answered by
/// asking one test at a time, the second is a fact about the answer and is
/// answered by sending the same request again. Because the first now stops after
/// a single call and the second still spends the retry budget, the two are
/// distinguishable in the session log by this record rather than by counting
/// calls. `None` — no batched call was made — is the flag-off schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAuditTransport {
    /// One batched call answered the whole round.
    Batched,
    /// The batched answer was CUT OFF at its `max_tokens` cap
    /// (`finish_reason: "length"`) and was therefore not retried; the round was
    /// re-audited one test per call.
    TruncatedFallback,
    /// The batched answer was malformed WITHOUT being cut off. It was retried to
    /// the retry budget, and every test then fell back to the single-call
    /// failure shape.
    MalformedRetried,
    /// The batched call never produced a parseable answer at all — a transport
    /// error, or a prompt that cannot fit the context window (which is not
    /// retried either). Every test fell back to the single-call failure shape.
    Unavailable,
}

impl BatchAuditTransport {
    /// The token written into the session log's `intent.audit_batch` record.
    pub fn code(&self) -> &'static str {
        match self {
            BatchAuditTransport::Batched => "batched",
            BatchAuditTransport::TruncatedFallback => "truncated_fallback",
            BatchAuditTransport::MalformedRetried => "malformed_retried",
            BatchAuditTransport::Unavailable => "unavailable",
        }
    }
}

/// Output allowance the batched audit is given PER TEST in the round (§14.5).
///
/// [VANTA-WHY] The batch answers with one entry per test, so the cap the
/// single-test call pins is the wrong number for it: a three-test round asked
/// for one entry's worth of output, the JSON was cut off mid-object, and the
/// whole round's verdicts were lost to a parse failure. One entry per test is
/// what the request is for, so the cap is one test's worth per test.
///
/// [VANTA-WHY] 8192, raised from 2048 (§14.10 e-follow-up). The post-fix
/// Ornith-1.5 run on the toy was cut off at the old 6144 batch cap 21 times
/// across 8 of 12 runs, and the single-test pin (the schedule's own cap, the
/// same number) is what the fallback then inherits — so the raise covers both
/// transports. The biggest batched reply ends mid-sentence on the reasoning
/// ("Let me reconsider grounded once more…") with no JSON at all: the audit does
/// not stop deliberating at 2048 tokens, it stops writing.
pub const INTENT_AUDIT_BATCH_TOKENS_PER_TEST: u64 = 8_192;

/// Ceiling for the batched audit's output cap (§14.5).
///
/// [VANTA-WHY] 32768 — the full context window Vanta's local models are served
/// at (§9), raised from 16384 alongside the per-test allowance. It is now four
/// tests' worth of that allowance (8192 × 4), where under the old numbers it was
/// eight: a round larger than four is the case the per-test fallback below
/// exists for. The cap is deliberately allowed to reach the window rather than
/// stop at half of it: the alternative to a large cap is a truncated audit that
/// loses the whole round's verdicts, which is exactly the failure this raise is
/// for.
pub const INTENT_AUDIT_BATCH_TOKENS_CEILING: u64 = 32_768;

/// The `max_tokens` a batched audit of `tests` tests is given.
///
/// Never below one test's allowance: an empty round still asks a well-formed
/// question rather than for zero tokens, which no server can answer.
pub fn intent_audit_batch_max_tokens(tests: usize) -> u64 {
    INTENT_AUDIT_BATCH_TOKENS_PER_TEST
        .saturating_mul(tests as u64)
        .clamp(INTENT_AUDIT_BATCH_TOKENS_PER_TEST, INTENT_AUDIT_BATCH_TOKENS_CEILING)
}

/// One call for the whole round, to the batched auditor prompt (§14.5).
///
/// [VANTA-WHY] The batch is a TRANSPORT change, so the answer is turned into one
/// [`IntentVerdict`] per test, in the round's own order, and handed to the same
/// [`apply_intent_verdict`] the single-call path uses. An index that is unknown,
/// duplicated or missing fails the parse of the whole response (see
/// [`IntentBatchVerdict::by_index`]), which is what makes it retried like any
/// other malformed structured output instead of quietly demoting one test.
async fn attacker_check_intent_batch(
    cfg: &LlmConfig,
    issue_text: &str,
    original_source: &str,
    tests: &[GeneratedTest],
    grounded: bool,
) -> Result<Vec<IntentVerdict>, Box<dyn std::error::Error + Send + Sync>> {
    info!(
        tests = tests.len(),
        grounded,
        "batched intent auditor start"
    );

    let client = build_reqwest_client(cfg)?;

    let mut body = String::new();
    for (i, test) in tests.iter().enumerate() {
        body.push_str(&format!(
            "TEST {} — {}\nDeclared intent: {:?}\n```\n{}\n```\n\n",
            i + 1,
            test.test_name,
            test.intent,
            test.test_source
        ));
    }

    let closing_question = if grounded {
        format!(
            "Does the issue ask for the behaviour each of the {} tests asserts at the input it \
             asserts, does it require the behaviour at that input to change, and does each test \
             assert only behaviour the issue describes? Answer with one JSON entry per test.",
            tests.len()
        )
    } else {
        format!(
            "Does the issue ask for the behaviour each of the {} tests asserts at the input it \
             asserts, and does it require the behaviour at that input to change? Answer with one \
             JSON entry per test.",
            tests.len()
        )
    };

    let user_prompt = format!(
        "ISSUE:\n{}\n\nORIGINAL SOURCE:\n```\n{}\n```\n\n{} {} TESTS TO AUDIT:\n\n{}\n{}",
        issue_text,
        original_source,
        tests.len(),
        if tests.len() == 1 { "TEST IS" } else { "TESTS ARE" },
        body,
        closing_question
    );

    let sys_owned = intent_auditor_batch_system_prompt(grounded).to_string();
    let user_owned = user_prompt;
    let reprompt = if grounded {
        "REPROMPT: output ONLY {\"tests\": [{\"index\": 1, \"issue_asks_for_behaviour\": true|false, \"issue_requires_change\": true|false, \"assertions_grounded\": true|false|null, \"reason\": \"...\"}]} — one entry per numbered test, each index exactly once."
    } else {
        "REPROMPT: output ONLY {\"tests\": [{\"index\": 1, \"issue_asks_for_behaviour\": true|false, \"issue_requires_change\": true|false, \"reason\": \"...\"}]} — one entry per numbered test, each index exactly once."
    };
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            messages.push(json!({"role": "user", "content": reprompt}));
        }
        messages
    };

    // The same sampler pins as the single-test call, through the same table
    // (§14.11): `intent_audit` is one call site in §4.3's schedule, and batching
    // changes how many times it is called, not how each call is sampled. The
    // OUTPUT cap is the one number that must scale, because the batch's answer is
    // one entry per test (§14.5), so it is written over the schedule's cap after
    // the pins are applied — the batch keeps `8192 * tests` (ceilinged) and the
    // preset's temperature, which is what `run.summary` records separately.
    let mut extras = crate::llm::with_sampling(
        json!({
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
        }),
        "intent_audit",
    );
    extras["max_tokens"] = json!(intent_audit_batch_max_tokens(tests.len()));

    let count = tests.len();
    let parse_closure = |raw: &str| -> Result<Vec<IntentVerdict>, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        let batch = serde_json::from_str::<IntentBatchVerdict>(cleaned).map_err(|e| {
            format!(
                "INTENT AUDIT BATCH PARSE FAIL (line {} col {}): {}",
                e.line(),
                e.column(),
                e
            )
        })?;
        batch.by_index(count)
    };

    let verdicts = structured_call_with_retry(
        &client,
        cfg,
        cfg.retry_attempts,
        "intent_audit_batch",
        "intent_audit",
        build_messages,
        extras,
        parse_closure,
    )
    .await?;

    info!(tests = verdicts.len(), "batched intent auditor done");
    Ok(verdicts)
}

/// Audit a whole round, in ONE call when `batch` is set (§14.5) and one call per
/// test otherwise.
///
/// Scrapped tests are dropped from the returned list; their audit records (with
/// `scrapped = true`) are still returned for logging. Either way the records
/// come out in the order the tests went in, so a session log cannot be a
/// permutation of another run's.
///
/// The fourth element reports what happened to the batched call, for the run's
/// log (§14.5). It is `None` when the flag was off, because then no batched call
/// was made and the flag-off log stays what it was.
pub async fn audit_tests_intent(
    cfg: &LlmConfig,
    issue_text: &str,
    original_source: &str,
    tests: &[GeneratedTest],
    grounded: bool,
    batch: bool,
) -> (
    Vec<GeneratedTest>,
    Vec<IntentRelabel>,
    Vec<IntentCheckRecord>,
    Option<BatchAuditTransport>,
) {
    // [VANTA-WHY] §14.5 — with the flag off this is the v0.2 path unchanged: the
    // same per-test call, through the same function, in the same order.
    if !batch {
        let (audited, relabels, records) =
            audit_tests_intent_per_test(cfg, issue_text, original_source, tests, grounded).await;
        return (audited, relabels, records, None);
    }

    let (verdicts, transport): (Vec<IntentVerdict>, BatchAuditTransport) =
        match attacker_check_intent_batch(cfg, issue_text, original_source, tests, grounded).await {
            Ok(verdicts) => (verdicts, BatchAuditTransport::Batched),
            Err(e) if e.to_string().contains(TRUNCATED_OUTPUT_PREFIX) => {
                // [VANTA-WHY] §14.5 — a truncated batch is the one batch failure
                // the per-test path FIXES, because it is caused by the request
                // asking for N answers in one capped generation. `max_tokens` is
                // pinned, so the same batch is cut off at the same point: the
                // call is not retried at all (see `structured_call_with_retry`),
                // and falling through to the per-test loop asks the same
                // questions one at a time — each well inside the single call's
                // own 8192 cap — so the round keeps its verdicts instead of losing
                // all of them to a cap no retry of the batch can clear.
                warn!(
                    tests = tests.len(),
                    error = %e,
                    "batched intent audit was CUT OFF at max_tokens — re-auditing the round one \
                     test per call"
                );
                let (audited, relabels, records) =
                    audit_tests_intent_per_test(cfg, issue_text, original_source, tests, grounded)
                        .await;
                return (audited, relabels, records, Some(BatchAuditTransport::TruncatedFallback));
            }
            Err(e) => {
                // [VANTA-WHY] §14.5 — one failed batch call must not be one
                // verdict for the whole round. Every test falls back to exactly
                // what a failed single call produces (see
                // `audit_unavailable_verdict`), so a broken auditor demotes the
                // round instead of scrapping it and cannot manufacture a reject.
                warn!(
                    tests = tests.len(),
                    error = %e,
                    "batched intent audit unavailable — every test in the round takes the \
                     single-call failure path"
                );
                // Which of the two retried-and-failed shapes this is changes what
                // a reader should do about it: a malformed answer is the model's,
                // and the retries spent on it are the point of the retry budget;
                // a transport failure is the machine's. `PARSE ERR` is the marker
                // `structured_call_with_retry` puts on an attempt that reached
                // the parser — i.e. the model DID answer, and the answer was
                // wrong — so it is what separates them.
                let transport = if e.to_string().contains(PARSE_FAILURE_PREFIX) {
                    BatchAuditTransport::MalformedRetried
                } else {
                    BatchAuditTransport::Unavailable
                };
                let verdicts: Vec<IntentVerdict> =
                    tests.iter().map(|_| audit_unavailable_verdict(&e)).collect();
                (verdicts, transport)
            }
        };

    let mut audited = Vec::with_capacity(tests.len());
    let mut relabels = Vec::new();
    let mut records = Vec::new();
    for (test, verdict) in tests.iter().zip(verdicts) {
        let (updated, relabel, record) = apply_intent_verdict(test, issue_text, verdict);
        if let Some(updated) = updated {
            audited.push(updated);
        }
        if let Some(relabel) = relabel {
            relabels.push(relabel);
        }
        if let Some(record) = record {
            records.push(record);
        }
    }
    (audited, relabels, records, Some(transport))
}

/// Audit a round one test per call — the unbatched path.
///
/// [VANTA-WHY] Factored out of `audit_tests_intent` rather than duplicated: the
/// flag-off path and the truncated-batch fallback below must produce the SAME
/// per-test records, and two copies of this loop would be two definitions of
/// "what a per-test audit does".
async fn audit_tests_intent_per_test(
    cfg: &LlmConfig,
    issue_text: &str,
    original_source: &str,
    tests: &[GeneratedTest],
    grounded: bool,
) -> (
    Vec<GeneratedTest>,
    Vec<IntentRelabel>,
    Vec<IntentCheckRecord>,
) {
    let mut audited = Vec::with_capacity(tests.len());
    let mut relabels = Vec::new();
    let mut records = Vec::new();
    for test in tests {
        let (updated, relabel, record) =
            audit_test_intent(cfg, issue_text, original_source, test, grounded).await;
        if let Some(updated) = updated {
            audited.push(updated);
        }
        if let Some(relabel) = relabel {
            relabels.push(relabel);
        }
        if let Some(record) = record {
            records.push(record);
        }
    }
    (audited, relabels, records)
}

/// Whitespace-collapsed, markdown-free view of issue text for quote matching.
///
/// [VANTA-WHY] Issues wrap identifiers in markdown backticks, so an Attacker
/// that quotes the line literally *without* the backticks is not paraphrasing —
/// it is quoting. Dropping the ticks keeps such a quote admissible instead of
/// demoting a legitimate intent to BehaviourPreservation.
fn normalize_issue_text(text: &str) -> String {
    // Remove the ticks outright rather than replacing them with a space:
    // `"standard"`, must collapse to `"standard",` and not to `"standard" ,`.
    text.replace('`', "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Literal arguments appearing inside a call in the test source.
///
/// `assert shipping_band(500) == "light"` → `["500"]`.
///
/// [VANTA-WHY] These are the inputs a non-preservation test pins, and they are
/// the only deterministic link between a test and the issue sentence that is
/// supposed to authorise it. Without this, ANY verbatim sentence from the issue
/// is a licence: the fixture's real run quoted
/// "…without changing the existing behavior at the other thresholds." — a line
/// that forbids the very change the test encoded — and the quote check passed,
/// so the bad patch wrote its own acceptance test again.
fn asserted_input_literals(test_source: &str) -> Vec<String> {
    let chars: Vec<char> = test_source.chars().collect();
    let mut literals: Vec<String> = Vec::new();
    let mut depth = 0usize;
    let mut i = 0usize;

    while i < chars.len() {
        match chars[i] {
            '"' | '\'' => {
                let quote = chars[i];
                let start = i + 1;
                let mut j = start;
                while j < chars.len() && chars[j] != quote {
                    j += 1;
                }
                if depth > 0 {
                    let literal: String = chars[start..j.min(chars.len())].iter().collect();
                    let literal = literal.trim().to_string();
                    if literal.chars().count() >= 2 && !literals.contains(&literal) {
                        literals.push(literal);
                    }
                }
                i = j + 1;
                continue;
            }
            '(' => {
                depth += 1;
                i += 1;
                continue;
            }
            ')' => {
                depth = depth.saturating_sub(1);
                i += 1;
                continue;
            }
            c if c.is_ascii_digit() && depth > 0 => {
                let starts_token = chars
                    .get(i.wrapping_sub(1))
                    .map_or(true, |p| !(p.is_ascii_alphanumeric() || *p == '_' || *p == '.'));
                if starts_token {
                    let mut j = i;
                    while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
                        j += 1;
                    }
                    let literal: String = chars[i..j].iter().collect();
                    if !literals.contains(&literal) {
                        literals.push(literal);
                    }
                    i = j;
                    continue;
                }
            }
            _ => {}
        }
        i += 1;
    }

    literals
}

/// The literals an ASSERT statement pins — its inputs AND its expected value.
///
/// [VANTA-WHY] A10 — a "simplification" may delete the failing assertion
/// outright (`assert add(2, 3) == 5` → `add(2, 3)`, or `assert add(2, 3) is not
/// None`). Both still mention the same inputs, so `asserted_input_literals`
/// (call arguments only, which is all the duplicate-probe check needs) cannot
/// see the deletion: the body passes, and the Unknown looks resolved. The
/// contract a test owes is what its assert statements compare, so it is read off
/// THOSE lines: every string and number literal on a line that asserts.
fn asserted_contract_literals(test_source: &str) -> Vec<String> {
    let mut literals: Vec<String> = Vec::new();
    for line in test_source.lines() {
        if !line.trim_start().starts_with("assert") {
            continue;
        }
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            match chars[i] {
                '"' | '\'' => {
                    let quote = chars[i];
                    let start = i + 1;
                    let mut j = start;
                    while j < chars.len() && chars[j] != quote {
                        j += 1;
                    }
                    let literal: String = chars[start..j.min(chars.len())].iter().collect();
                    let literal = literal.trim().to_string();
                    if literal.chars().count() >= 2 && !literals.contains(&literal) {
                        literals.push(literal);
                    }
                    i = j + 1;
                }
                c if c.is_ascii_digit() => {
                    let starts_token = chars
                        .get(i.wrapping_sub(1))
                        .map_or(true, |p| !(p.is_ascii_alphanumeric() || *p == '_' || *p == '.'));
                    if starts_token {
                        let mut j = i;
                        while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
                            j += 1;
                        }
                        let literal: String = chars[i..j].iter().collect();
                        if !literals.contains(&literal) {
                            literals.push(literal);
                        }
                        i = j;
                    } else {
                        i += 1;
                    }
                }
                _ => i += 1,
            }
        }
    }
    literals
}

/// A10 — did the simplified test keep the contract the original test asserted?
///
/// True only when the simplified body still contains an assert and pins exactly
/// the literals — inputs and expected values — the original test's asserts did.
/// A body with no assert at all (`add(2, 3)`), or one that weakened the claim
/// (`assert add(2, 3) is not None`), answers false and must be SCRAPPED rather
/// than reported as a resolved Unknown.
pub fn simplification_keeps_the_asserted_contract(
    original_source: &str,
    simplified_source: &str,
) -> bool {
    let asserted = asserted_contract_literals(original_source);
    let simplified = asserted_contract_literals(simplified_source);
    !simplified.is_empty() && simplified == asserted
}

/// Does `quote` name `literal` as a standalone token?
///
/// Guards `500` against matching inside `1500`, and `2` against `20`.
fn quote_names_literal(quote: &str, literal: &str) -> bool {
    let mut from = 0usize;
    while let Some(offset) = quote[from..].find(literal) {
        let start = from + offset;
        let end = start + literal.len();
        let before_ok = quote[..start]
            .chars()
            .last()
            .map_or(true, |c| !(c.is_alphanumeric() || c == '_' || c == '.'));
        let after_ok = quote[end..]
            .chars()
            .next()
            .map_or(true, |c| !(c.is_alphanumeric() || c == '_' || c == '.'));
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

// [VANTA-WHY] There is deliberately NO deterministic relabel rule here any more.
// The quote-substring check (`relabel_test_to_issue_scope`) used to be the RULE:
// a non-preservation label survived iff its `issue_quote` appeared in the issue
// and named an asserted input. Sheriffing the label that way was still a lexical
// test pretending to be a semantic one — it can only ask whether a string
// appears, never whether the issue ASKS FOR the behaviour — and its own
// "anchored" case passed a quote that forbade the change it authorised. The
// substring evidence survives as `QuoteEvidence` on every `IntentCheckRecord`
// (a LOGGED FIELD, for readers and for re-analysis), while the decision is made
// by the patch-blind `attacker_check_intent` audit.

#[cfg(test)]
mod tests {
    use super::*;

    /// The real fixture issue, markdown and all — it is what exposed both
    /// defects: the backticked line and the sentence that forbids the change.
    const FIXTURE_ISSUE: &str = "`shipping_band(1000)` should return `\"standard\"`, not `\"heavy\"`.\n\nRight now the threshold logic treats weights strictly below `1000` grams as\n`\"standard\"` and everything else as `\"heavy\"`, so the exact `1000` gram edge\nfalls into the wrong band.\n\nPlease fix that boundary bug without changing the existing behavior at the\nother thresholds.\n";

    fn generated(intent: TestIntent, test_source: &str, issue_quote: Option<&str>) -> GeneratedTest {
        GeneratedTest {
            test_name: "test_boundary".into(),
            test_source: test_source.into(),
            intent,
            reason: "exercise a boundary".into(),
            issue_quote: issue_quote.map(str::to_string),
            relabeled: false,
        }
    }

    // ---- item 6: relabel routing (verdict in, routing out) ------------------

    fn verdict(asks: bool, requires_change: Option<bool>) -> IntentVerdict {
        IntentVerdict {
            issue_asks_for_behaviour: asks,
            issue_requires_change: requires_change,
            assertions_grounded: None,
            reason: "the issue says so".to_string(),
        }
    }

    /// [VANTA-WHY] §14.4 — the only difference is the third answer, so the
    /// grounded tests cannot accidentally pass by way of a change to the shared
    /// helper's other two fields.
    fn grounded_verdict(asks: bool, grounded: Option<bool>) -> IntentVerdict {
        IntentVerdict {
            issue_asks_for_behaviour: asks,
            issue_requires_change: None,
            assertions_grounded: grounded,
            reason: "the issue says so".to_string(),
        }
    }

    /// B v2's shape: the probe declares preservation but asserts the NEW
    /// behaviour, and the issue asks for it and requires it to change. Relabelling
    /// to FeatureAcceptance keeps the probe — scrapping it is what let the broken
    /// patch through.
    #[test]
    fn a_preservation_probe_asserting_a_required_new_behaviour_is_promoted() {
        assert_eq!(
            route_intent(
                &TestIntent::BehaviorPreservation,
                &verdict(true, Some(true)),
                false
            ),
            IntentRouting::PromoteToFeature
        );
    }

    /// The issue requires the behaviour to change but never asks for the asserted
    /// result, so the label is wrong and no honest one is left.
    #[test]
    fn a_preservation_probe_the_issue_requires_to_change_but_never_asks_for_is_scrapped() {
        assert_eq!(
            route_intent(
                &TestIntent::BehaviorPreservation,
                &verdict(false, Some(true)),
                false
            ),
            IntentRouting::ScrapRequiresChange
        );
    }

    /// An omitted answer is UNKNOWN, and unknown must never be read as "the issue
    /// requires a change" — that would relabel a possibly-valid probe on a parse
    /// artifact.
    #[test]
    fn a_preservation_probe_survives_an_unknown_or_negative_change_verdict() {
        assert_eq!(
            route_intent(
                &TestIntent::BehaviorPreservation,
                &verdict(true, None),
                false
            ),
            IntentRouting::Keep
        );
        assert_eq!(
            route_intent(
                &TestIntent::BehaviorPreservation,
                &verdict(false, Some(false)),
                false
            ),
            IntentRouting::Keep
        );
    }

    /// The oracle-loophole direction is deliberately KEPT: demoting a
    /// FeatureAcceptance test the issue does not ask for is safe because BATE then
    /// demands the test PASS on the original, so the demotion makes the test
    /// harder to admit, not easier.
    #[test]
    fn a_feature_acceptance_test_the_issue_does_not_ask_for_is_still_demoted() {
        assert_eq!(
            route_intent(
                &TestIntent::FeatureAcceptance,
                &verdict(false, Some(false)),
                false
            ),
            IntentRouting::DemoteToPreservation
        );
    }

    /// Item 6's ping-pong ceiling: a test that has already spent its one relabel
    /// cannot be relabelled again in either direction.
    #[test]
    fn a_test_that_already_carries_a_relabel_is_scrapped_rather_than_relabelled_again() {
        assert_eq!(
            route_intent(
                &TestIntent::BehaviorPreservation,
                &verdict(true, Some(true)),
                true
            ),
            IntentRouting::ScrapRelabelCeiling
        );
        assert_eq!(
            route_intent(
                &TestIntent::FeatureAcceptance,
                &verdict(false, Some(false)),
                true
            ),
            IntentRouting::ScrapRelabelCeiling
        );
    }

    /// The ceiling blocks RELABELS only: keeping a label is not a relabel, so a
    /// test that spent its one relabel still rides out the run.
    #[test]
    fn a_relabelled_test_whose_label_still_holds_is_kept() {
        assert_eq!(
            route_intent(&TestIntent::FeatureAcceptance, &verdict(true, None), true),
            IntentRouting::Keep
        );
    }

    // ---- item 7: the Attacker prompt's test-hygiene rules -------------------

    /// Each heretic run produced a generated test that failed for a reason that
    /// had nothing to do with the patch: a preservation probe that also asserted
    /// a new field's default (B v2), a test that instantiated Heretic's Settings
    /// and so parsed pytest's own argv and exited, and a fake tensor with no
    /// `.shape`. The prompt is the only place those mistakes can be prevented, so
    /// the rules that forbid them must be IN it — a rule dropped in a later edit
    /// would otherwise be invisible.
    #[test]
    fn the_attacker_prompt_forbids_the_test_hygiene_mistakes_the_heretic_runs_made() {
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("ONE INTENT PER TEST"),
            "the one-intent-per-test rule must be in the prompt"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("Do NOT mix a new-feature assertion with a preservation assertion"),
            "the mixed-intent prohibition must be explicit"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("PRESERVATION SPECIFICITY")
                && ATTACKER_SYSTEM_PROMPT
                    .contains("pin the INTERNAL calls of a function the issue says to change"),
            "a preservation probe must be told not to pin internals the issue rewires"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("TEST FIXTURES")
                && ATTACKER_SYSTEM_PROMPT.contains("read CLI arguments, environment variables or files")
                && ATTACKER_SYSTEM_PROMPT.contains("MagicMock"),
            "the fixture rules must forbid CLI/env/file-reading settings objects and name the safe alternative"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("implement EVERY attribute the code under test touches"),
            "fakes must be told to implement every attribute the code under test reads"
        );
    }

    /// [VANTA-WHY] §14.10 — the prompt used to say a BehaviorPreservation test
    /// "MUST PASS on BOTH original and patched code", while the BOUNDARY-PROBING
    /// RULE ordered a preservation probe at the exact boundary the patch changes.
    /// On a bad patch the two cannot both hold — the probe at the broken boundary
    /// IS the test that catches it, and it fails on the patched code by definition
    /// — and Ornith-1.0 answered the contradiction with a repetition loop that ran
    /// to the 32768-token cap. BATE checks a preservation test on the ORIGINAL
    /// tree only (`pipeline::bate`), so the both-trees wording was the bug, not the
    /// model. This pins the corrected semantics so the contradiction cannot return.
    #[test]
    fn the_attacker_prompt_does_not_forbid_the_test_that_catches_a_bad_patch() {
        assert!(
            !ATTACKER_SYSTEM_PROMPT.contains("MUST PASS on BOTH"),
            "the both-trees preservation rule is the contradiction that caused the runaways"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("MUST PASS on the ORIGINAL code"),
            "a preservation test must be told BATE only checks it on the original tree"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("FAILS on the PATCHED code")
                && ATTACKER_SYSTEM_PROMPT.contains("that failure IS the finding"),
            "a preservation failure on the patched tree is the finding, not a reason to drop the test"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT
                .contains("NEVER drop a boundary probe because you expect the patch to break it"),
            "the boundary probe must survive the patch it catches"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("BOUNDARY-PROBING RULE"),
            "the boundary-probing rule must still be in the prompt"
        );
    }

    /// [VANTA-WHY] A2 (§14.10) — only the BehaviorPreservation bullet carried the
    /// qualifier that makes a failure on the patched tree evidence rather than an
    /// error. The intent umbrella and the BugReproduction / FeatureAcceptance
    /// bullets still stated the expected original/patched results as a
    /// REQUIREMENT ("The test MUST FAIL on the ORIGINAL code and PASS on the
    /// PATCHED code"), which the BOUNDARY-PROBING RULE orders the very probes that
    /// break it. The run logs show models resolving that contradiction by
    /// inventing a rule to be violated ("violates the intent rule", "violates the
    /// \"MUST PASS on BOTH\" rule") and looping to the output cap, across models,
    /// binaries and attempts. Every intent rule must read as an expectation.
    #[test]
    fn every_intent_rule_reads_as_an_expectation_not_a_requirement() {
        for gone in [
            "each test MUST produce a DIFFERENT result on original vs patched",
            "The test MUST FAIL on the ORIGINAL code and PASS on the PATCHED code.",
            "The test MUST FAIL on the ORIGINAL code (feature absent) and PASS on the PATCHED code (feature present).",
        ] {
            assert!(
                !ATTACKER_SYSTEM_PROMPT.contains(gone),
                "a requirement-shaped intent rule must be gone from the prompt: {gone}"
            );
        }
        for present in [
            "it is an EXPECTATION, never a requirement the patch must satisfy",
            "On a correct patch it is FAIL on the ORIGINAL code and PASS on the PATCHED code",
            "On a correct patch: FAIL on the ORIGINAL code (feature absent) and PASS on the PATCHED code (feature present)",
        ] {
            assert!(
                ATTACKER_SYSTEM_PROMPT.contains(present),
                "the expectation-shaped intent rule must be in the prompt: {present}"
            );
        }
        assert_eq!(
            ATTACKER_SYSTEM_PROMPT
                .matches("that failure IS the finding")
                .count(),
            3,
            "all three intent bullets must carry the same qualifier the preservation \
             bullet already had"
        );
    }

    /// [VANTA-WHY] A3 (§14.10) — the closing line fixed the test COUNT but said
    /// nothing about what to spend the three tests on, and nothing about the cases
    /// the Attacker decided not to cover. A budget rule with no priority lets a
    /// round spend all three probes on the easy interior values and leave the
    /// boundary the patch actually changed untested, and a dropped case then
    /// leaves no trace in the run at all.
    #[test]
    fn the_prompt_orders_what_the_three_tests_are_spent_on() {
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("Write exactly 3 tests"),
            "the budget must stay exactly three tests"
        );
        assert!(
            !ATTACKER_SYSTEM_PROMPT.contains("Write exactly 3 tests. Label each one honestly"),
            "the priority order must come between the budget and the labelling rule"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("THIS order of priority")
                && ATTACKER_SYSTEM_PROMPT.contains("the exact comparison the patch changes"),
            "the prompt must spend the three tests on the changed comparison first"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("name what you left uncovered and why"),
            "an uncovered case must be named in the test's reason rather than dropped silently"
        );
    }

    /// [VANTA-WHY] A3 follow-up (§14.10) — the post-audit Ornith-1.5 run (default
    /// sampling, toy, three runs, all rejected, zero contradiction phrases) showed
    /// two loops the priority line left open.
    ///
    /// (a) The three priority items read as exclusive, so a slot was spent "covering"
    /// item (3) with a second wording of item (1). The reply quotes "(1) the exact
    /// comparison the patch changes, at its boundary value" 41 times and "(3) the
    /// code path the issue asks you to change" 39 times, then says outright "The code
    /// path the issue asks to change is the 1000 boundary. This is the same as test
    /// 1." — and still hunts for a third distinct test, because nothing told it a
    /// test may satisfy several items and count once.
    ///
    /// (b) With no rule for a leftover slot after the two that matter (1000
    /// BugReproduction, 500 BehaviorPreservation), the model cycled through filler
    /// candidates — 1001 (23 mentions), 0 (28), 2000 (19), 999 (19), -1 (15) — and
    /// quoted the issue's "without changing the existing behavior at the other
    /// thresholds." 180 times looking for thresholds that do not exist, saying "Hmm,
    /// I keep coming back to the same thing". The default must name a concrete way to
    /// fill the slot (a BehaviorPreservation probe at an input the ORIGINAL source
    /// answers differently) and pick the first one, not weigh alternatives.
    ///
    /// [VANTA-WHY] A21 follow-up (§14.10) — that leftover-slot default is not
    /// terminal. On the toy, a round-3 reply looped 45 times on the literal phrases
    /// "Let me pick the three most valuable tests", "Both are shared" and "-1 is
    /// shared": every meaningful input was already in the shared set from rounds 1
    /// and 2, so "an input where the ORIGINAL source returns a DIFFERENT result than
    /// at the inputs you have already tested" had no answer left, and the model kept
    /// hunting for a novel input it could not find. The fallback must terminate the
    /// search: an input inside a band that is ALREADY covered (a midpoint), chosen on
    /// the spot, and a repeat of an existing test is an ACCEPTABLE answer because the
    /// driver drops duplicates before BATE.
    #[test]
    fn the_prompt_says_the_priority_items_overlap_and_fills_a_leftover_slot() {
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("are NOT exclusive")
                && ATTACKER_SYSTEM_PROMPT.contains("you count it once")
                && ATTACKER_SYSTEM_PROMPT.contains("are the SAME test, not two"),
            "the priority items must be non-exclusive so a test covering two of them is one test"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT
                .contains("spend each remaining slot on one more BehaviorPreservation probe")
                && ATTACKER_SYSTEM_PROMPT
                    .contains("returns a DIFFERENT result than at the inputs you have already tested")
                && ATTACKER_SYSTEM_PROMPT.contains("Take the FIRST such input you find"),
            "a leftover slot must have a first-found default, not a search for a third thing to test"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("are FINAL")
                && ATTACKER_SYSTEM_PROMPT.contains("do not re-derive the plan"),
            "the plan must be chosen once and not re-derived while drafting"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("TERMINAL FALLBACK:")
                && ATTACKER_SYSTEM_PROMPT.contains("already covered by an existing test")
                && ATTACKER_SYSTEM_PROMPT
                    .contains("A repeat of an existing test is an ACCEPTABLE answer here"),
            "when every defaulted input is already covered the search must terminate on a \
             repeated input inside a covered band, because duplicates are dropped"
        );
    }

    /// [VANTA-WHY] A21 follow-up (§14.10) — the round >= 2 section and the system
    /// prompt's TERMINAL FALLBACK are one rule about the same late-round state and
    /// must not pull apart. A bare "DO NOT REPEAT ANY OF THESE" is unqualified in
    /// exactly the state the fallback exists for: a round whose shared set already
    /// covers every useful input, where the section told the model to find
    /// something new that does not exist. The section must keep the
    /// duplicate-is-not-an-attack fact (it is what makes a repeat harmless) and
    /// must send the exhausted case to the fallback rather than leaving the
    /// prohibition as the last word.
    #[test]
    fn the_round_two_section_points_at_the_terminal_fallback_instead_of_forbidding_repeats() {
        let section = already_probed_section(&[SharedProbe {
            test_name: "test_preserve_add_pair".to_string(),
            test_source: "from calc import add\n\ndef test_preserve_add_pair():\n    assert add(2, 3) == 5\n"
                .to_string(),
        }]);
        assert!(
            !section.contains("DO NOT REPEAT"),
            "an unqualified DO NOT REPEAT contradicts the terminal fallback in the very \
             state the fallback exists for"
        );
        assert!(
            section.contains("does not count as an attack")
                && section.contains("NOT already in this set"),
            "the section must keep the duplicate-is-not-an-attack fact and aim the probe \
             outside the shared set"
        );
        assert!(
            section.contains("TERMINAL FALLBACK") && section.contains("A repeat is acceptable there"),
            "the section must send the exhausted case to the terminal fallback the system \
             prompt defines"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("TERMINAL FALLBACK:")
                && ATTACKER_SYSTEM_PROMPT
                    .contains("A repeat of an existing test is an ACCEPTABLE answer here"),
            "the system prompt must still carry the terminal fallback the section points at"
        );
    }

    /// [VANTA-WHY] A21 follow-up (§14.10) — the 45x round-3 loop was the model
    /// re-listing and re-ranking candidate tests instead of writing what it had.
    /// The one-pass rule is what bounds that deliberation: candidates are listed
    /// once, a candidate already considered is not revisited, and three is the
    /// stop condition.
    #[test]
    fn the_prompt_limits_the_plan_to_one_pass() {
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("ONE PASS:")
                && ATTACKER_SYSTEM_PROMPT.contains("list candidate tests at most once")
                && ATTACKER_SYSTEM_PROMPT
                    .contains("Do not re-list, re-rank or re-check a candidate you have already considered")
                && ATTACKER_SYSTEM_PROMPT.contains("once you have three, write them"),
            "the plan must be one pass: no re-listing, no re-ranking, no re-checking"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("This order and these defaults are FINAL"),
            "the one-pass rule sits with the finality rule, not apart from it"
        );
    }

    /// [VANTA-WHY] A4 (§14.10) — the revise prompt ended with "If you cannot make
    /// the test match its intent, change the intent to match the test's actual
    /// behavior". That is an unconditional licence to relabel: the label can be
    /// moved to whatever the test happens to do, with no requirement that the
    /// ISSUE supports the new label, so a test whose input or expected value is
    /// wrong is laundered into a passing one. The label may still change — the
    /// declared intent really is sometimes the mistake — but only when the label
    /// is what is wrong.
    #[test]
    fn the_revise_prompt_does_not_license_relabelling_away_a_mismatch() {
        assert!(
            !ATTACKER_REVISE_SYSTEM_PROMPT
                .contains("change the intent to match the test's actual behavior"),
            "the unconditional relabel licence must be gone"
        );
        assert!(
            ATTACKER_REVISE_SYSTEM_PROMPT.contains("fix the TEST first")
                && ATTACKER_REVISE_SYSTEM_PROMPT.contains("Relabel only when the LABEL was the mistake")
                && ATTACKER_REVISE_SYSTEM_PROMPT
                    .contains("Relabelling is never a way to make a mismatch disappear"),
            "a mismatch must be repaired in the test, and a relabel must be tied to the label \
             being the mistake"
        );
    }

    /// [VANTA-WHY] A5 (§14.10) — the generate prompt made the `issue_quote` a
    /// REQUIREMENT ("that sentence MUST name the same input the test asserts")
    /// and then made the quote the licence for the label ("A test may be
    /// BugReproduction or FeatureAcceptance ONLY if the quoted issue sentence
    /// explicitly asks for that behaviour"). The quote is a recorded field, not a
    /// verdict: the label is decided by the patch-blind intent audit, and a
    /// requirement the prompt cannot check is a rule the model can only be
    /// punished for breaking. Say that the quote is evidence for the reader.
    #[test]
    fn the_issue_quote_is_recorded_evidence_not_the_licence_for_a_label() {
        for gone in [
            "and that sentence MUST name the same input the test asserts",
            "- A test may be BugReproduction or FeatureAcceptance ONLY if the quoted issue sentence explicitly asks for that behaviour AT THAT INPUT.",
        ] {
            assert!(
                !ATTACKER_SYSTEM_PROMPT.contains(gone),
                "the quote-as-requirement wording must be gone: {gone}"
            );
        }
        for present in [
            "The quote is RECORDED as evidence for the reader",
            "a patch-blind auditor, not your quote, decides whether the issue asks for that behaviour",
            "A patch-blind auditor checks that belief, not your quote",
        ] {
            assert!(
                ATTACKER_SYSTEM_PROMPT.contains(present),
                "the quote-as-evidence wording must be in the prompt: {present}"
            );
        }
    }

    /// [VANTA-WHY] A6 (§14.10) — "Use assertions (assert X == Y) so failures are
    /// clean, not crashes" was unconditional, but the corroboration rule needs the
    /// failure to look like the issue's own symptom: a test only corroborates a
    /// Logic rejection when its exception type is one the issue names
    /// (`issue_symptom_tokens`). A probe that asserts a neighbouring value raises a
    /// bare AssertionError instead, so a patch that reproduces the reported
    /// `NameError` is corroborated by nothing. The prompt has to let the issue
    /// override the style rule — the rule itself stays.
    #[test]
    fn the_assertion_rule_defers_to_a_symptom_the_issue_names() {
        assert!(
            !ATTACKER_SYSTEM_PROMPT.contains("- Use assertions (assert X == Y) so failures are clean, not crashes."),
            "the unconditional assertion rule must be gone"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("- Use assertions (assert X == Y) so failures are clean, not crashes")
                && ATTACKER_SYSTEM_PROMPT.contains("then the test must actually trigger THAT symptom"),
            "the assertion rule must stay and defer to a symptom the issue names"
        );
    }

    /// [VANTA-WHY] A11 (§14.10) — the prompt never said that the baseline may be
    /// measured more than once, so a test that depends on a clock, a random seed,
    /// the network or ordering could be written in good faith and then scrapped
    /// for flakiness with no rule it had broken. The remediation path (fresh
    /// sandbox, then simplify) exists precisely because this happens.
    #[test]
    fn the_prompt_warns_that_a_flaky_test_is_scrapped() {
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("BATE may run your test more than once")
                && ATTACKER_SYSTEM_PROMPT.contains("is flaky and will be scrapped"),
            "the prompt must say a non-deterministic test is scrapped, not blamed on the patch"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("Keep every test deterministic"),
            "the flakiness warning must come with the rule that prevents it"
        );
    }

    /// [VANTA-WHY] A12 (§14.10) — the INTENT-vs-API rule stopped at the
    /// FeatureAcceptance redirect, so the case it does not cover had no answer:
    /// a probe whose behaviour comes from a symbol the patch ADDS, when the issue
    /// asks for no new behaviour either. The Attacker's remaining options were all
    /// dishonest — a preservation label for new API, or an acceptance label with
    /// no issue sentence — and each is then scrapped by the audit, so the probe
    /// was written only to be thrown away. The rule needs its terminal clause.
    #[test]
    fn the_intent_vs_api_rule_ends_by_dropping_the_unlabelable_probe() {
        assert!(
            !ATTACKER_SYSTEM_PROMPT.contains("not BehaviorPreservation.\n\nONE INTENT PER TEST"),
            "the rule may no longer end at the FeatureAcceptance redirect"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("DROP the probe")
                && ATTACKER_SYSTEM_PROMPT
                    .contains("A test that cannot be labelled honestly is not a test"),
            "the unlabelable probe must be dropped rather than mislabelled"
        );
    }

    /// [VANTA-WHY] A13 (§14.10) — PRESERVATION SPECIFICITY banned pinning
    /// internals only in the case "the issue says to change" the function, and
    /// said nothing about how a preservation probe should read the behaviour it
    /// preserves. The B v1 caching probe was written by pinning an internal call's
    /// arguments, which is a preservation probe that cannot survive any rewire the
    /// issue asks for; the rule has to say the probe reads the observable result
    /// and, when only the internal call can be pinned, that there is no probe to
    /// write. The hygiene test pins the internal-calls phrase, so that sentence
    /// stays in the text verbatim.
    #[test]
    fn a_preservation_probe_reads_the_result_not_the_internal_calls() {
        assert!(
            !ATTACKER_SYSTEM_PROMPT.contains("the asserted input. It must NOT pin the INTERNAL calls"),
            "the rule must say how the result is read, not only what may not be pinned"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("through the observable behaviour, not through internal call arguments")
                && ATTACKER_SYSTEM_PROMPT.contains("there is no preservation probe to write"),
            "the probe must read the observable result, and an unpinnable clause must not be probed"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("pin the INTERNAL calls of a function the issue says to change"),
            "the sentence the test-hygiene test pins must survive this reword"
        );
    }

    /// [VANTA-WHY] A14 (§14.10) — both import-facing rules read as if the
    /// Attacker can see the whole repository. In review mode the ORIGINAL SOURCE
    /// is a narrowed excerpt of the files the fix touched, so "only import what
    /// you can see" and "may only exercise APIs that already exist in the ORIGINAL
    /// source" told the model to treat everything it cannot see as absent — and a
    /// symbol the excerpt merely omits then looks like one the patch adds, which
    /// relabels a preservation probe as FeatureAcceptance for no reason. Say the
    /// excerpt is partial and an unseen name is unknown, not absent.
    #[test]
    fn the_import_rules_allow_for_a_narrowed_excerpt_of_the_repo() {
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("The ORIGINAL SOURCE may be a NARROWED excerpt of the repository")
                && ATTACKER_SYSTEM_PROMPT
                    .contains("A name the excerpt does not show is UNKNOWN to you, not absent from the runtime"),
            "the import rules must say the source may be an excerpt and an unseen name is unknown"
        );
        assert!(
            !ATTACKER_SYSTEM_PROMPT
                .contains("ORIGINAL source. If the behaviour you want to check uses a symbol the patch ADDS")
                && ATTACKER_SYSTEM_PROMPT.contains("That source may be an excerpt (see IMPORT RULES)"),
            "the INTENT-vs-API rule must read the excerpt as unknown rather than absent"
        );
    }

    /// [VANTA-WHY] A17 (§14.10) — the Attacker prompt was written against one
    /// repository. It stated "Tests run with PYTHONPATH=src so `from
    /// heretic.modifier import X` works", and gave the defensive-import example
    /// as `import heretic.system as sysmod; fn = getattr(sysmod,
    /// "is_mps_available", None)`. Neither holds for a BugsInPy target: vanta sets
    /// no PYTHONPATH at all — it writes a root `conftest.py` that puts the repo
    /// root and `src/` on `sys.path`, and passes the real import path in the
    /// issue's `[IMPORT PATH HINT]` — and `heretic` / `is_mps_available` exist
    /// only in the repo the prompt was first written against. A model that trusts
    /// the PYTHONPATH line writes imports that hold in exactly one repository; the
    /// example invites the same. State the mechanism that actually runs the tests,
    /// and keep the example neutral.
    #[test]
    fn the_import_path_rules_describe_this_harness_not_one_repository() {
        assert!(
            !ATTACKER_SYSTEM_PROMPT.contains("PYTHONPATH=src")
                && !ATTACKER_SYSTEM_PROMPT.contains("Tests run with PYTHONPATH"),
            "vanta sets no PYTHONPATH, so the claim is false on every target and would \
             send the Attacker writing imports that hold in one repository only"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("Tests run from the REPOSITORY ROOT")
                && ATTACKER_SYSTEM_PROMPT.contains("conftest.py")
                && ATTACKER_SYSTEM_PROMPT
                    .contains("Never assume an environment variable such as PYTHONPATH is set"),
            "the rule must describe what the harness does instead"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT
                .contains(r#"import pkg.module as mod; fn = getattr(mod, "name", None)"#),
            "the defensive-import example must be project-neutral"
        );
        for prompt in [
            ATTACKER_SYSTEM_PROMPT,
            ATTACKER_REVISE_SYSTEM_PROMPT,
            INTENT_AUDITOR_SYSTEM_PROMPT,
            INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED,
        ] {
            assert!(
                !prompt.contains("heretic") && !prompt.contains("is_mps_available"),
                "no prompt may name a repository the run is not against"
            );
        }
    }

    /// [VANTA-WHY] A18 (§14.10) — the prompt opened by telling the Attacker to
    /// "Keep reasoning SHORT (under 100 words)". A thinking model cannot obey
    /// that: its reasoning is generated before the answer is written, against a
    /// token budget the prompt does not set, and the instruction can only be read
    /// as a limit on the answer that follows (Ornith reasoned ~30k tokens before
    /// any JSON appeared). The rule therefore did nothing except invite the model
    /// to truncate its answer to satisfy a word count, which reads to the caller
    /// as a cut-off reply. Keep the budget advice, drop the count.
    #[test]
    fn the_prompt_does_not_demand_a_word_count_a_thinking_model_cannot_keep() {
        assert!(
            !ATTACKER_SYSTEM_PROMPT.contains("under 100 words")
                && !ATTACKER_SYSTEM_PROMPT.contains("Keep reasoning SHORT"),
            "no word count on reasoning: a thinking model reasons before the answer and \
             cannot budget it here, and a model that tries will cut its own JSON short"
        );
        assert!(
            ATTACKER_SYSTEM_PROMPT.contains("Save output budget for the JSON answer"),
            "the budget advice must survive the removal of the count"
        );
    }

    /// [VANTA-WHY] A10 (§14.10) — the simplify prompt asked only for a clear
    /// PASS or FAIL and told the Attacker to "Keep the same intent", never that
    /// the ASSERTION had to survive the simplification. The Diagnostician then
    /// read a passing simplified test as "Unknown resolved", so a body reduced to
    /// `add(2, 3)` — the same inputs, no assertion — retired a real failure. The
    /// rule must now say the assertion and its expected value are kept.
    ///
    /// [VANTA-WHY] A10 follow-up: the reworded rule went into the hoisted
    /// constant while the sentence it replaced stayed beside it, and the first
    /// version of this test only banned the two sentences together — so it
    /// passed over a prompt that still licensed the reduction. It now reads the
    /// prompt through the accessor the call site uses and refuses the old phrase
    /// on its own.
    #[test]
    fn the_simplify_prompt_keeps_the_assertion_it_was_asked_to_simplify() {
        let prompt = attacker_simplify_system_prompt();
        assert!(
            !prompt.contains("Keep the same intent"),
            "the wording that let the assertion be reduced to a bare call must be gone"
        );
        assert!(
            prompt.contains("Use literal values."),
            "the simplification advice itself must survive the rewording"
        );
        for present in [
            "KEEP THE ASSERTION",
            "assert the SAME input and the SAME expected value",
            "it is then SCRAPPED and the failure stands",
        ] {
            assert!(
                prompt.contains(present),
                "the simplify prompt must require the assertion survive: {present}"
            );
        }
    }

    /// [VANTA-WHY] A10 (§14.10) — the deterministic half of the fix. A simplified
    /// test is allowed to pass only if it still asserts the SAME input and
    /// expected value; the integration test covers the routing, this pins the
    /// predicate it routes on.
    #[test]
    fn a_simplification_must_keep_the_input_and_expected_value_it_asserted() {
        let original = "from calc import add\n\ndef test_x():\n    assert add(2, 3) == 5\n";

        assert!(
            simplification_keeps_the_asserted_contract(
                original,
                "from calc import add\n\ndef test_x():\n    assert add(2, 3) == 5\n",
            ),
            "a simplification that keeps the assertion is allowed through"
        );
        assert!(
            !simplification_keeps_the_asserted_contract(
                original,
                "from calc import add\n\ndef test_x():\n    add(2, 3)\n",
            ),
            "dropping the assertion must be a scrap: the same inputs still appear, but nothing \
             is claimed about them"
        );
        assert!(
            !simplification_keeps_the_asserted_contract(
                original,
                "from calc import add\n\ndef test_x():\n    assert add(2, 3) is not None\n",
            ),
            "weakening the assertion loses the expected value, so it is not the same test"
        );
        assert!(
            !simplification_keeps_the_asserted_contract(
                original,
                "from calc import add\n\ndef test_x():\n    assert add(2, 4) == 5\n",
            ),
            "a different input is a different claim, not a simplification"
        );
        assert!(
            !simplification_keeps_the_asserted_contract(
                original,
                "from calc import add\n\ndef test_x():\n    assert True\n",
            ),
            "an assertion that pins nothing is not the same test"
        );
        assert!(
            simplification_keeps_the_asserted_contract(
                "def test_y():\n    assert shipping_band(500) == \"light\"\n",
                "def test_y():\n    assert shipping_band(500) == \"light\"\n",
            ),
            "a string expected value is part of the contract too"
        );
    }

    // ---- item 8: the intent auditor's threaded-value rule -------------------

    /// B v1's revised caching test insisted `get_responses_batched` is called
    /// with exactly the OLD arguments — but the issue was to THREAD the new value
    /// through that very function. Under the old auditor wording that probe could
    /// be answered "the issue does not ask for a change", so it landed on the
    /// preservation label and failed a correct patch. The auditor must be told
    /// that pinning an internal call the issue rewires REQUIRES a change.
    #[test]
    fn the_intent_auditor_is_told_a_pinned_internal_call_requires_a_change() {
        assert!(
            INTENT_AUDITOR_SYSTEM_PROMPT.contains("THREADED THROUGH")
                && INTENT_AUDITOR_SYSTEM_PROMPT
                    .contains("answer \"issue_requires_change\": true for it"),
            "the auditor must answer issue_requires_change=true when a preservation probe pins an \
             internal call the issue threads a new value through"
        );
    }

    // ---- §14.4: the grounded-assertions question ----------------------------

    /// PySnooper/3's false reject was a `BugReproduction` test that reproduced
    /// the reported `NameError` AND asserted output text — `'Starting var' in
    /// content` — that the issue never mentions. Every other arm accepts it, so
    /// without this routing it survives to round 4 and rejects a correct patch.
    ///
    /// The second half pins the PRECEDENCE: with `issue_asks_for_behaviour =
    /// false` the ordinary arms would demote this test, so `ScrapUngrounded`
    /// coming out at all proves it is checked before them.
    #[test]
    fn an_ungrounded_bug_reproduction_test_is_scrapped() {
        assert_eq!(
            route_intent(
                &TestIntent::BugReproduction,
                &grounded_verdict(true, Some(false)),
                false
            ),
            IntentRouting::ScrapUngrounded
        );

        let demote_shaped = IntentVerdict {
            issue_asks_for_behaviour: false,
            issue_requires_change: Some(false),
            assertions_grounded: Some(false),
            reason: "pins an invented output string".to_string(),
        };
        assert_eq!(
            route_intent(&TestIntent::BugReproduction, &demote_shaped, false),
            IntentRouting::ScrapUngrounded,
            "the ungrounded arm must be decided before the demotion the other \
             answers would have selected"
        );
    }

    /// The safety property that matters: a field NOBODY ASKED FOR must not be
    /// able to scrap a test earlier waves accepted. With the flag off the model
    /// never emits `assertions_grounded`, so `None` has to fall through to the
    /// ordinary arms — as does an explicit `true`.
    #[test]
    fn an_absent_or_positive_grounded_answer_never_scraps() {
        assert_eq!(
            route_intent(
                &TestIntent::BugReproduction,
                &grounded_verdict(true, None),
                false
            ),
            IntentRouting::Keep
        );
        assert_eq!(
            route_intent(
                &TestIntent::BugReproduction,
                &grounded_verdict(true, Some(true)),
                false
            ),
            IntentRouting::Keep
        );
        // The flag-off verdict has no opinion at all (see `verdict`).
        assert_eq!(
            route_intent(&TestIntent::BugReproduction, &verdict(true, None), false),
            IntentRouting::Keep
        );
    }

    /// Only `BugReproduction` is classified on the grounded answer. A
    /// `BehaviorPreservation`/`FeatureAcceptance` test answering
    /// `assertions_grounded = false` must NEVER reach `ScrapUngrounded` — its
    /// routing is whatever it was before this question existed, because the
    /// question is about a test that reproduces a bug and those two are the
    /// labels that carry a calibration risk of their own.
    #[test]
    fn an_ungrounded_answer_never_scraps_a_preservation_or_acceptance_test() {
        for grounded in [None, Some(false), Some(true)] {
            for asks in [false, true] {
                for intent in [TestIntent::BehaviorPreservation, TestIntent::FeatureAcceptance] {
                    assert_ne!(
                        route_intent(&intent, &grounded_verdict(asks, grounded), false),
                        IntentRouting::ScrapUngrounded,
                        "an ungrounded answer must not scrap a {intent:?} test \
                         (asks={asks}, assertions_grounded={grounded:?})"
                    );
                }
            }
        }

        // ...and the exact routings those two intents keep.
        assert_eq!(
            route_intent(
                &TestIntent::BehaviorPreservation,
                &grounded_verdict(true, Some(false)),
                false
            ),
            IntentRouting::Keep
        );
        // The demotion direction is deliberately unchanged: demoting already
        // makes a test harder to admit, so an ungrounded FeatureAcceptance test is
        // demoted rather than scrapped.
        assert_eq!(
            route_intent(
                &TestIntent::FeatureAcceptance,
                &grounded_verdict(false, Some(false)),
                false
            ),
            IntentRouting::DemoteToPreservation
        );
    }

    /// [VANTA-WHY] A19 (§14.10) — the auditor's Q1 for a preservation test. The
    /// post-fix coding run (toy, Ornith-1.5, 12 runs, all verdicts correct, 0
    /// Attacker truncations) shows the auditor burning its budget debating whether
    /// the issue's "without changing the existing behavior at the other
    /// thresholds" counts as the issue ASKING for the original result at 0/999/1002:
    /// the question is quoted 17x, "Preserving existing behaviour is always a valid
    /// answer" 23x, "FORBIDS the change; it never authorises it" 18x.
    ///
    /// The routing does not depend on that answer. For a BehaviorPreservation test
    /// Q1 is consulted only when `issue_requires_change == Some(true)`, and a
    /// "must not change" input is exactly one the issue requires no change at. So
    /// the prompt fixes both answers to false and forbids the deliberation.
    ///
    /// The value chosen is `issue_asks_for_behaviour: false` — what the
    /// "forbids, never authorises" rule already implies, and what
    /// `audit_unavailable_verdict` defaults to. Both values leave routing identical
    /// here (pinned below), so the choice is which one cannot manufacture a
    /// relabel: when `requires_change` is mis-answered `Some(true)`, `asks = false`
    /// SCRAPS and `asks = true` PROMOTES, and a promotion is a relabel.
    #[test]
    fn the_prompt_fixes_q1_for_a_preservation_test_the_issue_does_not_ask_to_change() {
        for prompt in [
            INTENT_AUDITOR_SYSTEM_PROMPT,
            INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED,
        ] {
            assert!(
                prompt.contains("when the asserted input is one the issue does NOT ask to change"),
                "every auditor prompt must fix both answers for a preservation input the issue \
                 does not ask to change"
            );
            assert!(
                prompt.contains("a clause that forbids a change never ASKS for the old result"),
                "the fixed answer must be justified by the forbids-never-authorises rule"
            );
            assert!(
                prompt.contains("Decide once and move on"),
                "the rule must forbid the deliberation the run showed"
            );
        }

        // The routing the fixed answer must not move: with the issue requiring no
        // change at the asserted input, a preservation test keeps whether the model
        // would have answered Q1 true or false.
        for asks in [false, true] {
            let v = IntentVerdict {
                issue_asks_for_behaviour: asks,
                issue_requires_change: Some(false),
                assertions_grounded: None,
                reason: "the issue does not ask for a change here".to_string(),
            };
            assert_eq!(
                route_intent(&TestIntent::BehaviorPreservation, &v, false),
                IntentRouting::Keep,
                "a preservation test at an input the issue does not ask to change must keep \
                 (asks={asks})"
            );
        }
        // ...and no answer to Q2 at all (the resolved-unknown case) keeps too.
        let unknown = IntentVerdict {
            issue_asks_for_behaviour: false,
            issue_requires_change: None,
            assertions_grounded: None,
            reason: "resolved".to_string(),
        };
        assert_eq!(
            route_intent(&TestIntent::BehaviorPreservation, &unknown, false),
            IntentRouting::Keep
        );
    }

    /// For a `BugReproduction` test the ungrounded answer decides alone: every
    /// combination of the other two answers still comes out `ScrapUngrounded`,
    /// so the test can never be quietly relabelled into something that keeps
    /// rejecting patches on an invented assertion.
    #[test]
    fn an_ungrounded_bug_reproduction_test_is_never_relabelled_instead_of_scrapped() {
        for asks in [false, true] {
            for requires_change in [None, Some(false), Some(true)] {
                let v = IntentVerdict {
                    issue_asks_for_behaviour: asks,
                    issue_requires_change: requires_change,
                    assertions_grounded: Some(false),
                    reason: "pins an invented output string".to_string(),
                };
                assert_eq!(
                    route_intent(&TestIntent::BugReproduction, &v, false),
                    IntentRouting::ScrapUngrounded
                );
            }
        }
    }

    /// The typed answer: the field is `Option<bool>`, so `Some(false)` (the
    /// model said ungrounded) must be distinguishable from `None` (nobody asked,
    /// or the model dropped the field) — the whole routing above keys on that
    /// distinction.
    #[test]
    fn the_grounded_answer_parses_present_absent_and_false() {
        let present = serde_json::from_str::<IntentVerdict>(
            r#"{"issue_asks_for_behaviour": true, "issue_requires_change": false, "assertions_grounded": false, "reason": "pins an invented string"}"#,
        )
        .expect("a verdict carrying assertions_grounded parses");
        assert_eq!(present.assertions_grounded, Some(false));

        let positive = serde_json::from_str::<IntentVerdict>(
            r#"{"issue_asks_for_behaviour": true, "issue_requires_change": false, "assertions_grounded": true, "reason": "only asserts no-raise"}"#,
        )
        .expect("a grounded verdict parses");
        assert_eq!(positive.assertions_grounded, Some(true));

        let absent = serde_json::from_str::<IntentVerdict>(
            r#"{"issue_asks_for_behaviour": true, "issue_requires_change": false, "reason": "flag off"}"#,
        )
        .expect("the flag-off shape still parses");
        assert_eq!(absent.assertions_grounded, None);
    }

    /// [VANTA-WHY] The grounded variant is a SEPARATE literal, so an edit to one
    /// prompt that is not mirrored in the other would silently make the two asks
    /// disagree about the rules they share. This fails on that edit.
    #[test]
    fn the_grounded_prompt_keeps_the_shared_rules() {
        const SHARED: [&str; 4] = [
            "The issue is the only source of requirements.",
            "A sentence that forbids changing behaviour",
            "THREADED THROUGH",
            "Preserving existing behaviour is always a valid answer.",
        ];
        for rule in SHARED {
            assert!(
                INTENT_AUDITOR_SYSTEM_PROMPT.contains(rule),
                "the base prompt must carry the shared rule: {rule}"
            );
            assert!(
                INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED.contains(rule),
                "the grounded prompt must carry the same shared rule verbatim: {rule}"
            );
        }

        assert!(
            INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED.contains("\"assertions_grounded\""),
            "the grounded prompt must ask the third question and pin its JSON key"
        );
        assert!(
            !INTENT_AUDITOR_SYSTEM_PROMPT.contains("assertions_grounded"),
            "the flag-off prompt must be unchanged — it may not mention the third question"
        );
        assert_eq!(
            intent_auditor_system_prompt(false),
            INTENT_AUDITOR_SYSTEM_PROMPT
        );
        assert_eq!(
            intent_auditor_system_prompt(true),
            INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED
        );
    }

    /// [VANTA-WHY] A7 (§14.10) — the closing line of the grounding rules was
    /// written as if the auditor had been handed several tests ("every test you
    /// are given"), in a prompt whose own first line says it is given ONE test,
    /// and it never named WHICH question it was closing. Read literally it asked
    /// for answers to a question about tests that were not in the message. Name
    /// question 3 and the test it belongs to.
    #[test]
    fn the_grounding_rule_names_the_question_and_the_test_it_closes() {
        for prompt in [
            INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED,
        ] {
            assert!(
                !prompt.contains("Answer this for every test you are given"),
                "the grounding rule may not address 'every test you are given': in the \
                 one-test prompt there is no such set, and in the batch prompt 'this' \
                 names no question"
            );
            assert!(
                prompt.contains("Answer question 3 for"),
                "the closing grounding rule must name the question it closes"
            );
        }
        assert!(
            INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED.contains("for the test you are given"),
            "the one-test prompt must speak about the single test it was given"
        );
        assert!(
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED.contains("for EACH numbered test"),
            "the batch prompt must speak about each numbered test"
        );
    }

    /// [VANTA-WHY] A19b (§14.10) — the grounded question was asked "for the test
    /// you are given, whatever its declared intent", but `route_intent` acts on
    /// `assertions_grounded` for ONE declared intent only: a BugReproduction test
    /// whose answer is `Some(false)`. The post-fix Ornith-1.5 run shows what that
    /// mismatch costs — the auditor cut off at its 6,144-token batch cap 21 times
    /// across 8 of 12 runs, and the tail of its replies is question 3 reasoned out
    /// for BehaviorPreservation tests ("Is 'light' at 0 a specific value the issue
    /// never states?") whose answer the pipeline then discards. Ask question 3
    /// only for a BugReproduction test and write null otherwise: a missing or null
    /// answer is `None`, which no route reads as ungrounded, so no decision moves.
    #[test]
    fn the_grounding_question_is_asked_only_for_a_bug_reproduction_test() {
        for prompt in [
            INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED,
        ] {
            assert!(
                prompt.contains("only when its declared intent is BugReproduction"),
                "a grounded prompt must scope question 3 to a BugReproduction test"
            );
            assert!(
                prompt.contains(r#"write "assertions_grounded": null"#),
                "a grounded prompt must say what to write for any other intent"
            );
            assert!(
                prompt.contains(r#""assertions_grounded": true|false|null"#),
                "the JSON contract must allow the null the other intents write"
            );
            assert!(
                !prompt.contains("whatever its declared intent"),
                "no grounded prompt may ask question 3 for every declared intent — the \
                 answer is routed on for a BugReproduction test alone"
            );
        }

        for prompt in [
            INTENT_AUDITOR_SYSTEM_PROMPT,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT,
        ] {
            assert!(
                !prompt.contains("assertions_grounded"),
                "the flag-off prompts may not gain the third question"
            );
        }
    }

    /// [VANTA-WHY] A19c (§14.10) — "without changing the existing behaviour at the
    /// other thresholds" is the single most-quoted sentence in the auditor's
    /// replies: the Q1 quote 168x, the follow-up run's 57x, "at the other
    /// thresholds" 180x. The model spends its budget deciding WHICH inputs count as
    /// "other thresholds" — is 0 one? 999? — a question the issue never frames. All
    /// four prompts now state the scope once, so there is nothing to enumerate.
    #[test]
    fn the_prompt_scopes_a_must_not_change_clause_to_every_unchanged_input() {
        for prompt in [
            INTENT_AUDITOR_SYSTEM_PROMPT,
            INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED,
        ] {
            assert!(
                prompt.contains(
                    r#""must not change" / "without changing X" clause covers EVERY input"#
                ),
                "every auditor prompt must state what a must-not-change clause covers"
            );
            assert!(
                prompt.contains(r#"do not debate which of them count as "thresholds""#),
                "the scope rule must forbid the threshold enumeration the replies show"
            );
        }
    }

    /// [VANTA-WHY] A19 (§14.10) — a)–c) are wording-only: they change which answers
    /// the auditor is TOLD to give, never which answer `route_intent` reads. This is
    /// the routing evidence for that claim, over the whole grid of intents, answers
    /// and relabel state.
    ///
    /// (b) makes every non-BugReproduction test answer question 3 "null" (`None`);
    /// (a) and (c) make a preservation input the issue does not ask to change answer
    /// `asks = false, requires_change = false`. Each fix is checked against the
    /// answers it replaces, so no fixture decision can move. BugReproduction is
    /// excluded from the (b) equivalence for the reason the fix keeps asking it
    /// there: `ScrapUngrounded` fires on `Some(false)` for that intent alone.
    #[test]
    fn the_wording_fixes_never_move_a_routing_decision() {
        let of = |asks: bool, rc: Option<bool>, grounded: Option<bool>| IntentVerdict {
            issue_asks_for_behaviour: asks,
            issue_requires_change: rc,
            assertions_grounded: grounded,
            reason: "the issue says so".to_string(),
        };

        for intent in [
            TestIntent::BehaviorPreservation,
            TestIntent::FeatureAcceptance,
        ] {
            for asks in [false, true] {
                for rc in [None, Some(false), Some(true)] {
                    for relabeled in [false, true] {
                        let null = route_intent(&intent, &of(asks, rc, None), relabeled);
                        for grounded in [Some(false), Some(true)] {
                            assert_eq!(
                                null,
                                route_intent(&intent, &of(asks, rc, grounded), relabeled),
                                "a null question-3 answer must route exactly where \
                                 {grounded:?} did for {intent:?} (asks={asks}, \
                                 requires_change={rc:?}, relabeled={relabeled})"
                            );
                        }
                    }
                }
            }
        }

        // (a)+(c): a preservation test at an input the issue does not ask to change.
        // The rule fixes asks = false, requires_change = false; all four answers the
        // model might otherwise have given route to the same place.
        for asks in [false, true] {
            for rc in [None, Some(false)] {
                assert_eq!(
                    route_intent(&TestIntent::BehaviorPreservation, &of(asks, rc, None), false),
                    IntentRouting::Keep,
                    "a preservation test at an input the issue does not ask to change must \
                     keep (asks={asks}, requires_change={rc:?})"
                );
            }
        }

        // The two answers the rule forbids, and why (a) fixes asks = false: with
        // requires_change mis-answered `true`, asks = false SCAPS and asks = true
        // PROMOTES — and a promotion is a relabel.
        assert_eq!(
            route_intent(
                &TestIntent::BehaviorPreservation,
                &of(false, Some(true), None),
                false
            ),
            IntentRouting::ScrapRequiresChange
        );
        assert_eq!(
            route_intent(
                &TestIntent::BehaviorPreservation,
                &of(true, Some(true), None),
                false
            ),
            IntentRouting::PromoteToFeature
        );
    }

    /// [VANTA-WHY] A8 (§14.10) — all four auditor prompts opened with "Answer two
    /// questions", a count that is only true of the two flag-off prompts: with
    /// `VANTA_GROUNDED_AUDIT=on` a third question follows and the JSON contract
    /// gains a third key. A model that trusts the count can answer the two named
    /// questions and stop, dropping `assertions_grounded` — the answer the whole
    /// grounded variant exists to collect. Let the heading promise the questions
    /// actually printed below it.
    #[test]
    fn the_auditor_heading_does_not_count_a_question_it_may_not_ask() {
        for prompt in [
            INTENT_AUDITOR_SYSTEM_PROMPT,
            INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED,
        ] {
            assert!(
                !prompt.contains("Answer two questions")
                    && !prompt.contains("Answer these two questions"),
                "an auditor prompt may not promise a fixed count of two questions: the \
                 grounded variants ask three"
            );
            assert!(
                prompt.contains("Answer these questions about the input"),
                "the heading must promise the questions it actually prints"
            );
        }
    }

    /// [VANTA-WHY] A9 (§14.10) — the batched prompts kept the single-test
    /// wording "for the input THIS test asserts". A batch message holds several
    /// numbered tests and no single "this" one, so the rule that stops a
    /// one-input issue from being generalised to every input named a test the
    /// auditor could not pick out — exactly the ambiguity the batching exists to
    /// remove. Say "that numbered test"; the one-test prompts keep "THIS test",
    /// where it is unambiguous.
    #[test]
    fn the_batch_rule_points_at_the_numbered_test_not_an_ambiguous_this() {
        for prompt in [
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED,
        ] {
            assert!(
                !prompt.contains("for the input THIS test asserts"),
                "a batch prompt may not say 'THIS test' when several are numbered"
            );
            assert!(
                prompt.contains("for the input THAT numbered test asserts"),
                "the batch rule must name the numbered test it is about"
            );
        }
        for prompt in [INTENT_AUDITOR_SYSTEM_PROMPT, INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED] {
            assert!(
                prompt.contains("for the input THIS test asserts"),
                "the one-test prompts must keep the wording that is unambiguous there"
            );
        }
    }

    // ---- §14.5 batched audit (schema, transport-neutral decision) -----------

    fn batch_entry(index: usize, asks: bool) -> IntentAuditEntry {
        IntentAuditEntry {
            index,
            issue_asks_for_behaviour: asks,
            issue_requires_change: None,
            assertions_grounded: None,
            reason: "the issue says so".to_string(),
        }
    }

    /// The batched answer is the single-test contract plus an index, so an entry
    /// is turned into exactly the verdict the single-test call would have parsed.
    #[test]
    fn the_batch_entry_carries_the_single_test_answers() {
        let entry = IntentAuditEntry {
            index: 3,
            issue_asks_for_behaviour: true,
            issue_requires_change: Some(false),
            assertions_grounded: Some(true),
            reason: "the issue names the input".to_string(),
        };
        assert_eq!(
            entry.verdict(),
            IntentVerdict {
                issue_asks_for_behaviour: true,
                issue_requires_change: Some(false),
                assertions_grounded: Some(true),
                reason: "the issue names the input".to_string(),
            }
        );
    }

    /// A model that numbers the tests in a different order (or lists them out of
    /// order) must still produce the round's own order: the index is what binds
    /// an answer to a test, not the position in the array.
    #[test]
    fn the_batch_maps_entries_back_onto_the_round_in_the_round_s_order() {
        let batch = IntentBatchVerdict {
            tests: vec![
                IntentAuditEntry {
                    index: 2,
                    ..batch_entry(2, false)
                },
                IntentAuditEntry {
                    index: 3,
                    ..batch_entry(3, true)
                },
                IntentAuditEntry {
                    index: 1,
                    ..batch_entry(1, true)
                },
            ],
        };
        let verdicts = batch.by_index(3).expect("every index appears once");
        assert_eq!(
            verdicts
                .iter()
                .map(|v| (v.issue_asks_for_behaviour, v.reason.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (true, "the issue says so"),
                (false, "the issue says so"),
                (true, "the issue says so"),
            ]
        );
    }

    /// One forgotten test would otherwise be silently demoted, which is exactly
    /// the "a decision nobody made" failure this project exists to prevent. The
    /// whole response is a parse failure, so the call is retried instead.
    ///
    /// [VANTA-WHY] An answer that OMITS a test cannot reach the `is missing` arm
    /// on its own: the round's size is checked first, and with the size enforced
    /// a gap in the numbering can only be filled by a duplicate or by an index
    /// outside `1..=count`. So the omission is asserted in the three shapes it can
    /// actually take — all of them a parse failure of the whole response, none of
    /// them a shrug about one test.
    #[test]
    fn an_answer_that_omits_a_test_fails_the_whole_batch() {
        let cases = [
            (
                vec![batch_entry(1, true), batch_entry(2, true)],
                "answered 2 test(s) but 3 were sent",
            ),
            (
                vec![
                    batch_entry(1, true),
                    batch_entry(2, true),
                    batch_entry(2, false),
                ],
                "index 2 appears twice",
            ),
            (
                vec![
                    batch_entry(1, true),
                    batch_entry(2, true),
                    batch_entry(4, true),
                ],
                "is not one of 1..=3",
            ),
        ];
        for (entries, expected) in cases {
            let batch = IntentBatchVerdict { tests: entries };
            let error = batch
                .by_index(3)
                .expect_err("a round of three tests needs three answers");
            assert!(error.contains(expected), "expected {expected:?}; got {error}");
        }
    }

    #[test]
    fn a_duplicate_index_fails_the_whole_batch() {
        let batch = IntentBatchVerdict {
            tests: vec![
                batch_entry(1, true),
                batch_entry(2, true),
                batch_entry(2, false),
            ],
        };
        let error = batch.by_index(3).expect_err("index 2 was answered twice");
        assert!(error.contains("index 2 appears twice"), "{error}");
    }

    /// Both ends of the range, because the prompt is 1-based: index 0 is off by
    /// one and `count + 1` is a test that was not sent.
    #[test]
    fn an_unknown_index_fails_the_whole_batch() {
        for index in [0_usize, 4] {
            let batch = IntentBatchVerdict {
                tests: vec![
                    batch_entry(1, true),
                    batch_entry(2, true),
                    batch_entry(index, true),
                ],
            };
            let error = batch
                .by_index(3)
                .expect_err("an index outside 1..=3 cannot be an answer");
            assert!(error.contains("is not one of 1..=3"), "{error}");
        }
    }

    /// The documented batch JSON parses, and an omitted `issue_requires_change`
    /// or `assertions_grounded` is UNKNOWN — never `false`. Reading a missing
    /// answer as a negative one is what would let a truncated response scrap a
    /// test or demote it.
    #[test]
    fn an_omitted_change_or_grounded_field_is_unknown_not_an_answer() {
        let raw = r#"{"tests": [
            {"index": 1, "issue_asks_for_behaviour": true, "reason": "a"},
            {"index": 2, "issue_asks_for_behaviour": false, "issue_requires_change": true,
             "assertions_grounded": false, "reason": "b"}
        ]}"#;
        let batch: IntentBatchVerdict = serde_json::from_str(raw).expect("the documented schema");
        let verdicts = batch.by_index(2).expect("every index appears once");
        assert_eq!(verdicts[0].issue_requires_change, None);
        assert_eq!(verdicts[0].assertions_grounded, None);
        assert_eq!(verdicts[1].issue_requires_change, Some(true));
        assert_eq!(verdicts[1].assertions_grounded, Some(false));
    }

    /// §14.8 — batched vs unbatched give the same per-test decisions and the same
    /// records. Both transports parse the SAME three answers into an
    /// [`IntentVerdict`] (the single call deserializes it directly, the batch maps
    /// its entries onto the round) and hand it to [`apply_intent_verdict`], so the
    /// equivalence is asserted over the decisions that come out of each shape.
    #[test]
    fn the_batched_and_single_test_answers_decide_the_same_way() {
        let round = vec![
            generated(
                TestIntent::BugReproduction,
                "def test_boundary():\n    assert shipping_band(1000) == \"standard\"\n",
                Some("`shipping_band(1000)` should return `\"standard\"`"),
            ),
            generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(500) == \"light\"\n",
                None,
            ),
            generated(
                TestIntent::BehaviorPreservation,
                "def test_boundary():\n    assert shipping_band(1000) == \"standard\"\n",
                None,
            ),
        ];
        // One answer per test, in the single-test JSON shape and the batched one.
        let answers = [
            (true, None, None),
            (false, None, None),
            (true, Some(true), None),
        ];
        let singles: Vec<IntentVerdict> = answers
            .iter()
            .map(|(asks, change, grounded)| IntentVerdict {
                issue_asks_for_behaviour: *asks,
                issue_requires_change: *change,
                assertions_grounded: *grounded,
                reason: "the issue says so".to_string(),
            })
            .collect();
        let batch_json = format!(
            "{{\"tests\": [{}]}}",
            answers
                .iter()
                .enumerate()
                .map(|(i, (asks, change, grounded))| format!(
                    "{{\"index\": {}, \"issue_asks_for_behaviour\": {}, ",
                    i + 1,
                    asks
                ) + &match (change, grounded) {
                    (Some(c), _) => format!("\"issue_requires_change\": {c}, "),
                    (None, Some(g)) => format!("\"assertions_grounded\": {g}, "),
                    (None, None) => String::new(),
                } + "\"reason\": \"the issue says so\"}")
                .collect::<Vec<_>>()
                .join(", ")
        );
        let batched: IntentBatchVerdict =
            serde_json::from_str(&batch_json).expect("the batch JSON is the documented schema");
        let batched_verdicts = batched.by_index(round.len()).expect("one entry per test");

        for ((test, single), batched) in round.iter().zip(&singles).zip(batched_verdicts) {
            assert_eq!(
                single, &batched,
                "the same three answers must parse to the same verdict either way"
            );
            assert_eq!(
                apply_intent_verdict(test, FIXTURE_ISSUE, single.clone()),
                apply_intent_verdict(test, FIXTURE_ISSUE, batched),
                "the decision and the record may not depend on the transport"
            );
        }
    }

    /// A failed call must produce the same verdict in both transports, and that
    /// verdict is the SAFE one: nothing is scrapped, and a non-preservation label
    /// that nobody verified does not stand — it is demoted, never admitted.
    #[test]
    fn an_unavailable_auditor_answers_unknown_in_the_batch_s_shape_too() {
        let unavailable = audit_unavailable_verdict(&"connection reset");
        assert!(!unavailable.issue_asks_for_behaviour);
        assert_eq!(unavailable.issue_requires_change, None);
        assert_eq!(unavailable.assertions_grounded, None);
        assert!(
            unavailable.reason.contains("connection reset"),
            "{}",
            unavailable.reason
        );

        // A probe that already declares preservation is untouched: there is
        // nothing an absent auditor could safely revise.
        let (kept, relabel, record) = apply_intent_verdict(
            &generated(
                TestIntent::BehaviorPreservation,
                "def test_boundary():\n    assert True\n",
                None,
            ),
            FIXTURE_ISSUE,
            audit_unavailable_verdict(&"connection reset"),
        );
        assert!(relabel.is_none());
        assert_eq!(
            kept.expect("the probe survives").intent,
            TestIntent::BehaviorPreservation
        );
        assert!(!record.expect("a record is always written").scrapped);

        // An unverified non-preservation label is demoted to preservation — the
        // BATE requirement that follows (baseline must PASS) is the one that
        // cannot admit a patch-introduced behaviour as if the issue asked for it.
        for intent in [TestIntent::BugReproduction, TestIntent::FeatureAcceptance] {
            let declared = format!("{intent:?}");
            let (updated, relabel, record) = apply_intent_verdict(
                &generated(intent, "def test_boundary():\n    assert True\n", None),
                FIXTURE_ISSUE,
                audit_unavailable_verdict(&"connection reset"),
            );
            assert_eq!(
                updated.expect("the test survives").intent,
                TestIntent::BehaviorPreservation,
                "{declared} may not keep an unverified label"
            );
            let relabel = relabel.expect("the demotion is a relabel and must be logged");
            assert_eq!(relabel.to_intent, TestIntent::BehaviorPreservation);
            assert!(!record.expect("a record is always written").scrapped);
        }
    }

    /// [VANTA-WHY] The batch prompts are SEPARATE literals, so an edit to one
    /// that is not mirrored in the others would silently make the asks disagree.
    #[test]
    fn the_batch_prompt_keeps_the_shared_rules() {
        const SHARED: [&str; 5] = [
            "The issue is the only source of requirements.",
            "A sentence that forbids changing behaviour",
            "THREADED THROUGH",
            "Preserving existing behaviour is always a valid answer.",
            "YOU MUST output valid JSON. First char '{', last char '}'.",
        ];
        for rule in SHARED {
            for prompt in [
                INTENT_AUDITOR_SYSTEM_PROMPT,
                INTENT_AUDITOR_SYSTEM_PROMPT_GROUNDED,
                INTENT_AUDITOR_BATCH_SYSTEM_PROMPT,
                INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED,
            ] {
                assert!(
                    prompt.contains(rule),
                    "every auditor prompt must carry the shared rule {rule}"
                );
            }
        }

        for prompt in [
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT,
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED,
        ] {
            assert!(
                prompt.contains("- Answer EVERY numbered test exactly once"),
                "a batch prompt must say that each numbered test is answered once"
            );
            assert!(
                prompt.contains(r#"{"tests": [{"index": 1"#),
                "a batch prompt must pin the batched JSON contract"
            );
        }
        assert!(
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED.contains("\"assertions_grounded\""),
            "the grounded batch prompt must ask the third question"
        );
        assert!(
            !INTENT_AUDITOR_BATCH_SYSTEM_PROMPT.contains("assertions_grounded"),
            "a batch run without the grounded flag may not scrap on an answer nobody asked for"
        );

        assert_eq!(
            intent_auditor_batch_system_prompt(false),
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT
        );
        assert_eq!(
            intent_auditor_batch_system_prompt(true),
            INTENT_AUDITOR_BATCH_SYSTEM_PROMPT_GROUNDED
        );
    }

    // ---- quote evidence (RECORDED, never the rule) --------------------------

    #[test]
    fn quote_evidence_reports_a_missing_quote() {
        let evidence = observe_quote_evidence(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(500) == \"light\"\n",
                None,
            ),
            FIXTURE_ISSUE,
        );
        assert!(!evidence.quote_present);
        assert!(!evidence.quote_in_issue);
        assert_eq!(evidence.asserted_inputs, vec!["500".to_string()]);
    }

    /// The exact failure the fixture matrix produced: a verbatim issue sentence
    /// that argues AGAINST the label. The evidence must make that visible.
    #[test]
    fn quote_evidence_shows_a_verbatim_quote_that_misses_the_asserted_input() {
        let evidence = observe_quote_evidence(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(500) == \"light\"\n",
                Some(
                    "Please fix that boundary bug without changing the existing behavior at the other thresholds.",
                ),
            ),
            FIXTURE_ISSUE,
        );
        assert!(evidence.quote_present);
        assert!(
            evidence.quote_in_issue,
            "the quote really is verbatim in the issue — which is exactly why a \
             substring check is not an endorsement"
        );
        assert!(
            !evidence.quote_names_asserted_input,
            "the quote is about other thresholds, not the asserted 500g input"
        );
    }

    #[test]
    fn quote_evidence_marks_a_neighbouring_input_as_unmatched() {
        let evidence = observe_quote_evidence(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(500) == \"light\"\n",
                Some("the exact `1000` gram edge falls into the wrong band."),
            ),
            FIXTURE_ISSUE,
        );
        assert!(evidence.quote_in_issue);
        assert!(!evidence.quote_names_asserted_input);
    }

    /// A quote that does name the asserted input is recorded as anchored, even
    /// though the Attacker dropped the markdown backticks.
    #[test]
    fn quote_evidence_accepts_a_quote_without_backticks() {
        for quote in [
            "shipping_band(1000) should return \"standard\", not \"heavy\".",
            "`shipping_band(1000)` should return `\"standard\"`, not `\"heavy\"`.",
        ] {
            let evidence = observe_quote_evidence(
                &generated(
                    TestIntent::BugReproduction,
                    "def test_boundary():\n    assert shipping_band(1000) == \"standard\"\n",
                    Some(quote),
                ),
                FIXTURE_ISSUE,
            );
            assert!(evidence.quote_in_issue, "quote={quote}");
            assert!(evidence.quote_names_asserted_input, "quote={quote}");
        }
    }

    #[test]
    fn quote_evidence_matching_normalizes_whitespace() {
        let evidence = observe_quote_evidence(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert shipping_band(1000) == \"standard\"\n",
                Some("shipping_band(1000)   should return  \"standard\""),
            ),
            "shipping_band(1000)\nshould return \"standard\" and not \"heavy\"",
        );
        assert!(evidence.quote_in_issue);
        assert!(evidence.quote_names_asserted_input);
    }

    /// A test with no literal call arguments cannot be anchored; the evidence
    /// must say so rather than claiming the quote matches.
    #[test]
    fn quote_evidence_reports_an_unanchorable_test() {
        let evidence = observe_quote_evidence(
            &generated(
                TestIntent::FeatureAcceptance,
                "def test_boundary():\n    assert compute() is not None\n",
                Some("A new compute helper must exist."),
            ),
            "A new compute helper must exist.",
        );
        assert!(evidence.quote_in_issue);
        assert!(evidence.asserted_inputs.is_empty());
        assert!(
            evidence.quote_names_asserted_input,
            "with nothing to anchor on, presence is all the evidence can report"
        );
    }

    #[test]
    fn asserted_input_literals_reads_call_arguments_only() {
        assert_eq!(
            asserted_input_literals("def t():\n    assert shipping_band(500) == \"light\"\n"),
            vec!["500".to_string()]
        );
        assert!(asserted_input_literals("def t():\n    assert add(2, 3) == 5\n")
            .iter()
            .all(|l| l == "2" || l == "3"));
        // `== 1500` is an expected value, not an argument.
        assert_eq!(
            asserted_input_literals("def t():\n    assert shipping_band(500) == 1500\n"),
            vec!["500".to_string()]
        );
    }

    #[test]
    fn quote_names_literal_respects_token_boundaries() {
        assert!(quote_names_literal("shipping_band(500) is light", "500"));
        assert!(!quote_names_literal("shipping_band(1500) is heavy", "500"));
        assert!(quote_names_literal("add(2, 3)", "2"));
        assert!(!quote_names_literal("add(20, 3)", "2"));
    }

    /// [VANTA-WHY] A19f (§14.10) — the recorded prompt hash is only worth having
    /// if it is the hash of the text the call sites SEND; otherwise it is another
    /// hand-maintained copy that can go stale exactly like the version label did
    /// (`3588728`, `d4ea314`). So the hash's ingredients are pinned against the
    /// three call sites: each slot is the value a call site assigns, in the order
    /// the accessor declares.
    ///
    /// [VANTA-WHY] The call-site lines are read out of this file's own source
    /// because `let system_prompt = ATTACKER_SYSTEM_PROMPT;` is the only thing
    /// that makes "the prompt the hash covers" and "the prompt the model
    /// receives" one text. A duplicated literal elsewhere in the module would
    /// satisfy every other assertion here and still let the two drift.
    #[test]
    fn attacker_prompt_hash_follows_the_call_sites() {
        let prompts = attacker_system_prompts();
        assert_eq!(
            prompts[0], ATTACKER_SYSTEM_PROMPT,
            "slot 0 must be the generate prompt"
        );
        assert_eq!(
            prompts[1], ATTACKER_REVISE_SYSTEM_PROMPT,
            "slot 1 must be the BATE-revision prompt"
        );
        assert_eq!(
            prompts[2],
            attacker_simplify_system_prompt(),
            "slot 2 must be the Unknown-path prompt"
        );

        let source = include_str!("attacker.rs");
        for call_site in [
            "let system_prompt = ATTACKER_SYSTEM_PROMPT;",
            "let system_prompt = ATTACKER_REVISE_SYSTEM_PROMPT;",
            "let system_prompt = attacker_simplify_system_prompt();",
        ] {
            assert!(
                source.contains(call_site),
                "every hashed Attacker prompt must be the one a call site sends; \
                 this file has no {call_site:?}"
            );
        }

        assert_eq!(
            crate::pipeline::prompt_hashes()
                .split(';')
                .find_map(|field| field.strip_prefix("attacker="))
                .expect("prompt_hashes must carry the attacker slot"),
            crate::core::prompt_content_hash(&prompts),
            "the recorded attacker hash must be the hash of the prompts its call sites send"
        );
    }

    /// [VANTA-WHY] §14.22 — `VANTA_ATTACKER_CONCISE` is the one PROMPT arm, and
    /// the single change that would quietly corrupt its accounting is the sentence
    /// leaking into a system prompt: the §14.10-f `attacker=` hash is taken over
    /// exactly the three system prompts, so a suffix that moved into one of them
    /// would change the hash while `run.summary` still reported the switch — two
    /// different sent texts sharing one recorded hash. So the sentence is pinned
    /// here twice: it must be the exact text the arm appends, and it must NOT be a
    /// substring of any prompt the hash covers. The arm rides the USER message by
    /// design; this test is what keeps that true.
    #[test]
    fn the_concise_suffix_is_the_sentence_and_is_not_in_a_system_prompt() {
        const SENTENCE: &str =
            "Once each test is written and checked once, output the JSON immediately; do not re-verify formatting.";
        let suffix = attacker_concise_suffix();
        assert_eq!(
            suffix,
            format!("\n\n{SENTENCE}"),
            "the arm's sentence is the text the model is sent; reword it here deliberately"
        );
        for (slot, prompt) in attacker_system_prompts().iter().enumerate() {
            assert!(
                !prompt.contains(SENTENCE),
                "system prompt slot {slot} is covered by the attacker= hash; the §14.22 \
                 sentence must ride the USER message, or the hash stops naming the prompts sent"
            );
        }
    }
}
