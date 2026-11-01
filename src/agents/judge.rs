// ============================================================================
// JUDGE AGENT — issue + context + candidate patch → verdict (M3 item 7)
// ============================================================================
//
// [VANTA-WHY] M3 item 7. The judge arm replaces the whole adversarial pipeline
// with ONE model call that reads the same material the `vanta` arm's roles read
// and answers a single question: does this candidate resolve the issue? It is
// the study's "LLM-as-judge" comparison against the test-driven arms, so it is
// deliberately NOT a pass/fail test run — no tool use, no execution — and its
// verdict is recorded under its own tokens (`JudgeApproved`/`JudgeRejected`/
// `JudgeFailed`) rather than borrowed from the test-driven ones, so the log
// always says which mechanism reached the answer.
//
// [VANTA-WHY] The judge is shown the SAME context selection and budget the
// `vanta` review arm is (plans/M3-plan.md item 7's fairness rule): the driver
// renders `select_review_prompt`'s output, the same value the Attacker, the
// auditor, BATE and the Diagnostician are shown, and passes it here verbatim.
// Sampling matches too — the same model, the same `reasoning_effort` the other
// call sites send, and the `judge` site's pins in `SAMPLING_SCHEDULE`.

use crate::config::LlmConfig;
use crate::llm::{build_reqwest_client, extract_json_candidate, structured_call_with_retry};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, info, trace};

// ============================================================================
// JUDGE_PROMPT_VERSION — the run-identity tag for the judge's prompt
// ============================================================================
//
// [VANTA-WHY] The plan requires the judge's prompt text to be versioned and
// hashed, exactly as the other prompts are, so a wording change is visible in
// the log instead of looking like "no change". Bump this whenever the system or
// user prompt text below changes; the content hash
// (`core::prompt_content_hash`) is the belt to this version tag's braces.
pub const JUDGE_PROMPT_VERSION: &str = "1";

// ============================================================================
// JUDGE_SYSTEM_PROMPT — the judging instruction, as a const
// ============================================================================
//
// [VANTA-WHY] Hoisted out of the call site for the same reason `CREATOR_SYSTEM_PROMPT`
// is: the hash must be taken over the text the model ACTUALLY sees, so "the
// prompt the call sends" and "the prompt the hash covers" are one value by
// construction.
pub const JUDGE_SYSTEM_PROMPT: &str = r#"You are Vanta Judge. You are given an issue, the repository context a developer would need to understand the change, and a candidate patch. Decide whether the candidate patch correctly resolves the issue WITHOUT breaking existing behaviour.

You are the ONLY judge. No test is run and you must not run anything: read the issue and the diff and reason about them. Judge the patch on its own merits — do not assume the patch author is right.

YOU MUST output valid JSON. NO EXCEPTIONS.
  1. First character of content MUST be '{'
  2. Last character MUST be the matching '}'
  3. No prose, no markdown fences
  4. Do not stop until the full '}' is written

Required JSON schema:
{
  "verdict":    "approve" | "reject",
  "confidence": number 0.0 to 1.0,
  "reason":     string — 1 to 3 sentences naming what in the diff decides it
}

Answer "approve" only when the diff genuinely fixes the issue and does not regress existing behaviour. When you are not sure, answer "reject"."#;

/// The judging prompt the call site sends.
///
/// Routed through an accessor so the hash is taken over the exact text the model
/// receives (see `creator_system_prompt` for the incident that makes that worth
/// pinning).
pub fn judge_system_prompt() -> &'static str {
    JUDGE_SYSTEM_PROMPT
}

// ============================================================================
// The USER prompt templates — one macro per block the call sends (C4)
// ============================================================================
//
// [VANTA-WHY] C4 — the judge's USER prompt used to be written inline at the call
// site, so only the system half of the prompt was templated and hashed: the
// frozen config's `[prompts]` block (via `judge_prompt_hash`) described a judge
// whose user wording could be rewritten — "the same selection the test-driven arm
// is shown", "decide whether this patch correctly resolves the issue" — without
// any recorded value moving. That is exactly the gap item 19 (fix 4) closed for
// the Attacker, the auditor and the Creator: a prompt edit has to be visible in
// the record, or two different experiments look identical. Each block is now a
// macro that IS the `format!` format string the builder expands AND IS the text
// the hash is taken over, so the two cannot drift the way a parallel copy would.

macro_rules! judge_issue_block {
    () => {
        "ISSUE:\n{}\n"
    };
}

macro_rules! judge_context_block {
    () => {
        "REPOSITORY CONTEXT (the same selection the test-driven arm is shown):\n{}\n"
    };
}

macro_rules! judge_patch_block {
    () => {
        "CANDIDATE PATCH:\n{}\n"
    };
}

/// The suite block exists only under `judge+suite`; the plain judge sends no such
/// message at all, which is the arms' whole difference.
macro_rules! judge_suite_block {
    () => {
        "NATIVE TEST SUITE RESULT ON THE CANDIDATE:\n{}\n"
    };
}

macro_rules! judge_closing_block {
    () => {
        "Decide whether this patch correctly resolves the issue without breaking behaviour. Answer with the JSON object described in your instructions."
    };
}

/// The third message on the retry attempt (see `build_messages`).
macro_rules! judge_reprompt_block {
    () => {
        "REPROMPT (attempt #2): Last response was not valid JSON or the `verdict` was not \"approve\" or \"reject\". Write ONLY a JSON object beginning with '{' and ending with '}'. Keys: verdict (\"approve\" or \"reject\"), confidence (0.0-1.0), reason."
    };
}

/// Every USER template the judge can send, in the order [`judge_user_prompt`]
/// emits them plus the retry tail.
///
/// [VANTA-WHY] C4 — the judge side of the `[prompts]` block. The retry tail is
/// here too: it is a message the model reads and it carries instructions ("write
/// ONLY a JSON object"), so a rewording of it moves verdicts and must move the
/// hash.
pub fn judge_user_prompts() -> [&'static str; 6] {
    [
        judge_issue_block!(),
        judge_context_block!(),
        judge_patch_block!(),
        judge_suite_block!(),
        judge_closing_block!(),
        judge_reprompt_block!(),
    ]
}

/// The judge's USER prompt for a request, in the exact form `judge_candidate`
/// sends.
///
/// [VANTA-WHY] C4, the same shape `creator_user_prompt` has and for the same
/// reason: `judge_prompt_hash` is taken over the templates below, so the builder
/// and the hash must read ONE text. A request built anywhere else could send
/// wording no recorded hash covers, and the judge row's provenance would describe
/// the wrong prompt.
pub fn judge_user_prompt(request: &JudgeRequest) -> String {
    let mut user_parts: Vec<String> = Vec::new();
    user_parts.push(format!(judge_issue_block!(), request.issue_text));
    user_parts.push(format!(judge_context_block!(), request.context));
    user_parts.push(format!(judge_patch_block!(), request.candidate_diff));
    if let Some(suite) = &request.suite_result {
        user_parts.push(format!(judge_suite_block!(), suite));
    }
    user_parts.push(judge_closing_block!().to_string());
    user_parts.join("\n")
}

/// The content hash of the judge's prompt TEMPLATES — system then user.
///
/// [VANTA-WHY] C4 — the value `configs/v1-m3.toml [prompts] judge` pins and the
/// judge row's `judge_prompt_hash` records. It covers the user templates now, so a
/// reworded user block moves it; before this, editing any of the six blocks left
/// the recorded value identical. Same system-then-user order and the same
/// `role_templates` helper every other role uses, so the judge cannot drift into a
/// different notion of "this role's prompt text" than the arm it is compared with.
///
/// Note this is deliberately NOT part of `pipeline::prompt_hashes()`: the judge
/// prompt is sent only on a judge arm, and every row — `session.start` included —
/// carries that string, so folding the judge in would claim every run sent a judge
/// prompt (see `driver::judge_prompt_keys`).
pub fn judge_prompt_hash() -> String {
    crate::core::prompt_content_hash(&crate::pipeline::role_templates(
        &[JUDGE_SYSTEM_PROMPT],
        &judge_user_prompts(),
    ))
}

// ============================================================================
// JudgeVerdict — the typed verdict
// ============================================================================
//
// [VANTA-WHY] An enum rather than a free string, so an unknown `verdict` value
// is a PARSE error and takes the retry path: "the model said something we cannot
// read as a verdict" is a different fact from "the model approved", and folding
// the first into either verdict would record a decision nobody made. Two failed
// attempts end the run as `JudgeFailed`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JudgeVerdict {
    Approve,
    Reject,
}

// ============================================================================
// JudgeRequest / JudgeResponse — the typed I/O of the judge agent
// ============================================================================

/// Everything the judge is shown. Every field is a value the driver already
/// built for the `vanta` arm, so the fairness rule (item 7) is enforced by the
/// shape: there is no field here the test-driven arm did not also have.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeRequest {
    pub issue_text: String,
    /// The rendered `select_review_prompt` selection — the same string the
    /// `vanta` review arm's roles are shown.
    pub context: String,
    /// The candidate patch (the diff under review).
    pub candidate_diff: String,
    /// The native test suite's result on the candidate, under `judge+suite`
    /// only: which tests failed, with the pass/fail summary. `None` under the
    /// plain judge, where no suite is run.
    pub suite_result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JudgeResponse {
    pub verdict: JudgeVerdict,
    pub confidence: f64,
    pub reason: String,
}

// ============================================================================
// JudgeError — a wrong ANSWER versus a broken CALL
// ============================================================================
//
// [VANTA-WHY] M3 item 7 — the two failures need opposite handling, so they are
// two types rather than one string. "The judge answered and its answer cannot be
// read as a verdict" is the judge arm's own outcome (`JudgeFailed`, its own exit
// code and its own bucket, counted and never retried as infrastructure). "The
// call never produced an answer at all" is every other call site's failure and
// ends the run as `InfraFailure`. Folding them together was the alternative: it
// would either hide a genuinely broken candidate behind `infra_failure`, or
// report a dead server as a verdict about the patch.
#[derive(Debug)]
pub enum JudgeError {
    /// The judge did not answer with a readable verdict — malformed JSON after
    /// the one retry, or an answer the server cut off at the cap so it was never
    /// handed to the parser. This is item 7's `JudgeFailed`.
    Malformed(String),
    /// The call itself failed — an HTTP error, the run's deadline, a prompt that
    /// cannot fit the window. No verdict was ever attempted.
    Call(String),
}

impl JudgeError {
    /// The message, whichever kind it is, for the session log.
    pub fn message(&self) -> &str {
        match self {
            JudgeError::Malformed(m) | JudgeError::Call(m) => m,
        }
    }
}

/// Did a `structured_call_with_retry` failure come from an answer the model DID
/// give, rather than from the call never completing?
///
/// [VANTA-WHY] M3 item 7 — the two markers the LLM layer writes are the whole
/// distinction: `PARSE ERR` is an answer that arrived and could not be read, and
/// `TRUNCATED` is an answer the server stopped mid-way (also not a verdict). An
/// HTTP error, the deadline marker and the context-overflow refusal carry
/// neither, so they stay infrastructure. This is a reading of the LLM layer's own
/// wording, kept here rather than in the driver so the classification lives
/// beside the prompt it is about.
fn is_unreadable_answer(message: &str) -> bool {
    message.contains("PARSE ERR") || message.contains("TRUNCATED")
}

// ============================================================================
// judge_candidate() — call the LLM once (plus one retry), get a typed verdict
// ============================================================================
//
// [VANTA-WHY] `max_attempts: 2`, not `cfg.retry_attempts`: the plan fixes the
// judge at "one call per verdict, plus one retry if the JSON is malformed". A
// third attempt would make the judge arm's call count depend on the invocation
// rather than on the arm, which is exactly what the arm registry exists to
// prevent.
pub async fn judge_candidate(
    cfg: &LlmConfig,
    request: &JudgeRequest,
) -> Result<JudgeResponse, JudgeError> {
    info!(
        issue_len = request.issue_text.len(),
        context_len = request.context.len(),
        diff_len = request.candidate_diff.len(),
        with_suite = request.suite_result.is_some(),
        "judge start"
    );

    let client = build_reqwest_client(cfg)
        .map_err(|e| JudgeError::Call(format!("could not build the HTTP client: {e}")))?;

    let system_prompt = judge_system_prompt();
    // [VANTA-WHY] C4 — built from the templates `judge_prompt_hash` covers, never
    // inline here: the recorded provenance has to describe the text that goes out.
    let user_prompt = judge_user_prompt(request);

    debug!(
        system_len = system_prompt.len(),
        user_len = user_prompt.len(),
        "judge prompts built"
    );

    let sys_owned = system_prompt.to_string();
    let user_owned = user_prompt.clone();
    let build_messages = move |attempt: u32| {
        let mut messages = vec![
            json!({"role": "system", "content": sys_owned.clone()}),
            json!({"role": "user", "content": user_owned.clone()}),
        ];
        if attempt >= 2 {
            // The template, not a copy: this is the third message the model reads,
            // so its wording is inside the hash too (C4).
            messages.push(json!({"role": "user", "content": judge_reprompt_block!()}));
        }
        messages
    };

    // [VANTA-WHY] The same `reasoning_effort` every other call site sends, and
    // the `judge` site's own pins from `SAMPLING_SCHEDULE` — the fairness rule
    // item 7 states (same model, effort and sampling as the arm it is compared
    // with).
    let extras = crate::llm::with_sampling(
        json!({
            "response_format": {"type": "json_object"},
            "reasoning_effort": "low",
        }),
        "judge",
    );

    let parse_closure = |raw: &str| -> Result<JudgeResponse, String> {
        let cleaned = extract_json_candidate(raw).map_err(|e| format!("{e}"))?;
        trace!(json = %cleaned.chars().take(300).collect::<String>(), "judge extracted json");
        serde_json::from_str::<JudgeResponse>(cleaned).map_err(|e| {
            format!("JUDGE PARSE FAIL (line {} col {}): {}", e.line(), e.column(), e)
        })
    };

    let typed: JudgeResponse = structured_call_with_retry(
        &client, cfg, 2, "", "judge", build_messages, extras, parse_closure
    )
    .await
    .map_err(|e| {
        let message = e.to_string();
        if is_unreadable_answer(&message) {
            JudgeError::Malformed(message)
        } else {
            JudgeError::Call(message)
        }
    })?;

    info!(
        verdict = ?typed.verdict,
        confidence = typed.confidence,
        "judge done"
    );

    Ok(typed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The verdict enum is the parse gate: an unknown value must not deserialize
    /// into a verdict, or a malformed answer would be recorded as a decision the
    /// model never made.
    #[test]
    fn an_unknown_verdict_is_not_a_judge_response() {
        let ok: JudgeResponse =
            serde_json::from_str(r#"{"verdict":"approve","confidence":0.9,"reason":"r"}"#).unwrap();
        assert_eq!(ok.verdict, JudgeVerdict::Approve);
        let reject: JudgeResponse =
            serde_json::from_str(r#"{"verdict":"reject","confidence":0.1,"reason":"r"}"#).unwrap();
        assert_eq!(reject.verdict, JudgeVerdict::Reject);
        assert!(
            serde_json::from_str::<JudgeResponse>(
                r#"{"verdict":"maybe","confidence":0.5,"reason":"r"}"#
            )
            .is_err(),
            "an unknown verdict must fail to parse so it takes the retry path"
        );
    }

    /// The judge site must be in the schedule, or `sampling_pin_table` panics when
    /// the request body is built.
    #[test]
    fn the_judge_site_is_in_the_sampling_schedule() {
        assert!(
            crate::llm::SAMPLING_SCHEDULE.iter().any(|(site, ..)| *site == "judge"),
            "`judge` must be a site in SAMPLING_SCHEDULE"
        );
    }

    fn request_with(suite_result: Option<&str>) -> JudgeRequest {
        JudgeRequest {
            issue_text: "the issue".to_string(),
            context: "the context".to_string(),
            candidate_diff: "--- a/f.py\n+++ b/f.py\n".to_string(),
            suite_result: suite_result.map(str::to_string),
        }
    }

    /// [VANTA-WHY] C4 — the fix's whole point: every block the judge can send is
    /// inside `judge_prompt_hash`, so a reworded user block moves the value the
    /// judge row records and the frozen config pins. Before C4 the user wording was
    /// inline at the call site and this loop had one member (the system prompt); a
    /// template that is not inside the hash is a prompt whose rewording the record
    /// cannot see, which is the gap item 19 (fix 4) closed for the other roles.
    #[test]
    fn every_judge_prompt_template_is_inside_the_judge_hash() {
        let mut all: Vec<&'static str> = vec![JUDGE_SYSTEM_PROMPT];
        all.extend(judge_user_prompts());
        assert_eq!(
            all.len(),
            7,
            "one system template plus the six user blocks (issue, context, patch, \
             suite, closing, retry tail)"
        );

        for (i, template) in all.iter().enumerate() {
            assert!(
                !template.is_empty(),
                "judge template {i} is empty; an empty template hashes to the \
                 e3b0c44298fc value this record exists to keep out"
            );
            let edited: Vec<String> = all
                .iter()
                .enumerate()
                .map(|(j, t)| if j == i { format!("{t} ") } else { t.to_string() })
                .collect();
            let edited_refs: Vec<&str> = edited.iter().map(String::as_str).collect();
            assert_ne!(
                crate::core::prompt_content_hash(&all),
                crate::core::prompt_content_hash(&edited_refs),
                "judge template {i} is not inside the hash: editing it changed nothing"
            );
        }

        assert_ne!(
            crate::core::prompt_content_hash(&[JUDGE_SYSTEM_PROMPT]),
            judge_prompt_hash(),
            "the recorded judge hash must cover the USER templates, not only the system one"
        );
    }

    /// [VANTA-WHY] C4 — the builder and the hash read one text. This pins the
    /// rendered prompt to the templates themselves (each block's literal text, with
    /// its placeholder filled, appears in the body in order), so a future edit that
    /// rebuilds the prompt somewhere else — the shape the bug had — cannot send
    /// wording the hash does not cover.
    #[test]
    fn the_user_prompt_is_the_blocks_the_judge_hash_covers() {
        let rendered = judge_user_prompt(&request_with(Some("1 failed, 4 passed")));
        let expected = [
            format!(judge_issue_block!(), "the issue"),
            format!(judge_context_block!(), "the context"),
            format!(judge_patch_block!(), "--- a/f.py\n+++ b/f.py\n"),
            format!(judge_suite_block!(), "1 failed, 4 passed"),
            judge_closing_block!().to_string(),
        ]
        .join("\n");
        assert_eq!(rendered, expected);
    }

    /// The plain judge is shown no suite result — that is `judge+suite`'s whole
    /// difference — so the optional block must be exactly that: absent, not empty.
    #[test]
    fn the_plain_judges_prompt_carries_no_suite_block() {
        let judged = judge_user_prompt(&request_with(Some("1 failed, 4 passed")));
        let plain = judge_user_prompt(&request_with(None));
        assert!(judged.contains("NATIVE TEST SUITE RESULT ON THE CANDIDATE:"));
        assert!(
            !plain.contains("NATIVE TEST SUITE RESULT"),
            "the plain judge must be sent no suite block at all; got {plain}"
        );
        assert!(
            plain.starts_with(&format!(judge_issue_block!(), "the issue")),
            "the plain judge still opens with the issue block; got {plain}"
        );
    }
}
