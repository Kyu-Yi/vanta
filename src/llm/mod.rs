// ============================================================================
// LLM MODULE — shared HTTP + structured-output cleaning helpers
// ============================================================================
//
// These helpers were PROMOTED from bootstrap/mod.rs (M0 spike) into a shared
// module so that Creator (agents/creator.rs), Attacker (agents/attacker.rs),
// and the M0 spike (bootstrap/mod.rs) can ALL reuse them without duplication.
//
// WHAT YOU'LL LEARN HERE
//
//   # | Rust concept                  → Where Vanta reuses it later
//  ---+-------------------------------+---------------------------------------
//   1 | reqwest::Client + json!()     → Every LLM call in M1-M5 goes through
//     |   (HTTP POST JSON body)        →   post_chat() — single source of truth
//  ---+-------------------------------+---------------------------------------
//   2 | extract_first_valid_json      → Brace-balanced multi-candidate scanner
//     |   (robust JSON extraction)    →   + serde try-parse on EACH candidate.
//     |                           →   The #1 robustness pattern for small models.
//  ---+-------------------------------+---------------------------------------
//   3 | structured_call_with_retry<T> → N-attempt retry loop with corrective
//     |   (generic async retry)       →   REPROMPT tails. p=0.5 → 3 attm = 87.5%.
// ============================================================================

use crate::config::{redact_secret, LlmConfig};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::OnceCell;
use tracing::{debug, info, trace, warn};

// [VANTA-WHY] §14.26 — the per-model-family request dialects of the hosted
// backend. A module rather than more of this file: it is a TABLE of documented
// facts about somebody else's API, and the one question it answers ("what may
// this model be sent?") has nothing to do with how a stream is reassembled.
pub mod capabilities;
pub mod pricing;

// Re-export the agent response types so bootstrap can use them too.
pub use crate::agents::attacker::{AttackerResponse, AttackerTestCase};
pub use crate::agents::creator::CreatorResponse;
pub use capabilities::{OpenAiFamily, ShapedBody};

// ============================================================================
// FORENSICS: write raw LLM responses + parsed JSON to disk for debugging
// ============================================================================
//
// [VANTA-WHY] User explicitly requested "extensive debugging and logging".
// When VANTA_DEBUG_DIR is set, every structured_call_with_retry attempt writes:
//   {debug_dir}/{type_name}[_{tag}]_round_{R}_attempt_{n}_raw_response.txt
//   {debug_dir}/{type_name}[_{tag}]_round_{R}_attempt_{n}_parsed.json
// where R is the PIPELINE ROUND (the driver's attempt index) and n is the
// structured-call RETRY index (1..=3).
//
// [VANTA-WHY] R used to be missing entirely, and n was the only number in the
// name. Every pipeline round calls the Attacker/BATE/intent-audit helpers again,
// each of which restarts n at 1, so round 2 overwrote round 1 and round 3
// overwrote round 2: the debug folder of a three-round run held round 3 three
// times over and the forensics for the failed rounds were gone. The round is a
// property of the RUN, not of one call, so it lives in a process-global that the
// driver sets once per attempt. The tag (a test name) disambiguates the per-test
// calls within one round.
static FORENSICS_ROUND: AtomicU32 = AtomicU32::new(0);

/// Tell the forensics writer which pipeline round we are in. Called once per
/// attempt by the driver, before any agent runs.
pub fn set_forensics_round(round: u32) {
    FORENSICS_ROUND.store(round, Ordering::Relaxed);
}

// ============================================================================
// RUN DEADLINE — Vanta's own wall clock (plans/bugsinpy.md §14.15)
// ============================================================================
//
// [VANTA-WHY] The harness kills a run at `RUN_TIMEOUT_SECS`. Nothing Vanta did
// ever ended a run on its own clock, so a run-away was stopped by `SIGKILL`
// mid-call: the process died between the terminal decision and the write of
// `run.summary`, and the batch recorded `no run.summary` — a row with no
// verdict, no cost and no prompt provenance. The remedy cannot be a verdict
// Vanta decides *after* being killed, so Vanta keeps its own deadline, set
// short of the harness's, and ends itself as a RECORDED outcome instead.
//
// [VANTA-WHY] Armed in the LLM layer rather than threaded through every call
// site: all five structured call sites converge on `structured_call_with_retry`,
// so one check there is a check before EVERY call — including the ones inside
// BATE and the Diagnostician, whose loops have no channel to the driver. A
// refused call costs one string, not one HTTP request, so a phase that keeps
// iterating after the deadline runs out of tests and exits at once instead of
// spending another ten minutes per test.

/// Marker `structured_call_with_retry` puts on a call it refused to make
/// because the run's deadline had passed (§14.15).
///
/// [VANTA-WHY] The driver has to tell this apart from a transport error: a
/// transport error is `InfraFailure` (exit 2, "the machine broke"), while a
/// deadline is `budget_exhausted` (exit 3, "the run ran out of clock"). The two
/// are opposite claims about the run, so the token is spelled out here rather
/// than matched by a sentence the driver would have to re-spell.
pub const DEADLINE_EXCEEDED_MARKER: &str = "deadline_exceeded";

/// The instant the run must stop making calls, or `None` when unarmed.
static DEADLINE_AT: Mutex<Option<Instant>> = Mutex::new(None);

/// Arm the run's deadline `secs` from now; `None` leaves it unarmed, which is
/// the default and the behaviour of every run before §14.15.
pub fn set_deadline_secs(secs: Option<u64>) {
    *DEADLINE_AT.lock().unwrap() = secs.map(|s| Instant::now() + Duration::from_secs(s));
}

/// Whether the run's deadline has passed. Always `false` when none is armed.
pub fn deadline_past() -> bool {
    DEADLINE_AT
        .lock()
        .unwrap()
        .map_or(false, |at| Instant::now() >= at)
}

// ============================================================================
// TOKEN ACCOUNTING — what each chat call really cost
// ============================================================================
//
// [VANTA-WHY] Before this, NO completion token count existed anywhere in the
// codebase and the only prompt count was `conservative_token_estimate`, which is
// computed from CHARACTERS and deliberately over-counts (see
// `TOKEN_ESTIMATE_BYTES_PER_TOKEN`: it ran ~1.5x high on real runs). A cost
// figure taken from the estimate is therefore not a measurement, and the paper's
// cost numbers would be inflated by that factor with no way to detect it. The
// estimate stays — the pre-flight guard needs a number BEFORE the request is
// sent — but it is now labelled as an estimate wherever it is recorded, and the
// server's own counts are what the cost tables are built from.
//
// [VANTA-WHY] A process-global accumulator rather than a return value. The
// counts are known at the bottom of `post_chat`, but the durable record is
// written by the driver, and between the two sit `structured_call_with_retry`
// and every agent without a channel for usage. Threading a sink through all of
// them would change agent signatures that decide verdicts; a global drained by
// the driver leaves every verdict path untouched. The cost is that the driver
// must `reset_call_usages` when a run starts (this process may run several
// sessions, and a test process certainly does).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallUsage {
    /// Monotonic index of the call within the run, so rows can be ordered
    /// without trusting wall-clock timestamps.
    pub seq: u64,
    /// The call site, named exactly as in `SAMPLING_SCHEDULE` for the pipeline;
    /// the M0 spike names its own (it is not part of the schedule).
    pub site: String,
    /// The pipeline round this call belongs to, 1-based, or 0 when no round was
    /// ever set (the M0 spike). The driver sets it once per attempt via
    /// `set_forensics_round(attempt + 1)`, so it is the same index the forensics
    /// filenames carry.
    ///
    /// [VANTA-WHY] §14.12 — "the same question was asked four times in ONE
    /// round" is only readable if the round travels with the call. Without it the
    /// reader has to reconstruct the round from the driver's log, which is
    /// exactly the reconstruction the runaway diagnosis could not do: the calls
    /// were on disk but their round was not.
    pub round: u32,
    /// Which retry of `structured_call_with_retry` produced this row, 1-based.
    ///
    /// [VANTA-WHY] §14.12 — attempt 1 is the pinned request, every later attempt
    /// is a retry. Two rows with the same `site` and `round` and different
    /// `attempt` ARE the retry loop; that is the shape the runaway report needs
    /// to see, and it cannot be inferred from `seq` alone because other sites
    /// interleave.
    pub attempt: u32,
    /// How many times THIS call was re-sent because the PROVIDER refused it while
    /// busy, rather than because the model answered badly (§14.30).
    ///
    /// [VANTA-WHY] A separate field from `attempt`, and that is the point: a 429
    /// is the service saying "not now" about a request the model never saw, so
    /// counting it into `attempt` would make a throttled call look like a model
    /// that needed two tries — the one reading this row exists to make
    /// impossible. The retries happen inside ONE attempt, against their own
    /// budget, and this says how many.
    pub provider_retries: u32,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    /// `usage.prompt_tokens_details.cached_tokens`, the subset of
    /// `prompt_tokens` the provider served from its prompt cache (§14.29).
    ///
    /// [VANTA-WHY] Per CALL, not per run, because a cache hit is a property of
    /// the prefix THIS body sent: the Creator, the Attacker's revise and the
    /// batch audit send different stable prefixes, so a run-level hit rate would
    /// average a warm site with a cold one and answer nothing. The hit RATE is
    /// derived from these rows, never stored.
    pub cached_tokens: Option<u64>,
    /// `usage.completion_tokens_details.reasoning_tokens`, the subset of
    /// `completion_tokens` spent thinking (§14.29).
    pub reasoning_tokens: Option<u64>,
    /// What this call cost in USD, `None` when no price applies — the local
    /// backend sells nothing, a server that published no `usage` block has no
    /// measured count to price, and a family this table cannot cite has no rate
    /// (see `llm::pricing`).
    ///
    /// [VANTA-WHY] `Option<f64>`, never `0.0` for "unknown". A cost column that
    /// renders "free" and "unpriced" as the same number cannot be summed, and the
    /// batch table would silently under-report the one figure money was spent on.
    pub cost_usd: Option<f64>,
    /// How `cost_usd` was arrived at (§14.29): the price row's
    /// `family/tier/date`, or one of `llm::pricing`'s refusal spellings
    /// (`not_billed`, `no_usage`, `unpriced_family`).
    ///
    /// [VANTA-WHY] A cost without a basis is a number nobody can check, and the
    /// price table is versioned precisely so that "which prices produced this?"
    /// has an answer. Carried per call because a run's rows can differ: one call
    /// can be priced while another fell back to the estimate.
    pub cost_basis: String,
    /// Where the numbers came from: `server-usage` (the `usage` block that
    /// `stream_options.include_usage` asks for), `server-timings` (llama-server's
    /// own final-frame `timings`), or `estimate` (the server published neither,
    /// so only the local pre-flight estimate exists — prompt only, no completion
    /// count). A cost table must be able to exclude `estimate` rows instead of
    /// silently averaging a measurement with a guess.
    pub source: String,
    /// The `seed` THIS call actually sent, `None` when the body carried no
    /// `seed` key at all.
    ///
    /// [VANTA-WHY] §14.5 — a truncated answer above temperature 0.0 is retried
    /// with the seed advanced (see `seed_for_attempt`), so "which seed did
    /// attempt 2 send?" is not answerable from `VANTA_LLM_SEED` alone. Carrying
    /// it on the per-call row is what makes the retry reproducible AFTER the
    /// fact: the sequence is a function of the pinned seed and the attempt
    /// index, and both are in the record.
    pub seed: Option<u64>,
    /// How the seed reached (or did not reach) this call's body (§14.23).
    ///
    /// [VANTA-WHY] §14.23 — the same reason §14.21's knobs carry a mechanism: a
    /// recorded value without one cannot be compared across backends. `llamacpp:seed`
    /// — the body carried a `seed` key. `unset` — no seed was asked for, so the body
    /// carried none (§4.4, D4's byte-identity rule). `unsupported` is reserved for a
    /// backend whose API has no seed field: the local transport never emits it, and it
    /// exists so a hosted run records a refusal rather than a silently missing key.
    pub seed_mechanism: String,
    /// The server's own `finish_reason` for this call, verbatim (`stop`,
    /// `length`, …), `None` when it published none.
    ///
    /// [VANTA-WHY] §14.12 — `length` is the FACT; `truncated` is that fact
    /// interpreted. Kept verbatim beside it so a call that stopped for a reason
    /// nobody anticipated is visible, instead of being silently binned as "not
    /// truncated" — which is all a boolean on its own can do.
    pub finish_reason: Option<String>,
    /// Wall time for this call, request sent to the stream closing, in
    /// milliseconds.
    ///
    /// [VANTA-WHY] §14.12 — the runaway is a story about TIME (5-10 minutes per
    /// long call) as much as about tokens, and the only wall time that existed
    /// was a `debug!` line nothing kept. Measured around the same `Instant` that
    /// line used, so the record and the log can never disagree.
    pub wall_ms: u64,
    /// Whether the server stopped this call at `max_tokens` (`finish_reason:
    /// "length"`).
    ///
    /// [VANTA-WHY] §14.5 — a model that "runs away" spends the whole output cap
    /// on reasoning and returns nothing parseable. Counting those per role is
    /// how the paper reports how often it happens, and a counter derived from
    /// the calls the run already records cannot disagree with the run it
    /// describes.
    pub truncated: bool,
    /// What this call's request ASKED FOR and what the body actually carried
    /// (§14.26).
    ///
    /// [VANTA-WHY] Per call, not per run, because the answer is a property of
    /// the body — the same model is sent different extras by different call
    /// sites (the batch audit scales its own `max_tokens`, the coding preset
    /// pins keys the pinned one does not). A run-level list would have to be the
    /// union of every call's, and a union is exactly what cannot answer "did
    /// THIS call's temperature reach the model?". `run.summary` carries the
    /// model-level rule beside it, so the two levels are both readable.
    pub request_shape: ShapedBody,
    /// The service tier this call was served on, as `run.summary` and the row
    /// spell it: `standard`/`flex` on the hosted backend, `unset` on the local
    /// one (§14.31).
    ///
    /// [VANTA-WHY] The name, not the enum, and `unset` rather than `standard`
    /// for a local run — the same call `cost_basis` makes with `not_billed`. A
    /// llama.cpp server has no tier and sells nothing, so recording `standard`
    /// would put a local row in the batch table's full-price bucket. Carried per
    /// call even though it is one value per run, because the row is where a
    /// reader looks up what priced it and `cost_basis` is read from the same
    /// place.
    pub service_tier: String,
}

static CALL_USAGES: Mutex<Vec<CallUsage>> = Mutex::new(Vec::new());
static USAGE_SEQ: AtomicU64 = AtomicU64::new(0);

/// How many calls have given up because BOTH the original request and the short
/// re-ask that follows a cut-off were cut off themselves (§14.5, item 2).
///
/// [VANTA-WHY] The per-call rows cannot answer this: `truncated` is a fact about
/// one HTTP call, while "cut off twice" is a fact about a `structured_call_with_retry`.
/// A run-level counter is the smallest thing that makes the number reportable,
/// and it is drained by the driver at the end of a run like the usage ledger.
static TRUNCATED_TWICE: AtomicU64 = AtomicU64::new(0);

/// How many times a call was re-sent because the PROVIDER was busy — a rate limit,
/// or a `flex` tier with no capacity — rather than because the model's answer was
/// wrong (§14.30, item 6).
///
/// [VANTA-WHY] Kept beside `TRUNCATED_TWICE` and for the same reason: the per-call
/// rows can carry this for calls that ANSWERED (`provider_retries` on
/// `CallUsage`), but a call whose provider never let it through writes no row at
/// all, and its retries still happened. A run-level counter is the only place
/// that number can live.
static PROVIDER_RETRIES: AtomicU64 = AtomicU64::new(0);
/// The same count, split by WHY the provider refused (§14.30).
///
/// [VANTA-WHY] The two are NOT the same operational fact: a rate limit means slow
/// down, and `Resource Unavailable` on the flex tier means the cheap capacity
/// this run asked for is gone. They are priced differently (`pricing::ServiceTier`)
/// and the fix differs, so a report that folded them into one number could not
/// tell which happened.
static PROVIDER_RETRIES_RATE_LIMIT: AtomicU64 = AtomicU64::new(0);
static PROVIDER_RETRIES_UNAVAILABLE: AtomicU64 = AtomicU64::new(0);

fn call_usages() -> MutexGuard<'static, Vec<CallUsage>> {
    // A panic in another thread must not make token accounting fatal: the lock is
    // pushed to and drained, never held across an await, so the data behind a
    // poisoned lock is still structurally sound.
    CALL_USAGES.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn record_call_usage(usage: CallUsage) {
    call_usages().push(usage);
}

/// Every chat call recorded so far, oldest first, and empty the accumulator.
///
/// Draining (rather than reading) is what stops a second run in the same process
/// from re-writing the first run's calls into its own record.
pub fn drain_call_usages() -> Vec<CallUsage> {
    std::mem::take(&mut *call_usages())
}

/// Drop any usage recorded by an earlier run in this process. Called by the
/// driver at the start of a session.
pub fn reset_call_usages() {
    call_usages().clear();
    USAGE_SEQ.store(0, Ordering::Relaxed);
    TRUNCATED_TWICE.store(0, Ordering::Relaxed);
    PROVIDER_RETRIES.store(0, Ordering::Relaxed);
    PROVIDER_RETRIES_RATE_LIMIT.store(0, Ordering::Relaxed);
    PROVIDER_RETRIES_UNAVAILABLE.store(0, Ordering::Relaxed);
}

/// How many times this run's calls were re-sent because the provider was busy,
/// as `(total, rate_limit, resource_unavailable)` (§14.30).
///
/// [VANTA-WHY] `total` is counted here rather than summed from the per-call rows,
/// and the two are deliberately not interchangeable: a call that never got an
/// answer writes no row, so the rows would under-report exactly the runs that were
/// throttled hardest. Read by the driver when it writes `run.summary`.
pub fn provider_retry_counts() -> (u64, u64, u64) {
    (
        PROVIDER_RETRIES.load(Ordering::Relaxed),
        PROVIDER_RETRIES_RATE_LIMIT.load(Ordering::Relaxed),
        PROVIDER_RETRIES_UNAVAILABLE.load(Ordering::Relaxed),
    )
}

fn note_provider_retry(class: ProviderRetryClass) {
    PROVIDER_RETRIES.fetch_add(1, Ordering::Relaxed);
    match class {
        ProviderRetryClass::RateLimit => &PROVIDER_RETRIES_RATE_LIMIT,
        ProviderRetryClass::ResourceUnavailable => &PROVIDER_RETRIES_UNAVAILABLE,
    }
    .fetch_add(1, Ordering::Relaxed);
}

/// How many calls in this run were cut off at `max_tokens` twice, including the
/// brief re-ask (§14.5, item 2). Read by the driver when it writes `run.summary`.
pub fn truncated_twice_calls() -> u64 {
    TRUNCATED_TWICE.load(Ordering::Relaxed)
}

fn note_truncated_twice() {
    TRUNCATED_TWICE.fetch_add(1, Ordering::Relaxed);
}

fn debug_dir() -> Option<PathBuf> {
    std::env::var("VANTA_DEBUG_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
}

fn write_forensics(type_name: &str, tag: &str, attempt: u32, suffix: &str, content: &str) {
    let Some(dir) = debug_dir() else { return };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        warn!(err = %e, "forensics: failed to create debug dir");
        return;
    }
    write_forensics_to(
        &dir,
        type_name,
        tag,
        FORENSICS_ROUND.load(Ordering::Relaxed),
        attempt,
        suffix,
        content,
    );
}

/// The one place a forensics filename is built. `round` is the pipeline round,
/// `attempt` the structured-call retry index.
///
/// [VANTA-WHY] `type_name` is sanitized too, because it is NOT always a bare
/// identifier: `structured_call_with_retry` derives it as the LAST `::`-segment
/// of `std::any::type_name::<T>()`, and for a generic answer that segment keeps
/// its angle bracket — `Vec<IntentVerdict>` yields `"IntentVerdict>"`, which the
/// batched intent audit writes as
/// `IntentVerdict>_intent_audit_batch_round_1_attempt_1_raw_response.txt`. A `>`
/// in a filename is shell-hostile, so the same rule that cleans a tag cleans it.
fn write_forensics_to(
    dir: &Path,
    type_name: &str,
    tag: &str,
    round: u32,
    attempt: u32,
    suffix: &str,
    content: &str,
) {
    let label = if tag.is_empty() {
        String::new()
    } else {
        format!("_{}", sanitize_tag(tag))
    };
    let filename = format!(
        "{}{label}_round_{round}_attempt_{attempt}_{suffix}",
        sanitize_tag(type_name)
    );
    let path = dir.join(filename);
    // [VANTA-WHY] The one write site every forensics file goes through, and so
    // the one place a credential inside a model's raw output can be stopped.
    // Redaction here rather than at each caller: a caller added later inherits
    // the rule instead of having to remember it. It is a no-op for content that
    // carries no `sk-` token, which is every local run.
    let content = redact_secret(content);
    match std::fs::write(&path, content) {
        Ok(_) => debug!(path = %path.display(), "forensics written"),
        Err(e) => warn!(err = %e, path = %path.display(), "forensics write failed"),
    }
}

/// A test name can carry `/`, `[`, `]`, spaces — none of which belong in a
/// filename. The same rule cleans the type name in a forensics filename, where a
/// `Vec<T>` answer contributes a trailing `>`.
fn sanitize_tag(tag: &str) -> String {
    tag.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect()
}

// ============================================================================
// HELPER: build_reqwest_client(&cfg) → reqwest::Client
// ============================================================================
pub fn build_reqwest_client(cfg: &LlmConfig) -> Result<reqwest::Client, reqwest::Error> {
    let mut headers = reqwest::header::HeaderMap::new();

    if let Some(key) = &cfg.api_key {
        use reqwest::header::{HeaderValue, AUTHORIZATION};
        let mut val = HeaderValue::from_str(&format!("Bearer {key}"))
            .expect("API key contains non-ASCII bytes");
        val.set_sensitive(true);
        headers.insert(AUTHORIZATION, val);
    }

    // [VANTA-WHY] There is deliberately NO `.timeout(...)` here. A single
    // total deadline measures the wrong thing: a 27B model legitimately
    // generating for 20 minutes is progress, and killing it at 600s left
    // llama-server generating an orphan while the retry opened a second slot
    // against the same KV cache ("Context size has been exceeded"), after which
    // the run crashed with no final decision. Liveness is enforced per call
    // instead — an IDLE timeout between stream chunks plus an overall backstop —
    // see `assemble_sse_stream`.
    reqwest::Client::builder().default_headers(headers).build()
}

// ============================================================================
// HELPER: strip_markdown_fences(&str) -> &str
// ============================================================================
pub fn strip_markdown_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s.strip_prefix("```json").unwrap_or(s).trim_start();
    let s = s.strip_prefix("```").unwrap_or(s).trim_start();
    let s = s.strip_suffix("```").unwrap_or(s).trim_end();
    s
}

const CHAT_OVERHEAD_TOKENS: u32 = 256;

fn render_messages_for_token_estimate(messages: &[Value]) -> String {
    messages
        .iter()
        .map(|message| {
            let role = message["role"].as_str().unwrap_or("unknown");
            let content = message["content"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| message["content"].to_string());
            format!("<{role}>\n{content}\n")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Bytes-per-token divisor for the pre-flight prompt estimate.
///
/// [VANTA-WHY] The estimate was `bytes / 3` and ran ~1.5× high — the heretic
/// run logged 84500 estimated against 55300 real prompt tokens (server usage),
/// i.e. ~4.6 bytes/token for this pipeline's mixed Rust/Python/English prompts.
/// A 1.5× over-count is not free: it is what refused prompts that would have
/// fit. `/ 4` keeps ~15% headroom over the measured ratio, so the guard stays
/// fail-closed (it must never under-count) while no longer tripping on a prompt
/// that fits. Every call now also logs the server's own `usage.prompt_tokens`
/// next to this estimate, so the ratio is re-measurable instead of assumed.
const TOKEN_ESTIMATE_BYTES_PER_TOKEN: u32 = 4;

fn conservative_token_estimate(text: &str) -> u32 {
    let bytes = text.as_bytes().len() as u32;
    if bytes == 0 {
        0
    } else {
        bytes / TOKEN_ESTIMATE_BYTES_PER_TOKEN
            + u32::from(bytes % TOKEN_ESTIMATE_BYTES_PER_TOKEN != 0)
    }
}

fn context_budget_exceeded(
    prompt_tokens: u32,
    requested_output_tokens: u32,
    context_window_tokens: u32,
) -> bool {
    prompt_tokens
        .saturating_add(requested_output_tokens)
        .saturating_add(CHAT_OVERHEAD_TOKENS)
        > context_window_tokens
}

fn tokenize_url_candidates(cfg: &LlmConfig) -> Vec<String> {
    let base = cfg.base_url.trim_end_matches('/');
    let mut urls = Vec::new();
    if let Some(root) = base.strip_suffix("/v1") {
        urls.push(format!("{root}/tokenize"));
    }
    urls.push(format!("{base}/tokenize"));
    urls.dedup();
    urls
}

async fn tokenize_via_server(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    text: &str,
) -> Option<u32> {
    for url in tokenize_url_candidates(cfg) {
        let Ok(resp) = client
            .post(&url)
            .json(&json!({ "content": text }))
            .send()
            .await
        else {
            continue;
        };
        if !resp.status().is_success() {
            continue;
        }
        let Ok(body) = resp.json::<Value>().await else {
            continue;
        };
        if let Some(tokens) = body["tokens"].as_array() {
            return Some(tokens.len() as u32);
        }
        if let Some(count) = body["count"].as_u64() {
            return Some(count as u32);
        }
        if let Some(count) = body["token_count"].as_u64() {
            return Some(count as u32);
        }
    }
    None
}

// ============================================================================
// HELPER: context_window_tokens(&client, &cfg) -> u32
// ============================================================================
//
// [VANTA-WHY] The context window used to be a compile-time 65536 unless
// VANTA_LLM_CONTEXT_TOKENS said otherwise. A wrong window is a wrong guard: too
// small and the pre-flight check refuses prompts that fit (the 1.5×-high
// estimate made that a real cost), too large and the check passes a prompt the
// server then rejects mid-run. llama-server knows its own value and publishes it
// at `/props` (`default_generation_settings.n_ctx`), so ask it — once per
// process, because /props is a genuine HTTP call and the answer cannot change
// inside one run.
static CONTEXT_WINDOW: OnceCell<u32> = OnceCell::const_new();

fn props_url_candidates(cfg: &LlmConfig) -> Vec<String> {
    let base = cfg.base_url.trim_end_matches('/');
    let mut urls = Vec::new();
    if let Some(root) = base.strip_suffix("/v1") {
        urls.push(format!("{root}/props"));
    }
    urls.push(format!("{base}/props"));
    urls.dedup();
    urls
}

async fn fetch_props(client: &reqwest::Client, cfg: &LlmConfig) -> Option<Value> {
    for url in props_url_candidates(cfg) {
        let Ok(resp) = client.get(&url).send().await else {
            continue;
        };
        if !resp.status().is_success() {
            continue;
        }
        let Ok(body) = resp.json::<Value>().await else {
            continue;
        };
        return Some(body);
    }
    None
}

async fn probe_server_context_window(client: &reqwest::Client, cfg: &LlmConfig) -> Option<u32> {
    let body = fetch_props(client, cfg).await?;
    if let Some(n_ctx) = body["default_generation_settings"]["n_ctx"].as_u64() {
        return Some(n_ctx as u32);
    }
    if let Some(n_ctx) = body["n_ctx"].as_u64() {
        return Some(n_ctx as u32);
    }
    None
}

/// The sampling the harness PINS, per call site (plans/bugsinpy.md §4.3).
///
/// [VANTA-WHY] §4.3's first draft said the harness sets no `temperature` /
/// `top_p` / `max_tokens` at all, and told the provenance block to record the
/// server's `/props` defaults instead. That was wrong about this codebase: every
/// M3 agent passes its own `temperature` / `max_tokens` through `extra_fields`,
/// which `post_chat` merges AFTER the body defaults — so the agent's value is
/// what the request carries, and `post_chat`'s own 0.2 / 16384 are dead for
/// every M3 call. Writing `default_generation_settings.temperature` into
/// `run.summary` would therefore have recorded a number the model never sampled
/// with.
///
/// [VANTA-WHY] There is also no single run-level value to record: the call sites
/// disagree on purpose (the intent audit is deterministic at 0.0, the Creator is
/// near-deterministic at 0.1, generation is 0.2, revision is 0.3). So the record
/// is per call site, and this table is it.
///
/// [VANTA-WHY] This table is the `pinned` preset — part of v0.2's behaviour, the
/// sampling the collected baseline ran under — and it is the DEFAULT:
/// `VANTA_SAMPLING` unset (or `pinned`) gives exactly these values at every call
/// site, byte for byte. `VANTA_SAMPLING=coding` (§14.11) overrides the sampler
/// pins of EVERY site, the auditor's row included: the audit judges the
/// Attacker's output, so it samples the way that output was sampled. The values
/// reach the request bodies through [`with_sampling`], and
/// `tests/m3_session_log.rs::the_recorded_sampling_schedule_matches_the_request_bodies`
/// compares the EFFECTIVE schedule against the bodies actually sent, which is the
/// guard against a call site and this table drifting apart.
pub const SAMPLING_SCHEDULE: &[(&str, f64, u32)] = &[
    ("creator", 0.1, 32768),
    ("attacker_generate", 0.2, 32768),
    ("attacker_revise", 0.3, 32768),
    ("attacker_simplify", 0.3, 32768),
    ("intent_audit", 0.0, 8192),
    // [VANTA-WHY] M3 item 7 — the judge's one call. Its temperature matches the
    // `attacker_generate` site, which is the arm the judge is compared with
    // (plans/M3-plan.md item 7's fairness rule: "same model, reasoning effort,
    // context selection and budget, and seed policy as the B3 arm"), so under
    // both presets the judge samples exactly as that arm's Attacker does. The
    // cap is the schedule's own 32,768: a verdict JSON is small, but the answer
    // is reasoned about before it is written, so the cap must not be the thing
    // that truncates the reasoning. The values are recorded only on judge rows —
    // `driver::sampling_schedule_keys` emits this site's keys only when the
    // run's arm is a judge arm — so every pre-existing `run.summary` is
    // byte-identical.
    ("judge", 0.2, 32768),
];

/// The `coding` preset's sampler pins, for EVERY site including the auditor (§14.11).
///
/// [VANTA-WHY] One temperature / top_p / top_k / min_p / presence_penalty set,
/// not a second schedule: the preset changes HOW a site samples, not how many
/// tokens it may emit (`max_tokens` stays the `pinned` row's, so the auditor keeps
/// its 8192-token cap). The auditor is covered like every other site: under
/// `coding` it samples at 0.6 alongside the Attacker whose tests it judges, which
/// is the condition the preset exists for — a pinned-0.0 auditor grades a noisy
/// Attacker's output as if that output were the only reading of the issue. A
/// truncated audit at 0.6 is NOT deterministic, so it takes the retry path with
/// the seed advanced, never the temperature-0.0 fail-fast
/// (`structured_call_with_retry` reads the temperature off the request itself).
const CODING_TEMPERATURE: f64 = 0.6;
const CODING_TOP_P: f64 = 0.95;
const CODING_TOP_K: u32 = 20;
const CODING_MIN_P: f64 = 0.0;
const CODING_PRESENCE_PENALTY: f64 = 0.0;

/// The sampling preset a run is under (`VANTA_SAMPLING`, §14.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingPreset {
    /// Today's per-site schedule — the default, byte-identical to v0.2.
    Pinned,
    /// Higher-temperature sampling for the generative roles.
    Coding,
}

impl SamplingPreset {
    /// Parse `VANTA_SAMPLING`. `None` (unset) is the default, `Pinned`.
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw.map(str::trim) {
            None | Some("") | Some("pinned") => Ok(Self::Pinned),
            Some("coding") => Ok(Self::Coding),
            Some(other) => Err(format!(
                "CONFIG_ERROR: VANTA_SAMPLING='{other}' is not a preset (expected pinned|coding)."
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Pinned => "pinned",
            Self::Coding => "coding",
        }
    }
}

/// The preset this process runs under, read from `VANTA_SAMPLING` once.
///
/// [VANTA-WHY] Read once, like the other run-identity switches: a call site late
/// in a run must not be able to disagree with the one that built the earlier
/// request bodies. An unparsable value is reported and falls back to `Pinned`
/// rather than failing a generation mid-flight — the effective preset is written
/// into `run.summary`, so a run that fell back says so.
pub fn sampling_preset() -> SamplingPreset {
    static PRESET: OnceLock<SamplingPreset> = OnceLock::new();
    *PRESET.get_or_init(|| {
        let raw = std::env::var("VANTA_SAMPLING").ok();
        match SamplingPreset::parse(raw.as_deref()) {
            Ok(preset) => preset,
            Err(e) => {
                warn!("{e} — falling back to `pinned`");
                SamplingPreset::Pinned
            }
        }
    })
}

/// One pinned sampler key's value, typed so it renders the same way a request
/// body carries it (`0.0` stays `0.0`).
#[derive(Debug, Clone, Copy)]
enum PinValue {
    Float(f64),
    Int(u32),
}

// ============================================================================
// PRESENCE PENALTY OVERRIDE — `VANTA_PRESENCE_PENALTY` (§14.16)
// ============================================================================
//
// [VANTA-WHY] The overnight batch's run-aways were not wrong answers; they were
// the SAME answer, rewritten 45-87 times until `max_tokens` cut the stream
// (§14.13's classification (i)). `presence_penalty` is the sampler parameter
// that penalises tokens already present in the output, so it is the one knob
// that addresses that failure mode directly — and whether it helps is an
// empirical question, not a design one. This switch exists so the question can
// be ANSWERED rather than argued: one batch arm with it at 0.0 and one at 1.0,
// everything else identical.
//
// [VANTA-WHY] It is an OVERRIDE of the pin table, not a third preset. A preset
// changes a whole sampler (temperature and the nucleus together); this changes
// exactly one parameter and leaves every other pin — including the preset's own
// — where it was. So it is applied AFTER the preset and replaces that one key.
//
// [VANTA-WHY] ZERO INSERTS NOTHING, which is what keeps the `pinned` preset
// byte-identical to v0.2. `Some(0.0)` and "unset" both leave the request body
// with no `presence_penalty` key at all, so `the_coding_preset_covers_every_site_including_the_auditor`'s
// "a pinned body carries no `presence_penalty`" still holds, and a `coding` run
// keeps the 0.0 it declares. That is also why the value is recorded at the RUN
// level as well as per site: zero and unset produce identical bodies, so the
// only way a batch table can tell those two arms apart is if `run.summary`
// carries what was ASKED FOR, not just what was sent.

/// The requested presence penalty, or `None` when the switch is unset. The RAW
/// request: `Some(0.0)` is "asked for zero", which is not the same claim as
/// "asked for nothing" even though the two produce the same bodies.
static PRESENCE_PENALTY: Mutex<Option<f64>> = Mutex::new(None);

/// Pin `presence_penalty` on every call site, or `None` to leave the pins alone.
pub fn set_presence_penalty(value: Option<f64>) {
    *PRESENCE_PENALTY.lock().unwrap() = value;
}

/// The presence penalty to pin, or `None` when there is nothing to pin. A zero
/// is "no pin", because inserting `presence_penalty: 0.0` would change every
/// body byte for a run that asked for the server's own behaviour.
fn presence_penalty_override() -> Option<f64> {
    PRESENCE_PENALTY
        .lock()
        .unwrap()
        .filter(|penalty| *penalty != 0.0)
}

// ============================================================================
// PER-SITE OUTPUT CAP OVERRIDE — `VANTA_MAX_TOKENS_<SITE>` (§14.20)
// ============================================================================
//
// [VANTA-WHY] §14.14 caps how MANY revise/simplify calls one attempt may make;
// this caps how LONG one may run. They answer the same question — "is the
// Attacker stuck?" — from two sides, and a single revised test that spends
// 32,768 tokens re-checking its own formatting is a run-away the call budget
// cannot see, because it is one call.
//
// [VANTA-WHY] It is an OVERRIDE of the pin table, not a third preset, and it is
// applied after the preset and after §14.16's presence penalty, for the same
// reason: it changes exactly one key and leaves every other pin where it was. It
// REPLACES the existing `max_tokens` slot rather than appending, because two
// `max_tokens` keys in one JSON object is the second one silently winning and a
// record that named both would describe a cap nobody sent.
//
// [VANTA-WHY] The value is recorded twice and the two records say different
// things. The per-site `max_tokens_<site>` keys in `run.summary` come from the
// same table the bodies are built from, so they carry the EFFECTIVE cap; the
// run-level line below carries what was ASKED FOR, which is what identifies the
// arm — a reader comparing two batch directories needs to see the switch, not
// have to infer it from a number that happens to differ from the schedule.

/// The requested per-site output caps, keyed by call site. Empty = nothing asked.
///
/// [VANTA-WHY] A `Vec` of the table's own `&'static str` site names rather than a
/// `HashMap`: `HashMap::new` is not a constant function, so it cannot back a
/// `static`, and the site list is five entries long — a linear scan here is not a
/// cost worth a `LazyLock` for.
static MAX_TOKENS_OVERRIDES: Mutex<Vec<(&'static str, u32)>> = Mutex::new(Vec::new());

/// The environment variable that carries `site`'s output cap (`VANTA_MAX_TOKENS_<SITE>`).
///
/// [VANTA-WHY] Derived from the site name rather than spelled at each site, so
/// the switch and the table that applies it cannot drift: `attacker_revise`
/// is `VANTA_MAX_TOKENS_ATTACKER_REVISE` and nothing else can be.
pub fn max_tokens_env_var(site: &str) -> String {
    format!("VANTA_MAX_TOKENS_{}", site.to_uppercase())
}

/// Pin `max_tokens` per call site. An empty list leaves the pins alone.
pub fn set_max_tokens_overrides(overrides: Vec<(&'static str, u32)>) {
    *MAX_TOKENS_OVERRIDES.lock().unwrap() = overrides;
}

/// The requested cap for `site`, or `None` when the schedule's value stands.
///
/// `None` is every run before §14.20 and every run without the switch, so the
/// bodies those runs send are byte-identical to what they sent before it.
fn max_tokens_override(site: &str) -> Option<u32> {
    MAX_TOKENS_OVERRIDES
        .lock()
        .unwrap()
        .iter()
        .find(|(name, _)| *name == site)
        .map(|(_, cap)| *cap)
}

/// The overrides that were asked for, `site=cap` pairs in `SAMPLING_SCHEDULE` order.
///
/// [VANTA-WHY] Rendered in the schedule's own order rather than the order the
/// environment happened to arrive in, so two arms of the same ablation render
/// identically. `unset` is the absence of the switch, which is every run before
/// §14.20 — never an empty string, which would be a claim that something was
/// pinned.
pub fn max_tokens_overrides_text() -> String {
    let overrides = MAX_TOKENS_OVERRIDES.lock().unwrap();
    let pairs: Vec<String> = SAMPLING_SCHEDULE
        .iter()
        .filter_map(|(site, ..)| {
            overrides
                .iter()
                .find(|(name, _)| name == site)
                .map(|(_, cap)| format!("{site}={cap}"))
        })
        .collect();
    if pairs.is_empty() {
        "unset".to_string()
    } else {
        pairs.join(",")
    }
}

/// The sampler keys `site` pins under `preset`, in a fixed order.
///
/// [VANTA-WHY] One table feeds both the request body ([`with_sampling`]) and the
/// `run.summary` record (`driver::sampling_schedule_keys`) through the two
/// renderers below, so the number a session records cannot drift from the number
/// its bodies carried.
fn sampling_pin_table(preset: SamplingPreset, site: &str) -> Vec<(&'static str, PinValue)> {
    let (temperature, max_tokens) = SAMPLING_SCHEDULE
        .iter()
        .find(|(name, ..)| *name == site)
        .map(|(_, temperature, max_tokens)| (*temperature, *max_tokens))
        .unwrap_or_else(|| panic!("`{site}` is not a site in SAMPLING_SCHEDULE"));

    let mut pins = vec![
        ("temperature", PinValue::Float(temperature)),
        ("max_tokens", PinValue::Int(max_tokens)),
    ];
    if preset == SamplingPreset::Coding {
        pins[0] = ("temperature", PinValue::Float(CODING_TEMPERATURE));
        pins.push(("top_p", PinValue::Float(CODING_TOP_P)));
        pins.push(("top_k", PinValue::Int(CODING_TOP_K)));
        pins.push(("min_p", PinValue::Float(CODING_MIN_P)));
        pins.push(("presence_penalty", PinValue::Float(CODING_PRESENCE_PENALTY)));
    }
    // [VANTA-WHY] §14.16 — the measurement override, applied last so it is the
    // one key that wins over both presets. It REPLACES the coding preset's own
    // 0.0 rather than sitting beside it (two `presence_penalty` keys in one JSON
    // object is the second one silently winning), and inserts the key under
    // `pinned`, which otherwise carries none.
    if let Some(penalty) = presence_penalty_override() {
        match pins.iter_mut().find(|(key, _)| *key == "presence_penalty") {
            Some(slot) => slot.1 = PinValue::Float(penalty),
            None => pins.push(("presence_penalty", PinValue::Float(penalty))),
        }
    }
    // [VANTA-WHY] §14.20 — the output-cap override, applied last for §14.16's
    // reason and REPLACING the `max_tokens` slot the two builders above already
    // put there. It never appends: a second `max_tokens` key would be silently
    // won by whichever the JSON object kept, and the record would then name a cap
    // no body carried.
    if let Some(cap) = max_tokens_override(site) {
        match pins.iter_mut().find(|(key, _)| *key == "max_tokens") {
            Some(slot) => slot.1 = PinValue::Int(cap),
            None => pins.push(("max_tokens", PinValue::Int(cap))),
        }
    }
    pins
}

/// The pins as a JSON object, for a request body.
fn sampling_pins_json(preset: SamplingPreset, site: &str) -> Value {
    let mut object = serde_json::Map::new();
    for (key, value) in sampling_pin_table(preset, site) {
        let rendered = match value {
            PinValue::Float(v) => json!(v),
            PinValue::Int(v) => json!(v),
        };
        object.insert(key.to_string(), rendered);
    }
    Value::Object(object)
}

/// The same pins as `key=value` pairs for `run.summary`, values rendered exactly
/// as a body carries them.
pub fn sampling_pins_text(preset: SamplingPreset, site: &str) -> Vec<(&'static str, String)> {
    sampling_pin_table(preset, site)
        .into_iter()
        .map(|(key, value)| {
            let text = match value {
                PinValue::Float(v) => format!("{v:?}"),
                PinValue::Int(v) => v.to_string(),
            };
            (key, text)
        })
        .collect()
}

// ============================================================================
// REASONING CONTROL — `VANTA_REASONING_BUDGET` / `VANTA_REASONING_EFFORT` (§14.21)
// ============================================================================
//
// [VANTA-WHY] §14.13's output cap and its brief re-ask are a SAFETY NET: they
// decide what to do with an answer that has already spent its whole budget
// thinking. This is the control that acts earlier — it bounds the thinking
// itself, so the call that would have burned 32,768 tokens re-deriving its own
// plan does not get to.
//
// [VANTA-WHY] The interface is deliberately provider-neutral and deliberately
// small: a TOKEN BUDGET and an EFFORT LEVEL, both optional, both meaning the same
// thing whatever the transport. It is deliberately NOT "continue the answer from a
// partial prefix", which is the one mechanism a hosted reasoning API cannot be
// asked for — OpenAI does not hand back the reasoning tokens to continue from,
// and Anthropic will not accept a prefilled assistant turn while extended
// thinking is on — so a design that needed it could not port, and porting is the
// point: the same arm has to be runnable against llama.cpp today and against a
// hosted API later.
//
// [VANTA-WHY] Each backend implements the neutral pair its own way, which is why
// the record carries the MECHANISM beside the request. `llamacpp:noop` for the
// effort is the honest record of a knob that did nothing: llama.cpp's sampler has
// no effort level, and this model's chat template declares
// `supports_reasoning_effort: false`, so the six call sites' existing
// `reasoning_effort: "low"` is sent and then ignored. Recording that as an applied
// control would credit the run with an intervention it never made.
//
// [VANTA-WHY] The budget is PER THINKING BLOCK, not per completion, and the
// record's mechanism says so. llama.cpp's `reasoning_budget_tokens` installs a
// start/end-tag sampler that re-arms with a fresh window on every new start tag,
// so a model that closes its thinking block and opens another gets the budget
// again (measured through the router: 11,976 → 219 characters of reasoning with
// the budget at 64, and a reopened block is armed afresh). Nothing here therefore
// bounds a RESPONSE. The output cap plus §14.13's re-ask remains the bound on the
// call, exactly as it was; this knob only removes waste inside it.

/// The reasoning effort a run may ask for (§14.21, §14.28).
///
/// [VANTA-WHY] The names are llama.cpp's own — and the same names OpenAI uses
/// for `reasoning.effort` — so an arm that says `high` means the same thing on
/// either transport and only the MECHANISM differs. `none` is a level, not the
/// absence of the switch: unset is the absence.
///
/// [VANTA-WHY] §14.28 — six levels, not four. The hosted families document
/// `xhigh` and `max` (`gpt-6-luna` and `gpt-6.1-sol` both list them), so an arm
/// that cannot name them cannot run the top of the ladder this project is
/// measuring. `minimal` is deliberately NOT here: neither model page lists it,
/// and an enum is the wrong place to invent a level. Which of the six a given
/// model may be asked for is a per-family fact and lives in
/// [`capabilities::OpenAiFamily::accepted_efforts`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ReasoningEffort {
    /// Parse `VANTA_REASONING_EFFORT`.
    ///
    /// [VANTA-WHY] An unparsable value is an error, never a fallback: this is an
    /// ablation switch, and an arm that ran at a level nobody asked for is not
    /// the arm the operator compared against.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "none" => Ok(Self::None),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            "max" => Ok(Self::Max),
            other => Err(format!(
                "CONFIG_ERROR: VANTA_REASONING_EFFORT='{other}' is not a level \
                 (expected none|low|medium|high|xhigh|max)."
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// The sites a reasoning budget applies to (§14.21).
///
/// [VANTA-WHY] The Attacker's three call sites and nothing else: they are the
/// ones whose reasoning is the run-away §14.13 measured. The intent audit is
/// pinned at temperature 0.0 and a budget there would move a measurement the
/// design holds fixed, and the Creator is outside it too — a patch reasoned about
/// for fewer tokens is a different experiment, not the same one made cheaper.
pub const REASONING_BUDGET_SITES: &[&str] =
    &["attacker_generate", "attacker_revise", "attacker_simplify"];

/// The requested budget, or `None` when `VANTA_REASONING_BUDGET` is unset.
static REASONING_BUDGET: Mutex<Option<u32>> = Mutex::new(None);

/// The requested effort, or `None` when `VANTA_REASONING_EFFORT` is unset.
static REASONING_EFFORT: Mutex<Option<ReasoningEffort>> = Mutex::new(None);

/// Ask for a reasoning budget of `tokens` per thinking block; `None` is off.
pub fn set_reasoning_budget(tokens: Option<u32>) {
    *REASONING_BUDGET.lock().unwrap() = tokens;
}

/// Ask for a reasoning effort level; `None` is off.
pub fn set_reasoning_effort(effort: Option<ReasoningEffort>) {
    *REASONING_EFFORT.lock().unwrap() = effort;
}

fn reasoning_budget() -> Option<u32> {
    *REASONING_BUDGET.lock().unwrap()
}

fn reasoning_effort() -> Option<ReasoningEffort> {
    *REASONING_EFFORT.lock().unwrap()
}

/// The budget `site`'s request body carries, or `None` (§14.21).
fn budget_for_site(site: &str) -> Option<u32> {
    if REASONING_BUDGET_SITES.contains(&site) {
        reasoning_budget()
    } else {
        None
    }
}

/// llama.cpp implements the budget with one request field, per thinking block.
const LLAMACPP_BUDGET_MECHANISM: &str = "llamacpp:reasoning_budget_tokens";

/// llama.cpp implements the effort level with nothing at all.
const LLAMACPP_EFFORT_MECHANISM: &str = "llamacpp:noop";

/// OpenAI implements the effort with one request field, and it is not a no-op.
///
/// [VANTA-WHY] §14.28 — the hosted spelling has to be its own word because the
/// two mechanisms are not the same claim: `llamacpp:noop` says "the level was
/// sent and ignored", while this says "the level this run asked for is the level
/// the model reasoned at". Pooling runs that recorded one word across both
/// backends would credit the local arm with an intervention it never made.
pub const OPENAI_EFFORT_MECHANISM: &str = "openai:reasoning_effort";

/// The hosted backend has no equivalent of llama.cpp's token budget (§14.28).
///
/// [VANTA-WHY] `unsupported`, not `openai:...`: the request field is a llama.cpp
/// extension, so the honest record of a budget on a hosted model is that the
/// control was ASKED FOR and could not be expressed — the run is then visibly
/// not the arm the operator set, instead of silently being a different one.
pub const UNSUPPORTED_MECHANISM: &str = "unsupported";

/// What `reasoning_effort` records when the hosted run asked for no level.
///
/// [VANTA-WHY] A word, not `unset`: the two are different facts. `unset` would
/// say "no effort was in play", which is false — the model reasoned, at the
/// level its provider chose. The word is also the one the `unset` mechanism
/// pairs with, so a reader of one field can read the other.
pub const PROVIDER_DEFAULT_EFFORT: &str = "provider_default";

/// The effort level the Creator call site pins (Item S).
///
/// [VANTA-WHY] Item S — the Creator has hard-coded `reasoning_effort: "low"`
/// since v0.2, exactly like the five other call sites. On the local backend that
/// value is sent verbatim and ignored (`llamacpp:noop`); on a hosted reasoning
/// model it is a real intervention nobody chose, so `capabilities` drops it
/// unless the run asked for a level the family accepts (§14.28). Named here so
/// the request body the Creator builds and the `run.summary` field that
/// describes it read the SAME pin and cannot drift onto two spellings of "low".
pub const CREATOR_PINNED_EFFORT: &str = "low";

/// Apply the run's reasoning controls to ONE request body (§14.21).
///
/// [VANTA-WHY] Called at the tail of [`with_sampling`], and therefore on the body
/// the server actually receives — the same place §14.20's output cap is applied,
/// for the same reason: a value the record names must be a value a body carried.
/// An unset switch touches nothing, so a default run's bodies are byte-identical
/// to what they were before this existed.
pub fn apply_reasoning_control(object: &mut serde_json::Map<String, Value>, site: &str) {
    if let Some(tokens) = budget_for_site(site) {
        object.insert("reasoning_budget_tokens".to_string(), json!(tokens));
    }
    // [VANTA-WHY] The effort REPLACES a key the call site already set; it does
    // not invent one for a site that has none. The six call sites that carry
    // `reasoning_effort` are the ones this knob is about, and a site that never
    // asked for an effort must not be handed one by a run-level switch.
    if let Some(effort) = reasoning_effort() {
        if object.contains_key("reasoning_effort") {
            object.insert("reasoning_effort".to_string(), json!(effort.name()));
        }
    }
}

/// The budget that was ASKED FOR, as `run.summary` records it (§14.21).
pub fn reasoning_budget_text() -> String {
    reasoning_budget()
        .map(|tokens| tokens.to_string())
        .unwrap_or_else(|| "unset".to_string())
}

/// The mechanism that implements the budget, or `unset` when nothing was asked.
///
/// [VANTA-WHY] §14.28 — backend-dependent, because the same switch means two
/// different things: llama.cpp has a field for it, the hosted API does not.
/// `unsupported` is the record of a control the run asked for and could not
/// send; the body carries nothing either way, which is exactly why the field has
/// to say so.
pub fn reasoning_budget_mechanism(is_openai: bool) -> &'static str {
    if reasoning_budget().is_some() {
        if is_openai {
            UNSUPPORTED_MECHANISM
        } else {
            LLAMACPP_BUDGET_MECHANISM
        }
    } else {
        "unset"
    }
}

/// The effort a request body carries, as `run.summary` records it (§14.21).
///
/// [VANTA-WHY] The EFFECTIVE value, not the switch: the six call sites have
/// hard-coded `low` since v0.2, so a field that read `unset` on a default LOCAL
/// run would describe a run that sent no effort at all. Setting the switch to
/// `low` is therefore indistinguishable from leaving it unset — which is the
/// truth, since the two send byte-identical bodies.
///
/// [VANTA-WHY] §14.28 — on the hosted backend the same unset switch is NOT `low`:
/// the call sites' `low` is dropped before the request leaves (`capabilities`),
/// so the model reasons at whatever its provider default is. Recording `low`
/// there would name a level the model was never asked for, and
/// `provider_default` is the fact. The two backends' default runs are different
/// arms and the record has to be able to say so.
pub fn reasoning_effort_text(is_openai: bool) -> String {
    match reasoning_effort() {
        Some(effort) => effort.name().to_string(),
        None if is_openai => PROVIDER_DEFAULT_EFFORT.to_string(),
        None => "low".to_string(),
    }
}

/// The mechanism that implements the effort level (§14.21, §14.28).
///
/// [VANTA-WHY] The LOCAL branch is unchanged and unconditional: a llama.cpp
/// request carries an effort whatever the switch says (the six call sites
/// hard-code `low`), and a llama.cpp server ignores the field, so the mechanism
/// is `llamacpp:noop` for every local run — set or unset. Replacing that with
/// `unset` on a default run would claim the field was absent, and it is not.
///
/// [VANTA-WHY] §14.28 — the hosted branch is the one that depends on the switch,
/// because it is the one whose body depends on it: with the switch unset the
/// shaper has removed the call sites' `low`, so no mechanism exists and the
/// value field's `provider_default` says whose choice the level was.
pub fn reasoning_effort_mechanism(is_openai: bool) -> &'static str {
    if is_openai {
        if reasoning_effort().is_none() {
            "unset"
        } else {
            OPENAI_EFFORT_MECHANISM
        }
    } else {
        LLAMACPP_EFFORT_MECHANISM
    }
}

/// The effort level this run asked for, or `None` when the switch was unset
/// (§14.28).
///
/// [VANTA-WHY] The one public way to ask, so that the two places that must agree
/// read the same value: `post_chat_meta` hands it to the shaping table, and the
/// driver hands it to the run-level `request_shape_not_sent` rule. Both then
/// describe the same request, which is the only reason a record can be compared
/// across the two.
pub fn requested_effort() -> Option<&'static str> {
    reasoning_effort().map(|effort| effort.name())
}

/// Merge the active preset's sampler pins for `site` into a call site's extras
/// (§14.11).
///
/// [VANTA-WHY] Each call site keeps its own `response_format` /
/// `reasoning_effort`; only the sampler keys are applied here, so the preset
/// cannot drift between sites. Under `pinned` (the default) this inserts exactly
/// the `temperature` / `max_tokens` the call site used to hard-code, so a default
/// run's request bodies are unchanged. A `coding` run additionally pins `top_p` /
/// `top_k` / `min_p` / `presence_penalty` on every site, the auditor included —
/// only `max_tokens` is left to the call site, and only the batch audit scales it.
pub fn with_sampling(extras: Value, site: &str) -> Value {
    with_sampling_pins(extras, sampling_preset(), site)
}

/// [`with_sampling`] with the preset taken as an EXPLICIT input.
///
/// [VANTA-WHY] The preset is an argument rather than [`sampling_preset`] so a
/// unit test can name the preset it is about. `sampling_preset` memoises
/// `VANTA_SAMPLING` in a `OnceLock`, so a test that reached it only through
/// `with_sampling` silently changed meaning with the ambient environment — the
/// `coding` preset inserts `presence_penalty: 0.0` on every site, which broke a
/// `pinned` body's "no `presence_penalty` key" assertion. The production entry
/// point is unchanged: `with_sampling` still reads the run's preset here.
fn with_sampling_pins(mut extras: Value, preset: SamplingPreset, site: &str) -> Value {
    let pins = sampling_pins_json(preset, site);
    let object = extras
        .as_object_mut()
        .expect("call-site extras are always a JSON object");
    if let Value::Object(pins) = pins {
        for (key, value) in pins {
            object.insert(key, value);
        }
    }
    // [VANTA-WHY] §14.21 — the reasoning controls last, for §14.16's reason: they
    // touch only a `reasoning_*` key and leave every sampler pin where it was.
    // With both switches unset this call changes nothing.
    apply_reasoning_control(object, site);
    extras
}

/// The server's own `top_p` default — the sampler parameter no call site pins
/// under the `pinned` preset, so every request inherits whatever the server is
/// configured with.
///
/// [VANTA-WHY] This is the one value `run.summary` really does have to read from
/// `/props` (§4.3): `temperature` and `max_tokens` are pinned per call (see
/// `SAMPLING_SCHEDULE`), `top_p` is not, so the server's answer is the only
/// truthful source. `None` means the server did not publish the key — a minimal
/// stub, or a llama.cpp build that omits it — and it is recorded as unset rather
/// than replaced by a plausible default, because a fabricated sampler value in
/// the provenance block is precisely the kind of claim this project exists to
/// avoid.
///
/// [VANTA-WHY] §14.11 — `top_p` is the server's default only under the `pinned`
/// preset: under `VANTA_SAMPLING=coding` every site, the intent auditor included,
/// pins `top_p` itself, so this server value then describes no call in the run.
/// The pinned values are recorded per role as `top_p_<site>`, which is what keeps
/// the two facts separable in the record and neither a claim about a call it does
/// not describe.
pub async fn server_top_p_default(client: &reqwest::Client, cfg: &LlmConfig) -> Option<f64> {
    // [VANTA-WHY] The hosted backend publishes no /props at all, so there is
    // no "server default" to report. Returning None (leaving the manifest field
    // null with its note) is the honest answer; fetching
    // https://api.openai.com/props to get a 404 and then calling that None
    // would be the same value for the price of a network call and a lie about
    // where it came from.
    if cfg.is_openai() {
        return None;
    }
    let body = fetch_props(client, cfg).await?;
    body["default_generation_settings"]["top_p"].as_f64()
}

/// Resolve the model's context window for THIS run.
///
/// Order: `VANTA_LLM_CONTEXT_TOKENS` if set (it wins, but a disagreement with
/// the server is logged, because a stale override silently starves the guard),
/// else the server's `/props` `n_ctx`, else 65536 with a warning.
///
/// [VANTA-WHY] On the openai backend the /props step does not exist, so the
/// order collapses to "the configured value, else 65536 with a warning" — the
/// probe is skipped rather than attempted. See the note in the body.
pub async fn context_window_tokens(client: &reqwest::Client, cfg: &LlmConfig) -> u32 {
    *CONTEXT_WINDOW
        .get_or_init(|| async {
            // [VANTA-WHY] The hosted backend has no /props, so the probe is
            // SKIPPED rather than allowed to fail: asking
            // https://api.openai.com/props would be a wasted round trip whose
            // only outcome is the 404 the fallback already assumes. With
            // nothing to reconcile, VANTA_LLM_CONTEXT_TOKENS is simply taken
            // (item 1's rule), and an unset value keeps the same 65536 backstop
            // the local path uses — loudly, because that number is far below a
            // hosted model's real window and a silently-too-small guard would
            // refuse prompts that fit.
            if cfg.is_openai() {
                return match cfg.context_window_tokens {
                    Some(configured) => configured,
                    None => {
                        warn!(
                            "VANTA_LLM_CONTEXT_TOKENS unset on the openai backend (no /props \
                             to ask); falling back to 65536 — set VANTA_LLM_CONTEXT_TOKENS to \
                             the hosted model's real window"
                        );
                        65_536
                    }
                };
            }
            let probed = probe_server_context_window(client, cfg).await;
            match (cfg.context_window_tokens, probed) {
                (Some(configured), Some(n_ctx)) => {
                    if configured != n_ctx {
                        warn!(
                            configured_context_window = configured,
                            server_n_ctx = n_ctx,
                            "VANTA_LLM_CONTEXT_TOKENS disagrees with the server's /props n_ctx; \
                             using the configured value"
                        );
                    }
                    configured
                }
                (Some(configured), None) => configured,
                (None, Some(n_ctx)) => {
                    info!(
                        n_ctx,
                        "context window taken from the server's /props (VANTA_LLM_CONTEXT_TOKENS unset)"
                    );
                    n_ctx
                }
                (None, None) => {
                    warn!(
                        "could not read n_ctx from the server's /props; falling back to 65536"
                    );
                    65_536
                }
            }
        })
        .await
}

// ============================================================================
// HELPER: assemble_sse_stream — consume a streamed chat completion
// ============================================================================
//
// [VANTA-WHY] Every chat completion is requested with `stream: true` so that
// LIVENESS is observable. A non-streaming POST has exactly one observable event
// (the response) and therefore can only be policed by a total deadline — which
// cannot tell "still generating" from "dead", so it killed healthy 20-minute
// generations. A stream emits an event per token, so the same watchdog can ask
// the only question that matters: has ANYTHING arrived recently?
//
// Two independent limits, both enforced here:
//   * idle timeout  — no new chunk for N seconds → the model is not generating.
//   * overall backstop — the whole call may not exceed N seconds, so a server
//     that emits a token just often enough to defeat the idle timeout still
//     cannot hold the run forever.
//
// On either limit the byte stream is DROPPED before the error is returned. That
// drop is the cancel: hyper abandons the in-flight response body and closes the
// socket, which is what tells llama-server to stop generating, so no orphan task
// survives into the retry to fight over the shared KV cache.
#[derive(Debug, Default)]
struct StreamedCompletion {
    content: String,
    reasoning: String,
    /// From the `usage` block (`stream_options.include_usage`).
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    /// `usage.prompt_tokens_details.cached_tokens` — the share of `prompt_tokens`
    /// the provider served from its prompt cache (§14.29).
    ///
    /// [VANTA-WHY] Kept BESIDE `prompt_tokens` rather than subtracted from it,
    /// because the two are different facts: `prompt_tokens` is what the request
    /// spent, and this is how much of it was cheap. A cost column needs the second
    /// to apply the cached rate, and a cache diagnostic needs it to report a hit
    /// RATE — neither recovers it from the first.
    cached_tokens: Option<u64>,
    /// `usage.completion_tokens_details.reasoning_tokens` — the share of
    /// `completion_tokens` the model spent thinking (§14.29).
    ///
    /// [VANTA-WHY] A SUBSET of `completion_tokens`, not a sibling of it. Recorded
    /// so a run can report how much of its output budget went to reasoning, which
    /// is the measurable half of "the model ran away" (§14.5); it must never be
    /// ADDED to the completion count, or the same tokens are billed twice.
    reasoning_tokens: Option<u64>,
    /// From llama-server's own final frame, which carries `timings` whether or
    /// not the server honours `stream_options`. `prompt_n` is the number of
    /// prompt tokens it processed and `predicted_n` the number it generated.
    timings_prompt_tokens: Option<u64>,
    timings_completion_tokens: Option<u64>,
    /// The server's own `finish_reason`, when it published one. Only `length`
    /// changes what Vanta does with the reply: it means the answer was CUT OFF at
    /// `max_tokens`, which is a different fact from "the answer was malformed"
    /// and has a different fix.
    finish_reason: Option<String>,
}

/// Fold ONE `\n\n`-delimited SSE event into the accumulator.
/// Returns true when the stream announced `[DONE]`.
fn consume_sse_event(event: &str, out: &mut StreamedCompletion) -> Result<bool, String> {
    let mut done = false;
    for line in event.lines() {
        let Some(payload) = line.trim_end().strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() {
            continue;
        }
        if payload == "[DONE]" {
            done = true;
            continue;
        }
        let value: Value = match serde_json::from_str(payload) {
            Ok(value) => value,
            Err(e) => {
                // Keep-alive comments and vendor fields are not ours to parse;
                // a frame we cannot read is skipped rather than fatal.
                trace!(err = %e, "sse: skipping unparsable data frame");
                continue;
            }
        };
        // A server-side failure arrives INSIDE the stream, not as an HTTP
        // status: llama-server reports an overflowing KV cache this way. It must
        // be an error, not an empty completion, or the run would record a
        // nonsense answer.
        if let Some(error) = value.get("error") {
            let message = error["message"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| error.to_string());
            return Err(format!("LLM server reported an error mid-stream: {message}"));
        }
        if let Some(choice) = value["choices"].as_array().and_then(|c| c.first()) {
            let delta = &choice["delta"];
            if delta.is_object() {
                if let Some(s) = delta["content"].as_str() {
                    out.content.push_str(s);
                }
                if let Some(s) = delta["reasoning_content"].as_str() {
                    out.reasoning.push_str(s);
                }
            } else if let Some(message) = choice.get("message") {
                // A server that ignores `stream` but answers with an SSE
                // envelope still produces a usable response.
                if let Some(s) = message["content"].as_str() {
                    out.content.push_str(s);
                }
                if let Some(s) = message["reasoning_content"].as_str() {
                    out.reasoning.push_str(s);
                }
            }
            // [VANTA-WHY] Why the server stopped, when it says. `length` is the
            // one value the caller acts on: the answer was CUT OFF at
            // `max_tokens`, which is a different fact from "the answer was
            // malformed" — the fix is a smaller question or a bigger cap, not
            // another attempt at the same request. Recorded rather than acted on
            // here, so nothing about a completed call changes.
            if let Some(reason) = choice["finish_reason"].as_str() {
                out.finish_reason = Some(reason.to_string());
            }
        }
        if let Some(usage) = value.get("usage").filter(|u| u.is_object()) {
            if let Some(prompt_tokens) = usage["prompt_tokens"].as_u64() {
                out.prompt_tokens = Some(prompt_tokens);
            }
            if let Some(completion_tokens) = usage["completion_tokens"].as_u64() {
                out.completion_tokens = Some(completion_tokens);
            }
            if let Some(total_tokens) = usage["total_tokens"].as_u64() {
                out.total_tokens = Some(total_tokens);
            }
            // [VANTA-WHY] §14.29 — the two nested counts the hosted providers
            // publish, read from the objects they actually arrive in.
            // `cached_tokens` is the cheap share of the prompt and
            // `reasoning_tokens` the thinking share of the completion; neither is
            // a Chat Completions top-level field, so a flat read of the `usage`
            // object would find nothing and silently record "no cache, no
            // reasoning" for every hosted call. Absent keys stay absent: a
            // provider that reports neither must not be recorded as reporting
            // zero.
            if let Some(cached) = usage["prompt_tokens_details"]["cached_tokens"].as_u64() {
                out.cached_tokens = Some(cached);
            }
            if let Some(reasoning) = usage["completion_tokens_details"]["reasoning_tokens"].as_u64()
            {
                out.reasoning_tokens = Some(reasoning);
            }
        }
        // llama-server's non-OpenAI tail. Read even when `usage` arrived, as a
        // fallback for a build that ignores `stream_options`: without it, a
        // completion count would not exist at all on such a server.
        if let Some(timings) = value.get("timings").filter(|t| t.is_object()) {
            if let Some(prompt_n) = timings["prompt_n"].as_u64() {
                out.timings_prompt_tokens = Some(prompt_n);
            }
            if let Some(predicted_n) = timings["predicted_n"].as_u64() {
                out.timings_completion_tokens = Some(predicted_n);
            }
        }
    }
    Ok(done)
}

async fn assemble_sse_stream(
    mut chunks: Pin<Box<dyn Stream<Item = Result<Vec<u8>, String>> + Send>>,
    idle_timeout: Duration,
    max_call: Duration,
) -> Result<StreamedCompletion, String> {
    // tokio's clock, not std's: this instant is compared against deadlines
    // produced by tokio::time::timeout, so both must read the same clock.
    let started = tokio::time::Instant::now();
    let mut buffer = String::new();
    let mut out = StreamedCompletion::default();

    loop {
        let elapsed = started.elapsed();
        if elapsed >= max_call {
            return Err(format!(
                "LLM call abandoned: exceeded the overall backstop of {}s \
                 (VANTA_LLM_MAX_SECS) after {:.1}s; connection dropped to cancel the server task",
                max_call.as_secs(),
                elapsed.as_secs_f64()
            ));
        }
        // Never wait longer than the backstop has left, so a stream whose idle
        // timeout is larger than the remaining budget still ends on time.
        let wait = idle_timeout.min(max_call - elapsed);
        let chunk = match tokio::time::timeout(wait, chunks.next()).await {
            Err(_elapsed) => {
                drop(chunks);
                return Err(format!(
                    "LLM stream stalled: no new data for {}s (VANTA_LLM_IDLE_TIMEOUT_SECS); \
                     connection dropped to cancel the server task",
                    idle_timeout.as_secs()
                ));
            }
            Ok(None) => break,
            Ok(Some(Err(e))) => {
                drop(chunks);
                return Err(format!("LLM stream error: {e}"));
            }
            Ok(Some(Ok(bytes))) => bytes,
        };
        // SSE frames are delimited by a blank line; normalise CRLF so both
        // line endings split the same way.
        buffer.push_str(&String::from_utf8_lossy(&chunk).replace('\r', ""));
        while let Some(end) = buffer.find("\n\n") {
            let event: String = buffer[..end].to_string();
            buffer.drain(..end + 2);
            if consume_sse_event(&event, &mut out)? {
                return Ok(out);
            }
        }
    }

    Ok(out)
}

// ============================================================================
// HELPER: extract_first_json_object(&str) -> Option<&str> (kept for pedagogy)
// ============================================================================
#[allow(dead_code)]
pub fn extract_first_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut prev_was_backslash = false;
    let mut end_idx: Option<usize> = None;

    for (i, &b) in bytes[start..].iter().enumerate() {
        let abs_idx = start + i;
        match b {
            b'"' if !prev_was_backslash => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 { end_idx = Some(abs_idx); break; }
            }
            _ => {}
        }
        prev_was_backslash = b == b'\\' && !prev_was_backslash;
    }
    let end = end_idx?;
    Some(&s[start..=end])
}

// ============================================================================
// HELPER: extract_first_valid_json(&str) -> Option<&str>
//   Iterate ALL balanced brace pairs, try serde_json parse on each,
//   return first that parses. Solves "Rust code block in prose" failure.
// ============================================================================
pub fn extract_first_valid_json(s: &str) -> Option<&str> {
    let mut cursor: usize = 0;
    let s_bytes = s.as_bytes();
    let mut attempts: u32 = 0;
    const MAX_ATTEMPTS: u32 = 200;

    while cursor < s_bytes.len() && attempts < MAX_ATTEMPTS {
        attempts += 1;
        let rel_start = match s[cursor..].find('{') {
            Some(i) => i,
            None => break,
        };
        let abs_start = cursor + rel_start;

        let bytes = s.as_bytes();
        let mut depth: i32 = 0;
        let mut in_string = false;
        let mut prev_was_backslash = false;
        let mut abs_end: Option<usize> = None;

        for (i, &b) in bytes[abs_start..].iter().enumerate() {
            let abs = abs_start + i;
            match b {
                b'"' if !prev_was_backslash => in_string = !in_string,
                b'{' if !in_string => depth += 1,
                b'}' if !in_string => {
                    depth -= 1;
                    if depth == 0 { abs_end = Some(abs); break; }
                }
                _ => {}
            }
            prev_was_backslash = b == b'\\' && !prev_was_backslash;
        }

        let end = match abs_end {
            Some(e) => e,
            None => break,
        };

        let candidate = &s[abs_start..=end];
        let parse_result: Result<serde_json::Value, _> = serde_json::from_str(candidate);
        match parse_result {
            Ok(_v) => return Some(candidate),
            Err(_) => { cursor = end + 1; }
        }
    }
    None
}

// ============================================================================
// HELPER: extract_json_candidate(raw: &str) -> Result<&str, String>
//   4-stage pipeline: fences → prose scan → fallthrough → sanity.
// ============================================================================
pub fn extract_json_candidate<'a>(raw: &'a str) -> Result<&'a str, String> {
    let stage1 = strip_markdown_fences(raw);
    let stage2: &str = if let Some(json_slice) = extract_first_valid_json(stage1) {
        json_slice
    } else if stage1.starts_with('{') && stage1.ends_with('}') {
        stage1
    } else {
        return Err(format!(
            "NO valid JSON found across all brace-balanced candidates. starts={:?}, len={}",
            stage1.chars().take(200).collect::<String>(),
            stage1.len()
        ));
    };
    if stage2.is_empty() {
        return Err("EMPTY after all cleaning stages".into());
    }
    if !stage2.starts_with('{') || !stage2.ends_with('}') {
        return Err(format!(
            "candidate not JSON object (starts={:?}, ends={:?})",
            stage2.chars().next(), stage2.chars().last()
        ));
    }
    Ok(stage2)
}

// ============================================================================
// HELPER: post_chat(&client, &cfg, messages, extra_fields, site) -> Result<String>
//   Returns assistant content string (fusion: content + reasoning_content).
//   `site` is the call-site name from `SAMPLING_SCHEDULE`; it labels the token
//   accounting row this call records (see `CallUsage`).
// ============================================================================
pub async fn post_chat(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    messages: Vec<Value>,
    extra_fields: Value,
    site: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    post_chat_meta(
        client,
        cfg,
        messages,
        extra_fields,
        site,
        cfg.seed,
        // Not in a retry loop: this call is attempt 1 by construction.
        1,
    )
    .await
    .map(|(content, _finish_reason)| content)
}

/// As `post_chat`, but the server's own `finish_reason` comes back beside the
/// content, and the `seed` this attempt sends can be overridden.
///
/// [VANTA-WHY] §14.5 — the retry loop needs to tell a TRUNCATED answer from a
/// malformed one, and only the server knows which it was. Returning it beside
/// the content (rather than in an error) keeps every existing caller's behaviour
/// identical: a truncated answer still comes back as content, still fails to
/// parse, and is still retried; what changes is that the FINAL failure can say
/// why, and the one caller that can act on it — the batched audit, which can ask
/// the same round's questions one at a time — does.
///
/// [VANTA-WHY] `seed` is the value THIS attempt sends, not necessarily
/// `cfg.seed`: a retry of a call sampled above temperature 0.0 advances it, so
/// that the harness's single pinned seed does not reproduce the identical
/// cut-off prefix (see `seed_for_attempt`). `post_chat` passes `cfg.seed`
/// untouched, so attempt 1 of every call — and every call that never retries —
/// is byte-identical to what it was before this parameter existed. `None` means
/// "send no `seed` key at all", which is what keeps an unseeded run's bodies
/// byte-for-byte unchanged.
async fn post_chat_meta(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    messages: Vec<Value>,
    extra_fields: Value,
    site: &str,
    seed: Option<u64>,
    // Which retry of `structured_call_with_retry` this is, so the accounting row
    // can carry it (§14.12). `post_chat` — the one-shot wrapper the M0 spike and
    // bootstrap use — has no retry loop and passes 1.
    attempt: u32,
) -> Result<(String, Option<String>), Box<dyn Error + Send + Sync>> {
    let url = cfg.chat_completions_url();

    let body: Value = json!({
        "model": cfg.model_id,
        "messages": messages,
        // [VANTA-WHY] These two are the M0 spike's values, not the pipeline's:
        // every M3 agent passes its own through `extra_fields`, which is merged
        // below and therefore WINS. Kept as the fallback for a caller that sends
        // no `extras` (bootstrap does), and left at the values the baseline was
        // collected under rather than "tidied" — see `SAMPLING_SCHEDULE`.
        "max_tokens": 16384,
        "temperature": 0.2,
        // [VANTA-WHY] Streamed, always. The response is reassembled below into
        // exactly the same `===CONTENT===/===REASONING===` string a
        // non-streaming call produced, so every parser, retry and agent on top
        // of this function is unaffected — what changes is that liveness is now
        // observable while the model generates (see `assemble_sse_stream`).
        "stream": true,
        // [VANTA-WHY] A streamed response does not carry a `usage` block unless
        // it is asked for, so without this the completion token count does not
        // exist — and it cannot be derived from the content (bytes per token is
        // exactly the ratio the pre-flight estimate gets wrong). This is the one
        // request-body change token accounting needs; it asks the server to
        // measure, it does not change what is sampled.
        "stream_options": {"include_usage": true},
    });

    let mut full_body = body.as_object().cloned().unwrap_or_default();
    if let Some(obj) = extra_fields.as_object() {
        for (k, v) in obj {
            full_body.insert(k.clone(), v.clone());
        }
    }
    // [VANTA-WHY] The seed is inserted ONLY when one was asked for
    // (VANTA_LLM_SEED). An unset seed must leave the request body BYTE-for-byte
    // identical to what it was before this field existed — a `"seed": null`
    // would be a change to what every server sees, and `"seed": 0` would move
    // the sampling baseline the existing measurements were taken under. The
    // harness asserts the byte-identity; this is where it is kept.
    //
    // Placed after the `extra_fields` merge so a caller cannot smuggle one in
    // that disagrees with the configuration the run records as its provenance.
    //
    // [VANTA-WHY] §14.5 — `seed` is the ATTEMPT's seed, which the retry loop may
    // have advanced past the pinned one; `post_chat` passes `cfg.seed` as-is, so
    // the first attempt of every call sends exactly what it always did.
    if let Some(seed) = seed {
        full_body.insert("seed".to_string(), json!(seed));
    }

    // [VANTA-WHY] §14.26 — the output cap is read from the LLAMA.CPP spelling
    // BEFORE the body is shaped, because shaping renames that key for OpenAI.
    // Reading it after would silently fall back to 16384 on every hosted call:
    // the pre-flight guard would compare the prompt against a cap no request
    // carries, and a run whose call sites asked for 32768 would be refused (or
    // admitted) on a number that was never real.
    let requested_output_tokens = full_body
        .get(capabilities::LLAMACPP_OUTPUT_CAP_KEY)
        .and_then(Value::as_u64)
        .unwrap_or(16_384) as u32;

    // [VANTA-WHY] §14.26 — shape the body for the backend it is going to, ONCE,
    // here, on the object that is about to be serialised. Two facts come out and
    // both travel with the call: the body the server receives, and the record of
    // what was asked for versus what was sent. The local path takes the
    // `native` branch and is not touched at all — its bodies are pinned byte for
    // byte by the harness, and a filter that runs on them is a chance to change
    // them.
    // [VANTA-WHY] §14.29 — resolved ONCE, above the shaping branch, because two
    // later decisions read it: what the body may carry (§14.26/§14.28) and what
    // the call cost (§14.29). Computing it twice from the same model id would be
    // two chances for the shaping record and the price record to describe
    // different models.
    let family = OpenAiFamily::from_model_id(&cfg.model_id);
    // [VANTA-WHY] §14.31 — the service tier, when the run asked for `flex`.
    // Inserted BEFORE the shaping branch for §14.23's reason (the seed does the
    // same a few lines up): the record's `requested` list is the body as the
    // caller built it, so a key added after shaping would appear in `sent` but
    // not `requested` and `is_consistent` would have to be believed rather than
    // checked.
    //
    // [VANTA-WHY] Only `flex` is inserted, and only on the hosted backend.
    // `service_tier: "default"` is the API's own default, so sending it would be
    // a body change that means nothing — and an explicit `default` arm would
    // stop being byte-identical to an unset one, which is the property this
    // switch must not cost. The local branch of this `if` never runs, so a
    // llama.cpp body cannot gain the key.
    if cfg.is_openai() && cfg.service_tier == pricing::ServiceTier::Flex {
        full_body.insert(
            capabilities::SERVICE_TIER_KEY.to_string(),
            json!(cfg.service_tier.name()),
        );
    }
    let (full_body, request_shape) = if cfg.is_openai() {
        // [VANTA-WHY] §14.28 — the run's effort is passed IN rather than read by
        // the table, so "what did this run ask for?" is answered at the one call
        // site that has the run's configuration in hand, and the shaping function
        // stays a pure function of (body, family, request).
        let effort = requested_effort();
        let shaped = capabilities::shape_body_for_openai(&mut full_body, family, effort);
        (Value::Object(full_body), shaped)
    } else {
        let shaped = ShapedBody::native(&full_body);
        (Value::Object(full_body), shaped)
    };

    let context_window = context_window_tokens(client, cfg).await;
    let prompt_render = render_messages_for_token_estimate(&messages);
    let estimate = conservative_token_estimate(&prompt_render);
    let mut prompt_tokens = estimate;
    let mut token_source = "estimate";
    // [VANTA-WHY] Ask the server for an exact count only when the conservative
    // estimate is already about to reject. The /tokenize probe re-sends the whole
    // prompt — exactly the cost the narrowed review context exists to avoid — so
    // it must never run on the common path. This keeps the guard fail-closed
    // (the estimate over-counts) while still rescuing a borderline-but-fitting
    // request that the estimate would have refused.
    if context_budget_exceeded(estimate, requested_output_tokens, context_window) {
        if let Some(exact) = tokenize_via_server(client, cfg, &prompt_render).await {
            prompt_tokens = exact;
            token_source = "llama-server";
        }
    }
    info!(
        prompt_tokens,
        requested_output_tokens,
        context_window_tokens = context_window,
        token_source,
        "llm prompt budget"
    );
    if context_budget_exceeded(prompt_tokens, requested_output_tokens, context_window) {
        // [VANTA-WHY] This is a DETERMINISTIC local failure: the same prompt
        // against the same window fails identically on every attempt. Retrying
        // it three times only delayed the run and dressed a known-bad request up
        // as a flaky one, so `structured_call_with_retry` returns on the first
        // occurrence (it matches this exact prefix).
        return Err(format!(
            "{}: prompt_tokens={} + max_tokens={} + overhead={} > context_window={}",
            DETERMINISTIC_CONTEXT_PREFIX,
            prompt_tokens,
            requested_output_tokens,
            CHAT_OVERHEAD_TOKENS,
            context_window,
        )
        .into());
    }

    let t = Instant::now();
    // [VANTA-WHY] §14.30 — the send is its own retry loop now: a provider that
    // answers 429 is re-asked the SAME request against a budget that is not the
    // model's, so a busy minute cannot spend an attempt that exists to let the
    // model answer differently. `provider_retries` is what this call spent on
    // that ladder, and it travels into the row below rather than only into the
    // run's total, because "which call was throttled" is a fact about the call.
    let (resp, provider_retries) =
        send_with_provider_retries(client, cfg, &url, &full_body).await?;

    let status = resp.status();

    let byte_stream = resp.bytes_stream();
    let chunks: Pin<Box<dyn Stream<Item = Result<Vec<u8>, String>> + Send>> = Box::pin(
        byte_stream.map(|chunk| chunk.map(|b| b.to_vec()).map_err(|e| e.to_string())),
    );
    let streamed = assemble_sse_stream(
        chunks,
        Duration::from_secs(cfg.idle_timeout_secs),
        Duration::from_secs(cfg.max_call_secs),
    )
    .await?;

    let content = streamed.content.trim().to_string();
    let reasoning = streamed.reasoning.trim().to_string();

    if content.is_empty() && reasoning.is_empty() {
        return Err("Assistant response: BOTH content AND reasoning_content empty".into());
    }

    // [VANTA-WHY] The server's own count, printed beside ours. The local
    // estimate is what the pre-flight guard acts on, so the only way to know
    // whether its calibration is still right is to compare the two on real runs
    // — which is exactly the measurement that showed the old /3 heuristic was
    // ~1.5× high.
    match streamed.prompt_tokens {
        Some(actual) => {
            let ratio = if estimate == 0 {
                0.0
            } else {
                actual as f64 / f64::from(estimate)
            };
            info!(
                estimated_prompt_tokens = estimate,
                actual_prompt_tokens = actual,
                estimate_over_actual = format!("{ratio:.2}"),
                "llm prompt tokens: local estimate vs server usage"
            );
        }
        None => {
            info!(
                estimated_prompt_tokens = estimate,
                "llm prompt tokens: server returned no usage block, estimate only"
            );
        }
    }

    // ---- Token accounting (see `CallUsage`) ----
    // The numbers the server measured, in preference order, with the local
    // estimate kept ONLY as a labelled last resort. `total` is derived when the
    // server gives the two parts but not their sum, and the completion count is
    // derived from `total - prompt` when the server gives only those two — both
    // are arithmetic on measurements, not estimates.
    let measured_prompt = streamed.prompt_tokens.or(streamed.timings_prompt_tokens);
    let measured_completion = streamed
        .completion_tokens
        .or(streamed.timings_completion_tokens)
        .or_else(|| {
            streamed
                .total_tokens
                .zip(measured_prompt)
                .map(|(total, prompt)| total.saturating_sub(prompt))
        });
    let measured_total = streamed
        .total_tokens
        .or_else(|| match (measured_prompt, measured_completion) {
            (Some(prompt), Some(completion)) => Some(prompt + completion),
            _ => None,
        });
    let (prompt_tokens_recorded, usage_source) = match measured_prompt {
        Some(prompt) => (
            prompt,
            if streamed.prompt_tokens.is_some() {
                "server-usage"
            } else {
                "server-timings"
            },
        ),
        // Nothing from the server: record the estimate, but SAY it is one.
        None => (u64::from(estimate), "estimate"),
    };
    // [VANTA-WHY] §14.12 — the wall time is measured ONCE, here, and both the
    // record and the debug line below read this value: two `elapsed()` calls
    // would give the log and the session row different numbers for the same call.
    let wall_ms = t.elapsed().as_millis() as u64;
    // [VANTA-WHY] §14.29 — what this call cost and which price row said so.
    // The local backend is refused by NAME rather than by a zero rate: no tokens
    // are sold, and `not_billed` is a fact a batch table can report as such,
    // while `0.0` would put a free local run and a wrongly-priced hosted one in
    // the same column. The hosted branch lets `pricing` refuse for its own two
    // reasons (`no_usage` for an estimate-sourced call, `unpriced_family` for a
    // model this table has no rate for) instead of this call site guessing which
    // one applies.
    let (cost_usd, cost_basis) = if cfg.is_openai() {
        pricing::cost_for_family(
            family,
            // [VANTA-WHY] §14.31 — the tier THIS run asked for, read from the
            // same config field that decided the body's `service_tier` key a few
            // lines above. One source, so the row that prices a call and the
            // request a provider served cannot describe two different arms;
            // `flex` is documented at half the standard rates for the same
            // tokens, so pricing a flex run at the standard row would over-report
            // its cost by 2x.
            cfg.service_tier,
            usage_source,
            measured_prompt,
            streamed.cached_tokens,
            measured_completion,
        )
    } else {
        (None, pricing::COST_BASIS_NOT_BILLED.to_string())
    };
    let usage = CallUsage {
        seq: USAGE_SEQ.fetch_add(1, Ordering::Relaxed),
        site: site.to_string(),
        // [VANTA-WHY] §14.12 — the round comes from the process-global the driver
        // sets once per attempt BEFORE any agent runs (`set_forensics_round`), so
        // it is the same index `write_forensics` puts in the filename. Read here
        // rather than threaded through every agent: the round is a property of the
        // run, not of the call, and an agent signature changed for bookkeeping is
        // an agent signature that can change a verdict.
        round: FORENSICS_ROUND.load(Ordering::Relaxed),
        attempt,
        // [VANTA-WHY] §14.30 — how many times the provider was asked again before
        // it served THIS call. A fact about the call, recorded beside `attempt` so
        // a row that took a 429 on the way in is readable as such and cannot be
        // mistaken for a model that needed two tries.
        provider_retries,
        prompt_tokens: Some(prompt_tokens_recorded),
        completion_tokens: measured_completion,
        total_tokens: measured_total,
        // [VANTA-WHY] §14.29 — the two nested counts travel verbatim, `None` when
        // the provider published neither. Nothing here subtracts `reasoning` from
        // `completion` or `cached` from `prompt`: the recorded counts must be the
        // server's own numbers, and the arithmetic that reads them is a billing
        // concern, not a recording one.
        cached_tokens: streamed.cached_tokens,
        reasoning_tokens: streamed.reasoning_tokens,
        cost_usd,
        cost_basis,
        source: usage_source.to_string(),
        seed,
        // [VANTA-WHY] §14.23 — the mechanism follows the value that is actually in the
        // body, which is decided a few lines above: a `Some` seed was inserted as the
        // `seed` key, and `None` means no key at all. Nothing here can disagree with
        // what the request carried.
        //
        // [VANTA-WHY] §14.26 — the spelling is backend-dependent because the claim
        // is: `llamacpp:seed` and `openai:seed` are two different statements about
        // two different samplers, and a record that used one word for both could not
        // be pooled across backends. `unset` stays shared — "no seed was asked for"
        // means the same thing everywhere.
        seed_mechanism: if seed.is_some() {
            if cfg.is_openai() {
                OPENAI_SEED_MECHANISM
            } else {
                LLAMACPP_SEED_MECHANISM
            }
        } else {
            UNSET_SEED_MECHANISM
        }
        .to_string(),
        finish_reason: streamed.finish_reason.clone(),
        wall_ms,
        truncated: streamed.finish_reason.as_deref() == Some("length"),
        // [VANTA-WHY] §14.26 — what this body asked for and what it carried.
        request_shape,
        // [VANTA-WHY] §14.31 — the tier the PRICE ROW above was selected by,
        // spelled by the same helper `run.summary` uses. On the local backend it
        // is `unset`, because a llama.cpp server has no tier and the row's
        // `not_billed` basis says the same thing about money; on the hosted
        // backend it is exactly `cfg.service_tier`, i.e. the value that put the
        // `service_tier` key into the body this call sent.
        service_tier: cfg.service_tier_text().to_string(),
    };
    debug!(
        site,
        round = usage.round,
        attempt,
        // [VANTA-WHY] §14.30 — logged beside `attempt` so the two retry budgets are
        // told apart in the live log too: a call that reports `attempt=1` with
        // `provider_retries=2` was asked twice more because the service was busy,
        // never because the model answered badly.
        provider_retries,
        prompt_tokens = prompt_tokens_recorded,
        completion_tokens = ?measured_completion,
        token_source = usage_source,
        seed = ?seed,
        finish_reason = streamed.finish_reason.as_deref().unwrap_or("unset"),
        wall_ms,
        truncated = usage.truncated,
        // [VANTA-WHY] §14.29 — the cheap share of the prompt and the thinking
        // share of the output, plus what the call cost and which price row said
        // so. Logged beside the totals for §14.26's reason: a hosted run whose
        // price basis came out `unpriced_family` should say so while it happens,
        // not only in a session log somebody reads afterwards.
        cached_tokens = ?usage.cached_tokens,
        reasoning_tokens = ?usage.reasoning_tokens,
        cost_basis = %usage.cost_basis,
        // [VANTA-WHY] §14.31 — the tier, logged with the cost basis it selected:
        // a `flex` run that came out priced at `.../standard/...` is a mistake
        // worth seeing in the live log, not one to discover in a manifest.
        service_tier = %usage.service_tier,
        // [VANTA-WHY] §14.26 — logged as well as recorded, so a hosted run that
        // is silently missing a field says so while it happens rather than only
        // in the session log afterwards.
        dropped_fields = ?usage.request_shape.dropped,
        "llm call token usage recorded"
    );
    record_call_usage(usage);

    debug!(content_len = content.len(), reasoning_len = reasoning.len(), "model response received");
    trace!(content = %content.chars().take(500).collect::<String>(), "model content");
    trace!(reasoning = %reasoning.chars().take(500).collect::<String>(), "model reasoning");

    {
        use std::io::Write;
        if let Ok(mut f) = std::fs::File::create("/tmp/vanta_last_model_output.txt") {
            let _ = writeln!(f, "=== CONTENT ({len_c}) ===\n{content}\n\n=== REASONING ({len_r}) ===\n{reasoning}\n",
                len_c = content.len(), len_r = reasoning.len());
        }
    }

    let combined = format!(
        "===CONTENT===\n{}\n===END_CONTENT===\n\n===REASONING===\n{}\n===END_REASONING===",
        content, reasoning
    );

    debug!(
        status = status.as_u16(),
        elapsed_ms = wall_ms,
        total_tokens = streamed.total_tokens.unwrap_or(0),
        finish_reason = streamed.finish_reason.as_deref().unwrap_or("unset"),
        "streamed chat completion complete"
    );

    Ok((combined, streamed.finish_reason))
}

// ============================================================================
// HELPER: structured_call_with_retry<T> — N-attempt retry + typed parse
// ============================================================================
/// Prefix of the pre-flight budget failure. Kept as one constant because the
/// producer (`post_chat`) and the consumer (the retry loop below) must agree on
/// it exactly.
///
/// [VANTA-WHY] M3 item F5 — the driver names this failure as its own reason
/// (`prompt_over_budget`) instead of folding it into "the Creator call failed",
/// because a prompt that cannot fit says something about the RUN'S CONFIGURATION
/// and not about the candidate. Public for the same reason as
/// `TRUNCATED_OUTPUT_PREFIX`: so that second consumer does not re-spell the string.
pub const DETERMINISTIC_CONTEXT_PREFIX: &str = "Prompt would exceed model context";

/// Prefix of the failure a response cut off at `max_tokens` produces.
///
/// [VANTA-WHY] §14.5 — a truncated answer is a DIFFERENT fact from a malformed
/// one: the model was still producing when the output cap stopped it, so the
/// remedy is a smaller question or a bigger cap, and re-sending the same request
/// cannot fix it. The retry loop marks the final failure with this prefix and
/// the one consumer that can act on it — the batched audit, which can ask the
/// same round's questions one test at a time — matches it. Public so that
/// consumer does not have to re-spell the string.
pub const TRUNCATED_OUTPUT_PREFIX: &str = "LLM response truncated at max_tokens";

/// Marker `structured_call_with_retry` puts on an attempt that reached the
/// PARSER and did not satisfy it.
///
/// [VANTA-WHY] §14.5 — this is the difference between "the model answered, and
/// the answer was wrong" and "the model never answered" (a transport error, or a
/// context gate that never sent a request). Both exhaust the same retry budget
/// and produce the same final failure shape, and only one of them is the model's
/// behaviour, so a reader — and the batched audit's log — has to be able to tell
/// them apart. Public for the same reason as `TRUNCATED_OUTPUT_PREFIX`: so the
/// consumer does not have to re-spell the string.
pub const PARSE_FAILURE_PREFIX: &str = "PARSE ERR";

/// Marker `structured_call_with_retry` puts on a failure that was cut off at
/// `max_tokens` BOTH times it was asked — the original request and the brief
/// re-ask (§14.5, item 2).
///
/// [VANTA-WHY] A single cut-off is a fact about one generation; two in a row,
/// with a shorter instruction in between, is the call saying it cannot answer
/// this question inside its cap. That is a different fact from "the model
/// answered and the answer was wrong", and the reader has to be able to count
/// it, which is why the token is spelled out here rather than left implicit in
/// the error text.
pub const TRUNCATED_TWICE_MARKER: &str = "truncated_twice";

/// The short fixed instruction appended to the request that follows a cut-off
/// answer (§14.5, item 2).
///
/// [VANTA-WHY] A response stopped at `max_tokens` is never re-sent unchanged.
/// The runaway this fixes is a model that spends the ENTIRE output cap inside
/// its reasoning stream and never emits the JSON: re-sending the identical
/// prompt asks the identical question, and on the pinned temperature of the
/// auditor it reaches the identical prefix. The nudge changes the question
/// instead — answer directly, reason briefly, emit the JSON — so the retry has
/// something to change rather than another full-length generation to burn.
pub const TRUNCATION_NUDGE: &str = "Your previous reply was CUT OFF at the output limit before it \
finished, so it was discarded without being read. Answer the task directly: write at most three \
short sentences of reasoning, then emit the required JSON object immediately. Do not restate the \
task, do not repeat the reasoning, and do not explore alternatives — emit the JSON NOW.";

/// Append `TRUNCATION_NUDGE` as a trailing user turn.
///
/// [VANTA-WHY] An EXTRA turn rather than a rewrite of the existing prompt: every
/// call site builds its own messages through `build_messages`, and rewriting one
/// of them here would either need every site to hand its prompt over again or
/// would silently change the prefix a reader has already seen in the forensics.
/// A trailing instruction is additive, so attempt 1 of every call — the only
/// attempt a non-truncated run ever makes — is byte-identical to v0.2.
fn append_truncation_nudge(messages: &mut Vec<Value>) {
    messages.push(json!({
        "role": "user",
        "content": TRUNCATION_NUDGE,
    }));
}

/// The role a call site's truncations are counted under (§14.5).
///
/// [VANTA-WHY] The three roles the paper reports are `creator`, `attacker` and
/// `intent_audit`; the Attacker alone has four call sites
/// (`attacker_generate` / `_revise` / `_simplify`), and a run-away is a fact
/// about the AGENT, not about which of its calls it happened on. A site outside
/// the schedule (the M0 spike names its own) is counted as `other` rather than
/// folded into a role it does not belong to.
pub fn truncation_role(site: &str) -> &'static str {
    match site {
        "creator" => "creator",
        "attacker_generate" | "attacker_revise" | "attacker_simplify" => "attacker",
        "intent_audit" => "intent_audit",
        _ => "other",
    }
}

/// The mechanism strings the per-call `seed` record can carry (§14.23).
///
/// [VANTA-WHY] Kept as consts, not inlined: the value a row renders and the value a
/// test asserts have to be the same string, and the §14.21 knobs already set the
/// `backend:knob` spelling these follow.
pub const LLAMACPP_SEED_MECHANISM: &str = "llamacpp:seed";
pub const UNSET_SEED_MECHANISM: &str = "unset";

/// The mechanism a hosted `seed` records (§14.26).
///
/// [VANTA-WHY] `seed` is a documented Chat Completions parameter, and the
/// reference names `system_fingerprint` as its determinism companion — nothing in
/// the GPT-6 model docs says these models ignore it. So a seed that was inserted
/// into the body is recorded as reaching the model, and the `unsupported`
/// spelling stays reserved for the outcome it was written for.
///
/// [VANTA-WHY] §14.34 — the live probe (`scripts/openai_probe.py`) has now run,
/// and the provider ACCEPTED the key (`200`) on both `gpt-6-luna` and
/// `gpt-6.1-sol`. That is evidence the field TRAVELS, which is what this constant
/// claims — it is not evidence that one seed reproduces one completion, and the
/// probe cannot produce that evidence from a single sample. So the value stays
/// `openai:seed` (best effort, not a determinism claim); it becomes
/// `openai:seed-ignored` only if a same-seed repeat is later shown to differ,
/// which no run so far has tested.
pub const OPENAI_SEED_MECHANISM: &str = "openai:seed";

/// The seed POLICY every run from §14.23 on applies, recorded in `run.summary` as
/// `seed_policy` and in the manifest.
///
/// [VANTA-WHY] §14.23 — the offset is default-on, so a run's rows cannot say by
/// themselves whether the review rounds were independent draws or deterministic
/// duplicates. A row that predates this section has NO `seed_policy`, and the
/// section's own note says which batches those are: the value here is what makes
/// "before" and "after" impossible to pool by accident.
pub const SEED_POLICY: &str = "per_round_offset";

/// The `seed` one ROUND sends (§14.23).
///
/// [VANTA-WHY] §14.23 — the review loop re-asks the Attacker the same prompt once
/// per round, and with ONE pinned seed every round sampled identically: round 2 was
/// a deterministic duplicate of round 1, not a second, independent attack. Offsetting
/// by the round index makes rounds 2..n independent draws from the same pinned
/// baseline while keeping the run reproducible — the sequence is a function of
/// `VANTA_LLM_SEED` and the round index, and both are recorded.
///
/// [VANTA-WHY] Round 1 sends the configured seed UNCHANGED, so a run that never
/// reaches round 2 — and every single-round run — is byte-identical to one from
/// before this existed. `round` is `FORENSICS_ROUND`, the 1-based attempt index the
/// driver sets before any agent runs (`set_forensics_round`), and the same index the
/// `call.usage` row records; it is 0 outside the pipeline, and 0 is treated like
/// round 1 so the one-shot `post_chat` path is untouched.
fn seed_for_round(configured: Option<u64>, round: u32) -> Option<u64> {
    match configured {
        Some(seed) if round > 1 => Some(seed.wrapping_add(u64::from(round - 1))),
        other => other,
    }
}

/// The `seed` one attempt sends (§14.5).
///
/// [VANTA-WHY] A call's FIRST attempt always sends its round's seed unchanged — and
/// round 1's seed is the configured seed (§14.23) — so a call that never retries is
/// byte-identical to one from before this per-attempt offsetting existed.
///
/// [VANTA-WHY] A RETRY advances it by the attempt index, because the harness pins
/// ONE seed per round: with the same seed and the same prompt, even a
/// temperature-0.2 retry can come back identical — and be cut off at the
/// identical point — so the retry budget would be spent reproducing the same
/// prefix. Advancing the seed keeps the run reproducible (the sequence is a
/// function of the pinned seed, the round and the attempt number, and all three
/// are recorded) while letting the retry actually differ.
///
/// [VANTA-WHY] A call PINNED at temperature 0.0 is greedy: its retries are
/// deterministic whatever the seed says, so the seed is deliberately left alone
/// there. That is also the only place a truncated answer fails fast (see the
/// retry loop), so nothing depends on the seed moving for it.
fn seed_for_attempt(
    configured: Option<u64>,
    attempt: u32,
    deterministic_temperature: bool,
) -> Option<u64> {
    match configured {
        Some(seed) if attempt > 1 && !deterministic_temperature => {
            Some(seed.wrapping_add(u64::from(attempt - 1)))
        }
        other => other,
    }
}

/// Backoff before retry `attempt` (1-based): 500ms, 1s, 2s, 4s, 8s, 8s…
///
/// [VANTA-WHY] A retry that starts the instant the previous connection dropped
/// races the server: llama-server still holds the slot (and the KV cache) for a
/// moment after the socket closes, so an immediate retry piles a second
/// generation onto a cache that has not been freed — the "Context size has been
/// exceeded" cascade. The pause is what lets the cancel land.
const RETRY_BACKOFF_BASE_MS: u64 = 500;
const RETRY_BACKOFF_MAX_MS: u64 = 8_000;

async fn back_off(attempt: u32) {
    let shift = attempt.saturating_sub(1).min(4);
    let delay = Duration::from_millis((RETRY_BACKOFF_BASE_MS << shift).min(RETRY_BACKOFF_MAX_MS));
    debug!(
        attempt,
        delay_ms = delay.as_millis() as u64,
        "backing off before the next attempt"
    );
    tokio::time::sleep(delay).await;
}

/// Why a provider refused a call while busy (§14.30, item 6).
///
/// [VANTA-WHY] The two are distinguished because the run's report has to be able
/// to say which happened — a rate limit is this run going too fast, an
/// unavailable `flex` tier is the cheap capacity being gone — and because the
/// second is the failure mode §14.30's tier switch (`VANTA_OPENAI_SERVICE_TIER`,
/// item 7) makes reachable. Both arrive as HTTP 429; only the body says which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderRetryClass {
    /// A plain rate limit: too many requests per unit time.
    RateLimit,
    /// `429 Resource Unavailable` — the service tier this request asked for has
    /// no capacity right now. The documented failure of `service_tier: flex`.
    ResourceUnavailable,
}

impl ProviderRetryClass {
    /// The class as the record spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::RateLimit => "rate_limit",
            Self::ResourceUnavailable => "resource_unavailable",
        }
    }
}

/// How many times one attempt re-sends a request a busy provider refused (§14.30).
///
/// [VANTA-WHY] Its own budget, separate from `max_attempts`: those attempts exist
/// to let the MODEL answer differently, and a 429 is not an answer. Charging a
/// throttle to the model's budget would let a busy minute turn a call that was
/// never refused by the model into "the model failed twice". Four, with the
/// ladder below, is under a minute of waiting — long enough to ride out the
/// documented `flex` capacity blip, short enough that a provider that is really
/// down still fails the call rather than the run's deadline (§14.15).
pub const PROVIDER_RETRY_MAX: u32 = 4;
const PROVIDER_RETRY_BACKOFF_BASE_MS: u64 = 1_000;
const PROVIDER_RETRY_BACKOFF_MAX_MS: u64 = 30_000;

/// The backoff before provider retry `retry` (1-based): 1s, 2s, 4s, 8s, 16s…
///
/// [VANTA-WHY] A LADDER, not a fixed pause, and a longer base than `back_off`'s
/// 500 ms: a rate limit is a window that has to ROLL OVER, so retrying at the
/// model-retry cadence would spend the whole budget inside one window and come
/// back to the same 429. The `flex` blip is the same shape at a coarser scale.
async fn provider_back_off(retry: u32) {
    let shift = retry.saturating_sub(1).min(5);
    let delay = Duration::from_millis(
        (PROVIDER_RETRY_BACKOFF_BASE_MS << shift).min(PROVIDER_RETRY_BACKOFF_MAX_MS),
    );
    debug!(
        retry,
        delay_ms = delay.as_millis() as u64,
        "backing off before re-sending to a busy provider"
    );
    tokio::time::sleep(delay).await;
}

/// The HTTP status in a `post_chat_meta` transport error, when it has one.
///
/// `None` for the two errors that never reached a status line — the header
/// timeout and the connection failure — which are transport faults, not the
/// provider declining, and must keep their existing treatment.
fn http_status_code(err: &str) -> Option<u16> {
    err.strip_prefix("HTTP ERROR ")?.split_whitespace().next()?.parse().ok()
}

/// Whether a transport error is the PROVIDER asking to be asked again later
/// (§14.30).
///
/// [VANTA-WHY] Narrower than "any 5xx" ON PURPOSE. llama-server answers a
/// request that does not fit its KV cache with a 5xx, and a retry of that is a
/// second full generation onto the same full cache — the "Context size has been
/// exceeded" cascade `back_off` exists to avoid. A 429 is the one status that
/// means "the request was fine, the service was busy", which is why only it is
/// retried here. `insufficient_quota` is excluded because it arrives as a 429
/// too and is NOT transient: the account is out of credit, and four retries
/// would only add a minute to a failure that waits for a human.
pub fn provider_retry_class(err: &str) -> Option<ProviderRetryClass> {
    if http_status_code(err)? != 429 {
        return None;
    }
    if err.to_ascii_lowercase().contains("insufficient_quota") {
        return None;
    }
    if err.to_ascii_lowercase().contains("resource unavailable") {
        Some(ProviderRetryClass::ResourceUnavailable)
    } else {
        Some(ProviderRetryClass::RateLimit)
    }
}

/// Whether a transport error is the PROVIDER asking to be asked again later,
/// and this attempt still has budget to do so (§14.30).
///
/// [VANTA-WHY] The CLASS and the BUDGET are one decision, in one function,
/// because they are checked at the same instant and a test must be able to
/// assert the bound without waiting out the ladder: `Some(class)` is "back off
/// and re-send", `None` is "return this error as the failure". Keeping the
/// comparison here rather than inline in the loop is what makes
/// "the ladder stops at exactly `PROVIDER_RETRY_MAX`" a statement a unit test
/// can check.
fn provider_retry_decision(err: &str, retries_so_far: u32) -> Option<ProviderRetryClass> {
    match provider_retry_class(err) {
        Some(class) if retries_so_far < PROVIDER_RETRY_MAX => Some(class),
        _ => None,
    }
}

/// POST the body, re-sending it while the provider is busy (§14.30).
///
/// Returns the successful response beside how many times the request was re-sent
/// for it. Every non-429 failure — including a 429 past `PROVIDER_RETRY_MAX` —
/// comes back as the same error the single-shot version produced, so the caller's
/// existing handling (and the failure text a reader has seen before) is unchanged.
async fn send_with_provider_retries(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    url: &str,
    full_body: &Value,
) -> Result<(reqwest::Response, u32), Box<dyn std::error::Error + Send + Sync>> {
    let mut retries = 0_u32;
    loop {
        // The idle timeout covers the WAIT FOR HEADERS too: a server that accepts
        // the connection and then says nothing is as dead as one that stops
        // mid-stream. Dropping this future closes the socket.
        let send_future = client.post(url).json(full_body).send();
        let resp = match tokio::time::timeout(
            Duration::from_secs(cfg.idle_timeout_secs),
            send_future,
        )
        .await
        {
            Err(_elapsed) => {
                return Err(format!(
                    "HTTP POST to {url} timed out: no response headers for {}s (VANTA_LLM_IDLE_TIMEOUT_SECS); connection dropped",
                    cfg.idle_timeout_secs
                )
                .into());
            }
            Ok(Err(e)) => return Err(format!("HTTP POST to {url} FAILED: {e}").into()),
            Ok(Ok(resp)) => resp,
        };

        let status = resp.status();
        if status.is_success() {
            return Ok((resp, retries));
        }
        let code = status.as_u16();
        let txt = resp.text().await.unwrap_or_else(|_| String::from("<could not read body>"));
        // [VANTA-WHY] The error body is the provider's text, and a provider is
        // free to quote the credential it just rejected back at us
        // ("Incorrect API key provided: sk-…"). Only the body is shown: the
        // request headers are never formatted into this message, and the
        // Authorization header is built with `set_sensitive(true)` besides, so
        // reqwest will not log it either. `redact_secret` closes the remaining
        // hole — a key echoed *inside* the body.
        //
        // `chars().take(300)` rather than `&txt[..300]`: the byte slice panics
        // when byte 300 lands inside a multi-byte character, which turns "the
        // provider returned an error" into "Vanta crashed". Taken BEFORE the
        // classification below so the string the classifier reads is the one a
        // reader would see.
        let redacted = redact_secret(&txt);
        let snippet: String = redacted.chars().take(300).collect();
        let err = format!("HTTP ERROR {code} from {url}. Body snippet: {snippet}");

        match provider_retry_decision(&err, retries) {
            Some(class) => {
                retries += 1;
                note_provider_retry(class);
                warn!(
                    code,
                    class = class.name(),
                    retries,
                    max = PROVIDER_RETRY_MAX,
                    "the provider refused this call while busy — backing off and re-sending the \
                     SAME request (this is NOT a model retry)"
                );
                provider_back_off(retries).await;
                continue;
            }
            None => return Err(err.into()),
        }
    }
}

pub async fn structured_call_with_retry<Fm, Fp, T>(
    client: &reqwest::Client,
    cfg: &LlmConfig,
    max_attempts: u32,
    forensics_tag: &str,
    // Call-site name for the token-accounting row; forwarded to `post_chat`.
    site: &str,
    mut build_messages: Fm,
    extras_for_attempt: Value,
    mut parse_into_typed: Fp,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>>
where
    Fm: FnMut(u32) -> Vec<Value>,
    Fp: FnMut(&str) -> Result<T, String>,
    T: std::fmt::Debug + serde::Serialize,
{
    let mut last_err: Option<String> = None;
    // [VANTA-WHY] §14.5 — whether the most recent completed attempt was cut off
    // at `max_tokens`. Kept so the FINAL failure can say so: every attempt that
    // ends this way parses as nothing and looks, from the parse error alone,
    // exactly like an answer the model got wrong. Only the server's own
    // `finish_reason` separates the two, and the two have opposite remedies.
    let mut last_was_truncated = false;
    // [VANTA-WHY] §14.5 item 2 — how many completed attempts the server cut off
    // at `max_tokens`. The FIRST one earns the brief re-ask below; the SECOND one
    // is the call saying it cannot answer inside its cap, and ends the call.
    let mut truncations = 0_u32;
    // Whether the next request must carry `TRUNCATION_NUDGE`. Set by a cut-off
    // and never cleared: a call site that has once spent its whole cap on
    // reasoning gets the brief instruction on every later attempt too.
    let mut nudge_next = false;
    // Whether the call is giving up BECAUSE it was cut off twice (rather than
    // because its attempt budget ran out), so the final failure can say which.
    let mut truncated_twice = false;
    // Attempts that actually reached the parse step, so the failure can report
    // how many were spent rather than how many were allowed: a call that stops
    // early (a cut-off answer) did not fail `max_attempts` times.
    let mut attempts_made = 0_u32;

    // [VANTA-WHY] §14.5 — whether this call is PINNED at temperature 0.0, and
    // therefore answers identically on every attempt. It is read from the
    // request's own `temperature`, because that is what the server samples with:
    // every M3 agent passes its pin through `extra_fields`, which wins over
    // `post_chat`'s own body default, so a caller that sends no extras samples at
    // that default (0.2) and is NOT deterministic.
    //
    // [VANTA-WHY] This is still the scope of `seed_for_attempt`'s "leave the seed
    // alone" rule: a greedy retry is deterministic whatever the seed says, and
    // the brief re-ask below is what changes that question anyway.
    let deterministic_temperature = extras_for_attempt
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(0.2)
        <= 0.0;

    for attempt in 1..=max_attempts {
        // [VANTA-WHY] §14.15 — the run's own wall clock, checked before EVERY
        // request. This is the one place every structured call site passes
        // through, so it is the one place a check is a check before all of them —
        // including the calls BATE's revision loop and the Diagnostician's
        // Unknown path make from inside loops the driver cannot see. Refusing
        // here costs a string rather than a ten-minute generation, and the error
        // carries `DEADLINE_EXCEEDED_MARKER` so the driver can end the run as
        // `budget_exhausted` rather than blaming the machine.
        if deadline_past() {
            warn!(
                site,
                attempt,
                "the run's wall-clock deadline has passed — refusing to start this call \
                 ({DEADLINE_EXCEEDED_MARKER})"
            );
            return Err(format!(
                "the run's wall-clock deadline (plans/bugsinpy.md §14.15) passed before this \
                 `{site}` call could start, so it was never made ({DEADLINE_EXCEEDED_MARKER})"
            )
            .into());
        }
        let mut messages = build_messages(attempt);
        attempts_made = attempt;
        if nudge_next {
            // [VANTA-WHY] §14.5 item 2 — a cut-off answer is never re-sent
            // unchanged. The nudge makes the retry a DIFFERENT question, which is
            // the only kind that can come back different on a pinned temperature.
            append_truncation_nudge(&mut messages);
        }
        if attempt > 1 {
            warn!(attempt, max_attempts, "retrying structured call");
        }
        // [VANTA-WHY] §14.23 — the ROUND offset is applied first and the ATTEMPT
        // offset on top of it, so the two sequences compose: round 1's first attempt
        // is the pinned seed unchanged (byte-identical to before either existed),
        // round 2 starts one past it, and a retry within either round advances by the
        // retry index. `FORENSICS_ROUND` is the round the driver set for this attempt
        // — the same index the `call.usage` row carries.
        let attempt_seed = seed_for_attempt(
            seed_for_round(cfg.seed, FORENSICS_ROUND.load(Ordering::Relaxed)),
            attempt,
            deterministic_temperature,
        );

        let (raw, finish_reason) = match post_chat_meta(
            client,
            cfg,
            messages,
            extras_for_attempt.clone(),
            site,
            attempt_seed,
            // The retry index, so the row can say which attempt this was (§14.12).
            attempt,
        )
        .await
        {
            Ok(pair) => pair,
            Err(http_e) => {
                let http_e = http_e.to_string();
                // [VANTA-WHY] A prompt that cannot fit the context window fails
                // IDENTICALLY on every attempt — it is decided by arithmetic
                // before a byte leaves the process. Retrying it three times
                // cannot change the outcome; it only pretended the failure was
                // transient and delayed the run. Fail once, loudly, with what to
                // change.
                if http_e.contains(DETERMINISTIC_CONTEXT_PREFIX) {
                    return Err(format!(
                        "{http_e}\nThis is a deterministic local failure — the prompt does not fit \
                         the model's context window — so it is NOT retried. Raise the server's n_ctx \
                         (or VANTA_LLM_CONTEXT_TOKENS to match it), or shrink the prompt."
                    )
                    .into());
                }
                let msg = format!("attempt {attempt} HTTP err: {http_e}");
                warn!(%msg);
                last_err = Some(msg);
                last_was_truncated = false;
                if attempt < max_attempts {
                    back_off(attempt).await;
                }
                continue;
            }
        };
        // The stream completed, so it has an answer to judge: `length` here means
        // the answer was CUT OFF, not that it was wrong.
        last_was_truncated = finish_reason.as_deref() == Some("length");

        let type_name = std::any::type_name::<T>().split("::").last().unwrap_or("T");

        // [FORENSICS] Write the raw model response to disk before parsing,
        // so we can inspect exactly what the model returned if parsing fails.
        // A cut-off answer is written here too: it is the evidence a runaway
        // report is built from, and it must be on disk even though nothing below
        // reads it.
        write_forensics(type_name, forensics_tag, attempt, "raw_response.txt", &raw);

        // [VANTA-WHY] §14.5 item 2 — a reply the server stopped at `max_tokens` is
        // NEVER handed to the parser. It is a prefix of the intended answer, so a
        // parse that happens to succeed (the JSON landed before the cap did) would
        // silently admit a half-written object and a parse that fails would bill
        // the model for a mistake it did not make. The first cut-off earns a
        // brief re-ask; the second ends the call with the site's ordinary failure
        // value, which is what every caller already handles conservatively.
        if last_was_truncated {
            truncations += 1;
            warn!(
                attempt,
                truncations,
                "the server cut this answer off at max_tokens — it is NOT parsed, and the next \
                 request carries a brief `answer directly` instruction"
            );
            last_err = Some(format!(
                "attempt {attempt} TRUNCATED: the server stopped this answer at the request's \
                 `max_tokens` cap (finish_reason=length), so the reply is a prefix of the intended \
                 one and was not parsed"
            ));
            if truncations >= 2 {
                // [VANTA-WHY] §14.5 item 2 — cut off twice, with a shorter
                // instruction in between. Re-sending the identical request would
                // spend another full-length generation to reach the same prefix,
                // so the call gives up instead: this is the conservative failure
                // value the step already has, and it is the same one everywhere.
                truncated_twice = true;
                note_truncated_twice();
                warn!(
                    site,
                    attempt,
                    "this call was cut off TWICE — giving up with the site's ordinary failure \
                     value ({TRUNCATED_TWICE_MARKER})"
                );
                break;
            }
            nudge_next = true;
            if attempt < max_attempts {
                back_off(attempt).await;
            }
            continue;
        }

        match parse_into_typed(&raw) {
            Ok(typed) => {
                debug!(attempt, type_name, "parsed typed response");
                // [FORENSICS] Write the successfully parsed typed result as JSON.
                if let Ok(json) = serde_json::to_string_pretty(&serde_json::to_value(&typed).unwrap_or(json!(null))) {
                    write_forensics(type_name, forensics_tag, attempt, "parsed.json", &json);
                }
                return Ok(typed);
            }
            Err(parse_err) => {
                let preview: String = raw.chars().take(240).collect();
                last_err = Some(format!(
                    "attempt {attempt} PARSE ERR: {}\n  model-preview: {preview}",
                    parse_err
                ));
                // [VANTA-WHY] §14.5 — a malformed answer that was NOT cut off is
                // the model's own mistake, not the cap's: it kept its retries, in
                // the v0.2 order, with the seed advanced each time (see
                // `seed_for_attempt`) so a pinned seed cannot reproduce the same
                // mistake forever. A CUT-OFF answer never reaches this arm at all
                // (it is handled above), so the remedy the two need — a smaller
                // question versus a correct one — can no longer be confused.
                if attempt < max_attempts {
                    back_off(attempt).await;
                }
            }
        }
    }

    let mut message = format!(
        "structured_call_with_retry {}/{} attempts FAILED. Last:\n{}",
        attempts_made,
        max_attempts,
        last_err.as_deref().unwrap_or("(no details)")
    );
    // [VANTA-WHY] §14.5 — say WHY the last attempt failed, when the server said.
    // Without this a truncated answer is indistinguishable from an answer the
    // model got wrong, and the two have opposite remedies: this one is fixed by
    // asking a smaller question, which is exactly what the batched audit can do
    // (one test per call) and must do, instead of losing the whole round.
    if last_was_truncated {
        let tail = if truncated_twice {
            "The answer was cut off TWICE — including once after being re-asked with a short \
             `answer directly, keep the reasoning brief, emit the JSON` instruction — so the \
             identical request is NOT re-sent: this call takes the conservative failure value its \
             call site already has for a failed step. Ask for less, or raise the cap."
                .to_string()
        } else if nudge_next {
            "The answer was cut off and the brief re-ask was NOT made: this call's attempt budget \
             is one. Ask for less, or raise the cap."
                .to_string()
        } else {
            "The answer was cut off on the attempt that ended the call. Ask for less, or raise \
             the cap."
                .to_string()
        };
        let marker = if truncated_twice {
            format!(" {TRUNCATED_TWICE_MARKER}")
        } else {
            String::new()
        };
        message.push_str(&format!(
            "\n{TRUNCATED_OUTPUT_PREFIX}: the server stopped this answer at the request's \
             `max_tokens` cap, so the reply is a prefix of the intended one and could not be \
             parsed.{marker} {tail}"
        ));
    }
    Err(message.into())
}

#[cfg(test)]
mod _cleaning_tests {
    use super::{
        context_budget_exceeded, conservative_token_estimate, extract_first_json_object,
        extract_json_candidate, strip_markdown_fences,
    };

    #[test]
    fn strips_all_fence_variants() {
        assert_eq!(strip_markdown_fences("```json\n{}\n```"), "{}");
        assert_eq!(strip_markdown_fences("```{}\n```"), "{}");
        assert_eq!(strip_markdown_fences("{}\n"), "{}");
    }

    #[test]
    fn extracts_plain_json() {
        assert_eq!(extract_first_json_object("{\"a\":1}"), Some("{\"a\":1}"));
    }

    #[test]
    fn extracts_json_after_prose() {
        let s = "Sure! Here you go:\n{\"x\":\"hello\",\"y\":[1,2]}";
        assert_eq!(extract_first_json_object(s), Some("{\"x\":\"hello\",\"y\":[1,2]}"));
    }

    #[test]
    fn pipeline_handles_prose_then_fenced_json() {
        let s = "Okay let's see...\n```json\n{\"ok\":true}\n```\nHope that helps";
        let r = extract_json_candidate(s).unwrap();
        assert_eq!(r, "{\"ok\":true}");
    }

    #[test]
    fn pipeline_handles_plain_json_no_wrapper() {
        let s = "{\"a\": 1, \"b\": [1,2,3]}";
        let r = extract_json_candidate(s).unwrap();
        assert_eq!(r, s);
    }

    #[test]
    fn pipeline_rejects_non_object() {
        assert!(extract_json_candidate("just text").is_err());
    }

    #[test]
    fn conservative_token_estimate_grows_with_input() {
        assert!(
            conservative_token_estimate("a b c d e f g h")
                > conservative_token_estimate("a b")
        );
    }

    #[test]
    fn context_budget_exceeded_counts_prompt_output_and_overhead() {
        assert!(context_budget_exceeded(60_000, 6_000, 65_536));
        assert!(!context_budget_exceeded(20_000, 2_000, 65_536));
    }

    // ---- Fix 4 regression: round-numbered forensics do not overwrite ----

    /// The old filename carried ONLY the structured-call retry index (1..=3),
    /// which restarts every round — so a 2-round run wrote round 2 over round 1.
    /// Both rounds' files must survive now.
    #[test]
    fn forensics_from_two_rounds_both_exist() {
        use super::{write_forensics_to};
        use std::collections::BTreeSet;

        let dir = tempfile::tempdir().unwrap();
        for round in 1..=2u32 {
            write_forensics_to(
                dir.path(),
                "IntentVerdict",
                "test_print_responses",
                round,
                1,
                "parsed.json",
                "{}",
            );
        }

        let names: BTreeSet<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names.len(), 2, "a later round overwrote an earlier one: {names:?}");
        assert!(
            names.iter().any(|n| n.contains("round_1_attempt_1")),
            "{names:?}"
        );
        assert!(
            names.iter().any(|n| n.contains("round_2_attempt_1")),
            "{names:?}"
        );
        assert!(
            names.iter().all(|n| n.contains("test_print_responses")),
            "intent verdicts must carry the test name: {names:?}"
        );
    }

    #[test]
    fn forensics_tag_is_sanitized() {
        assert_eq!(super::sanitize_tag("tests/x.py::test[1]"), "tests_x.py__test_1_");
    }

    /// [VANTA-WHY] e (§14.10) — the batched intent audit writes its raw replies
    /// as
    /// `IntentVerdict>_intent_audit_batch_round_1_attempt_1_raw_response.txt`. The
    /// `>` is not in the tag; it is the last `::`-segment of
    /// `std::any::type_name::<Vec<IntentVerdict>>()`, which keeps the closing
    /// angle bracket of the generic. A `>` in a filename is shell-hostile (it
    /// redirects), so the same rule that cleans a tag must clean the type name.
    #[test]
    fn forensics_type_name_is_sanitized() {
        use super::write_forensics_to;

        let dir = tempfile::tempdir().unwrap();
        write_forensics_to(
            dir.path(),
            "IntentVerdict>",
            "intent_audit_batch",
            1,
            1,
            "raw_response.txt",
            "{}",
        );

        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names.len(), 1, "one file per call: {names:?}");
        let name = &names[0];
        assert!(
            !name.contains('>'),
            "a forensics filename may not carry a '>': {name}"
        );
        assert_eq!(
            name,
            "IntentVerdict__intent_audit_batch_round_1_attempt_1_raw_response.txt",
            "the '>' must be replaced by '_', not dropped or left in place"
        );
    }
}

// ============================================================================
// Streaming / liveness tests
// ============================================================================
//
// [VANTA-WHY] These run on tokio's PAUSED clock (`start_paused = true`), which
// is the only way to assert the thing the incident was about: a stream that
// keeps producing tokens for 700 seconds must survive, while one that stops
// producing must be abandoned. Real time would make both untestable. The stub
// here is an in-memory byte stream rather than a socket, because the clock is
// what is under test, not the transport.
#[cfg(test)]
mod _streaming_tests {
    use super::{StreamedCompletion, assemble_sse_stream, consume_sse_event};
    use futures::Stream;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Context, Poll};
    use std::time::Duration;

    type Chunks = Pin<Box<dyn Stream<Item = Result<Vec<u8>, String>> + Send>>;

    /// One SSE data frame carrying the single token `x`, exactly as
    /// llama-server emits it under `stream: true`.
    fn token_event() -> Vec<u8> {
        b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n".to_vec()
    }

    /// A stream that emits one token and then never produces anything again —
    /// the stall the old total-timeout could not tell apart from slow work. It
    /// records whether it was DROPPED, because that drop is the cancel: hyper
    /// abandons the response body and closes the socket.
    struct StallingStream {
        sent: bool,
        dropped: Arc<AtomicBool>,
    }

    impl Stream for StallingStream {
        type Item = Result<Vec<u8>, String>;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            if !self.sent {
                self.sent = true;
                return Poll::Ready(Some(Ok(token_event())));
            }
            Poll::Pending
        }
    }

    impl Drop for StallingStream {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_slow_but_steady_stream_completes_past_six_hundred_simulated_seconds() {
        const TOKENS: u32 = 700;
        let stream: Chunks = Box::pin(futures::stream::unfold(0_u32, |i| async move {
            if i >= TOKENS {
                return None;
            }
            // One token per simulated second: the whole generation spans 700
            // SIMULATED seconds, well past the 600s total timeout that used to
            // kill exactly this call, while no individual gap ever approaches
            // the 180s idle timeout.
            tokio::time::sleep(Duration::from_secs(1)).await;
            Some((Ok(token_event()), i + 1))
        }));

        let out = assemble_sse_stream(
            stream,
            Duration::from_secs(180),
            Duration::from_secs(3600),
        )
        .await
        .expect("a stream that keeps producing tokens must never be abandoned");

        assert_eq!(
            out.content,
            "x".repeat(TOKENS as usize),
            "every delta must be assembled into the response"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_stalled_stream_fires_the_idle_timeout_and_the_connection_is_dropped() {
        let dropped = Arc::new(AtomicBool::new(false));
        let stream: Chunks = Box::pin(StallingStream {
            sent: false,
            dropped: Arc::clone(&dropped),
        });

        let err = assemble_sse_stream(
            stream,
            Duration::from_secs(180),
            Duration::from_secs(3600),
        )
        .await
        .expect_err("a stream silent for the idle timeout must be abandoned");

        assert!(err.contains("stalled"), "got: {err}");
        assert!(
            dropped.load(Ordering::SeqCst),
            "the byte stream must be dropped before the error is returned — that drop is what \
             closes the socket and cancels the server's generation"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_overall_backstop_ends_a_stream_that_never_stalls() {
        // A server that emits a token just often enough to defeat the idle
        // timeout still must not be able to hold the run forever.
        let stream: Chunks = Box::pin(futures::stream::unfold(0_u32, |i| async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Some((Ok(token_event()), i + 1))
        }));

        let err = assemble_sse_stream(
            stream,
            Duration::from_secs(180),
            Duration::from_secs(120),
        )
        .await
        .expect_err("the overall backstop must end the call");

        assert!(err.contains("overall backstop"), "got: {err}");
    }

    #[test]
    fn an_error_frame_inside_the_stream_is_an_error_not_an_empty_answer() {
        let mut out = StreamedCompletion::default();
        let err = consume_sse_event(
            "data: {\"error\":{\"message\":\"Context size has been exceeded\"}}",
            &mut out,
        )
        .expect_err("a server-reported failure must not be read as an empty completion");
        assert!(err.contains("Context size has been exceeded"), "got: {err}");
    }

    #[test]
    fn deltas_usage_and_done_are_read_from_the_stream() {
        let mut out = StreamedCompletion::default();
        assert!(
            !consume_sse_event(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
                &mut out
            )
            .unwrap()
        );
        assert!(
            !consume_sse_event(
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"total_tokens\":22}}",
                &mut out
            )
            .unwrap()
        );
        assert!(consume_sse_event("data: [DONE]", &mut out).unwrap());
        assert_eq!(out.content, "hi");
        assert_eq!(out.prompt_tokens, Some(11));
        assert_eq!(out.total_tokens, Some(22));
    }
}

// ============================================================================
// §14.15 — the run deadline's own switch
// ============================================================================
//
// [VANTA-WHY] The deadline is a process-global (there is one run per process), so
// this module owns it end to end and leaves it disarmed. Nothing else arms it or
// reads it outside `structured_call_with_retry`, which no unit test drives — it
// needs a server — so the global is this module's alone even under the parallel
// test harness.
#[cfg(test)]
mod _deadline_tests {
    use super::{deadline_past, set_deadline_secs};

    #[test]
    fn an_unarmed_deadline_never_passes_and_an_armed_one_passes_when_it_is_due() {
        // The default, and the behaviour of every run before §14.15: no clock.
        set_deadline_secs(None);
        assert!(!deadline_past(), "an unarmed deadline must never end a run");

        // Armed an hour out, a call may still start.
        set_deadline_secs(Some(3600));
        assert!(!deadline_past(), "a deadline an hour away is not here yet");

        // Armed zero seconds out it is due the instant it is set, so the very next
        // call is refused. This is the point of the check being `>=` and not `>`:
        // "no clock left" must refuse, not permit one more call.
        set_deadline_secs(Some(0));
        assert!(deadline_past(), "a deadline set at `now` must already be due");

        // Disarming is total — a cleared deadline is a run with no clock, which is
        // the state this test must leave behind.
        set_deadline_secs(None);
        assert!(!deadline_past(), "disarming must return the run to 'no clock'");
    }
}

// ============================================================================
// §14.16 — the presence-penalty override's own switch
// ============================================================================
//
// [VANTA-WHY] The override is a process-global, so this module owns it end to end
// and leaves it cleared. The one hazard is a parallel test that builds a request
// body: `with_sampling` is called at the five structured call sites, all of which
// need a live server, so no unit test reaches them. The window is the length of
// this test either way.
#[cfg(test)]
mod _presence_penalty_tests {
    use super::{
        SAMPLING_SCHEDULE, SamplingPreset, sampling_pins_text, set_presence_penalty,
        with_sampling_pins,
    };
    use serde_json::json;

    /// The pinned `presence_penalty` for `site`, or `None` when the body carries
    /// no such key — which is the claim "the pins were left alone".
    fn pinned(preset: SamplingPreset, site: &str) -> Option<String> {
        sampling_pins_text(preset, site)
            .into_iter()
            .find(|(key, _)| *key == "presence_penalty")
            .map(|(_, value)| value)
    }

    #[test]
    fn zero_and_unset_pin_nothing_and_a_non_zero_pins_every_site() {
        // The default, and every run before §14.16.
        set_presence_penalty(None);
        for (site, ..) in SAMPLING_SCHEDULE {
            assert_eq!(
                pinned(SamplingPreset::Pinned, site),
                None,
                "an unset override must leave a pinned body byte-identical ({site})"
            );
            assert!(
                with_sampling_pins(json!({}), SamplingPreset::Pinned, site)
                    .get("presence_penalty")
                    .is_none(),
                "an unset override must insert no `presence_penalty` key ({site})"
            );
        }

        // Zero is the measurement arm's control: it must pin nothing either, or
        // the control's bodies would differ from the baseline's by one key.
        set_presence_penalty(Some(0.0));
        for (site, ..) in SAMPLING_SCHEDULE {
            assert_eq!(
                pinned(SamplingPreset::Pinned, site),
                None,
                "a zero override pins nothing — `0.0` is the server's own default \
                 ({site})"
            );
        }

        // A non-zero value pins EVERY site, the auditor included: the repetition
        // happens in the generative roles and the auditor reads their output, so
        // an override that skipped a site would be a different experiment.
        set_presence_penalty(Some(1.0));
        for (site, ..) in SAMPLING_SCHEDULE {
            assert_eq!(
                pinned(SamplingPreset::Pinned, site),
                Some("1.0".to_string()),
                "a non-zero override must reach every site ({site})"
            );
            assert_eq!(
                with_sampling_pins(json!({}), SamplingPreset::Pinned, site)["presence_penalty"],
                json!(1.0),
                "the override must reach the request body itself ({site})"
            );
        }

        // Under `coding` the override REPLACES the preset's own 0.0 rather than
        // adding a second key: two `presence_penalty` keys in one object is the
        // later one silently winning, and a record that named both would be a
        // schedule nobody sent.
        for (site, ..) in SAMPLING_SCHEDULE {
            let text = sampling_pins_text(SamplingPreset::Coding, site);
            assert_eq!(
                text.iter().filter(|(key, _)| *key == "presence_penalty").count(),
                1,
                "the override must not duplicate the coding preset's key ({site})"
            );
            assert_eq!(
                pinned(SamplingPreset::Coding, site),
                Some("1.0".to_string()),
                "the override must win over the coding preset's 0.0 ({site})"
            );
            // The preset's OTHER keys are untouched — this changes one parameter,
            // it is not a third preset.
            assert_eq!(
                text.iter()
                    .find(|(key, _)| *key == "temperature")
                    .map(|(_, value)| value.as_str()),
                Some("0.6"),
                "the override must not move the preset's temperature ({site})"
            );
        }

        // A cleared override leaves the coding preset's own 0.0 in place, which is
        // the state this test must leave behind.
        set_presence_penalty(None);
        assert_eq!(
            pinned(SamplingPreset::Coding, "creator"),
            Some("0.0".to_string()),
            "coding declares its own `presence_penalty`, override or not"
        );
        assert_eq!(pinned(SamplingPreset::Pinned, "creator"), None);
    }
}

// ============================================================================
// §14.20 — the per-site output cap's own switch
// ============================================================================
//
// [VANTA-WHY] One test, not four, for the reason the block above is one test:
// `MAX_TOKENS_OVERRIDES` is a process-global and the harness runs these in
// parallel threads, so two tests that each set and clear it race — the clearing
// one wins the `max_tokens` slot out from under the asserting one and the
// failure reads as a broken override rather than a broken test. The override is
// owned here end to end and left cleared.
#[cfg(test)]
mod _max_tokens_override_tests {
    use super::{
        SAMPLING_SCHEDULE, SamplingPreset, max_tokens_env_var, max_tokens_overrides_text,
        sampling_pins_text, set_max_tokens_overrides, with_sampling_pins,
    };
    use serde_json::json;

    /// The `max_tokens` a body would carry for `site`.
    fn cap(preset: SamplingPreset, site: &str) -> Option<u32> {
        sampling_pins_text(preset, site)
            .into_iter()
            .find(|(key, _)| *key == "max_tokens")
            .map(|(_, value)| value.parse().expect("a max_tokens pin is an integer"))
    }

    #[test]
    fn the_output_cap_override_replaces_one_slot_and_leaves_the_rest_on_the_schedule() {
        // The switch is named after the site it caps, so it cannot be spelled for
        // a site the pin table would panic on.
        assert_eq!(
            max_tokens_env_var("attacker_revise"),
            "VANTA_MAX_TOKENS_ATTACKER_REVISE"
        );
        assert_eq!(max_tokens_env_var("creator"), "VANTA_MAX_TOKENS_CREATOR");

        // (1) The default and every run before §14.20: nothing asked for, so every
        //     body and every record stays on the schedule, byte for byte.
        set_max_tokens_overrides(Vec::new());
        for (site, _, scheduled) in SAMPLING_SCHEDULE {
            assert_eq!(
                cap(SamplingPreset::Pinned, site),
                Some(*scheduled),
                "an unset override must leave the schedule's cap ({site})"
            );
            assert_eq!(
                with_sampling_pins(json!({}), SamplingPreset::Pinned, site)["max_tokens"],
                json!(*scheduled),
                "the unset body must carry the schedule's cap ({site})"
            );
        }
        assert_eq!(max_tokens_overrides_text(), "unset");

        // (2) The arm: two sites capped. The override REPLACES the `max_tokens`
        //     slot rather than appending — a duplicate key is silently won by
        //     whichever the object kept, and a record naming both would describe a
        //     cap nobody sent — and it wins over the coding preset too.
        set_max_tokens_overrides(vec![
            ("attacker_revise", 12288),
            ("attacker_simplify", 12288),
        ]);
        for preset in [SamplingPreset::Pinned, SamplingPreset::Coding] {
            for site in ["attacker_revise", "attacker_simplify"] {
                let text = sampling_pins_text(preset, site);
                assert_eq!(
                    text.iter().filter(|(key, _)| *key == "max_tokens").count(),
                    1,
                    "the override must not duplicate `max_tokens` ({site})"
                );
                assert_eq!(cap(preset, site), Some(12288), "({site})");
                assert_eq!(
                    with_sampling_pins(json!({}), preset, site)["max_tokens"],
                    json!(12288),
                    "the override must reach the request body itself ({site})"
                );
            }
        }

        // The three sites the switch does not name keep the schedule, so an arm
        // that caps two sites leaves the others alone.
        assert_eq!(cap(SamplingPreset::Pinned, "creator"), Some(32768));
        assert_eq!(cap(SamplingPreset::Pinned, "attacker_generate"), Some(32768));
        assert_eq!(cap(SamplingPreset::Pinned, "intent_audit"), Some(8192));

        // The run-level line names what was ASKED FOR, in schedule order, which is
        // what identifies the arm in a batch table.
        assert_eq!(
            max_tokens_overrides_text(),
            "attacker_revise=12288,attacker_simplify=12288"
        );

        // (3) §14.20 must not be revise-only: the PySnooper smoke's round-2
        //     `attacker_generate` spent its whole 32,768-token cap without
        //     answering, so the first generation is addressable too.
        set_max_tokens_overrides(vec![("attacker_generate", 16384)]);
        assert_eq!(cap(SamplingPreset::Pinned, "attacker_generate"), Some(16384));
        assert_eq!(
            with_sampling_pins(json!({}), SamplingPreset::Pinned, "attacker_generate")["max_tokens"],
            json!(16384)
        );
        assert_eq!(max_tokens_overrides_text(), "attacker_generate=16384");

        // The cleared state this test has to leave behind.
        set_max_tokens_overrides(Vec::new());
        assert_eq!(cap(SamplingPreset::Pinned, "attacker_revise"), Some(32768));
        assert_eq!(max_tokens_overrides_text(), "unset");
    }
}

// ============================================================================
// REASONING CONTROL (§14.21) — tests
// ============================================================================
//
// [VANTA-WHY] One test, for the reason the block above is one test: both
// `REASONING_BUDGET` and `REASONING_EFFORT` are process-globals and the harness
// runs these in parallel threads, so two tests that each set and clear them race
// and the failure reads as a broken control rather than a broken test. The switch
// and the effort are owned here end to end and left cleared.
#[cfg(test)]
mod _reasoning_control_tests {
    use super::{
        OPENAI_EFFORT_MECHANISM, PROVIDER_DEFAULT_EFFORT, REASONING_BUDGET_SITES, ReasoningEffort,
        SAMPLING_SCHEDULE, SamplingPreset, UNSUPPORTED_MECHANISM, reasoning_budget_mechanism,
        reasoning_budget_text, reasoning_effort_mechanism, reasoning_effort_text,
        set_reasoning_budget, set_reasoning_effort, with_sampling_pins,
    };
    use serde_json::json;

    #[test]
    fn the_reasoning_switches_reach_the_three_attacker_sites_and_nothing_else() {
        // The allow-list is the Attacker's three call sites and only those, in the
        // schedule's own order: the audit holds a fixed measurement and the Creator
        // is a different experiment, not the same one made cheaper.
        assert_eq!(
            REASONING_BUDGET_SITES,
            &["attacker_generate", "attacker_revise", "attacker_simplify"]
        );

        // (1) The default and every run before §14.21: nothing asked for, so no body
        //     carries a reasoning control at all and the record reads the two
        //     mechanisms the transport has always used.
        set_reasoning_budget(None);
        set_reasoning_effort(None);
        for (site, _, _) in SAMPLING_SCHEDULE {
            let body = with_sampling_pins(
                json!({"reasoning_effort": "low"}),
                SamplingPreset::Pinned,
                site,
            );
            assert_eq!(
                body.get("reasoning_budget_tokens"),
                None,
                "an unset budget must leave the body untouched ({site})"
            );
            assert_eq!(
                body["reasoning_effort"],
                json!("low"),
                "an unset effort must not change the site's own level ({site})"
            );
        }
        assert_eq!(reasoning_budget_text(), "unset");
        assert_eq!(reasoning_budget_mechanism(false), "unset");
        // The EFFECTIVE effort, not the switch: `low` is what the six call sites
        // send, so `unset` here would describe a run that sent no effort at all.
        assert_eq!(reasoning_effort_text(false), "low");
        assert_eq!(reasoning_effort_mechanism(false), "llamacpp:noop");
        // §14.28 — the SAME unset switch on the hosted backend is a different
        // arm: the call sites' `low` is dropped before the request leaves, so the
        // model reasons at its provider's default and the record says so.
        assert_eq!(
            reasoning_effort_text(true),
            PROVIDER_DEFAULT_EFFORT,
            "the hosted default is the provider's choice, not `low`"
        );
        assert_eq!(reasoning_effort_mechanism(true), "unset");

        // (2) The budget arm reaches exactly the three allow-listed sites, and names
        //     both the value it asked for and the mechanism that carried it.
        set_reasoning_budget(Some(12000));
        for (site, _, _) in SAMPLING_SCHEDULE {
            let body = with_sampling_pins(json!({}), SamplingPreset::Pinned, site);
            if REASONING_BUDGET_SITES.contains(site) {
                assert_eq!(
                    body["reasoning_budget_tokens"],
                    json!(12000),
                    "the budget must reach the request body ({site})"
                );
            } else {
                assert_eq!(
                    body.get("reasoning_budget_tokens"),
                    None,
                    "the budget must not reach a site outside the allow-list ({site})"
                );
            }
        }
        assert_eq!(reasoning_budget_text(), "12000");
        assert_eq!(
            reasoning_budget_mechanism(false),
            "llamacpp:reasoning_budget_tokens"
        );
        // §14.28 — the hosted API defines no such field, so the same switch there
        // records a control that could not be expressed. The body carries nothing
        // either way, which is exactly why the mechanism has to say so.
        assert_eq!(reasoning_budget_mechanism(true), UNSUPPORTED_MECHANISM);
        assert_eq!(reasoning_budget_text(), "12000", "the ask is still recorded");

        // (3) The effort REPLACES a key the call site already set and never invents
        //     one for a site that had none — the six `low` call sites are the ones
        //     this knob is about. The recorded mechanism says the transport ignores
        //     the level (`llamacpp:noop`), which is the truth for this template.
        set_reasoning_effort(Some(ReasoningEffort::High));
        let pinned = with_sampling_pins(
            json!({"reasoning_effort": "low"}),
            SamplingPreset::Pinned,
            "attacker_revise",
        );
        assert_eq!(pinned["reasoning_effort"], json!("high"));
        let invented = with_sampling_pins(json!({}), SamplingPreset::Pinned, "attacker_revise");
        assert_eq!(
            invented.get("reasoning_effort"),
            None,
            "the effort must not be invented for a site that never asked for one"
        );
        assert_eq!(reasoning_effort_text(false), "high");
        assert_eq!(reasoning_effort_mechanism(false), "llamacpp:noop");
        // §14.28 — on the hosted backend the same ask IS the intervention, and
        // the mechanism says which transport carried it.
        assert_eq!(reasoning_effort_text(true), "high");
        assert_eq!(reasoning_effort_mechanism(true), OPENAI_EFFORT_MECHANISM);

        // `none` is a LEVEL, not the absence of the switch: it is a body that asks
        // the backend to reason not at all, and the record must say so.
        set_reasoning_effort(Some(ReasoningEffort::None));
        assert_eq!(
            with_sampling_pins(
                json!({"reasoning_effort": "low"}),
                SamplingPreset::Pinned,
                "attacker_revise",
            )["reasoning_effort"],
            json!("none")
        );
        assert_eq!(reasoning_effort_text(false), "none");
        assert_eq!(reasoning_effort_text(true), "none");

        // An unparsable level is an error, never a fallback: an ablation arm must run
        // at the level it named, or the comparison means nothing.
        assert!(ReasoningEffort::parse("bogus").is_err());
        assert!(ReasoningEffort::parse("").is_err());
        assert_eq!(ReasoningEffort::parse("medium").unwrap().name(), "medium");
        // §14.28 — the full documented ladder, and NOTHING beyond it: the two
        // hosted families list `xhigh` and `max`, and neither lists `minimal`.
        assert_eq!(ReasoningEffort::parse("xhigh").unwrap().name(), "xhigh");
        assert_eq!(ReasoningEffort::parse("max").unwrap().name(), "max");
        assert!(
            ReasoningEffort::parse("minimal").is_err(),
            "`minimal` is in neither model's ladder, so this enum must not invent it"
        );

        // The cleared state this test has to leave behind.
        set_reasoning_budget(None);
        set_reasoning_effort(None);
        assert_eq!(reasoning_budget_text(), "unset");
        assert_eq!(reasoning_budget_mechanism(false), "unset");
        assert_eq!(reasoning_budget_mechanism(true), "unset");
        assert_eq!(reasoning_effort_text(false), "low");
        assert_eq!(reasoning_effort_mechanism(true), "unset");
    }
}

// ============================================================================
// PER-ROUND SEEDS (§14.23) — tests
// ============================================================================
//
// [VANTA-WHY] `seed_for_round` is a pure function of its two arguments, so this
// needs no global and cannot race the parallel harness. The const spellings are
// pinned here too: the string a `call.usage` row renders and the string a batch
// table groups by have to be the same literal.
#[cfg(test)]
mod _seed_round_tests {
    use super::{
        LLAMACPP_SEED_MECHANISM, SEED_POLICY, UNSET_SEED_MECHANISM, seed_for_round,
    };

    #[test]
    fn round_one_is_unchanged_and_every_later_round_is_offset_by_its_index() {
        // The record that tells a post-§14.23 row from a pre-§14.23 one, and the
        // two mechanisms a row can name.
        assert_eq!(SEED_POLICY, "per_round_offset");
        assert_eq!(LLAMACPP_SEED_MECHANISM, "llamacpp:seed");
        assert_eq!(UNSET_SEED_MECHANISM, "unset");

        // (1) Round 1 sends the configured seed UNCHANGED, so a single-round run —
        //     and every run that never reaches round 2 — is byte-identical to one
        //     from before this offsetting existed.
        assert_eq!(seed_for_round(Some(7), 1), Some(7));

        // (2) Round `r` is the pinned seed plus `r - 1`, so rounds 2..n are distinct
        //     draws from the same pinned baseline instead of deterministic copies.
        assert_eq!(seed_for_round(Some(7), 2), Some(8));
        assert_eq!(seed_for_round(Some(7), 3), Some(9));
        assert_eq!(seed_for_round(Some(0), 5), Some(4));

        // (3) Round 0 is the driver-less state (`post_chat`, the M0 spike,
        //     bootstrap), and it must behave exactly like round 1 — otherwise a
        //     one-shot caller would sample at a seed nobody asked for.
        assert_eq!(seed_for_round(Some(7), 0), Some(7));

        // (4) An unseeded run stays unseeded in every round: `None` means "send no
        //     `seed` key at all" (§4.4), and no round may invent one.
        assert_eq!(seed_for_round(None, 1), None);
        assert_eq!(seed_for_round(None, 9), None);

        // (5) The offset is wrapping, matching `seed_for_attempt`, so the largest
        //     seed does not panic in debug around the round arithmetic.
        assert_eq!(seed_for_round(Some(u64::MAX), 2), Some(0));
    }
}

// [VANTA-WHY] §14.30 — the provider-retry DECISION, tested without the ladder.
// The mocked server in `m3_session_log.rs` proves the re-send happens; what it
// cannot cheaply prove is the BOUND, because reaching it means waiting out four
// backoffs (1+2+4+8s). Factoring the class-plus-budget check out of the send loop
// is what lets these assertions be instant and exhaustive: every status a
// provider can answer with, and every point on the budget.
#[cfg(test)]
mod _provider_retry_tests {
    use super::{
        PROVIDER_RETRY_MAX, ProviderRetryClass, provider_retry_class, provider_retry_decision,
    };

    fn err(status: u16, body: &str) -> String {
        format!("HTTP ERROR {status} from http://example.invalid/v1/chat/completions. Body snippet: {body}")
    }

    #[test]
    fn only_a_429_the_service_asks_us_to_repeat_is_retried() {
        // (1) A 429 is the one status that means "the request was fine, the service
        //     was busy" — and it is retried, in both of its documented flavours.
        assert!(matches!(
            provider_retry_class(&err(429, "Rate limit reached for gpt-6-luna")),
            Some(ProviderRetryClass::RateLimit)
        ));
        assert!(matches!(
            provider_retry_class(&err(429, "Resource Unavailable: no capacity")),
            Some(ProviderRetryClass::ResourceUnavailable)
        ));

        // (2) `insufficient_quota` arrives as a 429 and is NOT transient: the
        //     account is out of credit, so four retries would only add a minute to
        //     a failure that is waiting for a human.
        assert!(
            provider_retry_class(&err(429, "You exceeded your current quota: insufficient_quota")).is_none(),
            "an out-of-credit 429 must not be retried"
        );

        // (3) Everything else keeps its existing treatment. The 5xx case is the
        //     one that matters: llama-server answers an over-long prompt with a
        //     5xx, and retrying that is a second full generation onto a full
        //     KV cache — the cascade `back_off` exists to avoid.
        for status in [400, 401, 404, 422, 500, 502, 503] {
            assert!(
                provider_retry_class(&err(status, "some provider text")).is_none(),
                "only a 429 may be retried; {status} must not be"
            );
        }
        // The two transport faults never reached a status line at all.
        assert!(provider_retry_class("HTTP POST to http://x timed out: no response headers").is_none());
        assert!(provider_retry_class("HTTP POST to http://x FAILED: connection refused").is_none());
    }

    #[test]
    fn the_ladder_stops_at_exactly_the_budget() {
        let busy = err(429, "Resource Unavailable");
        // Every retry inside the budget is allowed...
        for made in 0..PROVIDER_RETRY_MAX {
            assert!(
                matches!(
                    provider_retry_decision(&busy, made),
                    Some(ProviderRetryClass::ResourceUnavailable)
                ),
                "retry {made} is still within the budget"
            );
        }
        // ...and the one that would exceed it is refused, so the call fails with
        // the provider's own error rather than looping.
        assert!(
            provider_retry_decision(&busy, PROVIDER_RETRY_MAX).is_none(),
            "the ladder must stop at PROVIDER_RETRY_MAX so a provider that is really down \
             fails the call instead of running out the clock"
        );
        assert!(provider_retry_decision(&busy, PROVIDER_RETRY_MAX + 7).is_none());
    }
}
