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

use crate::config::LlmConfig;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::sync::OnceCell;
use tracing::{debug, info, trace, warn};

// Re-export the agent response types so bootstrap can use them too.
pub use crate::agents::attacker::{AttackerResponse, AttackerTestCase};
pub use crate::agents::creator::CreatorResponse;

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
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
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
    /// Whether the server stopped this call at `max_tokens` (`finish_reason:
    /// "length"`).
    ///
    /// [VANTA-WHY] §14.5 — a model that "runs away" spends the whole output cap
    /// on reasoning and returns nothing parseable. Counting those per role is
    /// how the paper reports how often it happens, and a counter derived from
    /// the calls the run already records cannot disagree with the run it
    /// describes.
    pub truncated: bool,
}

static CALL_USAGES: Mutex<Vec<CallUsage>> = Mutex::new(Vec::new());
static USAGE_SEQ: AtomicU64 = AtomicU64::new(0);

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
    let filename = format!("{type_name}{label}_round_{round}_attempt_{attempt}_{suffix}");
    let path = dir.join(filename);
    match std::fs::write(&path, content) {
        Ok(_) => debug!(path = %path.display(), "forensics written"),
        Err(e) => warn!(err = %e, path = %path.display(), "forensics write failed"),
    }
}

/// A test name can carry `/`, `[`, `]`, spaces — none of which belong in a
/// filename.
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
/// [VANTA-WHY] These are part of v0.2's behaviour — the sampling the collected
/// baseline ran under — so they are NOT configurable and no env var may override
/// them; moving them would move the control the pilot is measured against. The
/// values are duplicated at the call sites rather than read from this table on
/// purpose: `tests/m3_session_log.rs::the_recorded_sampling_schedule_matches_the_request_bodies`
/// compares this table against the bodies actually sent, which is the guard
/// against the two copies drifting.
pub const SAMPLING_SCHEDULE: &[(&str, f64, u32)] = &[
    ("creator", 0.1, 32768),
    ("attacker_generate", 0.2, 32768),
    ("attacker_revise", 0.3, 32768),
    ("attacker_simplify", 0.3, 32768),
    ("intent_audit", 0.0, 2048),
];

/// The server's own `top_p` default — the sampler parameter NO call site pins,
/// so every request inherits whatever the server is configured with.
///
/// [VANTA-WHY] This is the one value `run.summary` really does have to read from
/// `/props` (§4.3): `temperature` and `max_tokens` are pinned per call (see
/// `SAMPLING_SCHEDULE`), `top_p` is not, so the server's answer is the only
/// truthful source. `None` means the server did not publish the key — a minimal
/// stub, or a llama.cpp build that omits it — and it is recorded as unset rather
/// than replaced by a plausible default, because a fabricated sampler value in
/// the provenance block is precisely the kind of claim this project exists to
/// avoid.
pub async fn server_top_p_default(client: &reqwest::Client, cfg: &LlmConfig) -> Option<f64> {
    let body = fetch_props(client, cfg).await?;
    body["default_generation_settings"]["top_p"].as_f64()
}

/// Resolve the model's context window for THIS run.
///
/// Order: `VANTA_LLM_CONTEXT_TOKENS` if set (it wins, but a disagreement with
/// the server is logged, because a stale override silently starves the guard),
/// else the server's `/props` `n_ctx`, else 65536 with a warning.
pub async fn context_window_tokens(client: &reqwest::Client, cfg: &LlmConfig) -> u32 {
    *CONTEXT_WINDOW
        .get_or_init(|| async {
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
    let full_body = Value::Object(full_body);

    let context_window = context_window_tokens(client, cfg).await;
    let requested_output_tokens = full_body["max_tokens"].as_u64().unwrap_or(16_384) as u32;
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
            "Prompt would exceed model context: prompt_tokens={} + max_tokens={} + overhead={} > context_window={}",
            prompt_tokens,
            requested_output_tokens,
            CHAT_OVERHEAD_TOKENS,
            context_window,
        )
        .into());
    }

    let t = Instant::now();
    // The idle timeout covers the WAIT FOR HEADERS too: a server that accepts
    // the connection and then says nothing is as dead as one that stops
    // mid-stream. Dropping this future closes the socket.
    let send_future = client.post(&url).json(&full_body).send();
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
    if !status.is_success() {
        let code = status.as_u16();
        let txt = resp.text().await.unwrap_or_else(|_| String::from("<could not read body>"));
        return Err(format!(
            "HTTP ERROR {code} from {url}. Body snippet: {}",
            &txt[..300.min(txt.len())]
        ).into());
    }

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
    let usage = CallUsage {
        seq: USAGE_SEQ.fetch_add(1, Ordering::Relaxed),
        site: site.to_string(),
        prompt_tokens: Some(prompt_tokens_recorded),
        completion_tokens: measured_completion,
        total_tokens: measured_total,
        source: usage_source.to_string(),
        seed,
        truncated: streamed.finish_reason.as_deref() == Some("length"),
    };
    debug!(
        site,
        prompt_tokens = prompt_tokens_recorded,
        completion_tokens = ?measured_completion,
        token_source = usage_source,
        seed = ?seed,
        truncated = usage.truncated,
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

    let ms = t.elapsed().as_millis();
    debug!(
        status = status.as_u16(),
        elapsed_ms = ms,
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
const DETERMINISTIC_CONTEXT_PREFIX: &str = "Prompt would exceed model context";

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

/// The `seed` one attempt sends (§14.5).
///
/// [VANTA-WHY] Attempt 1 always sends the configured seed unchanged, so a call
/// that never retries is byte-identical to one from before this offsetting
/// existed.
///
/// [VANTA-WHY] A RETRY advances it by the attempt index, because the harness
/// pins ONE seed per run: with the same seed and the same prompt, even a
/// temperature-0.2 retry can come back identical — and be cut off at the
/// identical point — so the retry budget would be spent reproducing the same
/// prefix. Advancing the seed keeps the run reproducible (the sequence is a
/// function of the pinned seed and the attempt number, and both are recorded)
/// while letting the retry actually differ.
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
    // Whether the loop stopped ON a truncation (rather than exhausting its
    // budget), so the final failure can say which of the two happened.
    let mut stopped_on_truncation = false;
    // Attempts that actually reached the parse step, so the failure can report
    // how many were spent rather than how many were allowed: a call that stops
    // early (a cut-off answer at temperature 0.0) did not fail `max_attempts`
    // times.
    let mut attempts_made = 0_u32;

    // [VANTA-WHY] §14.5 — whether this call is PINNED at temperature 0.0, and
    // therefore answers identically on every attempt. It is read from the
    // request's own `temperature`, because that is what the server samples with:
    // every M3 agent passes its pin through `extra_fields`, which wins over
    // `post_chat`'s own body default, so a caller that sends no extras samples at
    // that default (0.2) and is NOT deterministic.
    //
    // [VANTA-WHY] This is the scope of the fail-fast below. The "same request is
    // cut off at the same point" argument holds only at temperature 0.0: the
    // Creator and the Attacker run at 0.1-0.3, where a cut-off answer CAN come
    // back different — and a model that spends the whole output cap on reasoning
    // returns nothing parseable however many times it is asked, which is exactly
    // the case a retry exists for. Applying the fail-fast to them turned one
    // model run-away into a dead run.
    let deterministic_temperature = extras_for_attempt
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(0.2)
        <= 0.0;

    for attempt in 1..=max_attempts {
        let messages = build_messages(attempt);
        attempts_made = attempt;
        if attempt > 1 {
            warn!(attempt, max_attempts, "retrying structured call");
        }
        let attempt_seed = seed_for_attempt(cfg.seed, attempt, deterministic_temperature);

        let (raw, finish_reason) = match post_chat_meta(
            client,
            cfg,
            messages,
            extras_for_attempt.clone(),
            site,
            attempt_seed,
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
        if last_was_truncated {
            warn!(
                attempt,
                "the server cut this answer off at max_tokens — it cannot be parsed as written"
            );
        }

        let type_name = std::any::type_name::<T>().split("::").last().unwrap_or("T");

        // [FORENSICS] Write the raw model response to disk before parsing,
        // so we can inspect exactly what the model returned if parsing fails.
        write_forensics(type_name, forensics_tag, attempt, "raw_response.txt", &raw);

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
                // [VANTA-WHY] §14.5 — a CUT-OFF answer fails fast ONLY where a
                // retry is provably deterministic: a call PINNED at temperature
                // 0.0 samples greedily, so `max_tokens` being pinned per call
                // site means the same request is cut off at the same point every
                // time, and each retry is a full-length generation spent to
                // reach the identical prefix. Failing here is what lets the
                // callers that can ask a SMALLER question (the batched audit,
                // one test per call) do it on the first attempt instead of the
                // fourth.
                //
                // [VANTA-WHY] Above temperature 0.0 the same request CAN come
                // back different — that is the whole reason the pipeline samples
                // the Creator and Attacker at 0.1-0.3 — so a cut-off answer keeps
                // its retries, with the seed advanced each time (see
                // `seed_for_attempt`). A model that spends the entire output cap
                // on reasoning and returns no content is precisely the transient
                // failure a retry exists for; refusing to retry it turned one
                // run-away into a dead run. A malformed answer that was NOT cut
                // off is a different fact again — the model answered, and the
                // answer was wrong — so it keeps its retries too.
                if last_was_truncated && deterministic_temperature {
                    stopped_on_truncation = true;
                    break;
                }
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
        let tail = if stopped_on_truncation {
            "The call was NOT retried: `max_tokens` is pinned for this call site AND this call is \
             pinned at temperature 0.0, so the same request is cut off at the same point every \
             time. Ask for less, or raise the cap."
                .to_string()
        } else if attempts_made > 1 {
            format!(
                "This call site samples above temperature 0.0, so a retry can come back \
                 different: each attempt advanced the seed by its attempt index, and both are \
                 recorded with the call. {} retries were spent and the last attempt was cut off \
                 as well. Ask for less, or raise the cap.",
                attempts_made - 1
            )
        } else {
            "This call site samples above temperature 0.0, so a retry could come back different, \
             but no retry was made: the attempt budget is one. Ask for less, or raise the cap."
                .to_string()
        };
        message.push_str(&format!(
            "\n{TRUNCATED_OUTPUT_PREFIX}: the server stopped this answer at the request's \
             `max_tokens` cap, so the reply is a prefix of the intended one and could not be \
             parsed. {tail}"
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
