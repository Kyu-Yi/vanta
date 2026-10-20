// ============================================================================
// config/mod.rs — LlmConfig + from_env() constructor
// ============================================================================
//
// WHAT YOU'LL LEARN HERE
//
//   # | Rust concept              → Where Vanta reuses it later
//  ---+---------------------------+-------------------------------------------
//   1 | std::env::var             → M3 CLI: read 8+ env vars (VANTA_* pattern)
//     |   (KEY_ENV)               →   for config, API keys, secrets
//  ---+---------------------------+-------------------------------------------
//   2 | Option<String> api_key    → Spec: local model (llama-server) runs with
//     |                           →   NO auth; api_key is None for local.
//     |                           →   Anthropic tier uses Some(sk-ant-...).
//  ---+---------------------------+-------------------------------------------
//   3 | Default trait impl        → struct default constructor via
//     |   impl Default for T     →   Default::default() or ..Default::default()
//     |                           →   struct literal spread operator.
//  ---+---------------------------+-------------------------------------------
//   4 | impl Display for T        → println!("{config}") pretty printing
//     |   (manual fmt() impl)     → M3 run banner: prints resolved config
//  ---+---------------------------+-------------------------------------------
//   5 | Result<String, From<...>> → Env var: VarError → our String Err
//     |   + map_err               →   with human context (VANTA_LLM_...)
// ============================================================================

// ----------------------------------------------------------------------------
// [SYNTAX] use std::env; — brings the std env module into local scope.
// env::var("NAME") returns Result<String, env::VarError>:
//   Ok(value)      → env var WAS set, value is an owned String of its content
//   Err(VarError::NotPresent)    → env var was not set at all
//   Err(VarError::NotUnicode(s)) → env var set but contained non-UTF8 bytes
//
// [VANTA-WHY] We use environment variables for ALL configuration per M1
// Master Plan constraint "env var config, no YAML/TOML for M0-M2". For M3
// we'll introduce a real config file format; env vars are the zero-dependency
// quickstart option that work in CI, shell scripts, and launchd services.
// ----------------------------------------------------------------------------
use std::env;
use std::fmt;

// ----------------------------------------------------------------------------
// LlmConfig — every parameter for calling ANY LLM backend (local or remote)
// ----------------------------------------------------------------------------
//
// Shape matches the OpenAI-compatible REST API convention used by:
//   • llama-server /v1 endpoint   (local, Metal GPU, Qwen3.5-4B GGUF)
//   • Ollama openai compat layer  (alt local runner)
//   • OpenAI platform             (gpt-4o tier later)
//   • Anthropic via proxy         (uncommon; convert OpenAI format at proxy)
//
// This single struct replaces 30+ different "API client" crates in one shot.
//
// Serde derives: YES because in M3 we save/load session configs to JSON.
//
// [PITFALL] Debug derive on api_key: Debug prints {api_key: Some("sk-ant-123...")}
// to console. If someone does println!("{cfg:?}") in CI, the secret leaks to
// logs. We COULD make a custom Debug that redacts. For now: api_key is
// Option<String> derive Debug, but you MUST NEVER format a full LlmConfig
// with {:?} in production output; use the Display impl below which omits it.
// ----------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq)]
pub struct LlmConfig {
    // [SYNTAX] base_url: String = HTTP scheme + host + port + API prefix.
    // [EXAMPLE] Default for local llama-server is "http://localhost:8080/v1".
    // [EXAMPLE] For OpenAI it's "https://api.openai.com/v1".
    // [EXAMPLE] For Anthropic-compatible proxy it's "https://api.anthropic.com/v1".
    //
    // [VANTA-WHY] The /v1 prefix is mandatory for OpenAI-compatible endpoints.
    // llama-server mounts the Chat Completions POST at:
    //   POST {base_url}/chat/completions
    // So if base_url is missing /v1, POST {base_url}/chat/completions 404s.
    pub base_url: String,

    // [SYNTAX] model_id: String = logical model name string the backend uses.
    // [EXAMPLE] "Qwen3.5-4B" for llama-server (user-facing model name)
    // [EXAMPLE] "gpt-4o" for OpenAI, "claude-3-5-sonnet-20240620" for Anthropic
    //
    // [VANTA-WHY] When you run llama-server --model foo.gguf --alias Qwen4B
    // the --alias string is what shows up as the model ID in responses and
    // what clients are expected to send in request JSON. If your request
    // sends a model ID the server doesn't recognize → 400 Bad Request.
    pub model_id: String,

    // [SYNTAX] api_key: Option<String> = "maybe a key exists, maybe not".
    //
    // [VANTA-WHY] SPECIFICALLY TO ANSWER YOUR QUESTION: "for the anthropic api
    // key can we use a local model i have installed in my llama.cpp folder" —
    // YES, 100% yes. This is exactly why api_key is Option.
    //
    //   Local llama-server (M0, M1, M2 experiments): api_key = None
    //     → llama-server DOES NOT REQUIRE auth. HTTP requests omit
    //       Authorization header entirely. Works on your machine offline.
    //
    //   Tier experiments M3+: api_key = Some(secret)
    //     → Anthropic / OpenAI / etc. require Authorization: Bearer $KEY.
    //     → We read the key from env var so it's never committed to Git.
    //
    // [PITFALL] Never store a literal API key in source code. Never commit a
    // file with a key to GitHub. The correct approach is (and will always be):
    //   export VANTA_API_KEY='sk-ant-XXXX'   # shell env, per-session
    //   → code reads env::var() here
    pub api_key: Option<String>,

    // [SYNTAX] idle_timeout_secs: u64 = seconds with NO NEW TOKEN before the
    // stream is abandoned. Default 180. Override with VANTA_LLM_IDLE_TIMEOUT_SECS.
    //
    // [VANTA-WHY] There used to be ONE total timeout of 600s for the whole
    // request. A watch-the-clock deadline is the wrong shape for a generation
    // that legitimately takes 15 minutes: it killed a Creator call that was
    // still producing tokens, and because the connection was never dropped,
    // llama-server kept generating the orphan while the retry started a second
    // generation beside it until the shared KV cache overflowed. Progress, not
    // elapsed time, is what tells a slow generation from a dead one.
    pub idle_timeout_secs: u64,

    // [SYNTAX] max_call_secs: u64 = overall backstop for ONE streamed call.
    // Default 3600. Override with VANTA_LLM_MAX_SECS.
    //
    // [VANTA-WHY] Not redundant with the idle timeout: a server that keeps
    // dribbling one token forever would never trip the idle timer, so a single
    // call still needs an outer bound. It is deliberately 20x the idle default
    // — it exists to catch a pathological stream, not to second-guess a slow
    // model.
    pub max_call_secs: u64,

    // [SYNTAX] retry_attempts: u32 = max structured_call_with_retry attempts.
    // Default: 3. Per Spec §1 reliability requirements (3 retry budget).
    pub retry_attempts: u32,

    // [VANTA-WHY] Prompt-budget preflight needs the model's total context
    // window so Vanta can fail before sending an oversized request.
    //
    // [VANTA-WHY] Option, not a u32 defaulting to 65536. A hardcoded window is
    // a guess about someone else's server: llama-server is started with a
    // specific `--ctx-size`, and Vanta guessing low refuses prompts that fit
    // while guessing high lets an oversized one through. None means "ask the
    // server's /props" (see llm::context_window_tokens); Some(v) is
    // VANTA_LLM_CONTEXT_TOKENS, which wins but is checked against /props.
    pub context_window_tokens: Option<u32>,

    // [SYNTAX] seed: Option<u64> = "maybe the caller pinned a sampler seed".
    // Set with VANTA_LLM_SEED. None = the field is ABSENT from the request body.
    //
    // [VANTA-WHY] None must mean ABSENT, not `seed: 0`. The measurement
    // baseline this project is compared against was collected with whatever
    // the server's own default sampling was; a body that suddenly carried a
    // seed would move the control rather than repeat it. So the key is added
    // only when the caller asked for it (llm::post_chat), and "unset
    // reproduces today byte-for-byte" is an assertion, not an intention.
    //
    // [VANTA-WHY] A seed is a reproducibility HINT, not a guarantee. llama.cpp
    // perturbs sampling through the prompt cache, continuous batching, and a
    // non-deterministic float reduction order across threads, so the same seed
    // is a *repeat key* for a run — never a claim of bit-exact reruns.
    //
    // [VANTA-WHY] Option<u64> and no "-1 = random" sentinel: llama.cpp's seed
    // is 64-bit, and "let the server decide" is exactly what an absent key
    // already says without inventing a magic number that a stricter backend
    // might read as a real seed.
    pub seed: Option<u64>,
}

// ----------------------------------------------------------------------------
// Default impl for LlmConfig — Sensible DEFAULTS for M0 Qwen3.5-4B local setup
// ----------------------------------------------------------------------------
//
// [SYNTAX] impl Default for LlmConfig { fn default() -> Self { ... } }
//   Implements the std::default::Default trait. Callers write:
//     let cfg = LlmConfig::default();
//
//   Or in a struct literal with partial overrides:
//     let cfg = LlmConfig {
//         model_id: "Ornith-8B".to_string(),
//         ..Default::default()  // fill ALL UNSPECIFIED fields from default()
//     };
//
// [VANTA-WHY of defaults] Per user feedback:
//   "please use smaller model possibly my qwen4B or ornith or next coder
//    since this is testing phase and this can be changed at any time"
// Defaults are Qwen3.5-4B on localhost:8080 NO API KEY = zero-setup run for
// your M0 testing. Swap via env vars any time.
impl Default for LlmConfig {
    fn default() -> Self {
        LlmConfig {
            // llama-server prints "listening on http://127.0.0.1:8080" (IPv4 literal,
            // NOT "localhost"). Using the IPv4 literal avoids:
            //   (a) reqwest's DNS resolver preferring IPv6 ::1 (which llama-server
            //       does NOT bind to by default, causing Connection refused)
            //   (b) /etc/hosts localhost aliases
            // The /v1 prefix is mandatory for OpenAI-compatible endpoints.
            base_url: "http://127.0.0.1:8080/v1".to_string(),

            // llama-server will use whatever model you passed with --model;
            // the model_id field in requests is validated but not strictly
            // required. We use a descriptive string matching the GGUF.
            // User can override: export VANTA_LLM_MODEL="qwen3.6-35b-a3b" etc.
            // [VANTA-WHY] 4b-class models burn tokens on reasoning and often
            // fail to emit valid JSON content. 27b+ is needed for reliable
            // structured output + patch generation.
            model_id: "qwen3.8-27b".to_string(),

            // Local = NO auth needed. So None.
            // If setting a remote key: env::var("VANTA_API_KEY") returns Some.
            api_key: None,

            // Seconds with no new token before a stream is abandoned.
            idle_timeout_secs: 180,

            // Overall backstop for one streamed call.
            max_call_secs: 3600,

            // Spec §1: 3-attempt retry budget for structured LLM calls.
            retry_attempts: 3,

            // None = read n_ctx from the server's /props (llm module), falling
            // back to 65536 only when the server cannot be asked.
            context_window_tokens: None,

            // None = no `seed` key in the request; the server samples with its
            // own default, which is what the existing baseline was measured
            // under. Set VANTA_LLM_SEED to pin one.
            seed: None,
        }
    }
}

// ----------------------------------------------------------------------------
// Display impl for LlmConfig — pretty print, NEVER leaks api_key
// ----------------------------------------------------------------------------
//
// [SYNTAX] impl fmt::Display for LlmConfig { fn fmt(...) -> fmt::Result }
//   Display = "{}" format. Debug = "{:?}" format. Display is for end-user
//   facing pretty output. Debug is for developer dumps (may show more).
//
// [VANTA-WHY] Whenever the M1+ program starts it prints a banner:
//   ┌─ Vanta run starting ──────────────────────────┐
//   │ LLM : http://localhost:8080/v1 @ Qwen3.5-4B   │
//   │ Key : (not set, local mode)                    │
//   │ Timeout : 600s                                 │
//   └───────────────────────────────────────────────┘
// Use Display for that. Never Debug, since Debug would print the key literal.
impl fmt::Display for LlmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key_display = match &self.api_key {
            Some(k) if k.len() >= 8 => format!("{}...{}", &k[..4], &k[k.len()-4..]),
            Some(_)                 => "(< 8 chars, redacted)".to_string(),
            None                    => "(not set — local / no-auth mode)".to_string(),
        };
        write!(f,
            "LlmConfig:\n  \
            endpoint       : {}\n  \
            model          : {}\n  \
            api_key        : {}\n  \
            idle_timeout   : {}s (abandon after no new token)\n  \
            max_call       : {}s (backstop per call)\n  \
            retry_attempts : {}\n  \
            context_window : {}\n  \
            seed           : {}",
            self.base_url, self.model_id, key_display,
            self.idle_timeout_secs, self.max_call_secs, self.retry_attempts,
            match self.context_window_tokens {
                Some(v) => format!("{v} (VANTA_LLM_CONTEXT_TOKENS)"),
                None => "auto (server /props, fallback 65536)".to_string(),
            },
            // [VANTA-WHY] Printed because a run's sampling is part of its
            // identity: two runs whose banners differ only here are not the
            // same experiment, and the banner is the only place a human sees it
            // before the session log exists.
            match self.seed {
                Some(v) => format!("{v} (VANTA_LLM_SEED)"),
                None => "unset (server default)".to_string(),
            }
        )
    }
}

impl LlmConfig {
    // ------------------------------------------------------------------------
    // from_env() — primary constructor: read env vars, fall back to defaults
    // ------------------------------------------------------------------------
    //
    // [SYNTAX] pub fn from_env() -> Result<Self, String>
    //   Associated function (no &self param) — call as LlmConfig::from_env().
    //   Returns Result because env::var can fail with non-UTF8 values (which
    //   we escalate to human-readable Err messages instead of panicking).
    //
    // [ENV VAR CONTRACT]
    //
    //   Variable name              | Purpose                           | Default
    //   ---------------------------+-----------------------------------+----------
    //   VANTA_LLM_BASE_URL         | HTTP endpoint /v1 prefix           | localhost:8080/v1
    //   VANTA_LLM_MODEL            | Model identifier string           | Qwen3.5-4B
    //   VANTA_API_KEY              | Bearer token / API key (optional) | None
    //   VANTA_LLM_IDLE_TIMEOUT_SECS| Seconds with NO new token          | 180
    //   VANTA_LLM_MAX_SECS         | Overall backstop per LLM call      | 3600
    //   VANTA_LLM_CONTEXT_TOKENS   | Model context window (else /props) | unset
    //   VANTA_LLM_RETRY_ATTEMPTS   | structured_call retry budget       | 3
    //   VANTA_LLM_SEED             | Sampler seed (else server default) | unset
    //
    // All of them are optional to set (defaults used). If any of them contain
    // non-UTF-8 bytes (shouldn't happen in normal shell use), we return Err
    // rather than silently continuing with malformed config.
    //
    pub fn from_env() -> Result<Self, String> {
        // Start with defaults, then override field by field if env was set.
        let mut cfg = LlmConfig::default();

        // ---- base_url override ---------------------------------------------
        // [SYNTAX] match env::var("NAME") { Ok(v) => ..., Err(e) => ... }
        // We explicitly handle the two VarError cases to produce clear messages.
        match env::var("VANTA_LLM_BASE_URL") {
            Ok(val) if !val.trim().is_empty() => {
                cfg.base_url = val.trim_end_matches('/').to_string();
                // [VANTA-WHY] trim trailing '/' so later concatenation doesn't
                // produce "http://x:8080/v1//chat/completions" (double-slash).
                // Most HTTP clients tolerate double slashes, but not all, and
                // it looks ugly in logs. Safer to normalize at source.
            }
            Ok(_) => {
                // Env var set but empty → treat as unset and keep default.
                // No error: user may have done `export VANTA_LLM_BASE_URL=`.
            }
            Err(env::VarError::NotPresent) => { /* keep default */ }
            Err(env::VarError::NotUnicode(_)) => {
                return Err("CONFIG_ERROR: VANTA_LLM_BASE_URL env var contained \
                            non-UTF-8 bytes. Please re-export with ASCII content.".to_string());
            }
        }

        // ---- model_id override ---------------------------------------------
        match env::var("VANTA_LLM_MODEL") {
            Ok(val) if !val.trim().is_empty() => {
                cfg.model_id = val.trim().to_string();
            }
            Ok(_) => { /* keep default */ }
            Err(env::VarError::NotPresent) => { /* keep default */ }
            Err(env::VarError::NotUnicode(_)) => {
                return Err("CONFIG_ERROR: VANTA_LLM_MODEL env var contained \
                            non-UTF-8 bytes.".to_string());
            }
        }

        // ---- api_key override (OPTIONAL, None default) ---------------------
        match env::var("VANTA_API_KEY") {
            Ok(val) if !val.trim().is_empty() => {
                cfg.api_key = Some(val.trim().to_string());
            }
            Ok(_) => {
                // Env var set but empty string → treat as "no key". Don't wrap
                // empty in Some("") or Authorization header becomes
                // "Bearer " (malformed). Explicitly set to None.
                cfg.api_key = None;
            }
            Err(env::VarError::NotPresent) => { /* keep default None */ }
            Err(env::VarError::NotUnicode(_)) => {
                return Err("CONFIG_ERROR: VANTA_API_KEY env var contained \
                            non-UTF-8 bytes.".to_string());
            }
        }

        // ---- idle_timeout_secs override -------------------------------------
        match env::var("VANTA_LLM_IDLE_TIMEOUT_SECS") {
            Ok(val) if !val.trim().is_empty() => {
                let parsed: u64 = val.trim().parse::<u64>()
                    .map_err(|e| format!(
                        "CONFIG_ERROR: VANTA_LLM_IDLE_TIMEOUT_SECS='{}' not a \
                         valid non-negative integer: {e}", val.trim()))?;
                if parsed == 0 {
                    return Err("CONFIG_ERROR: VANTA_LLM_IDLE_TIMEOUT_SECS must be > 0. \
                                (0 would mean 'abort instantly' — almost certainly a typo.)"
                               .to_string());
                }
                cfg.idle_timeout_secs = parsed;
            }
            Ok(_) => { /* keep default */ }
            Err(env::VarError::NotPresent) => { /* keep default */ }
            Err(env::VarError::NotUnicode(_)) => {
                return Err("CONFIG_ERROR: VANTA_LLM_IDLE_TIMEOUT_SECS env var contained \
                            non-UTF-8 bytes.".to_string());
            }
        }

        // ---- max_call_secs override -----------------------------------------
        match env::var("VANTA_LLM_MAX_SECS") {
            Ok(val) if !val.trim().is_empty() => {
                let parsed: u64 = val.trim().parse::<u64>()
                    .map_err(|e| format!(
                        "CONFIG_ERROR: VANTA_LLM_MAX_SECS='{}' not a valid \
                         non-negative integer: {e}", val.trim()))?;
                if parsed == 0 {
                    return Err("CONFIG_ERROR: VANTA_LLM_MAX_SECS must be > 0.".to_string());
                }
                cfg.max_call_secs = parsed;
            }
            Ok(_) => { /* keep default */ }
            Err(env::VarError::NotPresent) => { /* keep default */ }
            Err(env::VarError::NotUnicode(_)) => {
                return Err("CONFIG_ERROR: VANTA_LLM_MAX_SECS env var contained \
                            non-UTF-8 bytes.".to_string());
            }
        }

        // ---- retry_attempts override ----------------------------------------
        match env::var("VANTA_LLM_RETRY_ATTEMPTS") {
            Ok(val) if !val.trim().is_empty() => {
                let parsed: u32 = val.trim().parse::<u32>()
                    .map_err(|e| format!(
                        "CONFIG_ERROR: VANTA_LLM_RETRY_ATTEMPTS='{}' not a valid \
                         non-negative integer: {e}", val.trim()))?;
                if parsed == 0 {
                    return Err("CONFIG_ERROR: VANTA_LLM_RETRY_ATTEMPTS must be > 0.".to_string());
                }
                cfg.retry_attempts = parsed;
            }
            Ok(_) => { /* keep default */ }
            Err(env::VarError::NotPresent) => { /* keep default */ }
            Err(env::VarError::NotUnicode(_)) => {
                return Err("CONFIG_ERROR: VANTA_LLM_RETRY_ATTEMPTS env var contained \
                            non-UTF-8 bytes.".to_string());
            }
        }

        match env::var("VANTA_LLM_CONTEXT_TOKENS") {
            Ok(val) if !val.trim().is_empty() => {
                let parsed: u32 = val.trim().parse::<u32>().map_err(|e| {
                    format!(
                        "CONFIG_ERROR: VANTA_LLM_CONTEXT_TOKENS='{}' not a valid positive integer: {e}",
                        val.trim()
                    )
                })?;
                if parsed == 0 {
                    return Err(
                        "CONFIG_ERROR: VANTA_LLM_CONTEXT_TOKENS must be > 0.".to_string()
                    );
                }
                cfg.context_window_tokens = Some(parsed);
            }
            Ok(_) => { /* keep default */ }
            Err(env::VarError::NotPresent) => { /* keep default */ }
            Err(env::VarError::NotUnicode(_)) => {
                return Err("CONFIG_ERROR: VANTA_LLM_CONTEXT_TOKENS env var contained non-UTF-8 bytes.".to_string());
            }
        }

        // ---- seed override (OPTIONAL, None default) -------------------------
        // [VANTA-WHY] No "must be > 0" check, unlike the timeouts above: 0 is a
        // perfectly good seed, and rejecting it would be inventing a constraint
        // the sampler does not have. Only "set but not a number" is an error —
        // a silently ignored seed would make two runs that claim the same seed
        // differ for a reason no artifact records.
        match env::var("VANTA_LLM_SEED") {
            Ok(val) if !val.trim().is_empty() => {
                let parsed: u64 = val.trim().parse::<u64>().map_err(|e| {
                    format!(
                        "CONFIG_ERROR: VANTA_LLM_SEED='{}' not a valid non-negative integer: {e}",
                        val.trim()
                    )
                })?;
                cfg.seed = Some(parsed);
            }
            Ok(_) => { /* keep default None */ }
            Err(env::VarError::NotPresent) => { /* keep default None */ }
            Err(env::VarError::NotUnicode(_)) => {
                return Err("CONFIG_ERROR: VANTA_LLM_SEED env var contained non-UTF-8 bytes.".to_string());
            }
        }

        Ok(cfg)
    }

    // ------------------------------------------------------------------------
    // chat_completions_url() — helper: full POST URL for OpenAI-compatible endpoint
    // ------------------------------------------------------------------------
    //
    // [SYNTAX] pub fn chat_completions_url(&self) -> String
    //   &self = shared reference; method call syntax: cfg.chat_completions_url().
    //
    // [VANTA-WHY] Concatenates base_url (which always ends WITHOUT trailing
    // slash because we trim in from_env) + literal "/chat/completions" =
    // the single exact URL for POSTing chat completion request bodies in M1+.
    //
    // Example: default LlmConfig → "http://localhost:8080/v1/chat/completions"
    pub fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

// ============================================================================
// UNIT TESTS (compiled only under cargo test, not shipped binary)
// ============================================================================
//
// [SYNTAX] #[cfg(test)] = compile this module ONLY when running tests via
// `cargo test`. cfg(test) = conditional compilation on the "test" config flag.
// This keeps test code out of release builds.
//
// You can run these tests specifically with: cargo test -p vanta config::tests
// from the vanta project root.
#[cfg(test)]
mod tests {
    // [SYNTAX] use super::*; = "bring everything from parent module (LlmConfig,
    // Default, etc.) into scope inside this child test module". Standard idiom.
    use super::*;

    // [SYNTAX] std::sync::OnceLock = cell that can be written EXACTLY once and
    // read many times. Used for "run once at start of test process" setup.
    // We use it to snapshot the ENV VAR state before tests, since they
    // temporarily SET/UNSET env vars (which is process-global state — ordering
    // can leak between tests if you don't clean up).
    //
    // Each test uses env guards (set_var + RAII reset) instead of OnceLock;
    // but we keep the default test here as reference for you.
    use std::sync::OnceLock;
    static _INIT_ENV_BACKUP: OnceLock<()> = OnceLock::new();

    #[test]
    fn test_defaults_m0_local_profile() {
        // Default should match local 127.0.0.1:8080 no-key profile.
        // 27b-class model for reliable structured output.
        let d = LlmConfig::default();
        assert_eq!(d.base_url, "http://127.0.0.1:8080/v1");
        assert_eq!(d.model_id, "qwen3.8-27b");
        assert!(d.api_key.is_none(), "default must be no-auth local mode");
        assert_eq!(d.idle_timeout_secs, 180);
        assert_eq!(d.max_call_secs, 3600);
        // [SYNTAX] None = "do not guess the server's window; ask /props".
        assert_eq!(d.context_window_tokens, None);
        // [VANTA-WHY] None means the request body carries no `seed` key at all.
        // A default of Some(0) would silently change every existing run's body.
        assert_eq!(d.seed, None);
    }

    #[test]
    fn test_display_never_leaks_key() {
        // Short key should be fully redacted (only "< 8 chars" message shown).
        let short = LlmConfig {
            api_key: Some("123".to_string()),
            ..Default::default()
        };
        let disp = format!("{short}");
        assert!(!disp.contains("123"), "short api key leaked into Display! {disp}");

        // Long key → only first 4 and last 4 chars shown.
        let long = LlmConfig {
            api_key: Some("sk-ant-12345678-ABCD".to_string()),
            ..Default::default()
        };
        let disp = format!("{long}");
        assert!( disp.contains("sk-a...ABCD"),
            "expected truncated api key mask in Display, got: {disp}");
        assert!(!disp.contains("12345678"),
            "middle of api key leaked! {disp}");

        // None → "(not set" message.
        let none = LlmConfig::default();
        let disp = format!("{none}");
        assert!(disp.contains("not set") || disp.contains("local"),
            "None api key should show local/not-set msg: {disp}");
    }

    #[test]
    fn test_chat_completions_url_no_double_slash() {
        // Default → no double slash after /v1
        // Uses IPv4 literal 127.0.0.1 (not localhost) to match llama-server
        // bind address (see Default impl comment).
        let d = LlmConfig::default();
        assert_eq!(d.chat_completions_url(),
                   "http://127.0.0.1:8080/v1/chat/completions");
        assert!(!d.chat_completions_url().contains("//chat"),
            "URL has double slash: {}", d.chat_completions_url());
    }

    #[test]
    fn test_context_window_tokens_from_env() {
        unsafe {
            env::set_var("VANTA_LLM_CONTEXT_TOKENS", "131072");
        }
        let cfg = LlmConfig::from_env().unwrap();
        assert_eq!(cfg.context_window_tokens, Some(131_072));
        unsafe {
            env::remove_var("VANTA_LLM_CONTEXT_TOKENS");
        }
    }
}
