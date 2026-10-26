//! Per-model-family request dialects for the hosted backend (plans/bugsinpy.md
//! §14.26).
//!
//! [VANTA-WHY] Why a table rather than a chain of `if cfg.is_openai()` tests at
//! the call sites: the two backends do not merely differ in a URL, they accept
//! different FIELDS, and *which* fields is a property of the MODEL. `top_k` and
//! `min_p` are llama.cpp sampler knobs that OpenAI's Chat Completions endpoint
//! does not define at all — they are not "wrong values", they are not
//! parameters. `temperature` and `top_p` ARE defined, and the GPT-6 reasoning
//! families reject them with `Unsupported value: 'temperature'` whenever
//! reasoning is on. A field list scattered across the agents could not answer
//! "what will this model actually receive?" without reading every call site, and
//! the run's record has to answer exactly that for the model it names.
//!
//! [VANTA-WHY] The table is consulted at ONE place — [`shape_body_for_openai`],
//! called from `post_chat_meta` on the body it is about to send — rather than
//! where the call sites build their extras. A filter applied at the call sites
//! would leave `post_chat_meta`'s own M0 defaults (`max_tokens`, `temperature`)
//! unfiltered, and the two could drift: the record would then describe a body
//! different from the one the server received.
//!
//! [VANTA-WHY] The llama.cpp path does NOT go through this module. Its request
//! bodies are pinned by tests byte for byte, and a "filter" that runs on them is
//! a chance to change them. `post_chat_meta` branches on `cfg.is_openai()` and
//! leaves a local body untouched, recording `requested == sent` instead.
//!
//! ## Where each drop comes from (checked against the OpenAI docs, 2026-10-26)
//!
//! * `top_k`, `min_p` — llama.cpp sampler knobs. `docs/models/gpt-6-luna` and
//!   `docs/models/gpt-6.1-sol` list no such parameter and the Chat Completions
//!   reference defines neither. Sending one is a 400 at best.
//! * `reasoning_budget_tokens` — llama.cpp's own extension. No OpenAI endpoint
//!   defines it (§14.21's budget; §14.27 records the `unsupported` mechanism).
//! * `temperature`, `top_p`, `logprobs`, `top_logprobs` — defined by Chat
//!   Completions, and explicitly to be removed for a reasoning model:
//!   `docs/models/gpt-6-luna` says *"`reasoning.effort` supports `none`, `low`,
//!   `medium` (default), `high`, `xhigh`, and `max`"* and the sampling fields are
//!   rejected while effort is anything but `none`. Vanta never runs these at
//!   effort `none`, so this is an unconditional drop for the two known families.
//! * `presence_penalty`, `frequency_penalty` — standard Chat Completions
//!   parameters, also rejected by these reasoning models.
//!
//! `max_tokens` is not dropped but RENAMED: `max_completion_tokens` is the
//! current field and the legacy spelling is the one that errors on a reasoning
//! model. `seed` is KEPT — the Chat Completions reference defines it, and
//! `system_fingerprint` is documented as its determinism companion. Nothing in
//! the docs says these models ignore it; the live probe
//! (`scripts/openai_probe.py`, the operator's call) is what would move it to
//! `unsupported`.
//!
//! ## What is deliberately NOT modelled
//!
//! * **Tools.** Neither family is sent any: Sol's page says *"Use the Responses
//!   API for tool calling. Chat Completions is supported without tool calling"*
//!   and Luna's allows CC function calling *only* with effort `none`. Vanta
//!   sends no `tools` on either backend, so there is nothing to filter.
//! * **`response_format`.** Both list "Structured outputs: Supported", so the
//!   call sites' `{"type": "json_object"}` is carried as-is.
//! * **`stream` / `stream_options`.** Both list "Streaming: Supported" and the
//!   Chat Completions reference defines `stream_options.include_usage` (the last
//!   chunk carries `usage`, and is empty otherwise). No Responses API is needed.
//!
//! ## `reasoning_effort` (§14.28)
//!
//! The effort levels are the one field whose accepted VALUES — not just whose
//! presence — differ between the two families, so they are a per-family list
//! rather than a shared constant. `docs/models/gpt-6-luna` gives the ladder as
//! `none`, `low`, `medium` (default), `high`, `xhigh`, `max`;
//! `docs/models/gpt-6.1-sol` gives `low`, `medium` (default), `high`, `xhigh`,
//! `max` and rejects `none`/`minimal`. The union — the values
//! `VANTA_REASONING_EFFORT` may name — is therefore six names, and each family
//! is checked against its own five-or-six. `minimal` is NOT in the union: it
//! appears in neither ladder, and inventing a level no doc lists is exactly the
//! guess this table exists to avoid.
//!
//! [VANTA-WHY] The effort is handled HERE rather than at the call sites even
//! though §14.21 put the budget there, because the two differ in kind: the
//! budget is an ADDITION, so `apply_reasoning_control` inserting a key is enough
//! for the local backend and the shaping table drops it for the hosted one. The
//! effort is a SUBSTITUTION: all six call sites already hard-code
//! `reasoning_effort: "low"`, and on a hosted model that value is a level nobody
//! asked for. It has to be REMOVED when the run did not ask, and the removal has
//! to happen on the same body the record describes — which is this module's
//! choke point, not the six call sites.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A model family whose request dialect this table knows.
///
/// [VANTA-WHY] Resolved from the model id, not from a config switch: the id is
/// what the request carries, so it is the only thing that can disagree with the
/// endpoint that will read it. `VANTA_LLM_MODEL` selects the family by naming
/// the model — there is no second place to say "this is Luna".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiFamily {
    Gpt6Luna,
    Gpt61Sol,
    /// A model id this table does not know.
    ///
    /// [VANTA-WHY] Not a silent "assume it behaves like the known ones". The
    /// drops that are per-FAMILY claims (sampling rejected *while reasoning is
    /// on*) are then NOT applied, because the table has no evidence for them on
    /// an unknown model, while the drops that are claims about the API itself
    /// (a llama.cpp knob is not a parameter of any OpenAI endpoint) still are.
    /// The record names the family, so an abstention is visible rather than
    /// indistinguishable from a decision.
    Unknown,
}

impl OpenAiFamily {
    /// The family a model id belongs to.
    ///
    /// Longest prefix first, so a future `gpt-6.1-sol-2026-10-01` snapshot lands
    /// on Sol rather than being captured by a shorter, unrelated prefix.
    pub fn from_model_id(model_id: &str) -> Self {
        let id = model_id.trim().to_ascii_lowercase();
        if id.starts_with("gpt-6.1-sol") {
            Self::Gpt61Sol
        } else if id.starts_with("gpt-6-luna") {
            Self::Gpt6Luna
        } else {
            Self::Unknown
        }
    }

    /// The family as the record names it — the model id prefix, or `unknown`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Gpt6Luna => "gpt-6-luna",
            Self::Gpt61Sol => "gpt-6.1-sol",
            Self::Unknown => "unknown",
        }
    }

    /// Whether this family rejects the sampling fields while reasoning is on.
    ///
    /// [VANTA-WHY] The evidence is per-model and is quoted in the module note.
    /// `Unknown` answers `false` on purpose: withholding a parameter is only
    /// right when the docs say to withhold it, and for an unknown model there
    /// are no docs to say so. The cost of being wrong here is a 400 the run
    /// records loudly; the cost of guessing the other way is a silently
    /// different sampling schedule.
    pub fn rejects_sampling_while_reasoning(self) -> bool {
        matches!(self, Self::Gpt6Luna | Self::Gpt61Sol)
    }

    /// The `reasoning_effort` values this family accepts, per its model page.
    ///
    /// [VANTA-WHY] `Unknown` answers the FULL union. The two directions of being
    /// wrong are not symmetric here: an accepted-but-unlisted level is a level
    /// the operator asked for and cannot run, while an unknown model refusing a
    /// level would be this table vetoing a request on evidence it does not have.
    /// The table abstains — it sends what was asked and lets the server answer.
    pub fn accepted_efforts(self) -> &'static [&'static str] {
        match self {
            Self::Gpt6Luna => EFFORT_LADDER,
            // Sol's page omits `none` (and rejects `minimal`), so the ladder
            // shared with Luna has one name fewer.
            Self::Gpt61Sol => SOL_EFFORT_LADDER,
            Self::Unknown => EFFORT_LADDER,
        }
    }

    /// Whether `level` is one of [`Self::accepted_efforts`].
    pub fn accepts_effort(self, level: &str) -> bool {
        self.accepted_efforts().contains(&level)
    }
}

/// Fields that are not Chat Completions parameters at all, whatever the model.
///
/// [VANTA-WHY] These are dropped for EVERY OpenAI family, `Unknown` included: the
/// claim is about the API, not about a model, so a proxy that accepts them is
/// still not something this table should assume. `(key, why)` — the reason is
/// carried so a reader of this file does not have to reconstruct the evidence,
/// and so a test can assert the reason text rather than only the key list.
pub const NOT_A_CHAT_PARAMETER: &[(&str, &str)] = &[
    (
        "top_k",
        "not a Chat Completions parameter (a llama.cpp sampler knob)",
    ),
    (
        "min_p",
        "not a Chat Completions parameter (a llama.cpp sampler knob)",
    ),
    (
        "reasoning_budget_tokens",
        "not a Chat Completions parameter (a llama.cpp sampler knob; §14.21)",
    ),
];

/// Fields the known GPT-6 families reject BECAUSE reasoning is on.
///
/// [VANTA-WHY] Held separately from [`NOT_A_CHAT_PARAMETER`] because these ARE
/// parameters of the API — they are withheld for a per-model reason, and so the
/// `Unknown` family keeps them. Merging the two lists would make the table
/// unable to represent "I do not know this model" as anything but "drop
/// everything", which is the guess this table exists to avoid.
pub const REJECTED_WHILE_REASONING: &[(&str, &str)] = &[
    (
        "temperature",
        "rejected while reasoning is on (Unsupported value: 'temperature')",
    ),
    ("top_p", "rejected while reasoning is on"),
    ("presence_penalty", "rejected while reasoning is on"),
    ("frequency_penalty", "rejected while reasoning is on"),
    ("logprobs", "rejected while reasoning is on"),
    ("top_logprobs", "rejected while reasoning is on"),
];

/// The llama.cpp spelling of the output cap, and the OpenAI one it becomes.
pub const LLAMACPP_OUTPUT_CAP_KEY: &str = "max_tokens";
pub const OPENAI_OUTPUT_CAP_KEY: &str = "max_completion_tokens";

/// The `reasoning_effort` key, under the name both backends spell it (§14.28).
///
/// [VANTA-WHY] Shared by name on purpose: `llama.cpp` and OpenAI happen to agree
/// on the spelling, which is why §14.21's switch could be provider-neutral in the
/// first place. The constant exists so the drop and the replacement cannot drift
/// onto two spellings of the same field.
pub const REASONING_EFFORT_KEY: &str = "reasoning_effort";

/// The `service_tier` key, and the one value this project ever puts in it
/// (§14.31).
///
/// [VANTA-WHY] Here rather than spelled at the insert site because this module is
/// where "which fields does this family accept?" is answered, and `service_tier`
/// is the one such field that is neither dropped nor renamed: Chat Completions
/// defines it, both GPT-6 families document `flex`, and it changes the PRICE
/// rather than the answer. So it is absent from [`unsent_keys`] on purpose, and
/// the constant sits beside the two keys that ARE filtered so a reader can see
/// the tier is the deliberate exception.
///
/// [VANTA-WHY] `default` and unset insert NOTHING — `service_tier: "default"` is
/// the API's own default, so sending it would be a body change that means
/// nothing and would stop an explicitly-`default` arm from being byte-identical
/// to an unset one. Only `flex` is ever spelled into a body.
pub const SERVICE_TIER_KEY: &str = "service_tier";

/// Every level `VANTA_REASONING_EFFORT` may name — the UNION of the two ladders.
///
/// [VANTA-WHY] Ordered as the model pages list them (cheapest first). `none` is
/// here because Luna documents it; Sol's page does not, so the per-family list
/// below is what decides whether a given model may be asked for it.
pub const EFFORT_LADDER: &[&str] = &["none", "low", "medium", "high", "xhigh", "max"];

/// [`EFFORT_LADDER`] without `none` — what `gpt-6.1-sol` accepts.
///
/// [VANTA-WHY] Spelled as a suffix rather than filtered at runtime, so the two
/// lists are visible side by side in one file: the difference between the
/// families is one name, and a reader has to be able to see that at a glance.
pub const SOL_EFFORT_LADDER: &[&str] = &["low", "medium", "high", "xhigh", "max"];

/// Every field this table refuses to send to `family`, `(key, why)`, in a fixed
/// order — the single source of truth `shape_body_for_openai` drops from and
/// `run.summary` names.
pub fn unsent_keys(family: OpenAiFamily) -> Vec<(&'static str, &'static str)> {
    let mut keys = NOT_A_CHAT_PARAMETER.to_vec();
    if family.rejects_sampling_while_reasoning() {
        keys.extend_from_slice(REJECTED_WHILE_REASONING);
    }
    keys
}

/// Whether a run's requested effort survives shaping for `family` (§14.28).
///
/// [VANTA-WHY] The ONE predicate both halves of the §14.28 decision read: the
/// per-body drop inside [`shape_body_for_openai`] and the run-level
/// `request_shape_not_sent` list the driver writes. Written once because the two
/// must agree — a run-level rule that named `reasoning_effort` while the body
/// still carried it (or the reverse) is exactly the drift the request-shaping
/// record exists to make impossible.
///
/// [VANTA-WHY] A level the family rejects and a level nobody asked for answer the
/// same here on purpose: both mean "this body must not carry an effort". The
/// first cannot reach a real run — the driver refuses it at start-up — but the
/// predicate is shared with the run-level record, which must not depend on a
/// check made somewhere else in the pipeline.
pub fn keeps_effort(family: OpenAiFamily, requested_effort: Option<&str>) -> bool {
    requested_effort.is_some_and(|level| family.accepts_effort(level))
}

/// What a request body asked for and what it actually carries (§14.26).
///
/// [VANTA-WHY] Both sides are recorded, not just the outcome. "Which fields did
/// this model receive?" is answerable from the body alone and is therefore not
/// the reason this exists; the reason is "which fields did the harness ASK for
/// and NOT get?" — a run whose `temperature` never reached the model is a
/// different experiment from one whose `temperature` did, and from a record
/// carrying only the sent list the two look identical.
///
/// Every list holds plain key names. A reason with spaces inside a `key=value`
/// summary line would run past the value the readers parse
/// (`scripts/metrics.py::summary_field` stops at whitespace), so the reasons
/// stay in the constants above and only the names travel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapedBody {
    /// The keys the caller's body asked for, in body order.
    pub requested: Vec<String>,
    /// The keys the body carries after shaping, in body order.
    pub sent: Vec<String>,
    /// The keys that were dropped, in body order.
    pub dropped: Vec<String>,
    /// `from->to` for every renamed key, in a fixed order.
    pub renamed: Vec<String>,
}

impl ShapedBody {
    /// The record for a body nothing was changed about.
    ///
    /// [VANTA-WHY] Used on the llama.cpp path, where the two sides are equal by
    /// construction. Recording `requested == sent` there is not a formality: it
    /// is what makes the local and hosted arms comparable in one table, and it
    /// is the only way a reader can tell "nothing was filtered" from "the
    /// backend records nothing".
    pub fn native(body: &Map<String, Value>) -> Self {
        let keys: Vec<String> = body.keys().cloned().collect();
        Self {
            requested: keys.clone(),
            sent: keys,
            dropped: Vec::new(),
            renamed: Vec::new(),
        }
    }

    /// The four fields as one `key=value` fragment for a session-log row.
    ///
    /// An empty list renders `none` rather than an empty value, so the field is
    /// always present and always readable — `dropped_fields=` would look like a
    /// truncated line rather than "nothing was dropped".
    pub fn render(&self) -> String {
        fn list(items: &[String]) -> String {
            if items.is_empty() {
                "none".to_string()
            } else {
                items.join(",")
            }
        }
        format!(
            "requested_fields={} sent_fields={} dropped_fields={} renamed_fields={}",
            list(&self.requested),
            list(&self.sent),
            list(&self.dropped),
            list(&self.renamed)
        )
    }

    /// The recomputed set of keys the body carries, or `false` if the run this
    /// record describes could not have produced it.
    ///
    /// [VANTA-WHY] A self-check, not a display helper: it is what lets a test
    /// assert `requested - dropped + renamed == sent` on a REAL recorded row
    /// rather than on a table the test built itself. The arithmetic is the
    /// definition of "nothing was dropped silently".
    pub fn is_consistent(&self) -> bool {
        let mut expected: Vec<String> = self
            .requested
            .iter()
            .filter(|key| !self.dropped.contains(key))
            .filter(|key| !self.renamed.iter().any(|pair| pair.starts_with(&format!("{key}->"))))
            .cloned()
            .collect();
        for pair in &self.renamed {
            if let Some((_, to)) = pair.split_once("->") {
                expected.push(to.to_string());
            }
        }
        expected.sort();
        let mut sent = self.sent.clone();
        sent.sort();
        expected == sent
    }
}

/// Rewrite a llama.cpp-shaped body into what `family`'s endpoint accepts, and
/// report both sides (§14.26, §14.28).
///
/// Drops every key of [`unsent_keys`], renames `max_tokens` to
/// `max_completion_tokens`, and decides the `reasoning_effort` (§14.28).
/// Nothing else is touched: the two known families accept every remaining field,
/// and inventing further edits here would be this table guessing at behaviour no
/// doc states.
///
/// `requested_effort` is the run's `VANTA_REASONING_EFFORT`, or `None` when the
/// switch was unset. It is a parameter rather than a read of the llm module's
/// global so the decision is visible at the call site and a test can exercise
/// every combination without touching process state.
pub fn shape_body_for_openai(
    body: &mut Map<String, Value>,
    family: OpenAiFamily,
    requested_effort: Option<&str>,
) -> ShapedBody {
    let requested: Vec<String> = body.keys().cloned().collect();
    let mut dropped = Vec::new();
    let mut renamed = Vec::new();

    for (key, _why) in unsent_keys(family) {
        if body.remove(key).is_some() {
            dropped.push(key.to_string());
        }
    }
    // [VANTA-WHY] §14.28 — the call sites' `reasoning_effort: "low"` is a
    // llama.cpp fact, not this run's request: it is sent because a local server
    // ignores it (§14.21's `llamacpp:noop`), and on a hosted reasoning model it
    // would be a real intervention nobody chose. So the field goes out ONLY when
    // the run asked for a level this family accepts, and the value it carries is
    // the one `apply_reasoning_control` substituted earlier in the same request.
    //
    // [VANTA-WHY] Refusing an accepted-but-unasked-for level and refusing a
    // rejected one land in the SAME branch deliberately: both mean "this body
    // must not carry an effort", and both are recorded as a drop. A level the
    // family rejects is a `CONFIG_ERROR` at run start (driver), so a real run
    // cannot reach that half — but a filter that would send it if it did is not
    // a filter. The predicate is shared with the run-level rule (`keeps_effort`).
    if !keeps_effort(family, requested_effort) && body.remove(REASONING_EFFORT_KEY).is_some() {
        dropped.push(REASONING_EFFORT_KEY.to_string());
    }
    // [VANTA-WHY] `insert` onto the key we just removed, so the rename cannot
    // leave both spellings in one body — two output caps in one JSON object is
    // the second one silently winning, and the record would then name a cap no
    // server applied.
    if let Some(cap) = body.remove(LLAMACPP_OUTPUT_CAP_KEY) {
        body.insert(OPENAI_OUTPUT_CAP_KEY.to_string(), cap);
        renamed.push(format!(
            "{LLAMACPP_OUTPUT_CAP_KEY}->{OPENAI_OUTPUT_CAP_KEY}"
        ));
    }

    ShapedBody {
        requested,
        sent: body.keys().cloned().collect(),
        dropped,
        renamed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A body shaped exactly the way `post_chat_meta` shapes one: the M0
    /// defaults, a call site's extras, and a seed.
    fn a_llamacpp_shaped_body() -> Map<String, Value> {
        let body = json!({
            "model": "gpt-6-luna",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32768,
            "temperature": 0.2,
            "stream": true,
            "stream_options": {"include_usage": true},
            "response_format": {"type": "json_object"},
            "top_k": 20,
            "min_p": 0.0,
            "reasoning_effort": "low",
            "seed": 4242,
        });
        body.as_object().unwrap().clone()
    }

    #[test]
    fn the_family_is_read_from_the_model_id_and_the_longest_prefix_wins() {
        assert_eq!(
            OpenAiFamily::from_model_id("gpt-6-luna"),
            OpenAiFamily::Gpt6Luna
        );
        assert_eq!(
            OpenAiFamily::from_model_id("gpt-6.1-sol"),
            OpenAiFamily::Gpt61Sol
        );
        // A dated snapshot is the same family.
        assert_eq!(
            OpenAiFamily::from_model_id("gpt-6.1-sol-2026-10-01"),
            OpenAiFamily::Gpt61Sol
        );
        // Case and stray whitespace come from the environment.
        assert_eq!(
            OpenAiFamily::from_model_id("  GPT-6-Luna  "),
            OpenAiFamily::Gpt6Luna
        );
        // The local default model id, and anything else this table has not
        // been taught, must NOT be mistaken for a known family.
        assert_eq!(
            OpenAiFamily::from_model_id("qwen3.8-27b"),
            OpenAiFamily::Unknown
        );
        assert_eq!(OpenAiFamily::from_model_id(""), OpenAiFamily::Unknown);
        assert_eq!(OpenAiFamily::from_model_id("gpt-6"), OpenAiFamily::Unknown);
    }

    #[test]
    fn the_two_known_families_drop_the_sampling_fields_and_an_unknown_one_keeps_them() {
        assert!(OpenAiFamily::Gpt6Luna.rejects_sampling_while_reasoning());
        assert!(OpenAiFamily::Gpt61Sol.rejects_sampling_while_reasoning());
        // [VANTA-WHY] The abstention is the point: withholding a parameter the
        // docs do not say to withhold is the silent-guess failure mode.
        assert!(!OpenAiFamily::Unknown.rejects_sampling_while_reasoning());

        let luna: Vec<&str> = unsent_keys(OpenAiFamily::Gpt6Luna)
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        for key in [
            "top_k",
            "min_p",
            "reasoning_budget_tokens",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
        ] {
            assert!(luna.contains(&key), "Luna must not be sent `{key}`");
        }

        let unknown: Vec<&str> = unsent_keys(OpenAiFamily::Unknown)
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(unknown, vec!["top_k", "min_p", "reasoning_budget_tokens"]);
    }

    #[test]
    fn shaping_drops_the_unparameterized_keys_renames_the_cap_and_keeps_the_rest() {
        let mut body = a_llamacpp_shaped_body();
        // The run asked for `low` — the level the body already carried — so the
        // effort is kept (§14.28).
        let shaped = shape_body_for_openai(&mut body, OpenAiFamily::Gpt6Luna, Some("low"));

        for key in ["top_k", "min_p", "temperature", "presence_penalty"] {
            assert!(
                !body.contains_key(key),
                "`{key}` must not survive shaping: {body:?}"
            );
        }
        assert!(
            !body.contains_key(LLAMACPP_OUTPUT_CAP_KEY),
            "the legacy cap spelling must not survive alongside the new one"
        );
        assert_eq!(body[OPENAI_OUTPUT_CAP_KEY], json!(32768));

        // The fields the model does accept are untouched.
        assert_eq!(body["model"], json!("gpt-6-luna"));
        assert_eq!(body["seed"], json!(4242));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["reasoning_effort"], json!("low"));
        assert_eq!(body["response_format"], json!({"type": "json_object"}));

        assert_eq!(
            shaped.renamed,
            vec!["max_tokens->max_completion_tokens".to_string()]
        );
        assert!(shaped.dropped.contains(&"temperature".to_string()));
        // The record and the body cannot disagree about the end state.
        assert!(shaped.is_consistent(), "record: {shaped:?}");
        assert_eq!(
            shaped.sent,
            body.keys().cloned().collect::<Vec<String>>(),
            "the `sent` list must be the body's own keys"
        );
        // ... and both sides of the question are on the record.
        assert!(shaped.requested.contains(&"temperature".to_string()));
        assert!(!shaped.sent.contains(&"temperature".to_string()));
        assert!(shaped.render().contains("dropped_fields="));
        // The fragment is FOUR whitespace-separated `key=value` tokens, one per
        // list, and no list may contribute a space of its own: the run-summary
        // reader splits the whole line on whitespace, so a space inside a value
        // would run on into the next key (§14.11's dot hazard, one character over).
        let rendered = shaped.render();
        let tokens: Vec<&str> = rendered.split_whitespace().collect();
        assert_eq!(tokens.len(), 4, "one token per key: {rendered}");
        assert!(tokens.iter().all(|token| token.contains('=')), "{rendered}");
    }

    #[test]
    fn a_body_that_asked_for_nothing_unusual_shapes_to_itself() {
        // A minimal body: no sampler keys at all, so the drop list has nothing
        // to remove and only the cap is renamed.
        let body = json!({"model": "gpt-6-luna", "max_tokens": 8192});
        let mut body = body.as_object().unwrap().clone();
        let shaped = shape_body_for_openai(&mut body, OpenAiFamily::Gpt6Luna, None);
        assert!(shaped.dropped.is_empty(), "nothing was there to drop");
        assert_eq!(
            shaped.renamed,
            vec!["max_tokens->max_completion_tokens".to_string()]
        );
        assert!(shaped.is_consistent());
    }

    #[test]
    fn the_native_record_says_requested_equals_sent_and_drops_nothing() {
        let body = a_llamacpp_shaped_body();
        let native = ShapedBody::native(&body);
        assert_eq!(native.requested, native.sent);
        assert!(native.dropped.is_empty());
        assert!(native.renamed.is_empty());
        assert!(native.is_consistent());
        assert!(native.render().contains("dropped_fields=none"));
        assert!(native.render().contains("renamed_fields=none"));
    }

    #[test]
    fn is_consistent_rejects_a_record_that_lost_a_key_silently() {
        // The guard has to be able to FAIL: a record whose `sent` list omits a
        // requested key that was never dropped is exactly the drift this field
        // exists to make impossible.
        let mut body = a_llamacpp_shaped_body();
        let mut shaped = shape_body_for_openai(&mut body, OpenAiFamily::Gpt6Luna, None);
        assert!(shaped.is_consistent());
        shaped.sent.push("a_key_no_body_carried".to_string());
        assert!(!shaped.is_consistent());
    }

    /// §14.28: the effort goes out only when the RUN asked for a level this
    /// family accepts — the call sites' `low` is a llama.cpp fact.
    #[test]
    fn the_effort_is_dropped_unless_the_run_asked_for_a_level_that_is_accepted() {
        // (a) The switch was unset: the call site's `low` must NOT reach a hosted
        // model, and the drop must be on the record rather than silent.
        let mut body = a_llamacpp_shaped_body();
        let shaped = shape_body_for_openai(&mut body, OpenAiFamily::Gpt6Luna, None);
        assert!(
            !body.contains_key(REASONING_EFFORT_KEY),
            "an unasked-for effort is a real intervention on a hosted model: {body:?}"
        );
        assert!(
            shaped.dropped.contains(&REASONING_EFFORT_KEY.to_string()),
            "the drop is a fact about the run and belongs on the record: {shaped:?}"
        );
        assert!(shaped.requested.contains(&REASONING_EFFORT_KEY.to_string()));
        assert!(shaped.is_consistent());

        // (b) The run asked, the family accepts: the level is kept as sent.
        let mut body = a_llamacpp_shaped_body();
        let shaped = shape_body_for_openai(&mut body, OpenAiFamily::Gpt6Luna, Some("high"));
        assert!(body.contains_key(REASONING_EFFORT_KEY));
        assert!(
            !shaped.dropped.contains(&REASONING_EFFORT_KEY.to_string()),
            "an accepted, asked-for level is not a drop: {shaped:?}"
        );
        assert!(shaped.is_consistent());

        // (c) The run asked, the family rejects it. A real run fails at config
        // time; the shaper must still refuse to send it.
        let mut body = a_llamacpp_shaped_body();
        let shaped = shape_body_for_openai(&mut body, OpenAiFamily::Gpt61Sol, Some("none"));
        assert!(
            !body.contains_key(REASONING_EFFORT_KEY),
            "Sol rejects `none`: {body:?}"
        );
        assert!(shaped.dropped.contains(&REASONING_EFFORT_KEY.to_string()));
        assert!(shaped.is_consistent());
    }

    /// §14.31 — the tier is the one body field that is filtered by NEITHER list.
    #[test]
    fn the_service_tier_is_never_dropped_on_any_family() {
        // [VANTA-WHY] The guard has to be able to FAIL: if a later edit moved
        // `service_tier` into `NOT_A_CHAT_PARAMETER` (it is not a llama.cpp knob)
        // or into `REJECTED_WHILE_REASONING` (both families document `flex`), a
        // hosted run that asked for the half-price tier would silently be served
        // at full price while its record still said `flex`. That is the exact
        // shape of a wrong cost column.
        for family in [
            OpenAiFamily::Gpt6Luna,
            OpenAiFamily::Gpt61Sol,
            OpenAiFamily::Unknown,
        ] {
            let mut body = a_llamacpp_shaped_body();
            body.insert(SERVICE_TIER_KEY.to_string(), json!("flex"));
            let shaped = shape_body_for_openai(&mut body, family, None);
            assert!(
                body.contains_key(SERVICE_TIER_KEY),
                "`{SERVICE_TIER_KEY}` is a Chat Completions parameter: {body:?}"
            );
            assert!(
                !shaped.dropped.contains(&SERVICE_TIER_KEY.to_string()),
                "the tier must not be on the drop list: {shaped:?}"
            );
            assert!(shaped.requested.contains(&SERVICE_TIER_KEY.to_string()));
            assert!(shaped.sent.contains(&SERVICE_TIER_KEY.to_string()));
            assert!(shaped.is_consistent(), "record: {shaped:?}");
        }
    }

    /// §14.28: the two ladders, spelled out here rather than read from the table
    /// so a drifted table cannot keep passing against itself.
    #[test]
    fn each_family_lists_its_own_effort_ladder_and_minimal_is_in_neither() {
        for level in ["none", "low", "medium", "high", "xhigh", "max"] {
            assert!(
                OpenAiFamily::Gpt6Luna.accepts_effort(level),
                "Luna documents `{level}`"
            );
        }
        for level in ["low", "medium", "high", "xhigh", "max"] {
            assert!(
                OpenAiFamily::Gpt61Sol.accepts_effort(level),
                "Sol documents `{level}`"
            );
        }
        assert!(
            !OpenAiFamily::Gpt61Sol.accepts_effort("none"),
            "Sol's page omits `none`"
        );
        // `minimal` appears in neither ladder, so it is in neither enum nor
        // accepted set — the union is the six names above and nothing more.
        for family in [
            OpenAiFamily::Gpt6Luna,
            OpenAiFamily::Gpt61Sol,
            OpenAiFamily::Unknown,
        ] {
            assert!(!family.accepts_effort("minimal"));
            assert!(!family.accepts_effort(""));
            assert!(!family.accepts_effort("LOW"), "levels are lower-case");
        }
        assert_eq!(EFFORT_LADDER.len(), 6);
        assert_eq!(SOL_EFFORT_LADDER.len(), 5);
        // The unknown family abstains: it accepts the whole union rather than
        // vetoing a documented level on evidence it does not have.
        assert_eq!(
            OpenAiFamily::Unknown.accepted_efforts(),
            EFFORT_LADDER,
            "an unknown model must not be refused a level by this table"
        );
    }
}
