//! The versioned price table and the per-call cost arithmetic
//! (plans/bugsinpy.md §14.29).
//!
//! [VANTA-WHY] A cost figure is a claim about money, so every number here is a
//! claim about a SAFE that has to be citable. Each row therefore carries the
//! source URL and the date it was read, rather than a bare float: a table of
//! floats with no provenance cannot be checked, and a price that moved under a
//! reader would silently re-price a batch that already ran. The date is not
//! decoration — it travels into `cost_basis` on every priced call, so a row in a
//! manifest says WHICH price table priced it.
//!
//! [VANTA-WHY] One file, both families, both tiers. `scripts/metrics.py` and the
//! Rust pipeline must agree about what a token cost, and the only way to
//! guarantee that with two languages in the loop is to have ONE table. It lives
//! here, in the language that measures the tokens (the per-call counts exist
//! nowhere else), and Python reads the RESULT — `run.summary`'s `cost_usd=` —
//! instead of recomputing it from prices it would have to keep in sync.
//!
//! [VANTA-WHY] Prices are per 1,000,000 tokens and are divided out at the very
//! end, in one place (`cost_usd`), so a rate and its unit cannot drift apart
//! between rows.
//!
//! ## What the arithmetic bills, and what it deliberately does not
//!
//! * **Uncached input** — `prompt_tokens - cached_tokens`. `cached_tokens` is a
//!   SUBSET of `prompt_tokens` (the `usage` block reports the cache read as part
//!   of the prompt), so billing the prompt total at the input rate would charge
//!   the cached share at full price and delete the discount the table exists to
//!   express.
//! * **Cached input** — `cached_tokens`, at the cached rate.
//! * **Output** — `completion_tokens`, at the output rate. `reasoning_tokens` is
//!   a SUBSET of `completion_tokens` (the `usage` block nests it under
//!   `completion_tokens_details`), so it is recorded for the report and NOT added
//!   to the bill a second time.
//! * **Cache WRITES** are documented at 1.25x the uncached input rate for these
//!   models, but they are NOT in this table: the `usage` block these endpoints
//!   return reports no write count (there is no `cache_write_tokens` beside
//!   `cached_tokens`), so there is no measured quantity to multiply. A column no
//!   response can fill would be a number this table invented.
//! * **The long-context reprice** (an input past the model's own threshold
//!   re-prices the whole request) is likewise documented and NOT modelled, for
//!   the same reason: the source this table was read from states the rule but not
//!   the repriced rates, and inventing them would put a fabricated number in a
//!   cost column. A run that crosses the threshold still records its
//!   `llm_prompt_tokens`, so the crossing is visible; see
//!   [`LONG_CONTEXT_THRESHOLD_TOKENS`].
//!
//! [VANTA-WHY] The local backend is not priced at all, and that is a fact rather
//! than a gap: `LlmConfig.backend` names the CONTRACT, and a llama.cpp server is
//! a process on this machine — no tokens are sold. `not_billed` is recorded so a
//! batch table can tell "zero because it was free" from "zero because the record
//! forgot".

use serde::{Deserialize, Serialize};

use crate::llm::capabilities::OpenAiFamily;

/// The date every row below was read, and the page it was read from.
///
/// [VANTA-WHY] One constant pair for the table rather than a copy per row: the
/// rows were read together, and a per-row date that was written twice is a fact
/// that can disagree with itself. A row that grows its own source (a negotiated
/// price, a different vendor) gets its own field in [`PriceRow`] instead.
pub const PRICE_TABLE_DATE: &str = "2026-10-26";
pub const PRICE_TABLE_SOURCE: &str = "https://platform.openai.com/docs/pricing";

/// Which service tier a request was served on (§14.30, item 7).
///
/// [VANTA-WHY] A tier is a PRICE, not a queue: `flex` is documented at half the
/// standard rates for exactly the same tokens, so the tier has to be known before
/// a request is priced. Recording it per row (item 7) lets a batch table split
/// cheap-and-slow rows from full-price ones without re-deriving which requests
/// asked for `flex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceTier {
    /// `service_tier=default|auto` — the standard rates.
    Standard,
    /// `service_tier=flex` — half the standard rates.
    Flex,
}

impl ServiceTier {
    /// The tier as the record spells it, matching `VANTA_OPENAI_SERVICE_TIER`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Flex => "flex",
        }
    }
}

/// One price row: a family, a tier, and what a million tokens of each kind cost.
///
/// [VANTA-WHY] `source` and `date` are the provenance the section's own note
/// requires "per entry", and they are FIELDS rather than a doc comment so a test
/// can assert them and `cost_basis` can name them. A row whose provenance lived
/// only in prose is a row a future edit can re-price without touching a claim.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriceRow {
    pub family: OpenAiFamily,
    pub service_tier: ServiceTier,
    /// USD per 1,000,000 uncached input tokens.
    pub input_per_mtok: f64,
    /// USD per 1,000,000 cached input tokens (a cache READ).
    pub cached_input_per_mtok: f64,
    /// USD per 1,000,000 output tokens, `reasoning_tokens` included.
    pub output_per_mtok: f64,
    pub source: &'static str,
    pub date: &'static str,
}

/// The input length past which the models re-price the whole request.
///
/// [VANTA-WHY] Recorded and NOT used, deliberately. The rule is real and the
/// threshold is published, but the repriced RATES were not in the source this
/// table was read from, and a cost column is the last place to guess. Keeping the
/// number here means the caveat in the module note is checkable against a
/// constant rather than being a sentence somebody has to remember, and item 5's
/// smoke run (`VANTA_LLM_CONTEXT_TOKENS=131072`) sits well below it.
pub const LONG_CONTEXT_THRESHOLD_TOKENS: u64 = 272_000;

/// Every (family, tier) pair this project can price, in a fixed order.
///
/// [VANTA-WHY] `standard` first, `flex` second, Luna before Sol: a fixed order
/// means a reader diffing this file sees a rate change rather than a reordering,
/// and a test can assert the whole table without a sort.
pub const PRICE_TABLE: &[PriceRow] = &[
    PriceRow {
        family: OpenAiFamily::Gpt6Luna,
        service_tier: ServiceTier::Standard,
        input_per_mtok: 0.10,
        cached_input_per_mtok: 0.01,
        output_per_mtok: 0.50,
        source: PRICE_TABLE_SOURCE,
        date: PRICE_TABLE_DATE,
    },
    PriceRow {
        family: OpenAiFamily::Gpt6Luna,
        service_tier: ServiceTier::Flex,
        // `flex` is the standard rates halved, applied per column.
        input_per_mtok: 0.05,
        cached_input_per_mtok: 0.005,
        output_per_mtok: 0.25,
        source: PRICE_TABLE_SOURCE,
        date: PRICE_TABLE_DATE,
    },
    PriceRow {
        family: OpenAiFamily::Gpt61Sol,
        service_tier: ServiceTier::Standard,
        input_per_mtok: 2.00,
        cached_input_per_mtok: 0.10,
        output_per_mtok: 10.00,
        source: PRICE_TABLE_SOURCE,
        date: PRICE_TABLE_DATE,
    },
    PriceRow {
        family: OpenAiFamily::Gpt61Sol,
        service_tier: ServiceTier::Flex,
        input_per_mtok: 1.00,
        cached_input_per_mtok: 0.05,
        output_per_mtok: 5.00,
        source: PRICE_TABLE_SOURCE,
        date: PRICE_TABLE_DATE,
    },
];

/// The row that prices `family` on `tier`, or `None` when this table has no
/// evidence for that pair.
///
/// [VANTA-WHY] `None` rather than a fallback to the nearest family: an unknown
/// model's tokens have no price this project can cite, and pricing them at Luna's
/// rate would put a fabricated figure in the batch's cost column. The caller
/// records `unpriced_family` instead, which is a hole a reader can see.
pub fn price_row(family: OpenAiFamily, tier: ServiceTier) -> Option<&'static PriceRow> {
    PRICE_TABLE
        .iter()
        .find(|row| row.family == family && row.service_tier == tier)
}

/// The `cost_basis` a run records when the backend sells nothing (§14.29).
pub const COST_BASIS_NOT_BILLED: &str = "not_billed";
/// The `cost_basis` a call records when the server published no `usage` block, so
/// the only prompt figure is the character-derived estimate.
pub const COST_BASIS_NO_USAGE: &str = "no_usage";
/// The `cost_basis` a call records when no row prices its family/tier pair.
pub const COST_BASIS_UNPRICED_FAMILY: &str = "unpriced_family";

/// What one priced call cost, and how the record names the basis it used.
///
/// [VANTA-WHY] `(usd, basis)` and not a bare float: `unset` is a fact about the
/// record ("no number was computed") while `0.0` is a fact about the money ("this
/// call was priced and came to nothing"), and a cost column that renders both as
/// `0` cannot be audited. The basis is the row identity in `family/tier/date`
/// form, so a manifest row points at the exact entry that priced it.
pub type CallCost = (Option<f64>, String);

/// Price one call's measured counts.
///
/// `prompt`, `cached` and `completion` are the `usage` block's own numbers —
/// `cached` is the subset already inside `prompt`, and `reasoning` is NOT a
/// parameter because it is a subset of `completion` and billing it again would
/// double-charge the same tokens (see the module note).
pub fn cost_usd(
    row: &PriceRow,
    prompt_tokens: u64,
    cached_tokens: u64,
    completion_tokens: u64,
) -> f64 {
    // `saturating_sub`, not `-`: a provider that ever reported more cached tokens
    // than prompt tokens would otherwise make this subtraction panic in debug and
    // wrap in release, turning a billing oddity into a wrong bill or a dead run.
    let uncached_input = prompt_tokens.saturating_sub(cached_tokens);
    let per_token = |count: u64, rate_per_mtok: f64| count as f64 * rate_per_mtok / 1_000_000.0;
    per_token(uncached_input, row.input_per_mtok)
        + per_token(cached_tokens, row.cached_input_per_mtok)
        + per_token(completion_tokens, row.output_per_mtok)
}

/// Price a call by FAMILY and TIER, or explain why no price exists.
///
/// [VANTA-WHY] The `estimate`-sourced case is a parameter rather than something
/// the caller resolves first, because "the only prompt figure is a character
/// guess" and "the server published numbers" are two different reasons to refuse
/// a price and the record has to name the right one. A cost derived from the
/// estimate would be inflated by the same factor the estimate is (§4.3 measured
/// it ~1.5x high), and the paper's cost column is the one number nobody can
/// re-derive from the tokens afterwards.
pub fn cost_for_family(
    family: OpenAiFamily,
    tier: ServiceTier,
    usage_source: &str,
    prompt_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    completion_tokens: Option<u64>,
) -> CallCost {
    if usage_source == "estimate" {
        return (None, COST_BASIS_NO_USAGE.to_string());
    }
    let Some(row) = price_row(family, tier) else {
        return (None, COST_BASIS_UNPRICED_FAMILY.to_string());
    };
    let basis = format!("{}/{}/{}", family.name(), tier.name(), row.date);
    let Some(prompt) = prompt_tokens else {
        // Measured completion with no measured prompt is not a shape the usage
        // block produces; refusing to price it is the same call as refusing the
        // estimate above rather than billing the output alone as if the input
        // were free.
        return (None, COST_BASIS_NO_USAGE.to_string());
    };
    let cached = cached_tokens.unwrap_or(0).min(prompt);
    let completion = completion_tokens.unwrap_or(0);
    (Some(cost_usd(row, prompt, cached, completion)), basis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_and_tier_pair_in_the_table_is_priceable_and_provenanced() {
        for family in [OpenAiFamily::Gpt6Luna, OpenAiFamily::Gpt61Sol] {
            for tier in [ServiceTier::Standard, ServiceTier::Flex] {
                let row = price_row(family, tier)
                    .unwrap_or_else(|| panic!("no price row for {family:?}/{tier:?}"));
                assert_eq!(row.source, PRICE_TABLE_SOURCE, "a row must name its source");
                assert_eq!(row.date, PRICE_TABLE_DATE, "a row must name its date");
                assert!(row.input_per_mtok > 0.0 && row.output_per_mtok > 0.0);
                // A cached read is always cheaper than an uncached one, or the
                // discount the column expresses does not exist.
                assert!(row.cached_input_per_mtok < row.input_per_mtok);
            }
        }
        // The unknown family has no row: this table prices what it has evidence
        // for, and refuses the rest loudly via `unpriced_family`.
        assert!(price_row(OpenAiFamily::Unknown, ServiceTier::Standard).is_none());
    }

    #[test]
    fn flex_is_exactly_half_of_standard_on_every_column() {
        for family in [OpenAiFamily::Gpt6Luna, OpenAiFamily::Gpt61Sol] {
            let standard = price_row(family, ServiceTier::Standard).unwrap();
            let flex = price_row(family, ServiceTier::Flex).unwrap();
            assert_eq!(flex.input_per_mtok, standard.input_per_mtok / 2.0);
            assert_eq!(flex.cached_input_per_mtok, standard.cached_input_per_mtok / 2.0);
            assert_eq!(flex.output_per_mtok, standard.output_per_mtok / 2.0);
        }
    }

    #[test]
    fn a_million_uncached_input_tokens_cost_the_input_rate_exactly() {
        let row = price_row(OpenAiFamily::Gpt6Luna, ServiceTier::Standard).unwrap();
        let usd = cost_usd(row, 1_000_000, 0, 0);
        assert!((usd - 0.10).abs() < 1e-12, "got {usd}");
    }

    #[test]
    fn cached_tokens_are_billed_at_the_cached_rate_and_not_twice() {
        // 777 prompt of which 700 cached, 123 out: Luna standard.
        let row = price_row(OpenAiFamily::Gpt6Luna, ServiceTier::Standard).unwrap();
        let usd = cost_usd(row, 777, 700, 123);
        let expected = 77.0 * 0.10 / 1e6 + 700.0 * 0.01 / 1e6 + 123.0 * 0.50 / 1e6;
        assert!((usd - expected).abs() < 1e-15, "got {usd}, expected {expected}");
        // Billing the prompt whole would be strictly more expensive: the cached
        // share must be a discount, never a second charge on top of it.
        assert!(usd < cost_usd(row, 777, 0, 123));
    }

    #[test]
    fn a_cached_count_above_the_prompt_cannot_panic_or_overbill() {
        // Not a shape a provider should send; the point is that a billing oddity
        // must not be a crash and must not charge negative input. `cost_usd` is the
        // raw arithmetic on the numbers it is handed — it cannot know which of them
        // is a subset of which — so the clamp to the prompt lives in
        // `cost_for_family`, and each layer is asserted for its own contract.
        let row = price_row(OpenAiFamily::Gpt61Sol, ServiceTier::Standard).unwrap();
        let usd = cost_usd(row, 10, 999, 0);
        // `saturating_sub` leaves no uncached input to bill, so no term is
        // negative; the cached share is still billed at the cached rate.
        let expected = 999.0 * 0.10 / 1e6;
        assert!((usd - expected).abs() < 1e-15, "got {usd}");
        assert!(usd >= 0.0, "no term may be negative: got {usd}");

        // Through the public entry the same oddity is clamped to the prompt, so a
        // call is never billed for more cached tokens than it had.
        let (usd, basis) = cost_for_family(
            OpenAiFamily::Gpt61Sol,
            ServiceTier::Standard,
            "server-usage",
            Some(10),
            Some(999),
            Some(0),
        );
        let expected = 10.0 * 0.10 / 1e6;
        assert!((usd.unwrap() - expected).abs() < 1e-15, "got {usd:?}");
        assert_eq!(basis, "gpt-6.1-sol/standard/2026-10-26");
    }

    #[test]
    fn the_basis_names_the_family_the_tier_and_the_date() {
        let (usd, basis) = cost_for_family(
            OpenAiFamily::Gpt61Sol,
            ServiceTier::Flex,
            "server-usage",
            Some(1_000_000),
            Some(0),
            Some(1_000_000),
        );
        assert_eq!(basis, "gpt-6.1-sol/flex/2026-10-26");
        let expected = 1.00 + 5.00;
        assert!((usd.unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn an_estimate_sourced_call_is_refused_rather_than_priced() {
        let (usd, basis) = cost_for_family(
            OpenAiFamily::Gpt6Luna,
            ServiceTier::Standard,
            "estimate",
            Some(4_662),
            None,
            None,
        );
        assert!(usd.is_none(), "a character-derived count must not be priced");
        assert_eq!(basis, COST_BASIS_NO_USAGE);
    }

    #[test]
    fn an_unpriced_family_says_so_instead_of_borrowing_a_neighbours_rate() {
        let (usd, basis) = cost_for_family(
            OpenAiFamily::Unknown,
            ServiceTier::Standard,
            "server-usage",
            Some(1_000),
            None,
            Some(10),
        );
        assert!(usd.is_none());
        assert_eq!(basis, COST_BASIS_UNPRICED_FAMILY);
    }

    #[test]
    fn the_long_context_threshold_sits_above_the_smoke_runs_window() {
        // 131072 is the window item 5's smoke run is launched with; the rule the
        // module note declines to model must not be reachable inside it.
        assert!(u64::from(131_072_u32) < LONG_CONTEXT_THRESHOLD_TOKENS);
    }
}
