// ============================================================================
// GATEKEEPER — deterministic approve/reject on required signals (Spec §4, §6)
// ============================================================================
//
// [VANTA-WHY] Spec §4: "Gatekeeper: Opens a PR only when every required signal
// passes under repository policy." The Gatekeeper NEVER trusts the LLM. Every
// decision rests on deterministic evidence: test results, static analysis diff,
// and (V2) verifier outcomes.
//
// V1 decision rule (no verifier track yet — that's M4):
//   APPROVE if:
//     - All shared-set tests PASS on the patched code
//     - No NEW static errors introduced (ruff/mypy diff vs baseline)
//     - VerificationPolicy allows it (Counterexample would reject, but we have
//       no verifier yet, so this is a no-op for V1)
//   REJECT otherwise.
//
// [VANTA-WHY] This is the component that makes Vanta a VERIFICATION LAYER
// rather than a repair loop. A patch from ANY source (Vanta's Creator, another
// agent, a human) goes through the same Gatekeeper.
// ============================================================================

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::core::VerificationPolicy;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GatekeeperVerdict {
    Approved,
    Rejected { reason: String },
}

pub struct Gatekeeper {
    policy: VerificationPolicy,
}

impl Gatekeeper {
    pub fn new(policy: VerificationPolicy) -> Self {
        Gatekeeper { policy }
    }

    /// Evaluate a patch against the required signals.
    ///
    /// - `all_shared_tests_passed`: every test in the fixed shared set passed
    ///   on the patched code.
    /// - `new_static_errors`: true if ruff/mypy found NEW errors in the patched
    ///   code that weren't in the baseline.
    pub fn evaluate(
        &self,
        all_shared_tests_passed: bool,
        new_static_errors: bool,
    ) -> GatekeeperVerdict {
        // [VANTA-WHY] Required signal 1: all shared tests pass.
        // This is non-negotiable regardless of policy — a patch that breaks
        // existing tests cannot ship.
        if !all_shared_tests_passed {
            warn!("Gatekeeper REJECTED: not all shared tests passed");
            return GatekeeperVerdict::Rejected {
                reason: "Not all shared-set tests passed on patched code".to_string(),
            };
        }

        // [VANTA-WHY] Required signal 2: no new static errors.
        // Pre-existing diagnostics are baseline noise (Spec §5: "only newly
        // introduced diagnostics are patch-caused"). New ones mean the patch
        // introduced a syntax/type/lint problem.
        if new_static_errors {
            warn!("Gatekeeper REJECTED: new static errors introduced");
            return GatekeeperVerdict::Rejected {
                reason: "Patch introduces new static analysis errors (ruff/mypy)".to_string(),
            };
        }

        // [VANTA-WHY] V2 will add verifier outcome checks here:
        //   - Counterexample → always reject (regardless of policy)
        //   - Inconclusive/Unsupported → block if Required, continue if BestEffort
        // For V1 (no verifier track), we only check tests + static analysis.

        info!(
            "Gatekeeper APPROVED: all tests pass, no new static errors (policy={:?})",
            self.policy
        );
        GatekeeperVerdict::Approved
    }
}

impl Default for Gatekeeper {
    fn default() -> Self {
        Gatekeeper::new(VerificationPolicy::BestEffort)
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approve_when_all_pass_no_static_errors() {
        let gk = Gatekeeper::default();
        assert_eq!(
            gk.evaluate(true, false),
            GatekeeperVerdict::Approved
        );
    }

    #[test]
    fn test_reject_when_tests_fail() {
        let gk = Gatekeeper::default();
        match gk.evaluate(false, false) {
            GatekeeperVerdict::Rejected { .. } => {}
            _ => panic!("expected rejection"),
        }
    }

    #[test]
    fn test_reject_when_new_static_errors() {
        let gk = Gatekeeper::default();
        match gk.evaluate(true, true) {
            GatekeeperVerdict::Rejected { .. } => {}
            _ => panic!("expected rejection"),
        }
    }
}
