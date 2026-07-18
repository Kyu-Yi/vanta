#[derive(Debug)]
pub enum FailureClass {
    Infrastructure,
    Static,
    Logic,
    Unknown,
}

#[derive(Debug)]
pub enum TestIntent {
    BugReproduction,
    FeatureAcceptance,
    BehaviorPreservation,
}

#[derive(Debug)]
pub enum VerificationOutcome {
    VerifiedWithinBounds,
    Counterexample,
    Inconclusive,
    Unsupported,
    Timeout,
    ToolFailure,
}
