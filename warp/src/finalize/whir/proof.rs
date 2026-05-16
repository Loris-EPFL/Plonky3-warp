use super::*;

/// PCS proof for the accumulator codeword opening `f_hat(alpha) = mu`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "PcsProof: Serialize + serde::de::DeserializeOwned")]
pub struct WhirAccumulatorOpeningProof<PcsProof> {
    /// Opening proof produced by the underlying multilinear PCS.
    pub pcs_proof: PcsProof,
}

/// WHIR-facing proof of the terminal Boolean PESAT claim.
///
/// The Boolean equation over the terminal message subspace is reduced by a
/// sumcheck to one terminal witness claim. In systematic RS mode, the local
/// prover obtains that message from the committed codeword's systematic
/// coordinates and checks that its terminal codeword is `C(w)`. Verifier-side
/// codeword consistency still belongs to the root exact-codeword bridge, or to
/// an equivalent backend guarantee.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    bound = "EF: Serialize + serde::de::DeserializeOwned, PcsProof: Serialize + serde::de::DeserializeOwned"
)]
pub struct WhirPesatProof<EF, PcsProof> {
    /// Sumcheck over the PESAT witness hypercube.
    pub decider_sumcheck: SumcheckProof<EF>,
    /// Claimed terminal witness value at the sampled point.
    pub terminal_values: Vec<EF>,
    /// PCS opening proof for terminal values on the systematic RS oracle.
    pub pcs_proof: PcsProof,
}

/// WHIR-facing terminal WARP proof fragment.
///
/// This is the reusable assembly point for the WHIR-native terminal checks.
/// The two subproofs certify the two non-local equations against the same
/// accumulator commitment:
///
/// ```text
///     f_hat(alpha) = mu
///     BooleanPb(beta, w_sys(f)) = eta
/// ```
///
/// Soundness depends on the `Pcs` commitment being the same public commitment
/// layout as the accumulator's `rt`. A PCS that commits to a fresh unrelated
/// oracle must not be used here. Full WARP `DACC` soundness additionally needs
/// verifier-side codeword consistency `f = C(w)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    bound = "EF: Serialize + serde::de::DeserializeOwned, PcsProof: Serialize + serde::de::DeserializeOwned"
)]
pub struct WhirWarpFinalizerProof<EF, PcsProof> {
    /// Opening proof for `f_hat(alpha) = mu`.
    pub accumulator_opening: WhirAccumulatorOpeningProof<PcsProof>,
    /// PESAT decider proof for `Pb(beta, C^{-1}(f)) = eta`.
    pub pesat: WhirPesatProof<EF, PcsProof>,
}
