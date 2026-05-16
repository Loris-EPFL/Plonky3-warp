//! Native WARP-to-WHIR compiler scaffolding.
//!
//! This module is the RS-only compiler boundary described in
//! `warp/docs/native-whir-compiler.md`. It starts at the algebraic statement
//! level: WARP evaluation obligations are converted into WHIR linear
//! Sigma-IOP constraints rather than into PCS opening calls.
//!
//! The main invariant is that WARP and WHIR refer to one Reed-Solomon codeword
//! per oracle. WARP's root IOP records obligations such as "open `u = C(w)` at
//! index `i`" or "open the accumulator MLE at `alpha`". WHIR commits to the
//! encoded initial RS oracle for the same `w`, and this compiler rewrites WARP
//! obligations as WHIR linear-Sigma constraints over the committed message
//! representation whenever possible:
//!
//! - fresh base WARP inputs must satisfy `u = C(w)` before WHIR commits;
//! - accumulator codeword-MLE claims use the adjoint weights of the same RS
//!   encoder;
//! - all touched oracle commitments are absorbed before reduction challenges
//!   are sampled.
//!
//! This is a prover-side invariant and a proximity-sound verifier statement,
//! not an exact full-table equality theorem. WHIR proves proximity/opening
//! soundness for the committed RS oracle. A reduction that wants to invoke
//! WARP's exact source-paper `MT.Commit`/`MT.Open` transcript must first pass
//! through the source-WARP projection bridge: outside MMCS binding failure, an
//! exact-codeword bridge must identify the full committed table with `C(w)`.
//! For base WARP slots, that bridge also relies on the alphabet condition that
//! the committed table is genuinely `F`-valued, not merely an extension-field
//! RS word extracted by WHIR over `EF`. The native API enforces this on the
//! prover path by committing base oracles through `Mmcs<F>` data; an untyped
//! extension-only commitment would need a separate subfield proof.
//!
//! The proof-system half of this module proves the recorded linear
//! oracle-opening part of the root IOP with one precommitted WHIR
//! linear-Sigma residual proof.
//! It does not prove the nonlinear terminal PESAT equation
//! `Pb(beta, w) = eta`; that belongs to the configured WARP finalizer. Older
//! codeword-domain and limb fallback paths were removed so every linear-root
//! proof uses the same WARP/WHIR RS code.
//!
//! Soundness note: "one precommitted WHIR linear-Sigma residual proof" means
//! one WHIR proof object after WARP has batched its claims and after the
//! underlying RS-oracle roots were already bound. In the source-theorem view,
//! this is a one-round linear-Sigma IOP with per-slot answer arrays `A_a[...]`
//! and a `V_poly` check of the residual sum; WHIR then samples its own
//! combination challenge and proximity-tests the virtual combination of the
//! committed oracles. It is a precommitted specialization of WHIR's
//! linear-Sigma compiler, not a new PCS-style opening theorem. The proof still
//! executes WHIR's ordinary constrained-RS protocol internally: initial
//! folding, every configured intermediate STIR/proximity round,
//! OOD/query-combination checks, and the final folding phase. The WARP root
//! compiler relies on those WHIR round-by-round errors for proximity/opening
//! soundness; the WARP-specific sumcheck only reduces the recorded linear
//! opening claims to the `V_poly` residual query supplied to WHIR. It does not,
//! by itself, certify entrywise equality between the full MMCS table and the
//! extracted nearby RS codeword.

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;
use p3_whir::constraints::statement::{
    BatchedLinearSigmaOpeningClaim, BatchedLinearSigmaReductionProof, EqStatement,
    LinearSigmaConstraint, LinearSigmaReductionError, LinearSigmaReductionProof,
    LinearSigmaStatement,
};
use p3_whir::pcs::proof::WhirProof;
use p3_whir::pcs::verifier::errors::VerifierError as WhirVerifierError;
use p3_whir::pcs::{
    WhirBatchedDeferredProverOracle, WhirBatchedDeferredVerifierOracle, WhirDeferredProverData,
    WhirExtensionDeferredProverData, WhirPcs, WhirSharedBaseDeferredProverData,
};
use p3_whir::sumcheck::lagrange::extrapolate_01inf;
use p3_whir::sumcheck::strategy::VariableOrder;
use p3_whir::sumcheck::{SumcheckData, SumcheckError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::code::ReedSolomonCode;
use crate::root_iop::{
    RootIopBoundCommitment, RootIopBoundTranscript, RootIopError, RootIopOpeningClaim,
    RootIopOpeningPoint, RootIopOpeningValue, RootIopOracleField, RootIopOracleValues,
};

mod domain;

mod types;
pub use types::{
    NativeWarpWhirClaimCompileError, NativeWarpWhirLinearOpeningProof,
    NativeWarpWhirRootBaseProverData, NativeWarpWhirRootBatchedOpeningProof,
    NativeWarpWhirRootCommitment, NativeWarpWhirRootExtensionProverData,
    NativeWarpWhirRootOracleProverData, NativeWarpWhirRootProof, NativeWarpWhirRootProofError,
    NativeWarpWhirRootProverData, NativeWarpWhirRootReductionError,
    NativeWarpWhirRootSharedBaseProverData,
};

/// Native WARP linear-opening root proof system using WHIR over the same RS
/// oracle as WARP.
pub struct NativeWarpWhirRootProofSystem<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// WHIR PCS configured for the WARP message length and RS rate.
    message_pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    /// Algebraic compiler from WARP root claims to WHIR linear-Sigma claims.
    compiler: NativeWarpWhirCompiler<'a, F, Dft>,
    /// Seed challenger used to derive independent role-separated WHIR transcripts.
    challenger_seed: Challenger,
}

/// Explicit alias for the linear-opening role of
/// [`NativeWarpWhirRootProofSystem`].
///
/// The historical type name is kept for compatibility. This alias documents
/// the security boundary: the WHIR compiler authenticates linear opening
/// claims, while the terminal PESAT equation is handled by a finalizer.
pub type NativeWarpWhirLinearOpeningProofSystem<
    'a,
    F,
    EF,
    MT,
    Challenger,
    Dft,
    const DIGEST_ELEMS: usize,
> = NativeWarpWhirRootProofSystem<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>;

mod statement;
pub use statement::{NativeWarpWhirEvalClaim, NativeWarpWhirOracleStatement};

/// Compiler helper for WARP over Plonky3's Reed-Solomon code.
///
/// WARP's RS specialization and WHIR's initial oracle are both statements
/// about one smooth Reed-Solomon code. This compiler works in the message
/// representation `w` whose encoded oracle is `C(w)`, and uses
/// [`ReedSolomonCode`] to express every WARP codeword query as a linear query
/// over that same initial polynomial. The source-paper proof states this in
/// WHIR's Boolean multilinear basis; this implementation may store `w` in
/// coefficient or systematic coordinates, so the returned weights are already
/// transported through the corresponding coordinate map.
///
/// The compiler deliberately exposes only the linear/proximity part. Exact
/// equality between a verifier's full committed table and the RS codeword
/// selected by WHIR extraction is an external bridge condition, not a
/// consequence of compiling the claims.
/// In coefficient layout those are coefficient-coordinate weights; in
/// systematic layout they are the corresponding Lagrange weights on the
/// message subgroup. No second code is introduced.
pub struct NativeWarpWhirCompiler<'a, F, Dft>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    code: &'a ReedSolomonCode<F, Dft>,
}

impl<'a, F, Dft> NativeWarpWhirCompiler<'a, F, Dft>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a compiler for one WARP RS code.
    pub const fn new(code: &'a ReedSolomonCode<F, Dft>) -> Self {
        Self { code }
    }

    /// Return the RS code this compiler targets.
    pub const fn code(&self) -> &'a ReedSolomonCode<F, Dft> {
        self.code
    }

    /// Convert a folded-codeword evaluation claim into a WHIR linear-Sigma
    /// constraint.
    ///
    /// This is the WHIR paper's basic evaluation-as-Sigma-query identity:
    /// `f_hat(z) = v` is
    /// `sum_b eq(z, b) * f_hat(b) = v`.
    ///
    /// # Panics
    ///
    /// Panics if `claim.point` is not a point in the codeword hypercube.
    pub fn eval_claim_constraint<EF>(
        &self,
        claim: &NativeWarpWhirEvalClaim<EF>,
    ) -> LinearSigmaConstraint<EF>
    where
        EF: ExtensionField<F>,
    {
        assert_eq!(
            claim.point.num_variables(),
            self.code.log_codeword_len(),
            "WARP/WHIR evaluation point must have log_n variables",
        );
        let mut eq = EqStatement::initialize(self.code.log_codeword_len());
        eq.add_evaluated_constraint(claim.point.clone(), claim.value);
        LinearSigmaConstraint::from_eq_statement::<F>(&eq, EF::ONE)
    }

    /// Convert multiple folded-codeword evaluation claims into separate WHIR
    /// linear-Sigma constraints.
    ///
    /// WHIR's multi-constrained RS layer is responsible for the later random
    /// batching. Keeping these claims separate here preserves the binding
    /// point required by Construction 5.5.
    pub fn eval_claim_statement<EF>(
        &self,
        claims: &[NativeWarpWhirEvalClaim<EF>],
    ) -> NativeWarpWhirOracleStatement<EF>
    where
        EF: ExtensionField<F>,
    {
        let mut statement = LinearSigmaStatement::initialize(self.code.log_codeword_len());
        for claim in claims {
            statement.add_constraint(self.eval_claim_constraint(claim));
        }
        NativeWarpWhirOracleStatement::new(statement)
    }

    /// Convert a systematic witness-MLE evaluation claim into a codeword
    /// linear-Sigma constraint.
    ///
    /// In systematic mode, the witness/message MLE point `y` is lifted to the
    /// codeword point `(y, 0, ..., 0)`. This is useful for terminal opening
    /// claims emitted by finalizer sumchecks. It does not by itself prove the
    /// nonlinear PESAT equation `Pb(beta, C^{-1}(f)) = eta`.
    ///
    /// # Panics
    ///
    /// Panics if the RS code is not systematic or if `message_point` has the
    /// wrong arity.
    pub fn systematic_message_eval_constraint<EF>(
        &self,
        message_point: &[EF],
        value: EF,
    ) -> LinearSigmaConstraint<EF>
    where
        EF: ExtensionField<F>,
    {
        let point = self.code.systematic_message_point(message_point);
        self.eval_claim_constraint(&NativeWarpWhirEvalClaim { point, value })
    }

    fn check_claim_oracles_bound<EF, Comm>(
        &self,
        oracles: &[(RootIopBoundCommitment<Comm>, RootIopOracleValues<F, EF>)],
        claims: &[RootIopOpeningClaim<F, EF>],
    ) -> Result<(), NativeWarpWhirRootReductionError>
    where
        EF: ExtensionField<F>,
    {
        for claim in claims {
            if !oracles
                .iter()
                .any(|(commitment, _)| commitment.oracle_id == claim.oracle_id)
            {
                return Err(NativeWarpWhirRootReductionError::UnknownOracle(
                    claim.oracle_id,
                ));
            }
        }
        Ok(())
    }

    fn check_unique_bound_oracle_ids<EF, Comm>(
        &self,
        oracles: &[(RootIopBoundCommitment<Comm>, RootIopOracleValues<F, EF>)],
    ) -> Result<(), NativeWarpWhirRootReductionError>
    where
        EF: ExtensionField<F>,
    {
        let mut seen = Vec::new();
        for (commitment, _) in oracles {
            if seen.contains(&commitment.oracle_id) {
                return Err(NativeWarpWhirRootReductionError::DuplicateOracle(
                    commitment.oracle_id,
                ));
            }
            seen.push(commitment.oracle_id);
        }
        Ok(())
    }

    fn check_unique_public_oracle_ids<Comm>(
        &self,
        commitments: &[RootIopBoundCommitment<Comm>],
    ) -> Result<(), NativeWarpWhirRootReductionError> {
        let mut seen = Vec::new();
        for commitment in commitments {
            if seen.contains(&commitment.oracle_id) {
                return Err(NativeWarpWhirRootReductionError::DuplicateOracle(
                    commitment.oracle_id,
                ));
            }
            seen.push(commitment.oracle_id);
        }
        Ok(())
    }

    fn check_claim_oracles_public<EF, Comm>(
        &self,
        commitments: &[RootIopBoundCommitment<Comm>],
        claims: &[RootIopOpeningClaim<F, EF>],
    ) -> Result<(), NativeWarpWhirRootReductionError>
    where
        EF: ExtensionField<F>,
    {
        for claim in claims {
            if !commitments
                .iter()
                .any(|commitment| commitment.oracle_id == claim.oracle_id)
            {
                return Err(NativeWarpWhirRootReductionError::UnknownOracle(
                    claim.oracle_id,
                ));
            }
        }
        Ok(())
    }

    fn check_bound_oracle_shape<EF, Comm>(
        &self,
        commitment: &RootIopBoundCommitment<Comm>,
        values: Option<&RootIopOracleValues<F, EF>>,
    ) -> Result<(), NativeWarpWhirRootReductionError>
    where
        EF: ExtensionField<F>,
    {
        if commitment.log_len != self.code.log_codeword_len() {
            return Err(NativeWarpWhirRootReductionError::OracleLogLengthMismatch {
                oracle_id: commitment.oracle_id,
                expected: self.code.log_codeword_len(),
                actual: commitment.log_len,
            });
        }

        match values {
            Some(RootIopOracleValues::Base(values)) => {
                if commitment.field != RootIopOracleField::Base {
                    return Err(NativeWarpWhirRootReductionError::OracleValueFieldMismatch(
                        commitment.oracle_id,
                    ));
                }
                if values.len() != self.code.codeword_len() {
                    return Err(
                        NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                            oracle_id: commitment.oracle_id,
                            expected: self.code.codeword_len(),
                            actual: values.len(),
                        },
                    );
                }
            }
            Some(RootIopOracleValues::Extension(values)) => {
                if commitment.field != RootIopOracleField::Extension {
                    return Err(NativeWarpWhirRootReductionError::OracleValueFieldMismatch(
                        commitment.oracle_id,
                    ));
                }
                if values.len() != self.code.codeword_len() {
                    return Err(
                        NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                            oracle_id: commitment.oracle_id,
                            expected: self.code.codeword_len(),
                            actual: values.len(),
                        },
                    );
                }
            }
            None => {}
        }

        Ok(())
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    NativeWarpWhirRootProofSystem<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord + PrimeCharacteristicRing + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F> + TwoAdicField + Serialize + serde::de::DeserializeOwned,
    MT: Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    WhirExtensionDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a native WARP root proof system.
    ///
    /// `message_pcs` must be configured with `code.log_msg_len()` variables
    /// and the same RS rate as the WARP code. WHIR commits to the encoded RS
    /// oracle for `w`, and WARP codeword openings are compiled using the same
    /// [`ReedSolomonCode`] generator. Verifier soundness remains WHIR
    /// proximity soundness unless an exact-codeword bridge is supplied.
    pub fn new(
        message_pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        code: &'a ReedSolomonCode<F, Dft>,
        challenger_seed: Challenger,
    ) -> Self {
        assert_eq!(
            message_pcs.num_variables(),
            code.log_msg_len(),
            "WHIR variable count must match WARP RS message dimension",
        );
        assert_eq!(
            message_pcs.starting_log_inv_rate(),
            code.log_inv_rate(),
            "WHIR starting inverse rate must match WARP RS code",
        );
        Self {
            message_pcs,
            compiler: NativeWarpWhirCompiler::new(code),
            challenger_seed,
        }
    }

    /// Commit a base-field fresh WARP input with one shared RS encoding.
    ///
    /// The supplied `codeword` must equal `C(message)` on the prover side.
    /// WHIR then commits to its encoded initial oracle for `message`. During
    /// proof generation, codeword-index claims are transformed into
    /// constrained-RS claims over the same message representation. The verifier
    /// still relies on WHIR proximity plus the source-WARP projection bridge
    /// required before applying WARP's exact `MT.Commit`/`MT.Open` theorem. In
    /// particular, the base-field alphabet condition comes from using an
    /// `F`-valued base commitment, not from WHIR's extension-field proximity
    /// extraction alone.
    pub fn commit_base_message_oracle(
        &self,
        oracle_id: usize,
        codeword: Vec<F>,
        message: Vec<F>,
    ) -> Result<
        (
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        ),
        NativeWarpWhirRootProofError,
    > {
        if codeword.len() != self.compiler.code().codeword_len() {
            return Err(
                NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                    oracle_id,
                    expected: self.compiler.code().codeword_len(),
                    actual: codeword.len(),
                }
                .into(),
            );
        }
        if message.len() != self.compiler.code().msg_len() {
            return Err(
                NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                    oracle_id,
                    expected: self.compiler.code().msg_len(),
                    actual: message.len(),
                }
                .into(),
            );
        }
        self.ensure_base_codeword_matches_message(oracle_id, &codeword, &message)?;
        let mut challenger = self.base_oracle_challenger(oracle_id);
        let (commitment, prover_data) = self
            .message_pcs
            .commit_deferred(RowMajorMatrix::new(message.clone(), 1), &mut challenger);
        Ok((
            RootIopBoundCommitment {
                oracle_id,
                log_len: self.compiler.code().log_codeword_len(),
                field: RootIopOracleField::Base,
                commitment: NativeWarpWhirRootCommitment::BaseMessage(commitment),
            },
            NativeWarpWhirRootOracleProverData {
                oracle_id,
                data: NativeWarpWhirRootProverData::Base(NativeWarpWhirRootBaseProverData {
                    prover_data,
                    challenger,
                    message,
                }),
            },
        ))
    }

    /// Commit several base-field fresh WARP inputs under one WHIR/MMCS root.
    ///
    /// The shared root authenticates the whole ordered stack of base RS
    /// oracles. Each returned root-IOP commitment carries that same Merkle root
    /// plus a distinct column index. The column index is part of the
    /// Fiat-Shamir payload, so swapping columns changes the transcript. A
    /// source-WARP projection must treat this root as the stack commitment, not
    /// as `MT.Commit` applied independently to each column.
    pub fn commit_shared_base_message_oracles(
        &self,
        inputs: Vec<(usize, Vec<F>, Vec<F>)>,
    ) -> Result<
        Vec<(
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        )>,
        NativeWarpWhirRootProofError,
    > {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let width = inputs.len();
        let mut matrices = Vec::with_capacity(width);
        for (oracle_id, codeword, message) in &inputs {
            if codeword.len() != self.compiler.code().codeword_len() {
                return Err(
                    NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                        oracle_id: *oracle_id,
                        expected: self.compiler.code().codeword_len(),
                        actual: codeword.len(),
                    }
                    .into(),
                );
            }
            if message.len() != self.compiler.code().msg_len() {
                return Err(
                    NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                        oracle_id: *oracle_id,
                        expected: self.compiler.code().msg_len(),
                        actual: message.len(),
                    }
                    .into(),
                );
            }
            self.ensure_base_codeword_matches_message(*oracle_id, codeword, message)?;
            matrices.push(RowMajorMatrix::new(message.clone(), 1));
        }

        let mut challenger = self.challenger_seed.clone();
        challenger.observe(F::from_u64(domain::ROOT_WHIR_BASE_ORACLE));
        challenger.observe(F::from_usize(width));
        for (oracle_id, _, _) in &inputs {
            challenger.observe(F::from_usize(*oracle_id));
        }
        // This commitment is shared across columns. The root transcript binds
        // the role tag, width, and ordered oracle ids before WHIR samples any
        // commitment-dependent randomness, and each public commitment below
        // carries `(root, column, width)` so columns cannot be swapped.
        let encoded = self.message_pcs.encode_base_batch_initial_oracles(matrices);
        let (root, shared) = self
            .message_pcs
            .commit_base_batch_encoded_deferred(encoded, &mut challenger);

        let mut out = Vec::with_capacity(width);
        for (column, (oracle_id, _codeword, message)) in inputs.into_iter().enumerate() {
            let commitment = RootIopBoundCommitment {
                oracle_id,
                log_len: self.compiler.code().log_codeword_len(),
                field: RootIopOracleField::Base,
                commitment: NativeWarpWhirRootCommitment::BaseMessageShared {
                    root: root.clone(),
                    column,
                    width,
                },
            };
            let prover_data = NativeWarpWhirRootOracleProverData {
                oracle_id,
                data: NativeWarpWhirRootProverData::BaseShared(
                    NativeWarpWhirRootSharedBaseProverData {
                        shared: shared.clone(),
                        column,
                        width,
                        message,
                    },
                ),
            };
            out.push((commitment, prover_data));
        }

        Ok(out)
    }

    /// Commit an extension-field accumulator codeword for the WARP root IOP.
    ///
    /// The codeword is decoded back to the RS message and re-encoded before
    /// commitment. Non-codewords are rejected by this prover API. The verifier
    /// side still gets WHIR proximity/opening soundness; exact full-table
    /// equality is a separate bridge condition.
    pub fn commit_extension_oracle(
        &self,
        oracle_id: usize,
        codeword: Vec<EF>,
    ) -> Result<
        (
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        ),
        NativeWarpWhirRootProofError,
    > {
        if codeword.len() != self.compiler.code().codeword_len() {
            return Err(
                NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                    oracle_id,
                    expected: self.compiler.code().codeword_len(),
                    actual: codeword.len(),
                }
                .into(),
            );
        }

        let message = self.compiler.code().message_from_codeword(&codeword);
        self.commit_extension_oracle_with_message(oracle_id, codeword, message)
    }

    /// Commit an extension-field accumulator when both the WARP codeword and
    /// its RS message representation are already available.
    ///
    /// This is the fast path for accumulation code that already carries
    /// `w_merged` alongside `f_merged = C(w_merged)`: it avoids decoding from
    /// the codeword, but still enforces the single-RS invariant before WHIR
    /// commits.
    pub fn commit_extension_oracle_with_message(
        &self,
        oracle_id: usize,
        codeword: Vec<EF>,
        message: Vec<EF>,
    ) -> Result<
        (
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        ),
        NativeWarpWhirRootProofError,
    > {
        if codeword.len() != self.compiler.code().codeword_len() {
            return Err(
                NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                    oracle_id,
                    expected: self.compiler.code().codeword_len(),
                    actual: codeword.len(),
                }
                .into(),
            );
        }
        if message.len() != self.compiler.code().msg_len() {
            return Err(
                NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                    oracle_id,
                    expected: self.compiler.code().msg_len(),
                    actual: message.len(),
                }
                .into(),
            );
        }
        self.ensure_extension_codeword_matches_message(oracle_id, &codeword, &message)?;
        let challenger = self.extension_oracle_challenger(oracle_id);
        self.commit_extension_message_oracle_with_challenger(oracle_id, message, challenger)
    }

    /// Commit an extension-field accumulator through WHIR's initial-message
    /// oracle path when the RS message is already available.
    ///
    /// This helper is only for callers that do not have a WARP codeword in
    /// hand. If both representations are available, use
    /// [`Self::commit_extension_oracle_with_message`] so the codeword/message
    /// consistency check is enforced at commit time.
    pub fn commit_extension_message_oracle(
        &self,
        oracle_id: usize,
        message: Vec<EF>,
    ) -> Result<
        (
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        ),
        NativeWarpWhirRootProofError,
    > {
        let challenger = self.extension_oracle_challenger(oracle_id);
        self.commit_extension_message_oracle_with_challenger(oracle_id, message, challenger)
    }

    fn commit_extension_message_oracle_with_challenger(
        &self,
        oracle_id: usize,
        message: Vec<EF>,
        mut challenger: Challenger,
    ) -> Result<
        (
            RootIopBoundCommitment<NativeWarpWhirRootCommitment<MT::Commitment>>,
            NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        ),
        NativeWarpWhirRootProofError,
    > {
        if message.len() != self.compiler.code().msg_len() {
            return Err(
                NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                    oracle_id,
                    expected: self.compiler.code().msg_len(),
                    actual: message.len(),
                }
                .into(),
            );
        }

        let encoded = self
            .message_pcs
            .encode_extension_initial_oracle(RowMajorMatrix::new(message.clone(), 1));
        let (commitment, prover_data) = self
            .message_pcs
            .commit_extension_encoded_deferred(encoded, &mut challenger);
        Ok((
            RootIopBoundCommitment {
                oracle_id,
                log_len: self.compiler.code().log_codeword_len(),
                field: RootIopOracleField::Extension,
                commitment: NativeWarpWhirRootCommitment::ExtensionMessage(commitment),
            },
            NativeWarpWhirRootOracleProverData {
                oracle_id,
                data: NativeWarpWhirRootProverData::ExtensionMessage(
                    NativeWarpWhirRootExtensionProverData {
                        prover_data,
                        challenger,
                        message,
                    },
                ),
            },
        ))
    }

    /// Prove the recorded WARP root IOP with one WHIR batched opening.
    pub fn prove(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, NativeWarpWhirRootCommitment<MT::Commitment>>,
        prover_data: &[NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>],
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<NativeWarpWhirRootProof<F, EF, MT>, NativeWarpWhirRootProofError> {
        let opening = self.prove_direct_batched_root(
            transcript,
            prover_data,
            challenger,
            reduction_pow_bits,
        )?;
        Ok(NativeWarpWhirRootProof { opening })
    }

    /// Verify the recorded WARP root IOP with one WHIR batched opening.
    pub fn verify(
        &self,
        expected_commitments: &[RootIopBoundCommitment<
            NativeWarpWhirRootCommitment<MT::Commitment>,
        >],
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &NativeWarpWhirRootProof<F, EF, MT>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<(), NativeWarpWhirRootProofError> {
        self.verify_direct_batched_root(
            expected_commitments,
            expected_claims,
            &proof.opening,
            challenger,
            reduction_pow_bits,
        )
    }

    fn prove_direct_batched_root(
        &self,
        transcript: &RootIopBoundTranscript<F, EF, NativeWarpWhirRootCommitment<MT::Commitment>>,
        prover_data: &[NativeWarpWhirRootOracleProverData<F, EF, MT, Challenger, DIGEST_ELEMS>],
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<NativeWarpWhirRootBatchedOpeningProof<F, EF, MT>, NativeWarpWhirRootProofError>
    {
        self.compiler
            .check_unique_bound_oracle_ids(&transcript.oracles)?;
        self.compiler
            .check_claim_oracles_bound(&transcript.oracles, &transcript.claims)?;

        let mut commitments_to_observe = Vec::new();
        let mut statements = Vec::new();
        let mut polys = Vec::new();
        let mut whir_oracles = Vec::new();
        for (commitment, values) in &transcript.oracles {
            if !claims_include_oracle(&transcript.claims, commitment.oracle_id) {
                continue;
            }
            self.compiler
                .check_bound_oracle_shape::<EF, _>(commitment, Some(values))?;

            match (&commitment.commitment, values) {
                (
                    NativeWarpWhirRootCommitment::BaseMessage(_),
                    RootIopOracleValues::Base(values),
                ) => {
                    let message =
                        self.base_message_for_oracle(prover_data, commitment.oracle_id)?;
                    if message.len() != self.compiler.code().msg_len() {
                        return Err(
                            NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                                oracle_id: commitment.oracle_id,
                                expected: self.compiler.code().msg_len(),
                                actual: message.len(),
                            }
                            .into(),
                        );
                    }
                    self.ensure_base_codeword_matches_message(
                        commitment.oracle_id,
                        values,
                        message,
                    )?;
                    let oracle_data = prover_data
                        .iter()
                        .find(|data| data.oracle_id == commitment.oracle_id)
                        .ok_or(NativeWarpWhirRootProofError::MissingProverData(
                            commitment.oracle_id,
                        ))?;
                    let NativeWarpWhirRootProverData::Base(data) = &oracle_data.data else {
                        return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                            commitment.oracle_id,
                        ));
                    };
                    commitments_to_observe.push(commitment);
                    statements.push(self.compact_base_message_claim_statement(
                        &transcript.claims,
                        commitment.oracle_id,
                    )?);
                    polys.push(NativeWarpDirectBatchedResidualPoly::Base(message));
                    whir_oracles.push(NativeWarpBatchedResidualProverOracle::Base(
                        data.prover_data.clone(),
                    ));
                }
                (
                    NativeWarpWhirRootCommitment::BaseMessageShared { column, width, .. },
                    RootIopOracleValues::Base(values),
                ) => {
                    let message =
                        self.base_message_for_oracle(prover_data, commitment.oracle_id)?;
                    if message.len() != self.compiler.code().msg_len() {
                        return Err(
                            NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                                oracle_id: commitment.oracle_id,
                                expected: self.compiler.code().msg_len(),
                                actual: message.len(),
                            }
                            .into(),
                        );
                    }
                    self.ensure_base_codeword_matches_message(
                        commitment.oracle_id,
                        values,
                        message,
                    )?;
                    let oracle_data = prover_data
                        .iter()
                        .find(|data| data.oracle_id == commitment.oracle_id)
                        .ok_or(NativeWarpWhirRootProofError::MissingProverData(
                            commitment.oracle_id,
                        ))?;
                    let NativeWarpWhirRootProverData::BaseShared(data) = &oracle_data.data else {
                        return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                            commitment.oracle_id,
                        ));
                    };
                    if data.column != *column || data.width != *width {
                        return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                            commitment.oracle_id,
                        ));
                    }
                    commitments_to_observe.push(commitment);
                    statements.push(self.compact_base_message_claim_statement(
                        &transcript.claims,
                        commitment.oracle_id,
                    )?);
                    polys.push(NativeWarpDirectBatchedResidualPoly::Base(message));
                    whir_oracles.push(NativeWarpBatchedResidualProverOracle::SharedBase {
                        shared: data.shared.clone(),
                        column: *column,
                        width: *width,
                    });
                }
                (
                    NativeWarpWhirRootCommitment::ExtensionMessage(_),
                    RootIopOracleValues::Extension(values),
                ) => {
                    let message =
                        self.extension_message_for_oracle(prover_data, commitment.oracle_id)?;
                    if message.len() != self.compiler.code().msg_len() {
                        return Err(
                            NativeWarpWhirRootReductionError::OracleValueLengthMismatch {
                                oracle_id: commitment.oracle_id,
                                expected: self.compiler.code().msg_len(),
                                actual: message.len(),
                            }
                            .into(),
                        );
                    }
                    self.ensure_extension_codeword_matches_message(
                        commitment.oracle_id,
                        values,
                        message,
                    )?;
                    let oracle_data = prover_data
                        .iter()
                        .find(|data| data.oracle_id == commitment.oracle_id)
                        .ok_or(NativeWarpWhirRootProofError::MissingProverData(
                            commitment.oracle_id,
                        ))?;
                    let NativeWarpWhirRootProverData::ExtensionMessage(data) = &oracle_data.data
                    else {
                        return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                            commitment.oracle_id,
                        ));
                    };
                    commitments_to_observe.push(commitment);
                    statements.push(self.compact_extension_message_claim_statement(
                        &transcript.claims,
                        commitment.oracle_id,
                    )?);
                    polys.push(NativeWarpDirectBatchedResidualPoly::Extension(message));
                    whir_oracles.push(NativeWarpBatchedResidualProverOracle::Extension(
                        data.prover_data.clone(),
                    ));
                }
                _ => {
                    return Err(NativeWarpWhirRootProofError::OracleKindMismatch(
                        commitment.oracle_id,
                    ));
                }
            }
        }

        if statements.is_empty() {
            return Err(NativeWarpWhirClaimCompileError::EmptyOracle(usize::MAX).into());
        }

        // Fiat-Shamir binding point for the direct path. All public WARP root
        // commitments are absorbed before the batching challenge used by
        // `prove_compact_batched_root_reduction` is sampled.
        for commitment in commitments_to_observe {
            observe_native_root_commitment::<F, Challenger, MT::Commitment>(challenger, commitment);
        }

        // The compact reducer converts all per-oracle WARP constraints into
        // the residual query of the one-round source linear-Sigma IOP. The
        // grouped WHIR adapter packages the corresponding source answer
        // arrays as one virtual initial-oracle opening, then runs WHIR's own
        // OOD/proximity combination against the already committed RS messages.
        let (reduction, opening_claim) = prove_compact_batched_root_reduction::<F, EF, Dft, _>(
            self.compiler.code(),
            &statements,
            &polys,
            challenger,
            reduction_pow_bits,
        )?;
        let opening = self.message_pcs.open_grouped_batched_deferred(
            Self::group_prover_oracles(whir_oracles, &opening_claim.coeffs)?,
            opening_claim.point,
            opening_claim.value,
            challenger,
        )?;

        Ok(NativeWarpWhirRootBatchedOpeningProof { reduction, opening })
    }

    fn verify_direct_batched_root(
        &self,
        expected_commitments: &[RootIopBoundCommitment<
            NativeWarpWhirRootCommitment<MT::Commitment>,
        >],
        expected_claims: &[RootIopOpeningClaim<F, EF>],
        proof: &NativeWarpWhirRootBatchedOpeningProof<F, EF, MT>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<(), NativeWarpWhirRootProofError> {
        self.compiler
            .check_unique_public_oracle_ids(expected_commitments)?;
        self.compiler
            .check_claim_oracles_public(expected_commitments, expected_claims)?;

        let mut statements = Vec::new();
        let mut commitments = Vec::new();
        for commitment in expected_commitments {
            if !claims_include_oracle(expected_claims, commitment.oracle_id) {
                continue;
            }
            self.compiler
                .check_bound_oracle_shape::<EF, _>(commitment, None)?;

            match &commitment.commitment {
                NativeWarpWhirRootCommitment::BaseMessage(commitment_root) => {
                    observe_native_root_commitment::<F, Challenger, MT::Commitment>(
                        challenger, commitment,
                    );
                    statements.push(self.compact_base_message_claim_statement(
                        expected_claims,
                        commitment.oracle_id,
                    )?);
                    commitments.push(NativeWarpBatchedResidualCommitment::Base(
                        commitment_root.clone(),
                    ));
                }
                NativeWarpWhirRootCommitment::BaseMessageShared {
                    root,
                    column,
                    width,
                } => {
                    observe_native_root_commitment::<F, Challenger, MT::Commitment>(
                        challenger, commitment,
                    );
                    statements.push(self.compact_base_message_claim_statement(
                        expected_claims,
                        commitment.oracle_id,
                    )?);
                    commitments.push(NativeWarpBatchedResidualCommitment::SharedBase {
                        root: root.clone(),
                        column: *column,
                        width: *width,
                    });
                }
                NativeWarpWhirRootCommitment::ExtensionMessage(commitment_root) => {
                    observe_native_root_commitment::<F, Challenger, MT::Commitment>(
                        challenger, commitment,
                    );
                    statements.push(self.compact_extension_message_claim_statement(
                        expected_claims,
                        commitment.oracle_id,
                    )?);
                    commitments.push(NativeWarpBatchedResidualCommitment::Extension(
                        commitment_root.clone(),
                    ));
                }
            }
        }

        // Same binding order as the prover: for each touched oracle, absorb
        // the WARP metadata plus the WHIR commitment before deriving the
        // compact batching challenges.
        let opening_claim = verify_compact_batched_root_reduction::<F, EF, Dft, _>(
            self.compiler.code(),
            &statements,
            &proof.reduction,
            challenger,
            reduction_pow_bits,
        )?;
        let whir_oracles = Self::group_verifier_oracles(commitments, &opening_claim.coeffs)?;
        self.message_pcs
            .verify_batched_deferred(
                &whir_oracles,
                opening_claim.point,
                opening_claim.value,
                &proof.opening,
                challenger,
            )
            .map_err(NativeWarpWhirRootProofError::BatchedOpening)
    }

    fn group_prover_oracles(
        oracles: Vec<NativeWarpBatchedResidualProverOracle<F, EF, MT, DIGEST_ELEMS>>,
        coeffs: &[EF],
    ) -> Result<
        Vec<WhirBatchedDeferredProverOracle<F, EF, MT, DIGEST_ELEMS>>,
        LinearSigmaReductionError,
    > {
        if oracles.len() != coeffs.len() {
            return Err(LinearSigmaReductionError::ArityMismatch {
                expected: oracles.len(),
                actual: coeffs.len(),
            });
        }

        let mut grouped = Vec::new();
        for (oracle, &coeff) in oracles.into_iter().zip(coeffs) {
            match oracle {
                NativeWarpBatchedResidualProverOracle::Base(data) => {
                    grouped.push(WhirBatchedDeferredProverOracle::Base { coeff, data });
                }
                NativeWarpBatchedResidualProverOracle::Extension(data) => {
                    grouped.push(WhirBatchedDeferredProverOracle::Extension { coeff, data });
                }
                NativeWarpBatchedResidualProverOracle::SharedBase {
                    shared,
                    column,
                    width,
                } => {
                    if column >= width {
                        return Err(LinearSigmaReductionError::ArityMismatch {
                            expected: width,
                            actual: column + 1,
                        });
                    }
                    let mut inserted = false;
                    for existing in &mut grouped {
                        if let WhirBatchedDeferredProverOracle::SharedBase { coeffs, data } =
                            existing
                        {
                            if Arc::ptr_eq(data, &shared) {
                                if coeffs.len() != width {
                                    return Err(LinearSigmaReductionError::ArityMismatch {
                                        expected: width,
                                        actual: coeffs.len(),
                                    });
                                }
                                coeffs[column] += coeff;
                                inserted = true;
                                break;
                            }
                        }
                    }
                    if !inserted {
                        let mut coeffs = EF::zero_vec(width);
                        coeffs[column] = coeff;
                        grouped.push(WhirBatchedDeferredProverOracle::SharedBase {
                            coeffs,
                            data: shared,
                        });
                    }
                }
            }
        }

        Ok(grouped)
    }

    fn group_verifier_oracles(
        commitments: Vec<NativeWarpBatchedResidualCommitment<MT::Commitment>>,
        coeffs: &[EF],
    ) -> Result<Vec<WhirBatchedDeferredVerifierOracle<EF, MT::Commitment>>, LinearSigmaReductionError>
    {
        if commitments.len() != coeffs.len() {
            return Err(LinearSigmaReductionError::ArityMismatch {
                expected: commitments.len(),
                actual: coeffs.len(),
            });
        }

        let mut grouped = Vec::new();
        for (commitment, &coeff) in commitments.into_iter().zip(coeffs) {
            match commitment {
                NativeWarpBatchedResidualCommitment::Base(commitment) => {
                    grouped.push(WhirBatchedDeferredVerifierOracle::Base { coeff, commitment });
                }
                NativeWarpBatchedResidualCommitment::Extension(commitment) => {
                    grouped
                        .push(WhirBatchedDeferredVerifierOracle::Extension { coeff, commitment });
                }
                NativeWarpBatchedResidualCommitment::SharedBase {
                    root,
                    column,
                    width,
                } => {
                    if column >= width {
                        return Err(LinearSigmaReductionError::ArityMismatch {
                            expected: width,
                            actual: column + 1,
                        });
                    }
                    let mut inserted = false;
                    for existing in &mut grouped {
                        if let WhirBatchedDeferredVerifierOracle::SharedBase {
                            coeffs,
                            commitment,
                        } = existing
                        {
                            if *commitment == root {
                                if coeffs.len() != width {
                                    return Err(LinearSigmaReductionError::ArityMismatch {
                                        expected: width,
                                        actual: coeffs.len(),
                                    });
                                }
                                coeffs[column] += coeff;
                                inserted = true;
                                break;
                            }
                        }
                    }
                    if !inserted {
                        let mut coeffs = EF::zero_vec(width);
                        coeffs[column] = coeff;
                        grouped.push(WhirBatchedDeferredVerifierOracle::SharedBase {
                            coeffs,
                            commitment: root,
                        });
                    }
                }
            }
        }

        Ok(grouped)
    }

    fn compact_base_message_claim_statement(
        &self,
        claims: &[RootIopOpeningClaim<F, EF>],
        oracle_id: usize,
    ) -> Result<NativeWarpCompactRootStatement<EF>, NativeWarpWhirClaimCompileError> {
        let mut statement =
            NativeWarpCompactRootStatement::initialize(self.compiler.code().log_msg_len());
        for claim in claims.iter().filter(|claim| claim.oracle_id == oracle_id) {
            let value = match &claim.value {
                RootIopOpeningValue::Base(value) => EF::from(*value),
                _ => {
                    return Err(NativeWarpWhirClaimCompileError::OracleFieldMismatch(
                        oracle_id,
                    ));
                }
            };
            match &claim.point {
                RootIopOpeningPoint::Index(index) | RootIopOpeningPoint::RsCodewordIndex(index) => {
                    if *index >= self.compiler.code().codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::IndexOutOfBounds {
                            oracle_id,
                            index: *index,
                        });
                    }
                    statement.add_index(*index, value);
                }
                RootIopOpeningPoint::Mle(_) => {
                    return Err(NativeWarpWhirClaimCompileError::UnsupportedBaseMle(
                        oracle_id,
                    ));
                }
            }
        }

        if statement.is_empty() {
            return Err(NativeWarpWhirClaimCompileError::EmptyOracle(oracle_id));
        }
        Ok(statement)
    }

    fn compact_extension_message_claim_statement(
        &self,
        claims: &[RootIopOpeningClaim<F, EF>],
        oracle_id: usize,
    ) -> Result<NativeWarpCompactRootStatement<EF>, NativeWarpWhirClaimCompileError> {
        let mut statement =
            NativeWarpCompactRootStatement::initialize(self.compiler.code().log_msg_len());
        for claim in claims.iter().filter(|claim| claim.oracle_id == oracle_id) {
            let value = match &claim.value {
                RootIopOpeningValue::Extension(value) => *value,
                _ => {
                    return Err(NativeWarpWhirClaimCompileError::OracleFieldMismatch(
                        oracle_id,
                    ));
                }
            };
            match &claim.point {
                RootIopOpeningPoint::Index(index) | RootIopOpeningPoint::RsCodewordIndex(index) => {
                    if *index >= self.compiler.code().codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::IndexOutOfBounds {
                            oracle_id,
                            index: *index,
                        });
                    }
                    statement.add_index(*index, value);
                }
                RootIopOpeningPoint::Mle(point) => {
                    if point.len() != self.compiler.code().log_codeword_len() {
                        return Err(NativeWarpWhirClaimCompileError::PointArityMismatch {
                            oracle_id,
                        });
                    }
                    statement.add_mle(point.clone(), value);
                }
            }
        }

        if statement.is_empty() {
            return Err(NativeWarpWhirClaimCompileError::EmptyOracle(oracle_id));
        }
        Ok(statement)
    }

    fn base_message_for_oracle<'b>(
        &self,
        prover_data: &'b [NativeWarpWhirRootOracleProverData<
            F,
            EF,
            MT,
            Challenger,
            DIGEST_ELEMS,
        >],
        oracle_id: usize,
    ) -> Result<&'b [F], NativeWarpWhirRootProofError> {
        let oracle_data = prover_data
            .iter()
            .find(|data| data.oracle_id == oracle_id)
            .ok_or(NativeWarpWhirRootProofError::MissingProverData(oracle_id))?;
        match &oracle_data.data {
            NativeWarpWhirRootProverData::Base(data) => Ok(data.message.as_slice()),
            NativeWarpWhirRootProverData::BaseShared(data) => Ok(data.message.as_slice()),
            NativeWarpWhirRootProverData::ExtensionMessage(_) => {
                Err(NativeWarpWhirRootProofError::OracleKindMismatch(oracle_id))
            }
        }
    }

    fn extension_message_for_oracle<'b>(
        &self,
        prover_data: &'b [NativeWarpWhirRootOracleProverData<
            F,
            EF,
            MT,
            Challenger,
            DIGEST_ELEMS,
        >],
        oracle_id: usize,
    ) -> Result<&'b [EF], NativeWarpWhirRootProofError> {
        let oracle_data = prover_data
            .iter()
            .find(|data| data.oracle_id == oracle_id)
            .ok_or(NativeWarpWhirRootProofError::MissingProverData(oracle_id))?;
        match &oracle_data.data {
            NativeWarpWhirRootProverData::ExtensionMessage(data) => Ok(data.message.as_slice()),
            NativeWarpWhirRootProverData::Base(_) | NativeWarpWhirRootProverData::BaseShared(_) => {
                Err(NativeWarpWhirRootProofError::OracleKindMismatch(oracle_id))
            }
        }
    }

    fn ensure_base_codeword_matches_message(
        &self,
        oracle_id: usize,
        codeword: &[F],
        message: &[F],
    ) -> Result<(), NativeWarpWhirRootProofError> {
        let expected_codeword = self.compiler.code().encode(message);
        if expected_codeword.as_slice() != codeword {
            return Err(NativeWarpWhirRootReductionError::EncodingMismatch { oracle_id }.into());
        }
        Ok(())
    }

    fn ensure_extension_codeword_matches_message(
        &self,
        oracle_id: usize,
        codeword: &[EF],
        message: &[EF],
    ) -> Result<(), NativeWarpWhirRootProofError> {
        let expected_codeword = self.compiler.code().encode_algebra(message);
        if expected_codeword.as_slice() != codeword {
            return Err(NativeWarpWhirRootReductionError::EncodingMismatch { oracle_id }.into());
        }
        Ok(())
    }

    fn base_oracle_challenger(&self, oracle_id: usize) -> Challenger {
        let mut challenger = self.challenger_seed.clone();
        challenger.observe(F::from_u64(domain::ROOT_WHIR_BASE_ORACLE));
        challenger.observe(F::from_usize(oracle_id));
        challenger
    }

    fn extension_oracle_challenger(&self, oracle_id: usize) -> Challenger {
        let mut challenger = self.challenger_seed.clone();
        challenger.observe(F::from_u64(domain::ROOT_WHIR_EXTENSION_ORACLE));
        challenger.observe(F::from_usize(oracle_id));
        challenger
    }
}

mod direct;
use direct::{
    NativeWarpBatchedResidualCommitment, NativeWarpBatchedResidualProverOracle,
    NativeWarpCompactRootStatement, NativeWarpDirectBatchedResidualPoly,
    observe_native_root_commitment, prove_compact_batched_root_reduction,
    verify_compact_batched_root_reduction,
};

/// Build claims from parallel point/value lists.
///
/// # Panics
///
/// Panics if the two slices have different lengths.
pub fn eval_claims_from_parts<EF: Field>(
    points: &[Point<EF>],
    values: &[EF],
) -> Vec<NativeWarpWhirEvalClaim<EF>> {
    assert_eq!(
        points.len(),
        values.len(),
        "WARP/WHIR claim point/value count mismatch",
    );
    points
        .iter()
        .cloned()
        .zip(values.iter().copied())
        .map(|(point, value)| NativeWarpWhirEvalClaim { point, value })
        .collect()
}

fn claims_include_oracle<F, EF>(claims: &[RootIopOpeningClaim<F, EF>], oracle_id: usize) -> bool
where
    F: Field,
    EF: ExtensionField<F>,
{
    claims.iter().any(|claim| claim.oracle_id == oracle_id)
}

#[cfg(test)]
mod tests;
