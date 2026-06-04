use alloc::format;
use alloc::string::ToString;
use core::fmt::Debug;
use core::ops::Deref;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, PackedValue, TwoAdicField};
use p3_multilinear_util::point::Point;

use crate::constraints::statement::EqStatement;
use crate::parameters::WhirConfig;
use crate::pcs::proof::WhirProof;
use crate::pcs::verifier::errors::VerifierError;

/// Parsed commitment extracted from the verifier's transcript.
///
/// Contains the Merkle root and the OOD equality constraints
/// needed for verification.
#[derive(Debug, Clone)]
pub struct ParsedCommitment<F, D> {
    /// Merkle root of the committed evaluation table.
    pub root: D,
    /// OOD challenge points and their claimed evaluations.
    pub ood_statement: EqStatement<F>,
}

impl<F, D> ParsedCommitment<F, D>
where
    F: Field,
{
    /// Parse a commitment from the proof and transcript state.
    pub fn parse<EF, MT: Mmcs<F>, Challenger>(
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        num_variables: usize,
        ood_samples: usize,
    ) -> Result<ParsedCommitment<EF, MT::Commitment>, VerifierError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        Self::parse_with_round(proof, challenger, num_variables, ood_samples, None)
    }

    /// Parse a commitment for a specific round (or initial if `None`).
    pub fn parse_with_round<EF, MT: Mmcs<F>, Challenger>(
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        num_variables: usize,
        ood_samples: usize,
        round_index: Option<usize>,
    ) -> Result<ParsedCommitment<EF, MT::Commitment>, VerifierError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        // Extract root and OOD answers from either the initial commitment or a round.
        let (root, ood_answers) = match round_index {
            None => (
                proof
                    .initial_commitment
                    .clone()
                    .ok_or_else(|| VerifierError::MalformedProof {
                        details: "missing initial commitment".to_string(),
                    })?,
                proof.initial_ood_answers.clone(),
            ),
            Some(idx) => {
                let round_proof = proof
                    .rounds
                    .get(idx)
                    .ok_or(VerifierError::InvalidRoundIndex { index: idx })?;
                (
                    round_proof.commitment.clone().ok_or_else(|| {
                        VerifierError::MalformedProof {
                            details: format!("missing commitment for round {idx}"),
                        }
                    })?,
                    round_proof.ood_answers.clone(),
                )
            }
        };

        // Observe the Merkle root in the transcript.
        challenger.observe(root.clone());

        // Reconstruct equality constraints from OOD challenge points and answers.
        let mut ood_statement = EqStatement::initialize(num_variables);
        for i in 0..ood_samples {
            let point = challenger.sample_algebra_element();
            let point = Point::expand_from_univariate(point, num_variables);
            let eval =
                ood_answers
                    .get(i)
                    .copied()
                    .ok_or_else(|| VerifierError::MalformedProof {
                        details: format!("missing OOD answer {i} for commitment"),
                    })?;
            challenger.observe_algebra_element(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        }

        Ok(ParsedCommitment {
            root,
            ood_statement,
        })
    }
}

/// Lightweight wrapper for parsing commitment data during verification.
#[derive(Debug)]
pub struct CommitmentReader<'a, EF, F, MT: Mmcs<F>, Challenger>(
    &'a WhirConfig<EF, F, MT, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, MT, Challenger> CommitmentReader<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: Mmcs<F>,
{
    pub const fn new(params: &'a WhirConfig<EF, F, MT, Challenger>) -> Self {
        Self(params)
    }

    /// Parse the initial commitment from the proof and verifier transcript.
    pub fn parse_commitment<W, const DIGEST_ELEMS: usize>(
        &self,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<ParsedCommitment<EF, MT::Commitment>, VerifierError>
    where
        W: PackedValue<Value = W> + Eq + Copy,
        Challenger: CanObserve<MT::Commitment>,
    {
        ParsedCommitment::<_, MT::Commitment>::parse(
            proof,
            challenger,
            self.num_variables,
            self.commitment_ood_samples,
        )
    }
}

impl<EF, F, MT, Challenger> Deref for CommitmentReader<'_, EF, F, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    type Target = WhirConfig<EF, F, MT, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
