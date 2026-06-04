//! Small Fiat-Shamir sumcheck helper for WHIR-native table proofs.
//!
//! This module is intentionally independent from WARP. Circuit-table proving
//! needs a local sumcheck format that can be reused by primitive table checks
//! and by the Poseidon2 AIR adapter without depending on the WARP crate.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

use crate::whir_native::WhirNativeCircuitError;

/// Sumcheck proof with each round polynomial represented by evaluations at
/// the integer points `0, 1, ..., degree`.
///
/// Evaluation form avoids interpolation in the prover and supports arbitrary
/// low-degree constraints. The verifier evaluates a round polynomial at the
/// Fiat-Shamir challenge by Lagrange interpolation over this fixed domain.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: Serialize", deserialize = "EF: Deserialize<'de>"))]
pub struct WhirNativeSumcheckProof<EF> {
    pub degree: usize,
    pub round_evals: Vec<Vec<EF>>,
}

impl<EF> WhirNativeSumcheckProof<EF> {
    pub const fn new(degree: usize) -> Self {
        Self {
            degree,
            round_evals: Vec::new(),
        }
    }

    pub const fn num_rounds(&self) -> usize {
        self.round_evals.len()
    }
}

/// Prove a sumcheck for a polynomial supplied as a round evaluator.
///
/// `round_eval(round, prefix, t, suffix)` must evaluate the true polynomial
/// after fixing the already-sampled prefix challenges, setting the current
/// variable to `t`, and using the Boolean suffix assignment encoded by
/// `suffix`.
pub fn prove_sumcheck<F, EF, Challenger, Eval>(
    num_variables: usize,
    degree: usize,
    initial_claim: EF,
    challenger: &mut Challenger,
    mut round_eval: Eval,
) -> Result<(WhirNativeSumcheckProof<EF>, Point<EF>, EF), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    Eval: FnMut(usize, &[EF], EF, usize) -> EF,
{
    let mut proof = WhirNativeSumcheckProof::new(degree);
    let mut claim = initial_claim;
    let mut prefix = Vec::with_capacity(num_variables);

    for round in 0..num_variables {
        let suffix_vars = num_variables - round - 1;
        let suffix_count = 1usize << suffix_vars;
        let mut evals = Vec::with_capacity(degree + 1);

        for point in 0..=degree {
            let t = EF::from(F::from_u64(point as u64));
            let value = (0..suffix_count)
                .map(|suffix| round_eval(round, &prefix, t, suffix))
                .sum();
            evals.push(value);
        }

        if evals[0] + evals[1] != claim {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "sumcheck round {round} inconsistent with claimed sum"
            )));
        }

        challenger.observe_algebra_slice(&evals);
        let challenge = challenger.sample_algebra_element();
        claim = lagrange_eval_on_zero_to_degree::<F, EF>(&evals, challenge);
        prefix.push(challenge);
        proof.round_evals.push(evals);
    }

    Ok((proof, Point::new(prefix), claim))
}

/// Replay a sumcheck proof and return the terminal random point and claim.
pub fn verify_sumcheck<F, EF, Challenger>(
    proof: &WhirNativeSumcheckProof<EF>,
    num_variables: usize,
    initial_claim: EF,
    challenger: &mut Challenger,
) -> Result<(Point<EF>, EF), WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    if proof.round_evals.len() != num_variables {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "sumcheck round count mismatch: expected {num_variables}, got {}",
            proof.round_evals.len()
        )));
    }
    if proof.degree == 0 {
        return Err(WhirNativeCircuitError::ConstraintViolation(String::from(
            "sumcheck degree must be at least 1",
        )));
    }
    let expected_round_evals = proof.degree.checked_add(1).ok_or_else(|| {
        WhirNativeCircuitError::ConstraintViolation(String::from("sumcheck degree is too large"))
    })?;

    let mut claim = initial_claim;
    let mut point = Vec::with_capacity(num_variables);
    for (round, evals) in proof.round_evals.iter().enumerate() {
        if evals.len() != expected_round_evals {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "sumcheck round {round} degree mismatch: expected {}, got {}",
                expected_round_evals,
                evals.len()
            )));
        }
        if evals[0] + evals[1] != claim {
            return Err(WhirNativeCircuitError::ConstraintViolation(format!(
                "sumcheck round {round} failed consistency check"
            )));
        }

        challenger.observe_algebra_slice(evals);
        let challenge = challenger.sample_algebra_element();
        claim = lagrange_eval_on_zero_to_degree::<F, EF>(evals, challenge);
        point.push(challenge);
    }

    Ok((Point::new(point), claim))
}

/// Evaluate the unique degree-`n` polynomial passing through
/// `(0, evals[0]), ..., (n, evals[n])`.
pub fn lagrange_eval_on_zero_to_degree<F, EF>(evals: &[EF], x: EF) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    let degree = evals.len() - 1;
    let mut result = EF::ZERO;
    for i in 0..=degree {
        let xi = EF::from(F::from_u64(i as u64));
        let mut num = EF::ONE;
        let mut den = EF::ONE;
        for j in 0..=degree {
            if i == j {
                continue;
            }
            let xj = EF::from(F::from_u64(j as u64));
            num *= x - xj;
            den *= xi - xj;
        }
        result += evals[i] * num * den.inverse();
    }
    result
}

/// Build the full multilinear point used during a prover sumcheck round.
///
/// Variables are ordered consistently with [`Point::hypercube`]: the suffix
/// integer is decoded as a big-endian Boolean suffix.
pub fn point_from_prefix_current_suffix<F, EF>(
    prefix: &[EF],
    current: EF,
    suffix: usize,
    suffix_vars: usize,
) -> Point<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut point = Vec::with_capacity(prefix.len() + 1 + suffix_vars);
    point.extend_from_slice(prefix);
    point.push(current);
    point.extend(Point::hypercube(suffix, suffix_vars).as_slice());
    Point::new(point)
}

/// Evaluate the known multilinear extension of a Boolean evaluation vector at
/// an extension-field point.
pub fn eval_known_mle<F, EF>(evals: &[EF], point: &Point<EF>) -> Result<EF, WhirNativeCircuitError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    if evals.len() != (1usize << point.num_variables()) {
        return Err(WhirNativeCircuitError::ConstraintViolation(format!(
            "known MLE size mismatch: {} evals for {} variables",
            evals.len(),
            point.num_variables()
        )));
    }
    if evals.len() == 1 {
        return Ok(evals[0]);
    }

    let mut point_iter = point.as_slice().iter().rev();
    let Some(&first_r) = point_iter.next() else {
        return Err(WhirNativeCircuitError::ConstraintViolation(String::from(
            "non-constant known MLE has no point coordinates",
        )));
    };
    let mut layer = Vec::with_capacity(evals.len() / 2);
    for pair in evals.chunks_exact(2) {
        layer.push(pair[0] + first_r * (pair[1] - pair[0]));
    }

    let mut next = Vec::new();
    for &r in point_iter {
        next.clear();
        next.reserve(layer.len() / 2);
        for pair in layer.chunks_exact(2) {
            next.push(pair[0] + r * (pair[1] - pair[0]));
        }
        core::mem::swap(&mut layer, &mut next);
    }
    debug_assert_eq!(layer.len(), 1);
    Ok(layer[0])
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16};
    use p3_challenger::{
        CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::poly::Poly;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type InnerChallenger = DuplexChallenger<F, Perm, 16, 8>;

    #[derive(Clone)]
    struct TestChallenger(InnerChallenger);

    impl TestChallenger {
        fn new() -> Self {
            Self(InnerChallenger::new(default_babybear_poseidon2_16()))
        }
    }

    impl CanObserve<F> for TestChallenger {
        fn observe(&mut self, value: F) {
            self.0.observe(value);
        }
    }

    impl CanSample<F> for TestChallenger {
        fn sample(&mut self) -> F {
            self.0.sample()
        }
    }

    impl CanSampleBits<usize> for TestChallenger {
        fn sample_bits(&mut self, bits: usize) -> usize {
            self.0.sample_bits(bits)
        }
    }

    impl FieldChallenger<F> for TestChallenger {}

    impl GrindingChallenger for TestChallenger {
        type Witness = F;

        fn grind(&mut self, bits: usize) -> Self::Witness {
            self.0.grind(bits)
        }
    }

    #[test]
    fn known_mle_matches_poly_eval_ext() {
        let evals = (0..8)
            .map(|i| EF::from(F::from_u64(3 * i + 2)))
            .collect::<Vec<_>>();
        let point = Point::new(vec![
            EF::from(F::from_u64(5)),
            EF::from(F::from_u64(7)),
            EF::from(F::from_u64(11)),
        ]);

        let expected = Poly::new(evals.clone()).eval_ext::<F>(&point);
        let actual = eval_known_mle::<F, EF>(&evals, &point).expect("valid MLE");

        assert_eq!(actual, expected);
    }

    #[test]
    fn honest_quadratic_sumcheck_round_trips() {
        // P(x, y) = x + y - xy has sum 3 over {0,1}^2.
        let initial_claim = EF::from(F::from_u64(3));
        let mut prover_challenger = TestChallenger::new();
        let (proof, point, terminal_claim) = prove_sumcheck::<F, EF, _, _>(
            2,
            2,
            initial_claim,
            &mut prover_challenger,
            |round, prefix, t, suffix| {
                let (x, y) = if round == 0 {
                    let y = EF::from(F::from_bool(suffix == 1));
                    (t, y)
                } else {
                    let x = prefix[0];
                    (x, t)
                };
                x + y - x * y
            },
        )
        .expect("prove sumcheck");

        let mut verifier_challenger = TestChallenger::new();
        let (verifier_point, verifier_claim) =
            verify_sumcheck::<F, EF, _>(&proof, 2, initial_claim, &mut verifier_challenger)
                .expect("verify sumcheck");

        assert_eq!(point, verifier_point);
        assert_eq!(terminal_claim, verifier_claim);
    }

    #[test]
    fn tampered_round_fails() {
        let initial_claim = EF::from(F::from_u64(3));
        let mut challenger = TestChallenger::new();
        let (mut proof, _, _) = prove_sumcheck::<F, EF, _, _>(
            2,
            2,
            initial_claim,
            &mut challenger,
            |round, prefix, t, suffix| {
                let (x, y) = if round == 0 {
                    let y = EF::from(F::from_bool(suffix == 1));
                    (t, y)
                } else {
                    let x = prefix[0];
                    (x, t)
                };
                x + y - x * y
            },
        )
        .expect("prove sumcheck");

        proof.round_evals[0][0] += EF::ONE;
        let mut verifier_challenger = TestChallenger::new();
        verify_sumcheck::<F, EF, _>(&proof, 2, initial_claim, &mut verifier_challenger)
            .expect_err("tampered proof must fail");
    }

    #[test]
    fn degree_zero_sumcheck_rejects_without_panic() {
        let initial_claim = EF::from(F::from_u64(3));
        let proof = WhirNativeSumcheckProof {
            degree: 0,
            round_evals: vec![vec![initial_claim]],
        };
        let mut verifier_challenger = TestChallenger::new();

        let err = verify_sumcheck::<F, EF, _>(&proof, 1, initial_claim, &mut verifier_challenger)
            .expect_err("degree-zero proof must be rejected");

        match err {
            WhirNativeCircuitError::ConstraintViolation(message) => {
                assert!(message.contains("degree"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
