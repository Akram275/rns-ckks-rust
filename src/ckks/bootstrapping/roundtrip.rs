use crate::ckks::bootstrapping::coeff_to_slot::{coeff_to_slot, CoeffToSlotPlan, CoeffToSlotPrecomputed};
use crate::ckks::bootstrapping::eval_mod::{eval_mod, EvalModPlan, EvalModPrecomputed};
use crate::ckks::bootstrapping::linear_transform::generate_diagonal_transform_rotation_keys;
use crate::ckks::bootstrapping::mod_raise::{mod_raise, ModRaisedCiphertext};
use crate::ckks::bootstrapping::keys::BootstrapKeySet;
use crate::ckks::bootstrapping::slot_to_coeff::{slot_to_coeff, SlotToCoeffPlan, SlotToCoeffPrecomputed};
use crate::rns::{RnsCiphertext, RnsCkksContext};
use num_complex::Complex64;

pub fn bootstrap_identity_roundtrip(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    key_pair: &crate::rns::RnsKeyPair,
    active_slots: usize,
) -> Result<RnsCiphertext, String> {
    let raised = mod_raise(context, ciphertext);
    bootstrap_identity_roundtrip_from_mod_raise(context, &raised, key_pair, active_slots)
}

pub fn bootstrap_exact_sparse_eval_mod_roundtrip(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    key_pair: &crate::rns::RnsKeyPair,
    active_slots: usize,
    sine_scale: f64,
    polynomial_degree: usize,
) -> Result<RnsCiphertext, String> {
    let raised = mod_raise(context, ciphertext);
    bootstrap_exact_sparse_eval_mod_roundtrip_from_mod_raise(
        context,
        &raised,
        key_pair,
        active_slots,
        sine_scale,
        polynomial_degree,
    )
}

pub fn bootstrap_identity_roundtrip_from_mod_raise(
    context: &RnsCkksContext,
    raised: &ModRaisedCiphertext,
    key_pair: &crate::rns::RnsKeyPair,
    active_slots: usize,
) -> Result<RnsCiphertext, String> {
    let projected = raised.project_to_q_ciphertext();

    let coeff_to_slot_plan = CoeffToSlotPlan::from_ciphertext(context, &projected, active_slots)?;
    let identity = identity_matrix(active_slots);

    let coeff_to_slot_precomputed = CoeffToSlotPrecomputed::new(
        context,
        coeff_to_slot_plan,
        &identity,
        projected.scale_bits,
    )?;
    let coeff_to_slot_rotation_keys = generate_diagonal_transform_rotation_keys(
        context,
        &key_pair.secret_key,
        &key_pair.public_key,
        projected.level,
        active_slots,
    );

    let slots = coeff_to_slot(
        context,
        &projected,
        &coeff_to_slot_precomputed,
        &coeff_to_slot_rotation_keys,
    )?;
    let slot_to_coeff_plan = SlotToCoeffPlan::from_ciphertext(context, &slots, active_slots)?;
    let slot_to_coeff_precomputed = SlotToCoeffPrecomputed::new(
        context,
        slot_to_coeff_plan,
        &identity,
        slots.scale_bits,
    )?;
    let slot_to_coeff_rotation_keys = generate_diagonal_transform_rotation_keys(
        context,
        &key_pair.secret_key,
        &key_pair.public_key,
        slots.level,
        active_slots,
    );

    slot_to_coeff(
        context,
        &slots,
        &slot_to_coeff_precomputed,
        &slot_to_coeff_rotation_keys,
    )
}

pub fn bootstrap_exact_sparse_eval_mod_roundtrip_from_mod_raise(
    context: &RnsCkksContext,
    raised: &ModRaisedCiphertext,
    key_pair: &crate::rns::RnsKeyPair,
    active_slots: usize,
    sine_scale: f64,
    polynomial_degree: usize,
) -> Result<RnsCiphertext, String> {
    let projected = raised.project_to_q_ciphertext();

    let coeff_to_slot_plan = CoeffToSlotPlan::from_ciphertext(context, &projected, active_slots)?;
    let coeff_to_slot_precomputed = CoeffToSlotPrecomputed::exact(
        context,
        coeff_to_slot_plan,
        projected.scale_bits,
    )?;
    let coeff_to_slot_rotation_keys = generate_diagonal_transform_rotation_keys(
        context,
        &key_pair.secret_key,
        &key_pair.public_key,
        projected.level,
        active_slots,
    );
    let slots = coeff_to_slot(
        context,
        &projected,
        &coeff_to_slot_precomputed,
        &coeff_to_slot_rotation_keys,
    )?;

    let eval_mod_plan = EvalModPlan::from_ciphertext(
        context,
        &slots,
        active_slots,
        1.0,
        sine_scale,
        polynomial_degree,
    )?;
    let eval_mod_precomputed = EvalModPrecomputed::sine_taylor(eval_mod_plan)?;
    let eval_mod_keys = BootstrapKeySet::for_eval_mod(
        context,
        key_pair,
        slots.level,
        eval_mod_precomputed.coefficients.len(),
    );
    let evaluated = eval_mod(
        context,
        &slots,
        &eval_mod_precomputed,
        &eval_mod_keys.eval_mod_relinearization_keys,
    )?;

    let slot_to_coeff_plan = SlotToCoeffPlan::from_ciphertext(context, &evaluated, active_slots)?;
    let slot_to_coeff_precomputed = SlotToCoeffPrecomputed::exact(
        context,
        slot_to_coeff_plan,
        evaluated.scale_bits,
    )?;
    let slot_to_coeff_rotation_keys = generate_diagonal_transform_rotation_keys(
        context,
        &key_pair.secret_key,
        &key_pair.public_key,
        evaluated.level,
        active_slots,
    );

    slot_to_coeff(
        context,
        &evaluated,
        &slot_to_coeff_precomputed,
        &slot_to_coeff_rotation_keys,
    )
}

fn identity_matrix(dimension: usize) -> Vec<Vec<Complex64>> {
    (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|col| {
                    if row == col {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{bootstrap_exact_sparse_eval_mod_roundtrip, bootstrap_identity_roundtrip};
    use crate::ckks::bootstrapping::{exact_coeff_to_slot_matrix, exact_slot_to_coeff_matrix};
    use crate::ckks::bootstrapping::linear_transform::repeat_block_slots;
    use crate::ckks::bootstrapping::eval_mod::sine_taylor_coefficients;
    use crate::rns::{RnsCkksContext, RnsCkksParams};
    use num_complex::Complex64;

    #[test]
    fn bootstrap_identity_roundtrip_preserves_q_chain_message() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let input = vec![0.5_f64, -0.25, 0.125, 0.75, -0.5, 0.25, -0.125, 0.375];
        let packed = repeat_block_slots(&input, context.num_slots());
        let ciphertext = context.encrypt(&context.encode_real(&packed), &key_pair.public_key);

        let recovered_ct = bootstrap_identity_roundtrip(&context, &ciphertext, &key_pair, input.len())
            .expect("identity bootstrapping roundtrip must succeed");
        let recovered = context.decode_real_at_scale(
            &context.decrypt(&recovered_ct, &key_pair.secret_key),
            recovered_ct.scale_bits,
        );

        for (index, &expected) in input.iter().enumerate() {
            assert!(
                (recovered[index] - expected).abs() <= 2e-2,
                "expected {expected}, got {} at slot {index}",
                recovered[index]
            );
        }
    }

    #[test]
    fn bootstrap_exact_sparse_eval_mod_matches_plaintext_pipeline() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let active_slots = 8;
        let sine_scale = 1.0;
        let polynomial_degree = 5;
        let input = vec![
            Complex64::new(0.03125, -0.015625),
            Complex64::new(-0.05, 0.0125),
            Complex64::new(0.08, -0.02),
            Complex64::new(-0.02, 0.01),
            Complex64::new(0.0125, -0.03125),
            Complex64::new(-0.04, 0.02),
            Complex64::new(0.025, -0.0125),
            Complex64::new(-0.01, 0.04),
        ];
        let repeated = repeat_complex_slots(&input, context.num_slots());
        let ciphertext = context.encrypt(&context.encode(&repeated), &key_pair.public_key);

        let recovered_ct = bootstrap_exact_sparse_eval_mod_roundtrip(
            &context,
            &ciphertext,
            &key_pair,
            active_slots,
            sine_scale,
            polynomial_degree,
        )
        .expect("exact sparse EvalMod bootstrapping roundtrip must succeed");
        let recovered = context.decode_at_scale(
            &context.decrypt(&recovered_ct, &key_pair.secret_key),
            recovered_ct.scale_bits,
        );

        let coeff_to_slot = exact_coeff_to_slot_matrix(&context, active_slots)
            .expect("exact CoeffToSlot matrix must build");
        let slot_to_coeff = exact_slot_to_coeff_matrix(&context, active_slots)
            .expect("exact SlotToCoeff matrix must build");
        let coeff_slots = apply_matrix(&coeff_to_slot, &input);
        let coefficients = sine_taylor_coefficients(polynomial_degree, sine_scale);
        let evaluated_slots: Vec<Complex64> = coeff_slots
            .iter()
            .map(|&value| evaluate_polynomial(&coefficients, value))
            .collect();
        let expected = apply_matrix(&slot_to_coeff, &evaluated_slots);

        for (index, expected_slot) in expected.iter().enumerate() {
            assert!(
                (recovered[index] - *expected_slot).norm() <= 5e-2,
                "expected {expected_slot:?}, got {:?} at slot {index}",
                recovered[index]
            );
        }
    }

    fn repeat_complex_slots(values: &[Complex64], slot_count: usize) -> Vec<Complex64> {
        (0..slot_count)
            .map(|index| values[index % values.len()])
            .collect()
    }

    fn apply_matrix(matrix: &[Vec<Complex64>], vector: &[Complex64]) -> Vec<Complex64> {
        matrix
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(lhs, rhs)| *lhs * *rhs).sum())
            .collect()
    }

    fn evaluate_polynomial(coefficients: &[f64], value: Complex64) -> Complex64 {
        coefficients
            .iter()
            .rev()
            .fold(Complex64::new(0.0, 0.0), |accumulator, &coefficient| {
                accumulator * value + Complex64::new(coefficient, 0.0)
            })
    }
}