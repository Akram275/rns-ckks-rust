use crate::ckks::bootstrapping::coeff_to_slot::{coeff_to_slot, CoeffToSlotPlan, CoeffToSlotPrecomputed};
use crate::ckks::bootstrapping::eval_mod::{eval_mod, EvalModPlan, EvalModPrecomputed};
use crate::ckks::bootstrapping::exact_transforms::{
    exact_dense_coeff_to_slot_matrices,
    exact_dense_slot_to_coeff_matrices,
};
use crate::ckks::bootstrapping::linear_transform::{
    apply_diagonal_linear_transform,
    pack_diagonal_transform_plaintexts,
    DiagonalTransformRotationKeys,
};
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

pub fn bootstrap_exact_dense_eval_mod_roundtrip(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    key_pair: &crate::rns::RnsKeyPair,
    sine_scale: f64,
    polynomial_degree: usize,
) -> Result<RnsCiphertext, String> {
    let raised = mod_raise(context, ciphertext);
    bootstrap_exact_dense_eval_mod_roundtrip_from_mod_raise(
        context,
        &raised,
        key_pair,
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

pub fn bootstrap_exact_dense_eval_mod_roundtrip_from_mod_raise(
    context: &RnsCkksContext,
    raised: &ModRaisedCiphertext,
    key_pair: &crate::rns::RnsKeyPair,
    sine_scale: f64,
    polynomial_degree: usize,
) -> Result<RnsCiphertext, String> {
    let projected = raised.project_to_q_ciphertext();
    let conjugation_key = context.generate_conjugation_key_at_level(
        &key_pair.secret_key,
        &key_pair.public_key,
        projected.level,
    );
    let conjugated = context.conjugate(&projected, &conjugation_key);

    let dense_slots = dense_coeff_to_slot_exact(
        context,
        &projected,
        &conjugated,
        &key_pair.secret_key,
        &key_pair.public_key,
    )?;

    let eval_mod_plan = EvalModPlan::from_ciphertext(
        context,
        &dense_slots.0,
        context.num_slots(),
        1.0,
        sine_scale,
        polynomial_degree,
    )?;
    let eval_mod_precomputed = EvalModPrecomputed::sine_taylor(eval_mod_plan)?;
    let eval_mod_keys = BootstrapKeySet::for_eval_mod(
        context,
        key_pair,
        dense_slots.0.level,
        eval_mod_precomputed.coefficients.len(),
    );
    let evaluated_upper = eval_mod(
        context,
        &dense_slots.0,
        &eval_mod_precomputed,
        &eval_mod_keys.eval_mod_relinearization_keys,
    )?;
    let evaluated_lower = eval_mod(
        context,
        &dense_slots.1,
        &eval_mod_precomputed,
        &eval_mod_keys.eval_mod_relinearization_keys,
    )?;

    dense_slot_to_coeff_exact(
        context,
        &evaluated_upper,
        &evaluated_lower,
        &key_pair.secret_key,
        &key_pair.public_key,
    )
}

fn dense_coeff_to_slot_exact(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    conjugated_ciphertext: &RnsCiphertext,
    secret_key: &crate::rns::RnsSecretKey,
    public_key: &crate::rns::RnsPublicKey,
) -> Result<(RnsCiphertext, RnsCiphertext), String> {
    let matrices = exact_dense_coeff_to_slot_matrices(context)?;
    let slot_count = context.num_slots();
    let rotation_keys = dense_rotation_keys(context, ciphertext, slot_count, secret_key, public_key);

    let upper = apply_dense_branch(
        context,
        ciphertext,
        conjugated_ciphertext,
        &matrices.upper_from_slots,
        &matrices.upper_from_conjugates,
        &rotation_keys,
    );
    let lower = apply_dense_branch(
        context,
        ciphertext,
        conjugated_ciphertext,
        &matrices.lower_from_slots,
        &matrices.lower_from_conjugates,
        &rotation_keys,
    );

    Ok((upper, lower))
}

fn dense_slot_to_coeff_exact(
    context: &RnsCkksContext,
    upper: &RnsCiphertext,
    lower: &RnsCiphertext,
    secret_key: &crate::rns::RnsSecretKey,
    public_key: &crate::rns::RnsPublicKey,
) -> Result<RnsCiphertext, String> {
    let matrices = exact_dense_slot_to_coeff_matrices(context)?;
    let slot_count = context.num_slots();
    let upper_keys = dense_rotation_keys(context, upper, slot_count, secret_key, public_key);
    let lower_keys = dense_rotation_keys(context, lower, slot_count, secret_key, public_key);

    let upper_term = apply_single_dense_transform(
        context,
        upper,
        &matrices.slots_from_upper,
        &upper_keys,
    );
    let lower_term = apply_single_dense_transform(
        context,
        lower,
        &matrices.slots_from_lower,
        &lower_keys,
    );

    Ok(upper_term.add(&lower_term))
}

fn apply_dense_branch(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    conjugated_ciphertext: &RnsCiphertext,
    from_slots: &[Vec<Complex64>],
    from_conjugates: &[Vec<Complex64>],
    rotation_keys: &DiagonalTransformRotationKeys,
) -> RnsCiphertext {
    let direct = apply_single_dense_transform(context, ciphertext, from_slots, rotation_keys);
    let conjugate = apply_single_dense_transform(context, conjugated_ciphertext, from_conjugates, rotation_keys);
    direct.add(&conjugate)
}

fn apply_single_dense_transform(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    matrix: &[Vec<Complex64>],
    rotation_keys: &DiagonalTransformRotationKeys,
) -> RnsCiphertext {
    let diagonals = pack_diagonal_transform_plaintexts(
        context,
        matrix,
        ciphertext.level,
        ciphertext.scale_bits,
    );
    apply_diagonal_linear_transform(context, ciphertext, &diagonals, rotation_keys)
}

fn dense_rotation_keys(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    dimension: usize,
    secret_key: &crate::rns::RnsSecretKey,
    public_key: &crate::rns::RnsPublicKey,
) -> DiagonalTransformRotationKeys {
    generate_diagonal_transform_rotation_keys(
        context,
        secret_key,
        public_key,
        ciphertext.level,
        dimension,
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
    use super::{
        bootstrap_exact_dense_eval_mod_roundtrip,
        bootstrap_exact_sparse_eval_mod_roundtrip,
        dense_coeff_to_slot_exact,
        dense_slot_to_coeff_exact,
        bootstrap_identity_roundtrip,
    };
    use crate::ckks::bootstrapping::mod_raise::mod_raise;
    use crate::ckks::bootstrapping::{
        eval_mod,
        exact_coeff_to_slot_matrix,
        exact_dense_coeff_to_slot_matrices,
        exact_dense_slot_to_coeff_matrices,
        exact_slot_to_coeff_matrix,
        BootstrapKeySet,
        EvalModPlan,
        EvalModPrecomputed,
    };
    use crate::ckks::bootstrapping::linear_transform::repeat_block_slots;
    use crate::ckks::bootstrapping::eval_mod::sine_taylor_coefficients;
    use crate::rns::{RnsCkksContext, RnsCkksParams, RnsPrime};
    use num_complex::Complex64;

    fn sample_dense_bootstrap_context() -> RnsCkksContext {
        RnsCkksContext::new(RnsCkksParams {
            poly_degree: 16,
            primes: vec![
                RnsPrime::new(998_244_353, 3),
                RnsPrime::new(1_004_535_809, 3),
                RnsPrime::new(469_762_049, 3),
                RnsPrime::new(754_974_721, 11),
                RnsPrime::new(1_224_736_769, 3),
                RnsPrime::new(2_013_265_921, 31),
            ],
            aux_primes: vec![RnsPrime::new(167_772_161, 3)],
            scale_bits: 28,
        })
        .expect("sample dense bootstrap CKKS context must build")
    }

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

    #[test]
    fn dense_coeff_to_slot_exact_matches_plaintext_pipeline() {
        let context = sample_dense_bootstrap_context();
        let key_pair = context.keygen(16);
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
        let ciphertext = context.encrypt(&context.encode(&input), &key_pair.public_key);
        let projected = mod_raise(&context, &ciphertext).project_to_q_ciphertext();
        let conjugation_key = context.generate_conjugation_key_at_level(
            &key_pair.secret_key,
            &key_pair.public_key,
            projected.level,
        );
        let conjugated = context.conjugate(&projected, &conjugation_key);

        let (upper_ct, lower_ct) = dense_coeff_to_slot_exact(
            &context,
            &projected,
            &conjugated,
            &key_pair.secret_key,
            &key_pair.public_key,
        )
        .expect("dense exact CoeffToSlot must succeed");
        let upper = context.decode_at_scale(
            &context.decrypt(&upper_ct, &key_pair.secret_key),
            upper_ct.scale_bits,
        );
        let lower = context.decode_at_scale(
            &context.decrypt(&lower_ct, &key_pair.secret_key),
            lower_ct.scale_bits,
        );

        let coeff_to_slot = exact_dense_coeff_to_slot_matrices(&context)
            .expect("exact dense CoeffToSlot matrices must build");
        let input_conjugates = conjugate_vector(&input);
        let expected_upper = add_vectors(
            &apply_matrix(&coeff_to_slot.upper_from_slots, &input),
            &apply_matrix(&coeff_to_slot.upper_from_conjugates, &input_conjugates),
        );
        let expected_lower = add_vectors(
            &apply_matrix(&coeff_to_slot.lower_from_slots, &input),
            &apply_matrix(&coeff_to_slot.lower_from_conjugates, &input_conjugates),
        );

        for (index, expected_slot) in expected_upper.iter().enumerate() {
            assert!(
                (upper[index] - *expected_slot).norm() <= 7e-2,
                "upper expected {expected_slot:?}, got {:?} at slot {index}",
                upper[index]
            );
            assert!(
                (lower[index] - expected_lower[index]).norm() <= 7e-2,
                "lower expected {:?}, got {:?} at slot {index}",
                expected_lower[index],
                lower[index]
            );
        }
    }

    #[test]
    fn dense_coeff_to_slot_then_slot_to_coeff_exact_recovers_input() {
        let context = sample_dense_bootstrap_context();
        let key_pair = context.keygen(16);
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
        let ciphertext = context.encrypt(&context.encode(&input), &key_pair.public_key);
        let projected = mod_raise(&context, &ciphertext).project_to_q_ciphertext();
        let conjugation_key = context.generate_conjugation_key_at_level(
            &key_pair.secret_key,
            &key_pair.public_key,
            projected.level,
        );
        let conjugated = context.conjugate(&projected, &conjugation_key);
        let (upper_ct, lower_ct) = dense_coeff_to_slot_exact(
            &context,
            &projected,
            &conjugated,
            &key_pair.secret_key,
            &key_pair.public_key,
        )
        .expect("dense exact CoeffToSlot must succeed");

        let recovered_ct = dense_slot_to_coeff_exact(
            &context,
            &upper_ct,
            &lower_ct,
            &key_pair.secret_key,
            &key_pair.public_key,
        )
        .expect("dense exact SlotToCoeff must succeed");
        let recovered = context.decode_at_scale(
            &context.decrypt(&recovered_ct, &key_pair.secret_key),
            recovered_ct.scale_bits,
        );

        for (index, expected_slot) in input.iter().enumerate() {
            assert!(
                (recovered[index] - *expected_slot).norm() <= 7e-2,
                "expected {expected_slot:?}, got {:?} at slot {index}",
                recovered[index]
            );
        }
    }

    #[test]
    fn dense_coeff_to_slot_then_eval_mod_matches_plaintext_pipeline() {
        let context = sample_dense_bootstrap_context();
        let key_pair = context.keygen(16);
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
        let ciphertext = context.encrypt(&context.encode(&input), &key_pair.public_key);
        let projected = mod_raise(&context, &ciphertext).project_to_q_ciphertext();
        let conjugation_key = context.generate_conjugation_key_at_level(
            &key_pair.secret_key,
            &key_pair.public_key,
            projected.level,
        );
        let conjugated = context.conjugate(&projected, &conjugation_key);
        let (upper_ct, lower_ct) = dense_coeff_to_slot_exact(
            &context,
            &projected,
            &conjugated,
            &key_pair.secret_key,
            &key_pair.public_key,
        )
        .expect("dense exact CoeffToSlot must succeed");

        let eval_mod_plan = EvalModPlan::from_ciphertext(
            &context,
            &upper_ct,
            context.num_slots(),
            1.0,
            sine_scale,
            polynomial_degree,
        )
        .expect("EvalMod plan construction must succeed");
        let eval_mod_precomputed = EvalModPrecomputed::sine_taylor(eval_mod_plan)
            .expect("EvalMod sine Taylor precomputation must succeed");
        let eval_mod_keys = BootstrapKeySet::for_eval_mod(
            &context,
            &key_pair,
            upper_ct.level,
            eval_mod_precomputed.coefficients.len(),
        );
        let evaluated_upper_ct = eval_mod(
            &context,
            &upper_ct,
            &eval_mod_precomputed,
            &eval_mod_keys.eval_mod_relinearization_keys,
        )
        .expect("EvalMod on dense upper branch must succeed");
        let evaluated_lower_ct = eval_mod(
            &context,
            &lower_ct,
            &eval_mod_precomputed,
            &eval_mod_keys.eval_mod_relinearization_keys,
        )
        .expect("EvalMod on dense lower branch must succeed");
        let evaluated_upper = context.decode_at_scale(
            &context.decrypt(&evaluated_upper_ct, &key_pair.secret_key),
            evaluated_upper_ct.scale_bits,
        );
        let evaluated_lower = context.decode_at_scale(
            &context.decrypt(&evaluated_lower_ct, &key_pair.secret_key),
            evaluated_lower_ct.scale_bits,
        );

        let coeff_to_slot = exact_dense_coeff_to_slot_matrices(&context)
            .expect("exact dense CoeffToSlot matrices must build");
        let input_conjugates = conjugate_vector(&input);
        let upper = add_vectors(
            &apply_matrix(&coeff_to_slot.upper_from_slots, &input),
            &apply_matrix(&coeff_to_slot.upper_from_conjugates, &input_conjugates),
        );
        let lower = add_vectors(
            &apply_matrix(&coeff_to_slot.lower_from_slots, &input),
            &apply_matrix(&coeff_to_slot.lower_from_conjugates, &input_conjugates),
        );
        let coefficients = sine_taylor_coefficients(polynomial_degree, sine_scale);
        let expected_upper: Vec<Complex64> = upper
            .iter()
            .map(|&value| evaluate_polynomial(&coefficients, value))
            .collect();
        let expected_lower: Vec<Complex64> = lower
            .iter()
            .map(|&value| evaluate_polynomial(&coefficients, value))
            .collect();

        for (index, expected_slot) in expected_upper.iter().enumerate() {
            assert!(
                (evaluated_upper[index] - *expected_slot).norm() <= 7e-2,
                "upper expected {expected_slot:?}, got {:?} at slot {index}",
                evaluated_upper[index]
            );
            assert!(
                (evaluated_lower[index] - expected_lower[index]).norm() <= 7e-2,
                "lower expected {:?}, got {:?} at slot {index}",
                expected_lower[index],
                evaluated_lower[index]
            );
        }
    }

    #[test]
    fn bootstrap_exact_dense_eval_mod_matches_plaintext_pipeline() {
        let context = sample_dense_bootstrap_context();
        let key_pair = context.keygen(16);

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
        let ciphertext = context.encrypt(&context.encode(&input), &key_pair.public_key);

        let recovered_ct = bootstrap_exact_dense_eval_mod_roundtrip(
            &context,
            &ciphertext,
            &key_pair,
            sine_scale,
            polynomial_degree,
        )
        .expect("exact dense EvalMod bootstrapping roundtrip must succeed");
        let recovered = context.decode_at_scale(
            &context.decrypt(&recovered_ct, &key_pair.secret_key),
            recovered_ct.scale_bits,
        );

        let coeff_to_slot = exact_dense_coeff_to_slot_matrices(&context)
            .expect("exact dense CoeffToSlot matrices must build");
        let slot_to_coeff = exact_dense_slot_to_coeff_matrices(&context)
            .expect("exact dense SlotToCoeff matrices must build");
        let input_conjugates = conjugate_vector(&input);
        let upper = add_vectors(
            &apply_matrix(&coeff_to_slot.upper_from_slots, &input),
            &apply_matrix(&coeff_to_slot.upper_from_conjugates, &input_conjugates),
        );
        let lower = add_vectors(
            &apply_matrix(&coeff_to_slot.lower_from_slots, &input),
            &apply_matrix(&coeff_to_slot.lower_from_conjugates, &input_conjugates),
        );
        let coefficients = sine_taylor_coefficients(polynomial_degree, sine_scale);
        let evaluated_upper: Vec<Complex64> = upper
            .iter()
            .map(|&value| evaluate_polynomial(&coefficients, value))
            .collect();
        let evaluated_lower: Vec<Complex64> = lower
            .iter()
            .map(|&value| evaluate_polynomial(&coefficients, value))
            .collect();
        let expected = add_vectors(
            &apply_matrix(&slot_to_coeff.slots_from_upper, &evaluated_upper),
            &apply_matrix(&slot_to_coeff.slots_from_lower, &evaluated_lower),
        );

        for (index, expected_slot) in expected.iter().enumerate() {
            assert!(
                (recovered[index] - *expected_slot).norm() <= 7e-2,
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

    fn add_vectors(lhs: &[Complex64], rhs: &[Complex64]) -> Vec<Complex64> {
        lhs.iter().zip(rhs.iter()).map(|(left, right)| *left + *right).collect()
    }

    fn conjugate_vector(values: &[Complex64]) -> Vec<Complex64> {
        values.iter().map(|value| value.conj()).collect()
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