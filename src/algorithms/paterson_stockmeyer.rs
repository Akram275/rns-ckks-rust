//Paterson-Stockmeyer evaluation of a plaintext polynomial on an encrypted CKKS ciphertext.
//
// Contract:
// - `coefficients[i]` is the real coefficient of `x^i`,
// - the polynomial is evaluated slotwise on the encrypted input ciphertext,
// - coefficients are plaintext scalars shared across all slots,
// - ciphertext-ciphertext multiplications use level-specific relinearization keys,
// - ciphertext levels are aligned by modulus dropping,
// - ciphertext scales are aligned by lifting with plaintext-one multiplications.

use std::collections::BTreeMap;

use num_complex::Complex64;

use crate::rns::ops::multiply_plain_rns;
use crate::rns::{RnsCiphertext, RnsCkksContext, RnsKeySwitchingKey, RnsPolynomial, RnsPublicKey, RnsSecretKey};

pub fn paterson_stockmeyer_block_size(term_count: usize) -> usize {
    assert!(term_count > 0, "term_count must be positive");

    let mut block = 1usize;
    while block * block < term_count {
        block += 1;
    }
    block
}

pub fn paterson_stockmeyer_eval(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    coefficients: &[f64],
    relin_keys: &BTreeMap<usize, RnsKeySwitchingKey>,
) -> RnsCiphertext {
    assert!(!coefficients.is_empty(), "polynomial must contain at least one coefficient");

    let block_size = paterson_stockmeyer_block_size(coefficients.len());
    let block_count = coefficients.len().div_ceil(block_size);

    if coefficients.len() == 1 {
        return constant_ciphertext(
            context,
            ciphertext.level,
            ciphertext.scale_bits,
            coefficients[0],
        );
    }

    let working_scale_bits = paterson_stockmeyer_working_scale_bits(context, ciphertext.level, coefficients.len());
    let coefficient_scale_bits = working_scale_bits
        .checked_sub(ciphertext.scale_bits)
        .expect("Paterson-Stockmeyer working scale must dominate the input ciphertext scale");

    let baby_powers = precompute_baby_powers(
        context,
        ciphertext,
        block_size,
        ciphertext.scale_bits,
        relin_keys,
    );
    let giant_step = baby_powers
        .last()
        .expect("non-constant polynomial must produce a giant step")
        .clone();

    assert!(
        giant_step.level + 1 >= block_count,
        "polynomial degree is too large for the available CKKS level budget"
    );

    let top_level = giant_step.level;
    let blocks: Vec<RnsCiphertext> = (0..block_count)
        .map(|block_index| {
            let remaining_outer_mults = block_count - 1 - block_index;
            let target_level = top_level - remaining_outer_mults;
            evaluate_block(
                context,
                &baby_powers,
                coefficients,
                block_index,
                block_size,
                target_level,
                working_scale_bits,
                coefficient_scale_bits,
            )
        })
        .collect();

    let mut result = blocks
        .last()
        .expect("Paterson-Stockmeyer requires at least one block")
        .clone();

    for block in blocks[..blocks.len() - 1].iter().rev() {
        let aligned_giant_step = lift_ciphertext_scale(
            context,
            &align_ciphertext_to_level(&giant_step, result.level),
            result.scale_bits,
        );
        let relin_key = relin_keys
            .get(&result.level)
            .unwrap_or_else(|| panic!("missing Paterson-Stockmeyer relinearization key at level {}", result.level));
        let multiplied = context.mult_relinearized(&result, &aligned_giant_step, relin_key);
        let lifted = lift_ciphertext_scale(context, &multiplied, result.scale_bits);
        result = lifted.add(block);
    }

    result
}

pub fn paterson_stockmeyer_required_levels(ciphertext_level: usize, term_count: usize) -> Vec<usize> {
    assert!(term_count > 0, "term_count must be positive");
    if term_count == 1 {
        return Vec::new();
    }

    let block_size = paterson_stockmeyer_block_size(term_count);
    let block_count = term_count.div_ceil(block_size);
    let top_level = ciphertext_level
        .checked_sub(block_size - 1)
        .expect("polynomial degree is too large for the available CKKS level budget");
    let final_level = top_level
        .checked_sub(block_count - 1)
        .expect("polynomial degree is too large for the available CKKS level budget");

    ((final_level + 1)..=ciphertext_level).rev().collect()
}

pub fn generate_paterson_stockmeyer_relinearization_keys(
    context: &RnsCkksContext,
    secret_key: &RnsSecretKey,
    public_key: &RnsPublicKey,
    ciphertext_level: usize,
    term_count: usize,
) -> BTreeMap<usize, RnsKeySwitchingKey> {
    paterson_stockmeyer_required_levels(ciphertext_level, term_count)
        .into_iter()
        .map(|level| {
            (
                level,
                context.generate_relinearization_key_at_level(secret_key, public_key, level),
            )
        })
        .collect()
}

fn precompute_baby_powers(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    block_size: usize,
    working_scale_bits: usize,
    relin_keys: &BTreeMap<usize, RnsKeySwitchingKey>,
) -> Vec<RnsCiphertext> {
    let mut powers = Vec::with_capacity(block_size);
    let lifted_base = lift_ciphertext_scale(context, ciphertext, working_scale_bits);
    powers.push(lifted_base.clone());

    for exponent in 2..=block_size {
        let previous = powers
            .last()
            .expect("a previous baby power must exist")
            .clone();
        let aligned_previous = lift_ciphertext_scale(context, &previous, working_scale_bits);
        let aligned_base = align_ciphertext_to_level(&lifted_base, previous.level);
        let relin_key = relin_keys
            .get(&previous.level)
            .unwrap_or_else(|| panic!("missing Paterson-Stockmeyer relinearization key at level {}", previous.level));
        let next_power = context.mult_relinearized(&aligned_previous, &aligned_base, relin_key);
        debug_assert_eq!(next_power.level + exponent - 1, ciphertext.level);
        powers.push(lift_ciphertext_scale(context, &next_power, working_scale_bits));
    }

    powers
}

fn evaluate_block(
    context: &RnsCkksContext,
    baby_powers: &[RnsCiphertext],
    coefficients: &[f64],
    block_index: usize,
    block_size: usize,
    target_level: usize,
    working_scale_bits: usize,
    coefficient_scale_bits: usize,
) -> RnsCiphertext {
    let mut accumulator = constant_ciphertext(context, target_level, working_scale_bits, 0.0);

    for offset in 0..block_size {
        let coeff_index = block_index * block_size + offset;
        if coeff_index >= coefficients.len() {
            break;
        }

        let coefficient = coefficients[coeff_index];
        if coefficient == 0.0 {
            continue;
        }

        if offset == 0 {
            accumulator = add_constant_term(context, &accumulator, coefficient);
            continue;
        }

        let aligned_power = align_block_term_to_level(
            context,
            &baby_powers[offset - 1],
            target_level,
            baby_powers[offset - 1].scale_bits,
        );
        let scaled_term = mult_plain_slots_real_at_scale(
            context,
            &aligned_power,
            &constant_slots(context, coefficient),
            coefficient_scale_bits,
        );
        accumulator = accumulator.add(&scaled_term);
    }

    accumulator
}

fn align_ciphertext_to_level(
    ciphertext: &RnsCiphertext,
    target_level: usize,
) -> RnsCiphertext {
    assert!(target_level <= ciphertext.level, "target level cannot exceed ciphertext level");

    ciphertext.truncate_levels(target_level + 1)
}

fn lift_ciphertext_scale(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    target_scale_bits: usize,
) -> RnsCiphertext {
    assert!(
        ciphertext.scale_bits <= target_scale_bits,
        "cannot lower ciphertext scale with a plaintext-one lift"
    );
    if ciphertext.scale_bits == target_scale_bits {
        return ciphertext.clone();
    }

    let plain = constant_plaintext(
        context,
        1.0,
        ciphertext.level,
        target_scale_bits - ciphertext.scale_bits,
    );
    let level_context = context
        .level_context(ciphertext.level)
        .expect("ciphertext level context must exist for scale lifting");
    let lifted = multiply_plain_rns(ciphertext, &plain, &level_context);
    RnsCiphertext::new(lifted.c0, lifted.c1, target_scale_bits)
}

fn mult_plain_slots_real_at_scale(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    plain_slots: &[f64],
    plain_scale_bits: usize,
) -> RnsCiphertext {
    let complex_slots: Vec<Complex64> = plain_slots
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect();
    let plain = context.encode_at_level_and_scale(&complex_slots, ciphertext.level, plain_scale_bits);
    let level_context = context
        .level_context(ciphertext.level)
        .expect("ciphertext level context must exist for plaintext multiplication");
    let multiplied = multiply_plain_rns(ciphertext, &plain, &level_context);
    RnsCiphertext::new(
        multiplied.c0,
        multiplied.c1,
        ciphertext.scale_bits + plain_scale_bits,
    )
}

fn align_block_term_to_level(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    target_level: usize,
    working_scale_bits: usize,
) -> RnsCiphertext {
    assert!(target_level <= ciphertext.level, "target level cannot exceed ciphertext level");

    let aligned = ciphertext.truncate_levels(target_level + 1);
    lift_ciphertext_scale(context, &aligned, working_scale_bits)
}

fn paterson_stockmeyer_working_scale_bits(
    context: &RnsCkksContext,
    ciphertext_level: usize,
    term_count: usize,
) -> usize {
    paterson_stockmeyer_required_levels(ciphertext_level, term_count)
        .into_iter()
        .map(|level| {
            let dropped_prime = context
                .level_context(level)
                .expect("ciphertext level context must exist for working-scale selection")
                .primes()
                .last()
                .expect("Paterson-Stockmeyer requires a last prime for every multiplication level")
                .modulus;
            (u64::BITS as usize - 1) - dropped_prime.leading_zeros() as usize
        })
        .min()
        .expect("Paterson-Stockmeyer multiplication levels must be non-empty")
}

fn constant_ciphertext(
    context: &RnsCkksContext,
    level: usize,
    scale_bits: usize,
    constant: f64,
) -> RnsCiphertext {
    let level_context = context
        .level_context(level)
        .expect("ciphertext level context must exist for constant construction");
    let zero = level_context.zero_poly();
    let ciphertext = RnsCiphertext::new(zero.clone(), zero, scale_bits);
    add_constant_term(context, &ciphertext, constant)
}

fn add_constant_term(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    constant: f64,
) -> RnsCiphertext {
    if constant == 0.0 {
        return ciphertext.clone();
    }

    let plain = constant_plaintext(context, constant, ciphertext.level, ciphertext.scale_bits);
    RnsCiphertext::new(ciphertext.c0.add(&plain), ciphertext.c1.clone(), ciphertext.scale_bits)
}

fn constant_plaintext(
    context: &RnsCkksContext,
    constant: f64,
    level: usize,
    scale_bits: usize,
) -> RnsPolynomial {
    let slots = vec![Complex64::new(constant, 0.0); context.num_slots()];
    context.encode_at_level_and_scale(&slots, level, scale_bits)
}

fn constant_slots(context: &RnsCkksContext, constant: f64) -> Vec<f64> {
    vec![constant; context.num_slots()]
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        evaluate_block,
        precompute_baby_powers,
        generate_paterson_stockmeyer_relinearization_keys,
        paterson_stockmeyer_block_size,
        paterson_stockmeyer_eval,
        paterson_stockmeyer_required_levels,
    };
    use crate::rns::{RnsCiphertext, RnsCkksContext, RnsCkksParams, RnsKeyPair, RnsKeySwitchingKey};

    #[test]
    fn paterson_stockmeyer_block_size_is_ceil_sqrt() {
        assert_eq!(paterson_stockmeyer_block_size(1), 1);
        assert_eq!(paterson_stockmeyer_block_size(2), 2);
        assert_eq!(paterson_stockmeyer_block_size(4), 2);
        assert_eq!(paterson_stockmeyer_block_size(5), 3);
        assert_eq!(paterson_stockmeyer_block_size(9), 3);
    }

    #[test]
    fn paterson_stockmeyer_required_levels_cover_all_multiplication_levels() {
        assert_eq!(paterson_stockmeyer_required_levels(7, 1), Vec::<usize>::new());
        assert_eq!(paterson_stockmeyer_required_levels(7, 5), vec![7, 6, 5]);
    }

    #[test]
    fn paterson_stockmeyer_eval_matches_slotwise_polynomial_on_realistic_params() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let coefficients = [0.75_f64, -0.5, 1.25, -0.25, 0.125];
        let input = [0.125_f64, -0.2, 0.05, 0.3];
        let expected: Vec<f64> = input
            .iter()
            .map(|&value| {
                coefficients
                    .iter()
                    .enumerate()
                    .map(|(power, &coeff)| coeff * value.powi(power as i32))
                    .sum()
            })
            .collect();

        let mut padded = vec![0.0_f64; context.num_slots()];
        padded[..input.len()].copy_from_slice(&input);
        let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
        let relin_keys = generate_paterson_stockmeyer_relinearization_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            coefficients.len(),
        );

        let evaluated = paterson_stockmeyer_eval(&context, &ciphertext, &coefficients, &relin_keys);
        let recovered = context.decode_real_at_scale(
            &context.decrypt(&evaluated, &key_pair.secret_key),
            evaluated.scale_bits,
        );

        for (index, &expected_value) in expected.iter().enumerate() {
            assert!(
                (recovered[index] - expected_value).abs() <= 2e-2,
                "expected {expected_value}, got {} at slot {index}",
                recovered[index]
            );
        }
    }

    #[test]
    fn paterson_stockmeyer_matches_horner_reference_on_realistic_params() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let coefficients = [0.75_f64, -0.5, 1.25, -0.25, 0.125];
        let input = [0.125_f64, -0.2, 0.05, 0.3];

        let mut padded = vec![0.0_f64; context.num_slots()];
        padded[..input.len()].copy_from_slice(&input);
        let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
        let relin_keys = generate_paterson_stockmeyer_relinearization_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            coefficients.len(),
        );
        let mut horner_relin_keys = relin_keys.clone();
        horner_relin_keys.insert(
            4,
            context.generate_relinearization_key_at_level(&key_pair.secret_key, &key_pair.public_key, 4),
        );

        let paterson_stockmeyer = paterson_stockmeyer_eval(&context, &ciphertext, &coefficients, &relin_keys);
        let horner = horner_reference_eval(
            &context,
            &ciphertext,
            &coefficients,
            &horner_relin_keys,
            &key_pair,
        );

        let ps_slots = context.decode_real_at_scale(
            &context.decrypt(&paterson_stockmeyer, &key_pair.secret_key),
            paterson_stockmeyer.scale_bits,
        );
        let horner_slots = context.decode_real_at_scale(
            &context.decrypt(&horner, &key_pair.secret_key),
            horner.scale_bits,
        );

        for index in 0..input.len() {
            assert!(
                (ps_slots[index] - horner_slots[index]).abs() <= 2e-2,
                "PS/Horner mismatch at slot {index}: {} vs {}",
                ps_slots[index],
                horner_slots[index]
            );
        }
    }

    #[test]
    fn paterson_stockmeyer_baby_powers_match_monomial_reference() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let input = [0.125_f64, -0.2, 0.05, 0.3];
        let mut padded = vec![0.0_f64; context.num_slots()];
        padded[..input.len()].copy_from_slice(&input);
        let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
        let relin_keys = generate_paterson_stockmeyer_relinearization_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            5,
        );
        let baby_powers = precompute_baby_powers(&context, &ciphertext, 3, ciphertext.scale_bits, &relin_keys);

        let monomial = horner_reference_eval(
            &context,
            &ciphertext,
            &[0.0, 0.0, 1.0],
            &relin_keys,
            &key_pair,
        );

        let baby_slots = context.decode_real_at_scale(
            &context.decrypt(&super::align_ciphertext_to_level(&baby_powers[1], monomial.level), &key_pair.secret_key),
            baby_powers[1].scale_bits,
        );
        let monomial_slots = context.decode_real_at_scale(
            &context.decrypt(&monomial, &key_pair.secret_key),
            monomial.scale_bits,
        );

        for index in 0..input.len() {
            assert!(
                (baby_slots[index] - monomial_slots[index]).abs() <= 2e-2,
                "baby x^2 mismatch at slot {index}: {} vs {}",
                baby_slots[index],
                monomial_slots[index]
            );
        }
    }

    #[test]
    fn paterson_stockmeyer_giant_step_matches_cubic_reference() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let input = [0.125_f64, -0.2, 0.05, 0.3];
        let mut padded = vec![0.0_f64; context.num_slots()];
        padded[..input.len()].copy_from_slice(&input);
        let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
        let relin_keys = generate_paterson_stockmeyer_relinearization_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            5,
        );
        let baby_powers = precompute_baby_powers(&context, &ciphertext, 3, ciphertext.scale_bits, &relin_keys);
        let giant_step = baby_powers.last().expect("giant step must exist");

        let recovered = context.decode_real_at_scale(
            &context.decrypt(giant_step, &key_pair.secret_key),
            giant_step.scale_bits,
        );

        for (index, &value) in input.iter().enumerate() {
            let expected = value.powi(3);
            assert!(
                (recovered[index] - expected).abs() <= 2e-2,
                "giant step mismatch at slot {index}: {} vs {}",
                recovered[index],
                expected
            );
        }
    }

    #[test]
    fn paterson_stockmeyer_outer_giant_product_matches_expected_term() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let coefficients = [0.75_f64, -0.5, 1.25, -0.25, 0.125];
        let input = [0.125_f64, -0.2, 0.05, 0.3];
        let mut padded = vec![0.0_f64; context.num_slots()];
        padded[..input.len()].copy_from_slice(&input);
        let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
        let relin_keys = generate_paterson_stockmeyer_relinearization_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            coefficients.len(),
        );
        let working_scale_bits = super::paterson_stockmeyer_working_scale_bits(
            &context,
            ciphertext.level,
            coefficients.len(),
        );
        let baby_powers = precompute_baby_powers(&context, &ciphertext, 3, ciphertext.scale_bits, &relin_keys);
        let giant_step = baby_powers.last().expect("giant step must exist").clone();
        let block1 = evaluate_block(
            &context,
            &baby_powers,
            &coefficients,
            1,
            3,
            5,
            working_scale_bits,
            working_scale_bits - ciphertext.scale_bits,
        );
        let aligned_giant_step = super::lift_ciphertext_scale(
            &context,
            &super::align_ciphertext_to_level(&giant_step, block1.level),
            block1.scale_bits,
        );
        let multiplied = context.mult_relinearized(
            &block1,
            &aligned_giant_step,
            relin_keys.get(&block1.level).expect("missing giant-step relin key"),
        );
        let outer_term = super::lift_ciphertext_scale(&context, &multiplied, block1.scale_bits);
        let recovered = context.decode_real_at_scale(
            &context.decrypt(&outer_term, &key_pair.secret_key),
            outer_term.scale_bits,
        );

        for (index, &value) in input.iter().enumerate() {
            let expected = -0.25 * value.powi(3) + 0.125 * value.powi(4);
            assert!(
                (recovered[index] - expected).abs() <= 2e-2,
                "outer giant term mismatch at slot {index}: {} vs {}",
                recovered[index],
                expected
            );
        }
    }

    #[test]
    fn paterson_stockmeyer_block0_terms_match_expected_values() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let input = [0.125_f64, -0.2, 0.05, 0.3];
        let mut padded = vec![0.0_f64; context.num_slots()];
        padded[..input.len()].copy_from_slice(&input);
        let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
        let relin_keys = generate_paterson_stockmeyer_relinearization_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            5,
        );
        let working_scale_bits = super::paterson_stockmeyer_working_scale_bits(&context, ciphertext.level, 5);
        let coefficient_scale_bits = working_scale_bits - ciphertext.scale_bits;
        let baby_powers = precompute_baby_powers(&context, &ciphertext, 3, ciphertext.scale_bits, &relin_keys);

        let linear_term = super::mult_plain_slots_real_at_scale(
            &context,
            &super::align_block_term_to_level(&context, &baby_powers[0], 4, working_scale_bits),
            &vec![-0.5_f64; context.num_slots()],
            coefficient_scale_bits,
        );
        let quadratic_term = super::mult_plain_slots_real_at_scale(
            &context,
            &super::align_block_term_to_level(&context, &baby_powers[1], 4, working_scale_bits),
            &vec![1.25_f64; context.num_slots()],
            coefficient_scale_bits,
        );

        let linear_slots = context.decode_real_at_scale(
            &context.decrypt(&linear_term, &key_pair.secret_key),
            linear_term.scale_bits,
        );
        let quadratic_slots = context.decode_real_at_scale(
            &context.decrypt(&quadratic_term, &key_pair.secret_key),
            quadratic_term.scale_bits,
        );

        for (index, &value) in input.iter().enumerate() {
            let expected_linear = -0.5 * value;
            let expected_quadratic = 1.25 * value.powi(2);
            assert!(
                (linear_slots[index] - expected_linear).abs() <= 2e-2,
                "linear term mismatch at slot {index}: {} vs {}",
                linear_slots[index],
                expected_linear
            );
            assert!(
                (quadratic_slots[index] - expected_quadratic).abs() <= 2e-2,
                "quadratic term mismatch at slot {index}: {} vs {}",
                quadratic_slots[index],
                expected_quadratic
            );
        }
    }

    fn horner_reference_eval(
        context: &RnsCkksContext,
        ciphertext: &RnsCiphertext,
        coefficients: &[f64],
        relin_keys: &BTreeMap<usize, RnsKeySwitchingKey>,
        key_pair: &RnsKeyPair,
    ) -> RnsCiphertext {
        let working_scale_bits = super::paterson_stockmeyer_working_scale_bits(
            context,
            ciphertext.level,
            coefficients.len(),
        );
        let x = super::lift_ciphertext_scale(context, ciphertext, working_scale_bits);
        let mut result = super::constant_ciphertext(
            context,
            ciphertext.level,
            working_scale_bits,
            *coefficients.last().expect("coefficients must be non-empty"),
        );

        for &coefficient in coefficients[..coefficients.len() - 1].iter().rev() {
            let aligned_x = super::lift_ciphertext_scale(
                context,
                &super::align_ciphertext_to_level(&x, result.level),
                result.scale_bits,
            );
            let relin_key = relin_keys
                .get(&result.level)
                .unwrap_or_else(|| panic!("missing Horner relinearization key at level {}", result.level));
            let multiplied = context.mult_relinearized(&result, &aligned_x, relin_key);
            let lifted = super::lift_ciphertext_scale(context, &multiplied, result.scale_bits);
            result = super::add_constant_term(context, &lifted, coefficient);
        }

        let decrypted = context.decode_real_at_scale(
            &context.decrypt(&result, &key_pair.secret_key),
            result.scale_bits,
        );
        debug_assert!(decrypted[0].is_finite());
        result
    }
}