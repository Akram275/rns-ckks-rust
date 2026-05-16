//Paterson-Stockmeyer evaluation of a plaintext polynomial on an encrypted CKKS ciphertext.
//
// Contract:
// - `coefficients[i]` is the real coefficient of `x^i`,
// - the polynomial is evaluated slotwise on the encrypted input ciphertext,
// - coefficients are plaintext scalars shared across all slots,
// - ciphertext-ciphertext multiplications use level-specific relinearization keys,
// - CKKS level alignment is handled by repeated multiplication by the plaintext value `1`.
//
//This first implementation uses a conservative block schedule with `m = 1`.
//That is the Horner boundary case of Paterson-Stockmeyer, chosen here because
//the current CKKS surface does not yet expose an explicit modulus-switching
//primitive for aggressively reusing larger baby-step blocks.

use std::collections::BTreeMap;

use num_complex::Complex64;

use crate::rns::{RnsCiphertext, RnsCkksContext, RnsKeySwitchingKey, RnsPolynomial, RnsPublicKey, RnsSecretKey};

pub fn paterson_stockmeyer_block_size(term_count: usize) -> usize {
    assert!(term_count > 0, "term_count must be positive");
    1
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
        return constant_ciphertext(context, ciphertext.level, ciphertext.scale_bits, coefficients[0]);
    }

    let baby_powers = precompute_baby_powers(context, ciphertext, block_size, relin_keys);
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
            )
        })
        .collect();

    let mut result = blocks
        .last()
        .expect("Paterson-Stockmeyer requires at least one block")
        .clone();

    for block in blocks[..blocks.len() - 1].iter().rev() {
        let aligned_giant_step = align_ciphertext_to_level(context, &giant_step, result.level);
        let relin_key = relin_keys
            .get(&result.level)
            .unwrap_or_else(|| panic!("missing Paterson-Stockmeyer relinearization key at level {}", result.level));
        let multiplied = context.mult_relinearized(&result, &aligned_giant_step, relin_key);
        result = multiplied.add(block);
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
    relin_keys: &BTreeMap<usize, RnsKeySwitchingKey>,
) -> Vec<RnsCiphertext> {
    let mut powers = Vec::with_capacity(block_size);
    powers.push(ciphertext.clone());

    for exponent in 2..=block_size {
        let previous = powers
            .last()
            .expect("a previous baby power must exist")
            .clone();
        let aligned_base = align_ciphertext_to_level(context, ciphertext, previous.level);
        let relin_key = relin_keys
            .get(&previous.level)
            .unwrap_or_else(|| panic!("missing Paterson-Stockmeyer relinearization key at level {}", previous.level));
        let next_power = context.mult_relinearized(&previous, &aligned_base, relin_key);
        debug_assert_eq!(next_power.level + exponent - 1, ciphertext.level);
        powers.push(next_power);
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
) -> RnsCiphertext {
    let mut accumulator: Option<RnsCiphertext> = None;
    let mut constant_term = 0.0_f64;

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
            constant_term = coefficient;
            continue;
        }

        let aligned_power = align_ciphertext_to_level(context, &baby_powers[offset - 1], target_level + 1);
        let scaled_term = context.mult_plain_slots_real(&aligned_power, &constant_slots(context, coefficient));
        accumulator = Some(match accumulator {
            Some(current) => current.add(&scaled_term),
            None => scaled_term,
        });
    }

    let mut accumulator = accumulator.unwrap_or_else(|| {
        let fallback_scale_bits = baby_powers
            .first()
            .map(|power| align_ciphertext_to_level(context, power, target_level).scale_bits)
            .unwrap_or(context.params().scale_bits);
        constant_ciphertext(context, target_level, fallback_scale_bits, 0.0)
    });
    if constant_term != 0.0 {
        accumulator = add_constant_term(context, &accumulator, constant_term);
    }
    accumulator
}

fn align_ciphertext_to_level(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    target_level: usize,
) -> RnsCiphertext {
    assert!(target_level <= ciphertext.level, "target level cannot exceed ciphertext level");

    let mut aligned = ciphertext.clone();
    while aligned.level > target_level {
        aligned = level_down_by_one(context, &aligned);
    }
    aligned
}

fn level_down_by_one(context: &RnsCkksContext, ciphertext: &RnsCiphertext) -> RnsCiphertext {
    context.mult_plain_slots_real(ciphertext, &constant_slots(context, 1.0))
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
    use super::{
        generate_paterson_stockmeyer_relinearization_keys,
        paterson_stockmeyer_block_size,
        paterson_stockmeyer_eval,
        paterson_stockmeyer_required_levels,
    };
    use crate::rns::{RnsCkksContext, RnsCkksParams};

    #[test]
    fn paterson_stockmeyer_block_size_is_currently_conservative() {
        assert_eq!(paterson_stockmeyer_block_size(1), 1);
        assert_eq!(paterson_stockmeyer_block_size(2), 1);
        assert_eq!(paterson_stockmeyer_block_size(4), 1);
        assert_eq!(paterson_stockmeyer_block_size(5), 1);
        assert_eq!(paterson_stockmeyer_block_size(9), 1);
    }

    #[test]
    fn paterson_stockmeyer_required_levels_cover_all_multiplication_levels() {
        assert_eq!(paterson_stockmeyer_required_levels(7, 1), Vec::<usize>::new());
        assert_eq!(paterson_stockmeyer_required_levels(7, 5), vec![7, 6, 5, 4]);
    }

    #[test]
    fn paterson_stockmeyer_eval_matches_affine_polynomial_on_realistic_params() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let coefficients = [0.75_f64, -0.5];
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
                (recovered[index] - expected_value).abs() <= 1.5e-2,
                "expected {expected_value}, got {} at slot {index}",
                recovered[index]
            );
        }
    }
}