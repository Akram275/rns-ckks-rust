use std::collections::BTreeMap;

use crate::rns::{RnsCkksContext, RnsCiphertext, RnsPublicKey, RnsRotationKey, RnsSecretKey};

pub fn plaintext_ciphertext_dot_product_rotation_steps(active_slots: usize) -> Vec<usize> {
    assert!(active_slots > 0, "active_slots must be positive");

    let mut steps = Vec::new();
    let mut step = 1usize;
    let fold_width = active_slots.next_power_of_two();
    while step < fold_width {
        steps.push(step);
        step <<= 1;
    }
    steps
}

pub fn generate_plaintext_ciphertext_dot_product_rotation_keys(
    context: &RnsCkksContext,
    secret_key: &RnsSecretKey,
    public_key: &RnsPublicKey,
    level: usize,
    active_slots: usize,
) -> BTreeMap<usize, RnsRotationKey> {
    plaintext_ciphertext_dot_product_rotation_steps(active_slots)
        .into_iter()
        .map(|step| {
            (
                step,
                context.generate_rotation_key_at_level(secret_key, public_key, step as isize, level),
            )
        })
        .collect()
}

pub fn plaintext_ciphertext_dot_product(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    plaintext_slots: &[f64],
    rotation_keys: &BTreeMap<usize, RnsRotationKey>,
) -> RnsCiphertext {
    assert!(
        plaintext_slots.len() <= context.num_slots(),
        "plaintext slot vector cannot exceed the CKKS slot count"
    );
    assert!(!plaintext_slots.is_empty(), "plaintext slot vector cannot be empty");

    let mut padded_slots = vec![0.0_f64; context.num_slots()];
    padded_slots[..plaintext_slots.len()].copy_from_slice(plaintext_slots);

    let mut accumulator = context.mult_plain_slots_real(ciphertext, &padded_slots);
    for step in plaintext_ciphertext_dot_product_rotation_steps(plaintext_slots.len()) {
        let rotation_key = rotation_keys
            .get(&step)
            .unwrap_or_else(|| panic!("missing rotation key for step {step}"));
        let rotated = context.rotate(&accumulator, rotation_key);
        accumulator = accumulator.add(&rotated);
    }

    accumulator
}

#[cfg(test)]
mod tests {
    use super::{
        plaintext_ciphertext_dot_product,
        plaintext_ciphertext_dot_product_rotation_steps,
    };
    use crate::rns::{RnsCkksContext, RnsCkksParams, RnsCiphertext, RnsKeySwitchingKey, RnsRotationKey, RNS_DECOMP_BITS};
    use num_bigint::BigUint;

    fn num_decomp_digits(total_modulus_bits: u64, decomp_bits: usize) -> usize {
        (total_modulus_bits as usize + decomp_bits - 1) / decomp_bits
    }

    fn noiseless_rotation_key(
        context: &RnsCkksContext,
        secret_key: &crate::rns::RnsSecretKey,
        steps: isize,
        level: usize,
    ) -> RnsRotationKey {
        let galois_element = context.galois_element_for_rotation(steps);
        let source = secret_key
            .apply_galois_automorphism(galois_element)
            .poly
            .truncate_levels(level + 1);
        let level_context = context
            .level_context(level)
            .expect("dot-product level context must exist");
        let modulus = level_context.total_modulus();
        let digits = num_decomp_digits(modulus.bits(), RNS_DECOMP_BITS);
        let mut base_power = BigUint::from(1u64);
        let base = BigUint::from(1u64) << RNS_DECOMP_BITS;
        let source_coeffs = source.reconstruct_coefficients(&level_context);
        let zero = level_context.zero_poly();
        let mut ksk = Vec::with_capacity(digits);

        for _ in 0..digits {
            let coeffs: Vec<BigUint> = source_coeffs
                .iter()
                .map(|coeff| (coeff * &base_power) % &modulus)
                .collect();
            let plaintext = level_context.poly_from_biguint_coeffs(&coeffs);
            ksk.push(RnsCiphertext::new(plaintext, zero.clone(), 0));
            base_power = (&base_power * &base) % &modulus;
        }

        RnsRotationKey {
            galois_element,
            keyswitch_key: RnsKeySwitchingKey {
                ksk,
                decomp_bits: RNS_DECOMP_BITS,
                level,
            },
        }
    }

    #[test]
    fn plaintext_ciphertext_dot_product_steps_follow_powers_of_two() {
        assert_eq!(plaintext_ciphertext_dot_product_rotation_steps(1), Vec::<usize>::new());
        assert_eq!(plaintext_ciphertext_dot_product_rotation_steps(4), vec![1, 2]);
        assert_eq!(plaintext_ciphertext_dot_product_rotation_steps(5), vec![1, 2, 4]);
    }

    #[test]
    fn plaintext_ciphertext_dot_product_matches_expected_scalar_on_realistic_params() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);
        let encrypted_slots = [0.125, -0.4, 0.03125, -0.125];
        let plaintext_slots = [2.0, -1.0, 0.5, 4.0];
        let expected: f64 = encrypted_slots
            .iter()
            .zip(plaintext_slots.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum();

        let mut padded_encrypted_slots = vec![0.0_f64; context.num_slots()];
        padded_encrypted_slots[..encrypted_slots.len()].copy_from_slice(&encrypted_slots);
        let ciphertext = context.encrypt(&context.encode_real(&padded_encrypted_slots), &key_pair.public_key);
        let rotation_keys = plaintext_ciphertext_dot_product_rotation_steps(plaintext_slots.len())
            .into_iter()
            .map(|step| (step, noiseless_rotation_key(&context, &key_pair.secret_key, step as isize, ciphertext.level - 1)))
            .collect();

        let dot_product = plaintext_ciphertext_dot_product(
            &context,
            &ciphertext,
            &plaintext_slots,
            &rotation_keys,
        );
        let recovered = context.decode_real_at_scale(
            &context.decrypt(&dot_product, &key_pair.secret_key),
            dot_product.scale_bits,
        );

        assert!((recovered[0] - expected).abs() <= 1.5e-2, "expected {expected}, got {}", recovered[0]);
    }
}