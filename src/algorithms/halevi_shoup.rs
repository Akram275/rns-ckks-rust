//! Halevi-Shoup plaintext-ciphertext dot product.
//!
//! Packing contract:
//! - the encrypted vector occupies the first `active_slots` slots,
//! - the plaintext vector is provided as a standard real slice and is padded
//!   with zeros to the full CKKS slot count,
//! - the output is a rotated-and-summed ciphertext whose slot `0` contains the
//!   dot product, while the remaining active slots contain the same scalar under
//!   the usual binary fold pattern.

use std::collections::BTreeMap;

use crate::rns::{RnsCkksContext, RnsCiphertext, RnsPublicKey, RnsRotationKey, RnsSecretKey};

pub fn halevi_shoup_dot_product_rotation_steps(active_slots: usize) -> Vec<usize> {
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

pub fn generate_halevi_shoup_dot_product_rotation_keys(
    context: &RnsCkksContext,
    secret_key: &RnsSecretKey,
    public_key: &RnsPublicKey,
    level: usize,
    active_slots: usize,
) -> BTreeMap<usize, RnsRotationKey> {
    halevi_shoup_dot_product_rotation_steps(active_slots)
        .into_iter()
        .map(|step| {
            (
                step,
                context.generate_rotation_key_at_level(secret_key, public_key, step as isize, level),
            )
        })
        .collect()
}

pub fn halevi_shoup_dot_product(
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
    for step in halevi_shoup_dot_product_rotation_steps(plaintext_slots.len()) {
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
        generate_halevi_shoup_dot_product_rotation_keys,
        halevi_shoup_dot_product,
        halevi_shoup_dot_product_rotation_steps,
    };
    use crate::rns::{RnsCkksContext, RnsCkksParams};

    #[test]
    fn halevi_shoup_dot_product_rotation_steps_follow_powers_of_two() {
        assert_eq!(halevi_shoup_dot_product_rotation_steps(1), Vec::<usize>::new());
        assert_eq!(halevi_shoup_dot_product_rotation_steps(4), vec![1, 2]);
        assert_eq!(halevi_shoup_dot_product_rotation_steps(5), vec![1, 2, 4]);
    }

    #[test]
    fn halevi_shoup_dot_product_matches_expected_scalar_on_realistic_params() {
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
        let rotation_keys = generate_halevi_shoup_dot_product_rotation_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level - 1,
            plaintext_slots.len(),
        );

        let dot_product = halevi_shoup_dot_product(
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