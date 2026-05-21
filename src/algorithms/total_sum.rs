//! Logarithmic total sum of the leading active CKKS slots.
//!
//! Packing contract:
//! - the values to sum occupy the first `active_slots` slots,
//! - if `active_slots` is not a power of two, the caller should zero the tail
//!   up to `active_slots.next_power_of_two()` before invoking this routine,
//! - the output follows the standard binary-fold pattern, so slot `0` contains
//!   the total sum and several leading slots repeat partial or total sums.

use std::collections::BTreeMap;

use crate::rns::{RnsCkksContext, RnsCiphertext, RnsPublicKey, RnsRotationKey, RnsSecretKey};

pub fn total_sum_rotation_steps(active_slots: usize) -> Vec<usize> {
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

pub fn generate_total_sum_rotation_keys(
    context: &RnsCkksContext,
    secret_key: &RnsSecretKey,
    public_key: &RnsPublicKey,
    level: usize,
    active_slots: usize,
) -> BTreeMap<usize, RnsRotationKey> {
    total_sum_rotation_steps(active_slots)
        .into_iter()
        .map(|step| {
            (
                step,
                context.generate_rotation_key_at_level(secret_key, public_key, step as isize, level),
            )
        })
        .collect()
}

pub fn total_sum(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    active_slots: usize,
    rotation_keys: &BTreeMap<usize, RnsRotationKey>,
) -> RnsCiphertext {
    let mut accumulator = ciphertext.clone();
    for step in total_sum_rotation_steps(active_slots) {
        let rotation_key = rotation_keys
            .get(&step)
            .unwrap_or_else(|| panic!("missing total-sum rotation key for step {step}"));
        let rotated = context.rotate(&accumulator, rotation_key);
        accumulator = accumulator.add(&rotated);
    }
    accumulator
}

#[cfg(test)]
mod tests {
    use super::{
        generate_total_sum_rotation_keys,
        total_sum,
        total_sum_rotation_steps,
    };
    use crate::rns::{RnsCkksContext, RnsCkksParams};

    #[test]
    fn total_sum_rotation_steps_follow_powers_of_two() {
        assert_eq!(total_sum_rotation_steps(1), Vec::<usize>::new());
        assert_eq!(total_sum_rotation_steps(4), vec![1, 2]);
        assert_eq!(total_sum_rotation_steps(5), vec![1, 2, 4]);
    }

    #[test]
    fn total_sum_matches_expected_scalar_on_realistic_params() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);
        let active_slots = 8usize;
        let input = [0.125_f64, -0.4, 0.03125, -0.125, 0.375, -0.25, 0.5, 0.0625];
        let expected: f64 = input.iter().sum();

        let mut padded = vec![0.0_f64; context.num_slots()];
        padded[..input.len()].copy_from_slice(&input);
        let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
        let rotation_keys = generate_total_sum_rotation_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            active_slots,
        );

        let total = total_sum(&context, &ciphertext, active_slots, &rotation_keys);
        let recovered = context.decode_real_at_scale(
            &context.decrypt(&total, &key_pair.secret_key),
            total.scale_bits,
        );

        assert!((recovered[0] - expected).abs() <= 1.5e-2, "expected {expected}, got {}", recovered[0]);
    }
}