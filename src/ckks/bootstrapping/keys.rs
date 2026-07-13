use std::collections::BTreeMap;

use crate::algorithms::paterson_stockmeyer::generate_paterson_stockmeyer_relinearization_keys;
use crate::ckks::bootstrapping::linear_transform::generate_diagonal_transform_rotation_keys;
use crate::rns::{RnsCkksContext, RnsKeyPair, RnsKeySwitchingKey, RnsRotationKey};

#[derive(Clone, Debug, Default)]
pub struct BootstrapKeySet {
    pub coeff_to_slot_rotation_keys: BTreeMap<usize, RnsRotationKey>,
    pub slot_to_coeff_rotation_keys: BTreeMap<usize, RnsRotationKey>,
    pub eval_mod_relinearization_keys: BTreeMap<usize, RnsKeySwitchingKey>,
    pub conjugation_key: Option<RnsRotationKey>,
}

impl BootstrapKeySet {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn for_eval_mod(
        context: &RnsCkksContext,
        key_pair: &RnsKeyPair,
        ciphertext_level: usize,
        polynomial_term_count: usize,
    ) -> Self {
        Self {
            eval_mod_relinearization_keys: generate_paterson_stockmeyer_relinearization_keys(
                context,
                &key_pair.secret_key,
                &key_pair.public_key,
                ciphertext_level,
                polynomial_term_count,
            ),
            ..Self::default()
        }
    }

    pub fn for_coeff_to_slot(
        context: &RnsCkksContext,
        key_pair: &RnsKeyPair,
        ciphertext_level: usize,
        active_slots: usize,
    ) -> Self {
        let rotation_keys = generate_diagonal_transform_rotation_keys(
            context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext_level,
            active_slots,
        );
        Self {
            coeff_to_slot_rotation_keys: rotation_keys.by_step,
            ..Self::default()
        }
    }

    pub fn for_slot_to_coeff(
        context: &RnsCkksContext,
        key_pair: &RnsKeyPair,
        ciphertext_level: usize,
        active_slots: usize,
    ) -> Self {
        let rotation_keys = generate_diagonal_transform_rotation_keys(
            context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext_level,
            active_slots,
        );
        Self {
            slot_to_coeff_rotation_keys: rotation_keys.by_step,
            ..Self::default()
        }
    }

    pub fn for_dense_coeff_to_slot(
        context: &RnsCkksContext,
        key_pair: &RnsKeyPair,
        ciphertext_level: usize,
    ) -> Self {
        let rotation_keys = generate_diagonal_transform_rotation_keys(
            context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext_level,
            context.num_slots(),
        );
        Self {
            coeff_to_slot_rotation_keys: rotation_keys.by_step,
            conjugation_key: Some(context.generate_conjugation_key_at_level(
                &key_pair.secret_key,
                &key_pair.public_key,
                ciphertext_level,
            )),
            ..Self::default()
        }
    }

    pub fn for_dense_slot_to_coeff(
        context: &RnsCkksContext,
        key_pair: &RnsKeyPair,
        ciphertext_level: usize,
    ) -> Self {
        let rotation_keys = generate_diagonal_transform_rotation_keys(
            context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext_level,
            context.num_slots(),
        );
        Self {
            slot_to_coeff_rotation_keys: rotation_keys.by_step,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BootstrapKeySet;
    use crate::algorithms::paterson_stockmeyer::paterson_stockmeyer_required_levels;
    use crate::rns::{RnsCkksContext, RnsCkksParams, RnsPrime};

    fn sample_key_context() -> RnsCkksContext {
        RnsCkksContext::new(RnsCkksParams {
            poly_degree: 16,
            primes: vec![
                RnsPrime::new(998_244_353, 3),
                RnsPrime::new(1_004_535_809, 3),
                RnsPrime::new(469_762_049, 3),
                RnsPrime::new(754_974_721, 11),
            ],
            aux_primes: vec![RnsPrime::new(167_772_161, 3)],
            scale_bits: 28,
        })
        .expect("sample key CKKS context must build")
    }

    #[test]
    fn eval_mod_key_scaffold_matches_paterson_stockmeyer_levels() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let keys = BootstrapKeySet::for_eval_mod(&context, &key_pair, 7, 4);
        let expected_levels = paterson_stockmeyer_required_levels(7, 4);

        assert_eq!(keys.eval_mod_relinearization_keys.len(), expected_levels.len());
        for level in expected_levels {
            assert!(keys.eval_mod_relinearization_keys.contains_key(&level));
        }
        assert!(keys.coeff_to_slot_rotation_keys.is_empty());
        assert!(keys.slot_to_coeff_rotation_keys.is_empty());
        assert!(keys.conjugation_key.is_none());
    }

    #[test]
    fn dense_coeff_to_slot_key_scaffold_includes_rotations_and_conjugation() {
        let context = sample_key_context();
        let key_pair = context.keygen(16);

        let keys = BootstrapKeySet::for_dense_coeff_to_slot(&context, &key_pair, 3);

        assert_eq!(keys.coeff_to_slot_rotation_keys.len(), context.num_slots() - 1);
        assert!(keys.conjugation_key.is_some());
        assert!(keys.slot_to_coeff_rotation_keys.is_empty());
        assert!(keys.eval_mod_relinearization_keys.is_empty());
    }

    #[test]
    fn coeff_to_slot_key_scaffold_includes_sparse_rotations_only() {
        let context = sample_key_context();
        let key_pair = context.keygen(16);

        let keys = BootstrapKeySet::for_coeff_to_slot(&context, &key_pair, 3, 4);

        assert_eq!(keys.coeff_to_slot_rotation_keys.len(), 3);
        assert!(keys.slot_to_coeff_rotation_keys.is_empty());
        assert!(keys.eval_mod_relinearization_keys.is_empty());
        assert!(keys.conjugation_key.is_none());
    }

    #[test]
    fn dense_slot_to_coeff_key_scaffold_includes_rotations() {
        let context = sample_key_context();
        let key_pair = context.keygen(16);

        let keys = BootstrapKeySet::for_dense_slot_to_coeff(&context, &key_pair, 3);

        assert_eq!(keys.slot_to_coeff_rotation_keys.len(), context.num_slots() - 1);
        assert!(keys.coeff_to_slot_rotation_keys.is_empty());
        assert!(keys.conjugation_key.is_none());
    }

    #[test]
    fn slot_to_coeff_key_scaffold_includes_sparse_rotations_only() {
        let context = sample_key_context();
        let key_pair = context.keygen(16);

        let keys = BootstrapKeySet::for_slot_to_coeff(&context, &key_pair, 3, 4);

        assert_eq!(keys.slot_to_coeff_rotation_keys.len(), 3);
        assert!(keys.coeff_to_slot_rotation_keys.is_empty());
        assert!(keys.eval_mod_relinearization_keys.is_empty());
        assert!(keys.conjugation_key.is_none());
    }
}