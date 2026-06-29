use std::collections::BTreeMap;

use crate::algorithms::paterson_stockmeyer::generate_paterson_stockmeyer_relinearization_keys;
use crate::rns::{RnsCkksContext, RnsKeyPair, RnsKeySwitchingKey, RnsRotationKey};

#[derive(Clone, Debug, Default)]
pub struct BootstrapKeySet {
    pub coeff_to_slot_rotation_keys: BTreeMap<usize, RnsRotationKey>,
    pub slot_to_coeff_rotation_keys: BTreeMap<usize, RnsRotationKey>,
    pub eval_mod_relinearization_keys: BTreeMap<usize, RnsKeySwitchingKey>,
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
}

#[cfg(test)]
mod tests {
    use super::BootstrapKeySet;
    use crate::algorithms::paterson_stockmeyer::paterson_stockmeyer_required_levels;
    use crate::rns::{RnsCkksContext, RnsCkksParams};

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
    }
}