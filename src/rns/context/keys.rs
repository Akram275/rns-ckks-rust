use super::{RnsCkksContext, RnsPublicKey, RnsSecretKey, RnsKeyPair, RnsKeySwitchingKey, RnsPolynomial, RNS_DECOMP_BITS};
use crate::rns::keys::rns_keygen;
use crate::rns::keyswitching::generate_keyswitch_key_rns;

impl RnsCkksContext {
    pub fn keygen(&self, hamming_weight: usize) -> RnsKeyPair {
        rns_keygen(&self.rns_context, hamming_weight)
    }

    pub fn generate_keyswitch_key(
        &self,
        source: &RnsPolynomial,
        target_public_key: &RnsPublicKey,
    ) -> RnsKeySwitchingKey {
        generate_keyswitch_key_rns(source, target_public_key, self.rns_context(), RNS_DECOMP_BITS)
    }

    pub fn generate_keyswitch_key_at_level(
        &self,
        source: &RnsPolynomial,
        target_public_key: &RnsPublicKey,
        level: usize,
    ) -> RnsKeySwitchingKey {
        let level_context = self
            .level_context(level)
            .expect("keyswitch level context must exist");
        let source = source.truncate_levels(level + 1);
        let target_public_key = target_public_key.truncate_levels(level + 1);

        generate_keyswitch_key_rns(&source, &target_public_key, &level_context, RNS_DECOMP_BITS)
    }

    pub fn generate_relinearization_key(
        &self,
        secret_key: &RnsSecretKey,
        public_key: &RnsPublicKey,
    ) -> RnsKeySwitchingKey {
        let source = secret_key.poly.multiply_ntt(&secret_key.poly, self.rns_context());
        self.generate_keyswitch_key(&source, public_key)
    }

    pub fn generate_relinearization_key_at_level(
        &self,
        secret_key: &RnsSecretKey,
        public_key: &RnsPublicKey,
        level: usize,
    ) -> RnsKeySwitchingKey {
        let level_context = self
            .level_context(level)
            .expect("relinearization level context must exist");
        let truncated_secret = secret_key.poly.truncate_levels(level + 1);
        let source = truncated_secret.multiply_ntt(&truncated_secret, &level_context);
        let target_public_key = public_key.truncate_levels(level + 1);

        generate_keyswitch_key_rns(&source, &target_public_key, &level_context, RNS_DECOMP_BITS)
    }
}