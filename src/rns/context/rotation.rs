use super::{RnsCiphertext, RnsCkksContext, RnsPolynomial, RnsPublicKey, RnsQuadraticCiphertext, RnsSecretKey};
use crate::rns::{galois_element_for_rotation, RnsRotationKey};
use crate::rns::keyswitching::{keyswitch_rns, keyswitch_rns_hybrid};

impl RnsCkksContext {
    pub fn apply_galois_automorphism(&self, poly: &RnsPolynomial, galois_element: usize) -> RnsPolynomial {
        let modulus = 2 * self.params().poly_degree;
        assert!(galois_element < modulus, "galois element must be reduced modulo 2N");
        poly.apply_galois_automorphism(galois_element)
    }

    pub fn rotate_plain_raw(&self, poly: &RnsPolynomial, steps: isize) -> RnsPolynomial {
        poly.rotate_raw(self.params().poly_degree, steps)
    }

    pub fn rotate_secret_key_raw(&self, secret_key: &RnsSecretKey, steps: isize) -> RnsSecretKey {
        secret_key.rotate_raw(self.params().poly_degree, steps)
    }

    pub fn rotate_raw(&self, ciphertext: &RnsCiphertext, steps: isize) -> RnsCiphertext {
        ciphertext.rotate_raw(self.params().poly_degree, steps)
    }

    pub fn generate_rotation_key(
        &self,
        secret_key: &RnsSecretKey,
        public_key: &RnsPublicKey,
        steps: isize,
    ) -> RnsRotationKey {
        let galois_element = self.galois_element_for_rotation(steps);
        let rotated_secret_key = secret_key.apply_galois_automorphism(galois_element);
        let keyswitch_key = if self.has_aux_basis() {
            self.generate_eval_keyswitch_key(&rotated_secret_key.poly, secret_key)
        } else {
            self.generate_keyswitch_key(&rotated_secret_key.poly, public_key)
        };
        RnsRotationKey {
            galois_element,
            keyswitch_key,
        }
    }

    pub fn generate_rotation_key_at_level(
        &self,
        secret_key: &RnsSecretKey,
        public_key: &RnsPublicKey,
        steps: isize,
        level: usize,
    ) -> RnsRotationKey {
        let galois_element = self.galois_element_for_rotation(steps);
        let rotated_secret_key = secret_key.apply_galois_automorphism(galois_element);
        let keyswitch_key = if self.has_aux_basis() {
            self.generate_eval_keyswitch_key_at_level(&rotated_secret_key.poly, secret_key, level)
        } else {
            self.generate_keyswitch_key_at_level(&rotated_secret_key.poly, public_key, level)
        };
        RnsRotationKey {
            galois_element,
            keyswitch_key,
        }
    }

    pub fn rotate(&self, ciphertext: &RnsCiphertext, rotation_key: &RnsRotationKey) -> RnsCiphertext {
        let raw_rotated = ciphertext.apply_galois_automorphism(rotation_key.galois_element);
        assert_eq!(rotation_key.keyswitch_key.level, ciphertext.level, "rotation key level must match the ciphertext level");
        let switched = if rotation_key.keyswitch_key.aux_level_count == 0 {
            let context = self
                .level_context(ciphertext.level)
                .expect("rotation level context must exist");
            keyswitch_rns(&raw_rotated.c1, &rotation_key.keyswitch_key, &context)
        } else {
            let q_context = self
                .level_context(ciphertext.level)
                .expect("rotation level context must exist");
            let eval_key_context = self
                .eval_key_level_context(ciphertext.level)
                .expect("hybrid rotation requires an eval-key level context");
            keyswitch_rns_hybrid(&raw_rotated.c1, &rotation_key.keyswitch_key, &q_context, &eval_key_context)
        };

        RnsCiphertext::new(
            raw_rotated.c0.add(&switched.c0),
            switched.c1,
            raw_rotated.scale_bits,
        )
    }

    pub fn rotate_quadratic_raw(
        &self,
        ciphertext: &RnsQuadraticCiphertext,
        steps: isize,
    ) -> RnsQuadraticCiphertext {
        ciphertext.rotate_raw(self.params().poly_degree, steps)
    }

    pub fn galois_element_for_rotation(&self, steps: isize) -> usize {
        galois_element_for_rotation(self.params().poly_degree, steps)
    }
}