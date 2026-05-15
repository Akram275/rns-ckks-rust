use num_complex::Complex64;

use super::{RnsCkksContext, RnsCiphertext, RnsKeySwitchingKey, RnsPolynomial, RnsPublicKey, RnsQuadraticCiphertext, RnsSecretKey};
use crate::rns::keyswitching::{keyswitch_rns, keyswitch_rns_hybrid};
use crate::rns::ops::{decrypt_quadratic_rns, decrypt_rns, encrypt_rns, multiply_ciphertexts_rns, multiply_plain_rns, rescale_ciphertext_rns, rescale_quadratic_ciphertext_rns};

impl RnsCkksContext {
    pub fn encrypt(&self, plaintext: &RnsPolynomial, public_key: &RnsPublicKey) -> RnsCiphertext {
        encrypt_rns(plaintext, public_key, &self.rns_context, self.params.scale_bits)
    }

    pub fn decrypt(&self, ciphertext: &RnsCiphertext, secret_key: &RnsSecretKey) -> RnsPolynomial {
        let context = self
            .level_context(ciphertext.level)
            .expect("ciphertext level context must exist for decryption");
        let secret_key = RnsSecretKey::new(secret_key.poly.truncate_levels(ciphertext.level + 1));
        decrypt_rns(ciphertext, &secret_key, &context)
    }

    pub fn mult_plain(&self, ciphertext: &RnsCiphertext, plain: &RnsPolynomial) -> RnsCiphertext {
        let context = self
            .level_context(ciphertext.level)
            .expect("ciphertext level context must exist for plaintext multiplication");
        let multiplied = multiply_plain_rns(ciphertext, plain, &context);
        let next_scale_bits = ciphertext.scale_bits + ciphertext.scale_bits;
        RnsCiphertext {
            scale_bits: next_scale_bits,
            ..multiplied
        }
    }

    pub fn rescale(&self, ciphertext: &RnsCiphertext) -> RnsCiphertext {
        let context = self
            .level_context(ciphertext.level)
            .expect("ciphertext level context must exist for rescaling");
        rescale_ciphertext_rns(ciphertext, &context)
    }

    pub fn mult_plain_slots(&self, ciphertext: &RnsCiphertext, plain_slots: &[Complex64]) -> RnsCiphertext {
        let plain = self.encode_at_level_and_scale(plain_slots, ciphertext.level, ciphertext.scale_bits);
        let multiplied = self.mult_plain(ciphertext, &plain);
        self.rescale(&multiplied)
    }

    pub fn mult_plain_slots_real(&self, ciphertext: &RnsCiphertext, plain_slots: &[f64]) -> RnsCiphertext {
        let complex_slots: Vec<Complex64> = plain_slots.iter().map(|&value| Complex64::new(value, 0.0)).collect();
        self.mult_plain_slots(ciphertext, &complex_slots)
    }

    pub fn mult_plain_slots_preserve_scale(&self, ciphertext: &RnsCiphertext, plain_slots: &[Complex64]) -> RnsCiphertext {
        let plain = self.encode_at_level_and_scale(plain_slots, ciphertext.level, 0);
        let context = self
            .level_context(ciphertext.level)
            .expect("ciphertext level context must exist for plaintext multiplication");
        multiply_plain_rns(ciphertext, &plain, &context)
    }

    pub fn mult_plain_slots_real_preserve_scale(&self, ciphertext: &RnsCiphertext, plain_slots: &[f64]) -> RnsCiphertext {
        let complex_slots: Vec<Complex64> = plain_slots.iter().map(|&value| Complex64::new(value, 0.0)).collect();
        self.mult_plain_slots_preserve_scale(ciphertext, &complex_slots)
    }

    pub fn mult(&self, lhs: &RnsCiphertext, rhs: &RnsCiphertext) -> RnsQuadraticCiphertext {
        assert_eq!(lhs.level, rhs.level, "ciphertexts must be at the same level to multiply");
        assert_eq!(lhs.scale_bits, rhs.scale_bits, "ciphertexts must share the same scale to multiply");
        let context = self
            .level_context(lhs.level)
            .expect("ciphertext level context must exist for ciphertext multiplication");
        multiply_ciphertexts_rns(lhs, rhs, &context)
    }

    pub fn decrypt_quadratic(
        &self,
        ciphertext: &RnsQuadraticCiphertext,
        secret_key: &RnsSecretKey,
    ) -> RnsPolynomial {
        let context = self
            .level_context(ciphertext.level)
            .expect("quadratic ciphertext level context must exist for decryption");
        let truncated_secret = secret_key.poly.truncate_levels(ciphertext.level + 1);
        let secret_key = RnsSecretKey::new(truncated_secret);
        decrypt_quadratic_rns(ciphertext, &secret_key, &context)
    }

    pub fn rescale_quadratic(&self, ciphertext: &RnsQuadraticCiphertext) -> RnsQuadraticCiphertext {
        let context = self
            .level_context(ciphertext.level)
            .expect("quadratic ciphertext level context must exist for rescaling");
        rescale_quadratic_ciphertext_rns(ciphertext, &context)
    }

    pub fn mult_slots(&self, lhs: &RnsCiphertext, rhs: &RnsCiphertext) -> RnsQuadraticCiphertext {
        let multiplied = self.mult(lhs, rhs);
        self.rescale_quadratic(&multiplied)
    }

    pub fn keyswitch(&self, poly: &RnsPolynomial, ksk: &RnsKeySwitchingKey) -> RnsCiphertext {
        let q_context = self
            .level_context(poly.levels() - 1)
            .expect("keyswitch level context must exist");
        if ksk.aux_level_count == 0 {
            keyswitch_rns(poly, ksk, &q_context)
        } else {
            let eval_key_context = self
                .eval_key_level_context(poly.levels() - 1)
                .expect("hybrid keyswitching requires an eval-key level context");
            keyswitch_rns_hybrid(poly, ksk, &q_context, &eval_key_context)
        }
    }

    pub fn relinearize(
        &self,
        ciphertext: &RnsQuadraticCiphertext,
        relin_key: &RnsKeySwitchingKey,
    ) -> RnsCiphertext {
        let switched = self.keyswitch(&ciphertext.c2, relin_key);
        RnsCiphertext::new(
            ciphertext.c0.add(&switched.c0),
            ciphertext.c1.add(&switched.c1),
            ciphertext.scale_bits,
        )
    }

    pub fn mult_relinearized(
        &self,
        lhs: &RnsCiphertext,
        rhs: &RnsCiphertext,
        relin_key: &RnsKeySwitchingKey,
    ) -> RnsCiphertext {
        let multiplied = self.mult(lhs, rhs);
        let relinearized = self.relinearize(&multiplied, relin_key);
        self.rescale(&relinearized)
    }
}