use num_complex::Complex64;

use super::basis::{default_rns_primes, RnsContext, RnsPrime};
use super::ciphertexts::{RnsCiphertext, RnsQuadraticCiphertext};
use super::encoding::{decode_integer_coeffs, encode_integer_coeffs};
use super::keys::{rns_keygen, RnsKeyPair, RnsPublicKey, RnsSecretKey};
use super::keyswitching::{generate_keyswitch_key_rns, RnsKeySwitchingKey};
use super::ops::{decrypt_quadratic_rns, decrypt_rns, encrypt_rns, keyswitch_rns, multiply_ciphertexts_rns, multiply_plain_rns, rescale_ciphertext_rns, rescale_quadratic_ciphertext_rns};
use super::polynomial::RnsPolynomial;
use super::RNS_DECOMP_BITS;

#[derive(Clone, Debug)]
pub struct RnsCkksParams {
    pub poly_degree: usize,
    pub primes: Vec<RnsPrime>,
    pub aux_primes: Vec<RnsPrime>,
    pub scale_bits: usize,
}

impl RnsCkksParams {
    pub fn toy() -> Self {
        Self {
            poly_degree: 8,
            primes: default_rns_primes(),
            aux_primes: Vec::new(),
            scale_bits: 30,
        }
    }

    pub fn small() -> Self {
        Self {
            poly_degree: 1024,
            primes: default_rns_primes(),
            aux_primes: Vec::new(),
            scale_bits: 40,
        }
    }

    pub fn realistic_8_level() -> Self {
        Self {
            poly_degree: 1 << 15,
            primes: vec![
                RnsPrime::new(1_125_899_908_022_273, 3),
                RnsPrime::new(1_125_899_908_612_097, 3),
                RnsPrime::new(1_125_899_909_398_529, 3),
                RnsPrime::new(1_125_899_910_316_033, 5),
                RnsPrime::new(1_125_899_911_168_001, 3),
                RnsPrime::new(1_125_899_911_364_609, 3),
                RnsPrime::new(1_125_899_913_527_297, 3),
                RnsPrime::new(1_125_899_914_510_337, 3),
            ],
            aux_primes: vec![
                RnsPrime::new(562_949_954_142_209, 3),
                RnsPrime::new(562_949_955_125_249, 3),
            ],
            scale_bits: 45,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.scale_bits == 0 {
            return Err("scale_bits must be greater than zero".to_string());
        }
        if self.scale_bits >= 60 {
            return Err("scale_bits must stay below 60 bits per rescale step in this RNS scaffold".to_string());
        }
        for aux_prime in &self.aux_primes {
            if self.primes.iter().any(|prime| prime.modulus == aux_prime.modulus) {
                return Err(format!(
                    "auxiliary eval-key modulus {} must be distinct from the ciphertext modulus chain",
                    aux_prime.modulus
                ));
            }
        }

        RnsContext::new(self.poly_degree, self.primes.clone()).map(|_| ())?;
        if !self.aux_primes.is_empty() {
            RnsContext::new(self.poly_degree, self.aux_primes.clone()).map(|_| ())?;
            let mut extended_primes = self.primes.clone();
            extended_primes.extend(self.aux_primes.clone());
            RnsContext::new(self.poly_degree, extended_primes).map(|_| ())?;
        }

        Ok(())
    }

    pub fn scale(&self) -> f64 {
        2f64.powi(self.scale_bits as i32)
    }
}

#[derive(Clone, Debug)]
pub struct RnsCkksContext {
    params: RnsCkksParams,
    rns_context: RnsContext,
    aux_context: Option<RnsContext>,
    eval_key_context: Option<RnsContext>,
}

impl RnsCkksContext {
    pub fn new(params: RnsCkksParams) -> Result<Self, String> {
        params.validate()?;
        let rns_context = RnsContext::new(params.poly_degree, params.primes.clone())?;
        let aux_context = if params.aux_primes.is_empty() {
            None
        } else {
            Some(RnsContext::new(params.poly_degree, params.aux_primes.clone())?)
        };
        let eval_key_context = if params.aux_primes.is_empty() {
            None
        } else {
            let mut extended_primes = params.primes.clone();
            extended_primes.extend(params.aux_primes.clone());
            Some(RnsContext::new(params.poly_degree, extended_primes)?)
        };
        Ok(Self {
            params,
            rns_context,
            aux_context,
            eval_key_context,
        })
    }

    pub fn params(&self) -> &RnsCkksParams {
        &self.params
    }

    pub fn rns_context(&self) -> &RnsContext {
        &self.rns_context
    }

    pub fn aux_context(&self) -> Option<&RnsContext> {
        self.aux_context.as_ref()
    }

    pub fn eval_key_context(&self) -> Option<&RnsContext> {
        self.eval_key_context.as_ref()
    }

    pub fn has_aux_basis(&self) -> bool {
        self.aux_context.is_some()
    }

    pub fn num_slots(&self) -> usize {
        self.params.poly_degree / 2
    }

    pub fn levels(&self) -> usize {
        self.rns_context.levels()
    }

    pub fn total_modulus_bits(&self) -> u64 {
        self.rns_context.total_modulus_bits()
    }

    pub fn aux_modulus_bits(&self) -> Option<u64> {
        self.aux_context.as_ref().map(RnsContext::total_modulus_bits)
    }

    pub fn eval_key_modulus_bits(&self) -> Option<u64> {
        self.eval_key_context.as_ref().map(RnsContext::total_modulus_bits)
    }

    pub fn level_context(&self, level: usize) -> Result<RnsContext, String> {
        self.rns_context.at_level(level + 1)
    }

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
        self.generate_keyswitch_key(source, target_public_key)
            .truncate_levels(level + 1)
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
        self.generate_relinearization_key(secret_key, public_key)
            .truncate_levels(level + 1)
    }

    pub fn encode(&self, slots: &[Complex64]) -> RnsPolynomial {
        self.encode_at_level_and_scale(slots, self.levels() - 1, self.params.scale_bits)
    }

    pub fn encode_real(&self, slots: &[f64]) -> RnsPolynomial {
        let complex_slots: Vec<Complex64> = slots.iter().map(|&value| Complex64::new(value, 0.0)).collect();
        self.encode(&complex_slots)
    }

    pub fn decode(&self, poly: &RnsPolynomial) -> Vec<Complex64> {
        self.decode_at_scale(poly, self.params.scale_bits)
    }

    pub fn encode_at_level_and_scale(
        &self,
        slots: &[Complex64],
        level: usize,
        scale_bits: usize,
    ) -> RnsPolynomial {
        let level_context = self
            .level_context(level)
            .expect("requested level context must exist for encoding");
        let coeffs = encode_integer_coeffs(self.params.poly_degree, 2f64.powi(scale_bits as i32), slots);
        level_context.poly_from_bigint_coeffs(&coeffs)
    }

    pub fn decode_at_scale(&self, poly: &RnsPolynomial, scale_bits: usize) -> Vec<Complex64> {
        let context = self
            .level_context(poly.levels() - 1)
            .expect("polynomial level context must exist for decoding");
        let coeffs_biguint = poly.reconstruct_coefficients(&context);
        decode_integer_coeffs(
            self.params.poly_degree,
            2f64.powi(scale_bits as i32),
            &context.total_modulus(),
            &coeffs_biguint,
        )
    }

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
        let context = self
            .level_context(poly.levels() - 1)
            .expect("keyswitch level context must exist");
        keyswitch_rns(poly, ksk, &context)
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