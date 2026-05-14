use num_complex::Complex64;

use super::{RnsCkksContext, RnsPolynomial};
use crate::rns::encoding::{decode_integer_coeffs, encode_integer_coeffs};

impl RnsCkksContext {
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
}