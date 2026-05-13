use crate::math::canonical_embedding::{canonical_embedding_backward, canonical_embedding_forward};
use num_bigint::{BigInt, BigUint};
use num_complex::Complex64;
use num_traits::ToPrimitive;
use std::f64::consts::PI;

use super::polynomial::mod_q_biguint_to_centered;

pub(crate) fn encode_integer_coeffs(poly_degree: usize, scale: f64, slots: &[Complex64]) -> Vec<BigInt> {
    let slots_n = poly_degree / 2;
    assert_eq!(slots.len(), slots_n, "slot count must be N/2");

    let mut evals = vec![Complex64::new(0.0, 0.0); poly_degree];
    let slot_exponents = slot_exponents(poly_degree);
    for (value, &exponent) in slots.iter().zip(slot_exponents.iter()) {
        let index = exponent_to_eval_index(exponent);
        let conjugate_index = exponent_to_eval_index((2 * poly_degree - exponent) % (2 * poly_degree));
        evals[index] = *value;
        evals[conjugate_index] = value.conj();
    }

    let xi = primitive_2n_root(poly_degree);
    canonical_embedding_backward(&evals)
        .into_iter()
        .enumerate()
        .map(|(index, coeff)| {
            let twisted = coeff * xi.powi(-(index as i32));
            BigInt::from(twisted.re.mul_add(scale, 0.0).round() as i128)
        })
        .collect()
}

pub(crate) fn decode_integer_coeffs(
    poly_degree: usize,
    scale: f64,
    total_modulus: &BigUint,
    coeffs_mod_q: &[BigUint],
) -> Vec<Complex64> {
    let inv_delta = 1.0 / scale;
    let xi = primitive_2n_root(poly_degree);
    let coeffs: Vec<Complex64> = coeffs_mod_q
        .iter()
        .map(|value| {
            let centered = mod_q_biguint_to_centered(value, total_modulus);
            Complex64::new(
                centered
                    .to_f64()
                    .expect("decoded coefficient must fit in f64")
                    * inv_delta,
                0.0,
            )
        })
        .collect();

    let b_coeffs: Vec<Complex64> = coeffs
        .into_iter()
        .enumerate()
        .map(|(index, coeff)| coeff * xi.powi(index as i32))
        .collect();
    let evals = canonical_embedding_forward(&b_coeffs);

    slot_exponents(poly_degree)
        .into_iter()
        .map(|exponent| evals[exponent_to_eval_index(exponent)])
        .collect()
}

fn slot_exponents(poly_degree: usize) -> Vec<usize> {
    let modulus = 2 * poly_degree;
    let mut exponents = Vec::with_capacity(poly_degree / 2);
    let mut current = 1usize;

    for _ in 0..(poly_degree / 2) {
        exponents.push(current);
        current = (current * 5) % modulus;
    }

    exponents
}

fn exponent_to_eval_index(exponent: usize) -> usize {
    assert!(exponent % 2 == 1, "evaluation exponents must be odd");
    (exponent - 1) / 2
}

fn primitive_2n_root(n: usize) -> Complex64 {
    Complex64::from_polar(1.0, PI / n as f64)
}