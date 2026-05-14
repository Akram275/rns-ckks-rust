use crate::poly::ModularPolynomial;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::basis::RnsContext;
use super::polynomial::RnsPolynomial;

#[derive(Clone, Debug)]
pub struct RnsSecretKey {
    pub poly: RnsPolynomial,
}

impl RnsSecretKey {
    pub fn new(poly: RnsPolynomial) -> Self {
        Self { poly }
    }

    pub fn sample_sparse_ternary(context: &RnsContext, hamming_weight: usize) -> Self {
        assert!(
            hamming_weight <= context.poly_degree(),
            "hamming_weight must be <= poly_degree"
        );
        let mut rng = rand::thread_rng();
        let mut coeffs = vec![0i8; context.poly_degree()];
        let mut placed = 0usize;

        while placed < hamming_weight {
            let index = rng.gen_range(0..context.poly_degree());
            if coeffs[index] == 0 {
                coeffs[index] = if rng.gen_bool(0.5) { 1 } else { -1 };
                placed += 1;
            }
        }

        let residues = context
            .primes()
            .iter()
            .map(|prime| {
                let lifted: Vec<u64> = coeffs
                    .iter()
                    .map(|&value| match value {
                        -1 => prime.modulus - 1,
                        0 => 0,
                        1 => 1,
                        _ => unreachable!("ternary secret coefficients must be in {{-1,0,1}}"),
                    })
                    .collect();
                ModularPolynomial::new(lifted, prime.modulus)
            })
            .collect();
        Self::new(RnsPolynomial::new(context, residues))
    }
}

#[derive(Clone, Debug)]
pub struct RnsPublicKey {
    pub p0: RnsPolynomial,
    pub p1: RnsPolynomial,
}

impl RnsPublicKey {
    pub fn new(p0: RnsPolynomial, p1: RnsPolynomial) -> Self {
        assert_eq!(p0.levels(), p1.levels(), "public key components must have the same number of levels");
        assert_eq!(p0.degree(), p1.degree(), "public key components must have the same degree");
        Self { p0, p1 }
    }

    pub fn truncate_levels(&self, level_count: usize) -> Self {
        Self {
            p0: self.p0.truncate_levels(level_count),
            p1: self.p1.truncate_levels(level_count),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RnsKeyPair {
    pub secret_key: RnsSecretKey,
    pub public_key: RnsPublicKey,
    pub error: RnsPolynomial,
}

pub(crate) fn sample_gaussian_i64(rng: &mut impl Rng) -> i64 {
    let normal = Normal::new(0.0_f64, 3.19_f64).expect("valid gaussian parameters");
    normal.sample(rng).round() as i64
}

pub(crate) fn sample_uniform_poly(context: &RnsContext, rng: &mut impl Rng) -> RnsPolynomial {
    let residues = context
        .primes()
        .iter()
        .map(|prime| {
            let coeffs: Vec<u64> = (0..context.poly_degree())
                .map(|_| rng.gen::<u64>() % prime.modulus)
                .collect();
            ModularPolynomial::new(coeffs, prime.modulus)
        })
        .collect();
    RnsPolynomial::new(context, residues)
}

pub(crate) fn sample_error_poly(context: &RnsContext, rng: &mut impl Rng) -> RnsPolynomial {
    let coeffs: Vec<i64> = (0..context.poly_degree())
        .map(|_| sample_gaussian_i64(rng))
        .collect();
    let residues = context
        .primes()
        .iter()
        .map(|prime| {
            let coeffs: Vec<u64> = coeffs
                .iter()
                .map(|&value| {
                    if value >= 0 {
                        value as u64
                    } else {
                        (prime.modulus as i128 + value as i128) as u64
                    }
                })
                .collect();
            ModularPolynomial::new(coeffs, prime.modulus)
        })
        .collect();
    RnsPolynomial::new(context, residues)
}

pub(crate) fn sample_ternary_poly(context: &RnsContext, rng: &mut impl Rng) -> RnsPolynomial {
    let coeffs: Vec<i8> = (0..context.poly_degree())
        .map(|_| match rng.gen_range(0..3) {
            0 => 0,
            1 => 1,
            _ => -1,
        })
        .collect();
    let residues = context
        .primes()
        .iter()
        .map(|prime| {
            let lifted: Vec<u64> = coeffs
                .iter()
                .map(|&value| match value {
                    -1 => prime.modulus - 1,
                    0 => 0,
                    1 => 1,
                    _ => unreachable!("ternary mask coefficients must be in {{-1,0,1}}"),
                })
                .collect();
            ModularPolynomial::new(lifted, prime.modulus)
        })
        .collect();
    RnsPolynomial::new(context, residues)
}

pub(crate) fn rns_keygen(context: &RnsContext, hamming_weight: usize) -> RnsKeyPair {
    let mut rng = rand::thread_rng();
    let secret_key = RnsSecretKey::sample_sparse_ternary(context, hamming_weight);
    let p1 = sample_uniform_poly(context, &mut rng);
    let error = sample_error_poly(context, &mut rng);
    let p1s = p1.multiply_ntt(&secret_key.poly, context);
    let p0 = error.sub(&p1s);
    let public_key = RnsPublicKey::new(p0, p1);

    RnsKeyPair {
        secret_key,
        public_key,
        error,
    }
}