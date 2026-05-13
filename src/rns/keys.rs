use crate::poly::ModularPolynomial;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::basis::RnsContext;
use super::ciphertexts::RnsCiphertext;
use super::ops::encrypt_rns;
use super::polynomial::{mod_q_biguint_to_centered, RnsPolynomial};

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

#[derive(Clone, Debug)]
pub struct RnsKeySwitchingKey {
    pub ksk: Vec<RnsCiphertext>,
    pub decomp_bits: usize,
    pub level: usize,
}

impl RnsKeySwitchingKey {
    pub fn drop_last_level(&self) -> Self {
        assert!(self.level > 0, "cannot drop the last keyswitch level");
        Self {
            ksk: self.ksk.iter().map(RnsCiphertext::drop_last_level).collect(),
            decomp_bits: self.decomp_bits,
            level: self.level - 1,
        }
    }

    pub fn truncate_levels(&self, level_count: usize) -> Self {
        assert!(level_count > 0, "level_count must be positive");
        assert!(level_count <= self.level + 1, "level_count cannot exceed keyswitch levels");

        let mut truncated = self.clone();
        while truncated.level + 1 > level_count {
            truncated = truncated.drop_last_level();
        }
        truncated
    }
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

pub(crate) fn generate_keyswitch_key_rns(
    source: &RnsPolynomial,
    target_public_key: &RnsPublicKey,
    context: &RnsContext,
    decomp_bits: usize,
) -> RnsKeySwitchingKey {
    assert_eq!(source.levels(), context.levels(), "source levels must match keyswitch context");
    let modulus = context.total_modulus();
    let digits = num_decomp_digits(modulus.bits(), decomp_bits);
    let source_coeffs = source.reconstruct_coefficients(context);
    let mut base_power = BigUint::one();
    let base = BigUint::one() << decomp_bits;
    let mut ksk = Vec::with_capacity(digits);

    for _ in 0..digits {
        let coeffs: Vec<BigUint> = source_coeffs
            .iter()
            .map(|coeff| (coeff * &base_power) % &modulus)
            .collect();
        let plaintext = context.poly_from_biguint_coeffs(&coeffs);
        ksk.push(encrypt_rns(&plaintext, target_public_key, context, 0));
        base_power = (&base_power * &base) % &modulus;
    }

    RnsKeySwitchingKey {
        ksk,
        decomp_bits,
        level: context.levels() - 1,
    }
}

pub(crate) fn decompose_poly_rns_with_levels(
    poly: &RnsPolynomial,
    context: &RnsContext,
    decomp_bits: usize,
    levels: usize,
) -> Vec<RnsPolynomial> {
    let coeffs = poly.reconstruct_coefficients(context);
    let mut per_digit = vec![vec![0i64; context.poly_degree()]; levels];

    for (index, coeff) in coeffs.iter().enumerate() {
        let centered = mod_q_biguint_to_centered(coeff, &context.total_modulus());
        for (digit_index, digit) in balanced_digits_bigint(&centered, levels, decomp_bits)
            .into_iter()
            .enumerate()
        {
            per_digit[digit_index][index] = digit;
        }
    }

    per_digit
        .into_iter()
        .map(|digit_coeffs| {
            let digit_bigints: Vec<BigInt> = digit_coeffs.into_iter().map(BigInt::from).collect();
            context.poly_from_bigint_coeffs(&digit_bigints)
        })
        .collect()
}

pub(crate) fn balanced_digits_bigint(value: &BigInt, levels: usize, decomp_bits: usize) -> Vec<i64> {
    let base_i64 = 1i64 << decomp_bits;
    let half_i64 = base_i64 >> 1;
    let base = BigInt::from(base_i64);
    let mut current = value.clone();
    let mut digits = Vec::with_capacity(levels);

    for _ in 0..levels {
        let mut digit = (&current % &base)
            .to_i64()
            .expect("decomposition digit must fit in i64");
        if digit >= half_i64 {
            digit -= base_i64;
        } else if digit < -half_i64 {
            digit += base_i64;
        }
        digits.push(digit);
        current = (current - BigInt::from(digit)) / &base;
    }

    digits
}

pub(crate) fn num_decomp_digits(total_modulus_bits: u64, decomp_bits: usize) -> usize {
    (total_modulus_bits as usize + decomp_bits - 1) / decomp_bits
}