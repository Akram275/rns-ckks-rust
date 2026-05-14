use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive};

use super::basis::RnsContext;
use super::ciphertexts::RnsCiphertext;
use super::keys::RnsPublicKey;
use super::ops::encrypt_rns;
use super::polynomial::{mod_q_biguint_to_centered, RnsPolynomial};

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