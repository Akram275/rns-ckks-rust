use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;

use super::basis::{mod_inv_u64, RnsContext};
use super::ciphertexts::RnsCiphertext;
use super::keys::{sample_error_poly, sample_uniform_poly, RnsPublicKey, RnsSecretKey};
use super::ops::encrypt_rns;
use super::polynomial::{mod_q_biguint_to_centered, RnsPolynomial};
use crate::poly::ModularPolynomial;

#[derive(Clone, Debug)]
pub struct RnsKeySwitchingKey {
    pub ksk: Vec<RnsCiphertext>,
    pub decomp_bits: usize,
    pub level: usize,
    pub aux_level_count: usize,
}

impl RnsKeySwitchingKey {
    pub fn drop_last_level(&self) -> Self {
        assert!(self.level > 0, "cannot drop the last keyswitch level");
        assert_eq!(self.aux_level_count, 0, "dropping levels is only supported for Q-only keyswitch keys");
        Self {
            ksk: self.ksk.iter().map(RnsCiphertext::drop_last_level).collect(),
            decomp_bits: self.decomp_bits,
            level: self.level - 1,
            aux_level_count: 0,
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
    let mut base_powers = Vec::with_capacity(digits);
    for _ in 0..digits {
        base_powers.push(base_power.clone());
        base_power = (&base_power * &base) % &modulus;
    }
    let ksk = base_powers
        .into_par_iter()
        .map(|base_power| {
            let coeffs: Vec<BigUint> = source_coeffs
                .iter()
                .map(|coeff| (coeff * &base_power) % &modulus)
                .collect();
            let plaintext = context.poly_from_biguint_coeffs(&coeffs);
            encrypt_rns(&plaintext, target_public_key, context, 0)
        })
        .collect();

    RnsKeySwitchingKey {
        ksk,
        decomp_bits,
        level: context.levels() - 1,
        aux_level_count: 0,
    }
}

pub(crate) fn generate_keyswitch_key_rns_hybrid(
    source: &RnsPolynomial,
    target_secret_key: &RnsSecretKey,
    q_context: &RnsContext,
    eval_key_context: &RnsContext,
    decomp_bits: usize,
) -> RnsKeySwitchingKey {
    assert_eq!(source.levels(), q_context.levels(), "source levels must match the active Q context");
    assert!(eval_key_context.levels() > q_context.levels(), "hybrid keyswitching requires an extended evaluation basis");

    let aux_level_count = eval_key_context.levels() - q_context.levels();
    let aux_modulus = eval_key_context.total_modulus() / q_context.total_modulus();
    let digits = num_decomp_digits(q_context.total_modulus_bits(), decomp_bits);
    let source_coeffs = source.reconstruct_coefficients(q_context);
    let q_modulus = q_context.total_modulus();
    let aux_modulus_bigint = BigInt::from_biguint(Sign::Plus, aux_modulus);
    let target_secret = lift_poly_to_context(&target_secret_key.poly.truncate_levels(q_context.levels()), q_context, eval_key_context);
    let mut base_power = BigInt::one();
    let base = BigInt::one() << decomp_bits;
    let mut base_powers = Vec::with_capacity(digits);
    for _ in 0..digits {
        base_powers.push(base_power.clone());
        base_power *= &base;
    }
    let ksk = base_powers
        .into_par_iter()
        .map(|base_power| {
            let coeffs: Vec<BigInt> = source_coeffs
                .iter()
                .map(|coeff| mod_q_biguint_to_centered(coeff, &q_modulus) * &base_power * &aux_modulus_bigint)
                .collect();
            let plaintext = eval_key_context.poly_from_bigint_coeffs(&coeffs);
            let mut rng = rand::thread_rng();
            let a1 = sample_uniform_poly(eval_key_context, &mut rng);
            let error = sample_error_poly(eval_key_context, &mut rng);
            let a1s = a1.multiply_ntt(&target_secret, eval_key_context);
            let a0 = plaintext.add(&error).sub(&a1s);
            RnsCiphertext::new(a0, a1, 0)
        })
        .collect();

    RnsKeySwitchingKey {
        ksk,
        decomp_bits,
        level: q_context.levels() - 1,
        aux_level_count,
    }
}

pub(crate) fn keyswitch_rns(
    poly: &RnsPolynomial,
    ksk: &RnsKeySwitchingKey,
    context: &RnsContext,
) -> RnsCiphertext {
    assert_eq!(ksk.level, context.levels() - 1, "keyswitch key level must match the active modulus chain");
    let active_digits = num_decomp_digits(context.total_modulus_bits(), ksk.decomp_bits);
    assert!(ksk.ksk.len() >= active_digits, "keyswitch key must cover the active modulus chain");
    let digits = decompose_poly_rns_with_levels(poly, context, ksk.decomp_bits, ksk.ksk.len());
    let contributions: Vec<(RnsPolynomial, RnsPolynomial)> = digits
        .par_iter()
        .zip(ksk.ksk.par_iter())
        .map(|(digit_poly, key_ct)| {
            (
                digit_poly.multiply_ntt(&key_ct.c0, context),
                digit_poly.multiply_ntt(&key_ct.c1, context),
            )
        })
        .collect();
    let mut result_c0 = RnsPolynomial::zero(context);
    let mut result_c1 = RnsPolynomial::zero(context);
    for (c0, c1) in contributions {
        result_c0 = result_c0.add(&c0);
        result_c1 = result_c1.add(&c1);
    }

    RnsCiphertext::new(result_c0, result_c1, poly.levels())
}

pub(crate) fn keyswitch_rns_hybrid(
    poly: &RnsPolynomial,
    ksk: &RnsKeySwitchingKey,
    q_context: &RnsContext,
    eval_key_context: &RnsContext,
) -> RnsCiphertext {
    assert_eq!(ksk.level, q_context.levels() - 1, "keyswitch key level must match the active Q chain");
    assert_eq!(ksk.aux_level_count, eval_key_context.levels() - q_context.levels(), "eval-key context must match the keyswitch key auxiliary basis");
    let active_digits = num_decomp_digits(q_context.total_modulus_bits(), ksk.decomp_bits);
    assert!(ksk.ksk.len() >= active_digits, "keyswitch key must cover the active modulus chain");

    let digits = decompose_poly_rns_with_levels(poly, q_context, ksk.decomp_bits, active_digits);
    let contributions: Vec<(RnsPolynomial, RnsPolynomial)> = digits
        .par_iter()
        .zip(ksk.ksk.par_iter())
        .map(|(digit_poly, key_ct)| {
            let lifted_digit = lift_poly_to_context(digit_poly, q_context, eval_key_context);
            (
                lifted_digit.multiply_ntt(&key_ct.c0, eval_key_context),
                lifted_digit.multiply_ntt(&key_ct.c1, eval_key_context),
            )
        })
        .collect();
    let mut result_c0 = RnsPolynomial::zero(eval_key_context);
    let mut result_c1 = RnsPolynomial::zero(eval_key_context);
    for (c0, c1) in contributions {
        result_c0 = result_c0.add(&c0);
        result_c1 = result_c1.add(&c1);
    }

    let aux_modulus = eval_key_context.total_modulus() / q_context.total_modulus();
    let c0 = mod_down_eval_to_q(&result_c0, q_context, eval_key_context, &aux_modulus);
    let c1 = mod_down_eval_to_q(&result_c1, q_context, eval_key_context, &aux_modulus);

    RnsCiphertext::new(c0, c1, poly.levels())
}

pub(crate) fn decompose_poly_rns_with_levels(
    poly: &RnsPolynomial,
    context: &RnsContext,
    decomp_bits: usize,
    levels: usize,
) -> Vec<RnsPolynomial> {
    let base = 1u64 << decomp_bits;
    let precomp = BasisConversionPrecomp::new(context.primes().iter().map(|prime| prime.modulus).collect(), base);
    let coeff_digits: Vec<Vec<i64>> = (0..context.poly_degree())
        .into_par_iter()
        .map(|coeff_index| {
            let residues: Vec<u64> = poly
                .residues()
                .iter()
                .map(|residue| residue.coeffs()[coeff_index])
                .collect();
            decompose_residues_balanced(&residues, &precomp, levels)
        })
        .collect();

    let mut per_digit_residues = vec![vec![vec![0u64; context.poly_degree()]; context.levels()]; levels];
    for (coeff_index, digits) in coeff_digits.into_iter().enumerate() {
        for (digit_index, digit) in digits.into_iter().enumerate() {
            for (prime_index, prime) in context.primes().iter().enumerate() {
                per_digit_residues[digit_index][prime_index][coeff_index] = signed_digit_mod_q(digit, prime.modulus);
            }
        }
    }

    per_digit_residues
        .into_iter()
        .map(|digit_residues| {
            let residues = digit_residues
                .into_iter()
                .zip(context.primes().iter())
                .map(|(coeffs, prime)| ModularPolynomial::new(coeffs, prime.modulus))
                .collect();
            RnsPolynomial::new(context, residues)
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

fn lift_poly_to_context(poly: &RnsPolynomial, source_context: &RnsContext, target_context: &RnsContext) -> RnsPolynomial {
    assert!(target_context.levels() >= source_context.levels(), "target context must extend the source basis");
    let source_moduli: Vec<u64> = source_context.primes().iter().map(|prime| prime.modulus).collect();
    let target_moduli: Vec<u64> = target_context.primes()[source_context.levels()..]
        .iter()
        .map(|prime| prime.modulus)
        .collect();
    if target_moduli.is_empty() {
        return poly.clone();
    }

    let source_precomp = BasisConversionPrecomp::new(source_moduli.clone(), 1u64 << 30);
    let converted: Vec<Vec<u64>> = (0..source_context.poly_degree())
        .into_par_iter()
        .map(|coeff_index| {
            let residues: Vec<u64> = poly
                .residues()
                .iter()
                .map(|residue| residue.coeffs()[coeff_index])
                .collect();
            target_moduli
                .iter()
                .map(|&target_modulus| convert_centered_residues_to_target(&residues, &source_precomp, target_modulus))
                .collect()
        })
        .collect();

    let mut residues = poly.residues().to_vec();
    for (target_offset, target_modulus) in target_moduli.iter().enumerate() {
        let coeffs: Vec<u64> = converted.iter().map(|row| row[target_offset]).collect();
        residues.push(ModularPolynomial::new(coeffs, *target_modulus));
    }

    RnsPolynomial::new(target_context, residues)
}

fn mod_down_eval_to_q(
    poly: &RnsPolynomial,
    q_context: &RnsContext,
    eval_key_context: &RnsContext,
    aux_modulus: &BigUint,
) -> RnsPolynomial {
    let q_levels = q_context.levels();
    let aux_moduli: Vec<u64> = eval_key_context.primes()[q_levels..]
        .iter()
        .map(|prime| prime.modulus)
        .collect();
    let aux_precomp = BasisConversionPrecomp::new(aux_moduli, 1u64 << 30);

    let residues = q_context
        .primes()
        .iter()
        .enumerate()
        .map(|(prime_index, prime)| {
            let p_mod_q = (aux_modulus % BigUint::from(prime.modulus))
                .to_u64()
                .expect("P mod q_i must fit in u64");
            let p_inv_mod_q = mod_inv_u64(p_mod_q, prime.modulus)
                .expect("P must be invertible modulo each q_i");
            let coeffs: Vec<u64> = (0..q_context.poly_degree())
                .into_par_iter()
                .map(|coeff_index| {
                    let aux_residues: Vec<u64> = poly.residues()[q_levels..]
                        .iter()
                        .map(|residue| residue.coeffs()[coeff_index])
                        .collect();
                    let centered_p_mod_q = convert_centered_residues_to_target(
                        &aux_residues,
                        &aux_precomp,
                        prime.modulus,
                    );
                    let numerator = sub_mod_u64(poly.residues()[prime_index].coeffs()[coeff_index], centered_p_mod_q, prime.modulus);
                    ((numerator as u128 * p_inv_mod_q as u128) % prime.modulus as u128) as u64
                })
                .collect();
            ModularPolynomial::new(coeffs, prime.modulus)
        })
        .collect();

    RnsPolynomial::new(q_context, residues)
}

#[derive(Clone, Debug)]
struct BasisConversionPrecomp {
    moduli: Vec<u64>,
    inverses: Vec<Vec<u64>>,
    modulus_base_digits: Vec<u64>,
    half_modulus_base_digits: Vec<u64>,
    base: u64,
}

impl BasisConversionPrecomp {
    fn new(moduli: Vec<u64>, base: u64) -> Self {
        let mut inverses = vec![Vec::new(); moduli.len()];
        for i in 0..moduli.len() {
            for j in 0..i {
                inverses[i].push(
                    mod_inv_u64(moduli[j] % moduli[i], moduli[i])
                        .expect("RNS moduli must be pairwise coprime"),
                );
            }
        }

        let mut modulus_base_digits = vec![1u64];
        for &modulus in &moduli {
            mul_add_small_base(&mut modulus_base_digits, modulus, 0, base);
        }
        trim_base_digits(&mut modulus_base_digits);

        let mut half_modulus_base_digits = modulus_base_digits.clone();
        div2_base_digits(&mut half_modulus_base_digits, base);
        trim_base_digits(&mut half_modulus_base_digits);

        Self {
            moduli,
            inverses,
            modulus_base_digits,
            half_modulus_base_digits,
            base,
        }
    }
}

fn convert_residues_to_targets(
    residues: &[u64],
    precomp: &BasisConversionPrecomp,
    target_moduli: &[u64],
) -> Vec<u64> {
    let mixed = mixed_radix_digits(residues, precomp);
    target_moduli
        .iter()
        .map(|&target_modulus| evaluate_mixed_radix_mod(&mixed, &precomp.moduli, target_modulus))
        .collect()
}

fn convert_centered_residues_to_target(
    residues: &[u64],
    precomp: &BasisConversionPrecomp,
    target_modulus: u64,
) -> u64 {
    let mixed = mixed_radix_digits(residues, precomp);
    let canonical_base_digits = canonical_digits_from_mixed_radix(&mixed, &precomp.moduli, precomp.base);
    let canonical_mod_target = evaluate_mixed_radix_mod(&mixed, &precomp.moduli, target_modulus);
    if cmp_base_digits(&canonical_base_digits, &precomp.half_modulus_base_digits) == std::cmp::Ordering::Greater {
        let modulus_mod_target = base_digits_mod_modulus(&precomp.modulus_base_digits, precomp.base, target_modulus);
        sub_mod_u64(canonical_mod_target, modulus_mod_target, target_modulus)
    } else {
        canonical_mod_target
    }
}

fn decompose_residues_balanced(
    residues: &[u64],
    precomp: &BasisConversionPrecomp,
    levels: usize,
) -> Vec<i64> {
    let mixed = mixed_radix_digits(residues, precomp);
    let canonical = canonical_digits_from_mixed_radix(&mixed, &precomp.moduli, precomp.base);
    let signed_digits = if cmp_base_digits(&canonical, &precomp.half_modulus_base_digits) == std::cmp::Ordering::Greater {
        let magnitude = sub_base_digits(&precomp.modulus_base_digits, &canonical, precomp.base);
        magnitude.into_iter().map(|digit| -(digit as i64)).collect()
    } else {
        canonical.into_iter().map(|digit| digit as i64).collect()
    };
    normalize_balanced_digits(signed_digits, levels, precomp.base as i64)
}

fn mixed_radix_digits(residues: &[u64], precomp: &BasisConversionPrecomp) -> Vec<u64> {
    assert_eq!(residues.len(), precomp.moduli.len(), "residue count must match the basis size");
    let mut digits = residues.to_vec();
    for i in 1..digits.len() {
        let modulus = precomp.moduli[i];
        let mut value = digits[i];
        for j in 0..i {
            value = sub_mod_u64(value, digits[j], modulus);
            value = mul_mod_u64(value, precomp.inverses[i][j], modulus);
        }
        digits[i] = value;
    }
    digits
}

fn evaluate_mixed_radix_mod(mixed: &[u64], moduli: &[u64], target_modulus: u64) -> u64 {
    if mixed.is_empty() {
        return 0;
    }

    let mut acc = mixed[mixed.len() - 1] % target_modulus;
    for index in (0..mixed.len() - 1).rev() {
        acc = ((acc as u128 * (moduli[index] % target_modulus) as u128) % target_modulus as u128) as u64;
        acc = add_mod_u64(acc, mixed[index] % target_modulus, target_modulus);
    }
    acc
}

fn canonical_digits_from_mixed_radix(mixed: &[u64], moduli: &[u64], base: u64) -> Vec<u64> {
    let mut digits = vec![0u64];
    for index in (0..mixed.len()).rev() {
        mul_add_small_base(&mut digits, moduli[index], mixed[index], base);
    }
    trim_base_digits(&mut digits);
    digits
}

fn normalize_balanced_digits(mut digits: Vec<i64>, levels: usize, base: i64) -> Vec<i64> {
    let half = base >> 1;
    digits.resize(levels + 1, 0);
    let mut result = Vec::with_capacity(levels);
    let mut carry = 0i64;

    for index in 0..levels {
        let value = digits[index] + carry;
        let mut digit = value % base;
        if digit >= half {
            digit -= base;
        } else if digit < -half {
            digit += base;
        }
        carry = (value - digit) / base;
        result.push(digit);
    }

    debug_assert_eq!(carry + digits[levels], 0, "balanced decomposition exceeded the requested digit count");
    result
}

fn signed_digit_mod_q(digit: i64, modulus: u64) -> u64 {
    if digit >= 0 {
        digit as u64 % modulus
    } else {
        let magnitude = ((-digit) as u64) % modulus;
        if magnitude == 0 {
            0
        } else {
            modulus - magnitude
        }
    }
}

fn mul_add_small_base(digits: &mut Vec<u64>, multiplier: u64, addend: u64, base: u64) {
    let mut carry = addend as u128;
    for digit in digits.iter_mut() {
        let value = *digit as u128 * multiplier as u128 + carry;
        *digit = (value % base as u128) as u64;
        carry = value / base as u128;
    }
    while carry > 0 {
        digits.push((carry % base as u128) as u64);
        carry /= base as u128;
    }
}

fn div2_base_digits(digits: &mut [u64], base: u64) {
    let mut carry = 0u64;
    for digit in digits.iter_mut().rev() {
        let value = carry as u128 * base as u128 + *digit as u128;
        *digit = (value / 2) as u64;
        carry = (value % 2) as u64;
    }
}

fn trim_base_digits(digits: &mut Vec<u64>) {
    while digits.len() > 1 && digits.last() == Some(&0) {
        digits.pop();
    }
}

fn cmp_base_digits(lhs: &[u64], rhs: &[u64]) -> std::cmp::Ordering {
    let lhs_len = lhs.iter().rposition(|&digit| digit != 0).map(|index| index + 1).unwrap_or(0);
    let rhs_len = rhs.iter().rposition(|&digit| digit != 0).map(|index| index + 1).unwrap_or(0);
    if lhs_len != rhs_len {
        return lhs_len.cmp(&rhs_len);
    }
    for index in (0..lhs_len).rev() {
        if lhs[index] != rhs[index] {
            return lhs[index].cmp(&rhs[index]);
        }
    }
    std::cmp::Ordering::Equal
}

fn sub_base_digits(lhs: &[u64], rhs: &[u64], base: u64) -> Vec<u64> {
    let mut result = Vec::with_capacity(lhs.len());
    let mut borrow = 0i128;
    for index in 0..lhs.len() {
        let lhs_digit = lhs[index] as i128;
        let rhs_digit = rhs.get(index).copied().unwrap_or(0) as i128;
        let mut value = lhs_digit - rhs_digit - borrow;
        if value < 0 {
            value += base as i128;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result.push(value as u64);
    }
    debug_assert_eq!(borrow, 0, "base digit subtraction underflowed");
    trim_base_digits(&mut result);
    result
}

fn base_digits_mod_modulus(digits: &[u64], base: u64, modulus: u64) -> u64 {
    let mut acc = 0u64;
    for &digit in digits.iter().rev() {
        acc = ((acc as u128 * (base % modulus) as u128) % modulus as u128) as u64;
        acc = add_mod_u64(acc, digit % modulus, modulus);
    }
    acc
}

fn add_mod_u64(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    let sum = lhs as u128 + rhs as u128;
    if sum >= modulus as u128 {
        (sum - modulus as u128) as u64
    } else {
        sum as u64
    }
}

fn sub_mod_u64(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    if lhs >= rhs {
        lhs - rhs
    } else {
        modulus - (rhs - lhs)
    }
}

fn mul_mod_u64(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((lhs as u128 * rhs as u128) % modulus as u128) as u64
}