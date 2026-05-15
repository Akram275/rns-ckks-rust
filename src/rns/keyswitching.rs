use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, ToPrimitive};

use super::basis::RnsContext;
use super::ciphertexts::RnsCiphertext;
use super::keys::{sample_error_poly, sample_uniform_poly, RnsPublicKey, RnsSecretKey};
use super::ops::encrypt_rns;
use super::polynomial::{centered_bigint_to_mod_q, mod_q_biguint_to_centered, RnsPolynomial};
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
    let mut rng = rand::thread_rng();
    let mut ksk = Vec::with_capacity(digits);

    for _ in 0..digits {
        let coeffs: Vec<BigInt> = source_coeffs
            .iter()
            .map(|coeff| mod_q_biguint_to_centered(coeff, &q_modulus) * &base_power * &aux_modulus_bigint)
            .collect();
        let plaintext = eval_key_context.poly_from_bigint_coeffs(&coeffs);
        let a1 = sample_uniform_poly(eval_key_context, &mut rng);
        let error = sample_error_poly(eval_key_context, &mut rng);
        let a1s = a1.multiply_ntt(&target_secret, eval_key_context);
        let a0 = plaintext.add(&error).sub(&a1s);
        ksk.push(RnsCiphertext::new(a0, a1, 0));
        base_power *= &base;
    }

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
    let mut result_c0 = RnsPolynomial::zero(context);
    let mut result_c1 = RnsPolynomial::zero(context);

    for (digit_poly, key_ct) in digits.iter().zip(ksk.ksk.iter()) {
        result_c0 = result_c0.add(&digit_poly.multiply_ntt(&key_ct.c0, context));
        result_c1 = result_c1.add(&digit_poly.multiply_ntt(&key_ct.c1, context));
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
    let mut result_c0 = RnsPolynomial::zero(eval_key_context);
    let mut result_c1 = RnsPolynomial::zero(eval_key_context);

    for (digit_poly, key_ct) in digits.iter().zip(ksk.ksk.iter()) {
        let lifted_digit = lift_poly_to_context(digit_poly, q_context, eval_key_context);
        result_c0 = result_c0.add(&lifted_digit.multiply_ntt(&key_ct.c0, eval_key_context));
        result_c1 = result_c1.add(&lifted_digit.multiply_ntt(&key_ct.c1, eval_key_context));
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

fn lift_poly_to_context(poly: &RnsPolynomial, source_context: &RnsContext, target_context: &RnsContext) -> RnsPolynomial {
    let coeffs = poly.reconstruct_coefficients(source_context);
    let modulus = source_context.total_modulus();
    let centered: Vec<BigInt> = coeffs
        .iter()
        .map(|coeff| mod_q_biguint_to_centered(coeff, &modulus))
        .collect();
    target_context.poly_from_bigint_coeffs(&centered)
}

fn mod_down_eval_to_q(
    poly: &RnsPolynomial,
    q_context: &RnsContext,
    eval_key_context: &RnsContext,
    aux_modulus: &BigUint,
) -> RnsPolynomial {
    let q_levels = q_context.levels();
    let aux_context = RnsContext::new(
        q_context.poly_degree(),
        eval_key_context.primes()[q_levels..].to_vec(),
    )
    .expect("auxiliary context must be derivable from the eval-key basis");
    let aux_poly = RnsPolynomial::new(&aux_context, poly.residues()[q_levels..].to_vec());
    let p_residues = aux_poly.reconstruct_coefficients(&aux_context);
    let p_modulus = aux_context.total_modulus();

    let residues = q_context
        .primes()
        .iter()
        .enumerate()
        .map(|(prime_index, prime)| {
            let p_mod_q = (aux_modulus % BigUint::from(prime.modulus))
                .to_u64()
                .expect("P mod q_i must fit in u64");
            let p_inv_mod_q = super::basis::mod_inv_u64(p_mod_q, prime.modulus)
                .expect("P must be invertible modulo each q_i");
            let coeffs: Vec<u64> = poly.residues()[prime_index]
                .coeffs()
                .iter()
                .zip(p_residues.iter())
                .map(|(&q_coeff, p_coeff)| {
                    let centered_p = mod_q_biguint_to_centered(p_coeff, &p_modulus);
                    let p_lift_mod_q = centered_bigint_to_mod_q(&centered_p, &BigUint::from(prime.modulus))
                        .to_u64()
                        .expect("lifted P residue must fit in u64");
                    let numerator = if q_coeff >= p_lift_mod_q {
                        q_coeff - p_lift_mod_q
                    } else {
                        prime.modulus - (p_lift_mod_q - q_coeff)
                    };
                    ((numerator as u128 * p_inv_mod_q as u128) % prime.modulus as u128) as u64
                })
                .collect();
            ModularPolynomial::new(coeffs, prime.modulus)
        })
        .collect();

    RnsPolynomial::new(q_context, residues)
}