use num_bigint::{BigInt, Sign};
use num_traits::Zero;

use super::basis::RnsContext;
use super::ciphertexts::{RnsCiphertext, RnsQuadraticCiphertext};
use super::keys::{sample_error_poly, sample_ternary_poly, RnsPublicKey, RnsSecretKey};
use super::keyswitching::{decompose_poly_rns_with_levels, num_decomp_digits, RnsKeySwitchingKey};
use super::polynomial::{mod_q_biguint_to_centered, RnsPolynomial};

pub(crate) fn encrypt_rns(
    plaintext: &RnsPolynomial,
    public_key: &RnsPublicKey,
    context: &RnsContext,
    scale_bits: usize,
) -> RnsCiphertext {
    assert_eq!(plaintext.levels(), context.levels(), "plaintext levels must match the active RNS context");
    assert_eq!(public_key.p0.levels(), context.levels(), "public key levels must match the active RNS context");

    let mut rng = rand::thread_rng();
    let u = sample_ternary_poly(context, &mut rng);
    let e0 = sample_error_poly(context, &mut rng);
    let e1 = sample_error_poly(context, &mut rng);

    encrypt_rns_with_samples(plaintext, public_key, context, scale_bits, &u, &e0, &e1)
}

pub(crate) fn encrypt_rns_with_samples(
    plaintext: &RnsPolynomial,
    public_key: &RnsPublicKey,
    context: &RnsContext,
    scale_bits: usize,
    u: &RnsPolynomial,
    e0: &RnsPolynomial,
    e1: &RnsPolynomial,
) -> RnsCiphertext {
    assert_eq!(u.levels(), context.levels(), "mask levels must match the active RNS context");
    assert_eq!(e0.levels(), context.levels(), "error levels must match the active RNS context");
    assert_eq!(e1.levels(), context.levels(), "error levels must match the active RNS context");

    let c0 = public_key.p0.multiply_ntt(u, context).add(e0).add(plaintext);
    let c1 = public_key.p1.multiply_ntt(u, context).add(e1);

    RnsCiphertext::new(c0, c1, scale_bits)
}

pub(crate) fn decrypt_rns(
    ciphertext: &RnsCiphertext,
    secret_key: &RnsSecretKey,
    context: &RnsContext,
) -> RnsPolynomial {
    assert_eq!(ciphertext.c0.levels(), context.levels(), "ciphertext levels must match the active RNS context");
    let c1s = ciphertext.c1.multiply_ntt(&secret_key.poly, context);
    ciphertext.c0.add(&c1s)
}

pub(crate) fn multiply_plain_rns(
    ciphertext: &RnsCiphertext,
    plain: &RnsPolynomial,
    context: &RnsContext,
) -> RnsCiphertext {
    assert_eq!(ciphertext.c0.levels(), context.levels(), "ciphertext levels must match the multiplication context");
    assert_eq!(plain.levels(), context.levels(), "plaintext levels must match the multiplication context");
    RnsCiphertext::new(
        ciphertext.c0.multiply_ntt(plain, context),
        ciphertext.c1.multiply_ntt(plain, context),
        ciphertext.scale_bits,
    )
}

pub(crate) fn rescale_ciphertext_rns(ciphertext: &RnsCiphertext, context: &RnsContext) -> RnsCiphertext {
    assert!(context.levels() > 1, "rescale requires at least two levels");
    let dropped_context = context
        .at_level(context.levels() - 1)
        .expect("dropped level context must exist");
    let dropped_prime = context.primes().last().expect("rescale requires a last prime").modulus;
    let dropped_prime_bigint = BigInt::from(dropped_prime);
    let dropped_prime_bits = (u64::BITS as usize - 1) - dropped_prime.leading_zeros() as usize;

    let rescale_poly = |poly: &RnsPolynomial| {
        let coeffs = poly.reconstruct_coefficients(context);
        let rounded: Vec<BigInt> = coeffs
            .iter()
            .map(|coeff| {
                let centered = mod_q_biguint_to_centered(coeff, &context.total_modulus());
                div_round_nearest(&centered, &dropped_prime_bigint)
            })
            .collect();
        dropped_context.poly_from_bigint_coeffs(&rounded)
    };

    RnsCiphertext {
        c0: rescale_poly(&ciphertext.c0),
        c1: rescale_poly(&ciphertext.c1),
        scale_bits: ciphertext.scale_bits.saturating_sub(dropped_prime_bits),
        level: ciphertext.level - 1,
    }
}

pub(crate) fn multiply_ciphertexts_rns(
    lhs: &RnsCiphertext,
    rhs: &RnsCiphertext,
    context: &RnsContext,
) -> RnsQuadraticCiphertext {
    assert_eq!(lhs.c0.levels(), context.levels(), "lhs levels must match multiplication context");
    assert_eq!(rhs.c0.levels(), context.levels(), "rhs levels must match multiplication context");

    let c0 = lhs.c0.multiply_ntt(&rhs.c0, context);
    let c1 = lhs
        .c0
        .multiply_ntt(&rhs.c1, context)
        .add(&lhs.c1.multiply_ntt(&rhs.c0, context));
    let c2 = lhs.c1.multiply_ntt(&rhs.c1, context);

    RnsQuadraticCiphertext::new(c0, c1, c2, lhs.scale_bits + rhs.scale_bits)
}

pub(crate) fn decrypt_quadratic_rns(
    ciphertext: &RnsQuadraticCiphertext,
    secret_key: &RnsSecretKey,
    context: &RnsContext,
) -> RnsPolynomial {
    assert_eq!(ciphertext.c0.levels(), context.levels(), "quadratic ciphertext levels must match the active RNS context");
    let s2 = secret_key.poly.multiply_ntt(&secret_key.poly, context);
    ciphertext
        .c0
        .add(&ciphertext.c1.multiply_ntt(&secret_key.poly, context))
        .add(&ciphertext.c2.multiply_ntt(&s2, context))
}

pub(crate) fn rescale_quadratic_ciphertext_rns(
    ciphertext: &RnsQuadraticCiphertext,
    context: &RnsContext,
) -> RnsQuadraticCiphertext {
    assert!(context.levels() > 1, "quadratic rescale requires at least two levels");
    let dropped_context = context
        .at_level(context.levels() - 1)
        .expect("dropped quadratic level context must exist");
    let dropped_prime = context.primes().last().expect("quadratic rescale requires a last prime").modulus;
    let dropped_prime_bigint = BigInt::from(dropped_prime);
    let dropped_prime_bits = (u64::BITS as usize - 1) - dropped_prime.leading_zeros() as usize;

    let rescale_poly = |poly: &RnsPolynomial| {
        let coeffs = poly.reconstruct_coefficients(context);
        let rounded: Vec<BigInt> = coeffs
            .iter()
            .map(|coeff| {
                let centered = mod_q_biguint_to_centered(coeff, &context.total_modulus());
                div_round_nearest(&centered, &dropped_prime_bigint)
            })
            .collect();
        dropped_context.poly_from_bigint_coeffs(&rounded)
    };

    RnsQuadraticCiphertext {
        c0: rescale_poly(&ciphertext.c0),
        c1: rescale_poly(&ciphertext.c1),
        c2: rescale_poly(&ciphertext.c2),
        scale_bits: ciphertext.scale_bits.saturating_sub(dropped_prime_bits),
        level: ciphertext.level - 1,
    }
}

pub(crate) fn keyswitch_rns(poly: &RnsPolynomial, ksk: &RnsKeySwitchingKey, context: &RnsContext) -> RnsCiphertext {
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

pub(crate) fn div_round_nearest(value: &BigInt, divisor: &BigInt) -> BigInt {
    assert!(divisor > &BigInt::zero(), "divisor must be positive");
    let half = divisor >> 1usize;
    if value.sign() == Sign::Minus {
        (value - &half) / divisor
    } else {
        (value + &half) / divisor
    }
}