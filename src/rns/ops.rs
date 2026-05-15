use super::basis::{mod_inv_u64, RnsContext};
use super::ciphertexts::{RnsCiphertext, RnsQuadraticCiphertext};
use super::keys::{sample_error_poly, sample_ternary_poly, RnsPublicKey, RnsSecretKey};
use super::polynomial::RnsPolynomial;
use crate::poly::ModularPolynomial;
use rayon::prelude::*;

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
    let dropped_prime_bits = (u64::BITS as usize - 1) - dropped_prime.leading_zeros() as usize;

    RnsCiphertext {
        c0: rescale_poly_rns(&ciphertext.c0, &dropped_context, dropped_prime),
        c1: rescale_poly_rns(&ciphertext.c1, &dropped_context, dropped_prime),
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
    let dropped_prime_bits = (u64::BITS as usize - 1) - dropped_prime.leading_zeros() as usize;

    RnsQuadraticCiphertext {
        c0: rescale_poly_rns(&ciphertext.c0, &dropped_context, dropped_prime),
        c1: rescale_poly_rns(&ciphertext.c1, &dropped_context, dropped_prime),
        c2: rescale_poly_rns(&ciphertext.c2, &dropped_context, dropped_prime),
        scale_bits: ciphertext.scale_bits.saturating_sub(dropped_prime_bits),
        level: ciphertext.level - 1,
    }
}

fn rescale_poly_rns(
    poly: &RnsPolynomial,
    dropped_context: &RnsContext,
    dropped_prime: u64,
) -> RnsPolynomial {
    let dropped_coeffs = poly.residues().last().expect("rescale requires a dropped residue").coeffs();
    let dropped_half = dropped_prime >> 1;

    let residues = poly
        .residues()
        .par_iter()
        .take(dropped_context.levels())
        .zip(dropped_context.primes().par_iter())
        .map(|(residue, prime)| {
            let q_i = prime.modulus;
            let dropped_prime_mod_q_i = dropped_prime % q_i;
            let dropped_prime_inv = mod_inv_u64(dropped_prime_mod_q_i, q_i)
                .expect("dropped prime must be invertible modulo the surviving basis");
            let coeffs: Vec<u64> = residue
                .coeffs()
                .iter()
                .zip(dropped_coeffs.iter())
                .map(|(&coeff, &dropped_coeff)| {
                    let centered_dropped_mod_q_i = centered_residue_mod_prime(
                        dropped_coeff,
                        dropped_prime,
                        dropped_half,
                        q_i,
                    );
                    let numerator = sub_mod_u64(coeff, centered_dropped_mod_q_i, q_i);
                    ((numerator as u128 * dropped_prime_inv as u128) % q_i as u128) as u64
                })
                .collect();
            ModularPolynomial::new(coeffs, q_i)
        })
        .collect();

    RnsPolynomial::new(dropped_context, residues)
}

fn centered_residue_mod_prime(value: u64, modulus: u64, half: u64, target_modulus: u64) -> u64 {
    if value <= half {
        value % target_modulus
    } else {
        let distance = (modulus - value) % target_modulus;
        if distance == 0 {
            0
        } else {
            target_modulus - distance
        }
    }
}

fn sub_mod_u64(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    if lhs >= rhs {
        lhs - rhs
    } else {
        modulus - (rhs - lhs)
    }
}