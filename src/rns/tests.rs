use super::basis::default_rns_primes;
use super::keys::sample_error_poly;
use super::ops::encrypt_rns_with_samples;
use super::{RnsCkksContext, RnsCkksParams, RnsCiphertext, RnsContext, RnsPolynomial, RnsPrime, RnsSecretKey};
use num_bigint::BigUint;
use num_complex::Complex64;

fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
    (a - b).norm() <= eps
}

fn sample_context() -> RnsContext {
    RnsContext::new(
        8,
        vec![
            RnsPrime::new(998_244_353, 3),
            RnsPrime::new(1_004_535_809, 3),
            RnsPrime::new(469_762_049, 3),
        ],
    )
    .expect("sample RNS context must be valid")
}

fn sample_ckks_context() -> RnsCkksContext {
    RnsCkksContext::new(RnsCkksParams {
        poly_degree: 8,
        primes: default_rns_primes()[..3].to_vec(),
        aux_primes: Vec::new(),
        scale_bits: 40,
    })
    .expect("sample RNS CKKS context must be valid")
}

#[test]
fn context_builds_ntt_plans_for_each_prime() {
    let ctx = sample_context();
    assert_eq!(ctx.levels(), 3);
    assert_eq!(ctx.ntt_plans().len(), 3);
    assert_eq!(ctx.poly_degree(), 8);
}

#[test]
fn rns_polynomial_addition_reconstructs_correctly() {
    let ctx = sample_context();
    let lhs = ctx.poly_from_coeffs(&[10, 20, 30, 40]);
    let rhs = ctx.poly_from_coeffs(&[3, 4, 5, 6]);
    let sum = lhs.add(&rhs);
    let reconstructed = sum.reconstruct_coefficients(&ctx);

    assert_eq!(reconstructed[0], BigUint::from(13u64));
    assert_eq!(reconstructed[1], BigUint::from(24u64));
    assert_eq!(reconstructed[2], BigUint::from(35u64));
    assert_eq!(reconstructed[3], BigUint::from(46u64));
}

#[test]
fn rns_polynomial_multiplication_matches_per_level_reference() {
    let ctx = sample_context();
    let lhs = ctx.poly_from_coeffs(&[1, 2, 3, 4]);
    let rhs = ctx.poly_from_coeffs(&[4, 3, 2, 1]);
    let product = lhs.multiply_ntt(&rhs, &ctx);

    for ((lhs_level, rhs_level), product_level) in lhs
        .residues()
        .iter()
        .zip(rhs.residues().iter())
        .zip(product.residues().iter())
    {
        let expected = lhs_level.multiply_naive(rhs_level);
        assert_eq!(&expected, product_level);
    }
}

#[test]
fn dropping_rns_level_removes_last_modulus() {
    let ctx = sample_context();
    let poly = ctx.poly_from_coeffs(&[1, 2, 3]);
    let dropped = poly.drop_last_level();

    assert_eq!(dropped.levels(), 2);
    assert_eq!(dropped.residues()[0].modulus(), 998_244_353);
    assert_eq!(dropped.residues()[1].modulus(), 1_004_535_809);
}

#[test]
fn biguint_coefficients_reduce_into_each_residue() {
    let ctx = sample_context();
    let big = ctx.total_modulus() + BigUint::from(17u64);
    let poly = RnsPolynomial::from_biguint_coeffs(&ctx, &[big]);

    for residue in poly.residues() {
        assert_eq!(residue.coeffs()[0], 17);
    }
}

#[test]
fn rns_ckks_context_reports_slot_count_and_scale() {
    let ctx = sample_ckks_context();
    assert_eq!(ctx.num_slots(), 4);
    assert_eq!(ctx.params().scale_bits, 40);
    assert_eq!(ctx.rns_context().levels(), 3);
    assert!(!ctx.has_aux_basis());
}

#[test]
fn default_rns_ckks_presets_build() {
    let toy = RnsCkksContext::new(RnsCkksParams::toy()).expect("toy RNS CKKS params must build");
    let small = RnsCkksContext::new(RnsCkksParams::small()).expect("small RNS CKKS params must build");

    assert_eq!(toy.num_slots(), 4);
    assert_eq!(small.num_slots(), 512);
    assert_eq!(toy.levels(), 6);
    assert!(small.total_modulus_bits() >= 170);
}

#[test]
fn realistic_rns_ckks_params_build_for_n15_and_8_levels() {
    let realistic = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS params must build");

    assert_eq!(realistic.params().poly_degree, 1 << 15);
    assert_eq!(realistic.num_slots(), 1 << 14);
    assert_eq!(realistic.levels(), 8);
    assert_eq!(realistic.params().scale_bits, 45);
    assert!(realistic.total_modulus_bits() >= 400);
    assert!(realistic.total_modulus_bits() <= 410);
    assert!(realistic.has_aux_basis());
    assert_eq!(realistic.aux_context().expect("realistic preset must have aux basis").levels(), 2);
    assert!(realistic.aux_modulus_bits().expect("realistic preset must report aux modulus bits") >= 98);
    assert!(realistic.eval_key_modulus_bits().expect("realistic preset must report eval-key modulus bits") >= 480);
}

#[test]
fn realistic_eval_key_basis_is_disjoint_and_extends_q_chain() {
    let realistic = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS params must build");
    let q_moduli: Vec<u64> = realistic.rns_context().primes().iter().map(|prime| prime.modulus).collect();
    let p_moduli: Vec<u64> = realistic
        .aux_context()
        .expect("realistic preset must have aux basis")
        .primes()
        .iter()
        .map(|prime| prime.modulus)
        .collect();
    let eval_key_context = realistic
        .eval_key_context()
        .expect("realistic preset must have eval-key context");

    assert!(p_moduli.iter().all(|modulus| !q_moduli.contains(modulus)));
    assert_eq!(eval_key_context.levels(), q_moduli.len() + p_moduli.len());
    assert_eq!(
        eval_key_context.primes()[..q_moduli.len()]
            .iter()
            .map(|prime| prime.modulus)
            .collect::<Vec<_>>(),
        q_moduli
    );
    assert_eq!(
        eval_key_context.primes()[q_moduli.len()..]
            .iter()
            .map(|prime| prime.modulus)
            .collect::<Vec<_>>(),
        p_moduli
    );
}

#[test]
fn params_validation_rejects_overlapping_q_and_p_moduli() {
    let err = RnsCkksParams {
        poly_degree: 8,
        primes: vec![RnsPrime::new(998_244_353, 3)],
        aux_primes: vec![RnsPrime::new(998_244_353, 3)],
        scale_bits: 30,
    }
    .validate()
    .expect_err("Q and P chains must be disjoint");

    assert!(err.contains("distinct from the ciphertext modulus chain"));
}

#[test]
fn level_context_returns_prefix_chain() {
    let ctx = sample_ckks_context();
    let level_one = ctx.level_context(1).expect("level 1 context must exist");

    assert_eq!(level_one.levels(), 2);
    assert_eq!(level_one.primes()[0].modulus, 998_244_353);
    assert_eq!(level_one.primes()[1].modulus, 1_004_535_809);
}

#[test]
fn rns_ciphertext_addition_is_component_wise() {
    let ctx = sample_context();
    let lhs = RnsCiphertext::new(
        ctx.poly_from_coeffs(&[1, 2, 3]),
        ctx.poly_from_coeffs(&[4, 5, 6]),
        40,
    );
    let rhs = RnsCiphertext::new(
        ctx.poly_from_coeffs(&[10, 20, 30]),
        ctx.poly_from_coeffs(&[40, 50, 60]),
        40,
    );

    let sum = lhs.add(&rhs);
    let c0 = sum.c0.reconstruct_coefficients(&ctx);
    let c1 = sum.c1.reconstruct_coefficients(&ctx);

    assert_eq!(c0[0], BigUint::from(11u64));
    assert_eq!(c0[1], BigUint::from(22u64));
    assert_eq!(c1[0], BigUint::from(44u64));
    assert_eq!(c1[1], BigUint::from(55u64));
}

#[test]
fn rns_ciphertext_drop_last_level_tracks_level() {
    let ctx = sample_context();
    let ct = RnsCiphertext::new(
        ctx.poly_from_coeffs(&[1, 2, 3]),
        ctx.poly_from_coeffs(&[4, 5, 6]),
        40,
    );

    let dropped = ct.drop_last_level();

    assert_eq!(dropped.level, 1);
    assert_eq!(dropped.c0.levels(), 2);
    assert_eq!(dropped.c1.levels(), 2);
}

#[test]
fn sparse_ternary_rns_secret_key_has_expected_weight() {
    let ctx = sample_context();
    let secret_key = RnsSecretKey::sample_sparse_ternary(&ctx, 3);
    let nonzero = secret_key
        .poly
        .residues()[0]
        .coeffs()
        .iter()
        .filter(|&&coeff| coeff != 0)
        .count();

    assert_eq!(nonzero, 3);
}

#[test]
fn rns_keygen_satisfies_public_key_relation_per_residue() {
    let ctx = sample_ckks_context();
    let pair = ctx.keygen(3);
    let relation = pair
        .public_key
        .p0
        .add(&pair.public_key.p1.multiply_ntt(&pair.secret_key.poly, ctx.rns_context()));

    for (got, expected) in relation.residues().iter().zip(pair.error.residues().iter()) {
        assert_eq!(got, expected);
    }
}

#[test]
fn rns_encode_decode_round_trip_recovers_slots() {
    let ctx = sample_ckks_context();
    let slots = vec![
        Complex64::new(0.125, -0.25),
        Complex64::new(-0.75, 0.5),
        Complex64::new(0.0, 0.3125),
        Complex64::new(0.625, -0.4375),
    ];

    let poly = ctx.encode(&slots);
    let recovered = ctx.decode(&poly);

    for (&expected, &actual) in slots.iter().zip(recovered.iter()) {
        assert!(approx_eq(expected, actual, 1e-3), "expected {expected:?}, got {actual:?}");
    }
}

#[test]
fn rns_encode_real_matches_complex_zero_imag() {
    let ctx = sample_ckks_context();
    let real_slots = vec![0.125, -0.75, 0.0, 0.625];
    let complex_slots: Vec<Complex64> = real_slots
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect();

    let poly_from_real = ctx.encode_real(&real_slots);
    let poly_from_complex = ctx.encode(&complex_slots);

    assert_eq!(poly_from_real, poly_from_complex);
}

#[test]
fn rns_encrypt_decrypt_round_trip_recovers_plaintext() {
    let ctx = sample_ckks_context();
    let pair = ctx.keygen(3);
    let slots = vec![
        Complex64::new(0.125, -0.25),
        Complex64::new(-0.0625, 0.1875),
        Complex64::new(0.03125, 0.09375),
        Complex64::new(-0.125, -0.0625),
    ];

    let plaintext = ctx.encode(&slots);
    let ciphertext = ctx.encrypt(&plaintext, &pair.public_key);
    let decrypted = ctx.decrypt(&ciphertext, &pair.secret_key);
    let recovered = ctx.decode(&decrypted);

    for (&expected, &actual) in slots.iter().zip(recovered.iter()) {
        assert!(approx_eq(expected, actual, 5e-3), "expected {expected:?}, got {actual:?}");
    }
}

#[test]
fn rns_ciphertext_has_context_level_and_scale() {
    let ctx = sample_ckks_context();
    let pair = ctx.keygen(3);
    let plaintext = ctx.encode_real(&[0.125, -0.25, 0.0625, 0.375]);
    let ciphertext = ctx.encrypt(&plaintext, &pair.public_key);

    assert_eq!(ciphertext.level, ctx.levels() - 1);
    assert_eq!(ciphertext.scale_bits, ctx.params().scale_bits);
    assert_eq!(ciphertext.c0.levels(), ctx.levels());
    assert_eq!(ciphertext.c1.levels(), ctx.levels());
}

#[test]
fn rns_encrypt_decrypt_noiseless_matches_exact_plaintext() {
    let ctx = sample_ckks_context();
    let pair = ctx.keygen(3);
    let plaintext = ctx.encode(&[
        Complex64::new(0.125, -0.25),
        Complex64::new(-0.0625, 0.1875),
        Complex64::new(0.03125, 0.09375),
        Complex64::new(-0.125, -0.0625),
    ]);
    let mask = ctx.rns_context().poly_from_coeffs(&[1, 0, 1, 0, 0, 0, 0, 0]);
    let zero = ctx.rns_context().zero_poly();

    let ciphertext = encrypt_rns_with_samples(
        &plaintext,
        &pair.public_key,
        ctx.rns_context(),
        ctx.params().scale_bits,
        &mask,
        &zero,
        &zero,
    );
    let decrypted = ctx.decrypt(&ciphertext, &pair.secret_key);
    let expected = plaintext.add(&pair.error.multiply_ntt(&mask, ctx.rns_context()));

    assert_eq!(decrypted, expected);
}

#[test]
fn realistic_params_round_trip_encode_encrypt_decrypt_decode() {
    let ctx = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS params must build");
    let pair = ctx.keygen(64);
    let mut slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    slots[0] = Complex64::new(0.125, -0.25);
    slots[1] = Complex64::new(-0.0625, 0.1875);
    slots[2] = Complex64::new(0.03125, 0.09375);
    slots[3] = Complex64::new(-0.125, -0.0625);

    let plaintext = ctx.encode(&slots);
    let mut mask_coeffs = vec![0u64; ctx.params().poly_degree];
    mask_coeffs[0] = 1;
    let mask = ctx.rns_context().poly_from_coeffs(&mask_coeffs);
    let mut rng = rand::thread_rng();
    let e0 = sample_error_poly(ctx.rns_context(), &mut rng);
    let e1 = sample_error_poly(ctx.rns_context(), &mut rng);

    let ciphertext = encrypt_rns_with_samples(
        &plaintext,
        &pair.public_key,
        ctx.rns_context(),
        ctx.params().scale_bits,
        &mask,
        &e0,
        &e1,
    );
    let decrypted = ctx.decrypt(&ciphertext, &pair.secret_key);
    let recovered = ctx.decode(&decrypted);

    for (&expected, &actual) in slots.iter().zip(recovered.iter()).take(4) {
        assert!(approx_eq(expected, actual, 2e-3), "expected {expected:?}, got {actual:?}");
    }
}

#[test]
fn realistic_params_mult_plain_slots_rescales_and_matches_hadamard_product() {
    let ctx = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS params must build");
    let pair = ctx.keygen(64);

    let mut msg_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    msg_slots[0] = Complex64::new(0.125, -0.25);
    msg_slots[1] = Complex64::new(-0.0625, 0.1875);
    msg_slots[2] = Complex64::new(0.03125, 0.09375);
    msg_slots[3] = Complex64::new(-0.125, -0.0625);

    let mut plain_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    plain_slots[0] = Complex64::new(0.5, 0.25);
    plain_slots[1] = Complex64::new(-0.25, 0.125);
    plain_slots[2] = Complex64::new(0.125, -0.25);
    plain_slots[3] = Complex64::new(-0.5, 0.0);

    let plaintext = ctx.encode(&msg_slots);
    let ciphertext = ctx.encrypt(&plaintext, &pair.public_key);
    let multiplied = ctx.mult_plain_slots(&ciphertext, &plain_slots);
    let decrypted = ctx.decrypt(&multiplied, &pair.secret_key);
    let recovered = ctx.decode_at_scale(&decrypted, multiplied.scale_bits);

    assert_eq!(multiplied.level, 6);
    assert!(multiplied.scale_bits < ciphertext.scale_bits * 2);

    for ((&lhs, &rhs), &actual) in msg_slots
        .iter()
        .zip(plain_slots.iter())
        .zip(recovered.iter())
        .take(4)
    {
        let expected = lhs * rhs;
        assert!(approx_eq(expected, actual, 6e-3), "expected {expected:?}, got {actual:?}");
    }
}

#[test]
fn realistic_params_mult_slots_rescales_and_matches_hadamard_product() {
    let ctx = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS params must build");
    let pair = ctx.keygen(64);

    let mut lhs_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    lhs_slots[0] = Complex64::new(0.125, -0.25);
    lhs_slots[1] = Complex64::new(-0.0625, 0.1875);
    lhs_slots[2] = Complex64::new(0.03125, 0.09375);
    lhs_slots[3] = Complex64::new(-0.125, -0.0625);

    let mut rhs_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    rhs_slots[0] = Complex64::new(0.5, 0.25);
    rhs_slots[1] = Complex64::new(-0.25, 0.125);
    rhs_slots[2] = Complex64::new(0.125, -0.25);
    rhs_slots[3] = Complex64::new(-0.5, 0.0);

    let lhs = ctx.encrypt(&ctx.encode(&lhs_slots), &pair.public_key);
    let rhs = ctx.encrypt(&ctx.encode(&rhs_slots), &pair.public_key);
    let multiplied = ctx.mult_slots(&lhs, &rhs);
    let decrypted = ctx.decrypt_quadratic(&multiplied, &pair.secret_key);
    let recovered = ctx.decode_at_scale(&decrypted, multiplied.scale_bits);

    assert_eq!(multiplied.level, 6);
    assert!(multiplied.scale_bits < lhs.scale_bits + rhs.scale_bits);

    for ((&lhs_slot, &rhs_slot), &actual) in lhs_slots
        .iter()
        .zip(rhs_slots.iter())
        .zip(recovered.iter())
        .take(4)
    {
        let expected = lhs_slot * rhs_slot;
        assert!(approx_eq(expected, actual, 1.2e-2), "expected {expected:?}, got {actual:?}");
    }
}

#[test]
fn realistic_params_mult_slots_is_commutative_after_decrypt() {
    let ctx = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS params must build");
    let pair = ctx.keygen(64);

    let mut lhs_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    lhs_slots[0] = Complex64::new(0.125, -0.25);
    lhs_slots[1] = Complex64::new(-0.0625, 0.1875);

    let mut rhs_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    rhs_slots[0] = Complex64::new(0.5, 0.25);
    rhs_slots[1] = Complex64::new(-0.25, 0.125);

    let lhs = ctx.encrypt(&ctx.encode(&lhs_slots), &pair.public_key);
    let rhs = ctx.encrypt(&ctx.encode(&rhs_slots), &pair.public_key);
    let forward = ctx.mult_slots(&lhs, &rhs);
    let reverse = ctx.mult_slots(&rhs, &lhs);

    let forward_slots = ctx.decode_at_scale(&ctx.decrypt_quadratic(&forward, &pair.secret_key), forward.scale_bits);
    let reverse_slots = ctx.decode_at_scale(&ctx.decrypt_quadratic(&reverse, &pair.secret_key), reverse.scale_bits);

    for (&forward_value, &reverse_value) in forward_slots.iter().zip(reverse_slots.iter()).take(4) {
        assert!(approx_eq(forward_value, reverse_value, 1.2e-2), "expected {forward_value:?}, got {reverse_value:?}");
    }
}

#[test]
fn realistic_params_mult_slots_tracks_quadratic_shape() {
    let ctx = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS params must build");
    let pair = ctx.keygen(64);

    let mut lhs_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    let mut rhs_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    lhs_slots[0] = Complex64::new(0.25, 0.0);
    rhs_slots[0] = Complex64::new(0.5, 0.0);

    let lhs = ctx.encrypt(&ctx.encode(&lhs_slots), &pair.public_key);
    let rhs = ctx.encrypt(&ctx.encode(&rhs_slots), &pair.public_key);
    let raw_product = ctx.mult(&lhs, &rhs);
    let scaled_product = ctx.rescale_quadratic(&raw_product);

    assert_eq!(raw_product.level, 7);
    assert_eq!(raw_product.scale_bits, 90);
    assert_eq!(scaled_product.level, 6);
    assert!(scaled_product.scale_bits < raw_product.scale_bits);
    assert_eq!(scaled_product.c0.levels(), 7);
    assert_eq!(scaled_product.c1.levels(), 7);
    assert_eq!(scaled_product.c2.levels(), 7);
}

#[test]
fn realistic_params_relinearization_matches_quadratic_decryption() {
    let ctx = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS params must build");
    let pair = ctx.keygen(64);
    let relin_key = ctx.generate_relinearization_key(&pair.secret_key, &pair.public_key);

    let mut lhs_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    let mut rhs_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    lhs_slots[0] = Complex64::new(0.125, -0.25);
    lhs_slots[1] = Complex64::new(-0.0625, 0.1875);
    rhs_slots[0] = Complex64::new(0.5, 0.25);
    rhs_slots[1] = Complex64::new(-0.25, 0.125);

    let lhs = ctx.encrypt(&ctx.encode(&lhs_slots), &pair.public_key);
    let rhs = ctx.encrypt(&ctx.encode(&rhs_slots), &pair.public_key);
    let multiplied = ctx.mult(&lhs, &rhs);
    let relinearized = ctx.relinearize(&multiplied, &relin_key);

    let quadratic_slots = ctx.decode_at_scale(
        &ctx.decrypt_quadratic(&multiplied, &pair.secret_key),
        multiplied.scale_bits,
    );
    let linear_slots = ctx.decode_at_scale(&ctx.decrypt(&relinearized, &pair.secret_key), relinearized.scale_bits);

    for (&quadratic, &linear) in quadratic_slots.iter().zip(linear_slots.iter()).take(4) {
        assert!(approx_eq(quadratic, linear, 1.5e-2), "expected {quadratic:?}, got {linear:?}");
    }
}

#[test]
fn realistic_params_mult_relinearized_matches_hadamard_product() {
    let ctx = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS params must build");
    let pair = ctx.keygen(64);
    let relin_key = ctx.generate_relinearization_key(&pair.secret_key, &pair.public_key);

    let mut lhs_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    let mut rhs_slots = vec![Complex64::new(0.0, 0.0); ctx.num_slots()];
    lhs_slots[0] = Complex64::new(0.125, -0.25);
    lhs_slots[1] = Complex64::new(-0.0625, 0.1875);
    lhs_slots[2] = Complex64::new(0.03125, 0.09375);
    rhs_slots[0] = Complex64::new(0.5, 0.25);
    rhs_slots[1] = Complex64::new(-0.25, 0.125);
    rhs_slots[2] = Complex64::new(0.125, -0.25);

    let lhs = ctx.encrypt(&ctx.encode(&lhs_slots), &pair.public_key);
    let rhs = ctx.encrypt(&ctx.encode(&rhs_slots), &pair.public_key);
    let product = ctx.mult_relinearized(&lhs, &rhs, &relin_key);
    let recovered = ctx.decode_at_scale(&ctx.decrypt(&product, &pair.secret_key), product.scale_bits);

    assert_eq!(product.level, 6);

    for ((&lhs_slot, &rhs_slot), &actual) in lhs_slots.iter().zip(rhs_slots.iter()).zip(recovered.iter()).take(4) {
        let expected = lhs_slot * rhs_slot;
        assert!(approx_eq(expected, actual, 1.8e-2), "expected {expected:?}, got {actual:?}");
    }
}