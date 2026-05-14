use ckks::algorithms::halevi_shoup::{
    plaintext_ciphertext_dot_product,
    plaintext_ciphertext_dot_product_rotation_steps,
};
use ckks::rns::{RnsCkksContext, RnsCkksParams, RnsCiphertext, RnsKeySwitchingKey, RnsRotationKey, RnsSecretKey};
use num_bigint::BigUint;

const NOISELESS_DECOMP_BITS: usize = 30;

fn num_decomp_digits(total_modulus_bits: u64, decomp_bits: usize) -> usize {
    (total_modulus_bits as usize + decomp_bits - 1) / decomp_bits
}

fn noiseless_rotation_key(
    context: &RnsCkksContext,
    secret_key: &RnsSecretKey,
    steps: isize,
    level: usize,
) -> RnsRotationKey {
    let galois_element = context.galois_element_for_rotation(steps);
    let source = secret_key
        .apply_galois_automorphism(galois_element)
        .poly
        .truncate_levels(level + 1);
    let level_context = context
        .level_context(level)
        .expect("dot-product level context must exist");
    let modulus = level_context.total_modulus();
    let digits = num_decomp_digits(modulus.bits(), NOISELESS_DECOMP_BITS);
    let mut base_power = BigUint::from(1u64);
    let base = BigUint::from(1u64) << NOISELESS_DECOMP_BITS;
    let source_coeffs = source.reconstruct_coefficients(&level_context);
    let zero = level_context.zero_poly();
    let mut ksk = Vec::with_capacity(digits);

    for _ in 0..digits {
        let coeffs: Vec<BigUint> = source_coeffs
            .iter()
            .map(|coeff| (coeff * &base_power) % &modulus)
            .collect();
        let plaintext = level_context.poly_from_biguint_coeffs(&coeffs);
        ksk.push(RnsCiphertext::new(plaintext, zero.clone(), 0));
        base_power = (&base_power * &base) % &modulus;
    }

    RnsRotationKey {
        galois_element,
        keyswitch_key: RnsKeySwitchingKey {
            ksk,
            decomp_bits: NOISELESS_DECOMP_BITS,
            level,
        },
    }
}

fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);

    let encrypted_slots = [0.125, -0.4, 0.03125, -0.125];
    let plaintext_slots = [2.0, -1.0, 0.5, 4.0];
    let expected: f64 = encrypted_slots
        .iter()
        .zip(plaintext_slots.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum();

    let mut padded_encrypted_slots = vec![0.0_f64; context.num_slots()];
    padded_encrypted_slots[..encrypted_slots.len()].copy_from_slice(&encrypted_slots);
    let ciphertext = context.encrypt(&context.encode_real(&padded_encrypted_slots), &key_pair.public_key);
    let rotation_keys = plaintext_ciphertext_dot_product_rotation_steps(plaintext_slots.len())
        .into_iter()
        .map(|step| (step, noiseless_rotation_key(&context, &key_pair.secret_key, step as isize, ciphertext.level - 1)))
        .collect();

    let dot_product = plaintext_ciphertext_dot_product(
        &context,
        &ciphertext,
        &plaintext_slots,
        &rotation_keys,
    );
    let recovered = context.decode_real_at_scale(
        &context.decrypt(&dot_product, &key_pair.secret_key),
        dot_product.scale_bits,
    );

    println!("RNS CKKS Halevi-Shoup plaintext-ciphertext dot product");
    println!("expected scalar = {expected}");
    println!("slot[0] output = {}", recovered[0]);
}