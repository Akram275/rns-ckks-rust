use ckks::algorithms::halevi_shoup::{
    generate_plaintext_ciphertext_dot_product_rotation_keys,
    plaintext_ciphertext_dot_product,
};
use ckks::rns::{RnsCkksContext, RnsCkksParams};

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
    let rotation_keys = generate_plaintext_ciphertext_dot_product_rotation_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        ciphertext.level - 1,
        plaintext_slots.len(),
    );

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