use ckks::algorithms::total_sum::{generate_total_sum_rotation_keys, total_sum};
use ckks::rns::{RnsCkksContext, RnsCkksParams};

fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);

    let active_slots = 8usize;
    let input = [0.125_f64, -0.4, 0.03125, -0.125, 0.375, -0.25, 0.5, 0.0625];
    let expected: f64 = input.iter().sum();

    let mut padded = vec![0.0_f64; context.num_slots()];
    padded[..input.len()].copy_from_slice(&input);
    let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
    let rotation_keys = generate_total_sum_rotation_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        ciphertext.level,
        active_slots,
    );

    let total = total_sum(&context, &ciphertext, active_slots, &rotation_keys);
    let recovered = context.decode_real_at_scale(
        &context.decrypt(&total, &key_pair.secret_key),
        total.scale_bits,
    );

    println!("RNS CKKS total sum over the first {active_slots} slots");
    println!("expected total = {expected}");
    println!("slot[0] output = {}", recovered[0]);
}