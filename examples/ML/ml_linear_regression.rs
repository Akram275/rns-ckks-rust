use ckks::algorithms::halevi_shoup::{
    generate_halevi_shoup_dot_product_rotation_keys,
    halevi_shoup_dot_product,
};
use ckks::rns::{RnsCkksContext, RnsCkksParams};

fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);

    // Model: y = w^T x with the bias folded into the last encrypted slot.
    let features = [0.35_f64, -0.2, 0.15, 1.0];
    let weights = [1.25_f64, -0.75, 0.5, 0.1];
    let expected: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(feature, weight)| feature * weight)
        .sum();

    let mut padded_features = vec![0.0_f64; context.num_slots()];
    padded_features[..features.len()].copy_from_slice(&features);
    let ciphertext = context.encrypt(&context.encode_real(&padded_features), &key_pair.public_key);
    let rotation_keys = generate_halevi_shoup_dot_product_rotation_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        ciphertext.level - 1,
        weights.len(),
    );

    let prediction = halevi_shoup_dot_product(&context, &ciphertext, &weights, &rotation_keys);
    let recovered = context.decode_real_at_scale(
        &context.decrypt(&prediction, &key_pair.secret_key),
        prediction.scale_bits,
    );

    println!("Encrypted linear regression inference");
    println!("features (bias in final slot) = {:?}", features);
    println!("weights = {:?}", weights);
    println!("expected prediction = {expected}");
    println!("recovered slot[0] = {}", recovered[0]);
}
