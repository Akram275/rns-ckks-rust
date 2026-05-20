use ckks::algorithms::halevi_shoup::{
    generate_halevi_shoup_dot_product_rotation_keys,
    halevi_shoup_dot_product,
};
use ckks::algorithms::paterson_stockmeyer::{
    generate_paterson_stockmeyer_relinearization_keys,
    paterson_stockmeyer_eval,
};
use ckks::rns::{RnsCkksContext, RnsCkksParams};

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);

    // Model: sigma(w^T x) with the bias folded into the last encrypted slot.
    let features = [0.4_f64, -0.3, 0.2, 1.0];
    let weights = [0.8_f64, -0.6, 0.4, -0.05];
    let linear_score: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(feature, weight)| feature * weight)
        .sum();

    // Cubic approximation around zero: sigmoid(x) ~= 0.5 + x/4 - x^3/48.
    let sigmoid_poly = [0.5_f64, 0.25, 0.0, -1.0 / 48.0];
    let expected_polynomial = sigmoid_poly
        .iter()
        .enumerate()
        .map(|(power, coefficient)| coefficient * linear_score.powi(power as i32))
        .sum::<f64>();
    let expected_sigmoid = sigmoid(linear_score);

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
    let linear_prediction = halevi_shoup_dot_product(&context, &ciphertext, &weights, &rotation_keys);

    let relin_keys = generate_paterson_stockmeyer_relinearization_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        linear_prediction.level,
        sigmoid_poly.len(),
    );
    let logistic_prediction = paterson_stockmeyer_eval(
        &context,
        &linear_prediction,
        &sigmoid_poly,
        &relin_keys,
    );
    let recovered = context.decode_real_at_scale(
        &context.decrypt(&logistic_prediction, &key_pair.secret_key),
        logistic_prediction.scale_bits,
    );

    println!("Encrypted logistic regression inference with Paterson-Stockmeyer activation");
    println!("features (bias in final slot) = {:?}", features);
    println!("weights = {:?}", weights);
    println!("linear score = {linear_score}");
    println!("expected sigmoid(score) = {expected_sigmoid}");
    println!("expected polynomial approximation = {expected_polynomial}");
    println!("recovered slot[0] = {}", recovered[0]);
}
