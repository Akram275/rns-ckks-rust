use ckks::algorithms::paterson_stockmeyer::{
    generate_paterson_stockmeyer_relinearization_keys,
    paterson_stockmeyer_eval,
};
use ckks::rns::{RnsCkksContext, RnsCkksParams};

fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);

    let coefficients = [0.75_f64, -0.5, 1.25, -0.25, 0.125];
    let input = [0.125_f64, -0.2, 0.05, 0.3];
    let expected: Vec<f64> = input
        .iter()
        .map(|&value| {
            coefficients
                .iter()
                .enumerate()
                .map(|(power, &coeff)| coeff * value.powi(power as i32))
                .sum()
        })
        .collect();

    let mut padded = vec![0.0_f64; context.num_slots()];
    padded[..input.len()].copy_from_slice(&input);
    let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
    let relin_keys = generate_paterson_stockmeyer_relinearization_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        ciphertext.level,
        coefficients.len(),
    );

    let evaluated = paterson_stockmeyer_eval(&context, &ciphertext, &coefficients, &relin_keys);
    let recovered = context.decode_real_at_scale(
        &context.decrypt(&evaluated, &key_pair.secret_key),
        evaluated.scale_bits,
    );

    println!("RNS CKKS Paterson-Stockmeyer polynomial evaluation");
    println!("polynomial coefficients = {:?}", coefficients);
    println!("expected first slots = {:?}", expected);
    println!("recovered first slots = {:?}", &recovered[..input.len()]);
}