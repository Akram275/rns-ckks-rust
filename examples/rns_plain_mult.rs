use ckks::rns::{RnsCkksContext, RnsCkksParams};



pub fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level()).expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);
    let mut enc_slots = vec![0.0_f64; context.num_slots()];
    enc_slots[0] = 0.125;
    enc_slots[1] = -0.4;
    enc_slots[2] = 0.03125;
    enc_slots[3] = -0.125;

    let mut plain_slots = vec![0.0_f64; context.num_slots()];
    plain_slots[0] = 2.0;
    plain_slots[1] = 2.0;
    plain_slots[2] = 2.0;
    plain_slots[3] = 2.0;
    
    let plain_encoded = context.encode_real(&plain_slots);
    let ciphertext = context.encrypt(&context.encode_real(&enc_slots), &key_pair.public_key);
    let product = context.mult_plain(&ciphertext, &plain_encoded);
    let recovered = context.decode_real_at_scale(
        &context.decrypt(&product, &key_pair.secret_key),
        product.scale_bits,
    );

    println!("RNS CKKS ciphertext-plaintext Hadamard product after relinearization");
    println!("remaining level = {}, scale_bits = {}", product.level, product.scale_bits);
    for index in 0..4 {
        println!(
            "slot[{}] expected = {:?}, output = {:?}",
            index,
            enc_slots[index] * plain_slots[index],
            recovered[index]
        );
    }
}    