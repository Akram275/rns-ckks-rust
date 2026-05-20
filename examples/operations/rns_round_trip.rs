use ckks::rns::{RnsCkksContext, RnsCkksParams};
use num_complex::Complex64;

fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);

    let mut slots = vec![Complex64::new(0.0, 0.0); context.num_slots()];
    slots[0] = Complex64::new(0.125, -0.25);
    slots[1] = Complex64::new(-0.0625, 0.1875);
    slots[2] = Complex64::new(0.03125, 0.09375);
    slots[3] = Complex64::new(-0.125, -0.0625);

    let plaintext = context.encode(&slots);
    let ciphertext = context.encrypt(&plaintext, &key_pair.public_key);
    let decrypted = context.decrypt(&ciphertext, &key_pair.secret_key);
    let recovered = context.decode(&decrypted);

    println!(
        "RNS CKKS round trip with N = {} and log2(Q) ~= {}",
        context.params().poly_degree,
        context.total_modulus_bits()
    );
    for index in 0..4 {
        println!(
            "slot[{index}] input = {:?}, output = {:?}",
            slots[index], recovered[index]
        );
    }
}