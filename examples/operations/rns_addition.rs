use ckks::rns::{RnsCkksContext, RnsCkksParams};
use num_complex::Complex64;

fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);

    let mut lhs_slots = vec![Complex64::new(0.0, 0.0); context.num_slots()];
    let mut rhs_slots = vec![Complex64::new(0.0, 0.0); context.num_slots()];
    lhs_slots[0] = Complex64::new(0.125, -0.25);
    lhs_slots[1] = Complex64::new(-0.0625, 0.1875);
    lhs_slots[2] = Complex64::new(0.03125, 0.09375);
    lhs_slots[3] = Complex64::new(-0.125, -0.0625);
    rhs_slots[0] = Complex64::new(0.5, 0.25);
    rhs_slots[1] = Complex64::new(-0.25, 0.125);
    rhs_slots[2] = Complex64::new(0.125, -0.25);
    rhs_slots[3] = Complex64::new(-0.5, 0.0);

    let lhs = context.encrypt(&context.encode(&lhs_slots), &key_pair.public_key);
    let rhs = context.encrypt(&context.encode(&rhs_slots), &key_pair.public_key);
    let sum = lhs.add(&rhs);
    let recovered = context.decode_at_scale(
        &context.decrypt(&sum, &key_pair.secret_key),
        sum.scale_bits,
    );

    println!("RNS CKKS ciphertext addition at level {}", sum.level);
    println!("scale_bits = {}", sum.scale_bits);
    for index in 0..4 {
        println!(
            "slot[{index}] expected = {:?}, output = {:?}",
            lhs_slots[index] + rhs_slots[index],
            recovered[index]
        );
    }
}