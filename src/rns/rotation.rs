use super::ciphertexts::{RnsCiphertext, RnsQuadraticCiphertext};
use super::keys::RnsSecretKey;
use super::keyswitching::RnsKeySwitchingKey;
use super::polynomial::RnsPolynomial;

pub const GALOIS_GENERATOR: usize = 5;

#[derive(Clone, Debug)]
pub struct RnsRotationKey {
    pub galois_element: usize,
    pub keyswitch_key: RnsKeySwitchingKey,
}

pub fn galois_element_for_rotation(poly_degree: usize, steps: isize) -> usize {
    let modulus = 2 * poly_degree;
    let generator = GALOIS_GENERATOR % modulus;
    assert_eq!(generator % 2, 1, "Galois generator must be odd modulo 2N");

    if steps >= 0 {
        pow_mod_usize(generator, steps as usize, modulus)
    } else {
        let generator_inv = mod_inv_usize(generator, modulus)
            .expect("Galois generator must be invertible modulo 2N");
        pow_mod_usize(generator_inv, (-steps) as usize, modulus)
    }
}

impl RnsPolynomial {
    pub fn rotate_raw(&self, poly_degree: usize, steps: isize) -> Self {
        self.apply_galois_automorphism(galois_element_for_rotation(poly_degree, steps))
    }
}

impl RnsSecretKey {
    pub fn apply_galois_automorphism(&self, galois_element: usize) -> Self {
        Self::new(self.poly.apply_galois_automorphism(galois_element))
    }

    pub fn rotate_raw(&self, poly_degree: usize, steps: isize) -> Self {
        self.apply_galois_automorphism(galois_element_for_rotation(poly_degree, steps))
    }
}

impl RnsCiphertext {
    pub fn rotate_raw(&self, poly_degree: usize, steps: isize) -> Self {
        self.apply_galois_automorphism(galois_element_for_rotation(poly_degree, steps))
    }
}

impl RnsQuadraticCiphertext {
    pub fn rotate_raw(&self, poly_degree: usize, steps: isize) -> Self {
        self.apply_galois_automorphism(galois_element_for_rotation(poly_degree, steps))
    }
}

fn pow_mod_usize(base: usize, exponent: usize, modulus: usize) -> usize {
    let mut result = 1usize;
    let mut base_power = base % modulus;
    let mut exp = exponent;

    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base_power as u128) % modulus as u128) as usize;
        }
        base_power = ((base_power as u128 * base_power as u128) % modulus as u128) as usize;
        exp >>= 1;
    }

    result
}

fn mod_inv_usize(value: usize, modulus: usize) -> Option<usize> {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (modulus as i128, value as i128);

    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }

    if r != 1 {
        return None;
    }

    if t < 0 {
        t += modulus as i128;
    }

    Some((t % modulus as i128) as usize)
}