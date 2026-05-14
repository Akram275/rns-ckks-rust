//This is the implementation of the RNSCKKSParams structure object, and the definition of few param sets
//The validation funcions checks that each prime is ditinct, and that it is able to support the NTT for the given polynomial degree via the test q_i = 1 mod 2n.


use super::{RnsCkksParams, RnsContext};
use crate::rns::basis::{default_rns_primes, RnsPrime};

impl RnsCkksParams {
    pub fn toy() -> Self {
        Self {
            poly_degree: 8,
            primes: default_rns_primes(),
            aux_primes: Vec::new(),
            scale_bits: 30,
        }
    }

    pub fn small() -> Self {
        Self {
            poly_degree: 1024,
            primes: default_rns_primes(),
            aux_primes: Vec::new(),
            scale_bits: 40,
        }
    }

    pub fn realistic_8_level() -> Self {
        Self {
            poly_degree: 1 << 15,
            primes: vec![
                RnsPrime::new(1_125_899_908_022_273, 3),
                RnsPrime::new(1_125_899_908_612_097, 3),
                RnsPrime::new(1_125_899_909_398_529, 3),
                RnsPrime::new(1_125_899_910_316_033, 5),
                RnsPrime::new(1_125_899_911_168_001, 3),
                RnsPrime::new(1_125_899_911_364_609, 3),
                RnsPrime::new(1_125_899_913_527_297, 3),
                RnsPrime::new(1_125_899_914_510_337, 3),
            ],
            aux_primes: vec![
                RnsPrime::new(562_949_954_142_209, 3),
                RnsPrime::new(562_949_955_125_249, 3),
            ],
            scale_bits: 45,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.scale_bits == 0 {
            return Err("scale_bits must be greater than zero".to_string());
        }
        if self.scale_bits >= 60 {
            return Err("scale_bits must stay below 60 bits per rescale step in this RNS scaffold".to_string());
        }
        for aux_prime in &self.aux_primes {
            if self.primes.iter().any(|prime| prime.modulus == aux_prime.modulus) {
                return Err(format!(
                    "auxiliary eval-key modulus {} must be distinct from the ciphertext modulus chain",
                    aux_prime.modulus
                ));
            }
        }

        RnsContext::new(self.poly_degree, self.primes.clone()).map(|_| ())?;
        if !self.aux_primes.is_empty() {
            RnsContext::new(self.poly_degree, self.aux_primes.clone()).map(|_| ())?;
            let mut extended_primes = self.primes.clone();
            extended_primes.extend(self.aux_primes.clone());
            RnsContext::new(self.poly_degree, extended_primes).map(|_| ())?;
        }

        Ok(())
    }

    pub fn scale(&self) -> f64 {
        2f64.powi(self.scale_bits as i32)
    }
}