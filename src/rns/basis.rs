use crate::math::ntt::NttPlan;
use num_bigint::{BigInt, BigUint};
use num_traits::One;

use super::polynomial::RnsPolynomial;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RnsPrime {
    pub modulus: u64,
    pub primitive_root: u64,
}

impl RnsPrime {
    pub fn new(modulus: u64, primitive_root: u64) -> Self {
        assert!(modulus > 1, "modulus must be greater than 1");
        assert!(primitive_root > 1, "primitive_root must be greater than 1");
        Self {
            modulus,
            primitive_root,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RnsContext {
    poly_degree: usize,
    primes: Vec<RnsPrime>,
    plans: Vec<NttPlan>,
}

impl RnsContext {
    pub fn new(poly_degree: usize, primes: Vec<RnsPrime>) -> Result<Self, String> {
        if poly_degree == 0 || !poly_degree.is_power_of_two() {
            return Err("poly_degree must be a non-zero power of two".to_string());
        }
        if primes.is_empty() {
            return Err("RNS basis must contain at least one prime".to_string());
        }

        for (index, prime) in primes.iter().enumerate() {
            if primes.iter().take(index).any(|other| other.modulus == prime.modulus) {
                return Err(format!("duplicate RNS modulus {}", prime.modulus));
            }
        }

        let mut plans = Vec::with_capacity(primes.len());
        for prime in &primes {
            plans.push(
                NttPlan::new(poly_degree, prime.modulus, prime.primitive_root)
                    .map_err(|err| format!("invalid RNS prime {}: {err}", prime.modulus))?,
            );
        }

        Ok(Self {
            poly_degree,
            primes,
            plans,
        })
    }

    pub fn poly_degree(&self) -> usize {
        self.poly_degree
    }

    pub fn levels(&self) -> usize {
        self.primes.len()
    }

    pub fn primes(&self) -> &[RnsPrime] {
        &self.primes
    }

    pub fn ntt_plans(&self) -> &[NttPlan] {
        &self.plans
    }

    pub fn total_modulus(&self) -> BigUint {
        self.primes
            .iter()
            .fold(BigUint::one(), |acc, prime| acc * BigUint::from(prime.modulus))
    }

    pub fn total_modulus_bits(&self) -> u64 {
        self.total_modulus().bits()
    }

    pub fn at_level(&self, level_count: usize) -> Result<Self, String> {
        if level_count == 0 || level_count > self.levels() {
            return Err(format!(
                "level_count must be in 1..={}, got {level_count}",
                self.levels()
            ));
        }
        Self::new(self.poly_degree, self.primes[..level_count].to_vec())
    }

    pub fn zero_poly(&self) -> RnsPolynomial {
        RnsPolynomial::zero(self)
    }

    pub fn poly_from_coeffs(&self, coeffs: &[u64]) -> RnsPolynomial {
        RnsPolynomial::from_coeffs(self, coeffs)
    }

    pub fn poly_from_biguint_coeffs(&self, coeffs: &[BigUint]) -> RnsPolynomial {
        RnsPolynomial::from_biguint_coeffs(self, coeffs)
    }

    pub fn poly_from_bigint_coeffs(&self, coeffs: &[BigInt]) -> RnsPolynomial {
        RnsPolynomial::from_bigint_coeffs(self, coeffs)
    }
}

pub(crate) fn mod_inv_u64(value: u64, modulus: u64) -> Option<u64> {
    if value == 0 {
        return None;
    }

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

    Some((t % modulus as i128) as u64)
}

pub(crate) fn default_rns_primes() -> Vec<RnsPrime> {
    vec![
        RnsPrime::new(998_244_353, 3),
        RnsPrime::new(1_004_535_809, 3),
        RnsPrime::new(469_762_049, 3),
        RnsPrime::new(754_974_721, 11),
        RnsPrime::new(167_772_161, 3),
        RnsPrime::new(1_224_736_769, 3),
    ]
}