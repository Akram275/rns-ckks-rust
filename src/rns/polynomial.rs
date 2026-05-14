use crate::poly::ModularPolynomial;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{ToPrimitive, Zero};

use super::basis::{mod_inv_u64, RnsContext};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RnsPolynomial {
    residues: Vec<ModularPolynomial>,
}

impl RnsPolynomial {
    pub fn new(context: &RnsContext, residues: Vec<ModularPolynomial>) -> Self {
        assert_eq!(
            residues.len(),
            context.levels(),
            "residue count must match the RNS basis length"
        );

        for (residue, prime) in residues.iter().zip(context.primes()) {
            assert_eq!(
                residue.degree(),
                context.poly_degree(),
                "all residues must have the context polynomial degree"
            );
            assert_eq!(
                residue.modulus(),
                prime.modulus,
                "residue modulus must match the corresponding RNS prime"
            );
        }

        Self { residues }
    }

    pub fn zero(context: &RnsContext) -> Self {
        let residues = context
            .primes()
            .iter()
            .map(|prime| ModularPolynomial::zero(context.poly_degree(), prime.modulus))
            .collect();
        Self { residues }
    }

    pub fn from_coeffs(context: &RnsContext, coeffs: &[u64]) -> Self {
        let mut padded = coeffs.to_vec();
        padded.resize(context.poly_degree(), 0);
        let residues = context
            .primes()
            .iter()
            .map(|prime| ModularPolynomial::new(padded.clone(), prime.modulus))
            .collect();
        Self { residues }
    }

    pub fn from_biguint_coeffs(context: &RnsContext, coeffs: &[BigUint]) -> Self {
        let mut padded = coeffs.to_vec();
        padded.resize(context.poly_degree(), BigUint::zero());
        let residues = context
            .primes()
            .iter()
            .map(|prime| {
                let reduced: Vec<u64> = padded
                    .iter()
                    .map(|value| {
                        (value % BigUint::from(prime.modulus))
                            .to_u64()
                            .expect("reduced coefficient must fit in u64")
                    })
                    .collect();
                ModularPolynomial::new(reduced, prime.modulus)
            })
            .collect();
        Self { residues }
    }

    pub fn from_bigint_coeffs(context: &RnsContext, coeffs: &[BigInt]) -> Self {
        let modulus = context.total_modulus();
        let coeffs_mod_q: Vec<BigUint> = coeffs
            .iter()
            .map(|value| centered_bigint_to_mod_q(value, &modulus))
            .collect();
        Self::from_biguint_coeffs(context, &coeffs_mod_q)
    }

    pub fn levels(&self) -> usize {
        self.residues.len()
    }

    pub fn degree(&self) -> usize {
        self.residues.first().map(ModularPolynomial::degree).unwrap_or(0)
    }

    pub fn residues(&self) -> &[ModularPolynomial] {
        &self.residues
    }

    pub fn add(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let residues = self
            .residues
            .iter()
            .zip(other.residues.iter())
            .map(|(lhs, rhs)| lhs.add(rhs))
            .collect();
        Self { residues }
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let residues = self
            .residues
            .iter()
            .zip(other.residues.iter())
            .map(|(lhs, rhs)| lhs.sub(rhs))
            .collect();
        Self { residues }
    }

    pub fn multiply_ntt(&self, other: &Self, context: &RnsContext) -> Self {
        self.assert_compatible(other);
        assert_eq!(
            self.levels(),
            context.levels(),
            "context level count must match polynomial levels"
        );
        let residues = self
            .residues
            .iter()
            .zip(other.residues.iter())
            .zip(context.ntt_plans().iter())
            .map(|((lhs, rhs), plan)| lhs.multiply_ntt(rhs, plan))
            .collect();
        Self { residues }
    }

    pub fn apply_galois_automorphism(&self, galois_element: usize) -> Self {
        let residues = self
            .residues
            .iter()
            .map(|residue| residue.apply_galois_automorphism(galois_element))
            .collect();
        Self { residues }
    }

    pub fn drop_last_level(&self) -> Self {
        assert!(self.levels() > 1, "cannot drop the last RNS level");
        let mut residues = self.residues.clone();
        residues.pop();
        Self { residues }
    }

    pub fn truncate_levels(&self, level_count: usize) -> Self {
        assert!(level_count > 0, "level_count must be positive");
        assert!(level_count <= self.levels(), "level_count cannot exceed polynomial levels");
        Self {
            residues: self.residues[..level_count].to_vec(),
        }
    }

    pub fn reconstruct_coefficients(&self, context: &RnsContext) -> Vec<BigUint> {
        assert_eq!(
            self.levels(),
            context.levels(),
            "context level count must match polynomial levels"
        );

        let total_modulus = context.total_modulus();
        let mut coeffs = Vec::with_capacity(context.poly_degree());

        for coeff_index in 0..context.poly_degree() {
            let mut acc = BigUint::zero();
            for (residue, prime) in self.residues.iter().zip(context.primes()) {
                let modulus_big = BigUint::from(prime.modulus);
                let partial = &total_modulus / &modulus_big;
                let partial_mod = (&partial % &modulus_big)
                    .to_u64()
                    .expect("partial modulus reduction must fit in u64");
                let inverse = mod_inv_u64(partial_mod, prime.modulus)
                    .expect("RNS moduli must be pairwise coprime");
                let term = BigUint::from(residue.coeffs()[coeff_index])
                    * &partial
                    * BigUint::from(inverse);
                acc += term;
            }
            coeffs.push(acc % &total_modulus);
        }

        coeffs
    }

    fn assert_compatible(&self, other: &Self) {
        assert_eq!(
            self.levels(),
            other.levels(),
            "RNS polynomials must have the same number of levels"
        );
        assert_eq!(
            self.degree(),
            other.degree(),
            "RNS polynomials must have the same degree"
        );

        for (lhs, rhs) in self.residues.iter().zip(other.residues.iter()) {
            assert_eq!(lhs.modulus(), rhs.modulus(), "RNS residues must align by modulus");
        }
    }
}

pub(crate) fn centered_bigint_to_mod_q(value: &BigInt, modulus: &BigUint) -> BigUint {
    let modulus_bigint = BigInt::from_biguint(Sign::Plus, modulus.clone());
    let reduced = ((value % &modulus_bigint) + &modulus_bigint) % &modulus_bigint;
    reduced.to_biguint().expect("reduced value must be non-negative")
}

pub(crate) fn mod_q_biguint_to_centered(value: &BigUint, modulus: &BigUint) -> BigInt {
    let half = modulus >> 1usize;
    let value_bigint = BigInt::from_biguint(Sign::Plus, value.clone());
    if value <= &half {
        value_bigint
    } else {
        value_bigint - BigInt::from_biguint(Sign::Plus, modulus.clone())
    }
}