// Modular polynomial arithmetic built on top of an NTT plan.

use crate::math::ntt::NttPlan;

/// A polynomial with coefficients reduced modulo a fixed integer modulus.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModularPolynomial {
    coeffs: Vec<u64>,
    modulus: u64,
}

impl ModularPolynomial {
    /// Create a polynomial from raw coefficients and reduce them modulo `modulus`.
    pub fn new(coeffs: Vec<u64>, modulus: u64) -> Self {
        assert!(modulus > 1, "modulus must be greater than 1");
        let coeffs = coeffs.into_iter().map(|value| value % modulus).collect();
        Self { coeffs, modulus }
    }

    /// Create a zero polynomial of a fixed degree.
    pub fn zero(degree: usize, modulus: u64) -> Self {
        Self::new(vec![0; degree], modulus)
    }

    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len()
    }

    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Coefficient-wise addition modulo `modulus`.
    pub fn add(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let coeffs = self
            .coeffs
            .iter()
            .zip(&other.coeffs)
            .map(|(&lhs, &rhs)| add_mod(lhs, rhs, self.modulus))
            .collect();
        Self::new(coeffs, self.modulus)
    }

    /// Coefficient-wise subtraction modulo `modulus`.
    pub fn sub(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let coeffs = self
            .coeffs
            .iter()
            .zip(&other.coeffs)
            .map(|(&lhs, &rhs)| sub_mod(lhs, rhs, self.modulus))
            .collect();
        Self::new(coeffs, self.modulus)
    }

    /// Naive negacyclic multiplication modulo x^n + 1 and coefficient modulus.
    pub fn multiply_naive(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let n = self.degree();
        let mut result = vec![0u64; n];

        let lhs_nonzero: Vec<(usize, u64)> = self
            .coeffs
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, coeff)| coeff != 0)
            .collect();
        let rhs_nonzero: Vec<(usize, u64)> = other
            .coeffs
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, coeff)| coeff != 0)
            .collect();

        if lhs_nonzero.is_empty() || rhs_nonzero.is_empty() {
            return Self::zero(n, self.modulus);
        }

        let (outer, inner) = if lhs_nonzero.len() <= rhs_nonzero.len() {
            (&lhs_nonzero, &rhs_nonzero)
        } else {
            (&rhs_nonzero, &lhs_nonzero)
        };

        for &(i, lhs_coeff) in outer {
            for &(j, rhs_coeff) in inner {
                let sum_idx = i + j;
                let product = mul_mod(lhs_coeff, rhs_coeff, self.modulus);
                if sum_idx < n {
                    result[sum_idx] = add_mod(result[sum_idx], product, self.modulus);
                } else {
                    // x^n = -1 in Z_q[X]/(X^n + 1)
                    let dst = sum_idx - n;
                    result[dst] = sub_mod(result[dst], product, self.modulus);
                }
            }
        }

        Self::new(result, self.modulus)
    }

    /// Apply the Galois automorphism sigma_m: X -> X^m in Z_q[X]/(X^n + 1).
    pub fn apply_galois_automorphism(&self, galois_element: usize) -> Self {
        assert!(galois_element % 2 == 1, "galois automorphisms require an odd exponent");
        let n = self.degree();
        let modulus_2n = 2 * n;
        let mut result = vec![0u64; n];

        for (index, &coeff) in self.coeffs.iter().enumerate() {
            if coeff == 0 {
                continue;
            }

            let mapped = (index * galois_element) % modulus_2n;
            if mapped < n {
                result[mapped] = add_mod(result[mapped], coeff, self.modulus);
            } else {
                result[mapped - n] = sub_mod(result[mapped - n], coeff, self.modulus);
            }
        }

        Self::new(result, self.modulus)
    }

    //This is the standard used polynomial multiplication in all the library
    //It uses the fundamental relation: a(x)b(x) = NTT^{-1}(NTT(a) pointwise_mult NTT(b))
    pub fn multiply_ntt(&self, other: &Self, plan: &NttPlan) -> Self {
        self.assert_compatible(other);
        assert_eq!(self.degree(), plan.size(), "NTT plan size must match polynomial degree");
        assert_eq!(self.modulus, plan.modulus(), "NTT plan modulus must match polynomial modulus");

        let psi = plan.negacyclic_twiddle();
        let psi_inv = plan.negacyclic_twiddle_inv();
        let modulus = self.modulus;
        let n = self.degree();

        let mut lhs = self.coeffs.clone();
        let mut rhs = other.coeffs.clone();
        twist_in_place(&mut lhs, psi, modulus);
        twist_in_place(&mut rhs, psi, modulus);

        plan.forward(&mut lhs);
        plan.forward(&mut rhs);

        for (lhs_coeff, rhs_coeff) in lhs.iter_mut().zip(rhs.iter()) {
            *lhs_coeff = mul_mod(*lhs_coeff, *rhs_coeff, modulus);
        }

        plan.inverse(&mut lhs);
        twist_in_place(&mut lhs, psi_inv, modulus);

        debug_assert_eq!(lhs.len(), n);
        Self::new(lhs, modulus)
    }

    fn assert_compatible(&self, other: &Self) {
        assert_eq!(self.modulus, other.modulus, "polynomials must share the same modulus");
        assert_eq!(self.degree(), other.degree(), "polynomials must have the same degree");
    }
}

fn twist_in_place(coeffs: &mut [u64], twiddle: u64, modulus: u64) {
    let mut power = 1u64;
    for coeff in coeffs.iter_mut() {
        *coeff = mul_mod(*coeff, power, modulus);
        power = mul_mod(power, twiddle, modulus);
    }
}

fn add_mod(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    let sum = lhs as u128 + rhs as u128;
    if sum >= modulus as u128 {
        (sum - modulus as u128) as u64
    } else {
        sum as u64
    }
}

fn sub_mod(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    if lhs >= rhs {
        lhs - rhs
    } else {
        modulus - (rhs - lhs)
    }
}

fn mul_mod(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((lhs as u128 * rhs as u128) % modulus as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::ModularPolynomial;
    use crate::math::ntt::NttPlan;

    #[test]
    fn modular_polynomial_addition_and_subtraction_work() {
        let modulus = 17;
        let lhs = ModularPolynomial::new(vec![14, 1, 9, 0], modulus);
        let rhs = ModularPolynomial::new(vec![5, 16, 12, 3], modulus);

        assert_eq!(lhs.add(&rhs).coeffs(), &[2, 0, 4, 3]);
        assert_eq!(lhs.sub(&rhs).coeffs(), &[9, 2, 14, 14]);
    }

    #[test]
    fn ntt_multiplication_matches_naive_negacyclic_multiplication() {
        let modulus = 998_244_353;
        let plan = NttPlan::new(8, modulus, 3).expect("valid NTT plan");
        let lhs = ModularPolynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8], modulus);
        let rhs = ModularPolynomial::new(vec![8, 7, 6, 5, 4, 3, 2, 1], modulus);

        let expected = ModularPolynomial::new(vec![998_244_193, 998_244_243, 998_244_297, 0, 56, 110, 160, 204], modulus);
        let actual = lhs.multiply_ntt(&rhs, &plan);

        assert_eq!(actual, expected);
    }

    #[test]
    fn galois_automorphism_matches_x_to_x_power_m() {
        let modulus = 17;
        let poly = ModularPolynomial::new(vec![1, 2, 3, 4], modulus);

        let automorphed = poly.apply_galois_automorphism(3);

        assert_eq!(automorphed.coeffs(), &[1, 13, 2, 14]);
    }
}