// Canonical embedding for polynomials via evaluation/interpolation on roots of unity.
//
// This implements the dense O(n^2) DFT/IDFT-style maps used as the mathematical
// foundation for CKKS encoding. Later encoding will scale and round these values.

use num_complex::Complex64;
use std::f64::consts::PI;

/// Return a primitive n-th root of unity: exp(2*pi*i/n).
pub fn primitive_nth_root_of_unity(n: usize) -> Complex64 {
    assert!(n > 0, "n must be positive");
    let theta = 2.0 * PI / n as f64;
    Complex64::from_polar(1.0, theta)
}

/// Forward canonical embedding: evaluate a polynomial at n roots of unity.
///
/// Given coefficients a_0..a_{n-1}, this returns values:
/// y_k = sum_{j=0}^{n-1} a_j * w^{j*k}, for k in 0..n-1,
/// where w = exp(2*pi*i/n).
pub fn canonical_embedding_forward(coeffs: &[Complex64]) -> Vec<Complex64> {
    let n = coeffs.len();
    assert!(n > 0, "coeff vector must be non-empty");

    let mut values = coeffs.to_vec();
    fft_in_place(&mut values, false);
    values
}

// Backward canonical embedding (interpolation): recover coefficients of a 
// polynomial whose evaluation at the n primitive 2n roots for unity gives the input complex values.
//
// This is the inverse map of `canonical_embedding_forward`, computed as:
// a_j = (1/n) * sum_{k=0}^{n-1} y_k * w^{-j*k}.
pub fn canonical_embedding_backward(values: &[Complex64]) -> Vec<Complex64> {
    let n = values.len();
    assert!(n > 0, "value vector must be non-empty");

    let mut coeffs = values.to_vec();
    fft_in_place(&mut coeffs, true);
    coeffs
}

fn fft_in_place(values: &mut [Complex64], inverse: bool) {
    let n = values.len();
    assert!(n.is_power_of_two(), "FFT size must be a power of two");

    bit_reverse_permute(values);

    let mut len = 2;
    while len <= n {
        let angle = if inverse {
            -2.0 * PI / len as f64
        } else {
            2.0 * PI / len as f64
        };
        let w_len = Complex64::from_polar(1.0, angle);

        for chunk_start in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for i in 0..(len / 2) {
                let u = values[chunk_start + i];
                let v = values[chunk_start + i + len / 2] * w;
                values[chunk_start + i] = u + v;
                values[chunk_start + i + len / 2] = u - v;
                w *= w_len;
            }
        }

        len *= 2;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for value in values.iter_mut() {
            *value *= scale;
        }
    }
}

fn bit_reverse_permute(values: &mut [Complex64]) {
    let n = values.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{canonical_embedding_backward, canonical_embedding_forward};
    use num_complex::Complex64;

    fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
        (a - b).norm() <= eps
    }

    #[test]
    fn forward_backward_round_trip_recovers_coeffs() {
        let coeffs = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-0.5, 0.75),
            Complex64::new(3.25, 1.5),
            Complex64::new(-2.0, 0.0),
            Complex64::new(0.0, -4.0),
            Complex64::new(7.5, 2.5),
            Complex64::new(-1.25, 0.125),
        ];

        let values = canonical_embedding_forward(&coeffs);
        let recovered = canonical_embedding_backward(&values);

        for (&expected, &actual) in coeffs.iter().zip(recovered.iter()) {
            assert!(approx_eq(expected, actual, 1e-9));
        }
    }

    #[test]
    fn constant_polynomial_maps_to_constant_values() {
        let coeffs = vec![Complex64::new(5.0, -2.0); 8];
        let values = canonical_embedding_forward(&coeffs);

        // For p(x) = c + c x + ... + c x^(n-1):
        // p(1) = n*c, and p(w^k)=0 for k != 0.
        assert!(approx_eq(values[0], Complex64::new(40.0, -16.0), 1e-9));
        for value in values.iter().skip(1) {
            assert!(approx_eq(*value, Complex64::new(0.0, 0.0), 1e-9));
        }
    }
}
