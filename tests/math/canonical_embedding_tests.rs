use ckks::math::canonical_embedding::{canonical_embedding_backward, canonical_embedding_forward};
use num_complex::Complex64;

fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
    (a - b).norm() <= eps
}

#[test]
fn canonical_embedding_round_trip() {
    let coeffs = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(-2.0, 0.5),
        Complex64::new(0.25, -1.25),
        Complex64::new(3.0, 1.0),
        Complex64::new(-4.5, 2.5),
        Complex64::new(6.0, 0.0),
        Complex64::new(1.5, -3.5),
        Complex64::new(-0.75, 0.125),
    ];

    let values = canonical_embedding_forward(&coeffs);
    let recovered = canonical_embedding_backward(&values);

    for (&expected, &actual) in coeffs.iter().zip(recovered.iter()) {
        assert!(approx_eq(expected, actual, 1e-9));
    }
}

#[test]
fn backward_of_forward_matches_original_for_real_coeffs() {
    let coeffs: Vec<Complex64> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        .into_iter()
        .map(|x| Complex64::new(x, 0.0))
        .collect();

    let recovered = canonical_embedding_backward(&canonical_embedding_forward(&coeffs));

    for (&expected, &actual) in coeffs.iter().zip(recovered.iter()) {
        assert!(approx_eq(expected, actual, 1e-9));
    }
}
