use ckks::math::ntt::NttPlan;
use ckks::poly::ModularPolynomial;

#[test]
fn modular_polynomial_reduces_coefficients() {
    let poly = ModularPolynomial::new(vec![17, 34, 35, 99], 17);
    assert_eq!(poly.coeffs(), &[0, 0, 1, 14]);
}

#[test]
fn modular_polynomial_add_sub_consistency() {
    let modulus = 97;
    let a = ModularPolynomial::new(vec![96, 10, 5, 42, 0, 1, 2, 3], modulus);
    let b = ModularPolynomial::new(vec![5, 90, 90, 70, 1, 2, 3, 4], modulus);

    let sum = a.add(&b);
    let back = sum.sub(&b);

    assert_eq!(back, a);
}

#[test]
fn ntt_multiplication_matches_naive() {
    let modulus = 998_244_353;
    let plan = NttPlan::new(8, modulus, 3).expect("valid NTT plan");

    let lhs = ModularPolynomial::new(vec![11, 22, 33, 44, 55, 66, 77, 88], modulus);
    let rhs = ModularPolynomial::new(vec![3, 1, 4, 1, 5, 9, 2, 6], modulus);

    let expected = lhs.multiply_naive(&rhs);
    let actual = lhs.multiply_ntt(&rhs, &plan);

    assert_eq!(actual, expected);
}
