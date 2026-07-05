use num_complex::Complex64;
use std::f64::consts::PI;

use crate::math::canonical_embedding::{canonical_embedding_backward, canonical_embedding_forward};
use crate::rns::RnsCkksContext;

#[derive(Clone, Debug, PartialEq)]
pub struct DenseCoeffToSlotMatrices {
    pub upper_from_slots: Vec<Vec<Complex64>>,
    pub upper_from_conjugates: Vec<Vec<Complex64>>,
    pub lower_from_slots: Vec<Vec<Complex64>>,
    pub lower_from_conjugates: Vec<Vec<Complex64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseSlotToCoeffMatrices {
    pub slots_from_upper: Vec<Vec<Complex64>>,
    pub slots_from_lower: Vec<Vec<Complex64>>,
}

pub fn exact_coeff_to_slot_matrix(
    context: &RnsCkksContext,
    active_slots: usize,
) -> Result<Vec<Vec<Complex64>>, String> {
    validate_active_slots(context, active_slots)?;

    let matrix = (0..active_slots)
        .map(|column| {
            let mut basis = vec![Complex64::new(0.0, 0.0); active_slots];
            basis[column] = Complex64::new(1.0, 0.0);
            compressed_coeffs_to_slots(context, active_slots, &basis)
        })
        .collect::<Vec<_>>();

    Ok(transpose_columns(matrix))
}

pub fn exact_slot_to_coeff_matrix(
    context: &RnsCkksContext,
    active_slots: usize,
) -> Result<Vec<Vec<Complex64>>, String> {
    validate_active_slots(context, active_slots)?;

    let matrix = (0..active_slots)
        .map(|column| {
            let mut basis = vec![Complex64::new(0.0, 0.0); active_slots];
            basis[column] = Complex64::new(1.0, 0.0);
            slots_to_compressed_coeffs(context, active_slots, &basis)
        })
        .collect::<Vec<_>>();

    Ok(transpose_columns(matrix))
}

pub fn exact_dense_coeff_to_slot_matrices(
    context: &RnsCkksContext,
) -> Result<DenseCoeffToSlotMatrices, String> {
    let slot_count = context.num_slots();
    let mut upper_from_slots = zero_matrix(slot_count);
    let mut upper_from_conjugates = zero_matrix(slot_count);
    let mut lower_from_slots = zero_matrix(slot_count);
    let mut lower_from_conjugates = zero_matrix(slot_count);
    let imag_unit = Complex64::new(0.0, 1.0);

    for column in 0..slot_count {
        let mut real_basis = vec![Complex64::new(0.0, 0.0); slot_count];
        real_basis[column] = Complex64::new(1.0, 0.0);
        let (real_upper, real_lower) = dense_coeff_to_slot_halves(context, &real_basis);

        let mut imag_basis = vec![Complex64::new(0.0, 0.0); slot_count];
        imag_basis[column] = imag_unit;
        let (imag_upper, imag_lower) = dense_coeff_to_slot_halves(context, &imag_basis);

        for row in 0..slot_count {
            upper_from_slots[row][column] = (real_upper[row] - imag_unit * imag_upper[row]) * 0.5;
            upper_from_conjugates[row][column] = (real_upper[row] + imag_unit * imag_upper[row]) * 0.5;
            lower_from_slots[row][column] = (real_lower[row] - imag_unit * imag_lower[row]) * 0.5;
            lower_from_conjugates[row][column] = (real_lower[row] + imag_unit * imag_lower[row]) * 0.5;
        }
    }

    Ok(DenseCoeffToSlotMatrices {
        upper_from_slots,
        upper_from_conjugates,
        lower_from_slots,
        lower_from_conjugates,
    })
}

pub fn exact_dense_slot_to_coeff_matrices(
    context: &RnsCkksContext,
) -> Result<DenseSlotToCoeffMatrices, String> {
    let slot_count = context.num_slots();
    let poly_degree = context.params().poly_degree;
    let mut slots_from_upper = zero_matrix(slot_count);
    let mut slots_from_lower = zero_matrix(slot_count);

    for column in 0..slot_count {
        let mut upper_coeffs = vec![0.0_f64; poly_degree];
        upper_coeffs[column] = 1.0;
        let upper_slots = dense_real_coeffs_to_slots(context, &upper_coeffs);
        for row in 0..slot_count {
            slots_from_upper[row][column] = upper_slots[row];
        }

        let mut lower_coeffs = vec![0.0_f64; poly_degree];
        lower_coeffs[column + slot_count] = 1.0;
        let lower_slots = dense_real_coeffs_to_slots(context, &lower_coeffs);
        for row in 0..slot_count {
            slots_from_lower[row][column] = lower_slots[row];
        }
    }

    Ok(DenseSlotToCoeffMatrices {
        slots_from_upper,
        slots_from_lower,
    })
}

fn validate_active_slots(context: &RnsCkksContext, active_slots: usize) -> Result<(), String> {
    if active_slots == 0 {
        return Err("active_slots must be positive".to_string());
    }
    if !active_slots.is_power_of_two() {
        return Err("active_slots must be a power of two".to_string());
    }
    if active_slots > context.num_slots() {
        return Err(format!(
            "active_slots cannot exceed the CKKS slot count {}; got {active_slots}",
            context.num_slots()
        ));
    }
    Ok(())
}

fn compressed_coeffs_to_slots(
    context: &RnsCkksContext,
    active_slots: usize,
    compressed_coeffs: &[Complex64],
) -> Vec<Complex64> {
    let real_coeffs = expand_sparse_compressed_coeffs(context, active_slots, compressed_coeffs);
    dense_complex_coeffs_to_slots(context, &real_coeffs)
        .into_iter()
        .take(active_slots)
        .collect()
}

fn slots_to_compressed_coeffs(
    context: &RnsCkksContext,
    active_slots: usize,
    slots: &[Complex64],
) -> Vec<Complex64> {
    let full_slots = repeat_complex_block(slots, context.num_slots());
    let real_coeffs = dense_slots_to_real_coeffs(context, &full_slots);

    compress_sparse_real_coeffs(context, active_slots, &real_coeffs)
}

fn dense_coeff_to_slot_halves(
    context: &RnsCkksContext,
    slots: &[Complex64],
) -> (Vec<Complex64>, Vec<Complex64>) {
    let real_coeffs = dense_slots_to_real_coeffs(context, slots);
    let slot_count = context.num_slots();
    let upper = real_coeffs[..slot_count]
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect();
    let lower = real_coeffs[slot_count..]
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect();
    (upper, lower)
}

fn dense_slots_to_real_coeffs(
    context: &RnsCkksContext,
    slots: &[Complex64],
) -> Vec<f64> {
    assert_eq!(slots.len(), context.num_slots(), "slot vector must match the CKKS slot count");

    let mut evaluations = vec![Complex64::new(0.0, 0.0); context.params().poly_degree];
    let slot_exponents = slot_exponents(context.params().poly_degree);
    for (value, exponent) in slots.iter().zip(slot_exponents.into_iter()) {
        let index = exponent_to_eval_index(exponent);
        let conjugate_index = exponent_to_eval_index((2 * context.params().poly_degree - exponent) % (2 * context.params().poly_degree));
        evaluations[index] = *value;
        evaluations[conjugate_index] = value.conj();
    }

    let xi = primitive_2n_root(context.params().poly_degree);
    canonical_embedding_backward(&evaluations)
        .into_iter()
        .enumerate()
        .map(|(index, coeff)| (coeff * xi.powi(-(index as i32))).re)
        .collect()
}

fn dense_real_coeffs_to_slots(
    context: &RnsCkksContext,
    coeffs: &[f64],
) -> Vec<Complex64> {
    let complex_coeffs: Vec<Complex64> = coeffs.iter().map(|&value| Complex64::new(value, 0.0)).collect();
    dense_complex_coeffs_to_slots(context, &complex_coeffs)
}

fn dense_complex_coeffs_to_slots(
    context: &RnsCkksContext,
    coeffs: &[Complex64],
) -> Vec<Complex64> {
    assert_eq!(coeffs.len(), context.params().poly_degree, "coefficient vector must match the polynomial degree");

    let xi = primitive_2n_root(context.params().poly_degree);
    let twisted: Vec<Complex64> = coeffs
        .iter()
        .enumerate()
        .map(|(index, coeff)| *coeff * xi.powi(index as i32))
        .collect();
    let evaluations = canonical_embedding_forward(&twisted);
    let slot_exponents = slot_exponents(context.params().poly_degree);

    slot_exponents
        .into_iter()
        .map(|exponent| evaluations[exponent_to_eval_index(exponent)])
        .collect()
}

fn expand_sparse_compressed_coeffs(
    context: &RnsCkksContext,
    active_slots: usize,
    compressed_coeffs: &[Complex64],
) -> Vec<Complex64> {
    assert_eq!(compressed_coeffs.len(), active_slots, "compressed coefficient block must match active_slots");

    let stride = context.num_slots() / active_slots;
    let mut coeffs = vec![Complex64::new(0.0, 0.0); context.params().poly_degree];
    for (index, value) in compressed_coeffs.iter().enumerate() {
        coeffs[index * stride] = Complex64::new(value.re, 0.0);
        coeffs[(index + active_slots) * stride] = Complex64::new(value.im, 0.0);
    }
    coeffs
}

fn compress_sparse_real_coeffs(
    context: &RnsCkksContext,
    active_slots: usize,
    coeffs: &[f64],
) -> Vec<Complex64> {
    let stride = context.num_slots() / active_slots;
    (0..active_slots)
        .map(|index| Complex64::new(coeffs[index * stride], coeffs[(index + active_slots) * stride]))
        .collect()
}

fn repeat_complex_block(values: &[Complex64], slot_count: usize) -> Vec<Complex64> {
    assert!(!values.is_empty(), "input block cannot be empty");
    assert!(values.len() <= slot_count, "input block cannot exceed slot count");
    assert_eq!(slot_count % values.len(), 0, "slot count must be a multiple of the active block length");

    (0..slot_count)
        .map(|index| values[index % values.len()])
        .collect()
}

fn slot_exponents(poly_degree: usize) -> Vec<usize> {
    let modulus = 2 * poly_degree;
    let mut exponents = Vec::with_capacity(poly_degree / 2);
    let mut current = 1usize;

    for _ in 0..(poly_degree / 2) {
        exponents.push(current);
        current = (current * 5) % modulus;
    }

    exponents
}

fn exponent_to_eval_index(exponent: usize) -> usize {
    assert!(exponent % 2 == 1, "evaluation exponents must be odd");
    (exponent - 1) / 2
}

fn primitive_2n_root(poly_degree: usize) -> Complex64 {
    Complex64::from_polar(1.0, PI / poly_degree as f64)
}

fn zero_matrix(dimension: usize) -> Vec<Vec<Complex64>> {
    vec![vec![Complex64::new(0.0, 0.0); dimension]; dimension]
}

fn transpose_columns(columns: Vec<Vec<Complex64>>) -> Vec<Vec<Complex64>> {
    let dimension = columns.len();
    let mut rows = vec![vec![Complex64::new(0.0, 0.0); dimension]; dimension];
    for (column_index, column) in columns.into_iter().enumerate() {
        for (row_index, value) in column.into_iter().enumerate() {
            rows[row_index][column_index] = value;
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::{
        exact_coeff_to_slot_matrix,
        exact_dense_coeff_to_slot_matrices,
        exact_dense_slot_to_coeff_matrices,
        exact_slot_to_coeff_matrix,
    };
    use crate::ckks::bootstrapping::coeff_to_slot::{coeff_to_slot, CoeffToSlotPlan, CoeffToSlotPrecomputed};
    use crate::ckks::bootstrapping::linear_transform::generate_diagonal_transform_rotation_keys;
    use crate::ckks::bootstrapping::slot_to_coeff::{slot_to_coeff, SlotToCoeffPlan, SlotToCoeffPrecomputed};
    use crate::rns::{RnsCkksContext, RnsCkksParams, RnsPrime};
    use num_complex::Complex64;

    fn sample_dense_context() -> RnsCkksContext {
        RnsCkksContext::new(RnsCkksParams {
            poly_degree: 16,
            primes: vec![
                RnsPrime::new(998_244_353, 3),
                RnsPrime::new(1_004_535_809, 3),
                RnsPrime::new(469_762_049, 3),
            ],
            aux_primes: Vec::new(),
            scale_bits: 30,
        })
        .expect("sample dense CKKS context must build")
    }

    #[test]
    fn exact_sparse_transform_matrices_are_mutual_inverses() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let active_slots = 8;

        let coeff_to_slot = exact_coeff_to_slot_matrix(&context, active_slots)
            .expect("exact CoeffToSlot matrix must build");
        let slot_to_coeff = exact_slot_to_coeff_matrix(&context, active_slots)
            .expect("exact SlotToCoeff matrix must build");

        let basis = vec![
            Complex64::new(0.5, -0.125),
            Complex64::new(-0.25, 0.75),
            Complex64::new(0.125, 0.5),
            Complex64::new(0.875, -0.25),
            Complex64::new(-0.5, 0.125),
            Complex64::new(0.375, -0.5),
            Complex64::new(-0.125, 0.25),
            Complex64::new(0.625, 0.375),
        ];

        let slots = apply_matrix(&coeff_to_slot, &basis);
        let recovered = apply_matrix(&slot_to_coeff, &slots);

        for (index, (expected, actual)) in basis.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (*actual - *expected).norm() <= 1e-8,
                "expected {expected:?}, got {actual:?} at entry {index}"
            );
        }
    }

    #[test]
    fn exact_coeff_to_slot_executes_as_expected() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);
        let active_slots = 8;
        let input = vec![
            Complex64::new(0.5, -0.125),
            Complex64::new(-0.25, 0.75),
            Complex64::new(0.125, 0.5),
            Complex64::new(0.875, -0.25),
            Complex64::new(-0.5, 0.125),
            Complex64::new(0.375, -0.5),
            Complex64::new(-0.125, 0.25),
            Complex64::new(0.625, 0.375),
        ];
        let repeated = repeat_complex_slots(&input, context.num_slots());
        let ciphertext = context.encrypt(&context.encode(&repeated), &key_pair.public_key);
        let expected = apply_matrix(
            &exact_coeff_to_slot_matrix(&context, active_slots)
                .expect("exact CoeffToSlot matrix must build"),
            &input,
        );

        let coeff_to_slot_plan = CoeffToSlotPlan::from_ciphertext(&context, &ciphertext, active_slots)
            .expect("CoeffToSlot plan construction must succeed");
        let coeff_to_slot_precomputed = CoeffToSlotPrecomputed::exact(
            &context,
            coeff_to_slot_plan,
            ciphertext.scale_bits,
        )
        .expect("exact CoeffToSlot precomputation must succeed");
        let coeff_to_slot_rotation_keys = generate_diagonal_transform_rotation_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            active_slots,
        );
        let slots = coeff_to_slot(
            &context,
            &ciphertext,
            &coeff_to_slot_precomputed,
            &coeff_to_slot_rotation_keys,
        )
        .expect("CoeffToSlot execution must succeed");
        let recovered = context.decode_at_scale(
            &context.decrypt(&slots, &key_pair.secret_key),
            slots.scale_bits,
        );

        for (index, expected) in expected.iter().enumerate() {
            assert!(
                (recovered[index] - *expected).norm() <= 4e-2,
                "expected {expected:?}, got {:?} at slot {index}",
                recovered[index]
            );
        }
    }

    #[test]
    fn exact_slot_to_coeff_executes_as_expected() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);
        let active_slots = 8;
        let input = vec![
            Complex64::new(0.25, 0.5),
            Complex64::new(-0.75, 0.125),
            Complex64::new(0.625, -0.375),
            Complex64::new(-0.125, -0.25),
            Complex64::new(0.875, 0.25),
            Complex64::new(-0.5, 0.625),
            Complex64::new(0.125, -0.75),
            Complex64::new(0.375, 0.125),
        ];
        let repeated = repeat_complex_slots(&input, context.num_slots());
        let ciphertext = context.encrypt(&context.encode(&repeated), &key_pair.public_key);
        let expected = apply_matrix(
            &exact_slot_to_coeff_matrix(&context, active_slots)
                .expect("exact SlotToCoeff matrix must build"),
            &input,
        );

        let slot_to_coeff_plan = SlotToCoeffPlan::from_ciphertext(&context, &ciphertext, active_slots)
            .expect("SlotToCoeff plan construction must succeed");
        let slot_to_coeff_precomputed = SlotToCoeffPrecomputed::exact(
            &context,
            slot_to_coeff_plan,
            ciphertext.scale_bits,
        )
        .expect("exact SlotToCoeff precomputation must succeed");
        let slot_to_coeff_rotation_keys = generate_diagonal_transform_rotation_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            active_slots,
        );
        let transformed = slot_to_coeff(
            &context,
            &ciphertext,
            &slot_to_coeff_precomputed,
            &slot_to_coeff_rotation_keys,
        )
        .expect("SlotToCoeff execution must succeed");
        let recovered = context.decode_at_scale(
            &context.decrypt(&transformed, &key_pair.secret_key),
            transformed.scale_bits,
        );

        for (index, expected) in expected.iter().enumerate() {
            assert!(
                (recovered[index] - *expected).norm() <= 4e-2,
                "expected {expected:?}, got {:?} at slot {index}",
                recovered[index]
            );
        }
    }

    #[test]
    fn exact_dense_block_matrices_roundtrip_plaintext_slots() {
        let context = sample_dense_context();
        let coeff_to_slot = exact_dense_coeff_to_slot_matrices(&context)
            .expect("dense CoeffToSlot matrices must build");
        let slot_to_coeff = exact_dense_slot_to_coeff_matrices(&context)
            .expect("dense SlotToCoeff matrices must build");
        let slots = vec![
            Complex64::new(0.5, -0.125),
            Complex64::new(-0.25, 0.75),
            Complex64::new(0.125, 0.5),
            Complex64::new(0.875, -0.25),
            Complex64::new(-0.5, 0.125),
            Complex64::new(0.375, -0.5),
            Complex64::new(-0.125, 0.25),
            Complex64::new(0.625, 0.375),
        ];

        let upper = add_vectors(
            &apply_matrix(&coeff_to_slot.upper_from_slots, &slots),
            &apply_matrix(&coeff_to_slot.upper_from_conjugates, &conjugate_vector(&slots)),
        );
        let lower = add_vectors(
            &apply_matrix(&coeff_to_slot.lower_from_slots, &slots),
            &apply_matrix(&coeff_to_slot.lower_from_conjugates, &conjugate_vector(&slots)),
        );
        let recovered = add_vectors(
            &apply_matrix(&slot_to_coeff.slots_from_upper, &upper),
            &apply_matrix(&slot_to_coeff.slots_from_lower, &lower),
        );

        for (index, (expected, actual)) in slots.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (*actual - *expected).norm() <= 1e-8,
                "expected {expected:?}, got {actual:?} at slot {index}"
            );
        }
    }

    #[test]
    fn exact_dense_coeff_to_slot_blocks_match_direct_real_coefficients() {
        let context = sample_dense_context();
        let coeff_to_slot = exact_dense_coeff_to_slot_matrices(&context)
            .expect("dense CoeffToSlot matrices must build");
        let slots = vec![
            Complex64::new(0.25, 0.5),
            Complex64::new(-0.75, 0.125),
            Complex64::new(0.625, -0.375),
            Complex64::new(-0.125, -0.25),
            Complex64::new(0.875, 0.25),
            Complex64::new(-0.5, 0.625),
            Complex64::new(0.125, -0.75),
            Complex64::new(0.375, 0.125),
        ];

        let expected_coeffs = super::dense_slots_to_real_coeffs(&context, &slots);
        let upper = add_vectors(
            &apply_matrix(&coeff_to_slot.upper_from_slots, &slots),
            &apply_matrix(&coeff_to_slot.upper_from_conjugates, &conjugate_vector(&slots)),
        );
        let lower = add_vectors(
            &apply_matrix(&coeff_to_slot.lower_from_slots, &slots),
            &apply_matrix(&coeff_to_slot.lower_from_conjugates, &conjugate_vector(&slots)),
        );

        for index in 0..context.num_slots() {
            assert!((upper[index].re - expected_coeffs[index]).abs() <= 1e-8);
            assert!(upper[index].im.abs() <= 1e-8);
            assert!((lower[index].re - expected_coeffs[index + context.num_slots()]).abs() <= 1e-8);
            assert!(lower[index].im.abs() <= 1e-8);
        }
    }

    fn apply_matrix(matrix: &[Vec<Complex64>], vector: &[Complex64]) -> Vec<Complex64> {
        matrix
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(lhs, rhs)| *lhs * *rhs).sum())
            .collect()
    }

    fn add_vectors(lhs: &[Complex64], rhs: &[Complex64]) -> Vec<Complex64> {
        lhs.iter().zip(rhs.iter()).map(|(left, right)| *left + *right).collect()
    }

    fn conjugate_vector(values: &[Complex64]) -> Vec<Complex64> {
        values.iter().map(|value| value.conj()).collect()
    }

    fn repeat_complex_slots(values: &[Complex64], slot_count: usize) -> Vec<Complex64> {
        (0..slot_count)
            .map(|index| values[index % values.len()])
            .collect()
    }
}