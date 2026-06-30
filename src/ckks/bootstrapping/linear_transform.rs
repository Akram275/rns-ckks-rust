use std::collections::BTreeMap;

use num_complex::Complex64;

use crate::rns::{RnsCkksContext, RnsCiphertext, RnsPolynomial, RnsPublicKey, RnsRotationKey, RnsSecretKey};

#[derive(Clone, Debug)]
pub struct DiagonalTransformRotationKeys {
    pub by_step: BTreeMap<usize, RnsRotationKey>,
}

#[derive(Clone, Debug)]
pub struct DiagonalTransformPlaintexts {
    pub diagonals: Vec<RnsPolynomial>,
    pub dimension: usize,
    level: usize,
    scale_bits: usize,
}

pub fn diagonal_transform_rotation_steps(dimension: usize) -> Vec<usize> {
    assert!(dimension > 0, "transform dimension must be positive");
    (1..dimension).collect()
}

pub fn generate_diagonal_transform_rotation_keys(
    context: &RnsCkksContext,
    secret_key: &RnsSecretKey,
    public_key: &RnsPublicKey,
    ciphertext_level: usize,
    dimension: usize,
) -> DiagonalTransformRotationKeys {
    let mut by_step = BTreeMap::new();
    for step in diagonal_transform_rotation_steps(dimension) {
        by_step.insert(
            step,
            context.generate_rotation_key_at_level(secret_key, public_key, step as isize, ciphertext_level),
        );
    }

    DiagonalTransformRotationKeys { by_step }
}

pub fn repeat_block_slots(values: &[f64], slot_count: usize) -> Vec<f64> {
    assert!(!values.is_empty(), "input block cannot be empty");
    assert!(values.len() <= slot_count, "input block cannot exceed slot count");
    assert_eq!(slot_count % values.len(), 0, "slot count must be a multiple of the active block length");

    (0..slot_count)
        .map(|index| values[index % values.len()])
        .collect()
}

pub fn pack_diagonal_transform_plaintexts(
    context: &RnsCkksContext,
    matrix: &[Vec<f64>],
    level: usize,
    scale_bits: usize,
) -> DiagonalTransformPlaintexts {
    let dimension = validate_square_matrix(matrix, context.num_slots());
    let diagonals = (0..dimension)
        .map(|step| {
            let slots = packed_diagonal_slots(matrix, step, context.num_slots());
            let complex_slots: Vec<Complex64> = slots
                .into_iter()
                .map(|value| Complex64::new(value, 0.0))
                .collect();
            context.encode_at_level_and_scale(&complex_slots, level, scale_bits)
        })
        .collect();

    DiagonalTransformPlaintexts {
        diagonals,
        dimension,
        level,
        scale_bits,
    }
}

pub fn apply_diagonal_linear_transform(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    diagonals: &DiagonalTransformPlaintexts,
    rotation_keys: &DiagonalTransformRotationKeys,
) -> RnsCiphertext {
    assert_eq!(diagonals.level, ciphertext.level, "packed diagonals must match the ciphertext level");
    assert_eq!(diagonals.scale_bits, ciphertext.scale_bits, "packed diagonals must match the ciphertext scale");

    let initial = context.mult_plain(ciphertext, &diagonals.diagonals[0]);
    let mut accumulator = context.rescale(&initial);

    for step in diagonal_transform_rotation_steps(diagonals.dimension) {
        let rotation_key = rotation_keys
            .by_step
            .get(&step)
            .unwrap_or_else(|| panic!("missing input rotation key for step {step}"));
        let rotated = context.rotate(ciphertext, rotation_key);
        let term = context.mult_plain(&rotated, &diagonals.diagonals[step]);
        let term = context.rescale(&term);
        accumulator = accumulator.add(&term);
    }

    accumulator
}

fn validate_square_matrix(matrix: &[Vec<f64>], slot_count: usize) -> usize {
    assert!(!matrix.is_empty(), "matrix cannot be empty");
    let dimension = matrix.len();
    assert!(dimension.is_power_of_two(), "matrix dimension must be a power of two");
    assert!(dimension <= slot_count, "matrix dimension cannot exceed slot count");
    for row in matrix {
        assert_eq!(row.len(), dimension, "current diagonal transform implementation expects a square matrix");
    }
    dimension
}

fn packed_diagonal_slots(matrix: &[Vec<f64>], diagonal: usize, slot_count: usize) -> Vec<f64> {
    let dimension = matrix.len();
    let mut slots = vec![0.0; slot_count];
    for row in 0..dimension {
        slots[row] = matrix[row][(diagonal + row) % dimension];
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::{
        apply_diagonal_linear_transform,
        diagonal_transform_rotation_steps,
        generate_diagonal_transform_rotation_keys,
        pack_diagonal_transform_plaintexts,
        repeat_block_slots,
    };
    use crate::rns::{RnsCkksContext, RnsCkksParams};

    #[test]
    fn diagonal_transform_rotation_steps_cover_nonzero_diagonals() {
        assert_eq!(diagonal_transform_rotation_steps(1), Vec::<usize>::new());
        assert_eq!(diagonal_transform_rotation_steps(4), vec![1, 2, 3]);
    }

    #[test]
    fn diagonal_transform_matches_expected_on_realistic_params() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);
        let matrix = vec![
            vec![1.0, 0.5, -0.25, 0.0],
            vec![0.0, -1.0, 0.75, 0.25],
            vec![0.5, 0.0, 1.25, -0.5],
            vec![-0.25, 0.125, 0.0, 0.75],
        ];
        let vector = vec![0.5, -0.25, 0.125, 0.75];
        let expected: Vec<f64> = matrix
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(lhs, rhs)| lhs * rhs).sum())
            .collect();

        let packed_vector = repeat_block_slots(&vector, context.num_slots());
        let ciphertext = context.encrypt(&context.encode_real(&packed_vector), &key_pair.public_key);
        let rotation_keys = generate_diagonal_transform_rotation_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            matrix.len(),
        );
        let diagonals = pack_diagonal_transform_plaintexts(&context, &matrix, ciphertext.level, ciphertext.scale_bits);

        let product = apply_diagonal_linear_transform(
            &context,
            &ciphertext,
            &diagonals,
            &rotation_keys,
        );
        let recovered = context.decode_real_at_scale(
            &context.decrypt(&product, &key_pair.secret_key),
            product.scale_bits,
        );

        for (index, &expected_value) in expected.iter().enumerate() {
            assert!(
                (recovered[index] - expected_value).abs() <= 2e-2,
                "expected {expected_value}, got {} at slot {index}",
                recovered[index]
            );
            assert!(
                recovered[index + matrix.len()].abs() <= 2e-2,
                "expected zero-padded tail, got {} at slot {}",
                recovered[index + matrix.len()],
                index + matrix.len()
            );
        }
    }
}