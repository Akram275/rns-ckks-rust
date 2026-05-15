//! Matrix-local packed-diagonal plaintext-matrix encrypted-vector product.
//!
//! Packing contract:
//! - the matrix is square of dimension `n`, with `n` a power of two,
//! - only the first `n` slots of each packed diagonal plaintext are used; the
//!   remaining CKKS slots are zero-padded,
//! - the encrypted input vector is repeated blockwise across the full slot
//!   count so that ordinary CKKS rotations realize cyclic shifts of the active
//!   `n`-slot block,
//! - the first `n` output slots contain the matrix-vector product and the tail
//!   is expected to stay near zero.

use std::collections::BTreeMap;

use num_complex::Complex64;

use crate::rns::{RnsCkksContext, RnsCiphertext, RnsPolynomial, RnsPublicKey, RnsRotationKey, RnsSecretKey};

#[derive(Clone, Debug)]
pub struct PackedDiagonalRotationKeys {
    pub by_step: BTreeMap<usize, RnsRotationKey>,
}

#[derive(Clone, Debug)]
pub struct PackedDiagonalPlaintexts {
    pub diagonals: Vec<RnsPolynomial>,
    pub dimension: usize,
    level: usize,
    scale_bits: usize,
}

pub fn packed_diagonal_rotation_steps(row_count: usize) -> Vec<usize> {
    assert!(row_count > 0, "matrix row count must be positive");
    (1..row_count).collect()
}

pub fn generate_packed_diagonal_rotation_keys(
    context: &RnsCkksContext,
    secret_key: &RnsSecretKey,
    public_key: &RnsPublicKey,
    ciphertext_level: usize,
    row_count: usize,
) -> PackedDiagonalRotationKeys {
    let mut by_step = BTreeMap::new();
    for step in packed_diagonal_rotation_steps(row_count) {
        by_step.insert(
            step,
            context.generate_rotation_key_at_level(secret_key, public_key, step as isize, ciphertext_level),
        );
    }

    PackedDiagonalRotationKeys { by_step }
}

pub fn repeat_real_vector_blocks(values: &[f64], slot_count: usize) -> Vec<f64> {
    assert!(!values.is_empty(), "input vector cannot be empty");
    assert!(values.len() <= slot_count, "input vector cannot exceed slot count");
    assert_eq!(slot_count % values.len(), 0, "slot count must be a multiple of the active vector length");

    (0..slot_count)
        .map(|index| values[index % values.len()])
        .collect()
}

pub fn pack_diagonal_plaintexts(
    context: &RnsCkksContext,
    matrix: &[Vec<f64>],
    level: usize,
    scale_bits: usize,
) -> PackedDiagonalPlaintexts {
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

    PackedDiagonalPlaintexts {
        diagonals,
        dimension,
        level,
        scale_bits,
    }
}

pub fn packed_diagonal_matvec_prepacked(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    diagonals: &PackedDiagonalPlaintexts,
    rotation_keys: &PackedDiagonalRotationKeys,
) -> RnsCiphertext {
    assert_eq!(diagonals.level, ciphertext.level, "packed diagonals must match the ciphertext level");
    assert_eq!(diagonals.scale_bits, ciphertext.scale_bits, "packed diagonals must match the ciphertext scale");

    let initial = context.mult_plain(ciphertext, &diagonals.diagonals[0]);
    let mut accumulator = context.rescale(&initial);

    for step in packed_diagonal_rotation_steps(diagonals.dimension) {
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

pub fn packed_diagonal_matvec(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    matrix: &[Vec<f64>],
    rotation_keys: &PackedDiagonalRotationKeys,
) -> RnsCiphertext {
    let diagonals = pack_diagonal_plaintexts(context, matrix, ciphertext.level, ciphertext.scale_bits);
    packed_diagonal_matvec_prepacked(context, ciphertext, &diagonals, rotation_keys)
}

fn validate_square_matrix(matrix: &[Vec<f64>], slot_count: usize) -> usize {
    assert!(!matrix.is_empty(), "matrix cannot be empty");
    let dimension = matrix.len();
    assert!(dimension.is_power_of_two(), "matrix dimension must be a power of two");
    assert!(dimension <= slot_count, "matrix dimension cannot exceed slot count");
    for row in matrix {
        assert_eq!(row.len(), dimension, "current packed diagonal implementation expects a square matrix");
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
        generate_packed_diagonal_rotation_keys,
        pack_diagonal_plaintexts,
        packed_diagonal_matvec,
        packed_diagonal_matvec_prepacked,
        packed_diagonal_rotation_steps,
        repeat_real_vector_blocks,
    };
    use crate::rns::{RnsCkksContext, RnsCkksParams};

    #[test]
    fn packed_diagonal_rotation_steps_cover_nonzero_diagonals() {
        assert_eq!(packed_diagonal_rotation_steps(1), Vec::<usize>::new());
        assert_eq!(packed_diagonal_rotation_steps(4), vec![1, 2, 3]);
    }

    #[test]
    fn packed_diagonal_matvec_matches_expected_on_realistic_params() {
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

        let packed_vector = repeat_real_vector_blocks(&vector, context.num_slots());
        let ciphertext = context.encrypt(&context.encode_real(&packed_vector), &key_pair.public_key);
        let rotation_keys = generate_packed_diagonal_rotation_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            matrix.len(),
        );
        let diagonals = pack_diagonal_plaintexts(&context, &matrix, ciphertext.level, ciphertext.scale_bits);

        let product = packed_diagonal_matvec_prepacked(
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

        let product_from_matrix = packed_diagonal_matvec(&context, &ciphertext, &matrix, &rotation_keys);
        let recovered_from_matrix = context.decode_real_at_scale(
            &context.decrypt(&product_from_matrix, &key_pair.secret_key),
            product_from_matrix.scale_bits,
        );
        for (index, &expected_value) in expected.iter().enumerate() {
            assert!(
                (recovered_from_matrix[index] - expected_value).abs() <= 2e-2,
                "matrix API expected {expected_value}, got {} at slot {index}",
                recovered_from_matrix[index]
            );
        }
    }
}