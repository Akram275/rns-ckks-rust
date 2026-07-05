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

use num_complex::Complex64;

use crate::ckks::bootstrapping::linear_transform::{
    apply_diagonal_linear_transform,
    diagonal_transform_rotation_steps,
    generate_diagonal_transform_rotation_keys,
    pack_diagonal_transform_plaintexts,
    repeat_block_slots,
    DiagonalTransformPlaintexts,
    DiagonalTransformRotationKeys,
};
use crate::rns::{RnsCkksContext, RnsCiphertext, RnsPublicKey, RnsSecretKey};

pub type PackedDiagonalRotationKeys = DiagonalTransformRotationKeys;
pub type PackedDiagonalPlaintexts = DiagonalTransformPlaintexts;

pub fn packed_diagonal_rotation_steps(row_count: usize) -> Vec<usize> {
    diagonal_transform_rotation_steps(row_count)
}

pub fn generate_packed_diagonal_rotation_keys(
    context: &RnsCkksContext,
    secret_key: &RnsSecretKey,
    public_key: &RnsPublicKey,
    ciphertext_level: usize,
    row_count: usize,
) -> PackedDiagonalRotationKeys {
    generate_diagonal_transform_rotation_keys(context, secret_key, public_key, ciphertext_level, row_count)
}

pub fn repeat_real_vector_blocks(values: &[f64], slot_count: usize) -> Vec<f64> {
    repeat_block_slots(values, slot_count)
}

pub fn pack_diagonal_plaintexts(
    context: &RnsCkksContext,
    matrix: &[Vec<f64>],
    level: usize,
    scale_bits: usize,
) -> PackedDiagonalPlaintexts {
    let complex_matrix: Vec<Vec<Complex64>> = matrix
        .iter()
        .map(|row| row.iter().map(|&value| Complex64::new(value, 0.0)).collect())
        .collect();
    pack_diagonal_transform_plaintexts(context, &complex_matrix, level, scale_bits)
}

pub fn packed_diagonal_matvec_prepacked(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    diagonals: &PackedDiagonalPlaintexts,
    rotation_keys: &PackedDiagonalRotationKeys,
) -> RnsCiphertext {
    apply_diagonal_linear_transform(context, ciphertext, diagonals, rotation_keys)
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