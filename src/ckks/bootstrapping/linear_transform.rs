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
    baby_step: usize,
    giant_step_diagonals: Vec<Vec<Option<RnsPolynomial>>>,
    level: usize,
    scale_bits: usize,
}

pub fn diagonal_transform_rotation_steps(dimension: usize) -> Vec<usize> {
    assert!(dimension > 0, "transform dimension must be positive");
    (1..dimension).collect()
}

pub fn diagonal_transform_bsgs_rotation_steps(dimension: usize) -> Vec<usize> {
    assert!(dimension > 0, "transform dimension must be positive");
    if dimension == 1 {
        return Vec::new();
    }

    let baby_step = diagonal_transform_bsgs_baby_step(dimension);
    let giant_step_count = (dimension + baby_step - 1) / baby_step;
    let mut steps = BTreeMap::new();

    for baby in 1..baby_step.min(dimension) {
        steps.insert(baby, ());
    }
    for giant_index in 1..giant_step_count {
        let step = giant_index * baby_step;
        if step < dimension {
            steps.insert(step, ());
        }
    }

    steps.into_keys().collect()
}

pub fn generate_diagonal_transform_rotation_keys(
    context: &RnsCkksContext,
    secret_key: &RnsSecretKey,
    public_key: &RnsPublicKey,
    ciphertext_level: usize,
    dimension: usize,
) -> DiagonalTransformRotationKeys {
    let mut by_step = BTreeMap::new();
    for step in diagonal_transform_bsgs_rotation_steps(dimension) {
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
    matrix: &[Vec<Complex64>],
    level: usize,
    scale_bits: usize,
) -> DiagonalTransformPlaintexts {
    let dimension = validate_square_matrix(matrix, context.num_slots());
    let baby_step = diagonal_transform_bsgs_baby_step(dimension);
    let diagonals = (0..dimension)
        .map(|step| {
            let slots = packed_diagonal_slots(matrix, step, context.num_slots());
            context.encode_at_level_and_scale(&slots, level, scale_bits)
        })
        .collect();
    let giant_step_diagonals = pack_bsgs_diagonal_transform_plaintexts(
        context,
        matrix,
        dimension,
        baby_step,
        level,
        scale_bits,
    );

    DiagonalTransformPlaintexts {
        diagonals,
        dimension,
        baby_step,
        giant_step_diagonals,
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

    let baby_rotations = precompute_baby_rotations(context, ciphertext, diagonals, rotation_keys);
    // Rotate each unrescaled inner sum while it still sits at the level the
    // giant-step rotation keys were generated for; rescale only once at the end.
    let mut giant_accumulators = diagonals
        .giant_step_diagonals
        .iter()
        .enumerate()
        .filter_map(|(giant_index, giant_diagonals)| {
            apply_giant_step_terms(context, &baby_rotations, giant_diagonals).map(|inner_sum| {
                if giant_index == 0 {
                    inner_sum
                } else {
                    let giant_step = giant_index * diagonals.baby_step;
                    let rotation_key = rotation_keys
                        .by_step
                        .get(&giant_step)
                        .unwrap_or_else(|| panic!("missing input rotation key for BSGS giant step {giant_step}"));
                    context.rotate(&inner_sum, rotation_key)
                }
            })
        });

    let mut accumulator = giant_accumulators
        .next()
        .expect("diagonal transform must contain at least one non-empty giant-step block");
    for term in giant_accumulators {
        accumulator = accumulator.add(&term);
    }

    context.rescale(&accumulator)
}

fn diagonal_transform_bsgs_baby_step(dimension: usize) -> usize {
    let mut baby_step = 1usize;
    while baby_step * baby_step < dimension {
        baby_step += 1;
    }
    baby_step
}

fn pack_bsgs_diagonal_transform_plaintexts(
    context: &RnsCkksContext,
    matrix: &[Vec<Complex64>],
    dimension: usize,
    baby_step: usize,
    level: usize,
    scale_bits: usize,
) -> Vec<Vec<Option<RnsPolynomial>>> {
    let giant_step_count = (dimension + baby_step - 1) / baby_step;
    (0..giant_step_count)
        .map(|giant_index| {
            let giant_shift = giant_index * baby_step;
            (0..baby_step)
                .map(|baby_index| {
                    let diagonal_step = giant_shift + baby_index;
                    if diagonal_step >= dimension {
                        None
                    } else {
                        let slots = packed_bsgs_diagonal_slots(
                            matrix,
                            diagonal_step,
                            giant_shift,
                            context.num_slots(),
                        );
                        Some(context.encode_at_level_and_scale(&slots, level, scale_bits))
                    }
                })
                .collect()
        })
        .collect()
}

fn packed_bsgs_diagonal_slots(
    matrix: &[Vec<Complex64>],
    diagonal_step: usize,
    giant_shift: usize,
    slot_count: usize,
) -> Vec<Complex64> {
    let dimension = matrix.len();
    let shift = giant_shift % dimension;
    // Tiled with period `dimension` (not zero-padded) because the giant-step
    // rotation is a full-slot-count rotate applied to an already plaintext-masked
    // ciphertext; correctness requires the mask to stay periodic like the input
    // (see the BSGS derivation in docs/bootstrapping.tex), otherwise the giant
    // rotation shifts the active block out of its expected window.
    (0..slot_count)
        .map(|slot| {
            let row = slot % dimension;
            let source_row = (row + dimension - shift) % dimension;
            matrix[source_row][(diagonal_step + source_row) % dimension]
        })
        .collect()
}

fn precompute_baby_rotations(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    diagonals: &DiagonalTransformPlaintexts,
    rotation_keys: &DiagonalTransformRotationKeys,
) -> Vec<RnsCiphertext> {
    let mut baby_rotations = Vec::with_capacity(diagonals.baby_step.min(diagonals.dimension));
    baby_rotations.push(ciphertext.clone());

    for baby_step in 1..diagonals.baby_step.min(diagonals.dimension) {
        let rotation_key = rotation_keys
            .by_step
            .get(&baby_step)
            .unwrap_or_else(|| panic!("missing input rotation key for BSGS baby step {baby_step}"));
        baby_rotations.push(context.rotate(ciphertext, rotation_key));
    }

    baby_rotations
}

fn apply_giant_step_terms(
    context: &RnsCkksContext,
    baby_rotations: &[RnsCiphertext],
    giant_diagonals: &[Option<RnsPolynomial>],
) -> Option<RnsCiphertext> {
    // Left unrescaled so the caller can rotate at the pre-rescale level before
    // the single final rescale.
    let mut terms = giant_diagonals
        .iter()
        .enumerate()
        .filter_map(|(baby_index, diagonal)| {
            diagonal.as_ref().map(|diagonal| context.mult_plain(&baby_rotations[baby_index], diagonal))
        });

    let mut accumulator = terms.next()?;
    for term in terms {
        accumulator = accumulator.add(&term);
    }
    Some(accumulator)
}

fn validate_square_matrix(matrix: &[Vec<Complex64>], slot_count: usize) -> usize {
    assert!(!matrix.is_empty(), "matrix cannot be empty");
    let dimension = matrix.len();
    assert!(dimension.is_power_of_two(), "matrix dimension must be a power of two");
    assert!(dimension <= slot_count, "matrix dimension cannot exceed slot count");
    for row in matrix {
        assert_eq!(row.len(), dimension, "current diagonal transform implementation expects a square matrix");
    }
    dimension
}

fn packed_diagonal_slots(matrix: &[Vec<Complex64>], diagonal: usize, slot_count: usize) -> Vec<Complex64> {
    let dimension = matrix.len();
    let mut slots = vec![Complex64::new(0.0, 0.0); slot_count];
    for row in 0..dimension {
        slots[row] = matrix[row][(diagonal + row) % dimension];
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::{
        apply_diagonal_linear_transform,
        diagonal_transform_bsgs_rotation_steps,
        diagonal_transform_rotation_steps,
        generate_diagonal_transform_rotation_keys,
        pack_diagonal_transform_plaintexts,
        repeat_block_slots,
    };
    use crate::rns::{RnsCkksContext, RnsCkksParams};
    use num_complex::Complex64;

    #[test]
    fn diagonal_transform_rotation_steps_cover_nonzero_diagonals() {
        assert_eq!(diagonal_transform_rotation_steps(1), Vec::<usize>::new());
        assert_eq!(diagonal_transform_rotation_steps(4), vec![1, 2, 3]);
    }

    #[test]
    fn diagonal_transform_bsgs_steps_cover_baby_and_giant_rotations() {
        assert_eq!(diagonal_transform_bsgs_rotation_steps(1), Vec::<usize>::new());
        assert_eq!(diagonal_transform_bsgs_rotation_steps(4), vec![1, 2]);
        assert_eq!(diagonal_transform_bsgs_rotation_steps(8), vec![1, 2, 3, 6]);
    }

    #[test]
    fn diagonal_transform_matches_expected_on_realistic_params() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);
        let matrix = vec![
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(-0.25, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.75, 0.0),
                Complex64::new(0.25, 0.0),
            ],
            vec![
                Complex64::new(0.5, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.25, 0.0),
                Complex64::new(-0.5, 0.0),
            ],
            vec![
                Complex64::new(-0.25, 0.0),
                Complex64::new(0.125, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.75, 0.0),
            ],
        ];
        let vector = vec![0.5, -0.25, 0.125, 0.75];
        let expected: Vec<f64> = matrix
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(lhs, rhs)| lhs.re * rhs).sum())
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
            // The BSGS giant-step rotation is applied to an already-masked
            // ciphertext, so the mask (and thus the output) must stay periodic
            // with period `dimension` rather than zero-padded past it.
            assert!(
                (recovered[index + matrix.len()] - expected_value).abs() <= 2e-2,
                "expected periodic tail matching {expected_value}, got {} at slot {}",
                recovered[index + matrix.len()],
                index + matrix.len()
            );
        }
    }
}