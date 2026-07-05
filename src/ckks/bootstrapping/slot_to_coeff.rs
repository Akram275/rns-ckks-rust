//! SlotToCoeff scaffolding for CKKS bootstrapping.
//!
//! This module will host the slot-to-coefficient linear transform plan built on
//! top of the shared diagonal linear-transform engine.
//!
//! Intended contents:
//! - transform-plan parameters for the active slot regime,
//! - prepacked diagonal plaintexts for the inverse transform,
//! - rotation-key requirements for the transform execution,
//! - the eventual `slot_to_coeff(...)` evaluator entry point.

use num_complex::Complex64;

use crate::ckks::bootstrapping::exact_transforms::exact_slot_to_coeff_matrix;
use crate::ckks::bootstrapping::linear_transform::{
    apply_diagonal_linear_transform,
    diagonal_transform_rotation_steps,
    pack_diagonal_transform_plaintexts,
    DiagonalTransformPlaintexts,
    DiagonalTransformRotationKeys,
};
use crate::ckks::bootstrapping::BootstrapParameters;
use crate::rns::{RnsCiphertext, RnsCkksContext};

/// Static planning data for a SlotToCoeff transform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotToCoeffPlan {
    pub bootstrap: BootstrapParameters,
    pub transform_dimension: usize,
    pub rotation_steps: Vec<usize>,
}

impl SlotToCoeffPlan {
    pub fn new(bootstrap: BootstrapParameters) -> Self {
        let transform_dimension = bootstrap.active_slots;
        let rotation_steps = diagonal_transform_rotation_steps(transform_dimension);

        Self {
            bootstrap,
            transform_dimension,
            rotation_steps,
        }
    }

    pub fn from_ciphertext(
        context: &RnsCkksContext,
        ciphertext: &RnsCiphertext,
        active_slots: usize,
    ) -> Result<Self, String> {
        Ok(Self::new(BootstrapParameters::from_ciphertext(
            context,
            ciphertext,
            active_slots,
        )?))
    }
}

/// Precomputed diagonal plaintext data for executing a SlotToCoeff transform.
#[derive(Clone, Debug)]
pub struct SlotToCoeffPrecomputed {
    pub plan: SlotToCoeffPlan,
    pub diagonals: DiagonalTransformPlaintexts,
}

impl SlotToCoeffPrecomputed {
    pub fn new(
        context: &RnsCkksContext,
        plan: SlotToCoeffPlan,
        matrix: &[Vec<Complex64>],
        ciphertext_scale_bits: usize,
    ) -> Result<Self, String> {
        validate_transform_matrix(matrix, plan.transform_dimension)?;

        let diagonals = pack_diagonal_transform_plaintexts(
            context,
            matrix,
            plan.bootstrap.ciphertext_level,
            ciphertext_scale_bits,
        );

        Ok(Self { plan, diagonals })
    }

    pub fn exact(
        context: &RnsCkksContext,
        plan: SlotToCoeffPlan,
        ciphertext_scale_bits: usize,
    ) -> Result<Self, String> {
        let matrix = exact_slot_to_coeff_matrix(context, plan.transform_dimension)?;
        Self::new(context, plan, &matrix, ciphertext_scale_bits)
    }
}

/// Runtime configuration for applying a SlotToCoeff transform to a ciphertext.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotToCoeffRuntime {
    pub plan: SlotToCoeffPlan,
    pub ciphertext_level: usize,
    pub ciphertext_scale_bits: usize,
}

impl SlotToCoeffRuntime {
    pub fn new(plan: SlotToCoeffPlan, ciphertext: &RnsCiphertext) -> Result<Self, String> {
        if ciphertext.level != plan.bootstrap.ciphertext_level {
            return Err(format!(
                "ciphertext level {} did not match SlotToCoeff plan level {}",
                ciphertext.level,
                plan.bootstrap.ciphertext_level,
            ));
        }

        Ok(Self {
            plan,
            ciphertext_level: ciphertext.level,
            ciphertext_scale_bits: ciphertext.scale_bits,
        })
    }
}

pub fn slot_to_coeff(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    precomputed: &SlotToCoeffPrecomputed,
    rotation_keys: &DiagonalTransformRotationKeys,
) -> Result<RnsCiphertext, String> {
    let _runtime = SlotToCoeffRuntime::new(precomputed.plan.clone(), ciphertext)?;
    Ok(apply_diagonal_linear_transform(
        context,
        ciphertext,
        &precomputed.diagonals,
        rotation_keys,
    ))
}

fn validate_transform_matrix(matrix: &[Vec<Complex64>], expected_dimension: usize) -> Result<(), String> {
    if matrix.is_empty() {
        return Err("SlotToCoeff transform matrix cannot be empty".to_string());
    }
    if matrix.len() != expected_dimension {
        return Err(format!(
            "SlotToCoeff transform matrix dimension {} did not match plan dimension {}",
            matrix.len(),
            expected_dimension,
        ));
    }
    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != expected_dimension {
            return Err(format!(
                "SlotToCoeff transform row {row_index} had length {}, expected {}",
                row.len(),
                expected_dimension,
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{slot_to_coeff, SlotToCoeffPlan, SlotToCoeffPrecomputed, SlotToCoeffRuntime};
    use crate::ckks::bootstrapping::linear_transform::{
        generate_diagonal_transform_rotation_keys,
        repeat_block_slots,
    };
    use crate::ckks::bootstrapping::BootstrapParameters;
    use num_complex::Complex64;
    use crate::rns::{RnsCkksContext, RnsCkksParams};

    #[test]
    fn slot_to_coeff_plan_follows_bootstrap_active_slots() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let bootstrap = BootstrapParameters::new(&context, 7, 8)
            .expect("realistic preset must support bootstrap parameters");

        let plan = SlotToCoeffPlan::new(bootstrap.clone());

        assert_eq!(plan.bootstrap, bootstrap);
        assert_eq!(plan.transform_dimension, 8);
        assert_eq!(plan.rotation_steps, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn slot_to_coeff_runtime_requires_matching_level() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let mut padded = vec![0.0_f64; context.num_slots()];
        padded[0] = 0.125;
        let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
        let plan = SlotToCoeffPlan::from_ciphertext(&context, &ciphertext, 8)
            .expect("plan construction must succeed");

        let runtime = SlotToCoeffRuntime::new(plan.clone(), &ciphertext)
            .expect("runtime construction must succeed");
        assert_eq!(runtime.plan, plan);
        assert_eq!(runtime.ciphertext_level, ciphertext.level);
        assert_eq!(runtime.ciphertext_scale_bits, ciphertext.scale_bits);

        let lowered = ciphertext.drop_last_level();
        let error = SlotToCoeffRuntime::new(plan, &lowered)
            .expect_err("mismatched level must be rejected");
        assert!(error.contains("did not match SlotToCoeff plan level"));
    }

    #[test]
    fn slot_to_coeff_precompute_validates_and_packs_diagonals() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let bootstrap = BootstrapParameters::new(&context, 7, 8)
            .expect("realistic preset must support bootstrap parameters");
        let plan = SlotToCoeffPlan::new(bootstrap);

        let identity: Vec<Vec<Complex64>> = (0..8)
            .map(|row| {
                (0..8)
                    .map(|col| if row == col { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) })
                    .collect()
            })
            .collect();

        let precomputed = SlotToCoeffPrecomputed::new(
            &context,
            plan.clone(),
            &identity,
            context.params().scale_bits,
        )
        .expect("identity transform precomputation must succeed");

        assert_eq!(precomputed.plan, plan);
        assert_eq!(precomputed.diagonals.dimension, 8);
        assert_eq!(precomputed.diagonals.diagonals.len(), 8);

        let bad_matrix = vec![vec![Complex64::new(1.0, 0.0); 4]; 4];
        let error = SlotToCoeffPrecomputed::new(
            &context,
            precomputed.plan.clone(),
            &bad_matrix,
            context.params().scale_bits,
        )
        .expect_err("dimension mismatch must be rejected");
        assert!(error.contains("did not match plan dimension"));
    }

    #[test]
    fn slot_to_coeff_executes_precomputed_identity_transform() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let input = vec![0.5_f64, -0.25, 0.125, 0.75, -0.5, 0.25, -0.125, 0.375];
        let packed = repeat_block_slots(&input, context.num_slots());
        let ciphertext = context.encrypt(&context.encode_real(&packed), &key_pair.public_key);
        let plan = SlotToCoeffPlan::from_ciphertext(&context, &ciphertext, input.len())
            .expect("plan construction must succeed");

        let identity: Vec<Vec<Complex64>> = (0..input.len())
            .map(|row| {
                (0..input.len())
                    .map(|col| if row == col { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) })
                    .collect()
            })
            .collect();
        let precomputed = SlotToCoeffPrecomputed::new(
            &context,
            plan,
            &identity,
            ciphertext.scale_bits,
        )
        .expect("precomputation must succeed");
        let rotation_keys = generate_diagonal_transform_rotation_keys(
            &context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            input.len(),
        );

        let transformed = slot_to_coeff(&context, &ciphertext, &precomputed, &rotation_keys)
            .expect("SlotToCoeff execution must succeed");
        let recovered = context.decode_real_at_scale(
            &context.decrypt(&transformed, &key_pair.secret_key),
            transformed.scale_bits,
        );

        for (index, &expected) in input.iter().enumerate() {
            assert!(
                (recovered[index] - expected).abs() <= 2e-2,
                "expected {expected}, got {} at slot {index}",
                recovered[index]
            );
        }
    }
}