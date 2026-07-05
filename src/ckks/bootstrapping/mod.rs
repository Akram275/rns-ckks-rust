//! CKKS bootstrapping namespace.
//!
//! This module will host the shared infrastructure and concrete variants for
//! CKKS bootstrapping, starting from the canonical ModRaise -> CoeffToSlot ->
//! EvalMod -> SlotToCoeff decomposition.

pub mod coeff_to_slot;
pub mod eval_mod;
pub mod keys;
pub mod linear_transform;
pub mod mod_raise;
pub mod params;
pub mod roundtrip;
pub mod slot_to_coeff;

pub use eval_mod::{
	eval_mod,
	sine_taylor_coefficients,
	EvalModPlan,
	EvalModPrecomputed,
	EvalModRuntime,
};
pub use keys::BootstrapKeySet;
pub use linear_transform::{
	apply_diagonal_linear_transform,
	diagonal_transform_rotation_steps,
	generate_diagonal_transform_rotation_keys,
	pack_diagonal_transform_plaintexts,
	repeat_block_slots,
	DiagonalTransformPlaintexts,
	DiagonalTransformRotationKeys,
};
pub use params::BootstrapParameters;
pub use roundtrip::bootstrap_identity_roundtrip;
pub use slot_to_coeff::{
	slot_to_coeff,
	SlotToCoeffPlan,
	SlotToCoeffPrecomputed,
	SlotToCoeffRuntime,
};