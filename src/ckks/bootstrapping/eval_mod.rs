//! EvalMod scaffolding for CKKS bootstrapping.
//!
//! In the canonical CKKS bootstrap, this stage approximates modular reduction
//! in the slot domain. A common first approximation path is an odd
//! trigonometric polynomial derived from `sin(2 pi x)` or a related rescaled
//! variant. For now, this module provides:
//! - a validated planning surface for the bootstrap level budget,
//! - a polynomial precomputation container for sine-style approximations,
//! - a first evaluator entry point built on Paterson-Stockmeyer.

use crate::algorithms::paterson_stockmeyer::{
	paterson_stockmeyer_eval,
	paterson_stockmeyer_required_levels,
};
use crate::ckks::bootstrapping::BootstrapParameters;
use crate::rns::{RnsCiphertext, RnsCkksContext, RnsKeySwitchingKey};

use std::collections::BTreeMap;

/// Static planning data for an EvalMod stage.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalModPlan {
	pub bootstrap: BootstrapParameters,
	pub input_scale: f64,
	pub sine_scale: f64,
	pub polynomial_degree: usize,
}

impl EvalModPlan {
	pub fn new(
		bootstrap: BootstrapParameters,
		input_scale: f64,
		sine_scale: f64,
		polynomial_degree: usize,
	) -> Result<Self, String> {
		if input_scale <= 0.0 {
			return Err("EvalMod input_scale must be positive".to_string());
		}
		if sine_scale == 0.0 {
			return Err("EvalMod sine_scale must be non-zero".to_string());
		}
		if polynomial_degree == 0 {
			return Err("EvalMod polynomial_degree must be positive".to_string());
		}
		if polynomial_degree % 2 == 0 {
			return Err("EvalMod polynomial_degree must be odd for a sine approximation".to_string());
		}

		Ok(Self {
			bootstrap,
			input_scale,
			sine_scale,
			polynomial_degree,
		})
	}

	pub fn from_ciphertext(
		context: &RnsCkksContext,
		ciphertext: &RnsCiphertext,
		active_slots: usize,
		input_scale: f64,
		sine_scale: f64,
		polynomial_degree: usize,
	) -> Result<Self, String> {
		Self::new(
			BootstrapParameters::from_ciphertext(context, ciphertext, active_slots)?,
			input_scale,
			sine_scale,
			polynomial_degree,
		)
	}

	pub fn polynomial_term_count(&self) -> usize {
		self.polynomial_degree + 1
	}

	pub fn required_levels(&self) -> Vec<usize> {
		paterson_stockmeyer_required_levels(
			self.bootstrap.ciphertext_level,
			self.polynomial_term_count(),
		)
	}
	}

/// Precomputed polynomial data for executing EvalMod.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalModPrecomputed {
	pub plan: EvalModPlan,
	pub coefficients: Vec<f64>,
}

impl EvalModPrecomputed {
	pub fn new(plan: EvalModPlan, coefficients: Vec<f64>) -> Result<Self, String> {
		validate_sine_coefficients(plan.polynomial_degree, &coefficients)?;
		Ok(Self { plan, coefficients })
	}

	pub fn sine_taylor(plan: EvalModPlan) -> Result<Self, String> {
		let coefficients = sine_taylor_coefficients(plan.polynomial_degree, plan.sine_scale);
		Self::new(plan, coefficients)
	}
	}

/// Runtime configuration for applying EvalMod to a ciphertext.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalModRuntime {
	pub ciphertext_level: usize,
	pub ciphertext_scale_bits: usize,
	pub required_relinearization_levels: Vec<usize>,
}

impl EvalModRuntime {
	pub fn new(plan: &EvalModPlan, ciphertext: &RnsCiphertext) -> Result<Self, String> {
		if ciphertext.level != plan.bootstrap.ciphertext_level {
			return Err(format!(
				"ciphertext level {} did not match EvalMod plan level {}",
				ciphertext.level,
				plan.bootstrap.ciphertext_level,
			));
		}

		Ok(Self {
			ciphertext_level: ciphertext.level,
			ciphertext_scale_bits: ciphertext.scale_bits,
			required_relinearization_levels: plan.required_levels(),
		})
	}
	}

pub fn eval_mod(
	context: &RnsCkksContext,
	ciphertext: &RnsCiphertext,
	precomputed: &EvalModPrecomputed,
	relinearization_keys: &BTreeMap<usize, RnsKeySwitchingKey>,
) -> Result<RnsCiphertext, String> {
	let runtime = EvalModRuntime::new(&precomputed.plan, ciphertext)?;
	for level in &runtime.required_relinearization_levels {
		if !relinearization_keys.contains_key(level) {
			return Err(format!(
				"missing EvalMod relinearization key at level {level}"
			));
		}
	}

	Ok(paterson_stockmeyer_eval(
		context,
		ciphertext,
		&precomputed.coefficients,
		relinearization_keys,
	))
}

pub fn sine_taylor_coefficients(polynomial_degree: usize, sine_scale: f64) -> Vec<f64> {
	assert!(polynomial_degree > 0, "polynomial_degree must be positive");
	assert!(polynomial_degree % 2 == 1, "polynomial_degree must be odd");

	let mut coefficients = vec![0.0_f64; polynomial_degree + 1];
	for odd_degree in (1..=polynomial_degree).step_by(2) {
		let sign = if ((odd_degree - 1) / 2) % 2 == 0 {
			1.0
		} else {
			-1.0
		};
		coefficients[odd_degree] = sign * sine_scale.powi(odd_degree as i32) / factorial(odd_degree);
	}
	coefficients
}

fn validate_sine_coefficients(polynomial_degree: usize, coefficients: &[f64]) -> Result<(), String> {
	if coefficients.len() != polynomial_degree + 1 {
		return Err(format!(
			"EvalMod coefficient vector length {} did not match polynomial degree {}",
			coefficients.len(),
			polynomial_degree,
		));
	}
	for even_degree in (2..=polynomial_degree).step_by(2) {
		if coefficients[even_degree] != 0.0 {
			return Err(format!(
				"EvalMod sine approximation requires zero coefficient for even degree {even_degree}"
			));
		}
	}
	Ok(())
}

fn factorial(value: usize) -> f64 {
	(1..=value).fold(1.0_f64, |accumulator, factor| accumulator * factor as f64)
}

#[cfg(test)]
mod tests {
	use super::{
		eval_mod,
		sine_taylor_coefficients,
		EvalModPlan,
		EvalModPrecomputed,
		EvalModRuntime,
	};
	use crate::ckks::bootstrapping::{BootstrapKeySet, BootstrapParameters};
	use crate::rns::{RnsCkksContext, RnsCkksParams};

	#[test]
	fn sine_taylor_coefficients_match_expected_low_degree_terms() {
		let coefficients = sine_taylor_coefficients(5, 2.0);

		assert_eq!(coefficients.len(), 6);
		assert_eq!(coefficients[0], 0.0);
		assert_eq!(coefficients[2], 0.0);
		assert_eq!(coefficients[4], 0.0);
		assert!((coefficients[1] - 2.0).abs() <= 1e-12);
		assert!((coefficients[3] + 8.0 / 6.0).abs() <= 1e-12);
		assert!((coefficients[5] - 32.0 / 120.0).abs() <= 1e-12);
	}

	#[test]
	fn eval_mod_plan_tracks_required_levels() {
		let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
			.expect("realistic RNS CKKS params must build");
		let bootstrap = BootstrapParameters::new(&context, 7, 8)
			.expect("realistic preset must support bootstrap parameters");

		let plan = EvalModPlan::new(bootstrap, 1.0, 1.0, 5)
			.expect("EvalMod plan construction must succeed");

		assert_eq!(plan.polynomial_term_count(), 6);
		assert_eq!(plan.required_levels(), vec![7, 6, 5]);
	}

	#[test]
	fn eval_mod_runtime_requires_matching_level() {
		let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
			.expect("realistic RNS CKKS params must build");
		let key_pair = context.keygen(64);

		let mut padded = vec![0.0_f64; context.num_slots()];
		padded[0] = 0.125;
		let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);
		let plan = EvalModPlan::from_ciphertext(&context, &ciphertext, 8, 1.0, 1.0, 5)
			.expect("plan construction must succeed");

		let runtime = EvalModRuntime::new(&plan, &ciphertext)
			.expect("runtime construction must succeed");
		assert_eq!(runtime.ciphertext_level, ciphertext.level);

		let lowered = ciphertext.drop_last_level();
		let error = EvalModRuntime::new(&plan, &lowered)
			.expect_err("mismatched level must be rejected");
		assert!(error.contains("did not match EvalMod plan level"));
	}

	#[test]
	fn eval_mod_executes_sine_polynomial_on_realistic_params() {
		let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
			.expect("realistic RNS CKKS params must build");
		let key_pair = context.keygen(64);

		let input = [0.03125_f64, -0.05, 0.08, -0.02];
		let mut padded = vec![0.0_f64; context.num_slots()];
		padded[..input.len()].copy_from_slice(&input);
		let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);

		let plan = EvalModPlan::from_ciphertext(&context, &ciphertext, 8, 1.0, 1.0, 5)
			.expect("plan construction must succeed");
		let precomputed = EvalModPrecomputed::sine_taylor(plan)
			.expect("sine Taylor precomputation must succeed");
		let keys = BootstrapKeySet::for_eval_mod(
			&context,
			&key_pair,
			ciphertext.level,
			precomputed.coefficients.len(),
		);

		let evaluated = eval_mod(
			&context,
			&ciphertext,
			&precomputed,
			&keys.eval_mod_relinearization_keys,
		)
		.expect("EvalMod execution must succeed");

		let recovered = context.decode_real_at_scale(
			&context.decrypt(&evaluated, &key_pair.secret_key),
			evaluated.scale_bits,
		);

		for (index, &value) in input.iter().enumerate() {
			let expected = value - value.powi(3) / 6.0 + value.powi(5) / 120.0;
			assert!(
				(recovered[index] - expected).abs() <= 2e-2,
				"expected {expected}, got {} at slot {index}",
				recovered[index]
			);
		}
	}

	#[test]
	fn eval_mod_rejects_missing_relinearization_levels() {
		let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
			.expect("realistic RNS CKKS params must build");
		let key_pair = context.keygen(64);

		let input = [0.03125_f64, -0.05, 0.08, -0.02];
		let mut padded = vec![0.0_f64; context.num_slots()];
		padded[..input.len()].copy_from_slice(&input);
		let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);

		let plan = EvalModPlan::from_ciphertext(&context, &ciphertext, 8, 1.0, 1.0, 5)
			.expect("plan construction must succeed");
		let precomputed = EvalModPrecomputed::sine_taylor(plan)
			.expect("sine Taylor precomputation must succeed");

		let error = eval_mod(
			&context,
			&ciphertext,
			&precomputed,
			&Default::default(),
		)
		.expect_err("missing relinearization levels must be rejected");

		assert!(error.contains("missing EvalMod relinearization key"));
	}
}