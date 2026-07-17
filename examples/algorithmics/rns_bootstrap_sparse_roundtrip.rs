use std::time::{Duration, Instant};

use ckks::ckks::bootstrapping::{
    coeff_to_slot_exact,
    eval_mod,
    sine_taylor_coefficients,
    slot_to_coeff_exact,
    BootstrapKeySet,
    EvalModPlan,
    EvalModPrecomputed,
    exact_coeff_to_slot_matrix,
    exact_slot_to_coeff_matrix,
};
use ckks::ckks::bootstrapping::mod_raise::mod_raise;
use ckks::rns::{RnsCkksContext, RnsCkksParams};
use num_complex::Complex64;

const ACTIVE_SLOTS: usize = 8;
const SINE_SCALE: f64 = 1.0;
const POLYNOMIAL_DEGREE: usize = 5;
const ROUNDTRIP_TOLERANCE: f64 = 5e-2;

fn main() {
    let total_start = Instant::now();

    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);

    let message = vec![
        Complex64::new(0.03125, -0.015625),
        Complex64::new(-0.05, 0.0125),
        Complex64::new(0.08, -0.02),
        Complex64::new(-0.02, 0.01),
        Complex64::new(0.0125, -0.03125),
        Complex64::new(-0.04, 0.02),
        Complex64::new(0.025, -0.0125),
        Complex64::new(-0.01, 0.04),
    ];
    let mask = vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.75, 0.0),
        Complex64::new(0.6, 0.0),
        Complex64::new(0.8, 0.0),
        Complex64::new(0.7, 0.0),
        Complex64::new(0.55, 0.0),
        Complex64::new(0.65, 0.0),
        Complex64::new(0.9, 0.0),
    ];

    let repeated_message = repeat_complex_block(&message, context.num_slots());
    let repeated_mask = repeat_complex_block(&mask, context.num_slots());
    let ciphertext = context.encrypt(&context.encode(&repeated_message), &key_pair.public_key);
    let plaintext_after_compute: Vec<Complex64> = message
        .iter()
        .zip(mask.iter())
        .map(|(value, scale)| *value * *scale)
        .collect();

    let (computed_ciphertext, compute_time) = time(|| {
        context.mult_plain_slots(&ciphertext, &repeated_mask)
    });
    let computed_plain = context.decode_at_scale(
        &context.decrypt(&computed_ciphertext, &key_pair.secret_key),
        computed_ciphertext.scale_bits,
    );

    let (raised, mod_raise_time) = time(|| mod_raise(&context, &computed_ciphertext));
    let projected = raised.project_to_q_ciphertext();

    let (slot_ciphertext, coeff_to_slot_time) = time(|| {
        coeff_to_slot_exact(&context, &projected, &key_pair, ACTIVE_SLOTS)
            .expect("CoeffToSlot stage must succeed")
    });

    let (eval_mod_precomputed, eval_mod_setup_time) = time(|| {
        let plan = EvalModPlan::from_ciphertext(
            &context,
            &slot_ciphertext,
            ACTIVE_SLOTS,
            1.0,
            SINE_SCALE,
            POLYNOMIAL_DEGREE,
        )
        .expect("EvalMod plan construction must succeed");
        EvalModPrecomputed::sine_taylor(plan)
            .expect("EvalMod precomputation must succeed")
    });
    let (eval_mod_keys, eval_mod_key_time) = time(|| {
        BootstrapKeySet::for_eval_mod(
            &context,
            &key_pair,
            slot_ciphertext.level,
            eval_mod_precomputed.coefficients.len(),
        )
    });
    let (evaluated_ciphertext, eval_mod_time) = time(|| {
        eval_mod(
            &context,
            &slot_ciphertext,
            &eval_mod_precomputed,
            &eval_mod_keys.eval_mod_relinearization_keys,
        )
        .expect("EvalMod stage must succeed")
    });

    let (bootstrapped_ciphertext, slot_to_coeff_time) = time(|| {
        slot_to_coeff_exact(&context, &evaluated_ciphertext, &key_pair, ACTIVE_SLOTS)
            .expect("SlotToCoeff stage must succeed")
    });

    let total_time = total_start.elapsed();
    let recovered = context.decode_at_scale(
        &context.decrypt(&bootstrapped_ciphertext, &key_pair.secret_key),
        bootstrapped_ciphertext.scale_bits,
    );

    let coeff_to_slot_matrix = exact_coeff_to_slot_matrix(&context, ACTIVE_SLOTS)
        .expect("exact sparse CoeffToSlot matrix must build");
    let slot_to_coeff_matrix = exact_slot_to_coeff_matrix(&context, ACTIVE_SLOTS)
        .expect("exact sparse SlotToCoeff matrix must build");
    let coefficients = sine_taylor_coefficients(POLYNOMIAL_DEGREE, SINE_SCALE);
    let expected_slots = apply_matrix(&coeff_to_slot_matrix, &plaintext_after_compute);
    let expected_evaluated_slots: Vec<Complex64> = expected_slots
        .iter()
        .map(|&value| evaluate_polynomial(&coefficients, value))
        .collect();
    let expected = apply_matrix(&slot_to_coeff_matrix, &expected_evaluated_slots);

    let max_error = recovered
        .iter()
        .zip(expected.iter())
        .take(ACTIVE_SLOTS)
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0_f64, f64::max);

    println!("RNS CKKS sparse bootstrap roundtrip");
    println!(
        "parameters: N = {}, slots = {}, ciphertext levels = {}, scale_bits = {}",
        context.params().poly_degree,
        context.num_slots(),
        context.levels(),
        context.params().scale_bits,
    );
    println!(
        "bootstrap config: active_slots = {}, sine_scale = {}, polynomial_degree = {}",
        ACTIVE_SLOTS,
        SINE_SCALE,
        POLYNOMIAL_DEGREE,
    );
    println!();
    println!("pre-bootstrap computation:");
    println!("  encrypted plaintext multiplication + rescale: {:?}", compute_time);
    println!("  level after compute: {}", computed_ciphertext.level);
    println!("  first active slots after compute: {:?}", &computed_plain[..ACTIVE_SLOTS]);
    println!();
    println!("bootstrap stage timings:");
    println!("  ModRaise: {:?}", mod_raise_time);
    println!("  CoeffToSlot: {:?}", coeff_to_slot_time);
    println!("  EvalMod precompute: {:?}", eval_mod_setup_time);
    println!("  EvalMod key scaffold: {:?}", eval_mod_key_time);
    println!("  EvalMod execution: {:?}", eval_mod_time);
    println!("  SlotToCoeff: {:?}", slot_to_coeff_time);
    println!("  total example wall time: {:?}", total_time);
    println!();
    println!("level flow:");
    println!("  encrypted input level: {}", ciphertext.level);
    println!("  after compute level: {}", computed_ciphertext.level);
    println!("  ModRaise projected Q level: {}", projected.level);
    println!("  CoeffToSlot output level: {}", slot_ciphertext.level);
    println!("  EvalMod output level: {}", evaluated_ciphertext.level);
    println!("  SlotToCoeff output level: {}", bootstrapped_ciphertext.level);
    println!();
    println!("expected first active slots: {:?}", &expected[..ACTIVE_SLOTS]);
    println!("recovered first active slots: {:?}", &recovered[..ACTIVE_SLOTS]);
    println!("maximum active-slot error: {:.6e}", max_error);

    assert!(
        max_error <= ROUNDTRIP_TOLERANCE,
        "bootstrapping roundtrip error {:.6e} exceeded tolerance {:.6e}",
        max_error,
        ROUNDTRIP_TOLERANCE,
    );
}

fn time<T>(operation: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let output = operation();
    (output, start.elapsed())
}

fn repeat_complex_block(values: &[Complex64], slot_count: usize) -> Vec<Complex64> {
    assert!(!values.is_empty(), "input block cannot be empty");
    assert!(values.len() <= slot_count, "input block cannot exceed slot count");
    assert_eq!(slot_count % values.len(), 0, "slot count must be a multiple of the input block length");

    (0..slot_count)
        .map(|index| values[index % values.len()])
        .collect()
}

fn apply_matrix(matrix: &[Vec<Complex64>], vector: &[Complex64]) -> Vec<Complex64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector.iter()).map(|(lhs, rhs)| *lhs * *rhs).sum())
        .collect()
}

fn evaluate_polynomial(coefficients: &[f64], value: Complex64) -> Complex64 {
    coefficients
        .iter()
        .rev()
        .fold(Complex64::new(0.0, 0.0), |accumulator, &coefficient| {
            accumulator * value + Complex64::new(coefficient, 0.0)
        })
}