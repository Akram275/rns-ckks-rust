use std::time::{Duration, Instant};

use ckks::algorithms::halevi_shoup::{
    generate_halevi_shoup_dot_product_rotation_keys,
    halevi_shoup_dot_product,
};
use ckks::algorithms::paterson_stockmeyer::{
    generate_paterson_stockmeyer_relinearization_keys,
    paterson_stockmeyer_eval,
};
use ckks::rns::{RnsCiphertext, RnsCkksContext, RnsCkksParams};

const FEATURE_DIM: usize = 6;
const SUPPORT_VECTOR_COUNT: usize = 50;

struct KernelRun {
    ciphertext: RnsCiphertext,
    elapsed: Duration,
}

struct SupportVector {
    features: Vec<f64>,
    label: f64,
}

fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);

    let query = vec![0.24_f64, -0.18, 0.31, -0.22, 0.17, 0.09];
    let support_vectors = build_support_vectors(SUPPORT_VECTOR_COUNT, FEATURE_DIM);

    let expected_linear = plain_linear_svm_score(&support_vectors, &query);
    let expected_sigmoid = plain_sigmoid_svm_score(&support_vectors, &query);
    let expected_polynomial = plain_polynomial_svm_score(&support_vectors, &query);

    let mut padded_query = vec![0.0_f64; context.num_slots()];
    padded_query[..query.len()].copy_from_slice(&query);
    let query_ciphertext = context.encrypt(&context.encode_real(&padded_query), &key_pair.public_key);

    let rotation_key_start = Instant::now();
    let rotation_keys = generate_halevi_shoup_dot_product_rotation_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        query_ciphertext.level - 1,
        query.len(),
    );
    let rotation_key_time = rotation_key_start.elapsed();

    let ps_poly = sigmoid_polynomial_coefficients();
    let relin_key_start = Instant::now();
    let relin_keys = generate_paterson_stockmeyer_relinearization_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        query_ciphertext.level - 1,
        ps_poly.len(),
    );
    let relin_key_time = relin_key_start.elapsed();

    let linear = timed_linear_kernel_svm(&context, &query_ciphertext, &support_vectors, &rotation_keys);
    let sigmoid = timed_sigmoid_kernel_svm(
        &context,
        &query_ciphertext,
        &support_vectors,
        &rotation_keys,
        &relin_keys,
        &ps_poly,
    );
    let polynomial = timed_polynomial_kernel_svm(
        &context,
        &query_ciphertext,
        &support_vectors,
        &rotation_keys,
        &relin_keys,
    );

    let recovered_linear = decrypt_slot_zero(&context, &linear.ciphertext, &key_pair.secret_key);
    let recovered_sigmoid = decrypt_slot_zero(&context, &sigmoid.ciphertext, &key_pair.secret_key);
    let recovered_polynomial = decrypt_slot_zero(&context, &polynomial.ciphertext, &key_pair.secret_key);

    println!("Encrypted binary SVM inference with {} plaintext support vectors", SUPPORT_VECTOR_COUNT);
    println!("query = {:?}", query);
    println!("shared rotation-key generation time = {:.3} ms", to_ms(rotation_key_time));
    println!("shared Paterson-Stockmeyer relin-key generation time = {:.3} ms", to_ms(relin_key_time));
    println!();
    print_kernel_result("linear", expected_linear, recovered_linear, linear.elapsed);
    print_kernel_result("sigmoid-approx", expected_sigmoid, recovered_sigmoid, sigmoid.elapsed);
    print_kernel_result("polynomial-cubic", expected_polynomial, recovered_polynomial, polynomial.elapsed);
}

fn timed_linear_kernel_svm(
    context: &RnsCkksContext,
    query_ciphertext: &RnsCiphertext,
    support_vectors: &[SupportVector],
    rotation_keys: &std::collections::BTreeMap<usize, ckks::rns::RnsRotationKey>,
) -> KernelRun {
    let start = Instant::now();
    let mut terms = support_vectors.iter().map(|support_vector| {
        let dot = halevi_shoup_dot_product(context, query_ciphertext, &support_vector.features, rotation_keys);
        apply_label(context, &dot, support_vector.label)
    });
    let mut accumulator = terms.next().expect("at least one support vector must exist");
    for term in terms {
        accumulator = accumulator.add(&term);
    }
    KernelRun {
        ciphertext: accumulator,
        elapsed: start.elapsed(),
    }
}

fn timed_sigmoid_kernel_svm(
    context: &RnsCkksContext,
    query_ciphertext: &RnsCiphertext,
    support_vectors: &[SupportVector],
    rotation_keys: &std::collections::BTreeMap<usize, ckks::rns::RnsRotationKey>,
    relin_keys: &std::collections::BTreeMap<usize, ckks::rns::RnsKeySwitchingKey>,
    sigmoid_polynomial: &[f64],
) -> KernelRun {
    let start = Instant::now();
    let mut terms = support_vectors.iter().map(|support_vector| {
        let dot = halevi_shoup_dot_product(context, query_ciphertext, &support_vector.features, rotation_keys);
        let kernel_value = paterson_stockmeyer_eval(context, &dot, sigmoid_polynomial, relin_keys);
        apply_label(context, &kernel_value, support_vector.label)
    });
    let mut accumulator = terms.next().expect("at least one support vector must exist");
    for term in terms {
        accumulator = accumulator.add(&term);
    }
    KernelRun {
        ciphertext: accumulator,
        elapsed: start.elapsed(),
    }
}

fn timed_polynomial_kernel_svm(
    context: &RnsCkksContext,
    query_ciphertext: &RnsCiphertext,
    support_vectors: &[SupportVector],
    rotation_keys: &std::collections::BTreeMap<usize, ckks::rns::RnsRotationKey>,
    relin_keys: &std::collections::BTreeMap<usize, ckks::rns::RnsKeySwitchingKey>,
) -> KernelRun {
    let cubic_polynomial = [0.0_f64, 0.0, 0.0, 1.0];
    let start = Instant::now();
    let mut terms = support_vectors.iter().map(|support_vector| {
        let dot = halevi_shoup_dot_product(context, query_ciphertext, &support_vector.features, rotation_keys);
        let kernel_value = paterson_stockmeyer_eval(context, &dot, &cubic_polynomial, relin_keys);
        apply_label(context, &kernel_value, support_vector.label)
    });
    let mut accumulator = terms.next().expect("at least one support vector must exist");
    for term in terms {
        accumulator = accumulator.add(&term);
    }
    KernelRun {
        ciphertext: accumulator,
        elapsed: start.elapsed(),
    }
}

fn apply_label(context: &RnsCkksContext, ciphertext: &RnsCiphertext, label: f64) -> RnsCiphertext {
    context.mult_plain_slots_real_preserve_scale(ciphertext, &constant_slots(context.num_slots(), label))
}

fn decrypt_slot_zero(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    secret_key: &ckks::rns::RnsSecretKey,
) -> f64 {
    context.decode_real_at_scale(&context.decrypt(ciphertext, secret_key), ciphertext.scale_bits)[0]
}

fn plain_linear_svm_score(support_vectors: &[SupportVector], query: &[f64]) -> f64 {
    support_vectors
        .iter()
        .map(|support_vector| support_vector.label * dot_product(&support_vector.features, query))
        .sum()
}

fn plain_sigmoid_svm_score(support_vectors: &[SupportVector], query: &[f64]) -> f64 {
    let coefficients = sigmoid_polynomial_coefficients();
    support_vectors
        .iter()
        .map(|support_vector| {
            let dot = dot_product(&support_vector.features, query);
            support_vector.label * evaluate_polynomial(&coefficients, dot)
        })
        .sum()
}

fn plain_polynomial_svm_score(support_vectors: &[SupportVector], query: &[f64]) -> f64 {
    support_vectors
        .iter()
        .map(|support_vector| {
            let dot = dot_product(&support_vector.features, query);
            support_vector.label * dot.powi(3)
        })
        .sum()
}

fn sigmoid_polynomial_coefficients() -> [f64; 4] {
    [0.5_f64, 0.25, 0.0, -1.0 / 48.0]
}

fn dot_product(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs.iter()).map(|(left, right)| left * right).sum()
}

fn evaluate_polynomial(coefficients: &[f64], value: f64) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .map(|(power, coefficient)| coefficient * value.powi(power as i32))
        .sum()
}

fn build_support_vectors(count: usize, dimension: usize) -> Vec<SupportVector> {
    (0..count)
        .map(|support_index| {
            let features = (0..dimension)
                .map(|feature_index| {
                    let base = ((support_index + 3 * feature_index) % 11) as f64 - 5.0;
                    let sign = if (support_index + feature_index) % 2 == 0 { 1.0 } else { -1.0 };
                    sign * base / 32.0
                })
                .collect();
            let label = if support_index % 2 == 0 { 1.0 } else { -1.0 };
            SupportVector { features, label }
        })
        .collect()
}

fn constant_slots(slot_count: usize, value: f64) -> Vec<f64> {
    vec![value; slot_count]
}

fn print_kernel_result(name: &str, expected: f64, recovered: f64, elapsed: Duration) {
    let expected_sign = if expected >= 0.0 { 1 } else { -1 };
    let recovered_sign = if recovered >= 0.0 { 1 } else { -1 };
    println!("{name} kernel");
    println!("  expected score = {expected}");
    println!("  recovered score = {recovered}");
    println!("  expected sign = {expected_sign}, recovered sign = {recovered_sign}");
    println!("  inference time = {:.3} ms", to_ms(elapsed));
    println!();
}

fn to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}