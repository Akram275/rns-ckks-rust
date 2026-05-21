use std::time::{Duration, Instant};

use ckks::algorithms::halevi_shoup::{
    generate_halevi_shoup_dot_product_rotation_keys,
    halevi_shoup_dot_product_rotation_steps,
};
use ckks::algorithms::packed_diagonal::{
    generate_packed_diagonal_rotation_keys,
    pack_diagonal_plaintexts,
    packed_diagonal_matvec_prepacked,
    repeat_real_vector_blocks,
};
use ckks::algorithms::paterson_stockmeyer::{
    generate_paterson_stockmeyer_relinearization_keys,
    paterson_stockmeyer_block_size,
    paterson_stockmeyer_eval,
};
use ckks::rns::{RnsCiphertext, RnsCkksContext, RnsCkksParams, RnsRotationKey};

const FEATURE_DIM: usize = 6;
const SUPPORT_VECTOR_COUNT: usize = 50;
const PADDED_SUPPORT_VECTOR_COUNT: usize = SUPPORT_VECTOR_COUNT.next_power_of_two();

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
    let support_matrix = build_support_matrix(&support_vectors, PADDED_SUPPORT_VECTOR_COUNT);
    let label_mask = label_mask_slots(&support_vectors, PADDED_SUPPORT_VECTOR_COUNT, context.num_slots());

    let packed_query = repeat_real_vector_blocks(
        &pad_query_to_dimension(&query, PADDED_SUPPORT_VECTOR_COUNT),
        context.num_slots(),
    );
    let diagonals = pack_diagonal_plaintexts(
        &context,
        &support_matrix,
        context.levels() - 1,
        context.params().scale_bits,
    );
    let query_ciphertext = context.encrypt(&context.encode_real(&packed_query), &key_pair.public_key);

    let matvec_key_start = Instant::now();
    let matvec_rotation_keys = generate_packed_diagonal_rotation_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        query_ciphertext.level,
        PADDED_SUPPORT_VECTOR_COUNT,
    );
    let matvec_key_time = matvec_key_start.elapsed();

    let linear_sum_key_start = Instant::now();
    let linear_sum_rotation_keys = generate_halevi_shoup_dot_product_rotation_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        query_ciphertext.level - 1,
        PADDED_SUPPORT_VECTOR_COUNT,
    );
    let linear_sum_key_time = linear_sum_key_start.elapsed();

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
    let nonlinear_level = paterson_stockmeyer_output_level(query_ciphertext.level - 1, ps_poly.len());
    let nonlinear_sum_key_start = Instant::now();
    let nonlinear_sum_rotation_keys = generate_halevi_shoup_dot_product_rotation_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        nonlinear_level,
        PADDED_SUPPORT_VECTOR_COUNT,
    );
    let nonlinear_sum_key_time = nonlinear_sum_key_start.elapsed();

    let linear = timed_linear_kernel_svm(
        &context,
        &query_ciphertext,
        &diagonals,
        &matvec_rotation_keys,
        &linear_sum_rotation_keys,
        &label_mask,
    );
    let sigmoid = timed_sigmoid_kernel_svm(
        &context,
        &query_ciphertext,
        &diagonals,
        &matvec_rotation_keys,
        &nonlinear_sum_rotation_keys,
        &label_mask,
        &relin_keys,
        &ps_poly,
    );
    let polynomial = timed_polynomial_kernel_svm(
        &context,
        &query_ciphertext,
        &diagonals,
        &matvec_rotation_keys,
        &nonlinear_sum_rotation_keys,
        &label_mask,
        &relin_keys,
    );

    let recovered_linear = decrypt_slot_zero(&context, &linear.ciphertext, &key_pair.secret_key);
    let recovered_sigmoid = decrypt_slot_zero(&context, &sigmoid.ciphertext, &key_pair.secret_key);
    let recovered_polynomial = decrypt_slot_zero(&context, &polynomial.ciphertext, &key_pair.secret_key);

    println!("Encrypted binary SVM inference with {} plaintext support vectors", SUPPORT_VECTOR_COUNT);
    println!("packed support-matrix dimension = {}", PADDED_SUPPORT_VECTOR_COUNT);
    println!("query = {:?}", query);
    println!("shared matrix-product rotation-key generation time = {:.3} ms", to_ms(matvec_key_time));
    println!("shared linear total-sum key generation time = {:.3} ms", to_ms(linear_sum_key_time));
    println!("shared Paterson-Stockmeyer relin-key generation time = {:.3} ms", to_ms(relin_key_time));
    println!("shared nonlinear total-sum key generation time = {:.3} ms", to_ms(nonlinear_sum_key_time));
    println!();
    print_kernel_result("linear", expected_linear, recovered_linear, linear.elapsed);
    print_kernel_result("sigmoid-approx", expected_sigmoid, recovered_sigmoid, sigmoid.elapsed);
    print_kernel_result("polynomial-cubic", expected_polynomial, recovered_polynomial, polynomial.elapsed);
}

fn timed_linear_kernel_svm(
    context: &RnsCkksContext,
    query_ciphertext: &RnsCiphertext,
    diagonals: &ckks::algorithms::packed_diagonal::PackedDiagonalPlaintexts,
    matvec_rotation_keys: &ckks::algorithms::packed_diagonal::PackedDiagonalRotationKeys,
    sum_rotation_keys: &std::collections::BTreeMap<usize, RnsRotationKey>,
    label_mask: &[f64],
) -> KernelRun {
    let start = Instant::now();
    let dot_products = packed_diagonal_matvec_prepacked(context, query_ciphertext, diagonals, matvec_rotation_keys);
    let weighted = apply_slot_weights(context, &dot_products, label_mask);
    let accumulator = total_sum_slots(context, &weighted, PADDED_SUPPORT_VECTOR_COUNT, sum_rotation_keys);
    KernelRun {
        ciphertext: accumulator,
        elapsed: start.elapsed(),
    }
}

fn timed_sigmoid_kernel_svm(
    context: &RnsCkksContext,
    query_ciphertext: &RnsCiphertext,
    diagonals: &ckks::algorithms::packed_diagonal::PackedDiagonalPlaintexts,
    matvec_rotation_keys: &ckks::algorithms::packed_diagonal::PackedDiagonalRotationKeys,
    sum_rotation_keys: &std::collections::BTreeMap<usize, RnsRotationKey>,
    label_mask: &[f64],
    relin_keys: &std::collections::BTreeMap<usize, ckks::rns::RnsKeySwitchingKey>,
    sigmoid_polynomial: &[f64],
) -> KernelRun {
    let start = Instant::now();
    let dot_products = packed_diagonal_matvec_prepacked(context, query_ciphertext, diagonals, matvec_rotation_keys);
    let kernel_values = paterson_stockmeyer_eval(context, &dot_products, sigmoid_polynomial, relin_keys);
    let weighted = apply_slot_weights(context, &kernel_values, label_mask);
    let accumulator = total_sum_slots(context, &weighted, PADDED_SUPPORT_VECTOR_COUNT, sum_rotation_keys);
    KernelRun {
        ciphertext: accumulator,
        elapsed: start.elapsed(),
    }
}

fn timed_polynomial_kernel_svm(
    context: &RnsCkksContext,
    query_ciphertext: &RnsCiphertext,
    diagonals: &ckks::algorithms::packed_diagonal::PackedDiagonalPlaintexts,
    matvec_rotation_keys: &ckks::algorithms::packed_diagonal::PackedDiagonalRotationKeys,
    sum_rotation_keys: &std::collections::BTreeMap<usize, RnsRotationKey>,
    label_mask: &[f64],
    relin_keys: &std::collections::BTreeMap<usize, ckks::rns::RnsKeySwitchingKey>,
) -> KernelRun {
    let cubic_polynomial = [0.0_f64, 0.0, 0.0, 1.0];
    let start = Instant::now();
    let dot_products = packed_diagonal_matvec_prepacked(context, query_ciphertext, diagonals, matvec_rotation_keys);
    let kernel_values = paterson_stockmeyer_eval(context, &dot_products, &cubic_polynomial, relin_keys);
    let weighted = apply_slot_weights(context, &kernel_values, label_mask);
    let accumulator = total_sum_slots(context, &weighted, PADDED_SUPPORT_VECTOR_COUNT, sum_rotation_keys);
    KernelRun {
        ciphertext: accumulator,
        elapsed: start.elapsed(),
    }
}

fn apply_slot_weights(context: &RnsCkksContext, ciphertext: &RnsCiphertext, weights: &[f64]) -> RnsCiphertext {
    context.mult_plain_slots_real_preserve_scale(ciphertext, weights)
}

fn total_sum_slots(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    active_slots: usize,
    rotation_keys: &std::collections::BTreeMap<usize, RnsRotationKey>,
) -> RnsCiphertext {
    let mut accumulator = ciphertext.clone();
    for step in halevi_shoup_dot_product_rotation_steps(active_slots) {
        let rotation_key = rotation_keys
            .get(&step)
            .unwrap_or_else(|| panic!("missing total-sum rotation key for step {step}"));
        let rotated = context.rotate(&accumulator, rotation_key);
        accumulator = accumulator.add(&rotated);
    }
    accumulator
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

fn build_support_matrix(support_vectors: &[SupportVector], dimension: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0_f64; dimension]; dimension];
    for (row, support_vector) in support_vectors.iter().enumerate() {
        matrix[row][..support_vector.features.len()].copy_from_slice(&support_vector.features);
    }
    matrix
}

fn pad_query_to_dimension(query: &[f64], dimension: usize) -> Vec<f64> {
    let mut padded = vec![0.0_f64; dimension];
    padded[..query.len()].copy_from_slice(query);
    padded
}

fn label_mask_slots(
    support_vectors: &[SupportVector],
    active_slots: usize,
    slot_count: usize,
) -> Vec<f64> {
    let mut first_block = vec![0.0_f64; active_slots];
    for (index, support_vector) in support_vectors.iter().enumerate() {
        first_block[index] = support_vector.label;
    }
    repeat_real_vector_blocks(&first_block, slot_count)
}

fn paterson_stockmeyer_output_level(ciphertext_level: usize, term_count: usize) -> usize {
    if term_count == 1 {
        return ciphertext_level;
    }

    let block_size = paterson_stockmeyer_block_size(term_count);
    let block_count = term_count.div_ceil(block_size);
    let top_level = ciphertext_level - (block_size - 1);
    top_level - (block_count - 1)
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