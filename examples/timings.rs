use std::time::{Duration, Instant};

use ckks::algorithms::halevi_shoup::{
    generate_halevi_shoup_dot_product_rotation_keys,
    halevi_shoup_dot_product_rotation_steps,
};
use ckks::algorithms::packed_diagonal::{
    generate_packed_diagonal_rotation_keys,
    pack_diagonal_plaintexts,
    packed_diagonal_rotation_steps,
    repeat_real_vector_blocks,
};
use ckks::rns::{RnsCiphertext, RnsCkksContext, RnsCkksParams, RnsKeySwitchingKey, RnsRotationKey};

struct TimingBreakdown {
    operation: Duration,
    key_generation: Option<Duration>,
    keyswitching: Option<Duration>,
    rescaling: Option<Duration>,
}

fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
    let key_pair = context.keygen(64);

    let vector_a = vec![0.5, -0.25, 0.125, 0.75];
    let vector_b = vec![1.25, -0.5, 0.375, -0.125];
    let plaintext_vector = vec![2.0, -1.0, 0.5, 4.0];
    let matrix = vec![
        vec![1.0, 0.5, -0.25, 0.0],
        vec![0.0, -1.0, 0.75, 0.25],
        vec![0.5, 0.0, 1.25, -0.5],
        vec![-0.25, 0.125, 0.0, 0.75],
    ];

    let mut padded_a = vec![0.0; context.num_slots()];
    padded_a[..vector_a.len()].copy_from_slice(&vector_a);
    let mut padded_b = vec![0.0; context.num_slots()];
    padded_b[..vector_b.len()].copy_from_slice(&vector_b);

    let ciphertext_a = context.encrypt(&context.encode_real(&padded_a), &key_pair.public_key);
    let ciphertext_b = context.encrypt(&context.encode_real(&padded_b), &key_pair.public_key);

    let addition = time_ciphertext_addition(&ciphertext_a, &ciphertext_b);
    let plain_product = time_plain_ciphertext_product(&context, &ciphertext_a, &plaintext_vector);
    let encrypted_product = time_ciphertext_ciphertext_product(&context, &ciphertext_a, &ciphertext_b, &key_pair);
    let plain_dot = time_plain_ciphertext_dot_product(&context, &ciphertext_a, &plaintext_vector, &key_pair);
    let encrypted_dot = time_ciphertext_ciphertext_dot_product(&context, &ciphertext_a, &ciphertext_b, &key_pair, vector_a.len());

    let repeated_matrix_input = repeat_real_vector_blocks(&vector_a, context.num_slots());
    let matrix_ciphertext = context.encrypt(&context.encode_real(&repeated_matrix_input), &key_pair.public_key);
    let matrix_product = time_plain_matrix_ciphertext_product(&context, &matrix_ciphertext, &matrix, &key_pair);

    println!("RNS CKKS realistic timings");
    println!("parameters: N = {}, levels = {}, scale_bits = {}", context.params().poly_degree, context.levels(), context.params().scale_bits);
    println!("timings exclude setup work such as secret/public key generation, encryption, and plaintext packing, except for the requested keyswitch-key generation phase");
    println!();
    print_breakdown("1. ciphertext addition", &addition);
    print_breakdown("2. plaintext-ciphertext product", &plain_product);
    print_breakdown("3. ciphertext-ciphertext product", &encrypted_product);
    print_breakdown("4. plaintext-ciphertext dot product", &plain_dot);
    print_breakdown("5. ciphertext-ciphertext dot product", &encrypted_dot);
    print_breakdown("6. plaintext-matrix ciphertext product", &matrix_product);
}

fn time_ciphertext_addition(lhs: &RnsCiphertext, rhs: &RnsCiphertext) -> TimingBreakdown {
    let operation = time(|| lhs.add(rhs)).1;
    TimingBreakdown {
        operation,
        key_generation: None,
        keyswitching: None,
        rescaling: None,
    }
}

fn time_plain_ciphertext_product(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    plaintext_slots: &[f64],
) -> TimingBreakdown {
    let mut padded = vec![0.0; context.num_slots()];
    padded[..plaintext_slots.len()].copy_from_slice(plaintext_slots);
    let plain = context.encode_real(&padded).truncate_levels(ciphertext.level + 1);

    let (multiplied, operation) = time(|| context.mult_plain(ciphertext, &plain));
    let (_, rescaling) = time(|| context.rescale(&multiplied));

    TimingBreakdown {
        operation,
        key_generation: None,
        keyswitching: None,
        rescaling: Some(rescaling),
    }
}

fn time_ciphertext_ciphertext_product(
    context: &RnsCkksContext,
    lhs: &RnsCiphertext,
    rhs: &RnsCiphertext,
    key_pair: &ckks::rns::RnsKeyPair,
) -> TimingBreakdown {
    let (_, key_generation) = time(|| {
        context.generate_relinearization_key(&key_pair.secret_key, &key_pair.public_key)
    });
    let relin_key = context.generate_relinearization_key(&key_pair.secret_key, &key_pair.public_key);

    let (quadratic, operation) = time(|| context.mult(lhs, rhs));
    let (relinearized, keyswitching, relinearize_overhead) = timed_relinearize(context, &quadratic, &relin_key);
    let (_, rescaling) = time(|| context.rescale(&relinearized));

    TimingBreakdown {
        operation: operation + relinearize_overhead,
        key_generation: Some(key_generation),
        keyswitching: Some(keyswitching),
        rescaling: Some(rescaling),
    }
}

fn time_plain_ciphertext_dot_product(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    plaintext_slots: &[f64],
    key_pair: &ckks::rns::RnsKeyPair,
) -> TimingBreakdown {
    let rotation_level = ciphertext.level - 1;
    let (rotation_keys, key_generation) = time(|| {
        generate_halevi_shoup_dot_product_rotation_keys(
            context,
            &key_pair.secret_key,
            &key_pair.public_key,
            rotation_level,
            plaintext_slots.len(),
        )
    });

    let mut padded = vec![0.0; context.num_slots()];
    padded[..plaintext_slots.len()].copy_from_slice(plaintext_slots);
    let plain = context.encode_real(&padded).truncate_levels(ciphertext.level + 1);
    let (multiplied, mult_time) = time(|| context.mult_plain(ciphertext, &plain));
    let (mut accumulator, rescaling) = time(|| context.rescale(&multiplied));

    let mut operation = mult_time;
    let mut keyswitching = Duration::ZERO;
    for step in halevi_shoup_dot_product_rotation_steps(plaintext_slots.len()) {
        let rotation_key = rotation_keys
            .get(&step)
            .unwrap_or_else(|| panic!("missing rotation key for step {step}"));
        let (rotated, rotate_operation, rotate_keyswitch) = timed_rotate(context, &accumulator, rotation_key);
        keyswitching += rotate_keyswitch;
        operation += rotate_operation;
        let (next_accumulator, add_time) = time(|| accumulator.add(&rotated));
        operation += add_time;
        accumulator = next_accumulator;
    }

    TimingBreakdown {
        operation,
        key_generation: Some(key_generation),
        keyswitching: Some(keyswitching),
        rescaling: Some(rescaling),
    }
}

fn time_ciphertext_ciphertext_dot_product(
    context: &RnsCkksContext,
    lhs: &RnsCiphertext,
    rhs: &RnsCiphertext,
    key_pair: &ckks::rns::RnsKeyPair,
    active_slots: usize,
) -> TimingBreakdown {
    let (relin_key, relin_key_generation) = time(|| {
        context.generate_relinearization_key(&key_pair.secret_key, &key_pair.public_key)
    });

    let (quadratic, mult_time) = time(|| context.mult(lhs, rhs));
    let (relinearized, relin_keyswitch, relinearize_overhead) = timed_relinearize(context, &quadratic, &relin_key);
    let (hadamard, rescaling) = time(|| context.rescale(&relinearized));

    let (rotation_keys, rotation_key_generation) = time(|| {
        generate_halevi_shoup_dot_product_rotation_keys(
            context,
            &key_pair.secret_key,
            &key_pair.public_key,
            hadamard.level,
            active_slots,
        )
    });

    let mut accumulator = hadamard;
    let mut operation = mult_time + relinearize_overhead;
    let mut keyswitching = relin_keyswitch;
    for step in halevi_shoup_dot_product_rotation_steps(active_slots) {
        let rotation_key = rotation_keys
            .get(&step)
            .unwrap_or_else(|| panic!("missing rotation key for step {step}"));
        let (rotated, rotate_operation, rotate_keyswitch) = timed_rotate(context, &accumulator, rotation_key);
        keyswitching += rotate_keyswitch;
        operation += rotate_operation;
        let (next_accumulator, add_time) = time(|| accumulator.add(&rotated));
        operation += add_time;
        accumulator = next_accumulator;
    }

    TimingBreakdown {
        operation,
        key_generation: Some(relin_key_generation + rotation_key_generation),
        keyswitching: Some(keyswitching),
        rescaling: Some(rescaling),
    }
}

fn time_plain_matrix_ciphertext_product(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    matrix: &[Vec<f64>],
    key_pair: &ckks::rns::RnsKeyPair,
) -> TimingBreakdown {
    let (rotation_keys, key_generation) = time(|| {
        generate_packed_diagonal_rotation_keys(
            context,
            &key_pair.secret_key,
            &key_pair.public_key,
            ciphertext.level,
            matrix.len(),
        )
    });
    let diagonals = pack_diagonal_plaintexts(context, matrix, ciphertext.level, ciphertext.scale_bits);

    let (initial_mult, initial_mult_time) = time(|| context.mult_plain(ciphertext, &diagonals.diagonals[0]));
    let (mut accumulator, initial_rescale) = time(|| context.rescale(&initial_mult));

    let mut operation = initial_mult_time;
    let mut keyswitching = Duration::ZERO;
    let mut rescaling = initial_rescale;
    for step in packed_diagonal_rotation_steps(diagonals.dimension) {
        let rotation_key = rotation_keys
            .by_step
            .get(&step)
            .unwrap_or_else(|| panic!("missing input rotation key for step {step}"));
        let (rotated, rotate_operation, rotate_keyswitch) = timed_rotate(context, ciphertext, rotation_key);
        keyswitching += rotate_keyswitch;
        operation += rotate_operation;

        let (term_mult, term_mult_time) = time(|| context.mult_plain(&rotated, &diagonals.diagonals[step]));
        let (term, term_rescale) = time(|| context.rescale(&term_mult));
        let (next_accumulator, add_time) = time(|| accumulator.add(&term));

        operation += term_mult_time + add_time;
        rescaling += term_rescale;
        accumulator = next_accumulator;
    }

    TimingBreakdown {
        operation,
        key_generation: Some(key_generation),
        keyswitching: Some(keyswitching),
        rescaling: Some(rescaling),
    }
}

fn timed_rotate(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    rotation_key: &RnsRotationKey,
) -> (RnsCiphertext, Duration, Duration) {
    let (raw_rotated, raw_operation) = time(|| ciphertext.apply_galois_automorphism(rotation_key.galois_element));
    let (switched, keyswitching) = time(|| context.keyswitch(&raw_rotated.c1, &rotation_key.keyswitch_key));
    let (rotated, combine_time) = time(|| {
        RnsCiphertext::new(
            raw_rotated.c0.add(&switched.c0),
            switched.c1,
            raw_rotated.scale_bits,
        )
    });
    (rotated, raw_operation + combine_time, keyswitching)
}

fn timed_relinearize(
    context: &RnsCkksContext,
    quadratic: &ckks::rns::RnsQuadraticCiphertext,
    relin_key: &RnsKeySwitchingKey,
) -> (RnsCiphertext, Duration, Duration) {
    let (switched, keyswitching) = time(|| context.keyswitch(&quadratic.c2, relin_key));
    let (relinearized, combine_time) = time(|| {
        RnsCiphertext::new(
            quadratic.c0.add(&switched.c0),
            quadratic.c1.add(&switched.c1),
            quadratic.scale_bits,
        )
    });
    (relinearized, keyswitching, combine_time)
}

fn print_breakdown(label: &str, breakdown: &TimingBreakdown) {
    println!("{label}");
    println!("  operation: {}", format_duration(breakdown.operation));
    println!(
        "  keyswitch key generation: {}",
        breakdown
            .key_generation
            .map(format_duration)
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!(
        "  keyswitching: {}",
        breakdown
            .keyswitching
            .map(format_duration)
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!(
        "  rescaling: {}",
        breakdown
            .rescaling
            .map(format_duration)
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!();
}

fn format_duration(duration: Duration) -> String {
    format!("{:.3} ms", duration.as_secs_f64() * 1_000.0)
}

fn time<T>(operation: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let value = operation();
    (value, start.elapsed())
}