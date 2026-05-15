use ckks::algorithms::packed_diagonal::{
    generate_packed_diagonal_matrix_vector_rotation_keys,
    plaintext_matrix_ciphertext_vector_product,
    repeat_real_vector_in_slots,
};
use ckks::rns::{RnsCkksContext, RnsCkksParams};

fn main() {
    let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
        .expect("realistic RNS CKKS parameters must build");
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

    let packed_vector = repeat_real_vector_in_slots(&vector, context.num_slots());
    let ciphertext = context.encrypt(&context.encode_real(&packed_vector), &key_pair.public_key);
    let rotation_keys = generate_packed_diagonal_matrix_vector_rotation_keys(
        &context,
        &key_pair.secret_key,
        &key_pair.public_key,
        ciphertext.level,
        matrix.len(),
    );

    let product = plaintext_matrix_ciphertext_vector_product(
        &context,
        &ciphertext,
        &matrix,
        &rotation_keys,
    );
    let recovered = context.decode_real_at_scale(
        &context.decrypt(&product, &key_pair.secret_key),
        product.scale_bits,
    );

    println!("RNS CKKS packed-diagonal plaintext-matrix encrypted-vector product");
    println!("expected first block = {:?}", expected);
    println!("recovered first block = {:?}", &recovered[..matrix.len()]);
}