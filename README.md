# RNS CKKS in Rust

A Rust implementation of RNS CKKS (Cheon, Kim, Kim, Song) homomorphic encryption.

The current codebase focuses on:

- RNS modulus-chain contexts
- CKKS encode and decode
- Encrypt and decrypt round trips
- Ciphertext-plaintext Hadamard multiplication
- Ciphertext-ciphertext Hadamard multiplication
- Relinearization 
- NTT-backed polynomial-polynomial multiplication

## Running tests

```bash
cargo test -- --nocapture
```

## Running examples

```bash
cargo run --example rns_round_trip
cargo run --example rns_addition
cargo run --example rns_hadamard_product
cargo run --example rns_halevi_shoup_dot_product
cargo run --example rns_packed_diagonal_matvec
cargo run --example rns_paterson_stockmeyer
cargo run --example rns_total_sum
cargo run --example timings
```

## Layout

- `src/rns/`: RNS CKKS implementation
- `src/algorithms/`: higher-level encrypted linear algebra and polynomial evaluation
- `src/math/`: NTT and canonical embedding helpers
- `src/poly/`: modular polynomial arithmetic
- `examples/operations/`: core CKKS operation examples
- `examples/algorithmics/`: higher-level CKKS algorithm examples
- `examples/ML/`: placeholder for upcoming ML-oriented examples
- `tests/`: integration tests for math primitives

## License

This project is licensed under the MIT License. See the LICENSE file for details.