# RNS CKKS in Rust

This repository contains a Rust implementation of an RNS-based CKKS homomorphic encryption scaffold with realistic parameter support.

The current codebase focuses on:

- RNS modulus-chain contexts
- CKKS encode and decode
- Encrypt and decrypt round trips
- Ciphertext-plaintext Hadamard multiplication
- Ciphertext-ciphertext Hadamard multiplication
- Relinearization scaffolding
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
```

## Layout

- `src/rns/`: RNS CKKS implementation
- `src/math/`: NTT and canonical embedding helpers
- `src/poly/`: modular polynomial arithmetic
- `examples/`: realistic RNS examples
- `tests/`: integration tests for math primitives