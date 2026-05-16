//! High-level FHE algorithms built on top of the RNS CKKS primitives.
//!
//! These routines assume the caller already chose a compatible data layout.
//! In particular, each module documents the packing contract expected by its
//! rotations and plaintext masks.

pub mod halevi_shoup;
pub mod paterson_stockmeyer;
pub mod packed_diagonal;