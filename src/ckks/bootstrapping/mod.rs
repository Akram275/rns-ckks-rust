//! CKKS bootstrapping namespace.
//!
//! This module will host the shared infrastructure and concrete variants for
//! CKKS bootstrapping, starting from the canonical ModRaise -> CoeffToSlot ->
//! EvalMod -> SlotToCoeff decomposition.

pub mod keys;
pub mod mod_raise;
pub mod params;

pub use keys::BootstrapKeySet;
pub use params::BootstrapParameters;