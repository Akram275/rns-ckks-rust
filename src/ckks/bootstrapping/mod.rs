//! CKKS bootstrapping namespace.
//!
//! This module will host the shared infrastructure and concrete variants for
//! CKKS bootstrapping, starting from the canonical ModRaise -> CoeffToSlot ->
//! EvalMod -> SlotToCoeff decomposition.