//! CoeffToSlot scaffolding for CKKS bootstrapping.
//!
//! This module will host the coefficient-to-slot linear transform plan built on
//! top of the shared diagonal linear-transform engine.
//!
//! Intended contents:
//! - transform-plan parameters for the active slot regime,
//! - prepacked diagonal plaintexts for the forward transform,
//! - rotation-key requirements for the transform execution,
//! - the eventual `coeff_to_slot(...)` evaluator entry point.

/// Static planning data for a CoeffToSlot transform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoeffToSlotPlan {}

/// Precomputed diagonal plaintext data for executing a CoeffToSlot transform.
#[derive(Clone, Debug)]
pub struct CoeffToSlotPrecomputed {}

/// Runtime configuration for applying a CoeffToSlot transform to a ciphertext.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoeffToSlotRuntime {}