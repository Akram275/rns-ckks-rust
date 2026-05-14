mod construction;
mod encoding;
mod keys;
mod ops;
mod params;
mod rotation;

use super::basis::{RnsContext, RnsPrime};
use super::ciphertexts::{RnsCiphertext, RnsQuadraticCiphertext};
use super::keys::{RnsKeyPair, RnsPublicKey, RnsSecretKey};
use super::keyswitching::RnsKeySwitchingKey;
use super::polynomial::RnsPolynomial;
use super::RNS_DECOMP_BITS;

#[derive(Clone, Debug)]
pub struct RnsCkksParams {
    pub poly_degree: usize,
    pub primes: Vec<RnsPrime>,
    pub aux_primes: Vec<RnsPrime>,
    pub scale_bits: usize,
}

#[derive(Clone, Debug)]
pub struct RnsCkksContext {
    params: RnsCkksParams,
    rns_context: RnsContext,
    aux_context: Option<RnsContext>,
    eval_key_context: Option<RnsContext>,
}