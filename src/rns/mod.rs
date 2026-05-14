pub(crate) const RNS_DECOMP_BITS: usize = 30;

mod basis;
mod ciphertexts;
mod context;
mod encoding;
mod keys;
mod keyswitching;
mod ops;
mod polynomial;

pub use basis::{RnsContext, RnsPrime};
pub use ciphertexts::{RnsCiphertext, RnsQuadraticCiphertext};
pub use context::{RnsCkksContext, RnsCkksParams};
pub use keys::{RnsKeyPair, RnsPublicKey, RnsSecretKey};
pub use keyswitching::RnsKeySwitchingKey;
pub use polynomial::RnsPolynomial;

#[cfg(test)]
mod tests;
