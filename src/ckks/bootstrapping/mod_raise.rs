use crate::rns::keyswitching::lift_poly_to_context;
use crate::rns::{RnsCiphertext, RnsCkksContext, RnsSecretKey};

#[derive(Clone, Debug)]
pub struct ModRaisedCiphertext {
    pub ciphertext: RnsCiphertext,
    q_level: usize,
    aux_level_count: usize,
}

impl ModRaisedCiphertext {
    pub fn q_level(&self) -> usize {
        self.q_level
    }

    pub fn aux_level_count(&self) -> usize {
        self.aux_level_count
    }

    pub fn eval_level(&self) -> usize {
        self.ciphertext.level
    }

    pub fn project_to_q_ciphertext(&self) -> RnsCiphertext {
        self.ciphertext.truncate_levels(self.q_level + 1)
    }
}

pub fn mod_raise(context: &RnsCkksContext, ciphertext: &RnsCiphertext) -> ModRaisedCiphertext {
    let q_context = context
        .level_context(ciphertext.level)
        .expect("ciphertext level context must exist for ModRaise");
    let eval_key_context = context
        .eval_key_level_context(ciphertext.level)
        .expect("ModRaise requires an auxiliary evaluation basis");

    let raised_c0 = lift_poly_to_context(&ciphertext.c0, &q_context, &eval_key_context);
    let raised_c1 = lift_poly_to_context(&ciphertext.c1, &q_context, &eval_key_context);

    ModRaisedCiphertext {
        ciphertext: RnsCiphertext::new(raised_c0, raised_c1, ciphertext.scale_bits),
        q_level: ciphertext.level,
        aux_level_count: eval_key_context.levels() - q_context.levels(),
    }
}

//Used for testing only
pub fn decrypt_mod_raised(
    context: &RnsCkksContext,
    ciphertext: &ModRaisedCiphertext,
    secret_key: &RnsSecretKey,
) -> crate::rns::RnsPolynomial {
    let eval_key_context = context
        .eval_key_level_context(ciphertext.q_level)
        .expect("ModRaise decryption requires an auxiliary evaluation basis");
    let q_context = context
        .level_context(ciphertext.q_level)
        .expect("ciphertext Q level context must exist for ModRaise decryption");
    let lifted_secret = RnsSecretKey::new(lift_poly_to_context(
        &secret_key.poly.truncate_levels(ciphertext.q_level + 1),
        &q_context,
        &eval_key_context,
    ));

    crate::rns::ops::decrypt_rns(&ciphertext.ciphertext, &lifted_secret, &eval_key_context)
}

#[cfg(test)]
mod tests {
    use super::{decrypt_mod_raised, mod_raise};
    use crate::rns::{RnsCkksContext, RnsCkksParams};

    #[test]
    fn mod_raise_preserves_round_trip_on_realistic_params() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let input = [0.125_f64, -0.4, 0.03125, -0.125];
        let mut padded = vec![0.0_f64; context.num_slots()];
        padded[..input.len()].copy_from_slice(&input);
        let ciphertext = context.encrypt(&context.encode_real(&padded), &key_pair.public_key);

        let raised = mod_raise(&context, &ciphertext);
        assert_eq!(raised.q_level(), ciphertext.level);
        assert!(raised.aux_level_count() > 0);

        let decrypted = decrypt_mod_raised(&context, &raised, &key_pair.secret_key);
        let recovered = context.decode_real_at_scale(
            &decrypted.truncate_levels(raised.q_level() + 1),
            raised.ciphertext.scale_bits,
        );

        for (index, &expected) in input.iter().enumerate() {
            assert!(
                (recovered[index] - expected).abs() <= 2e-2,
                "expected {expected}, got {} at slot {index}",
                recovered[index]
            );
        }
    }
}