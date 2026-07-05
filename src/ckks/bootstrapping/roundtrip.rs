use crate::ckks::bootstrapping::coeff_to_slot::{coeff_to_slot, CoeffToSlotPlan, CoeffToSlotPrecomputed};
use crate::ckks::bootstrapping::linear_transform::generate_diagonal_transform_rotation_keys;
use crate::ckks::bootstrapping::mod_raise::{mod_raise, ModRaisedCiphertext};
use crate::ckks::bootstrapping::slot_to_coeff::{slot_to_coeff, SlotToCoeffPlan, SlotToCoeffPrecomputed};
use crate::rns::{RnsCiphertext, RnsCkksContext};

pub fn bootstrap_identity_roundtrip(
    context: &RnsCkksContext,
    ciphertext: &RnsCiphertext,
    key_pair: &crate::rns::RnsKeyPair,
    active_slots: usize,
) -> Result<RnsCiphertext, String> {
    let raised = mod_raise(context, ciphertext);
    bootstrap_identity_roundtrip_from_mod_raise(context, &raised, key_pair, active_slots)
}

pub fn bootstrap_identity_roundtrip_from_mod_raise(
    context: &RnsCkksContext,
    raised: &ModRaisedCiphertext,
    key_pair: &crate::rns::RnsKeyPair,
    active_slots: usize,
) -> Result<RnsCiphertext, String> {
    let projected = raised.project_to_q_ciphertext();

    let coeff_to_slot_plan = CoeffToSlotPlan::from_ciphertext(context, &projected, active_slots)?;
    let identity = identity_matrix(active_slots);

    let coeff_to_slot_precomputed = CoeffToSlotPrecomputed::new(
        context,
        coeff_to_slot_plan,
        &identity,
        projected.scale_bits,
    )?;
    let coeff_to_slot_rotation_keys = generate_diagonal_transform_rotation_keys(
        context,
        &key_pair.secret_key,
        &key_pair.public_key,
        projected.level,
        active_slots,
    );

    let slots = coeff_to_slot(
        context,
        &projected,
        &coeff_to_slot_precomputed,
        &coeff_to_slot_rotation_keys,
    )?;
    let slot_to_coeff_plan = SlotToCoeffPlan::from_ciphertext(context, &slots, active_slots)?;
    let slot_to_coeff_precomputed = SlotToCoeffPrecomputed::new(
        context,
        slot_to_coeff_plan,
        &identity,
        slots.scale_bits,
    )?;
    let slot_to_coeff_rotation_keys = generate_diagonal_transform_rotation_keys(
        context,
        &key_pair.secret_key,
        &key_pair.public_key,
        slots.level,
        active_slots,
    );

    slot_to_coeff(
        context,
        &slots,
        &slot_to_coeff_precomputed,
        &slot_to_coeff_rotation_keys,
    )
}

fn identity_matrix(dimension: usize) -> Vec<Vec<f64>> {
    (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|col| if row == col { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::bootstrap_identity_roundtrip;
    use crate::ckks::bootstrapping::linear_transform::repeat_block_slots;
    use crate::rns::{RnsCkksContext, RnsCkksParams};

    #[test]
    fn bootstrap_identity_roundtrip_preserves_q_chain_message() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");
        let key_pair = context.keygen(64);

        let input = vec![0.5_f64, -0.25, 0.125, 0.75, -0.5, 0.25, -0.125, 0.375];
        let packed = repeat_block_slots(&input, context.num_slots());
        let ciphertext = context.encrypt(&context.encode_real(&packed), &key_pair.public_key);

        let recovered_ct = bootstrap_identity_roundtrip(&context, &ciphertext, &key_pair, input.len())
            .expect("identity bootstrapping roundtrip must succeed");
        let recovered = context.decode_real_at_scale(
            &context.decrypt(&recovered_ct, &key_pair.secret_key),
            recovered_ct.scale_bits,
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