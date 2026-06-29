use crate::rns::{RnsCiphertext, RnsCkksContext};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootstrapParameters {
    pub ciphertext_level: usize,
    pub q_level_count: usize,
    pub eval_level_count: usize,
    pub aux_level_count: usize,
    pub slot_count: usize,
    pub active_slots: usize,
}

impl BootstrapParameters {
    pub fn new(
        context: &RnsCkksContext,
        ciphertext_level: usize,
        active_slots: usize,
    ) -> Result<Self, String> {
        if !context.has_aux_basis() {
            return Err("bootstrapping requires an auxiliary evaluation basis".to_string());
        }
        if ciphertext_level >= context.levels() {
            return Err(format!(
                "ciphertext level must be in 0..{}, got {ciphertext_level}",
                context.levels()
            ));
        }
        if active_slots == 0 {
            return Err("active_slots must be positive".to_string());
        }
        if !active_slots.is_power_of_two() {
            return Err("active_slots must be a power of two".to_string());
        }
        if active_slots > context.num_slots() {
            return Err(format!(
                "active_slots cannot exceed the CKKS slot count {}; got {active_slots}",
                context.num_slots()
            ));
        }

        let q_level_count = ciphertext_level + 1;
        let eval_level_count = context.eval_key_level_context(ciphertext_level)?.levels();
        let aux_level_count = eval_level_count - q_level_count;

        Ok(Self {
            ciphertext_level,
            q_level_count,
            eval_level_count,
            aux_level_count,
            slot_count: context.num_slots(),
            active_slots,
        })
    }

    pub fn from_ciphertext(
        context: &RnsCkksContext,
        ciphertext: &RnsCiphertext,
        active_slots: usize,
    ) -> Result<Self, String> {
        Self::new(context, ciphertext.level, active_slots)
    }
}

#[cfg(test)]
mod tests {
    use super::BootstrapParameters;
    use crate::rns::{RnsCkksContext, RnsCkksParams};

    #[test]
    fn realistic_bootstrap_parameters_capture_q_and_eval_sizes() {
        let context = RnsCkksContext::new(RnsCkksParams::realistic_8_level())
            .expect("realistic RNS CKKS params must build");

        let params = BootstrapParameters::new(&context, 7, 8)
            .expect("realistic preset must support bootstrap parameters");

        assert_eq!(params.ciphertext_level, 7);
        assert_eq!(params.q_level_count, 8);
        assert_eq!(params.eval_level_count, 10);
        assert_eq!(params.aux_level_count, 2);
        assert_eq!(params.active_slots, 8);
    }
}