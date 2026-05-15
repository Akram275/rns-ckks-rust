//This file contains the implementation of the RnsCKKSContext structure object
//It is an encapsulation of the parameters set and the NTT plan, 
// which will be used to generate the key set and encoder/decoder objects. 
//

use super::{RnsCkksContext, RnsCkksParams, RnsContext};

impl RnsCkksContext {
    pub fn new(params: RnsCkksParams) -> Result<Self, String> {
        params.validate()?;
        let rns_context = RnsContext::new(params.poly_degree, params.primes.clone())?;
        let aux_context = if params.aux_primes.is_empty() {
            None
        } else {
            Some(RnsContext::new(params.poly_degree, params.aux_primes.clone())?)
        };
        let eval_key_context = if params.aux_primes.is_empty() {
            None
        } else {
            let mut extended_primes = params.primes.clone();
            extended_primes.extend(params.aux_primes.clone());
            Some(RnsContext::new(params.poly_degree, extended_primes)?)
        };
        Ok(Self {
            params,
            rns_context,
            aux_context,
            eval_key_context,
        })
    }

    pub fn params(&self) -> &RnsCkksParams {
        &self.params
    }

    pub fn rns_context(&self) -> &RnsContext {
        &self.rns_context
    }

    pub fn aux_context(&self) -> Option<&RnsContext> {
        self.aux_context.as_ref()
    }

    pub fn eval_key_context(&self) -> Option<&RnsContext> {
        self.eval_key_context.as_ref()
    }

    pub fn has_aux_basis(&self) -> bool {
        self.aux_context.is_some()
    }

    pub fn num_slots(&self) -> usize {
        self.params.poly_degree / 2
    }

    pub fn levels(&self) -> usize {
        self.rns_context.levels()
    }

    pub fn total_modulus_bits(&self) -> u64 {
        self.rns_context.total_modulus_bits()
    }

    pub fn aux_modulus_bits(&self) -> Option<u64> {
        self.aux_context.as_ref().map(RnsContext::total_modulus_bits)
    }

    pub fn eval_key_modulus_bits(&self) -> Option<u64> {
        self.eval_key_context.as_ref().map(RnsContext::total_modulus_bits)
    }

    pub fn level_context(&self, level: usize) -> Result<RnsContext, String> {
        self.rns_context.at_level(level + 1)
    }

    pub fn eval_key_level_context(&self, level: usize) -> Result<RnsContext, String> {
        if self.params.aux_primes.is_empty() {
            return Err("eval-key level context requires an auxiliary basis".to_string());
        }
        if level >= self.params.primes.len() {
            return Err(format!(
                "level must be in 0..{}, got {level}",
                self.params.primes.len()
            ));
        }

        let mut primes = self.params.primes[..level + 1].to_vec();
        primes.extend(self.params.aux_primes.clone());
        RnsContext::new(self.params.poly_degree, primes)
    }
}