//! Number-theoretic transform helpers for modular polynomial multiplication.

/// Plan for a radix-2 NTT over a prime modulus.
#[derive(Clone, Debug)]
pub struct NttPlan {
    modulus: u64,
    size: usize,
    root: u64,
    root_inv: u64,
    negacyclic_twiddle: u64,
    negacyclic_twiddle_inv: u64,
    size_inv: u64,
}

impl NttPlan {
    /// Build an NTT plan for a power-of-two transform size.
    pub fn new(size: usize, modulus: u64, primitive_root: u64) -> Result<Self, String> {
        if size == 0 || !size.is_power_of_two() {
            return Err("NTT size must be a non-zero power of two".to_string());
        }
        if modulus < 3 {
            return Err("modulus must be an odd prime larger than 2".to_string());
        }
        if (modulus - 1) % size as u64 != 0 {
            return Err("size must divide modulus - 1".to_string());
        }
        if (modulus - 1) % (2 * size as u64) != 0 {
            return Err("2 * size must divide modulus - 1 for negacyclic NTTs".to_string());
        }

        let exponent = (modulus - 1) / size as u64;
        let root = mod_pow(primitive_root % modulus, exponent, modulus);
        if mod_pow(root, size as u64, modulus) != 1 {
            return Err("primitive_root does not generate a size-th root of unity".to_string());
        }
        if size > 1 && mod_pow(root, (size / 2) as u64, modulus) == 1 {
            return Err("root is not primitive for the requested size".to_string());
        }

        let root_inv = mod_inv(root, modulus).ok_or_else(|| "root is not invertible".to_string())?;
        let size_inv = mod_inv(size as u64, modulus).ok_or_else(|| "size is not invertible modulo modulus".to_string())?;
        let negacyclic_twiddle = mod_pow(primitive_root % modulus, (modulus - 1) / (2 * size as u64), modulus);
        if mod_pow(negacyclic_twiddle, (2 * size) as u64, modulus) != 1
            || mod_pow(negacyclic_twiddle, size as u64, modulus) != modulus - 1
        {
            return Err("primitive_root does not generate a valid negacyclic twiddle".to_string());
        }
        let negacyclic_twiddle_inv = mod_inv(negacyclic_twiddle, modulus)
            .ok_or_else(|| "negacyclic twiddle is not invertible".to_string())?;

        Ok(Self {
            modulus,
            size,
            root,
            root_inv,
            negacyclic_twiddle,
            negacyclic_twiddle_inv,
            size_inv,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    pub fn negacyclic_twiddle(&self) -> u64 {
        self.negacyclic_twiddle
    }

    pub fn negacyclic_twiddle_inv(&self) -> u64 {
        self.negacyclic_twiddle_inv
    }

    pub fn forward(&self, values: &mut [u64]) {
        assert_eq!(values.len(), self.size, "NTT input length mismatch");
        ntt_in_place(values, self.modulus, self.root);
    }

    pub fn inverse(&self, values: &mut [u64]) {
        assert_eq!(values.len(), self.size, "INTT input length mismatch");
        ntt_in_place(values, self.modulus, self.root_inv);
        for value in values.iter_mut() {
            *value = mod_mul(*value, self.size_inv, self.modulus);
        }
    }
}

fn ntt_in_place(values: &mut [u64], modulus: u64, root: u64) {
    let n = values.len();
    bit_reverse_permute(values);

    let mut len = 2;
    while len <= n {
        let step = n / len;
        let w_len = mod_pow(root, step as u64, modulus);

        for chunk_start in (0..n).step_by(len) {
            let mut w = 1u64;
            for i in 0..(len / 2) {
                let u = values[chunk_start + i];
                let v = mod_mul(values[chunk_start + i + len / 2], w, modulus);
                values[chunk_start + i] = mod_add(u, v, modulus);
                values[chunk_start + i + len / 2] = mod_sub(u, v, modulus);
                w = mod_mul(w, w_len, modulus);
            }
        }

        len *= 2;
    }
}

fn bit_reverse_permute(values: &mut [u64]) {
    let n = values.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            values.swap(i, j);
        }
    }
}

fn mod_add(a: u64, b: u64, modulus: u64) -> u64 {
    let sum = a as u128 + b as u128;
    if sum >= modulus as u128 {
        (sum - modulus as u128) as u64
    } else {
        sum as u64
    }
}

fn mod_sub(a: u64, b: u64, modulus: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        modulus - (b - a)
    }
}

fn mod_mul(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, modulus);
        }
        base = mod_mul(base, base, modulus);
        exp >>= 1;
    }
    result
}

fn mod_inv(value: u64, modulus: u64) -> Option<u64> {
    if value == 0 {
        return None;
    }

    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (modulus as i128, value as i128);

    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }

    if r != 1 {
        return None;
    }

    if t < 0 {
        t += modulus as i128;
    }

    Some((t % modulus as i128) as u64)
}

#[cfg(test)]
mod tests {
    use super::NttPlan;

    #[test]
    fn ntt_round_trip_restores_values() {
        let plan = NttPlan::new(8, 998_244_353, 3).expect("valid NTT plan");
        let mut values = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let original = values.clone();

        plan.forward(&mut values);
        plan.inverse(&mut values);

        assert_eq!(values, original);
    }

    #[test]
    fn ntt_plan_rejects_modulus_without_negacyclic_support() {
        let err = NttPlan::new(8, 41, 6).expect_err("41 only supports an 8-point cyclic root, not a 16-th negacyclic twiddle");

        assert!(err.contains("2 * size must divide modulus - 1"));
    }
}