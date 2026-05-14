use super::polynomial::RnsPolynomial;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RnsCiphertext {
    pub c0: RnsPolynomial,
    pub c1: RnsPolynomial,
    pub scale_bits: usize,
    pub level: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RnsQuadraticCiphertext {
    pub c0: RnsPolynomial,
    pub c1: RnsPolynomial,
    pub c2: RnsPolynomial,
    pub scale_bits: usize,
    pub level: usize,
}

impl RnsQuadraticCiphertext {
    pub fn new(c0: RnsPolynomial, c1: RnsPolynomial, c2: RnsPolynomial, scale_bits: usize) -> Self {
        assert_eq!(c0.levels(), c1.levels(), "quadratic ciphertext components must share the same level count");
        assert_eq!(c0.levels(), c2.levels(), "quadratic ciphertext components must share the same level count");
        assert_eq!(c0.degree(), c1.degree(), "quadratic ciphertext components must share the same degree");
        assert_eq!(c0.degree(), c2.degree(), "quadratic ciphertext components must share the same degree");
        let level = c0.levels() - 1;
        Self {
            c0,
            c1,
            c2,
            scale_bits,
            level,
        }
    }

    pub fn apply_galois_automorphism(&self, galois_element: usize) -> Self {
        Self {
            c0: self.c0.apply_galois_automorphism(galois_element),
            c1: self.c1.apply_galois_automorphism(galois_element),
            c2: self.c2.apply_galois_automorphism(galois_element),
            scale_bits: self.scale_bits,
            level: self.level,
        }
    }
}

impl RnsCiphertext {
    pub fn new(c0: RnsPolynomial, c1: RnsPolynomial, scale_bits: usize) -> Self {
        assert_eq!(c0.levels(), c1.levels(), "ciphertext components must have the same number of RNS levels");
        assert_eq!(c0.degree(), c1.degree(), "ciphertext components must have the same degree");
        let level = c0.levels() - 1;
        Self {
            c0,
            c1,
            scale_bits,
            level,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.scale_bits, other.scale_bits, "ciphertexts must share the same scale to add");
        assert_eq!(self.level, other.level, "ciphertexts must be at the same level to add");
        Self {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
            scale_bits: self.scale_bits,
            level: self.level,
        }
    }

    pub fn apply_galois_automorphism(&self, galois_element: usize) -> Self {
        Self {
            c0: self.c0.apply_galois_automorphism(galois_element),
            c1: self.c1.apply_galois_automorphism(galois_element),
            scale_bits: self.scale_bits,
            level: self.level,
        }
    }

    pub fn drop_last_level(&self) -> Self {
        assert!(self.level > 0, "cannot drop the last ciphertext level");
        Self {
            c0: self.c0.drop_last_level(),
            c1: self.c1.drop_last_level(),
            scale_bits: self.scale_bits,
            level: self.level - 1,
        }
    }

    pub fn truncate_levels(&self, level_count: usize) -> Self {
        assert!(level_count > 0, "level_count must be positive");
        assert!(level_count <= self.c0.levels(), "level_count cannot exceed ciphertext levels");
        Self {
            c0: self.c0.truncate_levels(level_count),
            c1: self.c1.truncate_levels(level_count),
            scale_bits: self.scale_bits,
            level: level_count - 1,
        }
    }
}