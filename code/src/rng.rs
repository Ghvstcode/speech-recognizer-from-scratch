//! A tiny pseudo-random number generator.
//!
//! We need randomness to initialise weights, shuffle training data and add
//! noise. We do not need it to be cryptographically secure; we need it to be
//! fast, decent, and *reproducible*: the same seed must give the same run.
//!
//! The generator is xoshiro256**, seeded with SplitMix64. Both are public
//! domain algorithms by Blackman and Vigna and fit in a few lines.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Create a generator from a seed. Any seed works, including 0.
    pub fn new(seed: u64) -> Self {
        // SplitMix64 stretches one 64-bit seed into the 256 bits of state
        // xoshiro needs, and guarantees the state is never all zeros.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// A uniform float in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        // Use the top 24 bits: an f32 has 24 bits of mantissa.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A uniform float in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// A uniform integer in `[0, n)`. `n` must be greater than zero.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0) has no valid answer");
        // The modulo bias is negligible for the sizes we use (n << 2^64).
        (self.next_u64() % n as u64) as usize
    }

    /// A sample from the standard normal distribution (mean 0, variance 1),
    /// using the Box-Muller transform.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7); // avoid ln(0)
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Shuffle a slice in place (Fisher-Yates).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }

    /// Split off an independent generator, e.g. to hand one to each thread.
    pub fn fork(&mut self) -> Rng {
        Rng::new(self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let (mut a, mut b) = (Rng::new(42), Rng::new(42));
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_stays_in_range_and_looks_uniform() {
        let mut rng = Rng::new(1);
        let n = 100_000;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let x = rng.uniform();
            assert!((0.0..1.0).contains(&x));
            sum += x as f64;
        }
        assert!((sum / n as f64 - 0.5).abs() < 0.01);
    }

    #[test]
    fn normal_has_roughly_zero_mean_and_unit_variance() {
        let mut rng = Rng::new(7);
        let n = 200_000;
        let (mut sum, mut sq) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let x = rng.normal() as f64;
            sum += x;
            sq += x * x;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }

    #[test]
    fn shuffle_keeps_every_element() {
        let mut rng = Rng::new(3);
        let mut v: Vec<usize> = (0..50).collect();
        rng.shuffle(&mut v);
        assert_ne!(v, (0..50).collect::<Vec<_>>());
        v.sort();
        assert_eq!(v, (0..50).collect::<Vec<_>>());
    }
}
