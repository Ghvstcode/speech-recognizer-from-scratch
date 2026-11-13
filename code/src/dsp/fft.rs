//! Complex numbers, the discrete Fourier transform, and the FFT.
//!
//! The DFT answers one question about a block of `n` samples: *how much of
//! each frequency is in here?* It does so by comparing the block against `n`
//! sinusoids, the ones that complete exactly 0, 1, 2, ... `n - 1` cycles over
//! the block. The comparison is a sum of products (a correlation), and to
//! compare against a cosine and a sine of the same frequency in one go we use
//! a complex number: `e^(-i a) = cos(a) - i sin(a)`.
//!
//! ```text
//! X[k] = sum over t of  x[t] * e^(-2 pi i k t / n)        k = 0 .. n-1
//! ```
//!
//! `X[k]` is called *bin* `k`. With a sample rate of `sr`, bin `k` stands for
//! the frequency `k * sr / n` Hz. Its length `|X[k]|` says how strong that
//! frequency is and its angle says where in its cycle it starts.
//!
//! # Scaling convention
//!
//! As in most libraries, the forward transform ([`dft`], [`fft`]) is the bare
//! sum above with no scaling, and the inverse ([`ifft`]) divides by `n`. So:
//!
//! * a constant signal of value `c` gives `X[0] = c * n`;
//! * a real sine of amplitude `a` that completes exactly `k` cycles gives
//!   `|X[k]| = |X[n-k]| = a * n / 2` (its energy is split between a positive
//!   and a negative frequency), and nothing anywhere else;
//! * Parseval: `sum |x[t]|^2 = (1/n) sum |X[k]|^2`.

use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

/// A complex number `re + i * im`, with just the operations the FFT needs.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Complex {
    /// The real part: how much the signal looks like a cosine.
    pub re: f32,
    /// The imaginary part: how much it looks like a (negated) sine.
    pub im: f32,
}

impl Complex {
    /// The number 0.
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    /// `re + i * im`.
    pub fn new(re: f32, im: f32) -> Self {
        Complex { re, im }
    }

    /// The point on the unit circle at `angle` radians:
    /// `e^(i angle) = cos(angle) + i sin(angle)`.
    ///
    /// The angle is an `f64` because for large transforms the angles get very
    /// close together, and we want the rounding to happen once, at the end.
    pub fn from_angle(angle: f64) -> Self {
        Complex { re: angle.cos() as f32, im: angle.sin() as f32 }
    }

    /// The complex conjugate: the same point mirrored in the real axis.
    pub fn conj(self) -> Self {
        Complex { re: self.re, im: -self.im }
    }

    /// The squared length `re^2 + im^2`. For a spectrum bin this is its *power*.
    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// The length (magnitude) `sqrt(re^2 + im^2)`.
    pub fn abs(self) -> f32 {
        self.norm_sqr().sqrt()
    }

    /// Multiply by a real number.
    pub fn scale(self, k: f32) -> Self {
        Complex { re: self.re * k, im: self.im * k }
    }
}

impl Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex { re: self.re + o.re, im: self.im + o.im }
    }
}

impl Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex { re: self.re - o.re, im: self.im - o.im }
    }
}

impl Mul for Complex {
    type Output = Complex;
    /// `(a + ib)(c + id) = (ac - bd) + i(ad + bc)`, because `i * i = -1`.
    /// Geometrically: lengths multiply and angles add.
    fn mul(self, o: Complex) -> Complex {
        Complex { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}

/// The discrete Fourier transform, straight from the definition.
///
/// Two nested loops, `n * n` complex multiplications: fine for 256 samples,
/// painful for 4096. It works for any length, and it is the reference the
/// fast version is tested against.
pub fn dft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let mut output = Vec::with_capacity(n);
    for k in 0..n {
        // Correlate the input with a sinusoid that makes k turns over the block.
        let mut sum = Complex::ZERO;
        for (t, &x) in input.iter().enumerate() {
            // The angle is -2 pi k t / n. Only the position on the circle
            // matters, so we reduce k*t modulo n first, with exact integer
            // arithmetic, and keep the angle small and accurate.
            let turns = ((k * t) % n) as f64 / n as f64;
            sum = sum + x * Complex::from_angle(-2.0 * PI * turns);
        }
        output.push(sum);
    }
    output
}

/// The fast Fourier transform, in place. `buf.len()` must be a power of two.
///
/// Same result as [`dft`], in `n log2(n)` steps instead of `n^2`. See
/// [`Fft`] if you transform many buffers of the same length.
pub fn fft(buf: &mut [Complex]) {
    Fft::new(buf.len()).forward(buf);
}

/// The inverse FFT, in place: turns a spectrum back into samples.
/// `buf.len()` must be a power of two. Includes the `1/n` scaling, so
/// `ifft(fft(x)) == x`.
pub fn ifft(buf: &mut [Complex]) {
    Fft::new(buf.len()).inverse(buf);
}

/// An FFT of one fixed length, with its table of sines and cosines computed
/// once up front.
///
/// The transform itself needs no trigonometry at all, only this table. When
/// we extract features we run hundreds of FFTs of the same length per second
/// of audio, so building the table once is the single biggest speed-up.
#[derive(Clone, Debug)]
pub struct Fft {
    n: usize,
    /// "Twiddle factors": `twiddles[j] = e^(-2 pi i j / n)` for `j < n/2`,
    /// the first half of the `n` points evenly spaced round the unit circle.
    twiddles: Vec<Complex>,
}

impl Fft {
    /// Prepare transforms of length `n`, which must be a power of two.
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT length must be a power of two, got {n}");
        let twiddles = (0..n / 2).map(|j| Complex::from_angle(-2.0 * PI * j as f64 / n as f64)).collect();
        Fft { n, twiddles }
    }

    /// The transform length this plan was built for.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always false: a plan has a length of at least 1. (Clippy insists that
    /// anything with a `len` also has an `is_empty`.)
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Forward transform, in place.
    ///
    /// The idea (Cooley and Tukey, 1965): a DFT of length `n` can be put
    /// together from the DFT of the even-numbered samples and the DFT of the
    /// odd-numbered samples, each of length `n/2`:
    ///
    /// ```text
    /// X[k]       = Even[k] + w^k * Odd[k]
    /// X[k + n/2] = Even[k] - w^k * Odd[k]        w = e^(-2 pi i / n)
    /// ```
    ///
    /// Each half can be split the same way, down to length 1, where the DFT
    /// of a single sample is the sample itself. [`fft_recursive`] does exactly
    /// that. Here we do the same work bottom-up, without recursion and
    /// without allocating: first put the samples in the order the recursion
    /// would end up reading them, then combine neighbours into transforms of
    /// length 2, those into length 4, and so on.
    pub fn forward(&self, buf: &mut [Complex]) {
        let n = self.n;
        assert_eq!(buf.len(), n, "this FFT was planned for length {n}, got a buffer of {}", buf.len());
        if n < 2 {
            return;
        }

        // Step 1: reorder. Splitting into even/odd again and again sorts the
        // samples by their index *with its bits reversed* (even/odd is the
        // lowest bit, and it is decided first). So sample 1 = 0b001 of 8
        // moves to position 0b100 = 4.
        let bits = n.trailing_zeros();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - bits);
            if i < j {
                buf.swap(i, j);
            }
        }

        // Step 2: combine. `len` is the size of the transforms being built.
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            // We need e^(-2 pi i j / len); the table has steps of 1/n of a
            // turn, so we walk through it in strides of n / len.
            let stride = n / len;
            for block in buf.chunks_exact_mut(len) {
                let (even, odd) = block.split_at_mut(half);
                for j in 0..half {
                    let rotated = odd[j] * self.twiddles[j * stride];
                    let e = even[j];
                    even[j] = e + rotated; // the "butterfly": one sum,
                    odd[j] = e - rotated; //  one difference, same two inputs
                }
            }
            len *= 2;
        }
    }

    /// Inverse transform, in place, including the `1/n` scaling.
    ///
    /// The inverse DFT is the forward DFT with the sign of every angle
    /// flipped. Flipping the sign of an angle is what `conj` does, so:
    /// conjugate the input, run the forward transform, conjugate the output.
    pub fn inverse(&self, buf: &mut [Complex]) {
        for x in buf.iter_mut() {
            *x = x.conj();
        }
        self.forward(buf);
        let scale = 1.0 / self.n as f32;
        for x in buf.iter_mut() {
            *x = x.conj().scale(scale);
        }
    }

    /// Power spectrum of a real frame: `|X[k]|^2` for `k = 0 ..= n/2`,
    /// written to `out`. The frame is zero-padded up to the FFT length.
    ///
    /// `scratch` is working memory that the caller keeps between calls so
    /// that we do not allocate once per frame.
    pub fn power_spectrum_into(&self, frame: &[f32], scratch: &mut Vec<Complex>, out: &mut [f32]) {
        assert!(frame.len() <= self.n, "frame of {} samples does not fit in an FFT of length {}", frame.len(), self.n);
        assert_eq!(out.len(), self.n / 2 + 1, "a length-{} FFT has {} bins up to Nyquist", self.n, self.n / 2 + 1);
        scratch.clear();
        scratch.extend(frame.iter().map(|&x| Complex::new(x, 0.0)));
        scratch.resize(self.n, Complex::ZERO);
        self.forward(scratch);
        for (power, bin) in out.iter_mut().zip(scratch.iter()) {
            *power = bin.norm_sqr();
        }
    }
}

/// The FFT exactly as the derivation goes: split into even and odd samples,
/// transform each half, combine. Allocates at every level, so it is slower
/// than [`fft`]; it is here because it is the easiest version to understand.
/// `input.len()` must be a power of two.
pub fn fft_recursive(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two, got {n}");
    if n == 1 {
        return vec![input[0]];
    }
    let even: Vec<Complex> = input.iter().step_by(2).copied().collect();
    let odd: Vec<Complex> = input.iter().skip(1).step_by(2).copied().collect();
    let (even, odd) = (fft_recursive(&even), fft_recursive(&odd));

    let mut output = vec![Complex::ZERO; n];
    for k in 0..n / 2 {
        let rotated = odd[k] * Complex::from_angle(-2.0 * PI * k as f64 / n as f64);
        output[k] = even[k] + rotated;
        output[k + n / 2] = even[k] - rotated;
    }
    output
}

/// Power spectrum of a real frame: `|X[k]|^2` for `k = 0 ..= n_fft/2`.
///
/// The frame is padded with zeros up to `n_fft` (a power of two, at least
/// `frame.len()`). Only the first `n_fft/2 + 1` bins are returned: for real
/// input the rest are their mirror image and carry no new information. The
/// last bin is the Nyquist frequency, half the sample rate.
///
/// No scaling is applied: a sine of amplitude `a` exactly on bin `k`, filling
/// the whole FFT, gives `(a * n_fft / 2)^2` in that bin.
pub fn power_spectrum(frame: &[f32], n_fft: usize) -> Vec<f32> {
    let mut out = vec![0.0; n_fft / 2 + 1];
    Fft::new(n_fft).power_spectrum_into(frame, &mut Vec::new(), &mut out);
    out
}

/// The frequency in Hz that bin `k` of a length-`n_fft` transform stands for.
pub fn bin_frequency(k: usize, n_fft: usize, sample_rate: u32) -> f32 {
    k as f32 * sample_rate as f32 / n_fft as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn real(values: &[f32]) -> Vec<Complex> {
        values.iter().map(|&v| Complex::new(v, 0.0)).collect()
    }

    fn random(n: usize, seed: u64) -> Vec<Complex> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| Complex::new(rng.normal(), rng.normal())).collect()
    }

    fn assert_close(a: &[Complex], b: &[Complex], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((*x - *y).abs() < tolerance, "element {i}: {x:?} vs {y:?}");
        }
    }

    #[test]
    fn complex_arithmetic() {
        let (a, b) = (Complex::new(1.0, 2.0), Complex::new(3.0, -1.0));
        assert_eq!(a + b, Complex::new(4.0, 1.0));
        assert_eq!(a - b, Complex::new(-2.0, 3.0));
        assert_eq!(a * b, Complex::new(5.0, 5.0)); // (1+2i)(3-i) = 3 - i + 6i + 2
        assert_eq!(Complex::new(0.0, 1.0) * Complex::new(0.0, 1.0), Complex::new(-1.0, 0.0)); // i*i = -1
        assert_eq!(Complex::new(3.0, 4.0).abs(), 5.0);
        assert_eq!(a.conj(), Complex::new(1.0, -2.0));
        let quarter_turn = Complex::from_angle(PI / 2.0);
        assert!((quarter_turn - Complex::new(0.0, 1.0)).abs() < 1e-6);
    }

    #[test]
    fn dft_of_four_samples_by_hand() {
        // x = [1, 2, 3, 4]: X[0] = 10, X[1] = -2+2i, X[2] = -2, X[3] = -2-2i.
        let expected = [Complex::new(10.0, 0.0), Complex::new(-2.0, 2.0), Complex::new(-2.0, 0.0), Complex::new(-2.0, -2.0)];
        let input = real(&[1.0, 2.0, 3.0, 4.0]);
        assert_close(&dft(&input), &expected, 1e-5);
        let mut buf = input.clone();
        fft(&mut buf);
        assert_close(&buf, &expected, 1e-5);
    }

    #[test]
    fn impulse_has_a_flat_spectrum() {
        // A single click contains every frequency equally.
        let mut buf = vec![Complex::ZERO; 64];
        buf[0] = Complex::new(1.0, 0.0);
        fft(&mut buf);
        assert_close(&buf, &vec![Complex::new(1.0, 0.0); 64], 1e-6);
    }

    #[test]
    fn constant_signal_lands_in_bin_zero() {
        let mut buf = real(&[0.5; 32]);
        fft(&mut buf);
        assert!((buf[0] - Complex::new(16.0, 0.0)).abs() < 1e-5); // c * n
        assert!(buf[1..].iter().all(|x| x.abs() < 1e-5));
    }

    #[test]
    fn sine_lands_in_its_bin_with_magnitude_a_n_over_2() {
        // 1 kHz at 16 kHz in 512 samples completes exactly 32 cycles.
        let (n, sample_rate, amplitude) = (512, 16000, 0.8f32);
        let frame: Vec<f32> =
            (0..n).map(|t| amplitude * (2.0 * PI * 1000.0 * t as f64 / sample_rate as f64).sin() as f32).collect();
        let power = power_spectrum(&frame, n);
        assert_eq!(power.len(), 257);
        let peak = (0..power.len()).max_by(|&a, &b| power[a].total_cmp(&power[b])).unwrap();
        assert_eq!(peak, 32);
        assert_eq!(bin_frequency(peak, n, sample_rate), 1000.0);
        let expected_magnitude = amplitude * n as f32 / 2.0;
        assert!((power[32].sqrt() - expected_magnitude).abs() < 1e-2, "{} vs {expected_magnitude}", power[32].sqrt());
        // Everything else is rounding noise.
        for (k, &p) in power.iter().enumerate() {
            assert!(k == 32 || p < 1e-4, "bin {k} has power {p}");
        }
    }

    #[test]
    fn fft_agrees_with_dft_on_random_input() {
        for n in [1, 2, 4, 8, 64, 256, 1024] {
            let input = random(n, n as u64);
            let slow = dft(&input);
            let mut fast = input.clone();
            fft(&mut fast);
            assert_close(&fast, &slow, 2e-3);
            assert_close(&fft_recursive(&input), &slow, 2e-3);
        }
    }

    #[test]
    fn ifft_undoes_fft() {
        let input = random(512, 9);
        let mut buf = input.clone();
        fft(&mut buf);
        ifft(&mut buf);
        assert_close(&buf, &input, 1e-4);
    }

    #[test]
    fn parseval_energy_is_the_same_on_both_sides() {
        let input = random(256, 3);
        let time_energy: f32 = input.iter().map(|x| x.norm_sqr()).sum();
        let mut spectrum = input.clone();
        fft(&mut spectrum);
        let freq_energy: f32 = spectrum.iter().map(|x| x.norm_sqr()).sum::<f32>() / 256.0;
        assert!((time_energy - freq_energy).abs() / time_energy < 1e-5, "{time_energy} vs {freq_energy}");
    }

    #[test]
    fn dft_handles_lengths_that_are_not_powers_of_two() {
        // Length 3, x = [1, 0, 0] -> all ones; x = [1, 1, 1] -> [3, 0, 0].
        assert_close(&dft(&real(&[1.0, 0.0, 0.0])), &real(&[1.0, 1.0, 1.0]), 1e-6);
        assert_close(&dft(&real(&[1.0, 1.0, 1.0])), &real(&[3.0, 0.0, 0.0]), 1e-6);
    }

    #[test]
    fn power_spectrum_zero_pads_short_frames() {
        // [1, 1] padded to 4: X = [2, 1-i, 0, 1+i] -> powers 4, 2, 0.
        let power = power_spectrum(&[1.0, 1.0], 4);
        assert_eq!(power.len(), 3);
        for (p, expected) in power.iter().zip([4.0, 2.0, 0.0]) {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn fft_rejects_other_lengths() {
        fft(&mut [Complex::ZERO; 12]);
    }
}
