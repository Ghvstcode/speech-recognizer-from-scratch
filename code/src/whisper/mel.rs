//! Whisper's audio front end: 30 seconds of sound in, a 3000 x 80 log-mel
//! spectrogram out.
//!
//! This is chapters 2 to 4 again, with someone else's constants. A pretrained
//! network has only ever seen pictures produced by one exact recipe, so we
//! must follow that recipe to the letter; a "better" spectrogram would be a
//! worse input. The recipe, from OpenAI's reference `log_mel_spectrogram`:
//!
//! 1. Audio is 16 kHz mono. Pad with zeros, or cut, to exactly 30 s
//!    (480,000 samples).
//! 2. Short-time Fourier transform with a 400-sample (25 ms) window hopping
//!    160 samples (10 ms). The signal is first extended by 200 samples at
//!    each end by *reflection*, so frame `t` is centred on sample `160 * t`.
//!    The window is a *periodic* Hann window.
//! 3. Keep the power `|X|^2` of bins 0..=200, and drop the very last frame,
//!    leaving exactly 3000 frames.
//! 4. Multiply by an 80-band mel filterbank (the "Slaney" flavour, the
//!    default in the `librosa` library the reference used to build it).
//! 5. `log10`, with the power floored at `1e-10`.
//! 6. Clamp everything to at most 8 below the loudest value in the window
//!    (a dynamic range of 80 dB), then squash with `(x + 4) / 4`.
//!
//! ## Why a plain DFT and not the FFT from chapter 3
//!
//! Our radix-2 FFT needs a power-of-two length, and 400 is not one. Padding
//! to 512 would give 257 bins at different frequencies: a different picture.
//! The honest options are a mixed-radix FFT, Bluestein's algorithm, or the
//! "slow" DFT that correlates the frame against every sinusoid directly. We
//! use the last. With the sines and cosines tabulated once, each bin is two
//! dot products of length 400, the whole spectrogram is about half a billion
//! multiply-adds, and spread over the cores that takes a few hundredths of a
//! second: well under 1% of the total. Clarity wins.

use super::parallel::for_each_row_par;
use crate::mat::{dot, Mat};

/// Whisper only understands 16 kHz audio.
pub const SAMPLE_RATE: u32 = 16_000;
/// Samples per analysis frame (25 ms) and length of the Fourier transform.
pub const N_FFT: usize = 400;
/// Samples between the starts of consecutive frames (10 ms).
pub const HOP: usize = 160;
/// Frequency bins kept from each transform: 0 Hz up to and including 8 kHz.
pub const N_BINS: usize = N_FFT / 2 + 1;
/// The model always looks at exactly this much audio: 30 seconds.
pub const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
/// Spectrogram frames in one 30 s chunk.
pub const CHUNK_FRAMES: usize = CHUNK_SAMPLES / HOP;

/// Hertz to mels on the Slaney scale: linear below 1 kHz (200/3 Hz per mel),
/// logarithmic above (each 27 mels multiply the frequency by 6.4).
///
/// Chapter 4 used the other common formula, `2595 * log10(1 + f / 700)`.
/// Both bend the frequency axis the way the ear does; they are simply two
/// different curve fits, and Whisper was trained with this one.
pub fn hz_to_mel(hz: f64) -> f64 {
    const LINEAR_HZ_PER_MEL: f64 = 200.0 / 3.0;
    const BREAK_HZ: f64 = 1000.0;
    if hz < BREAK_HZ {
        hz / LINEAR_HZ_PER_MEL
    } else {
        BREAK_HZ / LINEAR_HZ_PER_MEL + (hz / BREAK_HZ).ln() / (6.4f64.ln() / 27.0)
    }
}

/// The inverse of `hz_to_mel`.
pub fn mel_to_hz(mel: f64) -> f64 {
    const LINEAR_HZ_PER_MEL: f64 = 200.0 / 3.0;
    const BREAK_HZ: f64 = 1000.0;
    let break_mel = BREAK_HZ / LINEAR_HZ_PER_MEL;
    if mel < break_mel {
        mel * LINEAR_HZ_PER_MEL
    } else {
        BREAK_HZ * ((mel - break_mel) * (6.4f64.ln() / 27.0)).exp()
    }
}

/// The mel filterbank, `n_mels x (n_fft / 2 + 1)`: row `m` holds the weights
/// that mix the power spectrum's bins into mel band `m`.
///
/// Band `m` is a triangle over frequency. Place `n_mels + 2` points evenly
/// *in mels* from 0 Hz to the Nyquist frequency; triangle `m` rises from
/// point `m` to a peak at point `m + 1` and falls back to zero at point
/// `m + 2`. Each triangle is then scaled by `2 / (its width in Hz)`, which
/// gives every triangle the same area. Without that, the wide high-frequency
/// bands would collect far more energy than the narrow low ones.
///
/// The reference implementation loads this matrix from a file
/// (`mel_filters.npz`) that was produced by `librosa.filters.mel(sr=16000,
/// n_fft=400, n_mels=80)`. That function is what we reproduce here.
pub fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Mat {
    let sample_rate = sample_rate as f64;
    let top_mel = hz_to_mel(sample_rate / 2.0);
    let edges_hz: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(top_mel * i as f64 / (n_mels + 1) as f64)).collect();
    Mat::from_fn(n_mels, n_fft / 2 + 1, |band, bin| {
        let hz = bin as f64 * sample_rate / n_fft as f64;
        let (low, peak, high) = (edges_hz[band], edges_hz[band + 1], edges_hz[band + 2]);
        let rising = (hz - low) / (peak - low);
        let falling = (high - hz) / (high - peak);
        let triangle = rising.min(falling).max(0.0);
        (triangle * 2.0 / (high - low)) as f32
    })
}

/// A periodic Hann window: `0.5 - 0.5 cos(2 pi n / N)`.
///
/// "Periodic" means the divisor is `N`, not the `N - 1` of the symmetric
/// window in chapter 2, so the window never quite returns to zero at its
/// right edge. That is the right shape when windows overlap and add, and it
/// is what `torch.hann_window` returns by default.
pub fn periodic_hann(len: usize) -> Vec<f32> {
    (0..len).map(|n| (0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / len as f64).cos()) as f32).collect()
}

/// Cut `samples` to `len`, or pad it with silence up to `len`.
pub fn pad_or_trim(samples: &[f32], len: usize) -> Vec<f32> {
    let mut out = samples[..samples.len().min(len)].to_vec();
    out.resize(len, 0.0);
    out
}

/// Extend a signal by `pad` samples at each end by mirroring it about its
/// first and last samples (which are not repeated): `c b | a b c d | c b`.
pub fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    assert!(pad < samples.len(), "cannot reflect {pad} samples of a {}-sample signal", samples.len());
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|i| samples[i]));
    out.extend_from_slice(samples);
    out.extend((1..=pad).map(|i| samples[n - 1 - i]));
    out
}

/// Everything the front end needs that does not depend on the audio,
/// computed once.
pub struct MelFrontEnd {
    /// Number of mel bands (80 for every Whisper model except large-v3).
    pub n_mels: usize,
    /// The periodic Hann window, `N_FFT` long.
    window: Vec<f32>,
    /// `cos_table[(k, n)] = cos(2 pi k n / N_FFT)`: the cosine at bin `k`'s frequency.
    cos_table: Mat,
    /// `sin_table[(k, n)] = sin(2 pi k n / N_FFT)`.
    sin_table: Mat,
    /// The mel filterbank, `n_mels x N_BINS`.
    filters: Mat,
}

impl MelFrontEnd {
    /// Build the window, the sinusoid tables and the filterbank.
    pub fn new(n_mels: usize) -> MelFrontEnd {
        // k * n can reach 200 * 399; reducing it modulo N first keeps the
        // angle small, where sin and cos are most accurate.
        let angle = |k: usize, n: usize| std::f64::consts::TAU * ((k * n) % N_FFT) as f64 / N_FFT as f64;
        MelFrontEnd {
            n_mels,
            window: periodic_hann(N_FFT),
            cos_table: Mat::from_fn(N_BINS, N_FFT, |k, n| angle(k, n).cos() as f32),
            sin_table: Mat::from_fn(N_BINS, N_FFT, |k, n| angle(k, n).sin() as f32),
            filters: mel_filterbank(n_mels, N_FFT, SAMPLE_RATE),
        }
    }

    /// The power spectrum `|X[k]|^2` of one windowed frame, for bins `0..N_BINS`.
    ///
    /// The DFT is `X[k] = sum_n x[n] (cos(2 pi k n / N) - i sin(2 pi k n / N))`:
    /// the real part is the frame's dot product with a cosine, the imaginary
    /// part (up to sign) its dot product with a sine.
    pub fn power_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        assert_eq!(frame.len(), N_FFT, "a frame is {N_FFT} samples");
        (0..N_BINS)
            .map(|k| {
                let real = dot(frame, self.cos_table.row(k));
                let imaginary = dot(frame, self.sin_table.row(k));
                real * real + imaginary * imaginary
            })
            .collect()
    }

    /// The model's input for one 30 s window: `CHUNK_FRAMES x n_mels`, one
    /// row per 10 ms frame. `samples` is padded with silence or cut to 30 s.
    pub fn log_mel(&self, samples: &[f32]) -> Mat {
        let audio = pad_or_trim(samples, CHUNK_SAMPLES);
        let padded = reflect_pad(&audio, N_FFT / 2);

        // Steps 2-5: each frame's log-mel energies. Frames are independent,
        // so they are computed in parallel. The reference computes one frame
        // more (centred on the last sample) and throws it away; we simply
        // never compute it.
        let mut mel = Mat::zeros(CHUNK_FRAMES, self.n_mels);
        for_each_row_par(&mut mel.data, self.n_mels, |t, mel_row| {
            let start = t * HOP;
            let frame: Vec<f32> = padded[start..start + N_FFT].iter().zip(&self.window).map(|(s, w)| s * w).collect();
            let power = self.power_spectrum(&frame);
            for (band, out) in mel_row.iter_mut().enumerate() {
                *out = dot(self.filters.row(band), &power).max(1e-10).log10();
            }
        });

        // Step 6: limit the dynamic range relative to the loudest cell, then
        // rescale so typical values land roughly in [-1, 1].
        let loudest = mel.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for value in mel.data.iter_mut() {
            *value = (value.max(loudest - 8.0) + 4.0) / 4.0;
        }
        mel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slaney_scale_known_points() {
        assert!((hz_to_mel(0.0)).abs() < 1e-12);
        assert!((hz_to_mel(200.0) - 3.0).abs() < 1e-12);
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-12);
        // One "log step" above the break: 6.4 kHz is exactly 27 mels above 1 kHz.
        assert!((hz_to_mel(6400.0) - 42.0).abs() < 1e-9);
        for hz in [0.0, 50.0, 999.0, 1000.0, 1001.0, 4000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-6, "round trip at {hz} Hz");
        }
    }

    #[test]
    fn filterbank_has_the_documented_shape_and_properties() {
        let fb = mel_filterbank(80, N_FFT, SAMPLE_RATE);
        assert_eq!((fb.rows, fb.cols), (80, 201));
        assert!(fb.data.iter().all(|&w| w >= 0.0));
        // No energy from 0 Hz or from the Nyquist bin: they sit exactly on
        // the outer feet of the first and last triangles.
        assert!((0..80).all(|m| fb[(m, 0)] == 0.0 && fb[(m, 200)] == 0.0));
        // Every band listens to something, and centres move upwards.
        let peak_bin = |m: usize| (0..201).max_by(|&a, &b| fb[(m, a)].total_cmp(&fb[(m, b)])).unwrap();
        for m in 0..80 {
            assert!(fb.row(m).iter().sum::<f32>() > 0.0, "band {m} is empty");
            if m > 0 {
                assert!(peak_bin(m) >= peak_bin(m - 1));
            }
        }
        // Area normalisation: a triangle of height 2/width has area 1 (in Hz).
        // Bins are 40 Hz apart, so for the wide high bands the sum of the
        // sampled weights times 40 should be close to 1.
        for m in 60..80 {
            let area = fb.row(m).iter().sum::<f32>() * 40.0;
            assert!((area - 1.0).abs() < 0.05, "band {m} has area {area}");
        }
    }

    #[test]
    fn filterbank_matches_values_derived_by_hand() {
        // Below 1 kHz the scale is linear, so the first triangles are easy to
        // work out on paper. Top mel = 15 + 27 ln(8)/ln(6.4) = 45.24564; the 82
        // edge points are 45.24564/81 = 0.558588 mel = 37.2392 Hz apart.
        // Band 0 spans 0..74.4784 Hz with its peak at 37.2392 Hz. Bin 1 is at
        // 40 Hz, on the falling side: (74.4784 - 40) / 37.2392 = 0.925863,
        // times the normalisation 2 / 74.4784 = 0.0268534, gives 0.0248626.
        let fb = mel_filterbank(80, N_FFT, SAMPLE_RATE);
        assert!((fb[(0, 1)] - 0.0248626).abs() < 1e-6, "got {}", fb[(0, 1)]);
        // Band 1 spans 37.2392..111.7176 Hz; bin 1 (40 Hz) is on its rising
        // side: (40 - 37.2392) / 37.2392 = 0.0741366, times 0.0268534 = 0.0019908.
        assert!((fb[(1, 1)] - 0.0019908).abs() < 1e-6, "got {}", fb[(1, 1)]);
        assert_eq!(fb[(0, 2)], 0.0);
    }

    #[test]
    fn hann_window_is_periodic() {
        let w = periodic_hann(400);
        assert_eq!(w[0], 0.0);
        assert!((w[200] - 1.0).abs() < 1e-7);
        assert!((w[100] - 0.5).abs() < 1e-7);
        // Periodic: w[N - n] == w[n], and the last sample is not zero.
        assert!((w[399] - w[1]).abs() < 1e-7 && w[399] > 0.0);
    }

    #[test]
    fn padding_helpers() {
        assert_eq!(pad_or_trim(&[1.0, 2.0, 3.0], 2), vec![1.0, 2.0]);
        assert_eq!(pad_or_trim(&[1.0, 2.0, 3.0], 5), vec![1.0, 2.0, 3.0, 0.0, 0.0]);
        assert_eq!(reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2), vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn power_spectrum_of_a_pure_tone_peaks_in_its_bin() {
        let front_end = MelFrontEnd::new(80);
        // A cosine that completes exactly 10 cycles in the frame: all of its
        // energy is in bin 10, where |X| = N/2 = 200 and |X|^2 = 40000.
        let tone: Vec<f32> = (0..N_FFT).map(|n| (std::f32::consts::TAU * 10.0 * n as f32 / N_FFT as f32).cos()).collect();
        let power = front_end.power_spectrum(&tone);
        assert!((power[10] - 40_000.0).abs() < 1.0, "got {}", power[10]);
        for (k, &p) in power.iter().enumerate() {
            assert!(k == 10 || p < 1e-3, "bin {k} leaked {p}");
        }
        // A constant signal is "0 Hz": everything lands in bin 0.
        let dc = front_end.power_spectrum(&vec![1.0; N_FFT]);
        assert!((dc[0] - 160_000.0).abs() < 1.0 && dc[1] < 1e-3);
    }

    #[test]
    fn log_mel_of_silence_and_of_a_tone() {
        let front_end = MelFrontEnd::new(80);
        // Silence: power 0 everywhere, floored to 1e-10, log10 = -10,
        // (-10 + 4) / 4 = -1.5 in every cell.
        let silence = front_end.log_mel(&[]);
        assert_eq!((silence.rows, silence.cols), (3000, 80));
        assert!(silence.data.iter().all(|&v| v == -1.5));

        // One second of a 1 kHz tone, then silence.
        let tone: Vec<f32> = (0..16_000).map(|n| 0.5 * (std::f32::consts::TAU * 1000.0 * n as f32 / 16_000.0).sin()).collect();
        let mel = front_end.log_mel(&tone);
        let (min, max) = mel.data.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        // The clamp limits the range to 8 in log10 units, 2 after the /4.
        assert!((max - min - 2.0).abs() < 1e-5, "range {min}..{max}");
        // The loudest band during the tone is the one whose triangle covers 1 kHz.
        let fb = mel_filterbank(80, N_FFT, SAMPLE_RATE);
        let expected_band = (0..80).max_by(|&a, &b| fb[(a, 25)].total_cmp(&fb[(b, 25)])).unwrap(); // bin 25 = 1 kHz
        let loudest_band = (0..80).max_by(|&a, &b| mel[(50, a)].total_cmp(&mel[(50, b)])).unwrap();
        assert_eq!(loudest_band, expected_band);
        // Long after the tone has stopped, the frames are at the floor.
        assert!(mel.row(2000).iter().all(|&v| v == min));
    }
}
