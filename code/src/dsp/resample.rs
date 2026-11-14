//! Changing the sample rate.
//!
//! Our features assume one particular sample rate (16 kHz for the models we
//! train). A microphone may deliver 44.1 or 48 kHz, and the digit dataset is
//! at 8 kHz, so sooner or later we have to convert.
//!
//! The sampling theorem says that samples taken at rate `sr` pin down exactly
//! one smooth curve that contains no frequencies above `sr / 2`, and tells us
//! how to draw it: put a `sinc` pulse (`sin(pi x) / (pi x)`) on every sample
//! and add them all up. Resampling is then simply reading that curve off at
//! the new sample times.
//!
//! Two practical details:
//!
//! * A sinc goes on forever, so we fade it out with a Hann window after a
//!   fixed number of zero crossings. This is a *windowed-sinc* filter.
//! * When going **down** in rate, anything above the *new* Nyquist frequency
//!   must be removed first, or it folds back and shows up as a false lower
//!   tone ("aliasing"). Stretching the sinc in time makes it a low-pass
//!   filter at the lower cutoff, so one kernel both smooths and interpolates.

use crate::dsp::window::hann_symmetric;
use std::f64::consts::PI;

/// How many zero crossings of the sinc we keep on each side of its peak.
/// More is more accurate (a sharper cutoff) and slower.
const ZERO_CROSSINGS: usize = 16;

/// The kernel is tabulated at this many points between two neighboring zero
/// crossings and read off by linear interpolation.
const TABLE_RESOLUTION: usize = 512;

/// We put the filter's cutoff a little below the Nyquist frequency, because
/// a finite filter cannot drop from "pass" to "block" instantly.
const ROLLOFF: f64 = 0.94;

/// Convert `samples` from `from_rate` to `to_rate` samples per second.
///
/// The result has `round(len * to_rate / from_rate)` samples. Frequencies
/// that both rates can represent are preserved; when downsampling, those
/// above the new Nyquist frequency are filtered out.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    assert!(from_rate > 0 && to_rate > 0, "sample rates must be positive");
    if from_rate == to_rate {
        return samples.to_vec();
    }

    // Cutoff as a fraction of the *input* Nyquist frequency: 1 (well, 0.94)
    // when upsampling, lower when downsampling.
    let cutoff = ROLLOFF * (to_rate as f64 / from_rate as f64).min(1.0);
    let kernel = SincKernel::new();
    // The kernel is `ZERO_CROSSINGS` zero crossings wide on each side; at a
    // lower cutoff the crossings are further apart, measured in input samples.
    let half_width = ZERO_CROSSINGS as f64 / cutoff;

    let out_len = ((samples.len() as u64 * to_rate as u64 + from_rate as u64 / 2) / from_rate as u64) as usize;
    let step = from_rate as f64 / to_rate as f64; // input samples per output sample
    let mut out = Vec::with_capacity(out_len);
    for m in 0..out_len {
        // Where output sample m falls on the input's time axis.
        let position = m as f64 * step;
        let first = (position - half_width).ceil().max(0.0) as usize;
        let last = ((position + half_width).floor() as usize).min(samples.len().saturating_sub(1));
        let mut sum = 0.0f64;
        for (k, &x) in samples.iter().enumerate().take(last + 1).skip(first) {
            // Distance from this input sample to the output position,
            // measured in zero crossings of the stretched sinc.
            let distance = (position - k as f64) * cutoff;
            sum += x as f64 * kernel.at(distance);
        }
        // A sinc stretched by 1/cutoff is 1/cutoff times as tall in total;
        // multiplying by the cutoff keeps the loudness the same.
        out.push((sum * cutoff) as f32);
    }
    out
}

/// A Hann-windowed sinc, `sinc(x) * w(x)` for `|x| <= ZERO_CROSSINGS`,
/// tabulated once so that the inner loop needs no trigonometry.
struct SincKernel {
    /// `table[i]` is the kernel at `x = i / TABLE_RESOLUTION`, for `x >= 0`
    /// (the kernel is symmetric).
    table: Vec<f64>,
}

impl SincKernel {
    fn new() -> Self {
        let len = ZERO_CROSSINGS * TABLE_RESOLUTION;
        // The right half of a symmetric Hann window: 1 at the center, 0 at the edge.
        let window = hann_symmetric(2 * len + 1);
        let mut table: Vec<f64> = (0..=len)
            .map(|i| {
                let x = i as f64 / TABLE_RESOLUTION as f64;
                let sinc = if i == 0 { 1.0 } else { (PI * x).sin() / (PI * x) };
                sinc * window[len + i] as f64
            })
            .collect();
        table.push(0.0); // one spare entry so `at` can always look one ahead
        SincKernel { table }
    }

    /// The kernel at `x` zero crossings from its center.
    fn at(&self, x: f64) -> f64 {
        let scaled = x.abs() * TABLE_RESOLUTION as f64;
        let index = scaled as usize;
        if index + 1 >= self.table.len() {
            return 0.0;
        }
        let fraction = scaled - index as f64;
        self.table[index] + fraction * (self.table[index + 1] - self.table[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::fft::power_spectrum;
    use crate::dsp::window::{apply, coherent_gain, hann};

    fn sine(frequency: f32, sample_rate: u32, len: usize) -> Vec<f32> {
        (0..len).map(|t| (2.0 * PI * frequency as f64 * t as f64 / sample_rate as f64).sin() as f32).collect()
    }

    /// Frequency (Hz) and amplitude of the strongest tone in the middle of a
    /// signal, measured with a Hann-windowed 4096-point FFT.
    fn strongest_tone(samples: &[f32], sample_rate: u32) -> (f32, f32) {
        let n = 4096;
        let start = (samples.len() - n) / 2;
        let window = hann(n);
        let power = power_spectrum(&apply(&samples[start..start + n], &window), n);
        let peak = (0..power.len()).max_by(|&a, &b| power[a].total_cmp(&power[b])).unwrap();
        // Undo the FFT's n/2 and the window's gain to get back an amplitude.
        let amplitude = power[peak].sqrt() / (n as f32 / 2.0 * coherent_gain(&window));
        (peak as f32 * sample_rate as f32 / n as f32, amplitude)
    }

    #[test]
    fn output_length_scales_with_the_rate() {
        assert_eq!(resample(&[0.0; 16000], 16000, 8000).len(), 8000);
        assert_eq!(resample(&[0.0; 8000], 8000, 16000).len(), 16000);
        assert_eq!(resample(&[0.0; 44100], 44100, 16000).len(), 16000);
        assert_eq!(resample(&[0.0; 100], 48000, 16000).len(), 33);
        assert_eq!(resample(&[0.25; 7], 8000, 8000), vec![0.25; 7]);
        assert!(resample(&[], 8000, 16000).is_empty());
    }

    #[test]
    fn a_tone_keeps_its_frequency_and_amplitude() {
        // 1 kHz: 4096-point FFT bins are 3.9 Hz apart at 16 kHz, 1.95 at 8 kHz.
        for (from, to) in [(16000, 8000), (8000, 16000), (44100, 16000), (48000, 16000), (16000, 44100)] {
            let out = resample(&sine(1000.0, from, from as usize), from, to);
            let (frequency, amplitude) = strongest_tone(&out, to);
            assert!((frequency - 1000.0).abs() <= to as f32 / 4096.0, "{from} -> {to}: {frequency} Hz");
            assert!((amplitude - 1.0).abs() < 0.03, "{from} -> {to}: amplitude {amplitude}");
        }
    }

    #[test]
    fn upsampling_passes_through_the_original_samples() {
        // Doubling the rate: every other output sample sits exactly on an
        // input sample, and for a signal well below Nyquist it should match.
        let input = sine(440.0, 8000, 2000);
        let out = resample(&input, 8000, 16000);
        for k in 200..1800 {
            assert!((out[2 * k] - input[k]).abs() < 2e-3, "sample {k}: {} vs {}", out[2 * k], input[k]);
        }
    }

    #[test]
    fn a_tone_above_the_new_nyquist_is_removed_not_aliased() {
        // 6 kHz cannot exist at 8 kHz (Nyquist 4 kHz). Without the low-pass
        // it would fold down to a full-strength false tone at 2 kHz.
        let out = resample(&sine(6000.0, 16000, 16000), 16000, 8000);
        let (_, amplitude) = strongest_tone(&out, 8000);
        assert!(amplitude < 0.01, "leftover amplitude {amplitude}");
        // Naive decimation (take every other sample) aliases at full strength.
        let naive: Vec<f32> = sine(6000.0, 16000, 16000).into_iter().step_by(2).collect();
        let (frequency, amplitude) = strongest_tone(&naive, 8000);
        assert!((frequency - 2000.0).abs() < 4.0 && amplitude > 0.9);
    }

    #[test]
    fn a_constant_stays_constant() {
        let out = resample(&[0.5; 4000], 16000, 8000);
        for &v in &out[200..1800] {
            assert!((v - 0.5).abs() < 2e-3, "{v}");
        }
    }
}
