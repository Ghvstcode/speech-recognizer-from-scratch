//! From samples to features: log-mel spectrograms and MFCCs.
//!
//! This is the front door of every recognizer in the book. Audio goes in, a
//! [`Mat`] with **one row per frame** comes out:
//!
//! ```text
//! samples ─ pre-emphasis ─ frames ─ Hann window ─ FFT ─ |X|^2 ─ mel filterbank ─ ln ─▶ log-mel
//!                                                                                  └─ DCT ─▶ MFCC
//! ```

use crate::dsp::fft::{Complex, Fft};
use crate::dsp::frame::{frames, preemphasis};
use crate::dsp::mel::mel_filterbank;
use crate::dsp::window::hann;
use crate::mat::Mat;
use std::f32::consts::PI;

/// Added to every mel energy before taking the logarithm, so that silence
/// gives `ln(1e-6) = -13.8` instead of minus infinity.
pub const LOG_FLOOR: f32 = 1e-6;

/// Everything that determines what the features look like.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureConfig {
    /// Sample rate of the audio in Hz: 16000 for Speech Commands, 8000 for
    /// the Free Spoken Digit Dataset.
    pub sample_rate: u32,
    /// Frame length in milliseconds. 25 ms is long enough to contain a few
    /// periods of the lowest voices, short enough that the mouth has hardly
    /// moved.
    pub frame_ms: f32,
    /// Time between the starts of consecutive frames, in milliseconds. 10 ms
    /// gives 100 feature rows per second.
    pub hop_ms: f32,
    /// Number of mel bands, i.e. the number of columns of the log-mel output.
    pub n_mels: usize,
    /// Lower edge of the lowest mel band in Hz.
    pub fmin: f32,
    /// Upper edge of the highest mel band in Hz.
    pub fmax: f32,
    /// Pre-emphasis coefficient (0.97 is customary; 0.0 disables it).
    pub preemphasis: f32,
}

impl FeatureConfig {
    /// The defaults used throughout the book: 25 ms frames every 10 ms,
    /// 40 mel bands from 20 Hz to half the sample rate, pre-emphasis 0.97.
    pub fn new(sample_rate: u32) -> Self {
        FeatureConfig {
            sample_rate,
            frame_ms: 25.0,
            hop_ms: 10.0,
            n_mels: 40,
            fmin: 20.0,
            fmax: sample_rate as f32 / 2.0,
            preemphasis: 0.97,
        }
    }

    /// Frame length in samples: 400 at 16 kHz, 200 at 8 kHz.
    pub fn frame_len(&self) -> usize {
        (self.sample_rate as f32 * self.frame_ms / 1000.0).round() as usize
    }

    /// Hop length in samples: 160 at 16 kHz, 80 at 8 kHz.
    pub fn hop_len(&self) -> usize {
        (self.sample_rate as f32 * self.hop_ms / 1000.0).round() as usize
    }

    /// FFT length: the next power of two at or above the frame length (our
    /// FFT only does powers of two). 512 at 16 kHz, 256 at 8 kHz. The frame
    /// is padded with zeros to fill it.
    pub fn n_fft(&self) -> usize {
        self.frame_len().next_power_of_two()
    }

    /// How many feature rows a clip of `n_samples` samples produces. See
    /// [`num_frames`](crate::dsp::frame::num_frames): only whole frames count.
    pub fn num_frames(&self, n_samples: usize) -> usize {
        crate::dsp::frame::num_frames(n_samples, self.frame_len(), self.hop_len())
    }
}

/// A feature extractor with everything that can be computed ahead of time
/// already computed: the window, the mel filterbank and the FFT's table.
///
/// Build one and reuse it for every clip. It is never modified after
/// construction, so many threads can share one `&FeatureExtractor`.
#[derive(Clone, Debug)]
pub struct FeatureExtractor {
    config: FeatureConfig,
    window: Vec<f32>,
    fft: Fft,
    /// `n_mels x (n_fft/2 + 1)`, see [`mel_filterbank`].
    filterbank: Mat,
}

impl FeatureExtractor {
    /// Precompute the tables for `config`.
    pub fn new(config: &FeatureConfig) -> Self {
        let n_fft = config.n_fft();
        FeatureExtractor {
            window: hann(config.frame_len()),
            fft: Fft::new(n_fft),
            filterbank: mel_filterbank(config.n_mels, n_fft, config.sample_rate, config.fmin, config.fmax),
            config: config.clone(),
        }
    }

    /// The configuration this extractor was built from.
    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }

    /// The mel filterbank in use (`n_mels x n_bins`).
    pub fn filterbank(&self) -> &Mat {
        &self.filterbank
    }

    /// Power spectrogram of `samples` (frames x `n_fft/2 + 1`), after
    /// pre-emphasis and Hann windowing: the stage just before the filterbank.
    pub fn power_spectrogram(&self, samples: &[f32]) -> Mat {
        let emphasized;
        let samples = if self.config.preemphasis != 0.0 {
            emphasized = preemphasis(samples, self.config.preemphasis);
            &emphasized[..]
        } else {
            samples
        };

        let all_frames = frames(samples, self.config.frame_len(), self.config.hop_len());
        let mut power = Mat::zeros(all_frames.len(), self.fft.len() / 2 + 1);
        let mut windowed = vec![0.0f32; self.window.len()];
        let mut scratch: Vec<Complex> = Vec::with_capacity(self.fft.len());
        for (i, frame) in all_frames.into_iter().enumerate() {
            for ((w, x), win) in windowed.iter_mut().zip(frame).zip(&self.window) {
                *w = x * win;
            }
            self.fft.power_spectrum_into(&windowed, &mut scratch, power.row_mut(i));
        }
        power
    }

    /// Log-mel spectrogram: frames x `n_mels`, each entry
    /// `ln(mel energy + 1e-6)`.
    pub fn log_mel(&self, samples: &[f32]) -> Mat {
        let power = self.power_spectrogram(samples);
        // (frames x bins) times (n_mels x bins) transposed: every output
        // entry is the dot product of one spectrum with one triangle.
        let mel_energy = power.matmul_t(&self.filterbank);
        mel_energy.map(|e| (e + LOG_FLOOR).ln())
    }

    /// MFCCs: frames x `n_coeffs`. See the free function [`mfcc`].
    pub fn mfcc(&self, samples: &[f32], n_coeffs: usize) -> Mat {
        self.log_mel(samples).matmul_t(&dct_matrix(n_coeffs, self.config.n_mels))
    }
}

/// Log-mel spectrogram of `samples`: frames x `cfg.n_mels`, natural log of
/// (mel energy + 1e-6).
///
/// "Mel energy" is the power spectrum `|X[k]|^2` of a pre-emphasized,
/// Hann-windowed frame, summed through each triangle of the mel filterbank.
/// Nothing is rescaled along the way, so for samples in `[-1, 1]` the values
/// run from -13.8 (digital silence) to roughly +10 (very loud).
///
/// This builds a [`FeatureExtractor`] every time. If you have more than one
/// clip, build the extractor once yourself.
pub fn log_mel(samples: &[f32], cfg: &FeatureConfig) -> Mat {
    FeatureExtractor::new(cfg).log_mel(samples)
}

/// Mel-frequency cepstral coefficients: frames x `n_coeffs`.
///
/// Each row of the log-mel spectrogram is run through a DCT ([`dct2`]) and
/// only the first `n_coeffs` outputs are kept (13 is traditional). The DCT
/// describes the *shape* of the spectrum with cosines of increasing
/// wiggliness: coefficient 0 is the overall level, the next few capture the
/// broad humps that tell vowels apart, and the ones we throw away hold the
/// fine ripple caused by the pitch of the voice, which we do not want. The
/// coefficients are also far less correlated with one another than
/// neighbouring mel bands are, which simple models (like the template
/// matcher of chapter 5) appreciate.
pub fn mfcc(samples: &[f32], cfg: &FeatureConfig, n_coeffs: usize) -> Mat {
    FeatureExtractor::new(cfg).mfcc(samples, n_coeffs)
}

/// The discrete cosine transform (type II, orthonormal) of `input`, keeping
/// the first `n_coeffs` coefficients:
///
/// ```text
/// c[k] = s(k) * sum over m of  x[m] * cos(pi * k * (m + 1/2) / M)
///
///        M = input.len(),  s(0) = sqrt(1/M),  s(k) = sqrt(2/M) for k > 0
/// ```
///
/// It is a cousin of the DFT that uses only cosines and so stays real. The
/// scale factors `s(k)` make it *orthonormal*: with `n_coeffs == M` the
/// transform preserves lengths and its inverse is its transpose.
pub fn dct2(input: &[f32], n_coeffs: usize) -> Vec<f32> {
    let m_total = input.len();
    assert!(n_coeffs <= m_total, "cannot keep {n_coeffs} DCT coefficients of only {m_total} inputs");
    let mut output = Vec::with_capacity(n_coeffs);
    for k in 0..n_coeffs {
        let mut sum = 0.0;
        for (m, &x) in input.iter().enumerate() {
            sum += x * (PI * k as f32 * (m as f32 + 0.5) / m_total as f32).cos();
        }
        output.push(sum * dct_scale(k, m_total));
    }
    output
}

/// The DCT as a matrix (`n_coeffs x n_inputs`): row `k` is the `k`-th cosine,
/// already scaled, so `dct2(x, n)` equals this matrix times `x`. Used to
/// transform all frames at once without recomputing any cosines.
pub fn dct_matrix(n_coeffs: usize, n_inputs: usize) -> Mat {
    assert!(n_coeffs <= n_inputs, "cannot keep {n_coeffs} DCT coefficients of only {n_inputs} inputs");
    Mat::from_fn(n_coeffs, n_inputs, |k, m| dct_scale(k, n_inputs) * (PI * k as f32 * (m as f32 + 0.5) / n_inputs as f32).cos())
}

/// The factor `s(k)` that makes the DCT orthonormal.
fn dct_scale(k: usize, n_inputs: usize) -> f32 {
    if k == 0 {
        (1.0 / n_inputs as f32).sqrt()
    } else {
        (2.0 / n_inputs as f32).sqrt()
    }
}

/// Delta features: how fast each feature is changing, frame by frame. Same
/// shape as the input.
///
/// A single frame is a snapshot; speech is movement. The delta of a feature
/// at frame `t` is the slope of the straight line that best fits its values
/// from frame `t - width` to `t + width`:
///
/// ```text
/// d[t] = sum for n = 1..=width of  n * (c[t+n] - c[t-n])
///        ───────────────────────────────────────────────
///                 2 * sum for n = 1..=width of n^2
/// ```
///
/// Frames before the start or after the end are taken to be copies of the
/// first or last frame. `width` is usually 2. Apply it twice to get
/// "delta-deltas" (acceleration).
pub fn deltas(features: &Mat, width: usize) -> Mat {
    assert!(width >= 1, "delta width must be at least 1");
    let denominator = 2.0 * (1..=width).map(|n| (n * n) as f32).sum::<f32>();
    let last = features.rows.saturating_sub(1);
    let mut out = Mat::zeros(features.rows, features.cols);
    for t in 0..features.rows {
        for n in 1..=width {
            let ahead = features.row((t + n).min(last));
            let behind = features.row(t.saturating_sub(n));
            for ((d, a), b) in out.row_mut(t).iter_mut().zip(ahead).zip(behind) {
                *d += n as f32 * (a - b) / denominator;
            }
        }
    }
    out
}

/// Normalize each column to mean 0 and variance 1 over the utterance.
///
/// A louder recording, a different microphone or a different room shifts and
/// stretches every column of the features. Subtracting each column's own
/// mean and dividing by its own standard deviation removes exactly that, so
/// what is left is how the features *move* within the utterance. (For MFCCs
/// this is "cepstral mean and variance normalization".)
///
/// A column that is constant is left at zero rather than divided by zero.
pub fn normalize(features: &mut Mat) {
    if features.rows == 0 {
        return;
    }
    let n = features.rows as f32;
    for c in 0..features.cols {
        let mean = (0..features.rows).map(|r| features[(r, c)]).sum::<f32>() / n;
        let variance = (0..features.rows).map(|r| (features[(r, c)] - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();
        let scale = if std > 1e-6 { 1.0 / std } else { 0.0 };
        for r in 0..features.rows {
            features[(r, c)] = (features[(r, c)] - mean) * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::mel::mel_points;
    use crate::rng::Rng;

    fn sine(frequency: f32, amplitude: f32, sample_rate: u32, len: usize) -> Vec<f32> {
        (0..len).map(|t| amplitude * (2.0 * PI * frequency * t as f32 / sample_rate as f32).sin()).collect()
    }

    #[test]
    fn config_derives_the_documented_sizes() {
        let cfg = FeatureConfig::new(16000);
        assert_eq!((cfg.frame_len(), cfg.hop_len(), cfg.n_fft()), (400, 160, 512));
        assert_eq!((cfg.n_mels, cfg.fmin, cfg.fmax, cfg.preemphasis), (40, 20.0, 8000.0, 0.97));
        assert_eq!(cfg.num_frames(16000), 98);
        let cfg = FeatureConfig::new(8000);
        assert_eq!((cfg.frame_len(), cfg.hop_len(), cfg.n_fft(), cfg.fmax), (200, 80, 256, 4000.0));
    }

    #[test]
    fn log_mel_has_one_row_per_frame_and_one_column_per_band() {
        let cfg = FeatureConfig::new(16000);
        let features = log_mel(&vec![0.0; 16000], &cfg);
        assert_eq!((features.rows, features.cols), (98, 40));
        // Silence: every entry is exactly the log of the floor.
        assert!(features.data.iter().all(|&v| (v - LOG_FLOOR.ln()).abs() < 1e-5));
        assert_eq!(log_mel(&[0.0; 399], &cfg).rows, 0);
    }

    #[test]
    fn a_pure_tone_lights_up_the_band_it_falls_in() {
        let cfg = FeatureConfig { preemphasis: 0.0, ..FeatureConfig::new(16000) };
        let features = log_mel(&sine(1000.0, 0.5, 16000, 16000), &cfg);
        // Which triangle has its peak nearest 1000 Hz?
        let points = mel_points(cfg.n_mels, cfg.fmin, cfg.fmax);
        let expected =
            (0..cfg.n_mels).min_by(|&a, &b| (points[a + 1] - 1000.0).abs().total_cmp(&(points[b + 1] - 1000.0).abs())).unwrap();
        for t in [0, 50, 97] {
            let row = features.row(t);
            let loudest = (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            assert_eq!(loudest, expected, "frame {t}");
        }
    }

    #[test]
    fn mel_energy_of_a_tone_matches_the_hand_calculation() {
        // A sine of amplitude a through a Hann window of length N peaks at
        // |X| = a * N / 4 (N/2 for the sine, times the window's gain of 1/2),
        // and the window spreads it over three bins: the centre one and its
        // two neighbours at half the height. 1000 Hz is exactly bin 32 of 512
        // only if the frame is 512 long, so use a 32 ms frame.
        let cfg = FeatureConfig { preemphasis: 0.0, frame_ms: 32.0, ..FeatureConfig::new(16000) };
        assert_eq!((cfg.frame_len(), cfg.n_fft()), (512, 512));
        let extractor = FeatureExtractor::new(&cfg);
        let features = extractor.log_mel(&sine(1000.0, 0.5, 16000, 4000));

        let fb = extractor.filterbank();
        let peak_power = (0.5f32 * 512.0 / 4.0).powi(2);
        for band in 0..cfg.n_mels {
            let expected_energy =
                fb[(band, 31)] * peak_power / 4.0 + fb[(band, 32)] * peak_power + fb[(band, 33)] * peak_power / 4.0;
            let expected = (expected_energy + LOG_FLOOR).ln();
            // Tolerance scaled to the size of the energy: bands the tone does
            // not reach hold only rounding noise.
            let got = features[(3, band)];
            if expected_energy > 1.0 {
                assert!((got - expected).abs() < 1e-2, "band {band}: {got} vs {expected}");
            } else {
                assert!(got < 0.0, "band {band} should be near silent, got {got}");
            }
        }
    }

    #[test]
    fn ten_times_the_amplitude_adds_ln_100_to_every_band() {
        let cfg = FeatureConfig::new(16000);
        let mut rng = Rng::new(4);
        let quiet: Vec<f32> = (0..4000).map(|_| 0.03 * rng.normal()).collect();
        let loud: Vec<f32> = quiet.iter().map(|x| 10.0 * x).collect();
        let (a, b) = (log_mel(&quiet, &cfg), log_mel(&loud, &cfg));
        for (x, y) in a.data.iter().zip(&b.data) {
            assert!((y - x - 100.0f32.ln()).abs() < 0.05, "{x} -> {y}");
        }
    }

    #[test]
    fn extractor_and_free_functions_agree() {
        let cfg = FeatureConfig::new(8000);
        let samples = sine(300.0, 0.3, 8000, 4000);
        let extractor = FeatureExtractor::new(&cfg);
        assert_eq!(extractor.log_mel(&samples), log_mel(&samples, &cfg));
        assert_eq!(extractor.mfcc(&samples, 13), mfcc(&samples, &cfg, 13));
        assert_eq!(extractor.power_spectrogram(&samples).cols, 129);
    }

    #[test]
    fn dct_by_hand() {
        // M = 2: c0 = (a + b) / sqrt(2), c1 = (a - b) / sqrt(2).
        let c = dct2(&[3.0, 1.0], 2);
        assert!((c[0] - 4.0 / 2.0f32.sqrt()).abs() < 1e-6 && (c[1] - 2.0 / 2.0f32.sqrt()).abs() < 1e-6);
        // A constant has only a zeroth coefficient: value * sqrt(M).
        let c = dct2(&[2.0; 16], 16);
        assert!((c[0] - 8.0).abs() < 1e-5);
        assert!(c[1..].iter().all(|v| v.abs() < 1e-5));
        // One full half-cosine is exactly basis function 1: c1 = sqrt(M/2).
        let half_cosine: Vec<f32> = (0..8).map(|m| (PI * (m as f32 + 0.5) / 8.0).cos()).collect();
        let c = dct2(&half_cosine, 8);
        assert!((c[1] - 2.0).abs() < 1e-5 && c[0].abs() < 1e-5 && c[2].abs() < 1e-5);
    }

    #[test]
    fn dct_is_orthonormal() {
        // D * D^T = identity, so lengths are preserved.
        let d = dct_matrix(40, 40);
        let product = d.matmul_t(&d);
        for r in 0..40 {
            for c in 0..40 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((product[(r, c)] - expected).abs() < 1e-4, "({r}, {c}) = {}", product[(r, c)]);
            }
        }
        let mut rng = Rng::new(2);
        let x: Vec<f32> = (0..40).map(|_| rng.normal()).collect();
        let c = dct2(&x, 40);
        let (ex, ec): (f32, f32) = (x.iter().map(|v| v * v).sum(), c.iter().map(|v| v * v).sum());
        assert!((ex - ec).abs() < 1e-3);
    }

    #[test]
    fn mfcc_is_the_truncated_dct_of_log_mel() {
        let cfg = FeatureConfig::new(16000);
        let mut rng = Rng::new(8);
        let samples: Vec<f32> = (0..8000).map(|_| 0.1 * rng.normal()).collect();
        let (mel, cepstra) = (log_mel(&samples, &cfg), mfcc(&samples, &cfg, 13));
        assert_eq!((cepstra.rows, cepstra.cols), (mel.rows, 13));
        for t in [0, 20, mel.rows - 1] {
            let expected = dct2(mel.row(t), 13);
            for (a, b) in cepstra.row(t).iter().zip(&expected) {
                assert!((a - b).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn deltas_of_a_ramp_are_its_slope() {
        // Column 0 rises by 3 per frame, column 1 is constant.
        let features = Mat::from_fn(10, 2, |r, c| if c == 0 { 3.0 * r as f32 } else { 7.0 });
        let d = deltas(&features, 2);
        assert_eq!((d.rows, d.cols), (10, 2));
        for t in 2..8 {
            assert!((d[(t, 0)] - 3.0).abs() < 1e-5);
        }
        assert!(d.data.iter().skip(1).step_by(2).all(|&v| v == 0.0));
        // At the edge, missing frames repeat the first frame:
        // t = 0: (1*(3-0) + 2*(6-0)) / 10 = 1.5
        assert!((d[(0, 0)] - 1.5).abs() < 1e-5);
        // Width 1 is half the difference of the neighbours.
        let d1 = deltas(&Mat::from_vec(3, 1, vec![1.0, 4.0, 9.0]), 1);
        assert_eq!(d1.data, vec![1.5, 4.0, 2.5]);
    }

    #[test]
    fn normalize_gives_zero_mean_and_unit_variance_per_column() {
        let mut features = Mat::from_vec(4, 3, vec![1.0, 10.0, 5.0, 2.0, 20.0, 5.0, 3.0, 30.0, 5.0, 4.0, 40.0, 5.0]);
        normalize(&mut features);
        // Column 0 is 1,2,3,4: mean 2.5, std sqrt(1.25). Column 1 is ten
        // times that, so it normalizes to the same thing.
        let expected = [-1.5, -0.5, 0.5, 1.5].map(|v| v / 1.25f32.sqrt());
        for r in 0..4 {
            assert!((features[(r, 0)] - expected[r]).abs() < 1e-5);
            assert!((features[(r, 1)] - expected[r]).abs() < 1e-5);
            assert_eq!(features[(r, 2)], 0.0); // constant column
        }
        normalize(&mut Mat::zeros(0, 3)); // must not panic
    }
}
