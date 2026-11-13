//! The spectrogram: a picture of how the spectrum changes over time.
//!
//! One FFT tells us which frequencies are in one short frame. Do it for
//! every frame, stack the results, and we get a grid with time along one
//! side and frequency along the other. This is the *short-time Fourier
//! transform* (STFT), and its squared magnitude is the power spectrogram.

use crate::dsp::fft::{Complex, Fft};
use crate::dsp::frame::frames;
use crate::mat::Mat;

/// Power spectrogram: one row per frame, one column per frequency bin
/// (`n_fft / 2 + 1` of them, from 0 Hz up to half the sample rate).
///
/// Each frame is `window.len()` samples long, starts `hop_len` samples after
/// the previous one, is multiplied by `window`, zero-padded to `n_fft` and
/// transformed. Frames follow the edge policy of
/// [`num_frames`](crate::dsp::frame::num_frames). The values are raw
/// `|X[k]|^2` with no scaling; see [`power_to_db`] for something to look at.
pub fn power_spectrogram(samples: &[f32], window: &[f32], hop_len: usize, n_fft: usize) -> Mat {
    let fft = Fft::new(n_fft);
    let n_bins = n_fft / 2 + 1;
    let all_frames = frames(samples, window.len(), hop_len);

    let mut out = Mat::zeros(all_frames.len(), n_bins);
    // Two buffers reused for every frame, so the loop does not allocate.
    let mut windowed = vec![0.0f32; window.len()];
    let mut scratch: Vec<Complex> = Vec::with_capacity(n_fft);
    for (i, frame) in all_frames.into_iter().enumerate() {
        for ((w, x), win) in windowed.iter_mut().zip(frame).zip(window) {
            *w = x * win;
        }
        fft.power_spectrum_into(&windowed, &mut scratch, out.row_mut(i));
    }
    out
}

/// The stretch of time, in seconds, that the frames of a spectrogram cover
/// when each frame is drawn as a cell one hop wide, centred on the middle of
/// the frame. Returns `(left edge of the first cell, right edge of the last)`.
///
/// Use it to label the time axis of a plot. The first cell does not start at
/// zero: frame 0 covers `0 .. frame_len` samples, so its centre is half a
/// frame in.
pub fn time_extent(n_frames: usize, frame_len: usize, hop_len: usize, sample_rate: u32) -> (f32, f32) {
    let first_centre = frame_len as f32 / 2.0;
    let half_hop = hop_len as f32 / 2.0;
    let start = first_centre - half_hop;
    let end = first_centre + n_frames.saturating_sub(1) as f32 * hop_len as f32 + half_hop;
    (start / sample_rate as f32, end / sample_rate as f32)
}

/// Convert one power value to decibels: `10 log10(power)`, with a floor at
/// -100 dB so that zero does not become minus infinity.
pub fn db(power: f32) -> f32 {
    10.0 * power.max(1e-10).log10()
}

/// Convert a power spectrogram to decibels **relative to its loudest cell**,
/// keeping only the top `dynamic_range_db`.
///
/// The loudest cell becomes 0 dB and everything else is negative; anything
/// more than `dynamic_range_db` below the peak is clipped to
/// `-dynamic_range_db`. Raw power spans a ratio of a billion to one, and
/// plotted directly the picture is one bright dot on a blank page. In
/// decibels, with the bottom cut off at 60 to 80 dB, the structure shows.
pub fn power_to_db(power: &Mat, dynamic_range_db: f32) -> Mat {
    let peak = power.data.iter().copied().fold(0.0f32, f32::max);
    let peak_db = db(peak);
    power.map(|p| (db(p) - peak_db).max(-dynamic_range_db))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::window::{hann, rectangular};
    use std::f32::consts::PI;

    #[test]
    fn shape_is_frames_by_bins() {
        let s = power_spectrogram(&vec![0.0; 16000], &hann(400), 160, 512);
        assert_eq!((s.rows, s.cols), (98, 257));
        assert_eq!(power_spectrogram(&[0.0; 100], &hann(400), 160, 512).rows, 0);
    }

    #[test]
    fn a_tone_that_changes_pitch_shows_up_in_the_right_bins() {
        // 0.5 s of 1000 Hz then 0.5 s of 3000 Hz at 16 kHz; 512-sample
        // frames, so bins are 31.25 Hz apart: bin 32, then bin 96.
        let samples: Vec<f32> = (0..16000)
            .map(|t| {
                let f = if t < 8000 { 1000.0 } else { 3000.0 };
                (2.0 * PI * f * t as f32 / 16000.0).sin()
            })
            .collect();
        let s = power_spectrogram(&samples, &rectangular(512), 512, 512);
        let loudest_bin = |row: usize| (0..s.cols).max_by(|&a, &b| s[(row, a)].total_cmp(&s[(row, b)])).unwrap();
        assert_eq!(loudest_bin(2), 32);
        assert_eq!(loudest_bin(s.rows - 2), 96);
        // Rectangular window, sine exactly on a bin: |X| = n/2 = 256.
        assert!((s[(2, 32)].sqrt() - 256.0).abs() < 0.1);
    }

    #[test]
    fn time_extent_centres_cells_on_frames() {
        // 98 frames of 400 samples every 160 at 16 kHz: the first frame is
        // centred on sample 200, the last on 200 + 97 * 160 = 15720.
        let (start, end) = time_extent(98, 400, 160, 16000);
        assert!((start - 120.0 / 16000.0).abs() < 1e-7);
        assert!((end - 15800.0 / 16000.0).abs() < 1e-7);
    }

    #[test]
    fn decibels_are_relative_to_the_peak_and_clipped() {
        let power = Mat::from_vec(1, 4, vec![100.0, 10.0, 1.0, 0.0]);
        let d = power_to_db(&power, 15.0);
        assert_eq!(d.data, vec![0.0, -10.0, -15.0, -15.0]);
        assert_eq!(db(1000.0), 30.0);
        assert_eq!(db(0.0), -100.0);
    }
}
