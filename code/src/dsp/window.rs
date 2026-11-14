//! Window functions.
//!
//! Cutting a frame out of a signal is the same as multiplying the signal by
//! a rectangle: 1 inside the frame, 0 outside. The Fourier transform then
//! treats the frame as if it repeated forever, and wherever the end of the
//! frame does not line up with its beginning it sees a sudden jump. That jump
//! smears energy over all frequencies ("spectral leakage").
//!
//! A window function fades the frame in and out so both ends meet at (or
//! near) zero. The jump disappears and so does most of the leakage. The price
//! is that each frequency peak gets a little wider.
//!
//! # Periodic or symmetric?
//!
//! There are two ways to sample the same bell-shaped curve with `n` points,
//! and they differ only in the denominator:
//!
//! * **symmetric** divides by `n - 1`. The first and the last sample are both
//!   exactly at the ends of the curve, so the window is a mirror image of
//!   itself. That is what you want when *designing a filter* (our resampler
//!   uses it).
//! * **periodic** divides by `n`. It is the symmetric window of length
//!   `n + 1` with its last sample dropped. Lay copies of it end to end and
//!   the curve continues smoothly, which is exactly how the FFT thinks about
//!   a frame. That is what you want for *spectral analysis*, so the plain
//!   [`hann`] and [`hamming`] here are periodic.
//!
//! For a 400-sample frame the two differ by less than 1%; the distinction
//! matters mostly when you compare numbers with another library.

use std::f32::consts::PI;

/// The periodic Hann window: `0.5 - 0.5 cos(2 pi i / n)`.
///
/// One full period of a raised cosine. It starts at exactly 0, peaks at 1 in
/// the middle, ends one step short of 0, and is our default for everything.
pub fn hann(n: usize) -> Vec<f32> {
    raised_cosine(n, n, 0.5)
}

/// The symmetric Hann window: `0.5 - 0.5 cos(2 pi i / (n - 1))`.
pub fn hann_symmetric(n: usize) -> Vec<f32> {
    raised_cosine(n, n.saturating_sub(1), 0.5)
}

/// The periodic Hamming window: `0.54 - 0.46 cos(2 pi i / n)`.
///
/// A Hann window lifted so that its ends sit at 0.08 instead of 0. That odd
/// choice cancels the largest side lobe of the spectrum, at the cost of side
/// lobes further out that fade more slowly. Classic speech front ends (HTK,
/// Kaldi) use it; the difference from Hann is small in practice.
pub fn hamming(n: usize) -> Vec<f32> {
    raised_cosine(n, n, 0.54)
}

/// The symmetric Hamming window: `0.54 - 0.46 cos(2 pi i / (n - 1))`.
pub fn hamming_symmetric(n: usize) -> Vec<f32> {
    raised_cosine(n, n.saturating_sub(1), 0.54)
}

/// The rectangular window: all ones, i.e. "no window at all". It is here so
/// that we can compare the others against it.
pub fn rectangular(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

/// The family both Hann and Hamming belong to:
/// `a - (1 - a) cos(2 pi i / denominator)` for `i` in `0..n`.
fn raised_cosine(n: usize, denominator: usize, a: f32) -> Vec<f32> {
    if denominator == 0 {
        return vec![1.0; n]; // a window of length 0 or 1 has no shape
    }
    (0..n).map(|i| a - (1.0 - a) * (2.0 * PI * i as f32 / denominator as f32).cos()).collect()
}

/// Multiply a frame by a window, sample by sample.
pub fn apply(frame: &[f32], window: &[f32]) -> Vec<f32> {
    assert_eq!(frame.len(), window.len(), "frame has {} samples but the window has {}", frame.len(), window.len());
    frame.iter().zip(window).map(|(x, w)| x * w).collect()
}

/// The average height of a window.
///
/// Windowing makes a frame quieter: a sine of amplitude 1 seen through a Hann
/// window shows up in the spectrum as if its amplitude were 0.5. Divide by
/// this number to undo that.
pub fn coherent_gain(window: &[f32]) -> f32 {
    window.iter().sum::<f32>() / window.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-6, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn hann_matches_hand_computed_values() {
        // Periodic, n = 4: 0.5 - 0.5 cos(pi i / 2) = 0, 0.5, 1, 0.5.
        assert_close(&hann(4), &[0.0, 0.5, 1.0, 0.5]);
        // Symmetric, n = 5: 0.5 - 0.5 cos(pi i / 2) = 0, 0.5, 1, 0.5, 0.
        assert_close(&hann_symmetric(5), &[0.0, 0.5, 1.0, 0.5, 0.0]);
    }

    #[test]
    fn hamming_matches_hand_computed_values() {
        assert_close(&hamming(4), &[0.08, 0.54, 1.0, 0.54]);
        assert_close(&hamming_symmetric(5), &[0.08, 0.54, 1.0, 0.54, 0.08]);
    }

    #[test]
    fn periodic_is_symmetric_of_one_more_with_the_last_dropped() {
        assert_close(&hann(64), &hann_symmetric(65)[..64]);
        assert_close(&hamming(64), &hamming_symmetric(65)[..64]);
    }

    #[test]
    fn symmetric_windows_are_mirror_images() {
        let w = hamming_symmetric(31);
        for i in 0..31 {
            assert!((w[i] - w[30 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn coherent_gains_are_the_textbook_ones() {
        assert!((coherent_gain(&rectangular(400)) - 1.0).abs() < 1e-6);
        assert!((coherent_gain(&hann(400)) - 0.5).abs() < 1e-5);
        assert!((coherent_gain(&hamming(400)) - 0.54).abs() < 1e-5);
    }

    #[test]
    fn tiny_windows_do_not_divide_by_zero() {
        assert_eq!(hann(0), Vec::<f32>::new());
        assert_eq!(hann_symmetric(1), vec![1.0]);
        assert_eq!(apply(&[2.0, 4.0], &[0.5, 0.25]), vec![1.0, 1.0]);
    }
}
