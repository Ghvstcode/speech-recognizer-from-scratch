//! The mel scale and the mel filterbank.
//!
//! The FFT spaces its bins evenly in Hz. Our ears do not work that way: the
//! step from 200 Hz to 400 Hz sounds huge (an octave), the step from 6200 Hz
//! to 6400 Hz is barely audible. The *mel scale* (from "melody") is a
//! re-labeling of the frequency axis on which equal distances sound equally
//! far apart: close to linear below about 1000 Hz, logarithmic above.
//!
//! A mel filterbank summarizes the few hundred FFT bins as a few dozen
//! numbers by adding up the energy in overlapping bands that are evenly
//! spaced *in mels*: narrow bands at the bottom, where the ear is picky, and
//! wide ones at the top, where it is not.

use crate::mat::Mat;

/// Convert a frequency in Hz to mels: `2595 * log10(1 + hz / 700)`.
///
/// This is the formula used by HTK (the Hidden Markov Model Toolkit), by
/// Kaldi and in most papers. The constants are chosen so that 1000 Hz comes
/// out as (almost exactly) 1000 mel.
///
/// It is not the only "mel scale". The experiments behind it are from the
/// 1930s and several curves have been fitted to them since. The other common
/// one is Slaney's (the default in librosa, and what Whisper was trained
/// with): exactly linear below 1000 Hz and logarithmic above. The two differ
/// by a few percent. What matters is to use the same one for training and
/// for recognition.
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mels back to Hz: the inverse of [`hz_to_mel`].
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// `n_mels + 2` frequencies in Hz, evenly spaced on the mel scale from `fmin`
/// to `fmax` inclusive.
///
/// Filter `m` of the filterbank starts at point `m`, peaks at point `m + 1`
/// and ends at point `m + 2`, so each filter's peak is where its neighbors
/// begin and end.
pub fn mel_points(n_mels: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let (mel_min, mel_max) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let step = (mel_max - mel_min) / (n_mels + 1) as f32;
    let mut points: Vec<f32> = (0..n_mels + 2).map(|i| mel_to_hz(mel_min + step * i as f32)).collect();
    // Converting to mels and back picks up a little rounding error. Pin the
    // two ends, so that the bank covers exactly fmin..fmax and not a hair more.
    points[0] = fmin;
    points[n_mels + 1] = fmax;
    points
}

/// A bank of `n_mels` triangular filters as a matrix: one row per filter,
/// one column per FFT bin (`n_fft / 2 + 1`). Multiplying it with a power
/// spectrum gives the mel energies.
///
/// Row `m` is a triangle over frequency: zero up to `points[m]`, rising in a
/// straight line (in Hz) to 1 at `points[m + 1]`, falling back to zero at
/// `points[m + 2]`, with the points from [`mel_points`]. Each weight is the
/// height of that triangle at the exact frequency of the bin.
///
/// # Normalization
///
/// Every triangle has **peak height 1** (the HTK convention). Two things
/// follow. Neighboring triangles add up to exactly 1 everywhere between the
/// first and the last peak, so no frequency in that range counts more than
/// another. And the wide filters at the top collect more bins than the
/// narrow ones at the bottom, so white noise gives mel energies that grow
/// with frequency. The other common choice (Slaney's, used by librosa and
/// Whisper) divides each triangle by its width so white noise comes out
/// flat. After the logarithm the two differ only by a constant per band,
/// which per-feature normalization or the first layer of a network absorbs.
pub fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32, fmin: f32, fmax: f32) -> Mat {
    assert!(fmin >= 0.0 && fmin < fmax, "need 0 <= fmin < fmax, got {fmin} and {fmax}");
    assert!(fmax <= sample_rate as f32 / 2.0 + 1e-3, "fmax {fmax} Hz is above the Nyquist frequency of {sample_rate} Hz audio");
    let points = mel_points(n_mels, fmin, fmax);
    let n_bins = n_fft / 2 + 1;
    let hz_per_bin = sample_rate as f32 / n_fft as f32;

    Mat::from_fn(n_mels, n_bins, |m, bin| {
        let (left, peak, right) = (points[m], points[m + 1], points[m + 2]);
        let hz = bin as f32 * hz_per_bin;
        let rising = (hz - left) / (peak - left);
        let falling = (right - hz) / (right - peak);
        // Inside the triangle the lower of the two lines is the triangle;
        // outside, one of them is negative and we clamp to zero.
        rising.min(falling).max(0.0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_reference_points() {
        assert_eq!(hz_to_mel(0.0), 0.0);
        // 2595 * log10(1 + 1000/700) = 2595 * 0.385351 = 999.99
        assert!((hz_to_mel(1000.0) - 1000.0).abs() < 0.1);
        // 2595 * log10(1 + 8000/700) = 2595 * 1.094421 = 2840.02
        assert!((hz_to_mel(8000.0) - 2840.02).abs() < 0.01);
    }

    #[test]
    fn mel_and_hz_are_inverses() {
        for hz in [0.0, 20.0, 440.0, 1000.0, 4000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < hz * 1e-4 + 1e-2, "{hz}");
        }
    }

    #[test]
    fn mel_points_are_evenly_spaced_in_mel_and_span_the_range() {
        let points = mel_points(40, 20.0, 8000.0);
        assert_eq!(points.len(), 42);
        assert!((points[0] - 20.0).abs() < 0.01 && (points[41] - 8000.0).abs() < 1.0);
        let mels: Vec<f32> = points.iter().map(|&hz| hz_to_mel(hz)).collect();
        let step = mels[1] - mels[0];
        for pair in mels.windows(2) {
            assert!((pair[1] - pair[0] - step).abs() < 0.05);
        }
        // In Hz the spacing keeps growing.
        assert!(points[41] - points[40] > 10.0 * (points[1] - points[0]));
    }

    #[test]
    fn a_tiny_filterbank_by_hand() {
        // 2 filters over 0..4000 Hz with 1000 Hz bins (n_fft = 8 at 8 kHz).
        // mel(4000) = 2146.06; a third of that is 715.35 mel = 620.6 Hz, two
        // thirds are 1430.7 mel = 1791.4 Hz. So the points are
        // 0, 620.6, 1791.4, 4000 Hz and the bins sit at 0, 1000, ..., 4000.
        let fb = mel_filterbank(2, 8, 8000, 0.0, 4000.0);
        assert_eq!((fb.rows, fb.cols), (2, 5));
        let expected = [
            // filter 0: up 0 -> 620.6, down 620.6 -> 1791.4
            [0.0, (1791.4 - 1000.0) / (1791.4 - 620.6), 0.0, 0.0, 0.0],
            // filter 1: up 620.6 -> 1791.4, down 1791.4 -> 4000
            [
                0.0,
                (1000.0 - 620.6) / (1791.4 - 620.6),
                (4000.0 - 2000.0) / (4000.0 - 1791.4),
                (4000.0 - 3000.0) / (4000.0 - 1791.4),
                0.0,
            ],
        ];
        for m in 0..2 {
            for bin in 0..5 {
                assert!(
                    (fb[(m, bin)] - expected[m][bin]).abs() < 1e-3,
                    "filter {m} bin {bin}: {} vs {}",
                    fb[(m, bin)],
                    expected[m][bin]
                );
            }
        }
    }

    /// Check the shape of every filter for one configuration.
    fn check_filters_are_triangles(sample_rate: u32, n_fft: usize) {
        let (n_mels, fmin, fmax) = (40, 20.0, sample_rate as f32 / 2.0);
        let fb = mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax);
        let points = mel_points(n_mels, fmin, fmax);
        let hz_per_bin = sample_rate as f32 / n_fft as f32;

        for m in 0..n_mels {
            let row = fb.row(m);
            assert!(row.iter().all(|w| (0.0..=1.0).contains(w)));
            let peak_bin = (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            // The tallest bin is the one nearest the filter's center frequency...
            assert!((peak_bin as f32 * hz_per_bin - points[m + 1]).abs() <= hz_per_bin, "filter {m} peaks at the wrong place");
            assert!(row[peak_bin] > 0.0, "filter {m} is empty: no FFT bin falls inside it");
            // ...weights rise up to it and fall after it...
            assert!(row[..=peak_bin].windows(2).all(|w| w[0] <= w[1]), "filter {m} does not rise monotonically");
            assert!(row[peak_bin..].windows(2).all(|w| w[0] >= w[1]), "filter {m} does not fall monotonically");
            // ...and it is zero outside its own band.
            for (bin, &w) in row.iter().enumerate() {
                let hz = bin as f32 * hz_per_bin;
                assert!(w == 0.0 || (hz > points[m] && hz < points[m + 2]), "filter {m} leaks to {hz} Hz");
            }
        }

        // Together the filters cover fmin..fmax: every bin strictly inside is
        // seen by some filter, and between the first and the last peak the
        // weights add up to exactly 1.
        for bin in 0..fb.cols {
            let hz = bin as f32 * hz_per_bin;
            let total: f32 = (0..n_mels).map(|m| fb[(m, bin)]).sum();
            if hz > fmin && hz < fmax {
                assert!(total > 0.0, "nothing covers {hz} Hz");
            } else {
                assert_eq!(total, 0.0, "{hz} Hz is outside fmin..fmax but has weight");
            }
            if hz >= points[1] && hz <= points[n_mels] {
                assert!((total - 1.0).abs() < 1e-4, "weights at {hz} Hz sum to {total}");
            }
        }
    }

    #[test]
    fn filters_are_triangles_that_cover_the_band_at_16_khz() {
        check_filters_are_triangles(16000, 512);
    }

    #[test]
    fn filters_are_triangles_that_cover_the_band_at_8_khz() {
        check_filters_are_triangles(8000, 256);
    }
}
