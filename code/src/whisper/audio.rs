//! Getting audio into the shape Whisper wants: 16 kHz, mono, `f32`.
//!
//! Whisper was trained on audio sampled 16,000 times a second and its front
//! end ([`super::mel`]) is built around that number, so every recording has
//! to arrive at that rate. Nothing new is needed for it: chapter 1's reader
//! ([`Wav`]) decodes the file (any of the PCM and float layouts it knows,
//! any number of channels, mixed down to mono) and the resampler from
//! [`crate::dsp::resample`] changes the rate if it is not already 16 kHz.
//! An 8 kHz telephone-quality clip therefore works too; it simply has
//! nothing above 4 kHz for the model to hear.
//!
//! Formats that are not WAV at all (MP3, FLAC, ...) are still out of scope.
//! Convert those first: `ffmpeg -i in.mp3 out.wav`.

use super::mel::SAMPLE_RATE;
use crate::dsp::resample::resample;
use crate::wav::Wav;
use std::io;
use std::path::Path;

/// Read a WAV file as mono samples in `[-1.0, 1.0]` at 16 kHz, resampling
/// if the file has another sample rate.
pub fn read_wav_16k_mono(path: &Path) -> io::Result<Vec<f32>> {
    Wav::read(path).map(|wav| to_16k(&wav))
}

/// Decode the bytes of a WAV file. See [`read_wav_16k_mono`].
pub fn parse_wav_16k_mono(bytes: &[u8]) -> io::Result<Vec<f32>> {
    Wav::from_bytes(bytes).map(|wav| to_16k(&wav))
}

/// The samples of `wav` at 16 kHz: unchanged if it already has that rate,
/// resampled otherwise.
pub fn to_16k(wav: &Wav) -> Vec<f32> {
    resample(&wav.samples, wav.sample_rate, SAMPLE_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A WAV file holding a 440 Hz tone of the given length.
    fn tone(rate: u32, seconds: f32) -> Wav {
        let n = (rate as f32 * seconds) as usize;
        Wav::new(rate, (0..n).map(|t| 0.5 * (std::f32::consts::TAU * 440.0 * t as f32 / rate as f32).sin()).collect())
    }

    #[test]
    fn sixteen_kilohertz_passes_through_untouched() {
        let wav = tone(16_000, 0.1);
        // Writing PCM16 rounds each sample to a multiple of 1/32768.
        let samples = parse_wav_16k_mono(&wav.to_bytes()).unwrap();
        assert_eq!(samples.len(), 1600);
        assert!(samples.iter().zip(&wav.samples).all(|(a, b)| (a - b).abs() < 1.0 / 32768.0));
    }

    #[test]
    fn other_rates_are_resampled_to_16k() {
        for rate in [8_000, 44_100, 48_000] {
            let samples = parse_wav_16k_mono(&tone(rate, 0.5).to_bytes()).unwrap();
            assert_eq!(samples.len(), 8000, "half a second from {rate} Hz");
            // Still the same 440 Hz tone: compare the middle with a tone
            // generated directly at 16 kHz (the edges ring a little).
            let direct = tone(16_000, 0.5);
            let worst = (2000..6000).map(|t| (samples[t] - direct.samples[t]).abs()).fold(0.0, f32::max);
            assert!(worst < 0.01, "{rate} Hz: off by {worst}");
        }
    }

    #[test]
    fn stereo_is_mixed_down_and_garbage_is_rejected() {
        // Two channels, 16 kHz, PCM16: left = 0.5, right = -0.25 for 4 frames.
        let mut bytes = b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x02\0".to_vec();
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&64_000u32.to_le_bytes());
        bytes.extend_from_slice(&[4, 0, 16, 0]);
        bytes.extend_from_slice(b"data\x10\0\0\0");
        for _ in 0..4 {
            bytes.extend_from_slice(&16384i16.to_le_bytes());
            bytes.extend_from_slice(&(-8192i16).to_le_bytes());
        }
        assert_eq!(parse_wav_16k_mono(&bytes).unwrap(), vec![0.125; 4]);
        assert!(parse_wav_16k_mono(b"not a wav file").is_err());
    }

    #[test]
    fn reads_from_disk() {
        let path = std::env::temp_dir().join(format!("speech-whisper-audio-test-{}.wav", std::process::id()));
        tone(8_000, 0.25).write(&path).unwrap();
        let samples = read_wav_16k_mono(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(samples.unwrap().len(), 4000);
        assert!(read_wav_16k_mono(&path).is_err());
    }
}
