//! A minimal WAV reader for the Whisper example.
//!
//! TODO(integration): replace with crate::wav::Wav::read
//!
//! Chapter 1 builds a complete WAV reader (`crate::wav`). This is a
//! deliberately narrow stand-in that accepts the one layout Whisper wants:
//! 16 kHz, mono, 16-bit PCM. Anything else is rejected with the `ffmpeg`
//! command that converts it.

use std::io;
use std::path::Path;

fn bad(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Read a 16 kHz mono PCM16 WAV file as samples in `[-1.0, 1.0]`.
pub fn read_wav_16k_mono(path: &Path) -> io::Result<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    parse_wav_16k_mono(&bytes).map_err(|e| {
        let hint = "convert with: ffmpeg -i in.any -ar 16000 -ac 1 -c:a pcm_s16le out.wav";
        io::Error::new(e.kind(), format!("{}: {e} ({hint})", path.display()))
    })
}

/// Parse the bytes of a WAV file. See `read_wav_16k_mono`.
pub fn parse_wav_16k_mono(bytes: &[u8]) -> io::Result<Vec<f32>> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file".into()));
    }
    let u16_at = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    let u32_at = |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);

    // A WAV file is a list of chunks: 4-byte id, 4-byte length, payload.
    let mut format_ok = false;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_at(pos + 4) as usize;
        let body = pos + 8;
        // Streaming encoders may not know the final size; clamp to the file.
        let end = body.saturating_add(size).min(bytes.len());
        if id == b"fmt " {
            if end - body < 16 {
                return Err(bad("fmt chunk too short".into()));
            }
            let (encoding, channels, rate, bits) = (u16_at(body), u16_at(body + 2), u32_at(body + 4), u16_at(body + 14));
            if encoding != 1 || channels != 1 || rate != 16_000 || bits != 16 {
                return Err(bad(format!(
                    "need 16000 Hz mono 16-bit PCM, found encoding {encoding}, {channels} channel(s), {rate} Hz, {bits} bits"
                )));
            }
            format_ok = true;
        } else if id == b"data" {
            if !format_ok {
                return Err(bad("data chunk before fmt chunk".into()));
            }
            let samples = bytes[body..end].as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b) as f32 / 32768.0);
            return Ok(samples.collect());
        }
        pos = end + (size & 1); // chunks are padded to an even length
    }
    Err(bad("no data chunk".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let mut out = b"RIFF\0\0\0\0WAVE".to_vec();
        out.extend_from_slice(b"LIST\x03\0\0\0abc\0"); // an odd-sized chunk to skip
        out.extend_from_slice(b"fmt \x10\0\0\0\x01\0");
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * 2).to_le_bytes());
        out.extend_from_slice(&[2, 0, 16, 0]);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(samples.len() as u32 * 2).to_le_bytes());
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    #[test]
    fn reads_samples_and_skips_other_chunks() {
        let samples = parse_wav_16k_mono(&wav(16_000, 1, &[0, 16384, -32768, 32767])).unwrap();
        assert_eq!(samples, vec![0.0, 0.5, -1.0, 32767.0 / 32768.0]);
    }

    #[test]
    fn rejects_other_formats() {
        assert!(parse_wav_16k_mono(&wav(8_000, 1, &[0])).is_err());
        assert!(parse_wav_16k_mono(&wav(16_000, 2, &[0, 0])).is_err());
        assert!(parse_wav_16k_mono(b"not a wav file").is_err());
    }
}
