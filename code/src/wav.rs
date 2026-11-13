//! Reading and writing WAV files, by hand.
//!
//! A WAV file is a *RIFF* container: a 12-byte header followed by a list of
//! *chunks*. Every chunk starts with a four-letter name and a four-byte
//! little-endian size, then that many bytes of payload:
//!
//! ```text
//! "RIFF" <file size - 8> "WAVE"
//! "fmt " <16>            format tag, channels, sample rate, ..., bits per sample
//! "LIST" <n>             (optional: title, artist... we skip it)
//! "data" <n>             the samples, interleaved channel by channel
//! ```
//!
//! We only care about two chunks, `fmt ` (how the samples are stored) and
//! `data` (the samples). Everything else is skipped. Whatever the file holds
//! (8/16/24/32-bit integers or 32-bit floats, one channel or several), we
//! hand back the same thing: mono `f32` samples in `[-1.0, 1.0]`.

use std::fs;
use std::io;
use std::path::Path;

/// Format tag for plain integer samples ("pulse code modulation").
const WAVE_FORMAT_PCM: u16 = 1;
/// Format tag for IEEE floating point samples.
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
/// Format tag that means "the real tag is further down, in the extension".
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// A mono recording: a sample rate and the samples themselves.
#[derive(Clone, Debug, PartialEq)]
pub struct Wav {
    /// Samples per second, e.g. 16000.
    pub sample_rate: u32,
    /// Air pressure over time, one number per sample, in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
}

/// How the samples were stored in the file, before we converted them.
/// `Wav` itself is always mono `f32`; this is only for the curious
/// (chapter 1 prints it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavFormat {
    /// How each number is encoded.
    pub encoding: Encoding,
    /// Number of interleaved channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Samples per second, per channel.
    pub sample_rate: u32,
    /// Bits used to store one sample of one channel.
    pub bits_per_sample: u16,
}

/// The two families of sample encodings we understand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Integers. 8-bit is unsigned, everything wider is signed.
    Pcm,
    /// IEEE floating point (32 or 64 bit).
    Float,
}

impl Wav {
    /// Wrap samples we already have in memory.
    pub fn new(sample_rate: u32, samples: Vec<f32>) -> Self {
        Wav { sample_rate, samples }
    }

    /// Read and decode a WAV file.
    pub fn read(path: impl AsRef<Path>) -> io::Result<Wav> {
        Self::read_with_format(path).map(|(wav, _)| wav)
    }

    /// Like [`Wav::read`], but also reports how the file stored its samples.
    pub fn read_with_format(path: impl AsRef<Path>) -> io::Result<(Wav, WavFormat)> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        Self::from_bytes_with_format(&bytes).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))
    }

    /// Decode a WAV file that is already in memory.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Wav> {
        Self::from_bytes_with_format(bytes).map(|(wav, _)| wav)
    }

    /// Like [`Wav::from_bytes`], but also reports how the file stored its samples.
    pub fn from_bytes_with_format(bytes: &[u8]) -> io::Result<(Wav, WavFormat)> {
        if bytes.len() < 12 || &bytes[0..4] != b"RIFF" {
            return Err(bad("not a WAV file: it does not start with \"RIFF\""));
        }
        if &bytes[8..12] != b"WAVE" {
            return Err(bad("a RIFF file, but not a WAVE one (it may be an AVI or WebP)"));
        }
        // Bytes 4..8 hold the size of the rest of the file. We ignore it:
        // programs that stream audio to a pipe do not know it in advance and
        // write 0 or 0xFFFFFFFF, and the real length of `bytes` is what counts.

        let mut format: Option<WavFormat> = None;
        let mut pos = 12;
        // Walk the chunk list. Each iteration looks at one 8-byte chunk header.
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let declared_size = read_u32(bytes, pos + 4) as usize;
            let body_start = pos + 8;
            // A chunk can claim more bytes than the file has (a recording that
            // was cut short, or a streamed file with a placeholder size).
            // Take what is really there.
            let body_end = body_start.saturating_add(declared_size).min(bytes.len());
            let body = &bytes[body_start..body_end];

            match id {
                b"fmt " => format = Some(parse_format(body)?),
                b"data" => {
                    let format = format.ok_or_else(|| bad("the \"data\" chunk came before the \"fmt \" chunk"))?;
                    let samples = decode_samples(body, &format);
                    return Ok((Wav { sample_rate: format.sample_rate, samples }, format));
                }
                _ => {} // "LIST", "fact", "cue ", "bext"...: not our business.
            }

            // Chunks are aligned to even offsets: an odd-sized chunk is
            // followed by one padding byte that its size does not count.
            pos = body_start + declared_size + (declared_size & 1);
        }

        Err(bad(if format.is_some() {
            "no \"data\" chunk: the file has a header but no samples"
        } else {
            "no \"fmt \" chunk found"
        }))
    }

    /// Encode as a 16-bit PCM mono WAV file, the most widely understood kind.
    ///
    /// Samples outside `[-1, 1]` are clipped. Each sample is rounded to one of
    /// 65,536 levels, so reading the file back gives values within half a
    /// level (1/65536) of what was written.
    pub fn to_bytes(&self) -> Vec<u8> {
        let data_size = (self.samples.len() * 2) as u32;
        let mut out = Vec::with_capacity(44 + self.samples.len() * 2);

        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_size).to_le_bytes()); // everything after these 8 bytes
        out.extend_from_slice(b"WAVE");

        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes()); // size of the fmt chunk
        out.extend_from_slice(&WAVE_FORMAT_PCM.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // channels
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&(self.sample_rate * 2).to_le_bytes()); // bytes per second
        out.extend_from_slice(&2u16.to_le_bytes()); // bytes per sample frame ("block align")
        out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_size.to_le_bytes());
        for &sample in &self.samples {
            // The mirror image of reading: multiply by 32768. That makes
            // +1.0 land on 32768, one past the largest i16, hence the clamp.
            let scaled = (sample * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
            out.extend_from_slice(&scaled.to_le_bytes());
        }
        out
    }

    /// Write a 16-bit PCM mono WAV file. See [`Wav::to_bytes`].
    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        fs::write(path, self.to_bytes())
    }

    /// Length of the recording in seconds.
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

/// Parse the body of a `fmt ` chunk.
fn parse_format(body: &[u8]) -> io::Result<WavFormat> {
    if body.len() < 16 {
        return Err(bad(format!("the \"fmt \" chunk is {} bytes long, it needs at least 16", body.len())));
    }
    let mut tag = read_u16(body, 0);
    let channels = read_u16(body, 2);
    let sample_rate = read_u32(body, 4);
    // bytes 8..12: average bytes per second, bytes 12..14: block align.
    // Both can be derived from the other fields, so we do not read them.
    let bits_per_sample = read_u16(body, 14);

    // WAVE_FORMAT_EXTENSIBLE is used for more than two channels or more than
    // 16 bits. The fields above keep their meaning; the real format tag moves
    // into the first two bytes of a 16-byte "sub-format" id at offset 24.
    if tag == WAVE_FORMAT_EXTENSIBLE {
        if body.len() < 26 {
            return Err(bad("WAVE_FORMAT_EXTENSIBLE header is cut short"));
        }
        tag = read_u16(body, 24);
    }

    let encoding = match tag {
        WAVE_FORMAT_PCM => Encoding::Pcm,
        WAVE_FORMAT_IEEE_FLOAT => Encoding::Float,
        6 => return Err(bad("A-law compressed audio is not supported, convert the file to PCM first")),
        7 => return Err(bad("mu-law compressed audio is not supported, convert the file to PCM first")),
        other => return Err(bad(format!("unsupported WAV format tag {other} (we read PCM = 1 and float = 3)"))),
    };
    if channels == 0 {
        return Err(bad("the header says the file has 0 channels"));
    }
    if sample_rate == 0 {
        return Err(bad("the header says the sample rate is 0"));
    }
    match (encoding, bits_per_sample) {
        (Encoding::Pcm, 8 | 16 | 24 | 32) | (Encoding::Float, 32 | 64) => {}
        (Encoding::Pcm, bits) => return Err(bad(format!("unsupported PCM bit depth {bits} (we read 8, 16, 24 and 32)"))),
        (Encoding::Float, bits) => return Err(bad(format!("unsupported float bit depth {bits} (we read 32 and 64)"))),
    }
    Ok(WavFormat { encoding, channels, sample_rate, bits_per_sample })
}

/// Turn the raw bytes of a `data` chunk into mono `f32` samples.
fn decode_samples(data: &[u8], format: &WavFormat) -> Vec<f32> {
    let bytes_per_sample = (format.bits_per_sample / 8) as usize;
    let channels = format.channels as usize;
    // One "sample frame" is one sample for every channel, stored side by side.
    let bytes_per_frame = bytes_per_sample * channels;

    // `chunks_exact` quietly drops a trailing partial frame, which is exactly
    // what we want for a file that was cut off mid-sample.
    let mut samples = Vec::with_capacity(data.len() / bytes_per_frame);
    for frame in data.chunks_exact(bytes_per_frame) {
        // Downmix to mono: the average of the channels.
        let mut sum = 0.0f32;
        for channel in frame.chunks_exact(bytes_per_sample) {
            sum += decode_one(channel, format.encoding);
        }
        samples.push(sum / channels as f32);
    }
    samples
}

/// Decode one little-endian sample of 1, 2, 3, 4 or 8 bytes to `[-1, 1]`.
fn decode_one(b: &[u8], encoding: Encoding) -> f32 {
    match (encoding, b.len()) {
        // 8-bit WAV is the odd one out: unsigned, with silence at 128.
        (Encoding::Pcm, 1) => (b[0] as f32 - 128.0) / 128.0,
        (Encoding::Pcm, 2) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
        // There is no i24, so we put the three bytes in the *top* of an i32
        // (which gets the sign right) and divide by 2^31 instead of 2^23.
        (Encoding::Pcm, 3) => i32::from_le_bytes([0, b[0], b[1], b[2]]) as f32 / 2_147_483_648.0,
        (Encoding::Pcm, 4) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
        (Encoding::Float, 4) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        (Encoding::Float, 8) => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
        _ => unreachable!("parse_format only lets supported sizes through"),
    }
}

fn read_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn bad(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a WAV file byte by byte, so the tests do not depend on `to_bytes`.
    fn build(tag: u16, channels: u16, rate: u32, bits: u16, extra_chunks: &[(&[u8; 4], &[u8])], data: &[u8]) -> Vec<u8> {
        let mut f = Vec::new();
        f.extend_from_slice(b"RIFF\0\0\0\0WAVE");
        f.extend_from_slice(b"fmt ");
        f.extend_from_slice(&16u32.to_le_bytes());
        f.extend_from_slice(&tag.to_le_bytes());
        f.extend_from_slice(&channels.to_le_bytes());
        f.extend_from_slice(&rate.to_le_bytes());
        let block = channels * bits / 8;
        f.extend_from_slice(&(rate * block as u32).to_le_bytes());
        f.extend_from_slice(&block.to_le_bytes());
        f.extend_from_slice(&bits.to_le_bytes());
        for (id, body) in extra_chunks {
            f.extend_from_slice(*id);
            f.extend_from_slice(&(body.len() as u32).to_le_bytes());
            f.extend_from_slice(body);
            if body.len() % 2 == 1 {
                f.push(0);
            }
        }
        f.extend_from_slice(b"data");
        f.extend_from_slice(&(data.len() as u32).to_le_bytes());
        f.extend_from_slice(data);
        let riff_size = (f.len() - 8) as u32;
        f[4..8].copy_from_slice(&riff_size.to_le_bytes());
        f
    }

    #[test]
    fn pcm16_values_are_scaled_by_32768() {
        // 0, +16384 (half scale), -32768 (most negative), +32767.
        let data = [0x00, 0x00, 0x00, 0x40, 0x00, 0x80, 0xFF, 0x7F];
        let wav = Wav::from_bytes(&build(1, 1, 8000, 16, &[], &data)).unwrap();
        assert_eq!(wav.sample_rate, 8000);
        assert_eq!(wav.samples, vec![0.0, 0.5, -1.0, 32767.0 / 32768.0]);
    }

    #[test]
    fn pcm8_is_unsigned_with_silence_at_128() {
        let wav = Wav::from_bytes(&build(1, 1, 8000, 8, &[], &[128, 255, 0, 192])).unwrap();
        assert_eq!(wav.samples, vec![0.0, 127.0 / 128.0, -1.0, 0.5]);
    }

    #[test]
    fn pcm24_and_pcm32_sign_extend_correctly() {
        // 24-bit: +0x400000 (half scale), -0x800000 (minimum), -1.
        let d24 = [0x00, 0x00, 0x40, 0x00, 0x00, 0x80, 0xFF, 0xFF, 0xFF];
        let wav = Wav::from_bytes(&build(1, 1, 44100, 24, &[], &d24)).unwrap();
        assert_eq!(wav.samples, vec![0.5, -1.0, -1.0 / 8_388_608.0]);

        let mut d32 = Vec::new();
        d32.extend_from_slice(&(1i32 << 30).to_le_bytes());
        d32.extend_from_slice(&i32::MIN.to_le_bytes());
        let wav = Wav::from_bytes(&build(1, 1, 44100, 32, &[], &d32)).unwrap();
        assert_eq!(wav.samples, vec![0.5, -1.0]);
    }

    #[test]
    fn float32_is_read_as_is() {
        let mut data = Vec::new();
        for v in [0.25f32, -0.75, 1.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let (wav, format) = Wav::from_bytes_with_format(&build(3, 1, 16000, 32, &[], &data)).unwrap();
        assert_eq!(wav.samples, vec![0.25, -0.75, 1.0]);
        assert_eq!(format.encoding, Encoding::Float);
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        // Two frames: (L=+0.5, R=-0.5) -> 0, (L=+0.5, R=+0.25) -> 0.375.
        let mut data = Vec::new();
        for v in [16384i16, -16384, 16384, 8192] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let (wav, format) = Wav::from_bytes_with_format(&build(1, 2, 8000, 16, &[], &data)).unwrap();
        assert_eq!(format.channels, 2);
        assert_eq!(wav.samples, vec![0.0, 0.375]);
    }

    #[test]
    fn unknown_and_odd_sized_chunks_are_skipped() {
        // A 5-byte LIST chunk needs one padding byte before the next chunk.
        let extra: [(&[u8; 4], &[u8]); 2] = [(b"LIST", b"hello"), (b"fact", &[1, 2, 3, 4])];
        let wav = Wav::from_bytes(&build(1, 1, 8000, 16, &extra, &[0x00, 0x40])).unwrap();
        assert_eq!(wav.samples, vec![0.5]);
    }

    #[test]
    fn extensible_format_looks_inside_the_subformat() {
        // Hand-written 40-byte fmt chunk: tag 0xFFFE, real tag (3 = float) in the GUID.
        let mut f = Vec::new();
        f.extend_from_slice(b"RIFF\0\0\0\0WAVEfmt ");
        f.extend_from_slice(&40u32.to_le_bytes());
        f.extend_from_slice(&0xFFFEu16.to_le_bytes());
        f.extend_from_slice(&1u16.to_le_bytes()); // channels
        f.extend_from_slice(&48000u32.to_le_bytes());
        f.extend_from_slice(&(48000u32 * 4).to_le_bytes());
        f.extend_from_slice(&4u16.to_le_bytes()); // block align
        f.extend_from_slice(&32u16.to_le_bytes()); // bits
        f.extend_from_slice(&22u16.to_le_bytes()); // extension size
        f.extend_from_slice(&32u16.to_le_bytes()); // valid bits
        f.extend_from_slice(&4u32.to_le_bytes()); // channel mask
        f.extend_from_slice(&[3, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
        f.extend_from_slice(b"data");
        f.extend_from_slice(&4u32.to_le_bytes());
        f.extend_from_slice(&0.5f32.to_le_bytes());
        let (wav, format) = Wav::from_bytes_with_format(&f).unwrap();
        assert_eq!(format, WavFormat { encoding: Encoding::Float, channels: 1, sample_rate: 48000, bits_per_sample: 32 });
        assert_eq!(wav.samples, vec![0.5]);
    }

    #[test]
    fn truncated_data_chunk_yields_the_samples_that_are_there() {
        let mut file = build(1, 1, 8000, 16, &[], &[0x00, 0x40, 0x00, 0xC0, 0x00, 0x40]);
        // Lie about the size (as a streaming encoder would) and cut the file
        // in the middle of the third sample.
        let size_at = file.len() - 6 - 4;
        file[size_at..size_at + 4].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        file.truncate(file.len() - 1);
        let wav = Wav::from_bytes(&file).unwrap();
        assert_eq!(wav.samples, vec![0.5, -0.5]);
    }

    #[test]
    fn errors_say_what_is_wrong() {
        let message = |bytes: &[u8]| Wav::from_bytes(bytes).unwrap_err().to_string();
        assert!(message(b"ID3\x03 this is an mp3").contains("RIFF"));
        assert!(message(b"RIFF\0\0\0\0AVI LIST").contains("WAVE"));
        assert!(message(&build(7, 1, 8000, 8, &[], &[0])).contains("mu-law"));
        assert!(message(&build(1, 1, 8000, 12, &[], &[0])).contains("bit depth 12"));
        assert!(message(b"RIFF\0\0\0\0WAVEdata\x02\0\0\0\0\0").contains("fmt "));
        let mut no_data = build(1, 1, 8000, 16, &[], &[]);
        no_data.truncate(36); // header and fmt chunk only
        assert!(message(&no_data).contains("no \"data\" chunk"));
    }

    #[test]
    fn write_then_read_round_trips_within_one_pcm16_step() {
        let original = Wav::new(16000, (0..1000).map(|i| (i as f32 * 0.05).sin() * 0.9).collect());
        let bytes = original.to_bytes();
        assert_eq!(bytes.len(), 44 + 2000);
        let back = Wav::from_bytes(&bytes).unwrap();
        assert_eq!(back.sample_rate, 16000);
        assert_eq!(back.samples.len(), 1000);
        for (a, b) in original.samples.iter().zip(&back.samples) {
            assert!((a - b).abs() <= 0.5 / 32768.0 + 1e-6, "{a} vs {b}");
        }
        assert!((original.duration_secs() - 0.0625).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_samples_are_clipped_on_write() {
        let back = Wav::from_bytes(&Wav::new(8000, vec![2.0, -2.0]).to_bytes()).unwrap();
        assert_eq!(back.samples, vec![32767.0 / 32768.0, -1.0]);
    }
}
