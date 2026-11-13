//! Running OpenAI's Whisper `tiny.en` from its published weights.
//!
//! Until now every number in our recognizer was one we trained. This module
//! runs a model somebody else trained, on 680,000 hours of audio, and the
//! point of the chapter is that nothing more is needed to do so than what
//! the reader can already write:
//!
//! | file | what it does |
//! |---|---|
//! | `json.rs` | a recursive-descent JSON parser |
//! | `safetensors.rs` | reads the weights file: a JSON header and raw `f32`s |
//! | `config.rs` | the model's sizes and decoding rules, from two JSON files |
//! | `audio.rs` | any WAV file to 16 kHz mono samples |
//! | `mel.rs` | Whisper's exact log-mel spectrogram |
//! | `layers.rs` | linear, convolution, GELU, layer norm, attention |
//! | `encoder.rs` | spectrogram to audio features |
//! | `decoder.rs` | audio features and tokens so far to next-token scores |
//! | `tokenizer.rs` | token ids back to text |
//! | `transcribe.rs` | the greedy decoding loop that ties it together |
//! | `parallel.rs` | one helper for using all CPU cores |
//!
//! ```no_run
//! use speech::whisper::{audio, model_dir, DecodeOptions, Whisper};
//! let model = Whisper::load(&model_dir())?;
//! let samples = audio::read_wav_16k_mono(std::path::Path::new("hello.wav"))?;
//! println!("{}", model.transcribe(&samples, &DecodeOptions::default()).text);
//! # Ok::<(), std::io::Error>(())
//! ```

pub mod audio;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod json;
pub mod layers;
pub mod mel;
pub mod parallel;
pub mod safetensors;
pub mod tokenizer;
pub mod transcribe;

pub use transcribe::{DecodeOptions, Transcript, Whisper};

use std::path::PathBuf;

/// Where the Whisper files live: `$SPEECH_DATA/whisper-tiny.en`, with
/// `SPEECH_DATA` defaulting to `../data` (relative to the `code/` directory).
pub fn model_dir() -> PathBuf {
    let data = std::env::var_os("SPEECH_DATA").map_or_else(|| PathBuf::from("../data"), PathBuf::from);
    data.join("whisper-tiny.en")
}
