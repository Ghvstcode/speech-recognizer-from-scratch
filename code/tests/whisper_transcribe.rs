//! End-to-end tests for the Whisper engine against the real weights.
//!
//! These need files that are not in the repository, so they skip (and say
//! so) unless both of these are available:
//!
//! - the model, in `$SPEECH_DATA/whisper-tiny.en/` (default `../data/...`);
//! - a WAV recording, named by `WHISPER_TEST_WAV` (any sample rate and
//!   channel count our reader understands; it is brought to 16 kHz mono).
//!   The recipe below writes 16 kHz mono directly, so the test does not
//!   depend on the resampler.
//!
//! Optionally `WHISPER_TEST_TEXT` gives words the transcript must contain
//! (compared ignoring case and punctuation):
//!
//! ```text
//! say -o fox.aiff "The quick brown fox jumps over the lazy dog."
//! ffmpeg -i fox.aiff -ar 16000 -ac 1 -c:a pcm_s16le fox.wav
//! WHISPER_TEST_WAV=fox.wav WHISPER_TEST_TEXT="quick brown fox" cargo test --test whisper_transcribe -- --nocapture
//! ```

use speech::whisper::{audio, model_dir, DecodeOptions, Whisper};
use std::path::PathBuf;

/// The model and the test recording, or `None` (with a note) if either is missing.
fn setup() -> Option<(Whisper, Vec<f32>)> {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: no Whisper weights in {}", dir.display());
        return None;
    }
    let Some(wav) = std::env::var_os("WHISPER_TEST_WAV").map(PathBuf::from) else {
        eprintln!("skipping: set WHISPER_TEST_WAV to a WAV file to run this test");
        return None;
    };
    let model = Whisper::load(&dir).expect("the model files should load");
    let samples = audio::read_wav_16k_mono(&wav).expect("WHISPER_TEST_WAV should be a WAV file our reader understands");
    Some((model, samples))
}

/// Lowercase letters, digits and single spaces only.
fn normalize(text: &str) -> String {
    let cleaned: String = text.chars().map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { ' ' }).collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn transcribes_the_test_recording() {
    let Some((model, samples)) = setup() else { return };
    let transcript = model.transcribe(&samples, &DecodeOptions::default());
    eprintln!("transcript: {}", transcript.text);
    assert!(!transcript.text.is_empty(), "the transcript is empty");
    assert!(transcript.windows.iter().all(|w| w.decoded.finished), "decoding ran into the length limit");
    if let Ok(expected) = std::env::var("WHISPER_TEST_TEXT") {
        let (heard, wanted) = (normalize(&transcript.text), normalize(&expected));
        assert!(heard.contains(&wanted), "expected to hear {wanted:?} in {heard:?}");
    }
}

#[test]
fn cached_and_uncached_decoding_choose_the_same_tokens() {
    let Some((model, samples)) = setup() else { return };
    // One window is enough, and the uncached decoder is slow.
    let cached = model.transcribe_window(&samples, &DecodeOptions { use_cache: true, trace_steps: 3 });
    let uncached = model.transcribe_window(&samples, &DecodeOptions { use_cache: false, trace_steps: 3 });
    assert_eq!(cached.decoded.tokens, uncached.decoded.tokens);
    // The probabilities behind the choices agree too, up to rounding.
    for (a, b) in cached.decoded.trace.iter().zip(&uncached.decoded.trace) {
        assert_eq!(a.top[0].0, b.top[0].0);
        assert!((a.top[0].1 - b.top[0].1).abs() < 1e-3, "{} vs {}", a.top[0].1, b.top[0].1);
    }
}

#[test]
fn every_tensor_in_the_checkpoint_is_used() {
    let dir = model_dir();
    let path = dir.join("model.safetensors");
    if !path.exists() {
        eprintln!("skipping: no Whisper weights in {}", dir.display());
        return;
    }
    use speech::whisper::config::ModelConfig;
    use speech::whisper::safetensors::SafeTensors;
    let config = ModelConfig::read(&dir.join("config.json")).unwrap();
    let mut weights = SafeTensors::read(&path).unwrap();
    assert_eq!(weights.infos.len(), 167);
    assert_eq!(weights.infos.iter().map(|i| i.element_count()).sum::<usize>(), 37_760_256);
    speech::whisper::encoder::Encoder::load(&mut weights, &config).unwrap();
    speech::whisper::decoder::Decoder::load(&mut weights, &config).unwrap();
    // If the model loaded but left tensors behind, we forgot a layer.
    let leftover: Vec<&str> = weights.names().collect();
    assert!(leftover.is_empty(), "unused tensors: {leftover:?}");
}
