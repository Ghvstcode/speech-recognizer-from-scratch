//! The whole recognizer: load the model, then audio in, text out.
//!
//! ```text
//! samples --(mel.rs)--> spectrogram --(encoder.rs)--> audio features
//!                                                          |
//!         prompt tokens --> decoder --> next token --+     |
//!               ^              ^---------------------|-----+
//!               +------ append, repeat --------------+
//!                                  |
//!                        tokens --(tokenizer.rs)--> text
//! ```
//!
//! The loop on the second line is *greedy decoding*: ask the decoder to
//! score every possible next token, take the single best one, append it,
//! ask again, and stop when the best token is `<|endoftext|>`. Chapter 10
//! used the same strategy for our CTC model, and met its limits; here the
//! model is strong enough that greedy is usually all it needs.

use super::config::{GenerationConfig, ModelConfig};
use super::decoder::Decoder;
use super::encoder::Encoder;
use super::layers::softmax_in_place;
use super::mel::{MelFrontEnd, CHUNK_SAMPLES};
use super::safetensors::SafeTensors;
use super::tokenizer::Tokenizer;
use crate::mat::Mat;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

/// A loaded Whisper model, ready to transcribe.
pub struct Whisper {
    /// The architecture's sizes.
    pub config: ModelConfig,
    /// The decoding rules.
    pub rules: GenerationConfig,
    /// Audio to log-mel spectrogram.
    pub front_end: MelFrontEnd,
    /// Spectrogram to audio features.
    pub encoder: Encoder,
    /// Audio features and tokens so far to next-token scores.
    pub decoder: Decoder,
    /// Token ids to text.
    pub tokenizer: Tokenizer,
}

/// Knobs for decoding.
#[derive(Clone, Debug)]
pub struct DecodeOptions {
    /// Use the key/value cache (`Decoder::step`). Turning it off runs the
    /// full `Decoder::forward` for every token: same answer, much slower,
    /// useful only to see what the cache buys.
    pub use_cache: bool,
    /// Record the top candidates for this many decoding steps (0 for none).
    pub trace_steps: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions { use_cache: true, trace_steps: 0 }
    }
}

/// What the decoder was thinking at one step: its favourite next tokens.
#[derive(Clone, Debug)]
pub struct StepTrace {
    /// Position in the token sequence that this step filled.
    pub position: usize,
    /// The five most probable tokens with their probabilities, best first.
    /// The first of these is the token greedy decoding chose.
    pub top: Vec<(usize, f32)>,
}

/// The result of decoding one window.
#[derive(Clone, Debug)]
pub struct Decoded {
    /// The generated tokens: without the prompt and without `<|endoftext|>`.
    pub tokens: Vec<usize>,
    /// Per-step candidates, if `DecodeOptions::trace_steps` asked for any.
    pub trace: Vec<StepTrace>,
    /// False if decoding stopped because the sequence hit its maximum
    /// length rather than because the model chose to end it.
    pub finished: bool,
}

/// How long each stage took.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// Computing the log-mel spectrogram.
    pub mel: Duration,
    /// Running the encoder.
    pub encoder: Duration,
    /// Running the decoding loop.
    pub decoder: Duration,
    /// Decoder steps taken (each produces one token, counting the final
    /// `<|endoftext|>`), for a tokens-per-second figure.
    pub decoder_steps: usize,
}

impl Timings {
    fn add(&mut self, other: &Timings) {
        self.mel += other.mel;
        self.encoder += other.encoder;
        self.decoder += other.decoder;
        self.decoder_steps += other.decoder_steps;
    }
}

/// One 30 s window's worth of results.
#[derive(Clone, Debug)]
pub struct WindowResult {
    /// The text of this window, trimmed.
    pub text: String,
    /// The tokens and trace behind the text.
    pub decoded: Decoded,
    /// Stage timings for this window.
    pub timings: Timings,
}

/// A complete transcript.
#[derive(Clone, Debug)]
pub struct Transcript {
    /// The windows' texts joined with spaces.
    pub text: String,
    /// The individual windows, in order.
    pub windows: Vec<WindowResult>,
    /// Stage timings summed over all windows.
    pub timings: Timings,
}

/// Index of the largest element.
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// Make the listed tokens impossible to choose by giving them a score of
/// negative infinity (probability zero after softmax).
pub fn suppress(logits: &mut [f32], tokens: &[usize]) {
    for &token in tokens {
        if let Some(logit) = logits.get_mut(token) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// The `k` most probable tokens under `softmax(logits)`, best first.
pub fn top_candidates(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut probabilities = logits.to_vec();
    softmax_in_place(&mut probabilities);
    let mut order: Vec<usize> = (0..probabilities.len()).collect();
    order.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]));
    order.into_iter().take(k).map(|token| (token, probabilities[token])).collect()
}

impl Whisper {
    /// Load a model from a directory holding `model.safetensors`,
    /// `config.json`, `generation_config.json`, `vocab.json` and (optionally)
    /// `added_tokens.json`, as published in the Hugging Face repository
    /// `openai/whisper-tiny.en`.
    pub fn load(dir: &Path) -> io::Result<Whisper> {
        let config = ModelConfig::read(&dir.join("config.json"))?;
        let rules = GenerationConfig::read(&dir.join("generation_config.json"))?;
        let mut weights = SafeTensors::read(&dir.join("model.safetensors"))?;
        let encoder = Encoder::load(&mut weights, &config)?;
        let decoder = Decoder::load(&mut weights, &config)?;
        let tokenizer = Tokenizer::load(dir, rules.end_token)?;
        let model = Whisper { front_end: MelFrontEnd::new(config.n_mels), config, rules, encoder, decoder, tokenizer };
        model.check_consistency()?;
        Ok(model)
    }

    /// Cross-check the configuration against the weights that were loaded.
    /// A mismatch here means the files do not belong together, and it is far
    /// better to say so now than to panic (or mumble nonsense) later.
    fn check_consistency(&self) -> io::Result<()> {
        let config = &self.config;
        let checks = [
            ("d_model vs token embedding width", config.d_model, self.decoder.width()),
            ("vocab_size vs token embedding rows", config.vocab_size, self.decoder.token_embedding.rows),
            ("max_target_positions vs decoder positions", config.text_positions, self.decoder.positions.rows),
            ("max_source_positions vs encoder positions", config.audio_positions, self.encoder.positions.rows),
            ("num_mel_bins vs conv1 input channels", config.n_mels, self.encoder.conv1.in_channels),
            ("d_model vs encoder width", config.d_model, self.encoder.positions.cols),
        ];
        for (what, expected, found) in checks {
            if expected != found {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("config and weights disagree on {what}: {expected} vs {found}")));
            }
        }
        if self.rules.max_length > self.decoder.max_tokens() || self.rules.prompt.len() >= self.rules.max_length {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "generation config: max_length does not fit the decoder"));
        }
        Ok(())
    }

    /// Greedy decoding of one window's audio features.
    pub fn greedy_decode(&self, audio_features: &Mat, options: &DecodeOptions) -> Decoded {
        let rules = &self.rules;
        let mut tokens = rules.prompt.clone();
        let mut cache = options.use_cache.then(|| self.decoder.start(audio_features));
        let mut trace = Vec::new();
        let mut finished = false;

        while tokens.len() < rules.max_length {
            // Scores for the token that would come after `tokens`.
            let mut logits = match &mut cache {
                Some(cache) => {
                    // Feed whatever the cache has not seen: the whole prompt
                    // the first time round, one new token ever after.
                    let mut logits = Vec::new();
                    while cache.tokens_seen < tokens.len() {
                        logits = self.decoder.step(tokens[cache.tokens_seen], cache);
                    }
                    logits
                }
                None => {
                    let hidden = self.decoder.forward(&tokens, audio_features);
                    self.decoder.logits(hidden.row(hidden.rows - 1))
                }
            };

            suppress(&mut logits, &rules.suppress);
            if tokens.len() == rules.prompt.len() {
                suppress(&mut logits, &rules.begin_suppress);
            }
            if trace.len() < options.trace_steps {
                trace.push(StepTrace { position: tokens.len(), top: top_candidates(&logits, 5) });
            }

            let next = argmax(&logits);
            if next == rules.end_token {
                finished = true;
                break;
            }
            tokens.push(next);
        }
        Decoded { tokens: tokens.split_off(rules.prompt.len()), trace, finished }
    }

    /// Transcribe up to 30 s of 16 kHz audio (longer input is cut, shorter
    /// input is padded with silence).
    pub fn transcribe_window(&self, samples: &[f32], options: &DecodeOptions) -> WindowResult {
        let started = Instant::now();
        let mel = self.front_end.log_mel(samples);
        let mel_done = Instant::now();
        let audio_features = self.encoder.forward(&mel);
        let encoder_done = Instant::now();
        let decoded = self.greedy_decode(&audio_features, options);
        let decoder_done = Instant::now();

        let timings = Timings {
            mel: mel_done - started,
            encoder: encoder_done - mel_done,
            decoder: decoder_done - encoder_done,
            decoder_steps: decoded.tokens.len() + usize::from(decoded.finished),
        };
        WindowResult { text: self.tokenizer.decode(&decoded.tokens).trim().to_string(), decoded, timings }
    }

    /// Transcribe 16 kHz mono audio of any length.
    ///
    /// The model sees 30 s at a time, so longer audio is cut into
    /// consecutive 30 s windows that are transcribed independently and
    /// joined with spaces.
    ///
    /// **Limitation.** The cuts fall wherever 30 s happen to end, possibly
    /// in the middle of a word, and each window starts without knowing what
    /// was said before. A word straddling a cut may come out wrong, twice,
    /// or not at all. The reference implementation avoids this by asking the
    /// model for timestamps and starting the next window where the last
    /// complete sentence ended, and by feeding the previous text in as a
    /// prompt. We do neither.
    pub fn transcribe(&self, samples: &[f32], options: &DecodeOptions) -> Transcript {
        let mut windows = Vec::new();
        let mut timings = Timings::default();
        for window in samples.chunks(CHUNK_SAMPLES) {
            let result = self.transcribe_window(window, options);
            timings.add(&result.timings);
            windows.push(result);
        }
        let texts: Vec<&str> = windows.iter().map(|w| w.text.as_str()).filter(|t| !t.is_empty()).collect();
        Transcript { text: texts.join(" "), windows, timings }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_the_first_largest() {
        assert_eq!(argmax(&[0.1, 3.0, -2.0, 3.0]), 1);
        assert_eq!(argmax(&[f32::NEG_INFINITY, -5.0]), 1);
    }

    #[test]
    fn suppressed_tokens_cannot_win() {
        let mut logits = vec![1.0, 9.0, 3.0];
        suppress(&mut logits, &[1, 99]); // out-of-range ids are ignored
        assert_eq!(argmax(&logits), 2);
        assert_eq!(logits[1], f32::NEG_INFINITY);
    }

    #[test]
    fn top_candidates_are_sorted_probabilities() {
        // exp of these is 1, 2, 5, and a suppressed 0: probabilities /8.
        let logits = [0.0, 2f32.ln(), 5f32.ln(), f32::NEG_INFINITY];
        let top = top_candidates(&logits, 2);
        assert_eq!(top[0].0, 2);
        assert!((top[0].1 - 0.625).abs() < 1e-6);
        assert_eq!(top[1].0, 1);
        assert!((top[1].1 - 0.25).abs() < 1e-6);
    }
}
