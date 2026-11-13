//! The two small JSON files that describe the model and how to decode with it.
//!
//! `config.json` holds the architecture's sizes (how many layers, how wide).
//! `generation_config.json` holds the decoding rules: which tokens start a
//! transcript, which token ends it, and which tokens are never allowed.
//! Reading them from the files, rather than hardcoding `tiny.en`'s numbers,
//! means the same code describes any Whisper model of this family.

use super::json::Json;
use std::io;
use std::path::Path;

fn bad(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn usize_field(json: &Json, key: &str) -> io::Result<usize> {
    json.get(key).and_then(Json::as_usize).ok_or_else(|| bad(format!("config: missing or non-integer `{key}`")))
}

fn usize_list(json: &Json, key: &str) -> io::Result<Vec<usize>> {
    let Some(value) = json.get(key) else { return Ok(Vec::new()) };
    let items = value.as_array().ok_or_else(|| bad(format!("config: `{key}` is not a list")))?;
    items.iter().map(|item| item.as_usize().ok_or_else(|| bad(format!("config: `{key}` holds a non-integer")))).collect()
}

/// The architecture's sizes, from `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// Mel bands in the input spectrogram (`num_mel_bins`): 80.
    pub n_mels: usize,
    /// Width of every row flowing through the model (`d_model`): 384 for tiny.
    pub d_model: usize,
    /// Transformer blocks in the encoder: 4 for tiny.
    pub encoder_layers: usize,
    /// Attention heads per encoder block: 6 for tiny.
    pub encoder_heads: usize,
    /// Transformer blocks in the decoder: 4 for tiny.
    pub decoder_layers: usize,
    /// Attention heads per decoder block: 6 for tiny.
    pub decoder_heads: usize,
    /// Number of distinct tokens: 51864 for the English-only models.
    pub vocab_size: usize,
    /// Encoder positions (`max_source_positions`): 1500, one per 20 ms.
    pub audio_positions: usize,
    /// Decoder positions (`max_target_positions`): 448 tokens at most.
    pub text_positions: usize,
}

impl ModelConfig {
    /// Extract the sizes from a parsed `config.json`.
    pub fn from_json(json: &Json) -> io::Result<ModelConfig> {
        Ok(ModelConfig {
            n_mels: usize_field(json, "num_mel_bins")?,
            d_model: usize_field(json, "d_model")?,
            encoder_layers: usize_field(json, "encoder_layers")?,
            encoder_heads: usize_field(json, "encoder_attention_heads")?,
            decoder_layers: usize_field(json, "decoder_layers")?,
            decoder_heads: usize_field(json, "decoder_attention_heads")?,
            vocab_size: usize_field(json, "vocab_size")?,
            audio_positions: usize_field(json, "max_source_positions")?,
            text_positions: usize_field(json, "max_target_positions")?,
        })
    }

    /// Read `config.json` from disk.
    pub fn read(path: &Path) -> io::Result<ModelConfig> {
        ModelConfig::from_json(&Json::read_file(path)?)
    }
}

/// The decoding rules, from `generation_config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationConfig {
    /// The tokens every transcript begins with, before the model has said
    /// anything. For `tiny.en`: `<|startoftranscript|>` then
    /// `<|notimestamps|>`. The first comes from `decoder_start_token_id`; the
    /// rest from `forced_decoder_ids`, a list of `[position, token]` pairs.
    /// (The multilingual models force a language and a task token here too.)
    pub prompt: Vec<usize>,
    /// `<|endoftext|>`: when the model picks this, the transcript is done.
    pub end_token: usize,
    /// Tokens the decoder may never produce: mostly punctuation-only tokens
    /// such as brackets and musical notes, with which the model would
    /// otherwise annotate non-speech sounds, plus the special control tokens.
    pub suppress: Vec<usize>,
    /// Tokens forbidden only as the *first* token after the prompt: a lone
    /// space and `<|endoftext|>`, so that the model cannot return an empty
    /// transcript without at least trying.
    pub begin_suppress: Vec<usize>,
    /// Upper limit on the sequence length, prompt included. It equals the
    /// number of decoder positions, so position embeddings can not run out.
    pub max_length: usize,
}

impl GenerationConfig {
    /// Extract the rules from a parsed `generation_config.json` (the same
    /// keys also appear in older `config.json` files).
    pub fn from_json(json: &Json) -> io::Result<GenerationConfig> {
        let mut prompt = vec![usize_field(json, "decoder_start_token_id")?];
        if let Some(forced) = json.get("forced_decoder_ids").and_then(Json::as_array) {
            for pair in forced {
                let (position, token) = match pair.as_array() {
                    Some([position, token]) => (position.as_usize(), token.as_usize()),
                    _ => (None, None),
                };
                // The list is in order; each entry must name the next position.
                match (position, token) {
                    (Some(position), Some(token)) if position == prompt.len() => prompt.push(token),
                    _ => return Err(bad("config: `forced_decoder_ids` is not a run of [position, token] pairs".into())),
                }
            }
        }
        Ok(GenerationConfig {
            prompt,
            end_token: usize_field(json, "eos_token_id")?,
            suppress: usize_list(json, "suppress_tokens")?,
            begin_suppress: usize_list(json, "begin_suppress_tokens")?,
            max_length: usize_field(json, "max_length")?,
        })
    }

    /// Read `generation_config.json` from disk.
    pub fn read(path: &Path) -> io::Result<GenerationConfig> {
        GenerationConfig::from_json(&Json::read_file(path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_model_sizes() {
        let json = Json::parse(
            r#"{"num_mel_bins": 80, "d_model": 384, "encoder_layers": 4, "encoder_attention_heads": 6,
                "decoder_layers": 4, "decoder_attention_heads": 6, "vocab_size": 51864,
                "max_source_positions": 1500, "max_target_positions": 448, "dropout": 0.0}"#,
        )
        .unwrap();
        let config = ModelConfig::from_json(&json).unwrap();
        assert_eq!((config.n_mels, config.d_model, config.vocab_size), (80, 384, 51864));
        assert_eq!((config.audio_positions, config.text_positions), (1500, 448));
        assert!(ModelConfig::from_json(&Json::parse("{}").unwrap()).is_err());
    }

    #[test]
    fn builds_the_prompt_from_start_token_and_forced_ids() {
        let json = Json::parse(
            r#"{"decoder_start_token_id": 50257, "forced_decoder_ids": [[1, 50362]], "eos_token_id": 50256,
                "suppress_tokens": [1, 2, 7], "begin_suppress_tokens": [220, 50256], "max_length": 448}"#,
        )
        .unwrap();
        let rules = GenerationConfig::from_json(&json).unwrap();
        assert_eq!(rules.prompt, vec![50257, 50362]);
        assert_eq!(rules.end_token, 50256);
        assert_eq!(rules.suppress, vec![1, 2, 7]);
        assert_eq!(rules.begin_suppress, vec![220, 50256]);
        assert_eq!(rules.max_length, 448);
    }

    #[test]
    fn rejects_forced_ids_that_skip_a_position() {
        let json = Json::parse(r#"{"decoder_start_token_id": 1, "forced_decoder_ids": [[2, 5]], "eos_token_id": 0, "max_length": 9}"#).unwrap();
        assert!(GenerationConfig::from_json(&json).is_err());
    }
}
