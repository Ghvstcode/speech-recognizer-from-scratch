//! The text decoder: given the audio features and the tokens so far, score
//! every possible next token.
//!
//! ```text
//! tokens so far                  n               (integers)
//!   token embedding + positional embedding
//!                                n x 384
//!   4 x block: causal self-attention   (look at earlier tokens)
//!              cross-attention         (look at the audio)
//!              MLP
//!   layer norm                   n x 384
//!   dot with every token embedding
//!                                n x 51864       "logits": a score per token
//! ```
//!
//! Row `t` of the output scores candidates for token `t + 1`. Generating a
//! transcript means running this in a loop: score, pick a token, append it,
//! score again.
//!
//! This file has the decoder twice. `forward` is the definition: it takes
//! the whole token sequence and recomputes everything, every time. `step` is
//! what we actually run: it takes *one* new token and reuses the work done
//! for the earlier ones, kept in a `DecoderCache`. They compute the same
//! numbers; the tests check that they do.

use super::config::ModelConfig;
use super::layers::{attend, LayerNorm, Mlp, MultiHeadAttention};
use super::parallel::for_each_row_par;
use super::safetensors::SafeTensors;
use crate::mat::{dot, Mat};
use std::io;

/// One decoder transformer block. An encoder block with one more sub-layer
/// in the middle: cross-attention, where the text consults the audio.
pub struct DecoderBlock {
    /// Normalization before self-attention (`self_attn_layer_norm`).
    pub self_attention_norm: LayerNorm,
    /// Each token looks at itself and the tokens before it (`self_attn`).
    pub self_attention: MultiHeadAttention,
    /// Normalization before cross-attention (`encoder_attn_layer_norm`).
    pub cross_attention_norm: LayerNorm,
    /// Each token looks at all 1500 audio positions (`encoder_attn`).
    pub cross_attention: MultiHeadAttention,
    /// Normalization before the MLP (`final_layer_norm`).
    pub mlp_norm: LayerNorm,
    /// The per-position network (`fc1`, `fc2`).
    pub mlp: Mlp,
}

impl DecoderBlock {
    /// Load the block stored under `prefix`, e.g. `model.decoder.layers.0`.
    pub fn load(weights: &mut SafeTensors, prefix: &str, n_heads: usize) -> io::Result<DecoderBlock> {
        Ok(DecoderBlock {
            self_attention_norm: LayerNorm::load(weights, &format!("{prefix}.self_attn_layer_norm"))?,
            self_attention: MultiHeadAttention::load(weights, &format!("{prefix}.self_attn"), n_heads)?,
            cross_attention_norm: LayerNorm::load(weights, &format!("{prefix}.encoder_attn_layer_norm"))?,
            cross_attention: MultiHeadAttention::load(weights, &format!("{prefix}.encoder_attn"), n_heads)?,
            mlp_norm: LayerNorm::load(weights, &format!("{prefix}.final_layer_norm"))?,
            mlp: Mlp::load(weights, prefix)?,
        })
    }
}

/// What one decoder block remembers between steps.
///
/// Attention needs a key row and a value row for everything it looks at.
/// - For **cross-attention** those come from the audio features, which do
///   not change while we decode: compute them once per 30 s window.
/// - For **self-attention** they come from the tokens. Because of the causal
///   mask, the rows belonging to earlier tokens can never be affected by
///   later ones, so they are final the moment they are computed. Each step
///   only appends one new row.
pub struct BlockCache {
    /// Self-attention keys, one row per token processed so far.
    pub self_keys: Mat,
    /// Self-attention values, one row per token processed so far.
    pub self_values: Mat,
    /// Cross-attention keys, one row per audio position. Fixed.
    pub cross_keys: Mat,
    /// Cross-attention values, one row per audio position. Fixed.
    pub cross_values: Mat,
}

/// The decoder's memory while transcribing one window: a `BlockCache` per
/// block, and how many tokens have been fed in.
pub struct DecoderCache {
    /// One cache per decoder block, in order.
    pub blocks: Vec<BlockCache>,
    /// Number of tokens processed so far; also the next token's position.
    pub tokens_seen: usize,
}

/// Append the rows of `new_rows` below the rows of `matrix`.
fn append_rows(matrix: &mut Mat, new_rows: &Mat) {
    assert_eq!(matrix.cols, new_rows.cols, "appending rows of a different width");
    matrix.data.extend_from_slice(&new_rows.data);
    matrix.rows += new_rows.rows;
}

/// The whole text decoder.
pub struct Decoder {
    /// One row per token in the vocabulary (`embed_tokens`), `vocab x d_model`.
    ///
    /// This matrix is used twice. On the way in, a token id selects its row.
    /// On the way out, the final hidden row is compared (by dot product)
    /// against *every* row to score each token as the continuation. Sharing
    /// one matrix for both jobs is called *weight tying*; it is why the
    /// checkpoint has no separate output layer.
    pub token_embedding: Mat,
    /// One learned row per position in the token sequence (`embed_positions`).
    pub positions: Mat,
    /// The transformer blocks.
    pub blocks: Vec<DecoderBlock>,
    /// Normalization after the last block (`layer_norm`).
    pub final_norm: LayerNorm,
}

impl Decoder {
    /// Load every tensor under `model.decoder.`.
    pub fn load(weights: &mut SafeTensors, config: &ModelConfig) -> io::Result<Decoder> {
        let blocks = (0..config.decoder_layers)
            .map(|i| DecoderBlock::load(weights, &format!("model.decoder.layers.{i}"), config.decoder_heads))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Decoder {
            token_embedding: weights.take_mat("model.decoder.embed_tokens.weight")?,
            positions: weights.take_mat("model.decoder.embed_positions.weight")?,
            blocks,
            final_norm: LayerNorm::load(weights, "model.decoder.layer_norm")?,
        })
    }

    /// Width of the rows flowing through the decoder.
    pub fn width(&self) -> usize {
        self.token_embedding.cols
    }

    /// The most tokens a sequence may hold: one per learned position.
    pub fn max_tokens(&self) -> usize {
        self.positions.rows
    }

    /// The decoder's input row for `token` at `position`: the token's
    /// embedding plus the position's embedding. The first token is at
    /// position 0.
    pub fn embed(&self, token: usize, position: usize) -> Vec<f32> {
        assert!(token < self.token_embedding.rows, "token id {token} is outside the vocabulary");
        assert!(position < self.positions.rows, "position {position} is beyond the model's {} positions", self.positions.rows);
        self.token_embedding.row(token).iter().zip(self.positions.row(position)).map(|(t, p)| t + p).collect()
    }

    /// The textbook decoder: process the whole token sequence from scratch.
    ///
    /// Returns the final hidden rows, `tokens.len() x d_model`. Pass the last
    /// row to `logits` to score the next token.
    pub fn forward(&self, tokens: &[usize], audio_features: &Mat) -> Mat {
        let rows: Vec<Vec<f32>> = tokens.iter().enumerate().map(|(position, &token)| self.embed(token, position)).collect();
        let mut x = Mat::from_rows(&rows);
        for block in &self.blocks {
            // Tokens look at earlier tokens. `causal = true` hides the future.
            let normed = block.self_attention_norm.forward(&x);
            x.add_assign(&block.self_attention.forward(&normed, &normed, true));
            // Tokens look at the audio.
            let normed = block.cross_attention_norm.forward(&x);
            x.add_assign(&block.cross_attention.forward(&normed, audio_features, false));
            // Each token digests what it gathered.
            let normed = block.mlp_norm.forward(&x);
            x.add_assign(&block.mlp.forward(&normed));
        }
        self.final_norm.forward(&x)
    }

    /// Score every vocabulary entry against one final hidden row:
    /// `logits[v] = hidden . token_embedding[v]`.
    ///
    /// These are *logits*, unnormalized scores; softmax turns them into
    /// probabilities. With 51864 tokens this is the most expensive single
    /// operation of a decoding step, so the vocabulary is split across cores.
    pub fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0; self.token_embedding.rows];
        for_each_row_par(&mut logits, 1, |token, logit| logit[0] = dot(hidden, self.token_embedding.row(token)));
        logits
    }

    /// Begin decoding a window: project the audio features into every
    /// block's cross-attention keys and values, once.
    pub fn start(&self, audio_features: &Mat) -> DecoderCache {
        let width = self.width();
        let blocks = self
            .blocks
            .iter()
            .map(|block| BlockCache {
                self_keys: Mat::zeros(0, width),
                self_values: Mat::zeros(0, width),
                cross_keys: block.cross_attention.key.forward(audio_features),
                cross_values: block.cross_attention.value.forward(audio_features),
            })
            .collect();
        DecoderCache { blocks, tokens_seen: 0 }
    }

    /// Feed one more token and return the logits for the token after it.
    ///
    /// This follows `forward` line for line, except that `x` is a single row
    /// (the new token) and the keys and values it attends to come from the
    /// cache. No causal mask is needed: the cache only *contains* the past.
    pub fn step(&self, token: usize, cache: &mut DecoderCache) -> Vec<f32> {
        let mut x = Mat::from_vec(1, self.width(), self.embed(token, cache.tokens_seen));
        for (block, memory) in self.blocks.iter().zip(&mut cache.blocks) {
            let attention = &block.self_attention;
            let normed = block.self_attention_norm.forward(&x);
            append_rows(&mut memory.self_keys, &attention.key.forward(&normed));
            append_rows(&mut memory.self_values, &attention.value.forward(&normed));
            let query = attention.query.forward(&normed);
            let attended = attend(&query, &memory.self_keys, &memory.self_values, attention.n_heads, false);
            x.add_assign(&attention.output.forward(&attended));

            let attention = &block.cross_attention;
            let normed = block.cross_attention_norm.forward(&x);
            let query = attention.query.forward(&normed);
            let attended = attend(&query, &memory.cross_keys, &memory.cross_values, attention.n_heads, false);
            x.add_assign(&attention.output.forward(&attended));

            let normed = block.mlp_norm.forward(&x);
            x.add_assign(&block.mlp.forward(&normed));
        }
        cache.tokens_seen += 1;
        self.logits(self.final_norm.forward(&x).row(0))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::whisper::layers::Linear;

    fn random_mat(rng: &mut Rng, rows: usize, cols: usize, scale: f32) -> Mat {
        Mat::from_fn(rows, cols, |_, _| scale * rng.normal())
    }

    fn random_linear(rng: &mut Rng, inputs: usize, outputs: usize, bias: bool) -> Linear {
        let bias = bias.then(|| (0..outputs).map(|_| 0.1 * rng.normal()).collect());
        Linear::new(random_mat(rng, outputs, inputs, 0.4), bias)
    }

    fn random_norm(rng: &mut Rng, width: usize) -> LayerNorm {
        LayerNorm { gain: (0..width).map(|_| 1.0 + 0.1 * rng.normal()).collect(), bias: (0..width).map(|_| 0.1 * rng.normal()).collect() }
    }

    fn random_attention(rng: &mut Rng, width: usize, n_heads: usize) -> MultiHeadAttention {
        MultiHeadAttention {
            query: random_linear(rng, width, width, true),
            key: random_linear(rng, width, width, false),
            value: random_linear(rng, width, width, true),
            output: random_linear(rng, width, width, true),
            n_heads,
        }
    }

    /// A small decoder with random weights: 2 blocks, width 8, 2 heads,
    /// 11 tokens, 6 positions.
    pub(crate) fn random_decoder(seed: u64) -> Decoder {
        let rng = &mut Rng::new(seed);
        let (width, n_heads) = (8, 2);
        let blocks = (0..2)
            .map(|_| DecoderBlock {
                self_attention_norm: random_norm(rng, width),
                self_attention: random_attention(rng, width, n_heads),
                cross_attention_norm: random_norm(rng, width),
                cross_attention: random_attention(rng, width, n_heads),
                mlp_norm: random_norm(rng, width),
                mlp: Mlp { expand: random_linear(rng, width, 16, true), contract: random_linear(rng, 16, width, true) },
            })
            .collect();
        Decoder {
            token_embedding: random_mat(rng, 11, width, 1.0),
            positions: random_mat(rng, 6, width, 1.0),
            blocks,
            final_norm: random_norm(rng, width),
        }
    }

    #[test]
    fn embedding_adds_token_and_position_rows() {
        let decoder = random_decoder(1);
        let row = decoder.embed(3, 2);
        for (c, value) in row.iter().enumerate() {
            assert_eq!(*value, decoder.token_embedding[(3, c)] + decoder.positions[(2, c)]);
        }
    }

    #[test]
    fn logits_are_dot_products_with_the_embedding_rows() {
        let decoder = random_decoder(2);
        let hidden: Vec<f32> = (0..8).map(|i| i as f32 - 3.0).collect();
        let logits = decoder.logits(&hidden);
        assert_eq!(logits.len(), 11);
        for (token, logit) in logits.iter().enumerate() {
            let by_hand: f32 = (0..8).map(|c| hidden[c] * decoder.token_embedding[(token, c)]).sum();
            assert!((logit - by_hand).abs() < 1e-4);
        }
    }

    #[test]
    fn stepping_with_the_cache_matches_the_full_forward_pass() {
        let decoder = random_decoder(3);
        let audio = random_mat(&mut Rng::new(4), 7, 8, 1.0);
        let tokens = [4, 0, 10, 4, 7];
        let mut cache = decoder.start(&audio);
        for (n, &token) in tokens.iter().enumerate() {
            let stepped = decoder.step(token, &mut cache);
            let hidden = decoder.forward(&tokens[..=n], &audio);
            let full = decoder.logits(hidden.row(n));
            for (a, b) in stepped.iter().zip(&full) {
                assert!((a - b).abs() < 1e-4, "step {n}: cached {a} vs full {b}");
            }
        }
        assert_eq!(cache.tokens_seen, 5);
        assert_eq!(cache.blocks[0].self_keys.rows, 5);
        assert_eq!(cache.blocks[0].cross_keys.rows, 7);
    }

    #[test]
    fn earlier_rows_do_not_depend_on_later_tokens() {
        // The causal mask in action: changing the last token must leave the
        // hidden rows of all earlier positions untouched.
        let decoder = random_decoder(5);
        let audio = random_mat(&mut Rng::new(6), 4, 8, 1.0);
        let a = decoder.forward(&[1, 2, 3, 4], &audio);
        let b = decoder.forward(&[1, 2, 3, 9], &audio);
        assert_eq!(a.data[..3 * 8], b.data[..3 * 8]);
        assert_ne!(a.row(3), b.row(3));
    }
}
