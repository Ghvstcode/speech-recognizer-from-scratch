//! The audio encoder: a spectrogram in, one feature row per 20 ms out.
//!
//! ```text
//! log-mel spectrogram            3000 x 80     one row per 10 ms
//!   conv1 (kernel 3)  + GELU     3000 x 384
//!   conv2 (kernel 3, stride 2) + GELU
//!                                1500 x 384    one row per 20 ms
//!   + positional embedding
//!   4 x transformer block        1500 x 384
//!   layer norm                   1500 x 384    "what was said, where"
//! ```
//!
//! The two convolutions look at a few neighbouring frames and halve the
//! frame rate; chapter 7's classifier started the same way. The transformer
//! blocks then let every 20 ms slice consult every other slice of the 30 s
//! window, which is how the model can use context from seconds away.

use super::config::ModelConfig;
use super::layers::{gelu_in_place, Conv1d, LayerNorm, Mlp, MultiHeadAttention};
use super::safetensors::SafeTensors;
use crate::mat::Mat;
use std::io;

/// One encoder transformer block: self-attention, then an MLP.
///
/// Both halves have the same shape: `x = x + f(norm(x))`. The `x +` is a
/// *residual connection*: each sub-layer only computes a correction that is
/// added to its input, so information (and, during training, gradients) can
/// flow straight through a deep stack. Normalising *before* the sub-layer
/// rather than after it is called the "pre-norm" arrangement. The original
/// transformer paper did it the other way round; Whisper, like most models
/// since, uses pre-norm, and the weights only make sense in that order.
pub struct EncoderBlock {
    /// Normalisation before self-attention (`self_attn_layer_norm`).
    pub attention_norm: LayerNorm,
    /// Each audio position gathers context from all the others (`self_attn`).
    pub attention: MultiHeadAttention,
    /// Normalisation before the MLP (`final_layer_norm`).
    pub mlp_norm: LayerNorm,
    /// The per-position network (`fc1`, `fc2`).
    pub mlp: Mlp,
}

impl EncoderBlock {
    /// Load the block stored under `prefix`, e.g. `model.encoder.layers.0`.
    pub fn load(weights: &mut SafeTensors, prefix: &str, n_heads: usize) -> io::Result<EncoderBlock> {
        Ok(EncoderBlock {
            attention_norm: LayerNorm::load(weights, &format!("{prefix}.self_attn_layer_norm"))?,
            attention: MultiHeadAttention::load(weights, &format!("{prefix}.self_attn"), n_heads)?,
            mlp_norm: LayerNorm::load(weights, &format!("{prefix}.final_layer_norm"))?,
            mlp: Mlp::load(weights, prefix)?,
        })
    }

    /// Apply the block in place.
    pub fn forward(&self, x: &mut Mat) {
        let normed = self.attention_norm.forward(x);
        x.add_assign(&self.attention.forward(&normed, &normed, false));
        let normed = self.mlp_norm.forward(x);
        x.add_assign(&self.mlp.forward(&normed));
    }
}

/// The whole audio encoder.
pub struct Encoder {
    /// First convolution: 80 mel bands to `d_model` channels, same length.
    pub conv1: Conv1d,
    /// Second convolution: stride 2, halving 3000 frames to 1500.
    pub conv2: Conv1d,
    /// One row per output position (`embed_positions`), added to the
    /// features. Attention itself is blind to order (shuffle the rows and
    /// each row's weighted average is unchanged), so position has to be
    /// written into the features themselves. In Whisper's encoder these rows
    /// are not learned: they are a fixed pattern of sines and cosines of
    /// different wavelengths, which the checkpoint stores for convenience.
    pub positions: Mat,
    /// The transformer blocks.
    pub blocks: Vec<EncoderBlock>,
    /// Normalisation after the last block (`layer_norm`).
    pub final_norm: LayerNorm,
}

impl Encoder {
    /// Load every tensor under `model.encoder.`.
    pub fn load(weights: &mut SafeTensors, config: &ModelConfig) -> io::Result<Encoder> {
        let blocks = (0..config.encoder_layers)
            .map(|i| EncoderBlock::load(weights, &format!("model.encoder.layers.{i}"), config.encoder_heads))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Encoder {
            conv1: Conv1d::load(weights, "model.encoder.conv1", 1, 1)?,
            conv2: Conv1d::load(weights, "model.encoder.conv2", 2, 1)?,
            positions: weights.take_mat("model.encoder.embed_positions.weight")?,
            blocks,
            final_norm: LayerNorm::load(weights, "model.encoder.layer_norm")?,
        })
    }

    /// The convolutional stem plus positions: `frames x n_mels` in,
    /// `frames / 2 x d_model` out.
    pub fn stem(&self, mel: &Mat) -> Mat {
        let mut x = self.conv1.forward(mel);
        gelu_in_place(&mut x);
        let mut x = self.conv2.forward(&x);
        gelu_in_place(&mut x);
        assert_eq!(
            (x.rows, x.cols),
            (self.positions.rows, self.positions.cols),
            "the encoder needs exactly {} spectrogram frames",
            2 * self.positions.rows
        );
        x.add_assign(&self.positions);
        x
    }

    /// Encode one 30 s spectrogram (`3000 x 80`) into audio features (`1500 x d_model`).
    pub fn forward(&self, mel: &Mat) -> Mat {
        let mut x = self.stem(mel);
        for block in &self.blocks {
            block.forward(&mut x);
        }
        self.final_norm.forward(&x)
    }
}
