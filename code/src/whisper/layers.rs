//! The building blocks of a transformer, forward pass only.
//!
//! There are five, and the reader has met relatives of most of them:
//!
//! - `Linear`: `y = x W^T + b`, the layer from chapter 6.
//! - `Conv1d`: the convolution over time from chapter 7.
//! - `gelu`: an activation function, a smoother cousin of ReLU.
//! - `LayerNorm`: rescale each row to mean 0 and variance 1.
//! - `MultiHeadAttention`: the new idea. Every row of a sequence looks at
//!   every row of another sequence and takes a weighted average of them.
//!
//! All of them map a `Mat` with one row per position (audio frame or token)
//! to another `Mat` with one row per position. Nothing here learns: there is
//! no backward pass, because someone else already did the training.

use super::parallel::for_each_row_par;
use super::safetensors::SafeTensors;
use crate::mat::{dot, Mat};
use std::io;

/// The error function `erf(x)`, which the standard library does not provide.
///
/// This is formula 7.1.26 from Abramowitz and Stegun's *Handbook of
/// Mathematical Functions*: a fifth-degree polynomial in `1 / (1 + p x)`
/// times `exp(-x^2)`. Its error is below `1.5e-7` everywhere, which is as
/// much precision as an `f32` holds anyway.
pub fn erf(x: f32) -> f32 {
    const P: f32 = 0.327_591_1;
    const A: [f32; 5] = [0.254_829_6, -0.284_496_74, 1.421_413_7, -1.453_152, 1.061_405_4];
    let magnitude = x.abs();
    let t = 1.0 / (1.0 + P * magnitude);
    let polynomial = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
    let value = 1.0 - polynomial * (-magnitude * magnitude).exp();
    value.copysign(x) // erf is odd: erf(-x) = -erf(x)
}

/// The Gaussian Error Linear Unit: `x * Phi(x)`, where `Phi(x)` is the
/// probability that a standard normal variable is below `x`.
///
/// Like ReLU it passes large positive inputs through and blocks large
/// negative ones, but it bends smoothly in between instead of having a kink.
///
/// There are two GELUs in the wild: this exact one, and a `tanh`-based
/// approximation that older models (GPT-2, BERT) were trained with. Whisper
/// uses PyTorch's default `nn.GELU()`, which is the exact, erf-based form,
/// so that is what we must compute. (The two differ by up to about 5e-4.
/// Small, but there is no reason to feed the model a function it was not
/// trained with.)
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Apply `gelu` to every element.
pub fn gelu_in_place(x: &mut Mat) {
    if x.cols == 0 {
        return;
    }
    for_each_row_par(&mut x.data, x.cols, |_, row| {
        for value in row.iter_mut() {
            *value = gelu(*value);
        }
    });
}

/// Turn a row of scores into probabilities: exponentiate and normalise.
///
/// Subtracting the largest score first changes nothing mathematically (it
/// cancels in the ratio) but keeps `exp` from overflowing. A score of
/// negative infinity gets probability exactly 0.
pub fn softmax_in_place(scores: &mut [f32]) {
    let largest = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for score in scores.iter_mut() {
        *score = (*score - largest).exp();
        total += *score;
    }
    for score in scores.iter_mut() {
        *score /= total;
    }
}

/// A fully connected layer: `y = x W^T + b`.
pub struct Linear {
    /// The weights **transposed**, `inputs x outputs`.
    ///
    /// Checkpoints store `W` as `outputs x inputs` (row `o` holds the weights
    /// of output `o`). We flip it once at load time so that the forward pass
    /// is the plain product `x * weight_t`, which `Mat::matmul_par` computes
    /// with its cache-friendly inner loop.
    pub weight_t: Mat,
    /// One bias per output, or `None` for a layer trained without bias.
    pub bias: Option<Vec<f32>>,
}

impl Linear {
    /// Build from a weight matrix in checkpoint layout (`outputs x inputs`).
    pub fn new(weight: Mat, bias: Option<Vec<f32>>) -> Linear {
        if let Some(bias) = &bias {
            assert_eq!(bias.len(), weight.rows, "one bias per output");
        }
        Linear { weight_t: weight.transpose(), bias }
    }

    /// Load `{prefix}.weight` and, if the file has one, `{prefix}.bias`.
    pub fn load(weights: &mut SafeTensors, prefix: &str) -> io::Result<Linear> {
        let weight = weights.take_mat(&format!("{prefix}.weight"))?;
        let bias_name = format!("{prefix}.bias");
        let bias = if weights.contains(&bias_name) { Some(weights.take_vec(&bias_name)?) } else { None };
        Ok(Linear::new(weight, bias))
    }

    /// Number of input features.
    pub fn inputs(&self) -> usize {
        self.weight_t.rows
    }

    /// Number of output features.
    pub fn outputs(&self) -> usize {
        self.weight_t.cols
    }

    /// Apply the layer to every row of `x`.
    pub fn forward(&self, x: &Mat) -> Mat {
        let mut y = x.matmul_par(&self.weight_t);
        if let Some(bias) = &self.bias {
            for row in y.data.chunks_exact_mut(bias.len()) {
                for (value, b) in row.iter_mut().zip(bias) {
                    *value += b;
                }
            }
        }
        y
    }
}

/// A convolution over time: every output row is a linear function of a few
/// neighbouring input rows.
///
/// That sentence is also the implementation. For each output position we
/// copy the `kernel` input rows it can see side by side into one long row
/// (a "patch"), and then a single `Linear` layer maps patches to outputs.
/// Unrolling a convolution into a matrix product like this is a standard
/// trick (it goes by the name *im2col*), and it means the heavy lifting is
/// again done by `matmul_par`.
pub struct Conv1d {
    /// Maps a patch (`kernel * in_channels` values) to `out_channels` values.
    pub patch_to_output: Linear,
    /// How many consecutive input rows each output looks at.
    pub kernel: usize,
    /// How far the window moves between outputs. Stride 2 halves the length.
    pub stride: usize,
    /// Rows of zeros imagined before the first and after the last input row.
    pub padding: usize,
    /// Width of each input row.
    pub in_channels: usize,
}

impl Conv1d {
    /// Build from PyTorch's layout: `weight` is `[out_channels, in_channels, kernel]`.
    ///
    /// Our patches are laid out time-major: all channels of the first input
    /// row, then all channels of the second, and so on. PyTorch's weight has
    /// the kernel index *last*, so we reorder each output's weights from
    /// `[channel][tap]` to `[tap][channel]` to line up with the patches.
    /// Getting this order wrong is one of the classic ways to produce a
    /// model that runs happily and outputs nonsense.
    pub fn new(shape: &[usize], weight: &[f32], bias: Vec<f32>, stride: usize, padding: usize) -> Conv1d {
        let [out_channels, in_channels, kernel] = shape[..] else {
            panic!("conv weight must be [out, in, kernel], found {shape:?}");
        };
        assert_eq!(weight.len(), out_channels * in_channels * kernel, "conv weight does not match its shape");
        let patch_weight = Mat::from_fn(out_channels, kernel * in_channels, |out, patch_index| {
            let (tap, channel) = (patch_index / in_channels, patch_index % in_channels);
            weight[(out * in_channels + channel) * kernel + tap]
        });
        Conv1d { patch_to_output: Linear::new(patch_weight, Some(bias)), kernel, stride, padding, in_channels }
    }

    /// Load `{prefix}.weight` and `{prefix}.bias`.
    pub fn load(weights: &mut SafeTensors, prefix: &str, stride: usize, padding: usize) -> io::Result<Conv1d> {
        let weight = weights.take(&format!("{prefix}.weight"))?;
        let bias = weights.take_vec(&format!("{prefix}.bias"))?;
        if weight.shape.len() != 3 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{prefix}.weight: expected 3 axes, found {:?}", weight.shape)));
        }
        Ok(Conv1d::new(&weight.shape, &weight.data, bias, stride, padding))
    }

    /// Number of output rows for `input_len` input rows.
    pub fn output_len(&self, input_len: usize) -> usize {
        (input_len + 2 * self.padding - self.kernel) / self.stride + 1
    }

    /// Gather the patches: row `t` is input rows `t * stride - padding ..`
    /// laid end to end, with zeros where the window hangs off either end.
    fn patches(&self, x: &Mat) -> Mat {
        let mut patches = Mat::zeros(self.output_len(x.rows), self.kernel * self.in_channels);
        for t in 0..patches.rows {
            for tap in 0..self.kernel {
                // Input row seen by this tap; may fall in the zero padding.
                let Some(source) = (t * self.stride + tap).checked_sub(self.padding) else { continue };
                if source < x.rows {
                    let at = tap * self.in_channels;
                    patches.row_mut(t)[at..at + self.in_channels].copy_from_slice(x.row(source));
                }
            }
        }
        patches
    }

    /// Convolve `x` (`time x in_channels`), giving `output_len x out_channels`.
    pub fn forward(&self, x: &Mat) -> Mat {
        assert_eq!(x.cols, self.in_channels, "conv expects {} channels, got {}", self.in_channels, x.cols);
        self.patch_to_output.forward(&self.patches(x))
    }
}

/// Layer normalisation: shift and scale each row to mean 0 and variance 1,
/// then apply a learned per-column gain and bias.
///
/// Deep stacks of layers are only trainable if the numbers flowing through
/// them stay in a sensible range; this is the transformer's way of ensuring
/// it. Unlike chapter 4's feature normalisation, which took statistics down
/// each *column* over time, this works across each *row*: every position is
/// normalised on its own, so the result does not depend on sequence length.
pub struct LayerNorm {
    /// Learned multiplier per column (`weight` in the checkpoint).
    pub gain: Vec<f32>,
    /// Learned offset per column.
    pub bias: Vec<f32>,
}

impl LayerNorm {
    /// Added to the variance so that a constant row does not divide by zero.
    /// This is PyTorch's default, and the value Whisper was trained with.
    pub const EPSILON: f32 = 1e-5;

    /// Load `{prefix}.weight` and `{prefix}.bias`.
    pub fn load(weights: &mut SafeTensors, prefix: &str) -> io::Result<LayerNorm> {
        Ok(LayerNorm { gain: weights.take_vec(&format!("{prefix}.weight"))?, bias: weights.take_vec(&format!("{prefix}.bias"))? })
    }

    /// Normalise every row of `x`.
    pub fn forward(&self, x: &Mat) -> Mat {
        assert_eq!(x.cols, self.gain.len(), "layer norm width mismatch");
        let mut out = x.clone();
        for row in out.data.chunks_exact_mut(x.cols) {
            let n = row.len() as f32;
            let mean = row.iter().sum::<f32>() / n;
            let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inverse_std = 1.0 / (variance + Self::EPSILON).sqrt();
            for ((value, gain), bias) in row.iter_mut().zip(&self.gain).zip(&self.bias) {
                *value = (*value - mean) * inverse_std * gain + bias;
            }
        }
        out
    }
}

/// Scaled dot-product attention with several heads: the core of the transformer.
///
/// We are given one *query* row per position that wants information, and one
/// *key* row and one *value* row per position that offers information. For a
/// query `q`:
///
/// 1. score every key by its dot product with the query, `q . k_j`, divided
///    by `sqrt(head_dim)` to keep the scores in a range where softmax is not
///    saturated;
/// 2. softmax the scores into weights that sum to 1;
/// 3. output the weighted average of the values, `sum_j weight_j * v_j`.
///
/// "Multi-head" means this is done several times in parallel on separate
/// slices of the columns: with 384 columns and 6 heads, head 0 uses columns
/// 0..64 of the queries, keys and values and writes columns 0..64 of the
/// output, head 1 uses 64..128, and so on. Each head can then attend to
/// something different (one to the current sound, one to the previous word).
///
/// With `causal` set, query `t` may only look at keys `0..=t`. A decoder
/// needs this when it processes a whole token sequence at once: token `t`
/// must not see the tokens that come after it, because when the model is
/// used for real those tokens do not exist yet.
pub fn attend(queries: &Mat, keys: &Mat, values: &Mat, n_heads: usize, causal: bool) -> Mat {
    let width = queries.cols;
    assert!(n_heads > 0 && width.is_multiple_of(n_heads), "{width} columns do not split into {n_heads} heads");
    assert_eq!((keys.cols, values.cols), (width, width), "queries, keys and values must be equally wide");
    assert_eq!(keys.rows, values.rows, "one value per key");
    let head_dim = width / n_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut out = Mat::zeros(queries.rows, width);
    // Each query's output row depends only on that query: one row per task.
    for_each_row_par(&mut out.data, width, |t, out_row| {
        let visible = if causal { (t + 1).min(keys.rows) } else { keys.rows };
        let mut weights = vec![0.0f32; visible];
        for head in 0..n_heads {
            let columns = head * head_dim..(head + 1) * head_dim;
            let query = &queries.row(t)[columns.clone()];
            for (j, weight) in weights.iter_mut().enumerate() {
                *weight = scale * dot(query, &keys.row(j)[columns.clone()]);
            }
            softmax_in_place(&mut weights);
            let out_head = &mut out_row[columns.clone()];
            for (j, &weight) in weights.iter().enumerate() {
                for (o, &v) in out_head.iter_mut().zip(&values.row(j)[columns.clone()]) {
                    *o += weight * v;
                }
            }
        }
    });
    out
}

/// An attention layer: four `Linear` layers around `attend`.
///
/// The queries, keys and values are not the raw inputs but learned linear
/// projections of them, and a final projection mixes the heads' outputs
/// back together.
pub struct MultiHeadAttention {
    /// Projects the rows that ask (`q_proj` in the checkpoint).
    pub query: Linear,
    /// Projects the rows that are searched (`k_proj`). In Whisper this one
    /// layer has **no bias**: adding the same vector to every key would shift
    /// all of a query's scores equally, and softmax ignores such a shift, so
    /// a bias here could never change the output.
    pub key: Linear,
    /// Projects the rows that are averaged (`v_proj`).
    pub value: Linear,
    /// Mixes the concatenated head outputs (`out_proj`).
    pub output: Linear,
    /// Number of heads the columns are split into.
    pub n_heads: usize,
}

impl MultiHeadAttention {
    /// Load the four projections stored under `{prefix}.q_proj` etc.
    pub fn load(weights: &mut SafeTensors, prefix: &str, n_heads: usize) -> io::Result<MultiHeadAttention> {
        Ok(MultiHeadAttention {
            query: Linear::load(weights, &format!("{prefix}.q_proj"))?,
            key: Linear::load(weights, &format!("{prefix}.k_proj"))?,
            value: Linear::load(weights, &format!("{prefix}.v_proj"))?,
            output: Linear::load(weights, &format!("{prefix}.out_proj"))?,
            n_heads,
        })
    }

    /// Let every row of `x` gather information from the rows of `source`.
    ///
    /// *Self-attention* passes the same matrix for both: the sequence looks
    /// at itself. *Cross-attention* passes the decoder's tokens as `x` and
    /// the encoder's audio features as `source`: this is the one place where
    /// the text side of the model listens to the audio.
    pub fn forward(&self, x: &Mat, source: &Mat, causal: bool) -> Mat {
        let queries = self.query.forward(x);
        let keys = self.key.forward(source);
        let values = self.value.forward(source);
        self.output.forward(&attend(&queries, &keys, &values, self.n_heads, causal))
    }
}

/// The small two-layer network inside every transformer block: widen
/// (384 to 1536), apply GELU, narrow back. Attention moves information
/// *between* positions; this is where each position digests what it received.
pub struct Mlp {
    /// The widening layer (`fc1`).
    pub expand: Linear,
    /// The narrowing layer (`fc2`).
    pub contract: Linear,
}

impl Mlp {
    /// Load `{prefix}.fc1` and `{prefix}.fc2`.
    pub fn load(weights: &mut SafeTensors, prefix: &str) -> io::Result<Mlp> {
        Ok(Mlp { expand: Linear::load(weights, &format!("{prefix}.fc1"))?, contract: Linear::load(weights, &format!("{prefix}.fc2"))? })
    }

    /// Apply the network to every row.
    pub fn forward(&self, x: &Mat) -> Mat {
        let mut hidden = self.expand.forward(x);
        gelu_in_place(&mut hidden);
        self.contract.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= tolerance, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn erf_matches_tabulated_values() {
        // From any table of the error function.
        for (x, expected) in [(0.0, 0.0), (0.5, 0.520_499_9), (1.0, 0.842_700_8), (2.0, 0.995_322_3), (-1.0, -0.842_700_8)] {
            assert!((erf(x) - expected).abs() < 3e-7, "erf({x}) = {}", erf(x));
        }
    }

    #[test]
    fn gelu_is_the_exact_form_not_the_tanh_approximation() {
        // x * Phi(x) with Phi from a normal table: Phi(1) = 0.8413447, Phi(2) = 0.9772499.
        assert_close(&[gelu(0.0), gelu(1.0), gelu(-1.0), gelu(2.0)], &[0.0, 0.841_344_7, -0.158_655_3, 1.954_499_7], 1e-6);
        // The tanh approximation gives 2.9963627 at x = 3; the exact value is 2.9959503.
        assert!((gelu(3.0) - 2.995_950_3).abs() < 1e-5);
        // Far out it behaves like ReLU.
        assert!((gelu(10.0) - 10.0).abs() < 1e-6 && gelu(-10.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_by_hand() {
        let mut scores = [0.0, 2f32.ln(), f32::NEG_INFINITY, 5f32.ln()];
        softmax_in_place(&mut scores);
        // exp gives 1, 2, 0, 5: divide by their sum, 8.
        assert_close(&scores, &[0.125, 0.25, 0.0, 0.625], 1e-6);
    }

    #[test]
    fn linear_applies_weights_then_bias() {
        // Two inputs, three outputs; rows of W are per-output weights.
        let layer = Linear::new(Mat::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 2.0, -1.0]), Some(vec![10.0, 20.0, 30.0]));
        assert_eq!((layer.inputs(), layer.outputs()), (2, 3));
        let y = layer.forward(&Mat::from_vec(2, 2, vec![3.0, 4.0, -1.0, 0.5]));
        assert_eq!(y.data, vec![13.0, 24.0, 32.0, 9.0, 20.5, 27.5]);
    }

    #[test]
    fn layer_norm_by_hand() {
        // Row [1, 2, 3, 4]: mean 2.5, variance 1.25, std 1.118034.
        let norm = LayerNorm { gain: vec![1.0, 1.0, 2.0, 1.0], bias: vec![0.0, 0.0, 0.0, 5.0] };
        let y = norm.forward(&Mat::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]));
        let z = [-1.341_641, -0.447_214, 0.447_214, 1.341_641]; // (x - 2.5) / 1.118034
        assert_close(&y.data, &[z[0], z[1], 2.0 * z[2], z[3] + 5.0], 1e-4);
        // A constant row has zero variance; epsilon keeps the result finite: all bias.
        let flat = norm.forward(&Mat::from_vec(1, 4, vec![7.0; 4]));
        assert_close(&flat.data, &[0.0, 0.0, 0.0, 5.0], 1e-6);
    }

    #[test]
    fn conv_slides_a_window_with_padding_and_stride() {
        // One input channel, one output channel, kernel [1, 10, 100], bias 0.5.
        let conv = Conv1d::new(&[1, 1, 3], &[1.0, 10.0, 100.0], vec![0.5], 1, 1);
        let x = Mat::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        // Position t sees (x[t-1], x[t], x[t+1]) with zeros off the ends.
        assert_eq!(conv.forward(&x).data, vec![210.5, 321.5, 432.5, 43.5]);
        // Stride 2 keeps positions 0 and 2.
        let strided = Conv1d::new(&[1, 1, 3], &[1.0, 10.0, 100.0], vec![0.5], 2, 1);
        assert_eq!(strided.forward(&x).data, vec![210.5, 432.5]);
        assert_eq!(strided.output_len(3000), 1500);
    }

    #[test]
    fn conv_reads_pytorch_weight_layout() {
        // Two input channels, two output channels, kernel 2, no padding.
        // PyTorch order is weight[out][in][tap].
        let weight = [
            1.0, 2.0, // out 0, in 0, taps 0 and 1
            3.0, 4.0, // out 0, in 1
            0.0, 1.0, // out 1, in 0
            -1.0, 0.0, // out 1, in 1
        ];
        let conv = Conv1d::new(&[2, 2, 2], &weight, vec![0.0, 0.0], 1, 0);
        // Rows are time steps; columns are channels (a, b).
        let x = Mat::from_vec(3, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        // out0[t] = 1*a[t] + 2*a[t+1] + 3*b[t] + 4*b[t+1];  out1[t] = a[t+1] - b[t]
        assert_eq!(conv.forward(&x).data, vec![1.0 + 4.0 + 30.0 + 80.0, 2.0 - 10.0, 2.0 + 6.0 + 60.0 + 120.0, 3.0 - 20.0]);
    }

    #[test]
    fn attention_with_equal_scores_averages_the_values() {
        // A zero query scores every key 0, so the weights are uniform.
        let queries = Mat::zeros(1, 2);
        let keys = Mat::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let values = Mat::from_vec(3, 2, vec![3.0, 0.0, 6.0, 30.0, 9.0, 60.0]);
        assert_close(&attend(&queries, &keys, &values, 1, false).data, &[6.0, 30.0], 1e-6);
    }

    #[test]
    fn attention_weights_follow_the_dot_products() {
        // One head of width 1, so the scale is 1/sqrt(1) = 1.
        // Query ln 3 against keys 0 and 1 gives scores 0 and ln 3:
        // weights 1/4 and 3/4, output 0.25 * 8 + 0.75 * 16 = 14.
        let queries = Mat::from_vec(1, 1, vec![3f32.ln()]);
        let keys = Mat::from_vec(2, 1, vec![0.0, 1.0]);
        let values = Mat::from_vec(2, 1, vec![8.0, 16.0]);
        assert_close(&attend(&queries, &keys, &values, 1, false).data, &[14.0], 1e-5);
    }

    #[test]
    fn heads_work_on_separate_column_slices_and_scale_by_head_width() {
        // Two heads of width 2 over 4 columns; the scale is 1/sqrt(2).
        // Head 0: query (s, s) with s = sqrt(2)*ln(3)/2 against keys (0,0)
        // and (1,1) gives raw scores 0 and 2s, scaled scores 0 and ln 3:
        // weights 1/4, 3/4. Head 1's query is zero: weights 1/2, 1/2.
        let s = 2f32.sqrt() * 3f32.ln() / 2.0;
        let queries = Mat::from_vec(1, 4, vec![s, s, 0.0, 0.0]);
        let keys = Mat::from_vec(2, 4, vec![0.0, 0.0, 9.0, 9.0, 1.0, 1.0, -9.0, 9.0]);
        let values = Mat::from_vec(2, 4, vec![4.0, 8.0, 100.0, 0.0, 8.0, 16.0, 200.0, 2.0]);
        assert_close(&attend(&queries, &keys, &values, 2, false).data, &[7.0, 14.0, 150.0, 1.0], 1e-4);
    }

    #[test]
    fn causal_attention_cannot_see_the_future() {
        let queries = Mat::zeros(3, 1); // uniform weights over whatever is visible
        let keys = Mat::zeros(3, 1);
        let values = Mat::from_vec(3, 1, vec![3.0, 9.0, 30.0]);
        // Row t averages values 0..=t.
        assert_close(&attend(&queries, &keys, &values, 1, true).data, &[3.0, 6.0, 14.0], 1e-5);
        assert_close(&attend(&queries, &keys, &values, 1, false).data, &[14.0, 14.0, 14.0], 1e-5);
    }

    #[test]
    fn attention_layer_composes_projections() {
        // Identity projections except the output, which doubles: with zero
        // queries the layer returns twice the mean of the source rows.
        let identity = || Linear::new(Mat::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]), None);
        let attention = MultiHeadAttention {
            query: Linear::new(Mat::zeros(2, 2), None),
            key: identity(),
            value: identity(),
            output: Linear::new(Mat::from_vec(2, 2, vec![2.0, 0.0, 0.0, 2.0]), Some(vec![1.0, 0.0])),
            n_heads: 2,
        };
        let x = Mat::from_vec(1, 2, vec![5.0, 5.0]);
        let source = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 6.0]);
        assert_close(&attention.forward(&x, &source, false).data, &[5.0, 8.0], 1e-5);
    }

    #[test]
    fn mlp_is_expand_gelu_contract() {
        let mlp = Mlp {
            expand: Linear::new(Mat::from_vec(2, 1, vec![1.0, -1.0]), None),
            contract: Linear::new(Mat::from_vec(1, 2, vec![1.0, 1.0]), Some(vec![0.5])),
        };
        // x = 1: hidden (1, -1), gelu gives (0.8413447, -0.1586553), sum 0.6826894, plus 0.5.
        assert_close(&mlp.forward(&Mat::from_vec(1, 1, vec![1.0])).data, &[1.182_689_4], 1e-6);
    }
}
