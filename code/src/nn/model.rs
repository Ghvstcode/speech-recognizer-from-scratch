//! Chaining layers into a model.
//!
//! A [`Model`] is a list of [`Layer`]s. The forward pass feeds the input to
//! the first layer, that layer's output to the second, and so on. The
//! backward pass does the same walk in reverse with gradients. That is the
//! chain rule, and it is the whole of backpropagation: each layer only has to
//! know how to turn "the gradient at my output" into "the gradient at my
//! input" (and, on the way, into the gradient of its own weights).
//!
//! Three types keep the bookkeeping straight:
//!
//! * [`Model`]: the weights. Never modified by a forward or backward pass.
//! * [`Cache`]: everything one forward pass computed that the matching
//!   backward pass will need. One per call, so one per thread.
//! * [`Grads`]: one gradient tensor per weight tensor, same shapes, same
//!   order. Backward passes *add* into it, so gradients of several examples
//!   accumulate into a minibatch gradient by themselves.

use super::activation;
use super::conv::Conv1d;
use super::gru::{BiGru, Direction, Gru, GruCache};
use super::linear::Linear;
use crate::mat::Mat;
use crate::rng::Rng;

/// One step of a model.
///
/// A plain enum rather than a trait: there are only a handful of layer types,
/// and a `match` shows all of them at once.
#[derive(Clone, Debug, PartialEq)]
pub enum Layer {
    /// `y = x W + b` on every row.
    Linear(Linear),
    /// `max(0, x)` on every element.
    Relu,
    /// `tanh(x)` on every element.
    Tanh,
    /// `1 / (1 + e^-x)` on every element.
    Sigmoid,
    /// Convolution over time; can shorten the sequence.
    Conv1d(Conv1d),
    /// A GRU reading first frame to last; outputs every hidden state.
    Gru(Gru),
    /// Two GRUs, one per direction; outputs both states side by side.
    BiGru(BiGru),
    /// Average over time: `T x C` becomes `1 x C`. Turns a sequence into one
    /// row that summarises the whole clip, for clip classifiers.
    MeanPool,
}

impl Layer {
    /// A [`Linear`] layer with Xavier initialisation (for output layers and
    /// layers followed by tanh or sigmoid).
    pub fn linear(n_in: usize, n_out: usize, rng: &mut Rng) -> Layer {
        Layer::Linear(Linear::new(n_in, n_out, rng))
    }

    /// A [`Linear`] layer with He initialisation (for layers followed by ReLU).
    pub fn linear_he(n_in: usize, n_out: usize, rng: &mut Rng) -> Layer {
        Layer::Linear(Linear::new_he(n_in, n_out, rng))
    }

    /// A [`Conv1d`] layer.
    pub fn conv1d(c_in: usize, c_out: usize, kernel: usize, stride: usize, padding: usize, rng: &mut Rng) -> Layer {
        Layer::Conv1d(Conv1d::new(c_in, c_out, kernel, stride, padding, rng))
    }

    /// A unidirectional [`Gru`] layer.
    pub fn gru(n_in: usize, hidden: usize, rng: &mut Rng) -> Layer {
        Layer::Gru(Gru::new(n_in, hidden, rng))
    }

    /// A bidirectional GRU layer ([`BiGru`]); its output is `2 * hidden` wide.
    pub fn bigru(n_in: usize, hidden: usize, rng: &mut Rng) -> Layer {
        Layer::BiGru(BiGru::new(n_in, hidden, rng))
    }

    /// A short lowercase name for the layer type.
    pub fn kind(&self) -> &'static str {
        match self {
            Layer::Linear(_) => "linear",
            Layer::Relu => "relu",
            Layer::Tanh => "tanh",
            Layer::Sigmoid => "sigmoid",
            Layer::Conv1d(_) => "conv1d",
            Layer::Gru(_) => "gru",
            Layer::BiGru(_) => "bigru",
            Layer::MeanPool => "meanpool",
        }
    }

    /// The layer type with its sizes, e.g. `Conv1d(40 -> 64, kernel 5, stride 2, padding 2)`.
    pub fn describe(&self) -> String {
        match self {
            Layer::Linear(l) => format!("Linear({} -> {})", l.n_in(), l.n_out()),
            Layer::Relu => "ReLU".to_string(),
            Layer::Tanh => "Tanh".to_string(),
            Layer::Sigmoid => "Sigmoid".to_string(),
            Layer::Conv1d(c) => format!("Conv1d({} -> {}, kernel {}, stride {}, padding {})", c.c_in, c.c_out, c.kernel, c.stride, c.padding),
            Layer::Gru(g) => format!("Gru({} -> {})", g.n_in(), g.hidden()),
            Layer::BiGru(g) => format!("BiGru({} -> 2 x {})", g.n_in(), g.hidden()),
            Layer::MeanPool => "MeanPool".to_string(),
        }
    }

    /// The layer's weight tensors, in a fixed order. Empty for layers with
    /// nothing to learn.
    pub fn params(&self) -> Vec<&Mat> {
        match self {
            Layer::Linear(l) => vec![&l.weight, &l.bias],
            Layer::Conv1d(c) => vec![&c.weight, &c.bias],
            Layer::Gru(g) => g.params(),
            Layer::BiGru(g) => [g.fwd.params(), g.bwd.params()].concat(),
            Layer::Relu | Layer::Tanh | Layer::Sigmoid | Layer::MeanPool => Vec::new(),
        }
    }

    /// Mutable access to the weight tensors, in the same order as [`Layer::params`].
    pub fn params_mut(&mut self) -> Vec<&mut Mat> {
        match self {
            Layer::Linear(l) => vec![&mut l.weight, &mut l.bias],
            Layer::Conv1d(c) => vec![&mut c.weight, &mut c.bias],
            Layer::Gru(g) => g.params_mut(),
            Layer::BiGru(g) => {
                let mut all = g.fwd.params_mut();
                all.extend(g.bwd.params_mut());
                all
            }
            Layer::Relu | Layer::Tanh | Layer::Sigmoid | Layer::MeanPool => Vec::new(),
        }
    }

    /// Names for the weight tensors, in the same order as [`Layer::params`].
    pub fn param_names(&self) -> Vec<&'static str> {
        match self {
            Layer::Linear(_) | Layer::Conv1d(_) => vec!["weight", "bias"],
            Layer::Gru(_) => vec!["w_x", "w_h", "b_x", "b_h"],
            Layer::BiGru(_) => vec!["fwd.w_x", "fwd.w_h", "fwd.b_x", "fwd.b_h", "bwd.w_x", "bwd.w_h", "bwd.b_x", "bwd.b_h"],
            Layer::Relu | Layer::Tanh | Layer::Sigmoid | Layer::MeanPool => Vec::new(),
        }
    }

    /// How many columns the layer expects, or `None` if it accepts any width.
    pub fn in_features(&self) -> Option<usize> {
        match self {
            Layer::Linear(l) => Some(l.n_in()),
            Layer::Conv1d(c) => Some(c.c_in),
            Layer::Gru(g) => Some(g.n_in()),
            Layer::BiGru(g) => Some(g.n_in()),
            Layer::Relu | Layer::Tanh | Layer::Sigmoid | Layer::MeanPool => None,
        }
    }

    /// How many columns the layer produces, or `None` if it keeps the width
    /// of its input.
    pub fn out_features(&self) -> Option<usize> {
        match self {
            Layer::Linear(l) => Some(l.n_out()),
            Layer::Conv1d(c) => Some(c.c_out),
            Layer::Gru(g) => Some(g.hidden()),
            Layer::BiGru(g) => Some(2 * g.hidden()),
            Layer::Relu | Layer::Tanh | Layer::Sigmoid | Layer::MeanPool => None,
        }
    }

    /// How many rows come out when `t` rows go in.
    pub fn out_len(&self, t: usize) -> usize {
        match self {
            Layer::Conv1d(c) => c.out_len(t),
            Layer::MeanPool => 1,
            _ => t,
        }
    }

    /// Compute the layer's output, plus whatever its backward pass will need
    /// beyond its input and output.
    fn forward(&self, x: &Mat) -> (Mat, LayerState) {
        match self {
            Layer::Linear(l) => (l.forward(x), LayerState::Nothing),
            Layer::Relu => (activation::relu_forward(x), LayerState::Nothing),
            Layer::Tanh => (activation::tanh_forward(x), LayerState::Nothing),
            Layer::Sigmoid => (activation::sigmoid_forward(x), LayerState::Nothing),
            Layer::Conv1d(c) => {
                let (y, patches) = c.forward(x);
                (y, LayerState::Conv1d(patches))
            }
            Layer::Gru(g) => {
                let cache = g.forward(x, Direction::Forward);
                (cache.h.clone(), LayerState::Gru(cache))
            }
            Layer::BiGru(g) => {
                let (y, fwd, bwd) = g.forward(x);
                (y, LayerState::BiGru(fwd, bwd))
            }
            Layer::MeanPool => (mean_pool_forward(x), LayerState::Nothing),
        }
    }

    /// Turn the gradient at the layer's output into the gradient at its
    /// input, adding the gradients of its own weights into `grads` on the way.
    /// `x` and `y` are the input and output of the matching forward call.
    fn backward(&self, x: &Mat, y: &Mat, state: &LayerState, d_out: &Mat, grads: &mut [Mat]) -> Mat {
        match (self, state) {
            (Layer::Linear(l), _) => l.backward(x, d_out, grads),
            (Layer::Relu, _) => activation::relu_backward(y, d_out),
            (Layer::Tanh, _) => activation::tanh_backward(y, d_out),
            (Layer::Sigmoid, _) => activation::sigmoid_backward(y, d_out),
            (Layer::Conv1d(c), LayerState::Conv1d(patches)) => c.backward(patches, x.rows, d_out, grads),
            (Layer::Gru(g), LayerState::Gru(cache)) => g.backward(x, cache, d_out, Direction::Forward, grads),
            (Layer::BiGru(g), LayerState::BiGru(fwd, bwd)) => g.backward(x, fwd, bwd, d_out, grads),
            (Layer::MeanPool, _) => mean_pool_backward(x.rows, d_out),
            _ => panic!("the cache does not belong to this model (layer {} got another layer's state)", self.kind()),
        }
    }
}

/// The average of all rows, as a single row.
fn mean_pool_forward(x: &Mat) -> Mat {
    let mut mean = Mat::zeros(1, x.cols);
    super::kernels::add_column_sums(x, &mut mean.data);
    if x.rows > 0 {
        mean.scale(1.0 / x.rows as f32);
    }
    mean
}

/// Every input row contributed `1/T` of itself to the mean, so every input
/// row receives `1/T` of the mean's gradient.
fn mean_pool_backward(t_len: usize, d_out: &Mat) -> Mat {
    assert_eq!(d_out.rows, 1, "MeanPool: the gradient must have exactly one row");
    let share = 1.0 / t_len.max(1) as f32;
    Mat::from_fn(t_len, d_out.cols, |_, c| d_out[(0, c)] * share)
}

/// What a layer remembers from its forward pass, besides its input and output.
#[derive(Clone, Debug)]
enum LayerState {
    Nothing,
    /// The windows the convolution gathered.
    Conv1d(Mat),
    Gru(GruCache),
    BiGru(GruCache, GruCache),
}

/// Everything one forward pass computed that the backward pass needs.
///
/// A cache belongs to the call that made it. Give each thread (each
/// utterance) its own.
#[derive(Clone, Debug)]
pub struct Cache {
    /// `activations[0]` is the model's input and `activations[i + 1]` the
    /// output of layer `i`, so the last one is the model's output.
    pub activations: Vec<Mat>,
    /// Per-layer extras (convolution windows, GRU gate values).
    states: Vec<LayerState>,
}

impl Cache {
    /// The model's output: the logits.
    pub fn output(&self) -> &Mat {
        self.activations.last().expect("a cache always holds at least the input")
    }

    /// Take the model's output out of the cache, discarding the rest.
    pub fn into_output(mut self) -> Mat {
        self.activations.pop().expect("a cache always holds at least the input")
    }
}

/// One gradient tensor for every weight tensor of a model, in the order of
/// [`Model::params`].
#[derive(Clone, Debug, PartialEq)]
pub struct Grads {
    /// `tensors[i]` has the shape of `model.params()[i]` and holds the
    /// gradient of the loss with respect to it.
    pub tensors: Vec<Mat>,
}

impl Grads {
    /// All-zero gradients shaped like the model's weights.
    pub fn zeros_like(model: &Model) -> Grads {
        Grads { tensors: model.params().iter().map(|p| Mat::zeros(p.rows, p.cols)).collect() }
    }

    /// Reset every gradient to zero, ready for the next minibatch.
    pub fn zero(&mut self) {
        for t in &mut self.tensors {
            t.fill(0.0);
        }
    }

    /// `self += other`, tensor by tensor. This is how the gradients computed
    /// by different threads are combined.
    pub fn add_assign(&mut self, other: &Grads) {
        assert_eq!(self.tensors.len(), other.tensors.len(), "Grads: the two sets of gradients come from different models");
        for (a, b) in self.tensors.iter_mut().zip(&other.tensors) {
            a.add_assign(b);
        }
    }

    /// Multiply every gradient by `k` (e.g. `1 / batch_size` to turn a sum
    /// into a mean).
    pub fn scale(&mut self, k: f32) {
        for t in &mut self.tensors {
            t.scale(k);
        }
    }

    /// The length of all gradients taken together as one long vector:
    /// the square root of the sum of every entry squared.
    pub fn global_norm(&self) -> f32 {
        // Summed in f64: millions of small squares lose digits in f32.
        let sum: f64 = self.tensors.iter().flat_map(|t| &t.data).map(|&g| g as f64 * g as f64).sum();
        sum.sqrt() as f32
    }

    /// True if no entry is NaN or infinite.
    pub fn is_finite(&self) -> bool {
        self.tensors.iter().flat_map(|t| &t.data).all(|g| g.is_finite())
    }
}

/// A stack of layers, applied in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// The layers, first to last.
    pub layers: Vec<Layer>,
}

impl Model {
    /// Build a model from its layers.
    ///
    /// Panics if two neighbouring layers disagree about the number of columns
    /// passed between them: better to find out here than in the middle of a
    /// training run.
    pub fn new(layers: Vec<Layer>) -> Model {
        let mut width: Option<usize> = None; // columns leaving the previous layer, once known
        for (i, layer) in layers.iter().enumerate() {
            if let (Some(have), Some(want)) = (width, layer.in_features()) {
                assert_eq!(have, want, "layer {i} ({}) expects {want} columns but the layer before it produces {have}", layer.describe());
            }
            width = layer.out_features().or(width);
        }
        Model { layers }
    }

    /// The forward pass. Returns a [`Cache`] holding the output
    /// (`cache.output()`) and everything `backward` will need.
    pub fn forward(&self, x: &Mat) -> Cache {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        let mut states = Vec::with_capacity(self.layers.len());
        activations.push(x.clone());
        for layer in &self.layers {
            let (y, state) = layer.forward(activations.last().expect("starts non-empty"));
            activations.push(y);
            states.push(state);
        }
        Cache { activations, states }
    }

    /// Just the output, for when no backward pass will follow (inference).
    pub fn predict(&self, x: &Mat) -> Mat {
        self.forward(x).into_output()
    }

    /// The backward pass.
    ///
    /// `d_out` is the gradient of the loss with respect to the model's output
    /// (what a loss function returns alongside the loss). The gradient of
    /// every weight tensor is **added** into `grads`; the gradient with
    /// respect to the model's input is returned (nobody trains on it, but the
    /// gradient checker wants to see it).
    pub fn backward(&self, cache: &Cache, d_out: &Mat, grads: &mut Grads) -> Mat {
        assert_eq!(cache.states.len(), self.layers.len(), "the cache does not belong to this model");
        let output = cache.output();
        assert_eq!((d_out.rows, d_out.cols), (output.rows, output.cols), "backward: the gradient is {}x{} but the model's output is {}x{}", d_out.rows, d_out.cols, output.rows, output.cols);

        let mut d = d_out.clone();
        // `grads.tensors` lists every layer's tensors one after another.
        // Walking the layers backward, we peel each layer's share off the end.
        let mut end = grads.tensors.len();
        for (i, layer) in self.layers.iter().enumerate().rev() {
            let start = end - layer.param_names().len();
            let (x, y) = (&cache.activations[i], &cache.activations[i + 1]);
            d = layer.backward(x, y, &cache.states[i], &d, &mut grads.tensors[start..end]);
            end = start;
        }
        assert_eq!(end, 0, "Grads: these gradients were made for a different model");
        d
    }

    /// Every weight tensor of the model, layer by layer.
    pub fn params(&self) -> Vec<&Mat> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    /// Mutable access to every weight tensor, in the same order as
    /// [`Model::params`]. This is what optimisers use.
    pub fn params_mut(&mut self) -> Vec<&mut Mat> {
        self.layers.iter_mut().flat_map(|l| l.params_mut()).collect()
    }

    /// A name for every weight tensor, in the same order as
    /// [`Model::params`], such as `2.bigru.fwd.w_h` (layer index, layer type,
    /// tensor).
    pub fn param_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for name in layer.param_names() {
                names.push(format!("{i}.{}.{name}", layer.kind()));
            }
        }
        names
    }

    /// Total number of learnable numbers.
    pub fn num_params(&self) -> usize {
        self.params().iter().map(|p| p.data.len()).sum()
    }

    /// How many output rows an input of `t` rows produces. With CTC this is
    /// the number of frames the loss has to fit the target into.
    pub fn out_len(&self, t: usize) -> usize {
        self.layers.iter().fold(t, |t, layer| layer.out_len(t))
    }

    /// A table of the layers and their sizes, for printing.
    pub fn summary(&self) -> String {
        let mut text = String::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let count: usize = layer.params().iter().map(|p| p.data.len()).sum();
            text.push_str(&format!("{i:>3}  {:<52} {count:>9} params\n", layer.describe()));
        }
        text.push_str(&format!("     {:<52} {:>9} params\n", "total", self.num_params()));
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_by_hand() {
        let x = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 6.0]);
        assert_eq!(mean_pool_forward(&x).data, vec![2.0, 4.0]);
        let d = mean_pool_backward(2, &Mat::from_vec(1, 2, vec![1.0, -4.0]));
        assert_eq!(d.data, vec![0.5, -2.0, 0.5, -2.0]);
    }

    #[test]
    fn two_layer_forward_and_backward_by_hand() {
        // x = [1, -2]  ->  Linear(W = [[1, 0], [0, 1]] (identity), b = [0.5, 0.5])  ->  ReLU
        // pre-activation = [1.5, -1.5], output = [1.5, 0].
        let linear = Linear { weight: Mat::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]), bias: Mat::from_vec(1, 2, vec![0.5, 0.5]) };
        let model = Model::new(vec![Layer::Linear(linear), Layer::Relu]);
        let cache = model.forward(&Mat::from_vec(1, 2, vec![1.0, -2.0]));
        assert_eq!(cache.output().data, vec![1.5, 0.0]);

        // Upstream gradient [1, 1]. The ReLU blocks the second entry, so the
        // gradient at the linear layer's output is [1, 0].
        let mut grads = Grads::zeros_like(&model);
        let d_in = model.backward(&cache, &Mat::from_vec(1, 2, vec![1.0, 1.0]), &mut grads);
        assert_eq!(grads.tensors[0].data, vec![1.0, 0.0, -2.0, 0.0]); // x^T [1, 0]
        assert_eq!(grads.tensors[1].data, vec![1.0, 0.0]);
        assert_eq!(d_in.data, vec![1.0, 0.0]);
    }

    #[test]
    fn shapes_names_and_counts() {
        let mut rng = Rng::new(1);
        let model = Model::new(vec![
            Layer::conv1d(40, 64, 5, 2, 2, &mut rng),
            Layer::Relu,
            Layer::bigru(64, 96, &mut rng),
            Layer::linear(192, 28, &mut rng),
        ]);
        // conv: 5*40*64 + 64 = 12864
        // each GRU: 64*288 + 96*288 + 288 + 288 = 46656, two of them = 93312
        // linear: 192*28 + 28 = 5404
        assert_eq!(model.num_params(), 12864 + 93312 + 5404);
        assert_eq!(model.params().len(), 12);
        assert_eq!(model.param_names()[3], "2.bigru.fwd.w_h");
        assert_eq!(model.out_len(100), 50);
        let out = model.predict(&Mat::zeros(100, 40));
        assert_eq!((out.rows, out.cols), (50, 28));
        assert!(model.summary().contains("BiGru(64 -> 2 x 96)"));
    }

    #[test]
    #[should_panic(expected = "expects 10 columns")]
    fn mismatched_layers_are_caught_at_construction() {
        let mut rng = Rng::new(1);
        Model::new(vec![Layer::linear(4, 8, &mut rng), Layer::Tanh, Layer::linear(10, 2, &mut rng)]);
    }

    #[test]
    fn grads_arithmetic() {
        let mut rng = Rng::new(1);
        let model = Model::new(vec![Layer::linear(2, 2, &mut rng)]);
        let mut g = Grads::zeros_like(&model);
        g.tensors[0].data = vec![3.0, 0.0, 0.0, 0.0];
        g.tensors[1].data = vec![0.0, 4.0];
        assert_eq!(g.global_norm(), 5.0);
        let copy = g.clone();
        g.add_assign(&copy);
        g.scale(0.5);
        assert_eq!(g, copy);
        g.zero();
        assert_eq!(g.global_norm(), 0.0);
    }
}
