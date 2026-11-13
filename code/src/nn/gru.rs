//! The gated recurrent unit (GRU), and backpropagation through time.
//!
//! A convolution sees a fixed window. A *recurrent* layer instead carries a
//! memory, the **hidden state** `h` (a row of `H` numbers), from one frame to
//! the next: at every frame it reads the new input and its own previous
//! state and writes a new state. In principle the state at frame 80 can
//! depend on something heard at frame 3.
//!
//! The simplest recurrent layer, `h_t = tanh(x_t W + h_{t-1} U)`, has trouble
//! learning that in practice: the state is completely rewritten at every
//! frame, so both the memory and the gradient fade quickly. The GRU (Cho et
//! al., 2014) fixes this with two **gates**, numbers between 0 and 1 computed
//! from the input and the old state, that control how the state is updated.
//!
//! # The equations
//!
//! We use the same variant as PyTorch's `nn.GRU` (the "v3" form from the
//! cuDNN paper), in which the reset gate is applied *after* the old state has
//! been multiplied by its weights. With `x` the current input frame, `h` the
//! previous state and `*` meaning element-by-element multiplication:
//!
//! ```text
//! r  = sigmoid(x W_xr + b_xr  +  h W_hr + b_hr)         reset gate
//! z  = sigmoid(x W_xz + b_xz  +  h W_hz + b_hz)         update gate
//! n  = tanh   (x W_xn + b_xn  +  r * (h W_hn + b_hn))   candidate state
//! h' = (1 - z) * n  +  z * h                            new state
//! ```
//!
//! Read the last line first: each entry of the new state is a blend of the
//! old entry and a freshly computed candidate, and `z` is the blend knob.
//! Where `z` is close to 1 the old value is simply copied forward, untouched,
//! for as many frames as `z` stays there. That copy is the trick: it is also
//! a path along which the gradient travels backward without shrinking. The
//! reset gate `r` decides how much of the old state the candidate is allowed
//! to see: with `r` near 0 the candidate is computed from the input alone.
//!
//! (The original paper's variant computes `n = tanh(x W_xn + (r * h) W_hn)`,
//! resetting *before* the multiplication. The two are equally good in
//! practice; ours has the advantage that all three `h W_h..` products can be
//! done as one, and that weights can be exchanged with PyTorch.)
//!
//! # How the weights are stored
//!
//! The three input weight matrices sit side by side in one `n_in x 3H`
//! matrix `w_x = [W_xr | W_xz | W_xn]`, and likewise `w_h` is `H x 3H`. One
//! product `x w_x` then gives the input's contribution to all three gates at
//! once, as a row of `3H` numbers that we cut into thirds.
//!
//! # Making it fast
//!
//! The input's half of the sums, `x w_x + b_x`, does not depend on the state,
//! so we compute it for *all* frames with a single matrix product before the
//! time loop starts. Inside the loop each frame then costs one vector-matrix
//! product (`h w_h`) and a few passes over `H` numbers, with no allocation.
//!
//! # Backpropagation through time
//!
//! The backward pass walks the frames in the opposite order. At each frame
//! the gradient of the state has two sources: the layer above (which read
//! `h_t` as this frame's output) and the *next* frame (which read `h_t` as
//! its previous state). We add them, push the sum back through the four
//! equations above, one line at a time in reverse, and end up with the
//! gradient of the pre-activation sums (the things inside `sigmoid(...)` and
//! `tanh(...)`) and the gradient to hand to the previous frame. The
//! derivation is spelled out, line by line, in the code of [`Gru::backward`].

use super::activation::sigmoid;
use super::init;
use super::kernels;
use crate::mat::Mat;
use crate::rng::Rng;

/// Which way a GRU walks along the sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// From the first frame to the last: the state at frame `t` summarises
    /// frames `0..=t`.
    Forward,
    /// From the last frame to the first: the state at frame `t` summarises
    /// frames `t..T`.
    Backward,
}

impl Direction {
    /// The frame visited at step number `step` of a `t_len`-frame sequence.
    fn frame(self, step: usize, t_len: usize) -> usize {
        match self {
            Direction::Forward => step,
            Direction::Backward => t_len - 1 - step,
        }
    }
}

/// tanh computed through the sigmoid: `tanh(x) = 2 * sigmoid(2x) - 1`.
///
/// The identity is exact (multiply the top and bottom of
/// `(e^x - e^-x) / (e^x + e^-x)` by `e^x`), and on the machine this was
/// written on the standard library's `tanh` turned out to be over ten times
/// slower than its `exp`. A GRU calls tanh once per state entry per frame,
/// enough for that to show up as a tenth of the training time.
fn fast_tanh(x: f32) -> f32 {
    2.0 * sigmoid(2.0 * x) - 1.0
}

// Positions of the four parameter tensors in `params()` and in the gradient
// slice handed to `backward`.
const W_X: usize = 0;
const W_H: usize = 1;
const B_X: usize = 2;
const B_H: usize = 3;

/// A single-layer GRU. Maps `T x n_in` to the sequence of its hidden states,
/// `T x hidden`.
#[derive(Clone, Debug, PartialEq)]
pub struct Gru {
    /// `n_in x 3H`: input weights for the reset, update and candidate parts,
    /// side by side.
    pub w_x: Mat,
    /// `H x 3H`: the same for the previous state.
    pub w_h: Mat,
    /// `1 x 3H`: biases added on the input side.
    pub b_x: Mat,
    /// `1 x 3H`: biases added on the state side. (Two bias rows look
    /// redundant, and for `r` and `z` they are; for the candidate they are
    /// not, because `b_hn` sits inside the reset gate's multiplication.)
    pub b_h: Mat,
}

/// What one run of a GRU has to remember for its backward pass.
#[derive(Clone, Debug)]
pub struct GruCache {
    /// `T x H`: the hidden state after each frame. This is also the layer's output.
    pub h: Mat,
    /// `T x 3H`: the values of `r`, `z` and `n` at each frame, side by side.
    gates: Mat,
    /// `T x H`: `h_prev W_hn + b_hn` at each frame, the quantity the reset
    /// gate multiplies.
    hn: Mat,
}

impl Gru {
    /// A GRU with `hidden` state entries.
    ///
    /// * Input weights: Xavier, because they feed sigmoids and a tanh.
    /// * Recurrent weights: each of the three `H x H` blocks starts as a
    ///   random *orthogonal* matrix, so that repeated multiplication over a
    ///   hundred frames neither shrinks nor blows up the state. See
    ///   [`init::orthogonal`].
    /// * Biases: zero.
    pub fn new(n_in: usize, hidden: usize, rng: &mut Rng) -> Self {
        let w_x = init::xavier_uniform(n_in, 3 * hidden, n_in, hidden, rng);
        let mut w_h = Mat::zeros(hidden, 3 * hidden);
        for block in 0..3 {
            let q = init::orthogonal(hidden, rng);
            for r in 0..hidden {
                w_h.row_mut(r)[block * hidden..(block + 1) * hidden].copy_from_slice(q.row(r));
            }
        }
        Gru { w_x, w_h, b_x: Mat::zeros(1, 3 * hidden), b_h: Mat::zeros(1, 3 * hidden) }
    }

    /// Number of values in each input frame.
    pub fn n_in(&self) -> usize {
        self.w_x.rows
    }

    /// Size of the hidden state, `H`.
    pub fn hidden(&self) -> usize {
        self.w_h.rows
    }

    /// The parameter tensors, in the order `[w_x, w_h, b_x, b_h]`.
    pub fn params(&self) -> Vec<&Mat> {
        vec![&self.w_x, &self.w_h, &self.b_x, &self.b_h]
    }

    /// Mutable access to the parameter tensors, in the same order.
    pub fn params_mut(&mut self) -> Vec<&mut Mat> {
        vec![&mut self.w_x, &mut self.w_h, &mut self.b_x, &mut self.b_h]
    }

    /// Run the GRU over a `T x n_in` sequence, starting from an all-zero
    /// state. The output is `cache.h` (`T x H`); row `t` is the state after
    /// frame `t` has been read, whichever direction we walk in.
    pub fn forward(&self, x: &Mat, direction: Direction) -> GruCache {
        assert_eq!(x.cols, self.n_in(), "Gru: input has {} columns, the layer expects {}", x.cols, self.n_in());
        let (t_len, h) = (x.rows, self.hidden());

        // The input's contribution to every gate at every frame, in one go.
        // We then overwrite it in place, frame by frame, with the gate values.
        let mut gates = kernels::matmul(x, &self.w_x);
        kernels::add_row_to_each(&mut gates, &self.b_x.data);

        let mut states = Mat::zeros(t_len, h);
        let mut hn = Mat::zeros(t_len, h);
        // Scratch buffers, allocated once and reused at every frame.
        let mut h_prev = vec![0.0; h]; // the state before this frame
        let mut from_state = vec![0.0; 3 * h]; // h_prev W_h + b_h

        for step in 0..t_len {
            let t = direction.frame(step, t_len);

            from_state.copy_from_slice(&self.b_h.data);
            kernels::add_vec_mat(&h_prev, &self.w_h, &mut from_state);
            let (state_r, rest) = from_state.split_at(h);
            let (state_z, state_n) = rest.split_at(h);

            let (gate_r, rest) = gates.row_mut(t).split_at_mut(h);
            let (gate_z, gate_n) = rest.split_at_mut(h);
            let h_new = states.row_mut(t);

            for j in 0..h {
                // On entry gate_r/z/n hold the input's half of each sum.
                let r = sigmoid(gate_r[j] + state_r[j]);
                let z = sigmoid(gate_z[j] + state_z[j]);
                let n = fast_tanh(gate_n[j] + r * state_n[j]);
                h_new[j] = (1.0 - z) * n + z * h_prev[j];
                gate_r[j] = r;
                gate_z[j] = z;
                gate_n[j] = n;
            }
            hn.row_mut(t).copy_from_slice(state_n);
            h_prev.copy_from_slice(h_new);
        }
        GruCache { h: states, gates, hn }
    }

    /// For every frame, the state the GRU held just *before* reading it: the
    /// output of the previous step in walking order, or zeros at the first.
    fn previous_states(&self, states: &Mat, direction: Direction) -> Mat {
        let t_len = states.rows;
        let mut previous = Mat::zeros(t_len, states.cols);
        for step in 1..t_len {
            let (t, t_before) = (direction.frame(step, t_len), direction.frame(step - 1, t_len));
            previous.row_mut(t).copy_from_slice(states.row(t_before));
        }
        previous
    }

    /// Backpropagation through time.
    ///
    /// `x` and `cache` are the input and the result of the matching `forward`
    /// call, `d_out` (`T x H`) is the gradient of the loss with respect to
    /// every output state. Adds the parameter gradients into `grads`
    /// (`[w_x, w_h, b_x, b_h]`) and returns the gradient at the input
    /// (`T x n_in`).
    pub fn backward(&self, x: &Mat, cache: &GruCache, d_out: &Mat, direction: Direction, grads: &mut [Mat]) -> Mat {
        let (t_len, h) = (x.rows, self.hidden());
        assert_eq!((d_out.rows, d_out.cols), (t_len, h), "Gru: gradient shape does not match the output");

        // The state each frame started from: row t holds the output of the
        // step before frame t (zeros for the very first step).
        let h_prev_all = self.previous_states(&cache.h, direction);

        // Gradients of the two sums inside the gates, for every frame:
        // (x W_x + b_x), the input side, and (h_prev W_h + b_h), the state
        // side. We fill these in inside the loop. Each side is a linear layer
        // in disguise, so afterwards the weight gradients fall out of a few
        // whole-sequence products, mirroring the forward pass.
        let mut d_input_side = Mat::zeros(t_len, 3 * h);
        let mut d_state_side = Mat::zeros(t_len, 3 * h);
        // W_h transposed, so that "push a gradient back through W_h" is the
        // same fast vector-matrix product the forward pass uses.
        let w_h_transposed = self.w_h.transpose();

        // Scratch buffers, reused at every frame.
        let mut from_future = vec![0.0; h]; // dLoss/dh_t arriving from the step processed after t
        let mut to_past = vec![0.0; h]; // the same thing, being built for the step before t

        // Walk the steps in the opposite order to the forward pass.
        for step in (0..t_len).rev() {
            let t = direction.frame(step, t_len);
            let h_prev = h_prev_all.row(t);
            let (gate_r, rest) = cache.gates.row(t).split_at(h);
            let (gate_z, gate_n) = rest.split_at(h);
            let state_n = cache.hn.row(t);
            let d_from_above = d_out.row(t);
            let (d_in_r, rest) = d_input_side.row_mut(t).split_at_mut(h);
            let (d_in_z, d_in_n) = rest.split_at_mut(h);
            let (d_st_r, rest) = d_state_side.row_mut(t).split_at_mut(h);
            let (d_st_z, d_st_n) = rest.split_at_mut(h);

            for j in 0..h {
                let (r, z, n) = (gate_r[j], gate_z[j], gate_n[j]);
                // Two readers of h_t: the layer above and the next step.
                let dh = d_from_above[j] + from_future[j];

                // h' = (1 - z) * n + z * h_prev
                let dn = dh * (1.0 - z);
                let dz = dh * (h_prev[j] - n);
                to_past[j] = dh * z; // the direct copy path; more is added below

                // n = tanh(a_n), with a_n = [input side] + r * state_n.
                // tanh'(a) = 1 - tanh(a)^2 = 1 - n^2
                let da_n = dn * (1.0 - n * n);
                let dr = da_n * state_n[j];

                // r = sigmoid(a_r), z = sigmoid(a_z). sigmoid' = s * (1 - s)
                let da_r = dr * r * (1.0 - r);
                let da_z = dz * z * (1.0 - z);

                // The input side and the state side enter a_r and a_z as a
                // plain sum, so both get the same gradient. In a_n the state
                // side is multiplied by r, so its gradient is too.
                d_in_r[j] = da_r;
                d_in_z[j] = da_z;
                d_in_n[j] = da_n;
                d_st_r[j] = da_r;
                d_st_z[j] = da_z;
                d_st_n[j] = da_n * r;
            }

            // The state side read h_prev through W_h, so its gradient flows
            // back to h_prev through W_h transposed: dh_prev += d W_h^T.
            kernels::add_vec_mat(d_state_side.row(t), &w_h_transposed, &mut to_past);

            // What we just built for the past is the future's message for the
            // next iteration.
            std::mem::swap(&mut from_future, &mut to_past);
        }

        // State side: (h_prev W_h + b_h) is a linear layer applied to h_prev.
        kernels::add_t_matmul(&h_prev_all, &d_state_side, &mut grads[W_H]);
        kernels::add_column_sums(&d_state_side, &mut grads[B_H].data);
        // Input side: (x W_x + b_x) is a linear layer applied to every frame.
        kernels::add_t_matmul(x, &d_input_side, &mut grads[W_X]);
        kernels::add_column_sums(&d_input_side, &mut grads[B_X].data);
        kernels::matmul_transposed(&d_input_side, &self.w_x)
    }
}

/// Two GRUs reading the same sequence in opposite directions.
///
/// A forward GRU's state at frame `t` knows only the past. For recognising
/// speech from a recording (rather than live) there is no reason to be so
/// strict: what follows a sound is as informative as what came before it. A
/// bidirectional GRU runs a second, independent GRU from the last frame to
/// the first and glues the two state sequences together, so the output at
/// frame `t` is `[state of the forward GRU | state of the backward GRU]`,
/// `2H` numbers that between them have seen the whole clip.
#[derive(Clone, Debug, PartialEq)]
pub struct BiGru {
    /// The GRU that reads frames first to last. Fills output columns `0..H`.
    pub fwd: Gru,
    /// The GRU that reads frames last to first. Fills output columns `H..2H`.
    pub bwd: Gru,
}

impl BiGru {
    /// Two independently initialised GRUs with `hidden` state entries each;
    /// the output has `2 * hidden` columns.
    pub fn new(n_in: usize, hidden: usize, rng: &mut Rng) -> Self {
        BiGru { fwd: Gru::new(n_in, hidden, rng), bwd: Gru::new(n_in, hidden, rng) }
    }

    /// Number of values in each input frame.
    pub fn n_in(&self) -> usize {
        self.fwd.n_in()
    }

    /// Size of each direction's hidden state. The output is twice as wide.
    pub fn hidden(&self) -> usize {
        self.fwd.hidden()
    }

    /// Run both directions. Returns the `T x 2H` output and the two caches.
    pub fn forward(&self, x: &Mat) -> (Mat, GruCache, GruCache) {
        let fwd = self.fwd.forward(x, Direction::Forward);
        let bwd = self.bwd.forward(x, Direction::Backward);
        let h = self.hidden();
        let mut out = Mat::zeros(x.rows, 2 * h);
        for t in 0..x.rows {
            let (left, right) = out.row_mut(t).split_at_mut(h);
            left.copy_from_slice(fwd.h.row(t));
            right.copy_from_slice(bwd.h.row(t));
        }
        (out, fwd, bwd)
    }

    /// Split the output gradient into its two halves, backpropagate each
    /// through its own GRU, and add the two input gradients (both GRUs read
    /// the same input). `grads` holds the forward GRU's four tensors followed
    /// by the backward GRU's four.
    pub fn backward(&self, x: &Mat, fwd_cache: &GruCache, bwd_cache: &GruCache, d_out: &Mat, grads: &mut [Mat]) -> Mat {
        let h = self.hidden();
        assert_eq!((d_out.rows, d_out.cols), (x.rows, 2 * h), "BiGru: gradient shape does not match the output");
        let d_fwd = Mat::from_fn(x.rows, h, |t, j| d_out[(t, j)]);
        let d_bwd = Mat::from_fn(x.rows, h, |t, j| d_out[(t, h + j)]);
        let (grads_fwd, grads_bwd) = grads.split_at_mut(4);
        let mut d_x = self.fwd.backward(x, fwd_cache, &d_fwd, Direction::Forward, grads_fwd);
        d_x.add_assign(&self.bwd.backward(x, bwd_cache, &d_bwd, Direction::Backward, grads_bwd));
        d_x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GRU with one input and one state entry, so every matrix is a number.
    fn scalar_gru() -> Gru {
        Gru {
            w_x: Mat::from_vec(1, 3, vec![0.5, -0.3, 0.8]),
            w_h: Mat::from_vec(1, 3, vec![0.1, 0.2, -0.4]),
            b_x: Mat::from_vec(1, 3, vec![0.1, 0.0, -0.1]),
            b_h: Mat::from_vec(1, 3, vec![0.0, 0.2, 0.3]),
        }
    }

    #[test]
    fn two_steps_by_hand() {
        // Step 1: x = 1, h = 0.
        //   r = sigmoid(0.5 + 0.1 + 0 + 0)          = sigmoid(0.6)  = 0.6456563
        //   z = sigmoid(-0.3 + 0 + 0 + 0.2)         = sigmoid(-0.1) = 0.4750208
        //   n = tanh(0.8 - 0.1 + r * (0 + 0.3))     = tanh(0.8936969) = 0.7132149
        //   h = (1 - z) * n + z * 0                 = 0.3744230
        // Step 2: x = -2, h = 0.3744230.
        //   r = sigmoid(-1 + 0.1 + 0.0374423)       = 0.2968052
        //   z = sigmoid(0.6 + 0.0748846 + 0.2)      = 0.7057611
        //   n = tanh(-1.6 - 0.1 + r * (-0.1497692 + 0.3)) = -0.9295964
        //   h = (1 - z) * n + z * 0.3744230         = -0.0092703
        let x = Mat::from_vec(2, 1, vec![1.0, -2.0]);
        let cache = scalar_gru().forward(&x, Direction::Forward);
        let expected_gates = [0.645_656_3, 0.475_020_8, 0.713_214_9, 0.296_805_2, 0.705_761_1, -0.929_596_4];
        for (got, want) in cache.gates.data.iter().zip(expected_gates) {
            assert!((got - want).abs() < 1e-6, "gate {got} vs {want}");
        }
        assert!((cache.h.data[0] - 0.374_423).abs() < 1e-6, "h1 = {}", cache.h.data[0]);
        assert!((cache.h.data[1] - -0.009_270_3).abs() < 1e-6, "h2 = {}", cache.h.data[1]);
    }

    #[test]
    fn backward_direction_is_forward_on_the_reversed_sequence() {
        let mut rng = Rng::new(3);
        let gru = Gru::new(3, 4, &mut rng);
        let x = Mat::from_fn(6, 3, |_, _| rng.normal());
        let x_reversed = Mat::from_fn(6, 3, |t, c| x[(5 - t, c)]);
        let a = gru.forward(&x, Direction::Backward);
        let b = gru.forward(&x_reversed, Direction::Forward);
        for t in 0..6 {
            assert_eq!(a.h.row(t), b.h.row(5 - t));
        }
    }

    #[test]
    fn matches_a_naive_implementation_of_the_equations() {
        // Recompute the GRU with explicit loops over single weights, straight
        // from the four equations, and compare.
        let mut rng = Rng::new(4);
        let (n_in, h) = (3, 5);
        let mut gru = Gru::new(n_in, h, &mut rng);
        for b in gru.b_x.data.iter_mut().chain(gru.b_h.data.iter_mut()) {
            *b = rng.normal() * 0.3; // non-zero biases, to test them too
        }
        let x = Mat::from_fn(7, n_in, |_, _| rng.normal());
        let cache = gru.forward(&x, Direction::Forward);

        let mut state = vec![0.0f32; h];
        for t in 0..x.rows {
            // sum_x[k] = x W_x + b_x and sum_h[k] = h W_h + b_h for k in 0..3H
            let mut sum_x = gru.b_x.data.clone();
            let mut sum_h = gru.b_h.data.clone();
            for k in 0..3 * h {
                for i in 0..n_in {
                    sum_x[k] += x[(t, i)] * gru.w_x[(i, k)];
                }
                for i in 0..h {
                    sum_h[k] += state[i] * gru.w_h[(i, k)];
                }
            }
            let mut new_state = vec![0.0f32; h];
            for j in 0..h {
                let r = sigmoid(sum_x[j] + sum_h[j]);
                let z = sigmoid(sum_x[h + j] + sum_h[h + j]);
                let n = (sum_x[2 * h + j] + r * sum_h[2 * h + j]).tanh();
                new_state[j] = (1.0 - z) * n + z * state[j];
            }
            state = new_state;
            for j in 0..h {
                assert!((cache.h[(t, j)] - state[j]).abs() < 1e-5, "t={t} j={j}");
            }
        }
    }

    #[test]
    fn recurrent_blocks_start_orthogonal() {
        let gru = Gru::new(4, 8, &mut Rng::new(5));
        for block in 0..3 {
            let q = Mat::from_fn(8, 8, |r, c| gru.w_h[(r, block * 8 + c)]);
            let qqt = q.matmul_t(&q);
            for i in 0..8 {
                assert!((qqt[(i, i)] - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn bigru_output_is_the_two_directions_side_by_side() {
        let mut rng = Rng::new(6);
        let bi = BiGru::new(3, 4, &mut rng);
        let x = Mat::from_fn(5, 3, |_, _| rng.normal());
        let (out, _, _) = bi.forward(&x);
        let f = bi.fwd.forward(&x, Direction::Forward);
        let b = bi.bwd.forward(&x, Direction::Backward);
        assert_eq!((out.rows, out.cols), (5, 8));
        for t in 0..5 {
            assert_eq!(&out.row(t)[..4], f.h.row(t));
            assert_eq!(&out.row(t)[4..], b.h.row(t));
        }
    }
}
