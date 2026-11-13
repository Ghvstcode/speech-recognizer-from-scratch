//! The linear (fully connected) layer: `y = x W + b`.
//!
//! Each row of the input is treated on its own. A row `x` with `n_in` numbers
//! becomes a row `y` with `n_out` numbers, and every output is a weighted sum
//! of all the inputs plus a bias:
//!
//! ```text
//! y[j] = b[j] + sum over i of  x[i] * W[i][j]
//! ```
//!
//! We store `W` with **one row per input and one column per output**
//! (`n_in x n_out`), so that for a whole `T x n_in` matrix of rows the forward
//! pass is literally the matrix product `X W`. (PyTorch stores the transpose,
//! `n_out x n_in`. Same numbers, other way up.)
//!
//! # The backward pass
//!
//! We are handed `dY`, the gradient of the loss with respect to our output
//! (same shape as `Y`), and have to produce three things:
//!
//! ```text
//! dW = X^T dY     W[i][j] touched the loss through every row t, by x[t][i] * dY[t][j]
//! db = column sums of dY     b[j] was added to y[t][j] in every row
//! dX = dY W^T     x[t][i] fed every output j of its row through W[i][j]
//! ```

use super::init;
use super::kernels;
use crate::mat::Mat;
use crate::rng::Rng;

/// A fully connected layer applied to every row of its input.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    /// `n_in x n_out`: row `i` holds the weights leaving input `i`.
    pub weight: Mat,
    /// `1 x n_out`: one bias per output.
    pub bias: Mat,
}

impl Linear {
    /// A layer with Xavier-initialised weights and zero biases.
    ///
    /// Xavier is the right choice when the layer is followed by tanh, sigmoid,
    /// or nothing (the last layer of a network, whose outputs are the logits).
    /// See [`init::xavier_uniform`] for why.
    pub fn new(n_in: usize, n_out: usize, rng: &mut Rng) -> Self {
        Linear { weight: init::xavier_uniform(n_in, n_out, n_in, n_out, rng), bias: Mat::zeros(1, n_out) }
    }

    /// A layer with He-initialised weights and zero biases, for layers
    /// followed by a ReLU. See [`init::he_uniform`] for why.
    pub fn new_he(n_in: usize, n_out: usize, rng: &mut Rng) -> Self {
        Linear { weight: init::he_uniform(n_in, n_out, n_in, rng), bias: Mat::zeros(1, n_out) }
    }

    /// Number of values in each input row.
    pub fn n_in(&self) -> usize {
        self.weight.rows
    }

    /// Number of values in each output row.
    pub fn n_out(&self) -> usize {
        self.weight.cols
    }

    /// `Y = X W + b` for a `T x n_in` input; returns `T x n_out`.
    pub fn forward(&self, x: &Mat) -> Mat {
        assert_eq!(x.cols, self.n_in(), "Linear: input has {} columns, the layer expects {}", x.cols, self.n_in());
        let mut y = kernels::matmul(x, &self.weight);
        kernels::add_row_to_each(&mut y, &self.bias.data);
        y
    }

    /// Given the input `x` that `forward` saw and the gradient `d_out` at the
    /// output, add the weight and bias gradients into `grads` (`[weight,
    /// bias]`) and return the gradient at the input.
    pub fn backward(&self, x: &Mat, d_out: &Mat, grads: &mut [Mat]) -> Mat {
        assert_eq!((d_out.rows, d_out.cols), (x.rows, self.n_out()), "Linear: gradient shape does not match the output");
        kernels::add_t_matmul(x, d_out, &mut grads[0]); // dW += X^T dY
        kernels::add_column_sums(d_out, &mut grads[1].data); // db += column sums of dY
        kernels::matmul_transposed(d_out, &self.weight) // dX = dY W^T
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2-in, 2-out layer small enough to do on paper.
    fn tiny() -> Linear {
        Linear { weight: Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]), bias: Mat::from_vec(1, 2, vec![0.5, -0.5]) }
    }

    #[test]
    fn forward_matches_hand_computation() {
        // Row [1, 1]:  y0 = 1*1 + 1*3 + 0.5 = 4.5    y1 = 1*2 + 1*4 - 0.5 = 5.5
        // Row [2, -1]: y0 = 2*1 - 1*3 + 0.5 = -0.5   y1 = 2*2 - 1*4 - 0.5 = -0.5
        let x = Mat::from_vec(2, 2, vec![1.0, 1.0, 2.0, -1.0]);
        assert_eq!(tiny().forward(&x).data, vec![4.5, 5.5, -0.5, -0.5]);
    }

    #[test]
    fn backward_matches_hand_computation() {
        let x = Mat::from_vec(2, 2, vec![1.0, 1.0, 2.0, -1.0]);
        let d_out = Mat::from_vec(2, 2, vec![1.0, 0.0, 0.5, -1.0]);
        let mut grads = vec![Mat::zeros(2, 2), Mat::zeros(1, 2)];
        let d_in = tiny().backward(&x, &d_out, &mut grads);
        // dW = X^T dY:
        //   dW[0][0] = 1*1 + 2*0.5 = 2     dW[0][1] = 1*0 + 2*(-1) = -2
        //   dW[1][0] = 1*1 - 1*0.5 = 0.5   dW[1][1] = 1*0 - 1*(-1) = 1
        assert_eq!(grads[0].data, vec![2.0, -2.0, 0.5, 1.0]);
        // db = column sums of dY
        assert_eq!(grads[1].data, vec![1.5, -1.0]);
        // dX = dY W^T:
        //   row 0: [1*1 + 0*2, 1*3 + 0*4] = [1, 3]
        //   row 1: [0.5*1 - 1*2, 0.5*3 - 1*4] = [-1.5, -2.5]
        assert_eq!(d_in.data, vec![1.0, 3.0, -1.5, -2.5]);
    }

    #[test]
    fn backward_accumulates_rather_than_overwrites() {
        let x = Mat::from_vec(1, 2, vec![1.0, 1.0]);
        let d_out = Mat::from_vec(1, 2, vec![1.0, 1.0]);
        let mut grads = vec![Mat::zeros(2, 2), Mat::zeros(1, 2)];
        tiny().backward(&x, &d_out, &mut grads);
        tiny().backward(&x, &d_out, &mut grads);
        assert_eq!(grads[1].data, vec![2.0, 2.0]);
    }
}
