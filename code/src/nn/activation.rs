//! Activation functions: the bends that make a network more than one big
//! matrix product.
//!
//! Stack two linear layers with nothing in between and you get... one linear
//! layer (`(x A) B = x (A B)`). An activation function is applied to every
//! number on its own, between the layers, and breaks that collapse.
//!
//! None of these has weights, so their backward pass only has to turn the
//! gradient at the output into the gradient at the input. Since each output
//! depends on exactly one input, that is a multiplication by the slope:
//!
//! ```text
//! d_in = d_out * slope of the function at the input
//! ```
//!
//! Conveniently, all three slopes can be computed from the *output* `y`
//! alone, so the backward functions take `y` and never need the input again.

use crate::mat::Mat;

/// The logistic sigmoid, `1 / (1 + e^-x)`: squashes any number into `(0, 1)`.
pub fn sigmoid(x: f32) -> f32 {
    // For very negative x, e^-x overflows to infinity and 1 / infinity is 0,
    // which is the right answer, so no special case is needed.
    1.0 / (1.0 + (-x).exp())
}

/// ReLU, `max(0, x)`: lets positive numbers through, replaces negative ones by 0.
pub fn relu_forward(x: &Mat) -> Mat {
    x.map(|v| v.max(0.0))
}

/// ReLU's slope is 1 where the input was positive (which is exactly where the
/// output is positive) and 0 elsewhere.
pub fn relu_backward(y: &Mat, d_out: &Mat) -> Mat {
    combine(y, d_out, |y, d| if y > 0.0 { d } else { 0.0 })
}

/// tanh: squashes any number into `(-1, 1)`.
pub fn tanh_forward(x: &Mat) -> Mat {
    x.map(f32::tanh)
}

/// The slope of tanh is `1 - tanh(x)^2 = 1 - y^2`.
pub fn tanh_backward(y: &Mat, d_out: &Mat) -> Mat {
    combine(y, d_out, |y, d| d * (1.0 - y * y))
}

/// The sigmoid applied to every element.
pub fn sigmoid_forward(x: &Mat) -> Mat {
    x.map(sigmoid)
}

/// The slope of the sigmoid is `y * (1 - y)`.
pub fn sigmoid_backward(y: &Mat, d_out: &Mat) -> Mat {
    combine(y, d_out, |y, d| d * y * (1.0 - y))
}

/// Build a matrix from two same-shaped ones, element by element.
fn combine(a: &Mat, b: &Mat, f: impl Fn(f32, f32) -> f32) -> Mat {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols), "activation: gradient shape does not match the output");
    Mat { rows: a.rows, cols: a.cols, data: a.data.iter().zip(&b.data).map(|(&x, &y)| f(x, y)).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_values() {
        let x = Mat::from_vec(1, 3, vec![-2.0, 0.0, 3.0]);
        assert_eq!(relu_forward(&x).data, vec![0.0, 0.0, 3.0]);
        let t = tanh_forward(&x);
        assert!((t.data[2] - 0.995_054_8).abs() < 1e-6 && t.data[1] == 0.0);
        let s = sigmoid_forward(&x);
        // sigmoid(0) = 1/2, sigmoid(-2) = 1 / (1 + e^2) = 0.1192029
        assert!((s.data[0] - 0.119_202_9).abs() < 1e-6 && s.data[1] == 0.5);
    }

    #[test]
    fn sigmoid_survives_extreme_inputs() {
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert_eq!(sigmoid(1000.0), 1.0);
    }

    #[test]
    fn backward_multiplies_by_the_slope() {
        let d = Mat::from_vec(1, 2, vec![10.0, 10.0]);
        // ReLU: output 0 blocks the gradient, positive output passes it.
        assert_eq!(relu_backward(&Mat::from_vec(1, 2, vec![0.0, 2.0]), &d).data, vec![0.0, 10.0]);
        // tanh: y = 0.5 has slope 1 - 0.25 = 0.75.
        assert_eq!(tanh_backward(&Mat::from_vec(1, 2, vec![0.0, 0.5]), &d).data, vec![10.0, 7.5]);
        // sigmoid: y = 0.5 has slope 0.25, y = 1 has slope 0.
        assert_eq!(sigmoid_backward(&Mat::from_vec(1, 2, vec![0.5, 1.0]), &d).data, vec![2.5, 0.0]);
    }
}
