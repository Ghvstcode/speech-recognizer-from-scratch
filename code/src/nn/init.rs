//! Choosing the starting weights.
//!
//! A network cannot start from all zeros: every unit in a layer would compute
//! the same thing, receive the same gradient, and stay identical forever. So
//! we start from small random numbers. *How* small matters a lot: too small
//! and the signal fades to nothing after a few layers, too large and it blows
//! up (or pins every tanh at +/-1, where its slope, and so its gradient, is
//! zero). The recipes below pick the scale so that a signal keeps roughly the
//! same size as it passes through a layer.

use crate::mat::Mat;
use crate::rng::Rng;

/// Xavier (Glorot) initialization: uniform in `[-a, a]` with
/// `a = sqrt(6 / (fan_in + fan_out))`.
///
/// `fan_in` is how many inputs feed one output, `fan_out` how many outputs one
/// input feeds. A uniform distribution on `[-a, a]` has variance `a^2 / 3`, so
/// this gives each weight variance `2 / (fan_in + fan_out)`: a compromise
/// between keeping the forward signal's size (which wants `1 / fan_in`) and
/// keeping the backward gradient's size (which wants `1 / fan_out`). It is the
/// right default for layers followed by tanh, sigmoid, or nothing at all.
pub fn xavier_uniform(rows: usize, cols: usize, fan_in: usize, fan_out: usize, rng: &mut Rng) -> Mat {
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    Mat::from_fn(rows, cols, |_, _| rng.range(-limit, limit))
}

/// He (Kaiming) initialization: uniform in `[-a, a]` with `a = sqrt(6 / fan_in)`.
///
/// A ReLU throws away the negative half of its input, which halves the
/// signal's variance. He initialization doubles the weights' variance
/// (`2 / fan_in`) to make up for it. Use it for layers followed by a ReLU.
pub fn he_uniform(rows: usize, cols: usize, fan_in: usize, rng: &mut Rng) -> Mat {
    let limit = (6.0 / fan_in as f32).sqrt();
    Mat::from_fn(rows, cols, |_, _| rng.range(-limit, limit))
}

/// A random `n x n` orthogonal matrix: its rows all have length 1 and are at
/// right angles to each other.
///
/// Multiplying a vector by an orthogonal matrix rotates it without changing
/// its length. A recurrent layer multiplies its state by the same matrix at
/// every time step, a hundred times in a row for a one-second clip; if that
/// matrix shrinks vectors even slightly, a hundred steps shrink them to
/// nothing (and the gradient with them), and if it stretches them they
/// explode. Starting from a pure rotation avoids both.
///
/// We build it with the Gram-Schmidt process: take random rows one at a time,
/// subtract from each whatever part of it points along the rows already
/// accepted, and scale what is left to length 1.
pub fn orthogonal(n: usize, rng: &mut Rng) -> Mat {
    let mut q = Mat::from_fn(n, n, |_, _| rng.normal());
    for i in 0..n {
        let (done, rest) = q.data.split_at_mut(i * n);
        let row = &mut rest[..n];
        // Two passes: after one, rounding error leaves a little of the earlier
        // rows behind; the second pass removes it.
        for _ in 0..2 {
            for earlier in done.chunks_exact(n) {
                let along = crate::mat::dot(row, earlier);
                for (v, &e) in row.iter_mut().zip(earlier) {
                    *v -= along * e;
                }
            }
        }
        let length = crate::mat::dot(row, row).sqrt();
        for v in row.iter_mut() {
            *v /= length;
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_inits_respect_their_limits() {
        let mut rng = Rng::new(1);
        let x = xavier_uniform(30, 50, 30, 50, &mut rng);
        let limit = (6.0f32 / 80.0).sqrt();
        assert!(x.data.iter().all(|v| v.abs() <= limit));
        // Variance of uniform[-a, a] is a^2 / 3 = 2 / (fan_in + fan_out) = 0.025.
        let var = x.data.iter().map(|v| v * v).sum::<f32>() / x.data.len() as f32;
        assert!((var - 0.025).abs() < 0.003, "variance {var}");

        let h = he_uniform(30, 50, 30, &mut rng);
        assert!(h.data.iter().all(|v| v.abs() <= (6.0f32 / 30.0).sqrt()));
    }

    #[test]
    fn orthogonal_times_its_transpose_is_identity() {
        let mut rng = Rng::new(2);
        let q = orthogonal(16, &mut rng);
        let qqt = q.matmul_t(&q);
        for r in 0..16 {
            for c in 0..16 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((qqt[(r, c)] - expected).abs() < 1e-5, "({r},{c}) = {}", qqt[(r, c)]);
            }
        }
    }
}
