//! The numerical gradient checker.
//!
//! Hand-written backward passes are easy to get subtly wrong: a missing
//! factor, a flipped sign, the wrong transpose. The nasty part is that a
//! network with a slightly wrong gradient often still *sort of* trains, just
//! worse, and nothing ever crashes. So before trusting a backward pass we
//! test it against a second opinion that is slow and crude but very hard to
//! get wrong.
//!
//! The second opinion is the definition of a derivative. To find out how the
//! loss responds to one weight `w`, nudge it up by a small `eps`, nudge it
//! down by `eps`, and look at the difference:
//!
//! ```text
//! d loss / d w  ~=  (loss(w + eps) - loss(w - eps)) / (2 * eps)
//! ```
//!
//! This "central difference" needs two full forward passes *per weight*,
//! which is hopeless for training but fine for a test on a tiny model.
//!
//! # Doing it in `f32`
//!
//! Textbooks suggest `eps = 1e-5` or smaller. That assumes 64-bit floats. Our
//! matrices are `f32`, which carries about 7 significant digits. The two
//! losses we subtract agree in most of their digits, so with a loss around 1
//! their difference is only known to about `1e-7`, and dividing by `2 * eps`
//! magnifies that uncertainty: with `eps = 1e-5` the "gradient" would be
//! noise of size `0.005`. A larger `eps` shrinks the noise but brings in the
//! other error, the curvature of the loss between the two nudged points,
//! which grows like `eps^2`. For `f32` the sweet spot is around
//! **`eps = 1e-2`**, where both errors are near `1e-4` to `1e-5`.
//!
//! In practice a correct backward pass then scores a relative error between
//! `1e-5` and `1e-3`, while a real bug scores `0.1` to `2`, so a tolerance
//! of **`1e-2`** separates them with room to spare on both sides.
//!
//! # Measuring agreement
//!
//! For each tensor we compare the whole vector of analytic gradients `a`
//! with the vector of numerical ones `n`:
//!
//! ```text
//! relative error = |a - n| / (|a| + |n|)        (|.| = length of the vector)
//! ```
//!
//! This is 0 for identical vectors, 1 if they point in opposite directions
//! (a sign error), and about 1/3 if one is twice the other (a missing factor
//! of 2). Comparing whole tensors rather than single entries keeps a few
//! near-zero entries, where rounding noise is all there is, from raising
//! false alarms. A tensor whose gradients are *all* tiny passes if no entry
//! is off by more than a small absolute amount.
//!
//! # The ReLU caveat
//!
//! ReLU has a kink at zero, and a central difference that straddles a kink
//! measures the average of two different slopes, which is the derivative of
//! neither side. With `eps = 1e-2` this is not a rare accident: a nudge of
//! that size moves the input of every unit a little, and in a ReLU network
//! with a few dozen units some unit often sits close enough to zero to be
//! pushed across. A perfectly correct backward pass can then fail the check,
//! typically with a relative error of a few percent in one or two tensors
//! while the others pass.
//!
//! Before hunting for a bug, run the check again with a smaller nudge
//! (`eps = 1e-3`: fewer units are within reach of their kink, at the price
//! of more rounding noise) or with another seed, so that the units sit
//! somewhere else. A false alarm from a kink shrinks or disappears; a real
//! bug stays at `0.1` or more whatever you do. Checking the same
//! architecture with tanh in place of ReLU settles it for the other layers.
//! `examples/ch06_gradcheck.rs` stages such a false alarm on purpose.

use super::model::{Grads, Model};
use crate::mat::Mat;
use crate::rng::Rng;
use std::fmt;

/// Settings for [`gradcheck`].
#[derive(Clone, Copy, Debug)]
pub struct GradCheckConfig {
    /// How far to nudge each value. `1e-2` suits `f32` for smooth models. A
    /// ReLU model can fail at that size without being wrong (a nudge crosses
    /// a kink); retry with `1e-3` or another seed. See the module docs.
    pub eps: f32,
    /// Largest relative error that still counts as agreement.
    pub tolerance: f64,
    /// A tensor also passes if no entry differs by more than this, however
    /// large the relative error (for tensors whose gradient is essentially
    /// zero, where relative error is meaningless).
    pub abs_tolerance: f64,
    /// Check at most this many entries per tensor, picked at random. Keeps
    /// the check fast on tensors with thousands of entries.
    pub max_per_tensor: usize,
    /// Seed for picking which entries to check.
    pub seed: u64,
}

impl Default for GradCheckConfig {
    fn default() -> Self {
        GradCheckConfig { eps: 1e-2, tolerance: 1e-2, abs_tolerance: 1e-4, max_per_tensor: 40, seed: 0 }
    }
}

/// The verdict for one tensor.
#[derive(Clone, Debug)]
pub struct TensorCheck {
    /// Which tensor: a parameter name such as `0.linear.weight`, or `input`.
    pub name: String,
    /// How many entries were compared.
    pub checked: usize,
    /// The largest difference between an analytic and a numerical entry.
    pub max_abs_diff: f64,
    /// `|a - n| / (|a| + |n|)` over the compared entries.
    pub rel_error: f64,
    /// Whether the two gradients agree within the tolerances.
    pub pass: bool,
}

/// The verdicts for every tensor of a model, plus its input.
#[derive(Clone, Debug)]
pub struct GradCheckReport {
    /// One entry per parameter tensor, then one for the input.
    pub checks: Vec<TensorCheck>,
}

impl GradCheckReport {
    /// True if every tensor passed.
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.pass)
    }

    /// The largest relative error among all tensors.
    pub fn worst_rel_error(&self) -> f64 {
        self.checks.iter().map(|c| c.rel_error).fold(0.0, f64::max)
    }
}

impl fmt::Display for GradCheckReport {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{:<24} {:>7} {:>12} {:>12}", "tensor", "checked", "max |diff|", "rel. error")?;
        for c in &self.checks {
            let verdict = if c.pass { "ok" } else { "MISMATCH" };
            writeln!(f, "{:<24} {:>7} {:>12.2e} {:>12.2e}  {verdict}", c.name, c.checked, c.max_abs_diff, c.rel_error)?;
        }
        Ok(())
    }
}

/// Estimate the gradient of `loss` at `values` by central differences.
///
/// `loss` is called with the nudged values, twice per entry, and returns the
/// loss as an `f64` (accumulate in `f64` if you can; every digit helps).
/// `values` is modified during the call and restored before it returns.
///
/// This is the building block; use [`gradcheck`] to check a whole [`Model`].
pub fn numerical_gradient(values: &mut [f32], eps: f32, mut loss: impl FnMut(&[f32]) -> f64) -> Vec<f32> {
    let mut gradient = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let original = values[i];
        let (up, down) = (original + eps, original - eps);
        values[i] = up;
        let loss_up = loss(values);
        values[i] = down;
        let loss_down = loss(values);
        values[i] = original;
        // Divide by the nudge we actually made: after rounding to f32,
        // `up - down` is not exactly `2 * eps`.
        gradient.push(((loss_up - loss_down) / (up as f64 - down as f64)) as f32);
    }
    gradient
}

/// Compare an analytic gradient with a numerical one and decide whether they
/// agree. See the module docs for how the relative error is defined.
pub fn compare_gradients(name: &str, analytic: &[f32], numeric: &[f32], tolerance: f64, abs_tolerance: f64) -> TensorCheck {
    assert_eq!(analytic.len(), numeric.len(), "compare_gradients: the two gradients have different lengths");
    let (mut diff_sq, mut a_sq, mut n_sq, mut max_abs_diff) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &n) in analytic.iter().zip(numeric) {
        let (a, n) = (a as f64, n as f64);
        diff_sq += (a - n) * (a - n);
        a_sq += a * a;
        n_sq += n * n;
        max_abs_diff = max_abs_diff.max((a - n).abs());
    }
    let denominator = a_sq.sqrt() + n_sq.sqrt();
    // Both gradients exactly zero is perfect agreement. (A NaN anywhere makes
    // the denominator NaN, which is not equal to zero, so NaN propagates into
    // the relative error and fails the check, as it should.)
    let rel_error = if denominator == 0.0 { 0.0 } else { diff_sq.sqrt() / denominator };
    let pass = rel_error.is_finite() && (rel_error <= tolerance || max_abs_diff <= abs_tolerance);
    TensorCheck { name: name.to_string(), checked: analytic.len(), max_abs_diff, rel_error, pass }
}

/// Check a model's backward pass against central differences, for every
/// parameter tensor and for the input.
///
/// `loss_fn` maps the model's output to `(loss, d loss / d output)`, exactly
/// like the loss functions used for training; wrap the target in a closure:
///
/// ```text
/// let report = gradcheck(&model, &x, |logits| cross_entropy(logits, &classes), &GradCheckConfig::default());
/// println!("{report}");
/// assert!(report.passed());
/// ```
pub fn gradcheck(model: &Model, x: &Mat, loss_fn: impl Fn(&Mat) -> (f32, Mat), config: &GradCheckConfig) -> GradCheckReport {
    // The answer under test: one forward and one backward pass.
    let cache = model.forward(x);
    let (_, d_logits) = loss_fn(cache.output());
    let mut grads = Grads::zeros_like(model);
    let d_input = model.backward(&cache, &d_logits, &mut grads);

    let mut rng = Rng::new(config.seed);
    let mut probe = model.clone(); // a copy whose weights we are free to nudge
    let mut checks = Vec::new();

    for (i, name) in model.param_names().iter().enumerate() {
        let picks = pick_entries(grads.tensors[i].data.len(), config.max_per_tensor, &mut rng);
        let mut numeric = Vec::with_capacity(picks.len());
        for &k in &picks {
            let original = probe.params()[i].data[k];
            let (up, down) = (original + config.eps, original - config.eps);
            probe.params_mut()[i].data[k] = up;
            let loss_up = loss_fn(&probe.predict(x)).0 as f64;
            probe.params_mut()[i].data[k] = down;
            let loss_down = loss_fn(&probe.predict(x)).0 as f64;
            probe.params_mut()[i].data[k] = original;
            numeric.push(((loss_up - loss_down) / (up as f64 - down as f64)) as f32);
        }
        let analytic: Vec<f32> = picks.iter().map(|&k| grads.tensors[i].data[k]).collect();
        checks.push(compare_gradients(name, &analytic, &numeric, config.tolerance, config.abs_tolerance));
    }

    // The same for the input.
    let picks = pick_entries(x.data.len(), config.max_per_tensor, &mut rng);
    let mut x_probe = x.clone();
    let mut numeric = Vec::with_capacity(picks.len());
    for &k in &picks {
        let original = x_probe.data[k];
        let (up, down) = (original + config.eps, original - config.eps);
        x_probe.data[k] = up;
        let loss_up = loss_fn(&model.predict(&x_probe)).0 as f64;
        x_probe.data[k] = down;
        let loss_down = loss_fn(&model.predict(&x_probe)).0 as f64;
        x_probe.data[k] = original;
        numeric.push(((loss_up - loss_down) / (up as f64 - down as f64)) as f32);
    }
    let analytic: Vec<f32> = picks.iter().map(|&k| d_input.data[k]).collect();
    checks.push(compare_gradients("input", &analytic, &numeric, config.tolerance, config.abs_tolerance));

    GradCheckReport { checks }
}

/// Choose which entries of a tensor with `len` entries to check: all of them
/// if there are few, otherwise `max` distinct ones at random.
fn pick_entries(len: usize, max: usize, rng: &mut Rng) -> Vec<usize> {
    let mut all: Vec<usize> = (0..len).collect();
    if len > max {
        rng.shuffle(&mut all);
        all.truncate(max);
        all.sort_unstable();
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerical_gradient_of_a_known_function() {
        // f(a, b) = a^2 + 3ab  has gradient (2a + 3b, 3a). At (2, -1): (1, 6).
        let mut point = vec![2.0f32, -1.0];
        let g = numerical_gradient(&mut point, 1e-2, |v| (v[0] as f64).powi(2) + 3.0 * v[0] as f64 * v[1] as f64);
        assert!((g[0] - 1.0).abs() < 1e-4 && (g[1] - 6.0).abs() < 1e-4, "{g:?}");
        assert_eq!(point, vec![2.0, -1.0], "values must be restored");
    }

    #[test]
    fn comparison_scores() {
        let a = [1.0f32, -2.0, 0.5];
        // Identical: error 0.
        assert_eq!(compare_gradients("t", &a, &a, 1e-2, 0.0).rel_error, 0.0);
        // Sign flipped: |a - (-a)| / (|a| + |a|) = 1.
        let flipped: Vec<f32> = a.iter().map(|v| -v).collect();
        let c = compare_gradients("t", &a, &flipped, 1e-2, 0.0);
        assert!((c.rel_error - 1.0).abs() < 1e-9 && !c.pass);
        // Off by a factor of two: |a| / (3 |a|) = 1/3.
        let doubled: Vec<f32> = a.iter().map(|v| 2.0 * v).collect();
        let c = compare_gradients("t", &a, &doubled, 1e-2, 0.0);
        assert!((c.rel_error - 1.0 / 3.0).abs() < 1e-9 && !c.pass);
        // Both essentially zero: passes on the absolute tolerance.
        assert!(compare_gradients("t", &[0.0, 0.0], &[1e-6, -1e-6], 1e-2, 1e-4).pass);
        // NaN never passes.
        assert!(!compare_gradients("t", &[f32::NAN], &[1.0], 1e-2, 1e-4).pass);
    }

    #[test]
    fn pick_entries_is_bounded_and_distinct() {
        let mut rng = Rng::new(1);
        assert_eq!(pick_entries(5, 40, &mut rng), vec![0, 1, 2, 3, 4]);
        let picks = pick_entries(1000, 40, &mut rng);
        assert_eq!(picks.len(), 40);
        assert!(picks.windows(2).all(|w| w[0] < w[1]));
    }
}
