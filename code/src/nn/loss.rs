//! Softmax and cross-entropy: turning scores into probabilities, and
//! probabilities into a number that says how wrong we were.
//!
//! A model's last layer produces one unbounded score per class for each row,
//! the **logits**. Softmax turns a row of logits into probabilities (all
//! positive, summing to 1) by exponentiating and normalising:
//!
//! ```text
//! p[c] = e^logit[c] / sum over k of e^logit[k]
//! ```
//!
//! The **cross-entropy** loss for a row whose correct class is `y` is
//! `-ln p[y]`: zero when the model gives the right class probability 1, and
//! growing without bound as that probability goes to 0.
//!
//! # Staying finite
//!
//! Computed as written, `e^logit` overflows an `f32` as soon as a logit
//! passes 88. The standard cure is to work with logarithms throughout:
//!
//! ```text
//! ln p[c] = logit[c] - ln(sum over k of e^logit[k])
//! ```
//!
//! and to compute that last term, the "log-sum-exp", by first pulling the
//! largest logit `m` out of the sum: `m + ln(sum of e^(logit[k] - m))`. Now
//! every exponent is zero or negative, so nothing can overflow, and the
//! largest term is exactly `e^0 = 1`, so the sum cannot underflow to zero
//! either.

use crate::mat::Mat;

/// `ln(sum of e^v)` over a slice, computed without overflow.
pub fn log_sum_exp(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max; // empty or all -infinity: the sum is 0 and its log -infinity
    }
    let sum: f32 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Replace a row of logits by its log-probabilities, in place.
fn log_softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    // Subtract the maximum first and the log of the sum second. Folding the
    // two into one number (the log-sum-exp) and subtracting that would be
    // the same on paper, but for logits in the thousands the small `ln(sum)`
    // would be rounded away inside the big `max`.
    let log_sum = row.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
    for v in row.iter_mut() {
        *v = (*v - max) - log_sum;
    }
}

/// Log-probabilities for every row of a logits matrix: each output row is the
/// natural log of the softmax of the matching input row.
pub fn log_softmax(logits: &Mat) -> Mat {
    let mut out = logits.clone();
    if out.cols == 0 {
        return out;
    }
    for row in out.data.chunks_exact_mut(logits.cols) {
        log_softmax_row(row);
    }
    out
}

/// Probabilities for every row of a logits matrix: each output row is
/// positive and sums to 1.
pub fn softmax(logits: &Mat) -> Mat {
    log_softmax(logits).map(f32::exp)
}

/// Index of the largest value in a slice (the first one, if there are ties).
/// The class a row of logits "votes" for.
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// Softmax cross-entropy for one row of logits and the index of the correct
/// class. Returns `(loss, d loss / d logits)`.
///
/// The gradient is famously simple: `p - onehot(class)`, that is, the
/// predicted probabilities, minus 1 at the correct class. If the model says
/// "70% cat" and the answer is cat, the cat logit's gradient is `0.7 - 1 =
/// -0.3` (push it up) and every other logit's gradient is its probability
/// (push it down, in proportion to how much it was wrongly believed).
pub fn cross_entropy_row(logits: &[f32], class: usize) -> (f32, Vec<f32>) {
    assert!(class < logits.len(), "cross_entropy: class {class} but there are only {} logits", logits.len());
    let mut log_p = logits.to_vec();
    log_softmax_row(&mut log_p);
    let loss = -log_p[class];
    let mut d_logits: Vec<f32> = log_p.iter().map(|&lp| lp.exp()).collect(); // p
    d_logits[class] -= 1.0;
    (loss, d_logits)
}

/// Softmax cross-entropy for a whole matrix of logits, one correct class per
/// row. Returns `(sum of the rows' losses, d loss / d logits)`.
///
/// This has the same signature as the CTC loss of chapter 8, so the two can
/// be swapped in a training loop. For a clip classifier the logits have a
/// single row and `classes` a single entry. The loss is a *sum* over rows;
/// divide by the number of rows yourself if you want a per-row average.
pub fn cross_entropy(logits: &Mat, classes: &[usize]) -> (f32, Mat) {
    assert_eq!(logits.rows, classes.len(), "cross_entropy: {} rows of logits but {} class labels", logits.rows, classes.len());
    let mut total = 0.0;
    let mut d_logits = Mat::zeros(logits.rows, logits.cols);
    for (r, &class) in classes.iter().enumerate() {
        let (loss, d_row) = cross_entropy_row(logits.row(r), class);
        total += loss;
        d_logits.row_mut(r).copy_from_slice(&d_row);
    }
    (total, d_logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_by_hand() {
        // Logits [0, ln 3]: e^0 = 1 and e^(ln 3) = 3, so p = [1/4, 3/4].
        let logits = Mat::from_vec(1, 2, vec![0.0, 3.0f32.ln()]);
        let p = softmax(&logits);
        assert!((p.data[0] - 0.25).abs() < 1e-6 && (p.data[1] - 0.75).abs() < 1e-6);
        let lp = log_softmax(&logits);
        assert!((lp.data[0] - 0.25f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn cross_entropy_by_hand() {
        // Same logits, correct class 1: loss = -ln(3/4), gradient = p - [0, 1].
        let (loss, d) = cross_entropy_row(&[0.0, 3.0f32.ln()], 1);
        assert!((loss - (4.0f32 / 3.0).ln()).abs() < 1e-6);
        assert!((d[0] - 0.25).abs() < 1e-6 && (d[1] + 0.25).abs() < 1e-6);
        // Uniform logits over 4 classes: loss = ln 4 whatever the class.
        let (loss, _) = cross_entropy_row(&[2.0; 4], 3);
        assert!((loss - 4.0f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn huge_logits_do_not_overflow() {
        let (loss, d) = cross_entropy_row(&[1000.0, 0.0, -1000.0], 0);
        assert_eq!(loss, 0.0);
        assert!(d.iter().all(|v| v.is_finite()));
        let (loss, d) = cross_entropy_row(&[1000.0, 0.0, -1000.0], 2);
        assert_eq!(loss, 2000.0);
        assert!(d.iter().all(|v| v.is_finite()));
        let p = softmax(&Mat::from_vec(1, 2, vec![-5000.0, -5000.0]));
        assert_eq!(p.data, vec![0.5, 0.5]);
    }

    #[test]
    fn matrix_version_sums_the_rows() {
        let logits = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 0.5, -1.0, 0.0]);
        let (total, d) = cross_entropy(&logits, &[2, 0]);
        let (l0, d0) = cross_entropy_row(logits.row(0), 2);
        let (l1, d1) = cross_entropy_row(logits.row(1), 0);
        assert!((total - (l0 + l1)).abs() < 1e-6);
        assert_eq!(d.row(0), &d0[..]);
        assert_eq!(d.row(1), &d1[..]);
        // Each row of the gradient sums to zero (probabilities sum to 1, minus the 1).
        assert!(d.row(0).iter().sum::<f32>().abs() < 1e-6);
    }

    #[test]
    fn argmax_picks_the_first_maximum() {
        assert_eq!(argmax(&[0.1, 0.7, 0.7, -2.0]), 1);
    }
}
