//! Connectionist temporal classification (CTC): a loss for when we know *what*
//! was said but not *when*.
//!
//! The network looks at T frames of audio and, for every frame, outputs a
//! probability for each of C classes. The target is a much shorter sequence of
//! labels, say `c a t`. Nobody tells us which frames belong to the `c`.
//!
//! CTC's answer has three parts.
//!
//! 1. Add one extra class, the **blank** (class 0 here), meaning "nothing new
//!    to report in this frame".
//! 2. Call a sequence of T classes, one per frame, an **alignment** (or path).
//!    An alignment *collapses* to a labelling by first merging runs of the
//!    same class and then deleting blanks: `c c _ a a a _ t` and
//!    `_ c a _ _ t t _` both collapse to `c a t`. See [`collapse`].
//! 3. The probability of an alignment is the product of its per-frame
//!    probabilities. The probability of the *target* is the sum of that over
//!    **every** alignment that collapses to it. The loss is minus the
//!    logarithm of that sum.
//!
//! The number of alignments explodes with T, so we cannot list them (we do,
//! in the tests, for T up to 6, to check the clever way against the stupid
//! way). The clever way is dynamic programming over a grid called the
//! **lattice**, and it is the same idea as the DTW grid from chapter 5 with
//! `min` replaced by `sum`.
//!
//! ## The lattice
//!
//! Put a blank between every pair of target labels and at both ends:
//! `c a t` becomes `_ c _ a _ t _`, the **extended target**, of length
//! `S = 2 * len + 1` ([`extend_with_blanks`]). Every valid alignment is a walk
//! through a grid with one row per frame and one column per position in the
//! extended target, that
//!
//! * starts in column 0 (the leading blank) or column 1 (the first label),
//! * at each frame stays in its column, moves one column right, or skips over
//!   a blank by moving two columns right, the skip being allowed only when it
//!   lands on a label different from the label it left (otherwise `a _ a`
//!   would merge into a single `a`),
//! * ends in the last column (trailing blank) or the one before (last label).
//!
//! ## Conventions used throughout this file
//!
//! * **Blank is class 0.** Target labels are class ids in `1..C`.
//! * **Lattice orientation:** every lattice is a `Mat` with **one row per
//!   frame** (`T` rows) and **one column per extended-target position** (`S`
//!   columns). `lattice[(t, s)]` talks about frame `t` and position `s`.
//!   (Figures usually draw time left to right; transpose for that.)
//! * **Log space.** A path probability is a product of T numbers below 1, and
//!   for T = 500 that underflows even a 64-bit float. We store logarithms
//!   instead: products become sums, and an impossible cell holds `-inf`, the
//!   logarithm of zero. Adding two probabilities stored as logarithms is
//!   [`log_add`]. The recursions need a few such additions per cell, so they
//!   do them a row at a time: step out of log space *relative to the row's
//!   largest value*, add ordinary numbers, and step back in.
//! * **f64 inside, f32 outside.** The lattices are computed in `f64` and
//!   converted to `f32` on the way out. The inputs and the gradient are `f32`
//!   like the rest of the crate, but the loss is a small difference of large
//!   sums, and the numerical gradient check in the tests needs more digits
//!   than `f32` carries. [`ctc_loss_f64`] returns the loss before rounding.

use crate::mat::Mat;

/// The class index reserved for the CTC blank.
pub const BLANK: usize = 0;

/// The logarithm of zero: the log-probability of something impossible.
const LOG_ZERO: f64 = f64::NEG_INFINITY;

/// `ln(e^a + e^b)`: add two probabilities that are stored as logarithms.
///
/// Computing `e^a` directly would underflow to zero for the very negative
/// numbers we deal with. Instead factor out the larger of the two:
/// `ln(e^a + e^b) = hi + ln(1 + e^(lo - hi))`. The exponent `lo - hi` is never
/// positive, so `e^(lo - hi)` is between 0 and 1 and nothing can overflow.
pub fn log_add(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if lo == LOG_ZERO {
        return hi; // adding zero probability changes nothing (and avoids inf - inf)
    }
    hi + (lo - hi).exp().ln_1p()
}

/// Merge repeated classes, then drop blanks: the map from an alignment (one
/// class per frame) to the labelling it stands for.
///
/// `[1, 1, 0, 1, 2, 2, 0]` becomes `[1, 1, 2]`: the two leading 1s merge, the
/// blank separates them from the next 1, and the 2s merge.
pub fn collapse(path: &[usize]) -> Vec<usize> {
    let mut labels = Vec::new();
    let mut previous = BLANK;
    for &class in path {
        if class != BLANK && class != previous {
            labels.push(class);
        }
        previous = class;
    }
    labels
}

/// Interleave blanks: `[c, a, t]` becomes `[_, c, _, a, _, t, _]`.
/// An empty target becomes a single blank.
pub fn extend_with_blanks(target: &[usize]) -> Vec<usize> {
    let mut extended = Vec::with_capacity(2 * target.len() + 1);
    extended.push(BLANK);
    for &label in target {
        extended.push(label);
        extended.push(BLANK);
    }
    extended
}

/// Turn each row of logits into log-probabilities:
/// `log_softmax(z)[k] = z[k] - ln(sum over j of e^z[j])`.
///
/// After this, `e^row[k]` is a probability and each row's probabilities sum
/// to 1. The sum is computed with the largest logit factored out, for the
/// same reason as in [`log_add`].
pub fn log_softmax(logits: &Mat) -> Mat {
    let log_probs = log_softmax_f64(logits);
    Mat::from_vec(logits.rows, logits.cols, log_probs.iter().map(|&v| v as f32).collect())
}

/// [`log_softmax`] in double precision, as a flat row-major buffer.
fn log_softmax_f64(logits: &Mat) -> Vec<f64> {
    let mut out = Vec::with_capacity(logits.data.len());
    for t in 0..logits.rows {
        let row = logits.row(t);
        let max = row.iter().fold(f32::NEG_INFINITY, |m, &z| m.max(z)) as f64;
        let mut sum = 0.0;
        for &z in row {
            sum += (z as f64 - max).exp();
        }
        let log_sum = max + sum.ln();
        out.extend(row.iter().map(|&z| z as f64 - log_sum));
    }
    out
}

/// Widen a matrix of f32 log-probabilities to a flat f64 buffer.
fn widen(log_probs: &Mat) -> Vec<f64> {
    log_probs.data.iter().map(|&v| v as f64).collect()
}

/// Wrap a flat f64 lattice as an f32 `Mat` with the given shape.
fn narrow(rows: usize, cols: usize, values: &[f64]) -> Mat {
    Mat::from_vec(rows, cols, values.iter().map(|&v| v as f32).collect())
}

/// Panic with a helpful message if `target` cannot be a CTC target for a
/// network with `classes` outputs.
fn check_target(target: &[usize], classes: usize) {
    for &label in target {
        assert!(label != BLANK, "ctc: the target must not contain the blank (class 0)");
        assert!(label < classes, "ctc: target label {label} but the network only has {classes} classes");
    }
}

/// May a path jump from position `s - 2` straight to position `s`, skipping
/// the blank between them?
///
/// Only when `s` holds a label (odd positions do) that differs from the label
/// two positions back. Skipping the blank between two equal labels would glue
/// them into one when the path is collapsed.
fn can_skip_into(extended: &[usize], s: usize) -> bool {
    s >= 2 && extended[s] != BLANK && extended[s] != extended[s - 2]
}

/// Below this, a sum of rescaled probabilities is too close to the bottom of
/// `f64`'s range to trust; see [`log_of_sum`].
const SAFE_SUM: f64 = 1e-200;

/// Leave log space for one row of a lattice: fill `scaled[s]` with
/// `e^(row[s] - m)`, where `m` is the largest value in the row, and return
/// `m`.
///
/// Subtracting the maximum first is what makes this safe. The largest entry
/// becomes `e^0 = 1` and every other entry lands between 0 and 1, however
/// negative the logarithms were. It is the trick from [`log_add`], applied to
/// a whole row at once so that we pay for one `exp` per cell rather than one
/// per addition.
fn exp_relative_to_max(row: &[f64], scaled: &mut [f64]) -> f64 {
    let row_max = row.iter().fold(LOG_ZERO, |m, &v| m.max(v));
    for (p, &log_value) in scaled.iter_mut().zip(row) {
        *p = (log_value - row_max).exp();
    }
    row_max
}

/// Go back to log space after adding up rescaled probabilities (see
/// [`exp_relative_to_max`]): the fast lane of the recursions.
///
/// `sum` is a sum of two or three entries of `scaled`, and `cells` says which
/// ones. Normally the answer is just `row_max + ln(sum)`: one `ln` instead of
/// a [`log_add`] per term. That is exact unless the cells involved are so far
/// below the row's maximum (a factor of 10^200 or more) that their rescaled
/// values are running out of floating-point range. For those cells, and for
/// cells nothing can reach (`sum == 0`), we take the careful lane instead:
/// `log_add` on the logarithms themselves, which cannot underflow.
fn log_of_sum(sum: f64, row_max: f64, row: &[f64], cells: &[usize]) -> f64 {
    if sum > SAFE_SUM {
        row_max + sum.ln()
    } else {
        cells.iter().fold(LOG_ZERO, |total, &s| log_add(total, row[s]))
    }
}

/// For every position of the extended target: may a path skip into it from
/// two positions back? (See [`can_skip_into`]; worked out once, used in every
/// frame.)
fn skip_table(extended: &[usize]) -> Vec<bool> {
    (0..extended.len()).map(|s| can_skip_into(extended, s)).collect()
}

/// The forward recursion, in double precision. Returns a flat `T x S` buffer.
///
/// `alpha[t][s]` = log of the total probability of all the ways to emit the
/// first `t + 1` frames and be at position `s` of the extended target at
/// frame `t`.
///
/// ```text
/// alpha[0][0] = lp[0][blank]
/// alpha[0][1] = lp[0][first label]          (everything else is impossible)
///
/// alpha[t][s] = lp[t][ext[s]] + ln( e^alpha[t-1][s]          stay
///                                 + e^alpha[t-1][s-1]        advance
///                                 + e^alpha[t-1][s-2] )      skip a blank, if allowed
/// ```
fn forward(lp: &[f64], classes: usize, extended: &[usize]) -> Vec<f64> {
    let s_len = extended.len();
    let t_len = lp.len() / classes;
    let mut alpha = vec![LOG_ZERO; t_len * s_len];
    if t_len == 0 {
        return alpha;
    }
    // Frame 0: a path starts on the leading blank or on the first label.
    alpha[0] = lp[BLANK];
    if s_len > 1 {
        alpha[1] = lp[extended[1]];
    }
    let skip = skip_table(extended);
    let mut scaled = vec![0.0; s_len];
    for t in 1..t_len {
        // Split the buffer so we can read row t-1 while writing row t.
        let (done, rest) = alpha.split_at_mut(t * s_len);
        let previous = &done[(t - 1) * s_len..];
        let current = &mut rest[..s_len];
        let frame = &lp[t * classes..(t + 1) * classes];
        let row_max = exp_relative_to_max(previous, &mut scaled);
        if row_max == LOG_ZERO {
            break; // nothing is reachable any more; the rest stays at -inf
        }
        for s in 0..s_len {
            // Add up the cells of the previous frame a path can arrive from.
            let arriving = if skip[s] {
                // stay, advance, or skip a blank
                log_of_sum(scaled[s] + scaled[s - 1] + scaled[s - 2], row_max, previous, &[s, s - 1, s - 2])
            } else if s >= 1 {
                // stay or advance
                log_of_sum(scaled[s] + scaled[s - 1], row_max, previous, &[s, s - 1])
            } else {
                previous[0] // the leading blank can only stay
            };
            current[s] = arriving + frame[extended[s]];
        }
    }
    alpha
}

/// The backward recursion, in double precision. Returns a flat `T x S` buffer.
///
/// `beta[t][s]` = log of the total probability of all the ways to emit frames
/// `t + 1 ..` and finish properly, given that we are at position `s` at
/// frame `t`. Note what is *not* included: the emission at frame `t` itself.
/// That is already inside `alpha[t][s]`, so `alpha[t][s] + beta[t][s]` counts
/// every frame exactly once.
///
/// ```text
/// beta[T-1][S-1] = beta[T-1][S-2] = ln 1 = 0     (the two legal end positions)
///
/// beta[t][s] = ln( e^(lp[t+1][ext[s]]   + beta[t+1][s])         stay
///                + e^(lp[t+1][ext[s+1]] + beta[t+1][s+1])       advance
///                + e^(lp[t+1][ext[s+2]] + beta[t+1][s+2]) )     skip a blank, if allowed
/// ```
///
/// It is the forward recursion run from the other end of the lattice.
fn backward(lp: &[f64], classes: usize, extended: &[usize]) -> Vec<f64> {
    let s_len = extended.len();
    let t_len = lp.len() / classes;
    let mut beta = vec![LOG_ZERO; t_len * s_len];
    if t_len == 0 {
        return beta;
    }
    // Last frame: a path may end on the trailing blank or on the last label.
    let last_row = (t_len - 1) * s_len;
    beta[last_row + s_len - 1] = 0.0;
    if s_len > 1 {
        beta[last_row + s_len - 2] = 0.0;
    }
    let skip = skip_table(extended);
    let mut onward = vec![LOG_ZERO; s_len];
    let mut scaled = vec![0.0; s_len];
    for t in (0..t_len - 1).rev() {
        // Split the buffer so we can read row t+1 while writing row t.
        let (head, done) = beta.split_at_mut((t + 1) * s_len);
        let current = &mut head[t * s_len..];
        let next = &done[..s_len];
        let next_frame = &lp[(t + 1) * classes..(t + 2) * classes];
        // onward[s] = "emit ext[s] at frame t+1, then finish from there".
        for s in 0..s_len {
            onward[s] = next_frame[extended[s]] + next[s];
        }
        let row_max = exp_relative_to_max(&onward, &mut scaled);
        if row_max == LOG_ZERO {
            break; // no way to finish from here on back; the rest stays at -inf
        }
        for s in 0..s_len {
            // Add up the cells of the next frame a path can leave to.
            current[s] = if s + 2 < s_len && skip[s + 2] {
                // stay, advance, or skip a blank
                log_of_sum(scaled[s] + scaled[s + 1] + scaled[s + 2], row_max, &onward, &[s, s + 1, s + 2])
            } else if s + 1 < s_len {
                // stay or advance
                log_of_sum(scaled[s] + scaled[s + 1], row_max, &onward, &[s, s + 1])
            } else {
                onward[s] // the trailing blank can only stay
            };
        }
    }
    beta
}

/// `ln p(target)` read off a forward lattice: the paths that end at frame
/// `T - 1` on the trailing blank, plus those that end on the last label.
fn total_log_prob(alpha: &[f64], t_len: usize, s_len: usize) -> f64 {
    if t_len == 0 {
        // No frames: the only thing we can have "said" is nothing.
        return if s_len == 1 { 0.0 } else { LOG_ZERO };
    }
    let last_row = &alpha[(t_len - 1) * s_len..];
    if s_len == 1 {
        last_row[0]
    } else {
        log_add(last_row[s_len - 1], last_row[s_len - 2])
    }
}

/// The forward (alpha) lattice for `target`, given `T x C` log-probabilities
/// (rows already log-softmaxed, see [`log_softmax`]).
///
/// Returns a `T x S` matrix, `S = 2 * target.len() + 1`: row `t` is a frame,
/// column `s` a position in the extended target. Element `(t, s)` is the log
/// of the summed probability of every partial alignment of frames `0..=t`
/// that sits at position `s` at frame `t`. Unreachable cells are `-inf`.
pub fn forward_log(log_probs: &Mat, target: &[usize]) -> Mat {
    check_target(target, log_probs.cols);
    let extended = extend_with_blanks(target);
    let alpha = forward(&widen(log_probs), log_probs.cols, &extended);
    narrow(log_probs.rows, extended.len(), &alpha)
}

/// The backward (beta) lattice, same shape and orientation as
/// [`forward_log`]. Element `(t, s)` is the log of the summed probability of
/// every way to complete the target over frames `t + 1 ..`, starting from
/// position `s` at frame `t`. The emission at frame `t` is *not* included, so
/// `forward_log + backward_log`, cell by cell, is the log-probability of all
/// complete alignments that pass through that cell.
pub fn backward_log(log_probs: &Mat, target: &[usize]) -> Mat {
    check_target(target, log_probs.cols);
    let extended = extend_with_blanks(target);
    let beta = backward(&widen(log_probs), log_probs.cols, &extended);
    narrow(log_probs.rows, extended.len(), &beta)
}

/// `ln p(target | log_probs)`: the log of the summed probability of every
/// alignment that collapses to `target`. `-inf` if there is no such alignment.
pub fn log_likelihood(log_probs: &Mat, target: &[usize]) -> f32 {
    check_target(target, log_probs.cols);
    let extended = extend_with_blanks(target);
    let alpha = forward(&widen(log_probs), log_probs.cols, &extended);
    total_log_prob(&alpha, log_probs.rows, extended.len()) as f32
}

/// The occupancy (gamma) lattice: where the probability mass flows.
///
/// Element `(t, s)` is the probability that an alignment of `target` is at
/// extended-target position `s` at frame `t`, given that the alignment is one
/// of the valid ones:
///
/// ```text
/// gamma[t][s] = exp( alpha[t][s] + beta[t][s] - ln p(target) )
/// ```
///
/// Every valid alignment is in exactly one position at each frame, so each
/// row sums to 1. Same shape and orientation as [`forward_log`], but these
/// are plain probabilities, not logarithms. If the target is impossible the
/// lattice is all zeros.
pub fn occupancy(log_probs: &Mat, target: &[usize]) -> Mat {
    check_target(target, log_probs.cols);
    let extended = extend_with_blanks(target);
    let lp = widen(log_probs);
    let alpha = forward(&lp, log_probs.cols, &extended);
    let beta = backward(&lp, log_probs.cols, &extended);
    let total = total_log_prob(&alpha, log_probs.rows, extended.len());
    let mut gamma = Mat::zeros(log_probs.rows, extended.len());
    if total == LOG_ZERO {
        return gamma;
    }
    for (g, (a, b)) in gamma.data.iter_mut().zip(alpha.iter().zip(&beta)) {
        *g = (a + b - total).exp() as f32;
    }
    gamma
}

/// The CTC loss in double precision, without the gradient:
/// `-ln p(target | logits)`.
///
/// Same number as the first element of [`ctc_loss`] before it is rounded to
/// `f32`. A numerical gradient check subtracts two nearly equal losses, and
/// the extra digits make that subtraction trustworthy.
pub fn ctc_loss_f64(logits: &Mat, target: &[usize]) -> f64 {
    check_target(target, logits.cols);
    let extended = extend_with_blanks(target);
    let alpha = forward(&log_softmax_f64(logits), logits.cols, &extended);
    -total_log_prob(&alpha, logits.rows, extended.len())
}

/// The CTC loss and its gradient.
///
/// `logits` is `T x C`, straight from the network (this function applies
/// log-softmax itself). `target` holds class ids in `1..C`; blank is class 0.
/// Returns `(loss, gradient)` where `loss = -ln p(target | logits)` and
/// `gradient` is `T x C`, the derivative of the loss with respect to every
/// logit.
///
/// **Impossible targets.** A target of length L needs at least L frames, plus
/// one more for every pair of equal neighbouring labels (they need a blank
/// between them). If T is too small no alignment exists, `p(target) = 0`, and
/// the loss is `+inf`. In that case the gradient is all zeros; a training
/// loop should check `loss.is_finite()` and skip the example.
///
/// # Where the gradient comes from
///
/// Write `y[t][k]` for the softmax probability of class `k` at frame `t`, and
/// `p` for `p(target)`. Fix a frame `t` and a class `k`. For any position `s`
/// whose symbol is `k`, the paths through cell `(t, s)` have total probability
/// `exp(alpha[t][s] + beta[t][s])`, and every one of them contains the factor
/// `y[t][k]` exactly once. A product that contains `y` once has derivative
/// `product / y` with respect to `y`, so
///
/// ```text
/// d p / d y[t][k] = (1 / y[t][k]) * sum over s with ext[s] == k of exp(alpha[t][s] + beta[t][s])
/// ```
///
/// With `loss = -ln p`, and `d ln y = dy / y`, the `1 / y` cancels:
///
/// ```text
/// d loss / d ln y[t][k] = - sum over s with ext[s] == k of gamma[t][s]  =  -occ[t][k]
/// ```
///
/// where `gamma` is the [`occupancy`] lattice and `occ[t][k]` is the
/// probability that a valid alignment emits class `k` at frame `t`. Finally
/// the logits `z` feed the log-softmax, `ln y[j] = z[j] - ln(sum of e^z)`,
/// whose derivative is `d ln y[j] / d z[k] = (1 if j == k else 0) - y[k]`.
/// Chain the two:
///
/// ```text
/// d loss / d z[t][k] = sum over j of -occ[t][j] * ((j == k) - y[t][k])
///                    = -occ[t][k] + y[t][k] * (sum over j of occ[t][j])
///                    = y[t][k] - occ[t][k]            (the occupancies of a frame sum to 1)
/// ```
///
/// **The gradient is `softmax - occupancy`**: what the network said, minus
/// what the valid alignments wanted it to say. It is the same shape as the
/// cross-entropy gradient `softmax - one_hot` from chapter 6, with the hard
/// one-hot target replaced by a soft one that CTC worked out for itself.
pub fn ctc_loss(logits: &Mat, target: &[usize]) -> (f32, Mat) {
    check_target(target, logits.cols);
    let (t_len, classes) = (logits.rows, logits.cols);
    let extended = extend_with_blanks(target);
    let s_len = extended.len();

    let lp = log_softmax_f64(logits);
    let alpha = forward(&lp, classes, &extended);
    let total = total_log_prob(&alpha, t_len, s_len);
    let mut gradient = Mat::zeros(t_len, classes);
    if total == LOG_ZERO {
        return (f32::INFINITY, gradient);
    }
    let beta = backward(&lp, classes, &extended);

    for t in 0..t_len {
        let row = gradient.row_mut(t);
        // Start from the softmax probabilities y[t][k] ...
        for (g, &log_y) in row.iter_mut().zip(&lp[t * classes..(t + 1) * classes]) {
            *g = (log_y as f32).exp();
        }
        // ... and subtract the occupancy of every lattice cell from the
        // class that cell emits. (The gradient is f32, so f32's exp will do.)
        for s in 0..s_len {
            let log_gamma = alpha[t * s_len + s] + beta[t * s_len + s] - total;
            row[extended[s]] -= (log_gamma as f32).exp();
        }
    }
    // A probability is at most 1, so the loss is at least 0; rounding can
    // leave a perfectly confident, correct input a hair below that.
    ((-total).max(0.0) as f32, gradient)
}

/// The single most probable alignment of a target to the frames.
#[derive(Clone, Debug, PartialEq)]
pub struct Alignment {
    /// For each frame, the position in the extended target (see
    /// [`extend_with_blanks`]) the best path is at. Even positions are blanks;
    /// odd position `s` is target label number `(s - 1) / 2`.
    pub positions: Vec<usize>,
    /// For each frame, the class the best path emits there (0 for blank).
    /// `collapse(&classes)` gives back the target.
    pub classes: Vec<usize>,
    /// The log-probability of this one path. Never more than
    /// [`log_likelihood`], which sums over all paths.
    pub log_prob: f32,
}

/// Forced alignment: the most probable single path that collapses to
/// `target` (the Viterbi path), or `None` if no path does.
///
/// This is the forward recursion with "sum over the ways to get here"
/// replaced by "best of the ways to get here", plus a note in every cell of
/// which predecessor was best, so that we can walk back from the end, exactly
/// like the DTW backtrace.
pub fn best_alignment(log_probs: &Mat, target: &[usize]) -> Option<Alignment> {
    check_target(target, log_probs.cols);
    let extended = extend_with_blanks(target);
    let (t_len, s_len) = (log_probs.rows, extended.len());
    if t_len == 0 {
        let nothing = Alignment { positions: Vec::new(), classes: Vec::new(), log_prob: 0.0 };
        return if target.is_empty() { Some(nothing) } else { None };
    }

    // best[t][s]: log-probability of the best partial path ending at (t, s).
    // came_from[t][s]: the position that path was at in frame t - 1.
    let mut best = vec![LOG_ZERO; t_len * s_len];
    let mut came_from = vec![0usize; t_len * s_len];
    best[0] = log_probs[(0, BLANK)] as f64;
    if s_len > 1 {
        best[1] = log_probs[(0, extended[1])] as f64;
    }
    for t in 1..t_len {
        for s in 0..s_len {
            let mut from = s; // stay
            if s >= 1 && best[(t - 1) * s_len + s - 1] > best[(t - 1) * s_len + from] {
                from = s - 1; // advance
            }
            if can_skip_into(&extended, s) && best[(t - 1) * s_len + s - 2] > best[(t - 1) * s_len + from] {
                from = s - 2; // skip a blank
            }
            came_from[t * s_len + s] = from;
            best[t * s_len + s] = best[(t - 1) * s_len + from] + log_probs[(t, extended[s])] as f64;
        }
    }

    // The path ends on the trailing blank or on the last label, whichever is better.
    let last_row = (t_len - 1) * s_len;
    let mut s = s_len - 1;
    if s_len > 1 && best[last_row + s_len - 2] > best[last_row + s] {
        s = s_len - 2;
    }
    let log_prob = best[last_row + s];
    if log_prob == LOG_ZERO {
        return None;
    }
    let mut positions = vec![0; t_len];
    for t in (0..t_len).rev() {
        positions[t] = s;
        s = came_from[t * s_len + s];
    }
    let classes = positions.iter().map(|&s| extended[s]).collect();
    Some(Alignment { positions, classes, log_prob: log_prob as f32 })
}

/// Call `visit` with every possible alignment of `frames` frames over
/// `classes` classes: all `classes^frames` of them, counting like an odometer
/// from `[0, 0, .., 0]` to `[C-1, C-1, .., C-1]`.
///
/// This is the brute force that the forward algorithm exists to avoid. It is
/// here for the book's toy examples and for the tests, which check the
/// lattice against it; do not call it with more than a handful of frames.
pub fn for_each_alignment(frames: usize, classes: usize, mut visit: impl FnMut(&[usize])) {
    if classes == 0 {
        return;
    }
    let mut path = vec![0; frames];
    loop {
        visit(&path);
        // Add one to the last digit, carrying leftwards.
        let mut digit = frames;
        loop {
            if digit == 0 {
                return; // carried off the left end: every path has been visited
            }
            digit -= 1;
            path[digit] += 1;
            if path[digit] < classes {
                break;
            }
            path[digit] = 0;
        }
    }
}

/// The probability of one alignment: the product over frames of the
/// probability of the class it picks in that frame (computed as the exp of a
/// sum of logs).
pub fn alignment_log_prob(log_probs: &Mat, path: &[usize]) -> f64 {
    assert_eq!(path.len(), log_probs.rows, "an alignment has one class per frame");
    path.iter().enumerate().map(|(t, &class)| log_probs[(t, class)] as f64).sum()
}

/// Every alignment that collapses to `target`, with its log-probability, by
/// brute force (see [`for_each_alignment`]). Small inputs only.
pub fn alignments_of(log_probs: &Mat, target: &[usize]) -> Vec<(Vec<usize>, f64)> {
    let mut found = Vec::new();
    for_each_alignment(log_probs.rows, log_probs.cols, |path| {
        if collapse(path) == target {
            found.push((path.to_vec(), alignment_log_prob(log_probs, path)));
        }
    });
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn random_logits(rng: &mut Rng, frames: usize, classes: usize, scale: f32) -> Mat {
        Mat::from_fn(frames, classes, |_, _| scale * rng.normal())
    }

    /// p(target) by summing over every alignment: the definition of CTC.
    fn brute_force_probability(log_probs: &Mat, target: &[usize]) -> f64 {
        alignments_of(log_probs, target).iter().map(|(_, lp)| lp.exp()).sum()
    }

    #[test]
    fn collapse_merges_repeats_then_drops_blanks() {
        assert_eq!(collapse(&[1, 1, 0, 1, 2, 2, 0]), vec![1, 1, 2]);
        assert_eq!(collapse(&[0, 0, 0]), Vec::<usize>::new());
        assert_eq!(collapse(&[]), Vec::<usize>::new());
        assert_eq!(collapse(&[3, 3, 3]), vec![3]);
        assert_eq!(collapse(&[3, 0, 3]), vec![3, 3]);
    }

    #[test]
    fn extended_target_interleaves_blanks() {
        assert_eq!(extend_with_blanks(&[3, 1, 20]), vec![0, 3, 0, 1, 0, 20, 0]);
        assert_eq!(extend_with_blanks(&[]), vec![0]);
    }

    #[test]
    fn log_add_adds_probabilities() {
        assert!((log_add(0.3f64.ln(), 0.2f64.ln()) - 0.5f64.ln()).abs() < 1e-12);
        assert_eq!(log_add(LOG_ZERO, -1.5), -1.5);
        assert_eq!(log_add(-1.5, LOG_ZERO), -1.5);
        assert_eq!(log_add(LOG_ZERO, LOG_ZERO), LOG_ZERO);
        // Far beyond what exp() could represent directly.
        assert!((log_add(-2000.0, -2000.0) - (-2000.0 + 2f64.ln())).abs() < 1e-9);
    }

    #[test]
    fn log_softmax_rows_are_distributions() {
        let logits = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 1000.0, 1000.0, -1000.0]);
        let lp = log_softmax(&logits);
        for t in 0..2 {
            let sum: f32 = lp.row(t).iter().map(|v| v.exp()).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
        // e^3 / (e^1 + e^2 + e^3) = 0.66524...
        assert!((lp[(0, 2)].exp() - 0.665_240_96).abs() < 1e-6);
        // Huge logits must not overflow: two equal winners get half each.
        assert!((lp[(1, 0)] - 0.5f32.ln()).abs() < 1e-6);
    }

    /// Two frames, classes {blank, a}, target "a". Blank has probability 0.6
    /// and `a` 0.4 in both frames. The alignments of "a" are
    ///   a _ : 0.4 * 0.6 = 0.24
    ///   _ a : 0.6 * 0.4 = 0.24
    ///   a a : 0.4 * 0.4 = 0.16
    /// so p("a") = 0.64, and p("") = p(_ _) = 0.36.
    #[test]
    fn hand_computed_two_frame_example() {
        let lp = Mat::from_vec(2, 2, vec![0.6f32.ln(), 0.4f32.ln(), 0.6f32.ln(), 0.4f32.ln()]);
        assert!((log_likelihood(&lp, &[1]).exp() - 0.64).abs() < 1e-6);
        assert!((log_likelihood(&lp, &[]).exp() - 0.36).abs() < 1e-6);
        // The alpha lattice for "a" (extended: _ a _), as probabilities:
        //   frame 0:  0.6   0.4   0
        //   frame 1:  0.36  0.40  0.24      (0.4 = (0.6 + 0.4) * 0.4;  0.24 = 0.4 * 0.6)
        let alpha = forward_log(&lp, &[1]);
        let expected = [0.6, 0.4, 0.0, 0.36, 0.4, 0.24];
        for (a, e) in alpha.data.iter().zip(expected) {
            assert!((a.exp() - e).abs() < 1e-6, "{} vs {e}", a.exp());
        }
        // Occupancy at frame 0: paths starting with blank (_ a: 0.24) against
        // paths starting with a (a _, a a: 0.40), out of 0.64.
        let gamma = occupancy(&lp, &[1]);
        assert!((gamma[(0, 0)] - 0.24 / 0.64).abs() < 1e-6);
        assert!((gamma[(0, 1)] - 0.40 / 0.64).abs() < 1e-6);
    }

    /// (a) The forward algorithm must agree with summing over every alignment.
    #[test]
    fn loss_matches_brute_force_over_all_alignments() {
        let mut rng = Rng::new(21);
        let targets: [&[usize]; 9] = [&[], &[1], &[2], &[1, 2], &[1, 1], &[2, 1, 2], &[1, 1, 1], &[3, 3, 1], &[1, 2, 3]];
        for (frames, classes) in [(1, 3), (3, 4), (5, 4), (6, 4), (7, 3)] {
            let logits = random_logits(&mut rng, frames, classes, 1.5);
            let lp = log_softmax(&logits);
            for target in targets {
                if target.iter().any(|&l| l >= classes) {
                    continue;
                }
                let brute = brute_force_probability(&lp, target);
                let (loss, _) = ctc_loss(&logits, target);
                if brute == 0.0 {
                    assert!(loss.is_infinite(), "T={frames} target={target:?}");
                } else {
                    let expected = -brute.ln();
                    assert!((loss as f64 - expected).abs() < 1e-4, "T={frames} target={target:?}: {loss} vs {expected}");
                    assert!((ctc_loss_f64(&logits, target) - expected).abs() < 1e-5);
                }
            }
        }
    }

    /// Summed over every possible labelling, the probabilities must come to 1:
    /// each alignment collapses to exactly one labelling.
    #[test]
    fn probabilities_of_all_labellings_sum_to_one() {
        let mut rng = Rng::new(22);
        let logits = random_logits(&mut rng, 4, 3, 1.0);
        let lp = log_softmax(&logits);
        let mut labellings = std::collections::BTreeSet::new();
        for_each_alignment(4, 3, |path| {
            labellings.insert(collapse(path));
        });
        let total: f64 = labellings.iter().map(|l| (log_likelihood(&lp, l) as f64).exp()).sum();
        assert!((total - 1.0).abs() < 1e-5, "{total}");
    }

    /// Central finite differences on every logit, with the loss in f64.
    fn numerical_gradient(logits: &Mat, target: &[usize]) -> Vec<f64> {
        let eps = 1e-3f32;
        let mut numeric = Vec::with_capacity(logits.data.len());
        for i in 0..logits.data.len() {
            let (mut plus, mut minus) = (logits.clone(), logits.clone());
            plus.data[i] += eps;
            minus.data[i] -= eps;
            // Divide by the step actually taken: `x + eps` is rounded to the
            // nearest f32, so the real step is not exactly 2 * eps.
            let step = plus.data[i] as f64 - minus.data[i] as f64;
            numeric.push((ctc_loss_f64(&plus, target) - ctc_loss_f64(&minus, target)) / step);
        }
        numeric
    }

    /// (b) The analytic gradient must match the numerical one.
    #[test]
    fn gradient_matches_finite_differences() {
        let mut rng = Rng::new(23);
        let cases: [(usize, usize, &[usize]); 6] =
            [(4, 3, &[1]), (6, 4, &[1, 2]), (7, 4, &[2, 2, 3]), (5, 3, &[]), (12, 6, &[3, 1, 1, 5, 2]), (30, 28, &[4, 2, 21, 21, 1, 9])];
        for (frames, classes, target) in cases {
            let logits = random_logits(&mut rng, frames, classes, 2.0);
            let (loss, gradient) = ctc_loss(&logits, target);
            assert!(loss.is_finite());
            let numeric = numerical_gradient(&logits, target);
            for (i, (&analytic, &numeric)) in gradient.data.iter().zip(&numeric).enumerate() {
                assert!((analytic as f64 - numeric).abs() < 2e-6, "T={frames} target={target:?} element {i}: analytic {analytic} vs numeric {numeric}");
            }
            // Shifting all logits of a frame by a constant does not change the
            // softmax, so every row of the gradient must sum to zero.
            for t in 0..frames {
                let row_sum: f32 = gradient.row(t).iter().sum();
                assert!(row_sum.abs() < 1e-5, "row {t} sums to {row_sum}");
            }
        }
    }

    /// (c) At every frame, summing alpha * beta across positions gives the
    /// same total: every valid path is somewhere at every frame.
    #[test]
    fn alpha_and_beta_agree_on_the_total_at_every_frame() {
        let mut rng = Rng::new(24);
        let cases: [(usize, usize, &[usize]); 4] = [(9, 5, &[1, 2, 2, 4]), (6, 3, &[]), (40, 28, &[5, 5, 9, 2, 1, 7]), (3, 3, &[1, 1])];
        for (frames, classes, target) in cases {
            let lp = log_softmax(&random_logits(&mut rng, frames, classes, 2.0));
            let (alpha, beta) = (forward_log(&lp, target), backward_log(&lp, target));
            let total = log_likelihood(&lp, target) as f64;
            assert!(total.is_finite());
            for t in 0..frames {
                let mut through_frame = LOG_ZERO;
                for s in 0..alpha.cols {
                    through_frame = log_add(through_frame, alpha[(t, s)] as f64 + beta[(t, s)] as f64);
                }
                assert!((through_frame - total).abs() < 1e-4, "frame {t}: {through_frame} vs {total}");
            }
            // The total can also be read off the first frame of beta alone.
            let mut from_beta = beta[(0, 0)] as f64 + lp[(0, BLANK)] as f64;
            if !target.is_empty() {
                from_beta = log_add(from_beta, beta[(0, 1)] as f64 + lp[(0, target[0])] as f64);
            }
            assert!((from_beta - total).abs() < 1e-4);
            // And each row of the occupancy lattice is a distribution.
            let gamma = occupancy(&lp, target);
            for t in 0..frames {
                let sum: f32 = gamma.row(t).iter().sum();
                assert!((sum - 1.0).abs() < 1e-4, "gamma row {t} sums to {sum}");
            }
        }
    }

    /// (d) Too few frames for the target: infinite loss, zero gradient.
    #[test]
    fn impossible_targets_give_infinite_loss_and_zero_gradient() {
        let mut rng = Rng::new(25);
        let logits = random_logits(&mut rng, 3, 4, 1.0);
        // Four labels cannot fit in three frames.
        let (loss, gradient) = ctc_loss(&logits, &[1, 2, 3, 1]);
        assert_eq!(loss, f32::INFINITY);
        assert!(gradient.data.iter().all(|&g| g == 0.0));
        assert_eq!((gradient.rows, gradient.cols), (3, 4));
        // "1 1" needs a blank between the two 1s: three frames, not two.
        let two_frames = random_logits(&mut rng, 2, 4, 1.0);
        assert_eq!(ctc_loss(&two_frames, &[1, 1]).0, f32::INFINITY);
        assert!(ctc_loss(&logits, &[1, 1]).0.is_finite());
        assert!(ctc_loss(&two_frames, &[1, 2]).0.is_finite());
        // The other views of an impossible target.
        let lp = log_softmax(&logits);
        assert_eq!(log_likelihood(&lp, &[1, 2, 3, 1]), f32::NEG_INFINITY);
        assert!(occupancy(&lp, &[1, 2, 3, 1]).data.iter().all(|&g| g == 0.0));
        assert!(best_alignment(&lp, &[1, 2, 3, 1]).is_none());
        // No frames at all: only the empty target is possible.
        let no_frames = Mat::zeros(0, 4);
        assert_eq!(ctc_loss(&no_frames, &[]).0, 0.0);
        assert_eq!(ctc_loss(&no_frames, &[1]).0, f32::INFINITY);
    }

    #[test]
    fn empty_target_is_the_all_blank_path() {
        let mut rng = Rng::new(26);
        let logits = random_logits(&mut rng, 8, 5, 1.0);
        let lp = log_softmax(&logits);
        let all_blank: f32 = (0..8).map(|t| lp[(t, BLANK)]).sum();
        let (loss, gradient) = ctc_loss(&logits, &[]);
        assert!((loss + all_blank).abs() < 1e-5);
        // Gradient is softmax - one_hot(blank): plain cross-entropy toward blank.
        for t in 0..8 {
            for k in 0..5 {
                let expected = lp[(t, k)].exp() - if k == BLANK { 1.0 } else { 0.0 };
                assert!((gradient[(t, k)] - expected).abs() < 1e-6);
            }
        }
    }

    /// 500 frames: in plain probabilities the path products underflow, in
    /// log space nothing special happens.
    #[test]
    fn long_inputs_stay_finite() {
        let mut rng = Rng::new(27);
        let logits = random_logits(&mut rng, 500, 28, 3.0);
        let target: Vec<usize> = (0..60).map(|i| 1 + (i * 7) % 27).collect();
        let (loss, gradient) = ctc_loss(&logits, &target);
        // p(target) is below e^-745, the smallest positive f64: as a plain
        // probability it would have been rounded to zero.
        assert!(loss.is_finite() && loss > 745.0, "loss {loss}");
        assert!(gradient.data.iter().all(|g| g.is_finite()));
        for t in 0..500 {
            let row_sum: f32 = gradient.row(t).iter().sum();
            assert!(row_sum.abs() < 1e-4);
        }
        // Confident, correct logits give a loss near zero rather than NaN.
        let mut confident = Mat::from_fn(500, 28, |_, _| -50.0);
        for t in 0..500 {
            let class = if t % 8 == 0 && t / 8 < target.len() { target[t / 8] } else { BLANK };
            confident[(t, class)] = 50.0;
        }
        let (loss, _) = ctc_loss(&confident, &target);
        assert!((0.0..1e-3).contains(&loss), "loss {loss}");
    }

    /// The forward recursion written the slow, obvious way: one `log_add`
    /// per term, never leaving log space. The real one must agree with it.
    fn forward_reference(lp: &[f64], classes: usize, extended: &[usize]) -> Vec<f64> {
        let (s_len, t_len) = (extended.len(), lp.len() / classes);
        let mut alpha = vec![LOG_ZERO; t_len * s_len];
        alpha[0] = lp[BLANK];
        if s_len > 1 {
            alpha[1] = lp[extended[1]];
        }
        for t in 1..t_len {
            for s in 0..s_len {
                let mut arriving = alpha[(t - 1) * s_len + s];
                if s >= 1 {
                    arriving = log_add(arriving, alpha[(t - 1) * s_len + s - 1]);
                }
                if can_skip_into(extended, s) {
                    arriving = log_add(arriving, alpha[(t - 1) * s_len + s - 2]);
                }
                alpha[t * s_len + s] = arriving + lp[t * classes + extended[s]];
            }
        }
        alpha
    }

    fn assert_lattices_agree(fast: &[f64], reference: &[f64]) {
        for (i, (&x, &y)) in fast.iter().zip(reference).enumerate() {
            let agree = if y == LOG_ZERO { x == LOG_ZERO } else { (x - y).abs() <= 1e-9 * (1.0 + y.abs()) };
            assert!(agree, "cell {i}: {x} vs {y}");
        }
    }

    /// The per-row rescaling must give the same lattice as plain `log_add`,
    /// including when some cells sit absurdly far below the row's maximum.
    #[test]
    fn rescaled_rows_agree_with_plain_log_add() {
        let mut rng = Rng::new(29);
        let target = [1, 2, 2, 1, 2, 1, 1, 2];
        let extended = extend_with_blanks(&target);
        // An ordinary input.
        let lp = log_softmax_f64(&random_logits(&mut rng, 40, 3, 2.0));
        assert_lattices_agree(&forward(&lp, 3, &extended), &forward_reference(&lp, 3, &extended));

        // A network that is certain of "blank" (by a margin of 100 per frame,
        // a factor of 10^43) for the first 30 frames, then just as certain of
        // each label in turn. After 30 frames the cells that have already
        // emitted labels are up to 10^-347 times less likely than the cell
        // that has not: too small to survive rescaling by the row maximum.
        let mut logits = Mat::from_fn(50, 3, |_, _| -50.0);
        for t in 0..50 {
            let class = if t >= 30 && t % 2 == 0 && (t - 30) / 2 < target.len() { target[(t - 30) / 2] } else { BLANK };
            logits[(t, class)] = 50.0;
        }
        let lp = log_softmax_f64(&logits);
        let (alpha, reference) = (forward(&lp, 3, &extended), forward_reference(&lp, 3, &extended));
        assert!(reference[29 * extended.len() + 15] < -750.0 && reference[29 * extended.len() + 15].is_finite());
        assert_lattices_agree(&alpha, &reference);
        // The backward pass is the forward pass on the time-reversed problem,
        // so `alpha + beta` summed over a frame must still give the total.
        let beta = backward(&lp, 3, &extended);
        let total = total_log_prob(&alpha, 50, extended.len());
        for t in 0..50 {
            let row = t * extended.len();
            let through = (0..extended.len()).fold(LOG_ZERO, |sum, s| log_add(sum, alpha[row + s] + beta[row + s]));
            assert!((through - total).abs() < 1e-9, "frame {t}: {through} vs {total}");
        }
        // This network is right, so the loss is tiny; shifting the target by
        // one label makes it confidently wrong: a huge but finite loss.
        assert!(ctc_loss(&logits, &target).0 < 1e-3);
        let wrong = [2, 1, 1, 2, 1, 2, 2, 1];
        let (loss, gradient) = ctc_loss(&logits, &wrong);
        assert!(loss.is_finite() && loss > 300.0, "loss {loss}");
        assert!(gradient.data.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn best_alignment_is_the_most_probable_path() {
        let mut rng = Rng::new(28);
        let targets: [&[usize]; 5] = [&[], &[1], &[1, 2], &[2, 2], &[1, 2, 1]];
        for target in targets {
            let lp = log_softmax(&random_logits(&mut rng, 6, 3, 1.5));
            let alignment = best_alignment(&lp, target).unwrap();
            assert_eq!(collapse(&alignment.classes), target);
            assert_eq!(alignment.classes, alignment.positions.iter().map(|&s| extend_with_blanks(target)[s]).collect::<Vec<_>>());
            // Compare with the best of all alignments found by brute force.
            let best = alignments_of(&lp, target).into_iter().map(|(_, lp)| lp).fold(LOG_ZERO, f64::max);
            assert!((alignment.log_prob as f64 - best).abs() < 1e-5);
            assert!(alignment.log_prob <= log_likelihood(&lp, target) + 1e-6);
        }
    }

    #[test]
    fn odometer_visits_every_alignment_once() {
        let mut seen = std::collections::BTreeSet::new();
        let mut count = 0;
        for_each_alignment(3, 4, |path| {
            seen.insert(path.to_vec());
            count += 1;
        });
        assert_eq!((count, seen.len()), (64, 64));
        let mut zero_frames = 0;
        for_each_alignment(0, 4, |path| {
            assert!(path.is_empty());
            zero_frames += 1;
        });
        assert_eq!(zero_frames, 1); // the one empty alignment
    }

    #[test]
    #[should_panic(expected = "must not contain the blank")]
    fn blank_in_target_is_a_programmer_error() {
        ctc_loss(&Mat::zeros(4, 3), &[1, 0, 2]);
    }
}
