//! Minibatch gradients on many threads.
//!
//! Training looks at a *minibatch* of examples per step (a few dozen
//! utterances), and averages their gradients. The examples do not interact:
//! each one is its own forward pass, loss and backward pass against the same,
//! unchanging weights. That makes the job trivially parallel, and because a
//! [`Model`] is only ever read during those passes, Rust lets us share one
//! `&Model` across all threads with no locks at all. Each thread owns a
//! private [`Grads`] to accumulate into; at the end we add the threads'
//! results together.

use super::model::{Grads, Model};
use crate::mat::Mat;

/// What [`batch_gradients`] hands back.
#[derive(Clone, Debug)]
pub struct BatchOutcome {
    /// The gradient of the loss, **averaged** over the examples that were
    /// used. All zeros if none were.
    pub grads: Grads,
    /// The loss, averaged over the examples that were used (0 if none were).
    pub loss: f32,
    /// How many examples contributed.
    pub used: usize,
    /// How many examples were left out because their loss, or its gradient,
    /// was not a finite number.
    pub skipped: usize,
}

/// The number of threads worth using on this machine.
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Compute the average loss and gradient over a minibatch, using up to
/// `n_threads` threads.
///
/// * `batch` is a list of `(input, target)` pairs. The target can be anything:
///   a slice of class indices, a character sequence, a single number.
/// * `loss_fn(logits, target)` returns the loss for one example and its
///   gradient with respect to the logits. [`cross_entropy`](super::cross_entropy)
///   and the CTC loss of chapter 8 both already have this shape and can be
///   passed in directly.
///
/// **Skipping bad examples.** If a loss comes back infinite or NaN (CTC
/// returns infinity when a transcript cannot possibly fit into the number of
/// frames available), that example is left out, counted in `skipped`, and
/// the rest of the batch is unaffected. One bad clip must not poison a step.
///
/// The result does not depend on thread timing: thread `k` always handles
/// examples `k`, `k + n_threads`, `k + 2 * n_threads`, ... and the threads'
/// gradients are added in order, so the same batch gives bit-for-bit the same
/// gradient every time (for a given `n_threads`).
pub fn batch_gradients<T, L>(model: &Model, batch: &[(&Mat, &T)], n_threads: usize, loss_fn: L) -> BatchOutcome
where
    T: ?Sized + Sync,
    L: Fn(&Mat, &T) -> (f32, Mat) + Sync,
{
    let n_threads = n_threads.clamp(1, batch.len().max(1));

    let mut totals: Vec<WorkerTotal> = Vec::with_capacity(n_threads);
    if n_threads == 1 {
        totals.push(run_worker(model, batch, 0, 1, &loss_fn));
    } else {
        // A scope guarantees every thread has finished before it returns,
        // which is what allows the threads to borrow `model` and `batch`.
        std::thread::scope(|scope| {
            let loss_fn = &loss_fn;
            let handles: Vec<_> = (0..n_threads).map(|k| scope.spawn(move || run_worker(model, batch, k, n_threads, loss_fn))).collect();
            for handle in handles {
                totals.push(handle.join().expect("a worker thread panicked"));
            }
        });
    }

    // Add up what the threads found.
    let mut totals = totals.into_iter();
    let mut sum = totals.next().expect("there is always at least one worker");
    for other in totals {
        sum.grads.add_assign(&other.grads);
        sum.loss += other.loss;
        sum.used += other.used;
        sum.skipped += other.skipped;
    }
    if sum.used > 0 {
        sum.grads.scale(1.0 / sum.used as f32);
    }
    let mean_loss = if sum.used > 0 { sum.loss / sum.used as f64 } else { 0.0 };
    BatchOutcome { grads: sum.grads, loss: mean_loss as f32, used: sum.used, skipped: sum.skipped }
}

/// One thread's running totals (sums, not yet averages).
struct WorkerTotal {
    grads: Grads,
    loss: f64,
    used: usize,
    skipped: usize,
}

/// Process examples `first`, `first + stride`, `first + 2 * stride`, ...
fn run_worker<T, L>(model: &Model, batch: &[(&Mat, &T)], first: usize, stride: usize, loss_fn: &L) -> WorkerTotal
where
    T: ?Sized,
    L: Fn(&Mat, &T) -> (f32, Mat),
{
    let mut total = WorkerTotal { grads: Grads::zeros_like(model), loss: 0.0, used: 0, skipped: 0 };
    for &(input, target) in batch.iter().skip(first).step_by(stride) {
        let cache = model.forward(input);
        let (loss, d_logits) = loss_fn(cache.output(), target);
        if !loss.is_finite() || d_logits.data.iter().any(|d| !d.is_finite()) {
            total.skipped += 1;
            continue; // before backward, so nothing of it reaches the gradients
        }
        model.backward(&cache, &d_logits, &mut total.grads);
        total.loss += loss as f64;
        total.used += 1;
    }
    total
}
