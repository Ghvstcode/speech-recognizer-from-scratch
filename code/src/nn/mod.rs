//! A small neural-network library with hand-written backpropagation.
//!
//! There is no automatic differentiation here. Every layer has a `forward`
//! function that computes its output and a `backward` function, written by
//! hand, that answers one question: "if the loss changes by this much when my
//! output changes, how much does it change when my input and my weights
//! change?" Chaining those answers from the last layer to the first is all
//! backpropagation is.
//!
//! # The shape of the data
//!
//! Everything flows as a [`Mat`](crate::mat::Mat) with **one row per frame**:
//! a model maps `T x n_features` to *logits* `T' x n_classes`. (For a batch of
//! independent points, as in the chapter 6 toys, "one row per example" works
//! the same way: none of the row-wise layers care what a row means.)
//!
//! # Parameters are separate from per-call state
//!
//! A [`Model`] only holds weights. Running it does not change it:
//!
//! ```text
//! let cache = model.forward(&x);              // &self: remembers what backward needs
//! let (loss, d_logits) = loss_fn(cache.output(), &target);
//! model.backward(&cache, &d_logits, &mut grads);   // &self: adds into `grads`
//! optimizer.step(&mut model, &grads);         // the only place weights change
//! ```
//!
//! Because `forward` and `backward` take `&self`, fourteen threads can each
//! push a different utterance through the *same* model at the same time, each
//! with its own [`Cache`] and its own [`Grads`]. [`batch_gradients`] does
//! exactly that.
//!
//! # Map of the module
//!
//! | file | what is in it |
//! |---|---|
//! | `kernels.rs` | the few inner loops everything else is made of |
//! | `init.rs` | how to pick the starting weights (Xavier, He, orthogonal) |
//! | `linear.rs` | the fully connected layer |
//! | `activation.rs` | ReLU, tanh, sigmoid |
//! | `conv.rs` | 1-D convolution over time |
//! | `gru.rs` | the gated recurrent unit and its bidirectional pair |
//! | `model.rs` | `Layer`, `Model`, `Cache`, `Grads`: chaining layers together |
//! | `loss.rs` | softmax, log-softmax, cross-entropy |
//! | `optim.rs` | SGD, Adam, gradient clipping, a learning-rate schedule |
//! | `batch.rs` | gradients for a whole minibatch, one thread per slice of it |
//! | `gradcheck.rs` | the numerical gradient checker |
//! | `io.rs` | saving and loading a model |

pub mod activation;
pub mod batch;
pub mod conv;
pub mod gradcheck;
pub mod gru;
pub mod init;
pub mod io;
pub mod kernels;
pub mod linear;
pub mod loss;
pub mod model;
pub mod optim;

pub use batch::{batch_gradients, default_threads, BatchOutcome};
pub use conv::Conv1d;
pub use gradcheck::{compare_gradients, gradcheck, numerical_gradient, GradCheckConfig, GradCheckReport, TensorCheck};
pub use gru::{BiGru, Direction, Gru, GruCache};
pub use linear::Linear;
pub use loss::{argmax, cross_entropy, cross_entropy_row, log_softmax, softmax};
pub use model::{Cache, Grads, Layer, Model};
pub use optim::{clip_global_norm, Adam, LrSchedule, Sgd};

#[cfg(test)]
mod tests;
