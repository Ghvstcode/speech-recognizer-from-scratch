//! Chapter 6: what the gradient checker looks like when it catches a bug.
//!
//! We define a toy "layer" right here, a single neuron
//!
//! ```text
//! y    = tanh(w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + b)
//! loss = (y - target)^2 / 2
//! ```
//!
//! and write its backward pass by hand three times: once correctly, once
//! with a flipped sign, and once forgetting the slope of tanh. Then we ask
//! the numerical gradient checker about each.
//!
//! A fourth case shows the checker being wrong instead: a ReLU neuron with a
//! correct backward pass, placed so close to its kink that the nudge steps
//! across it. The checker reports a mismatch, and a smaller nudge makes the
//! false alarm go away.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example ch06_gradcheck              (all four)
//! cargo run --release --example ch06_gradcheck -- sign      (just one: none | sign | slope | kink)
//! ```

use speech::nn::{compare_gradients, numerical_gradient};

const USAGE: &str = "usage: ch06_gradcheck [none|sign|slope|kink]";

/// Which mistake to plant in the backward pass.
#[derive(Clone, Copy, PartialEq)]
enum Bug {
    /// No mistake.
    None,
    /// The weight gradient has the wrong sign.
    Sign,
    /// The slope of tanh, `1 - y^2`, was forgotten.
    Slope,
}

/// What to run: one of the three tanh checks, or the ReLU false alarm.
#[derive(Clone, Copy)]
enum Case {
    Tanh(Bug),
    Kink,
}

/// The input and target of our single training example.
const X: [f32; 3] = [0.5, -1.0, 2.0];
const TARGET: f32 = 0.3;

/// Forward pass. `params` is `[w0, w1, w2, b]`. Returns `(y, loss)`.
fn forward(params: &[f32]) -> (f32, f32) {
    let sum = params[0] * X[0] + params[1] * X[1] + params[2] * X[2] + params[3];
    let y = sum.tanh();
    (y, 0.5 * (y - TARGET) * (y - TARGET))
}

/// Backward pass, by hand. Returns `d loss / d [w0, w1, w2, b]`.
fn backward(params: &[f32], bug: Bug) -> Vec<f32> {
    let (y, _) = forward(params);
    // loss = (y - target)^2 / 2   =>   d loss / d y = y - target
    let d_y = y - TARGET;
    // y = tanh(sum)   =>   d loss / d sum = d_y * (1 - y^2)
    let d_sum = if bug == Bug::Slope { d_y } else { d_y * (1.0 - y * y) };
    // sum = w . x + b   =>   d loss / d w[i] = d_sum * x[i],  d loss / d b = d_sum
    let sign = if bug == Bug::Sign { -1.0 } else { 1.0 };
    vec![sign * d_sum * X[0], sign * d_sum * X[1], sign * d_sum * X[2], d_sum]
}

/// Print the two gradients side by side and return whether they agree
/// (relative error at most 1e-2, the checker's default tolerance).
fn compare(analytic: &[f32], numeric: &[f32]) -> (f64, bool) {
    let verdict = compare_gradients("neuron", analytic, numeric, 1e-2, 1e-4);
    println!("  parameter    backprop   numerical");
    for (i, name) in ["w[0]", "w[1]", "w[2]", "b"].iter().enumerate() {
        println!("  {name:<9} {:>11.6} {:>11.6}", analytic[i], numeric[i]);
    }
    (verdict.rel_error, verdict.pass)
}

fn check(title: &str, bug: Bug) {
    let mut params = vec![0.4f32, -0.2, 0.1, 0.05];
    let analytic = backward(&params, bug);
    // The second opinion: nudge each parameter by +/- 0.01 and watch the loss.
    let numeric = numerical_gradient(&mut params, 1e-2, |p| forward(p).1 as f64);

    println!("{title}\n");
    let (rel_error, pass) = compare(&analytic, &numeric);
    println!("\n  relative error {rel_error:.2e}  =>  {}\n", if pass { "ok, the two agree" } else { "MISMATCH, the backward pass is wrong" });
}

/// The same neuron with ReLU in place of tanh: `y = max(0, w . x + b)`.
/// Returns `(sum, loss)`; `sum` is the value ReLU is applied to.
fn forward_relu(params: &[f32]) -> (f32, f32) {
    let sum = params[0] * X[0] + params[1] * X[1] + params[2] * X[2] + params[3];
    let y = sum.max(0.0);
    (sum, 0.5 * (y - TARGET) * (y - TARGET))
}

/// Its backward pass, with no mistake in it. The slope of ReLU is 1 where
/// its input is positive and 0 where it is negative.
fn backward_relu(params: &[f32]) -> Vec<f32> {
    let (sum, _) = forward_relu(params);
    let y = sum.max(0.0);
    let d_sum = if sum > 0.0 { y - TARGET } else { 0.0 };
    vec![d_sum * X[0], d_sum * X[1], d_sum * X[2], d_sum]
}

/// The false alarm. The bias is chosen so that ReLU's input is +0.004: on
/// the sloped side of the kink, but closer to it than a nudge of 0.01
/// reaches. Each nudged pair of losses then has one foot on the slope and
/// one on the flat part, and their difference is the slope of neither.
fn kink() {
    let mut params = vec![0.4f32, -0.2, 0.1, -0.596];
    let analytic = backward_relu(&params);
    println!("4. No bug, but a false alarm: a ReLU neuron next to its kink\n");
    println!("  the value going into ReLU is {:+.4}; the backward pass is correct\n", forward_relu(&params).0);
    for eps in [1e-2f32, 1e-3] {
        let numeric = numerical_gradient(&mut params, eps, |p| forward_relu(p).1 as f64);
        println!("  nudging by eps = {eps}:");
        let (rel_error, pass) = compare(&analytic, &numeric);
        let verdict = if pass { "ok, the two agree" } else { "MISMATCH, although nothing is wrong: the nudge crossed the kink" };
        println!("\n  relative error {rel_error:.2e}  =>  {verdict}\n");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let which = match args.as_slice() {
        [] => vec![Case::Tanh(Bug::None), Case::Tanh(Bug::Sign), Case::Tanh(Bug::Slope), Case::Kink],
        [one] if one == "none" => vec![Case::Tanh(Bug::None)],
        [one] if one == "sign" => vec![Case::Tanh(Bug::Sign)],
        [one] if one == "slope" => vec![Case::Tanh(Bug::Slope)],
        [one] if one == "kink" => vec![Case::Kink],
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    };
    for case in which {
        match case {
            Case::Tanh(Bug::None) => check("1. The correct backward pass", Bug::None),
            Case::Tanh(Bug::Sign) => check("2. Bug: the weight gradients have the wrong sign", Bug::Sign),
            Case::Tanh(Bug::Slope) => check("3. Bug: forgot to multiply by the slope of tanh (1 - y^2)", Bug::Slope),
            Case::Kink => kink(),
        }
    }
}
