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
//! Run with:
//!
//! ```text
//! cargo run --release --example ch06_gradcheck              (all three)
//! cargo run --release --example ch06_gradcheck -- sign      (just one: none | sign | slope)
//! ```

use speech::nn::{compare_gradients, numerical_gradient};

const USAGE: &str = "usage: ch06_gradcheck [none|sign|slope]";

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

fn check(title: &str, bug: Bug) {
    let mut params = vec![0.4f32, -0.2, 0.1, 0.05];
    let analytic = backward(&params, bug);
    // The second opinion: nudge each parameter by +/- 0.01 and watch the loss.
    let numeric = numerical_gradient(&mut params, 1e-2, |p| forward(p).1 as f64);
    let verdict = compare_gradients("neuron", &analytic, &numeric, 1e-2, 1e-4);

    println!("{title}\n");
    println!("  parameter    backprop   numerical");
    for (i, name) in ["w[0]", "w[1]", "w[2]", "b"].iter().enumerate() {
        println!("  {name:<9} {:>11.6} {:>11.6}", analytic[i], numeric[i]);
    }
    println!("\n  relative error {:.2e}  =>  {}\n", verdict.rel_error, if verdict.pass { "ok, the two agree" } else { "MISMATCH, the backward pass is wrong" });
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let which = match args.as_slice() {
        [] => vec![Bug::None, Bug::Sign, Bug::Slope],
        [one] if one == "none" => vec![Bug::None],
        [one] if one == "sign" => vec![Bug::Sign],
        [one] if one == "slope" => vec![Bug::Slope],
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    };
    for bug in which {
        match bug {
            Bug::None => check("1. The correct backward pass", bug),
            Bug::Sign => check("2. Bug: the weight gradients have the wrong sign", bug),
            Bug::Slope => check("3. Bug: forgot to multiply by the slope of tanh (1 - y^2)", bug),
        }
    }
}
