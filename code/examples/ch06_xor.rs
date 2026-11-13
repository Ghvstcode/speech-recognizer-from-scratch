//! Chapter 6: the smallest network that needs a hidden layer.
//!
//! XOR ("one or the other but not both") is the classic problem a single
//! linear layer cannot solve: no straight line separates {(0,1), (1,0)} from
//! {(0,0), (1,1)}. A hidden layer of a few tanh units can.
//!
//! This trains `Linear(2 -> 4) -> Tanh -> Linear(4 -> 2)` with plain SGD,
//! after first checking the backward pass against numerical gradients.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example ch06_xor
//! cargo run --release --example ch06_xor -- --steps 2000 --lr 0.3 --seed 5
//! ```

use speech::mat::Mat;
use speech::nn::{argmax, cross_entropy, gradcheck, softmax, GradCheckConfig, Grads, Layer, Model, Sgd};
use speech::rng::Rng;

const USAGE: &str = "usage: ch06_xor [--steps 1000] [--lr 0.5] [--seed 1]";

fn parse_args() -> Result<(usize, f32, u64), String> {
    let (mut steps, mut lr, mut seed) = (1000usize, 0.5f32, 1u64);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))?;
        let bad = || format!("could not read {value:?} as a number for {}", args[i]);
        match args[i].as_str() {
            "--steps" => steps = value.parse().map_err(|_| bad())?,
            "--lr" => lr = value.parse().map_err(|_| bad())?,
            "--seed" => seed = value.parse().map_err(|_| bad())?,
            other => return Err(format!("unknown option {other:?}")),
        }
        i += 2;
    }
    Ok((steps, lr, seed))
}

fn main() {
    let (steps, lr, seed) = match parse_args() {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("error: {message}\n{USAGE}");
            std::process::exit(2);
        }
    };

    // The whole dataset: four points, one per row, and the class of each.
    let x = Mat::from_rows(&[vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]]);
    let classes = [0usize, 1, 1, 0];

    let mut rng = Rng::new(seed);
    let mut model = Model::new(vec![Layer::linear(2, 4, &mut rng), Layer::Tanh, Layer::linear(4, 2, &mut rng)]);
    print!("{}", model.summary());

    // Before training anything: is the backward pass right?
    println!("\nGradient check (analytic backprop vs. nudging each weight by +/- 0.01):\n");
    let report = gradcheck(&model, &x, |logits| cross_entropy(logits, &classes), &GradCheckConfig::default());
    print!("{report}");
    println!("=> {}\n", if report.passed() { "backprop agrees with the numerical gradient" } else { "MISMATCH: do not trust this backward pass" });

    // Plain gradient descent on all four points at once.
    let mut sgd = Sgd::new(lr, 0.0);
    let mut grads = Grads::zeros_like(&model);
    println!("Training with plain SGD, learning rate {lr}:\n");
    for step in 0..=steps {
        let cache = model.forward(&x);
        let (loss_sum, mut d_logits) = cross_entropy(cache.output(), &classes);
        if step % (steps / 10).max(1) == 0 {
            println!("step {step:>5}   loss {:.4}", loss_sum / 4.0);
        }
        d_logits.scale(1.0 / 4.0); // gradient of the *mean* loss over the four points
        grads.zero();
        model.backward(&cache, &d_logits, &mut grads);
        sgd.step(&mut model, &grads);
    }

    println!("\n  a  b    p(a xor b)   prediction");
    let probabilities = softmax(&model.predict(&x));
    let mut correct = 0;
    for (r, &class) in classes.iter().enumerate() {
        let predicted = argmax(probabilities.row(r));
        correct += (predicted == class) as usize;
        println!("  {}  {}    {:.3}        {predicted}  {}", x[(r, 0)], x[(r, 1)], probabilities[(r, 1)], if predicted == class { "correct" } else { "WRONG" });
    }
    println!("\naccuracy: {correct}/4");
}
