//! Chapter 8: CTC on examples small enough to check by hand.
//!
//! ```text
//! cargo run --release --example ch08_ctc_toy            # the walkthrough
//! cargo run --release --example ch08_ctc_toy -- bench   # time ctc_loss
//! ```
//!
//! Part 1 takes a 5-frame, 3-class table of probabilities and the target
//! "ab", lists every alignment, and shows that the forward algorithm gets the
//! same total without listing anything. Part 2 treats a grid of logits as the
//! thing to be learned and runs gradient descent on the CTC loss, so we can
//! watch an alignment for "cat" appear out of nothing.

use speech::ctc::{self, BLANK};
use speech::decode::{greedy, Alphabet};
use speech::mat::Mat;
use speech::rng::Rng;

const USAGE: &str = "usage: ch08_ctc_toy [bench]";

/// Class names for part 1: blank, a, b.
const TOY_SYMBOLS: [char; 3] = ['_', 'a', 'b'];

fn toy_string(classes: &[usize]) -> String {
    classes.iter().map(|&c| TOY_SYMBOLS[c]).collect()
}

fn part_one() {
    println!("== Part 1: every alignment, then the forward algorithm ==\n");

    // Probabilities for 5 frames over {blank, a, b}; each row sums to 1.
    let probabilities: [[f32; 3]; 5] = [[0.3, 0.6, 0.1], [0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.2, 0.1, 0.7], [0.6, 0.1, 0.3]];
    let log_probs = Mat::from_fn(5, 3, |t, k| probabilities[t][k].ln());
    let target = [1, 2]; // "ab"

    println!("The network's output, one row per frame:");
    println!("  frame   p(_)   p(a)   p(b)");
    for (t, row) in probabilities.iter().enumerate() {
        println!("    {t}     {:.2}   {:.2}   {:.2}", row[0], row[1], row[2]);
    }

    println!("\nEvery alignment that collapses to \"{}\":", toy_string(&target));
    let alignments = ctc::alignments_of(&log_probs, &target);
    let mut brute_force = 0.0;
    for (path, log_prob) in &alignments {
        let factors: Vec<String> = path.iter().enumerate().map(|(t, &k)| format!("{:.1}", probabilities[t][k])).collect();
        println!("  {}   {} = {:.5}", toy_string(path), factors.join(" x "), log_prob.exp());
        brute_force += log_prob.exp();
    }
    println!("  {} alignments (out of 3^5 = 243 paths), total probability {brute_force:.6}", alignments.len());

    println!("\nThe forward lattice alpha, as probabilities (rows: frames, columns: positions in _a_b_):");
    let extended = ctc::extend_with_blanks(&target);
    let alpha = ctc::forward_log(&log_probs, &target);
    print!("  frame ");
    for (s, &class) in extended.iter().enumerate() {
        print!("   s={s} ({}) ", TOY_SYMBOLS[class]);
    }
    println!();
    for t in 0..alpha.rows {
        print!("    {t}   ");
        for s in 0..alpha.cols {
            print!("  {:.6} ", alpha[(t, s)].exp());
        }
        println!();
    }
    let last = alpha.rows - 1;
    let from_lattice = alpha[(last, alpha.cols - 1)].exp() + alpha[(last, alpha.cols - 2)].exp();
    println!("  p(\"ab\") = alpha[4][s=4] + alpha[4][s=3] = {from_lattice:.6}");
    println!("  brute force said                          {brute_force:.6}");

    // ctc_loss takes logits. The log of a probability table is a perfectly
    // good set of logits: softmax gives the same table back.
    let logits = log_probs.clone();
    let (loss, gradient) = ctc::ctc_loss(&logits, &target);
    println!("\nLoss = -ln p(\"ab\") = {loss:.6}   (and -ln {brute_force:.6} = {:.6})", -brute_force.ln());

    println!("\nWhere the probability flows: the occupancy lattice gamma (each row sums to 1):");
    let gamma = ctc::occupancy(&log_probs, &target);
    for t in 0..gamma.rows {
        let cells: Vec<String> = gamma.row(t).iter().map(|g| format!("{g:.3}")).collect();
        println!("    {t}     {}", cells.join("  "));
    }
    let best = ctc::best_alignment(&log_probs, &target).expect("the target fits in 5 frames");
    println!("  most probable single alignment: {}  (p = {:.5})", toy_string(&best.classes), best.log_prob.exp());

    println!("\nGradient check: d loss / d logit, analytic (softmax - occupancy) against central differences:");
    println!("  frame class   analytic    numeric");
    let eps = 1e-3f32;
    let mut worst = 0.0f64;
    for t in 0..logits.rows {
        for k in 0..logits.cols {
            let (mut plus, mut minus) = (logits.clone(), logits.clone());
            plus[(t, k)] += eps;
            minus[(t, k)] -= eps;
            let step = plus[(t, k)] as f64 - minus[(t, k)] as f64;
            let numeric = (ctc::ctc_loss_f64(&plus, &target) - ctc::ctc_loss_f64(&minus, &target)) / step;
            let analytic = gradient[(t, k)] as f64;
            worst = worst.max((analytic - numeric).abs());
            println!("    {t}     {}    {analytic:>+9.6}  {numeric:>+9.6}", TOY_SYMBOLS[k]);
        }
    }
    println!("  largest difference: {worst:.2e}");
}

/// One row of an ASCII heatmap: darker characters for larger probabilities.
fn heat_row(values: impl Iterator<Item = f32>) -> String {
    const SHADES: [char; 10] = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    values.map(|p| SHADES[((p * 10.0) as usize).min(9)]).collect()
}

/// Softmax probabilities of a logits matrix.
fn softmax(logits: &Mat) -> Mat {
    ctc::log_softmax(logits).map(f32::exp)
}

fn part_two() {
    println!("\n== Part 2: gradient descent on the logits themselves ==\n");
    let text = "cat";
    let target = Alphabet::encode(text);
    let (frames, classes) = (16, Alphabet::SIZE);
    println!("Target \"{text}\" = classes {target:?}. The parameters are a {frames} x {classes} grid of logits,");
    println!("started near zero (every class equally likely in every frame). No audio, no network:");
    println!("just the loss pulling on its own input. Heatmaps show p(class) per frame, time left to right,");
    println!("for the blank and the three target letters (' ' is below 0.1, '@' is above 0.9).\n");

    let mut rng = Rng::new(8);
    let mut logits = Mat::from_fn(frames, classes, |_, _| 0.01 * rng.normal());
    let learning_rate = 0.5;
    let shown: Vec<usize> = std::iter::once(BLANK).chain(target.iter().copied()).collect();

    for step in 0..=200 {
        let (loss, gradient) = ctc::ctc_loss(&logits, &target);
        if [0, 5, 10, 20, 40, 80, 200].contains(&step) {
            let probabilities = softmax(&logits);
            let decoded = Alphabet::decode(&greedy(&ctc::log_softmax(&logits)));
            println!("step {step:>3}   loss {loss:>8.4}   p(target) {:>6.4}   greedy decode: \"{decoded}\"", (-loss).exp());
            for &class in &shown {
                let row = heat_row((0..frames).map(|t| probabilities[(t, class)]));
                println!("    {} |{row}|", Alphabet::symbol(class));
            }
            let best_path: String = speech::decode::best_path(&ctc::log_softmax(&logits)).iter().map(|&c| Alphabet::symbol(c)).collect();
            println!("  best |{best_path}|\n");
        }
        // Plain gradient descent: step against the gradient.
        for (z, g) in logits.data.iter_mut().zip(&gradient.data) {
            *z -= learning_rate * g;
        }
    }
}

/// Time `ctc_loss` on an input the size of a real training example.
fn bench() {
    let (frames, classes, target_len) = (100, Alphabet::SIZE, 20);
    let mut rng = Rng::new(1);
    let logits = Mat::from_fn(frames, classes, |_, _| rng.normal());
    let target: Vec<usize> = (0..target_len).map(|_| 1 + rng.below(classes - 1)).collect();
    let calls = 2000;
    let mut best = f64::INFINITY;
    let mut checksum = 0.0;
    // Best of several rounds: the least disturbed by whatever else is running.
    for _ in 0..10 {
        let start = std::time::Instant::now();
        for _ in 0..calls {
            let (loss, gradient) = ctc::ctc_loss(std::hint::black_box(&logits), std::hint::black_box(&target));
            checksum += loss + gradient.data[0];
        }
        best = best.min(start.elapsed().as_secs_f64() / calls as f64);
    }
    println!("ctc_loss, T = {frames}, C = {classes}, target length {target_len}: {:.1} microseconds per call (best of 10 rounds of {calls}; checksum {checksum:.1})", best * 1e6);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        [] => {
            part_one();
            part_two();
        }
        ["bench"] => bench(),
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}
