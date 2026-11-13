//! Chapter 6: a three-armed spiral, the toy problem that needs real depth.
//!
//! Points lie along three interleaved spiral arms; the class of a point is
//! the arm it belongs to. No straight lines separate them, and not even one
//! hidden layer does it comfortably. We train
//!
//! ```text
//! Linear(2 -> 32) -> ReLU -> Linear(32 -> 32) -> ReLU -> Linear(32 -> 3)
//! ```
//!
//! with plain minibatch SGD, watch the loss fall, measure accuracy on points
//! the network never trained on, and draw the regions it has learned.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example ch06_spiral
//! cargo run --release --example ch06_spiral -- --epochs 500 --lr 0.1 --seed 3
//! ```

use speech::mat::Mat;
use speech::nn::{argmax, cross_entropy, Grads, Layer, Model, Sgd};
use speech::rng::Rng;

const USAGE: &str = "usage: ch06_spiral [--epochs 300] [--lr 0.2] [--seed 1]";
const ARMS: usize = 3;
const BATCH: usize = 32;

fn parse_args() -> Result<(usize, f32, u64), String> {
    let (mut epochs, mut lr, mut seed) = (300usize, 0.2f32, 1u64);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))?;
        let bad = || format!("could not read {value:?} as a number for {}", args[i]);
        match args[i].as_str() {
            "--epochs" => epochs = value.parse().map_err(|_| bad())?,
            "--lr" => lr = value.parse().map_err(|_| bad())?,
            "--seed" => seed = value.parse().map_err(|_| bad())?,
            other => return Err(format!("unknown option {other:?}")),
        }
        i += 2;
    }
    Ok((epochs, lr, seed))
}

/// `per_arm` noisy points along each of the three arms: `(x, y)` and the arm.
fn make_spiral(per_arm: usize, rng: &mut Rng) -> (Vec<[f32; 2]>, Vec<usize>) {
    let (mut points, mut arms) = (Vec::new(), Vec::new());
    for arm in 0..ARMS {
        for i in 0..per_arm {
            let radius = (i as f32 + 0.5) / per_arm as f32; // 0 at the centre, 1 at the tip
            // Each arm starts a third of a turn after the previous one and
            // winds a bit more than half a turn on its way out.
            let angle = arm as f32 * std::f32::consts::TAU / ARMS as f32 + radius * 4.0 + 0.15 * rng.normal();
            points.push([radius * angle.cos(), radius * angle.sin()]);
            arms.push(arm);
        }
    }
    (points, arms)
}

/// Stack the chosen points into a matrix, one point per row.
fn rows_of(points: &[[f32; 2]], which: &[usize]) -> Mat {
    Mat::from_fn(which.len(), 2, |r, c| points[which[r]][c])
}

/// Fraction of points whose highest logit is the right arm.
fn accuracy(model: &Model, points: &[[f32; 2]], arms: &[usize]) -> f32 {
    let all: Vec<usize> = (0..points.len()).collect();
    let logits = model.predict(&rows_of(points, &all));
    let correct = arms.iter().enumerate().filter(|&(r, &arm)| argmax(logits.row(r)) == arm).count();
    correct as f32 / points.len() as f32
}

/// Draw which arm the model predicts at every position of a grid, with the
/// training points on top as the digits 0, 1, 2.
fn draw_regions(model: &Model, points: &[[f32; 2]], arms: &[usize]) {
    const WIDTH: usize = 64;
    const HEIGHT: usize = 30;
    let to_xy = |col: usize, row: usize| [-1.05 + 2.1 * col as f32 / (WIDTH - 1) as f32, 1.05 - 2.1 * row as f32 / (HEIGHT - 1) as f32];
    let mut picture = vec![vec![' '; WIDTH]; HEIGHT];
    for (row, line) in picture.iter_mut().enumerate() {
        let xs = Mat::from_fn(WIDTH, 2, |col, c| to_xy(col, row)[c]);
        let logits = model.predict(&xs);
        for (col, cell) in line.iter_mut().enumerate() {
            *cell = [' ', '.', '+'][argmax(logits.row(col))];
        }
    }
    for (p, &arm) in points.iter().zip(arms) {
        let col = ((p[0] + 1.05) / 2.1 * (WIDTH - 1) as f32).round() as usize;
        let row = ((1.05 - p[1]) / 2.1 * (HEIGHT - 1) as f32).round() as usize;
        picture[row.min(HEIGHT - 1)][col.min(WIDTH - 1)] = char::from(b'0' + arm as u8);
    }
    for line in picture {
        println!("  {}", line.into_iter().collect::<String>());
    }
}

fn main() {
    let (epochs, lr, seed) = match parse_args() {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("error: {message}\n{USAGE}");
            std::process::exit(2);
        }
    };

    let mut rng = Rng::new(seed);
    let (train_points, train_arms) = make_spiral(100, &mut rng);
    let (test_points, test_arms) = make_spiral(50, &mut rng);

    // He initialisation for the two layers that feed a ReLU, Xavier for the last.
    let mut model = Model::new(vec![
        Layer::linear_he(2, 32, &mut rng),
        Layer::Relu,
        Layer::linear_he(32, 32, &mut rng),
        Layer::Relu,
        Layer::linear(32, ARMS, &mut rng),
    ]);
    print!("{}", model.summary());

    println!();
    println!("Training on {} points with plain SGD (learning rate {lr}, minibatches of {BATCH}):\n", train_points.len());
    let mut sgd = Sgd::new(lr, 0.0);
    let mut grads = Grads::zeros_like(&model);
    let mut order: Vec<usize> = (0..train_points.len()).collect();
    for epoch in 1..=epochs {
        rng.shuffle(&mut order); // a different split into minibatches every epoch
        let mut epoch_loss = 0.0;
        for batch in order.chunks(BATCH) {
            let x = rows_of(&train_points, batch);
            let arms: Vec<usize> = batch.iter().map(|&i| train_arms[i]).collect();

            let cache = model.forward(&x);
            let (loss_sum, mut d_logits) = cross_entropy(cache.output(), &arms);
            epoch_loss += loss_sum;
            d_logits.scale(1.0 / batch.len() as f32); // mean over the minibatch

            grads.zero();
            model.backward(&cache, &d_logits, &mut grads);
            sgd.step(&mut model, &grads);
        }
        if epoch == 1 || epoch % (epochs / 10).max(1) == 0 {
            println!("epoch {epoch:>4}   loss {:.4}   train accuracy {:>5.1}%", epoch_loss / train_points.len() as f32, 100.0 * accuracy(&model, &train_points, &train_arms));
        }
    }

    println!("\naccuracy on {} points it never saw: {:.1}%\n", test_points.len(), 100.0 * accuracy(&model, &test_points, &test_arms));
    println!("What the network learned (background: predicted arm; digits: training points):\n");
    draw_regions(&model, &train_points, &train_arms);
}
