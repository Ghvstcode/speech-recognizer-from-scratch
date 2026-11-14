//! Chapter 6: a neural network recognizes the digits chapter 5 matched by
//! template.
//!
//! ```text
//! cargo run --release --example ch06_digits                 # train, measure, save the model, draw the figures
//! cargo run --release --example ch06_digits -- classify ../samples/seven_jackson.wav
//! ```
//!
//! The network is a multi-layer perceptron,
//!
//! ```text
//! Linear(96 -> 256) -> ReLU -> Linear(256 -> 128) -> ReLU -> Linear(128 -> 10)
//! ```
//!
//! trained with plain minibatch SGD. An MLP wants every input to have the
//! same size, and recordings do not, so most of this program's thinking goes
//! into [`FixedInput`]: how to squeeze a clip of any length into 96 numbers.
//!
//! What the full run prints, in order:
//!
//! 0. The gradient check (always first: if backprop is wrong, nothing after
//!    it means anything).
//! 1. Which fixed-size input works best.
//! 2. What helps on speakers the network has never heard.
//! 3. The final recipe: loss and accuracy per epoch, then the comparison
//!    with DTW on the official split and with each speaker left out.
//! 4. What one answer costs, next to DTW.
//!
//! Every training run is repeated with three seeds, because one run of a
//! small network on a small dataset can be a lucky one.

use speech::data::fsdd::{self, Clip, MfccFrontEnd, MfccRecipe, Split, DIGIT_NAMES, N_DIGITS, SAMPLE_RATE};
use speech::dsp::features::{FeatureConfig, FeatureExtractor};
use speech::dsp::resample::resample;
use speech::dtw::TemplateBank;
use speech::mat::Mat;
use speech::nn::{argmax, cross_entropy, gradcheck, softmax, GradCheckConfig, Grads, Layer, Model, Sgd};
use speech::plot::{BarChart, Color, Figure, Heatmap, LinePlot, Orientation, Series};
use speech::rng::Rng;
use speech::wav::Wav;
use std::io;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: ch06_digits                      train, compare with DTW, save the model, write the figures
       ch06_digits classify FILE.wav    load the saved model and recognize one recording";

/// Where the figures go, relative to `code/`.
const FIGURES: &str = "../figures";
/// Where the trained model is saved, relative to `code/`.
const MODEL_PATH: &str = "../models/digits-mlp.bin";
/// The seeds every experiment is repeated with.
const SEEDS: [u64; 3] = [1, 2, 3];

// ---------------------------------------------------------------------------
// From a clip of any length to a fixed number of inputs
// ---------------------------------------------------------------------------

/// The ways of making a fixed-size input that this program compares.
#[derive(Clone, Copy, Debug, PartialEq)]
enum InputKind {
    /// Cut the clip's MFCC frames into this many equal slices of time and
    /// average the frames inside each slice.
    MfccSlices(usize),
    /// The same with the 40 log-mel bands instead of 12 MFCCs.
    LogMelSlices(usize),
    /// Forget the order of the frames: only the mean and the standard
    /// deviation of every MFCC over the whole clip.
    MfccMeanAndSpread,
}

/// The input the final model uses: 8 slices x 12 MFCCs = 96 numbers.
const FINAL_INPUT: InputKind = InputKind::MfccSlices(8);

/// Turns recordings into fixed-size network inputs.
struct FixedInput {
    kind: InputKind,
    /// Chapter 5's features: 12 MFCCs per frame, silence trimmed.
    mfcc: MfccFrontEnd,
    /// For the log-mel variant.
    log_mel: FeatureExtractor,
}

impl FixedInput {
    fn new(kind: InputKind) -> FixedInput {
        FixedInput {
            kind,
            mfcc: MfccFrontEnd::new(&MfccRecipe::standard(), SAMPLE_RATE),
            log_mel: FeatureExtractor::new(&FeatureConfig::new(SAMPLE_RATE)),
        }
    }

    fn describe(&self) -> String {
        match self.kind {
            InputKind::MfccSlices(n) => format!("{n} time slices x 12 MFCCs"),
            InputKind::LogMelSlices(n) => format!("{n} time slices x 40 log-mel bands"),
            InputKind::MfccMeanAndSpread => "mean and spread of 12 MFCCs over time".to_string(),
        }
    }

    /// The clip's features before they are made fixed-size: one row per frame.
    fn frames(&self, samples: &[f32]) -> Mat {
        match self.kind {
            InputKind::MfccSlices(_) | InputKind::MfccMeanAndSpread => self.mfcc.features(samples),
            InputKind::LogMelSlices(_) => {
                let speech = self.mfcc.speech(samples); // the same trimming as for the MFCCs
                let frame_len = self.log_mel.config().frame_len();
                if speech.len() >= frame_len {
                    self.log_mel.log_mel(speech)
                } else {
                    // Shorter than one frame: pad with silence, as the MFCC front end does.
                    self.log_mel.log_mel(&[speech, &vec![0.0; frame_len - speech.len()]].concat())
                }
            }
        }
    }

    /// The network input for one recording.
    fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let frames = self.frames(samples);
        match self.kind {
            InputKind::MfccSlices(n) | InputKind::LogMelSlices(n) => slice_averages(&frames, n).data,
            InputKind::MfccMeanAndSpread => mean_and_spread(&frames),
        }
    }
}

/// Squeeze (or stretch) a feature sequence to exactly `n_slices` rows.
///
/// Lay the clip's `T` frames end to end and cut the line into `n_slices`
/// equal pieces. Each output row is the average of the frames in one piece,
/// and a frame that straddles a cut counts toward both sides in proportion
/// to how much of it lies on each. With `T = 41` and 8 slices every slice
/// averages 5.125 frames; with `T = 4` every slice is half of one frame, so
/// each frame is simply repeated twice.
///
/// This is a crude cousin of chapter 5's time warping: it stretches every
/// clip *uniformly* to the same length. A word said twice as slowly lands in
/// the same slices; a word with one drawn-out vowel does not.
fn slice_averages(frames: &Mat, n_slices: usize) -> Mat {
    assert!(frames.rows > 0, "cannot slice a clip with no frames");
    let frames_per_slice = frames.rows as f32 / n_slices as f32;
    let mut out = Mat::zeros(n_slices, frames.cols);
    for slice in 0..n_slices {
        let (from, to) = (slice as f32 * frames_per_slice, (slice + 1) as f32 * frames_per_slice);
        let last_frame = (to.ceil() as usize).min(frames.rows);
        let mut total_weight = 0.0;
        for t in from.floor() as usize..last_frame {
            // How much of frame t, which covers [t, t + 1), lies inside [from, to).
            let weight = (to.min(t as f32 + 1.0) - from.max(t as f32)).max(0.0);
            total_weight += weight;
            for (sum, value) in out.row_mut(slice).iter_mut().zip(frames.row(t)) {
                *sum += weight * value;
            }
        }
        for sum in out.row_mut(slice) {
            *sum /= total_weight;
        }
    }
    out
}

/// The mean of every column followed by the standard deviation of every
/// column: what the clip sounds like on average, with all timing discarded.
fn mean_and_spread(frames: &Mat) -> Vec<f32> {
    let n = frames.rows as f32;
    let column = |c: usize| (0..frames.rows).map(move |r| frames[(r, c)]);
    let means: Vec<f32> = (0..frames.cols).map(|c| column(c).sum::<f32>() / n).collect();
    let spreads = (0..frames.cols).map(|c| (column(c).map(|v| (v - means[c]).powi(2)).sum::<f32>() / n).sqrt());
    means.iter().copied().chain(spreads.collect::<Vec<f32>>()).collect()
}

/// Input normalization: the mean and standard deviation of every input
/// position, measured **on the training clips only**.
///
/// SGD takes the same size of step for every weight, which works best when
/// every input has about the same scale: mean 0, spread 1. The test clips
/// are shifted and scaled with the training set's numbers. Using their own
/// would let information about the test set leak into the model.
struct Standardizer {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Standardizer {
    fn fit(inputs: &[Vec<f32>], train: &[usize]) -> Standardizer {
        let (size, n) = (inputs[0].len(), train.len() as f32);
        let mean: Vec<f32> = (0..size).map(|i| train.iter().map(|&p| inputs[p][i]).sum::<f32>() / n).collect();
        let std = (0..size).map(|i| (train.iter().map(|&p| (inputs[p][i] - mean[i]).powi(2)).sum::<f32>() / n).sqrt().max(1e-6));
        Standardizer { std: std.collect(), mean }
    }

    /// The standardized inputs of the clips at `positions`, one clip per row.
    fn rows(&self, inputs: &[Vec<f32>], positions: &[usize]) -> Mat {
        Mat::from_fn(positions.len(), self.mean.len(), |r, c| (inputs[positions[r]][c] - self.mean[c]) / self.std[c])
    }

    /// A copy of `model` that does the standardizing itself, so that it can
    /// be saved as one file and fed raw inputs.
    ///
    /// The first layer computes `((x - mean) / std) W + b`. Multiplying out,
    /// that is `x (W / std) + (b - (mean / std) W)`: the same layer with
    /// each row of `W` divided by that input's `std`, and a shifted bias.
    fn bake_into(&self, model: &Model) -> Model {
        let mut baked = model.clone();
        let Layer::Linear(first) = &mut baked.layers[0] else { panic!("the first layer must be linear") };
        for i in 0..first.weight.rows {
            for j in 0..first.weight.cols {
                first.weight[(i, j)] /= self.std[i];
                first.bias[(0, j)] -= self.mean[i] * first.weight[(i, j)];
            }
        }
        baked
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// Everything that defines a training run, apart from the data and the seed.
#[derive(Clone, Debug)]
struct Recipe {
    /// Sizes of the two hidden layers.
    hidden: (usize, usize),
    epochs: usize,
    /// SGD step size. Divided by ten for the last third of the epochs, so
    /// that the weights settle instead of bouncing around the minimum.
    learning_rate: f32,
    batch_size: usize,
    /// Standard deviation of the random noise added to every (standardized)
    /// input value each time a clip is used for training. 0 disables it.
    ///
    /// This is our whole data augmentation. With it, the network never sees
    /// exactly the same input twice and cannot simply memorize the 2,500
    /// training clips. (The two usual audio augmentations would do
    /// next to nothing here: shifting a clip in time is undone by trimming
    /// the silence, and changing its volume adds the same amount to every
    /// log-mel band, which only moves MFCC 0, the one we dropped.)
    input_noise: f32,
    /// If not 0, every weight is shrunk by `learning_rate * weight_decay` of
    /// itself at every step ("weight decay"). Only used by experiment 2.
    weight_decay: f32,
}

impl Recipe {
    /// Plain SGD and nothing else.
    fn plain() -> Recipe {
        Recipe { hidden: (256, 128), epochs: 40, learning_rate: 0.05, batch_size: 32, input_noise: 0.0, weight_decay: 0.0 }
    }

    /// The recipe the chapter settles on: plain SGD plus input noise.
    fn settled() -> Recipe {
        Recipe { input_noise: 0.6, ..Recipe::plain() }
    }
}

/// A fresh, randomly initialized network. He initialization for the layers
/// that feed a ReLU, Xavier for the last.
fn new_model(n_inputs: usize, hidden: (usize, usize), rng: &mut Rng) -> Model {
    Model::new(vec![
        Layer::linear_he(n_inputs, hidden.0, rng),
        Layer::Relu,
        Layer::linear_he(hidden.0, hidden.1, rng),
        Layer::Relu,
        Layer::linear(hidden.1, N_DIGITS, rng),
    ])
}

/// Loss and accuracy at the end of one epoch.
#[derive(Clone, Copy, Debug)]
struct Epoch {
    /// Mean cross-entropy over the epoch's minibatches (on the noisy inputs).
    train_loss: f32,
    /// Percent of training clips classified correctly (clean inputs).
    train_accuracy: f32,
    test_loss: f32,
    test_accuracy: f32,
}

/// What a training run produced.
struct Trained {
    /// Expects standardized inputs.
    model: Model,
    standardizer: Standardizer,
    /// One entry per epoch. The test columns are for watching only: no
    /// decision during training looks at them.
    history: Vec<Epoch>,
    /// Predicted digit for every test clip, after the last epoch.
    predictions: Vec<usize>,
    seconds: f64,
}

impl Trained {
    fn test_accuracy(&self) -> f32 {
        self.history[self.history.len() - 1].test_accuracy
    }
}

/// Mean loss and percent correct of `model` on standardized inputs `x`.
fn evaluate(model: &Model, x: &Mat, digits: &[usize]) -> (f32, f32, Vec<usize>) {
    let logits = model.predict(x);
    let (loss_sum, _) = cross_entropy(&logits, digits);
    let predictions: Vec<usize> = (0..logits.rows).map(|r| argmax(logits.row(r))).collect();
    let correct = predictions.iter().zip(digits).filter(|(p, d)| p == d).count();
    (loss_sum / digits.len() as f32, 100.0 * correct as f32 / digits.len() as f32, predictions)
}

/// Train a network on the `train` clips of `split` and watch it on the
/// `test` clips. The same seed always gives the same model.
fn train(inputs: &[Vec<f32>], clips: &[Clip], split: &Split, recipe: &Recipe, seed: u64, print_epochs: bool) -> Trained {
    let start = Instant::now();
    let mut rng = Rng::new(seed);
    let standardizer = Standardizer::fit(inputs, &split.train);
    let digits_of = |positions: &[usize]| positions.iter().map(|&p| clips[p].digit).collect::<Vec<usize>>();
    let (train_x, train_digits) = (standardizer.rows(inputs, &split.train), digits_of(&split.train));
    let (test_x, test_digits) = (standardizer.rows(inputs, &split.test), digits_of(&split.test));

    let mut model = new_model(train_x.cols, recipe.hidden, &mut rng);
    let mut sgd = Sgd::new(recipe.learning_rate, 0.0); // momentum 0: plain SGD
    let mut grads = Grads::zeros_like(&model);
    let mut order: Vec<usize> = (0..train_x.rows).collect();
    let mut history = Vec::new();
    let mut predictions = Vec::new();

    for epoch in 1..=recipe.epochs {
        if epoch == recipe.epochs * 2 / 3 + 1 {
            sgd.lr = recipe.learning_rate / 10.0;
        }
        rng.shuffle(&mut order); // a different split into minibatches every epoch
        let mut loss_sum = 0.0;
        for batch in order.chunks(recipe.batch_size) {
            let mut x = Mat::from_fn(batch.len(), train_x.cols, |r, c| train_x[(batch[r], c)]);
            if recipe.input_noise > 0.0 {
                for value in &mut x.data {
                    *value += recipe.input_noise * rng.normal();
                }
            }
            let digits: Vec<usize> = batch.iter().map(|&i| train_digits[i]).collect();

            let cache = model.forward(&x);
            let (batch_loss, mut d_logits) = cross_entropy(cache.output(), &digits);
            loss_sum += batch_loss;
            d_logits.scale(1.0 / batch.len() as f32); // mean over the minibatch

            grads.zero();
            model.backward(&cache, &d_logits, &mut grads);
            sgd.step(&mut model, &grads);
            if recipe.weight_decay > 0.0 {
                let keep = 1.0 - sgd.lr * recipe.weight_decay;
                // Weight matrices only; a bias has a single row and is left alone.
                model.params_mut().into_iter().filter(|p| p.rows > 1).for_each(|weights| weights.scale(keep));
            }
        }

        let (_, train_accuracy, _) = evaluate(&model, &train_x, &train_digits);
        let (test_loss, test_accuracy, test_predictions) = evaluate(&model, &test_x, &test_digits);
        let stats = Epoch { train_loss: loss_sum / train_x.rows as f32, train_accuracy, test_loss, test_accuracy };
        if print_epochs {
            println!(
                "   epoch {epoch:>2}   train loss {:.4}  accuracy {:>5.1}%   |   test loss {:.4}  accuracy {:>5.1}%",
                stats.train_loss, stats.train_accuracy, stats.test_loss, stats.test_accuracy
            );
        }
        history.push(stats);
        predictions = test_predictions;
    }
    Trained { model, standardizer, history, predictions, seconds: start.elapsed().as_secs_f64() }
}

/// The splits every experiment is run on: the official one, then one per
/// held-out speaker.
fn all_splits(clips: &[Clip]) -> Vec<(String, Split)> {
    let mut splits = vec![("official".to_string(), fsdd::official_split(clips))];
    splits.extend(fsdd::leave_one_speaker_out(clips));
    splits
}

/// Test accuracies of one recipe on every split, for every seed:
/// `result[split][seed]`. The 21 training runs are spread over all cores.
fn accuracy_on_all_splits(inputs: &[Vec<f32>], clips: &[Clip], recipe: &Recipe) -> Vec<Vec<Trained>> {
    let splits = all_splits(clips);
    let jobs: Vec<(usize, u64)> = (0..splits.len()).flat_map(|s| SEEDS.map(|seed| (s, seed))).collect();
    let mut runs = fsdd::map_clips(&jobs, |&(s, seed)| train(inputs, clips, &splits[s].1, recipe, seed, false)).into_iter();
    splits.iter().map(|_| runs.by_ref().take(SEEDS.len()).collect()).collect()
}

/// Mean, smallest and largest of a few numbers.
fn spread(values: &[f32]) -> (f32, f32, f32) {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    (mean, values.iter().copied().fold(f32::INFINITY, f32::min), values.iter().copied().fold(f32::NEG_INFINITY, f32::max))
}

/// "81.3% (80.6-82.0)": the mean over the seeds, then the lowest and highest.
fn with_spread(values: &[f32]) -> String {
    let (mean, low, high) = spread(values);
    format!("{mean:>5.1}% ({low:.1}-{high:.1})")
}

/// Summarize [`accuracy_on_all_splits`]: per seed, the official-split
/// accuracy and the mean over the six held-out speakers.
fn official_and_unseen(runs: &[Vec<Trained>]) -> (Vec<f32>, Vec<f32>) {
    let official = runs[0].iter().map(Trained::test_accuracy).collect();
    let unseen = (0..SEEDS.len()).map(|seed| runs[1..].iter().map(|split| split[seed].test_accuracy()).sum::<f32>() / (runs.len() - 1) as f32);
    (official, unseen.collect())
}

// ---------------------------------------------------------------------------
// The experiments
// ---------------------------------------------------------------------------

/// Experiment 0: compare backprop's gradients with numerical ones on the
/// very model we are about to train, fed four real clips.
///
/// ReLU makes this check a little delicate. The numerical gradient nudges a
/// weight up and down and assumes the loss is smooth in between, but a ReLU
/// has a kink at zero: if the nudge carries one of the 384 hidden units
/// across it, the estimate is off although backprop is right. We therefore
/// nudge by less than the checker's default (0.001 instead of 0.01), and if
/// a tensor still disagrees we repeat the check with every ReLU replaced by
/// the smooth tanh. The linear layers and the loss, where the real bugs
/// hide, are the same code in both; a mismatch that survives the swap is a
/// bug, and we refuse to train.
fn gradient_check(inputs: &[Vec<f32>], clips: &[Clip]) {
    println!("0. Gradient check (backprop against nudging the weights, on four real clips)\n");
    let split = fsdd::official_split(clips);
    let standardizer = Standardizer::fit(inputs, &split.train);
    // Four clips of four different digits, spread across the training set.
    let positions: Vec<usize> = split.train.iter().copied().step_by(split.train.len() / 4 + 1).collect();
    let x = standardizer.rows(inputs, &positions);
    let digits: Vec<usize> = positions.iter().map(|&p| clips[p].digit).collect();
    let model = new_model(x.cols, Recipe::settled().hidden, &mut Rng::new(SEEDS[0]));
    let check = |model: &Model, config: &GradCheckConfig| {
        let report = gradcheck(model, &x, |logits| cross_entropy(logits, &digits), config);
        for line in report.to_string().lines() {
            println!("   {line}");
        }
        report.passed()
    };

    if !check(&model, &GradCheckConfig { eps: 1e-3, ..GradCheckConfig::default() }) {
        println!("\n   A mismatch. A nudge may have pushed a ReLU across its kink; the same network with tanh instead:\n");
        let smooth = Model::new(model.layers.iter().map(|layer| if *layer == Layer::Relu { Layer::Tanh } else { layer.clone() }).collect());
        assert!(check(&smooth, &GradCheckConfig::default()), "the gradient check failed: do not train with a broken backward pass");
    }
    println!("\n   backprop and the numerical gradient agree\n");
}

/// Experiment 1: the same network and the same plain training on four
/// different fixed-size inputs.
fn compare_inputs(clips: &[Clip]) {
    println!("1. Which fixed-size input? (plain SGD, mean over seeds {SEEDS:?} with lowest-highest)\n");
    println!("   {:<40} {:>6}   {:<22} {:<22}", "input", "size", "official split", "unseen speaker");
    for kind in [InputKind::MfccSlices(8), InputKind::MfccSlices(16), InputKind::LogMelSlices(8), InputKind::MfccMeanAndSpread] {
        let input = FixedInput::new(kind);
        let inputs = fsdd::map_clips(clips, |clip| input.compute(&clip.samples));
        let (official, unseen) = official_and_unseen(&accuracy_on_all_splits(&inputs, clips, &Recipe::plain()));
        println!("   {:<40} {:>6}   {:<22} {:<22}", input.describe(), inputs[0].len(), with_spread(&official), with_spread(&unseen));
    }
    println!();
}

/// Experiment 2: what moves the unseen-speaker number, on the chosen input.
fn compare_recipes(inputs: &[Vec<f32>], clips: &[Clip]) {
    println!("2. What helps on unseen speakers? ({}; mean over seeds with lowest-highest)\n", FixedInput::new(FINAL_INPUT).describe());
    println!("   {:<40} {:<22} {:<22}", "training recipe", "official split", "unseen speaker");
    let plain = Recipe::plain();
    let settled = Recipe::settled();
    let recipes = [
        ("plain SGD, hidden layers 256 and 128", plain.clone()),
        ("  smaller network: 32 and 32", Recipe { hidden: (32, 32), ..plain.clone() }),
        ("  10 epochs instead of 40", Recipe { epochs: 10, ..plain.clone() }),
        ("  + input noise 0.3", Recipe { input_noise: 0.3, ..plain.clone() }),
        ("  + input noise 0.6 (settled recipe)", settled.clone()),
        ("  + input noise 1.0", Recipe { input_noise: 1.0, ..plain }),
        ("  + input noise 0.6 + weight decay 0.01", Recipe { weight_decay: 0.01, ..settled }),
    ];
    for (name, recipe) in recipes {
        let (official, unseen) = official_and_unseen(&accuracy_on_all_splits(inputs, clips, &recipe));
        println!("   {name:<40} {:<22} {:<22}", with_spread(&official), with_spread(&unseen));
    }
    println!();
}

/// Chapter 5's recognizer on the same split, for the comparison: every
/// training clip is a template, the nearest one under DTW decides.
fn dtw_predictions(clips: &[Clip], mfcc: &[Mat], split: &Split) -> Vec<usize> {
    let mut bank = TemplateBank::new();
    for &p in &split.train {
        bank.add(clips[p].digit, mfcc[p].clone());
    }
    let queries: Vec<Mat> = split.test.iter().map(|&p| mfcc[p].clone()).collect();
    bank.classify_many(&queries, 1).into_iter().map(|(digit, _)| digit).collect()
}

fn percent_correct(clips: &[Clip], positions: &[usize], predicted: &[usize]) -> f32 {
    100.0 * positions.iter().zip(predicted).filter(|&(&p, &digit)| clips[p].digit == digit).count() as f32 / positions.len() as f32
}

/// The numbers behind the comparison table and the bar chart.
struct Comparison {
    /// "official", then the held-out speakers.
    names: Vec<String>,
    dtw: Vec<f32>,
    /// Mean over the seeds.
    mlp: Vec<f32>,
}

/// Experiment 3, second half: DTW and the MLP side by side on every split.
fn compare_with_dtw(clips: &[Clip], runs: &[Vec<Trained>]) -> Comparison {
    let front_end = MfccFrontEnd::new(&MfccRecipe::standard(), SAMPLE_RATE);
    let mfcc = fsdd::map_clips(clips, |clip| front_end.features(&clip.samples));
    let mut comparison = Comparison { names: Vec::new(), dtw: Vec::new(), mlp: Vec::new() };

    println!("\n   DTW against the MLP, same clips, same splits (MLP: mean over seeds {SEEDS:?} with lowest-highest)\n");
    println!("   {:<28} {:>9}   {:<22} {:>8}", "test clips", "DTW 1-NN", "MLP", "MLP-DTW");
    for ((name, split), split_runs) in all_splits(clips).iter().zip(runs) {
        let dtw = percent_correct(clips, &split.test, &dtw_predictions(clips, &mfcc, split));
        let mlp: Vec<f32> = split_runs.iter().map(Trained::test_accuracy).collect();
        let label = if name == "official" { "official split (300 clips)".to_string() } else { format!("unseen speaker: {name}") };
        println!("   {label:<28} {dtw:>8.1}%   {:<22} {:>+8.1}", with_spread(&mlp), spread(&mlp).0 - dtw);
        comparison.names.push(name.clone());
        comparison.dtw.push(dtw);
        comparison.mlp.push(spread(&mlp).0);
    }
    let dtw_unseen = spread(&comparison.dtw[1..]).0;
    let (_, mlp_unseen) = official_and_unseen(runs);
    println!("   {:<28} {dtw_unseen:>8.1}%   {:<22} {:>+8.1}", "unseen speaker: mean", with_spread(&mlp_unseen), spread(&mlp_unseen).0 - dtw_unseen);
    comparison.names.push("mean".to_string());
    comparison.dtw.push(dtw_unseen);
    comparison.mlp.push(spread(&mlp_unseen).0);
    comparison
}

/// Experiment 4: seconds of work per answer, on one core, for both
/// recognizers; and how much each has to keep in memory.
fn cost(clips: &[Clip], input: &FixedInput, baked: &Model, training_seconds: f64) {
    println!("\n4. What one answer costs (one core, official split, averaged over the 300 test clips)\n");
    let split = fsdd::official_split(clips);
    let microseconds = |work: &mut dyn FnMut(&Clip)| {
        let start = Instant::now();
        split.test.iter().for_each(|&p| work(&clips[p]));
        1e6 * start.elapsed().as_secs_f64() / split.test.len() as f64
    };

    // The MLP: features, then one forward pass.
    let features = microseconds(&mut |clip| drop(input.compute(&clip.samples)));
    let as_row = |x: Vec<f32>| Mat::from_vec(1, x.len(), x);
    let prepared: Vec<Mat> = split.test.iter().map(|&p| as_row(input.compute(&clips[p].samples))).collect();
    let start = Instant::now();
    for x in &prepared {
        baked.predict(x);
    }
    let forward = 1e6 * start.elapsed().as_secs_f64() / prepared.len() as f64;

    // DTW: features, then one DTW per template.
    let mfcc = fsdd::map_clips(clips, |clip| input.mfcc.features(&clip.samples));
    let mut bank = TemplateBank::new();
    split.train.iter().for_each(|&p| bank.add(clips[p].digit, mfcc[p].clone()));
    let dtw_features = microseconds(&mut |clip| drop(input.mfcc.features(&clip.samples)));
    let start = Instant::now();
    for &p in &split.test {
        bank.classify(&mfcc[p]);
    }
    let matching = 1e6 * start.elapsed().as_secs_f64() / split.test.len() as f64;

    let bank_numbers: usize = split.train.iter().map(|&p| mfcc[p].data.len()).sum();
    println!("   {:<26} {:>14} {:>16} {:>12} {:>20}", "", "features", "deciding", "total", "numbers to store");
    println!("   {:<26} {:>11.0} us {:>13.0} us {:>9.0} us {:>20}", "DTW, 2700 templates", dtw_features, matching, dtw_features + matching, bank_numbers);
    println!("   {:<26} {:>11.0} us {:>13.1} us {:>9.0} us {:>20}", "MLP", features, forward, features + forward, baked.num_params());
    println!(
        "\n   Deciding is {:.0} times faster with the MLP ({:.0} times counting the features), and it keeps {:.0} times fewer numbers.",
        matching / forward,
        (dtw_features + matching) / (features + forward),
        bank_numbers as f64 / baked.num_params() as f64
    );
    println!("   The price was paid once, in advance: {training_seconds:.1} s of training on one core. DTW needs no training at all.");
}

// ---------------------------------------------------------------------------
// Figures
// ---------------------------------------------------------------------------

/// Loss and accuracy per epoch: on the training clips, on test clips of
/// known speakers (official split), and on unseen speakers (mean of the six
/// leave-one-speaker-out runs). All from the first seed.
fn draw_training_curve(runs: &[Vec<Trained>], path: &str) -> io::Result<()> {
    let official = &runs[0][0].history;
    let epochs: Vec<f32> = (1..=official.len()).map(|e| e as f32).collect();
    let column = |f: fn(&Epoch) -> f32| official.iter().map(f).collect::<Vec<f32>>();
    // Average the six held-out-speaker runs epoch by epoch.
    let unseen = |f: fn(&Epoch) -> f32| -> Vec<f32> {
        (0..official.len()).map(|e| runs[1..].iter().map(|split| f(&split[0].history[e])).sum::<f32>() / (runs.len() - 1) as f32).collect()
    };
    let drop_at = (official.len() * 2 / 3) as f32 + 0.5;
    Figure::new("Training the digit network: it learns the training speakers perfectly, strangers less well")
        .desc("Two line plots against the training epoch. Top: the cross-entropy loss falls towards zero on the training clips and on test clips of the same speakers, but on unseen speakers it stops falling early and stays far higher. Bottom: accuracy reaches 100 percent on training clips and nearly that on test clips of known speakers, and levels off near 80 percent on unseen speakers.")
        .panel(
            LinePlot::new("loss (cross-entropy)")
                .series(Series::new("training clips", &epochs, &column(|e| e.train_loss)).color(Color::Ink48))
                .series(Series::new("test clips, known speakers", &epochs, &column(|e| e.test_loss)).color(Color::Green).width(2.0))
                .series(Series::new("test clips, unseen speaker", &epochs, &unseen(|e| e.test_loss)).color(Color::Amber).width(2.0))
                .vline(drop_at, "learning rate / 10")
                .x_range(0.0, official.len() as f32 + 0.5)
                .y_range(0.0, 2.0)
                .y_label("loss")
                .height(170.0),
        )
        .panel(
            LinePlot::new("accuracy")
                .series(Series::new("training clips", &epochs, &column(|e| e.train_accuracy)).color(Color::Ink48))
                .series(Series::new("test clips, known speakers", &epochs, &column(|e| e.test_accuracy)).color(Color::Green).width(2.0))
                .series(Series::new("test clips, unseen speaker", &epochs, &unseen(|e| e.test_accuracy)).color(Color::Amber).width(2.0))
                .vline(drop_at, "")
                .x_range(0.0, official.len() as f32 + 0.5)
                .y_range(40.0, 102.0)
                .x_label("epoch")
                .y_label("accuracy (%)")
                .legend(false)
                .height(170.0),
        )
        .share_x()
        .save(path)
}

/// DTW and MLP accuracy side by side for every split.
fn draw_dtw_vs_mlp(comparison: &Comparison, path: &str) -> io::Result<()> {
    let (mut labels, mut values) = (Vec::new(), Vec::new());
    for (i, name) in comparison.names.iter().enumerate() {
        let group = match name.as_str() {
            "official" => "known".to_string(),
            "mean" => "MEAN".to_string(),
            speaker => speaker.to_string(),
        };
        labels.push(format!("{group} DTW"));
        labels.push(format!("{group} MLP"));
        values.extend([comparison.dtw[i], comparison.mlp[i]]);
    }
    let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    let mut chart = BarChart::new("DTW (gray) against the MLP (green): known speakers, each speaker unseen, and the unseen mean", &labels, &values)
        .y_range(0.0, 115.0)
        .y_label("accuracy (%)")
        .x_label("who is being recognized (\"known\": official split; a name: that speaker was left out)")
        .show_values(1)
        .height(230.0);
    for mlp_bar in (1..values.len()).step_by(2) {
        chart = chart.highlight(mlp_bar, Color::Green);
    }
    Figure::new("DTW (gray) against the MLP (green): known speakers, each speaker unseen, and the unseen mean")
        .desc("A bar chart with a pair of bars, DTW and MLP, for eight settings. On the official split with known speakers both reach about 100 percent. With a speaker left out both drop, to between 55 and 92 percent depending on the speaker, and their means over the six speakers are within a point or two of each other.")
        .panel(chart)
        .save(path)
}

/// The model's ten probabilities for a clip, as a bar chart panel.
fn probability_panel(title: &str, probabilities: &[f32], said: usize) -> BarChart {
    let mut chart = BarChart::new(title, &DIGIT_NAMES, probabilities).y_range(0.0, 1.1).y_label("probability").show_values(2).height(140.0);
    let heard = argmax(probabilities);
    chart = chart.highlight(said, Color::Green);
    if heard != said {
        chart = chart.highlight(heard, Color::Red);
    }
    chart
}

/// The softmax output for two test clips: the book's sample "seven", and the
/// test clip the model is least sure about.
fn draw_probabilities(clips: &[Clip], inputs: &[Vec<f32>], baked: &Model, path: &str) -> io::Result<()> {
    let split = fsdd::official_split(clips);
    let probabilities = |p: usize| softmax(&baked.predict(&Mat::from_vec(1, inputs[p].len(), inputs[p].clone()))).data;
    let sample = clips.iter().position(|c| c.file_name() == "7_jackson_0.wav").expect("the sample clip is in the dataset");
    // Least sure: the test clip whose largest probability is smallest.
    let top = |p: usize| probabilities(p).into_iter().fold(0.0, f32::max);
    let unsure = split.test.iter().copied().min_by(|&a, &b| top(a).total_cmp(&top(b))).expect("the test set is not empty");
    let title = |p: usize| format!("{} (\"{}\")", clips[p].file_name(), DIGIT_NAMES[clips[p].digit]);
    Figure::new("What the network says: ten probabilities that sum to 1")
        .desc("Two bar charts of the network's softmax output over the ten digits. For a clear recording of seven nearly all the probability is on seven. For the test clip the network is least sure about, the probability is split between two digits.")
        .panel(probability_panel(&format!("a clear case: {}", title(sample)), &probabilities(sample), clips[sample].digit))
        .panel(probability_panel(&format!("the test clip it is least sure about: {}", title(unsure)), &probabilities(unsure), clips[unsure].digit))
        .save(path)
}

/// Three clips of different lengths, each as its MFCC frames and as the 8
/// slices the network gets.
fn draw_fixed_size_input(clips: &[Clip], input: &FixedInput, path: &str) -> io::Result<()> {
    let InputKind::MfccSlices(n_slices) = input.kind else { panic!("this figure shows the sliced MFCC input") };
    let names = ["7_jackson_0.wav", "7_jackson_37.wav", "0_jackson_0.wav"];
    let frames: Vec<Mat> = names
        .iter()
        .map(|name| input.frames(&clips.iter().find(|c| c.file_name() == *name).expect("the clip is in the dataset").samples))
        .collect();
    let longest = frames.iter().map(|f| f.rows).max().unwrap_or(1);
    let mut figure = Figure::new(&format!("Any length in, {n_slices} x 12 numbers out: the network's fixed-size input"))
        .desc("Six heatmaps in three pairs. Each pair shows one recording: first its MFCC frames, 41, 30 and 55 frames long, drawn to scale, then the same recording averaged into 8 equal slices of time. After slicing, all three recordings have the same size, and the two recordings of seven look alike.");
    for (name, frames) in names.iter().zip(&frames) {
        // Pad the shorter clips with "no data" so that all three are drawn to
        // the same time scale.
        let padded = Mat::from_fn(longest, frames.cols, |t, c| if t < frames.rows { frames[(t, c)] } else { f32::NAN });
        let picture = |title: &str, values: &Mat, x_label: &str| {
            // Row 0 of our features is MFCC 1 (coefficient 0 was dropped).
            Heatmap::new(title, values)
                .orientation(Orientation::TimeOnX)
                .value_range(-8.0, 8.0)
                .y_range(0.5, 12.5)
                .x_label(x_label)
                .y_label("MFCC")
                .height(48.0)
        };
        figure = figure
            .panel(picture(&format!("{name}: {} frames", frames.rows), &padded, "frame (10 ms each)"))
            .panel(picture(&format!("the same clip as {} numbers", n_slices * frames.cols), &slice_averages(frames, n_slices), "slice of time"));
    }
    figure.save(path)
}

// ---------------------------------------------------------------------------

/// Read a recording and bring it to the dataset's sample rate.
fn read_at_8k(path: &str) -> io::Result<Vec<f32>> {
    let wav = Wav::read(path)?;
    println!("{path}: {:.2} s at {} Hz", wav.samples.len() as f32 / wav.sample_rate as f32, wav.sample_rate);
    if wav.sample_rate == SAMPLE_RATE {
        return Ok(wav.samples);
    }
    println!("resampling from {} Hz to {SAMPLE_RATE} Hz, the rate the model was trained on", wav.sample_rate);
    Ok(resample(&wav.samples, wav.sample_rate, SAMPLE_RATE))
}

/// Load the saved model and print its ten probabilities for one recording.
fn classify(path: &str) -> io::Result<()> {
    let samples = read_at_8k(path)?;
    let model = Model::load(MODEL_PATH).map_err(|e| io::Error::new(e.kind(), format!("{MODEL_PATH}: {e} (run ch06_digits without arguments to train it)")))?;
    let input = FixedInput::new(FINAL_INPUT);
    let start = Instant::now();
    let x = input.compute(&samples);
    if model.layers[0].in_features() != Some(x.len()) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{MODEL_PATH} was not trained on {} inputs", x.len())));
    }
    let logits = model.predict(&Mat::from_vec(1, x.len(), x));
    let elapsed = start.elapsed();
    let probabilities = softmax(&logits);

    println!("loaded {MODEL_PATH}: {} weights\n", model.num_params());
    let heard = argmax(probabilities.row(0));
    for (digit, &p) in probabilities.row(0).iter().enumerate() {
        let bar = "#".repeat((40.0 * p).round() as usize);
        println!("   {:<6} {:>6.1}%  {bar}{}", DIGIT_NAMES[digit], 100.0 * p, if digit == heard { "  <--" } else { "" });
    }
    println!("\nIt says \"{}\" (digit {heard}), in {:.0} microseconds including the features.", DIGIT_NAMES[heard], 1e6 * elapsed.as_secs_f64());
    Ok(())
}

fn run_experiments() -> io::Result<()> {
    let clips = fsdd::load()?;
    let input = FixedInput::new(FINAL_INPUT);
    let inputs = fsdd::map_clips(&clips, |clip| input.compute(&clip.samples));
    println!("Free Spoken Digit Dataset: {} clips. Network input: {} = {} numbers per clip.\n", clips.len(), input.describe(), inputs[0].len());

    gradient_check(&inputs, &clips);
    compare_inputs(&clips);
    compare_recipes(&inputs, &clips);

    let recipe = Recipe::settled();
    println!("3. The settled recipe: SGD, learning rate {}, minibatches of {}, {} epochs, input noise {}\n", recipe.learning_rate, recipe.batch_size, recipe.epochs, recipe.input_noise);
    let official = fsdd::official_split(&clips);
    let model = new_model(inputs[0].len(), recipe.hidden, &mut Rng::new(SEEDS[0]));
    for line in model.summary().lines() {
        println!("   {line}");
    }
    println!("\n   Official split, seed {}: {} training clips, {} test clips\n", SEEDS[0], official.train.len(), official.test.len());
    let shown = train(&inputs, &clips, &official, &recipe, SEEDS[0], true);
    println!("\n   trained in {:.1} s on one core", shown.seconds);

    let runs = accuracy_on_all_splits(&inputs, &clips, &recipe);
    let comparison = compare_with_dtw(&clips, &runs);

    // Save the model we just watched train, with its input normalization
    // folded into the first layer, and make sure the file gives the same
    // answers as the model in memory.
    let baked = shown.standardizer.bake_into(&shown.model);
    std::fs::create_dir_all("../models")?;
    baked.save(MODEL_PATH)?;
    let reloaded = Model::load(MODEL_PATH)?;
    let raw_test = Mat::from_fn(official.test.len(), inputs[0].len(), |r, c| inputs[official.test[r]][c]);
    let logits = reloaded.predict(&raw_test);
    let from_file: Vec<usize> = (0..logits.rows).map(|r| argmax(logits.row(r))).collect();
    assert_eq!(from_file, shown.predictions, "the saved model disagrees with the trained one");
    let bytes = std::fs::metadata(MODEL_PATH)?.len();
    println!("\n   saved {MODEL_PATH} ({} weights, {bytes} bytes; official-split test accuracy {:.1}%)", baked.num_params(), shown.test_accuracy());

    cost(&clips, &input, &baked, shown.seconds);

    println!("\nFigures:\n");
    draw_training_curve(&runs, &format!("{FIGURES}/ch06-training-curve.svg"))?;
    draw_dtw_vs_mlp(&comparison, &format!("{FIGURES}/ch06-dtw-vs-mlp.svg"))?;
    draw_probabilities(&clips, &inputs, &baked, &format!("{FIGURES}/ch06-digit-probabilities.svg"))?;
    draw_fixed_size_input(&clips, &input, &format!("{FIGURES}/ch06-fixed-size-input.svg"))?;
    for name in ["training-curve", "dtw-vs-mlp", "digit-probabilities", "fixed-size-input"] {
        println!("   wrote {FIGURES}/ch06-{name}.svg");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let outcome = match args.as_slice() {
        [] => run_experiments(),
        ["classify", path] => classify(path),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
