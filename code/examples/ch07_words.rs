//! Chapter 7: a 35-word classifier, three ways.
//!
//! Trains three networks on Google Speech Commands, all on the same 40-band
//! log-mel features of one-second clips, and compares them:
//!
//! ```text
//! mlp        flatten 98 x 40 -> Linear -> ReLU -> Linear -> ReLU -> Linear(35)
//! conv       Conv1d -> ReLU -> Conv1d -> ReLU -> Conv1d -> ReLU -> MeanPool -> Linear(35)
//! conv+gru   Conv1d -> ReLU -> BiGru -> MeanPool -> Linear(35)
//! ```
//!
//! The MLP is what chapter 6 would build: every (frame, band) cell gets its
//! own weight, so a word that starts 50 ms later looks like a different
//! input. The other two slide the same weights along the time axis.
//!
//! ```text
//! cargo run --release --example ch07_words -- train                 # all three, about 10 minutes
//! cargo run --release --example ch07_words -- train --model conv+gru --epochs 4
//! cargo run --release --example ch07_words -- classify ../samples/seven.wav
//! ```
//!
//! `train` writes the best conv+gru model to `../models/words.bin` and three
//! figures to `../figures/` (only when all three models were trained).

use speech::data::pipeline::{load_audio_16k, models_dir, FrontEnd, WordClassifier};
use speech::data::speech_commands::{dataset_dir, parallel_map, LoadOptions, SpeechCommands, Split, WORDS};
use speech::dsp::features::normalize;
use speech::mat::Mat;
use speech::nn::{argmax, batch_gradients, clip_global_norm, cross_entropy, Adam, Layer, LrSchedule, Model};
use speech::plot::{BarChart, Color, Figure, Heatmap, LinePlot, Series};
use speech::rng::Rng;
use std::time::Instant;

const USAGE: &str = "usage: ch07_words train [--model mlp|conv|conv+gru|all] [--epochs 20] [--batch 64] [--lr 0.002]
                        [--threads 10] [--max-per-word N] [--seed 7] [--no-save] [--no-augment]
       ch07_words classify <file.wav>... [--model-file ../models/words.bin]";

/// Frames in one second of audio at a 10 ms hop with 25 ms windows.
const FRAMES: usize = 98;

/// The three architectures of the chapter.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arch {
    Mlp,
    Conv,
    ConvGru,
}

impl Arch {
    fn name(self) -> &'static str {
        match self {
            Arch::Mlp => "mlp",
            Arch::Conv => "conv",
            Arch::ConvGru => "conv+gru",
        }
    }

    /// Build the network with freshly drawn weights.
    fn build(self, rng: &mut Rng) -> Model {
        let features = FrontEnd::N_FEATURES;
        match self {
            // Sees the whole clip as one row of 98 x 40 = 3,920 numbers.
            Arch::Mlp => Model::new(vec![
                Layer::linear_he(FRAMES * features, 64, rng),
                Layer::Relu,
                Layer::linear_he(64, 64, rng),
                Layer::Relu,
                Layer::linear(64, WORDS.len(), rng),
            ]),
            // Three convolutions, each looking at 5 neighbouring frames of
            // the layer below; the first two halve the frame rate. One
            // output frame sees 29 input frames, about 0.3 s.
            Arch::Conv => Model::new(vec![
                Layer::conv1d(features, 64, 5, 2, 2, rng),
                Layer::Relu,
                Layer::conv1d(64, 96, 5, 2, 2, rng),
                Layer::Relu,
                Layer::conv1d(96, 128, 5, 1, 2, rng),
                Layer::Relu,
                Layer::MeanPool,
                Layer::linear(128, WORDS.len(), rng),
            ]),
            // One convolution to find local patterns and halve the frame
            // rate, then a GRU in each direction that can remember the
            // whole word.
            Arch::ConvGru => Model::new(vec![
                Layer::conv1d(features, 64, 5, 2, 2, rng),
                Layer::Relu,
                Layer::bigru(64, 128, rng),
                Layer::MeanPool,
                Layer::linear(256, WORDS.len(), rng),
            ]),
        }
    }

    /// The MLP wants the clip as a single row; the others as frames x bands.
    fn shape_input(self, features: &Mat) -> Mat {
        match self {
            Arch::Mlp => Mat::from_vec(1, features.rows * features.cols, features.data.clone()),
            _ => features.clone(),
        }
    }
}

struct Options {
    archs: Vec<Arch>,
    epochs: usize,
    batch: usize,
    lr: f32,
    threads: usize,
    max_per_word: Option<usize>,
    seed: u64,
    save: bool,
    /// Whether training clips are randomly shifted and masked.
    augment: bool,
}

/// One labelled set of clips: features and class ids.
struct Set {
    features: Vec<Mat>,
    labels: Vec<usize>,
}

/// What we remember about one training run.
struct RunResult {
    arch: Arch,
    params: usize,
    /// Mean training loss per epoch.
    train_loss: Vec<f32>,
    /// Training accuracy per epoch (measured while training, so slightly pessimistic).
    train_acc: Vec<f32>,
    /// Validation accuracy per epoch.
    valid_acc: Vec<f32>,
    best_epoch: usize,
    test_acc: f32,
    seconds_per_epoch: f32,
    model: Model,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("train") => parse_train(&args[1..]).and_then(|options| train_all(&options).map_err(|e| e.to_string())),
        Some("classify") => classify(&args[1..]),
        _ => Err("expected `train` or `classify`".to_string()),
    };
    if let Err(message) = result {
        eprintln!("error: {message}\n{USAGE}");
        std::process::exit(2);
    }
}

fn parse_train(args: &[String]) -> Result<Options, String> {
    let mut o = Options { archs: vec![Arch::Mlp, Arch::Conv, Arch::ConvGru], epochs: 20, batch: 64, lr: 2e-3, threads: 10, max_per_word: None, seed: 7, save: true, augment: true };
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--no-save" {
            o.save = false;
            i += 1;
            continue;
        }
        if args[i] == "--no-augment" {
            o.augment = false;
            i += 1;
            continue;
        }
        let value = args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))?;
        let bad = || format!("{}: cannot use {value:?}", args[i]);
        match args[i].as_str() {
            "--model" => {
                o.archs = match value.as_str() {
                    "mlp" => vec![Arch::Mlp],
                    "conv" => vec![Arch::Conv],
                    "conv+gru" | "gru" => vec![Arch::ConvGru],
                    "all" => vec![Arch::Mlp, Arch::Conv, Arch::ConvGru],
                    _ => return Err(bad()),
                }
            }
            "--epochs" => o.epochs = value.parse().map_err(|_| bad())?,
            "--batch" => o.batch = value.parse().map_err(|_| bad())?,
            "--lr" => o.lr = value.parse().map_err(|_| bad())?,
            "--threads" => o.threads = value.parse().map_err(|_| bad())?,
            "--max-per-word" => o.max_per_word = Some(value.parse().map_err(|_| bad())?),
            "--seed" => o.seed = value.parse().map_err(|_| bad())?,
            other => return Err(format!("unknown option {other:?}")),
        }
        i += 2;
    }
    if o.epochs == 0 || o.batch == 0 || o.threads == 0 {
        return Err("epochs, batch and threads must be at least 1".to_string());
    }
    Ok(o)
}

/// Load the dataset, compute every clip's features once, and sort them into
/// the three parts of the split.
fn load_sets(options: &Options) -> std::io::Result<[Set; 3]> {
    let started = Instant::now();
    let data = SpeechCommands::load_with(&dataset_dir(), &LoadOptions { threads: options.threads, max_per_word: options.max_per_word, only: None })?;
    let (n_train, n_valid, n_test) = data.split_sizes();
    let (s_train, s_valid, s_test) = data.speakers_per_split();
    println!("loaded {} clips in {:.1} s ({:.2} GB of 16-bit audio)", data.clips.len(), started.elapsed().as_secs_f32(), data.audio_bytes() as f64 / 1e9);
    println!("  train      {n_train:>6} clips, {s_train:>4} speakers");
    println!("  validation {n_valid:>6} clips, {s_valid:>4} speakers");
    println!("  test       {n_test:>6} clips, {s_test:>4} speakers");

    let started = Instant::now();
    let front_end = FrontEnd::new();
    let features: Vec<Mat> = parallel_map(&data.clips, options.threads, |_, clip| {
        let mut features = front_end.extractor().log_mel(&clip.audio_one_second());
        normalize(&mut features);
        features
    });
    let bytes: usize = features.iter().map(|m| m.data.len() * 4).sum();
    println!("computed {} x {} log-mel features for every clip in {:.1} s ({:.2} GB)\n", features[0].rows, features[0].cols, started.elapsed().as_secs_f32(), bytes as f64 / 1e9);

    let mut sets = [Split::Train, Split::Validation, Split::Test].map(|_| Set { features: Vec::new(), labels: Vec::new() });
    for (clip, features) in data.clips.iter().zip(features) {
        let set = &mut sets[match clip.split {
            Split::Train => 0,
            Split::Validation => 1,
            Split::Test => 2,
        }];
        set.features.push(features);
        set.labels.push(clip.word);
    }
    Ok(sets)
}

/// Slide a clip's features earlier or later by `shift` frames, repeating
/// the edge frame into the space that opens up.
///
/// A cheap stand-in for "the speaker started a little earlier or later".
/// The convolution and the GRU hardly care; the MLP, which has a separate
/// weight for every frame, cares a great deal, and without this trick it
/// simply memorises where each training clip happened to start.
fn shift_in_time(features: &Mat, shift: i64) -> Mat {
    let last = features.rows as i64 - 1;
    Mat::from_fn(features.rows, features.cols, |r, c| features[((r as i64 - shift).clamp(0, last) as usize, c)])
}

/// Hide a random stretch of time and a random band of frequencies by
/// setting them to zero (which, after normalization, is each band's average
/// value): up to `max_frames` consecutive frames and up to `max_bands`
/// neighbouring mel bands.
///
/// The network sees every training clip once per epoch, and after a few
/// epochs it starts to recognize the *clips* rather than the words. Hiding
/// a different part each time makes every epoch's version of a clip a
/// little different and forces the network to use all of the evidence, not
/// one tell-tale corner of the spectrogram. (The idea is known as
/// SpecAugment; Park et al., 2019.)
fn mask_random_patch(features: &mut Mat, max_frames: usize, max_bands: usize, rng: &mut Rng) {
    let frames = rng.below(max_frames + 1);
    let first_frame = rng.below(features.rows - frames + 1);
    for r in first_frame..first_frame + frames {
        features.row_mut(r).fill(0.0);
    }
    let bands = rng.below(max_bands + 1);
    let first_band = rng.below(features.cols - bands + 1);
    for r in 0..features.rows {
        features.row_mut(r)[first_band..first_band + bands].fill(0.0);
    }
}

/// The predicted class of every clip in `set`.
fn predict(model: &Model, arch: Arch, set: &Set, threads: usize) -> Vec<usize> {
    parallel_map(&set.features, threads, |_, features| argmax(model.predict(&arch.shape_input(features)).row(0)))
}

fn accuracy(predicted: &[usize], labels: &[usize]) -> f32 {
    predicted.iter().zip(labels).filter(|(p, l)| p == l).count() as f32 / labels.len().max(1) as f32
}

/// Train one architecture and measure it.
fn train_one(arch: Arch, sets: &[Set; 3], options: &Options) -> RunResult {
    let [train, valid, test] = sets;
    let mut rng = Rng::new(options.seed);
    let mut model = arch.build(&mut rng);
    println!("== {} ==", arch.name());
    print!("{}", model.summary());

    let steps_per_epoch = train.labels.len().div_ceil(options.batch);
    let schedule = LrSchedule { peak_lr: options.lr, final_lr: options.lr / 50.0, warmup_steps: 200.min(steps_per_epoch), total_steps: options.epochs * steps_per_epoch };
    let mut adam = Adam::new(options.lr);
    let mut order: Vec<usize> = (0..train.labels.len()).collect();
    let mut result = RunResult {
        arch,
        params: model.num_params(),
        train_loss: Vec::new(),
        train_acc: Vec::new(),
        valid_acc: Vec::new(),
        best_epoch: 0,
        test_acc: 0.0,
        seconds_per_epoch: 0.0,
        model: model.clone(),
    };
    let mut step = 0;
    let mut training_seconds = 0.0;

    for epoch in 1..=options.epochs {
        let started = Instant::now();
        rng.shuffle(&mut order);
        let (mut loss_sum, mut correct, mut seen) = (0.0f64, 0usize, 0usize);
        for batch in order.chunks(options.batch) {
            // Prepare this minibatch's inputs: shift each clip by up to
            // 100 ms either way, hide a random patch of it, and flatten it
            // if the model is the MLP.
            let inputs: Vec<Mat> = batch
                .iter()
                .map(|&i| {
                    if !options.augment {
                        return arch.shape_input(&train.features[i]);
                    }
                    let mut features = shift_in_time(&train.features[i], rng.below(21) as i64 - 10);
                    mask_random_patch(&mut features, 12, 6, &mut rng);
                    arch.shape_input(&features)
                })
                .collect();
            let targets: Vec<[usize; 1]> = batch.iter().map(|&i| [train.labels[i]]).collect();
            let pairs: Vec<(&Mat, &[usize])> = inputs.iter().zip(&targets).map(|(x, t)| (x, &t[..])).collect();

            // The loss function also counts correct answers on the side, so
            // we get a training accuracy without a second forward pass.
            let hits = std::sync::atomic::AtomicUsize::new(0);
            let mut outcome = batch_gradients(&model, &pairs, options.threads, |logits: &Mat, target: &[usize]| {
                if argmax(logits.row(0)) == target[0] {
                    hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                cross_entropy(logits, target)
            });
            clip_global_norm(&mut outcome.grads, 5.0);
            adam.lr = schedule.lr_at(step);
            adam.step(&mut model, &outcome.grads);
            step += 1;

            loss_sum += outcome.loss as f64 * outcome.used as f64;
            correct += hits.into_inner();
            seen += outcome.used;
        }
        let epoch_seconds = started.elapsed().as_secs_f32();
        training_seconds += epoch_seconds;

        let valid_acc = accuracy(&predict(&model, arch, valid, options.threads), &valid.labels);
        result.train_loss.push((loss_sum / seen as f64) as f32);
        result.train_acc.push(correct as f32 / seen as f32);
        result.valid_acc.push(valid_acc);
        // Keep a copy of the weights whenever validation accuracy improves:
        // the last epoch is not always the best one.
        let is_best = result.best_epoch == 0 || valid_acc > result.valid_acc[result.best_epoch - 1];
        if is_best {
            result.best_epoch = epoch;
            result.model = model.clone();
        }
        println!(
            "epoch {epoch:>2}   loss {:.4}   train {:>5.2}%   validation {:>5.2}%{}   lr {:.5}   {:.1} s ({:.0} clips/s)",
            result.train_loss[epoch - 1],
            100.0 * result.train_acc[epoch - 1],
            100.0 * valid_acc,
            if is_best { " *" } else { "  " },
            adam.lr,
            epoch_seconds,
            seen as f32 / epoch_seconds
        );
    }

    // The test set is touched exactly once, with the weights that did best
    // on validation.
    result.seconds_per_epoch = training_seconds / options.epochs as f32;
    result.test_acc = accuracy(&predict(&result.model, arch, test, options.threads), &test.labels);
    println!("best validation accuracy {:.2}% at epoch {}; test accuracy with those weights: {:.2}%\n", 100.0 * result.valid_acc[result.best_epoch - 1], result.best_epoch, 100.0 * result.test_acc);
    result
}

fn train_all(options: &Options) -> std::io::Result<()> {
    let sets = load_sets(options)?;
    let results: Vec<RunResult> = options.archs.iter().map(|&arch| train_one(arch, &sets, options)).collect();

    println!("| model    | parameters | best validation | test accuracy | time per epoch |");
    println!("|----------|-----------:|----------------:|--------------:|---------------:|");
    for r in &results {
        println!("| {:<8} | {:>10} | {:>14.2}% | {:>12.2}% | {:>12.1} s |", r.arch.name(), r.params, 100.0 * r.valid_acc[r.best_epoch - 1], 100.0 * r.test_acc, r.seconds_per_epoch);
    }
    println!();

    // Look closer at the last model trained (conv+gru when all three ran).
    let Some(best) = results.last() else { return Ok(()) };
    let test = &sets[2];
    let predicted = predict(&best.model, best.arch, test, options.threads);
    let mut confusion = Mat::zeros(WORDS.len(), WORDS.len());
    for (&p, &l) in predicted.iter().zip(&test.labels) {
        confusion[(l, p)] += 1.0;
    }
    print_confusions(&confusion, best.arch);

    if options.save && best.arch == Arch::ConvGru {
        std::fs::create_dir_all(models_dir())?;
        let path = WordClassifier::default_path();
        best.model.save(&path)?;
        println!("\nsaved the {} model to {}", best.arch.name(), path.display());
    }
    if options.save && results.len() == 3 {
        save_figures(&results, &confusion)?;
    }
    Ok(())
}

/// Accuracy per word, and the pairs that are mixed up most often.
fn print_confusions(confusion: &Mat, arch: Arch) {
    let mut per_word: Vec<(f32, &str)> = (0..WORDS.len()).map(|w| (confusion[(w, w)] / confusion.row(w).iter().sum::<f32>().max(1.0), WORDS[w])).collect();
    per_word.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!("{}: hardest words on the test set: {}", arch.name(), per_word.iter().take(6).map(|(acc, word)| format!("{word} {:.1}%", 100.0 * acc)).collect::<Vec<_>>().join(", "));
    println!("{}: easiest words on the test set: {}", arch.name(), per_word.iter().rev().take(6).map(|(acc, word)| format!("{word} {:.1}%", 100.0 * acc)).collect::<Vec<_>>().join(", "));

    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for said in 0..WORDS.len() {
        for heard in 0..WORDS.len() {
            if said != heard && confusion[(said, heard)] > 0.0 {
                pairs.push((confusion[(said, heard)], said, heard));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("most frequent mistakes (said -> heard, count of test clips):");
    for (count, said, heard) in pairs.iter().take(12) {
        println!("  {:<8} -> {:<8} {:>3}   ({:.1}% of the \"{}\" clips)", WORDS[*said], WORDS[*heard], count, 100.0 * count / confusion.row(*said).iter().sum::<f32>(), WORDS[*said]);
    }
}

fn save_figures(results: &[RunResult], confusion: &Mat) -> std::io::Result<()> {
    let colors = [Color::Ink48, Color::Amber, Color::Green];
    let epochs: Vec<f32> = (1..=results[0].valid_acc.len()).map(|e| e as f32).collect();
    let percent = |values: &[f32]| values.iter().map(|v| 100.0 * v).collect::<Vec<f32>>();

    // Training curve: validation accuracy (solid) and training accuracy
    // (dashed, kept out of the legend) per model.
    let mut accuracy_plot = LinePlot::new("accuracy on validation clips (solid) and on training clips as seen during training (dashed)").y_label("accuracy (%)").x_label("epoch").y_range(0.0, 100.0).height(260.0);
    let mut loss_plot = LinePlot::new("training loss").y_label("cross-entropy").x_label("epoch").height(150.0);
    for (r, &color) in results.iter().zip(&colors) {
        accuracy_plot = accuracy_plot
            .series(Series::new(r.arch.name(), &epochs, &percent(&r.valid_acc)).color(color).width(2.0).dots())
            .series(Series::new("", &epochs, &percent(&r.train_acc)).color(color).dashed());
        loss_plot = loss_plot.series(Series::new(r.arch.name(), &epochs, &r.train_loss).color(color));
    }
    Figure::new("Three networks learning 35 words")
        .desc("Validation accuracy (solid) and training accuracy (dashed) per epoch for an MLP, a convolutional network and a convolution plus GRU network, with the training loss below.")
        .panel(accuracy_plot)
        .panel(loss_plot.legend(false))
        .share_x()
        .save("../figures/ch07-training-curve.svg")?;

    // Architecture comparison.
    let labels: Vec<String> = results.iter().map(|r| format!("{} ({}k weights)", r.arch.name(), (r.params + 500) / 1000)).collect();
    let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    let test_acc = percent(&results.iter().map(|r| r.test_acc).collect::<Vec<_>>());
    BarChart::new("Test accuracy on 35 words: layers that understand time win", &labels, &test_acc)
        .y_label("test accuracy (%)")
        .y_range(0.0, 100.0)
        .show_values(1)
        .highlight(2, Color::Green)
        .highlight(1, Color::Amber)
        .height(260.0)
        .save("../figures/ch07-architectures.svg")?;

    // Confusion matrix, each row scaled to sum to 1 and then square-rooted
    // so that the rare mistakes are visible next to the dominant diagonal.
    // Below it, the twelve most frequent mistakes spelled out.
    let scaled = Mat::from_fn(WORDS.len(), WORDS.len(), |r, c| (confusion[(r, c)] / confusion.row(r).iter().sum::<f32>().max(1.0)).sqrt());
    let short: Vec<String> = WORDS.iter().map(|w| w.chars().take(3).collect()).collect();
    let short: Vec<&str> = short.iter().map(String::as_str).collect();
    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for said in 0..WORDS.len() {
        for heard in (0..WORDS.len()).filter(|&heard| heard != said) {
            pairs.push((100.0 * confusion[(said, heard)] / confusion.row(said).iter().sum::<f32>().max(1.0), said, heard));
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    pairs.truncate(12);
    let pair_labels: Vec<String> = pairs.iter().map(|&(_, said, heard)| format!("{}\u{2192}{}", WORDS[said], WORDS[heard])).collect();
    let pair_labels: Vec<&str> = pair_labels.iter().map(String::as_str).collect();
    let pair_values: Vec<f32> = pairs.iter().map(|p| p.0).collect();
    Figure::new("Which words the conv+gru network mixes up (test set)")
        .desc("A 35 by 35 confusion matrix with a strong diagonal, and a bar chart of the twelve most frequent confusions, such as tree heard as three and follow heard as four.")
        .panel(
            Heatmap::new("what was said (rows) against what was heard (columns, first three letters); darkness is the square root of the fraction", &scaled)
                .row_labels(&WORDS)
                .col_labels(&short)
                .y_label("said")
                .x_label("heard")
                .value_range(0.0, 1.0)
                .height(640.0),
        )
        .panel(BarChart::new("the twelve most frequent mistakes: said \u{2192} heard", &pair_labels, &pair_values).y_label("% of that word's clips").show_values(1).height(150.0))
        .save("../figures/ch07-confusion.svg")?;
    println!("wrote ../figures/ch07-training-curve.svg, ch07-architectures.svg, ch07-confusion.svg");
    Ok(())
}

/// `classify`: print the five most probable words for each file.
fn classify(args: &[String]) -> Result<(), String> {
    let mut model_path = WordClassifier::default_path();
    let mut files = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model-file" {
            model_path = args.get(i + 1).ok_or("--model-file needs a path")?.into();
            i += 2;
        } else {
            files.push(&args[i]);
            i += 1;
        }
    }
    if files.is_empty() {
        return Err("classify needs at least one WAV file".to_string());
    }
    let classifier = WordClassifier::load(&model_path).map_err(|e| format!("{e} (train it first with `ch07_words train`)"))?;
    for file in files {
        let audio = load_audio_16k(file).map_err(|e| e.to_string())?;
        println!("{file}  ({:.2} s)", audio.len() as f32 / 16000.0);
        for (word, probability) in classifier.top_k(&WordClassifier::fit_one_second(&audio), 5) {
            println!("  {word:<9} {:>5.1}%  {}", 100.0 * probability, "#".repeat((40.0 * probability).round() as usize));
        }
    }
    Ok(())
}
