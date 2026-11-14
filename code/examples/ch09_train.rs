//! Chapter 9: training the recognizer.
//!
//! Trains a character-level CTC model that listens to several words in a
//! row and spells them out:
//!
//! ```text
//! audio ─▶ log-mel (40 x T) ─▶ Conv1d, stride 2 ─▶ ReLU ─▶ BiGru ─▶ BiGru ─▶ Linear(28) ─▶ CTC
//!                                 (T/2 frames, 20 ms each)                    blank, space, a..z
//! ```
//!
//! There is no dataset of multi-word utterances, so every training example
//! is built on the spot from single-word clips (see `data/synth.rs`): a
//! fresh batch each step, never the same utterance twice.
//!
//! ```text
//! cargo run --release --example ch09_train                # the full run: 40,000 steps, about 40 minutes on 10 threads
//! cargo run --release --example ch09_train -- --quick     # 4,000 steps, about 5 minutes: already spells most words
//! ```
//!
//! The full run that produced `models/recognizer.bin` reached a word error
//! rate of 16% on the test set with greedy decoding (10% with the lexicon
//! decoder of chapter 10); the quick run reaches 22% (12%). The numbers a
//! run prints are reproducible bit for bit for a given `--threads`.
//!
//! While it runs it prints the loss every 100 steps, and at regular
//! intervals what it currently hears in three fixed validation utterances.
//! Watching those three lines go from nothing, to stray letters, to words
//! is the point of the chapter.

use speech::ctc::ctc_loss;
use speech::data::pipeline::FrontEnd;
use speech::data::speech_commands::{dataset_dir, parallel_map, LoadOptions, SpeechCommands, Split};
use speech::data::synth::{Corpus, SynthConfig, Synthesizer, TrimConfig, Utterance, EVAL_UTTERANCES};
use speech::decode::{best_path, greedy, log_softmax, Alphabet};
use speech::mat::Mat;
use speech::metrics::ErrorTally;
use speech::nn::{batch_gradients, clip_global_norm, Adam, Layer, LrSchedule, Model};
use speech::rng::Rng;
use std::io::Write as _;
use std::time::Instant;

const USAGE: &str = "usage: ch09_train [--quick] [--steps N] [--batch 32] [--lr 0.002] [--conv 96] [--hidden 128] [--layers 2]
                  [--threads 10] [--seed 9] [--eval-every 1000] [--max-words 5] [--max-per-word N]
                  [--out ../models/recognizer.bin] [--snapshots ../models/recognizer-posteriors.txt]
                  [--init model.bin]   (start from saved weights instead of random ones)";

struct Options {
    steps: usize,
    batch: usize,
    lr: f32,
    conv: usize,
    hidden: usize,
    layers: usize,
    threads: usize,
    seed: u64,
    eval_every: usize,
    max_words: usize,
    max_per_word: Option<usize>,
    /// Start from these saved weights instead of random ones.
    init: Option<String>,
    out: String,
    snapshots: String,
}

fn parse_options() -> Result<Options, String> {
    let mut o = Options {
        steps: 40_000,
        batch: 32,
        lr: 2e-3,
        conv: 96,
        hidden: 128,
        layers: 2,
        threads: 10,
        seed: 9,
        eval_every: 1000,
        max_words: 5,
        max_per_word: None,
        init: None,
        out: "../models/recognizer.bin".to_string(),
        snapshots: "../models/recognizer-posteriors.txt".to_string(),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--quick" {
            // A tenth of the steps: enough to get past the "gibberish"
            // stage and well into spelling.
            o.steps = 4000;
            o.eval_every = 500;
            i += 1;
            continue;
        }
        let value = args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))?;
        let bad = || format!("{}: cannot use {value:?}", args[i]);
        match args[i].as_str() {
            "--steps" => o.steps = value.parse().map_err(|_| bad())?,
            "--batch" => o.batch = value.parse().map_err(|_| bad())?,
            "--lr" => o.lr = value.parse().map_err(|_| bad())?,
            "--conv" => o.conv = value.parse().map_err(|_| bad())?,
            "--hidden" => o.hidden = value.parse().map_err(|_| bad())?,
            "--layers" => o.layers = value.parse().map_err(|_| bad())?,
            "--threads" => o.threads = value.parse().map_err(|_| bad())?,
            "--seed" => o.seed = value.parse().map_err(|_| bad())?,
            "--eval-every" => o.eval_every = value.parse().map_err(|_| bad())?,
            "--max-words" => o.max_words = value.parse().map_err(|_| bad())?,
            "--max-per-word" => o.max_per_word = Some(value.parse().map_err(|_| bad())?),
            "--init" => o.init = Some(value.clone()),
            "--out" => o.out = value.clone(),
            "--snapshots" => o.snapshots = value.clone(),
            other => return Err(format!("unknown option {other:?}")),
        }
        i += 2;
    }
    if o.steps == 0 || o.batch == 0 || o.threads == 0 || o.layers == 0 || o.eval_every == 0 {
        return Err("steps, batch, threads, layers and eval-every must be at least 1".to_string());
    }
    Ok(o)
}

/// The recognizer's network.
///
/// * The convolution looks at 5 frames (50 ms) at a time and steps 2 frames
///   forward per output, so the GRUs run at 50 frames per second instead of
///   100: half the work, and still several frames per letter.
/// * Each BiGru reads the utterance in both directions, so the decision
///   about a frame can use what comes after it as well as what came before.
///   The second one works on the first one's output.
/// * The linear layer turns each frame's state into 28 scores: blank,
///   space, a to z.
fn build_model(options: &Options, rng: &mut Rng) -> Model {
    let mut layers = vec![Layer::conv1d(FrontEnd::N_FEATURES, options.conv, 5, 2, 2, rng), Layer::Relu];
    let mut width = options.conv;
    for _ in 0..options.layers {
        layers.push(Layer::bigru(width, options.hidden, rng));
        width = 2 * options.hidden;
    }
    layers.push(Layer::linear(width, Alphabet::SIZE, rng));
    Model::new(layers)
}

/// An evaluation example: features computed once, plus the truth.
struct Example {
    features: Mat,
    text: String,
}

fn examples_from(utterances: &[Utterance], front_end: &FrontEnd, threads: usize) -> Vec<Example> {
    parallel_map(utterances, threads, |_, utt| Example { features: front_end.features(&utt.samples), text: utt.text() })
}

/// What the model currently hears in one example, by greedy decoding: the
/// most probable class of every frame, repeats merged, blanks dropped
/// (chapter 10 looks at this, and at better decoders, in detail).
fn hear(model: &Model, features: &Mat) -> String {
    let text = Alphabet::decode(&greedy(&log_softmax(&model.predict(features))));
    // A space spelled before the first word or after the last is not an error.
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Greedy-decode every example and count the errors.
fn evaluate(model: &Model, examples: &[Example], threads: usize) -> ErrorTally {
    let hypotheses = parallel_map(examples, threads, |_, example| hear(model, &example.features));
    let mut tally = ErrorTally::new();
    for (example, hypothesis) in examples.iter().zip(&hypotheses) {
        tally.add(&example.text, hypothesis);
    }
    tally
}

/// The most probable class of every output frame as one character each,
/// with `_` for the blank: the raw material greedy decoding collapses.
fn frame_string(model: &Model, features: &Mat) -> String {
    best_path(&log_softmax(&model.predict(features))).iter().map(|&class| if class == Alphabet::SPACE { '|' } else { Alphabet::symbol(class) }).collect()
}

/// Append the posteriors of one utterance to the snapshot file, so that a
/// figure can later show them sharpening. Plain text: a header line
/// `snapshot <step> <rows> <cols>`, then one line of probabilities per frame.
fn write_snapshot(file: &mut std::fs::File, model: &Model, features: &Mat, step: usize) -> std::io::Result<()> {
    let probabilities = log_softmax(&model.predict(features)).map(f32::exp);
    writeln!(file, "snapshot {step} {} {}", probabilities.rows, probabilities.cols)?;
    for t in 0..probabilities.rows {
        let row: Vec<String> = probabilities.row(t).iter().map(|p| format!("{p:.4}")).collect();
        writeln!(file, "{}", row.join(" "))?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let options = match parse_options() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("error: {message}\n{USAGE}");
            std::process::exit(2);
        }
    };
    let threads = options.threads;
    let front_end = FrontEnd::new();

    // ---- data ----------------------------------------------------------
    let started = Instant::now();
    let data = SpeechCommands::load_with(&dataset_dir(), &LoadOptions { threads, max_per_word: options.max_per_word, only: None })?;
    let noise = speech::data::speech_commands::load_background_noise(&dataset_dir())?;
    println!("loaded {} clips and {} noise recordings in {:.1} s", data.clips.len(), noise.len(), started.elapsed().as_secs_f32());
    let started = Instant::now();
    let corpus = Corpus::new(&data, &noise, &TrimConfig::default(), threads);
    assert!(corpus.speakers_are_disjoint(), "a speaker appears in more than one part of the split");
    for (name, part) in [("train", &corpus.train), ("validation", &corpus.validation), ("test", &corpus.test)] {
        println!(
            "  {name:<10} {:>6} words from {:>4} speakers, {:>5.1} min of speech after trimming ({} clips rejected), {:>4.0} s of noise",
            part.words.len(),
            part.words.num_speakers(),
            part.words.total_seconds() / 60.0,
            part.words.rejected,
            part.noise.tracks.iter().map(|(_, s)| s.len()).sum::<usize>() as f32 / 16000.0
        );
    }
    println!("trimmed every clip to its word in {:.1} s", started.elapsed().as_secs_f32());

    // Fixed evaluation sets, and the plain one-second test clips.
    let started = Instant::now();
    let validation = examples_from(&corpus.validation_set(EVAL_UTTERANCES, threads), &front_end, threads);
    let test = examples_from(&corpus.test_set(EVAL_UTTERANCES, threads), &front_end, threads);
    let single_clips: Vec<_> = data.split(Split::Test).collect();
    let single: Vec<Example> = parallel_map(&single_clips, threads, |_, clip| Example { features: front_end.features(&clip.audio()), text: speech::data::speech_commands::WORDS[clip.word].to_string() });
    drop(single_clips);
    drop(data); // the raw clips are no longer needed: 3 GB back
    println!("built {} validation and {} test utterances, and features of {} single-word test clips, in {:.1} s", validation.len(), test.len(), single.len(), started.elapsed().as_secs_f32());
    let quick_validation = &validation[..validation.len().min(500)];

    // Three utterances to watch, with 2, 3 and 4 words, and one of them to
    // take posterior snapshots of.
    let watched: Vec<&Example> = [2, 3, 4].iter().filter_map(|&n| validation.iter().find(|e| e.text.split(' ').count() == n)).collect();
    let snapshot_index = validation.iter().position(|e| e.text.split(' ').count() == 3 && e.features.rows < 280).unwrap_or(0);
    let snapshot_steps = [0, 100, 300, 600, 1000, 2000, 4000, options.steps];
    let mut snapshot_file = std::fs::File::create(&options.snapshots)?;
    writeln!(snapshot_file, "# posteriors of validation utterance {snapshot_index} (seed {}) at several points of training", speech::data::synth::VALIDATION_SEED)?;
    writeln!(snapshot_file, "utterance {snapshot_index}")?;
    writeln!(snapshot_file, "text {}", validation[snapshot_index].text)?;

    // ---- model ---------------------------------------------------------
    let mut rng = Rng::new(options.seed);
    let mut model = match &options.init {
        Some(path) => Model::load(path)?,
        None => build_model(&options, &mut rng),
    };
    println!();
    print!("{}", model.summary());
    let schedule = LrSchedule { peak_lr: options.lr, final_lr: options.lr / 20.0, warmup_steps: 500.min(options.steps / 4).max(1), total_steps: options.steps };
    let mut adam = Adam::new(options.lr);
    let recipe = SynthConfig { max_words: options.max_words, ..SynthConfig::default() };
    println!("\nrecipe: {recipe:?}");
    println!("training: {} steps of {} utterances, Adam, peak learning rate {} (warm-up {} steps, cosine decay to {:.5}), gradient clipping at norm {CLIP}, {} threads\n", options.steps, options.batch, options.lr, schedule.warmup_steps, schedule.final_lr, threads);
    let synthesizer = Synthesizer::new(&corpus.train.words, &corpus.train.noise, recipe);

    // ---- the loop ------------------------------------------------------
    let training_started = Instant::now();
    let (mut window_started, mut window_loss, mut window_norm, mut window_used, mut window_skipped) = (Instant::now(), 0.0f64, 0.0f64, 0usize, 0usize);
    let batch_slots: Vec<usize> = (0..options.batch).collect();
    for step in 0..=options.steps {
        // Checkpoint duties first, so that step 0 shows the untrained model.
        if snapshot_steps.contains(&step) {
            write_snapshot(&mut snapshot_file, &model, &validation[snapshot_index].features, step)?;
        }
        if step % options.eval_every == 0 || step == options.steps || [100, 200, 300, 500, 700].contains(&step) {
            let tally = evaluate(&model, quick_validation, threads);
            println!("--- step {step}: validation CER {:.1}%  WER {:.1}%  ({} utterances, greedy, {:.1} min elapsed)", 100.0 * tally.cer(), 100.0 * tally.wer(), tally.utterances, training_started.elapsed().as_secs_f32() / 60.0);
            for example in &watched {
                println!("      truth: {:<28} heard: {}", example.text, hear(&model, &example.features));
            }
            println!("      frames: {}", frame_string(&model, &watched[0].features));
            model.save(&options.out)?;
        }
        if step == options.steps {
            break;
        }

        // Build this step's batch: synthesize and extract features, in parallel.
        let first = (step * options.batch) as u64;
        let batch: Vec<(Mat, Vec<usize>)> = parallel_map(&batch_slots, threads, |_, &slot| {
            let utterance = synthesizer.utterance_at(options.seed, first + slot as u64);
            (front_end.features(&utterance.samples), Alphabet::encode(&utterance.text()))
        });
        let pairs: Vec<(&Mat, &[usize])> = batch.iter().map(|(x, target)| (x, &target[..])).collect();

        // Forward, CTC loss, backward for every utterance; average; step.
        let mut outcome = batch_gradients(&model, &pairs, threads, ctc_loss);
        let norm = clip_global_norm(&mut outcome.grads, CLIP);
        adam.lr = schedule.lr_at(step);
        adam.step(&mut model, &outcome.grads);

        window_loss += outcome.loss as f64 * outcome.used as f64;
        window_norm += norm as f64;
        window_used += outcome.used;
        window_skipped += outcome.skipped;
        if (step + 1) % LOG_EVERY == 0 || step == 0 {
            let steps_in_window = if step == 0 { 1 } else { LOG_EVERY };
            println!(
                "step {:>6}   loss {:>7.3}   lr {:.5}   grad norm {:>6.2}   {:>5.0} utt/s{}",
                step + 1,
                window_loss / window_used.max(1) as f64,
                adam.lr,
                window_norm / steps_in_window as f64,
                (window_used + window_skipped) as f32 / window_started.elapsed().as_secs_f32(),
                if window_skipped > 0 { format!("   ({window_skipped} skipped)") } else { String::new() }
            );
            (window_started, window_loss, window_norm, window_used, window_skipped) = (Instant::now(), 0.0, 0.0, 0, 0);
        }
    }
    let minutes = training_started.elapsed().as_secs_f32() / 60.0;
    println!("\ntrained on {} utterances in {minutes:.1} minutes", options.steps * options.batch);

    // ---- final numbers (greedy decoding; chapter 10 does better) ---------
    println!("\n| set (greedy decoding)                 | utterances |   CER |   WER | exact match |");
    println!("|---------------------------------------|-----------:|------:|------:|------------:|");
    for (name, examples) in [("synthesized validation", &validation), ("synthesized test", &test), ("single-word test clips, as recorded", &single)] {
        let tally = evaluate(&model, examples, threads);
        println!("| {name:<37} | {:>10} | {:>4.1}% | {:>4.1}% | {:>10.1}% |", tally.utterances, 100.0 * tally.cer(), 100.0 * tally.wer(), 100.0 * tally.exact_match_rate());
    }
    model.save(&options.out)?;
    println!("\nsaved the model to {} and posterior snapshots to {}", options.out, options.snapshots);
    Ok(())
}

/// Gradients longer than this are scaled down to this length.
const CLIP: f32 = 5.0;
/// Print a progress line every this many steps.
const LOG_EVERY: usize = 100;
