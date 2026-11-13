//! Chapter 7: how fast is our network?
//!
//! Builds a model shaped like the recognizer we are heading for,
//!
//! ```text
//! Conv1d(40 -> 64, kernel 5, stride 2) -> ReLU -> BiGru(64 -> 2 x 96) -> Linear(192 -> 28)
//! ```
//!
//! and measures how many utterances per second it can push through a full
//! training step's worth of work (forward pass, loss, backward pass), first
//! on one thread and then on all of them with `batch_gradients`. The input
//! is random noise and the targets are random: speed does not care.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example ch07_bench
//! cargo run --release --example ch07_bench -- --hidden 128 --frames 150 --threads 8
//! ```

use speech::mat::Mat;
use speech::nn::{batch_gradients, cross_entropy, default_threads, Grads, Layer, Model};
use speech::rng::Rng;
use std::time::Instant;

struct Options {
    frames: usize,
    features: usize,
    conv_channels: usize,
    hidden: usize,
    classes: usize,
    utterances: usize,
    threads: usize,
    seconds: f32,
}

const USAGE: &str = "usage: ch07_bench [--frames 100] [--features 40] [--conv 64] [--hidden 96] [--classes 28]
                  [--utterances 256] [--threads N] [--seconds 3]";

fn parse_options() -> Result<Options, String> {
    let mut o = Options { frames: 100, features: 40, conv_channels: 64, hidden: 96, classes: 28, utterances: 256, threads: default_threads(), seconds: 3.0 };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))?;
        let number = || value.parse::<usize>().map_err(|_| format!("{} needs a whole number, got {value:?}", args[i]));
        match args[i].as_str() {
            "--frames" => o.frames = number()?,
            "--features" => o.features = number()?,
            "--conv" => o.conv_channels = number()?,
            "--hidden" => o.hidden = number()?,
            "--classes" => o.classes = number()?,
            "--utterances" => o.utterances = number()?,
            "--threads" => o.threads = number()?,
            "--seconds" => o.seconds = value.parse().map_err(|_| format!("--seconds needs a number, got {value:?}"))?,
            other => return Err(format!("unknown option {other:?}")),
        }
        i += 2;
    }
    if o.frames < 5 || o.utterances == 0 || o.threads == 0 {
        return Err("need at least 5 frames, 1 utterance and 1 thread".to_string());
    }
    Ok(o)
}

fn main() {
    let o = match parse_options() {
        Ok(o) => o,
        Err(message) => {
            eprintln!("error: {message}\n{USAGE}");
            std::process::exit(2);
        }
    };

    let mut rng = Rng::new(7);
    let model = Model::new(vec![
        Layer::conv1d(o.features, o.conv_channels, 5, 2, 2, &mut rng),
        Layer::Relu,
        Layer::bigru(o.conv_channels, o.hidden, &mut rng),
        Layer::linear(2 * o.hidden, o.classes, &mut rng),
    ]);
    print!("{}", model.summary());

    // Random "utterances" and a random class for each output frame.
    let out_frames = model.out_len(o.frames);
    let inputs: Vec<Mat> = (0..o.utterances).map(|_| Mat::from_fn(o.frames, o.features, |_, _| rng.normal())).collect();
    let targets: Vec<Vec<usize>> = (0..o.utterances).map(|_| (0..out_frames).map(|_| rng.below(o.classes)).collect()).collect();
    let batch: Vec<(&Mat, &[usize])> = inputs.iter().zip(&targets).map(|(x, t)| (x, &t[..])).collect();
    println!("\n{} utterances of {} frames x {} features ({} output frames each)\n", o.utterances, o.frames, o.features, out_frames);

    // Where does the time go? One thread, forward and backward timed apart.
    let mut grads = Grads::zeros_like(&model);
    let (mut forward_s, mut backward_s, mut done) = (0.0f64, 0.0f64, 0usize);
    let started = Instant::now();
    while started.elapsed().as_secs_f32() < o.seconds {
        for &(x, target) in &batch {
            let t0 = Instant::now();
            let cache = model.forward(x);
            let t1 = Instant::now();
            let (_, d_logits) = cross_entropy(cache.output(), target);
            model.backward(&cache, &d_logits, &mut grads);
            forward_s += (t1 - t0).as_secs_f64();
            backward_s += t1.elapsed().as_secs_f64();
            done += 1;
        }
    }
    let single = done as f64 / (forward_s + backward_s);
    println!("1 thread:   {single:>8.0} utterances/s   (forward {:.2} ms, loss + backward {:.2} ms per utterance)", 1e3 * forward_s / done as f64, 1e3 * backward_s / done as f64);

    // The same work through the threaded helper, as training will do it.
    let (mut done, started) = (0usize, Instant::now());
    while started.elapsed().as_secs_f32() < o.seconds {
        let out = batch_gradients(&model, &batch, o.threads, cross_entropy);
        done += out.used;
    }
    let threaded = done as f64 / started.elapsed().as_secs_f64();
    println!("{:>2} threads: {threaded:>8.0} utterances/s   ({:.1}x)", o.threads, threaded / single);

    let audio_seconds = o.frames as f64 * 0.010; // 10 ms per frame
    println!("\nAt 10 ms per frame an utterance is {audio_seconds:.1} s of audio, so training runs at {:.0}x real time,", threaded * audio_seconds);
    println!("and one pass over 85,000 such clips would take about {:.0} s.", 85_000.0 / threaded);
}
