//! Chapter 12: transcribe a WAV file with Whisper tiny.en, using nothing but
//! the code in `src/whisper/`.
//!
//! ```text
//! cargo run --release --example ch12_whisper -- path/to/audio.wav
//! cargo run --release --example ch12_whisper -- --verbose audio.wav
//! cargo run --release --example ch12_whisper -- --no-cache audio.wav
//! ```
//!
//! The audio must be 16 kHz, mono, 16-bit PCM:
//! `ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le out.wav`.
//! The model files are looked for in `$SPEECH_DATA/whisper-tiny.en`
//! (`../data/whisper-tiny.en` by default).

use speech::mat::Mat;
use speech::whisper::transcribe::{Timings, WindowResult};
use speech::whisper::{audio, model_dir, DecodeOptions, Whisper};
use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

/// How many decoding steps `--verbose` shows candidates for.
const TRACED_STEPS: usize = 6;

fn usage() -> ExitCode {
    eprintln!("usage: ch12_whisper [--verbose] [--no-cache] audio.wav [more.wav ...]");
    eprintln!("  --verbose   print token ids, per-step candidates and tensor statistics");
    eprintln!("  --no-cache  decode without the key/value cache (slow; same result)");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let mut options = DecodeOptions::default();
    let mut verbose = false;
    let mut paths = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--verbose" => verbose = true,
            "--no-cache" => options.use_cache = false,
            flag if flag.starts_with('-') => return usage(),
            _ => paths.push(arg),
        }
    }
    if paths.is_empty() {
        return usage();
    }
    if verbose {
        options.trace_steps = TRACED_STEPS;
    }

    let dir = model_dir();
    let started = Instant::now();
    let model = match Whisper::load(&dir) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("cannot load the model from {}: {error}", dir.display());
            eprintln!("(set SPEECH_DATA to the directory that contains whisper-tiny.en/)");
            return ExitCode::FAILURE;
        }
    };
    let load_time = started.elapsed();
    println!("loaded {} in {}", dir.display(), seconds(load_time));
    if verbose {
        let c = &model.config;
        println!(
            "  encoder: {} layers, decoder: {} layers, width {}, {} heads, {} mel bands, {} tokens",
            c.encoder_layers, c.decoder_layers, c.d_model, c.encoder_heads, c.n_mels, c.vocab_size
        );
        println!("  prompt: {}", describe_tokens(&model, &model.rules.prompt));
    }

    for path in &paths {
        let samples = match audio::read_wav_16k_mono(Path::new(path)) {
            Ok(samples) => samples,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        };
        let audio_seconds = samples.len() as f64 / 16_000.0;
        println!();
        println!("{path}: {audio_seconds:.2} s of audio");
        if verbose {
            print_statistics(&model, &samples);
        }

        let transcript = model.transcribe(&samples, &options);
        if verbose {
            for (index, window) in transcript.windows.iter().enumerate() {
                print_window_details(&model, index, window);
            }
        }
        println!();
        println!("  {}", transcript.text);
        println!();
        print_timings(&transcript.timings, audio_seconds);
    }
    ExitCode::SUCCESS
}

fn seconds(duration: Duration) -> String {
    format!("{:.3} s", duration.as_secs_f64())
}

/// Token ids with their text, e.g. `50257 <|startoftranscript|>, 50362 <|notimestamps|>`.
fn describe_tokens(model: &Whisper, tokens: &[usize]) -> String {
    let parts: Vec<String> = tokens.iter().map(|&t| format!("{t} {:?}", model.tokenizer.piece_text(t))).collect();
    parts.join(", ")
}

/// Mean, standard deviation, minimum and maximum of a matrix: the first
/// thing to look at when a model's output makes no sense.
fn statistics(name: &str, m: &Mat) {
    let n = m.data.len() as f64;
    let mean = m.data.iter().map(|&v| v as f64).sum::<f64>() / n;
    let variance = m.data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let (min, max) = m.data.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    println!("  {name:<28} {:>5} x {:<4} mean {mean:>8.4}  std {:>7.4}  min {min:>8.4}  max {max:>8.4}", m.rows, m.cols, variance.sqrt());
}

/// Tensor statistics along the path of the first window.
fn print_statistics(model: &Whisper, samples: &[f32]) {
    let first_window = &samples[..samples.len().min(speech::whisper::mel::CHUNK_SAMPLES)];
    let mel = model.front_end.log_mel(first_window);
    statistics("log-mel spectrogram", &mel);
    let mut x = model.encoder.stem(&mel);
    statistics("after convolutions+positions", &x);
    for (index, block) in model.encoder.blocks.iter().enumerate() {
        block.forward(&mut x);
        statistics(&format!("after encoder block {index}"), &x);
    }
    statistics("encoder output", &model.encoder.final_norm.forward(&x));
}

fn print_window_details(model: &Whisper, index: usize, window: &WindowResult) {
    println!();
    println!("  window {index}: {} tokens{}", window.decoded.tokens.len(), if window.decoded.finished { "" } else { " (hit the length limit)" });
    println!("  ids: {:?}", window.decoded.tokens);
    for step in &window.decoded.trace {
        let candidates: Vec<String> =
            step.top.iter().map(|&(token, p)| format!("{:?} {:.1}%", model.tokenizer.piece_text(token), 100.0 * p)).collect();
        println!("    position {:>2}: {}", step.position, candidates.join("  "));
    }
}

fn print_timings(timings: &Timings, audio_seconds: f64) {
    let total = timings.mel + timings.encoder + timings.decoder;
    let tokens_per_second = timings.decoder_steps as f64 / timings.decoder.as_secs_f64().max(1e-9);
    println!("  mel spectrogram  {:>9}", seconds(timings.mel));
    println!("  encoder          {:>9}", seconds(timings.encoder));
    println!("  decoder          {:>9}   {} tokens, {:.0} tokens/s", seconds(timings.decoder), timings.decoder_steps, tokens_per_second);
    println!("  total            {:>9}   {:.1}x faster than real time", seconds(total), audio_seconds / total.as_secs_f64().max(1e-9));
}
