//! Chapter 1: sound is just numbers.
//!
//! Reads a WAV file with our own parser, prints what is inside, and draws the
//! waveform in the terminal. With `--beep OUT.wav` it also writes a one-second
//! 440 Hz sine wave you can listen to.
//!
//! ```text
//! cargo run --release --example ch01_wav
//! cargo run --release --example ch01_wav -- ../samples/yes.wav
//! cargo run --release --example ch01_wav -- --beep beep.wav
//! ```

use speech::plot::{waveform_ascii, LinePlot, Series};
use speech::wav::{Encoding, Wav};
use std::f32::consts::PI;
use std::io;
use std::process::ExitCode;

const USAGE: &str = "usage: cargo run --release --example ch01_wav -- [FILE.wav] [--beep OUT.wav] [--figures DIR]";
const DEFAULT_CLIP: &str = "../samples/seven.wav";

fn main() -> ExitCode {
    // Parse the command line by hand: one optional file, two optional flags.
    let mut clip: Option<String> = None;
    let mut beep_path: Option<String> = None;
    let mut figures_dir: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--beep" => beep_path = args.next(),
            "--figures" => figures_dir = args.next(),
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            flag if flag.starts_with('-') => {
                eprintln!("unknown option {flag}\n{USAGE}");
                return ExitCode::FAILURE;
            }
            _ => clip = Some(arg),
        }
    }
    // The book's figures are drawn from the default clip. If you point the
    // example at your own file, figures are only written where you ask.
    if clip.is_none() && figures_dir.is_none() {
        figures_dir = Some("../figures".to_string());
    }
    let clip = clip.unwrap_or_else(|| DEFAULT_CLIP.to_string());

    match run(&clip, beep_path.as_deref(), figures_dir.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn run(clip: &str, beep_path: Option<&str>, figures_dir: Option<&str>) -> io::Result<()> {
    let (wav, format) = Wav::read_with_format(clip)?;

    println!("file         {clip}");
    let encoding = match format.encoding {
        Encoding::Pcm => "integer PCM",
        Encoding::Float => "floating point",
    };
    println!("stored as    {}-bit {encoding}, {} channel(s)", format.bits_per_sample, format.channels);
    println!("sample rate  {} Hz", wav.sample_rate);
    println!("samples      {}", wav.samples.len());
    println!("duration     {:.3} s", wav.duration_secs());
    if wav.samples.is_empty() {
        println!("(the file has a header but no audio)");
        return Ok(());
    }

    let min = wav.samples.iter().copied().fold(f32::INFINITY, f32::min);
    let max = wav.samples.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = wav.samples.iter().sum::<f32>() / wav.samples.len() as f32;
    println!("min / max    {min:+.4} / {max:+.4}");
    println!("mean         {mean:+.5}   (close to zero: air pressure wobbles around its resting value)");

    // The loudest 20 ms: somewhere in the middle of a vowel.
    let window = (wav.sample_rate as usize / 50).min(wav.samples.len());
    let loudest = loudest_stretch(&wav.samples, window);
    let shown = 10.min(window);
    println!("\n{shown} samples from the loudest part, starting at sample {loudest}:");
    let as_text: Vec<String> = wav.samples[loudest..loudest + shown].iter().map(|s| format!("{s:+.4}")).collect();
    println!("  {}", as_text.join(" "));
    let as_int: Vec<String> =
        wav.samples[loudest..loudest + shown].iter().map(|s| format!("{:+7}", (s * 32768.0).round() as i32)).collect();
    println!("  {}   (as 16-bit integers)", as_int.join(""));

    println!("\nwaveform ({:.2} s across, +1 at the top, -1 at the bottom; scaled to the peak if quiet):\n", wav.duration_secs());
    print!("{}", waveform_ascii(&wav.samples, 72, 9));

    if let Some(path) = beep_path {
        let beep = sine_beep(440.0, 1.0, 16000);
        beep.write(path)?;
        println!(
            "\nwrote {path}: {:.1} s of 440 Hz at {} Hz, {} samples",
            beep.duration_secs(),
            beep.sample_rate,
            beep.samples.len()
        );
        // Read our own file back, as a check that reader and writer agree.
        let back = Wav::read(path)?;
        let worst = beep.samples.iter().zip(&back.samples).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!(
            "read it back: {} samples, largest difference {worst:.6} (one 16-bit step is {:.6})",
            back.samples.len(),
            1.0 / 32768.0
        );
    }

    if let Some(dir) = figures_dir {
        let name = std::path::Path::new(clip).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        draw_waveform(&wav, &name, &format!("{dir}/ch01-waveform.svg"))?;
        draw_zoom(&wav, &name, loudest, window, &format!("{dir}/ch01-waveform-zoom.svg"))?;
    }
    Ok(())
}

/// A sine wave: `amplitude * sin(2 pi f t)`, with a short fade in and out so
/// that it does not start and stop with a click.
fn sine_beep(frequency: f32, seconds: f32, sample_rate: u32) -> Wav {
    let n = (seconds * sample_rate as f32) as usize;
    let fade = sample_rate as usize / 100; // 10 ms
    let samples = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let envelope = (i.min(n - 1 - i) as f32 / fade as f32).min(1.0);
            0.5 * envelope * (2.0 * PI * frequency * t).sin()
        })
        .collect();
    Wav::new(sample_rate, samples)
}

/// Start index of the `len` consecutive samples with the most energy.
fn loudest_stretch(samples: &[f32], len: usize) -> usize {
    let energy = |s: &[f32]| s.iter().map(|x| x * x).sum::<f32>();
    let mut current = energy(&samples[..len]);
    let (mut best, mut best_start) = (current, 0);
    // Slide the window one sample at a time: add the new one, drop the old one.
    for start in 1..=samples.len() - len {
        current += samples[start + len - 1].powi(2) - samples[start - 1].powi(2);
        if current > best {
            (best, best_start) = (current, start);
        }
    }
    best_start
}

fn draw_waveform(wav: &Wav, name: &str, path: &str) -> io::Result<()> {
    let times: Vec<f32> = (0..wav.samples.len()).map(|i| i as f32 / wav.sample_rate as f32).collect();
    LinePlot::new(&format!("The waveform of {name}: {} numbers", wav.samples.len()))
        .x_label("time (s)")
        .y_label("amplitude")
        .series(Series::new("", &times, &wav.samples).width(1.0))
        .x_range(0.0, wav.duration_secs())
        .y_range(-1.0, 1.0)
        .save(path)?;
    println!("\nwrote {path}");
    Ok(())
}

fn draw_zoom(wav: &Wav, name: &str, start: usize, len: usize, path: &str) -> io::Result<()> {
    let slice = &wav.samples[start..start + len];
    let times_ms: Vec<f32> = (0..len).map(|i| (start + i) as f32 * 1000.0 / wav.sample_rate as f32).collect();
    LinePlot::new(&format!("Zooming in on {name}: {len} samples, {:.0} ms", len as f32 * 1000.0 / wav.sample_rate as f32))
        .x_label("time (ms)")
        .y_label("amplitude")
        .series(Series::new("", &times_ms, slice).width(1.0).dots())
        .save(path)?;
    println!("wrote {path}");
    Ok(())
}
