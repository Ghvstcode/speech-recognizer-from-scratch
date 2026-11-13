//! Chapter 4: hearing like a human.
//!
//! Prints the mel scale and the filterbank built on it, turns a clip into a
//! log-mel spectrogram and into MFCCs, and times the whole feature
//! extraction.
//!
//! ```text
//! cargo run --release --example ch04_mel
//! cargo run --release --example ch04_mel -- ../samples/seven_jackson.wav
//! ```

use speech::dsp::features::{dct2, deltas, log_mel, mfcc, normalize, FeatureConfig, FeatureExtractor};
use speech::dsp::mel::{hz_to_mel, mel_filterbank, mel_points, mel_to_hz};
use speech::dsp::spectrogram::{power_to_db, time_extent};
use speech::plot::{heatmap_ascii, Color, Figure, Heatmap, LinePlot, Orientation, Series};
use speech::wav::Wav;
use std::hint::black_box;
use std::io;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: cargo run --release --example ch04_mel -- [FILE.wav] [--figures DIR]";
const DEFAULT_CLIP: &str = "../samples/seven.wav";
const N_MFCC: usize = 13;

fn main() -> ExitCode {
    let mut clip: Option<String> = None;
    let mut figures_dir: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
    // The book's figures come from the default clip; see ch01_wav.
    if clip.is_none() && figures_dir.is_none() {
        figures_dir = Some("../figures".to_string());
    }
    let clip = clip.unwrap_or_else(|| DEFAULT_CLIP.to_string());

    match run(&clip, figures_dir.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn run(clip: &str, figures_dir: Option<&str>) -> io::Result<()> {
    let wav = Wav::read(clip)?;
    let cfg = FeatureConfig::new(wav.sample_rate);

    // --- The mel scale. ---
    println!("the mel scale, mel = 2595 * log10(1 + hz / 700):\n");
    println!("      Hz      mel    back to Hz");
    for hz in [0.0f32, 100.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0] {
        println!("  {hz:6.0}   {:6.1}   {:8.1}", hz_to_mel(hz), mel_to_hz(hz_to_mel(hz)));
    }
    println!(
        "  100 Hz -> 200 Hz is {:.0} mel; 7900 Hz -> 8000 Hz is {:.0} mel",
        hz_to_mel(200.0) - hz_to_mel(100.0),
        hz_to_mel(8000.0) - hz_to_mel(7900.0)
    );

    // --- The filterbank. ---
    let filterbank = mel_filterbank(cfg.n_mels, cfg.n_fft(), cfg.sample_rate, cfg.fmin, cfg.fmax);
    let points = mel_points(cfg.n_mels, cfg.fmin, cfg.fmax);
    println!(
        "\nfilterbank for {} Hz audio: {} triangles over {} FFT bins ({} Hz to {} Hz)",
        cfg.sample_rate, filterbank.rows, filterbank.cols, cfg.fmin, cfg.fmax
    );
    println!("  filter   from Hz   peak Hz   to Hz   bins it touches");
    for m in [0, 1, 2, cfg.n_mels / 2, cfg.n_mels - 2, cfg.n_mels - 1] {
        let touched = filterbank.row(m).iter().filter(|&&w| w > 0.0).count();
        println!("  {m:6}   {:7.0}   {:7.0}   {:5.0}   {touched:5}", points[m], points[m + 1], points[m + 2]);
    }

    // --- Features of the clip. ---
    let extractor = FeatureExtractor::new(&cfg);
    let power = extractor.power_spectrogram(&wav.samples);
    let mel = extractor.log_mel(&wav.samples);
    let cepstra = extractor.mfcc(&wav.samples, N_MFCC);
    println!("\n{clip}: {} samples at {} Hz ({:.2} s)", wav.samples.len(), wav.sample_rate, wav.duration_secs());
    println!("  frames of {} samples every {}, FFT length {}", cfg.frame_len(), cfg.hop_len(), cfg.n_fft());
    println!("  power spectrogram  {} x {}   (frames x FFT bins)", power.rows, power.cols);
    println!("  log-mel            {} x {}    (frames x mel bands)", mel.rows, mel.cols);
    println!("  MFCC               {} x {}    (frames x coefficients)", cepstra.rows, cepstra.cols);
    if mel.rows == 0 {
        println!("  the clip is shorter than one frame");
        return Ok(());
    }
    let (lo, hi) = mel.data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    println!("  log-mel values run from {lo:.1} to {hi:.1} (silence would be ln(1e-6) = {:.1})", 1e-6f32.ln());

    let frame_energy = |r: usize| mel.row(r).iter().sum::<f32>();
    let loudest = (0..mel.rows).max_by(|&a, &b| frame_energy(a).total_cmp(&frame_energy(b))).unwrap();
    let fmt = |values: &[f32]| values.iter().map(|v| format!("{v:6.1}")).collect::<Vec<_>>().join(" ");
    println!("\n  loudest frame (#{loudest}):");
    println!("    log-mel, first 10 bands:  {}", fmt(&mel.row(loudest)[..10.min(mel.cols)]));
    println!("    its {N_MFCC} MFCCs:             {}", fmt(cepstra.row(loudest)));
    println!("    DCT of the log-mel row:   {}   (the same, computed directly)", fmt(&dct2(mel.row(loudest), N_MFCC)));

    // What a model typically gets: MFCCs plus their deltas, normalized.
    let mut stacked = cepstra.clone();
    normalize(&mut stacked);
    let d = deltas(&stacked, 2);
    println!("    normalized MFCCs:         {}", fmt(stacked.row(loudest)));
    println!("    their deltas:             {}", fmt(d.row(loudest)));

    println!("\n  log-mel spectrogram (time left to right, band 0 at the bottom, darker = more energy):\n");
    for text_row in heatmap_ascii(&mel, 72, 14, Orientation::TimeOnX).lines() {
        println!("  {text_row}");
    }

    // --- How long does it take? ---
    let runs = 200;
    let reused = time_per_run(runs, || {
        black_box(extractor.log_mel(black_box(&wav.samples)));
    });
    let from_scratch = time_per_run(runs, || {
        black_box(log_mel(black_box(&wav.samples), &cfg));
    });
    let mfcc_time = time_per_run(runs, || {
        black_box(mfcc(black_box(&wav.samples), &cfg, N_MFCC));
    });
    println!("\ntiming on this {:.2} s clip (best of 5 batches of {runs} runs):", wav.duration_secs());
    println!(
        "  log-mel with a reused FeatureExtractor   {:7.1} us   ({:.0}x faster than real time)",
        reused * 1e6,
        wav.duration_secs() as f64 / reused
    );
    println!("  log_mel(), building its tables each time {:7.1} us", from_scratch * 1e6);
    println!("  mfcc(), building its tables each time    {:7.1} us", mfcc_time * 1e6);

    if let Some(dir) = figures_dir {
        let name = std::path::Path::new(clip).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        draw_mel_scale(&format!("{dir}/ch04-mel-scale.svg"))?;
        draw_filterbank(&cfg, &format!("{dir}/ch04-filterbank.svg"))?;
        draw_log_mel(&cfg, &name, &power, &mel, &format!("{dir}/ch04-logmel.svg"))?;
        draw_mfcc(&cfg, &name, &stacked, &format!("{dir}/ch04-mfcc.svg"))?;
    }
    Ok(())
}

/// Seconds per call of `work`: the fastest of five batches of `runs` calls.
/// Taking the best batch filters out the moments when the machine was busy
/// with something else.
fn time_per_run(runs: usize, mut work: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..runs {
            work();
        }
        best = best.min(start.elapsed().as_secs_f64() / runs as f64);
    }
    best
}

fn draw_mel_scale(path: &str) -> io::Result<()> {
    let hz: Vec<f32> = (0..=160).map(|i| i as f32 * 50.0).collect();
    let mel: Vec<f32> = hz.iter().map(|&f| hz_to_mel(f)).collect();
    // For comparison: a straight line with the slope the curve has at 0 Hz,
    // i.e. what the scale would be if pitch perception were linear.
    let slope = hz_to_mel(1.0);
    let straight: Vec<f32> = hz.iter().map(|&f| f * slope).collect();
    LinePlot::new("The mel scale: equal steps sound equally far apart")
        .x_label("frequency (Hz)")
        .y_label("pitch (mel)")
        .series(Series::new("mel = 2595 log10(1 + Hz / 700)", &hz, &mel).width(2.0))
        .series(Series::new("if hearing were linear", &hz, &straight).color(Color::Ink30).dashed())
        .y_range(0.0, 3000.0)
        .annotate(1900.0, 800.0, "1000 Hz = 1000 mel")
        .series(Series::new("", &[1000.0], &[hz_to_mel(1000.0)]).color(Color::Amber).dots_only())
        .height(260.0)
        .save(path)?;
    println!("\nwrote {path}");
    Ok(())
}

fn draw_filterbank(cfg: &FeatureConfig, path: &str) -> io::Result<()> {
    let filterbank = mel_filterbank(cfg.n_mels, cfg.n_fft(), cfg.sample_rate, cfg.fmin, cfg.fmax);
    let points = mel_points(cfg.n_mels, cfg.fmin, cfg.fmax);
    let mut plot = LinePlot::new(&format!("The mel filterbank: {} triangles, narrow at the bottom, wide at the top", cfg.n_mels))
        .x_label("frequency (Hz)")
        .y_label("weight")
        .x_range(0.0, cfg.fmax)
        .y_range(0.0, 1.08)
        .height(200.0);
    // Draw each triangle from its three corner frequencies (the matrix only
    // holds its values at the FFT bins, which would round off the tips).
    let highlighted = cfg.n_mels / 2;
    for m in 0..cfg.n_mels {
        let color = if m == highlighted { Color::Amber } else { Color::Green };
        let width = if m == highlighted { 2.0 } else { 1.0 };
        plot =
            plot.series(Series::new("", &[points[m], points[m + 1], points[m + 2]], &[0.0, 1.0, 0.0]).color(color).width(width));
    }
    // The weights the highlighted filter really has, one dot per FFT bin.
    let hz_per_bin = cfg.sample_rate as f32 / cfg.n_fft() as f32;
    let (bins, weights): (Vec<f32>, Vec<f32>) = filterbank
        .row(highlighted)
        .iter()
        .enumerate()
        .filter(|(_, &w)| w > 0.0)
        .map(|(bin, &w)| (bin as f32 * hz_per_bin, w))
        .unzip();
    plot = plot.series(Series::new("", &bins, &weights).color(Color::Amber).dots_only());
    plot.annotate(points[highlighted + 1], 1.0, &format!("filter {highlighted}: one weight per FFT bin")).save(path)?;
    println!("wrote {path}");
    Ok(())
}

fn draw_log_mel(cfg: &FeatureConfig, name: &str, power: &speech::mat::Mat, mel: &speech::mat::Mat, path: &str) -> io::Result<()> {
    let (t0, t1) = time_extent(power.rows, cfg.frame_len(), cfg.hop_len(), cfg.sample_rate);
    Figure::new(&format!("{name}: {} FFT bins squeezed into {} mel bands", power.cols, mel.cols))
        .desc("Top: the power spectrogram of the word seven with a linear frequency axis from 0 to 8000 hertz. Bottom: the log-mel spectrogram of the same clip, 40 mel bands. The lower frequencies, where the vowels' structure is, take up most of the mel picture; the upper half of the linear spectrogram is squeezed into the top few bands.")
        .panel(
            Heatmap::new(&format!("spectrogram (after pre-emphasis), {} bins, linear in Hz", power.cols), &power_to_db(power, 70.0))
                .orientation(Orientation::TimeOnX)
                .x_range(t0, t1)
                .y_range(0.0, cfg.sample_rate as f32 / 2.0)
                .y_label("frequency (Hz)")
                .colorbar("dB")
                .height(190.0),
        )
        .panel(
            Heatmap::new(&format!("log-mel spectrogram, {} bands", mel.cols), mel)
                .orientation(Orientation::TimeOnX)
                .x_range(t0, t1)
                .x_label("time (s)")
                .y_label("mel band")
                .colorbar("ln energy")
                .height(190.0),
        )
        .share_x()
        .save(path)?;
    println!("wrote {path}");
    Ok(())
}

fn draw_mfcc(cfg: &FeatureConfig, name: &str, normalized: &speech::mat::Mat, path: &str) -> io::Result<()> {
    let (t0, t1) = time_extent(normalized.rows, cfg.frame_len(), cfg.hop_len(), cfg.sample_rate);
    Heatmap::new(&format!("{name}: {} MFCCs per frame, each normalized over the clip", normalized.cols), normalized)
        .orientation(Orientation::TimeOnX)
        .x_range(t0, t1)
        .value_range(-2.5, 2.5)
        .x_label("time (s)")
        .y_label("coefficient")
        .colorbar("std. dev.")
        .height(182.0)
        .save(path)?;
    println!("wrote {path}");
    Ok(())
}
