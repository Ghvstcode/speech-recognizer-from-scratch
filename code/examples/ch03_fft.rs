//! Chapter 3: the Fourier transform, by hand.
//!
//! Checks that the slow DFT and the fast FFT agree and times them against
//! each other, takes the spectrum of a signal whose contents we know, shows
//! what a window does to spectral leakage, and draws the spectrogram of a
//! spoken word.
//!
//! ```text
//! cargo run --release --example ch03_fft
//! cargo run --release --example ch03_fft -- ../samples/yes.wav
//! ```

use speech::dsp::fft::{bin_frequency, dft, fft, ifft, power_spectrum, Complex, Fft};
use speech::dsp::spectrogram::{power_spectrogram, power_to_db, time_extent};
use speech::dsp::window::{apply, coherent_gain, hann, rectangular};
use speech::plot::{heatmap_ascii, Color, Figure, Heatmap, LinePlot, Orientation, Series};
use speech::rng::Rng;
use speech::wav::Wav;
use std::f64::consts::PI;
use std::hint::black_box;
use std::io;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: cargo run --release --example ch03_fft -- [FILE.wav] [--figures DIR]";
const DEFAULT_CLIP: &str = "../samples/seven.wav";

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
    benchmark();
    two_tones(figures_dir)?;
    leakage(figures_dir)?;
    spectrogram(clip, figures_dir)
}

/// A sine wave sampled at `sample_rate`.
fn sine(frequency: f32, amplitude: f32, sample_rate: u32, len: usize) -> Vec<f32> {
    // The phase is computed in f64: in f32 its rounding error alone would
    // show up in the leakage plot as a noise floor around -110 dB.
    (0..len).map(|t| amplitude * (2.0 * PI * frequency as f64 * t as f64 / sample_rate as f64).sin() as f32).collect()
}

/// DFT against FFT on random input: do they agree, and how long do they take?
fn benchmark() {
    println!("DFT (n^2) against FFT (n log n) on random complex input:\n");
    println!("     n      naive DFT            FFT    speed-up    n^2 / (n log2 n)    largest difference");
    let mut rng = Rng::new(3);
    for n in [256usize, 512, 1024, 2048, 4096] {
        let input: Vec<Complex> = (0..n).map(|_| Complex::new(rng.normal(), rng.normal())).collect();

        // The DFT is slow enough that a handful of runs gives a steady number.
        let dft_runs = (1 << 22) / (n * n) + 1;
        let start = Instant::now();
        let mut slow = Vec::new();
        for _ in 0..dft_runs {
            slow = dft(black_box(&input));
        }
        let dft_time = start.elapsed().as_secs_f64() / dft_runs as f64;

        // The FFT needs thousands of runs to be measurable at all. We time
        // the transform itself, with its table of sines built beforehand.
        let plan = Fft::new(n);
        let fft_runs = 2000;
        let mut fast = input.clone();
        let start = Instant::now();
        for _ in 0..fft_runs {
            fast.copy_from_slice(&input);
            plan.forward(black_box(&mut fast));
        }
        let fft_time = start.elapsed().as_secs_f64() / fft_runs as f64;

        let difference = slow.iter().zip(&fast).map(|(a, b)| (*a - *b).abs()).fold(0.0, f32::max);
        let theory = n as f64 / (n as f64).log2();
        println!(
            "  {n:4}   {:>10}     {:>10}    {:>7.0}x    {:>12.0}x        {difference:.1e}",
            format_time(dft_time),
            format_time(fft_time),
            dft_time / fft_time,
            theory
        );
    }

    // And back again.
    let original: Vec<Complex> = (0..1024).map(|_| Complex::new(rng.normal(), 0.0)).collect();
    let mut round_trip = original.clone();
    fft(&mut round_trip);
    ifft(&mut round_trip);
    let worst = original.iter().zip(&round_trip).map(|(a, b)| (*a - *b).abs()).fold(0.0, f32::max);
    println!("\nifft(fft(x)) for 1024 random samples: largest error {worst:.1e}");
}

fn format_time(seconds: f64) -> String {
    if seconds >= 1e-3 {
        format!("{:.2} ms", seconds * 1e3)
    } else {
        format!("{:.2} us", seconds * 1e6)
    }
}

/// The spectrum of a signal we built ourselves, so we know the answer.
fn two_tones(figures_dir: Option<&str>) -> io::Result<()> {
    let (sample_rate, n) = (16000u32, 1024usize);
    // With 1024 samples at 16 kHz the bins are 15.625 Hz apart, so 500 Hz
    // and 1500 Hz land exactly on bins 32 and 96.
    let signal: Vec<f32> =
        sine(500.0, 1.0, sample_rate, n).iter().zip(&sine(1500.0, 0.5, sample_rate, n)).map(|(a, b)| a + b).collect();
    let power = power_spectrum(&signal, n);
    // |X[k]| = amplitude * n / 2, so amplitude = 2 |X[k]| / n.
    let amplitudes: Vec<f32> = power.iter().map(|p| 2.0 * p.sqrt() / n as f32).collect();

    println!("\ntwo tones, 500 Hz (amplitude 1.0) + 1500 Hz (amplitude 0.5), {n} samples at {sample_rate} Hz:");
    println!("  {} bins from 0 to {} Hz, {} Hz apart", power.len(), sample_rate / 2, sample_rate as f32 / n as f32);
    let mut order: Vec<usize> = (0..amplitudes.len()).collect();
    order.sort_by(|&a, &b| amplitudes[b].total_cmp(&amplitudes[a]));
    for &k in order.iter().take(3) {
        println!("  bin {k:3} = {:6.1} Hz   amplitude {:.4}", bin_frequency(k, n, sample_rate), amplitudes[k]);
    }
    println!("  (the third strongest bin is rounding noise)");

    if let Some(dir) = figures_dir {
        let path = format!("{dir}/ch03-spectrum-two-tones.svg");
        let freqs: Vec<f32> = (0..amplitudes.len()).map(|k| bin_frequency(k, n, sample_rate)).collect();
        let times_ms: Vec<f32> = (0..=320).map(|t| t as f32 * 1000.0 / sample_rate as f32).collect();
        Figure::new("Two tones in, two spikes out")
            .desc("Top: 20 milliseconds of a signal that is the sum of a 500 hertz sine of amplitude 1 and a 1500 hertz sine of amplitude 0.5. Bottom: its spectrum, which is zero everywhere except for a spike of height 1 at 500 hertz and one of height 0.5 at 1500 hertz.")
            .panel(LinePlot::new("the signal: 20 ms of 500 Hz + 1500 Hz").x_label("time (ms)").y_label("amplitude").series(Series::new("", &times_ms, &signal[..=320])).height(130.0))
            .panel(
                LinePlot::new("its spectrum")
                    .x_label("frequency (Hz)")
                    .y_label("amplitude")
                    .series(Series::new("", &freqs, &amplitudes))
                    .x_range(0.0, 4000.0)
                    .y_range(0.0, 1.15)
                    .annotate(500.0, 1.0, "500 Hz")
                    .annotate(1500.0, 0.5, "1500 Hz")
                    .height(170.0),
            )
            .save(&path)?;
        println!("  wrote {path}");
    }
    Ok(())
}

/// What happens when a tone does not complete a whole number of cycles.
fn leakage(figures_dir: Option<&str>) -> io::Result<()> {
    let (sample_rate, n) = (16000u32, 512usize);
    let hz_per_bin = sample_rate as f32 / n as f32;
    // Bin 32 is exactly 1000 Hz. Halfway to bin 33 is the worst case.
    let off_bin = 32.5 * hz_per_bin;
    let tone = sine(off_bin, 1.0, sample_rate, n);

    // Spectrum in dB relative to a full-scale sine sitting exactly on a bin.
    let spectrum_db = |window: &[f32]| -> Vec<f32> {
        let reference = n as f32 / 2.0 * coherent_gain(window);
        power_spectrum(&apply(&tone, window), n).iter().map(|p| 20.0 * (p.sqrt() / reference).max(1e-7).log10()).collect()
    };
    let (rect_db, hann_db) = (spectrum_db(&rectangular(n)), spectrum_db(&hann(n)));

    println!("\nleakage: a {off_bin:.1} Hz tone falls between bins 32 and 33 of a {n}-point FFT.");
    println!("  level in dB (0 = the tone's true level)   at the peak   10 bins away   50 bins away   100 bins away");
    for (name, db) in [("rectangular", &rect_db), ("Hann", &hann_db)] {
        println!(
            "  {name:<12}                                 {:6.1}         {:6.1}         {:6.1}          {:6.1}",
            db[32], db[42], db[82], db[132]
        );
    }

    if let Some(dir) = figures_dir {
        let path = format!("{dir}/ch03-leakage.svg");
        let freqs: Vec<f32> = (0..rect_db.len()).map(|k| k as f32 * hz_per_bin).collect();
        LinePlot::new(&format!("Leakage: one {off_bin:.0} Hz tone that falls between two bins"))
            .x_label("frequency (Hz)")
            .y_label("level (dB)")
            .series(Series::new("rectangular window", &freqs, &rect_db).color(Color::Amber))
            .series(Series::new("Hann window", &freqs, &hann_db).color(Color::Green).width(2.0))
            .x_range(0.0, 4000.0)
            .y_range(-140.0, 5.0)
            .vline(off_bin, "the tone")
            .height(260.0)
            .save(&path)?;
        println!("  wrote {path}");
    }
    Ok(())
}

/// The spectrogram of a real recording.
fn spectrogram(clip: &str, figures_dir: Option<&str>) -> io::Result<()> {
    let wav = Wav::read(clip)?;
    let sr = wav.sample_rate;
    let frame_len = (sr as f32 * 0.025).round() as usize;
    let hop_len = (sr as f32 * 0.010).round() as usize;
    let n_fft = frame_len.next_power_of_two();

    let power = power_spectrogram(&wav.samples, &hann(frame_len), hop_len, n_fft);
    let db = power_to_db(&power, 70.0);
    println!("\nspectrogram of {clip}:");
    println!("  {frame_len}-sample Hann frames every {hop_len} samples, FFT length {n_fft}");
    println!(
        "  {} frames x {} bins; a bin is {:.2} Hz wide, the top one is {} Hz",
        power.rows,
        power.cols,
        sr as f32 / n_fft as f32,
        sr / 2
    );
    if power.rows == 0 {
        println!("  the clip is shorter than one frame");
        return Ok(());
    }

    // Where is the energy? The strongest bin of the loudest frame.
    let frame_power = |r: usize| power.row(r).iter().sum::<f32>();
    let loudest = (0..power.rows).max_by(|&a, &b| frame_power(a).total_cmp(&frame_power(b))).unwrap();
    let row = power.row(loudest);
    let peak_bin = (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
    println!(
        "  loudest frame: #{loudest} ({:.2} s), strongest at bin {peak_bin} = {:.0} Hz",
        (loudest * hop_len) as f32 / sr as f32,
        bin_frequency(peak_bin, n_fft, sr)
    );

    println!("\n  time left to right, 0 Hz at the bottom to {} Hz at the top, darker = louder (70 dB range):\n", sr / 2);
    for text_row in heatmap_ascii(&db, 72, 16, Orientation::TimeOnX).lines() {
        println!("  {text_row}");
    }

    if let Some(dir) = figures_dir {
        let path = format!("{dir}/ch03-spectrogram.svg");
        // Both panels show exactly the stretch of time the frames cover.
        let (t0, t1) = time_extent(power.rows, frame_len, hop_len, sr);
        let times: Vec<f32> = (0..wav.samples.len()).map(|i| i as f32 / sr as f32).collect();
        let name = std::path::Path::new(clip).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        Figure::new(&format!("{name}: the same second of sound, twice"))
            .desc("Top: the waveform of a one-second recording of the word seven. Bottom: its spectrogram, time on the horizontal axis and frequency from 0 to 8000 hertz on the vertical axis, darker meaning more energy. The hiss of the s shows as a cloud above 4000 hertz, the two vowels as stacks of dark horizontal bands below 3000 hertz.")
            .panel(LinePlot::new("waveform").y_label("amplitude").series(Series::new("", &times, &wav.samples).width(1.0)).x_range(t0, t1).height(110.0))
            .panel(
                Heatmap::new("spectrogram", &db)
                    .orientation(Orientation::TimeOnX)
                    .x_range(t0, t1)
                    .y_range(0.0, sr as f32 / 2.0)
                    .x_label("time (s)")
                    .y_label("frequency (Hz)")
                    .colorbar("dB")
                    .height(260.0),
            )
            .share_x()
            .save(&path)?;
        println!("\n  wrote {path}");
    }
    Ok(())
}
