//! Chapter 2: slicing time.
//!
//! Cuts a clip into frames, measures how loud each one is, finds where the
//! speech starts and ends, and shows what a window function does to a frame.
//!
//! ```text
//! cargo run --release --example ch02_frames
//! cargo run --release --example ch02_frames -- ../samples/happy.wav
//! ```

use speech::dsp::frame::{
    energy_db, find_speech, frames, num_frames, preemphasis, zero_crossing_rate, EndpointConfig, EndpointEvent, Endpointer,
};
use speech::dsp::window::{apply, coherent_gain, hamming, hann, rectangular};
use speech::plot::{sparkline, Color, Figure, LinePlot, Series};
use speech::wav::Wav;
use std::io;
use std::process::ExitCode;

const USAGE: &str = "usage: cargo run --release --example ch02_frames -- [FILE.wav] [--figures DIR]";
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
    let wav = Wav::read(clip)?;
    let sr = wav.sample_rate;
    let secs = |sample: usize| sample as f32 / sr as f32;
    println!("{clip}: {} samples at {sr} Hz, {:.2} s", wav.samples.len(), wav.duration_secs());

    // --- Framing: 25 ms frames, a new one every 10 ms. ---
    let frame_len = (sr as f32 * 0.025).round() as usize;
    let hop_len = (sr as f32 * 0.010).round() as usize;
    let all_frames = frames(&wav.samples, frame_len, hop_len);
    println!("\nframing: {frame_len} samples per frame (25 ms), hop {hop_len} samples (10 ms)");
    println!(
        "  1 + ({} - {frame_len}) / {hop_len} = {} frames (the function says {})",
        wav.samples.len(),
        1 + (wav.samples.len().max(frame_len) - frame_len) / hop_len,
        num_frames(wav.samples.len(), frame_len, hop_len)
    );
    if all_frames.is_empty() {
        println!("  the clip is shorter than one frame; nothing more to show");
        return Ok(());
    }

    // --- Loudness and zero crossings, frame by frame. ---
    let levels: Vec<f32> = all_frames.iter().map(|f| energy_db(f)).collect();
    let crossings: Vec<f32> = all_frames.iter().map(|f| zero_crossing_rate(f)).collect();
    println!("\nenergy per frame, in dB (one bar per 10 ms):");
    println!("  {}", sparkline(&levels));
    println!("zero-crossing rate per frame:");
    println!("  {}", sparkline(&crossings));
    let loudest = (0..levels.len()).max_by(|&a, &b| levels[a].total_cmp(&levels[b])).unwrap();
    let hissiest = (0..levels.len())
        .filter(|&i| levels[i] > levels[loudest] - 30.0)
        .max_by(|&a, &b| crossings[a].total_cmp(&crossings[b]))
        .unwrap();
    println!(
        "  loudest frame:  #{loudest:<3} at {:.2} s  {:6.1} dB  zero-crossing rate {:.2}",
        secs(loudest * hop_len),
        levels[loudest],
        crossings[loudest]
    );
    println!(
        "  hissiest frame: #{hissiest:<3} at {:.2} s  {:6.1} dB  zero-crossing rate {:.2}",
        secs(hissiest * hop_len),
        levels[hissiest],
        crossings[hissiest]
    );

    // --- Endpoint detection: the whole clip at once... ---
    let speech = find_speech(&wav.samples, sr);
    println!("\nendpoint detection:");
    match speech {
        Some((start, end)) => {
            println!(
                "  speech from {:.2} s to {:.2} s ({} of {} samples kept)",
                secs(start),
                secs(end),
                end - start,
                wav.samples.len()
            );
            // The same sparkline, with the detected stretch marked underneath.
            let marks: String = (0..levels.len())
                .map(|i| if (start..end).contains(&(i * hop_len + frame_len / 2)) { '▔' } else { ' ' })
                .collect();
            println!("  {}\n  {}", sparkline(&levels), marks.trim_end());
        }
        None => println!("  no speech found"),
    }

    // --- ...and the same detector fed block by block, as a microphone would. ---
    let mut endpointer = Endpointer::new(sr, EndpointConfig::default());
    println!("  streaming, {} samples ({} ms) at a time:", endpointer.block_len(), EndpointConfig::default().block_ms);
    let block_len = endpointer.block_len();
    let mut fed = 0;
    let report = |event: Option<EndpointEvent>, fed: usize| match event {
        Some(EndpointEvent::SpeechStart { sample }) => {
            println!("    at {:.2} s: speech started at {:.2} s", secs(fed), secs(sample))
        }
        Some(EndpointEvent::SpeechEnd { sample }) => println!("    at {:.2} s: speech ended at {:.2} s", secs(fed), secs(sample)),
        None => {}
    };
    for block in wav.samples.chunks_exact(block_len) {
        fed += block.len();
        report(endpointer.push(block), fed);
    }
    report(endpointer.finish(), fed);
    if let Some(floor) = endpointer.noise_floor_db() {
        println!("    noise floor it settled on: {floor:.1} dB");
    }

    // --- Windows. ---
    let (rect, hann_w, hamming_w) = (rectangular(frame_len), hann(frame_len), hamming(frame_len));
    println!("\nwindows of {frame_len} samples:      first   quarter   middle   average height");
    for (name, w) in [("rectangular", &rect), ("Hann", &hann_w), ("Hamming", &hamming_w)] {
        println!(
            "  {name:<12}            {:.3}     {:.3}    {:.3}        {:.3}",
            w[0],
            w[frame_len / 4],
            w[frame_len / 2],
            coherent_gain(w)
        );
    }
    let frame = all_frames[loudest];
    let windowed = apply(frame, &hann_w);
    println!(
        "  loudest frame, first and last sample: {:+.3} and {:+.3}; after the Hann window: {:+.3} and {:+.3}",
        frame[0],
        frame[frame_len - 1],
        windowed[0],
        windowed[frame_len - 1]
    );

    // --- Pre-emphasis. ---
    let emphasized = preemphasis(&wav.samples, 0.97);
    let zcr_before = zero_crossing_rate(frame);
    let zcr_after = zero_crossing_rate(&emphasized[loudest * hop_len..loudest * hop_len + frame_len]);
    println!("\npre-emphasis (0.97) on the loudest frame: zero-crossing rate {zcr_before:.2} -> {zcr_after:.2} (the high frequencies now carry more weight)");

    if let Some(dir) = figures_dir {
        draw_endpoints(&wav, &levels, frame_len, hop_len, speech, &format!("{dir}/ch02-energy-endpoints.svg"))?;
        draw_windows(frame_len, sr, &format!("{dir}/ch02-windows.svg"))?;
        draw_windowed_frame(frame, &hann_w, loudest, sr, &format!("{dir}/ch02-frame-windowed.svg"))?;
    }
    Ok(())
}

/// Waveform on top, loudness contour below, the detected speech shaded in both.
fn draw_endpoints(
    wav: &Wav,
    levels: &[f32],
    frame_len: usize,
    hop_len: usize,
    speech: Option<(usize, usize)>,
    path: &str,
) -> io::Result<()> {
    let sr = wav.sample_rate as f32;
    let sample_times: Vec<f32> = (0..wav.samples.len()).map(|i| i as f32 / sr).collect();
    // A frame's level is plotted at the time of the frame's centre.
    let frame_times: Vec<f32> = (0..levels.len()).map(|i| (i * hop_len + frame_len / 2) as f32 / sr).collect();

    let mut waveform = LinePlot::new("waveform")
        .y_label("amplitude")
        .series(Series::new("", &sample_times, &wav.samples).width(1.0))
        .x_range(0.0, wav.duration_secs())
        .height(150.0);
    let mut energy = LinePlot::new("energy per 25 ms frame")
        .x_label("time (s)")
        .y_label("energy (dB)")
        .series(Series::new("", &frame_times, levels).width(1.75))
        .x_range(0.0, wav.duration_secs())
        .height(170.0);
    if let Some((start, end)) = speech {
        let (start, end) = (start as f32 / sr, end as f32 / sr);
        waveform = waveform.span(start, end, "speech");
        energy = energy.span(start, end, "").vline(start, &format!("start {start:.2} s")).vline(end, &format!("end {end:.2} s"));
    }
    Figure::new("Finding the word: energy-based endpoint detection")
        .desc("Top: the waveform of a one-second recording of the word seven. Bottom: its energy in decibels for every 25 millisecond frame. The stretch the endpoint detector marked as speech is shaded in both.")
        .panel(waveform)
        .panel(energy)
        .share_x()
        .save(path)?;
    println!("\nwrote {path}");
    Ok(())
}

fn draw_windows(frame_len: usize, sample_rate: u32, path: &str) -> io::Result<()> {
    let times_ms: Vec<f32> = (0..frame_len).map(|i| i as f32 * 1000.0 / sample_rate as f32).collect();
    LinePlot::new(&format!("Three windows for a 25 ms frame ({frame_len} samples)"))
        .x_label("time within the frame (ms)")
        .y_label("weight")
        .series(Series::new("Hann", &times_ms, &hann(frame_len)).width(2.0))
        .series(Series::new("Hamming", &times_ms, &hamming(frame_len)))
        .series(Series::new("rectangular", &times_ms, &rectangular(frame_len)).color(Color::Ink48).dashed())
        .y_range(0.0, 1.1)
        .save(path)?;
    println!("wrote {path}");
    Ok(())
}

fn draw_windowed_frame(frame: &[f32], window: &[f32], frame_index: usize, sample_rate: u32, path: &str) -> io::Result<()> {
    let times_ms: Vec<f32> = (0..frame.len()).map(|i| i as f32 * 1000.0 / sample_rate as f32).collect();
    let peak = frame.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let envelope: Vec<f32> = window.iter().map(|w| w * peak).collect();
    LinePlot::new(&format!("Frame {frame_index} before and after the Hann window"))
        .x_label("time within the frame (ms)")
        .y_label("amplitude")
        .series(Series::new("raw frame", &times_ms, frame).color(Color::Ink30).width(1.25))
        .series(Series::new("frame \u{00D7} window", &times_ms, &apply(frame, window)).color(Color::Green))
        .series(Series::new("window (scaled)", &times_ms, &envelope).color(Color::Amber).dashed().width(1.25))
        .save(path)?;
    println!("wrote {path}");
    Ok(())
}
