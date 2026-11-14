//! Chapter 11: the live pipeline, run on a recording, drawn.
//!
//! Plays a WAV file into exactly the machinery `speech listen` uses (an
//! `AudioSource`, the `Segmenter` with its streaming endpoint detector, the
//! recognizer on every finished utterance) and draws what happened: the
//! loudness of the stream over time, where utterances were cut out, and
//! what was recognized in each.
//!
//! ```text
//! cargo run --release --example ch11_endpointing
//! cargo run --release --example ch11_endpointing -- my_recording.wav --decoder greedy
//! ```
//!
//! Writes `../figures/ch11-endpointing-live.svg`.

use speech::data::pipeline::{Decoder, Recognizer};
use speech::mic::{AudioSource, Segment, Segmenter, SegmenterConfig, WavPlayback, SAMPLE_RATE};
use speech::plot::{Color, Figure, LinePlot, Series};
use std::time::Instant;

const USAGE: &str = "usage: ch11_endpointing [file.wav] [--decoder greedy|beam|lexicon] [--out ../figures/ch11-endpointing-live.svg]";

fn main() -> std::io::Result<()> {
    let (mut file, mut decoder, mut out) = ("../samples/utterances_with_pauses.wav".to_string(), Decoder::from_name("lexicon").expect("a known name"), "../figures/ch11-endpointing-live.svg".to_string());
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut rest = args.iter();
    while let Some(argument) = rest.next() {
        match (argument.as_str(), argument.starts_with("--").then(|| rest.next()).flatten()) {
            ("--decoder", Some(name)) if Decoder::from_name(name).is_some() => decoder = Decoder::from_name(name).expect("just checked"),
            ("--out", Some(path)) => out = path.clone(),
            (path, None) if !path.starts_with("--") => file = path.to_string(),
            _ => {
                eprintln!("error: cannot use {argument:?}\n{USAGE}");
                std::process::exit(2);
            }
        }
    }
    let recognizer = Recognizer::load(Recognizer::default_path())?;

    // The live loop, written out (speech::mic::listen does the same; here
    // we also want to record the level of every chunk for the plot).
    let chunk = (SAMPLE_RATE / 100) as usize; // 10 ms, one endpointer block
    let mut source = WavPlayback::open(&file, chunk, false)?;
    let mut segmenter = Segmenter::new(SegmenterConfig::default());
    let (mut levels, mut active) = (Vec::new(), Vec::new());
    let mut heard: Vec<(Segment, String, f32)> = Vec::new();
    let recognize = |segment: Segment, heard: &mut Vec<(Segment, String, f32)>| {
        let started = Instant::now();
        let text = recognizer.transcribe(&segment.samples, &decoder);
        heard.push((segment, text, 1000.0 * started.elapsed().as_secs_f32()));
    };
    let mut samples = Vec::new();
    loop {
        samples.clear();
        if source.read(&mut samples)? == 0 {
            break;
        }
        for segment in segmenter.push(&samples) {
            recognize(segment, &mut heard);
        }
        levels.push(segmenter.level_db().max(-80.0));
        active.push(segmenter.in_speech());
    }
    if let Some(segment) = segmenter.finish() {
        recognize(segment, &mut heard);
    }

    let seconds = |sample: usize| sample as f32 / SAMPLE_RATE as f32;
    println!("{file}: {:.1} s, {} utterances found", seconds(segmenter.samples_seen()), heard.len());
    for (segment, text, ms) in &heard {
        println!("  [{:>6.2} s - {:>6.2} s]  {text}   ({ms:.0} ms)", seconds(segment.start), seconds(segment.end));
    }

    let times: Vec<f32> = (0..levels.len()).map(|i| (i as f32 + 0.5) * 0.01).collect();
    let duration = seconds(segmenter.samples_seen());
    let mut level_plot = LinePlot::new("loudness of every 10 ms block; shaded: the utterances that were cut out, with the recognized text")
        .y_label("level (dB)")
        .series(Series::new("", &times, &levels).color(Color::Ink70).width(1.0))
        .x_range(0.0, duration)
        .y_range(-80.0, 0.0)
        .height(200.0);
    for (segment, text, _) in &heard {
        level_plot = level_plot.span(seconds(segment.start), seconds(segment.end), if text.is_empty() { "(nothing)" } else { text });
    }
    // When did the detector consider itself "in speech"? It notices the
    // start a few blocks late and the end 600 ms late: that is the price of
    // not being fooled by clicks and by pauses between words.
    let state: Vec<f32> = active.iter().map(|&a| if a { 1.0 } else { 0.0 }).collect();
    let state_plot = LinePlot::new("the detector's state: it notices the end of speech 600 ms after the last loud block")
        .y_label("in speech?")
        .x_label("time (s)")
        .series(Series::new("", &times, &state).color(Color::Green).width(1.5))
        .x_range(0.0, duration)
        .y_range(-0.1, 1.1)
        .height(70.0);
    Figure::new("Listening live: from an endless stream to utterances to text")
        .desc("Top: the level in decibels of a 17 second recording containing four utterances separated by pauses, with the four detected segments shaded and labelled with the recognized words. Bottom: the endpoint detector's speech/silence state over time.")
        .panel(level_plot)
        .panel(state_plot)
        .share_x()
        .save(&out)?;
    println!("wrote {out}");
    Ok(())
}
