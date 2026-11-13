//! Chapter 9: the figures, and the sample utterances readers can try.
//!
//! Run after `ch09_train` (it reads the model, the training log and the
//! posterior snapshots that program wrote):
//!
//! ```text
//! cargo run --release --example ch09_figures
//! ```
//!
//! Writes `../figures/ch09-synthesis.svg`, `ch09-training-curve.svg`,
//! `ch09-posteriors-over-training.svg` and `ch09-alignment.svg`, and exports
//! a few synthesized *test* utterances to `../samples/` so that
//! `speech transcribe` can be tried without downloading the dataset.

use speech::data::pipeline::{Decoder, FrontEnd, Recognizer};
use speech::data::speech_commands::{dataset_dir, load_background_noise, LoadOptions, SpeechCommands, Split, SAMPLE_RATE, WORDS};
use speech::data::synth::{locate_word, Corpus, SynthConfig, Synthesizer, TrimConfig, Utterance, EVAL_UTTERANCES, TEST_SEED};
use speech::decode::Alphabet;
use speech::mat::Mat;
use speech::plot::{Color, Figure, Heatmap, LinePlot, Orientation, Series};
use speech::wav::Wav;
use std::io::{self, Write};

const LOG_PATH: &str = "../models/recognizer-training.log";
const SNAPSHOT_PATH: &str = "../models/recognizer-posteriors.txt";

/// Thin a waveform for plotting: keep the minimum and maximum of every
/// bucket, so that the outline survives while the file stays small.
fn outline(samples: &[f32], buckets: usize) -> (Vec<f32>, Vec<f32>) {
    let per_bucket = samples.len().div_ceil(buckets).max(1);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (b, bucket) in samples.chunks(per_bucket).enumerate() {
        let (lo, hi) = bucket.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &s| (lo.min(s), hi.max(s)));
        let t = (b * per_bucket) as f32 / SAMPLE_RATE as f32;
        xs.extend([t, t + 0.5 * per_bucket as f32 / SAMPLE_RATE as f32]);
        ys.extend([lo, hi]);
    }
    (xs, ys)
}

/// How an utterance is assembled: a raw clip and its trim, the words on a
/// timeline, the same with noise, and the features the network gets.
fn synthesis_figure(data: &SpeechCommands, corpus: &Corpus, front_end: &FrontEnd) -> io::Result<()> {
    // The same seed with and without noise gives the same words in the same
    // places (the words are drawn first), so we can show both versions.
    let recipe = SynthConfig { utterance_gain_db: (0.0, 0.0), ..SynthConfig::default() };
    let noisy_synth = Synthesizer::new(&corpus.test.words, &corpus.test.noise, SynthConfig { noise_prob: 1.0, snr_db: (10.0, 10.0), ..recipe.clone() });
    let clean_synth = Synthesizer::new(&corpus.test.words, &corpus.test.noise, SynthConfig { noise_prob: 0.0, ..recipe });
    // Look for a four-word utterance of a comfortable length.
    let index = (0..500u64).find(|&i| {
        let utt = clean_synth.utterance_at(77, i);
        utt.words.len() == 4 && (2.8..3.6).contains(&utt.seconds())
    });
    let index = index.expect("no suitable utterance among the first 500");
    let (clean, noisy) = (clean_synth.utterance_at(77, index), noisy_synth.utterance_at(77, index));
    assert_eq!(clean.words, noisy.words);
    let duration = clean.seconds();

    // One raw clip of the first word, and what the trimmer keeps of it.
    let clip = data.split(Split::Test).filter(|clip| clip.word == clean.words[0]).find(|clip| locate_word(&clip.audio(), &TrimConfig::default()).is_some_and(|(s, e)| (5000..9000).contains(&(e - s)) && s > 2500));
    let clip = clip.expect("no test clip of that word");
    let audio = clip.audio_one_second();
    let (start, end) = locate_word(&clip.audio(), &TrimConfig::default()).expect("checked above");
    let (xs, ys) = outline(&audio, 800);
    let raw_panel = LinePlot::new(&format!("1. a one-second clip of \"{}\": the endpoint detector finds the word, the rest is thrown away", WORDS[clip.word]))
        .y_label("amplitude")
        .x_label("time in the clip (s)")
        .series(Series::new("", &xs, &ys).color(Color::Ink70).width(1.0))
        .span(start as f32 / 16000.0, end as f32 / 16000.0, "kept")
        .x_range(0.0, 1.0)
        .height(110.0);

    let timeline = |title: &str, utt: &Utterance, color: Color| {
        let (xs, ys) = outline(&utt.samples, 1200);
        let mut plot = LinePlot::new(title).y_label("amplitude").series(Series::new("", &xs, &ys).color(color).width(1.0)).x_range(0.0, duration).y_range(-0.6, 0.6).height(120.0);
        for (&word, &(start, end)) in utt.words.iter().zip(&utt.spans) {
            plot = plot.span(start as f32 / 16000.0, end as f32 / 16000.0, WORDS[word]);
        }
        plot
    };
    let features = front_end.extractor().log_mel(&noisy.samples);
    Figure::new("Building a training utterance out of single-word clips")
        .desc("Four panels: a raw one-second clip with the kept part shaded; four trimmed words placed on a timeline with random gaps; the same timeline with background noise added at 10 dB signal-to-noise ratio; and the log-mel spectrogram of the result.")
        .panel(raw_panel)
        .panel(timeline(&format!("2. four trimmed words on a timeline, random pauses between them: \"{}\"", clean.text()), &clean, Color::Green).x_label("time (s)"))
        .panel(timeline(&format!("3. plus background noise ({}) at a signal-to-noise ratio of 10 dB", corpus.test.noise.tracks[noisy.noise_track.unwrap_or(0)].0.replace('_', " ")), &noisy, Color::Ink70).x_label("time (s)"))
        .panel(Heatmap::new("4. what the network is given: 40 log-mel bands every 10 ms (then normalized)", &features).orientation(Orientation::TimeOnX).x_range(0.0, duration).x_label("time (s)").y_label("mel band").height(130.0))
        .save("../figures/ch09-synthesis.svg")
}

/// Loss and validation error against the training step, read back from the
/// training log.
fn training_curve() -> io::Result<()> {
    let log = std::fs::read_to_string(LOG_PATH)?;
    let number_after = |line: &str, key: &str| -> Option<f32> { line.split(key).nth(1)?.split_whitespace().next()?.trim_end_matches('%').trim_end_matches(':').parse().ok() };
    let (mut steps, mut losses) = (Vec::new(), Vec::new());
    let (mut eval_steps, mut cers, mut wers) = (Vec::new(), Vec::new(), Vec::new());
    for line in log.lines() {
        if line.starts_with("step ") {
            if let (Some(step), Some(loss)) = (number_after(line, "step"), number_after(line, "loss")) {
                steps.push(step);
                losses.push(loss);
            }
        } else if line.starts_with("--- step ") {
            if let (Some(step), Some(cer), Some(wer)) = (number_after(line, "--- step"), number_after(line, "CER"), number_after(line, "WER")) {
                eval_steps.push(step);
                cers.push(cer.min(100.0));
                wers.push(wer.min(100.0));
            }
        }
    }
    if steps.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{LOG_PATH}: no training steps found")));
    }
    let last = *steps.last().unwrap_or(&1.0);
    Figure::new("Training the recognizer: loss, and errors on utterances it has never heard")
        .desc("Top: the CTC loss per utterance over training steps, falling steeply at first. Bottom: character and word error rate on validation utterances with greedy decoding.")
        .panel(LinePlot::new("CTC loss per utterance (average of the last 100 steps; the first value, off the chart, is about 400)").y_label("loss").series(Series::new("training loss", &steps, &losses).color(Color::Ink70)).y_range(0.0, 40.0).x_range(0.0, last).legend(false).height(170.0))
        .panel(
            LinePlot::new("error on 500 validation utterances, greedy decoding")
                .y_label("error rate (%)")
                .x_label("training step (32 utterances each)")
                .series(Series::new("word error rate", &eval_steps, &wers).color(Color::Amber).width(2.0).dots())
                .series(Series::new("character error rate", &eval_steps, &cers).color(Color::Green).width(2.0).dots())
                .y_range(0.0, 100.0)
                .x_range(0.0, last)
                .height(190.0),
        )
        .share_x()
        .save("../figures/ch09-training-curve.svg")
}

/// Reduce a frames x 28 grid of probabilities to the classes that matter
/// for one transcript: its own letters, the space, the blank, and one row
/// for all other letters together. Returns the grid and its row names,
/// ordered so that the blank is drawn at the top.
fn focus_on(probabilities: &Mat, text: &str) -> (Mat, Vec<String>) {
    let mut letters: Vec<usize> = Vec::new();
    for class in Alphabet::encode(text) {
        if class != Alphabet::SPACE && !letters.contains(&class) {
            letters.push(class);
        }
    }
    // Bottom to top: "other", the letters (last one lowest), space, blank.
    let mut classes: Vec<Option<usize>> = vec![None];
    classes.extend(letters.iter().rev().map(|&c| Some(c)));
    classes.extend([Some(Alphabet::SPACE), Some(Alphabet::BLANK)]);
    let names = classes
        .iter()
        .map(|class| match class {
            None => "other".to_string(),
            Some(Alphabet::BLANK) => "blank".to_string(),
            Some(Alphabet::SPACE) => "space".to_string(),
            Some(c) => Alphabet::symbol(*c).to_string(),
        })
        .collect();
    let grid = Mat::from_fn(probabilities.rows, classes.len(), |t, k| match classes[k] {
        Some(class) => probabilities[(t, class)],
        None => (2..Alphabet::SIZE).filter(|c| !letters.contains(c)).map(|c| probabilities[(t, c)]).sum(),
    });
    (grid, names)
}

/// The snapshots `ch09_train` saved: the transcript and `(step, probabilities)`.
fn read_snapshots() -> io::Result<(String, Vec<(usize, Mat)>)> {
    let bad = |what: &str| io::Error::new(io::ErrorKind::InvalidData, format!("{SNAPSHOT_PATH}: {what}"));
    let content = std::fs::read_to_string(SNAPSHOT_PATH)?;
    let mut lines = content.lines();
    let (mut text, mut snapshots) = (String::new(), Vec::new());
    while let Some(line) = lines.next() {
        if let Some(rest) = line.strip_prefix("text ") {
            text = rest.to_string();
        } else if let Some(rest) = line.strip_prefix("snapshot ") {
            let numbers: Vec<usize> = rest.split_whitespace().filter_map(|n| n.parse().ok()).collect();
            let [step, rows, cols] = numbers[..] else { return Err(bad("bad snapshot header")) };
            let mut data = Vec::with_capacity(rows * cols);
            for _ in 0..rows {
                data.extend(lines.next().ok_or_else(|| bad("file ends inside a snapshot"))?.split_whitespace().filter_map(|v| v.parse::<f32>().ok()));
            }
            if data.len() != rows * cols {
                return Err(bad("a snapshot has the wrong number of values"));
            }
            snapshots.push((step, Mat::from_vec(rows, cols, data)));
        }
    }
    Ok((text, snapshots))
}

/// The same utterance's posteriors at several points of training.
fn posteriors_over_training() -> io::Result<()> {
    let (text, snapshots) = read_snapshots()?;
    let wanted = [0usize, 100, 300, 1000];
    let mut chosen: Vec<&(usize, Mat)> = snapshots.iter().filter(|(step, _)| wanted.contains(step)).collect();
    if let Some(last) = snapshots.last() {
        if !chosen.iter().any(|(step, _)| *step == last.0) {
            chosen.push(last);
        }
    }
    let mut figure = Figure::new(&format!("What the network believes, frame by frame, as training goes on: \"{text}\""))
        .desc("Five heatmaps of per-frame class probabilities for one validation utterance, at training steps 0, 100, 300, 1000 and the end. Early on everything is smeared; then the blank takes over; then sharp spikes appear for each letter in order.");
    for (k, (step, probabilities)) in chosen.iter().enumerate() {
        let (grid, names) = focus_on(probabilities, &text);
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let duration = probabilities.rows as f32 * 0.02;
        let title = match *step {
            0 => "step 0: untrained, every frame a vague guess".to_string(),
            100 => "step 100: it has learned one thing, to say blank everywhere".to_string(),
            300 => "step 300: the first letters poke through".to_string(),
            1000 => "step 1,000: most letters are there, some still unsure".to_string(),
            s => format!("step {}: one sharp spike per letter, blank in between", thousands(s)),
        };
        let mut heatmap = Heatmap::new(&title, &grid).orientation(Orientation::TimeOnX).col_labels(&names).x_range(0.0, duration).value_range(0.0, 1.0).height(150.0);
        if k + 1 == chosen.len() {
            heatmap = heatmap.x_label("time (s)");
        }
        figure = figure.panel(heatmap);
    }
    figure.share_x().save("../figures/ch09-posteriors-over-training.svg")
}

/// 40000 -> "40,000".
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// The finished model on one test utterance: spectrogram above, posteriors
/// below, on the same time axis.
fn alignment_figure(recognizer: &Recognizer, utterance: &Utterance) -> io::Result<()> {
    let text = utterance.text();
    let heard = recognizer.transcribe(&utterance.samples, &Decoder::Greedy);
    let duration = utterance.seconds();
    let log_mel = recognizer.front_end.extractor().log_mel(&utterance.samples);
    let probabilities = recognizer.log_probs(&utterance.samples).map(f32::exp);
    let (grid, names) = focus_on(&probabilities, &text);
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    // The probability of "some letter or space" per frame: the spikes.
    let times: Vec<f32> = (0..probabilities.rows).map(|t| (t as f32 + 0.5) * duration / probabilities.rows as f32).collect();
    let not_blank: Vec<f32> = (0..probabilities.rows).map(|t| 1.0 - probabilities[(t, Alphabet::BLANK)]).collect();
    let mut spikes = LinePlot::new("probability that a frame is NOT blank: short spikes, one or two frames per letter").y_label("p(not blank)").series(Series::new("", &times, &not_blank).color(Color::Green)).x_range(0.0, duration).y_range(0.0, 1.0).height(90.0);
    for (&word, &(start, end)) in utterance.words.iter().zip(&utterance.spans) {
        spikes = spikes.span(start as f32 / 16000.0, end as f32 / 16000.0, WORDS[word]);
    }
    Figure::new(&format!("A test utterance through the finished recognizer. Said: \"{text}\". Heard: \"{heard}\"."))
        .desc("Three panels on a shared time axis: the log-mel spectrogram of a four-word utterance, the per-frame probabilities of the blank, the space and each letter of the transcript, and the probability of not-blank over time showing isolated spikes.")
        .panel(Heatmap::new("log-mel spectrogram (what goes in)", &log_mel).orientation(Orientation::TimeOnX).x_range(0.0, duration).y_label("mel band").height(130.0))
        .panel(Heatmap::new("per-frame probabilities, one output frame every 20 ms (what comes out)", &grid).orientation(Orientation::TimeOnX).col_labels(&names).x_range(0.0, duration).value_range(0.0, 1.0).height(190.0))
        .panel(spikes.x_label("time (s)"))
        .share_x()
        .save("../figures/ch09-alignment.svg")
}

/// Export a few test utterances (and one longer recording of several
/// utterances with pauses between them, for the live demo of chapter 11).
fn export_samples(recognizer: &Recognizer, test: &[Utterance]) -> io::Result<Vec<usize>> {
    // Readable examples: three or four words, mild noise or none, and ones
    // the recognizer gets right with plain greedy decoding. (The test set
    // also contains utterances it gets wrong; chapter 10 counts them.)
    let suitable = |utt: &Utterance, n: usize| utt.words.len() == n && utt.snr_db.is_none_or(|snr| snr >= 15.0) && recognizer.transcribe(&utt.samples, &Decoder::Greedy) == utt.text();
    let mut chosen: Vec<usize> = Vec::new();
    for n in [4, 3, 2, 5] {
        if let Some(i) = (0..test.len()).find(|&i| suitable(&test[i], n) && !chosen.contains(&i)) {
            chosen.push(i);
        }
    }
    let mut notes = String::from(
        "# Synthesized utterances\n\nMulti-word utterances for trying `speech transcribe` and `speech listen --from-file`\nwithout downloading any dataset. They were built by `code/examples/ch09_figures.rs` exactly\nlike the recognizer's test set (chapter 9): single-word clips from the **test** speakers of\nGoogle Speech Commands v0.02, trimmed, placed end to end with random pauses, with background\nnoise from the dataset's `_background_noise_` folder mixed in. The model never saw these\nspeakers or these stretches of noise in training.\n\n| File | Words | Background noise |\n|---|---|---|\n",
    );
    let mut sequence: Vec<f32> = Vec::new();
    let mut sequence_text = Vec::new();
    for &i in &chosen {
        let utt = &test[i];
        let name = format!("utt_{}.wav", utt.text().replace(' ', "_"));
        Wav::new(SAMPLE_RATE, utt.samples.clone()).write(format!("../samples/{name}"))?;
        let noise = utt.snr_db.map_or("none (faint hiss only)".to_string(), |snr| format!("SNR {snr:.0} dB"));
        notes.push_str(&format!("| `{name}` | \"{}\" | {noise} |\n", utt.text()));
        // For the long recording: this utterance, then 1.2 s of its own
        // quiet ending repeated (so the "silence" has the same faint hiss).
        sequence.extend_from_slice(&utt.samples);
        let tail = &utt.samples[utt.samples.len() - 800..];
        for k in 0..(1.2 * SAMPLE_RATE as f32) as usize {
            sequence.push(tail[k % tail.len()] * 0.5);
        }
        sequence_text.push(utt.text());
    }
    Wav::new(SAMPLE_RATE, sequence).write("../samples/utterances_with_pauses.wav")?;
    notes.push_str(&format!("| `utterances_with_pauses.wav` | the {} utterances above, one after another, 1.2 s of near-silence after each: \"{}\" | as above |\n", chosen.len(), sequence_text.join("\" ... \"")));
    notes.push_str(&format!(
        "\nThey are utterances {} of the fixed test set (seed {TEST_SEED}), chosen because they are short, not very\nnoisy, and recognized correctly: they are demonstrations, not a benchmark. Chapter 10 reports the\nerror rate over all {} test utterances.\n\n## License\n\nDerived from the Speech Commands dataset (Pete Warden, Google, 2018,\n<https://arxiv.org/abs/1804.03209>), which is licensed under\n[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).\nThese derived files are distributed under the same license. Changes made: clips were trimmed,\nchanged in speed by up to 10%, scaled, concatenated and mixed with the dataset's background noise.\n",
        chosen.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", "),
        test.len()
    ));
    std::fs::File::create("../samples/UTTERANCES.md")?.write_all(notes.as_bytes())?;
    Ok(chosen)
}

fn main() -> io::Result<()> {
    let threads = 8;
    let front_end = FrontEnd::new();
    let data = SpeechCommands::load_with(&dataset_dir(), &LoadOptions { threads, max_per_word: None, only: Some(vec![Split::Validation, Split::Test]) })?;
    let corpus = Corpus::new(&data, &load_background_noise(&dataset_dir())?, threads);

    synthesis_figure(&data, &corpus, &front_end)?;
    println!("wrote ../figures/ch09-synthesis.svg");
    training_curve()?;
    println!("wrote ../figures/ch09-training-curve.svg");
    posteriors_over_training()?;
    println!("wrote ../figures/ch09-posteriors-over-training.svg");

    let recognizer = Recognizer::load(Recognizer::default_path())?;
    let test = corpus.test_set(EVAL_UTTERANCES, threads);
    let chosen = export_samples(&recognizer, &test)?;
    println!("exported test utterances {chosen:?} to ../samples/ (see ../samples/UTTERANCES.md)");
    alignment_figure(&recognizer, &test[chosen[0]])?;
    println!("wrote ../figures/ch09-alignment.svg");
    Ok(())
}
