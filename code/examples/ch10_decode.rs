//! Chapter 10: three decoders, one network, measured.
//!
//! Loads the recognizer trained in chapter 9 and compares, on the fixed
//! test set of 2,000 synthesized utterances:
//!
//! * greedy decoding (best class per frame, collapse),
//! * prefix beam search at several beam widths,
//! * lexicon-constrained beam search over the 35 known words, with its one
//!   knob (the per-word bonus) tuned on the *validation* set.
//!
//! ```text
//! cargo run --release --example ch10_decode                      # the full comparison (needs the dataset)
//! cargo run --release --example ch10_decode -- files a.wav b.wav   # all decoders on your own recordings
//! ```
//!
//! The network runs once per utterance; every decoder then works on the
//! same grid of log-probabilities, so differences between rows of the table
//! are differences between decoders and nothing else.

use speech::data::pipeline::{decode, load_audio_16k, Decoder, Recognizer};
use speech::data::speech_commands::{dataset_dir, load_background_noise, parallel_map, LoadOptions, SpeechCommands, Split, WORDS};
use speech::data::synth::{Corpus, TrimConfig, SynthConfig, Synthesizer, Utterance, EVAL_UTTERANCES};
use speech::mat::Mat;
use speech::metrics::{align, format_alignment, words, EditOp, ErrorTally};
use speech::plot::{BarChart, Color, LinePlot, Series};
use std::collections::HashMap;
use std::time::Instant;

const USAGE: &str = "usage: ch10_decode [--model ../models/recognizer.bin] [--threads 10] [--no-figures]
       ch10_decode files <file.wav>... [--model ../models/recognizer.bin]";

/// One evaluation utterance after the network has run on it.
struct Scored {
    /// The truth.
    text: String,
    /// Number of words in the truth.
    n_words: usize,
    /// Signal-to-noise ratio, `None` when no background recording was mixed in.
    snr_db: Option<f32>,
    /// Whether all words came from one speaker.
    same_speaker: bool,
    /// The network's output: frames x 28 log-probabilities.
    log_probs: Mat,
}

fn score(recognizer: &Recognizer, utterances: &[Utterance], threads: usize) -> Vec<Scored> {
    parallel_map(utterances, threads, |_, utt| Scored { text: utt.text(), n_words: utt.words.len(), snr_db: utt.snr_db, same_speaker: utt.same_speaker, log_probs: recognizer.log_probs(&utt.samples) })
}

/// Decode every utterance of a set. Returns the hypotheses and the mean
/// decoding time per utterance in milliseconds (the decoder alone; the
/// network's time is not included).
fn decode_all(recognizer: &Recognizer, set: &[Scored], decoder: &Decoder, threads: usize) -> (Vec<String>, f32) {
    let results = parallel_map(set, threads, |_, scored| {
        let started = Instant::now();
        let text = decode(&scored.log_probs, decoder, &recognizer.lexicon);
        (text, started.elapsed().as_secs_f32())
    });
    let total: f32 = results.iter().map(|(_, seconds)| seconds).sum();
    (results.into_iter().map(|(text, _)| text).collect(), 1000.0 * total / set.len().max(1) as f32)
}

/// Error counts over the utterances selected by `keep`.
fn tally(set: &[Scored], hypotheses: &[String], keep: impl Fn(&Scored) -> bool) -> ErrorTally {
    let mut tally = ErrorTally::new();
    for (scored, hypothesis) in set.iter().zip(hypotheses).filter(|(scored, _)| keep(scored)) {
        tally.add(&scored.text, hypothesis);
    }
    tally
}

fn table_row(name: &str, tally: &ErrorTally, ms: f32) {
    println!("| {name:<34} | {:>5.2}% | {:>5.2}% | {:>10.1}% | {:>8.2} ms |", 100.0 * tally.cer(), 100.0 * tally.wer(), 100.0 * tally.exact_match_rate(), ms);
}

/// Count which reference word was replaced by which hypothesis word.
fn substitutions(set: &[Scored], hypotheses: &[String]) -> Vec<(usize, String, String)> {
    let mut counts: HashMap<(String, String), usize> = HashMap::new();
    for (scored, hypothesis) in set.iter().zip(hypotheses) {
        let (reference, hypothesis) = (words(&scored.text), words(hypothesis));
        let (mut i, mut j) = (0, 0);
        for op in align(&reference, &hypothesis) {
            match op {
                EditOp::Match => (i, j) = (i + 1, j + 1),
                EditOp::Substitute => {
                    *counts.entry((reference[i].clone(), hypothesis[j].clone())).or_insert(0) += 1;
                    (i, j) = (i + 1, j + 1);
                }
                EditOp::Delete => {
                    *counts.entry((reference[i].clone(), "(nothing)".to_string())).or_insert(0) += 1;
                    i += 1;
                }
                EditOp::Insert => {
                    *counts.entry(("(nothing)".to_string(), hypothesis[j].clone())).or_insert(0) += 1;
                    j += 1;
                }
            }
        }
    }
    let mut list: Vec<(usize, String, String)> = counts.into_iter().map(|((said, heard), count)| (count, said, heard)).collect();
    list.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
    list
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut model_path, mut threads, mut figures, mut files, mut files_mode) = (Recognizer::default_path(), 10usize, true, Vec::new(), false);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "files" if i == 0 => files_mode = true,
            "--no-figures" => figures = false,
            "--model" | "--threads" => {
                let Some(value) = args.get(i + 1) else {
                    eprintln!("error: {} needs a value\n{USAGE}", args[i]);
                    std::process::exit(2);
                };
                if args[i] == "--model" {
                    model_path = value.into();
                } else {
                    threads = value.parse().unwrap_or(10).max(1);
                }
                i += 1;
            }
            other if files_mode && !other.starts_with("--") => files.push(other.to_string()),
            other => {
                eprintln!("error: unexpected argument {other:?}\n{USAGE}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let recognizer = Recognizer::load(&model_path)?;
    let lexicon_decoder = Decoder::Lexicon { width: Decoder::DEFAULT_BEAM, word_bonus: Decoder::DEFAULT_WORD_BONUS };

    if files_mode {
        for file in &files {
            let log_probs = recognizer.log_probs(&load_audio_16k(file)?);
            println!("{file}");
            for decoder in [Decoder::Greedy, Decoder::Beam { width: 16 }, lexicon_decoder] {
                println!("  {:<8} {}", decoder.name(), decode(&log_probs, &decoder, &recognizer.lexicon));
            }
        }
        return Ok(());
    }

    // ---- data: the same fixed sets chapter 9 evaluated on ------------------
    let started = Instant::now();
    let data = SpeechCommands::load_with(&dataset_dir(), &LoadOptions { threads, max_per_word: None, only: Some(vec![Split::Validation, Split::Test]) })?;
    let corpus = Corpus::new(&data, &load_background_noise(&dataset_dir())?, &TrimConfig::default(), threads);
    let validation = score(&recognizer, &corpus.validation_set(EVAL_UTTERANCES, threads), threads);
    let test_utterances = corpus.test_set(EVAL_UTTERANCES, threads);
    let test = score(&recognizer, &test_utterances, threads);
    let audio_seconds: f32 = test_utterances.iter().map(Utterance::seconds).sum();
    println!("model: {} ({} weights)", model_path.display(), recognizer.model.num_params());
    println!("{} validation and {} test utterances ({:.0} min of test audio) scored in {:.1} s\n", validation.len(), test.len(), audio_seconds / 60.0, started.elapsed().as_secs_f32());

    // How long does the network itself take? One thread, 200 utterances.
    let started = Instant::now();
    for utt in &test_utterances[..200] {
        std::hint::black_box(recognizer.log_probs(&utt.samples));
    }
    let network_ms = 1000.0 * started.elapsed().as_secs_f32() / 200.0;
    let mean_seconds = test_utterances[..200].iter().map(Utterance::seconds).sum::<f32>() / 200.0;
    println!("features + network: {network_ms:.1} ms per utterance on one thread (mean length {mean_seconds:.2} s, so {:.0}x faster than real time)\n", 1000.0 * mean_seconds / network_ms);

    // ---- tune the word bonus on VALIDATION ---------------------------------
    println!("tuning the lexicon decoder's word bonus on the validation set (beam {}):", Decoder::DEFAULT_BEAM);
    let mut best_bonus = (f32::MAX, 0.0f32);
    for bonus in [-4.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0] {
        let (hypotheses, _) = decode_all(&recognizer, &validation, &Decoder::Lexicon { width: Decoder::DEFAULT_BEAM, word_bonus: bonus }, threads);
        let wer = tally(&validation, &hypotheses, |_| true).wer();
        println!("  word bonus {bonus:>+5.1}   validation WER {:.2}%", 100.0 * wer);
        if wer < best_bonus.0 {
            best_bonus = (wer, bonus);
        }
    }
    let bonus = best_bonus.1;
    println!("  -> using word bonus {bonus:+.1}\n");

    // ---- the comparison, on TEST -------------------------------------------
    let decoders: Vec<(String, Decoder)> = vec![
        ("greedy".to_string(), Decoder::Greedy),
        ("beam search, width 4".to_string(), Decoder::Beam { width: 4 }),
        ("beam search, width 16".to_string(), Decoder::Beam { width: 16 }),
        ("beam search, width 64".to_string(), Decoder::Beam { width: 64 }),
        ("lexicon beam, width 4".to_string(), Decoder::Lexicon { width: 4, word_bonus: bonus }),
        ("lexicon beam, width 16".to_string(), Decoder::Lexicon { width: 16, word_bonus: bonus }),
        ("lexicon beam, width 64".to_string(), Decoder::Lexicon { width: 64, word_bonus: bonus }),
    ];
    println!("| decoder (test set, {} utterances) |    CER |    WER | exact match | decode time |", test.len());
    println!("|------------------------------------|-------:|-------:|------------:|------------:|");
    let mut outputs: Vec<Vec<String>> = Vec::new();
    let mut tallies: Vec<ErrorTally> = Vec::new();
    for (name, decoder) in &decoders {
        let (hypotheses, ms) = decode_all(&recognizer, &test, decoder, threads);
        let t = tally(&test, &hypotheses, |_| true);
        table_row(name, &t, ms);
        outputs.push(hypotheses);
        tallies.push(t);
    }
    let (greedy, beam, lexicon) = (&outputs[0], &outputs[2], &outputs[5]);
    let empty = lexicon.iter().filter(|h| h.is_empty()).count();
    println!("\n(the lexicon decoder returned nothing for {empty} of {} utterances: no sequence of whole words survived in its beam)", test.len());

    // ---- aligned examples --------------------------------------------------
    println!("\nErrors greedy makes that the lexicon repairs:\n");
    let repaired: Vec<usize> = (0..test.len()).filter(|&i| greedy[i] != test[i].text && lexicon[i] == test[i].text).collect();
    for &i in repaired.iter().take(6) {
        println!("greedy:\n{}", format_alignment(&test[i].text, &greedy[i]));
        println!("lexicon:\n{}\n", format_alignment(&test[i].text, &lexicon[i]));
    }
    println!("Errors that survive the lexicon:\n");
    let surviving: Vec<usize> = (0..test.len()).filter(|&i| lexicon[i] != test[i].text).collect();
    for &i in surviving.iter().take(5) {
        println!("greedy:\n{}", format_alignment(&test[i].text, &greedy[i]));
        println!("lexicon:\n{}\n", format_alignment(&test[i].text, &lexicon[i]));
    }
    let broken = (0..test.len()).filter(|&i| greedy[i] == test[i].text && lexicon[i] != test[i].text).count();
    let beam_fixed = (0..test.len()).filter(|&i| greedy[i] != test[i].text && beam[i] == test[i].text).count();
    let beam_broke = (0..test.len()).filter(|&i| greedy[i] == test[i].text && beam[i] != test[i].text).count();
    println!("utterances greedy gets wrong and the lexicon gets right: {}; the other way round: {broken}", repaired.len());
    println!("utterances greedy gets wrong and plain beam search (16) gets right: {beam_fixed}; the other way round: {beam_broke}");
    println!("utterances still wrong with the lexicon: {}\n", surviving.len());

    // ---- breakdowns --------------------------------------------------------
    println!("| words in utterance | utterances | greedy WER | lexicon WER | greedy exact | lexicon exact |");
    println!("|-------------------:|-----------:|-----------:|------------:|-------------:|--------------:|");
    let max_words = test.iter().map(|s| s.n_words).max().unwrap_or(1);
    let (mut length_greedy, mut length_lexicon, mut length_any_error) = (Vec::new(), Vec::new(), Vec::new());
    for n in 1..=max_words {
        let (g, l) = (tally(&test, greedy, |s| s.n_words == n), tally(&test, lexicon, |s| s.n_words == n));
        println!("| {n:>18} | {:>10} | {:>9.2}% | {:>10.2}% | {:>11.1}% | {:>12.1}% |", g.utterances, 100.0 * g.wer(), 100.0 * l.wer(), 100.0 * g.exact_match_rate(), 100.0 * l.exact_match_rate());
        length_greedy.push(100.0 * g.wer());
        length_lexicon.push(100.0 * l.wer());
        length_any_error.push(100.0 * (1.0 - l.exact_match_rate()));
    }

    println!("\n| noise                      | utterances | greedy WER | lexicon WER |");
    println!("|----------------------------|-----------:|-----------:|------------:|");
    type NoiseFilter = fn(&Scored) -> bool;
    let noise_levels: [(&str, NoiseFilter); 5] = [
        ("none (hiss only)", |s| s.snr_db.is_none()),
        ("SNR 20 to 30 dB", |s| s.snr_db.is_some_and(|snr| snr >= 20.0)),
        ("SNR 15 to 20 dB", |s| s.snr_db.is_some_and(|snr| (15.0..20.0).contains(&snr))),
        ("SNR 10 to 15 dB", |s| s.snr_db.is_some_and(|snr| (10.0..15.0).contains(&snr))),
        ("SNR 5 to 10 dB", |s| s.snr_db.is_some_and(|snr| snr < 10.0)),
    ];
    for (name, keep) in noise_levels {
        let (g, l) = (tally(&test, greedy, keep), tally(&test, lexicon, keep));
        println!("| {name:<26} | {:>10} | {:>9.2}% | {:>10.2}% |", g.utterances, 100.0 * g.wer(), 100.0 * l.wer());
    }

    println!("\n| speakers                   | utterances | greedy WER | lexicon WER |");
    println!("|----------------------------|-----------:|-----------:|------------:|");
    for (name, same) in [("one speaker per utterance", true), ("a new speaker per word", false)] {
        // Single-word utterances have one speaker either way; leave them out.
        let (g, l) = (tally(&test, greedy, |s| s.same_speaker == same && s.n_words > 1), tally(&test, lexicon, |s| s.same_speaker == same && s.n_words > 1));
        println!("| {name:<26} | {:>10} | {:>9.2}% | {:>10.2}% |", g.utterances, 100.0 * g.wer(), 100.0 * l.wer());
    }

    println!("\nmost frequent word errors on the test set (said -> heard):");
    println!("  {:<34} lexicon", "greedy");
    let (greedy_subs, lexicon_subs) = (substitutions(&test, greedy), substitutions(&test, lexicon));
    for k in 0..12 {
        let show = |list: &[(usize, String, String)]| list.get(k).map_or(String::new(), |(count, said, heard)| format!("{count:>3}  {said} -> {heard}"));
        println!("  {:<34} {}", show(&greedy_subs), show(&lexicon_subs));
    }
    let in_vocabulary = greedy.iter().flat_map(|h| words(h)).filter(|w| WORDS.contains(&w.as_str())).count();
    let all_words = greedy.iter().flat_map(|h| words(h)).count();
    println!("\n{:.1}% of the words greedy decoding spells are not among the 35 words at all (e.g. \"lefd\"); the lexicon makes that impossible.", 100.0 * (1.0 - in_vocabulary as f32 / all_words.max(1) as f32));

    // ---- beyond what it was trained on -------------------------------------
    println!("\n| other test conditions                         | utterances | greedy WER | lexicon WER | lexicon exact |");
    println!("|-----------------------------------------------|-----------:|-----------:|------------:|--------------:|");
    let lexicon_tuned = Decoder::Lexicon { width: 16, word_bonus: bonus };
    let report = |name: &str, set: &[Scored]| {
        let (g, _) = decode_all(&recognizer, set, &Decoder::Greedy, threads);
        let (l, _) = decode_all(&recognizer, set, &lexicon_tuned, threads);
        let (g, l) = (tally(set, &g, |_| true), tally(set, &l, |_| true));
        println!("| {name:<45} | {:>10} | {:>9.2}% | {:>10.2}% | {:>12.1}% |", g.utterances, 100.0 * g.wer(), 100.0 * l.wer(), 100.0 * l.exact_match_rate());
    };
    // Longer utterances than any it saw in training (which stopped at 5 words).
    let long_recipe = SynthConfig { min_words: 6, max_words: 8, ..SynthConfig::default() };
    let long = Synthesizer::new(&corpus.test.words, &corpus.test.noise, long_recipe).utterances(3003, 0, 500, threads);
    report("6 to 8 words (training stopped at 5)", &score(&recognizer, &long, threads));
    // No pauses at all: every gap is an overlap of 20 to 40 ms.
    let tight_recipe = SynthConfig { min_words: 2, gap_ms: (-40.0, -20.0), ..SynthConfig::default() };
    let tight = Synthesizer::new(&corpus.test.words, &corpus.test.noise, tight_recipe).utterances(4004, 0, 500, threads);
    report("2 to 5 words, no pauses (clips overlap)", &score(&recognizer, &tight, threads));
    // Much more noise than in training.
    let loud_recipe = SynthConfig { noise_prob: 1.0, snr_db: (-5.0, 5.0), ..SynthConfig::default() };
    let loud = Synthesizer::new(&corpus.test.words, &corpus.test.noise, loud_recipe).utterances(5005, 0, 500, threads);
    report("SNR -5 to 5 dB (training stopped at 5)", &score(&recognizer, &loud, threads));
    // The original one-second test clips, untouched.
    let clips: Vec<_> = data.split(Split::Test).collect();
    let single: Vec<Scored> = parallel_map(&clips, threads, |_, clip| Scored { text: WORDS[clip.word].to_string(), n_words: 1, snr_db: None, same_speaker: true, log_probs: recognizer.log_probs(&clip.audio()) });
    report("single-word test clips, exactly as recorded", &single);

    if figures {
        let labels = ["greedy", "beam 4", "beam 16", "beam 64", "lexicon 4", "lexicon 16", "lexicon 64"];
        let wers: Vec<f32> = tallies.iter().map(|t| 100.0 * t.wer()).collect();
        BarChart::new("Word error rate on the test set, by decoder", &labels, &wers)
            .y_label("word error rate (%)")
            .show_values(2)
            .highlight(0, Color::Amber)
            .highlight(5, Color::Green)
            .height(240.0)
            .save("../figures/ch10-decoders.svg")?;
        let xs: Vec<f32> = (1..=max_words).map(|n| n as f32).collect();
        LinePlot::new("Per word, length hardly matters; per utterance, every extra word is another chance to slip")
            .x_label("words in the utterance")
            .y_label("error (%)")
            .series(Series::new("word error rate, greedy", &xs, &length_greedy).color(Color::Amber).width(2.0).dots())
            .series(Series::new("word error rate, lexicon", &xs, &length_lexicon).color(Color::Green).width(2.0).dots())
            .series(Series::new("utterances with any error, lexicon", &xs, &length_any_error).color(Color::Ink48).dashed().dots())
            .y_range(0.0, (length_any_error.iter().cloned().fold(0.0f32, f32::max) * 1.2).max(1.0))
            .x_range(1.0, max_words as f32)
            .height(240.0)
            .save("../figures/ch10-wer-by-length.svg")?;
        println!("\nwrote ../figures/ch10-decoders.svg and ../figures/ch10-wer-by-length.svg");
    }
    Ok(())
}
