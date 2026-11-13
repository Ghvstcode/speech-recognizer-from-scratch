//! Chapter 10: decoding on toy inputs.
//!
//! ```text
//! cargo run --release --example ch10_decode_toy
//! ```
//!
//! 1. A three-frame table where greedy decoding gives the wrong answer and
//!    prefix beam search the right one, with the beam printed after every
//!    frame.
//! 2. A noisy made-up network output for "stop go left", decoded greedily,
//!    with a plain beam search, and with a beam search restricted to a
//!    vocabulary.
//! 3. Scoring: reference and hypothesis lined up word by word, and WER / CER.

use speech::decode::{self, Alphabet, BeamEntry, Lexicon, LexiconOptions};
use speech::mat::Mat;
use speech::metrics::{self, ErrorTally};
use speech::rng::Rng;

/// The 35 words of the Speech Commands dataset: the vocabulary of the
/// recognizer we train in chapter 9.
const VOCABULARY: [&str; 35] = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes",
    "zero",
];

/// Class names for part 1: blank, a, b.
const TOY_SYMBOLS: [char; 3] = ['_', 'a', 'b'];

/// A toy labelling as text, with "" shown as (empty).
fn toy_string(labels: &[usize]) -> String {
    if labels.is_empty() {
        "(empty)".to_string()
    } else {
        labels.iter().map(|&c| TOY_SYMBOLS[c]).collect()
    }
}

fn print_beam(beam: &[BeamEntry]) {
    println!("      prefix    p(ends in blank)  p(ends in label)   total");
    for entry in beam {
        println!(
            "      {:<9} {:>12.4} {:>17.4} {:>11.4}",
            toy_string(&entry.labels),
            entry.log_p_blank.exp(),
            entry.log_p_non_blank.exp(),
            entry.log_total().exp()
        );
    }
}

fn part_one() {
    println!("== Part 1: greedy fails, beam search does not ==\n");
    let probabilities: [[f32; 3]; 3] = [[0.5, 0.4, 0.1], [0.5, 0.4, 0.1], [0.3, 0.1, 0.6]];
    let log_probs = Mat::from_fn(3, 3, |t, k| probabilities[t][k].ln());

    println!("  frame   p(_)   p(a)   p(b)");
    for (t, row) in probabilities.iter().enumerate() {
        println!("    {t}     {:.1}    {:.1}    {:.1}", row[0], row[1], row[2]);
    }

    let best_path = decode::best_path(&log_probs);
    let path_probability: f32 = best_path.iter().enumerate().map(|(t, &k)| probabilities[t][k]).product();
    println!("\nGreedy: the best class in each frame is {}, which collapses to \"{}\".", toy_string(&best_path), toy_string(&decode::greedy(&log_probs)));
    println!("That one alignment has probability {path_probability:.3}.");

    println!("\nThe truth, by adding up all 27 alignments:");
    for hypothesis in decode::all_labellings(&log_probs) {
        println!("    p({:<7}) = {:.4}", toy_string(&hypothesis.labels), hypothesis.log_prob.exp());
    }

    for width in [3, 10] {
        println!("\nPrefix beam search, beam width {width}:");
        let mut beam = decode::initial_beam();
        println!("    before any frame:");
        print_beam(&beam);
        for t in 0..log_probs.rows {
            beam = decode::beam_step(&beam, log_probs.row(t), width);
            println!("    after frame {t}:");
            print_beam(&beam);
        }
        println!("    answer: \"{}\" with probability {:.4}", toy_string(&beam[0].labels), beam[0].log_total().exp());
    }
}

/// Make up a network output for `text`: every character is held for two
/// frames, with a blank frame between characters, and the "network" is
/// right-but-unsure everywhere (Gaussian noise on the logits). On top of
/// the noise, each `(character index, wrong letter)` pair in `confusions`
/// pushes a wrong letter slightly ahead of the right one for that character,
/// the way a real model mishears similar sounds.
fn synthetic_log_probs(text: &str, confusions: &[(usize, char)], rng: &mut Rng) -> Mat {
    let labels = Alphabet::encode(text);
    let mut frame_classes = Vec::new();
    for &label in &labels {
        frame_classes.extend([label, label, Alphabet::BLANK]);
    }
    let mut logits = Mat::from_fn(frame_classes.len(), Alphabet::SIZE, |_, _| 0.8 * rng.normal());
    for (t, &class) in frame_classes.iter().enumerate() {
        logits[(t, class)] += 4.0;
    }
    for &(character, wrong) in confusions {
        let wrong_class = Alphabet::encode(&wrong.to_string())[0];
        for t in [3 * character, 3 * character + 1] {
            logits[(t, wrong_class)] = logits[(t, labels[character])] + 0.4;
        }
    }
    decode::log_softmax(&logits)
}

fn part_two() -> (String, String) {
    println!("\n== Part 2: a vocabulary to the rescue ==\n");
    let reference = "stop go left";
    // Mishear the "p" of stop as "b" and the "f" of left as "s".
    let confusions = [(3, 'b'), (10, 's')];
    let log_probs = synthetic_log_probs(reference, &confusions, &mut Rng::new(10));
    println!("A made-up, noisy network output for \"{reference}\": {} frames x {} classes.", log_probs.rows, log_probs.cols);
    println!("(We rigged it to slightly prefer \"b\" over the \"p\" of stop, and \"s\" over the \"f\" of left.)");
    println!("The most probable class in each frame:");
    let best_path: String = decode::best_path(&log_probs).iter().map(|&c| Alphabet::symbol(c)).collect();
    println!("    |{best_path}|");

    let greedy = Alphabet::decode(&decode::greedy(&log_probs));
    println!("\n  greedy:                    \"{greedy}\"");

    let beam = decode::prefix_beam_search(&log_probs, 32);
    println!("  prefix beam search (32):   \"{}\"   ln p = {:.3}", Alphabet::decode(&beam[0].labels), beam[0].log_prob);

    let lexicon = Lexicon::new(&VOCABULARY);
    let constrained = decode::lexicon_beam_search(&log_probs, &lexicon, &LexiconOptions::new(32));
    println!("  with the 35-word lexicon:  \"{}\"   ln p = {:.3}", constrained[0].text(), constrained[0].log_prob);
    let wide = decode::lexicon_beam_search(&log_probs, &lexicon, &LexiconOptions::new(1000));
    println!("\n  The five best word sequences, from a much wider beam (1000):");
    for hypothesis in wide.iter().take(5) {
        println!("    ln p = {:>8.3}   \"{}\"", hypothesis.log_prob, hypothesis.text());
    }
    (reference.to_string(), greedy)
}

fn part_three(reference: &str, greedy: &str) {
    println!("\n== Part 3: keeping score ==\n");
    let pairs = [
        (reference, greedy),
        ("turn the lights off", "turn light off now"),
        ("three two one go", "tree two one go"),
        ("yes", "no no no"),
        ("stop", "stop"),
    ];
    let mut tally = ErrorTally::new();
    for (reference, hypothesis) in pairs {
        println!("{}", metrics::format_alignment(reference, hypothesis));
        println!("     WER {:>5.1}%   CER {:>5.1}%\n", 100.0 * metrics::wer(reference, hypothesis), 100.0 * metrics::cer(reference, hypothesis));
        tally.add(reference, hypothesis);
    }
    println!("All {} utterances together:", tally.utterances);
    println!("  WER = {} word edits / {} reference words = {:.1}%", tally.word_edits, tally.reference_words, 100.0 * tally.wer());
    println!("  CER = {} character edits / {} reference characters = {:.1}%", tally.char_edits, tally.reference_chars, 100.0 * tally.cer());
    let mean_of_rates: f32 = pairs.iter().map(|(r, h)| metrics::wer(r, h)).sum::<f32>() / pairs.len() as f32;
    println!("  (the mean of the five per-utterance WERs would be {:.1}%: not the same thing)", 100.0 * mean_of_rates);
}

fn main() {
    if std::env::args().count() > 1 {
        eprintln!("usage: ch10_decode_toy   (takes no arguments)");
        std::process::exit(2);
    }
    part_one();
    let (reference, greedy) = part_two();
    part_three(&reference, &greedy);
}
