//! Chapter 5: recognizing spoken digits by template matching.
//!
//! Our first recognizer learns nothing. It keeps labeled recordings (the
//! *templates*), and to recognize a new recording it finds the template that
//! is closest under dynamic time warping and answers with that template's
//! digit. This program measures how far that idea goes on the Free Spoken
//! Digit Dataset:
//!
//! ```text
//! cargo run --release --example ch05_digits                 # every experiment, and the figures
//! cargo run --release --example ch05_digits -- classify ../samples/seven_jackson.wav
//! ```
//!
//! The experiments, in the order they are printed:
//!
//! 0. Which features? An ablation of the MFCC recipe, on training clips only.
//! 1. One template per digit, same speaker: the voice dialing setup.
//! 2. More templates per digit, same speaker.
//! 3. All six speakers, the dataset's official train/test split.
//! 4. A speaker the recognizer has never heard (leave one speaker out).
//! 5. What one answer costs.
//!
//! The dataset is looked for in `$SPEECH_DATA/fsdd/recordings`
//! (`../data/fsdd/recordings` by default). The experiments need it.
//! `classify` works without it too: it then falls back to the six digit
//! clips in `../samples/` as its templates, which is enough to tell a
//! "zero" from a "three" from a "seven" and to see the method at work.

use speech::data::fsdd::{self, Clip, MfccFrontEnd, MfccRecipe, Normalization, Split, DIGIT_NAMES, N_DIGITS, SAMPLE_RATE};
use speech::dsp::frame::find_speech;
use speech::dsp::resample::resample;
use speech::dtw::{dtw_distance, dtw_full, DtwResult, TemplateBank};
use speech::mat::Mat;
use speech::plot::{BarChart, Color, Figure, Heatmap, LinePlot, Orientation, Series};
use speech::rng::Rng;
use speech::wav::Wav;
use std::io;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: ch05_digits                      run every experiment and write the figures
       ch05_digits classify FILE.wav    recognize one recording and show its nearest templates";

/// Where the figures go, relative to `code/`.
const FIGURES: &str = "../figures";

/// The Sakoe-Chiba band used by the experiments: none. The ablation shows a
/// band changes the accuracy by nothing we can measure and halves the time;
/// we leave it off so that the recognizer is exactly the DTW of the chapter.
const BAND: Option<usize> = None;

/// Every recording of the dataset together with its features, computed once.
/// Experiments refer to clips by their position in these two lists.
struct Corpus {
    clips: Vec<Clip>,
    features: Vec<Mat>,
}

impl Corpus {
    /// Compute the features of every clip with `recipe`, on all cores.
    fn new(clips: Vec<Clip>, recipe: &MfccRecipe) -> Corpus {
        let front_end = MfccFrontEnd::new(recipe, SAMPLE_RATE);
        let features = fsdd::map_clips(&clips, |clip| front_end.features(&clip.samples));
        Corpus { clips, features }
    }

    /// A template bank holding the clips at `positions`.
    fn bank(&self, positions: &[usize], band: Option<usize>) -> TemplateBank {
        let mut bank = TemplateBank { templates: Vec::new(), band };
        for &p in positions {
            bank.add(self.clips[p].digit, self.features[p].clone());
        }
        bank
    }

    /// Recognize every test clip of `split` using its training clips as
    /// templates, letting the `k` nearest templates vote. Returns one
    /// predicted digit per test clip.
    fn recognize(&self, split: &Split, k: usize) -> Vec<usize> {
        let queries: Vec<Mat> = split.test.iter().map(|&p| self.features[p].clone()).collect();
        self.bank(&split.train, BAND).classify_many(&queries, k).into_iter().map(|(digit, _)| digit).collect()
    }

    /// Percentage of `predicted` digits that are what was really said in the
    /// clips at `positions`.
    fn percent_correct(&self, positions: &[usize], predicted: &[usize]) -> f32 {
        let correct = positions.iter().zip(predicted).filter(|&(&p, &digit)| self.clips[p].digit == digit).count();
        100.0 * correct as f32 / positions.len() as f32
    }

    /// Positions of all clips for which `keep` says yes.
    fn positions(&self, keep: impl Fn(&Clip) -> bool) -> Vec<usize> {
        (0..self.clips.len()).filter(|&p| keep(&self.clips[p])).collect()
    }
}

/// The DTW distance from every query to every template:
/// `table[q][t] = dtw_distance(queries[q], templates[t])`, computed on all
/// cores.
///
/// Several experiments ask "what if only *these* templates were allowed?"
/// many times over. The distances do not depend on the answer, so we compute
/// them once and then only pick the smallest among the allowed columns.
fn distance_table(features: &[Mat], queries: &[usize], templates: &[usize], band: Option<usize>) -> Vec<Vec<f32>> {
    fsdd::map_clips(queries, |&q| templates.iter().map(|&t| dtw_distance(&features[q], &features[t], band)).collect())
}

/// The column of the smallest distance in `row` among the columns `allowed`
/// accepts.
fn nearest_allowed(row: &[f32], allowed: impl Fn(usize) -> bool) -> usize {
    let mut best = None;
    for (column, &distance) in row.iter().enumerate() {
        if allowed(column) && best.is_none_or(|b: usize| distance < row[b]) {
            best = Some(column);
        }
    }
    best.expect("no template was allowed")
}

/// The time since `start`, for printing: "0.41 s", or "3 ms" when it is short.
fn seconds_since(start: Instant) -> String {
    let seconds = start.elapsed().as_secs_f64();
    if seconds < 0.1 {
        format!("{:.0} ms", 1000.0 * seconds)
    } else {
        format!("{seconds:.2} s")
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

// ---------------------------------------------------------------------------
// Experiment 0: which features?
// ---------------------------------------------------------------------------

/// One row of the ablation: the settled recipe with one thing changed.
struct Variant {
    name: &'static str,
    recipe: MfccRecipe,
    band: Option<usize>,
}

fn variants() -> Vec<Variant> {
    let settled = MfccRecipe::standard();
    let with = |name, recipe| Variant { name, recipe, band: None };
    vec![
        with("settled: 13 MFCCs, c0 dropped, trimmed", settled.clone()),
        with("  keep c0 (overall level)", MfccRecipe { keep_c0: true, ..settled.clone() }),
        with("  add deltas", MfccRecipe { deltas: true, ..settled.clone() }),
        with("  subtract the clip's mean", MfccRecipe { normalization: Normalization::Mean, ..settled.clone() }),
        with("  normalize mean and variance", MfccRecipe { normalization: Normalization::MeanVariance, ..settled.clone() }),
        with("  do not trim silence", MfccRecipe { trim: false, ..settled.clone() }),
        with("  6 coefficients", MfccRecipe { n_coeffs: 6, ..settled.clone() }),
        with("  8 coefficients", MfccRecipe { n_coeffs: 8, ..settled.clone() }),
        with("  20 coefficients", MfccRecipe { n_coeffs: 20, ..settled.clone() }),
        with("  30 coefficients", MfccRecipe { n_coeffs: 30, ..settled.clone() }),
        Variant { name: "  Sakoe-Chiba band of 2 frames", recipe: settled.clone(), band: Some(2) },
        Variant { name: "  Sakoe-Chiba band of 5 frames", recipe: settled.clone(), band: Some(5) },
        Variant { name: "  Sakoe-Chiba band of 10 frames", recipe: settled.clone(), band: Some(10) },
        with(
            "textbook: c0, deltas, mean+variance",
            MfccRecipe { keep_c0: true, deltas: true, normalization: Normalization::MeanVariance, ..settled },
        ),
    ]
}

/// Try each variant of the feature recipe on a *development* split made only
/// of training takes: takes 5 to 19 are the queries, takes 20 to 49 the
/// templates. The official test takes (0 to 4) are not touched, so choosing
/// the best row here does not flatter the test results below.
///
/// One table of distances per variant answers three questions at once,
/// depending on which templates each query is allowed to see:
///
/// * `1 own`: a single template per digit, from the query's own speaker;
/// * `known`: every template, the query's own speaker included;
/// * `unseen`: only templates from the other five speakers.
fn ablation(clips: &[Clip]) {
    println!("0. Which features? (development split: takes 5-19 are queries, takes 20-49 are templates)\n");
    println!("   {:<40} {:>5} {:>7} {:>7} {:>7} {:>8}", "recipe", "dims", "1 own", "known", "unseen", "time");
    for variant in variants() {
        let corpus = Corpus::new(clips.to_vec(), &variant.recipe);
        let queries = corpus.positions(|c| (5..20).contains(&c.index));
        let templates = corpus.positions(|c| c.index >= 20);
        let start = Instant::now();
        let table = distance_table(&corpus.features, &queries, &templates, variant.band);
        let time = seconds_since(start);

        // Percentage of queries whose nearest allowed template has the right digit.
        let accuracy = |allowed: &dyn Fn(&Clip, &Clip) -> bool| {
            let correct = queries.iter().zip(&table).filter(|&(&q, row)| {
                let query = &corpus.clips[q];
                let nearest = nearest_allowed(row, |t| allowed(query, &corpus.clips[templates[t]]));
                corpus.clips[templates[nearest]].digit == query.digit
            });
            100.0 * correct.count() as f32 / queries.len() as f32
        };
        let one_own = accuracy(&|query, template| template.speaker == query.speaker && template.index == 20);
        let known = accuracy(&|_, _| true);
        let unseen = accuracy(&|query, template| template.speaker != query.speaker);
        println!(
            "   {:<40} {:>5} {:>6.1}% {:>6.1}% {:>6.1}% {:>8}",
            variant.name,
            variant.recipe.n_features(),
            one_own,
            known,
            unseen,
            time
        );
    }
    println!();
}

// ---------------------------------------------------------------------------
// Experiments 1 and 2: one speaker's own voice
// ---------------------------------------------------------------------------

/// Experiment 1: for each speaker, one template per digit (training take 5,
/// the first one that is not a test clip), tested on that speaker's 50 test
/// clips.
fn one_template_per_digit(corpus: &Corpus) {
    println!("1. One template per digit, same speaker (template: take 5; test: takes 0-4)\n");
    let start = Instant::now();
    let mut accuracies = Vec::new();
    for speaker in fsdd::speakers(&corpus.clips) {
        let mut split = fsdd::same_speaker_split(&corpus.clips, &speaker);
        split.train.retain(|&p| corpus.clips[p].index == 5);
        let predicted = corpus.recognize(&split, 1);
        let accuracy = corpus.percent_correct(&split.test, &predicted);
        println!("   {speaker:<10} {:>2} templates, {:>2} test clips   {accuracy:>5.1}%", split.train.len(), split.test.len());
        accuracies.push(accuracy);
    }
    println!("   {:<10} {:>37.1}%   ({})\n", "mean", mean(&accuracies), seconds_since(start));
}

/// What experiment 2 found for one number of templates per digit.
struct TemplateCountResult {
    per_digit: usize,
    /// Accuracy over all 300 test clips, averaged over the random draws.
    mean: f32,
    /// The unluckiest and the luckiest draw.
    worst: f32,
    best: f32,
    /// Accuracy per speaker, averaged over the draws (same order as
    /// [`fsdd::speakers`]).
    per_speaker: Vec<f32>,
}

/// Experiment 2: the same setting with 1, 2, 5, 10, 20 and all 45 templates
/// per digit.
///
/// *Which* takes serve as templates matters when there are few of them, so
/// for every count we draw the templates at random 20 times and report the
/// average, the worst and the best draw.
fn more_templates(corpus: &Corpus) -> Vec<TemplateCountResult> {
    const COUNTS: [usize; 6] = [1, 2, 5, 10, 20, 45];
    const DRAWS: usize = 20;
    println!("2. More templates per digit, same speaker ({DRAWS} random draws of the templates for each count)\n");
    let start = Instant::now();
    let speakers = fsdd::speakers(&corpus.clips);
    // For every speaker: the split, and all test-to-training distances.
    let splits: Vec<Split> = speakers.iter().map(|s| fsdd::same_speaker_split(&corpus.clips, s)).collect();
    let tables: Vec<Vec<Vec<f32>>> = splits.iter().map(|s| distance_table(&corpus.features, &s.test, &s.train, BAND)).collect();

    let mut rng = Rng::new(5);
    let mut results = Vec::new();
    println!("   {:>9} {:>10} {:>8} {:>8} {:>8}", "templates", "per digit", "mean", "worst", "best");
    for per_digit in COUNTS {
        let draws = if per_digit == 45 { 1 } else { DRAWS }; // all 45 can only be drawn one way
        let mut per_draw = Vec::new();
        let mut per_speaker = vec![0.0; speakers.len()];
        for _ in 0..draws {
            let mut correct_in_draw = 0;
            for (s, (split, table)) in splits.iter().zip(&tables).enumerate() {
                // Pick `per_digit` random training takes of every digit.
                let mut chosen = vec![false; split.train.len()];
                for digit in 0..N_DIGITS {
                    let mut takes: Vec<usize> = (0..split.train.len()).filter(|&t| corpus.clips[split.train[t]].digit == digit).collect();
                    rng.shuffle(&mut takes);
                    for &t in &takes[..per_digit] {
                        chosen[t] = true;
                    }
                }
                let correct = split.test.iter().zip(table).filter(|&(&q, row)| {
                    let nearest = split.train[nearest_allowed(row, |t| chosen[t])];
                    corpus.clips[nearest].digit == corpus.clips[q].digit
                });
                let correct = correct.count();
                correct_in_draw += correct;
                per_speaker[s] += 100.0 * correct as f32 / split.test.len() as f32 / draws as f32;
            }
            per_draw.push(100.0 * correct_in_draw as f32 / 300.0);
        }
        let worst = per_draw.iter().copied().fold(f32::INFINITY, f32::min);
        let best = per_draw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let result = TemplateCountResult { per_digit, mean: mean(&per_draw), worst, best, per_speaker };
        println!("   {:>9} {:>10} {:>7.1}% {:>7.1}% {:>7.1}%", 10 * per_digit, per_digit, result.mean, result.worst, result.best);
        results.push(result);
    }
    println!("   ({})\n", seconds_since(start));
    results
}

// ---------------------------------------------------------------------------
// Experiments 3 and 4: many speakers
// ---------------------------------------------------------------------------

/// Experiment 3: the official split. All 2,700 training clips are templates.
fn all_speakers(corpus: &Corpus) {
    let split = fsdd::official_split(&corpus.clips);
    println!("3. All six speakers, official split ({} templates, {} test clips)\n", split.train.len(), split.test.len());
    for k in [1, 3, 5] {
        let start = Instant::now();
        let predicted = corpus.recognize(&split, k);
        let wrong = split.test.iter().zip(&predicted).filter(|&(&p, &digit)| corpus.clips[p].digit != digit).count();
        println!(
            "   {k}-nearest   {:>5.1}%   ({wrong} of {} wrong, {})",
            corpus.percent_correct(&split.test, &predicted),
            split.test.len(),
            seconds_since(start)
        );
    }
    println!();
}

/// What experiment 4 found.
struct UnseenSpeakerResult {
    /// 1-nearest accuracy per held-out speaker, in the order of [`fsdd::speakers`].
    per_speaker: Vec<f32>,
    /// `confusion[(said, heard)]` counts test clips over all six rounds.
    confusion: Mat,
}

/// Experiment 4: leave one speaker out. Templates come from five speakers;
/// all 500 clips of the sixth are the test.
///
/// The `takes 0-4` column scores the same 1-nearest answers on the held-out
/// speaker's 50 official test clips only. Those are the clips experiment 3
/// was tested on, and no clip among them took part in choosing the features
/// (experiment 0 used takes 5 to 49), so it is the like-for-like comparison
/// with experiment 3: same clips, with and without the speaker's own
/// templates in the bank.
fn unseen_speaker(corpus: &Corpus) -> UnseenSpeakerResult {
    println!("4. A speaker never heard before (leave one speaker out: 2,500 templates from five speakers, 500 test clips)\n");
    println!("   {:<10} {:>10} {:>10} {:>10} {:>10}", "held out", "1-nearest", "5-nearest", "takes 0-4", "time");
    let mut confusion = Mat::zeros(N_DIGITS, N_DIGITS);
    let (mut nearest_1, mut nearest_5, mut test_takes) = (Vec::new(), Vec::new(), Vec::new());
    for (speaker, split) in fsdd::leave_one_speaker_out(&corpus.clips) {
        let start = Instant::now();
        let predicted = corpus.recognize(&split, 1);
        let predicted_5 = corpus.recognize(&split, 5);
        for (&p, &digit) in split.test.iter().zip(&predicted) {
            confusion[(corpus.clips[p].digit, digit)] += 1.0;
        }
        nearest_1.push(corpus.percent_correct(&split.test, &predicted));
        nearest_5.push(corpus.percent_correct(&split.test, &predicted_5));
        // The official test takes among the 500, with the answers they got.
        let (official, official_predicted): (Vec<usize>, Vec<usize>) =
            split.test.iter().zip(&predicted).filter(|&(&p, _)| corpus.clips[p].is_official_test()).map(|(&p, &digit)| (p, digit)).unzip();
        test_takes.push(corpus.percent_correct(&official, &official_predicted));
        println!(
            "   {speaker:<10} {:>9.1}% {:>9.1}% {:>9.1}% {:>10}",
            nearest_1[nearest_1.len() - 1],
            nearest_5[nearest_5.len() - 1],
            test_takes[test_takes.len() - 1],
            seconds_since(start)
        );
    }
    println!("   {:<10} {:>9.1}% {:>9.1}% {:>9.1}%\n", "mean", mean(&nearest_1), mean(&nearest_5), mean(&test_takes));

    println!("   Confusions, 1-nearest, all six rounds together (row: digit said; column: digit heard):\n");
    println!("        {}", (0..N_DIGITS).map(|d| format!("{d:>5}")).collect::<String>());
    for said in 0..N_DIGITS {
        let row: String = (0..N_DIGITS).map(|heard| format!("{:>5}", confusion[(said, heard)])).collect();
        println!("     {said:>2} {row}   {:>5.1}% right", 100.0 * confusion[(said, said)] / confusion.row(said).iter().sum::<f32>());
    }
    println!();
    UnseenSpeakerResult { per_speaker: nearest_1, confusion }
}

// ---------------------------------------------------------------------------
// Experiment 5: the cost of an answer
// ---------------------------------------------------------------------------

/// Experiment 5: how long one answer takes on one core, as the bank grows.
fn cost(corpus: &Corpus) {
    println!("5. What one answer costs (one clip against N templates, one core)\n");
    let split = fsdd::official_split(&corpus.clips);
    let mut lengths: Vec<usize> = corpus.features.iter().map(|f| f.rows).collect();
    lengths.sort_unstable();
    let median = lengths[lengths.len() / 2];
    println!(
        "   A clip is {median} frames long (median; shortest {}, longest {}). One DTW of two such clips fills\n   {median} x {median} = {} cells, and each cell compares two frames of {} numbers.\n",
        lengths[0],
        lengths[lengths.len() - 1],
        median * median,
        corpus.features[0].cols
    );
    println!("   {:>10} {:>16} {:>22}", "templates", "ms per clip", "ms per clip, band 5");
    // Spread the queries and the templates over all speakers and digits.
    let queries: Vec<usize> = split.test.iter().copied().step_by(6).collect();
    for n_templates in [10, 100, 450, 2700] {
        let templates: Vec<usize> = split.train.iter().copied().step_by(split.train.len() / n_templates).take(n_templates).collect();
        let milliseconds_per_clip = |band| {
            let bank = corpus.bank(&templates, band);
            let start = Instant::now();
            for &q in &queries {
                bank.classify(&corpus.features[q]);
            }
            1000.0 * start.elapsed().as_secs_f64() / queries.len() as f64
        };
        println!("   {n_templates:>10} {:>16.2} {:>22.2}", milliseconds_per_clip(None), milliseconds_per_clip(Some(5)));
    }
    println!("\n   The time grows in step with the number of templates: every recording we add to get more");
    println!("   accurate makes every answer slower, and the bank is the whole dataset kept in memory.\n");
}

// ---------------------------------------------------------------------------
// Figures
// ---------------------------------------------------------------------------

/// The clip with this file name.
fn clip_named<'a>(corpus: &'a Corpus, name: &str) -> (&'a Clip, &'a Mat) {
    let position = corpus.clips.iter().position(|c| c.file_name() == name).unwrap_or_else(|| panic!("{name} is not in the dataset"));
    (&corpus.clips[position], &corpus.features[position])
}

/// The takes of "seven" the figures use: the clip that ships in `samples/`
/// and a faster take by the same speaker.
const SEVEN: &str = "7_jackson_0.wav";
const OTHER_SEVEN: &str = "7_jackson_37.wav";
/// A different word by the same speaker, for contrast.
const ZERO: &str = "0_jackson_0.wav";

/// Two takes of "seven" by the same speaker: same word, different lengths.
/// What the endpoint detector trims away as silence is shaded. (Shading what
/// it keeps would paint a whole panel: these clips arrive tightly cut.)
fn draw_two_sevens(corpus: &Corpus, path: &str) -> io::Result<()> {
    let (first, _) = clip_named(corpus, SEVEN);
    let (second, _) = clip_named(corpus, OTHER_SEVEN);
    let longest = first.duration().max(second.duration());
    let waveform = |clip: &Clip, color: Color| {
        let times: Vec<f32> = (0..clip.samples.len()).map(|i| i as f32 / clip.sample_rate as f32).collect();
        let (start, end) = find_speech(&clip.samples, clip.sample_rate).unwrap_or((0, clip.samples.len()));
        let (start, end) = (start as f32 / clip.sample_rate as f32, end as f32 / clip.sample_rate as f32);
        let mut plot = LinePlot::new(&format!("{}: {:.2} s of speech", clip.file_name(), end - start));
        // Anything shorter than a frame hop is not worth pointing at.
        if start > 0.01 {
            plot = plot.span_colored(0.0, start, "trimmed", Color::Ink48);
        }
        if clip.duration() - end > 0.01 {
            plot = plot.span_colored(end, clip.duration(), "trimmed", Color::Ink48);
        }
        plot.series(Series::new("", &times, &clip.samples).color(color).width(1.0))
            .x_range(0.0, longest * 1.02)
            .y_range(-0.6, 0.6)
            .x_label("time (s)")
            .y_label("amplitude")
            .height(130.0)
    };
    Figure::new("The same speaker says \"seven\" twice: never the same length, never the same timing")
        .desc("Two waveforms of the word seven spoken by the same person, on a shared time axis. The first take is 0.43 seconds of speech from end to end. The second is spoken faster: its loud part is over sooner, and the endpoint detector trims its quiet tail (shaded), leaving 0.32 seconds.")
        .panel(waveform(first, Color::Green))
        .panel(waveform(second, Color::Amber))
        .share_x()
        .save(path)
}

/// The accumulated-cost grid of two recordings with the warping path on top.
fn draw_grid(title: &str, desc: &str, a: (&Clip, &Mat), b: (&Clip, &Mat), darkest: f32, path: &str) -> io::Result<DtwResult> {
    let (result, accumulated) = dtw_full(a.1, b.1, BAND);
    Figure::new(&format!("{title} (cost {:.0}, normalized {:.2})", result.cost, result.normalized_cost))
        .desc(desc)
        .panel(
            Heatmap::new("", &accumulated)
                .orientation(Orientation::MatrixFlipped)
                .value_range(0.0, darkest)
                .colorbar("accumulated cost")
                .path(&result.path, Color::Amber)
                .x_label(&format!("frame of {}", b.0.file_name()))
                .y_label(&format!("frame of {}", a.0.file_name()))
                .height(380.0),
        )
        .save(path)?;
    Ok(result)
}

/// Both grid figures, on one color scale so that they can be compared.
fn draw_grids(corpus: &Corpus) -> io::Result<()> {
    let seven = clip_named(corpus, SEVEN);
    let other_seven = clip_named(corpus, OTHER_SEVEN);
    let zero = clip_named(corpus, ZERO);
    // One color scale for both figures: the darkest color stands for a
    // little more than the larger of the two final costs. (The far corners
    // of the grids, where one whole recording is matched to a single frame
    // of the other, cost far more than that; they are simply drawn darkest.)
    let final_cost = |a: &Mat, b: &Mat| dtw_full(a, b, BAND).0.cost;
    let darkest = 1.25 * final_cost(seven.1, other_seven.1).max(final_cost(seven.1, zero.1));

    let same = draw_grid(
        "Two takes of \"seven\": a cheap path exists",
        "Heatmap of the accumulated DTW cost between two recordings of the word seven by the same speaker, 41 and 30 frames long, with the cheapest warping path drawn on top. The path bends to absorb the difference in length and reaches the top-right corner at a low cost.",
        seven,
        other_seven,
        darkest,
        &format!("{FIGURES}/ch05-dtw-grid-same.svg"),
    )?;
    let different = draw_grid(
        "\"seven\" against \"zero\": every path is expensive",
        "Heatmap of the accumulated DTW cost between a recording of seven and a recording of zero by the same speaker, on the same color scale as the previous figure. The grid darkens much faster: the best path still has to pay for frames that do not match, and ends at about three times the cost.",
        seven,
        zero,
        darkest,
        &format!("{FIGURES}/ch05-dtw-grid-different.svg"),
    )?;
    println!(
        "   seven vs seven: cost {:.1}, normalized {:.3}, path of {} steps\n   seven vs zero:  cost {:.1}, normalized {:.3}, path of {} steps",
        same.cost,
        same.normalized_cost,
        same.path.len(),
        different.cost,
        different.normalized_cost,
        different.path.len()
    );
    Ok(())
}

fn draw_accuracy_vs_templates(results: &[TemplateCountResult], path: &str) -> io::Result<()> {
    let counts: Vec<f32> = results.iter().map(|r| r.per_digit as f32).collect();
    let column = |f: fn(&TemplateCountResult) -> f32| results.iter().map(f).collect::<Vec<f32>>();
    let mut plot = LinePlot::new("Same speaker: more templates per digit, fewer mistakes")
        .x_label("templates per digit (of the speaker's own voice)")
        .y_label("accuracy on 300 test clips (%)")
        .series(Series::new("luckiest draw of templates", &counts, &column(|r| r.best)).color(Color::Ink30).dashed())
        .series(Series::new("unluckiest draw", &counts, &column(|r| r.worst)).color(Color::Ink30).dashed())
        .series(Series::new("average over 20 draws", &counts, &column(|r| r.mean)).color(Color::Green).width(2.0).dots())
        .x_range(0.0, 47.0)
        .y_range(80.0, 104.0)
        .height(260.0);
    // Each note sits beside its point, in the gap between the average and
    // the unluckiest draw, where it covers neither a curve nor a dot. (The
    // offsets are in axis units: templates to the right, percent down.)
    for result in results.iter().filter(|r| [1, 5, 45].contains(&r.per_digit)) {
        let (right, down) = match result.per_digit {
            1 => (2.6, 2.6),
            5 => (1.5, 1.9),
            _ => (-0.2, -0.3),
        };
        plot = plot.annotate(result.per_digit as f32 + right, result.mean - down, &format!("{:.1}%", result.mean));
    }
    plot.save(path)
}

fn draw_speakers(own_voice: &[f32], unseen: &[f32], speakers: &[String], path: &str) -> io::Result<()> {
    let names: Vec<&str> = speakers.iter().map(String::as_str).collect();
    let chart = |title: &str, values: &[f32], color: Color| {
        BarChart::new(title, &names, values).color(color).y_range(0.0, 115.0).y_label("accuracy (%)").show_values(1).height(150.0)
    };
    Figure::new("A voice it knows, and a voice it has never heard")
        .desc("Two bar charts of DTW digit recognition accuracy for each of six speakers. With 45 templates per digit from the speaker's own voice every speaker is recognized almost perfectly. With templates from the other five speakers only, accuracy drops for everyone and falls to nearly half for two of the speakers.")
        .panel(chart("templates from the speaker's own voice (45 per digit, test takes 0-4)", own_voice, Color::Green))
        .panel(chart("templates from the other five speakers only (all 500 clips of the speaker)", unseen, Color::Amber))
        .save(path)
}

fn draw_confusion(confusion: &Mat, path: &str) -> io::Result<()> {
    let labels: Vec<String> = (0..N_DIGITS).map(|d| d.to_string()).collect();
    let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    Figure::new("Unseen speakers: which digit was said, and which digit DTW heard")
        .desc("A ten by ten confusion matrix of DTW digit recognition on speakers that are not among the templates, with the number of clips in every cell. Most of the 300 clips per digit lie on the diagonal; the off-diagonal cells show which digits are mistaken for which.")
        .panel(
            Heatmap::new("", confusion)
                .row_labels(&labels)
                .col_labels(&labels)
                .cell_text(0)
                .value_range(0.0, 300.0)
                .x_label("digit heard (nearest template)")
                .y_label("digit said")
                .height(400.0),
        )
        .save(path)
}

// ---------------------------------------------------------------------------
// classify: one file against the whole dataset
// ---------------------------------------------------------------------------

/// Where the clips that ship with the book are, relative to `code/`.
const SAMPLES: &str = "../samples";

/// The six Free Spoken Digit Dataset clips in `samples/`: file name, digit,
/// speaker. Each is take 0 of that digit by that speaker (see
/// `samples/README.md`).
const SAMPLE_TEMPLATES: [(&str, usize, &str); 6] = [
    ("zero_jackson.wav", 0, "jackson"),
    ("zero_george.wav", 0, "george"),
    ("three_jackson.wav", 3, "jackson"),
    ("three_george.wav", 3, "george"),
    ("seven_jackson.wav", 7, "jackson"),
    ("seven_george.wav", 7, "george"),
];

/// The clips of [`SAMPLE_TEMPLATES`], labeled as they are in the dataset.
fn sample_templates() -> io::Result<Vec<Clip>> {
    let mut clips = Vec::new();
    for (file, digit, speaker) in SAMPLE_TEMPLATES {
        let wav = Wav::read(format!("{SAMPLES}/{file}"))?;
        clips.push(Clip { digit, speaker: speaker.to_string(), index: 0, sample_rate: wav.sample_rate, samples: wav.samples });
    }
    Ok(clips)
}

/// The templates for `classify`: the whole dataset if it has been
/// downloaded, the six clips in `samples/` if not.
fn templates_for_classify() -> io::Result<Vec<Clip>> {
    let dataset = fsdd::recordings_dir();
    if dataset.is_dir() {
        return fsdd::load();
    }
    println!(
        "no dataset in {}: using the {} digit clips in {SAMPLES}/ as templates (zero, three and seven only)",
        dataset.display(),
        SAMPLE_TEMPLATES.len()
    );
    sample_templates()
}

/// Recognize the recording in `path` with every clip of the dataset as a
/// template (or, without the dataset, the six clips in `samples/`), and show
/// the work: the nearest templates, and how close the best template of each
/// digit came.
fn classify(path: &str) -> io::Result<()> {
    let wav = Wav::read(path)?;
    println!("{path}: {:.2} s at {} Hz", wav.samples.len() as f32 / wav.sample_rate as f32, wav.sample_rate);
    // The templates are 8 kHz recordings, so the query has to be too.
    let samples = if wav.sample_rate == SAMPLE_RATE {
        wav.samples
    } else {
        println!("resampling from {} Hz to {SAMPLE_RATE} Hz to match the templates", wav.sample_rate);
        resample(&wav.samples, wav.sample_rate, SAMPLE_RATE)
    };

    let recipe = MfccRecipe::standard();
    let start = Instant::now();
    let corpus = Corpus::new(templates_for_classify()?, &recipe);
    let query = MfccFrontEnd::new(&recipe, SAMPLE_RATE).features(&samples);
    println!("{} frames of {} MFCCs after trimming silence", query.rows, query.cols);
    println!("loaded {} templates and computed their features in {}", corpus.clips.len(), seconds_since(start));

    // A recording must not be its own template: finding yourself proves nothing.
    let templates = corpus.positions(|clip| clip.samples != samples);
    for clip in corpus.clips.iter().filter(|clip| clip.samples == samples) {
        println!("this recording is in the dataset as {}; leaving that one out of the templates", clip.file_name());
    }

    let start = Instant::now();
    let distances: Vec<f32> = fsdd::map_clips(&templates, |&t| dtw_distance(&query, &corpus.features[t], BAND));
    let elapsed = seconds_since(start);
    let mut ranked: Vec<(f32, &Clip)> = distances.iter().copied().zip(templates.iter().map(|&t| &corpus.clips[t])).collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));

    println!("\ncompared with {} templates in {elapsed} (all cores)\n", templates.len());
    if ranked.len() >= 10 {
        println!("The ten nearest templates:\n");
    } else {
        println!("All {} templates, nearest first:\n", ranked.len());
    }
    println!("   {:>4}  {:<22} {:<6} {:>9}", "rank", "template", "digit", "distance");
    for (rank, (distance, clip)) in ranked.iter().take(10).enumerate() {
        println!("   {:>4}  {:<22} {:<6} {distance:>9.3}", rank + 1, clip.file_name(), DIGIT_NAMES[clip.digit]);
    }

    println!("\nThe nearest template of each digit:\n");
    // `ranked` is sorted, so the first clip of each digit is that digit's nearest.
    let mut best_per_digit: Vec<&(f32, &Clip)> = Vec::new();
    for entry in &ranked {
        if best_per_digit.iter().all(|seen| seen.1.digit != entry.1.digit) {
            best_per_digit.push(entry);
        }
    }
    let furthest = best_per_digit[best_per_digit.len() - 1].0;
    for (distance, clip) in best_per_digit {
        let bar = "#".repeat((40.0 * distance / furthest).round() as usize);
        println!("   {:<6} {distance:>7.3}  {bar:<40}  {}", DIGIT_NAMES[clip.digit], clip.file_name());
    }

    let (distance, nearest) = ranked[0];
    println!("\nIt says \"{}\" (digit {}, distance {distance:.3} to {}).", DIGIT_NAMES[nearest.digit], nearest.digit, nearest.file_name());
    Ok(())
}

// ---------------------------------------------------------------------------

fn run_experiments() -> io::Result<()> {
    let start = Instant::now();
    let clips = fsdd::load()?;
    println!("Free Spoken Digit Dataset: {} clips, speakers {:?} (loaded in {})\n", clips.len(), fsdd::speakers(&clips), seconds_since(start));

    ablation(&clips);

    let recipe = MfccRecipe::standard();
    let start = Instant::now();
    let corpus = Corpus::new(clips, &recipe);
    println!("Features for the experiments: {} numbers per frame, computed for all clips in {}\n", recipe.n_features(), seconds_since(start));

    one_template_per_digit(&corpus);
    let by_template_count = more_templates(&corpus);
    all_speakers(&corpus);
    let unseen = unseen_speaker(&corpus);
    cost(&corpus);

    println!("Figures:\n");
    draw_two_sevens(&corpus, &format!("{FIGURES}/ch05-two-sevens.svg"))?;
    draw_grids(&corpus)?;
    draw_accuracy_vs_templates(&by_template_count, &format!("{FIGURES}/ch05-accuracy-vs-templates.svg"))?;
    let own_voice = &by_template_count[by_template_count.len() - 1].per_speaker; // all 45 templates per digit
    draw_speakers(own_voice, &unseen.per_speaker, &fsdd::speakers(&corpus.clips), &format!("{FIGURES}/ch05-speakers.svg"))?;
    draw_confusion(&unseen.confusion, &format!("{FIGURES}/ch05-confusion.svg"))?;
    for name in ["two-sevens", "dtw-grid-same", "dtw-grid-different", "accuracy-vs-templates", "speakers", "confusion"] {
        println!("   wrote {FIGURES}/ch05-{name}.svg");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let outcome = match args.as_slice() {
        [] => run_experiments(),
        ["classify", path] => classify(path),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
