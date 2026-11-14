//! The Free Spoken Digit Dataset (FSDD): "the MNIST of speech".
//!
//! 3,000 short recordings of the digits zero to nine, spoken by six people,
//! fifty times each, at 8 kHz. Every file name says what is in it:
//!
//! ```text
//! $SPEECH_DATA/fsdd/recordings/7_jackson_32.wav
//!                              │ │       └─ which take: 0..=49
//!                              │ └───────── who said it
//!                              └─────────── the digit that was said
//! ```
//!
//! This module does three jobs:
//!
//! * [`load`] reads every recording into memory as a [`Clip`].
//! * [`official_split`] and [`leave_speaker_out`] decide which clips we may
//!   learn from and which we test on. They return *positions* in the list of
//!   clips, so the same split can index the clips, their features, or
//!   anything else computed once per clip.
//! * [`MfccFrontEnd`] turns a clip into the feature picture chapters 5 and 6
//!   work with, and [`map_clips`] does that (or anything else) for all 3,000
//!   clips at once on every CPU core.
//!
//! # Two ways of holding clips back
//!
//! A recognizer has to be judged on recordings it has not seen, or the number
//! means nothing. There are two honest ways to do that here, and they answer
//! different questions:
//!
//! * **The official split.** The dataset's authors reserve takes 0 to 4 of
//!   every digit and speaker for testing (300 clips) and leave takes 5 to 49
//!   for training (2,700 clips). Every test clip's speaker is also in the
//!   training set. Question answered: *how well do we recognize people we
//!   have heard before?*
//! * **Leave one speaker out.** Train on five speakers, test on all 500
//!   clips of the sixth. Question answered: *how well do we recognize a
//!   stranger?* That is the question a real product faces, and it is much
//!   harder.

use crate::data::data_dir;
use crate::dsp::features::{deltas, normalize, FeatureConfig, FeatureExtractor};
use crate::dsp::frame::trim_silence;
use crate::mat::Mat;
use crate::wav::Wav;
use std::io;
use std::path::{Path, PathBuf};

/// The six speakers, in alphabetical order.
pub const SPEAKERS: [&str; 6] = ["george", "jackson", "lucas", "nicolas", "theo", "yweweler"];

/// Number of classes: the digits 0 to 9.
pub const N_DIGITS: usize = 10;

/// The sample rate of every recording in the dataset.
pub const SAMPLE_RATE: u32 = 8000;

/// Takes with an index below this are the official test set (0, 1, 2, 3, 4).
pub const TEST_TAKES: usize = 5;

/// The digits as words, for printing: `DIGIT_NAMES[7] == "seven"`.
pub const DIGIT_NAMES: [&str; N_DIGITS] = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];

/// One recording and what we know about it.
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    /// The digit that was said, 0 to 9. This is the label we try to predict.
    pub digit: usize,
    /// Who said it, e.g. `"jackson"`.
    pub speaker: String,
    /// Which take of this digit by this speaker, 0 to 49.
    pub index: usize,
    /// Samples per second (8000 for the whole dataset).
    pub sample_rate: u32,
    /// The audio, mono, in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
}

impl Clip {
    /// The clip's file name in the dataset, e.g. `7_jackson_32.wav`.
    pub fn file_name(&self) -> String {
        format!("{}_{}_{}.wav", self.digit, self.speaker, self.index)
    }

    /// True if the clip belongs to the dataset's official test set (takes 0
    /// to 4 of every digit and speaker).
    pub fn is_official_test(&self) -> bool {
        self.index < TEST_TAKES
    }

    /// Length of the recording in seconds.
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

/// Take a file name like `7_jackson_32.wav` apart into
/// `(digit, speaker, index)`. Returns `None` for anything that does not
/// follow the dataset's naming scheme (a stray `README`, say).
pub fn parse_file_name(name: &str) -> Option<(usize, String, usize)> {
    let stem = name.strip_suffix(".wav")?;
    let mut parts = stem.split('_');
    let (digit, speaker, index) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() || speaker.is_empty() {
        return None;
    }
    let digit: usize = digit.parse().ok()?;
    if digit >= N_DIGITS {
        return None;
    }
    Some((digit, speaker.to_string(), index.parse().ok()?))
}

/// Where the recordings are: `$SPEECH_DATA/fsdd/recordings`.
pub fn recordings_dir() -> PathBuf {
    data_dir().join("fsdd").join("recordings")
}

/// Read the whole dataset from [`recordings_dir`].
pub fn load() -> io::Result<Vec<Clip>> {
    load_from(&recordings_dir())
}

/// Read every `{digit}_{speaker}_{index}.wav` in `dir`.
///
/// The clips come back sorted by speaker, then digit, then take, whatever
/// order the file system lists them in, so that a run is reproducible: the
/// same seed picks the same clips on every machine.
pub fn load_from(dir: &Path) -> io::Result<Vec<Clip>> {
    let listing = std::fs::read_dir(dir).map_err(|e| {
        let hint = "The Free Spoken Digit Dataset goes in ../data/fsdd, next to the code/ directory. From code/, run:\n    \
                    git clone https://github.com/Jakobovski/free-spoken-digit-dataset ../data/fsdd\n\
                    The clips are then in ../data/fsdd/recordings/. To keep the data somewhere else, set the\n\
                    SPEECH_DATA environment variable to that directory and clone into $SPEECH_DATA/fsdd instead.";
        io::Error::new(e.kind(), format!("cannot open the digit recordings in {}: {e}\n{hint}", dir.display()))
    })?;
    let mut clips = Vec::new();
    for entry in listing {
        let path = entry?.path();
        let Some((digit, speaker, index)) = path.file_name().and_then(|n| n.to_str()).and_then(parse_file_name) else {
            continue; // not a recording
        };
        let wav = Wav::read(&path)?;
        clips.push(Clip { digit, speaker, index, sample_rate: wav.sample_rate, samples: wav.samples });
    }
    if clips.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound, format!("no recordings named like 7_jackson_32.wav in {}", dir.display())));
    }
    clips.sort_by(|a, b| (&a.speaker, a.digit, a.index).cmp(&(&b.speaker, b.digit, b.index)));
    Ok(clips)
}

/// The speakers that occur in `clips`, sorted, each once.
pub fn speakers(clips: &[Clip]) -> Vec<String> {
    let mut names: Vec<String> = clips.iter().map(|c| c.speaker.clone()).collect();
    names.sort();
    names.dedup();
    names
}

/// Which clips to learn from and which to test on, as positions in the list
/// of clips the split was made from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Split {
    /// Positions of the clips we may use as templates or training examples.
    pub train: Vec<usize>,
    /// Positions of the clips we are judged on. Never looked at before the
    /// final measurement.
    pub test: Vec<usize>,
}

/// Sort every clip into `train` or `test` according to `is_test`.
fn split_by(clips: &[Clip], is_test: impl Fn(&Clip) -> bool) -> Split {
    let mut split = Split { train: Vec::new(), test: Vec::new() };
    for (position, clip) in clips.iter().enumerate() {
        if is_test(clip) {
            split.test.push(position);
        } else {
            split.train.push(position);
        }
    }
    split
}

/// The dataset's own split: takes 0 to 4 are the test set, takes 5 to 49 the
/// training set. Every speaker appears on both sides.
pub fn official_split(clips: &[Clip]) -> Split {
    split_by(clips, Clip::is_official_test)
}

/// Hold one speaker out: all of that speaker's clips are the test set, and
/// everybody else's are the training set. The recognizer has never heard the
/// test speaker's voice.
pub fn leave_speaker_out(clips: &[Clip], speaker: &str) -> Split {
    split_by(clips, |clip| clip.speaker == speaker)
}

/// One [`leave_speaker_out`] split per speaker, with the name of the speaker
/// held out. Averaging a result over these six is "leave-one-speaker-out
/// cross-validation": every clip gets to be a test clip exactly once.
pub fn leave_one_speaker_out(clips: &[Clip]) -> Vec<(String, Split)> {
    speakers(clips).into_iter().map(|speaker| { let split = leave_speaker_out(clips, &speaker); (speaker, split) }).collect()
}

/// The official split of a single speaker's clips: train on that speaker's
/// takes 5 to 49, test on the same speaker's takes 0 to 4. This is the
/// "train it on your own voice" setting of 1980s voice dialing.
pub fn same_speaker_split(clips: &[Clip], speaker: &str) -> Split {
    let mut split = split_by(clips, Clip::is_official_test);
    split.train.retain(|&p| clips[p].speaker == speaker);
    split.test.retain(|&p| clips[p].speaker == speaker);
    split
}

/// Apply `f` to every item on all CPU cores and collect the results in
/// order: `out[i] == f(&items[i])`.
///
/// Computing features for 3,000 clips takes a few seconds on one core, and
/// every experiment needs the same features. So we compute them **once**,
/// with this function, and afterwards only pass positions around (see
/// [`Split`]). Each clip is independent of the others, which makes this the
/// easy kind of parallelism: cut the list into one chunk per thread and let
/// each thread fill in its own chunk of the output.
pub fn map_clips<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk_len = items.len().div_ceil(threads).max(1);
    let f = &f;
    let mut chunks: Vec<Vec<R>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = items.chunks(chunk_len).map(|chunk| scope.spawn(move || chunk.iter().map(f).collect::<Vec<R>>())).collect();
        for handle in handles {
            chunks.push(handle.join().expect("a worker thread panicked"));
        }
    });
    chunks.into_iter().flatten().collect()
}

/// How (and whether) to normalize each feature column over an utterance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalization {
    /// Leave the features as they are.
    None,
    /// Subtract each column's mean over the utterance ("cepstral mean
    /// normalization"). Removes what stays constant through the clip: the
    /// microphone, the room, and part of the speaker's voice.
    Mean,
    /// Subtract the mean and divide by the standard deviation
    /// ([`normalize`]). Also puts every column on the same scale, so that no
    /// single coefficient dominates the distance between two frames.
    MeanVariance,
}

/// The recipe that turns a recording into MFCC features. Every field is one
/// decision chapter 5 examines.
#[derive(Clone, Debug, PartialEq)]
pub struct MfccRecipe {
    /// Number of mel bands under the DCT.
    pub n_mels: usize,
    /// Number of cepstral coefficients computed, counting coefficient 0.
    pub n_coeffs: usize,
    /// Keep coefficient 0, the overall level of the frame's log-mel row? It
    /// follows the recording level, which says more about the distance to
    /// the microphone than about the word.
    pub keep_c0: bool,
    /// Append the delta of every coefficient (how fast it is changing),
    /// doubling the number of columns.
    pub deltas: bool,
    /// Per-utterance normalization of every column.
    pub normalization: Normalization,
    /// Cut leading and trailing silence off first
    /// ([`trim_silence`](crate::dsp::frame::trim_silence)).
    pub trim: bool,
}

impl MfccRecipe {
    /// The recipe chapter 5 settles on (see the ablation printed by
    /// `ch05_digits`), which turns out to be the plainest one: 13 MFCCs
    /// from 40 mel bands with coefficient 0 dropped, silence trimmed, no
    /// deltas and no normalization. 12 numbers per frame.
    pub fn standard() -> Self {
        MfccRecipe { n_mels: 40, n_coeffs: 13, keep_c0: false, deltas: false, normalization: Normalization::None, trim: true }
    }

    /// How many numbers describe one frame.
    pub fn n_features(&self) -> usize {
        let cepstra = if self.keep_c0 { self.n_coeffs } else { self.n_coeffs - 1 };
        if self.deltas {
            2 * cepstra
        } else {
            cepstra
        }
    }
}

/// An [`MfccRecipe`] with its tables precomputed for one sample rate. Build
/// it once; it is never modified, so all threads can share it.
#[derive(Clone, Debug)]
pub struct MfccFrontEnd {
    recipe: MfccRecipe,
    extractor: FeatureExtractor,
}

impl MfccFrontEnd {
    /// Prepare `recipe` for audio at `sample_rate` (25 ms frames every 10 ms).
    pub fn new(recipe: &MfccRecipe, sample_rate: u32) -> Self {
        assert!(recipe.n_coeffs >= 2 && recipe.n_coeffs <= recipe.n_mels, "need between 2 and n_mels cepstral coefficients");
        let config = FeatureConfig { n_mels: recipe.n_mels, ..FeatureConfig::new(sample_rate) };
        MfccFrontEnd { recipe: recipe.clone(), extractor: FeatureExtractor::new(&config) }
    }

    /// The recipe this front end follows.
    pub fn recipe(&self) -> &MfccRecipe {
        &self.recipe
    }

    /// The underlying feature extractor (for its frame and hop lengths).
    pub fn extractor(&self) -> &FeatureExtractor {
        &self.extractor
    }

    /// The part of `samples` the features are computed from: the stretch of
    /// speech if the recipe trims silence, everything otherwise.
    ///
    /// If trimming would leave less than one frame (the detector found only
    /// a click), the untrimmed recording is used instead.
    pub fn speech<'a>(&self, samples: &'a [f32]) -> &'a [f32] {
        if !self.recipe.trim {
            return samples;
        }
        let trimmed = trim_silence(samples, self.extractor.config().sample_rate);
        if trimmed.len() >= self.extractor.config().frame_len() {
            trimmed
        } else {
            samples
        }
    }

    /// The features of one recording: one row per frame,
    /// [`MfccRecipe::n_features`] columns. Always at least one row: a
    /// recording shorter than a single frame is padded with silence.
    pub fn features(&self, samples: &[f32]) -> Mat {
        let frame_len = self.extractor.config().frame_len();
        let speech = self.speech(samples);
        let padded;
        let speech = if speech.len() < frame_len {
            padded = [speech, &vec![0.0; frame_len - speech.len()]].concat();
            &padded[..]
        } else {
            speech
        };

        let cepstra = self.extractor.mfcc(speech, self.recipe.n_coeffs);
        let first = if self.recipe.keep_c0 { 0 } else { 1 };
        let kept = Mat::from_fn(cepstra.rows, cepstra.cols - first, |t, c| cepstra[(t, first + c)]);
        let mut features = if self.recipe.deltas { append_columns(&kept, &deltas(&kept, 2)) } else { kept };
        match self.recipe.normalization {
            Normalization::None => {}
            Normalization::Mean => subtract_column_means(&mut features),
            Normalization::MeanVariance => normalize(&mut features),
        }
        features
    }
}

/// Put two matrices with the same number of rows side by side.
pub fn append_columns(left: &Mat, right: &Mat) -> Mat {
    assert_eq!(left.rows, right.rows, "append_columns: {} rows and {} rows", left.rows, right.rows);
    Mat::from_fn(left.rows, left.cols + right.cols, |r, c| if c < left.cols { left[(r, c)] } else { right[(r, c - left.cols)] })
}

/// Subtract from every column its mean over all rows.
pub fn subtract_column_means(features: &mut Mat) {
    if features.rows == 0 {
        return;
    }
    for c in 0..features.cols {
        let mean = (0..features.rows).map(|r| features[(r, c)]).sum::<f32>() / features.rows as f32;
        for r in 0..features.rows {
            features[(r, c)] -= mean;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::features::mfcc;

    /// A clip with no audio, for testing the bookkeeping.
    fn clip(digit: usize, speaker: &str, index: usize) -> Clip {
        Clip { digit, speaker: speaker.to_string(), index, sample_rate: SAMPLE_RATE, samples: Vec::new() }
    }

    /// 2 speakers x 10 digits x 8 takes = 160 clips.
    fn small_dataset() -> Vec<Clip> {
        let mut clips = Vec::new();
        for speaker in ["ann", "bob"] {
            for digit in 0..N_DIGITS {
                for index in 0..8 {
                    clips.push(clip(digit, speaker, index));
                }
            }
        }
        clips
    }

    #[test]
    fn file_names_are_parsed_and_rebuilt() {
        assert_eq!(parse_file_name("7_jackson_32.wav"), Some((7, "jackson".to_string(), 32)));
        assert_eq!(clip(7, "jackson", 32).file_name(), "7_jackson_32.wav");
        for bad in ["README.md", "7_jackson.wav", "x_jackson_1.wav", "12_jackson_1.wav", "7_jackson_one.wav", "7_jackson_1_2.wav", "7__1.wav"] {
            assert_eq!(parse_file_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn official_split_puts_takes_0_to_4_in_the_test_set() {
        let clips = small_dataset();
        let split = official_split(&clips);
        // 5 of every 8 takes are test: 2 * 10 * 5 = 100, the other 60 train.
        assert_eq!((split.train.len(), split.test.len()), (60, 100));
        assert!(split.test.iter().all(|&p| clips[p].index < 5));
        assert!(split.train.iter().all(|&p| clips[p].index >= 5));
    }

    #[test]
    fn leaving_a_speaker_out_separates_the_voices() {
        let clips = small_dataset();
        let split = leave_speaker_out(&clips, "bob");
        assert_eq!((split.train.len(), split.test.len()), (80, 80));
        assert!(split.test.iter().all(|&p| clips[p].speaker == "bob"));
        assert!(split.train.iter().all(|&p| clips[p].speaker == "ann"));

        let all = leave_one_speaker_out(&clips);
        assert_eq!(all.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>(), ["ann", "bob"]);
        // Every clip is a test clip in exactly one of the splits.
        let mut times_tested = vec![0; clips.len()];
        for (_, split) in &all {
            for &p in &split.test {
                times_tested[p] += 1;
            }
        }
        assert!(times_tested.iter().all(|&n| n == 1));
    }

    #[test]
    fn same_speaker_split_never_mixes_speakers() {
        let clips = small_dataset();
        let split = same_speaker_split(&clips, "ann");
        assert_eq!((split.train.len(), split.test.len()), (30, 50));
        assert!(split.train.iter().chain(&split.test).all(|&p| clips[p].speaker == "ann"));
    }

    #[test]
    fn map_clips_keeps_the_order() {
        let numbers: Vec<usize> = (0..1000).collect();
        assert_eq!(map_clips(&numbers, |n| n * n), numbers.iter().map(|n| n * n).collect::<Vec<_>>());
        assert_eq!(map_clips(&[] as &[usize], |n| *n), Vec::<usize>::new());
        assert_eq!(map_clips(&[3usize], |n| n + 1), vec![4]);
    }

    #[test]
    fn loads_a_directory_sorted_and_skips_strangers() {
        let dir = std::env::temp_dir().join(format!("speech-fsdd-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Written in an order that is not the sorted one.
        for (name, level) in [("3_zoe_1.wav", 0.5), ("9_adam_0.wav", -0.25), ("3_zoe_0.wav", 0.25)] {
            Wav::new(SAMPLE_RATE, vec![level; 4]).write(dir.join(name)).unwrap();
        }
        std::fs::write(dir.join("README.md"), "not audio").unwrap();

        let clips = load_from(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(clips.iter().map(Clip::file_name).collect::<Vec<_>>(), ["9_adam_0.wav", "3_zoe_0.wav", "3_zoe_1.wav"]);
        assert_eq!(clips[0].sample_rate, SAMPLE_RATE);
        assert_eq!(clips[0].samples, vec![-0.25; 4]);
        assert_eq!(clips[0].duration(), 4.0 / 8000.0);
        assert_eq!(speakers(&clips), ["adam", "zoe"]);
        assert!(load_from(&dir).is_err(), "the directory is gone, so loading must fail");
    }

    /// A second of something speech-like enough to have a changing spectrum.
    fn chirp() -> Vec<f32> {
        (0..8000).map(|t| { let s = t as f32 / 8000.0; 0.3 * (std::f32::consts::TAU * (300.0 + 900.0 * s) * s).sin() }).collect()
    }

    #[test]
    fn plain_recipe_is_the_mfcc_without_its_first_column() {
        let recipe = MfccRecipe { n_mels: 40, n_coeffs: 13, keep_c0: false, deltas: false, normalization: Normalization::None, trim: false };
        let samples = chirp();
        let features = MfccFrontEnd::new(&recipe, SAMPLE_RATE).features(&samples);
        let reference = mfcc(&samples, &FeatureConfig::new(SAMPLE_RATE), 13);
        assert_eq!((features.rows, features.cols), (reference.rows, 12));
        assert_eq!(recipe.n_features(), 12);
        for t in [0, 40, reference.rows - 1] {
            assert_eq!(features.row(t), &reference.row(t)[1..]);
        }
    }

    #[test]
    fn standard_recipe_has_12_columns() {
        let recipe = MfccRecipe::standard();
        assert_eq!(recipe.n_features(), 12);
        assert_eq!(MfccFrontEnd::new(&recipe, SAMPLE_RATE).features(&chirp()).cols, 12);
    }

    #[test]
    fn deltas_and_normalization_give_24_standardized_columns() {
        let recipe = MfccRecipe { deltas: true, normalization: Normalization::MeanVariance, ..MfccRecipe::standard() };
        let features = MfccFrontEnd::new(&recipe, SAMPLE_RATE).features(&chirp());
        assert_eq!(features.cols, 24);
        assert_eq!(recipe.n_features(), 24);
        for c in 0..features.cols {
            let column: Vec<f32> = (0..features.rows).map(|r| features[(r, c)]).collect();
            let mean = column.iter().sum::<f32>() / column.len() as f32;
            let variance = column.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / column.len() as f32;
            assert!(mean.abs() < 1e-3 && (variance - 1.0).abs() < 1e-2, "column {c}: mean {mean}, variance {variance}");
        }
    }

    #[test]
    fn trimming_removes_the_silent_frames() {
        // Half a second of silence, a second of sound, half a second of silence.
        let mut samples = vec![0.0; 4000];
        samples.extend(chirp());
        samples.extend(vec![0.0; 4000]);
        let untrimmed = MfccRecipe { trim: false, ..MfccRecipe::standard() };
        let with = MfccFrontEnd::new(&MfccRecipe::standard(), SAMPLE_RATE).features(&samples);
        let without = MfccFrontEnd::new(&untrimmed, SAMPLE_RATE).features(&samples);
        assert_eq!(without.rows, 198); // 2 s: (16000 - 200) / 80 + 1
        // About 1 s of speech survives (plus the detector's padding).
        assert!((95..=115).contains(&with.rows), "{} frames after trimming", with.rows);
    }

    #[test]
    fn a_clip_shorter_than_one_frame_still_gives_one_row() {
        let features = MfccFrontEnd::new(&MfccRecipe::standard(), SAMPLE_RATE).features(&[0.1; 50]);
        assert_eq!((features.rows, features.cols), (1, 12));
        assert!(features.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn column_helpers_by_hand() {
        let left = Mat::from_vec(2, 1, vec![1.0, 2.0]);
        let right = Mat::from_vec(2, 2, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(append_columns(&left, &right).data, vec![1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
        let mut m = Mat::from_vec(2, 2, vec![1.0, 10.0, 3.0, 30.0]);
        subtract_column_means(&mut m);
        assert_eq!(m.data, vec![-1.0, -10.0, 1.0, 10.0]);
    }
}
