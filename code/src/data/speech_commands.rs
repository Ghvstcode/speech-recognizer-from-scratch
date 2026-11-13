//! Loading Google Speech Commands v2: 105,829 one-second clips of 35 words.
//!
//! The dataset is a directory per word, each full of files named
//! `{speaker}_nohash_{take}.wav`: `seven/0a7c2a8d_nohash_1.wav` is the second
//! time the speaker with the anonymous id `0a7c2a8d` said "seven". Two text
//! files, `validation_list.txt` and `testing_list.txt`, name the clips that
//! are set aside; everything they do not mention is training data.
//!
//! # Why the split matters
//!
//! The lists were made by hashing the *speaker id*, so each speaker lands in
//! exactly one of the three parts. A model that is tested on the test part
//! has therefore never heard those voices, which is the only honest way to
//! measure a recognizer: it will be used by people who were not in its
//! training data. We keep the official split so that our numbers can be
//! compared with published ones.
//!
//! # Memory
//!
//! Samples are kept as the 16-bit integers they are stored as on disk, two
//! bytes each instead of the four an `f32` takes: the whole dataset is then
//! about 3.4 GB in memory, and converting a clip to `f32` when it is used
//! costs next to nothing.

use crate::dsp::features::FeatureExtractor;
use crate::mat::Mat;
use crate::wav::Wav;
use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};

/// The 35 words, in alphabetical order. A word's position in this list is
/// its class id everywhere in the book.
pub const WORDS: [&str; 35] = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", "learn", "left", "marvin",
    "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
];

/// Every clip in the dataset is sampled at 16 kHz.
pub const SAMPLE_RATE: u32 = 16_000;

/// The length every clip is padded (or cut) to when a fixed size is needed:
/// one second.
pub const CLIP_SAMPLES: usize = 16_000;

/// The directory that holds the book's datasets: the `SPEECH_DATA`
/// environment variable, or `../data` (relative to `code/`) if it is not set.
pub fn data_dir() -> PathBuf {
    std::env::var_os("SPEECH_DATA").map_or_else(|| PathBuf::from("../data"), PathBuf::from)
}

/// Where Speech Commands lives: `$SPEECH_DATA/speech_commands`.
pub fn dataset_dir() -> PathBuf {
    data_dir().join("speech_commands")
}

/// The class id of a word, if it is one of the 35.
pub fn word_index(word: &str) -> Option<usize> {
    WORDS.iter().position(|&w| w == word)
}

/// Which of the three official parts a clip belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Split {
    /// What the model learns from (about 80% of the clips).
    Train,
    /// What we look at *during* training to pick settings and decide when to
    /// stop (about 10%).
    Validation,
    /// What we look at once, at the end, to report a number (about 10%).
    Test,
}

/// One recording of one word.
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    /// Class id: an index into [`WORDS`].
    pub word: usize,
    /// Index into [`SpeechCommands::speakers`].
    pub speaker: u32,
    /// The number after `nohash_` in the file name: which take of this word
    /// by this speaker.
    pub take: u32,
    /// Which part of the official split the clip belongs to.
    pub split: Split,
    /// The audio, 16 kHz mono, as stored on disk. At most one second.
    pub samples: Vec<i16>,
}

impl Clip {
    /// The audio as `f32` in `[-1, 1)`, at its original length.
    pub fn audio(&self) -> Vec<f32> {
        to_f32(&self.samples)
    }

    /// The audio as `f32`, padded with zeros at the end (or cut) to exactly
    /// one second, for models that need a fixed-size input.
    pub fn audio_one_second(&self) -> Vec<f32> {
        let mut audio = self.audio();
        audio.resize(CLIP_SAMPLES, 0.0);
        audio
    }
}

/// 16-bit samples to `f32` in `[-1, 1)`.
pub fn to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| s as f32 / 32768.0).collect()
}

/// `f32` samples to 16-bit, rounding and clamping.
pub fn to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|&s| (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect()
}

/// Split a file name like `0a7c2a8d_nohash_1.wav` into the speaker id
/// (`"0a7c2a8d"`) and the take number (`1`). `None` if the name does not
/// have that shape.
pub fn parse_file_name(file_name: &str) -> Option<(&str, u32)> {
    let stem = file_name.strip_suffix(".wav")?;
    let (speaker, take) = stem.split_once("_nohash_")?;
    if speaker.is_empty() {
        return None;
    }
    Some((speaker, take.parse().ok()?))
}

/// The two lists that define the official split, as sets of
/// `word/file.wav` paths.
#[derive(Clone, Debug, Default)]
pub struct SplitLists {
    validation: HashSet<String>,
    testing: HashSet<String>,
}

impl SplitLists {
    /// Read `validation_list.txt` and `testing_list.txt` from the dataset
    /// directory.
    pub fn read(root: &Path) -> io::Result<SplitLists> {
        let read_list = |name: &str| -> io::Result<String> {
            std::fs::read_to_string(root.join(name)).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", root.join(name).display())))
        };
        Ok(SplitLists::from_text(&read_list("validation_list.txt")?, &read_list("testing_list.txt")?))
    }

    /// Build the lists from the text of the two files (one path per line).
    pub fn from_text(validation: &str, testing: &str) -> SplitLists {
        let lines = |text: &str| text.lines().map(str::trim).filter(|line| !line.is_empty()).map(str::to_string).collect();
        SplitLists { validation: lines(validation), testing: lines(testing) }
    }

    /// The part a clip belongs to. `relative_path` is `word/file.wav`, as
    /// written in the lists. Anything in neither list is training data.
    pub fn split_of(&self, relative_path: &str) -> Split {
        if self.testing.contains(relative_path) {
            Split::Test
        } else if self.validation.contains(relative_path) {
            Split::Validation
        } else {
            Split::Train
        }
    }
}

/// Settings for [`SpeechCommands::load_with`].
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// How many threads read files at the same time.
    pub threads: usize,
    /// If set, keep at most this many clips of each word (the first ones in
    /// file-name order, which is random with respect to everything that
    /// matters because the names are hashes). For quick experiments.
    pub max_per_word: Option<usize>,
    /// If set, load only clips from these parts.
    pub only: Option<Vec<Split>>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions { threads: std::thread::available_parallelism().map_or(4, |n| n.get()).min(10), max_per_word: None, only: None }
    }
}

/// The dataset, in memory.
#[derive(Clone, Debug)]
pub struct SpeechCommands {
    /// Every clip, ordered by word and then by file name, so that the same
    /// directory always loads to the same list.
    pub clips: Vec<Clip>,
    /// The anonymous speaker ids from the file names. `Clip::speaker` is an
    /// index into this list.
    pub speakers: Vec<String>,
}

/// One file we intend to read: where it is and what its name told us.
struct Entry {
    path: PathBuf,
    word: usize,
    speaker: u32,
    take: u32,
    split: Split,
}

impl SpeechCommands {
    /// Load the whole dataset from [`dataset_dir`] with default options.
    pub fn load() -> io::Result<SpeechCommands> {
        SpeechCommands::load_with(&dataset_dir(), &LoadOptions::default())
    }

    /// Load the dataset found at `root`.
    ///
    /// Reading 105,000 small files is slow one at a time and fast in
    /// parallel, because most of the time goes into waiting for the disk and
    /// decoding. So we first make the list of files (cheap), then hand every
    /// thread its share of the list.
    pub fn load_with(root: &Path, options: &LoadOptions) -> io::Result<SpeechCommands> {
        let lists = SplitLists::read(root)?;
        let (entries, speakers) = list_entries(root, &lists, options)?;

        let n_threads = options.threads.clamp(1, entries.len().max(1));
        let share = entries.len().div_ceil(n_threads).max(1);
        let mut clips: Vec<Clip> = Vec::with_capacity(entries.len());
        // Each thread reads one contiguous stretch of the list, and the
        // stretches are joined in order, so the result does not depend on
        // which thread finished first.
        let results: Vec<io::Result<Vec<Clip>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = entries.chunks(share).map(|stretch| scope.spawn(move || stretch.iter().map(read_clip).collect())).collect();
            handles.into_iter().map(|handle| handle.join().expect("a loader thread panicked")).collect()
        });
        for result in results {
            clips.extend(result?);
        }
        Ok(SpeechCommands { clips, speakers })
    }

    /// The clips of one part of the split.
    pub fn split(&self, split: Split) -> impl Iterator<Item = &Clip> {
        self.clips.iter().filter(move |clip| clip.split == split)
    }

    /// How many clips each part has: `(train, validation, test)`.
    pub fn split_sizes(&self) -> (usize, usize, usize) {
        let count = |split| self.split(split).count();
        (count(Split::Train), count(Split::Validation), count(Split::Test))
    }

    /// How many distinct speakers each part has: `(train, validation, test)`.
    pub fn speakers_per_split(&self) -> (usize, usize, usize) {
        let count = |split| self.split(split).map(|clip| clip.speaker).collect::<HashSet<_>>().len();
        (count(Split::Train), count(Split::Validation), count(Split::Test))
    }

    /// Total size of the audio held in memory, in bytes.
    pub fn audio_bytes(&self) -> usize {
        self.clips.iter().map(|clip| clip.samples.len() * std::mem::size_of::<i16>()).sum()
    }
}

/// Walk the 35 word directories and decide, from names alone, what to load.
fn list_entries(root: &Path, lists: &SplitLists, options: &LoadOptions) -> io::Result<(Vec<Entry>, Vec<String>)> {
    let mut entries = Vec::new();
    let mut speakers: Vec<String> = Vec::new();
    let mut speaker_index: HashMap<String, u32> = HashMap::new();

    for (word, name) in WORDS.iter().enumerate() {
        let dir = root.join(name);
        let listing = std::fs::read_dir(&dir).map_err(|e| io::Error::new(e.kind(), format!("{}: {e} (is SPEECH_DATA set?)", dir.display())))?;
        let mut file_names: Vec<String> = Vec::new();
        for item in listing {
            file_names.push(item?.file_name().to_string_lossy().into_owned());
        }
        // The operating system lists files in no particular order; sorting
        // makes every run see the same dataset in the same order.
        file_names.sort();

        let mut kept = 0;
        for file_name in &file_names {
            let Some((speaker, take)) = parse_file_name(file_name) else { continue };
            let split = lists.split_of(&format!("{name}/{file_name}"));
            if options.only.as_ref().is_some_and(|wanted| !wanted.contains(&split)) {
                continue;
            }
            if options.max_per_word.is_some_and(|max| kept >= max) {
                break;
            }
            kept += 1;
            let speaker = *speaker_index.entry(speaker.to_string()).or_insert_with(|| {
                speakers.push(speaker.to_string());
                (speakers.len() - 1) as u32
            });
            entries.push(Entry { path: dir.join(file_name), word, speaker, take, split });
        }
    }
    Ok((entries, speakers))
}

/// Read one file from disk into a [`Clip`].
fn read_clip(entry: &Entry) -> io::Result<Clip> {
    let wav = Wav::read(&entry.path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", entry.path.display())))?;
    if wav.sample_rate != SAMPLE_RATE {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: expected 16 kHz audio, found {} Hz", entry.path.display(), wav.sample_rate)));
    }
    let mut samples = to_i16(&wav.samples);
    samples.truncate(CLIP_SAMPLES);
    Ok(Clip { word: entry.word, speaker: entry.speaker, take: entry.take, split: entry.split, samples })
}

/// The six long recordings in `_background_noise_` (dishes, a tap, an
/// exercise bike, a cat, and synthetic white and pink noise), as
/// `(name, samples)`, sorted by name.
pub fn load_background_noise(root: &Path) -> io::Result<Vec<(String, Vec<f32>)>> {
    let dir = root.join("_background_noise_");
    let mut names: Vec<String> = Vec::new();
    for item in std::fs::read_dir(&dir).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", dir.display())))? {
        let name = item?.file_name().to_string_lossy().into_owned();
        if name.ends_with(".wav") {
            names.push(name);
        }
    }
    names.sort();
    let mut tracks = Vec::new();
    for name in names {
        let wav = Wav::read(dir.join(&name))?;
        if wav.sample_rate != SAMPLE_RATE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{name}: expected 16 kHz audio")));
        }
        tracks.push((name.trim_end_matches(".wav").to_string(), wav.samples));
    }
    Ok(tracks)
}

/// Run `work` on every item of `items`, on `threads` threads, and return the
/// results in the order of the items.
///
/// The one parallel pattern this book needs outside of training: a list of
/// independent jobs (compute the features of each clip, synthesize each
/// utterance of a batch). Thread `k` takes one contiguous stretch of the
/// list, so the output order never depends on timing.
pub fn parallel_map<T: Sync, R: Send>(items: &[T], threads: usize, work: impl Fn(usize, &T) -> R + Sync) -> Vec<R> {
    let n_threads = threads.clamp(1, items.len().max(1));
    if n_threads == 1 {
        return items.iter().enumerate().map(|(i, item)| work(i, item)).collect();
    }
    let share = items.len().div_ceil(n_threads);
    let work = &work;
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(share)
            .enumerate()
            .map(|(k, stretch)| scope.spawn(move || stretch.iter().enumerate().map(|(i, item)| work(k * share + i, item)).collect::<Vec<R>>()))
            .collect();
        let mut all = Vec::with_capacity(items.len());
        for handle in handles {
            all.extend(handle.join().expect("a worker thread panicked"));
        }
        all
    })
}

/// Log-mel features of every clip (each padded to one second, so every
/// result is `98 x n_mels`), computed on `threads` threads.
///
/// This is the *feature cache*: a classifier looks at every clip once per
/// epoch, and recomputing 105,000 spectrograms each time would cost more
/// than the training itself. 105,000 clips x 98 frames x 40 numbers x 4
/// bytes is 1.6 GB, which fits in memory comfortably.
pub fn one_second_features(clips: &[&Clip], extractor: &FeatureExtractor, threads: usize) -> Vec<Mat> {
    parallel_map(clips, threads, |_, clip| extractor.log_mel(&clip.audio_one_second()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn there_are_35_sorted_distinct_words() {
        assert_eq!(WORDS.len(), 35);
        assert!(WORDS.windows(2).all(|pair| pair[0] < pair[1]), "WORDS must be sorted and distinct");
        assert_eq!(word_index("backward"), Some(0));
        assert_eq!(word_index("zero"), Some(34));
        assert_eq!(word_index("hello"), None);
    }

    #[test]
    fn file_names_give_speaker_and_take() {
        assert_eq!(parse_file_name("0a7c2a8d_nohash_1.wav"), Some(("0a7c2a8d", 1)));
        assert_eq!(parse_file_name("ffd2ba2f_nohash_12.wav"), Some(("ffd2ba2f", 12)));
        assert_eq!(parse_file_name("README.md"), None);
        assert_eq!(parse_file_name("_nohash_1.wav"), None);
        assert_eq!(parse_file_name("abc_nohash_x.wav"), None);
    }

    #[test]
    fn the_lists_decide_the_split_and_the_rest_is_training() {
        let lists = SplitLists::from_text("yes/aaaa_nohash_0.wav\nno/bbbb_nohash_1.wav\n", "yes/cccc_nohash_0.wav\n\n");
        assert_eq!(lists.split_of("yes/aaaa_nohash_0.wav"), Split::Validation);
        assert_eq!(lists.split_of("no/bbbb_nohash_1.wav"), Split::Validation);
        assert_eq!(lists.split_of("yes/cccc_nohash_0.wav"), Split::Test);
        assert_eq!(lists.split_of("yes/dddd_nohash_0.wav"), Split::Train);
        // Same file name under another word is a different clip.
        assert_eq!(lists.split_of("no/aaaa_nohash_0.wav"), Split::Train);
    }

    #[test]
    fn i16_round_trip_is_exact() {
        let original: Vec<i16> = vec![-32768, -1, 0, 1, 12345, 32767];
        assert_eq!(to_i16(&to_f32(&original)), original);
        assert_eq!(to_i16(&[2.0, -2.0]), vec![32767, -32768]);
    }

    #[test]
    fn parallel_map_keeps_order_for_any_thread_count() {
        let items: Vec<usize> = (0..103).collect();
        let expected: Vec<usize> = items.iter().map(|x| x * x + 1).collect();
        for threads in [1, 2, 7, 200] {
            let got = parallel_map(&items, threads, |i, &x| {
                assert_eq!(i, x, "the index passed to the closure is the item's position");
                x * x + 1
            });
            assert_eq!(got, expected, "threads = {threads}");
        }
        assert!(parallel_map(&[] as &[usize], 4, |_, &x| x).is_empty());
    }

    /// Build a miniature dataset on disk (two speakers in training, one in
    /// validation, one in test) and load it back.
    #[test]
    fn loads_a_miniature_dataset_with_the_right_split() {
        let root = std::env::temp_dir().join(format!("speech_commands_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for word in WORDS {
            std::fs::create_dir_all(root.join(word)).unwrap();
        }
        std::fs::create_dir_all(root.join("_background_noise_")).unwrap();
        let tone = |len: usize, value: f32| Wav::new(SAMPLE_RATE, vec![value; len]);
        // (word, speaker, take, samples, value)
        let files = [("yes", "aaaa", 0, 16000, 0.25), ("yes", "bbbb", 0, 8000, 0.5), ("yes", "cccc", 1, 16000, -0.25), ("no", "dddd", 0, 12000, 0.125), ("no", "aaaa", 3, 16000, 0.0)];
        for (word, speaker, take, len, value) in files {
            tone(len, value).write(root.join(word).join(format!("{speaker}_nohash_{take}.wav"))).unwrap();
        }
        std::fs::write(root.join("yes").join("notes.txt"), "not a clip").unwrap();
        std::fs::write(root.join("validation_list.txt"), "yes/cccc_nohash_1.wav\n").unwrap();
        std::fs::write(root.join("testing_list.txt"), "no/dddd_nohash_0.wav\n").unwrap();
        tone(32000, 0.01).write(root.join("_background_noise_").join("hum.wav")).unwrap();

        for threads in [1, 3] {
            let data = SpeechCommands::load_with(&root, &LoadOptions { threads, max_per_word: None, only: None }).unwrap();
            assert_eq!(data.clips.len(), 5);
            assert_eq!(data.split_sizes(), (3, 1, 1));
            assert_eq!(data.speakers_per_split(), (2, 1, 1));
            // Ordered by word ("no" is before "yes" in WORDS), then file name.
            let summary: Vec<(usize, &str, u32, Split, usize)> =
                data.clips.iter().map(|c| (c.word, data.speakers[c.speaker as usize].as_str(), c.take, c.split, c.samples.len())).collect();
            let (no, yes) = (word_index("no").unwrap(), word_index("yes").unwrap());
            assert_eq!(
                summary,
                vec![
                    (no, "aaaa", 3, Split::Train, 16000),
                    (no, "dddd", 0, Split::Test, 12000),
                    (yes, "aaaa", 0, Split::Train, 16000),
                    (yes, "bbbb", 0, Split::Train, 8000),
                    (yes, "cccc", 1, Split::Validation, 16000),
                ]
            );
            // The same speaker id under two words is one speaker.
            assert_eq!(data.clips[0].speaker, data.clips[2].speaker);
            assert_eq!(data.clips[3].samples[0], 16384); // 0.5 as a 16-bit sample
            assert_eq!(data.clips[3].audio_one_second().len(), CLIP_SAMPLES);
            assert_eq!(data.clips[3].audio_one_second()[8000], 0.0);
            assert_eq!(data.audio_bytes(), 2 * (16000 * 3 + 12000 + 8000));
        }

        let only_test = SpeechCommands::load_with(&root, &LoadOptions { threads: 2, max_per_word: None, only: Some(vec![Split::Test]) }).unwrap();
        assert_eq!(only_test.clips.len(), 1);
        let capped = SpeechCommands::load_with(&root, &LoadOptions { threads: 2, max_per_word: Some(1), only: None }).unwrap();
        assert_eq!(capped.clips.len(), 2);

        let noise = load_background_noise(&root).unwrap();
        assert_eq!(noise.len(), 1);
        assert_eq!((noise[0].0.as_str(), noise[0].1.len()), ("hum", 32000));

        let extractor = FeatureExtractor::new(&crate::dsp::features::FeatureConfig::new(SAMPLE_RATE));
        let clips: Vec<&Clip> = data_clips(&only_test);
        let features = one_second_features(&clips, &extractor, 2);
        assert_eq!((features[0].rows, features[0].cols), (98, 40));

        std::fs::remove_dir_all(&root).unwrap();
    }

    fn data_clips(data: &SpeechCommands) -> Vec<&Clip> {
        data.clips.iter().collect()
    }
}
