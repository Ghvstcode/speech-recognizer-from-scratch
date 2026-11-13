//! Making multi-word utterances out of single-word clips.
//!
//! Our recognizer should hear "seven three go left" and spell it out. To
//! learn that it needs thousands of hours... or at least many thousands of
//! examples of several words in a row, each with its transcript. Speech
//! Commands has no such thing: every clip is one word. So we *build* the
//! utterances: pick a few clips, cut the silence off each, lay them end to
//! end with pauses of random length, and add background noise. The
//! transcript comes for free, because we chose the words.
//!
//! ```text
//! clips      [ seven ]      [ three ]   [ go ]        each trimmed, sped up or slowed
//!                                                      down a little, set to a random level
//! timeline   ....[ seven ]..[ three ]........[ go ]........
//!            lead        gap         gap           trail
//! + noise    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   kitchen, tap, bike... at a random
//!                                                            signal-to-noise ratio
//! = utterance, with the transcript "seven three go"
//! ```
//!
//! # Why in the waveform and not in the features
//!
//! It would be faster to cut and paste *spectrograms*: the features of every
//! clip could be computed once. But three things would then be subtly wrong.
//! Noise adds to the signal *before* the logarithm, so it cannot be mixed
//! into log-mel features by adding. Pre-emphasis and the 25 ms analysis
//! window both look across the boundary between two words, so the frames at
//! a joint are not the frames of either clip. And the per-utterance
//! normalization depends on everything in the utterance. A microphone will
//! hand us a waveform, so we build waveforms and run the real front end on
//! them. The price is one feature extraction per training example, about
//! 0.4 ms per second of audio, which is less than the network then spends on
//! the same example.
//!
//! # What this cannot teach
//!
//! In fluent speech words run into each other ("go left" has no silence in
//! the middle, and the "o" is already bending towards the "l"). Glued
//! clips never do that, however short we make the gaps. The model learns to
//! recognize words spoken one... after... another, which is how people
//! speak to a machine when they want to be understood, but not how they
//! speak to each other.
//!
//! # Reproducibility
//!
//! Everything random is drawn from an [`Rng`] passed in by the caller, and
//! utterance number `i` of a seeded stream is always generated from
//! [`utterance_seed`]`(seed, i)`, no matter which thread makes it. The same
//! seed gives the same audio, bit for bit.

use super::speech_commands::{parallel_map, to_f32, to_i16, Clip, SAMPLE_RATE, WORDS};
use crate::dsp::frame::find_speech;
use crate::rng::Rng;

/// Milliseconds to a number of samples at 16 kHz.
fn ms_to_samples(ms: f32) -> usize {
    (ms.max(0.0) * SAMPLE_RATE as f32 / 1000.0).round() as usize
}

/// Root-mean-square level of a signal (0 for an empty one).
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// A gain in decibels as a factor: +6 dB is about 2x, -20 dB is 0.1x.
pub fn db_to_gain(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

// ---------------------------------------------------------------------------
// The word bank: every clip with its silence cut off
// ---------------------------------------------------------------------------

/// One word, cut out of its one-second clip.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    /// Class id: index into [`WORDS`].
    pub word: usize,
    /// Speaker index, as in [`Clip::speaker`].
    pub speaker: u32,
    /// The trimmed audio.
    pub samples: Vec<i16>,
}

/// How [`WordBank`] cuts words out of clips.
#[derive(Clone, Debug, PartialEq)]
pub struct TrimConfig {
    /// Extra audio kept on each side of what the endpoint detector found,
    /// in milliseconds. (The detector already keeps 30 ms of its own.)
    pub margin_ms: f32,
    /// A "word" shorter than this (margins included, so 100 ms of them are
    /// padding) is a click or a cough, not a word. Such clips are left out.
    pub min_word_ms: f32,
}

impl Default for TrimConfig {
    fn default() -> Self {
        TrimConfig { margin_ms: 20.0, min_word_ms: 200.0 }
    }
}

/// Where the word is inside a clip: `Some((start, end))` in samples, or
/// `None` if the clip should not be used.
///
/// This is the endpoint detector of chapter 2 ([`find_speech`]) plus a
/// margin. In roughly one clip in a hundred the detector finds nothing (the
/// recording is noise from end to end, or silent). We could keep the whole
/// second in that case, but a second of noise labelled "seven" would teach
/// the recognizer the wrong thing, so we drop those clips. Nobody will miss
/// them among 85,000 others.
pub fn locate_word(samples: &[f32], config: &TrimConfig) -> Option<(usize, usize)> {
    let (start, end) = find_speech(samples, SAMPLE_RATE)?;
    let margin = ms_to_samples(config.margin_ms);
    let (start, end) = (start.saturating_sub(margin), (end + margin).min(samples.len()));
    (end - start >= ms_to_samples(config.min_word_ms)).then_some((start, end))
}

/// All the words of one part of the dataset, trimmed and indexed so that we
/// can quickly draw "any clip of the word *go*" or "another clip by this
/// speaker".
#[derive(Clone, Debug)]
pub struct WordBank {
    /// The trimmed words.
    pub words: Vec<Word>,
    /// How many clips were left out because no word could be located.
    pub rejected: usize,
    /// `by_word[w]` lists the positions in `words` of every clip of word `w`.
    by_word: Vec<Vec<u32>>,
    /// One list per speaker: the positions in `words` of that speaker's clips.
    by_speaker: Vec<Vec<u32>>,
    /// For each entry of `words`, which list of `by_speaker` it is in.
    speaker_slot: Vec<u32>,
}

impl WordBank {
    /// Trim every clip (on `threads` threads) and index the result.
    pub fn from_clips(clips: &[&Clip], config: &TrimConfig, threads: usize) -> WordBank {
        let trimmed: Vec<Option<Word>> = parallel_map(clips, threads, |_, clip| {
            let audio = clip.audio();
            let (start, end) = locate_word(&audio, config)?;
            Some(Word { word: clip.word, speaker: clip.speaker, samples: clip.samples[start..end].to_vec() })
        });
        let rejected = trimmed.iter().filter(|word| word.is_none()).count();
        let mut bank = WordBank::from_words(trimmed.into_iter().flatten().collect());
        bank.rejected = rejected;
        bank
    }

    /// Index words that are already trimmed.
    pub fn from_words(words: Vec<Word>) -> WordBank {
        let mut by_word = vec![Vec::new(); WORDS.len()];
        let mut by_speaker: Vec<Vec<u32>> = Vec::new();
        let mut slot_of_speaker = std::collections::HashMap::new();
        let mut speaker_slot = Vec::with_capacity(words.len());
        for (position, word) in words.iter().enumerate() {
            by_word[word.word].push(position as u32);
            let slot = *slot_of_speaker.entry(word.speaker).or_insert_with(|| {
                by_speaker.push(Vec::new());
                by_speaker.len() - 1
            });
            by_speaker[slot].push(position as u32);
            speaker_slot.push(slot as u32);
        }
        WordBank { words, rejected: 0, by_word, by_speaker, speaker_slot }
    }

    /// Number of words in the bank.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Whether the bank has no words at all.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Number of distinct speakers.
    pub fn num_speakers(&self) -> usize {
        self.by_speaker.len()
    }

    /// The ids of all speakers in the bank.
    pub fn speaker_ids(&self) -> Vec<u32> {
        self.by_speaker.iter().map(|clips| self.words[clips[0] as usize].speaker).collect()
    }

    /// Total length of all words, in seconds.
    pub fn total_seconds(&self) -> f32 {
        self.words.iter().map(|word| word.samples.len()).sum::<usize>() as f32 / SAMPLE_RATE as f32
    }

    /// A random clip, chosen so that all 35 words are equally likely (the
    /// dataset has twice as many recordings of "zero" as of "follow"; we do
    /// not want the recognizer to inherit that preference).
    fn draw_any(&self, rng: &mut Rng) -> usize {
        loop {
            let candidates = &self.by_word[rng.below(WORDS.len())];
            if !candidates.is_empty() {
                return candidates[rng.below(candidates.len())] as usize;
            }
        }
    }

    /// A random clip by the same speaker as clip `like`.
    fn draw_same_speaker(&self, like: usize, rng: &mut Rng) -> usize {
        let theirs = &self.by_speaker[self.speaker_slot[like] as usize];
        theirs[rng.below(theirs.len())] as usize
    }
}

// ---------------------------------------------------------------------------
// Background noise
// ---------------------------------------------------------------------------

/// The background recordings we mix under the speech.
#[derive(Clone, Debug)]
pub struct NoiseBank {
    /// `(name, samples)` per recording.
    pub tracks: Vec<(String, Vec<f32>)>,
}

impl NoiseBank {
    /// Keep the stretch of every recording between the fractions `from` and
    /// `to` of its length.
    ///
    /// There are only six noise recordings, and no official split for them.
    /// So that evaluation never hears a stretch of noise the model trained
    /// on, we split each one by time ourselves: training gets the first 80%
    /// (`0.0..0.8`), validation the next 10%, test the last 10%.
    pub fn from_tracks(tracks: &[(String, Vec<f32>)], from: f32, to: f32) -> NoiseBank {
        assert!((0.0..=1.0).contains(&from) && from < to && to <= 1.0, "noise bank: bad range {from}..{to}");
        let part = |samples: &Vec<f32>| {
            let n = samples.len() as f32;
            samples[(from * n) as usize..(to * n) as usize].to_vec()
        };
        NoiseBank { tracks: tracks.iter().map(|(name, samples)| (name.clone(), part(samples))).collect() }
    }

    /// A bank with no recordings: utterances get only the faint hiss.
    pub fn empty() -> NoiseBank {
        NoiseBank { tracks: Vec::new() }
    }

    /// Copy `len` samples from a random position of track `track`, wrapping
    /// around at the end if the track is shorter than that.
    fn excerpt(&self, track: usize, len: usize, rng: &mut Rng) -> Vec<f32> {
        let samples = &self.tracks[track].1;
        let start = rng.below(samples.len());
        (0..len).map(|i| samples[(start + i) % samples.len()]).collect()
    }
}

// ---------------------------------------------------------------------------
// The recipe
// ---------------------------------------------------------------------------

/// Every knob of the synthesis. Ranges are `(low, high)` and values are
/// drawn uniformly between them unless stated otherwise.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthConfig {
    /// Fewest words in an utterance.
    pub min_words: usize,
    /// Most words in an utterance. The count is uniform between the two.
    pub max_words: usize,
    /// How often all words of an utterance come from one speaker. The rest
    /// of the time every word is drawn independently, so voices change mid
    /// utterance. One voice is what the recognizer will meet in use; mixed
    /// voices are harder and stop it from leaning on "the voice changed, so
    /// a new word began".
    pub same_speaker_prob: f32,
    /// Silence between two words, in milliseconds. Negative means the clips
    /// *overlap* by that much (their trimmed edges are near-silent, so a
    /// small overlap just removes the pause). Drawn as
    /// `low + (high - low) * u^2`: short gaps are more common than long ones.
    pub gap_ms: (f32, f32),
    /// Silence before the first word, in milliseconds.
    pub lead_ms: (f32, f32),
    /// Silence after the last word, in milliseconds.
    pub trail_ms: (f32, f32),
    /// Each word is brought to this RMS level (dB relative to full scale)
    /// before its random gain: the clips were recorded at wildly different
    /// volumes, more different than one person's words ever are.
    pub word_level_db: f32,
    /// Random gain per word, in dB around `word_level_db`.
    pub word_gain_db: (f32, f32),
    /// Playback speed per word: 1.1 is 10% faster (and higher pitched).
    pub speed: (f32, f32),
    /// Length of the fade-in and fade-out applied to each word, in
    /// milliseconds, so that a cut never produces a click.
    pub fade_ms: f32,
    /// How often a background recording is mixed in.
    pub noise_prob: f32,
    /// Signal-to-noise ratio when it is, in dB: speech power over noise
    /// power, with speech power measured over the words only (not the
    /// pauses). 30 dB is a quiet room; 5 dB is a running tap next to you.
    pub snr_db: (f32, f32),
    /// Level of a faint white hiss that is *always* added, in dB relative to
    /// full scale. Without it the pauses would be exact digital silence,
    /// something no microphone ever produces and which shows up in log-mel
    /// features as a wall of the smallest possible value.
    pub hiss_db: (f32, f32),
    /// Gain applied to the finished utterance, in dB: some people sit close
    /// to the microphone and some do not.
    pub utterance_gain_db: (f32, f32),
}

impl Default for SynthConfig {
    fn default() -> Self {
        SynthConfig {
            min_words: 1,
            max_words: 5,
            same_speaker_prob: 0.3,
            gap_ms: (-40.0, 500.0),
            lead_ms: (50.0, 500.0),
            trail_ms: (50.0, 500.0),
            word_level_db: -26.0,
            word_gain_db: (-5.0, 5.0),
            speed: (0.9, 1.1),
            fade_ms: 5.0,
            noise_prob: 0.8,
            snr_db: (5.0, 30.0),
            hiss_db: (-75.0, -55.0),
            utterance_gain_db: (-15.0, 0.0),
        }
    }
}

/// One synthesized utterance and everything we know about how it was made.
#[derive(Clone, Debug, PartialEq)]
pub struct Utterance {
    /// The audio, 16 kHz mono.
    pub samples: Vec<f32>,
    /// Class ids of the words, in order.
    pub words: Vec<usize>,
    /// Where each word sits in `samples`: `(start, end)` in samples.
    pub spans: Vec<(usize, usize)>,
    /// Whether all words came from one speaker.
    pub same_speaker: bool,
    /// The signal-to-noise ratio in dB, or `None` if no background recording
    /// was mixed in (only the faint hiss).
    pub snr_db: Option<f32>,
    /// Which background recording was used, if any.
    pub noise_track: Option<usize>,
}

impl Utterance {
    /// The transcript: the words joined by single spaces.
    pub fn text(&self) -> String {
        self.words.iter().map(|&w| WORDS[w]).collect::<Vec<_>>().join(" ")
    }

    /// Length in seconds.
    pub fn seconds(&self) -> f32 {
        self.samples.len() as f32 / SAMPLE_RATE as f32
    }

    /// The audio as 16-bit samples (two bytes each instead of four), for
    /// keeping a large fixed evaluation set in memory.
    pub fn samples_i16(&self) -> Vec<i16> {
        to_i16(&self.samples)
    }
}

/// Change the speed of a recording by reading it at a different rate:
/// output sample `i` is the input at position `i * speed`, interpolated
/// linearly between its two neighbours. `speed = 1.1` makes it 10% shorter
/// and, like a record played too fast, a little higher. (Linear
/// interpolation dulls the very highest frequencies slightly; for a change
/// of a few percent that is far below what the mel filterbank can see.)
pub fn change_speed(samples: &[f32], speed: f32) -> Vec<f32> {
    assert!(speed > 0.0, "change_speed: speed must be positive");
    if samples.len() < 2 || speed == 1.0 {
        return samples.to_vec();
    }
    let out_len = ((samples.len() - 1) as f32 / speed).floor() as usize + 1;
    (0..out_len)
        .map(|i| {
            let position = i as f32 * speed;
            let left = (position as usize).min(samples.len() - 2);
            let fraction = position - left as f32;
            samples[left] * (1.0 - fraction) + samples[left + 1] * fraction
        })
        .collect()
}

/// Fade the first and last `fade` samples in and out linearly.
fn fade_edges(samples: &mut [f32], fade: usize) {
    let fade = fade.min(samples.len() / 2);
    let n = samples.len();
    for i in 0..fade {
        let gain = (i + 1) as f32 / (fade + 1) as f32;
        samples[i] *= gain;
        samples[n - 1 - i] *= gain;
    }
}

/// The seed for utterance number `index` of the stream with seed `seed`.
///
/// Training makes its utterances on many threads. If they all drew from one
/// generator, the result would depend on which thread got there first.
/// Instead every utterance gets a generator of its own, seeded from the
/// stream's seed and its own number. (The odd constants only make sure that
/// neighbouring indices and neighbouring seeds give unrelated generators;
/// `Rng::new` scrambles the result further.)
pub fn utterance_seed(seed: u64, index: u64) -> u64 {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (index.wrapping_add(1)).wrapping_mul(0xD6E8_FEB8_6659_FD93)
}

/// Builds utterances from a [`WordBank`] and a [`NoiseBank`] following a
/// [`SynthConfig`].
#[derive(Clone, Debug)]
pub struct Synthesizer<'a> {
    /// Where the words come from.
    pub words: &'a WordBank,
    /// Where the background noise comes from.
    pub noise: &'a NoiseBank,
    /// The recipe.
    pub config: SynthConfig,
}

impl<'a> Synthesizer<'a> {
    /// A synthesizer over the given banks.
    pub fn new(words: &'a WordBank, noise: &'a NoiseBank, config: SynthConfig) -> Self {
        assert!(!words.is_empty(), "synthesizer: the word bank is empty");
        assert!(config.min_words >= 1 && config.min_words <= config.max_words, "synthesizer: bad word-count range");
        Synthesizer { words, noise, config }
    }

    /// Utterance number `index` of the stream with seed `seed`.
    pub fn utterance_at(&self, seed: u64, index: u64) -> Utterance {
        self.utterance(&mut Rng::new(utterance_seed(seed, index)))
    }

    /// `count` utterances of the stream with seed `seed`, starting at number
    /// `first`, made on `threads` threads. The result is the same for any
    /// number of threads.
    pub fn utterances(&self, seed: u64, first: u64, count: usize, threads: usize) -> Vec<Utterance> {
        let indices: Vec<u64> = (first..first + count as u64).collect();
        parallel_map(&indices, threads, |_, &index| self.utterance_at(seed, index))
    }

    /// Build one utterance, drawing every random choice from `rng`.
    pub fn utterance(&self, rng: &mut Rng) -> Utterance {
        let c = &self.config;

        // 1. Choose the clips.
        let n_words = c.min_words + rng.below(c.max_words - c.min_words + 1);
        let same_speaker = rng.uniform() < c.same_speaker_prob;
        let first = self.words.draw_any(rng);
        let mut chosen = vec![first];
        for _ in 1..n_words {
            chosen.push(if same_speaker { self.words.draw_same_speaker(first, rng) } else { self.words.draw_any(rng) });
        }

        // 2. Prepare each word: speed, level, soft edges.
        let mut pieces: Vec<Vec<f32>> = Vec::with_capacity(n_words);
        for &position in &chosen {
            let mut piece = change_speed(&to_f32(&self.words.words[position].samples), rng.range(c.speed.0, c.speed.1));
            let target = db_to_gain(c.word_level_db + rng.range(c.word_gain_db.0, c.word_gain_db.1));
            // A near-silent clip would need an absurd gain; cap it.
            let gain = (target / rms(&piece).max(1e-6)).min(100.0);
            piece.iter_mut().for_each(|s| *s *= gain);
            fade_edges(&mut piece, ms_to_samples(c.fade_ms));
            pieces.push(piece);
        }

        // 3. Lay them out on a timeline.
        let lead = ms_to_samples(rng.range(c.lead_ms.0, c.lead_ms.1));
        let mut spans = Vec::with_capacity(n_words);
        let mut cursor = lead as i64;
        for (i, piece) in pieces.iter().enumerate() {
            if i > 0 {
                let u = rng.uniform();
                let gap_ms = c.gap_ms.0 + (c.gap_ms.1 - c.gap_ms.0) * u * u;
                cursor += (gap_ms * SAMPLE_RATE as f32 / 1000.0).round() as i64;
            }
            let start = cursor.max(0) as usize;
            spans.push((start, start + piece.len()));
            cursor = (start + piece.len()) as i64;
        }
        let trail = ms_to_samples(rng.range(c.trail_ms.0, c.trail_ms.1));
        let total = spans.iter().map(|&(_, end)| end).max().unwrap_or(0) + trail;
        let mut samples = vec![0.0f32; total];
        for (piece, &(start, _)) in pieces.iter().zip(&spans) {
            // Adding (not copying) lets overlapping words blend.
            for (out, &s) in samples[start..].iter_mut().zip(piece) {
                *out += s;
            }
        }

        // 4. Background noise at a chosen signal-to-noise ratio.
        let speech_power = {
            let (sum, count) = pieces.iter().fold((0.0f32, 0usize), |(sum, count), piece| (sum + piece.iter().map(|s| s * s).sum::<f32>(), count + piece.len()));
            sum / count.max(1) as f32
        };
        let (mut snr_db, mut noise_track) = (None, None);
        if !self.noise.tracks.is_empty() && rng.uniform() < c.noise_prob {
            let track = rng.below(self.noise.tracks.len());
            let noise = self.noise.excerpt(track, total, rng);
            let snr = rng.range(c.snr_db.0, c.snr_db.1);
            let noise_power = rms(&noise).powi(2).max(1e-12);
            // We want speech_power / (k^2 * noise_power) = 10^(snr / 10).
            let k = (speech_power / (noise_power * 10f32.powf(snr / 10.0))).sqrt();
            for (out, &n) in samples.iter_mut().zip(&noise) {
                *out += k * n;
            }
            (snr_db, noise_track) = (Some(snr), Some(track));
        }

        // 5. The hiss that is always there. Uniform noise in [-0.5, 0.5) has
        // RMS 1/sqrt(12), so we scale by sqrt(12) to make the level exact.
        let hiss = db_to_gain(rng.range(c.hiss_db.0, c.hiss_db.1)) * 12f32.sqrt();
        for out in samples.iter_mut() {
            *out += hiss * (rng.uniform() - 0.5);
        }

        // 6. Overall level, and never clip.
        let mut gain = db_to_gain(rng.range(c.utterance_gain_db.0, c.utterance_gain_db.1));
        let peak = samples.iter().fold(0.0f32, |peak, &s| peak.max(s.abs())) * gain;
        if peak > 0.99 {
            gain *= 0.99 / peak;
        }
        samples.iter_mut().for_each(|s| *s *= gain);

        Utterance { samples, words: chosen.iter().map(|&p| self.words.words[p].word).collect(), spans, same_speaker, snr_db, noise_track }
    }
}

// ---------------------------------------------------------------------------
// Everything a training or evaluation program needs, loaded in one call
// ---------------------------------------------------------------------------

/// Seed of the fixed validation set of synthesized utterances.
pub const VALIDATION_SEED: u64 = 1001;
/// Seed of the fixed test set of synthesized utterances.
pub const TEST_SEED: u64 = 2002;
/// Number of utterances in each fixed evaluation set.
pub const EVAL_UTTERANCES: usize = 2000;

/// The words and noise for one part of the split.
#[derive(Clone, Debug)]
pub struct Part {
    /// Trimmed words of this part's speakers.
    pub words: WordBank,
    /// This part's stretch of every noise recording.
    pub noise: NoiseBank,
}

/// The three parts of the dataset, ready for synthesis, and kept strictly
/// apart: different speakers *and* different stretches of noise.
///
/// The rule that keeps our numbers honest is simple to state: nothing the
/// model is tested on may have been available to learn from. Words follow
/// the official speaker split. The noise recordings are split by time (see
/// [`NoiseBank::from_tracks`]). And the evaluation sets are generated from
/// fixed seeds ([`VALIDATION_SEED`], [`TEST_SEED`]), so every program and
/// every run measures on exactly the same 2,000 utterances.
#[derive(Clone, Debug)]
pub struct Corpus {
    /// What the model learns from.
    pub train: Part,
    /// What we tune on.
    pub validation: Part,
    /// What we report on.
    pub test: Part,
}

impl Corpus {
    /// Build the corpus from a loaded dataset and the noise recordings.
    pub fn new(data: &super::speech_commands::SpeechCommands, noise_tracks: &[(String, Vec<f32>)], threads: usize) -> Corpus {
        use super::speech_commands::Split;
        let part = |split: Split, from: f32, to: f32| {
            let clips: Vec<&Clip> = data.split(split).collect();
            Part { words: WordBank::from_clips(&clips, &TrimConfig::default(), threads), noise: NoiseBank::from_tracks(noise_tracks, from, to) }
        };
        Corpus { train: part(Split::Train, 0.0, 0.8), validation: part(Split::Validation, 0.8, 0.9), test: part(Split::Test, 0.9, 1.0) }
    }

    /// Load Speech Commands from `root` and build the corpus. The raw
    /// one-second clips are dropped again once the words are cut out.
    pub fn load(root: &std::path::Path, threads: usize, max_per_word: Option<usize>) -> std::io::Result<Corpus> {
        use super::speech_commands::{load_background_noise, LoadOptions, SpeechCommands};
        let data = SpeechCommands::load_with(root, &LoadOptions { threads, max_per_word, only: None })?;
        let noise = load_background_noise(root)?;
        Ok(Corpus::new(&data, &noise, threads))
    }

    /// The fixed validation set: `count` utterances from validation
    /// speakers, default recipe.
    pub fn validation_set(&self, count: usize, threads: usize) -> Vec<Utterance> {
        Synthesizer::new(&self.validation.words, &self.validation.noise, SynthConfig::default()).utterances(VALIDATION_SEED, 0, count, threads)
    }

    /// The fixed test set: `count` utterances from test speakers, default
    /// recipe.
    pub fn test_set(&self, count: usize, threads: usize) -> Vec<Utterance> {
        Synthesizer::new(&self.test.words, &self.test.noise, SynthConfig::default()).utterances(TEST_SEED, 0, count, threads)
    }

    /// True if no speaker appears in more than one part. (A check on the
    /// dataset's lists and on our loader; the unit tests and the training
    /// program both assert it.)
    pub fn speakers_are_disjoint(&self) -> bool {
        let ids = |part: &Part| part.words.speaker_ids().into_iter().collect::<std::collections::HashSet<u32>>();
        let (train, validation, test) = (ids(&self.train), ids(&self.validation), ids(&self.test));
        train.is_disjoint(&validation) && train.is_disjoint(&test) && validation.is_disjoint(&test)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bank of fake "words": word `w` by speaker `s` is a constant signal
    /// whose value encodes both, so we can tell afterwards what went where.
    fn toy_bank() -> WordBank {
        let mut words = Vec::new();
        for speaker in 0..4u32 {
            for word in 0..WORDS.len() {
                let value = 1000 + 100 * word as i16 + speaker as i16;
                words.push(Word { word, speaker, samples: vec![value; 1600 + 160 * (word % 3)] });
            }
        }
        WordBank::from_words(words)
    }

    /// A recipe with every random distortion switched off.
    fn plain_config() -> SynthConfig {
        SynthConfig {
            min_words: 1,
            max_words: 5,
            same_speaker_prob: 0.5,
            gap_ms: (10.0, 200.0),
            lead_ms: (20.0, 100.0),
            trail_ms: (20.0, 100.0),
            word_level_db: -20.0,
            word_gain_db: (0.0, 0.0),
            speed: (1.0, 1.0),
            fade_ms: 0.0,
            noise_prob: 0.0,
            snr_db: (10.0, 10.0),
            hiss_db: (-200.0, -200.0),
            utterance_gain_db: (0.0, 0.0),
        }
    }

    #[test]
    fn levels_and_gains() {
        assert!((rms(&[0.5, -0.5, 0.5, -0.5]) - 0.5).abs() < 1e-7);
        assert_eq!(rms(&[]), 0.0);
        assert!((db_to_gain(20.0) - 10.0).abs() < 1e-5);
        assert!((db_to_gain(-6.0206) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn change_speed_interpolates_a_ramp_exactly() {
        let ramp: Vec<f32> = (0..11).map(|i| i as f32).collect();
        // Twice as fast: every second sample.
        assert_eq!(change_speed(&ramp, 2.0), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        // Half speed: the midpoints appear.
        let slow = change_speed(&ramp[..3], 0.5);
        assert_eq!(slow, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        assert_eq!(change_speed(&ramp, 1.0), ramp);
    }

    #[test]
    fn locate_word_finds_a_burst_and_rejects_silence() {
        let mut clip = vec![0.0f32; 16000];
        let mut rng = Rng::new(1);
        for s in clip.iter_mut() {
            *s = 0.001 * (rng.uniform() - 0.5);
        }
        for (i, s) in clip[6000..10000].iter_mut().enumerate() {
            *s += 0.3 * (i as f32 * 0.2).sin();
        }
        let config = TrimConfig::default();
        let (start, end) = locate_word(&clip, &config).expect("the burst should be found");
        // 30 ms of detector padding plus our 20 ms margin, on each side.
        assert!((5000..=6000).contains(&start), "start = {start}");
        assert!((10000..=11000).contains(&end), "end = {end}");
        assert_eq!(locate_word(&vec![0.0; 16000], &config), None);
        // A 30 ms click is not a word.
        let mut click = vec![0.0f32; 16000];
        click[8000..8480].iter_mut().enumerate().for_each(|(i, s)| *s = 0.3 * (i as f32 * 0.2).sin());
        assert_eq!(locate_word(&click, &config), None);
    }

    #[test]
    fn the_bank_indexes_by_word_and_speaker() {
        let bank = toy_bank();
        assert_eq!(bank.len(), 4 * 35);
        assert_eq!(bank.num_speakers(), 4);
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let a = bank.draw_any(&mut rng);
            let b = bank.draw_same_speaker(a, &mut rng);
            assert_eq!(bank.words[a].speaker, bank.words[b].speaker);
        }
        // Every word is drawn about equally often.
        let mut counts = [0usize; 35];
        for _ in 0..35_000 {
            counts[bank.words[bank.draw_any(&mut rng)].word] += 1;
        }
        assert!(counts.iter().all(|&c| (800..1200).contains(&c)), "{counts:?}");
    }

    #[test]
    fn same_seed_same_utterance_and_threads_do_not_matter() {
        let (bank, noise) = (toy_bank(), NoiseBank { tracks: vec![("hum".to_string(), (0..5000).map(|i| (i as f32 * 0.01).sin() * 0.1).collect())] });
        let synth = Synthesizer::new(&bank, &noise, SynthConfig::default());
        let a = synth.utterance_at(7, 3);
        let b = synth.utterance_at(7, 3);
        assert_eq!(a, b);
        assert_ne!(a.samples, synth.utterance_at(7, 4).samples);
        assert_ne!(a.samples, synth.utterance_at(8, 3).samples);

        let one = synth.utterances(42, 100, 12, 1);
        let many = synth.utterances(42, 100, 12, 5);
        assert_eq!(one, many);
        assert_eq!(one[2], synth.utterance_at(42, 102));
    }

    #[test]
    fn the_transcript_matches_the_audio() {
        let (bank, noise) = (toy_bank(), NoiseBank::empty());
        let synth = Synthesizer::new(&bank, &noise, plain_config());
        let mut lengths_seen = [false; 6];
        for index in 0..300 {
            let utt = synth.utterance_at(1, index);
            lengths_seen[utt.words.len()] = true;
            assert_eq!(utt.words.len(), utt.spans.len());
            assert_eq!(utt.text().split(' ').count(), utt.words.len());
            for (k, (&word, &(start, end))) in utt.words.iter().zip(&utt.spans).enumerate() {
                // With gaps >= 10 ms the words do not overlap, so each span
                // holds exactly one constant "word". All were scaled to the
                // same RMS (-20 dB = 0.1), and the middle of the span tells
                // us nothing about identity any more... so recover identity
                // from the span length instead (1600 + 160 * (word % 3)).
                assert_eq!(end - start, 1600 + 160 * (word % 3), "word {k} of {:?}", utt.text());
                let level = rms(&utt.samples[start..end]);
                assert!((level - 0.1).abs() < 1e-3, "level {level}");
                if k > 0 {
                    let gap = start - utt.spans[k - 1].1;
                    assert!((160..=3200).contains(&gap), "gap of {gap} samples");
                    assert!(utt.samples[utt.spans[k - 1].1..start].iter().all(|&s| s.abs() < 1e-6), "the gap must be silent");
                }
            }
            // Lead and trail silence within the configured range.
            assert!((320..=1600).contains(&utt.spans[0].0));
            let trail = utt.samples.len() - utt.spans.last().unwrap().1;
            assert!((320..=1600).contains(&trail));
            assert_eq!(utt.snr_db, None);
            assert_eq!(utt.text(), utt.words.iter().map(|&w| WORDS[w]).collect::<Vec<_>>().join(" "));
        }
        assert_eq!(lengths_seen, [false, true, true, true, true, true], "all lengths 1..=5 should occur");
    }

    #[test]
    fn same_speaker_utterances_use_one_speaker() {
        // Make the speaker recoverable: give each speaker's clips a distinct length.
        let mut words = Vec::new();
        for speaker in 0..5u32 {
            for word in 0..WORDS.len() {
                words.push(Word { word, speaker, samples: vec![3000; 2000 + 100 * speaker as usize] });
            }
        }
        let (bank, noise) = (WordBank::from_words(words), NoiseBank::empty());
        let synth = Synthesizer::new(&bank, &noise, SynthConfig { min_words: 4, max_words: 4, ..plain_config() });
        let (mut same, mut mixed) = (0, 0);
        for index in 0..200 {
            let utt = synth.utterance_at(5, index);
            let lengths: Vec<usize> = utt.spans.iter().map(|&(s, e)| e - s).collect();
            let one_speaker = lengths.iter().all(|&l| l == lengths[0]);
            if utt.same_speaker {
                assert!(one_speaker, "marked same-speaker but lengths are {lengths:?}");
                same += 1;
            } else {
                mixed += 1;
            }
        }
        assert!(same > 60 && mixed > 60, "same {same}, mixed {mixed}");
    }

    #[test]
    fn noise_is_mixed_at_the_requested_snr() {
        let bank = toy_bank();
        let mut rng = Rng::new(9);
        let noise = NoiseBank { tracks: vec![("white".to_string(), (0..40_000).map(|_| rng.uniform() - 0.5).collect())] };
        let config = SynthConfig { noise_prob: 1.0, snr_db: (10.0, 10.0), min_words: 3, max_words: 3, ..plain_config() };
        let noisy = Synthesizer::new(&bank, &noise, config.clone()).utterance_at(2, 0);
        let clean = Synthesizer::new(&bank, &NoiseBank::empty(), config).utterance_at(2, 0);
        assert_eq!(noisy.snr_db, Some(10.0));
        assert_eq!(noisy.noise_track, Some(0));
        assert_eq!(noisy.words, clean.words);
        // What was added is the difference; its power should be a tenth of
        // the speech power (0.1^2), i.e. 10 dB below.
        let added: Vec<f32> = noisy.samples.iter().zip(&clean.samples).map(|(a, b)| a - b).collect();
        let measured_snr = 10.0 * (0.01 / rms(&added).powi(2)).log10();
        assert!((measured_snr - 10.0).abs() < 0.3, "measured {measured_snr} dB");
    }

    #[test]
    fn noise_bank_split_by_time_does_not_overlap() {
        let track: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let tracks = vec![("ramp".to_string(), track)];
        let train = NoiseBank::from_tracks(&tracks, 0.0, 0.8);
        let valid = NoiseBank::from_tracks(&tracks, 0.8, 0.9);
        let test = NoiseBank::from_tracks(&tracks, 0.9, 1.0);
        assert_eq!((train.tracks[0].1.len(), valid.tracks[0].1.len(), test.tracks[0].1.len()), (800, 100, 100));
        assert_eq!(train.tracks[0].1.last(), Some(&799.0));
        assert_eq!(valid.tracks[0].1.first(), Some(&800.0));
        assert_eq!(test.tracks[0].1.first(), Some(&900.0));
        // An excerpt longer than the track wraps around.
        let excerpt = valid.excerpt(0, 250, &mut Rng::new(1));
        assert!(excerpt.iter().all(|&s| (800.0..900.0).contains(&s)));
    }

    #[test]
    fn utterances_never_clip() {
        let (bank, noise) = (toy_bank(), NoiseBank::empty());
        let config = SynthConfig { word_level_db: -1.0, gap_ms: (-40.0, 0.0), utterance_gain_db: (0.0, 6.0), ..SynthConfig::default() };
        let synth = Synthesizer::new(&bank, &noise, config);
        for index in 0..50 {
            let utt = synth.utterance_at(11, index);
            assert!(utt.samples.iter().all(|s| s.abs() <= 0.9901), "peak above 0.99");
        }
    }
}
