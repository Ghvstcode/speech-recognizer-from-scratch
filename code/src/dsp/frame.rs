//! Cutting a signal into short frames, and what we can already tell from
//! them without any Fourier transform: how loud each frame is, how often it
//! crosses zero, and where the speech starts and stops.

/// How many frames [`frames`] produces.
///
/// # Edge policy
///
/// Frame `i` covers samples `i * hop_len .. i * hop_len + frame_len`. We only
/// keep frames that fit **entirely** inside the signal: nothing is padded,
/// and a few trailing samples that do not fill a frame are dropped. So
///
/// ```text
/// num_frames = 0                                        if n_samples < frame_len
///            = 1 + (n_samples - frame_len) / hop_len    otherwise (integer division)
/// ```
///
/// One second at 16 kHz with 25 ms frames every 10 ms: `1 + (16000 - 400) / 160 = 98`.
///
/// Other libraries pad the signal so that frames are centred on multiples of
/// the hop (and get `1 + n_samples / hop_len` frames). Ours is the simplest
/// rule that never invents samples.
pub fn num_frames(n_samples: usize, frame_len: usize, hop_len: usize) -> usize {
    assert!(frame_len > 0 && hop_len > 0, "frame_len and hop_len must be at least 1");
    if n_samples < frame_len {
        0
    } else {
        1 + (n_samples - frame_len) / hop_len
    }
}

/// Cut `samples` into overlapping frames of `frame_len` samples, each starting
/// `hop_len` samples after the previous one. See [`num_frames`] for the edge
/// policy. The frames borrow from `samples`; nothing is copied.
pub fn frames(samples: &[f32], frame_len: usize, hop_len: usize) -> Vec<&[f32]> {
    (0..num_frames(samples.len(), frame_len, hop_len)).map(|i| &samples[i * hop_len..i * hop_len + frame_len]).collect()
}

/// Pre-emphasis: `y[t] = x[t] - coeff * x[t-1]`, with `y[0] = x[0]`.
///
/// Speech has most of its energy at low frequencies; the higher ones, where
/// the consonants live, are much weaker. Subtracting most of the previous
/// sample cancels what changes slowly (low frequencies) and keeps what
/// changes quickly (high frequencies), which evens the spectrum out before
/// we analyse it. `coeff` is usually 0.97; 0.0 returns the signal unchanged.
pub fn preemphasis(samples: &[f32], coeff: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    let mut previous = 0.0;
    for &x in samples {
        out.push(x - coeff * previous);
        previous = x;
    }
    out
}

/// The energy of a frame: the mean of its squared samples.
///
/// (Strictly speaking this is *power*, energy per sample. Dividing by the
/// length makes frames of different sizes comparable.) A full-scale sine has
/// energy 0.5; silence has energy 0.
pub fn energy(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32
}

/// The energy of a frame in decibels: `10 log10(energy)`.
///
/// Ears judge loudness by ratios, not differences, and decibels turn ratios
/// into differences: +10 dB is always "ten times the energy". 0 dB here is a
/// full-scale square wave; a full-scale sine is -3 dB; quiet room noise in a
/// 16-bit recording sits around -60 dB. Digital silence would be minus
/// infinity, so we stop at -100 dB.
pub fn energy_db(frame: &[f32]) -> f32 {
    10.0 * (energy(frame) + 1e-10).log10()
}

/// Energy of every frame: the loudness contour of the signal.
pub fn short_time_energy(samples: &[f32], frame_len: usize, hop_len: usize) -> Vec<f32> {
    frames(samples, frame_len, hop_len).into_iter().map(energy).collect()
}

/// Zero-crossing rate: the fraction of neighbouring sample pairs that have
/// opposite signs, between 0 and 1.
///
/// A cheap stand-in for "how high-pitched is this frame": a sine of frequency
/// `f` crosses zero `2f` times per second, so its rate is `2f / sample_rate`.
/// Hissy sounds like "s" and "f" score high, vowels score low.
pub fn zero_crossing_rate(frame: &[f32]) -> f32 {
    if frame.len() < 2 {
        return 0.0;
    }
    let crossings = frame.windows(2).filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0)).count();
    crossings as f32 / (frame.len() - 1) as f32
}

/// Knobs of the endpoint detector. The defaults suit single words and short
/// phrases, on clean recordings and on a laptop microphone alike.
#[derive(Clone, Debug, PartialEq)]
pub struct EndpointConfig {
    /// Length of the blocks the detector measures, in milliseconds.
    pub block_ms: f32,
    /// A block is "loud" if it is this many dB above the noise floor.
    pub start_db: f32,
    /// Once speech has started, a block only has to be this many dB above the
    /// noise floor to count. Lower than `start_db`, so that a word that
    /// trails off is not cut short.
    pub end_db: f32,
    /// Blocks quieter than this (in dB, see [`energy_db`]) never count as
    /// loud, however quiet the background is. Without it, a background of
    /// pure digital silence would make the faintest rustle "10 dB above the
    /// noise".
    pub min_level_db: f32,
    /// Only used when the whole recording is available ([`find_speech`]):
    /// blocks more than this many dB below the loudest block of the clip
    /// never count as loud. It replaces `min_level_db`, which a live detector
    /// needs because it cannot know how loud the speaker is going to be.
    pub peak_range_db: f32,
    /// Speech starts once this many milliseconds in a row were loud. Guards
    /// against clicks and pops.
    pub min_speech_ms: f32,
    /// "Hangover": speech ends only after this many milliseconds in a row
    /// were quiet. Guards against cutting a word at a short internal pause,
    /// like the closed lips of the "p" in "happy".
    pub hangover_ms: f32,
    /// Extra audio kept before the first and after the last loud block, in
    /// milliseconds. The soft beginning of a word is often below threshold.
    pub padding_ms: f32,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        EndpointConfig {
            block_ms: 10.0,
            start_db: 10.0,
            end_db: 5.0,
            min_level_db: -60.0,
            peak_range_db: 40.0,
            min_speech_ms: 30.0,
            hangover_ms: 200.0,
            padding_ms: 30.0,
        }
    }
}

/// What the endpoint detector has to say after a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointEvent {
    /// Speech began at this sample (counted from the start of the stream,
    /// padding included, so it lies a little in the past).
    SpeechStart { sample: usize },
    /// Speech ended at this sample (exclusive, padding included).
    SpeechEnd { sample: usize },
}

/// A streaming endpoint detector: feed it blocks of audio as they arrive and
/// it tells you when speech starts and when it stops.
///
/// It is a two-state machine driven by one number per block, its energy in
/// decibels (measured after removing any constant offset):
///
/// ```text
///              min_speech_ms of loud blocks
///   SILENCE ─────────────────────────────────▶ SPEECH
///           ◀─────────────────────────────────
///              hangover_ms of quiet blocks
/// ```
///
/// "Loud" is relative to a *noise floor* the detector estimates itself:
/// while in silence, the floor drops immediately to any quieter block and
/// creeps up slowly towards louder ones. So it adapts to the room it is in.
/// The first block it hears is taken to be background.
///
/// ```
/// use speech::dsp::frame::{Endpointer, EndpointConfig, EndpointEvent};
/// let mut endpointer = Endpointer::new(16000, EndpointConfig::default());
/// let audio = vec![0.0f32; 16000];
/// for block in audio.chunks_exact(endpointer.block_len()) {
///     match endpointer.push(block) {
///         Some(EndpointEvent::SpeechStart { sample }) => println!("speech from {sample}"),
///         Some(EndpointEvent::SpeechEnd { sample }) => println!("speech until {sample}"),
///         None => {}
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Endpointer {
    config: EndpointConfig,
    block_len: usize,
    min_speech_blocks: usize,
    hangover_blocks: usize,
    padding_samples: usize,

    /// Estimated background level in dB; `None` until the first block.
    noise_db: Option<f32>,
    in_speech: bool,
    /// Blocks pushed so far.
    blocks_seen: usize,
    /// While in silence: how many loud blocks in a row we have just seen.
    loud_run: usize,
    /// While in speech: how many quiet blocks in a row we have just seen.
    quiet_run: usize,
}

impl Endpointer {
    /// A detector for audio at `sample_rate`.
    pub fn new(sample_rate: u32, config: EndpointConfig) -> Self {
        let ms_to_samples = |ms: f32| (ms * sample_rate as f32 / 1000.0).round() as usize;
        let block_len = ms_to_samples(config.block_ms).max(1);
        let ms_to_blocks = |ms: f32| ((ms / config.block_ms).round() as usize).max(1);
        Endpointer {
            block_len,
            min_speech_blocks: ms_to_blocks(config.min_speech_ms),
            hangover_blocks: ms_to_blocks(config.hangover_ms),
            padding_samples: ms_to_samples(config.padding_ms),
            config,
            noise_db: None,
            in_speech: false,
            blocks_seen: 0,
            loud_run: 0,
            quiet_run: 0,
        }
    }

    /// Start from a known noise floor (in dB) instead of learning it from the
    /// first block. Useful when the whole recording is available up front.
    pub fn with_noise_floor(mut self, noise_db: f32) -> Self {
        self.noise_db = Some(noise_db);
        self
    }

    /// The number of samples [`Endpointer::push`] expects per call.
    pub fn block_len(&self) -> usize {
        self.block_len
    }

    /// Whether we are currently inside speech.
    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// The current estimate of the background level in dB, if any block has
    /// been seen yet.
    pub fn noise_floor_db(&self) -> Option<f32> {
        self.noise_db
    }

    /// Feed the next block of audio (`block_len()` samples; the blocks do not
    /// overlap). Returns an event if this block changed the state.
    pub fn push(&mut self, block: &[f32]) -> Option<EndpointEvent> {
        let level = level_db(block);
        let noise = *self.noise_db.get_or_insert(level);
        let block_index = self.blocks_seen;
        self.blocks_seen += 1;

        if self.in_speech {
            let still_loud = level > (noise + self.config.end_db).max(self.config.min_level_db);
            if still_loud {
                self.quiet_run = 0;
                return None;
            }
            self.quiet_run += 1;
            if self.quiet_run < self.hangover_blocks {
                return None;
            }
            // Quiet for long enough. Speech ended where the quiet run began.
            self.in_speech = false;
            let last_loud_end = (block_index + 1 - self.quiet_run) * self.block_len;
            self.quiet_run = 0;
            return Some(EndpointEvent::SpeechEnd { sample: last_loud_end + self.padding_samples });
        }

        let loud = level > (noise + self.config.start_db).max(self.config.min_level_db);
        if !loud {
            self.loud_run = 0;
            self.adapt_noise_floor(level);
            return None;
        }
        self.loud_run += 1;
        if self.loud_run < self.min_speech_blocks {
            return None;
        }
        // Loud for long enough. Speech started where the loud run began.
        self.in_speech = true;
        let first_loud_start = (block_index + 1 - self.loud_run) * self.block_len;
        self.loud_run = 0;
        Some(EndpointEvent::SpeechStart { sample: first_loud_start.saturating_sub(self.padding_samples) })
    }

    /// Tell the detector the stream is over. If we were in the middle of
    /// speech, this closes it.
    pub fn finish(&mut self) -> Option<EndpointEvent> {
        if !self.in_speech {
            return None;
        }
        self.in_speech = false;
        let last_loud_end = (self.blocks_seen - self.quiet_run) * self.block_len;
        self.quiet_run = 0;
        Some(EndpointEvent::SpeechEnd { sample: last_loud_end + self.padding_samples })
    }

    /// Track the background level with a block we judged to be silence:
    /// follow it down at once, follow it up slowly, closing the gap with a
    /// time constant of one second. Slowly matters: a word that swells
    /// gently must not drag the floor up with it and go unnoticed.
    fn adapt_noise_floor(&mut self, level: f32) {
        let rise_per_block = self.config.block_ms / 1000.0;
        if let Some(noise) = self.noise_db.as_mut() {
            if level < *noise {
                *noise = level;
            } else {
                *noise += rise_per_block * (level - *noise);
            }
        }
    }
}

/// The loudness of a block for endpointing, in dB: like [`energy_db`], but
/// measured around the block's own average instead of around zero.
///
/// Cheap sound cards add a constant offset to every sample. It is inaudible,
/// but it has energy, and a recording with an offset of 0.03 looks like it
/// has a noise floor of -30 dB with the speech barely sticking out. Taking
/// the average out first (which makes this the *variance* of the block)
/// removes the offset and nothing else that matters.
fn level_db(block: &[f32]) -> f32 {
    if block.is_empty() {
        return -100.0;
    }
    let mean = block.iter().sum::<f32>() / block.len() as f32;
    let variance = block.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / block.len() as f32;
    10.0 * (variance + 1e-10).log10()
}

/// Find the stretch of speech in a recording: `Some((start, end))` as sample
/// indices (`end` exclusive), or `None` if nothing stands out from the
/// background.
///
/// This is [`Endpointer`] run over the whole clip, with one advantage a live
/// detector does not have: we can look at the entire recording first. So we
/// take the noise floor from its quietest tenth, and we judge loudness
/// relative to its loudest block rather than on an absolute scale (see
/// [`EndpointConfig::peak_range_db`]), which makes the result independent of
/// the recording level. If the detector fires several times (two words with
/// a long pause between them), the result spans from the first start to the
/// last end.
pub fn find_speech(samples: &[f32], sample_rate: u32) -> Option<(usize, usize)> {
    find_speech_with(samples, sample_rate, &EndpointConfig::default())
}

/// [`find_speech`] with your own settings.
pub fn find_speech_with(samples: &[f32], sample_rate: u32, config: &EndpointConfig) -> Option<(usize, usize)> {
    let block_len = Endpointer::new(sample_rate, config.clone()).block_len();
    let mut levels: Vec<f32> = samples.chunks_exact(block_len).map(level_db).collect();
    if levels.is_empty() {
        return None;
    }
    levels.sort_by(f32::total_cmp);
    // The noise floor: the level below which 10% of the blocks fall. A
    // percentile rather than the minimum, so that one freak block does not
    // decide everything.
    let noise_floor = levels[levels.len() / 10];
    let loudest = levels[levels.len() - 1];

    let config = EndpointConfig { min_level_db: loudest - config.peak_range_db, ..config.clone() };
    let mut endpointer = Endpointer::new(sample_rate, config).with_noise_floor(noise_floor);

    let (mut first_start, mut last_end) = (None, None);
    let events = samples.chunks_exact(block_len).filter_map(|block| endpointer.push(block)).collect::<Vec<_>>();
    for event in events.into_iter().chain(endpointer.finish()) {
        match event {
            EndpointEvent::SpeechStart { sample } => first_start = first_start.or(Some(sample)),
            EndpointEvent::SpeechEnd { sample } => last_end = Some(sample.min(samples.len())),
        }
    }
    Some((first_start?, last_end?))
}

/// The part of `samples` that [`find_speech`] considers speech. If it finds
/// none, the whole input is returned unchanged: better a clip with silence
/// around it than no clip at all.
pub fn trim_silence(samples: &[f32], sample_rate: u32) -> &[f32] {
    match find_speech(samples, sample_rate) {
        Some((start, end)) => &samples[start..end],
        None => samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use std::f32::consts::PI;

    #[test]
    fn frame_count_follows_the_documented_formula() {
        assert_eq!(num_frames(16000, 400, 160), 98);
        assert_eq!(num_frames(400, 400, 160), 1);
        assert_eq!(num_frames(399, 400, 160), 0);
        assert_eq!(num_frames(559, 400, 160), 1);
        assert_eq!(num_frames(560, 400, 160), 2);
        assert_eq!(num_frames(0, 400, 160), 0);
    }

    #[test]
    fn frames_overlap_and_drop_the_ragged_tail() {
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let f = frames(&samples, 4, 3);
        assert_eq!(f, vec![&[0.0, 1.0, 2.0, 3.0][..], &[3.0, 4.0, 5.0, 6.0][..], &[6.0, 7.0, 8.0, 9.0][..]]);
        assert_eq!(frames(&samples[..9], 4, 3).len(), 2); // sample 8 is dropped
    }

    #[test]
    fn preemphasis_by_hand() {
        let y = preemphasis(&[1.0, 1.0, 1.0, 0.0], 0.5);
        assert_eq!(y, vec![1.0, 0.5, 0.5, -0.5]);
        assert_eq!(preemphasis(&[0.3, -0.2], 0.0), vec![0.3, -0.2]);
    }

    #[test]
    fn preemphasis_removes_a_constant_offset_almost_entirely() {
        let y = preemphasis(&[1.0; 100], 0.97);
        assert!(y[1..].iter().all(|v| (v - 0.03).abs() < 1e-6));
    }

    #[test]
    fn energy_of_known_signals() {
        assert_eq!(energy(&[0.5, -0.5, 0.5, -0.5]), 0.25);
        assert_eq!(energy(&[]), 0.0);
        // A full-scale sine has mean square 1/2, which is -3.01 dB.
        let sine: Vec<f32> = (0..800).map(|t| (2.0 * PI * 10.0 * t as f32 / 800.0).sin()).collect();
        assert!((energy(&sine) - 0.5).abs() < 1e-4);
        assert!((energy_db(&sine) + 3.0103).abs() < 1e-2);
        assert!((energy_db(&[0.0; 10]) + 100.0).abs() < 1e-3);
        assert_eq!(short_time_energy(&[1.0, 1.0, 0.0, 0.0, 2.0, 2.0], 2, 2), vec![1.0, 0.0, 4.0]);
    }

    #[test]
    fn zero_crossing_rate_of_a_sine_is_twice_its_frequency() {
        assert_eq!(zero_crossing_rate(&[1.0, -1.0, 1.0, -1.0, 1.0]), 1.0);
        assert_eq!(zero_crossing_rate(&[0.1, 0.2, 0.3]), 0.0);
        // 440 Hz at 16 kHz: 2 * 440 / 16000 = 0.055.
        let sine: Vec<f32> = (0..16000).map(|t| (2.0 * PI * 440.0 * t as f32 / 16000.0 + 0.1).sin()).collect();
        assert!((zero_crossing_rate(&sine) - 0.055).abs() < 1e-3);
    }

    /// One second of faint noise at 16 kHz with a 440 Hz tone from `from` to `to`.
    fn tone_in_noise(from: usize, to: usize, noise_amplitude: f32) -> Vec<f32> {
        let mut rng = Rng::new(1);
        (0..16000)
            .map(|t| {
                let tone = if (from..to).contains(&t) { 0.3 * (2.0 * PI * 440.0 * t as f32 / 16000.0).sin() } else { 0.0 };
                tone + noise_amplitude * rng.normal()
            })
            .collect()
    }

    #[test]
    fn find_speech_brackets_a_tone_in_noise() {
        let samples = tone_in_noise(4800, 9600, 0.002);
        let (start, end) = find_speech(&samples, 16000).expect("the tone should be found");
        // Default padding is 30 ms = 480 samples; allow one 10 ms block of slack.
        assert!((4800 - 480 - 160..=4800).contains(&start), "start {start}");
        assert!((9600..=9600 + 480 + 160).contains(&end), "end {end}");
        assert_eq!(trim_silence(&samples, 16000).len(), end - start);
    }

    #[test]
    fn find_speech_finds_nothing_in_noise_or_silence() {
        assert_eq!(find_speech(&tone_in_noise(0, 0, 0.01), 16000), None);
        assert_eq!(find_speech(&vec![0.0; 16000], 16000), None);
        assert_eq!(find_speech(&[], 16000), None);
        // Nothing found: trim_silence hands the input back.
        assert_eq!(trim_silence(&[0.0; 800], 16000).len(), 800);
    }

    #[test]
    fn a_short_click_is_ignored_and_a_short_pause_is_bridged() {
        // A 10 ms click: shorter than min_speech_ms.
        assert_eq!(find_speech(&tone_in_noise(8000, 8160, 0.002), 16000), None);
        // Two tones 100 ms apart: one segment thanks to the hangover.
        let mut samples = tone_in_noise(3200, 6400, 0.002);
        let second = tone_in_noise(8000, 11200, 0.0);
        for (a, b) in samples.iter_mut().zip(&second) {
            *a += b;
        }
        let mut endpointer = Endpointer::new(16000, EndpointConfig::default());
        let events: Vec<_> = samples.chunks_exact(endpointer.block_len()).filter_map(|b| endpointer.push(b)).collect();
        assert_eq!(events.len(), 2, "{events:?}");
        assert!(matches!(events[0], EndpointEvent::SpeechStart { sample } if (2560..=3200).contains(&sample)));
        assert!(matches!(events[1], EndpointEvent::SpeechEnd { sample } if (11200..=11840).contains(&sample)));
    }

    #[test]
    fn a_constant_offset_does_not_hide_the_speech() {
        // The same tone in noise, with every sample shifted by 0.05: plain
        // energy would put the "silence" at -26 dB, 16 dB below nothing.
        let samples: Vec<f32> = tone_in_noise(4800, 9600, 0.002).iter().map(|s| s + 0.05).collect();
        assert!((energy_db(&samples[..160]) + 26.0).abs() < 0.5);
        let (start, end) = find_speech(&samples, 16000).expect("the tone should still be found");
        assert!((4800 - 640..=4800).contains(&start) && (9600..=9600 + 640).contains(&end), "{start}..{end}");
    }

    #[test]
    fn a_slow_fade_in_is_still_detected() {
        // A tone that takes 300 ms to swell from nothing to full strength
        // must not drag the noise floor up with it.
        let mut rng = Rng::new(2);
        let samples: Vec<f32> = (0..16000)
            .map(|t| {
                let swell = ((t as f32 - 4000.0) / 4800.0).clamp(0.0, 1.0);
                0.05 * swell * swell * (2.0 * PI * 300.0 * t as f32 / 16000.0).sin() + 0.002 * rng.normal()
            })
            .collect();
        let mut endpointer = Endpointer::new(16000, EndpointConfig::default());
        let events: Vec<_> = samples.chunks_exact(endpointer.block_len()).filter_map(|b| endpointer.push(b)).collect();
        assert!(matches!(events[..], [EndpointEvent::SpeechStart { sample }] if (4000..8000).contains(&sample)), "{events:?}");
    }

    #[test]
    fn streaming_detector_learns_the_noise_floor_and_reports_each_utterance() {
        // Two tones 400 ms apart: longer than the hangover, so two utterances.
        let mut samples = tone_in_noise(3200, 6400, 0.002);
        let second = tone_in_noise(12800, 16000, 0.0);
        for (a, b) in samples.iter_mut().zip(&second) {
            *a += b;
        }
        let mut endpointer = Endpointer::new(16000, EndpointConfig::default());
        assert_eq!(endpointer.block_len(), 160);
        let mut events: Vec<_> = samples.chunks_exact(endpointer.block_len()).filter_map(|b| endpointer.push(b)).collect();
        assert!(endpointer.in_speech(), "the stream ends in the middle of the second tone");
        events.extend(endpointer.finish());
        assert_eq!(events.len(), 4, "{events:?}");
        assert!(matches!(events[1], EndpointEvent::SpeechEnd { sample } if (6400..=7040).contains(&sample)));
        assert!(matches!(events[2], EndpointEvent::SpeechStart { sample } if (12160..=12800).contains(&sample)));
        // Noise of amplitude 0.002 is 20 log10(0.002) = -54 dB.
        let floor = endpointer.noise_floor_db().unwrap();
        assert!((-58.0..-50.0).contains(&floor), "noise floor {floor}");
    }
}
