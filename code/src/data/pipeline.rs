//! From audio to text: the front end, the network and the decoder as one
//! thing you can call.
//!
//! Training, evaluation, the `speech transcribe` command and the live
//! listener all need the same chain,
//!
//! ```text
//! samples ──log-mel──▶ 40 numbers per 10 ms ──normalize──▶ network ──log-softmax──▶ decoder ──▶ text
//! ```
//!
//! and they must agree on every detail of it: a model trained on normalized
//! 40-band features will produce nonsense if it is later fed 80 bands, or
//! features that were not normalized. So the chain is written down once,
//! here, and everybody calls it. (It lives next to the dataset code because
//! "how training examples are turned into network inputs" is part of the
//! data pipeline; the training program is its first user.)

use super::speech_commands::{CLIP_SAMPLES, SAMPLE_RATE, WORDS};
use crate::decode::{greedy, lexicon_beam_search, log_softmax, prefix_beam_search, Alphabet, Lexicon, LexiconOptions};
use crate::dsp::features::{normalize, FeatureConfig, FeatureExtractor};
use crate::dsp::frame::trim_silence;
use crate::dsp::resample::resample;
use crate::mat::Mat;
use crate::nn::{softmax, Model};
use crate::wav::Wav;
use std::io;
use std::path::{Path, PathBuf};

/// Where trained models are kept: the `SPEECH_MODELS` environment variable,
/// or `../models` (relative to `code/`) if it is not set.
pub fn models_dir() -> PathBuf {
    std::env::var_os("SPEECH_MODELS").map_or_else(|| PathBuf::from("../models"), PathBuf::from)
}

/// Read a WAV file and bring it to what our models expect: mono, 16 kHz.
/// ([`Wav::read`] already mixes stereo down to mono.)
pub fn load_audio_16k(path: impl AsRef<Path>) -> io::Result<Vec<f32>> {
    let path = path.as_ref();
    let wav = Wav::read(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    Ok(resample(&wav.samples, wav.sample_rate, SAMPLE_RATE))
}

/// The features both our models are trained on: 40 log-mel bands every
/// 10 ms from 25 ms windows of 16 kHz audio, each band normalized to mean 0
/// and variance 1 over the utterance.
#[derive(Clone, Debug)]
pub struct FrontEnd {
    extractor: FeatureExtractor,
}

impl Default for FrontEnd {
    fn default() -> Self {
        FrontEnd::new()
    }
}

impl FrontEnd {
    /// Number of features per frame.
    pub const N_FEATURES: usize = 40;

    /// The front end. Building it precomputes the window, the FFT tables
    /// and the mel filterbank, so make one and share it between threads.
    pub fn new() -> Self {
        let config = FeatureConfig::new(SAMPLE_RATE);
        assert_eq!(config.n_mels, Self::N_FEATURES, "the default feature configuration changed under us");
        FrontEnd { extractor: FeatureExtractor::new(&config) }
    }

    /// The underlying extractor (for plotting un-normalized spectrograms).
    pub fn extractor(&self) -> &FeatureExtractor {
        &self.extractor
    }

    /// Normalized log-mel features of an utterance: frames x 40.
    ///
    /// Why normalize per utterance? It removes the recording level and the
    /// tilt of the microphone, which differ between every two recordings
    /// and say nothing about the words. It needs the whole utterance, which
    /// is fine for us: the live listener waits for the speaker to finish
    /// before it recognizes anything.
    pub fn features(&self, samples: &[f32]) -> Mat {
        let mut features = self.extractor.log_mel(samples);
        normalize(&mut features);
        features
    }

    /// Number of frames [`FrontEnd::features`] returns for this many samples.
    pub fn num_frames(&self, n_samples: usize) -> usize {
        self.extractor.config().num_frames(n_samples)
    }
}

/// Which decoder turns the network's output into text (chapter 10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decoder {
    /// Best class per frame, then collapse.
    Greedy,
    /// Prefix beam search with this many prefixes.
    Beam {
        /// How many prefixes survive each frame.
        width: usize,
    },
    /// Beam search restricted to sequences of the 35 known words.
    Lexicon {
        /// How many prefixes survive each frame.
        width: usize,
        /// Score added per word (log units); see [`LexiconOptions`].
        word_bonus: f32,
    },
}

impl Decoder {
    /// The beam width that is used when none is asked for.
    pub const DEFAULT_BEAM: usize = 16;
    /// The word bonus tuned on the validation set in chapter 10 (it turned
    /// out to matter very little: anything from -2 to +2 gives the same
    /// error rate to within a tenth of a percent).
    pub const DEFAULT_WORD_BONUS: f32 = -2.0;

    /// The decoder with this name (`greedy`, `beam` or `lexicon`) and
    /// default settings.
    pub fn from_name(name: &str) -> Option<Decoder> {
        match name {
            "greedy" => Some(Decoder::Greedy),
            "beam" => Some(Decoder::Beam { width: Self::DEFAULT_BEAM }),
            "lexicon" => Some(Decoder::Lexicon { width: Self::DEFAULT_BEAM, word_bonus: Self::DEFAULT_WORD_BONUS }),
            _ => None,
        }
    }

    /// The decoder's name, as [`Decoder::from_name`] takes it.
    pub fn name(&self) -> &'static str {
        match self {
            Decoder::Greedy => "greedy",
            Decoder::Beam { .. } => "beam",
            Decoder::Lexicon { .. } => "lexicon",
        }
    }
}

/// The lexicon of the 35 Speech Commands words.
pub fn command_lexicon() -> Lexicon {
    Lexicon::new(&WORDS)
}

/// Decode a grid of log-probabilities (frames x 28) to text.
///
/// The lexicon decoder returns nothing when no sequence of whole words
/// survives in its beam. That happens when the network's favourite
/// spellings are all non-words and the beam is too narrow to keep a real
/// word alive next to them; we then search once more with a beam four times
/// as wide, and only if that fails too is the transcript empty.
pub fn decode(log_probs: &Mat, decoder: &Decoder, lexicon: &Lexicon) -> String {
    if log_probs.rows == 0 {
        return String::new();
    }
    match *decoder {
        Decoder::Greedy => tidy(&Alphabet::decode(&greedy(log_probs))),
        Decoder::Beam { width } => prefix_beam_search(log_probs, width).first().map_or_else(String::new, |best| tidy(&Alphabet::decode(&best.labels))),
        Decoder::Lexicon { width, word_bonus } => {
            let search = |width: usize| {
                let options = LexiconOptions { word_bonus, ..LexiconOptions::new(width) };
                lexicon_beam_search(log_probs, lexicon, &options).first().map(|best| best.text())
            };
            search(width).or_else(|| search(4 * width)).unwrap_or_default()
        }
    }
}

/// Remove spaces at the ends and runs of spaces in the middle. The network
/// sometimes spells a space for the silence before the first word; that is
/// not a transcription error anyone cares about.
fn tidy(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The trained character recognizer of chapters 8 to 10, ready to use.
#[derive(Clone, Debug)]
pub struct Recognizer {
    /// The network: features in, 28 logits per output frame out.
    pub model: Model,
    /// The matching front end.
    pub front_end: FrontEnd,
    /// The vocabulary for [`Decoder::Lexicon`].
    pub lexicon: Lexicon,
}

impl Recognizer {
    /// Wrap a model (for example one that is still being trained).
    pub fn new(model: Model) -> Recognizer {
        Recognizer { model, front_end: FrontEnd::new(), lexicon: command_lexicon() }
    }

    /// Load a model saved with [`Model::save`], checking that it has the
    /// shape of a recognizer (40 features in, 28 classes out).
    pub fn load(path: impl AsRef<Path>) -> io::Result<Recognizer> {
        let path = path.as_ref();
        let model = Model::load(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        check_shape(&model, FrontEnd::N_FEATURES, Alphabet::SIZE, path)?;
        Ok(Recognizer::new(model))
    }

    /// The default location: `recognizer.bin` in [`models_dir`].
    pub fn default_path() -> PathBuf {
        models_dir().join("recognizer.bin")
    }

    /// Per-frame log-probabilities for an utterance: output frames x 28.
    /// Empty (zero rows) if the audio is too short to give a single frame.
    pub fn log_probs(&self, samples: &[f32]) -> Mat {
        let features = self.front_end.features(samples);
        if features.rows == 0 || self.model.out_len(features.rows) == 0 {
            return Mat::zeros(0, Alphabet::SIZE);
        }
        log_softmax(&self.model.predict(&features))
    }

    /// Audio in, text out.
    pub fn transcribe(&self, samples: &[f32], decoder: &Decoder) -> String {
        decode(&self.log_probs(samples), decoder, &self.lexicon)
    }
}

/// The 35-word classifier of chapter 7, ready to use.
#[derive(Clone, Debug)]
pub struct WordClassifier {
    /// The network: 98 x 40 features in, one row of 35 logits out.
    pub model: Model,
    /// The matching front end.
    pub front_end: FrontEnd,
}

impl WordClassifier {
    /// Wrap a model.
    pub fn new(model: Model) -> WordClassifier {
        WordClassifier { model, front_end: FrontEnd::new() }
    }

    /// Load a model saved with [`Model::save`], checking its shape.
    pub fn load(path: impl AsRef<Path>) -> io::Result<WordClassifier> {
        let path = path.as_ref();
        let model = Model::load(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        check_shape(&model, FrontEnd::N_FEATURES, WORDS.len(), path)?;
        Ok(WordClassifier::new(model))
    }

    /// The default location: `words.bin` in [`models_dir`].
    pub fn default_path() -> PathBuf {
        models_dir().join("words.bin")
    }

    /// The features the classifier was trained on: exactly one second of
    /// audio (shorter clips are padded with silence at the end).
    pub fn features(&self, samples: &[f32]) -> Mat {
        let mut one_second = samples.to_vec();
        one_second.resize(CLIP_SAMPLES, 0.0);
        self.front_end.features(&one_second)
    }

    /// Bring any recording into the one-second shape: if it is longer, cut
    /// the silence off first and keep the first second of what remains.
    pub fn fit_one_second(samples: &[f32]) -> Vec<f32> {
        let speech = if samples.len() > CLIP_SAMPLES { trim_silence(samples, SAMPLE_RATE) } else { samples };
        let mut one_second = speech[..speech.len().min(CLIP_SAMPLES)].to_vec();
        one_second.resize(CLIP_SAMPLES, 0.0);
        one_second
    }

    /// The probability of each of the 35 words for a clip.
    pub fn probabilities(&self, samples: &[f32]) -> Vec<f32> {
        softmax(&self.model.predict(&self.features(samples))).row(0).to_vec()
    }

    /// The `k` most probable words with their probabilities, best first.
    pub fn top_k(&self, samples: &[f32], k: usize) -> Vec<(&'static str, f32)> {
        let probabilities = self.probabilities(samples);
        let mut order: Vec<usize> = (0..probabilities.len()).collect();
        order.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]));
        order.into_iter().take(k).map(|class| (WORDS[class], probabilities[class])).collect()
    }
}

/// Refuse a model file that cannot be what the caller thinks it is.
fn check_shape(model: &Model, n_in: usize, n_out: usize, path: &Path) -> io::Result<()> {
    let first_in = model.layers.iter().find_map(|layer| layer.in_features());
    let last_out = model.layers.iter().rev().find_map(|layer| layer.out_features());
    if first_in != Some(n_in) || last_out != Some(n_out) {
        let shape = |v: Option<usize>| v.map_or("?".to_string(), |v| v.to_string());
        let message = format!("{}: expected a model with {n_in} inputs and {n_out} outputs, found {} and {}", path.display(), shape(first_in), shape(last_out));
        return Err(io::Error::new(io::ErrorKind::InvalidData, message));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Layer;
    use crate::rng::Rng;

    fn tiny_recognizer() -> Recognizer {
        let mut rng = Rng::new(1);
        Recognizer::new(Model::new(vec![Layer::conv1d(40, 8, 5, 2, 2, &mut rng), Layer::Relu, Layer::bigru(8, 6, &mut rng), Layer::linear(12, Alphabet::SIZE, &mut rng)]))
    }

    #[test]
    fn front_end_gives_normalized_40_band_frames() {
        let front_end = FrontEnd::new();
        let mut rng = Rng::new(2);
        let samples: Vec<f32> = (0..16000).map(|_| 0.1 * rng.normal()).collect();
        let features = front_end.features(&samples);
        assert_eq!((features.rows, features.cols), (98, 40));
        assert_eq!(front_end.num_frames(16000), 98);
        for c in 0..40 {
            let mean = (0..98).map(|r| features[(r, c)]).sum::<f32>() / 98.0;
            let var = (0..98).map(|r| features[(r, c)].powi(2)).sum::<f32>() / 98.0;
            assert!(mean.abs() < 1e-3 && (var - 1.0).abs() < 1e-2, "column {c}: mean {mean}, variance {var}");
        }
        // Normalization makes the features blind to the recording level.
        let louder: Vec<f32> = samples.iter().map(|s| s * 3.0).collect();
        let louder = front_end.features(&louder);
        assert!(features.data.iter().zip(&louder.data).all(|(a, b)| (a - b).abs() < 1e-2));
    }

    #[test]
    fn log_probs_have_one_row_per_20_ms_and_sum_to_one() {
        let recognizer = tiny_recognizer();
        let mut rng = Rng::new(3);
        let samples: Vec<f32> = (0..24000).map(|_| 0.1 * rng.normal()).collect();
        let log_probs = recognizer.log_probs(&samples);
        assert_eq!((log_probs.rows, log_probs.cols), (74, 28)); // 148 frames, stride 2
        for t in 0..log_probs.rows {
            let total: f32 = log_probs.row(t).iter().map(|lp| lp.exp()).sum();
            assert!((total - 1.0).abs() < 1e-4);
        }
        // Too short for a single frame: no rows, and decoding gives "".
        assert_eq!(recognizer.log_probs(&[0.0; 100]).rows, 0);
        assert_eq!(recognizer.transcribe(&[0.0; 100], &Decoder::Greedy), "");
    }

    /// Log-probabilities that spell the given path, one class per frame,
    /// with probability 0.9 on it.
    fn spelled(path: &[usize]) -> Mat {
        Mat::from_fn(path.len(), Alphabet::SIZE, |t, c| if c == path[t] { 0.9f32.ln() } else { (0.1f32 / 27.0).ln() })
    }

    #[test]
    fn decoders_agree_on_a_clear_grid_and_the_lexicon_repairs_a_typo() {
        let lexicon = command_lexicon();
        let frames = |text: &str| -> Vec<usize> {
            // Every character twice, with a blank in between: "g g _ o o _ ..."
            Alphabet::encode(text).into_iter().flat_map(|c| [c, c, 0]).collect()
        };
        let clear = spelled(&frames("go left"));
        for decoder in [Decoder::Greedy, Decoder::Beam { width: 8 }, Decoder::Lexicon { width: 8, word_bonus: 0.0 }] {
            assert_eq!(decode(&clear, &decoder, &lexicon), "go left", "{}", decoder.name());
        }
        // "lefd" is not a word. Greedy reports what it sees; the lexicon
        // decoder has to find the nearest thing that is allowed.
        let typo = spelled(&frames("go lefd"));
        assert_eq!(decode(&typo, &Decoder::Greedy, &lexicon), "go lefd");
        assert_eq!(decode(&typo, &Decoder::Lexicon { width: 16, word_bonus: 0.0 }, &lexicon), "go left");
        assert_eq!(decode(&Mat::zeros(0, 28), &Decoder::Greedy, &lexicon), "");
    }

    #[test]
    fn decoder_names_round_trip() {
        for name in ["greedy", "beam", "lexicon"] {
            assert_eq!(Decoder::from_name(name).unwrap().name(), name);
        }
        assert_eq!(Decoder::from_name("viterbi"), None);
    }

    #[test]
    fn classifier_top_k_is_sorted_and_sums_to_at_most_one() {
        let mut rng = Rng::new(4);
        let model = Model::new(vec![Layer::conv1d(40, 8, 5, 2, 2, &mut rng), Layer::Relu, Layer::MeanPool, Layer::linear(8, 35, &mut rng)]);
        let classifier = WordClassifier::new(model);
        let samples: Vec<f32> = (0..9000).map(|_| 0.1 * rng.normal()).collect();
        let top = classifier.top_k(&samples, 5);
        assert_eq!(top.len(), 5);
        assert!(top.windows(2).all(|pair| pair[0].1 >= pair[1].1));
        assert!(top.iter().map(|&(_, p)| p).sum::<f32>() <= 1.0 + 1e-5);
        assert!((classifier.probabilities(&samples).iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert_eq!(WordClassifier::fit_one_second(&vec![0.1; 40000]).len(), CLIP_SAMPLES);
        assert_eq!(WordClassifier::fit_one_second(&[0.1; 10]).len(), CLIP_SAMPLES);
    }

    #[test]
    fn a_model_of_the_wrong_shape_is_refused() {
        let mut rng = Rng::new(5);
        let path = std::env::temp_dir().join(format!("pipeline_shape_test_{}.bin", std::process::id()));
        Model::new(vec![Layer::linear(13, 10, &mut rng)]).save(&path).unwrap();
        let error = Recognizer::load(&path).unwrap_err();
        assert!(error.to_string().contains("40 inputs and 28 outputs"), "{error}");
        tiny_recognizer().model.save(&path).unwrap();
        assert!(Recognizer::load(&path).is_ok());
        assert!(WordClassifier::load(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
