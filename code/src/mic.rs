//! Live audio without an audio library.
//!
//! Talking to a sound card is different on every operating system (Core
//! Audio, ALSA, PulseAudio, WASAPI...), and each of those APIs is a book of
//! its own. We sidestep all of them with one observation: `ffmpeg` already
//! knows how to open a microphone everywhere, and it can write what it hears
//! to its standard output as raw numbers. So we start `ffmpeg` as a **child
//! process** and read its output through a **pipe**:
//!
//! ```text
//! microphone ─▶ ffmpeg ──(pipe: 16-bit samples, 16,000 per second)──▶ our program
//! ```
//!
//! The format we ask for is the simplest there is, "s16le": every sample is
//! two bytes, a signed 16-bit integer, low byte first, no header, nothing
//! else. One second is 32,000 bytes.
//!
//! Three things in this module:
//!
//! * [`AudioSource`]: anything that yields audio a chunk at a time. The
//!   microphone ([`Microphone`]), a WAV file played back in chunks
//!   ([`WavPlayback`]), or raw PCM arriving on standard input
//!   ([`PcmStream`]). Because the rest of the program only sees the trait,
//!   the whole live path can be tested without a microphone.
//! * [`Segmenter`]: cuts the endless stream into utterances using the
//!   streaming endpoint detector of chapter 2.
//! * [`listen`]: the loop that connects the two.
//!
//! # The microphone permission on macOS
//!
//! The first time a program started from your terminal opens the microphone,
//! macOS asks whether the *terminal application* may use it. If the answer
//! was no (or the question never appeared), `ffmpeg` fails or records
//! silence. Allow it under System Settings → Privacy & Security →
//! Microphone, then restart the terminal.

use crate::dsp::frame::{EndpointConfig, EndpointEvent, Endpointer};
use crate::wav::Wav;
use std::io::{self, Read};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

/// The sample rate of everything in this module.
pub const SAMPLE_RATE: u32 = 16_000;

/// What to tell the user when the microphone cannot be opened.
pub const MICROPHONE_HELP: &str = "The microphone is read by running `ffmpeg`, so ffmpeg must be installed and on your PATH
(macOS: `brew install ffmpeg`; Debian/Ubuntu: `sudo apt install ffmpeg`).
On macOS the terminal application also needs permission to use the microphone:
System Settings -> Privacy & Security -> Microphone, enable your terminal, then restart it.
To try the program without a microphone, use `--from-file some.wav`.";

/// Something that produces 16 kHz mono audio a chunk at a time.
pub trait AudioSource {
    /// Wait until some audio is available and append it to `out`. Returns
    /// the number of samples appended; 0 means the stream has ended.
    fn read(&mut self, out: &mut Vec<f32>) -> io::Result<usize>;
}

// ---------------------------------------------------------------------------
// Raw PCM bytes to samples
// ---------------------------------------------------------------------------

/// Turns a stream of s16le bytes into samples.
///
/// A pipe delivers bytes in whatever portions it likes, so a read can end in
/// the middle of a sample. The decoder remembers such a leftover byte and
/// joins it with the first byte of the next portion.
#[derive(Clone, Debug, Default)]
pub struct PcmDecoder {
    leftover: Option<u8>,
}

impl PcmDecoder {
    /// A decoder with no pending byte.
    pub fn new() -> Self {
        PcmDecoder::default()
    }

    /// Decode `bytes`, appending samples in `[-1, 1)` to `out`. Returns how
    /// many samples were appended.
    pub fn push(&mut self, bytes: &[u8], out: &mut Vec<f32>) -> usize {
        let before = out.len();
        let mut bytes = bytes;
        if let (Some(low), Some((&high, rest))) = (self.leftover, bytes.split_first()) {
            out.push(i16::from_le_bytes([low, high]) as f32 / 32768.0);
            self.leftover = None;
            bytes = rest;
        }
        let (pairs, remainder) = bytes.as_chunks::<2>();
        if let [last] = remainder {
            self.leftover = Some(*last);
        }
        out.extend(pairs.iter().map(|&pair| i16::from_le_bytes(pair) as f32 / 32768.0));
        out.len() - before
    }
}

/// Raw s16le audio from any byte stream: a pipe, standard input, a file.
#[derive(Debug)]
pub struct PcmStream<R: Read> {
    reader: R,
    decoder: PcmDecoder,
    bytes: Vec<u8>,
}

impl<R: Read> PcmStream<R> {
    /// Read s16le samples from `reader`, at most `chunk_samples` per call.
    pub fn new(reader: R, chunk_samples: usize) -> Self {
        PcmStream { reader, decoder: PcmDecoder::new(), bytes: vec![0; 2 * chunk_samples.max(1)] }
    }
}

impl<R: Read> AudioSource for PcmStream<R> {
    fn read(&mut self, out: &mut Vec<f32>) -> io::Result<usize> {
        loop {
            let n = match self.reader.read(&mut self.bytes) {
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            if n == 0 {
                return Ok(0);
            }
            let decoded = self.decoder.push(&self.bytes[..n], out);
            // A single odd byte decodes to nothing; that is not the end.
            if decoded > 0 {
                return Ok(decoded);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A WAV file pretending to be a microphone
// ---------------------------------------------------------------------------

/// Plays a recording back in chunks, optionally at the speed a microphone
/// would deliver it.
#[derive(Clone, Debug)]
pub struct WavPlayback {
    samples: Vec<f32>,
    position: usize,
    chunk: usize,
    real_time: bool,
    started: Option<Instant>,
}

impl WavPlayback {
    /// Play `samples` (16 kHz mono) in chunks of `chunk_samples`. With
    /// `real_time`, each chunk is held back until the moment a live source
    /// would have produced it.
    pub fn new(samples: Vec<f32>, chunk_samples: usize, real_time: bool) -> Self {
        WavPlayback { samples, position: 0, chunk: chunk_samples.max(1), real_time, started: None }
    }

    /// Open a WAV file (any sample rate; it is resampled to 16 kHz).
    pub fn open(path: impl AsRef<Path>, chunk_samples: usize, real_time: bool) -> io::Result<Self> {
        let path = path.as_ref();
        let wav = Wav::read(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        Ok(WavPlayback::new(crate::dsp::resample::resample(&wav.samples, wav.sample_rate, SAMPLE_RATE), chunk_samples, real_time))
    }
}

impl AudioSource for WavPlayback {
    fn read(&mut self, out: &mut Vec<f32>) -> io::Result<usize> {
        let end = (self.position + self.chunk).min(self.samples.len());
        if self.real_time && end > self.position {
            // The chunk ending at sample `end` exists `end / 16000` seconds
            // after the recording began.
            let started = *self.started.get_or_insert_with(Instant::now);
            let due = Duration::from_secs_f64(end as f64 / SAMPLE_RATE as f64);
            if let Some(wait) = due.checked_sub(started.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        out.extend_from_slice(&self.samples[self.position..end]);
        let n = end - self.position;
        self.position = end;
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// The microphone: ffmpeg as a child process
// ---------------------------------------------------------------------------

/// The arguments that make `ffmpeg` read the default microphone on this
/// operating system.
///
/// * macOS: the AVFoundation framework; `":default"` means "no video
///   device, the default audio device".
/// * Linux: PulseAudio (which PipeWire also answers to). If your system has
///   neither, try `["-f", "alsa", "-i", "default"]` through
///   [`Microphone::with_input_args`].
/// * Windows: DirectShow needs the device's name; `ffmpeg -list_devices true
///   -f dshow -i dummy` prints the names. We try the generic
///   `audio=default`, which works on some systems only.
pub fn default_input_args() -> Vec<String> {
    let args: &[&str] = if cfg!(target_os = "macos") {
        &["-f", "avfoundation", "-i", ":default"]
    } else if cfg!(target_os = "windows") {
        &["-f", "dshow", "-i", "audio=default"]
    } else {
        &["-f", "pulse", "-i", "default"]
    };
    args.iter().map(|s| s.to_string()).collect()
}

/// The full `ffmpeg` command line: quiet, the given input, and raw 16 kHz
/// mono s16le on standard output (`-`).
pub fn ffmpeg_arguments(input_args: &[String]) -> Vec<String> {
    let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error", "-nostdin"].iter().map(|s| s.to_string()).collect();
    args.extend(input_args.iter().cloned());
    args.extend(["-ar", "16000", "-ac", "1", "-f", "s16le", "-"].iter().map(|s| s.to_string()));
    args
}

/// How much of a child's standard error we remember for an error message.
const STDERR_TAIL_BYTES: usize = 4096;

/// How long we wait for the last of a child's standard error after the
/// child itself has exited.
const STDERR_PATIENCE: Duration = Duration::from_secs(2);

/// Read `stderr` to its end on a thread of its own, keeping only the last
/// [`STDERR_TAIL_BYTES`]. The receiver gets that tail, as text, once the
/// stream has closed.
///
/// A pipe holds a limited number of bytes (64 KB is typical). A child that
/// writes more than that to a pipe nobody reads is put to sleep in the
/// middle of its write and never gets to produce audio or to exit. So
/// somebody has to keep reading stderr for as long as the child lives, and
/// the thread that reads the audio cannot do it: it is busy waiting on the
/// other pipe.
fn drain_stderr(mut stderr: impl Read + Send + 'static) -> Receiver<String> {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut tail = Vec::new();
        let mut buffer = [0u8; 4096];
        loop {
            match stderr.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    tail.extend_from_slice(&buffer[..n]);
                    if tail.len() > STDERR_TAIL_BYTES {
                        tail.drain(..tail.len() - STDERR_TAIL_BYTES);
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        // Nobody is listening any more if the `Microphone` was dropped.
        let _ = sender.send(String::from_utf8_lossy(&tail).into_owned());
    });
    receiver
}

/// A running `ffmpeg` whose output we read as audio.
///
/// Dropping it stops the recording: the child is killed and waited for, so
/// no orphaned `ffmpeg` keeps the microphone open after we are gone.
#[derive(Debug)]
pub struct Microphone {
    child: Child,
    stream: PcmStream<ChildStdout>,
    /// Delivers the end of the child's stderr once that stream has closed.
    complaint: Receiver<String>,
}

impl Microphone {
    /// Open the default microphone.
    pub fn open(chunk_samples: usize) -> io::Result<Microphone> {
        Microphone::with_input_args(&default_input_args(), chunk_samples)
    }

    /// Start `ffmpeg` with your own input arguments, for another device
    /// (`["-f", "alsa", "-i", "hw:1"]`) or, for testing the whole pipe
    /// mechanism without a microphone, a file read at real-time speed
    /// (`["-re", "-i", "recording.wav"]`).
    pub fn with_input_args(input_args: &[String], chunk_samples: usize) -> io::Result<Microphone> {
        Microphone::spawn(Command::new("ffmpeg").args(ffmpeg_arguments(input_args)), chunk_samples)
    }

    /// Start any command that writes s16le audio to its standard output.
    pub fn spawn(command: &mut Command, chunk_samples: usize) -> io::Result<Microphone> {
        // stdout is the audio; stderr is a second pipe, so we can show
        // ffmpeg's own error message if it gives up; stdin is closed so the
        // child never waits for (or steals) our keyboard input.
        let mut child = command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| {
            let program = command.get_program().to_string_lossy().into_owned();
            if e.kind() == io::ErrorKind::NotFound {
                io::Error::new(io::ErrorKind::NotFound, format!("could not start `{program}`: it is not installed or not on your PATH.\n{MICROPHONE_HELP}"))
            } else {
                io::Error::new(e.kind(), format!("could not start `{program}`: {e}"))
            }
        })?;
        let stdout = child.stdout.take().expect("stdout was requested as a pipe");
        let stderr = child.stderr.take().expect("stderr was requested as a pipe");
        // Two pipes need two readers: we read stdout, a thread reads stderr.
        Ok(Microphone { child, stream: PcmStream::new(stdout, chunk_samples), complaint: drain_stderr(stderr) })
    }

    /// Called when the pipe has closed: find out whether the child ended
    /// normally (a file source reached its end) or failed (no device, no
    /// permission), and in the second case turn the end of its stderr into
    /// our error.
    fn explain_end(&mut self) -> io::Result<usize> {
        let status = self.child.wait()?;
        if status.success() {
            return Ok(0);
        }
        // The child is gone, so its stderr closes and the tail arrives at
        // once. The time limit is for the odd case of a grandchild that
        // inherited the pipe and still holds it open.
        let complaint = self.complaint.recv_timeout(STDERR_PATIENCE).unwrap_or_default();
        Err(io::Error::other(format!("the recording process stopped ({status}): {}\n{MICROPHONE_HELP}", complaint.trim())))
    }
}

impl AudioSource for Microphone {
    fn read(&mut self, out: &mut Vec<f32>) -> io::Result<usize> {
        match self.stream.read(out)? {
            0 => self.explain_end(),
            n => Ok(n),
        }
    }
}

impl Drop for Microphone {
    fn drop(&mut self) {
        // Both calls fail harmlessly if the child has already exited. The
        // stderr thread is not joined: it ends by itself when the dead
        // child's pipe closes.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ---------------------------------------------------------------------------
// Cutting the stream into utterances
// ---------------------------------------------------------------------------

/// Settings of the [`Segmenter`].
#[derive(Clone, Debug, PartialEq)]
pub struct SegmenterConfig {
    /// The endpoint detector's settings. The default here waits for 600 ms
    /// of quiet before it calls an utterance finished (chapter 2 used
    /// 200 ms for single words): people pause between words, and we want
    /// "seven... three" to arrive as one utterance.
    pub endpoint: EndpointConfig,
    /// Audio kept from before the detected start of speech, in
    /// milliseconds. The detector fires a little late by design; the soft
    /// beginning of the word is in this pre-roll.
    pub pre_roll_ms: f32,
    /// Audio kept after the detected end, in milliseconds.
    pub post_roll_ms: f32,
    /// An utterance longer than this is cut off and recognized anyway, so
    /// that steady noise (or a monologue) cannot make us buffer forever.
    pub max_utterance_s: f32,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        SegmenterConfig { endpoint: EndpointConfig { hangover_ms: 600.0, ..EndpointConfig::default() }, pre_roll_ms: 150.0, post_roll_ms: 150.0, max_utterance_s: 10.0 }
    }
}

/// One utterance cut out of the stream.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// Position of the first sample, counted from the start of the stream.
    pub start: usize,
    /// Position one past the last sample.
    pub end: usize,
    /// The audio.
    pub samples: Vec<f32>,
    /// True if the utterance was cut because it reached the maximum length
    /// rather than because the speaker stopped.
    pub cut_short: bool,
}

/// Turns a stream of audio chunks into a stream of utterances.
///
/// Chunks of any size go in. Internally they are regrouped into the 10 ms
/// blocks the [`Endpointer`] wants. While nobody speaks we keep only the
/// most recent stretch of audio (for the pre-roll); from the start of
/// speech we keep everything until the detector says it is over.
#[derive(Clone, Debug)]
pub struct Segmenter {
    config: SegmenterConfig,
    endpointer: Endpointer,
    /// Audio we still hold. `buffer[0]` is sample number `buffer_start` of
    /// the stream.
    buffer: Vec<f32>,
    buffer_start: usize,
    /// How many samples of the stream the endpointer has been given.
    fed: usize,
    /// Where the current utterance began (already including pre-roll), if
    /// we are inside one.
    speech_start: Option<usize>,
    /// Level of the most recent block in dB, for a level meter.
    last_level_db: f32,
}

impl Segmenter {
    /// A segmenter for a 16 kHz stream.
    pub fn new(config: SegmenterConfig) -> Self {
        let endpointer = Endpointer::new(SAMPLE_RATE, config.endpoint.clone());
        Segmenter { config, endpointer, buffer: Vec::new(), buffer_start: 0, fed: 0, speech_start: None, last_level_db: -100.0 }
    }

    /// Whether we are in the middle of an utterance.
    pub fn in_speech(&self) -> bool {
        self.speech_start.is_some()
    }

    /// Loudness of the latest 10 ms block in dB (0 dB is full scale).
    pub fn level_db(&self) -> f32 {
        self.last_level_db
    }

    /// Total number of samples received so far.
    pub fn samples_seen(&self) -> usize {
        self.buffer_start + self.buffer.len()
    }

    fn ms(&self, ms: f32) -> usize {
        (ms * SAMPLE_RATE as f32 / 1000.0).round() as usize
    }

    /// Copy the stream samples `start..end` out of the buffer.
    fn cut(&self, start: usize, end: usize, cut_short: bool) -> Segment {
        let end = end.min(self.samples_seen());
        let start = start.max(self.buffer_start).min(end);
        Segment { start, end, samples: self.buffer[start - self.buffer_start..end - self.buffer_start].to_vec(), cut_short }
    }

    /// Feed the next chunk of audio. Returns the utterances that ended
    /// inside it (almost always none or one).
    pub fn push(&mut self, chunk: &[f32]) -> Vec<Segment> {
        self.buffer.extend_from_slice(chunk);
        let block_len = self.endpointer.block_len();
        let mut finished = Vec::new();

        // Hand every complete 10 ms block we have not yet looked at to the
        // endpoint detector.
        while self.fed + block_len <= self.samples_seen() {
            let offset = self.fed - self.buffer_start;
            let block = &self.buffer[offset..offset + block_len];
            let power = block.iter().map(|s| s * s).sum::<f32>() / block_len as f32;
            self.last_level_db = 10.0 * (power + 1e-10).log10();
            let event = self.endpointer.push(block);
            self.fed += block_len;

            match event {
                Some(EndpointEvent::SpeechStart { sample }) => {
                    self.speech_start = Some(sample.saturating_sub(self.ms(self.config.pre_roll_ms)).max(self.buffer_start));
                }
                Some(EndpointEvent::SpeechEnd { sample }) => {
                    if let Some(start) = self.speech_start.take() {
                        finished.push(self.cut(start, sample + self.ms(self.config.post_roll_ms), false));
                    }
                }
                None => {}
            }
            // Too long? Hand over what we have and start a new utterance
            // right here; the detector stays in its "speech" state.
            if let Some(start) = self.speech_start {
                if self.fed - start >= (self.config.max_utterance_s * SAMPLE_RATE as f32) as usize {
                    finished.push(self.cut(start, self.fed, true));
                    self.speech_start = Some(self.fed);
                }
            }
        }

        // Forget audio we can no longer need: everything before the current
        // utterance, or, in silence, everything but the last second.
        let keep_from = self.speech_start.unwrap_or_else(|| self.fed.saturating_sub(SAMPLE_RATE as usize));
        if keep_from > self.buffer_start {
            self.buffer.drain(..keep_from - self.buffer_start);
            self.buffer_start = keep_from;
        }
        finished
    }

    /// The stream is over. If someone was still speaking, return what we
    /// have of that last utterance.
    pub fn finish(&mut self) -> Option<Segment> {
        let event = self.endpointer.finish();
        let start = self.speech_start.take()?;
        let end = match event {
            Some(EndpointEvent::SpeechEnd { sample }) => sample + self.ms(self.config.post_roll_ms),
            _ => self.samples_seen(),
        };
        Some(self.cut(start, end, false))
    }
}

/// What [`listen`] reports while it runs.
#[derive(Debug)]
pub enum ListenEvent<'a> {
    /// A chunk was processed: the current input level and whether we are
    /// inside an utterance. For drawing a level meter.
    Level {
        /// Loudness of the latest block in dB (0 dB is full scale).
        db: f32,
        /// Whether speech is in progress.
        in_speech: bool,
    },
    /// An utterance is complete.
    Utterance(&'a Segment),
}

/// The live loop: read chunks from `source`, cut them into utterances, and
/// call `on_event` for each. Runs until the source ends (for a microphone:
/// until the program is interrupted with Ctrl-C, which also stops the
/// `ffmpeg` child because both belong to the terminal's foreground job).
/// Returns the number of samples heard.
pub fn listen(source: &mut dyn AudioSource, config: SegmenterConfig, mut on_event: impl FnMut(ListenEvent)) -> io::Result<usize> {
    let mut segmenter = Segmenter::new(config);
    let mut chunk = Vec::new();
    loop {
        chunk.clear();
        if source.read(&mut chunk)? == 0 {
            break;
        }
        for segment in segmenter.push(&chunk) {
            on_event(ListenEvent::Utterance(&segment));
        }
        on_event(ListenEvent::Level { db: segmenter.level_db(), in_speech: segmenter.in_speech() });
    }
    if let Some(segment) = segmenter.finish() {
        on_event(ListenEvent::Utterance(&segment));
    }
    Ok(segmenter.samples_seen())
}

/// A one-line level meter like `[######......] -23 dB *`, `width`
/// characters of bar covering -60 dB to 0 dB; the star means "speech".
pub fn level_meter(db: f32, in_speech: bool, width: usize) -> String {
    let filled = (((db + 60.0) / 60.0).clamp(0.0, 1.0) * width as f32).round() as usize;
    format!("[{}{}] {:>4.0} dB {}", "#".repeat(filled), ".".repeat(width - filled), db.max(-99.0), if in_speech { "*" } else { " " })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn to_bytes(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn pcm_decoder_handles_samples_split_across_reads() {
        let samples: Vec<i16> = vec![0, 1, -1, 16384, -32768, 32767, 258];
        let bytes = to_bytes(&samples);
        let expected: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        // Whatever the portion size, the same samples come out.
        for portion in [1, 2, 3, 5, 14] {
            let (mut decoder, mut out) = (PcmDecoder::new(), Vec::new());
            let total: usize = bytes.chunks(portion).map(|part| decoder.push(part, &mut out)).sum();
            assert_eq!(out, expected, "portion size {portion}");
            assert_eq!(total, samples.len());
        }
        // Low byte first: 0x0102 = 258 is stored as 02 01.
        assert_eq!(&bytes[12..], &[2, 1]);
        assert_eq!(expected[3], 0.5);
    }

    /// A reader that hands out its bytes in portions of a fixed odd size.
    struct Dribble {
        bytes: Vec<u8>,
        position: usize,
        portion: usize,
    }

    impl Read for Dribble {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = self.portion.min(buf.len()).min(self.bytes.len() - self.position);
            buf[..n].copy_from_slice(&self.bytes[self.position..self.position + n]);
            self.position += n;
            Ok(n)
        }
    }

    #[test]
    fn pcm_stream_reads_everything_then_reports_the_end() {
        let samples: Vec<i16> = (0..1001).map(|i| (i * 31 % 2000 - 1000) as i16).collect();
        for portion in [1, 7, 4096] {
            let mut stream = PcmStream::new(Dribble { bytes: to_bytes(&samples), position: 0, portion }, 64);
            let mut out = Vec::new();
            while stream.read(&mut out).unwrap() > 0 {}
            assert_eq!(out.len(), samples.len(), "portion {portion}");
            assert!(out.iter().zip(&samples).all(|(&f, &s)| f == s as f32 / 32768.0));
            assert_eq!(stream.read(&mut out).unwrap(), 0);
        }
    }

    #[test]
    fn wav_playback_delivers_every_sample_once_in_chunks() {
        let samples: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let mut playback = WavPlayback::new(samples.clone(), 160, false);
        let (mut out, mut sizes) = (Vec::new(), Vec::new());
        loop {
            let n = playback.read(&mut out).unwrap();
            if n == 0 {
                break;
            }
            sizes.push(n);
        }
        assert_eq!(out, samples);
        assert_eq!(sizes, vec![160, 160, 160, 160, 160, 160, 40]);
    }

    #[test]
    fn wav_playback_in_real_time_takes_as_long_as_the_audio() {
        // 0.2 s of audio in 20 ms chunks must take about 0.2 s to deliver.
        let mut playback = WavPlayback::new(vec![0.0; 3200], 320, true);
        let (started, mut out) = (Instant::now(), Vec::new());
        while playback.read(&mut out).unwrap() > 0 {}
        let took = started.elapsed().as_secs_f32();
        assert_eq!(out.len(), 3200);
        assert!((0.17..0.6).contains(&took), "took {took} s");
    }

    /// Faint noise with loud bursts at the given `(start, end)` positions.
    fn stream_with_bursts(len: usize, bursts: &[(usize, usize)]) -> Vec<f32> {
        let mut rng = Rng::new(5);
        let mut samples: Vec<f32> = (0..len).map(|_| 0.002 * (rng.uniform() - 0.5)).collect();
        for &(start, end) in bursts {
            for (i, s) in samples[start..end].iter_mut().enumerate() {
                *s += 0.3 * (i as f32 * 0.3).sin();
            }
        }
        samples
    }

    #[test]
    fn segmenter_finds_two_utterances_whatever_the_chunk_size() {
        // Two utterances; the first has a 300 ms pause inside, which must
        // NOT split it (the hangover is 600 ms).
        let bursts = [(16000, 24000), (28800, 36000), (64000, 76000)];
        let stream = stream_with_bursts(100_000, &bursts);
        let mut reference: Option<Vec<(usize, usize)>> = None;
        for chunk_len in [160, 1024, 333, 100_000] {
            let mut segmenter = Segmenter::new(SegmenterConfig::default());
            let mut segments = Vec::new();
            for chunk in stream.chunks(chunk_len) {
                segments.extend(segmenter.push(chunk));
            }
            segments.extend(segmenter.finish());
            assert_eq!(segments.len(), 2, "chunk {chunk_len}");
            // Starts: the burst's start minus 30 ms padding minus 150 ms pre-roll.
            let (a, b) = (&segments[0], &segments[1]);
            assert!((16000 - 3200..=16000 - 2400).contains(&a.start), "first start {}", a.start);
            assert!((36000 + 2400..=36000 + 3400).contains(&a.end), "first end {}", a.end);
            assert!((64000 - 3200..=64000 - 2400).contains(&b.start), "second start {}", b.start);
            assert!((76000 + 2400..=76000 + 3400).contains(&b.end), "second end {}", b.end);
            for segment in &segments {
                assert_eq!(segment.samples, &stream[segment.start..segment.end], "the audio must be the stream's own samples");
                assert!(!segment.cut_short);
            }
            // The cut points do not depend on how the stream was chunked.
            let bounds: Vec<(usize, usize)> = segments.iter().map(|s| (s.start, s.end)).collect();
            assert_eq!(*reference.get_or_insert(bounds.clone()), bounds, "chunk {chunk_len}");
            assert_eq!(segmenter.samples_seen(), stream.len());
        }
    }

    #[test]
    fn segmenter_closes_an_utterance_when_the_stream_ends() {
        let stream = stream_with_bursts(40_000, &[(16000, 40_000)]);
        let mut segmenter = Segmenter::new(SegmenterConfig::default());
        assert!(segmenter.push(&stream).is_empty());
        assert!(segmenter.in_speech());
        let last = segmenter.finish().expect("the unfinished utterance");
        assert_eq!(last.end, 40_000);
        assert!(last.start < 16000);
        assert!(!segmenter.in_speech());
    }

    #[test]
    fn segmenter_cuts_an_endless_utterance() {
        let stream = stream_with_bursts(80_000, &[(8000, 80_000)]);
        let config = SegmenterConfig { max_utterance_s: 2.0, ..SegmenterConfig::default() };
        let mut segmenter = Segmenter::new(config);
        let mut segments = Vec::new();
        for chunk in stream.chunks(512) {
            segments.extend(segmenter.push(chunk));
        }
        assert!(segments.len() >= 2);
        assert!(segments.iter().all(|s| s.cut_short && s.samples.len() == 32000));
        // Consecutive pieces touch: nothing lost, nothing repeated.
        assert_eq!(segments[0].end, segments[1].start);
    }

    #[test]
    fn segmenter_does_not_hoard_silence() {
        let mut segmenter = Segmenter::new(SegmenterConfig::default());
        for _ in 0..200 {
            segmenter.push(&[0.0; 1600]);
        }
        assert!(segmenter.buffer.len() <= 16000 + 1600, "kept {} samples of silence", segmenter.buffer.len());
        assert_eq!(segmenter.samples_seen(), 320_000);
    }

    #[test]
    fn listen_reports_levels_and_utterances() {
        let stream = stream_with_bursts(60_000, &[(16000, 30000)]);
        let mut source = WavPlayback::new(stream, 800, false);
        let (mut utterances, mut levels, mut loudest) = (0, 0, -100.0f32);
        let heard = listen(&mut source, SegmenterConfig::default(), |event| match event {
            ListenEvent::Utterance(segment) => {
                utterances += 1;
                assert!(segment.samples.len() > 14000);
            }
            ListenEvent::Level { db, .. } => {
                levels += 1;
                loudest = loudest.max(db);
            }
        })
        .unwrap();
        assert_eq!((heard, utterances, levels), (60_000, 1, 75));
        // A sine of amplitude 0.3 has power 0.045, which is -13.5 dB.
        assert!((loudest + 13.5).abs() < 0.5, "loudest {loudest}");
    }

    #[test]
    fn level_meter_draws_a_bar() {
        assert_eq!(level_meter(-30.0, false, 10), "[#####.....]  -30 dB  ");
        assert_eq!(level_meter(0.0, true, 4), "[####]    0 dB *");
        assert_eq!(level_meter(-80.0, false, 4), "[....]  -80 dB  ");
    }

    #[test]
    fn ffmpeg_arguments_ask_for_raw_16k_mono() {
        let args = ffmpeg_arguments(&["-f".to_string(), "pulse".to_string(), "-i".to_string(), "default".to_string()]);
        assert_eq!(args.join(" "), "-hide_banner -loglevel error -nostdin -f pulse -i default -ar 16000 -ac 1 -f s16le -");
        assert_eq!(default_input_args().len(), 4);
    }

    #[test]
    fn a_missing_program_gives_a_helpful_error() {
        let error = Microphone::spawn(&mut Command::new("surely-there-is-no-such-program-here"), 160).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(error.to_string().contains("not installed"), "{error}");
        assert!(error.to_string().contains("Microphone"), "{error}");
    }

    /// The child-process path, with `cat` standing in for `ffmpeg`: a
    /// process writes s16le bytes to a pipe, we read samples.
    #[cfg(unix)]
    #[test]
    fn a_child_process_is_read_through_a_pipe_and_failure_is_reported() {
        let samples: Vec<i16> = (0..5000).map(|i| (i % 300 - 150) as i16 * 50).collect();
        let path = std::env::temp_dir().join(format!("mic_test_{}.raw", std::process::id()));
        std::fs::write(&path, to_bytes(&samples)).unwrap();

        let mut microphone = Microphone::spawn(Command::new("cat").arg(&path), 256).unwrap();
        let mut out = Vec::new();
        while microphone.read(&mut out).unwrap() > 0 {}
        assert_eq!(out.len(), samples.len());
        assert!(out.iter().zip(&samples).all(|(&f, &s)| f == s as f32 / 32768.0));
        drop(microphone);
        std::fs::remove_file(&path).unwrap();

        // A child that fails: its stderr ends up in our error message.
        let mut broken = Microphone::spawn(Command::new("cat").arg("/no/such/file/anywhere"), 256).unwrap();
        let error = broken.read(&mut out).unwrap_err();
        assert!(error.to_string().contains("No such file"), "{error}");

        // Dropping a child that would run forever must not hang.
        let endless = Microphone::spawn(Command::new("sleep").arg("1000"), 256).unwrap();
        let started = Instant::now();
        drop(endless);
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// Run `scenario` on a thread and fail, instead of hanging the whole
    /// test run, if it has not finished after ten seconds.
    #[cfg(unix)]
    fn within_ten_seconds<T: Send + 'static>(scenario: impl FnOnce() -> T + Send + 'static) -> T {
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || sender.send(scenario()));
        receiver.recv_timeout(Duration::from_secs(10)).expect("the child process handling hung (or panicked)")
    }

    /// A megabyte of `x` on stderr: sixteen times what a pipe holds.
    #[cfg(unix)]
    const FLOOD: &str = "head -c 1048576 /dev/zero | tr '\\0' x >&2";

    #[cfg(unix)]
    #[test]
    fn a_child_that_floods_stderr_cannot_block_the_audio() {
        // The flood comes first, then one sample (258 = 02 01), then a last
        // line and a failing exit. Unless somebody reads stderr while we
        // wait for audio, the child sleeps in the flood forever.
        let (out, error) = within_ten_seconds(|| {
            let script = format!("{FLOOD}; printf '\\002\\001'; echo 'the last words' >&2; exit 3");
            let mut microphone = Microphone::spawn(Command::new("sh").args(["-c", &script]), 256).unwrap();
            let mut out = Vec::new();
            assert_eq!(microphone.read(&mut out).unwrap(), 1);
            (out, microphone.read(&mut Vec::new()).unwrap_err().to_string())
        });
        assert_eq!(out, vec![258.0 / 32768.0]);
        // The error carries the end of the complaint, not the megabyte.
        assert!(error.contains("exit status: 3"), "{}", &error[..error.len().min(200)]);
        assert!(error.contains("xxxxthe last words\n"), "the tail is missing");
        assert!(error.len() < STDERR_TAIL_BYTES + 1000, "{} bytes of error message", error.len());

        // A child that is merely talkative and then succeeds is a normal end.
        let samples = within_ten_seconds(|| {
            let script = format!("printf '\\002\\001'; {FLOOD}; printf '\\002\\001'");
            let mut microphone = Microphone::spawn(Command::new("sh").args(["-c", &script]), 256).unwrap();
            let mut out = Vec::new();
            while microphone.read(&mut out).unwrap() > 0 {}
            out
        });
        assert_eq!(samples, vec![258.0 / 32768.0; 2]);
    }

    #[cfg(unix)]
    #[test]
    fn dropping_a_child_that_floods_stderr_ends_the_child_and_the_thread() {
        // `yes` writes to stderr without end and never produces audio.
        let tail = within_ten_seconds(|| {
            let mut endless = Microphone::spawn(Command::new("sh").args(["-c", "exec yes >&2"]), 256).unwrap();
            // Keep the thread's result so we can see it finish after the drop.
            let complaint = std::mem::replace(&mut endless.complaint, mpsc::channel().1);
            std::thread::sleep(Duration::from_millis(100));
            drop(endless);
            complaint.recv_timeout(Duration::from_secs(5)).expect("the stderr thread must end once the child is dead")
        });
        // However long `yes` ran, only the tail was kept.
        assert_eq!(tail.len(), STDERR_TAIL_BYTES);
        assert!(tail.contains("y\ny\n"));
    }

    #[test]
    fn drain_stderr_keeps_only_the_tail() {
        let short = drain_stderr(io::Cursor::new(b"no such device\n".to_vec()));
        assert_eq!(short.recv().unwrap(), "no such device\n");
        let mut long = vec![b'a'; 3 * STDERR_TAIL_BYTES];
        long.extend_from_slice(b"the end");
        let tail = drain_stderr(io::Cursor::new(long)).recv().unwrap();
        assert_eq!(tail.len(), STDERR_TAIL_BYTES);
        assert!(tail.starts_with("aaa") && tail.ends_with("aaathe end"));
    }
}
