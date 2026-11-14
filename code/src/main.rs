//! `speech`: the recognizer of this book as a command-line tool.
//!
//! ```text
//! speech transcribe ../samples/utt_seven_three_go_left.wav
//! speech listen
//! speech words ../samples/seven.wav
//! ```
//!
//! Everything interesting happens in the library; this file parses
//! arguments, wires the pieces together and prints.

use speech::data::pipeline::{load_audio_16k, Decoder, Recognizer, WordClassifier};
use speech::mic::{level_meter, listen, AudioSource, ListenEvent, Microphone, PcmStream, SegmenterConfig, WavPlayback, MICROPHONE_HELP, SAMPLE_RATE};
use speech::whisper::{DecodeOptions, Whisper};
use std::io::{IsTerminal, Write};
use std::process::ExitCode;
use std::time::Instant;

const HELP: &str = "speech: a speech recognizer built from scratch

USAGE
    speech transcribe <file.wav>... [options]    write down what is said in each file
    speech listen [options]                      recognize from the microphone until Ctrl-C
    speech words <file.wav>... [--model PATH]    chapter 7's classifier: which of 35 words is this?
    speech help                                  show this text

OPTIONS FOR transcribe AND listen
    --decoder greedy|beam|lexicon   how the network's output becomes text (default: lexicon)
                                      greedy   best letter per frame; shows what the network really spells
                                      beam     prefix beam search over spellings
                                      lexicon  beam search that may only spell the 35 words it was trained on
    --beam N                        beam width for beam and lexicon (default: 16)
    --model PATH                    model file (default: ../models/recognizer.bin, or $SPEECH_MODELS/recognizer.bin)
    --whisper                       use the pretrained Whisper tiny.en of chapter 12 instead of our model
                                    (weights in $SPEECH_DATA/whisper-tiny.en; any English, much slower)

OPTIONS FOR listen
    --from-file FILE.wav            play a recording into the live pipeline instead of the microphone
    --fast                          with --from-file: as fast as possible instead of in real time
    --stdin                         read raw audio from standard input (16 kHz, mono, signed 16-bit little-endian)
    --ffmpeg-input \"ARGS\"           replace ffmpeg's input arguments, e.g. \"-f alsa -i hw:1\" for another
                                    device, or \"-re -i FILE.wav\" to test the ffmpeg pipe with a file
    --quiet                         no level meter, only the recognized text

Our own model knows 35 words (the Speech Commands vocabulary: the digits zero to nine, yes, no, up,
down, left, right, on, off, stop, go, backward, forward, follow, learn, visual, bed, bird, cat, dog,
happy, house, marvin, sheila, tree, wow), spoken one after another with short pauses.

The microphone is read through ffmpeg, which must be installed. On macOS the terminal needs
microphone permission (System Settings -> Privacy & Security -> Microphone).";

/// Whichever model does the recognizing.
enum Engine {
    /// The CTC model we trained in chapter 9, with a decoder from chapter 10.
    Ours(Box<Recognizer>, Decoder),
    /// The pretrained Whisper tiny.en model of chapter 12.
    Whisper(Box<Whisper>),
}

impl Engine {
    fn transcribe(&self, samples: &[f32]) -> String {
        match self {
            Engine::Ours(recognizer, decoder) => recognizer.transcribe(samples, decoder),
            Engine::Whisper(whisper) => whisper.transcribe(samples, &DecodeOptions::default()).text.trim().to_string(),
        }
    }

    fn describe(&self) -> String {
        match self {
            Engine::Ours(recognizer, decoder) => format!("our model ({} weights), {} decoder", recognizer.model.num_params(), decoder.name()),
            Engine::Whisper(_) => "Whisper tiny.en".to_string(),
        }
    }
}

/// Everything the command line can say.
#[derive(Default)]
struct Arguments {
    files: Vec<String>,
    decoder: Option<String>,
    beam: Option<usize>,
    model: Option<String>,
    whisper: bool,
    from_file: Option<String>,
    fast: bool,
    stdin: bool,
    ffmpeg_input: Option<String>,
    quiet: bool,
}

fn parse_arguments(args: &[String]) -> Result<Arguments, String> {
    let mut parsed = Arguments::default();
    let mut rest = args.iter();
    while let Some(argument) = rest.next() {
        // Options that take a value read it from the next argument.
        let mut value = |name: &str| rest.next().cloned().ok_or_else(|| format!("{name} needs a value"));
        match argument.as_str() {
            "--decoder" => parsed.decoder = Some(value("--decoder")?),
            "--beam" => parsed.beam = Some(value("--beam")?.parse().map_err(|_| "--beam needs a whole number".to_string())?),
            "--model" => parsed.model = Some(value("--model")?),
            "--from-file" => parsed.from_file = Some(value("--from-file")?),
            "--ffmpeg-input" => parsed.ffmpeg_input = Some(value("--ffmpeg-input")?),
            "--whisper" => parsed.whisper = true,
            "--fast" => parsed.fast = true,
            "--stdin" => parsed.stdin = true,
            "--quiet" => parsed.quiet = true,
            option if option.starts_with("--") => return Err(format!("unknown option {option}")),
            file => parsed.files.push(file.to_string()),
        }
    }
    Ok(parsed)
}

/// Load the model the arguments ask for.
fn load_engine(arguments: &Arguments) -> Result<Engine, String> {
    if arguments.whisper {
        let dir = speech::whisper::model_dir();
        let whisper = Whisper::load(&dir).map_err(|e| format!("could not load Whisper from {}: {e}", dir.display()))?;
        return Ok(Engine::Whisper(Box::new(whisper)));
    }
    let name = arguments.decoder.as_deref().unwrap_or("lexicon");
    let mut decoder = Decoder::from_name(name).ok_or_else(|| format!("unknown decoder {name:?} (choose greedy, beam or lexicon)"))?;
    if let Some(beam) = arguments.beam {
        if beam == 0 {
            return Err("--beam must be at least 1".to_string());
        }
        decoder = match decoder {
            Decoder::Greedy => Decoder::Greedy,
            Decoder::Beam { .. } => Decoder::Beam { width: beam },
            Decoder::Lexicon { word_bonus, .. } => Decoder::Lexicon { width: beam, word_bonus },
        };
    }
    let path = arguments.model.clone().map_or_else(Recognizer::default_path, Into::into);
    let recognizer = Recognizer::load(&path).map_err(|e| format!("{e}\n(train one with `cargo run --release --example ch09_train`, or pass --model)"))?;
    Ok(Engine::Ours(Box::new(recognizer), decoder))
}

/// `speech transcribe`: one line of text per file.
fn transcribe(arguments: &Arguments) -> Result<(), String> {
    if arguments.files.is_empty() {
        return Err("transcribe needs at least one WAV file".to_string());
    }
    let engine = load_engine(arguments)?;
    for file in &arguments.files {
        let samples = load_audio_16k(file).map_err(|e| e.to_string())?;
        let started = Instant::now();
        let text = engine.transcribe(&samples);
        let took = started.elapsed().as_secs_f32();
        let seconds = samples.len() as f32 / SAMPLE_RATE as f32;
        println!("{file}: {text}");
        eprintln!("    {seconds:.2} s of audio, {:.0} ms to recognize ({:.0}x real time), {}", 1000.0 * took, seconds / took.max(1e-6), engine.describe());
    }
    Ok(())
}

/// `speech listen`: the live loop.
fn listen_command(arguments: &Arguments) -> Result<(), String> {
    if !arguments.files.is_empty() {
        return Err(format!("listen takes no file arguments (did you mean --from-file {}?)", arguments.files[0]));
    }
    let engine = load_engine(arguments)?;

    // 20 ms chunks: small enough for a lively meter, large enough to be cheap.
    let chunk = (SAMPLE_RATE / 50) as usize;
    let (mut source, description): (Box<dyn AudioSource>, String) = if let Some(file) = &arguments.from_file {
        (Box::new(WavPlayback::open(file, chunk, !arguments.fast).map_err(|e| e.to_string())?), format!("the file {file}"))
    } else if arguments.stdin {
        (Box::new(PcmStream::new(std::io::stdin().lock(), chunk)), "standard input (raw 16 kHz mono s16le)".to_string())
    } else if let Some(input) = &arguments.ffmpeg_input {
        let input: Vec<String> = input.split_whitespace().map(str::to_string).collect();
        (Box::new(Microphone::with_input_args(&input, chunk).map_err(|e| e.to_string())?), format!("ffmpeg {}", input.join(" ")))
    } else {
        (Box::new(Microphone::open(chunk).map_err(|e| e.to_string())?), "the microphone (through ffmpeg)".to_string())
    };
    eprintln!("listening to {description} with {}. Ctrl-C to stop.", engine.describe());

    // The meter is drawn on stderr, over and over on one line, and only if
    // a person is looking at it.
    let show_meter = !arguments.quiet && std::io::stderr().is_terminal();
    let mut utterances = 0;
    let result = listen(source.as_mut(), SegmenterConfig::default(), |event| match event {
        ListenEvent::Level { db, in_speech } => {
            if show_meter {
                eprint!("\r{}", level_meter(db, in_speech, 30));
                let _ = std::io::stderr().flush();
            }
        }
        ListenEvent::Utterance(segment) => {
            let started = Instant::now();
            let text = engine.transcribe(&segment.samples);
            let took_ms = 1000.0 * started.elapsed().as_secs_f32();
            if show_meter {
                eprint!("\r{:60}\r", "");
            }
            let seconds = |sample: usize| sample as f32 / SAMPLE_RATE as f32;
            let shown = if text.is_empty() { "(nothing recognized)" } else { &text };
            println!("[{:>6.2} s - {:>6.2} s]  {shown}   ({took_ms:.0} ms{})", seconds(segment.start), seconds(segment.end), if segment.cut_short { ", cut at the maximum length" } else { "" });
            let _ = std::io::stdout().flush();
            utterances += 1;
        }
    });
    if show_meter {
        eprint!("\r{:60}\r", "");
    }
    let heard = result.map_err(|e| e.to_string())?;
    eprintln!("heard {:.1} s of audio, {utterances} utterance{}.", heard as f32 / SAMPLE_RATE as f32, if utterances == 1 { "" } else { "s" });
    if heard == 0 && arguments.from_file.is_none() && !arguments.stdin {
        eprintln!("No audio arrived at all.\n{MICROPHONE_HELP}");
    }
    Ok(())
}

/// `speech words`: the five most probable words for each clip.
fn words(arguments: &Arguments) -> Result<(), String> {
    if arguments.files.is_empty() {
        return Err("words needs at least one WAV file".to_string());
    }
    let path = arguments.model.clone().map_or_else(WordClassifier::default_path, Into::into);
    let classifier = WordClassifier::load(&path).map_err(|e| format!("{e}\n(train one with `cargo run --release --example ch07_words -- train`)"))?;
    for file in &arguments.files {
        let samples = load_audio_16k(file).map_err(|e| e.to_string())?;
        println!("{file}  ({:.2} s)", samples.len() as f32 / SAMPLE_RATE as f32);
        for (word, probability) in classifier.top_k(&WordClassifier::fit_one_second(&samples), 5) {
            println!("  {word:<9} {:>5.1}%  {}", 100.0 * probability, "#".repeat((40.0 * probability).round() as usize));
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        println!("{HELP}");
        return ExitCode::from(2);
    };
    if args.iter().any(|a| a == "--help" || a == "-h") || command == "help" {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }
    let result = parse_arguments(&args[1..]).and_then(|arguments| match command.as_str() {
        "transcribe" => transcribe(&arguments),
        "listen" => listen_command(&arguments),
        "words" => words(&arguments),
        other => Err(format!("unknown command {other:?}")),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}\n\nRun `speech help` for usage.");
            ExitCode::FAILURE
        }
    }
}
