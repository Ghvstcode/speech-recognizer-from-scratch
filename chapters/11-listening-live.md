## Listening Live

In the last two chapters we trained a network that spells and gave it three decoders. With the lexicon decoder it gets 9.7% of the words wrong on our synthesized test set (greedy: 16.4%). All of that happened on files: somebody had already decided where each utterance begins and ends, and the whole recording was sitting in memory before we looked at it.

A microphone does none of that for us. It produces 16,000 numbers every second for as long as it is plugged in, most of them room noise, and nobody marks the place where a person starts talking. Today the recognizer leaves the test set. We are going to read audio as it arrives, cut the stream into utterances with the endpoint detector we built back in [chapter 2](02-slicing-time.md), and print what was said a moment after the speaker stops. At the end there is a real command-line tool, `speech listen`, and Part III of the book is done.

### How Do You Read A Microphone With No Dependencies?

Here is our problem. Talking to a sound card is different on every operating system: Core Audio on macOS, ALSA and PulseAudio on Linux, WASAPI on Windows. Each of those is a large API, and every Rust program that records audio uses a crate that wraps them. Our `[dependencies]` section is empty and it is going to stay empty.

We could write the wrappers ourselves, but that would be a book about operating systems, and none of it would teach us anything about speech. So we do something humbler, and I want to be upfront that it bends our rule: we borrow a *program* in place of a library. `ffmpeg` already knows how to open a microphone on every system, and it can write what it hears to its standard output as raw numbers. We start `ffmpeg` as a **child process** and read its output through a **pipe**, the same mechanism your shell uses for `a | b`.

So "from scratch" has an asterisk in this chapter: the recognizer is ours down to the FFT, the capture is `ffmpeg`'s. The format we ask it for is the simplest one there is, called `s16le`: every sample is a **s**igned **16**-bit integer, **l**ittle-**e**ndian (low byte first), no header, nothing else. It is the data chunk of a chapter 1 WAV file without the file around it. At 16,000 samples per second, one second is 32,000 bytes.

![A diagram in two rows. Top row: a microphone feeds an ffmpeg child process, which writes raw samples into a pipe at 32,000 bytes per second; inside our program a PcmDecoder turns bytes into samples and a Segmenter cuts the stream into utterances. Bottom row, once per utterance: features, network, decoder, and a printed line of text.](../figures/ch11-live-pipeline.svg)
*Everything left of the pipe is borrowed. Everything right of it we have built, most of it in earlier chapters.*

The bottom row is old news: features from chapter 4, the network from chapter 9, a decoder from chapter 10. The new code is the top right corner, about 550 lines in `src/mic.rs`. Enough of the talk, let's write some code!

### Two Bytes Make A Sample, Usually

The first job is turning bytes into samples. Two bytes, low one first, make an `i16`; divide by 32,768 and we have our usual `f32` between -1 and 1. If the bytes came from a file, that would be the whole story.

A pipe is not a file, though. When we ask it for bytes it gives us however many happen to be there, and that number can be odd. Then the last byte we got is the low half of a sample whose high half has not arrived yet. If we dropped that byte, every sample after it would be assembled from the high byte of one sample and the low byte of the next, and the audio would turn into loud noise. (If you built the TCP stack, this is the same lesson as there: a byte stream has no message boundaries.) So the decoder remembers one leftover byte between calls:

```rust
// src/mic.rs

/// Turns a stream of s16le bytes into samples.
#[derive(Clone, Debug, Default)]
pub struct PcmDecoder {
    leftover: Option<u8>,
}

impl PcmDecoder {
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
```

If a byte is waiting from last time, it is joined with the first new byte. `as_chunks::<2>()` splits the rest into pairs plus a remainder of zero or one byte, and that one byte becomes the new leftover. (`as_chunks` needs Rust 1.88 or newer; `chunks_exact(2)` is the older spelling.)

How do we know it works? Feed the same bytes in portions of every awkward size and demand the same samples each time:

```rust
// src/mic.rs (tests)

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
```

Portion size 1 is the cruel one: every single sample is split.

### Anything That Yields Audio

Before we start any processes, a design decision that pays for itself all chapter. The rest of the program should not care where audio comes from. It wants one operation: "wait until there is some audio, then give it to me".

```rust
// src/mic.rs

/// Something that produces 16 kHz mono audio a chunk at a time.
pub trait AudioSource {
    /// Wait until some audio is available and append it to `out`. Returns
    /// the number of samples appended; 0 means the stream has ended.
    fn read(&mut self, out: &mut Vec<f32>) -> io::Result<usize>;
}
```

We will have three implementations. `PcmStream<R: Read>` wraps any byte stream (a pipe, standard input) around a `PcmDecoder`; its `read` calls the reader, decodes, and loops once more if all it got was a single odd byte, because zero samples must only ever mean "the end". `Microphone` is the `ffmpeg` child. And `WavPlayback` is a WAV file pretending to be a microphone, which is how we will test everything without making a sound:

```rust
// src/mic.rs

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
```

The interesting part is `real_time`. A microphone cannot deliver the third second of audio before three seconds have passed, and a faithful stand-in should not either. So each chunk is held back until its due time, measured from one fixed start. Sleeping until a deadline, and not for "20 ms per chunk", means that time we spend recognizing is not added on top: if we fall behind, `checked_sub` returns `None` and we do not sleep at all.

### Borrowing ffmpeg

Now the child process. We need a command line that says: be quiet, read the microphone, convert to 16 kHz mono `s16le`, write to standard output. Only the "read the microphone" part depends on the operating system, so it gets its own function:

```rust
// src/mic.rs

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
```

On macOS the capture framework is AVFoundation, and `":default"` reads as "no video device, the default audio device". On Linux we ask PulseAudio, which PipeWire also answers to. On Windows, DirectShow really wants the device's name; `audio=default` works on some systems only. `-ar 16000 -ac 1` does the resampling and the mixdown to one channel, so whatever the hardware runs at, we receive what the model was trained on.

Starting the child is `std::process::Command`, with care about its three standard streams:

```rust
// src/mic.rs

pub struct Microphone {
    child: Child,
    stream: PcmStream<ChildStdout>,
    /// Delivers the end of the child's stderr once that stream has closed.
    complaint: Receiver<String>,
}

impl Microphone {
    /// Start any command that writes s16le audio to its standard output.
    pub fn spawn(command: &mut Command, chunk_samples: usize) -> io::Result<Microphone> {
        // stdout is the audio; stderr is a second pipe, so we can show
        // ffmpeg's own error message if it gives up; stdin is closed so the
        // child never waits for (or steals) our keyboard input.
        let mut child = command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| {
            // ... a NotFound error becomes "it is not installed or not on your PATH" plus a hint
        })?;
        let stdout = child.stdout.take().expect("stdout was requested as a pipe");
        let stderr = child.stderr.take().expect("stderr was requested as a pipe");
        // Two pipes need two readers: we read stdout, a thread reads stderr.
        Ok(Microphone { child, stream: PcmStream::new(stdout, chunk_samples), complaint: drain_stderr(stderr) })
    }
}
```

`Microphone::open` calls this with `ffmpeg` and the default arguments; `with_input_args` lets you replace the input part.

Why a thread for standard error? A pipe is a buffer of limited size inside the operating system, typically 64 KB. When it is full, the writer is put to sleep until somebody reads. Suppose we left the child's stderr alone and only read it after the child had exited. A child that writes more than a pipe's worth of messages would fall asleep in the middle of one, and it would never write another sample or exit. We would be waiting for audio from a process that is waiting for us: a deadlock, and no error message to tell us so. `-loglevel error` keeps `ffmpeg` terse, but terse is not a bound, and `spawn` accepts any command at all. The rule for a child with two output pipes is that both must be read for as long as it lives. Our one thread is blocked on stdout, so stderr gets a thread of its own:

```rust
// src/mic.rs

/// How much of a child's standard error we remember for an error message.
const STDERR_TAIL_BYTES: usize = 4096;

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
```

The thread reads until the stream closes and throws away everything but the last 4 KB, so a child that complains for hours costs us no memory. The reason for a failure is usually in a program's last lines, and that is the part we keep. When the stream closes, which happens when the child exits, the thread sends the tail through a channel and ends.

The other end of that channel is used in `explain_end`. When the audio pipe closes, we `wait` for the child. If its exit status is a failure, we take the tail from the channel and return it as our error, with a hint about installing `ffmpeg` and about microphone permission:

```rust
// src/mic.rs (inside impl Microphone)

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
```

`STDERR_PATIENCE` is two seconds. We use `recv_timeout` and not a `join` on the thread because the thread ends when the *pipe* closes, and a pipe stays open as long as any process holds its writing end. If the child had started a process of its own that outlived it, a `join` would wait for that stranger.

Because `spawn` takes any command, the unit tests do not need `ffmpeg` at all. One uses `cat` on a file of raw bytes, and `cat` on a missing file to check that a failing child's complaint ends up in our error message. Another runs a shell script that writes a megabyte to standard error before its first sample; with nobody reading stderr that test never gets its sample, so it runs under a ten-second limit and fails if the limit is reached.

The last piece matters more than its few lines suggest:

```rust
// src/mic.rs

impl Drop for Microphone {
    fn drop(&mut self) {
        // Both calls fail harmlessly if the child has already exited. The
        // stderr thread is not joined: it ends by itself when the dead
        // child's pipe closes.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
```

A child process does not die because the `Child` value goes out of scope. Without `kill`, an error anywhere in our program would leave an `ffmpeg` running with the microphone open. Without `wait`, the killed child would linger in the process table as a zombie until we exit. One test spawns `sleep 1000`, drops it, and checks that the drop returns within five seconds. Another drops a child that is writing to standard error without end, and checks that the reading thread finishes too.

**Aside** <br>
`Drop` runs when our program returns or unwinds normally. Ctrl-C is different: the default handler ends the process on the spot and no destructors run. What should save us there is the terminal itself, which sends the interrupt to the whole foreground job, child included. I say "should" because it depends on how the program was started. If you ever suspect a leftover, `pgrep ffmpeg` tells you.

### Cutting The Stream Into Utterances

We have an endless stream. The recognizer wants utterances. Chapter 2 left us the tool for this, the streaming `Endpointer`: a two-state machine that takes 10 ms blocks and reports `SpeechStart` and `SpeechEnd` events, both a little late and both carrying the sample position where the thing actually happened. Back then we noted that whoever uses it live has to keep recent audio around, to go back and fetch the beginning of the word. The `Segmenter` is that somebody. It has four settings:

```rust
// src/mic.rs

impl Default for SegmenterConfig {
    fn default() -> Self {
        SegmenterConfig { endpoint: EndpointConfig { hangover_ms: 600.0, ..EndpointConfig::default() }, pre_roll_ms: 150.0, post_roll_ms: 150.0, max_utterance_s: 10.0 }
    }
}
```

* **Hangover, 600 ms.** In chapter 2 speech ended after 200 ms of quiet, which is right for one word. People pause between words, and our model was trained on several words with short pauses between them, so "seven... three" should arrive as one utterance. We wait 600 ms before believing the speaker is done.
* **Pre-roll and post-roll, 150 ms each.** Extra audio kept before the detected start and after the detected end, on top of the detector's own 30 ms padding. Soft consonants live there.
* **Forced cut, 10 s.** An utterance longer than this is handed over anyway. Hold that thought; we will see why it is needed.

The segmenter keeps a buffer of samples and three positions, all counted from the start of the stream: where the buffer begins (`buffer_start`), how far the endpointer has been fed (`fed`), and where the current utterance began, if we are in one (`speech_start`). Here is the heart of it:

```rust
// src/mic.rs

impl Segmenter {
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
}
```

Read it in three parts.

**Regrouping.** Chunks arrive in whatever size the source likes; the endpointer wants exactly 160 samples at a time, and its `push` panics on a block of any other length. The `while` loop feeds it every complete block and leaves a partial one in the buffer for the next call. This is `PcmDecoder`'s leftover byte again, one level up.

**Events.** On `SpeechStart` we remember where the utterance began, pre-roll included, but never earlier than the audio we still hold. On `SpeechEnd` we copy the samples out with `cut`, which also clamps the end to what has actually arrived. On the way we compute the block's level in decibels for a level meter.

**Forgetting.** A program that listens for hours must not grow. While nobody speaks we keep one second, far more than the 180 ms of pre-roll and padding we could ever reach back for. During an utterance we keep everything from its start.

There is also `finish`, for a stream that ends mid-word: it closes the endpointer and returns whatever was in progress.

The tests are the argument that this is right. `segmenter_finds_two_utterances_whatever_the_chunk_size` builds faint noise with three loud bursts, the first two only 300 ms apart, then feeds it in chunks of 160, 1024, 333 and 100,000 samples. Each time exactly two utterances must come out (the 300 ms pause stays inside the first), their samples must be the stream's own, and the cut points must be *identical* across chunk sizes. Others check that an endless burst is cut into touching 2-second pieces when the limit is set to 2 s, and that twenty seconds of silence leave no more than about a second in the buffer.

### The Loop

Connecting a source to the segmenter takes a dozen lines. The caller passes a closure, and we tell it about two kinds of events: a level after every chunk (for the meter), and each finished utterance.

```rust
// src/mic.rs

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
```

Notice that all the audio work happens on one thread. While the closure is busy recognizing, nobody reads the audio pipe. That is fine as long as recognition is quick: the operating system buffers the pipe for us, and we are about to see that an utterance takes a few milliseconds.

### From A Segment To Text

What does the closure do with a `Segment`? The same thing chapter 9 did with a file: it calls `Recognizer::transcribe` from `src/data/pipeline.rs`, the one place where the chain from samples to text is written down, so training, evaluation and the live loop cannot drift apart. Features, network, `log_softmax`, then one of chapter 10's decoders.

The first link of that chain is the front end: chapter 4's log-mel followed by `normalize`, which shifts and scales each of the 40 bands to mean 0 and variance 1 *over the utterance*. That removes the recording level and the tilt of the microphone. It also needs the whole utterance before it can produce a single frame, as chapters 4 and 9 both warned. Remember that; it decides how our live recognizer feels.

### The speech Binary

Until now every program in this book has been an example. The recognizer deserves a proper front door: `src/main.rs`, a binary called `speech` with three commands. `transcribe` writes down what is said in WAV files, `listen` is the live loop, and `words` runs chapter 7's 35-word classifier on a clip. The file parses arguments by hand and prints; the one part worth reading is how `listen` picks its source and what it does with events. (`engine` is a small enum that holds our `Recognizer` together with the chosen decoder.)

```rust
// src/main.rs (inside listen_command)

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
    // ...
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
            // ... wipe the meter line, then print "[start - end]  text   (N ms)" on stdout
        }
    });
```

The options that choose between them, as `speech help` prints them:

```
$ cargo run --release --bin speech -- help
...
OPTIONS FOR listen
    --from-file FILE.wav            play a recording into the live pipeline instead of the microphone
    --fast                          with --from-file: as fast as possible instead of in real time
    --stdin                         read raw audio from standard input (16 kHz, mono, signed 16-bit little-endian)
    --ffmpeg-input "ARGS"           replace ffmpeg's input arguments, e.g. "-f alsa -i hw:1" for another
                                    device, or "-re -i FILE.wav" to test the ffmpeg pipe with a file
    --quiet                         no level meter, only the recognized text
...
```

Four sources behind one `Box<dyn AudioSource>`: this is where the trait earns its keep. The meter is drawn on standard error, over and over on one line (`\r` returns to the start of the line without advancing), and only when standard error is a terminal. Recognized text goes to standard output, so `speech listen > notes.txt` saves the text and nothing else.

### Run It

First the file command, on the sample utterances that ship with the repository. From `code/`:

```
$ cargo run --release --bin speech -- transcribe ../samples/utt_*.wav
../samples/utt_down_five_three_sheila_nine.wav: down five three sheila nine
    4.83 s of audio, 19 ms to recognize (253x real time), our model (496508 weights), lexicon decoder
../samples/utt_four_one_visual.wav: four one visual
    2.92 s of audio, 9 ms to recognize (324x real time), our model (496508 weights), lexicon decoder
../samples/utt_two_yes_go_visual.wav: two yes go visual
    2.87 s of audio, 9 ms to recognize (337x real time), our model (496508 weights), lexicon decoder
../samples/utt_yes_right.wav: yes right
    2.06 s of audio, 5 ms to recognize (376x real time), our model (496508 weights), lexicon decoder
```

Timings are from my machine and yours will differ, but the order of magnitude is the point: a few milliseconds per utterance. (These samples were picked because they come out right. The honest error rate is chapter 10's.) The `words` command is there for comparison with chapter 7:

```
$ cargo run --release --bin speech -- words ../samples/go.wav
../samples/go.wav  (1.00 s)
  go         99.9%  ########################################
  no          0.0%  
...
```

Now the live path. `samples/utterances_with_pauses.wav` is 12.2 seconds long and holds four utterances with more than a second of quiet between them. `--from-file` plays it into the pipeline at the speed a microphone would:

```
$ cargo run --release --bin speech -- listen --from-file ../samples/utterances_with_pauses.wav
listening to the file ../samples/utterances_with_pauses.wav with our model (496508 weights), lexicon decoder. Ctrl-C to stop.
[  0.63 s -   2.78 s]  five cat up   (6 ms)
[  3.83 s -   6.03 s]  left right stop   (5 ms)
[  7.07 s -   9.55 s]  no down seven   (6 ms)
[ 10.70 s -  11.76 s]  one eight   (6 ms)
heard 12.2 s of audio, 4 utterances.
```

That is amazing. Nobody told the program where the utterances were. Go ahead and run it yourself: the lines appear one at a time, each a moment after its utterance ends. Add `--fast` and the same four lines come out at once (the whole run took half a second here), with identical cut points, because the segmenter counts samples and never looks at a clock.

![Two panels over a shared time axis of 12 seconds. Top: the level in decibels of every 10 ms block, with four shaded regions labeled five cat up, left right stop, no down seven and one eight. Bottom: the detector's in-speech state, a square wave that rises shortly after each shaded region begins and falls about half a second after it ends.](../figures/ch11-endpointing-live.svg)
*What the segmenter saw. Drawn by `cargo run --release --example ch11_endpointing`, which runs the same loop on any WAV file you give it.*

Look at the first utterance. "Five", "cat" and "up" are three separate humps with the level falling all the way back to the background between them, and they still arrive as one segment: that is the 600 ms hangover. The green line in the lower panel is the detector's state. It rises at about 0.84 s, yet the segment starts at 0.63 s; the difference is the 30 ms it took to be sure, the 30 ms of padding and the 150 ms of pre-roll, fetched from the buffer.

That run never started a process. The next one does: `--ffmpeg-input` replaces the microphone arguments, and `-re` tells `ffmpeg` to read a file at its natural speed. This is a real child, a real pipe and the real `Microphone` code, with a file where the device would be:

```
$ cargo run --release --bin speech -- listen --ffmpeg-input "-re -i ../samples/utterances_with_pauses.wav"
listening to ffmpeg -re -i ../samples/utterances_with_pauses.wav with our model (496508 weights), lexicon decoder. Ctrl-C to stop.
[  0.63 s -   2.78 s]  five cat up   (5 ms)
[  3.83 s -   6.03 s]  left right stop   (4 ms)
[  7.07 s -   9.55 s]  no down seven   (10 ms)
[ 10.70 s -  11.76 s]  one eight   (5 ms)
heard 12.2 s of audio, 4 utterances.
```

Same four lines, same positions to the hundredth of a second. And since the format is plain bytes, anything that can produce them can feed us through `--stdin`:

```
$ ffmpeg -loglevel error -i ../samples/utterances_with_pauses.wav -ar 16000 -ac 1 -f s16le - | cargo run --release --bin speech -- listen --stdin
```

It prints the same four lines again. When `ffmpeg` fails, we pass its complaint on. Here is a nonsense input:

```
$ cargo run --release --bin speech -- listen --ffmpeg-input "-f lavfi -i nosuchfilter"
listening to ffmpeg -f lavfi -i nosuchfilter with our model (496508 weights), lexicon decoder. Ctrl-C to stop.
error: the recording process stopped (exit status: 8): [AVFilterGraph @ 0x776c03a80] No such filter: 'nosuchfilter'
...
The microphone is read by running `ffmpeg`, so ffmpeg must be installed and on your PATH
(macOS: `brew install ffmpeg`; Debian/Ubuntu: `sudo apt install ffmpeg`).
...
```

### Your Microphone

Every output block above comes from a file or a pipe, because those are the runs that behave the same on your machine as on mine. The one link they do not exercise is `ffmpeg` opening an actual capture device. That depends on your operating system, your hardware and your permissions, and the default arguments in `default_input_args` follow `ffmpeg`'s documentation; they are a starting point, not something I can promise for your setup. So here is the command, what to expect, and what to check when it does not work.

```
$ cargo run --release --bin speech -- listen
```

You should see the `listening to the microphone (through ffmpeg)` line and then a meter that redraws 50 times a second. Its format is fixed by `level_meter`; the unit test pins it down with a 10-character bar (the binary uses 30):

```
[#####.....]  -30 dB  
```

The bar covers -60 dB to 0 dB, and a `*` after the number means the segmenter is inside an utterance. Say two or three words from the list with short pauses, "seven... three... stop", then be quiet. A little over half a second later a line in the same `[start - end]  text   (N ms)` format as above should replace the meter. Ctrl-C stops it.

If it does not work:

* **`could not start ffmpeg`.** It is not installed or not on your `PATH`. `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian and Ubuntu.
* **macOS: an error, or a meter stuck at the bottom.** The first time a program started from your terminal opens the microphone, macOS asks whether the *terminal application* may use it. If the answer was no, allow it under System Settings → Privacy & Security → Microphone and restart the terminal.
* **The wrong device, or no default.** Pass your own input arguments. Linux without PulseAudio: `--ffmpeg-input "-f alsa -i default"`, or `hw:1` for a second card. macOS: `ffmpeg -f avfoundation -list_devices true -i ""` lists the devices, then `--ffmpeg-input "-f avfoundation -i :1"` picks audio device 1. Windows: `ffmpeg -list_devices true -f dshow -i dummy` prints the names, then `--ffmpeg-input "-f dshow -i audio=NAME"`. The arguments are split at spaces, so a device name containing a space will not survive.
* **The star never appears.** Watch the number: speech has to rise 10 dB above the background and above -60 dB. Move closer or raise the input level.
* **Still stuck?** Record a WAV with any tool and use `--from-file`. That separates "the capture is broken" from "the recognizer is wrong".

There is one more way to try it, and it needs neither ffmpeg nor a terminal. Our crate has no dependencies, so it compiles to WebAssembly without changing a line, and the book's website runs the same recognizer inside a page. The browser supplies the microphone and hands us samples; everything after that is the code from this chapter and the ones before it: the resampler, the endpoint detector, the features, the network and the three decoders. The glue is a small crate, [`web/`](../web/src/lib.rs), and it is the only place in the repository with an `unsafe` block in it, because a pointer and a length coming from JavaScript cannot be turned into a slice any other way.

> **Play with it:** [the recognizer, in your browser](../interactive/listen.html). Say two or three of the 35 words, press Stop, and look at where the endpoint detector cut and which frames the letters came from. The page also has a few recordings to try if you would rather not use the microphone.

### What It Cannot Do

It listens. Now the fine print, because a demo that only gets shown its favorite inputs teaches nothing.

**Nothing appears while you speak.** Run the real-time file again with a stopwatch: the first line, for audio that ends at 2.78 s, prints 3.20 s into the run. The last loud block is at 2.60 s, and 600 ms of hangover later we know the speaker has stopped. Only then do we normalize, and normalization wants the whole utterance. Dictation software shows words while you are still talking; ours cannot, by construction.

**It knows 35 words, and has no way of saying "I don't know that one".** macOS has a speech synthesizer on the command line, `say`, so I had it speak a few phrases into 16 kHz WAV files (`say -v Samantha -o hello.wav --data-format=LEI16@16000 "hello world"`) and transcribed each with `--decoder greedy` and with the default. A comma makes the voice pause. (The voices differ between macOS versions and the recordings are not in the repository, so read the table as an illustration; yours may come out differently.)

```
said                      greedy               lexicon
seven, three, stop        seven three stop     seven three stop
hello world               o wo                 no on
turn, the, lights, off    ci dog right off     bird right off
seven, eleven, twelve     seven lmearte wow    seven marvin wow
go left                   zlla                 follow
go, left                  go left              go left
```

Greedy shows what the network really spells: fragments of sound, "o wo" for "hello world". The lexicon decoder may only output vocabulary words, so it reports the ones that score best on those same frames, "no on", with a straight face. (It does not correct greedy's string. It searches the frame probabilities again with only the 35 words allowed.) In-vocabulary words inside a foreign sentence often survive ("right off"). When the output surprises you, switch to `--decoder greedy`; it is the more honest of the two.

**It wants pauses.** Look at the last two rows. The model learned from single-word clips glued together, never from fluent speech where words run into each other. "Go left" in one breath comes out as "follow"; with a pause it is fine.

**A recording unlike the training data can fail outright.** `samples/seven_jackson.wav` is the word "seven" from chapter 5's digit dataset. `speech transcribe` hears "marvin" (greedy: "mn"), while `samples/seven.wav`, the same word from Speech Commands, comes out as "seven". The two clips differ in speaker, microphone and dataset, and the first was recorded at 8 kHz, the sample rate of a telephone line, so after upsampling it has nothing above 4 kHz. Ten of our 40 mel bands are centered above 4 kHz, and the model has never seen them empty. The missing band is only part of the story, though. `ch10_decode` also runs the 11,005 single-word test clips a second time after taking them down to 8 kHz and back, which changes nothing else, and the lexicon decoder's word error rate goes from 11.3% to 14.9%. Worse, and far from broken. The rest of what went wrong with this clip is a voice and a recording setup the model had never met, and one clip cannot tell us more than that.

**The noise floor only learns in silence.** Chapter 2's endpointer adapts its background estimate only while it believes nobody is speaking. Suppose a fan switches on and the background jumps by more than 10 dB. The detector takes that for speech, the floor is frozen during speech, and so it never finds out that this is the new quiet. It would stay "in speech" for as long as the fan runs, and the segmenter would buffer forever. That is what the 10-second forced cut is for: the audio is handed to the recognizer anyway, marked `cut at the maximum length`, and the buffer starts over. It is a guard, not a cure.

### Try It Yourself

1. **Split the utterances.** In `SegmenterConfig::default()` set `hangover_ms` to 200.0 and rerun `listen --fast --from-file ../samples/utterances_with_pauses.wav`. Every pause longer than 200 ms now ends an utterance. Count the segments, and see whether the transcripts get better or worse.
2. **Break the decoder on purpose.** In `PcmDecoder::push`, delete the three lines that store the leftover byte and run `cargo test pcm`. Then pipe the sample file through `--stdin` and see whether your operating system happens to deliver even-sized reads.
3. **Measure the delay.** Pipe the real-time run through a few lines of script that print the wall-clock time of every output line, and compare with the `[start - end]` stamps. Then predict what a 300 ms hangover does to the numbers, and check.
4. **Trigger the forced cut.** In the `ch11_endpointing` example, build the `Segmenter` with `max_utterance_s: 1.0` and run it on the sample file. Each utterance should come out as one-second pieces plus a remainder; look at what the recognizer makes of words cut in half.
5. **Your own voice, from a file.** Record yourself saying "one two three" with pauses, then without, and compare `--decoder greedy` with the default on both.

### Loose Ends

* **We borrowed the capture.** `ffmpeg` is a large program standing in for a small job. A from-scratch version would talk to Core Audio or ALSA directly.
* **One thread for audio.** Recognition runs inside the read loop. At a few milliseconds per utterance that is harmless; with a slow model, audio would queue up in the pipe and a reader thread with a channel would be the fix.
* **No streaming recognition.** Per-utterance normalization and a bidirectional network both need the end of the utterance before they can start. Printing words as they are spoken needs a running normalization and a model that only looks backward.
* **No rejection.** There is no confidence score and no "that was not one of my words".
* **Endpointing by loudness.** A door slam is speech to us. Real systems use a small trained voice activity detector.
* **Ctrl-C relies on the terminal** to stop the child, and `--ffmpeg-input` cannot express arguments that contain spaces.

What's next on our agenda? Let's take stock, because this is the end of Part III. Starting from an empty `Cargo.toml` we parsed WAV files, built an FFT and a mel filterbank, wrote layers with hand-derived gradients, implemented CTC, trained a network of 496,508 weights on a laptop, decoded its output three ways, and today connected it to a live stream. You can say "seven three stop" and watch the words appear. The recognizer is ours; only the capture is borrowed.

And we have just listed, with examples, everything it cannot do: 35 words, pauses required, "hello world" heard as "no on". Most of that is for lack of data and compute: fifteen hours of single words, where the serious systems had hundreds of thousands of hours of sentences. The rest is a small number of ideas we have not met yet, and they come with the model we are about to open. So in the next chapter we finally allow ourselves someone else's weights. We are going to open OpenAI's Whisper model file, find out what is inside, and start building the engine that runs it. `[dependencies]` stays empty.

### References

- [Corresponding code: `src/mic.rs`](../code/src/mic.rs), [`src/main.rs`](../code/src/main.rs), [`src/data/pipeline.rs`](../code/src/data/pipeline.rs) and [`examples/ch11_endpointing.rs`](../code/examples/ch11_endpointing.rs)
- [Chapter 2](02-slicing-time.md), where the `Endpointer` was built
- [`std::process::Command`](https://doc.rust-lang.org/std/process/struct.Command.html) and [`std::process::Child`](https://doc.rust-lang.org/std/process/struct.Child.html) in the Rust standard library; the `Child` page explains why a child must be waited for
- [ffmpeg devices documentation](https://ffmpeg.org/ffmpeg-devices.html): the `avfoundation`, `pulse`, `alsa` and `dshow` inputs and how to list devices
- [Pipeline (Unix)](https://en.wikipedia.org/wiki/Pipeline_(Unix)) and [Voice activity detection](https://en.wikipedia.org/wiki/Voice_activity_detection) on Wikipedia
- [`samples/UTTERANCES.md`](../samples/UTTERANCES.md): how the sample utterances were made

[← Previous: Decoding: From Probabilities To Words](10-decoding.md) · [Contents](../README.md) · [Next: Reading Someone Else's Weights →](12-reading-someone-elses-weights.md)
