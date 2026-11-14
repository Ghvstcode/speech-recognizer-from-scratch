# Trained models

Weights produced by the book's own training programs, small enough to keep in
the repository so that `speech transcribe`, `speech listen` and `speech words`
work right after `git clone`.

| File | What it is | Made by |
|---|---|---|
| `words.bin` | Chapter 7's 35-word classifier (conv + BiGRU + mean pooling) | `cargo run --release --example ch07_words -- train` |
| `recognizer.bin` | Chapters 8-10's character-level CTC recognizer | `cargo run --release --example ch09_train` |
| `recognizer-training.log` | Everything `ch09_train` printed during the run that produced `recognizer.bin` | the same run (`... > ../models/recognizer-training.log`) |
| `recognizer-posteriors.txt` | The per-frame probabilities of one validation utterance at several training steps (for the figure `ch09-posteriors-over-training.svg`) | the same run |
| `words-training.log` | Everything `ch07_words train` printed during the run that produced `words.bin` | `ch07_words train` |

## File format of the `.bin` files

The format written by `Model::save` and documented at the top of
`code/src/nn/io.rs`: the magic bytes `SPEECHNN`, a version number, then every
layer with its sizes and weights as little-endian `f32`. The file describes
the network completely (layer types and sizes included), so loading needs no
separate configuration.

## What is *not* in the file

A model file does not say what its inputs must look like. Both models expect
what `code/src/data/pipeline.rs` (`FrontEnd`) computes, and nothing else:

- 16 kHz mono audio (other rates are resampled on load),
- 40 log-mel bands (`FeatureConfig::new(16000)`: 25 ms Hann windows every 10 ms,
  512-point FFT, 20 Hz to 8 kHz, pre-emphasis 0.97, natural log of energy + 1e-6),
- each band normalized to mean 0, variance 1 over the utterance.

`words.bin` additionally expects exactly one second of audio (98 frames) and
answers with 35 scores in the order of `speech_commands::WORDS` (alphabetical).
`recognizer.bin` takes any length and answers with 28 scores per 20 ms in the
order of `decode::Alphabet`: blank, space, `a` to `z`.

## Format of `recognizer-posteriors.txt`

Plain text. A comment line, `utterance <index in the fixed validation set>`,
`text <transcript>`, then repeated blocks: a header `snapshot <step> <frames> <classes>`
followed by one line per frame with `<classes>` probabilities.
