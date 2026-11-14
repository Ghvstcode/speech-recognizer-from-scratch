# Synthesized utterances

Multi-word utterances for trying `speech transcribe` and `speech listen --from-file`
without downloading any dataset. They were built by `code/examples/ch09_figures.rs` exactly
like the recognizer's test set (chapter 9): single-word clips from the **test** speakers of
Google Speech Commands v0.02, trimmed, placed end to end with random pauses, with background
noise from the dataset's `_background_noise_` folder mixed in. The model never saw these
speakers or these stretches of noise in training.

| File | Words | Background noise |
|---|---|---|
| `utt_two_yes_go_visual.wav` | "two yes go visual" | SNR 20 dB |
| `utt_four_one_visual.wav` | "four one visual" | SNR 19 dB |
| `utt_yes_right.wav` | "yes right" | SNR 17 dB |
| `utt_down_five_three_sheila_nine.wav` | "down five three sheila nine" | SNR 17 dB |
| `utterances_with_pauses.wav` | four more utterances in one recording, more than a second of quiet between them: "five cat up" ... "left right stop" ... "no down seven" ... "one eight" | none (faint hiss only) |

The single utterances are numbers 9, 6, 4, 15 of the fixed test set (seed 2002). All of these files were
chosen because they are short, not very noisy, and recognized correctly by greedy decoding: they
are demonstrations, not a benchmark. Chapter 10 reports the
error rate over all 2000 test utterances.

## License

Derived from the Speech Commands dataset (Pete Warden, Google, 2018,
<https://arxiv.org/abs/1804.03209>), which is licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
These derived files are distributed under the same license. Changes made: clips were trimmed,
changed in speed by up to 10%, scaled, concatenated and mixed with the dataset's background noise.
