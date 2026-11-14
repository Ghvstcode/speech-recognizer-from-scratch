## Where To Go From Here

In the last chapter our own code ran somebody else's model, and a paragraph of English came out with commas in it. There is nothing left to build. This chapter has no new code at all. It does three things: it puts everything we measured on one page, it puts those numbers next to the ones the rest of the field reports, and it lists what we walked past on the way, with a pointer to the file in the repository where each missing piece would go.

It is short, and it is the chapter where we stop being pleased with ourselves for a moment.

### What We Built, On One Page

Fourteen chapters, about 18,500 lines of Rust in the library with its tests (and another 5,400 in the example programs), and an empty `[dependencies]` section. Here is every recognizer the book produced, with the number we measured for it.

| Chapter | Recognizer | Tested on | Result |
|---|---|---|---|
| [5](05-dynamic-time-warping.md) | DTW against 2,700 stored templates | spoken digits, speakers it has templates for | 100% (300 of 300) |
| | | a speaker it has never heard | 80.2% |
| [6](06-a-neural-network-from-nothing.md) | MLP, 59,018 weights | spoken digits, known speakers | 99.7% |
| | | a speaker it has never heard | 81.0% |
| [7](07-layers-that-understand-time.md) | convolution + GRU classifier, 170,851 weights | 11,005 one-word clips, 35 words, unseen speakers | 93.5% |
| [9](09-training-the-recognizer.md), [10](10-decoding.md) | CTC recognizer, 496,508 weights | 2,000 synthesized utterances of 1 to 5 words | WER 16.4% greedy, 15.6% beam search, 9.7% with the lexicon |
| [12](12-reading-someone-elses-weights.md), [13](13-a-transformer-that-listens.md) | Whisper tiny.en, 37.8 million weights (not ours) on our engine | a 20-second paragraph of synthetic speech | one phrase wrong; 0.51 s, 39.6 times faster than real time |

Under that table sits the part no row shows: a WAV parser, an FFT, a mel filterbank, a matrix type, backpropagation checked against finite differences, the CTC forward-backward algorithm, two beam searches, a microphone pipe, a JSON parser and a transformer forward pass. The timing in the last row is from the machine chapter 13 used, an Apple M4 Max; the encoder costs about a quarter of a second per 30-second window whatever is in it.

Read the table from top to bottom and one pattern repeats. Every time the number got better, it was because the test got harder and the data got bigger at the same step: six speakers, then two thousand, then the hundreds of thousands of hours OpenAI collected. Hold that thought.

### Honest Numbers

Our own model knows 35 words. On its home ground, utterances glued together from the same kind of clips it was trained on, the lexicon decoder gets three utterances in four exactly right, and 62% of the five-word ones. It confuses "three" with "tree" and "off" with "up". Say a word it does not know and the lexicon decoder turns it into ones it does: a synthetic voice saying "hello, world" came out as "follow wow". It has no way to say "I did not catch that". Fluent speech hurts it (the same synthetic voices got 9 phrases of 16 right without pauses, against 15 with them), noise hurts it (8.0% of words wrong with no background recording, 13.9% at the lowest signal-to-noise ratios we tested), and a recording from a different dataset can fail outright. Chapters 9 to 11 measured or showed each of these, and none of it is hidden.

Now the field. The usual yardstick for English is LibriSpeech (Panayotov et al., 2015): about 1,000 hours of people reading audiobooks aloud, with no fixed word list. The Whisper paper's appendix reports, for greedy decoding like ours, a word error rate of 5.6% on its `test-clean` set for `tiny.en`, the model we ran, and 2.7% for the largest model, `large-v2`, which has about 1.5 billion weights. Systems trained on LibriSpeech itself go lower still: the Conformer paper (Gulati et al., 2020) reports 2.1% on the same set, and 1.9% with a language model. For single words, the Audio Spectrogram Transformer paper (Gong et al., 2021) reports 98.1% on the 35-word Speech Commands task where our classifier scores 93.5%.

Put side by side:

| | Vocabulary | Training audio | Weights | Word error rate |
|---|---|---|---|---|
| Our CTC recognizer | 35 words | about 15 hours | 0.5 million | 9.7% on our own synthesized test, with a lexicon |
| Whisper tiny.en | open | 438,000 hours, the English part of 680,000 | 37.8 million | 5.6% on LibriSpeech test-clean (paper) |
| Whisper large-v2 | open | all 680,000 hours | about 1,550 million | 2.7% on LibriSpeech test-clean (paper) |

Three warnings about this table, because it flatters us. First, the last column does not compare like with like. Picking among 35 known words separated by pauses is a far easier task than transcribing a stranger reading a novel, and we do worse on the easy task than Whisper does on the hard one. Second, the Whisper figures are the paper's, scored after its own text normalization. The 680,000 hours are the whole multilingual collection. The paper counts 117,000 of them as speech in other languages and 125,000 as translation data, which leaves 438,000 hours of English speech with English transcripts, the figure the model card gives as well. We put that in the `tiny.en` row because it is the English-only model. Neither the paper nor the card states a separate number of hours for the English-only checkpoints, so read 438,000 as the most that row can mean. The two parameter counts come from different places: 37.8 million is what we counted in the `tiny.en` file in chapter 12 (the paper rounds its tiny models to 39 million), and 1,550 million is the paper's figure for the large model. We have not run LibriSpeech through our engine, and chapter 13 admitted that we never compared our outputs with the reference implementation number by number. Third, "15 hours" is the speech inside the training clips after trimming (906.7 minutes); the whole dataset is 105,829 one-second clips, about 29 hours with the silence.

So the distance in training audio is roughly a factor of 29,000 to `tiny.en` (438,000 hours against 15) and 45,000 to the multilingual models. In weights it is 76 to the smallest Whisper and 3,000 to the largest. And here is the claim this book has been building up to: the distance is a great deal of data and compute, and a small number of ideas that are within reach. Go through Whisper's pipeline and every stage has a small cousin somewhere in our crate.

| Whisper | Our small cousin | Where |
|---|---|---|
| 80-band log-mel spectrogram | 40-band log-mel spectrogram; the same Fourier analysis, by a 512-point FFT where Whisper's recipe needs a 400-point transform | `dsp/mel.rs`, chapter 4 |
| two convolutions, the second with stride 2 | one convolution with stride 2 | `nn/conv.rs`, chapter 7 |
| transformer encoder: every frame looks at every other frame | bidirectional GRU: every frame hears from both directions | `nn/gru.rs`, chapter 7 |
| decoder trained with cross-entropy to predict the next token | softmax and cross-entropy; CTC for the alignment | `nn/loss.rs`, `ctc.rs` |
| 51,864 tokens: 50,256 pieces of text, 1,608 control tokens | 28 symbols: letters, space, blank | `decode.rs` |
| a language model inside the decoder | a trie of 35 legal words | `decode.rs`, chapter 10 |
| beam search (in the reference implementation) | prefix beam search | `decode.rs`, chapter 10 |
| Adam-family optimizer, warm-up, gradient clipping | Adam, warm-up, gradient clipping | `nn/optim.rs` |

A cousin is not a twin. Two rows of that table are differences of design and not of size: attention in place of a GRU, and a decoder that reads its own output in place of CTC. Chapter 8 saw the second one coming. CTC treats frames as independent given the audio, so our network has no built-in sense that `q` is followed by `u` (what it knows of spelling it has to get from the sound), and we said a fix needs "a language model or a different kind of loss". Chapter 10's lexicon was the first, in miniature. Whisper's decoder is the second: each token is predicted from the audio *and* from every token already written. That is the difference between "lein" and "learn". It is also why Whisper says "you" to an empty room, while our lexicon decoder cannot write a word outside its list. (It can still write a listed word that nobody said: "hello world" came out as "follow wow".)

How much of the gap belongs to those two design choices and how much to the data, nothing we measured can say. We never trained an attention decoder on our 15 hours, and we never trained our CTC model on more. What the table does support is the smaller claim: no stage of Whisper is made of something we could not build.

### What We Skipped

A book that fits in a weekend's reading has to walk past most of the field. Here is what we walked past, grouped by where it would attach.

**In the decoder.**

- **Language models.** An n-gram model counts which words follow which in a pile of text; *shallow fusion* adds its log-probability to each hypothesis during beam search. Our `Lexicon` already accepts a log-probability per word (`with_log_probs`), and `lexicon_beam_search` already adds a word bonus at every word end, so the hook exists. We never used it, for an honest reason: our utterances are random words, and no language model can predict a coin toss.
- **Sub-word units.** We read Whisper's byte-pair vocabulary and never built one. Learning the merges from text and training a CTC model on pieces instead of letters is how real systems get past spelling.
- **Whisper's own decoding.** Chapter 13's Loose Ends list them: beam search, the temperature fallback for degenerate output, the no-speech check, timestamps, stitching long audio at the model's own timestamps, text prompts (our tokenizer only decodes), and the multilingual models.

**In training.**

- **Training attention.** Every transformer layer we wrote is forward only. An attention-based encoder-decoder (the lineage runs from *Listen, Attend and Spell* to Whisper) needs `backward` for attention and layer norm, and our gradient checker to keep us honest.
- **The transducer.** The RNN-Transducer adds a small network that sees the labels emitted so far, removing CTC's independence assumption while still producing output frame by frame. It is a common choice for streaming recognizers that run on the device.
- **Regularization.** Our `nn` module has no dropout, no batch norm and no trainable layer norm. Chapter 7 masked one random patch of each spectrogram, a first step toward SpecAugment, which masks several bands and stretches of time and warps the time axis; chapter 9's recognizer got noise and speed changes but no masking.
- **Real datasets.** We never trained on sentences. LibriSpeech and Mozilla's Common Voice are free to download.

**In the plumbing.**

- **Speed.** No SIMD intrinsics, no BLAS, no GPU, and an FFT that computes a mirror-image half it then discards (chapter 3) and only handles powers of two. Chapter 13 estimated that tuned kernels would be several times faster.
- **Quantization.** Whisper's weights are 151 MB of `f32`. Stored as 8-bit integers they would be a quarter of that, at a small cost in accuracy that we did not measure.
- **Voice activity detection.** Our endpointer only knows loudness (chapter 2), its thresholds were picked by hand, and it adapts its noise floor only during silence. Production detectors are small neural networks.
- **Streaming.** Our live loop waits for the speaker to stop, because normalization needs the whole utterance (chapter 4) and the GRU runs in both directions. Words appear after a pause, never while you talk.
- **Audio in.** Uncompressed WAV only (chapter 1), and the microphone arrives through a child `ffmpeg` process, not the operating system's audio interface.
- **Rejection.** Since chapter 5, no recognizer of ours has been able to answer "none of the above".

### Projects, Ranked By Effort

No list of small tweaks this time. These are larger, each starts from a specific file, and I have not done any of them, so the estimates are guesses.

**An afternoon.**

1. **A real FFT.** In `dsp/fft.rs`, add a transform for real input that computes only bins 0 to n/2, and switch `power_spectrum_into` to it. `cargo run --release --example ch03_fft` has the benchmark; the existing tests tell you whether you broke anything.
2. **"None of the above."** `TemplateBank::classify` in `dtw.rs` already returns the distance next to the label. Collect distances for right and wrong answers in `examples/ch05_digits.rs`, pick a threshold, and count how many errors turn into refusals and how many correct answers you lose.
3. **Masking for the recognizer.** Lift `mask_random_patch` out of `examples/ch07_words.rs` and apply it to the features in `examples/ch09_train.rs`. Budget another 40 minutes or so for the training run, then compare with 16.4% and 9.7%.

**A weekend.**

4. **A language model, and data that deserves one.** Change the synthesizer in `data/synth.rs` to draw word sequences from a small grammar (phone numbers; "go left three") in place of uniform random words. Count bigrams on the training text, and add their log-probabilities where `lexicon_beam_search` in `decode.rs` completes a word. Now the model has something to predict.
5. **Beam search for Whisper.** Next to `greedy_decode` in `whisper/transcribe.rs`, keep the best few token sequences alive, each with its own key/value cache. The test case is waiting in `samples/stop.wav`, where " Stop" was the third candidate at 9.3%, and chapter 13 worked out by hand what a search over whole transcripts should find there: "Jump!".
6. **Timestamps.** Drop `<|notimestamps|>` from the prompt, teach `tokenizer.rs` to read the time tokens, and print subtitles. That is only the way in. The reference also constrains the time tokens while it decodes (`ApplyTimestampRules` in its `decoding.py`: they come in pairs, they never run backwards, and the first one is bounded), and you will want the same rules. Then use the last timestamp to choose where the next 30-second window starts, which fixes the worst of our long-audio handling.
7. **Dropout and layer norm that train.** Add both as `Layer` variants in `nn/model.rs` with `backward`, and run `nn/gradcheck.rs` over them. Dropout needs its random mask held fixed while the checker runs.

**A month.**

8. **LibriSpeech with our CTC.** Train a character model on the 100-hour clean subset, starting from `examples/ch09_train.rs`. The files are FLAC, so either convert them with `ffmpeg` or write the decoder chapter 1 skipped. You will want a wider model, a faster `matmul` in `nn/kernels.rs`, and patience: this is where a laptop CPU starts to hurt. I do not know what error rate you will get.
9. **A transformer you train.** Write `backward` for the attention and layer norm in `whisper/layers.rs`, check them numerically, and train a two-layer encoder-decoder on chapter 9's synthesized utterances. Small enough to finish, and the first model in the repository that we could train to predict each token from the ones it has already written.
10. **A transducer loss.** A sibling of `ctc.rs`. The lattice becomes two-dimensional, frames by labels, and the forward-backward recursion carries over almost unchanged. Pair it with a one-directional GRU and a running normalization in `mic.rs`, and words appear while you are still speaking.
11. **Whisper in 8 bits.** Quantize the weights when `safetensors.rs` loads them, write an integer `matmul`, and measure what it costs on your own recordings.

If you only do one, do number 4 or number 9. They are the two places where the gap between our recognizer and Whisper is an idea you can hold in your hand.

### Closing The File

That is the book. We started with air pressure sampled 16,000 times a second and a promise not to import anything, and we ended with a program that writes down English. In between, the things that looked like magic turned out to be loops: a correlation with sinusoids, a cheapest path through a grid, a sum over alignments, a weighted average called attention.

I am not the ultimate authority on any of this. I learned it by building it, the same way you just did. There will be mistakes in these chapters, and the numbers in them are from one machine and one training run. If something does not reproduce, the code is the authority.

The best next step is not on the reading list below. Record yourself, run `speech listen`, watch it get a word wrong, and go find out why.

### References

Books.

- Rabiner and Juang, *Fundamentals of Speech Recognition*, Prentice Hall, 1993. The pre-neural field in one volume: filterbanks, cepstra, DTW, hidden Markov models. Chapters 2 to 5 of this book are its first half, in miniature.
- Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), third edition draft, free online. Its speech recognition chapter covers features, CTC, encoder-decoders and WER; its n-gram chapter is the language model we skipped.

Features and template matching.

- Davis and Mermelstein, "Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences", *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 1980. The paper that established MFCCs, and it evaluates them with DTW.
- Sakoe and Chiba, "Dynamic Programming Algorithm Optimization for Spoken Word Recognition", *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 1978. DTW, with the step patterns and the band from chapter 5.

Networks and training.

- Cho et al., [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), EMNLP 2014. Introduces the gated unit we call the GRU.
- Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), ICLR 2015. Short and readable; our `Adam::step` is its Algorithm 1.
- Park et al., [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779), Interspeech 2019. Masking the spectrogram, done properly.

Sequences without alignments.

- Graves, Fernández, Gomez and Schmidhuber, [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf), ICML 2006. Chapter 8's source.
- Hannun, [Sequence Modeling With CTC](https://distill.pub/2017/ctc/), Distill, 2017. The illustrated version; read it before the paper.
- Graves, [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711), ICML 2012 Workshop on Representation Learning. The transducer, project 10.
- Chan, Jaitly, Le and Vinyals, [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211), 2015. Attention-based recognition before transformers.

Transformers and Whisper.

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017. The transformer; after chapter 13 you can read it as a description of code you own.
- Radford, Kim, Xu, Brockman, McLeavey and Sutskever, [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356), 2022. The Whisper paper, and the source of the Whisper error rates above. Its section 4.5, on long-form transcription, is the list of things our loop does not do.
- Gulati et al., [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100), Interspeech 2020. Convolution and attention in one block; source of the 2.1% and 1.9% figures.
- Gong, Chung and Glass, [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778), Interspeech 2021. Source of the 35-word classification figure.

Data.

- Warden, [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209), 2018. Where our 35 words came from, and how they were collected.
- Panayotov, Chen, Povey and Khudanpur, "Librispeech: An ASR Corpus Based on Public Domain Audio Books", ICASSP 2015. The benchmark everyone quotes.
- Ardila et al., [Common Voice: A Massively-Multilingual Speech Corpus](https://arxiv.org/abs/1912.06670), LREC 2020. Volunteer recordings in many languages and accents.

Our code, one last time: [`code/src/`](../code/src/), [`code/examples/`](../code/examples/), and the interactive pages in [`interactive/`](../interactive/index.html).

[← Previous: A Transformer That Listens](13-a-transformer-that-listens.md) · [Contents](../README.md)
