//! A speech recognizer built from scratch, one chapter at a time.
//!
//! Every module here is written in the book. There are no dependencies:
//! if we need an FFT, a matrix multiply or a random number, we write it.

pub mod mat; // a tiny matrix type                       (chapter 6)
pub mod rng; // a tiny random number generator           (chapter 6)

pub mod wav; // reading and writing WAV files            (chapter 1)
pub mod dsp; // framing, windows, FFT, mel, MFCC         (chapters 2-4)
pub mod plot; // SVG and terminal plots for the book's figures

pub mod dtw; // dynamic time warping                     (chapter 5)
pub mod nn; // layers, backprop, optimizers              (chapters 6-7)
pub mod ctc; // connectionist temporal classification    (chapter 8)
pub mod decode; // greedy and beam search decoding       (chapter 10)
pub mod metrics; // edit distance, WER, CER              (chapter 10)

pub mod data; // dataset loaders                         (chapters 5, 7, 9)
pub mod mic; // live audio capture                       (chapter 11)
pub mod whisper; // running a pretrained model ourselves (chapter 12)
