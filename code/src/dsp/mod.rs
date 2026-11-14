//! Signal processing: from raw samples to the features a recognizer looks at.
//!
//! The modules are in the order the book builds them, and each one only uses
//! the ones above it:
//!
//! ```text
//! samples ──frame──▶ frames ──window──▶ ──fft──▶ power spectrum ──mel──▶ log-mel ──dct──▶ MFCC
//! ```

pub mod frame; // cutting a signal into short frames, energy, endpointing  (chapter 2)
pub mod window; // Hann, Hamming                                           (chapter 2)

pub mod fft; // complex numbers, the DFT and the FFT                       (chapter 3)
pub mod spectrogram; // the short-time Fourier transform as a picture      (chapter 3)

pub mod mel; // the mel scale and its triangular filterbank                (chapter 4)
pub mod features; // log-mel, MFCC, deltas, normalization                  (chapter 4)

pub mod resample; // changing the sample rate (used from chapter 5 on, whenever a clip's rate differs from the model's)
