//! Dataset loaders: where the recordings live and how to read them.

pub mod fsdd; // Free Spoken Digit Dataset: ten digits, six speakers (chapters 5-6)

use std::path::PathBuf;

/// The directory that holds the datasets: the `SPEECH_DATA` environment
/// variable if it is set, `../data` otherwise (the examples are run from
/// `code/`, and the data sits next to it, outside git).
pub fn data_dir() -> PathBuf {
    std::env::var_os("SPEECH_DATA").map_or_else(|| PathBuf::from("../data"), PathBuf::from)
}
