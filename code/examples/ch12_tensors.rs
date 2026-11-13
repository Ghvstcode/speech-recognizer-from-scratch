//! Chapter 12: list every tensor in a safetensors file.
//!
//! ```text
//! cargo run --release --example ch12_tensors                # the Whisper weights
//! cargo run --release --example ch12_tensors -- some.safetensors
//! ```
//!
//! Only the header is read, so this is instant even for a huge file.

use speech::whisper::{model_dir, safetensors};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = match args.as_slice() {
        [] => model_dir().join("model.safetensors"),
        [path] if !path.starts_with('-') => PathBuf::from(path),
        _ => {
            eprintln!("usage: ch12_tensors [path/to/model.safetensors]");
            return ExitCode::FAILURE;
        }
    };
    let infos = match safetensors::read_header(&path) {
        Ok(infos) => infos,
        Err(error) => {
            eprintln!("cannot read {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };

    println!("{}", path.display());
    println!("{:<58} {:<5} {:<16} {:>11}", "name", "dtype", "shape", "elements");
    let mut total = 0;
    for info in &infos {
        println!("{:<58} {:<5} {:<16} {:>11}", info.name, info.dtype, format!("{:?}", info.shape), info.element_count());
        total += info.element_count();
    }
    println!();
    println!("{} tensors, {} parameters ({:.1} million)", infos.len(), total, total as f64 / 1e6);
    ExitCode::SUCCESS
}
