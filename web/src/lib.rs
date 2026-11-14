//! The recognizer of chapters 9 to 11, compiled to WebAssembly so that a web
//! page can run it.
//!
//!     rustup target add wasm32-unknown-unknown
//!     cargo build --release --target wasm32-unknown-unknown --manifest-path web/Cargo.toml
//!
//! There is no wasm-bindgen here: the page and this module talk through a
//! handful of plain functions and the module's memory. To hand us bytes, the
//! page asks for a buffer with [`sr_alloc`], writes into it and passes the
//! pointer back. Our answer is a string of JSON the page reads at
//! [`sr_output`].
//!
//! The `speech` crate has no `unsafe` in it. This file has a little, because
//! turning "a pointer and a length from JavaScript" into a slice cannot be
//! done any other way. It is all in the four small functions at the bottom.

use speech::data::pipeline::{Decoder, Recognizer};
use speech::decode::Alphabet;
use speech::dsp::frame::find_speech;
use speech::dsp::resample::resample;
use speech::nn::{argmax, Model};
use std::cell::RefCell;

/// The rate our models were trained at.
const SAMPLE_RATE: u32 = 16_000;

thread_local! {
    static RECOGNIZER: RefCell<Option<Recognizer>> = const { RefCell::new(None) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn set_output(text: String) -> usize {
    OUTPUT.with(|out| {
        let mut out = out.borrow_mut();
        *out = text.into_bytes();
        out.len()
    })
}

fn error(message: &str) -> usize {
    let message: String = message.chars().filter(|&c| c != '"' && c != '\\' && !c.is_control()).collect();
    set_output(format!("{{\"error\":\"{message}\"}}"))
}

/// Load the weights (`models/recognizer.bin`, as bytes). Returns 0 if they
/// are a recognizer; otherwise the reason is at [`sr_output`].
#[no_mangle]
pub extern "C" fn sr_load_model(ptr: *const u8, len: usize) -> i32 {
    let mut bytes = bytes_from_page(ptr, len);
    let model = match Model::read_from(&mut bytes) {
        Ok(model) => model,
        Err(e) => {
            error(&format!("could not read the model: {e}"));
            return 1;
        }
    };
    RECOGNIZER.with(|r| *r.borrow_mut() = Some(Recognizer::new(model)));
    0
}

/// Transcribe `len` mono samples recorded at `sample_rate`.
///
/// The steps are those of `speech listen`: bring the audio to 16 kHz, find
/// where the speech is (chapter 2's endpoint detector), run the network on
/// that stretch, and decode its output three ways (chapter 10).
///
/// Returns the length of the JSON answer at [`sr_output`]:
/// `{"samples":N,"start":S,"end":E,"frames":"__tt_w...","greedy":"..","beam":"..","lexicon":".."}`
/// where `start` and `end` are sample positions at 16 kHz and `frames` is the
/// network's best symbol in every output frame, blanks included.
#[no_mangle]
pub extern "C" fn sr_transcribe(ptr: *const f32, len: usize, sample_rate: u32) -> usize {
    if sample_rate == 0 {
        return error("the sample rate is zero");
    }
    let audio = resample(samples_from_page(ptr, len), sample_rate, SAMPLE_RATE);
    RECOGNIZER.with(|r| {
        let recognizer = r.borrow();
        let Some(recognizer) = recognizer.as_ref() else {
            return error("the model is not loaded");
        };
        let Some((start, end)) = find_speech(&audio, SAMPLE_RATE) else {
            return set_output(format!("{{\"samples\":{},\"silent\":true}}", audio.len()));
        };
        let log_probs = recognizer.log_probs(&audio[start..end]);
        let frames: String = (0..log_probs.rows).map(|t| Alphabet::symbol(argmax(log_probs.row(t)))).collect();
        let lexicon = &recognizer.lexicon;
        let text = |name: &str| {
            let decoder = Decoder::from_name(name).expect("a decoder we know");
            speech::data::pipeline::decode(&log_probs, &decoder, lexicon)
        };
        // The alphabet is a to z, the space and the blank's underscore, so
        // none of these strings needs escaping.
        set_output(format!(
            "{{\"samples\":{},\"start\":{start},\"end\":{end},\"frames\":\"{frames}\",\"greedy\":\"{}\",\"beam\":\"{}\",\"lexicon\":\"{}\"}}",
            audio.len(),
            text("greedy"),
            text("beam"),
            text("lexicon"),
        ))
    })
}

/// Where the last answer is. Valid until the next call into the module.
#[no_mangle]
pub extern "C" fn sr_output() -> *const u8 {
    OUTPUT.with(|out| out.borrow().as_ptr())
}

// ---------------------------------------------------------------------------
// The unsafe corner: memory shared with the page.
// ---------------------------------------------------------------------------

/// Reserve `len` bytes for the page to write into, aligned for `f32`.
/// The page gives them back with [`sr_free`].
#[no_mangle]
pub extern "C" fn sr_alloc(len: usize) -> *mut u8 {
    let Ok(layout) = std::alloc::Layout::from_size_align(len.max(1), 8) else {
        return std::ptr::null_mut();
    };
    // SAFETY: the layout has a non-zero size.
    unsafe { std::alloc::alloc(layout) }
}

/// Give back a buffer from [`sr_alloc`].
///
/// # Safety
///
/// `ptr` must have come from `sr_alloc(len)` with this same `len`, and must
/// not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn sr_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    if let Ok(layout) = std::alloc::Layout::from_size_align(len.max(1), 8) {
        // SAFETY: the caller's promise above; `sr_alloc` used this layout.
        unsafe { std::alloc::dealloc(ptr, layout) }
    }
}

fn bytes_from_page<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if ptr.is_null() || len == 0 {
        return &[];
    }
    // SAFETY: the page promises `ptr` is a buffer of at least `len` bytes
    // from `sr_alloc` that it has filled and will not free during this call.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

fn samples_from_page<'a>(ptr: *const f32, len: usize) -> &'a [f32] {
    if ptr.is_null() || len == 0 {
        return &[];
    }
    // SAFETY: as above, for `len` samples of four bytes each; `sr_alloc`
    // aligns its buffers to eight bytes, which is enough for `f32`.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}
