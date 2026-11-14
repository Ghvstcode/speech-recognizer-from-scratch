//! Reading a `.safetensors` weights file by hand.
//!
//! A trained model is, physically, a few hundred named arrays of numbers.
//! The safetensors format stores them about as simply as it can be done:
//!
//! ```text
//! bytes 0..8      N, the length of the header, as a little-endian u64
//! bytes 8..8+N    the header: one JSON object
//! bytes 8+N..     the data: every tensor's numbers, back to back
//! ```
//!
//! The header maps each tensor's name to its element type, its shape and the
//! byte range its numbers occupy *within the data section*:
//!
//! ```text
//! { "model.encoder.conv1.bias": {"dtype":"F32", "shape":[384], "data_offsets":[118207488, 118209024]},
//!   ...,
//!   "__metadata__": {"format": "pt"} }
//! ```
//!
//! Numbers are little-endian and tensors are stored row-major (the last index
//! moves fastest), which is exactly the layout of our `Mat`. So "loading the
//! model" is: read 8 bytes, parse some JSON, and reinterpret byte ranges as
//! `f32`s. There is no code in the file, which is the "safe" in the name
//! (the older pickle-based formats could run arbitrary Python on load).
//!
//! No code does not mean no checking. Every number in the header is a claim
//! about the data section, and this reader accepts a file only if all the
//! claims hold together:
//!
//! - the header is at most 100 MB, starts with `{`, and names no tensor twice;
//! - `__metadata__`, if present, maps strings to strings;
//! - every element type is one the format defines;
//! - every byte range is exactly as long as the tensor's shape and type call
//!   for, with the multiplications checked for overflow;
//! - laid end to end, the byte ranges cover the data section exactly: no two
//!   overlap, there is no gap between them, and nothing is left over.
//!
//! The last rule is the format's own. It means a file can never expand into
//! more tensor data than it has bytes, and has nowhere to hide anything else.

use super::json::Json;
use crate::mat::Mat;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read};
use std::path::Path;

/// A header larger than this is certainly not a real model file.
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// One line of the header: where a tensor lives and what it looks like.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    /// The tensor's name, e.g. `model.encoder.conv1.weight`.
    pub name: String,
    /// Element type as spelled in the file: `F32`, `F16`, `I64`, ...
    pub dtype: String,
    /// Size along each axis. A plain number has shape `[]`.
    pub shape: Vec<usize>,
    /// First byte of the tensor, counted from the start of the data section.
    pub start: usize,
    /// One past the last byte, counted from the start of the data section.
    pub end: usize,
}

impl TensorInfo {
    /// How many numbers the tensor holds: the product of its shape. (For an
    /// entry read from a file, `parse_entry` has already checked that this
    /// product fits in a `usize`.)
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A tensor loaded into memory as `f32`s, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Size along each axis.
    pub shape: Vec<usize>,
    /// The numbers, last axis fastest.
    pub data: Vec<f32>,
}

impl Tensor {
    /// Turn a two-dimensional tensor into a `Mat` without copying.
    pub fn into_mat(self) -> io::Result<Mat> {
        match self.shape[..] {
            [rows, cols] => Ok(Mat::from_vec(rows, cols, self.data)),
            _ => Err(bad(format!("expected a 2-D tensor, found shape {:?}", self.shape))),
        }
    }

    /// Turn a one-dimensional tensor into a plain vector.
    pub fn into_vec(self) -> io::Result<Vec<f32>> {
        match self.shape[..] {
            [_] => Ok(self.data),
            _ => Err(bad(format!("expected a 1-D tensor, found shape {:?}", self.shape))),
        }
    }
}

/// Every floating point tensor of a safetensors file, loaded into memory.
pub struct SafeTensors {
    /// The header entries in file order (all of them, whatever their type).
    pub infos: Vec<TensorInfo>,
    tensors: BTreeMap<String, Tensor>,
}

fn bad(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("safetensors: {message}"))
}

/// Read just the header of a file: enough to list what is inside without
/// loading 150 MB of numbers. The header is checked against the size of the
/// file, so a listing that comes back describes data that is really there.
pub fn read_header(path: &Path) -> io::Result<Vec<TensorInfo>> {
    let mut file = std::fs::File::open(path)?;
    let file_bytes = file.metadata()?.len();
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)?;
    let length = u64::from_le_bytes(length_bytes);
    if length > MAX_HEADER_BYTES {
        return Err(bad(format!("header claims to be {length} bytes long")));
    }
    let mut header = vec![0u8; length as usize];
    file.read_exact(&mut header)?;
    // Both reads succeeded, so the file holds at least `8 + length` bytes.
    let data_bytes = file_bytes.saturating_sub(8 + length);
    parse_header_json(&header, data_bytes)
}

/// Split a whole file into (header entries, the data section).
pub fn parse_header(file: &[u8]) -> io::Result<(Vec<TensorInfo>, &[u8])> {
    if file.len() < 8 {
        return Err(bad("file is shorter than its 8-byte length prefix".into()));
    }
    let (length_bytes, rest) = file.split_at(8);
    let length = u64::from_le_bytes(length_bytes.try_into().expect("split_at(8) gives 8 bytes"));
    if length > MAX_HEADER_BYTES {
        return Err(bad(format!("header claims to be {length} bytes long")));
    }
    if length > rest.len() as u64 {
        return Err(bad(format!("header claims {length} bytes but only {} follow", rest.len())));
    }
    let (header, data) = rest.split_at(length as usize);
    Ok((parse_header_json(header, data.len() as u64)?, data))
}

/// Turn the header's JSON into a list of `TensorInfo`, in file order, and
/// check it against a data section of `data_bytes` bytes.
fn parse_header_json(header: &[u8], data_bytes: u64) -> io::Result<Vec<TensorInfo>> {
    if header.first() != Some(&b'{') {
        return Err(bad("header does not start with `{`".into()));
    }
    let text = std::str::from_utf8(header).map_err(|_| bad("header is not UTF-8".into()))?;
    let json = Json::parse(text)?;
    let entries = json.as_object().ok_or_else(|| bad("header is not a JSON object".into()))?;
    let mut infos = Vec::new();
    let mut seen = BTreeSet::new();
    for (name, entry) in entries {
        // Our `Json` keeps repeated keys; the format forbids them.
        if !seen.insert(name.as_str()) {
            return Err(bad(format!("the name `{name}` appears twice")));
        }
        if name == "__metadata__" {
            // Free-form notes about the file, not a tensor. The format
            // allows text here and nothing else.
            let notes = entry.as_object().ok_or_else(|| bad("`__metadata__` is not an object".into()))?;
            if notes.iter().any(|(_, value)| value.as_str().is_none()) {
                return Err(bad("`__metadata__` holds something other than strings".into()));
            }
            continue;
        }
        infos.push(parse_entry(name, entry)?);
    }
    check_layout(&infos, data_bytes)?;
    Ok(infos)
}

fn parse_entry(name: &str, entry: &Json) -> io::Result<TensorInfo> {
    let missing = |field: &str| bad(format!("tensor {name}: missing or malformed `{field}`"));
    let dtype = entry.get("dtype").and_then(Json::as_str).ok_or_else(|| missing("dtype"))?;
    let shape_json = entry.get("shape").and_then(Json::as_array).ok_or_else(|| missing("shape"))?;
    let shape: Option<Vec<usize>> = shape_json.iter().map(Json::as_usize).collect();
    let offsets = entry.get("data_offsets").and_then(Json::as_array).ok_or_else(|| missing("data_offsets"))?;
    let (start, end) = match offsets {
        [start, end] => (start.as_usize(), end.as_usize()),
        _ => (None, None),
    };
    let info = TensorInfo {
        name: name.to_string(),
        dtype: dtype.to_string(),
        shape: shape.ok_or_else(|| missing("shape"))?,
        start: start.ok_or_else(|| missing("data_offsets"))?,
        end: end.ok_or_else(|| missing("data_offsets"))?,
    };
    if bytes_per_element(&info.dtype).is_none() {
        return Err(bad(format!("tensor {name}: unknown dtype `{}`", info.dtype)));
    }
    if info.start > info.end {
        return Err(bad(format!("tensor {name}: byte range {}..{} runs backwards", info.start, info.end)));
    }
    if byte_count(&info.dtype, &info.shape) != Some(info.end - info.start) {
        return Err(bad(format!("tensor {name}: shape {:?} of {} does not match {} bytes", info.shape, info.dtype, info.end - info.start)));
    }
    Ok(info)
}

/// How many bytes a tensor of this type and shape occupies. `None` if the
/// type is unknown or the answer does not fit in a `usize`.
///
/// The shape comes from the file, so the product is computed with
/// `checked_mul`. Plain `*` on two sizes of 2^32 wraps around to 0 in a
/// release build (and panics in a debug build), and a wrapped product would
/// let a shape of 2^64 elements pass for an empty tensor.
fn byte_count(dtype: &str, shape: &[usize]) -> Option<usize> {
    let size = bytes_per_element(dtype)?;
    // Multiply the non-zero sizes even when a zero makes the tensor empty,
    // so that `[0, 2^40, 2^40]` is refused like `[2^40, 2^40]`.
    let bytes = shape.iter().filter(|&&n| n != 0).try_fold(size, |bytes, &n| bytes.checked_mul(n))?;
    Some(if shape.contains(&0) { 0 } else { bytes })
}

/// Check that the tensors' byte ranges tile the data section: sorted by
/// where they start, each begins where the one before ended, the first at
/// byte 0 and the last ending at `data_bytes`.
fn check_layout(infos: &[TensorInfo], data_bytes: u64) -> io::Result<()> {
    let mut by_offset: Vec<&TensorInfo> = infos.iter().collect();
    by_offset.sort_by_key(|info| (info.start, info.end));
    let mut covered = 0;
    for info in by_offset {
        if info.start < covered {
            return Err(bad(format!("tensor {}: byte range {}..{} overlaps another tensor", info.name, info.start, info.end)));
        }
        if info.start > covered {
            return Err(bad(format!("bytes {covered}..{} of the data belong to no tensor", info.start)));
        }
        covered = info.end;
    }
    if covered as u64 != data_bytes {
        return Err(bad(format!("the tensors cover {covered} bytes of data but the file holds {data_bytes}")));
    }
    Ok(())
}

/// Decode one IEEE 754 half-precision float (16 bits: 1 sign, 5 exponent,
/// 10 fraction). Some checkpoints are stored this way to halve their size.
/// `tiny.en` is not, but a loader that only handles the file in front of it
/// is a loader that breaks on the next one.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x3ff) as f32 / 1024.0;
    let magnitude = match exponent {
        // Exponent 0: "subnormal" numbers, no implicit leading 1.
        0 => fraction * 2f32.powi(-14),
        // Exponent all ones: infinity (fraction 0) or not-a-number.
        31 if fraction == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + fraction) * 2f32.powi(exponent - 15),
    };
    sign * magnitude
}

/// Reinterpret a tensor's bytes as `f32`s. `None` for types we do not load.
fn decode_floats(dtype: &str, bytes: &[u8]) -> Option<Vec<f32>> {
    match dtype {
        "F32" => Some(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()),
        "F16" => Some(bytes.as_chunks::<2>().0.iter().map(|b| f16_to_f32(u16::from_le_bytes(*b))).collect()),
        _ => None,
    }
}

/// Bytes per element for the types the format defines.
fn bytes_per_element(dtype: &str) -> Option<usize> {
    match dtype {
        "BOOL" | "U8" | "I8" => Some(1),
        "F16" | "I16" | "U16" | "BF16" => Some(2),
        "F32" | "I32" | "U32" => Some(4),
        "F64" | "I64" | "U64" => Some(8),
        _ => None,
    }
}

impl SafeTensors {
    /// Load a file from disk.
    pub fn read(path: &Path) -> io::Result<SafeTensors> {
        let with_path = |e: io::Error| io::Error::new(e.kind(), format!("{}: {e}", path.display()));
        // Check the header against the file's size first, so that a file
        // that fails the checks is turned away before we read all of it.
        read_header(path).map_err(with_path)?;
        let bytes = std::fs::read(path).map_err(with_path)?;
        SafeTensors::from_bytes(&bytes).map_err(with_path)
    }

    /// Load from a file's contents already in memory.
    ///
    /// Every `F32` and `F16` tensor is converted to `f32`. Tensors of other
    /// types (integers, say) stay listed in `infos` but are not loaded.
    pub fn from_bytes(file: &[u8]) -> io::Result<SafeTensors> {
        let (infos, data) = parse_header(file)?;
        let mut tensors = BTreeMap::new();
        for info in &infos {
            // `parse_header` has checked every byte range against `data`,
            // but slicing with `get` costs nothing and cannot panic.
            let bytes = data.get(info.start..info.end).ok_or_else(|| bad(format!("tensor {}: byte range is outside the file", info.name)))?;
            if let Some(values) = decode_floats(&info.dtype, bytes) {
                tensors.insert(info.name.clone(), Tensor { shape: info.shape.clone(), data: values });
            }
        }
        Ok(SafeTensors { infos, tensors })
    }

    /// Borrow a loaded tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Whether a floating point tensor of this name is (still) loaded.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Move a tensor out of the collection. The model takes ownership of each
    /// tensor this way, so the 150 MB of weights exist in memory only once.
    pub fn take(&mut self, name: &str) -> io::Result<Tensor> {
        self.tensors.remove(name).ok_or_else(|| bad(format!("no floating point tensor named `{name}`")))
    }

    /// `take` a 2-D tensor as a `Mat`.
    pub fn take_mat(&mut self, name: &str) -> io::Result<Mat> {
        self.take(name)?.into_mat().map_err(|e| bad(format!("`{name}`: {e}")))
    }

    /// `take` a 1-D tensor as a `Vec<f32>`.
    pub fn take_vec(&mut self, name: &str) -> io::Result<Vec<f32>> {
        self.take(name)?.into_vec().map_err(|e| bad(format!("`{name}`: {e}")))
    }

    /// Names of the tensors that are loaded and have not been taken yet.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a safetensors file in memory from (name, dtype, shape, raw bytes).
    pub(crate) fn build_file(tensors: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
        let mut header = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            let shape: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            header.push_str(&format!(
                ",\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
                shape.join(","),
                data.len()
            ));
        }
        header.push('}');
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&data);
        file
    }

    /// The little-endian bytes of a list of `f32`s.
    pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn loads_f32_tensors_with_shapes() {
        let file = build_file(&[
            ("w", "F32", &[2, 3], f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            ("b", "F32", &[2], f32_bytes(&[-0.5, 0.25])),
        ]);
        let mut st = SafeTensors::from_bytes(&file).unwrap();
        assert_eq!(st.infos.len(), 2);
        assert_eq!(st.infos[0].name, "w");
        assert_eq!((st.infos[1].start, st.infos[1].end), (24, 32));
        assert_eq!(st.get("b").unwrap().data, vec![-0.5, 0.25]);
        let w = st.take_mat("w").unwrap();
        assert_eq!((w.rows, w.cols), (2, 3));
        assert_eq!(w[(1, 0)], 4.0); // row-major: element (1,0) is the fourth number
        assert!(st.take("w").is_err(), "a tensor can only be taken once");
        assert_eq!(st.take_vec("b").unwrap(), vec![-0.5, 0.25]);
    }

    #[test]
    fn decodes_half_precision() {
        // Hand-assembled bit patterns: sign | exponent | fraction.
        assert_eq!(f16_to_f32(0x3C00), 1.0); // 0 01111 0000000000
        assert_eq!(f16_to_f32(0xC000), -2.0); // 1 10000 0000000000
        assert_eq!(f16_to_f32(0x3E00), 1.5); // 0 01111 1000000000
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // largest finite half
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());

        let file = build_file(&[("h", "F16", &[2], vec![0x00, 0x3C, 0x00, 0xC0])]);
        let st = SafeTensors::from_bytes(&file).unwrap();
        assert_eq!(st.get("h").unwrap().data, vec![1.0, -2.0]);
    }

    #[test]
    fn lists_but_does_not_load_integer_tensors() {
        let file = build_file(&[("ids", "I64", &[1], vec![7, 0, 0, 0, 0, 0, 0, 0])]);
        let st = SafeTensors::from_bytes(&file).unwrap();
        assert_eq!(st.infos[0].dtype, "I64");
        assert!(!st.contains("ids"));
    }

    #[test]
    fn rejects_inconsistent_files() {
        // Shape says 3 numbers, data holds 2.
        let wrong_shape = build_file(&[("w", "F32", &[3], f32_bytes(&[1.0, 2.0]))]);
        assert!(SafeTensors::from_bytes(&wrong_shape).is_err());
        // Data cut short.
        let mut truncated = build_file(&[("w", "F32", &[2], f32_bytes(&[1.0, 2.0]))]);
        truncated.truncate(truncated.len() - 1);
        assert!(SafeTensors::from_bytes(&truncated).is_err());
        // Header length pointing past the end of the file.
        assert!(SafeTensors::from_bytes(&[255, 0, 0, 0, 0, 0, 0, 0, b'{', b'}']).is_err());
        assert!(SafeTensors::from_bytes(&[1, 2, 3]).is_err());
    }

    /// A file from a header written out by hand and a data section.
    fn raw_file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(data);
        file
    }

    /// The error message for a file that must be refused.
    fn refusal(header: &str, data: &[u8]) -> String {
        match SafeTensors::from_bytes(&raw_file(header, data)) {
            Ok(_) => panic!("accepted a malformed file: {header}"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn rejects_shapes_whose_size_overflows() {
        // 2^32 x 2^32 elements: the product is 2^64, which wraps to 0 in a
        // `usize` and would match the empty byte range if we let it wrap.
        let message = refusal(r#"{"w":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#, &[]);
        assert!(message.contains("does not match"), "{message}");
        // The element count fits (2^62) but four bytes each does not.
        refusal(r#"{"w":{"dtype":"F32","shape":[2147483648,2147483648],"data_offsets":[0,0]}}"#, &[]);
        // A zero in the shape must not hide absurd sizes next to it.
        refusal(r#"{"w":{"dtype":"F32","shape":[0,4294967296,4294967296],"data_offsets":[0,0]}}"#, &[]);
        refusal(r#"{"w":{"dtype":"F32","shape":[4294967296,4294967296,0],"data_offsets":[0,0]}}"#, &[]);
        // Sizes and offsets too large for an `f64` to hold exactly.
        refusal(r#"{"w":{"dtype":"F32","shape":[1e30],"data_offsets":[0,4]}}"#, &[0; 4]);
        refusal(r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[1e30,1e30]}}"#, &[0; 4]);

        assert_eq!(byte_count("F32", &[2, 3]), Some(24));
        assert_eq!(byte_count("F64", &[]), Some(8)); // a plain number
        assert_eq!(byte_count("F32", &[5, 0]), Some(0));
        assert_eq!(byte_count("F32", &[usize::MAX]), None);
        assert_eq!(byte_count("U8", &[usize::MAX]), Some(usize::MAX));
        assert_eq!(byte_count("U8", &[usize::MAX, 2]), None);
        assert_eq!(byte_count("F99", &[1]), None);
    }

    #[test]
    fn rejects_tensors_that_do_not_tile_the_data() {
        let a = r#""a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}"#;
        // Two tensors claiming the same bytes.
        let message = refusal(&format!(r#"{{{a},"b":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#), &[0; 4]);
        assert!(message.contains("overlaps"), "{message}");
        // Partly overlapping.
        refusal(&format!(r#"{{{a},"b":{{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}}}"#), &[0; 6]);
        // A gap before the first tensor, and one between two tensors.
        let message = refusal(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#, &[0; 8]);
        assert!(message.contains("belong to no tensor"), "{message}");
        refusal(&format!(r#"{{{a},"b":{{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}}}"#), &[0; 12]);
        // Bytes after the last tensor.
        let message = refusal(&format!("{{{a}}}"), &[0; 8]);
        assert!(message.contains("cover 4 bytes"), "{message}");
        // A range that runs backwards, and one past the end of the data.
        refusal(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#, &[0; 4]);
        refusal(r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, &[0; 4]);

        // Header order need not be data order, and empty tensors are fine.
        let swapped = raw_file(
            r#"{"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"e":{"dtype":"F32","shape":[0],"data_offsets":[4,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &f32_bytes(&[1.0, 2.0]),
        );
        let st = SafeTensors::from_bytes(&swapped).unwrap();
        assert_eq!(st.get("a").unwrap().data, vec![1.0]);
        assert_eq!(st.get("b").unwrap().data, vec![2.0]);
        assert!(st.get("e").unwrap().data.is_empty());
        // A file with no tensors and no data is a valid, empty file.
        assert!(SafeTensors::from_bytes(&raw_file("{}", &[])).unwrap().infos.is_empty());
    }

    #[test]
    fn rejects_malformed_headers() {
        let a = r#""a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}"#;
        // The same name twice, even with ranges that would otherwise tile.
        let message = refusal(&format!(r#"{{{a},"a":{{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}}}"#), &[0; 8]);
        assert!(message.contains("appears twice"), "{message}");
        // An element type the format does not define.
        let message = refusal(r#"{"a":{"dtype":"F99","shape":[1],"data_offsets":[0,4]}}"#, &[0; 4]);
        assert!(message.contains("unknown dtype"), "{message}");
        // Not an object, not starting with `{`, missing or mistyped fields.
        refusal("[]", &[]);
        refusal(r#" {"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, &[0; 4]);
        refusal(r#"{"a":{"dtype":"F32","shape":[1]}}"#, &[0; 4]);
        refusal(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0]}}"#, &[0; 4]);
        refusal(r#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#, &[0; 4]);
        refusal(r#"{"a":{"dtype":"F32","shape":[1.5],"data_offsets":[0,4]}}"#, &[0; 4]);
        refusal(r#"{"a":{"dtype":"F32","shape":[01],"data_offsets":[0,4]}}"#, &[0; 4]);
        refusal(r#"{"a":7}"#, &[]);
        // `__metadata__` is for text only, and may appear once.
        refusal(r#"{"__metadata__":{"format":7}}"#, &[]);
        refusal(r#"{"__metadata__":[]}"#, &[]);
        refusal(r#"{"__metadata__":{},"__metadata__":{}}"#, &[]);
        assert!(SafeTensors::from_bytes(&raw_file(r#"{"__metadata__":{"format":"pt"}}"#, &[])).is_ok());
        // Trailing spaces after the header's closing brace are allowed.
        assert!(SafeTensors::from_bytes(&raw_file(&format!("{{{a}}}   "), &[0; 4])).is_ok());
    }

    #[test]
    fn never_panics_on_damaged_files() {
        // Take a good file and damage it in every simple way: cut it short at
        // each length, and overwrite each byte in turn. Every result must be
        // a clean `Ok` or `Err`; a panic fails the test.
        let good = build_file(&[
            ("w", "F32", &[2, 2], f32_bytes(&[1.0, 2.0, 3.0, 4.0])),
            ("h", "F16", &[2], vec![0x00, 0x3C, 0x00, 0xC0]),
        ]);
        assert!(SafeTensors::from_bytes(&good).is_ok());
        for length in 0..good.len() {
            assert!(SafeTensors::from_bytes(&good[..length]).is_err(), "accepted a file cut to {length} bytes");
        }
        for position in 0..good.len() {
            for replacement in [0x00, b'0', b'9', b'-', b'e', b'"', b'[', b'{', b',', 0xFF] {
                let mut damaged = good.clone();
                damaged[position] = replacement;
                let _ = SafeTensors::from_bytes(&damaged);
            }
        }
    }

    #[test]
    fn read_header_checks_the_file_size() {
        let dir = std::env::temp_dir().join(format!("speech-safetensors-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = build_file(&[("w", "F32", &[2], f32_bytes(&[1.0, 2.0]))]);
        let path = dir.join("good.safetensors");
        std::fs::write(&path, &good).unwrap();
        assert_eq!(read_header(&path).unwrap()[0].shape, vec![2]);
        assert_eq!(SafeTensors::read(&path).unwrap().get("w").unwrap().data, vec![1.0, 2.0]);
        // One byte of data missing, and one byte too many.
        std::fs::write(&path, &good[..good.len() - 1]).unwrap();
        assert!(read_header(&path).is_err());
        assert!(SafeTensors::read(&path).is_err());
        let mut longer = good.clone();
        longer.push(0);
        std::fs::write(&path, &longer).unwrap();
        assert!(read_header(&path).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
