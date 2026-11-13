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

use super::json::Json;
use crate::mat::Mat;
use std::collections::BTreeMap;
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
    /// How many numbers the tensor holds: the product of its shape.
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
/// loading 150 MB of numbers.
pub fn read_header(path: &Path) -> io::Result<Vec<TensorInfo>> {
    let mut file = std::fs::File::open(path)?;
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)?;
    let length = u64::from_le_bytes(length_bytes);
    if length > MAX_HEADER_BYTES {
        return Err(bad(format!("header claims to be {length} bytes long")));
    }
    let mut header = vec![0u8; length as usize];
    file.read_exact(&mut header)?;
    parse_header_json(&header)
}

/// Split a whole file into (header entries, the data section).
pub fn parse_header(file: &[u8]) -> io::Result<(Vec<TensorInfo>, &[u8])> {
    if file.len() < 8 {
        return Err(bad("file is shorter than its 8-byte length prefix".into()));
    }
    let (length_bytes, rest) = file.split_at(8);
    let length = u64::from_le_bytes(length_bytes.try_into().expect("split_at(8) gives 8 bytes"));
    if length > rest.len() as u64 {
        return Err(bad(format!("header claims {length} bytes but only {} follow", rest.len())));
    }
    let (header, data) = rest.split_at(length as usize);
    Ok((parse_header_json(header)?, data))
}

/// Turn the header's JSON into a list of `TensorInfo`, in file order.
fn parse_header_json(header: &[u8]) -> io::Result<Vec<TensorInfo>> {
    let text = std::str::from_utf8(header).map_err(|_| bad("header is not UTF-8".into()))?;
    let json = Json::parse(text)?;
    let entries = json.as_object().ok_or_else(|| bad("header is not a JSON object".into()))?;
    let mut infos = Vec::new();
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue; // free-form notes about the file, not a tensor
        }
        infos.push(parse_entry(name, entry)?);
    }
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
    Ok(TensorInfo {
        name: name.to_string(),
        dtype: dtype.to_string(),
        shape: shape.ok_or_else(|| missing("shape"))?,
        start: start.ok_or_else(|| missing("data_offsets"))?,
        end: end.ok_or_else(|| missing("data_offsets"))?,
    })
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
        let bytes = std::fs::read(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        SafeTensors::from_bytes(&bytes)
    }

    /// Load from a file's contents already in memory.
    ///
    /// Every `F32` and `F16` tensor is converted to `f32`. Tensors of other
    /// types (integers, say) stay listed in `infos` but are not loaded.
    pub fn from_bytes(file: &[u8]) -> io::Result<SafeTensors> {
        let (infos, data) = parse_header(file)?;
        let mut tensors = BTreeMap::new();
        for info in &infos {
            // Never trust offsets from a file: check them before slicing.
            if info.start > info.end || info.end > data.len() {
                return Err(bad(format!("tensor {}: byte range {}..{} is outside the file", info.name, info.start, info.end)));
            }
            let bytes = &data[info.start..info.end];
            if let Some(size) = bytes_per_element(&info.dtype) {
                if bytes.len() != info.element_count() * size {
                    return Err(bad(format!("tensor {}: shape {:?} does not match {} bytes", info.name, info.shape, bytes.len())));
                }
            }
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
}
