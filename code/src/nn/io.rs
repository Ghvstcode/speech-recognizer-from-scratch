//! Saving and loading a model.
//!
//! # The file format
//!
//! A deliberately tiny binary format. All integers are unsigned and
//! little-endian; all weights are little-endian IEEE-754 `f32`.
//!
//! ```text
//! bytes 0..8    magic: the ASCII text "SPEECHNN"
//! u32           format version (currently 1)
//! u32           number of layers
//! then, for each layer:
//!   u8          layer tag (see the table below)
//!   u32 ...     the layer's sizes (see the table below)
//!   tensors     the layer's weight tensors, in the order of `Layer::params`
//!
//! each tensor:
//!   u32         rows
//!   u32         cols
//!   f32 ...     rows * cols values, row after row
//! ```
//!
//! | tag | layer    | sizes (each a u32)                        | tensors |
//! |-----|----------|-------------------------------------------|---------|
//! | 1   | Linear   | n_in, n_out                               | weight (n_in x n_out), bias (1 x n_out) |
//! | 2   | ReLU     | none                                      | none |
//! | 3   | Tanh     | none                                      | none |
//! | 4   | Sigmoid  | none                                      | none |
//! | 5   | Conv1d   | c_in, c_out, kernel, stride, padding      | weight (kernel*c_in x c_out), bias (1 x c_out) |
//! | 6   | Gru      | n_in, hidden                              | w_x (n_in x 3H), w_h (H x 3H), b_x (1 x 3H), b_h (1 x 3H) |
//! | 7   | BiGru    | n_in, hidden                              | the forward GRU's four tensors, then the backward GRU's four |
//! | 8   | MeanPool | none                                      | none |
//!
//! The shapes of the tensors are implied by the sizes, and stored anyway:
//! the redundancy lets the loader notice a corrupt or mismatched file instead
//! of quietly reading garbage.

use super::conv::Conv1d;
use super::gru::{BiGru, Gru};
use super::linear::Linear;
use super::model::{Layer, Model};
use crate::mat::Mat;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// The first eight bytes of every model file.
pub const MAGIC: &[u8; 8] = b"SPEECHNN";
/// The format version this code writes and reads.
pub const VERSION: u32 = 1;

impl Model {
    /// Write the model to a file, replacing it if it exists.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut file = BufWriter::new(std::fs::File::create(path)?);
        self.write_to(&mut file)?;
        file.flush()
    }

    /// Read a model from a file written by [`Model::save`].
    pub fn load(path: impl AsRef<Path>) -> io::Result<Model> {
        Model::read_from(&mut BufReader::new(std::fs::File::open(path)?))
    }

    /// Write the model, in the file format, to anything writable (a file, a
    /// `Vec<u8>`).
    pub fn write_to(&self, out: &mut impl Write) -> io::Result<()> {
        out.write_all(MAGIC)?;
        write_u32(out, VERSION)?;
        write_u32(out, self.layers.len() as u32)?;
        for layer in &self.layers {
            out.write_all(&[layer_tag(layer)])?;
            for size in layer_sizes(layer) {
                write_u32(out, size as u32)?;
            }
            for tensor in layer.params() {
                write_tensor(out, tensor)?;
            }
        }
        Ok(())
    }

    /// Read a model, in the file format, from anything readable.
    pub fn read_from(input: &mut impl Read) -> io::Result<Model> {
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad_data("not a model file (the first 8 bytes are not \"SPEECHNN\")"));
        }
        let version = read_u32(input)?;
        if version != VERSION {
            return Err(bad_data(format!("model file has format version {version}, this code reads version {VERSION}")));
        }
        let n_layers = read_u32(input)? as usize;
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            layers.push(read_layer(input)?);
        }
        // Check the layers fit together ourselves: `Model::new` would panic,
        // and a bad file is not a programming error.
        let mut width: Option<usize> = None;
        for (i, layer) in layers.iter().enumerate() {
            if let (Some(have), Some(want)) = (width, layer.in_features()) {
                if have != want {
                    return Err(bad_data(format!("layer {i} expects {want} columns but the layer before it produces {have}")));
                }
            }
            width = layer.out_features().or(width);
        }
        Ok(Model { layers })
    }
}

fn bad_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn layer_tag(layer: &Layer) -> u8 {
    match layer {
        Layer::Linear(_) => 1,
        Layer::Relu => 2,
        Layer::Tanh => 3,
        Layer::Sigmoid => 4,
        Layer::Conv1d(_) => 5,
        Layer::Gru(_) => 6,
        Layer::BiGru(_) => 7,
        Layer::MeanPool => 8,
    }
}

fn layer_sizes(layer: &Layer) -> Vec<usize> {
    match layer {
        Layer::Linear(l) => vec![l.n_in(), l.n_out()],
        Layer::Conv1d(c) => vec![c.c_in, c.c_out, c.kernel, c.stride, c.padding],
        Layer::Gru(g) => vec![g.n_in(), g.hidden()],
        Layer::BiGru(g) => vec![g.n_in(), g.hidden()],
        Layer::Relu | Layer::Tanh | Layer::Sigmoid | Layer::MeanPool => Vec::new(),
    }
}

fn read_layer(input: &mut impl Read) -> io::Result<Layer> {
    let mut tag = [0u8; 1];
    input.read_exact(&mut tag)?;
    Ok(match tag[0] {
        1 => {
            let (n_in, n_out) = (read_size(input)?, read_size(input)?);
            Layer::Linear(Linear { weight: read_tensor(input, n_in, n_out)?, bias: read_tensor(input, 1, n_out)? })
        }
        2 => Layer::Relu,
        3 => Layer::Tanh,
        4 => Layer::Sigmoid,
        5 => {
            let (c_in, c_out, kernel, stride, padding) = (read_size(input)?, read_size(input)?, read_size(input)?, read_size(input)?, read_size(input)?);
            if kernel == 0 || stride == 0 {
                return Err(bad_data("Conv1d with a kernel or stride of 0"));
            }
            let rows = kernel.checked_mul(c_in).ok_or_else(|| bad_data("Conv1d sizes overflow"))?;
            Layer::Conv1d(Conv1d { c_in, c_out, kernel, stride, padding, weight: read_tensor(input, rows, c_out)?, bias: read_tensor(input, 1, c_out)? })
        }
        6 => {
            let (n_in, hidden) = (read_size(input)?, read_size(input)?);
            Layer::Gru(read_gru(input, n_in, hidden)?)
        }
        7 => {
            let (n_in, hidden) = (read_size(input)?, read_size(input)?);
            Layer::BiGru(BiGru { fwd: read_gru(input, n_in, hidden)?, bwd: read_gru(input, n_in, hidden)? })
        }
        8 => Layer::MeanPool,
        other => return Err(bad_data(format!("unknown layer tag {other}"))),
    })
}

fn read_gru(input: &mut impl Read, n_in: usize, hidden: usize) -> io::Result<Gru> {
    let wide = hidden.checked_mul(3).ok_or_else(|| bad_data("Gru sizes overflow"))?;
    Ok(Gru { w_x: read_tensor(input, n_in, wide)?, w_h: read_tensor(input, hidden, wide)?, b_x: read_tensor(input, 1, wide)?, b_h: read_tensor(input, 1, wide)? })
}

fn write_u32(out: &mut impl Write, value: u32) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

fn read_u32(input: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_size(input: &mut impl Read) -> io::Result<usize> {
    Ok(read_u32(input)? as usize)
}

fn write_tensor(out: &mut impl Write, tensor: &Mat) -> io::Result<()> {
    write_u32(out, tensor.rows as u32)?;
    write_u32(out, tensor.cols as u32)?;
    for value in &tensor.data {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

/// Read one tensor and insist that it has the shape the layer's sizes imply.
fn read_tensor(input: &mut impl Read, rows: usize, cols: usize) -> io::Result<Mat> {
    let (file_rows, file_cols) = (read_size(input)?, read_size(input)?);
    if (file_rows, file_cols) != (rows, cols) {
        return Err(bad_data(format!("tensor is {file_rows}x{file_cols} in the file but the layer needs {rows}x{cols}")));
    }
    let count = rows.checked_mul(cols).ok_or_else(|| bad_data("tensor size overflows"))?;
    // Read value by value rather than trusting `count` with one big
    // allocation: a corrupt header then fails with "unexpected end of file"
    // instead of trying to reserve gigabytes.
    let mut data = Vec::new();
    let mut bytes = [0u8; 4];
    for _ in 0..count {
        input.read_exact(&mut bytes)?;
        data.push(f32::from_le_bytes(bytes));
    }
    Ok(Mat::from_vec(rows, cols, data))
}
