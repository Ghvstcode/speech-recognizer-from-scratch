//! A tiny matrix type.
//!
//! Almost everything in a speech recognizer is a grid of numbers: a
//! spectrogram is (frames x frequencies), a layer's weights are
//! (outputs x inputs), the network's output is (frames x characters).
//! `Mat` is that grid: a row-major `Vec<f32>` plus its shape.

use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    /// Row-major storage: element (r, c) lives at `data[r * cols + c]`.
    pub data: Vec<f32>,
}

impl Mat {
    /// A `rows x cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Wrap an existing buffer. Panics if the length does not match the shape.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "buffer does not match {rows}x{cols}");
        Mat { rows, cols, data }
    }

    /// Build a matrix by calling `f(row, col)` for every element.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f32) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Mat { rows, cols, data }
    }

    /// Stack equally long rows into a matrix.
    pub fn from_rows(rows: &[Vec<f32>]) -> Self {
        let cols = rows.first().map_or(0, |r| r.len());
        let mut data = Vec::with_capacity(rows.len() * cols);
        for r in rows {
            assert_eq!(r.len(), cols, "ragged rows");
            data.extend_from_slice(r);
        }
        Mat { rows: rows.len(), cols, data }
    }

    /// Row `r` as a slice.
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Row `r` as a mutable slice.
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// The transpose: rows become columns.
    pub fn transpose(&self) -> Mat {
        Mat::from_fn(self.cols, self.rows, |r, c| self[(c, r)])
    }

    /// Apply `f` to every element, returning a new matrix.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Mat {
        Mat { rows: self.rows, cols: self.cols, data: self.data.iter().map(|&x| f(x)).collect() }
    }

    /// `self += other`, element by element.
    pub fn add_assign(&mut self, other: &Mat) {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols), "shape mismatch");
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }

    /// Multiply every element by `k`.
    pub fn scale(&mut self, k: f32) {
        for a in self.data.iter_mut() {
            *a *= k;
        }
    }

    /// Set every element to `value`.
    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    /// Matrix product `self * other`, on the calling thread.
    ///
    /// The loop order is i-k-j rather than the textbook i-j-k. The innermost
    /// loop then walks two rows left to right, which is kind to the cache and
    /// lets the compiler turn it into SIMD instructions.
    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows, "matmul: {}x{} * {}x{}", self.rows, self.cols, other.rows, other.cols);
        let mut out = Mat::zeros(self.rows, other.cols);
        for (a_row, out_row) in self.data.chunks_exact(self.cols.max(1)).zip(out.data.chunks_exact_mut(other.cols.max(1))) {
            matmul_row(a_row, other, out_row);
        }
        out
    }

    /// Matrix product `self * other`, splitting the rows of the result across
    /// all available CPU cores. Worth it for big products only (the Whisper
    /// chapter); for small ones the cost of starting threads dominates.
    pub fn matmul_par(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows, "matmul: {}x{} * {}x{}", self.rows, self.cols, other.rows, other.cols);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        if threads == 1 || self.rows < 2 * threads || self.cols == 0 || other.cols == 0 {
            return self.matmul(other);
        }
        let mut out = Mat::zeros(self.rows, other.cols);
        let rows_per_chunk = self.rows.div_ceil(threads);
        std::thread::scope(|scope| {
            let a_chunks = self.data.chunks(rows_per_chunk * self.cols);
            let out_chunks = out.data.chunks_mut(rows_per_chunk * other.cols);
            for (a_chunk, out_chunk) in a_chunks.zip(out_chunks) {
                scope.spawn(move || {
                    for (a_row, out_row) in a_chunk.chunks_exact(self.cols).zip(out_chunk.chunks_exact_mut(other.cols)) {
                        matmul_row(a_row, other, out_row);
                    }
                });
            }
        });
        out
    }

    /// `self^T * other` without building the transpose.
    /// Shows up in backprop: the gradient of a weight matrix is inputs^T * deltas.
    pub fn t_matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows, "t_matmul: row counts differ");
        let mut out = Mat::zeros(self.cols, other.cols);
        for r in 0..self.rows {
            let (a_row, b_row) = (self.row(r), other.row(r));
            for (i, &a) in a_row.iter().enumerate() {
                if a != 0.0 {
                    for (o, &b) in out.row_mut(i).iter_mut().zip(b_row) {
                        *o += a * b;
                    }
                }
            }
        }
        out
    }

    /// `self * other^T` without building the transpose: every output element
    /// is a dot product of a row of `self` with a row of `other`.
    pub fn matmul_t(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.cols, "matmul_t: column counts differ");
        Mat::from_fn(self.rows, other.rows, |r, c| dot(self.row(r), other.row(c)))
    }
}

/// One row of a matrix product: `out_row = a_row * b`.
fn matmul_row(a_row: &[f32], b: &Mat, out_row: &mut [f32]) {
    for (k, &a) in a_row.iter().enumerate() {
        if a == 0.0 {
            continue; // common after a ReLU, and free to skip
        }
        for (o, &bv) in out_row.iter_mut().zip(b.row(k)) {
            *o += a * bv;
        }
    }
}

/// Dot product of two equally long slices.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: lengths differ");
    // Four running sums instead of one: floating point addition is not
    // associative, so the compiler will not reorder a single sum into SIMD
    // on its own. Splitting it by hand gives it permission.
    let mut acc = [0.0f32; 4];
    let (a4, b4) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (a4.remainder(), b4.remainder());
    for (x, y) in a4.zip(b4) {
        for i in 0..4 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum = acc[0] + acc[1] + acc[2] + acc[3];
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

impl Index<(usize, usize)> for Mat {
    type Output = f32;
    fn index(&self, (r, c): (usize, usize)) -> &f32 {
        debug_assert!(r < self.rows && c < self.cols);
        &self.data[r * self.cols + c]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f32 {
        debug_assert!(r < self.rows && c < self.cols);
        &mut self.data[r * self.cols + c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a() -> Mat {
        Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }
    fn b() -> Mat {
        Mat::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    }

    #[test]
    fn matmul_matches_hand_computation() {
        assert_eq!(a().matmul(&b()), Mat::from_vec(2, 2, vec![58.0, 64.0, 139.0, 154.0]));
    }

    #[test]
    fn transposed_products_agree_with_explicit_transpose() {
        let (a, b) = (a(), b());
        assert_eq!(a.t_matmul(&b.transpose()), a.transpose().matmul(&b.transpose()));
        assert_eq!(a.matmul_t(&b.transpose()), a.matmul(&b));
    }

    #[test]
    fn parallel_matmul_agrees_with_serial() {
        let mut rng = crate::rng::Rng::new(5);
        let x = Mat::from_fn(301, 67, |_, _| rng.normal());
        let y = Mat::from_fn(67, 45, |_, _| rng.normal());
        let (s, p) = (x.matmul(&y), x.matmul_par(&y));
        for (u, v) in s.data.iter().zip(&p.data) {
            assert!((u - v).abs() < 1e-4);
        }
    }

    #[test]
    fn dot_handles_lengths_not_divisible_by_four() {
        let x: Vec<f32> = (1..=7).map(|v| v as f32).collect();
        assert_eq!(dot(&x, &x), 140.0);
    }
}
