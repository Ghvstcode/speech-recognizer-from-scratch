//! The handful of inner loops the layers are built from.
//!
//! `Mat` already has whole-matrix products. The loops here are the ones we
//! need *inside* a time loop, where we must not allocate a fresh matrix for
//! every frame: they all write into a buffer the caller already owns.
//!
//! Each one is written so that the innermost loop walks a few slices side by
//! side doing `out += a * b`. That pattern has no dependency between one
//! element and the next, so the compiler turns it into SIMD instructions
//! without being asked. (A dot product, by contrast, funnels everything into
//! one running sum, and each addition has to wait for the one before it.)

use crate::mat::Mat;

/// `out += k[0] * rows[0] + k[1] * rows[1] + k[2] * rows[2] + k[3] * rows[3]`.
///
/// The one loop where almost all of the training time is spent. Adding four
/// scaled rows per pass, instead of one, is the same arithmetic, but `out`
/// is read and written once per four rows rather than once per row, and that
/// reading and writing is most of what the one-row loop costs.
fn add_four_scaled_rows(out: &mut [f32], k: [f32; 4], rows: [&[f32]; 4]) {
    let [r0, r1, r2, r3] = rows;
    for ((((o, a), b), c), d) in out.iter_mut().zip(r0).zip(r1).zip(r2).zip(r3) {
        *o += k[0] * a + k[1] * b + k[2] * c + k[3] * d;
    }
}

/// Cut a slice holding four rows of `cols` values into its four rows.
fn four_rows(block: &[f32], cols: usize) -> [&[f32]; 4] {
    let (r0, rest) = block.split_at(cols);
    let (r1, rest) = rest.split_at(cols);
    let (r2, r3) = rest.split_at(cols);
    [r0, r1, r2, r3]
}

/// `out += x * w`, where `x` is a row vector with one entry per row of `w`.
///
/// This is one row of a matrix product. Read it as: "the result is a mix of
/// the rows of `w`, and `x` says how much of each row to take".
pub fn add_vec_mat(x: &[f32], w: &Mat, out: &mut [f32]) {
    assert_eq!(x.len(), w.rows, "add_vec_mat: vector has {} entries, matrix has {} rows", x.len(), w.rows);
    assert_eq!(out.len(), w.cols, "add_vec_mat: output has {} entries, matrix has {} columns", out.len(), w.cols);
    if w.cols == 0 {
        return;
    }
    // Take the rows of `w` four at a time...
    for (block, w_rows) in w.data.chunks_exact(4 * w.cols).enumerate() {
        let xs = &x[4 * block..4 * block + 4];
        add_four_scaled_rows(out, [xs[0], xs[1], xs[2], xs[3]], four_rows(w_rows, w.cols));
    }
    // ...and then whatever is left when the row count is not a multiple of four.
    let done = w.rows - w.rows % 4;
    for (&xk, w_row) in x[done..].iter().zip(w.data[done * w.cols..].chunks_exact(w.cols)) {
        for (o, &wv) in out.iter_mut().zip(w_row) {
            *o += xk * wv;
        }
    }
}

/// `out += a^T * b` for two row vectors: the *outer product*.
///
/// Element `(i, j)` of the result is `a[i] * b[j]`. This is the gradient of a
/// weight matrix for a single frame: (the layer's input) outer (the gradient
/// at the layer's output).
pub fn add_outer(a: &[f32], b: &[f32], out: &mut Mat) {
    assert_eq!((out.rows, out.cols), (a.len(), b.len()), "add_outer: shape mismatch");
    if b.is_empty() {
        return;
    }
    for (&ai, out_row) in a.iter().zip(out.data.chunks_exact_mut(b.len())) {
        for (o, &bj) in out_row.iter_mut().zip(b) {
            *o += ai * bj;
        }
    }
}

/// The matrix product `a * b`, one row at a time through [`add_vec_mat`].
///
/// Same result as [`Mat::matmul`], which is the version to read first if
/// you want to see how a matrix product works; this one exists so that the
/// layers' big products get the four-rows-at-a-time loop too.
pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.cols, b.rows, "matmul: {}x{} * {}x{}", a.rows, a.cols, b.rows, b.cols);
    let mut out = Mat::zeros(a.rows, b.cols);
    if a.cols == 0 || b.cols == 0 {
        return out;
    }
    for (a_row, out_row) in a.data.chunks_exact(a.cols).zip(out.data.chunks_exact_mut(b.cols)) {
        add_vec_mat(a_row, b, out_row);
    }
    out
}

/// `a * b^T`: like `Mat::matmul_t`, but faster for the shapes we meet in
/// backprop.
///
/// Every layer's backward pass ends by pushing the gradient through the
/// weights in the opposite direction, which is a product with the weight
/// matrix *transposed*. `Mat::matmul_t` does that with one dot product per
/// output element. A dot product adds everything into one running sum, so
/// each addition has to wait for the previous one. Building the transpose
/// once (cheap: it is just a copy) lets us use the `out += a * b` loop
/// instead, where nothing waits for anything.
pub fn matmul_transposed(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.cols, b.cols, "matmul_transposed: column counts differ ({} vs {})", a.cols, b.cols);
    matmul(a, &b.transpose())
}

/// `out += a^T * b` without allocating the product first.
///
/// The gradient of a weight matrix over a whole sequence: the sum, over
/// frames, of the outer products [`add_outer`] computes one at a time. Row
/// `i` of the result collects `a[t][i] * (row t of b)` from every frame `t`;
/// as in [`add_vec_mat`], we take the frames four at a time.
pub fn add_t_matmul(a: &Mat, b: &Mat, out: &mut Mat) {
    assert_eq!(a.rows, b.rows, "add_t_matmul: row counts differ ({} vs {})", a.rows, b.rows);
    assert_eq!((out.rows, out.cols), (a.cols, b.cols), "add_t_matmul: output shape mismatch");
    if a.cols == 0 || b.cols == 0 {
        return;
    }
    let mut a_blocks = a.data.chunks_exact(4 * a.cols);
    let mut b_blocks = b.data.chunks_exact(4 * b.cols);
    for (a4, b4) in (&mut a_blocks).zip(&mut b_blocks) {
        let [a0, a1, a2, a3] = four_rows(a4, a.cols);
        let b_rows = four_rows(b4, b.cols);
        for (i, out_row) in out.data.chunks_exact_mut(b.cols).enumerate() {
            add_four_scaled_rows(out_row, [a0[i], a1[i], a2[i], a3[i]], b_rows);
        }
    }
    for (a_row, b_row) in a_blocks.remainder().chunks_exact(a.cols).zip(b_blocks.remainder().chunks_exact(b.cols)) {
        add_outer(a_row, b_row, out);
    }
}

/// Add the same row vector to every row of `m` (how a bias is applied).
pub fn add_row_to_each(m: &mut Mat, row: &[f32]) {
    assert_eq!(m.cols, row.len(), "add_row_to_each: matrix has {} columns, row has {}", m.cols, row.len());
    if row.is_empty() {
        return;
    }
    for m_row in m.data.chunks_exact_mut(row.len()) {
        for (v, &b) in m_row.iter_mut().zip(row) {
            *v += b;
        }
    }
}

/// `out[c] += sum over rows of m[(r, c)]` (the gradient of a bias: it was
/// added to every row, so it collects the gradient of every row).
pub fn add_column_sums(m: &Mat, out: &mut [f32]) {
    assert_eq!(m.cols, out.len(), "add_column_sums: matrix has {} columns, output has {}", m.cols, out.len());
    if out.is_empty() {
        return;
    }
    for m_row in m.data.chunks_exact(out.len()) {
        for (o, &v) in out.iter_mut().zip(m_row) {
            *o += v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_vec_mat_matches_matmul_for_any_row_count() {
        let mut rng = crate::rng::Rng::new(1);
        // 1..=9 rows exercises both the four-at-a-time path and the leftovers.
        for rows in 1..=9 {
            let w = Mat::from_fn(rows, 5, |_, _| rng.normal());
            let x = Mat::from_fn(1, rows, |_, _| rng.normal());
            let expected = x.matmul(&w);
            let mut out = vec![0.0; 5];
            add_vec_mat(&x.data, &w, &mut out);
            for (a, b) in out.iter().zip(&expected.data) {
                assert!((a - b).abs() < 1e-5, "rows={rows}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn add_outer_by_hand() {
        let mut out = Mat::from_vec(2, 3, vec![1.0; 6]);
        add_outer(&[1.0, 2.0], &[3.0, 4.0, 5.0], &mut out);
        assert_eq!(out.data, vec![4.0, 5.0, 6.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn transposed_products_match_mat() {
        let mut rng = crate::rng::Rng::new(2);
        let a = Mat::from_fn(7, 6, |_, _| rng.normal());
        let b = Mat::from_fn(5, 6, |_, _| rng.normal());
        let (fast, slow) = (matmul_transposed(&a, &b), a.matmul_t(&b));
        for (x, y) in fast.data.iter().zip(&slow.data) {
            assert!((x - y).abs() < 1e-5);
        }
        for (x, y) in matmul(&a, &b.transpose()).data.iter().zip(&slow.data) {
            assert!((x - y).abs() < 1e-5);
        }
        // Row counts on both sides of a multiple of four.
        for rows in [1, 4, 7, 8] {
            let a = Mat::from_fn(rows, 6, |_, _| rng.normal());
            let c = Mat::from_fn(rows, 3, |_, _| rng.normal());
            let mut acc = Mat::zeros(6, 3);
            add_t_matmul(&a, &c, &mut acc);
            for (x, y) in acc.data.iter().zip(&a.t_matmul(&c).data) {
                assert!((x - y).abs() < 1e-5, "rows={rows}");
            }
        }
    }

    #[test]
    fn bias_helpers_by_hand() {
        let mut m = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        add_row_to_each(&mut m, &[10.0, 20.0]);
        assert_eq!(m.data, vec![11.0, 22.0, 13.0, 24.0]);
        let mut sums = vec![0.0; 2];
        add_column_sums(&m, &mut sums);
        assert_eq!(sums, vec![24.0, 46.0]);
    }
}
