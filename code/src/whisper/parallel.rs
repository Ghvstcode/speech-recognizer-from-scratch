//! One helper for spreading row-by-row work across CPU cores.
//!
//! Nearly every heavy loop in the Whisper engine has the same shape: "fill in
//! row `t` of an output matrix, and no row depends on any other row". The
//! spectrogram is like that (one row per frame), and so is attention (one row
//! per query). Rows that do not depend on each other can be computed on
//! different cores, and `std::thread::scope` lets us do that safely with no
//! `unsafe` and no dependencies: the output buffer is cut into disjoint
//! `&mut` chunks, one per thread, so the borrow checker can see that no two
//! threads ever write to the same place.

/// Below this many rows we do not bother starting threads: starting one costs
/// tens of microseconds, which is more than a few small rows are worth.
const MIN_ROWS_PER_THREAD: usize = 2;

/// Call `fill(row_index, row)` for every `cols`-wide row of `data`, using all
/// available cores. `fill` must compute a row from its index alone.
///
/// Panics if `data.len()` is not a multiple of `cols`.
pub fn for_each_row_par(data: &mut [f32], cols: usize, fill: impl Fn(usize, &mut [f32]) + Sync) {
    assert!(cols > 0 && data.len().is_multiple_of(cols), "buffer of {} is not rows of {cols}", data.len());
    let rows = data.len() / cols;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    if threads == 1 || rows < MIN_ROWS_PER_THREAD * threads {
        for (index, row) in data.chunks_exact_mut(cols).enumerate() {
            fill(index, row);
        }
        return;
    }
    let rows_per_thread = rows.div_ceil(threads);
    let fill = &fill;
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in data.chunks_mut(rows_per_thread * cols).enumerate() {
            scope.spawn(move || {
                let first_row = chunk_index * rows_per_thread;
                for (offset, row) in chunk.chunks_exact_mut(cols).enumerate() {
                    fill(first_row + offset, row);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_is_visited_once_with_its_own_index() {
        for rows in [0, 1, 5, 1000] {
            let cols = 3;
            let mut data = vec![0.0f32; rows * cols];
            for_each_row_par(&mut data, cols, |index, row| {
                for (c, value) in row.iter_mut().enumerate() {
                    *value += (index * 10 + c) as f32;
                }
            });
            for (i, value) in data.iter().enumerate() {
                assert_eq!(*value, ((i / cols) * 10 + i % cols) as f32);
            }
        }
    }
}
