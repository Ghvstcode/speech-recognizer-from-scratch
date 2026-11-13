//! Dynamic time warping: comparing two recordings that run at different speeds.
//!
//! Say "seven" twice and the two recordings will not have the same number of
//! frames, and even if they did, the "s" would be longer in one and the "e"
//! longer in the other. Comparing frame 10 with frame 10 is therefore
//! meaningless. Dynamic time warping (DTW) finds the best way to *stretch*
//! one sequence onto the other and reports how different they still are after
//! the stretching.
//!
//! The picture to keep in mind is a grid. Sequence `a` (n frames) runs down
//! the side, sequence `b` (m frames) runs along the top, and cell `(i, j)`
//! holds the distance between frame `i` of `a` and frame `j` of `b`. A
//! **warping path** is a walk from the top-left cell `(0, 0)` to the
//! bottom-right cell `(n-1, m-1)` that only ever steps right, down, or
//! diagonally down-right. Each cell on the walk says "these two frames are the
//! same moment of the word". The cost of a walk is the sum of the cells it
//! visits, and DTW finds the cheapest walk.
//!
//! There are astronomically many walks, but the cheapest one can be found by
//! filling in a second grid, `D`, one cell at a time:
//!
//! ```text
//! D[i][j] = d(i, j) + min( D[i-1][j-1],   diagonal: both sequences advance
//!                          D[i-1][j],     down:     a advances, b waits
//!                          D[i][j-1] )    right:    b advances, a waits
//! ```
//!
//! `D[i][j]` is the cost of the cheapest walk from `(0, 0)` to `(i, j)`, so the
//! answer we want is sitting in the bottom-right corner when we are done.

use crate::mat::Mat;

/// The distance between two feature frames: the ordinary straight-line
/// (Euclidean) distance, `sqrt(sum of squared differences)`.
///
/// This is the "local" distance, the number written in one cell of the grid.
pub fn frame_distance(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "frame_distance: frames have {} and {} features", x.len(), y.len());
    let mut sum = 0.0;
    for (p, q) in x.iter().zip(y) {
        sum += (p - q) * (p - q);
    }
    sum.sqrt()
}

/// The grid of local distances: element `(i, j)` is
/// `frame_distance(a.row(i), b.row(j))`. Rows follow `a`, columns follow `b`.
///
/// The DTW functions below compute these on the fly; this function exists so
/// that we can look at the grid (the book draws it).
pub fn local_distances(a: &Mat, b: &Mat) -> Mat {
    check_inputs(a, b);
    Mat::from_fn(a.rows, b.rows, |i, j| frame_distance(a.row(i), b.row(j)))
}

/// What DTW found out about a pair of sequences.
#[derive(Clone, Debug, PartialEq)]
pub struct DtwResult {
    /// Sum of the local distances along the cheapest warping path.
    pub cost: f32,
    /// `cost / (n + m)`: the cost made comparable across sequence lengths.
    /// See [`normalize_cost`].
    pub normalized_cost: f32,
    /// The cheapest warping path, as `(frame of a, frame of b)` pairs, from
    /// `(0, 0)` to `(n-1, m-1)`.
    pub path: Vec<(usize, usize)>,
}

/// Divide a DTW cost by `n + m`, the combined length of the two sequences.
///
/// Why normalize at all? The raw cost is a *sum* over the path, and a path
/// through a small grid has fewer cells to sum. When we compare one recording
/// against templates of different lengths, raw cost would favor the short
/// templates just for being short. Dividing by a length turns the sum into
/// something like an average distance per frame.
///
/// Why `n + m` rather than the number of cells on the path? Two reasons. The
/// path length depends on which path won, so "cost / path length" is not what
/// the recurrence minimized, and a longer, wigglier path could look better
/// after dividing than the path we actually found. And `n + m` is known
/// before we start, so the fast cost-only variant, which never traces the
/// path, can use the same normalization. (A path has between `max(n, m)` and
/// `n + m - 1` cells, so the two choices differ by less than a factor of
/// two.)
pub fn normalize_cost(cost: f32, n: usize, m: usize) -> f32 {
    cost / (n + m) as f32
}

/// Align two feature sequences (one row per frame) and return the cost and
/// the warping path.
pub fn dtw(a: &Mat, b: &Mat) -> DtwResult {
    dtw_full(a, b, None).0
}

/// Like [`dtw`], and also return the whole accumulated-cost grid `D`
/// (`n x m`, rows follow `a`, columns follow `b`). `D[(i, j)]` is the cost of
/// the cheapest path from `(0, 0)` to `(i, j)`.
pub fn dtw_with_matrix(a: &Mat, b: &Mat) -> (DtwResult, Mat) {
    dtw_full(a, b, None)
}

/// Like [`dtw`], restricted to a Sakoe-Chiba band of the given width
/// (see [`band_radius`]).
pub fn dtw_banded(a: &Mat, b: &Mat, band: usize) -> DtwResult {
    dtw_full(a, b, Some(band)).0
}

/// The general form behind [`dtw`], [`dtw_with_matrix`] and [`dtw_banded`]:
/// fill the accumulated-cost grid (inside the band, if there is one), then
/// trace the cheapest path back from the bottom-right corner.
///
/// Cells outside the band are left at `f32::INFINITY`.
pub fn dtw_full(a: &Mat, b: &Mat, band: Option<usize>) -> (DtwResult, Mat) {
    let accumulated = accumulate(a, b, band);
    let path = backtrace(&accumulated);
    let cost = accumulated[(a.rows - 1, b.rows - 1)];
    let result = DtwResult { cost, normalized_cost: normalize_cost(cost, a.rows, b.rows), path };
    (result, accumulated)
}

/// How far from the diagonal a banded DTW may stray.
///
/// A **Sakoe-Chiba band** only fills cells with `|i - j| <= radius`. It does
/// two jobs: it forbids absurd alignments (one frame of `a` stretched over
/// half of `b`), and it saves work, since cells outside the band are never
/// computed.
///
/// The end cell `(n-1, m-1)` is `|n - m|` away from the diagonal, so a band
/// narrower than that would cut the corner off and leave no path at all. We
/// widen the band to at least `|n - m|`. `None` means no band: every cell is
/// allowed.
pub fn band_radius(n: usize, m: usize, band: Option<usize>) -> usize {
    match band {
        Some(width) => width.max(n.abs_diff(m)),
        None => n.max(m),
    }
}

/// The columns of row `i` that lie inside the band: `i - radius ..= i + radius`,
/// clipped to the grid.
fn band_columns(i: usize, m: usize, radius: usize) -> std::ops::Range<usize> {
    i.saturating_sub(radius)..(i + radius + 1).min(m)
}

/// The smallest of the three neighbours a path can arrive from.
fn min3(diagonal: f32, up: f32, left: f32) -> f32 {
    diagonal.min(up).min(left)
}

/// Fill the accumulated-cost grid, row by row.
fn accumulate(a: &Mat, b: &Mat, band: Option<usize>) -> Mat {
    check_inputs(a, b);
    let (n, m) = (a.rows, b.rows);
    let radius = band_radius(n, m, band);
    let mut acc = Mat::from_fn(n, m, |_, _| f32::INFINITY);
    for i in 0..n {
        for j in band_columns(i, m, radius) {
            let local = frame_distance(a.row(i), b.row(j));
            // The cheapest way to arrive at (i, j). On the top row and left
            // column some neighbours do not exist; a missing neighbour is
            // "infinitely expensive", so `min` never picks it.
            let best_before = if i == 0 && j == 0 {
                0.0 // the start cell: nothing came before
            } else {
                let diagonal = if i > 0 && j > 0 { acc[(i - 1, j - 1)] } else { f32::INFINITY };
                let up = if i > 0 { acc[(i - 1, j)] } else { f32::INFINITY };
                let left = if j > 0 { acc[(i, j - 1)] } else { f32::INFINITY };
                min3(diagonal, up, left)
            };
            acc[(i, j)] = local + best_before;
        }
    }
    acc
}

/// Recover the cheapest path from a filled accumulated-cost grid.
///
/// We start at the end cell and repeatedly step to whichever of the three
/// neighbours (diagonal, up, left) has the smallest accumulated cost: that is
/// the neighbour the recurrence took its minimum from. On a tie we prefer the
/// diagonal, the step that advances both sequences.
fn backtrace(acc: &Mat) -> Vec<(usize, usize)> {
    let (mut i, mut j) = (acc.rows - 1, acc.cols - 1);
    let mut path = vec![(i, j)];
    while i > 0 || j > 0 {
        if i == 0 {
            j -= 1; // on the top row the only way back is left
        } else if j == 0 {
            i -= 1; // on the left column the only way back is up
        } else {
            let (diagonal, up, left) = (acc[(i - 1, j - 1)], acc[(i - 1, j)], acc[(i, j - 1)]);
            if diagonal <= up && diagonal <= left {
                i -= 1;
                j -= 1;
            } else if up <= left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        path.push((i, j));
    }
    path.reverse();
    path
}

/// The DTW cost alone (not normalized), without the path.
///
/// The recurrence for row `i` only reads row `i - 1` and the part of row `i`
/// already filled in, so two rows of memory are enough. That makes this the
/// fast path for classification, where we run DTW against thousands of
/// templates and only ever look at the final number. `band` is an optional
/// Sakoe-Chiba band width (see [`band_radius`]).
pub fn dtw_cost(a: &Mat, b: &Mat, band: Option<usize>) -> f32 {
    check_inputs(a, b);
    let (n, m) = (a.rows, b.rows);
    let radius = band_radius(n, m, band);
    let mut previous = vec![f32::INFINITY; m];
    let mut current = vec![f32::INFINITY; m];
    for i in 0..n {
        // Cells outside the band must read as infinity, including the ones
        // left over in this buffer from two rows ago.
        current.fill(f32::INFINITY);
        for j in band_columns(i, m, radius) {
            let local = frame_distance(a.row(i), b.row(j));
            let best_before = if i == 0 && j == 0 {
                0.0
            } else {
                let diagonal = if j > 0 { previous[j - 1] } else { f32::INFINITY };
                let up = previous[j];
                let left = if j > 0 { current[j - 1] } else { f32::INFINITY };
                min3(diagonal, up, left)
            };
            current[j] = local + best_before;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[m - 1]
}

/// The normalized DTW cost, `dtw_cost / (n + m)`: the number the classifier
/// ranks templates by.
pub fn dtw_distance(a: &Mat, b: &Mat, band: Option<usize>) -> f32 {
    normalize_cost(dtw_cost(a, b, band), a.rows, b.rows)
}

/// Panic with a helpful message if two sequences cannot be compared.
fn check_inputs(a: &Mat, b: &Mat) {
    assert!(a.rows > 0 && b.rows > 0, "dtw: sequences must have at least one frame (got {} and {})", a.rows, b.rows);
    assert_eq!(a.cols, b.cols, "dtw: frames have {} and {} features", a.cols, b.cols);
}

/// One labelled example: the features of a recording we know the answer for.
#[derive(Clone, Debug)]
pub struct Template {
    /// What was said, as a class id (for the digits dataset, the digit).
    pub label: usize,
    /// The recording's features, one row per frame.
    pub features: Mat,
}

/// A recognizer with no training at all: remember labelled recordings, and
/// classify a new one by finding the stored recordings it is closest to under
/// DTW.
#[derive(Clone, Debug, Default)]
pub struct TemplateBank {
    /// The stored examples.
    pub templates: Vec<Template>,
    /// Optional Sakoe-Chiba band width used for every comparison.
    pub band: Option<usize>,
}

impl TemplateBank {
    /// An empty bank that compares without a band.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty bank that compares inside a Sakoe-Chiba band of this width.
    pub fn with_band(band: usize) -> Self {
        TemplateBank { templates: Vec::new(), band: Some(band) }
    }

    /// Remember one labelled recording.
    pub fn add(&mut self, label: usize, features: Mat) {
        assert!(features.rows > 0, "a template needs at least one frame");
        self.templates.push(Template { label, features });
    }

    /// Number of stored templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// True if no template has been added yet.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// The `k` templates closest to `query`, nearest first, as
    /// `(label, normalized DTW cost)`.
    pub fn nearest(&self, query: &Mat, k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> =
            self.templates.iter().map(|t| (t.label, dtw_distance(query, &t.features, self.band))).collect();
        scored.sort_by(|x, y| x.1.total_cmp(&y.1));
        scored.truncate(k);
        scored
    }

    /// The label of the single closest template, and its normalized cost.
    /// Panics if the bank is empty.
    pub fn classify(&self, query: &Mat) -> (usize, f32) {
        self.classify_knn(query, 1)
    }

    /// Let the `k` closest templates vote. Returns the winning label and the
    /// normalized cost of that label's closest template.
    ///
    /// One odd template sitting close to the query can fool a 1-nearest
    /// classifier; asking several neighbours and taking the majority is more
    /// robust. If two labels get the same number of votes, the one whose
    /// closest template is nearer wins.
    pub fn classify_knn(&self, query: &Mat, k: usize) -> (usize, f32) {
        assert!(!self.is_empty(), "cannot classify with an empty template bank");
        assert!(k > 0, "k must be at least 1");
        let neighbours = self.nearest(query, k);
        // `neighbours` is sorted nearest first, so the first time we meet a
        // label is that label's closest template. Requiring strictly more
        // votes to take the lead means earlier (nearer) labels win ties.
        let mut winner = neighbours[0];
        let mut winner_votes = 0;
        for (index, &(label, cost)) in neighbours.iter().enumerate() {
            let first_time_seen = neighbours[..index].iter().all(|&(l, _)| l != label);
            if !first_time_seen {
                continue;
            }
            let votes = neighbours.iter().filter(|&&(l, _)| l == label).count();
            if votes > winner_votes {
                winner = (label, cost);
                winner_votes = votes;
            }
        }
        winner
    }

    /// Classify many recordings at once (k-nearest voting, `k = 1` for plain
    /// nearest template), spreading the work over all CPU cores.
    ///
    /// Each query costs one DTW per template, so a few hundred queries against
    /// a few thousand templates is millions of grids. The queries are
    /// independent of each other, which makes this the easiest kind of
    /// parallelism: cut the list of queries into one chunk per thread, and
    /// let each thread fill in its own chunk of the answers.
    pub fn classify_many(&self, queries: &[Mat], k: usize) -> Vec<(usize, f32)> {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let mut results = vec![(0, 0.0); queries.len()];
        if queries.is_empty() {
            return results;
        }
        let chunk_len = queries.len().div_ceil(threads);
        std::thread::scope(|scope| {
            for (query_chunk, result_chunk) in queries.chunks(chunk_len).zip(results.chunks_mut(chunk_len)) {
                scope.spawn(move || {
                    for (query, result) in query_chunk.iter().zip(result_chunk) {
                        *result = self.classify_knn(query, k);
                    }
                });
            }
        });
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// A sequence of 1-dimensional frames.
    fn seq(values: &[f32]) -> Mat {
        Mat::from_vec(values.len(), 1, values.to_vec())
    }

    fn random_seq(rng: &mut Rng, frames: usize, features: usize) -> Mat {
        Mat::from_fn(frames, features, |_, _| rng.normal())
    }

    fn close(x: f32, y: f32) -> bool {
        (x - y).abs() <= 1e-4 * (1.0 + x.abs().max(y.abs()))
    }

    #[test]
    fn frame_distance_is_euclidean() {
        assert_eq!(frame_distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
        assert_eq!(frame_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    /// a = [0, 2, 4] down the side, b = [0, 1, 4, 4] along the top.
    ///
    /// Local distances |a_i - b_j|:        Accumulated cost D:
    ///
    ///          b: 0  1  4  4                    0  1  4  4
    ///     a=0     0  1  4  4               0    0  1  5  9
    ///     a=2     2  1  2  2               2    2  1  3  5
    ///     a=4     4  3  0  0               4    6  4  1  1
    ///
    /// Working a few cells of D by hand:
    ///   D[0][j] is a running sum along the top row: 0, 0+1, 1+4, 5+4.
    ///   D[1][1] = 1 + min(D[0][0]=0, D[0][1]=1, D[1][0]=2) = 1
    ///   D[1][2] = 2 + min(1, 5, 1) = 3
    ///   D[2][2] = 0 + min(D[1][1]=1, D[1][2]=3, D[2][1]=4) = 1
    ///   D[2][3] = 0 + min(3, 5, 1) = 1
    ///
    /// Backtrace from (2,3): left to (2,2), diagonal to (1,1), diagonal to (0,0).
    #[test]
    fn hand_worked_grid() {
        let (a, b) = (seq(&[0.0, 2.0, 4.0]), seq(&[0.0, 1.0, 4.0, 4.0]));
        let local = local_distances(&a, &b);
        assert_eq!(local.data, vec![0.0, 1.0, 4.0, 4.0, 2.0, 1.0, 2.0, 2.0, 4.0, 3.0, 0.0, 0.0]);

        let (result, acc) = dtw_with_matrix(&a, &b);
        assert_eq!(acc.data, vec![0.0, 1.0, 5.0, 9.0, 2.0, 1.0, 3.0, 5.0, 6.0, 4.0, 1.0, 1.0]);
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.normalized_cost, 1.0 / 7.0);
        assert_eq!(result.path, vec![(0, 0), (1, 1), (2, 2), (2, 3)]);
        assert_eq!(dtw_cost(&a, &b, None), 1.0);
    }

    /// a = [1, 3], b = [2]: with a single column the only path goes straight
    /// down, so the cost is |1-2| + |3-2| = 2.
    #[test]
    fn single_frame_against_a_sequence() {
        let (a, b) = (seq(&[1.0, 3.0]), seq(&[2.0]));
        let result = dtw(&a, &b);
        assert_eq!(result.cost, 2.0);
        assert_eq!(result.path, vec![(0, 0), (1, 0)]);
        assert_eq!(dtw(&b, &a).path, vec![(0, 0), (0, 1)]);
    }

    #[test]
    fn a_sequence_against_itself_costs_nothing() {
        let mut rng = Rng::new(11);
        let a = random_seq(&mut rng, 17, 5);
        let result = dtw(&a, &a);
        assert_eq!(result.cost, 0.0);
        let diagonal: Vec<(usize, usize)> = (0..17).map(|i| (i, i)).collect();
        assert_eq!(result.path, diagonal);
    }

    #[test]
    fn cost_is_symmetric() {
        let mut rng = Rng::new(12);
        let (a, b) = (random_seq(&mut rng, 23, 4), random_seq(&mut rng, 31, 4));
        assert!(close(dtw(&a, &b).cost, dtw(&b, &a).cost));
        assert!(close(dtw_cost(&a, &b, Some(3)), dtw_cost(&b, &a, Some(3))));
    }

    /// Repeat every frame of `a` two or three times: a slowed-down copy. DTW
    /// should line each original frame up with its copies at zero cost, which
    /// is the whole point of warping.
    #[test]
    fn a_time_stretched_copy_costs_nothing() {
        let mut rng = Rng::new(13);
        let a = random_seq(&mut rng, 12, 3);
        let mut rows = Vec::new();
        for i in 0..a.rows {
            for _ in 0..(2 + i % 2) {
                rows.push(a.row(i).to_vec());
            }
        }
        let stretched = Mat::from_rows(&rows);
        assert_eq!(stretched.rows, 30);
        assert_eq!(dtw(&a, &stretched).cost, 0.0);
        // Comparing frame by frame, without warping, would be far from zero.
        let naive: f32 = (0..a.rows).map(|i| frame_distance(a.row(i), stretched.row(i))).sum();
        assert!(naive > 1.0);
    }

    #[test]
    fn rolling_rows_agree_with_the_full_grid() {
        let mut rng = Rng::new(14);
        for (n, m) in [(1, 1), (1, 9), (9, 1), (20, 20), (15, 40), (40, 15)] {
            let (a, b) = (random_seq(&mut rng, n, 6), random_seq(&mut rng, m, 6));
            for band in [None, Some(0), Some(2), Some(7)] {
                let (full, _) = dtw_full(&a, &b, band);
                assert_eq!(full.cost, dtw_cost(&a, &b, band), "n={n} m={m} band={band:?}");
            }
        }
    }

    #[test]
    fn a_wide_band_changes_nothing_and_a_narrow_one_never_helps() {
        let mut rng = Rng::new(15);
        let (a, b) = (random_seq(&mut rng, 30, 4), random_seq(&mut rng, 37, 4));
        let unbounded = dtw(&a, &b);
        // A band as wide as the grid allows every cell.
        let wide = dtw_banded(&a, &b, 40);
        assert_eq!(wide, unbounded);
        // A band just wide enough to contain the unbounded path finds that path.
        let needed = unbounded.path.iter().map(|&(i, j)| i.abs_diff(j)).max().unwrap();
        assert_eq!(dtw_banded(&a, &b, needed).cost, unbounded.cost);
        // A narrower band removes options, so the cost can only go up.
        let mut last = unbounded.cost;
        for band in (0..needed).rev() {
            let cost = dtw_banded(&a, &b, band).cost;
            assert!(cost >= last, "band {band}: {cost} < {last}");
            assert!(cost.is_finite());
            last = cost;
        }
    }

    #[test]
    fn band_cells_outside_are_infinite_and_path_stays_inside() {
        let mut rng = Rng::new(16);
        let (a, b) = (random_seq(&mut rng, 20, 2), random_seq(&mut rng, 24, 2));
        let (result, acc) = dtw_full(&a, &b, Some(1));
        // The band is widened to |20 - 24| = 4 so the corner is reachable.
        assert_eq!(band_radius(20, 24, Some(1)), 4);
        for i in 0..20 {
            for j in 0..24 {
                assert_eq!(acc[(i, j)].is_finite(), i.abs_diff(j) <= 4);
            }
        }
        assert!(result.path.iter().all(|&(i, j)| i.abs_diff(j) <= 4));
    }

    /// A warping path must start at (0,0), end at (n-1,m-1), and move by one
    /// of the three allowed steps each time: never backwards, never jumping.
    #[test]
    fn paths_are_monotone_and_connected() {
        let mut rng = Rng::new(17);
        for (n, m) in [(1, 1), (5, 1), (1, 5), (12, 19), (19, 12), (25, 25)] {
            let (a, b) = (random_seq(&mut rng, n, 3), random_seq(&mut rng, m, 3));
            for band in [None, Some(2)] {
                let (result, _) = dtw_full(&a, &b, band);
                let path = &result.path;
                assert_eq!(path[0], (0, 0));
                assert_eq!(*path.last().unwrap(), (n - 1, m - 1));
                for pair in path.windows(2) {
                    let step = (pair[1].0 - pair[0].0, pair[1].1 - pair[0].1);
                    assert!(matches!(step, (1, 1) | (1, 0) | (0, 1)), "bad step {step:?}");
                }
                // The cost is the sum of local distances along the path.
                let along: f32 = path.iter().map(|&(i, j)| frame_distance(a.row(i), b.row(j))).sum();
                assert!(close(along, result.cost));
            }
        }
    }

    /// Templates are noisy, differently stretched copies of three "words".
    fn toy_bank(rng: &mut Rng) -> (TemplateBank, Vec<Mat>) {
        let prototypes: Vec<Mat> = (0..3).map(|_| random_seq(rng, 10, 4)).collect();
        let mut bank = TemplateBank::with_band(8);
        for (label, prototype) in prototypes.iter().enumerate() {
            for copy in 0..4 {
                bank.add(label, noisy_stretch(rng, prototype, copy % 3));
            }
        }
        (bank, prototypes)
    }

    fn noisy_stretch(rng: &mut Rng, prototype: &Mat, extra: usize) -> Mat {
        let mut rows = Vec::new();
        for i in 0..prototype.rows {
            let repeats = 1 + usize::from(i % 3 < extra);
            for _ in 0..repeats {
                rows.push(prototype.row(i).iter().map(|v| v + 0.1 * rng.normal()).collect());
            }
        }
        Mat::from_rows(&rows)
    }

    #[test]
    fn nearest_template_classifier_recovers_labels() {
        let mut rng = Rng::new(18);
        let (bank, prototypes) = toy_bank(&mut rng);
        assert_eq!(bank.len(), 12);
        let mut queries = Vec::new();
        let mut truth = Vec::new();
        for round in 0..9 {
            let label = round % 3;
            queries.push(noisy_stretch(&mut rng, &prototypes[label], round % 3));
            truth.push(label);
        }
        for (query, &label) in queries.iter().zip(&truth) {
            assert_eq!(bank.classify(query).0, label);
            assert_eq!(bank.classify_knn(query, 3).0, label);
            let nearest = bank.nearest(query, 5);
            assert!(nearest.windows(2).all(|w| w[0].1 <= w[1].1), "not sorted");
        }
        // The threaded version gives exactly the same answers, in order.
        let serial: Vec<(usize, f32)> = queries.iter().map(|q| bank.classify(q)).collect();
        assert_eq!(bank.classify_many(&queries, 1), serial);
        assert!(bank.classify_many(&[], 1).is_empty());
    }

    /// Voting: the single nearest template says label 7, but the next two
    /// both say label 3, so with k = 3 the majority wins.
    #[test]
    fn k_nearest_voting_outvotes_a_single_outlier() {
        let mut bank = TemplateBank::new();
        bank.add(7, seq(&[1.0, 1.0])); // distance 0.0 per frame from the query
        bank.add(3, seq(&[1.1, 1.1]));
        bank.add(3, seq(&[1.2, 1.2]));
        bank.add(7, seq(&[5.0, 5.0]));
        let query = seq(&[1.0, 1.0]);
        assert_eq!(bank.classify(&query), (7, 0.0));
        let (label, cost) = bank.classify_knn(&query, 3);
        assert_eq!(label, 3);
        // Cost of label 3's closest template: (0.1 + 0.1) / (2 + 2).
        assert!(close(cost, 0.05));
        // With k = 2 the vote is tied 1-1 and the nearer label wins.
        assert_eq!(bank.classify_knn(&query, 2).0, 7);
    }
}
