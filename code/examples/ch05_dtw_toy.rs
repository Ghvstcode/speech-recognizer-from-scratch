//! Chapter 5: dynamic time warping on two tiny sequences, small enough to
//! check every number by hand.
//!
//! Run with the built-in example:
//!
//! ```text
//! cargo run --release --example ch05_dtw_toy
//! ```
//!
//! or with your own two sequences of numbers:
//!
//! ```text
//! cargo run --release --example ch05_dtw_toy -- 1,2,3,2 1,1,2,3,3,2
//! ```

use speech::dtw::{dtw_with_matrix, local_distances};
use speech::mat::Mat;

const USAGE: &str = "usage: ch05_dtw_toy [A B]\n  A, B: comma-separated numbers, e.g. 1,2,3,2 1,1,2,3,3,2";

/// Parse "1,2.5,3" into numbers.
fn parse_sequence(text: &str) -> Option<Vec<f32>> {
    let values: Option<Vec<f32>> = text.split(',').map(|v| v.trim().parse().ok()).collect();
    values.filter(|v| !v.is_empty())
}

/// A sequence of numbers as a feature matrix: one row per frame, one feature.
fn as_frames(values: &[f32]) -> Mat {
    Mat::from_vec(values.len(), 1, values.to_vec())
}

/// Print a grid with `a` down the side and `b` along the top. Cells on the
/// warping path are wrapped in [brackets].
fn print_grid(title: &str, a: &[f32], b: &[f32], grid: &Mat, path: &[(usize, usize)]) {
    println!("{title}");
    print!("          b:");
    for value in b {
        print!("{value:>6} ");
    }
    println!();
    for (i, value) in a.iter().enumerate() {
        print!("  a = {value:>4}  ");
        for j in 0..b.len() {
            if path.contains(&(i, j)) {
                print!("[{:>5}]", grid[(i, j)]);
            } else {
                print!(" {:>5} ", grid[(i, j)]);
            }
        }
        println!();
    }
    println!();
}

/// Print only the shape of the path: `#` where it goes, `.` elsewhere.
fn print_path(a: &[f32], b: &[f32], path: &[(usize, usize)]) {
    println!("The warping path (a runs down, b runs right):");
    for i in 0..a.len() {
        let row: String = (0..b.len()).map(|j| if path.contains(&(i, j)) { " #" } else { " ." }).collect();
        println!("  {row}");
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (a, b) = match args.as_slice() {
        [] => (vec![0.0, 1.0, 4.0, 6.0, 4.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 5.0, 4.0, 6.0, 6.0, 3.0, 1.0, 0.0]),
        [a, b] => match (parse_sequence(a), parse_sequence(b)) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
        },
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    };

    println!("a = {a:?}   ({} frames)", a.len());
    println!("b = {b:?}   ({} frames)", b.len());
    println!();

    let (frames_a, frames_b) = (as_frames(&a), as_frames(&b));
    let local = local_distances(&frames_a, &frames_b);
    let (result, accumulated) = dtw_with_matrix(&frames_a, &frames_b);

    print_grid("Local distance d(i, j) = |a[i] - b[j]|:", &a, &b, &local, &[]);
    print_grid("Accumulated cost D(i, j) = d(i, j) + min(D(i-1, j-1), D(i-1, j), D(i, j-1)), path in [brackets]:", &a, &b, &accumulated, &result.path);
    print_path(&a, &b, &result.path);

    println!("Path, as (frame of a, frame of b) pairs:");
    for &(i, j) in &result.path {
        println!("  a[{i}] = {:<4} <-> b[{j}] = {:<4}  distance {}", a[i], b[j], local[(i, j)]);
    }
    println!();
    println!("DTW cost (bottom-right corner of D): {}", result.cost);
    println!("Normalized by n + m = {} + {}:       {:.4}", a.len(), b.len(), result.normalized_cost);

    // What we would have got without warping: compare frame i with frame i
    // for as long as both sequences last.
    let naive: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
    println!("Frame-by-frame, no warping (first {} frames only): {naive}", a.len().min(b.len()));
}
