//! 1-D convolution over time.
//!
//! A linear layer looks at one frame at a time. A convolution looks at a
//! short *window* of `kernel` neighboring frames, and slides that window
//! along the sequence. The same weights are used at every position, so a
//! pattern (say, the sudden burst of a "t") is recognized wherever in the
//! clip it happens.
//!
//! # Shapes
//!
//! The input is `T x c_in` (frames by channels) and the output `T' x c_out`.
//! Two knobs change how many frames come out:
//!
//! * `padding` pretends there are that many all-zero frames before the first
//!   frame and after the last, so the window can hang over the edges.
//! * `stride` is how many frames the window moves between outputs. A stride
//!   of 2 halves the frame rate, which halves the work of every layer after.
//!
//! ```text
//! T' = floor((T + 2 * padding - kernel) / stride) + 1
//! ```
//!
//! Read it as: the padded sequence has `T + 2 * padding` frames; the window's
//! first frame can sit at positions `0, stride, 2 * stride, ...` as long as
//! its last frame still fits. If `T` is not a neat multiple of the stride,
//! the last frame or two simply never get a window of their own.
//!
//! # Convolution is a linear layer in disguise ("im2col")
//!
//! Output frame `t` is a weighted sum of everything in its window. So if we
//! copy each window out into one long row (`kernel * c_in` numbers: frame
//! after frame, laid end to end), then every output frame is just a linear
//! layer applied to its row:
//!
//! ```text
//! patches = one row per output frame, each the window it looks at   (T' x kernel*c_in)
//! Y       = patches W + b                                           (T' x c_out)
//! ```
//!
//! That costs some copying, but it turns the whole layer into the single
//! matrix product we already made fast, and it makes the backward pass
//! something we already know: backprop of a linear layer gives the gradient
//! of every patch, and each patch entry is a copy of one input entry, so we
//! *add* the patch gradients back onto the input frames they were copied
//! from (an input frame that appeared in three windows collects three
//! contributions).

use super::init;
use super::kernels;
use crate::mat::Mat;
use crate::rng::Rng;

/// A 1-D convolution over the time axis.
#[derive(Clone, Debug, PartialEq)]
pub struct Conv1d {
    /// Channels (columns) of the input.
    pub c_in: usize,
    /// Channels (columns) of the output.
    pub c_out: usize,
    /// How many consecutive frames one output looks at.
    pub kernel: usize,
    /// How many frames the window moves between outputs.
    pub stride: usize,
    /// How many zero frames are imagined on each end of the input.
    pub padding: usize,
    /// `(kernel * c_in) x c_out`. Row `j * c_in + c` holds the weights for
    /// channel `c` of the `j`-th frame in the window.
    pub weight: Mat,
    /// `1 x c_out`.
    pub bias: Mat,
}

impl Conv1d {
    /// A convolution with He-initialized weights (in all our models a
    /// convolution is followed by a ReLU) and zero biases.
    pub fn new(c_in: usize, c_out: usize, kernel: usize, stride: usize, padding: usize, rng: &mut Rng) -> Self {
        assert!(kernel >= 1 && stride >= 1, "Conv1d: kernel and stride must be at least 1");
        let fan_in = kernel * c_in; // every output is a sum over a whole window
        Conv1d { c_in, c_out, kernel, stride, padding, weight: init::he_uniform(fan_in, c_out, fan_in, rng), bias: Mat::zeros(1, c_out) }
    }

    /// How many output frames an input of `t` frames produces (0 if the input
    /// is too short for even one window).
    pub fn out_len(&self, t: usize) -> usize {
        let padded = t + 2 * self.padding;
        if padded < self.kernel {
            0
        } else {
            (padded - self.kernel) / self.stride + 1
        }
    }

    /// Which input frame does position `tap` of output frame `out_frame`'s
    /// window look at? `None` if that position is in the zero padding.
    fn input_frame(&self, out_frame: usize, tap: usize, t_in: usize) -> Option<usize> {
        let position_in_padded = out_frame * self.stride + tap;
        let frame = position_in_padded.checked_sub(self.padding)?;
        (frame < t_in).then_some(frame)
    }

    /// Copy every window out into its own row ("im2col").
    fn gather_patches(&self, x: &Mat) -> Mat {
        let mut patches = Mat::zeros(self.out_len(x.rows), self.kernel * self.c_in);
        for out_frame in 0..patches.rows {
            let patch = patches.row_mut(out_frame);
            for tap in 0..self.kernel {
                if let Some(frame) = self.input_frame(out_frame, tap, x.rows) {
                    patch[tap * self.c_in..(tap + 1) * self.c_in].copy_from_slice(x.row(frame));
                } // else: padding, and the patch is already zero there
            }
        }
        patches
    }

    /// The reverse of `gather_patches` for gradients: add every patch entry's
    /// gradient onto the input frame it was copied from ("col2im").
    fn scatter_patches(&self, d_patches: &Mat, t_in: usize) -> Mat {
        let mut d_x = Mat::zeros(t_in, self.c_in);
        for out_frame in 0..d_patches.rows {
            let d_patch = d_patches.row(out_frame);
            for tap in 0..self.kernel {
                if let Some(frame) = self.input_frame(out_frame, tap, t_in) {
                    let from = &d_patch[tap * self.c_in..(tap + 1) * self.c_in];
                    for (d, &g) in d_x.row_mut(frame).iter_mut().zip(from) {
                        *d += g;
                    }
                }
            }
        }
        d_x
    }

    /// Returns the output (`T' x c_out`) and the patch matrix, which
    /// `backward` needs.
    pub fn forward(&self, x: &Mat) -> (Mat, Mat) {
        assert_eq!(x.cols, self.c_in, "Conv1d: input has {} channels, the layer expects {}", x.cols, self.c_in);
        let patches = self.gather_patches(x);
        let mut y = kernels::matmul(&patches, &self.weight);
        kernels::add_row_to_each(&mut y, &self.bias.data);
        (y, patches)
    }

    /// Given the patches from `forward`, the number of input frames and the
    /// gradient at the output, add the weight and bias gradients into `grads`
    /// (`[weight, bias]`) and return the gradient at the input (`t_in x c_in`).
    pub fn backward(&self, patches: &Mat, t_in: usize, d_out: &Mat, grads: &mut [Mat]) -> Mat {
        assert_eq!((d_out.rows, d_out.cols), (patches.rows, self.c_out), "Conv1d: gradient shape does not match the output");
        // Exactly the linear layer's backward pass, with patches as the input...
        kernels::add_t_matmul(patches, d_out, &mut grads[0]);
        kernels::add_column_sums(d_out, &mut grads[1].data);
        let d_patches = kernels::matmul_transposed(d_out, &self.weight);
        // ...followed by undoing the copy.
        self.scatter_patches(&d_patches, t_in)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One channel in, one channel out, kernel [1, 2, 3], bias 0.5.
    fn tiny(stride: usize, padding: usize) -> Conv1d {
        Conv1d { c_in: 1, c_out: 1, kernel: 3, stride, padding, weight: Mat::from_vec(3, 1, vec![1.0, 2.0, 3.0]), bias: Mat::from_vec(1, 1, vec![0.5]) }
    }

    fn ramp() -> Mat {
        Mat::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
    }

    #[test]
    fn output_length_formula() {
        let mut rng = Rng::new(0);
        let conv = |k, s, p| Conv1d::new(1, 1, k, s, p, &mut Rng::new(0)).out_len(100);
        assert_eq!(conv(5, 2, 0), 48); // floor(95 / 2) + 1
        assert_eq!(conv(5, 2, 2), 50); // floor(99 / 2) + 1
        assert_eq!(conv(3, 1, 1), 100); // "same" padding
        assert_eq!(conv(3, 3, 0), 33); // 100 is not a multiple of 3: frame 99 is dropped
        assert_eq!(Conv1d::new(1, 1, 5, 1, 0, &mut rng).out_len(4), 0); // too short
    }

    #[test]
    fn forward_no_padding_by_hand() {
        // Windows [1,2,3], [2,3,4], [3,4,5] dotted with [1,2,3], plus 0.5.
        let (y, _) = tiny(1, 0).forward(&ramp());
        assert_eq!(y.data, vec![14.5, 20.5, 26.5]);
    }

    #[test]
    fn forward_with_stride_and_padding_by_hand() {
        // Padded input: 0 1 2 3 4 5 0. Stride 2 gives windows [0,1,2], [2,3,4], [4,5,0].
        let (y, patches) = tiny(2, 1).forward(&ramp());
        assert_eq!(patches.data, vec![0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 0.0]);
        assert_eq!(y.data, vec![8.5, 20.5, 14.5]);
    }

    #[test]
    fn backward_by_hand() {
        let conv = tiny(2, 1);
        let (_, patches) = conv.forward(&ramp());
        let d_out = Mat::from_vec(3, 1, vec![1.0, 10.0, 100.0]);
        let mut grads = vec![Mat::zeros(3, 1), Mat::zeros(1, 1)];
        let d_x = conv.backward(&patches, 5, &d_out, &mut grads);
        // Weight j sees the j-th entry of every window:
        //   dW[0] = 0*1 + 2*10 + 4*100 = 420
        //   dW[1] = 1*1 + 3*10 + 5*100 = 531
        //   dW[2] = 2*1 + 4*10 + 0*100 = 42
        assert_eq!(grads[0].data, vec![420.0, 531.0, 42.0]);
        assert_eq!(grads[1].data, vec![111.0]);
        // Input frame 0 (value 1) is the middle of window 0:            2*1          = 2
        // frame 1 is the end of window 0 and the start of window 1:     3*1 + 1*10   = 13
        // frame 2 is the middle of window 1:                            2*10         = 20
        // frame 3 is the end of window 1 and the start of window 2:     3*10 + 1*100 = 130
        // frame 4 is the middle of window 2:                            2*100        = 200
        assert_eq!(d_x.data, vec![2.0, 13.0, 20.0, 130.0, 200.0]);
    }

    #[test]
    fn multi_channel_patch_layout() {
        // Two channels, kernel 2: a patch is [frame t ch0, frame t ch1, frame t+1 ch0, frame t+1 ch1].
        let conv = Conv1d { c_in: 2, c_out: 1, kernel: 2, stride: 1, padding: 0, weight: Mat::from_vec(4, 1, vec![1.0, 10.0, 100.0, 1000.0]), bias: Mat::zeros(1, 1) };
        let x = Mat::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (y, _) = conv.forward(&x);
        assert_eq!(y.data, vec![4321.0, 6543.0]);
    }
}
