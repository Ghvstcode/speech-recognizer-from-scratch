//! Optimizers: turning gradients into better weights.
//!
//! A gradient says, for every weight, which direction makes the loss go *up*
//! fastest. Every optimizer here does some version of "take a small step the
//! other way". They differ in how they choose the size of the step.

use super::model::{Grads, Model};
use crate::mat::Mat;

/// Stochastic gradient descent, optionally with momentum.
///
/// Plain SGD is one line: `w -= lr * gradient`.
///
/// With momentum the optimizer keeps a running *velocity* for every weight:
///
/// ```text
/// v = momentum * v + gradient
/// w -= lr * v
/// ```
///
/// Picture a ball rolling downhill. Gradients that keep pointing the same way
/// build up speed; gradients that flip sign from step to step (the walls of a
/// narrow valley) cancel out. `momentum = 0.9` is the usual choice; `0.0`
/// gives plain SGD.
#[derive(Clone, Debug)]
pub struct Sgd {
    /// The learning rate: how big a step to take.
    pub lr: f32,
    /// How much of the previous velocity survives each step (0 disables momentum).
    pub momentum: f32,
    /// One velocity tensor per weight tensor; created on the first step.
    velocity: Vec<Mat>,
}

impl Sgd {
    /// An SGD optimizer. Pass `momentum = 0.0` for plain SGD.
    pub fn new(lr: f32, momentum: f32) -> Sgd {
        Sgd { lr, momentum, velocity: Vec::new() }
    }

    /// Update the model's weights in place from one set of gradients.
    pub fn step(&mut self, model: &mut Model, grads: &Grads) {
        let mut params = model.params_mut();
        assert_eq!(params.len(), grads.tensors.len(), "Sgd: the gradients do not match the model");
        if self.momentum == 0.0 {
            for (p, g) in params.iter_mut().zip(&grads.tensors) {
                for (w, &gw) in p.data.iter_mut().zip(&g.data) {
                    *w -= self.lr * gw;
                }
            }
            return;
        }
        if self.velocity.is_empty() {
            self.velocity = zeros_like(grads);
        }
        for ((p, g), v) in params.iter_mut().zip(&grads.tensors).zip(&mut self.velocity) {
            for ((w, &gw), vw) in p.data.iter_mut().zip(&g.data).zip(&mut v.data) {
                *vw = self.momentum * *vw + gw;
                *w -= self.lr * *vw;
            }
        }
    }
}

/// Adam (Kingma and Ba, 2014): SGD with momentum *and* a separate step size
/// for every single weight.
///
/// Adam keeps two running averages per weight:
///
/// ```text
/// m = beta1 * m + (1 - beta1) * g        the average gradient   (momentum)
/// v = beta2 * v + (1 - beta2) * g * g    the average squared gradient
/// ```
///
/// and steps by `lr * m / sqrt(v)`. Dividing by `sqrt(v)` normalizes the
/// step: a weight whose gradients are consistently tiny (common deep inside
/// a recurrent layer) gets its steps scaled up, and one with huge gradients
/// gets them scaled down, so that every weight moves by roughly `lr` per
/// step whatever the raw scale of its gradient. That makes Adam far less
/// fussy about the learning rate than SGD, which is why we use it for
/// everything after chapter 6.
///
/// **Bias correction.** Both averages start at zero, so for the first few
/// steps they underestimate (after one step, `m` is only `(1 - beta1) * g`,
/// a tenth of the gradient). Dividing by `1 - beta^t`, where `t` counts
/// steps, undoes exactly that; the correction fades to 1 as `t` grows.
///
/// **Weight decay** (optional) gently shrinks every weight toward zero at
/// each step, which discourages the model from leaning on a few huge
/// weights. We use the *decoupled* form (AdamW, Loshchilov and Hutter,
/// 2017): `w -= lr * weight_decay * w`, applied directly to the weight and
/// not mixed into the gradient, so that it is not distorted by the
/// `1 / sqrt(v)` scaling. Biases are not decayed.
#[derive(Clone, Debug)]
pub struct Adam {
    /// The learning rate. `1e-3` is a good first guess.
    pub lr: f32,
    /// Decay rate of the gradient average. Default 0.9.
    pub beta1: f32,
    /// Decay rate of the squared-gradient average. Default 0.999.
    pub beta2: f32,
    /// Added to `sqrt(v)` so we never divide by zero. Default 1e-8.
    pub eps: f32,
    /// Decoupled weight decay. Default 0 (off); 0.01 is a typical value.
    pub weight_decay: f32,
    /// How many steps have been taken.
    pub steps: u64,
    m: Vec<Mat>,
    v: Vec<Mat>,
}

impl Adam {
    /// Adam with the standard settings (`beta1 = 0.9`, `beta2 = 0.999`,
    /// `eps = 1e-8`, no weight decay). The fields are public: change them
    /// directly if you need something else.
    pub fn new(lr: f32) -> Adam {
        Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0, steps: 0, m: Vec::new(), v: Vec::new() }
    }

    /// Update the model's weights in place from one set of gradients.
    pub fn step(&mut self, model: &mut Model, grads: &Grads) {
        let mut params = model.params_mut();
        assert_eq!(params.len(), grads.tensors.len(), "Adam: the gradients do not match the model");
        if self.m.is_empty() {
            self.m = zeros_like(grads);
            self.v = zeros_like(grads);
        }
        self.steps += 1;
        // The bias corrections are the same for every weight, so compute them
        // once. (In f64: beta2^t is very close to 1 for small t.)
        let correct_m = 1.0 / (1.0 - (self.beta1 as f64).powi(self.steps as i32)) as f32;
        let correct_v = 1.0 / (1.0 - (self.beta2 as f64).powi(self.steps as i32)) as f32;

        for (((p, g), m), v) in params.iter_mut().zip(&grads.tensors).zip(&mut self.m).zip(&mut self.v) {
            // A tensor with one row is a bias; those are not decayed.
            let decay = if p.rows > 1 { self.lr * self.weight_decay } else { 0.0 };
            for (((w, &gw), mw), vw) in p.data.iter_mut().zip(&g.data).zip(&mut m.data).zip(&mut v.data) {
                *mw = self.beta1 * *mw + (1.0 - self.beta1) * gw;
                *vw = self.beta2 * *vw + (1.0 - self.beta2) * gw * gw;
                let m_hat = *mw * correct_m;
                let v_hat = *vw * correct_v;
                *w -= self.lr * m_hat / (v_hat.sqrt() + self.eps) + decay * *w;
            }
        }
    }
}

/// Zero tensors with the shapes of a set of gradients.
fn zeros_like(grads: &Grads) -> Vec<Mat> {
    grads.tensors.iter().map(|g| Mat::zeros(g.rows, g.cols)).collect()
}

/// Global-norm gradient clipping. Returns the norm *before* clipping (worth
/// logging: a norm that keeps growing is the first sign of trouble).
///
/// Recurrent networks occasionally produce an enormous gradient (the same
/// matrix is multiplied in a hundred times; sometimes that compounds). One
/// such step can throw the weights so far that training never recovers. The
/// cure: if the length of the whole gradient, all tensors taken together as
/// one vector, exceeds `max_norm`, scale all of it down so its length is
/// exactly `max_norm`. The direction is unchanged, only the step is shortened.
pub fn clip_global_norm(grads: &mut Grads, max_norm: f32) -> f32 {
    let norm = grads.global_norm();
    if norm > max_norm {
        grads.scale(max_norm / norm);
    }
    norm
}

/// A learning-rate schedule: linear warm-up, then cosine decay.
///
/// * **Warm-up.** For the first `warmup_steps` steps the rate climbs in a
///   straight line from 0 to `peak_lr`. Freshly initialized weights produce
///   large, erratic gradients (and Adam's averages are not yet trustworthy),
///   so we begin with baby steps.
/// * **Decay.** After that the rate glides down along half a cosine wave to
///   `final_lr` at `total_steps`, and stays there. Big steps early to cover
///   ground, small steps late to settle into a minimum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LrSchedule {
    /// The highest learning rate, reached at the end of warm-up.
    pub peak_lr: f32,
    /// The learning rate at `total_steps` and beyond.
    pub final_lr: f32,
    /// Number of warm-up steps.
    pub warmup_steps: usize,
    /// The step at which the decay reaches `final_lr`.
    pub total_steps: usize,
}

impl LrSchedule {
    /// The learning rate to use at `step` (counting from 0).
    pub fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.peak_lr * (step + 1) as f32 / self.warmup_steps as f32;
        }
        if step >= self.total_steps {
            return self.final_lr;
        }
        // How far through the decay we are, from 0 to 1.
        let progress = (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
        // cos goes from 1 to -1 over half a turn; shift and scale it to go from 1 to 0.
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.final_lr + (self.peak_lr - self.final_lr) * cosine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Layer, Linear};

    /// A model with a single 1x1 weight (value 1) and a bias (value 0).
    fn one_weight() -> Model {
        Model::new(vec![Layer::Linear(Linear { weight: Mat::from_vec(1, 1, vec![1.0]), bias: Mat::zeros(1, 1) })])
    }

    fn grads_of(model: &Model, weight_grad: f32) -> Grads {
        let mut g = Grads::zeros_like(model);
        g.tensors[0].data[0] = weight_grad;
        g
    }

    fn weight(model: &Model) -> f32 {
        model.params()[0].data[0]
    }

    #[test]
    fn plain_sgd_by_hand() {
        let mut model = one_weight();
        let g = grads_of(&model, 2.0);
        Sgd::new(0.1, 0.0).step(&mut model, &g);
        assert!((weight(&model) - 0.8).abs() < 1e-6); // 1 - 0.1 * 2
    }

    #[test]
    fn momentum_by_hand() {
        let mut model = one_weight();
        let g = grads_of(&model, 2.0);
        let mut sgd = Sgd::new(0.1, 0.5);
        sgd.step(&mut model, &g); // v = 2,           w = 1 - 0.2 = 0.8
        sgd.step(&mut model, &g); // v = 0.5*2 + 2 = 3, w = 0.8 - 0.3 = 0.5
        assert!((weight(&model) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn adam_first_steps_by_hand() {
        // With bias correction, the first step is lr * g / |g| = lr, whatever g is.
        let mut model = one_weight();
        let g = grads_of(&model, 123.0);
        let mut adam = Adam::new(0.01);
        adam.step(&mut model, &g);
        assert!((weight(&model) - 0.99).abs() < 1e-6, "{}", weight(&model));
        // Second step with the same gradient: m_hat = g and v_hat = g^2 again.
        adam.step(&mut model, &g);
        assert!((weight(&model) - 0.98).abs() < 1e-6, "{}", weight(&model));
    }

    #[test]
    fn adam_weight_decay_shrinks_weights_but_not_biases() {
        let mut model = Model::new(vec![Layer::Linear(Linear { weight: Mat::from_vec(2, 1, vec![1.0, 1.0]), bias: Mat::from_vec(1, 1, vec![1.0]) })]);
        let g = Grads::zeros_like(&model); // zero gradient: only decay acts
        let mut adam = Adam::new(0.1);
        adam.weight_decay = 0.5;
        adam.step(&mut model, &g);
        assert!((model.params()[0].data[0] - 0.95).abs() < 1e-6); // 1 - 0.1 * 0.5 * 1
        assert_eq!(model.params()[1].data[0], 1.0);
    }

    #[test]
    fn adam_finds_the_minimum_of_a_parabola() {
        // loss = (w - 3)^2, gradient = 2 (w - 3).
        let mut model = one_weight();
        let mut adam = Adam::new(0.1);
        for _ in 0..500 {
            let g = grads_of(&model, 2.0 * (weight(&model) - 3.0));
            adam.step(&mut model, &g);
        }
        assert!((weight(&model) - 3.0).abs() < 1e-2, "{}", weight(&model));
    }

    #[test]
    fn clipping_shortens_but_keeps_direction() {
        let model = Model::new(vec![Layer::Linear(Linear { weight: Mat::zeros(1, 1), bias: Mat::zeros(1, 1) })]);
        let mut g = Grads::zeros_like(&model);
        g.tensors[0].data[0] = 3.0;
        g.tensors[1].data[0] = 4.0;
        assert_eq!(clip_global_norm(&mut g, 10.0), 5.0); // short enough: untouched
        assert_eq!(g.tensors[0].data[0], 3.0);
        assert_eq!(clip_global_norm(&mut g, 1.0), 5.0);
        assert!((g.tensors[0].data[0] - 0.6).abs() < 1e-6 && (g.tensors[1].data[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn schedule_warms_up_then_decays() {
        let s = LrSchedule { peak_lr: 1.0, final_lr: 0.1, warmup_steps: 4, total_steps: 104 };
        assert_eq!(s.lr_at(0), 0.25);
        assert_eq!(s.lr_at(3), 1.0);
        assert_eq!(s.lr_at(4), 1.0); // start of the cosine
        assert!((s.lr_at(54) - 0.55).abs() < 1e-6); // halfway: midpoint of 1.0 and 0.1
        assert_eq!(s.lr_at(104), 0.1);
        assert_eq!(s.lr_at(10_000), 0.1);
    }
}
