//! Tests that exercise the pieces together: gradient checks for every layer,
//! save/load, the threaded batch helper, and an end-to-end "can it learn?".

use super::*;
use crate::mat::Mat;
use crate::rng::Rng;

fn random_mat(rows: usize, cols: usize, rng: &mut Rng) -> Mat {
    Mat::from_fn(rows, cols, |_, _| rng.normal())
}

fn random_classes(n: usize, n_classes: usize, rng: &mut Rng) -> Vec<usize> {
    (0..n).map(|_| rng.below(n_classes)).collect()
}

/// Biases start at zero, where some bugs cannot be seen. Shake everything up.
fn randomize_biases(model: &mut Model, rng: &mut Rng) {
    for p in model.params_mut() {
        if p.rows == 1 {
            for b in p.data.iter_mut() {
                *b = 0.3 * rng.normal();
            }
        }
    }
}

/// Gradcheck `model` on a random `t x n_in` input with a per-row
/// cross-entropy loss, and fail with the full report if anything is off.
fn assert_gradients_ok(mut model: Model, t: usize, n_in: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    randomize_biases(&mut model, &mut rng);
    let x = random_mat(t, n_in, &mut rng);
    let out = model.predict(&x);
    let classes = random_classes(out.rows, out.cols, &mut rng);
    let report = gradcheck(&model, &x, |logits| cross_entropy(logits, &classes), &GradCheckConfig::default());
    assert!(report.passed(), "gradient check failed for\n{}\n{report}", model.summary());
}

#[test]
fn gradcheck_linear() {
    let mut rng = Rng::new(10);
    assert_gradients_ok(Model::new(vec![Layer::linear(5, 4, &mut rng)]), 6, 5, 11);
}

#[test]
fn gradcheck_activations() {
    let mut rng = Rng::new(20);
    for activation in [Layer::Relu, Layer::Tanh, Layer::Sigmoid] {
        let model = Model::new(vec![Layer::linear_he(4, 6, &mut rng), activation, Layer::linear(6, 3, &mut rng)]);
        assert_gradients_ok(model, 5, 4, 21);
    }
}

#[test]
fn gradcheck_conv1d_across_strides_and_paddings() {
    let mut rng = Rng::new(30);
    // (kernel, stride, padding, T). Several have T not divisible by the
    // stride, and some leave trailing frames that no window covers.
    for (kernel, stride, padding, t) in [(3, 1, 0, 7), (3, 1, 1, 7), (5, 2, 2, 11), (5, 2, 0, 10), (3, 2, 1, 8), (4, 3, 0, 11), (4, 3, 2, 13), (1, 1, 0, 4), (3, 4, 1, 9)] {
        let model = Model::new(vec![Layer::conv1d(3, 4, kernel, stride, padding, &mut rng)]);
        assert_gradients_ok(model, t, 3, 31);
    }
}

#[test]
fn gradcheck_gru() {
    let mut rng = Rng::new(40);
    assert_gradients_ok(Model::new(vec![Layer::gru(3, 5, &mut rng)]), 9, 3, 41);
    // A single frame is an edge case of its own (no previous state at all).
    assert_gradients_ok(Model::new(vec![Layer::gru(3, 5, &mut rng)]), 1, 3, 42);
}

#[test]
fn gradcheck_bigru() {
    let mut rng = Rng::new(50);
    assert_gradients_ok(Model::new(vec![Layer::bigru(3, 4, &mut rng)]), 8, 3, 51);
}

#[test]
fn gradcheck_mean_pool() {
    let mut rng = Rng::new(60);
    let model = Model::new(vec![Layer::linear(4, 5, &mut rng), Layer::Tanh, Layer::MeanPool, Layer::linear(5, 3, &mut rng)]);
    assert_gradients_ok(model, 7, 4, 61);
}

#[test]
fn gradcheck_stacked_conv_gru_linear() {
    let mut rng = Rng::new(70);
    // T = 13 with stride 2: not divisible.
    let model = Model::new(vec![
        Layer::conv1d(4, 6, 3, 2, 1, &mut rng),
        Layer::Tanh,
        Layer::gru(6, 5, &mut rng),
        Layer::linear(5, 4, &mut rng),
    ]);
    assert_gradients_ok(model, 13, 4, 71);
}

#[test]
fn gradcheck_stacked_conv_relu_bigru_linear() {
    let mut rng = Rng::new(80);
    // The shape of the real recognizer, in miniature.
    let model = Model::new(vec![
        Layer::conv1d(5, 6, 5, 2, 2, &mut rng),
        Layer::Relu,
        Layer::bigru(6, 4, &mut rng),
        Layer::linear(8, 5, &mut rng),
    ]);
    assert_gradients_ok(model, 11, 5, 81);
}

#[test]
fn gradcheck_clip_classifier_with_every_layer_type() {
    let mut rng = Rng::new(90);
    let model = Model::new(vec![
        Layer::conv1d(3, 5, 3, 1, 1, &mut rng),
        Layer::Relu,
        Layer::bigru(5, 3, &mut rng),
        Layer::gru(6, 4, &mut rng),
        Layer::Sigmoid,
        Layer::MeanPool,
        Layer::linear_he(4, 6, &mut rng),
        Layer::Tanh,
        Layer::linear(6, 3, &mut rng),
    ]);
    assert_gradients_ok(model, 10, 3, 91);
}

#[test]
fn gradcheck_catches_a_wrong_gradient() {
    // Feed the checker a loss whose reported gradient is deliberately wrong
    // (twice the truth): every tensor must be flagged.
    let mut rng = Rng::new(100);
    let model = Model::new(vec![Layer::linear(3, 4, &mut rng), Layer::Tanh, Layer::linear(4, 3, &mut rng)]);
    let x = random_mat(4, 3, &mut rng);
    let classes = random_classes(4, 3, &mut rng);
    let wrong_loss = |logits: &Mat| {
        let (loss, mut d) = cross_entropy(logits, &classes);
        d.scale(2.0);
        (loss, d)
    };
    let report = gradcheck(&model, &x, wrong_loss, &GradCheckConfig::default());
    assert!(!report.passed());
    assert!(report.checks.iter().all(|c| !c.pass), "{report}");
}

#[test]
fn save_and_load_round_trip() {
    let mut rng = Rng::new(110);
    let mut model = Model::new(vec![
        Layer::conv1d(4, 6, 5, 2, 2, &mut rng),
        Layer::Relu,
        Layer::bigru(6, 5, &mut rng),
        Layer::gru(10, 4, &mut rng),
        Layer::Tanh,
        Layer::Sigmoid,
        Layer::MeanPool,
        Layer::linear(4, 3, &mut rng),
    ]);
    randomize_biases(&mut model, &mut rng);

    let path = std::env::temp_dir().join(format!("speech_nn_round_trip_{}.bin", std::process::id()));
    model.save(&path).unwrap();
    let loaded = Model::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(loaded, model);
    let x = random_mat(9, 4, &mut rng);
    assert_eq!(loaded.predict(&x), model.predict(&x));
}

#[test]
fn file_format_is_what_the_docs_say() {
    let linear = Linear { weight: Mat::from_vec(1, 2, vec![1.0, -2.0]), bias: Mat::from_vec(1, 2, vec![0.5, 0.0]) };
    let model = Model::new(vec![Layer::Linear(linear), Layer::Relu]);
    let mut bytes = Vec::new();
    model.write_to(&mut bytes).unwrap();
    let mut expected = Vec::new();
    expected.extend_from_slice(b"SPEECHNN");
    expected.extend_from_slice(&1u32.to_le_bytes()); // version
    expected.extend_from_slice(&2u32.to_le_bytes()); // two layers
    expected.push(1); // Linear
    expected.extend_from_slice(&1u32.to_le_bytes()); // n_in
    expected.extend_from_slice(&2u32.to_le_bytes()); // n_out
    for (rows, cols, values) in [(1u32, 2u32, [1.0f32, -2.0]), (1, 2, [0.5, 0.0])] {
        expected.extend_from_slice(&rows.to_le_bytes());
        expected.extend_from_slice(&cols.to_le_bytes());
        for v in values {
            expected.extend_from_slice(&v.to_le_bytes());
        }
    }
    expected.push(2); // ReLU
    assert_eq!(bytes, expected);
}

#[test]
fn loading_rejects_bad_files() {
    let mut rng = Rng::new(120);
    let model = Model::new(vec![Layer::linear(3, 2, &mut rng), Layer::Tanh]);
    let mut good = Vec::new();
    model.write_to(&mut good).unwrap();
    let load = |bytes: &[u8]| Model::read_from(&mut &bytes[..]);
    assert!(load(&good).is_ok());

    let invalid = |bytes: &[u8]| load(bytes).unwrap_err().kind() == std::io::ErrorKind::InvalidData;
    let mut wrong_magic = good.clone();
    wrong_magic[0] = b'X';
    assert!(invalid(&wrong_magic));
    let mut wrong_version = good.clone();
    wrong_version[8] = 99;
    assert!(invalid(&wrong_version));
    let mut wrong_tag = good.clone();
    wrong_tag[16] = 200;
    assert!(invalid(&wrong_tag));
    let mut wrong_shape = good.clone();
    wrong_shape[25] = 7; // the weight tensor's row count
    assert!(invalid(&wrong_shape));
    // Cut off in the middle of the weights.
    assert_eq!(load(&good[..good.len() - 6]).unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
}

/// A small conv + BiGRU clip classifier and a few random clips with labels.
fn clip_task(seed: u64) -> (Model, Vec<Mat>, Vec<Vec<usize>>) {
    let mut rng = Rng::new(seed);
    let model = Model::new(vec![
        Layer::conv1d(4, 8, 3, 2, 1, &mut rng),
        Layer::Relu,
        Layer::bigru(8, 6, &mut rng),
        Layer::MeanPool,
        Layer::linear(12, 3, &mut rng),
    ]);
    // Clips of different lengths, as real utterances are.
    let inputs: Vec<Mat> = (0..7).map(|i| random_mat(9 + i, 4, &mut rng)).collect();
    let targets: Vec<Vec<usize>> = (0..7).map(|i| vec![i % 3]).collect();
    (model, inputs, targets)
}

#[test]
fn threaded_batch_matches_one_at_a_time() {
    let (model, inputs, targets) = clip_task(130);
    let batch: Vec<(&Mat, &[usize])> = inputs.iter().zip(&targets).map(|(x, t)| (x, &t[..])).collect();

    // The reference: a plain loop.
    let mut expected = Grads::zeros_like(&model);
    let mut expected_loss = 0.0;
    for &(x, target) in &batch {
        let cache = model.forward(x);
        let (loss, d_logits) = cross_entropy(cache.output(), target);
        model.backward(&cache, &d_logits, &mut expected);
        expected_loss += loss;
    }
    expected.scale(1.0 / batch.len() as f32);
    expected_loss /= batch.len() as f32;

    for n_threads in [1, 2, 3, 16] {
        let out = batch_gradients(&model, &batch, n_threads, cross_entropy);
        assert_eq!((out.used, out.skipped), (7, 0));
        assert!((out.loss - expected_loss).abs() < 1e-5);
        for (a, b) in out.grads.tensors.iter().zip(&expected.tensors) {
            for (x, y) in a.data.iter().zip(&b.data) {
                assert!((x - y).abs() < 1e-5, "threads={n_threads}: {x} vs {y}");
            }
        }
    }
}

#[test]
fn batch_skips_non_finite_losses() {
    let (model, inputs, targets) = clip_task(140);
    let batch: Vec<(&Mat, &[usize])> = inputs.iter().zip(&targets).map(|(x, t)| (x, &t[..])).collect();
    // A loss that "fails" on the 10- and 12-frame clips, the way CTC returns
    // infinity for a transcript that cannot fit, or a bug returns NaN.
    let picky = |logits: &Mat, target: &[usize]| {
        let (loss, mut d) = cross_entropy(logits, target);
        match target[0] {
            1 => (f32::INFINITY, d),
            2 => {
                d.data[0] = f32::NAN;
                (loss, d)
            }
            _ => (loss, d),
        }
    };
    let out = batch_gradients(&model, &batch, 4, picky);
    // Classes cycle 0,1,2,0,1,2,0: three clips of class 0 survive.
    assert_eq!((out.used, out.skipped), (3, 4));
    assert!(out.loss.is_finite() && out.grads.is_finite());

    let survivors: Vec<(&Mat, &[usize])> = batch.iter().copied().filter(|(_, t)| t[0] == 0).collect();
    let expected = batch_gradients(&model, &survivors, 1, cross_entropy);
    assert!((out.loss - expected.loss).abs() < 1e-5);
    for (a, b) in out.grads.tensors.iter().zip(&expected.grads.tensors) {
        for (x, y) in a.data.iter().zip(&b.data) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    // A batch where everything fails yields zeros, not NaNs.
    let nothing = batch_gradients(&model, &batch, 4, |logits: &Mat, _: &[usize]| (f32::NAN, logits.clone()));
    assert_eq!((nothing.used, nothing.skipped, nothing.loss), (0, 7, 0.0));
    assert_eq!(nothing.grads.global_norm(), 0.0);
    // And an empty batch is fine too.
    let empty: Vec<(&Mat, &[usize])> = Vec::new();
    assert_eq!(batch_gradients(&model, &empty, 4, cross_entropy).used, 0);
}

#[test]
fn conv_gru_model_overfits_a_handful_of_sequences() {
    // The end-to-end proof: forward, loss, backward, threads, clipping and
    // Adam together must be able to memorize seven random clips.
    let (mut model, inputs, targets) = clip_task(150);
    let batch: Vec<(&Mat, &[usize])> = inputs.iter().zip(&targets).map(|(x, t)| (x, &t[..])).collect();
    let mut adam = Adam::new(0.02);
    let first = batch_gradients(&model, &batch, 1, cross_entropy).loss;
    let mut last = first;
    for _ in 0..300 {
        let mut out = batch_gradients(&model, &batch, 3, cross_entropy);
        clip_global_norm(&mut out.grads, 5.0);
        adam.step(&mut model, &out.grads);
        last = out.loss;
    }
    assert!(first > 0.5, "an untrained model should be near ln 3 = 1.1, got {first}");
    assert!(last < 0.01, "loss only fell from {first} to {last}");
    for (x, target) in &batch {
        assert_eq!(argmax(model.predict(x).row(0)), target[0]);
    }
}

#[test]
fn frame_level_model_overfits_too() {
    // Same idea with one label per output frame (the shape CTC training
    // has), SGD with momentum, and the learning-rate schedule.
    let mut rng = Rng::new(160);
    let mut model = Model::new(vec![
        Layer::conv1d(3, 8, 3, 1, 1, &mut rng),
        Layer::Tanh,
        Layer::gru(8, 8, &mut rng),
        Layer::linear(8, 4, &mut rng),
    ]);
    let inputs: Vec<Mat> = (0..3).map(|_| random_mat(8, 3, &mut rng)).collect();
    let targets: Vec<Vec<usize>> = (0..3).map(|_| random_classes(8, 4, &mut rng)).collect();
    let batch: Vec<(&Mat, &[usize])> = inputs.iter().zip(&targets).map(|(x, t)| (x, &t[..])).collect();

    let schedule = LrSchedule { peak_lr: 0.05, final_lr: 0.005, warmup_steps: 10, total_steps: 600 };
    let mut sgd = Sgd::new(0.0, 0.9);
    let mut last = f32::INFINITY;
    for step in 0..600 {
        sgd.lr = schedule.lr_at(step);
        let mut out = batch_gradients(&model, &batch, 2, cross_entropy);
        clip_global_norm(&mut out.grads, 5.0);
        sgd.step(&mut model, &out.grads);
        last = out.loss;
    }
    // The loss is a sum over 8 frames, so 0.05 is under 0.01 per frame.
    assert!(last < 0.05, "loss is still {last}");
}
