use deconvolution::preprocess::{apodize, edgetaper, estimate_nsr, normalize_range, window_edges};
use deconvolution::{Kernel2D, RangePolicy};
use ndarray::{array, Array2};

#[test]
fn window_edges_reduces_border_discontinuity_on_step_fixture() {
    let input = step_fixture(48, 48);
    let before = border_discontinuity(&input);

    let tapered = window_edges(&input, 6).unwrap();
    let after = border_discontinuity(&tapered);

    assert_eq!(tapered.dim(), input.dim());
    assert!(tapered.iter().all(|value| value.is_finite()));
    assert!(after < before);
}

#[test]
fn edgetaper_reduces_border_discontinuity_and_preserves_shape() {
    let input = step_fixture(64, 64);
    let psf = Kernel2D::new(array![
        [1.0_f32, 2.0_f32, 1.0_f32],
        [2.0_f32, 4.0_f32, 2.0_f32],
        [1.0_f32, 2.0_f32, 1.0_f32]
    ])
    .unwrap();

    let before = border_discontinuity(&input);
    let tapered = edgetaper(&input, &psf).unwrap();
    let after = border_discontinuity(&tapered);

    assert_eq!(tapered.dim(), input.dim());
    assert!(tapered.iter().all(|value| value.is_finite()));
    assert!(after < before);
}

#[test]
fn estimate_nsr_is_positive_finite_and_stable() {
    let input = noisy_gradient_fixture(40, 56);
    let nsr_a = estimate_nsr(&input).unwrap();
    let nsr_b = estimate_nsr(&input).unwrap();

    assert!(nsr_a.is_finite());
    assert!(nsr_b.is_finite());
    assert!(nsr_a > 0.0);
    assert!((nsr_a - nsr_b).abs() < 1e-8);
}

#[test]
fn apodize_and_normalize_range_preserve_dimensions_and_finiteness() {
    let input = noisy_gradient_fixture(32, 36);

    let apodized = apodize(&input).unwrap();
    assert_eq!(apodized.dim(), input.dim());
    assert!(apodized.iter().all(|value| value.is_finite()));

    let clamped01 = normalize_range(&apodized, RangePolicy::Clamp01).unwrap();
    assert_eq!(clamped01.dim(), input.dim());
    assert!(clamped01.iter().all(|value| value.is_finite()));
    assert!(clamped01.iter().all(|value| *value >= 0.0 && *value <= 1.0));

    let clamped11 = normalize_range(&apodized, RangePolicy::ClampNegPos1).unwrap();
    assert!(clamped11
        .iter()
        .all(|value| *value >= -1.0 && *value <= 1.0));
}

fn step_fixture(height: usize, width: usize) -> Array2<f32> {
    let mut image = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            image[[y, x]] = if x < width / 2 { 0.0 } else { 1.0 };
        }
    }
    image
}

fn noisy_gradient_fixture(height: usize, width: usize) -> Array2<f32> {
    let mut image = Array2::zeros((height, width));
    let denom_y = (height.saturating_sub(1).max(1)) as f32;
    let denom_x = (width.saturating_sub(1).max(1)) as f32;

    for y in 0..height {
        for x in 0..width {
            let gradient = 0.65 * (y as f32 / denom_y) + 0.35 * (x as f32 / denom_x);
            let phase = ((x + 3 * y) % 9) as f32;
            let noise = (phase - 4.0) * 0.005;
            image[[y, x]] = gradient + noise;
        }
    }

    image
}

fn border_discontinuity(input: &Array2<f32>) -> f32 {
    let (height, width) = input.dim();
    let mut sum = 0.0_f32;
    for y in 0..height {
        sum += (input[[y, 0]] - input[[y, width - 1]]).abs();
    }
    sum / (height as f32)
}
