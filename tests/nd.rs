use deconvolution::iterative::RichardsonLucy;
use deconvolution::nd;
use deconvolution::optimization::{Cmle, Gmle, Qmle};
use deconvolution::psf::basic::{delta2d, delta3d, gaussian3d, motion_linear};
use deconvolution::psf::init::uniform;
use deconvolution::simulate::blur::blur;
use deconvolution::simulate::noise::add_poisson_noise;
use deconvolution::simulate::phantom::{checkerboard_2d, phantom_3d};
use deconvolution::spectral::Wiener;
use deconvolution::{blind::BlindRichardsonLucy, Error, Kernel2D, Result};
use ndarray::{Array2, Array3, Axis};

#[test]
fn microscopy_wiener_3d_improves_phantom_restoration() {
    let sharp = phantom_3d((9, 40, 40)).unwrap();
    let psf_3d = gaussian3d((7, 9, 9), 1.4).unwrap();
    let projected_psf = project_psf3d(psf_3d.as_array()).unwrap();
    let degraded = blur_volume_slicewise(&sharp, &projected_psf).unwrap();

    let restored =
        nd::microscopy::wiener_with(&degraded, psf_3d.as_array(), &Wiener::new().nsr(2e-2))
            .unwrap();

    let baseline_mse = mse3(&sharp, &degraded).unwrap();
    let restored_mse = mse3(&sharp, &restored).unwrap();
    assert!(restored_mse < baseline_mse);
}

#[test]
fn microscopy_richardson_lucy_3d_improves_phantom_restoration() {
    let sharp = phantom_3d((9, 40, 40)).unwrap();
    let psf_3d = gaussian3d((7, 9, 9), 1.3).unwrap();
    let projected_psf = project_psf3d(psf_3d.as_array()).unwrap();
    let degraded = blur_volume_slicewise(&sharp, &projected_psf).unwrap();

    let (restored, report) = nd::microscopy::richardson_lucy_with(
        &degraded,
        psf_3d.as_array(),
        &RichardsonLucy::new()
            .iterations(18)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )
    .unwrap();

    let baseline_mse = mse3(&sharp, &degraded).unwrap();
    let restored_mse = mse3(&sharp, &restored).unwrap();
    assert!(restored_mse < baseline_mse);
    assert!(report.iterations >= 1);
}

#[test]
fn nd_known_psf_wiener_retains_fractional_precision() {
    let input = Array2::from_shape_fn((17, 19), |(y, x)| {
        0.07 + (((y * 19 + x) as f32 * 0.017_231 + 0.123_45) % 0.86)
    });
    let psf = delta2d((3, 3)).unwrap();
    let restored =
        nd::known_psf::wiener_with(&input, psf.as_array(), &Wiener::new().nsr(0.0)).unwrap();

    let direct_diff = max_abs_diff_2d(&input, &restored).unwrap();
    assert!(direct_diff < 1e-4);

    let quantized = input.mapv(quantize_u8_step);
    let quantized_diff = max_abs_diff_2d(&quantized, &restored).unwrap();
    assert!(quantized_diff > 5e-4);
}

#[test]
fn nd_microscopy_wiener_retains_fractional_precision() {
    let input = Array3::from_shape_fn((4, 9, 11), |(z, y, x)| {
        0.05 + (((z * 99 + y * 11 + x) as f32 * 0.013_579 + 0.234_56) % 0.88)
    });
    let psf = delta3d((3, 3, 3)).unwrap();
    let restored =
        nd::microscopy::wiener_with(&input, psf.as_array(), &Wiener::new().nsr(0.0)).unwrap();

    let direct_diff = max_abs_diff_3d(&input, &restored).unwrap();
    assert!(direct_diff < 1e-4);

    let quantized = input.mapv(quantize_u8_step);
    let quantized_diff = max_abs_diff_3d(&quantized, &restored).unwrap();
    assert!(quantized_diff > 5e-4);
}

#[test]
fn blind_nd_path_returns_normalized_psf() {
    let sharp = checkerboard_2d((56, 56), 4, 0.0, 1.0).unwrap();
    let true_psf = motion_linear(11.0, 28.0).unwrap();
    let blurred = blur(&sharp, &true_psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 40.0, 731).unwrap();
    let initial_psf = uniform(true_psf.dims()).unwrap();

    let output = nd::blind::richardson_lucy_with(
        &degraded,
        initial_psf.as_array(),
        &BlindRichardsonLucy::new()
            .iterations(20)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )
    .unwrap();

    assert_eq!(output.image.dim(), degraded.dim());
    assert!((output.psf.sum() - 1.0).abs() < 1e-6);
    assert!(output.psf.as_array().iter().all(|value| *value >= 0.0));
}

#[test]
fn nd_mle_family_improves_on_microscopy_volume_fixture() {
    let sharp = phantom_3d((9, 40, 40)).unwrap();
    let psf_3d = gaussian3d((7, 9, 9), 1.5).unwrap();
    let projected_psf = project_psf3d(psf_3d.as_array()).unwrap();
    let blurred = blur_volume_slicewise(&sharp, &projected_psf).unwrap();
    let degraded = add_poisson_noise_volume_slicewise(&blurred, 80.0, 9021).unwrap();

    let (cmle_restored, cmle_report) = nd::microscopy::cmle_with(
        &degraded,
        psf_3d.as_array(),
        &Cmle::new().iterations(5).snr(80.0).acuity(1.0),
    )
    .unwrap();
    let (gmle_restored, gmle_report) = nd::microscopy::gmle_with(
        &degraded,
        psf_3d.as_array(),
        &Gmle::new()
            .iterations(14)
            .snr(80.0)
            .acuity(0.85)
            .roughness(1.1),
    )
    .unwrap();
    let (qmle_restored, qmle_report) = nd::microscopy::qmle_with(
        &degraded,
        psf_3d.as_array(),
        &Qmle::new().iterations(4).snr(120.0).acuity(1.1),
    )
    .unwrap();

    let baseline_mse = mse3(&sharp, &degraded).unwrap();
    let cmle_mse = mse3(&sharp, &cmle_restored).unwrap();
    let gmle_mse = mse3(&sharp, &gmle_restored).unwrap();
    let qmle_mse = mse3(&sharp, &qmle_restored).unwrap();
    assert!(cmle_mse < baseline_mse);
    assert!(gmle_mse < baseline_mse);
    assert!(qmle_mse < baseline_mse);
    assert!(is_nonnegative_3d(&cmle_restored));
    assert!(is_nonnegative_3d(&gmle_restored));
    assert!(is_nonnegative_3d(&qmle_restored));
    assert!(is_finite_3d(&cmle_restored));
    assert!(is_finite_3d(&gmle_restored));
    assert!(is_finite_3d(&qmle_restored));
    assert!(cmle_report.iterations >= 1);
    assert!(gmle_report.iterations >= 1);
    assert!(qmle_report.iterations >= 1);
}

#[test]
fn nd_gmle_is_not_worse_than_nd_cmle_on_high_noise_volume() {
    let sharp = phantom_3d((9, 40, 40)).unwrap();
    let psf_3d = gaussian3d((7, 9, 9), 1.6).unwrap();
    let projected_psf = project_psf3d(psf_3d.as_array()).unwrap();
    let blurred = blur_volume_slicewise(&sharp, &projected_psf).unwrap();
    let degraded = add_poisson_noise_volume_slicewise(&blurred, 7.0, 12303).unwrap();

    let (cmle_restored, _) = nd::microscopy::cmle_with(
        &degraded,
        psf_3d.as_array(),
        &Cmle::new().iterations(22).snr(8.0).acuity(1.15),
    )
    .unwrap();
    let (gmle_restored, _) = nd::microscopy::gmle_with(
        &degraded,
        psf_3d.as_array(),
        &Gmle::new()
            .iterations(14)
            .snr(8.0)
            .acuity(0.8)
            .roughness(1.4),
    )
    .unwrap();

    let cmle_mse = mse3(&sharp, &cmle_restored).unwrap();
    let gmle_mse = mse3(&sharp, &gmle_restored).unwrap();
    assert!(gmle_mse <= cmle_mse * 1.01 + 1e-6);
}

fn project_psf3d(psf: &Array3<f32>) -> Result<Kernel2D> {
    if psf.is_empty() {
        return Err(Error::InvalidPsf);
    }
    if psf.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let mut projected = psf.sum_axis(Axis(0));
    let sum = projected.sum();
    if !sum.is_finite() || sum.abs() <= f32::EPSILON {
        return Err(Error::InvalidPsf);
    }
    for value in &mut projected {
        *value /= sum;
    }
    Kernel2D::new(projected)
}

fn blur_volume_slicewise(volume: &Array3<f32>, psf: &Kernel2D) -> Result<Array3<f32>> {
    if volume.is_empty() {
        return Err(Error::EmptyImage);
    }
    if volume.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let (depth, height, width) = volume.dim();
    let mut output = Array3::zeros((depth, height, width));
    for z in 0..depth {
        let slice = volume.index_axis(Axis(0), z).to_owned();
        let blurred = blur(&slice, psf)?;
        output.index_axis_mut(Axis(0), z).assign(&blurred);
    }
    Ok(output)
}

fn add_poisson_noise_volume_slicewise(
    volume: &Array3<f32>,
    peak: f32,
    seed: u64,
) -> Result<Array3<f32>> {
    if volume.is_empty() {
        return Err(Error::EmptyImage);
    }
    if volume.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    if !peak.is_finite() || peak <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    let (depth, height, width) = volume.dim();
    let mut output = Array3::zeros((depth, height, width));
    for z in 0..depth {
        let slice = volume.index_axis(Axis(0), z).to_owned();
        let slice_seed = seed.wrapping_add((z as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let noisy = add_poisson_noise(&slice, peak, slice_seed)?;
        output.index_axis_mut(Axis(0), z).assign(&noisy);
    }
    Ok(output)
}

fn mse3(lhs: &Array3<f32>, rhs: &Array3<f32>) -> Result<f32> {
    if lhs.dim() != rhs.dim() {
        return Err(Error::DimensionMismatch);
    }
    if lhs.is_empty() {
        return Err(Error::InvalidParameter);
    }

    let mut sum = 0.0_f32;
    let count = lhs.len() as f32;
    for ((z, y, x), value) in lhs.indexed_iter() {
        let diff = *value - rhs[[z, y, x]];
        sum += diff * diff;
    }
    Ok(sum / count)
}

fn max_abs_diff_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> Result<f32> {
    if lhs.dim() != rhs.dim() {
        return Err(Error::DimensionMismatch);
    }
    if lhs.is_empty() {
        return Err(Error::InvalidParameter);
    }

    let mut max_diff = 0.0_f32;
    for ((y, x), value) in lhs.indexed_iter() {
        let diff = (*value - rhs[[y, x]]).abs();
        if !diff.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        max_diff = max_diff.max(diff);
    }
    Ok(max_diff)
}

fn max_abs_diff_3d(lhs: &Array3<f32>, rhs: &Array3<f32>) -> Result<f32> {
    if lhs.dim() != rhs.dim() {
        return Err(Error::DimensionMismatch);
    }
    if lhs.is_empty() {
        return Err(Error::InvalidParameter);
    }

    let mut max_diff = 0.0_f32;
    for ((z, y, x), value) in lhs.indexed_iter() {
        let diff = (*value - rhs[[z, y, x]]).abs();
        if !diff.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        max_diff = max_diff.max(diff);
    }
    Ok(max_diff)
}

fn quantize_u8_step(value: f32) -> f32 {
    (value.clamp(0.0, 1.0) * 255.0).round() / 255.0
}

fn is_finite_3d(input: &Array3<f32>) -> bool {
    input.iter().all(|value| value.is_finite())
}

fn is_nonnegative_3d(input: &Array3<f32>) -> bool {
    input.iter().all(|value| *value >= 0.0)
}
