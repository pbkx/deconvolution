mod common;

use deconvolution::nd;
use deconvolution::otf::{otf2psf, otf2psf_3d, psf2otf, psf2otf_3d};
use deconvolution::psf::{gaussian2d, gaussian3d, Kernel3D};
use deconvolution::simulate::{
    add_poisson_noise, blur, checkerboard_2d, gaussian_blob_2d, phantom_3d,
};
use deconvolution::{
    qmle_with, richardson_lucy_with, wiener_with, Cmle, Error, Gmle, Qmle, RichardsonLucy, Wiener,
};
use image::{DynamicImage, Rgba, RgbaImage};
use ndarray::{Array3, Axis};

use common::{
    array_to_gray, gray_to_array, is_finite_2d, is_finite_3d, max_abs_diff_2d, mse_3d, psnr_2d,
};

#[test]
fn psf_otf_roundtrip_tolerances_hold() {
    let psf_2d = gaussian2d((9, 9), 1.5).unwrap();
    let otf_2d = psf2otf(&psf_2d, (48, 52)).unwrap();
    let recovered_2d = otf2psf(&otf_2d, psf_2d.dims()).unwrap();
    let max_err_2d = max_abs_diff_2d(psf_2d.as_array(), recovered_2d.as_array()).unwrap();
    assert!(max_err_2d <= 1e-3);

    let psf_3d = gaussian3d((7, 9, 9), 1.4).unwrap();
    let otf_3d = psf2otf_3d(&psf_3d, (9, 40, 40)).unwrap();
    let recovered_3d = otf2psf_3d(&otf_3d, psf_3d.dims()).unwrap();
    let mse_3d_value = mse_3d(psf_3d.as_array(), recovered_3d.as_array()).unwrap();
    assert!(mse_3d_value <= 1e-6);
}

#[test]
fn restoration_improvement_thresholds_hold() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.6).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 24.0, 4301).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let restored_wiener = wiener_with(&degraded_image, &psf, &Wiener::new().nsr(1e-2)).unwrap();
    let (restored_rl, _) = richardson_lucy_with(
        &degraded_image,
        &psf,
        &RichardsonLucy::new()
            .iterations(16)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )
    .unwrap();
    let (restored_qmle, _) = qmle_with(
        &degraded_image,
        &psf,
        &Qmle::new().iterations(10).snr(60.0).acuity(1.1),
    )
    .unwrap();

    let baseline_psnr = psnr_2d(&sharp, &gray_to_array(&degraded_image.to_luma8())).unwrap();
    let wiener_psnr = psnr_2d(&sharp, &gray_to_array(&restored_wiener.to_luma8())).unwrap();
    let rl_psnr = psnr_2d(&sharp, &gray_to_array(&restored_rl.to_luma8())).unwrap();
    let qmle_psnr = psnr_2d(&sharp, &gray_to_array(&restored_qmle.to_luma8())).unwrap();
    assert!(wiener_psnr > baseline_psnr);
    assert!(rl_psnr > baseline_psnr);
    assert!(qmle_psnr > baseline_psnr);
}

#[test]
fn finite_output_and_shape_preservation_hold_for_mle_matrix() {
    let sharp = gaussian_blob_2d((52, 60), 7.0).unwrap();
    let psf = gaussian2d((11, 11), 1.8).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 18.0, 9901).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (cmle_out, _) =
        deconvolution::cmle_with(&degraded_image, &psf, &Cmle::new().iterations(18).snr(18.0))
            .unwrap();
    let (gmle_out, _) = deconvolution::gmle_with(
        &degraded_image,
        &psf,
        &Gmle::new()
            .iterations(14)
            .snr(18.0)
            .roughness(1.1)
            .acuity(0.85),
    )
    .unwrap();
    let (qmle_out, _) = deconvolution::qmle_with(
        &degraded_image,
        &psf,
        &Qmle::new().iterations(9).snr(60.0).acuity(1.1),
    )
    .unwrap();

    let cmle_arr = gray_to_array(&cmle_out.to_luma8());
    let gmle_arr = gray_to_array(&gmle_out.to_luma8());
    let qmle_arr = gray_to_array(&qmle_out.to_luma8());
    assert_eq!(cmle_arr.dim(), sharp.dim());
    assert_eq!(gmle_arr.dim(), sharp.dim());
    assert_eq!(qmle_arr.dim(), sharp.dim());
    assert!(is_finite_2d(&cmle_arr));
    assert!(is_finite_2d(&gmle_arr));
    assert!(is_finite_2d(&qmle_arr));
    assert!(cmle_arr.iter().all(|value| *value >= 0.0));
    assert!(gmle_arr.iter().all(|value| *value >= 0.0));
    assert!(qmle_arr.iter().all(|value| *value >= 0.0));
}

#[test]
fn alpha_channel_behavior_is_preserved() {
    let mut rgba = RgbaImage::new(41, 29);
    for y in 0..29_u32 {
        for x in 0..41_u32 {
            let r = ((7 * x + 5 * y) % 256) as u8;
            let g = ((11 * x + 3 * y) % 256) as u8;
            let b = ((13 * x + 17 * y) % 256) as u8;
            let a = ((19 * x + 23 * y) % 256) as u8;
            rgba.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }

    let psf = gaussian2d((7, 7), 1.2).unwrap();
    let wiener = wiener_with(
        &DynamicImage::ImageRgba8(rgba.clone()),
        &psf,
        &Wiener::new(),
    )
    .unwrap()
    .to_rgba8();
    let (qmle, _) = qmle_with(
        &DynamicImage::ImageRgba8(rgba.clone()),
        &psf,
        &Qmle::new().iterations(8),
    )
    .unwrap();
    let qmle = qmle.to_rgba8();

    assert_eq!(wiener.dimensions(), rgba.dimensions());
    assert_eq!(qmle.dimensions(), rgba.dimensions());
    for y in 0..29_u32 {
        for x in 0..41_u32 {
            assert_eq!(wiener.get_pixel(x, y)[3], rgba.get_pixel(x, y)[3]);
            assert_eq!(qmle.get_pixel(x, y)[3], rgba.get_pixel(x, y)[3]);
        }
    }
}

#[test]
fn nd_volume_regression_metrics_hold() {
    let sharp = phantom_3d((9, 40, 40)).unwrap();
    let psf_3d = gaussian3d((7, 9, 9), 1.5).unwrap();
    let projected = project_psf3d(psf_3d.as_array()).unwrap();
    let blurred = blur_volume_slicewise(&sharp, &projected).unwrap();
    let degraded = add_poisson_noise_volume_slicewise(&blurred, 14.0, 2223).unwrap();

    let (restored, report) = nd::qmle_with(
        &degraded,
        psf_3d.as_array(),
        &Qmle::new().iterations(9).snr(60.0).acuity(1.1),
    )
    .unwrap();
    let baseline_mse = mse_3d(&sharp, &degraded).unwrap();
    let restored_mse = mse_3d(&sharp, &restored).unwrap();
    assert!(restored_mse < baseline_mse);
    assert!(is_finite_3d(&restored));
    assert!(restored.iter().all(|value| *value >= 0.0));
    assert!(report.iterations >= 1);
}

fn project_psf3d(psf: &Array3<f32>) -> deconvolution::Result<Kernel3D> {
    let mut kernel = Kernel3D::new(psf.to_owned())?;
    kernel.normalize()?;
    Ok(kernel)
}

fn blur_volume_slicewise(
    volume: &Array3<f32>,
    psf: &Kernel3D,
) -> deconvolution::Result<Array3<f32>> {
    if volume.is_empty() {
        return Err(Error::EmptyImage);
    }
    if volume.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let mut projected = psf.as_array().sum_axis(Axis(0));
    let sum = projected.sum();
    if !sum.is_finite() || sum.abs() <= f32::EPSILON {
        return Err(Error::InvalidPsf);
    }
    for value in &mut projected {
        *value /= sum;
    }
    let projected = deconvolution::Kernel2D::new(projected)?;

    let (depth, height, width) = volume.dim();
    let mut output = Array3::zeros((depth, height, width));
    for z in 0..depth {
        let slice = volume.index_axis(Axis(0), z).to_owned();
        let blurred = blur(&slice, &projected)?;
        output.index_axis_mut(Axis(0), z).assign(&blurred);
    }
    Ok(output)
}

fn add_poisson_noise_volume_slicewise(
    volume: &Array3<f32>,
    peak: f32,
    seed: u64,
) -> deconvolution::Result<Array3<f32>> {
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
        let noisy = add_poisson_noise(
            &slice,
            peak,
            seed.wrapping_add((z as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        )?;
        output.index_axis_mut(Axis(0), z).assign(&noisy);
    }
    Ok(output)
}
