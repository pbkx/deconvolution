use deconvolution::psf::gaussian2d;
use deconvolution::simulate::{add_poisson_noise, blur, checkerboard_2d};
use deconvolution::{
    damped_richardson_lucy_with, landweber, landweber_with, richardson_lucy, richardson_lucy_tv,
    richardson_lucy_tv_with, richardson_lucy_with, van_cittert, van_cittert_with, Landweber,
    RichardsonLucy, RichardsonLucyTv, VanCittert,
};
use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;

#[test]
fn richardson_lucy_improves_over_poisson_blurred_baseline() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.5).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 30.0, 2026).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (restored, report) = richardson_lucy_with(
        &degraded_image,
        &psf,
        &RichardsonLucy::new()
            .iterations(20)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )
    .unwrap();

    let baseline_array = gray_to_array(&degraded_image.to_luma8());
    let restored_array = gray_to_array(&restored.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline_array).unwrap();
    let restored_psnr = psnr(&sharp, &restored_array).unwrap();

    assert!(restored_psnr > baseline_psnr);
    assert!(report.iterations >= 1);
    assert!(report.estimated_nsr.is_none());
}

#[test]
fn richardson_lucy_output_is_nonnegative_and_finite() {
    let sharp = checkerboard_2d((48, 52), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.2).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 24.0, 73).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (restored, report) = richardson_lucy_with(
        &degraded_image,
        &psf,
        &RichardsonLucy::new()
            .iterations(16)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )
    .unwrap();

    let restored_array = gray_to_array(&restored.to_luma8());
    assert!(restored_array.iter().all(|value| *value >= 0.0));
    assert!(is_finite_2d(&restored_array));
    assert!(report
        .objective_history
        .iter()
        .all(|value| value.is_finite()));
    assert!(report
        .residual_history
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
fn richardson_lucy_is_deterministic() {
    let sharp = checkerboard_2d((40, 44), 5, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.4).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 20.0, 31337).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());
    let config = RichardsonLucy::new()
        .iterations(18)
        .filter_epsilon(1e-3)
        .collect_history(true);

    let (first, first_report) = richardson_lucy_with(&degraded_image, &psf, &config).unwrap();
    let (second, second_report) = richardson_lucy_with(&degraded_image, &psf, &config).unwrap();

    let first_arr = gray_to_array(&first.to_luma8());
    let second_arr = gray_to_array(&second.to_luma8());
    assert!(arrays_equal_2d(&first_arr, &second_arr));
    assert_eq!(first_report.stop_reason, second_report.stop_reason);
    assert_eq!(
        first_report.objective_history,
        second_report.objective_history
    );
    assert_eq!(
        first_report.residual_history,
        second_report.residual_history
    );
}

#[test]
fn richardson_lucy_default_path_runs_and_is_finite() {
    let sharp = checkerboard_2d((36, 36), 3, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.1).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 26.0, 9090).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (restored, report) = richardson_lucy(&degraded_image, &psf).unwrap();
    let restored_array = gray_to_array(&restored.to_luma8());
    assert!(is_finite_2d(&restored_array));
    assert!(report.iterations >= 1);
}

#[test]
fn damped_weighted_path_respects_mask() {
    let sharp = checkerboard_2d((48, 48), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.2).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 24.0, 1201).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());
    let degraded_luma = degraded_image.to_luma8();

    let mut weights = Array2::ones((48, 48));
    for y in 16..32 {
        for x in 16..32 {
            weights[[y, x]] = 0.0;
        }
    }

    let (restored, _) = damped_richardson_lucy_with(
        &degraded_image,
        &psf,
        &RichardsonLucy::new()
            .iterations(14)
            .damping(Some(0.15))
            .weights(weights)
            .filter_epsilon(1e-3)
            .collect_history(false),
    )
    .unwrap();
    let restored_luma = restored.to_luma8();

    for y in 16..32 {
        for x in 16..32 {
            let x_u32 = x as u32;
            let y_u32 = y as u32;
            assert_eq!(
                restored_luma.get_pixel(x_u32, y_u32)[0],
                degraded_luma.get_pixel(x_u32, y_u32)[0]
            );
        }
    }
}

#[test]
fn damping_reduces_noise_amplification_on_noisy_fixture() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((11, 11), 1.8).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 12.0, 7701).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let config = RichardsonLucy::new()
        .iterations(36)
        .filter_epsilon(1e-3)
        .collect_history(false);
    let (plain, _) = richardson_lucy_with(&degraded_image, &psf, &config).unwrap();
    let (damped, _) =
        damped_richardson_lucy_with(&degraded_image, &psf, &config.clone().damping(Some(0.2)))
            .unwrap();

    let plain_tv = total_variation(&gray_to_array(&plain.to_luma8()));
    let damped_tv = total_variation(&gray_to_array(&damped.to_luma8()));
    assert!(damped_tv < plain_tv);
}

#[test]
fn readout_noise_path_is_finite_and_deterministic() {
    let sharp = checkerboard_2d((44, 40), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.4).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 20.0, 411).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let config = RichardsonLucy::new()
        .iterations(20)
        .damping(Some(0.12))
        .readout_noise(0.01)
        .filter_epsilon(1e-3)
        .collect_history(true);
    let (first, first_report) =
        damped_richardson_lucy_with(&degraded_image, &psf, &config).unwrap();
    let (second, second_report) =
        damped_richardson_lucy_with(&degraded_image, &psf, &config).unwrap();

    let first_arr = gray_to_array(&first.to_luma8());
    let second_arr = gray_to_array(&second.to_luma8());
    assert!(is_finite_2d(&first_arr));
    assert!(is_finite_2d(&second_arr));
    assert!(arrays_equal_2d(&first_arr, &second_arr));
    assert_eq!(
        first_report.objective_history,
        second_report.objective_history
    );
    assert_eq!(
        first_report.residual_history,
        second_report.residual_history
    );
}

#[test]
fn richardson_lucy_tv_has_lower_total_variation_than_plain_rl() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((11, 11), 1.9).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 10.0, 202603).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (plain, _) = richardson_lucy_with(
        &degraded_image,
        &psf,
        &RichardsonLucy::new()
            .iterations(34)
            .filter_epsilon(1e-3)
            .collect_history(false),
    )
    .unwrap();
    let (tv, _) = richardson_lucy_tv_with(
        &degraded_image,
        &psf,
        &RichardsonLucyTv::new()
            .iterations(34)
            .filter_epsilon(1e-3)
            .tv_weight(2.5e-2)
            .tv_epsilon(1e-3)
            .collect_history(false),
    )
    .unwrap();

    let plain_tv = total_variation(&gray_to_array(&plain.to_luma8()));
    let tv_regularized_tv = total_variation(&gray_to_array(&tv.to_luma8()));
    assert!(tv_regularized_tv < plain_tv);
}

#[test]
fn richardson_lucy_tv_improves_over_blurred_baseline() {
    let sharp = checkerboard_2d((60, 60), 5, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.5).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 24.0, 8080).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (restored, _) = richardson_lucy_tv_with(
        &degraded_image,
        &psf,
        &RichardsonLucyTv::new()
            .iterations(24)
            .filter_epsilon(1e-3)
            .tv_weight(1.2e-2)
            .collect_history(false),
    )
    .unwrap();

    let baseline_array = gray_to_array(&degraded_image.to_luma8());
    let restored_array = gray_to_array(&restored.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline_array).unwrap();
    let restored_psnr = psnr(&sharp, &restored_array).unwrap();
    assert!(restored_psnr > baseline_psnr);
}

#[test]
fn richardson_lucy_tv_is_nonnegative_finite_and_deterministic() {
    let sharp = checkerboard_2d((46, 50), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.4).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 18.0, 91011).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());
    let config = RichardsonLucyTv::new()
        .iterations(22)
        .filter_epsilon(1e-3)
        .tv_weight(1.8e-2)
        .collect_history(true);

    let (first, first_report) = richardson_lucy_tv_with(&degraded_image, &psf, &config).unwrap();
    let (second, second_report) = richardson_lucy_tv_with(&degraded_image, &psf, &config).unwrap();
    let (default_path, _) = richardson_lucy_tv(&degraded_image, &psf).unwrap();

    let first_arr = gray_to_array(&first.to_luma8());
    let second_arr = gray_to_array(&second.to_luma8());
    let default_arr = gray_to_array(&default_path.to_luma8());
    assert!(first_arr.iter().all(|value| *value >= 0.0));
    assert!(is_finite_2d(&first_arr));
    assert!(is_finite_2d(&default_arr));
    assert!(arrays_equal_2d(&first_arr, &second_arr));
    assert_eq!(
        first_report.objective_history,
        second_report.objective_history
    );
    assert_eq!(
        first_report.residual_history,
        second_report.residual_history
    );
}

#[test]
fn landweber_residual_decreases_and_improves_over_blurred_baseline() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.6).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 24.0, 551).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (restored, report) = landweber_with(
        &degraded_image,
        &psf,
        &Landweber::new()
            .iterations(28)
            .step_size(None)
            .collect_history(true),
    )
    .unwrap();

    let baseline_array = gray_to_array(&degraded_image.to_luma8());
    let restored_array = gray_to_array(&restored.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline_array).unwrap();
    let restored_psnr = psnr(&sharp, &restored_array).unwrap();
    assert!(restored_psnr > baseline_psnr);
    assert!(is_finite_2d(&restored_array));
    assert!(report.objective_history.len() >= 2);
    assert!(report.residual_history.len() >= 2);
    assert!(
        report.objective_history[report.objective_history.len() - 1] < report.objective_history[0]
    );
}

#[test]
fn van_cittert_residual_decreases_and_improves_over_blurred_baseline() {
    let sharp = checkerboard_2d((60, 60), 5, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.5).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&blurred).unwrap());

    let (restored, report) = van_cittert_with(
        &degraded_image,
        &psf,
        &VanCittert::new()
            .iterations(20)
            .step_size(Some(1.0))
            .collect_history(true),
    )
    .unwrap();

    let baseline_array = gray_to_array(&degraded_image.to_luma8());
    let restored_array = gray_to_array(&restored.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline_array).unwrap();
    let restored_psnr = psnr(&sharp, &restored_array).unwrap();
    assert!(restored_psnr > baseline_psnr);
    assert!(is_finite_2d(&restored_array));
    assert!(report.objective_history.len() >= 2);
    assert!(report.residual_history.len() >= 2);
    assert!(
        report.objective_history[report.objective_history.len() - 1] < report.objective_history[0]
    );
}

#[test]
fn landweber_and_van_cittert_are_deterministic() {
    let sharp = checkerboard_2d((42, 46), 3, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.2).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 22.0, 4567).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let landweber_config = Landweber::new()
        .iterations(18)
        .step_size(None)
        .collect_history(true);
    let (landweber_first, landweber_first_report) =
        landweber_with(&degraded_image, &psf, &landweber_config).unwrap();
    let (landweber_second, landweber_second_report) =
        landweber_with(&degraded_image, &psf, &landweber_config).unwrap();

    assert!(arrays_equal_2d(
        &gray_to_array(&landweber_first.to_luma8()),
        &gray_to_array(&landweber_second.to_luma8())
    ));
    assert_eq!(
        landweber_first_report.objective_history,
        landweber_second_report.objective_history
    );
    assert_eq!(
        landweber_first_report.residual_history,
        landweber_second_report.residual_history
    );

    let van_cittert_config = VanCittert::new()
        .iterations(18)
        .step_size(None)
        .collect_history(true);
    let (van_first, van_first_report) =
        van_cittert_with(&degraded_image, &psf, &van_cittert_config).unwrap();
    let (van_second, van_second_report) =
        van_cittert_with(&degraded_image, &psf, &van_cittert_config).unwrap();

    assert!(arrays_equal_2d(
        &gray_to_array(&van_first.to_luma8()),
        &gray_to_array(&van_second.to_luma8())
    ));
    assert_eq!(
        van_first_report.objective_history,
        van_second_report.objective_history
    );
    assert_eq!(
        van_first_report.residual_history,
        van_second_report.residual_history
    );
}

#[test]
fn landweber_and_van_cittert_default_paths_are_finite() {
    let sharp = checkerboard_2d((40, 40), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.1).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 26.0, 999).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (landweber_restored, landweber_report) = landweber(&degraded_image, &psf).unwrap();
    let (van_restored, van_report) = van_cittert(&degraded_image, &psf).unwrap();

    assert!(is_finite_2d(&gray_to_array(&landweber_restored.to_luma8())));
    assert!(is_finite_2d(&gray_to_array(&van_restored.to_luma8())));
    assert!(landweber_report.iterations >= 1);
    assert!(van_report.iterations >= 1);
}

fn array_to_gray(input: &Array2<f32>) -> deconvolution::Result<GrayImage> {
    let (height, width) = input.dim();
    let width_u32 = u32::try_from(width).map_err(|_| deconvolution::Error::DimensionMismatch)?;
    let height_u32 = u32::try_from(height).map_err(|_| deconvolution::Error::DimensionMismatch)?;
    let mut image = GrayImage::new(width_u32, height_u32);

    for y in 0..height {
        let y_u32 = u32::try_from(y).map_err(|_| deconvolution::Error::DimensionMismatch)?;
        for x in 0..width {
            let x_u32 = u32::try_from(x).map_err(|_| deconvolution::Error::DimensionMismatch)?;
            let value = (input[[y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
            image.put_pixel(x_u32, y_u32, Luma([value]));
        }
    }

    Ok(image)
}

fn gray_to_array(input: &GrayImage) -> Array2<f32> {
    let width = input.width() as usize;
    let height = input.height() as usize;
    let mut output = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let x_u32 = x as u32;
            let y_u32 = y as u32;
            output[[y, x]] = f32::from(input.get_pixel(x_u32, y_u32)[0]) / 255.0;
        }
    }

    output
}

fn mse(lhs: &Array2<f32>, rhs: &Array2<f32>) -> deconvolution::Result<f32> {
    if lhs.dim() != rhs.dim() {
        return Err(deconvolution::Error::DimensionMismatch);
    }
    if lhs.is_empty() {
        return Err(deconvolution::Error::InvalidParameter);
    }

    let mut sum = 0.0_f32;
    let count = lhs.len() as f32;
    for ((y, x), value) in lhs.indexed_iter() {
        let diff = *value - rhs[[y, x]];
        sum += diff * diff;
    }
    Ok(sum / count)
}

fn psnr(lhs: &Array2<f32>, rhs: &Array2<f32>) -> deconvolution::Result<f32> {
    let mse_value = mse(lhs, rhs)?;
    if mse_value <= f32::EPSILON {
        return Ok(f32::INFINITY);
    }
    Ok(10.0 * (1.0 / mse_value).log10())
}

fn arrays_equal_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> bool {
    lhs.dim() == rhs.dim()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn is_finite_2d(input: &Array2<f32>) -> bool {
    input.iter().all(|value| value.is_finite())
}

fn total_variation(input: &Array2<f32>) -> f32 {
    let (height, width) = input.dim();
    let mut tv = 0.0_f32;
    for y in 0..height {
        for x in 0..width {
            if y + 1 < height {
                tv += (input[[y + 1, x]] - input[[y, x]]).abs();
            }
            if x + 1 < width {
                tv += (input[[y, x + 1]] - input[[y, x]]).abs();
            }
        }
    }
    tv
}
