use deconvolution::psf::gaussian2d;
use deconvolution::simulate::{add_poisson_noise, blur, checkerboard_2d};
use deconvolution::{richardson_lucy, richardson_lucy_with, RichardsonLucy};
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
