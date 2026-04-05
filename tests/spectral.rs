mod common;

use deconvolution::otf::psf2otf;
use deconvolution::psf::{delta2d, gaussian2d, motion_linear};
use deconvolution::simulate::{
    add_gaussian_noise, add_poisson_noise, add_readout_noise, blur, blur_otf, checkerboard_2d,
    degrade, gaussian_blob_2d, phantom_3d, rgb_edges_2d,
};
use deconvolution::{
    inverse_filter, inverse_filter_with, naive_inverse_filter, regularized_inverse_filter_with,
    tikhonov_inverse_filter_with, truncated_inverse_filter_with, wiener, wiener_with,
    InverseFilter, Padding, RegOperator2D, RegularizedInverseFilter, TikhonovInverseFilter,
    Transfer2D, Wiener,
};
use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};
use ndarray::{array, Array2};
use num_complex::Complex32;

use common::{arrays_differ_2d, arrays_equal_2d, arrays_equal_3d, is_finite_2d, is_finite_3d};

#[test]
fn same_seed_produces_same_noise_and_different_seed_changes_output() {
    let input = gaussian_blob_2d((48, 40), 6.0).unwrap();

    let g1 = add_gaussian_noise(&input, 0.05, 7).unwrap();
    let g2 = add_gaussian_noise(&input, 0.05, 7).unwrap();
    let g3 = add_gaussian_noise(&input, 0.05, 8).unwrap();
    assert!(arrays_equal_2d(&g1, &g2));
    assert!(arrays_differ_2d(&g1, &g3));

    let p1 = add_poisson_noise(&input, 32.0, 7).unwrap();
    let p2 = add_poisson_noise(&input, 32.0, 7).unwrap();
    let p3 = add_poisson_noise(&input, 32.0, 8).unwrap();
    assert!(arrays_equal_2d(&p1, &p2));
    assert!(arrays_differ_2d(&p1, &p3));

    let r1 = add_readout_noise(&input, 0.01, 19).unwrap();
    let r2 = add_readout_noise(&input, 0.01, 19).unwrap();
    let r3 = add_readout_noise(&input, 0.01, 20).unwrap();
    assert!(arrays_equal_2d(&r1, &r2));
    assert!(arrays_differ_2d(&r1, &r3));
}

#[test]
fn blur_and_degrade_preserve_dimensions() {
    let input = checkerboard_2d((32, 44), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.2).unwrap();

    let blurred = blur(&input, &psf).unwrap();
    assert_eq!(blurred.dim(), input.dim());
    assert!(is_finite_2d(&blurred));

    let otf = psf2otf(&psf, input.dim()).unwrap();
    let blurred_otf = blur_otf(&input, &otf).unwrap();
    assert_eq!(blurred_otf.dim(), input.dim());
    assert!(is_finite_2d(&blurred_otf));

    let degraded = degrade(&input, &psf, Some(0.03), Some(30.0), Some(0.01), 1234).unwrap();
    assert_eq!(degraded.dim(), input.dim());
    assert!(is_finite_2d(&degraded));
}

#[test]
fn fixtures_have_expected_shapes_ranges_and_finiteness() {
    let checker = checkerboard_2d((24, 30), 3, 0.25, 0.75).unwrap();
    assert_eq!(checker.dim(), (24, 30));
    assert!(checker.iter().all(|value| *value == 0.25 || *value == 0.75));
    assert!(is_finite_2d(&checker));

    let blob = gaussian_blob_2d((25, 25), 4.0).unwrap();
    assert_eq!(blob.dim(), (25, 25));
    assert!(blob.iter().all(|value| *value >= 0.0 && *value <= 1.0));
    assert!(blob[[12, 12]] >= blob[[0, 0]]);
    assert!(is_finite_2d(&blob));

    let rgb = rgb_edges_2d((18, 22)).unwrap();
    assert_eq!(rgb.width(), 22);
    assert_eq!(rgb.height(), 18);
    let p00 = rgb.get_pixel(0, 0).0;
    let p11 = rgb.get_pixel(21, 17).0;
    assert_ne!(p00, p11);

    let vol = phantom_3d((16, 20, 22)).unwrap();
    let vol_again = phantom_3d((16, 20, 22)).unwrap();
    assert_eq!(vol.dim(), (16, 20, 22));
    assert!(vol.iter().all(|value| *value >= 0.0 && *value <= 1.0));
    assert!(is_finite_3d(&vol));
    assert!(arrays_equal_3d(&vol, &vol_again));
}

#[test]
fn delta_psf_behaves_like_identity() {
    let mut input = GrayImage::new(24, 16);
    for y in 0..16_u32 {
        for x in 0..24_u32 {
            let value = ((13 * x + 7 * y) % 251) as u8;
            input.put_pixel(x, y, Luma([value]));
        }
    }
    let dynamic = DynamicImage::ImageLuma8(input.clone());
    let delta = delta2d((3, 3)).unwrap();

    let restored = naive_inverse_filter(&dynamic, &delta).unwrap();
    let restored_luma = restored.to_luma8();
    assert_eq!(restored_luma.dimensions(), input.dimensions());

    let mut max_abs_err = 0_i16;
    for y in 0..16_u32 {
        for x in 0..24_u32 {
            let lhs = i16::from(restored_luma.get_pixel(x, y)[0]);
            let rhs = i16::from(input.get_pixel(x, y)[0]);
            max_abs_err = max_abs_err.max((lhs - rhs).abs());
        }
    }
    assert!(max_abs_err <= 1);
}

#[test]
fn noiseless_blur_fixture_improves_over_blurred_baseline() {
    let sharp = checkerboard_2d((56, 56), 3, 0.0, 1.0).unwrap();
    let psf = gaussian2d((11, 11), 2.0).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();

    let sharp_img = array_to_gray(&sharp).unwrap();
    let blurred_img = array_to_gray(&blurred).unwrap();
    let config = InverseFilter::new()
        .stabilization_floor(1e-2)
        .truncation_cutoff(0.06)
        .padding(Padding::None);
    let restored = truncated_inverse_filter_with(
        &DynamicImage::ImageLuma8(blurred_img.clone()),
        &psf,
        &config,
    )
    .unwrap();
    let restored_array = gray_to_array(&restored.to_luma8());
    let sharp_array = gray_to_array(&sharp_img);
    let blurred_array = gray_to_array(&blurred_img);

    let blurred_mse = mse(&sharp_array, &blurred_array).unwrap();
    let restored_mse = mse(&sharp_array, &restored_array).unwrap();

    assert!(
        restored_mse < blurred_mse,
        "restored_mse={restored_mse}, blurred_mse={blurred_mse}"
    );
}

#[test]
fn stabilized_variants_avoid_non_finite_output() {
    let input = checkerboard_2d((40, 40), 4, 0.0, 1.0).unwrap();
    let blur_psf = motion_linear(13.0, 25.0).unwrap();
    let blurred = blur(&input, &blur_psf).unwrap();
    let blurred_img = DynamicImage::ImageLuma8(array_to_gray(&blurred).unwrap());

    let config = InverseFilter::new()
        .stabilization_floor(1e-2)
        .truncation_cutoff(5e-2)
        .padding(Padding::NextFastLen);

    let restored_inverse = inverse_filter_with(&blurred_img, &blur_psf, &config).unwrap();
    let restored_truncated =
        truncated_inverse_filter_with(&blurred_img, &blur_psf, &config).unwrap();

    assert_eq!(restored_inverse.to_luma8().dimensions(), (40, 40));
    assert_eq!(restored_truncated.to_luma8().dimensions(), (40, 40));
}

#[test]
fn dimensions_and_alpha_are_preserved() {
    let mut rgba = RgbaImage::new(31, 23);
    for y in 0..23_u32 {
        for x in 0..31_u32 {
            let r = ((11 * x + 5 * y) % 256) as u8;
            let g = ((3 * x + 17 * y) % 256) as u8;
            let b = ((7 * x + 13 * y) % 256) as u8;
            let a = ((19 * x + 29 * y) % 256) as u8;
            rgba.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }

    let psf = gaussian2d((7, 7), 1.2).unwrap();
    let restored = inverse_filter(&DynamicImage::ImageRgba8(rgba.clone()), &psf).unwrap();
    let restored_rgba = restored.to_rgba8();

    assert_eq!(restored_rgba.dimensions(), rgba.dimensions());
    for y in 0..23_u32 {
        for x in 0..31_u32 {
            assert_eq!(restored_rgba.get_pixel(x, y)[3], rgba.get_pixel(x, y)[3]);
        }
    }
}

#[test]
fn regularized_inverse_improves_over_plain_inverse_on_noisy_fixture() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.8).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let noisy = add_gaussian_noise(&blurred, 0.03, 404).unwrap();

    let sharp_img = array_to_gray(&sharp).unwrap();
    let noisy_img = DynamicImage::ImageLuma8(array_to_gray(&noisy).unwrap());

    let plain = inverse_filter_with(
        &noisy_img,
        &psf,
        &InverseFilter::new()
            .stabilization_floor(1e-3)
            .padding(Padding::NextFastLen),
    )
    .unwrap();
    let regularized = regularized_inverse_filter_with(
        &noisy_img,
        &psf,
        &RegularizedInverseFilter::new()
            .lambda(0.02)
            .stabilization_floor(1e-3)
            .padding(Padding::NextFastLen),
    )
    .unwrap();

    let sharp_array = gray_to_array(&sharp_img);
    let plain_array = gray_to_array(&plain.to_luma8());
    let regularized_array = gray_to_array(&regularized.to_luma8());
    let plain_mse = mse(&sharp_array, &plain_array).unwrap();
    let regularized_mse = mse(&sharp_array, &regularized_array).unwrap();

    assert!(regularized_mse < plain_mse);
}

#[test]
fn custom_regularizer_paths_are_shape_preserving_and_finite() {
    let input = checkerboard_2d((48, 40), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.4).unwrap();
    let blurred = blur(&input, &psf).unwrap();
    let noisy = add_gaussian_noise(&blurred, 0.02, 111).unwrap();
    let noisy_img = DynamicImage::ImageLuma8(array_to_gray(&noisy).unwrap());

    let reg_kernel = deconvolution::Kernel2D::new(array![
        [0.0_f32, -1.0_f32, 0.0_f32],
        [-1.0_f32, 4.0_f32, -1.0_f32],
        [0.0_f32, -1.0_f32, 0.0_f32]
    ])
    .unwrap();

    let out_kernel = regularized_inverse_filter_with(
        &noisy_img,
        &psf,
        &RegularizedInverseFilter::new()
            .lambda(0.015)
            .regularizer(RegOperator2D::CustomKernel(&reg_kernel))
            .padding(Padding::None),
    )
    .unwrap();

    let transfer = psf2otf(&reg_kernel, noisy.dim()).unwrap();
    let out_transfer = regularized_inverse_filter_with(
        &noisy_img,
        &psf,
        &RegularizedInverseFilter::new()
            .lambda(0.015)
            .regularizer(RegOperator2D::CustomTransfer(&transfer))
            .padding(Padding::None),
    )
    .unwrap();

    let kernel_arr = gray_to_array(&out_kernel.to_luma8());
    let transfer_arr = gray_to_array(&out_transfer.to_luma8());
    assert_eq!(kernel_arr.dim(), input.dim());
    assert_eq!(transfer_arr.dim(), input.dim());
    assert!(kernel_arr.iter().all(|value| value.is_finite()));
    assert!(transfer_arr.iter().all(|value| value.is_finite()));
}

#[test]
fn tikhonov_path_is_finite_and_shape_preserving() {
    let input = checkerboard_2d((40, 52), 5, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.1).unwrap();
    let blurred = blur(&input, &psf).unwrap();
    let noisy = add_gaussian_noise(&blurred, 0.02, 73).unwrap();
    let noisy_img = DynamicImage::ImageLuma8(array_to_gray(&noisy).unwrap());

    let output = tikhonov_inverse_filter_with(
        &noisy_img,
        &psf,
        &TikhonovInverseFilter::new()
            .lambda(0.01)
            .stabilization_floor(1e-3)
            .padding(Padding::NextFastLen),
    )
    .unwrap();

    let output_array = gray_to_array(&output.to_luma8());
    assert_eq!(output_array.dim(), input.dim());
    assert!(output_array.iter().all(|value| value.is_finite()));
}

#[test]
fn wiener_known_nsr_improves_psnr_on_noisy_fixture() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.6).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let noisy = add_gaussian_noise(&blurred, 0.04, 512).unwrap();

    let sharp_img = array_to_gray(&sharp).unwrap();
    let noisy_img = DynamicImage::ImageLuma8(array_to_gray(&noisy).unwrap());
    let inverse_out = inverse_filter_with(
        &noisy_img,
        &psf,
        &InverseFilter::new()
            .stabilization_floor(1e-3)
            .padding(Padding::NextFastLen),
    )
    .unwrap();
    let wiener_out = wiener_with(
        &noisy_img,
        &psf,
        &Wiener::new().nsr(0.01).padding(Padding::NextFastLen),
    )
    .unwrap();

    let sharp_arr = gray_to_array(&sharp_img);
    let inverse_arr = gray_to_array(&inverse_out.to_luma8());
    let wiener_arr = gray_to_array(&wiener_out.to_luma8());
    let inverse_psnr = psnr(&sharp_arr, &inverse_arr).unwrap();
    let wiener_psnr = psnr(&sharp_arr, &wiener_arr).unwrap();
    assert!(wiener_psnr > inverse_psnr);
}

#[test]
fn wiener_nsr_zero_matches_inverse_on_noiseless_fixture() {
    let sharp = checkerboard_2d((48, 48), 3, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.2).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let blurred_img = DynamicImage::ImageLuma8(array_to_gray(&blurred).unwrap());

    let inverse_out = inverse_filter_with(
        &blurred_img,
        &psf,
        &InverseFilter::new()
            .stabilization_floor(1e-3)
            .padding(Padding::None),
    )
    .unwrap();
    let wiener_out = wiener_with(
        &blurred_img,
        &psf,
        &Wiener::new().nsr(0.0).padding(Padding::None),
    )
    .unwrap();

    let inverse_arr = gray_to_array(&inverse_out.to_luma8());
    let wiener_arr = gray_to_array(&wiener_out.to_luma8());
    let delta = mse(&inverse_arr, &wiener_arr).unwrap();
    assert!(delta < 1e-4);
}

#[test]
fn wiener_dimensions_and_alpha_are_preserved() {
    let mut rgba = RgbaImage::new(33, 25);
    for y in 0..25_u32 {
        for x in 0..33_u32 {
            let r = ((7 * x + 5 * y) % 256) as u8;
            let g = ((11 * x + 3 * y) % 256) as u8;
            let b = ((13 * x + 17 * y) % 256) as u8;
            let a = ((19 * x + 23 * y) % 256) as u8;
            rgba.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }

    let psf = gaussian2d((7, 7), 1.1).unwrap();
    let restored = wiener(&DynamicImage::ImageRgba8(rgba.clone()), &psf).unwrap();
    let restored_rgba = restored.to_rgba8();
    assert_eq!(restored_rgba.dimensions(), rgba.dimensions());
    for y in 0..25_u32 {
        for x in 0..33_u32 {
            assert_eq!(restored_rgba.get_pixel(x, y)[3], rgba.get_pixel(x, y)[3]);
        }
    }
}

#[test]
fn wiener_autocorrelation_form_is_finite_and_shape_preserving() {
    let input = checkerboard_2d((40, 44), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.4).unwrap();
    let blurred = blur(&input, &psf).unwrap();
    let noisy_img = DynamicImage::ImageLuma8(array_to_gray(&blurred).unwrap());

    let dims = blurred.dim();
    let noise_psd = Transfer2D::new(Array2::from_elem(dims, Complex32::new(0.1, 0.0))).unwrap();
    let image_psd = Transfer2D::new(Array2::from_elem(dims, Complex32::new(1.0, 0.0))).unwrap();
    let restored = wiener_with(
        &noisy_img,
        &psf,
        &Wiener::new()
            .noise_autocorr(noise_psd)
            .image_autocorr(image_psd)
            .padding(Padding::None),
    )
    .unwrap();

    let restored_arr = gray_to_array(&restored.to_luma8());
    assert_eq!(restored_arr.dim(), input.dim());
    assert!(restored_arr.iter().all(|value| value.is_finite()));
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
