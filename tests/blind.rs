use deconvolution::blind::{
    maximum_likelihood, maximum_likelihood_with, parametric, parametric_with, richardson_lucy,
    richardson_lucy_with, BlindMaximumLikelihood, BlindOutput, BlindParametric, BlindReport,
    BlindRichardsonLucy, ParametricPsf,
};
use deconvolution::psf::{
    apply_constraint, apply_constraints, motion_linear, uniform, PsfConstraint,
};
use deconvolution::simulate::{add_poisson_noise, blur, checkerboard_2d};
use deconvolution::{Error, Kernel2D, StopReason};
use image::{DynamicImage, GrayImage, Luma};
use ndarray::{array, Array2};

#[test]
fn blind_output_and_report_structs_are_usable() {
    let report = BlindReport::new();
    assert_eq!(report.iterations, 0);
    assert_eq!(report.stop_reason, StopReason::MaxIterations);
    assert!(report.objective_history.is_empty());
    assert!(report.image_update_history.is_empty());
    assert!(report.psf_update_history.is_empty());

    let psf = Kernel2D::new(array![[1.0_f32]]).unwrap();
    let image = Array2::from_elem((2, 2), 0.5_f32);
    let output = BlindOutput { image, psf, report };
    assert_eq!(output.image.dim(), (2, 2));
    assert_eq!(output.psf.dims(), (1, 1));
}

#[test]
fn nonnegative_constraint_projects_negative_values() {
    let psf = Kernel2D::new(array![[-0.5_f32, 0.25_f32], [1.0_f32, -0.75_f32]]).unwrap();
    let projected = apply_constraint(&psf, &PsfConstraint::Nonnegative).unwrap();

    assert_eq!(projected.as_array()[[0, 0]], 0.0);
    assert_eq!(projected.as_array()[[0, 1]], 0.25);
    assert_eq!(projected.as_array()[[1, 0]], 1.0);
    assert_eq!(projected.as_array()[[1, 1]], 0.0);
}

#[test]
fn normalize_sum_constraint_produces_unit_sum() {
    let psf = Kernel2D::new(array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]]).unwrap();
    let projected = apply_constraint(&psf, &PsfConstraint::NormalizeSum).unwrap();
    assert!((projected.sum() - 1.0).abs() < 1e-6);
    assert!(projected.is_finite());
}

#[test]
fn support_mask_constraint_zeros_unsupported_entries() {
    let psf = Kernel2D::new(array![[0.2_f32, 0.3_f32], [0.4_f32, 0.1_f32]]).unwrap();
    let mask = array![[true, false], [false, true]];
    let projected = apply_constraint(&psf, &PsfConstraint::SupportMask(mask)).unwrap();

    assert_eq!(projected.as_array()[[0, 0]], 0.2);
    assert_eq!(projected.as_array()[[0, 1]], 0.0);
    assert_eq!(projected.as_array()[[1, 0]], 0.0);
    assert_eq!(projected.as_array()[[1, 1]], 0.1);
}

#[test]
fn constraints_can_be_composed_for_valid_projected_psf() {
    let psf = Kernel2D::new(array![[0.5_f32, -0.25_f32], [1.0_f32, -0.1_f32]]).unwrap();
    let mask = array![[true, false], [true, false]];
    let constraints = vec![
        PsfConstraint::Nonnegative,
        PsfConstraint::SupportMask(mask),
        PsfConstraint::NormalizeSum,
    ];
    let projected = apply_constraints(&psf, &constraints).unwrap();

    assert!(projected
        .as_array()
        .iter()
        .all(|value| *value >= 0.0 && value.is_finite()));
    assert_eq!(projected.as_array()[[0, 1]], 0.0);
    assert_eq!(projected.as_array()[[1, 1]], 0.0);
    assert!((projected.sum() - 1.0).abs() < 1e-6);
}

#[test]
fn support_mask_constraint_validates_shape() {
    let psf = Kernel2D::new(array![[1.0_f32, 0.0_f32], [0.0_f32, 1.0_f32]]).unwrap();
    let mask = Array2::from_elem((3, 3), true);
    let error = apply_constraint(&psf, &PsfConstraint::SupportMask(mask)).unwrap_err();
    assert_eq!(error, Error::DimensionMismatch);
}

#[test]
fn parametric_psfs_realize_to_finite_normalized_kernels() {
    let dims = (17_usize, 19_usize);
    let models = [
        ParametricPsf::Gaussian { sigma: 1.8 },
        ParametricPsf::MotionLinear {
            length: 11.0,
            angle_deg: 27.5,
        },
        ParametricPsf::Defocus { radius: 4.0 },
        ParametricPsf::OrientedGaussian {
            sigma_major: 2.4,
            sigma_minor: 1.1,
            angle_deg: 42.0,
        },
    ];

    for model in models {
        let kernel = model.realize(dims).unwrap();
        assert_eq!(kernel.dims(), dims);
        assert!(kernel.is_finite());
        assert!((kernel.sum() - 1.0).abs() < 1e-6);
    }
}

#[test]
fn parametric_psf_parameters_are_validated() {
    let invalid_gaussian = ParametricPsf::Gaussian { sigma: 0.0 };
    let invalid_motion = ParametricPsf::MotionLinear {
        length: -1.0,
        angle_deg: 0.0,
    };
    let invalid_defocus = ParametricPsf::Defocus { radius: 0.0 };
    let invalid_oriented = ParametricPsf::OrientedGaussian {
        sigma_major: 1.0,
        sigma_minor: 0.0,
        angle_deg: 0.0,
    };

    assert_eq!(
        invalid_gaussian.realize((9, 9)).unwrap_err(),
        Error::InvalidParameter
    );
    assert_eq!(
        invalid_motion.realize((9, 9)).unwrap_err(),
        Error::InvalidParameter
    );
    assert_eq!(
        invalid_defocus.realize((9, 9)).unwrap_err(),
        Error::InvalidParameter
    );
    assert_eq!(
        invalid_oriented.realize((9, 9)).unwrap_err(),
        Error::InvalidParameter
    );
}

#[test]
fn blind_richardson_lucy_improves_motion_blur_restoration() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let true_psf = motion_linear(11.0, 25.0).unwrap();
    let blurred = blur(&sharp, &true_psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 48.0, 991).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());
    let initial_psf = uniform(true_psf.dims()).unwrap();

    let output = richardson_lucy_with(
        &degraded_image,
        &initial_psf,
        &BlindRichardsonLucy::new()
            .iterations(24)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )
    .unwrap();

    let baseline = gray_to_array(&degraded_image.to_luma8());
    let restored = gray_to_array(&output.image.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline).unwrap();
    let restored_psnr = psnr(&sharp, &restored).unwrap();

    assert!(restored_psnr > baseline_psnr);
    assert_eq!(output.psf.dims(), true_psf.dims());
    assert!((output.psf.sum() - 1.0).abs() < 1e-6);
    assert!(output.psf.as_array().iter().all(|value| *value >= 0.0));
    assert!(output.report.iterations >= 1);
    assert_eq!(
        output.report.objective_history.len(),
        output.report.iterations
    );
    assert_eq!(
        output.report.image_update_history.len(),
        output.report.iterations
    );
    assert_eq!(
        output.report.psf_update_history.len(),
        output.report.iterations
    );
}

#[test]
fn blind_richardson_lucy_improves_psf_cross_correlation() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let true_psf = motion_linear(13.0, 35.0).unwrap();
    let blurred = blur(&sharp, &true_psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 64.0, 3117).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());
    let initial_psf = uniform(true_psf.dims()).unwrap();

    let initial_ncc =
        normalized_cross_correlation(true_psf.as_array(), initial_psf.as_array()).unwrap();
    let output = richardson_lucy_with(
        &degraded_image,
        &initial_psf,
        &BlindRichardsonLucy::new()
            .iterations(28)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )
    .unwrap();
    let final_ncc =
        normalized_cross_correlation(true_psf.as_array(), output.psf.as_array()).unwrap();

    assert!(final_ncc > initial_ncc);
}

#[test]
fn blind_richardson_lucy_support_mask_is_enforced() {
    let sharp = checkerboard_2d((56, 56), 4, 0.0, 1.0).unwrap();
    let true_psf = motion_linear(9.0, 0.0).unwrap();
    let blurred = blur(&sharp, &true_psf).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&blurred).unwrap());
    let initial_psf = uniform(true_psf.dims()).unwrap();

    let mut support = Array2::from_elem(true_psf.dims(), false);
    let (psf_h, psf_w) = true_psf.dims();
    let cy = psf_h / 2;
    for x in 0..psf_w {
        support[[cy, x]] = true;
    }

    let output = richardson_lucy_with(
        &degraded_image,
        &initial_psf,
        &BlindRichardsonLucy::new()
            .iterations(16)
            .filter_epsilon(1e-3)
            .support_mask(support.clone())
            .collect_history(true),
    )
    .unwrap();

    for y in 0..psf_h {
        for x in 0..psf_w {
            if !support[[y, x]] {
                assert!(output.psf.as_array()[[y, x]].abs() <= 1e-6);
            }
        }
    }
    assert!((output.psf.sum() - 1.0).abs() < 1e-6);
}

#[test]
fn blind_richardson_lucy_is_deterministic() {
    let sharp = checkerboard_2d((52, 52), 4, 0.0, 1.0).unwrap();
    let true_psf = motion_linear(9.0, 20.0).unwrap();
    let blurred = blur(&sharp, &true_psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 40.0, 44).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());
    let initial_psf = uniform(true_psf.dims()).unwrap();
    let config = BlindRichardsonLucy::new()
        .iterations(20)
        .filter_epsilon(1e-3)
        .collect_history(true);

    let first = richardson_lucy_with(&degraded_image, &initial_psf, &config).unwrap();
    let second = richardson_lucy_with(&degraded_image, &initial_psf, &config).unwrap();
    let default_path = richardson_lucy(&degraded_image, &initial_psf).unwrap();

    assert!(arrays_equal_2d_bits(
        &gray_to_array(&first.image.to_luma8()),
        &gray_to_array(&second.image.to_luma8())
    ));
    assert!(arrays_equal_2d_bits(
        first.psf.as_array(),
        second.psf.as_array()
    ));
    assert_eq!(
        first.report.objective_history,
        second.report.objective_history
    );
    assert_eq!(
        first.report.image_update_history,
        second.report.image_update_history
    );
    assert_eq!(
        first.report.psf_update_history,
        second.report.psf_update_history
    );
    assert_eq!(default_path.psf.dims(), initial_psf.dims());
}

#[test]
fn blind_maximum_likelihood_is_finite_and_restorative() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let true_psf = motion_linear(11.0, 30.0).unwrap();
    let blurred = blur(&sharp, &true_psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 48.0, 9303).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());
    let initial_psf = uniform(true_psf.dims()).unwrap();

    let output = maximum_likelihood_with(
        &degraded_image,
        &initial_psf,
        &BlindMaximumLikelihood::new()
            .iterations(24)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )
    .unwrap();

    let baseline = gray_to_array(&degraded_image.to_luma8());
    let restored = gray_to_array(&output.image.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline).unwrap();
    let restored_psnr = psnr(&sharp, &restored).unwrap();

    assert!(restored_psnr > baseline_psnr);
    assert!(output.psf.as_array().iter().all(|value| value.is_finite()));
    assert!((output.psf.sum() - 1.0).abs() < 1e-6);
    assert!(output.report.iterations >= 1);
}

#[test]
fn blind_maximum_likelihood_is_deterministic() {
    let sharp = checkerboard_2d((56, 56), 4, 0.0, 1.0).unwrap();
    let true_psf = motion_linear(9.0, 15.0).unwrap();
    let blurred = blur(&sharp, &true_psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 44.0, 1113).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());
    let initial_psf = uniform(true_psf.dims()).unwrap();
    let config = BlindMaximumLikelihood::new()
        .iterations(20)
        .filter_epsilon(1e-3)
        .collect_history(true);

    let first = maximum_likelihood_with(&degraded_image, &initial_psf, &config).unwrap();
    let second = maximum_likelihood_with(&degraded_image, &initial_psf, &config).unwrap();
    let default_path = maximum_likelihood(&degraded_image, &initial_psf).unwrap();

    assert!(arrays_equal_2d_bits(
        &gray_to_array(&first.image.to_luma8()),
        &gray_to_array(&second.image.to_luma8())
    ));
    assert!(arrays_equal_2d_bits(
        first.psf.as_array(),
        second.psf.as_array()
    ));
    assert_eq!(
        first.report.objective_history,
        second.report.objective_history
    );
    assert_eq!(
        first.report.image_update_history,
        second.report.image_update_history
    );
    assert_eq!(
        first.report.psf_update_history,
        second.report.psf_update_history
    );
    assert_eq!(default_path.psf.dims(), initial_psf.dims());
}

#[test]
fn blind_parametric_motion_estimate_improves_over_initial_guess() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let true_psf = motion_linear(13.0, 38.0).unwrap();
    let blurred = blur(&sharp, &true_psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 72.0, 707).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let initial_model = ParametricPsf::MotionLinear {
        length: 5.0,
        angle_deg: -25.0,
    };
    let initial_psf = initial_model.realize(true_psf.dims()).unwrap();
    let initial_ncc =
        normalized_cross_correlation(true_psf.as_array(), initial_psf.as_array()).unwrap();

    let output = parametric_with(
        &degraded_image,
        initial_model,
        true_psf.dims(),
        &BlindParametric::new()
            .iterations(12)
            .image_iterations(12)
            .initial_step_scale(0.45)
            .min_step_scale(0.02)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )
    .unwrap();
    let final_ncc =
        normalized_cross_correlation(true_psf.as_array(), output.psf.as_array()).unwrap();

    assert!(final_ncc > initial_ncc);
    assert!((output.psf.sum() - 1.0).abs() < 1e-6);
    assert!(output.psf.as_array().iter().all(|value| *value >= 0.0));

    let default_path = parametric(&degraded_image, initial_model, true_psf.dims()).unwrap();
    assert_eq!(default_path.psf.dims(), true_psf.dims());
}

#[test]
fn blind_parametric_is_deterministic() {
    let sharp = checkerboard_2d((60, 60), 4, 0.0, 1.0).unwrap();
    let true_psf = motion_linear(11.0, 32.0).unwrap();
    let blurred = blur(&sharp, &true_psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 60.0, 812).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());
    let initial_model = ParametricPsf::MotionLinear {
        length: 6.0,
        angle_deg: -20.0,
    };
    let config = BlindParametric::new()
        .iterations(10)
        .image_iterations(10)
        .initial_step_scale(0.4)
        .min_step_scale(0.02)
        .filter_epsilon(1e-3)
        .collect_history(true);

    let first = parametric_with(&degraded_image, initial_model, true_psf.dims(), &config).unwrap();
    let second = parametric_with(&degraded_image, initial_model, true_psf.dims(), &config).unwrap();

    assert!(arrays_equal_2d_bits(
        &gray_to_array(&first.image.to_luma8()),
        &gray_to_array(&second.image.to_luma8())
    ));
    assert!(arrays_equal_2d_bits(
        first.psf.as_array(),
        second.psf.as_array()
    ));
    assert_eq!(
        first.report.objective_history,
        second.report.objective_history
    );
    assert_eq!(
        first.report.image_update_history,
        second.report.image_update_history
    );
    assert_eq!(
        first.report.psf_update_history,
        second.report.psf_update_history
    );
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
            output[[y, x]] = f32::from(input.get_pixel(x as u32, y as u32)[0]) / 255.0;
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

fn normalized_cross_correlation(
    lhs: &Array2<f32>,
    rhs: &Array2<f32>,
) -> deconvolution::Result<f32> {
    if lhs.dim() != rhs.dim() {
        return Err(deconvolution::Error::DimensionMismatch);
    }
    if lhs.is_empty() {
        return Err(deconvolution::Error::InvalidParameter);
    }

    let mut lhs_norm = 0.0_f32;
    let mut rhs_norm = 0.0_f32;
    let mut cross = 0.0_f32;
    for ((y, x), lhs_value) in lhs.indexed_iter() {
        let rhs_value = rhs[[y, x]];
        cross += *lhs_value * rhs_value;
        lhs_norm += *lhs_value * *lhs_value;
        rhs_norm += rhs_value * rhs_value;
    }

    let denom = lhs_norm.sqrt() * rhs_norm.sqrt();
    if !denom.is_finite() || denom <= f32::EPSILON {
        return Err(deconvolution::Error::InvalidParameter);
    }

    let ncc = cross / denom;
    if !ncc.is_finite() {
        return Err(deconvolution::Error::NonFiniteInput);
    }
    Ok(ncc)
}

fn arrays_equal_2d_bits(lhs: &Array2<f32>, rhs: &Array2<f32>) -> bool {
    lhs.dim() == rhs.dim()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}
