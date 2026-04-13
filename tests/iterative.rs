use deconvolution::psf::gaussian2d;
use deconvolution::simulate::{add_poisson_noise, blur, checkerboard_2d};
use deconvolution::{
    bvls, bvls_with, cgls, cgls_with, damped_richardson_lucy_with, fista, fista_with, ictm,
    ictm_with, ista, ista_with, landweber, landweber_with, mrnsd, mrnsd_with, nnls, nnls_with,
    richardson_lucy, richardson_lucy_tv, richardson_lucy_tv_with, richardson_lucy_with,
    tikhonov_miller, tikhonov_miller_with, van_cittert, van_cittert_with, Bvls, Cgls, Fista, Ictm,
    Ista, Landweber, Mrnsd, Nnls, RichardsonLucy, RichardsonLucyTv, SparseBasis, TikhonovMiller,
    VanCittert,
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

#[test]
fn tikhonov_miller_improves_over_blurred_baseline_on_noisy_input() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.6).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 96.0, 42042).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (restored, report) = tikhonov_miller_with(
        &degraded_image,
        &psf,
        &TikhonovMiller::new()
            .iterations(18)
            .lambda(5.0e-4)
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
}

#[test]
fn ictm_is_nonnegative_and_improves_over_blurred_baseline_on_noisy_input() {
    let sharp = checkerboard_2d((62, 58), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.4).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 88.0, 90919).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (restored, report) = ictm_with(
        &degraded_image,
        &psf,
        &Ictm::new()
            .iterations(18)
            .lambda(6.0e-4)
            .collect_history(true),
    )
    .unwrap();

    let baseline_array = gray_to_array(&degraded_image.to_luma8());
    let restored_array = gray_to_array(&restored.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline_array).unwrap();
    let restored_psnr = psnr(&sharp, &restored_array).unwrap();
    assert!(restored_psnr > baseline_psnr);
    assert!(restored_array.iter().all(|value| *value >= 0.0));
    assert!(is_finite_2d(&restored_array));
    assert!(report.objective_history.len() >= 2);
}

#[test]
fn tikhonov_miller_and_ictm_default_paths_are_finite() {
    let sharp = checkerboard_2d((44, 44), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.2).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 24.0, 313).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (tikhonov_restored, tikhonov_report) = tikhonov_miller(&degraded_image, &psf).unwrap();
    let (ictm_restored, ictm_report) = ictm(&degraded_image, &psf).unwrap();

    let tikhonov_array = gray_to_array(&tikhonov_restored.to_luma8());
    let ictm_array = gray_to_array(&ictm_restored.to_luma8());
    assert!(is_finite_2d(&tikhonov_array));
    assert!(is_finite_2d(&ictm_array));
    assert!(ictm_array.iter().all(|value| *value >= 0.0));
    assert!(tikhonov_report.iterations >= 1);
    assert!(ictm_report.iterations >= 1);
}

#[test]
fn nnls_is_nonnegative_and_improves_over_blurred_baseline() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.6).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 28.0, 1119).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (restored, report) = nnls_with(
        &degraded_image,
        &psf,
        &Nnls::new()
            .iterations(24)
            .step_size(None)
            .collect_history(true),
    )
    .unwrap();

    let baseline_array = gray_to_array(&degraded_image.to_luma8());
    let restored_array = gray_to_array(&restored.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline_array).unwrap();
    let restored_psnr = psnr(&sharp, &restored_array).unwrap();
    assert!(restored_psnr > baseline_psnr);
    assert!(restored_array.iter().all(|value| *value >= 0.0));
    assert!(is_finite_2d(&restored_array));
    assert!(!report.objective_history.is_empty());
    assert!(!report.residual_history.is_empty());
    let first_objective = report.objective_history[0];
    let last_objective = *report.objective_history.last().unwrap();
    let first_residual = report.residual_history[0];
    let last_residual = *report.residual_history.last().unwrap();
    assert!(last_objective <= first_objective * 1.01 + 1e-4);
    assert!(last_residual <= first_residual * 1.01 + 1e-4);
}

#[test]
fn bvls_respects_custom_bounds_and_objective_stabilizes() {
    let sharp = checkerboard_2d((62, 58), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.4).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 24.0, 7772).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let lower_bound = 12.0_f32;
    let upper_bound = 240.0_f32;
    let (restored, report) = bvls_with(
        &degraded_image,
        &psf,
        &Bvls::new()
            .iterations(26)
            .lower_bound(lower_bound)
            .upper_bound(upper_bound)
            .collect_history(true),
    )
    .unwrap();

    let restored_array = gray_to_array(&restored.to_luma8());
    assert!(restored_array
        .iter()
        .all(|value| *value >= lower_bound / 255.0 && *value <= upper_bound / 255.0));
    assert!(is_finite_2d(&restored_array));
    assert!(objective_stabilizes(report.objective_history.as_slice()));
}

#[test]
fn nnls_and_bvls_are_deterministic() {
    let sharp = checkerboard_2d((46, 50), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.4).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 20.0, 90117).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let nnls_config = Nnls::new().iterations(20).collect_history(true);
    let (nnls_first, nnls_first_report) = nnls_with(&degraded_image, &psf, &nnls_config).unwrap();
    let (nnls_second, nnls_second_report) = nnls_with(&degraded_image, &psf, &nnls_config).unwrap();
    assert!(arrays_equal_2d(
        &gray_to_array(&nnls_first.to_luma8()),
        &gray_to_array(&nnls_second.to_luma8())
    ));
    assert_eq!(
        nnls_first_report.objective_history,
        nnls_second_report.objective_history
    );
    assert_eq!(
        nnls_first_report.residual_history,
        nnls_second_report.residual_history
    );

    let bvls_config = Bvls::new()
        .iterations(20)
        .lower_bound(0.0)
        .upper_bound(255.0)
        .collect_history(true);
    let (bvls_first, bvls_first_report) = bvls_with(&degraded_image, &psf, &bvls_config).unwrap();
    let (bvls_second, bvls_second_report) = bvls_with(&degraded_image, &psf, &bvls_config).unwrap();
    assert!(arrays_equal_2d(
        &gray_to_array(&bvls_first.to_luma8()),
        &gray_to_array(&bvls_second.to_luma8())
    ));
    assert_eq!(
        bvls_first_report.objective_history,
        bvls_second_report.objective_history
    );
    assert_eq!(
        bvls_first_report.residual_history,
        bvls_second_report.residual_history
    );
}

#[test]
fn nnls_and_bvls_default_paths_are_finite_and_improve_over_blurred_baseline() {
    let sharp = checkerboard_2d((44, 44), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.2).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&blurred).unwrap());

    let (nnls_restored, nnls_report) = nnls(&degraded_image, &psf).unwrap();
    let (bvls_restored, bvls_report) = bvls(&degraded_image, &psf).unwrap();

    let baseline_array = gray_to_array(&degraded_image.to_luma8());
    let nnls_array = gray_to_array(&nnls_restored.to_luma8());
    let bvls_array = gray_to_array(&bvls_restored.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline_array).unwrap();
    let nnls_psnr = psnr(&sharp, &nnls_array).unwrap();
    let bvls_psnr = psnr(&sharp, &bvls_array).unwrap();

    assert!(nnls_psnr > baseline_psnr);
    assert!(bvls_psnr > baseline_psnr);
    assert!(is_finite_2d(&nnls_array));
    assert!(is_finite_2d(&bvls_array));
    assert!(nnls_array.iter().all(|value| *value >= 0.0));
    assert!(bvls_array.iter().all(|value| *value >= 0.0));
    assert!(nnls_report.iterations >= 1);
    assert!(bvls_report.iterations >= 1);
}

#[test]
fn ista_and_fista_improve_over_blurred_baseline_and_are_finite() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.6).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 24.0, 1917).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let config_ista = Ista::new()
        .iterations(24)
        .lambda(0.35)
        .basis(SparseBasis::Pixel)
        .collect_history(true);
    let config_fista = Fista::new()
        .iterations(24)
        .lambda(0.35)
        .basis(SparseBasis::Pixel)
        .collect_history(true);
    let (ista_restored, ista_report) = ista_with(&degraded_image, &psf, &config_ista).unwrap();
    let (fista_restored, fista_report) = fista_with(&degraded_image, &psf, &config_fista).unwrap();

    let baseline_array = gray_to_array(&degraded_image.to_luma8());
    let ista_array = gray_to_array(&ista_restored.to_luma8());
    let fista_array = gray_to_array(&fista_restored.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline_array).unwrap();
    let ista_psnr = psnr(&sharp, &ista_array).unwrap();
    let fista_psnr = psnr(&sharp, &fista_array).unwrap();

    assert!(ista_psnr > baseline_psnr);
    assert!(fista_psnr > baseline_psnr);
    assert!(is_finite_2d(&ista_array));
    assert!(is_finite_2d(&fista_array));
    assert!(ista_array.iter().all(|value| *value >= 0.0));
    assert!(fista_array.iter().all(|value| *value >= 0.0));
    assert!(ista_report.objective_history.len() >= 2);
    assert!(fista_report.objective_history.len() >= 2);
}

#[test]
fn fista_converges_faster_than_ista_with_equal_iterations() {
    let sharp = checkerboard_2d((60, 60), 5, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.4).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&blurred).unwrap());

    let iterations = 20usize;
    let config_ista = Ista::new()
        .iterations(iterations)
        .lambda(0.3)
        .basis(SparseBasis::Pixel)
        .collect_history(true);
    let config_fista = Fista::new()
        .iterations(iterations)
        .lambda(0.3)
        .basis(SparseBasis::Pixel)
        .collect_history(true);
    let (_, ista_report) = ista_with(&degraded_image, &psf, &config_ista).unwrap();
    let (_, fista_report) = fista_with(&degraded_image, &psf, &config_fista).unwrap();

    let ista_last_objective = *ista_report.objective_history.last().unwrap();
    let fista_last_objective = *fista_report.objective_history.last().unwrap();
    let ista_last_residual = *ista_report.residual_history.last().unwrap();
    let fista_last_residual = *fista_report.residual_history.last().unwrap();
    assert!(
        fista_last_objective <= ista_last_objective || fista_last_residual <= ista_last_residual
    );
}

#[test]
fn haar_basis_paths_are_finite_and_deterministic() {
    let sharp = checkerboard_2d((46, 50), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.3).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 22.0, 55511).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let ista_config = Ista::new()
        .iterations(18)
        .lambda(0.3)
        .basis(SparseBasis::Haar)
        .collect_history(true);
    let fista_config = Fista::new()
        .iterations(18)
        .lambda(0.3)
        .basis(SparseBasis::Haar)
        .collect_history(true);

    let (ista_first, ista_first_report) = ista_with(&degraded_image, &psf, &ista_config).unwrap();
    let (ista_second, ista_second_report) = ista_with(&degraded_image, &psf, &ista_config).unwrap();
    let (fista_first, fista_first_report) =
        fista_with(&degraded_image, &psf, &fista_config).unwrap();
    let (fista_second, fista_second_report) =
        fista_with(&degraded_image, &psf, &fista_config).unwrap();

    let ista_first_array = gray_to_array(&ista_first.to_luma8());
    let ista_second_array = gray_to_array(&ista_second.to_luma8());
    let fista_first_array = gray_to_array(&fista_first.to_luma8());
    let fista_second_array = gray_to_array(&fista_second.to_luma8());
    assert!(is_finite_2d(&ista_first_array));
    assert!(is_finite_2d(&fista_first_array));
    assert!(arrays_equal_2d(&ista_first_array, &ista_second_array));
    assert!(arrays_equal_2d(&fista_first_array, &fista_second_array));
    assert_eq!(
        ista_first_report.objective_history,
        ista_second_report.objective_history
    );
    assert_eq!(
        fista_first_report.objective_history,
        fista_second_report.objective_history
    );
}

#[test]
fn ista_and_fista_default_paths_are_finite() {
    let sharp = checkerboard_2d((44, 44), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.2).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 20.0, 4777).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (ista_restored, ista_report) = ista(&degraded_image, &psf).unwrap();
    let (fista_restored, fista_report) = fista(&degraded_image, &psf).unwrap();

    let ista_array = gray_to_array(&ista_restored.to_luma8());
    let fista_array = gray_to_array(&fista_restored.to_luma8());
    assert!(is_finite_2d(&ista_array));
    assert!(is_finite_2d(&fista_array));
    assert!(ista_array.iter().all(|value| *value >= 0.0));
    assert!(fista_array.iter().all(|value| *value >= 0.0));
    assert!(ista_report.iterations >= 1);
    assert!(fista_report.iterations >= 1);
}

#[test]
fn mrnsd_is_nonnegative_and_improves_over_blurred_baseline() {
    let sharp = checkerboard_2d((64, 64), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.6).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 30.0, 8421).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (restored, report) = mrnsd_with(
        &degraded_image,
        &psf,
        &Mrnsd::new()
            .iterations(24)
            .step_size(Some(1.0))
            .collect_history(true),
    )
    .unwrap();

    let baseline_array = gray_to_array(&degraded_image.to_luma8());
    let restored_array = gray_to_array(&restored.to_luma8());
    let baseline_psnr = psnr(&sharp, &baseline_array).unwrap();
    let restored_psnr = psnr(&sharp, &restored_array).unwrap();
    assert!(restored_psnr > baseline_psnr);
    assert!(restored_array.iter().all(|value| *value >= 0.0));
    assert!(is_finite_2d(&restored_array));
    assert!(!report.objective_history.is_empty());
    assert!(!report.residual_history.is_empty());
    let first_objective = report.objective_history[0];
    let last_objective = *report.objective_history.last().unwrap();
    let first_residual = report.residual_history[0];
    let last_residual = *report.residual_history.last().unwrap();
    assert!(last_objective <= first_objective * 1.01 + 1e-4);
    assert!(last_residual <= first_residual * 1.01 + 1e-4);
}

#[test]
fn cgls_residual_stabilizes_and_improves_over_blurred_baseline() {
    let sharp = checkerboard_2d((62, 58), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.5).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&blurred).unwrap());

    let (restored, report) = cgls_with(
        &degraded_image,
        &psf,
        &Cgls::new()
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
    assert!(!report.objective_history.is_empty());
    assert!(!report.residual_history.is_empty());
    let first_objective = report.objective_history[0];
    let last_objective = *report.objective_history.last().unwrap();
    let first_residual = report.residual_history[0];
    let last_residual = *report.residual_history.last().unwrap();
    assert!(last_objective <= first_objective * 1.01 + 1e-4);
    assert!(last_residual <= first_residual * 1.01 + 1e-4);
}

#[test]
fn mrnsd_and_cgls_are_deterministic() {
    let sharp = checkerboard_2d((46, 50), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((9, 9), 1.4).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 24.0, 55123).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let mrnsd_config = Mrnsd::new()
        .iterations(18)
        .step_size(Some(1.0))
        .collect_history(true);
    let (mrnsd_first, mrnsd_first_report) =
        mrnsd_with(&degraded_image, &psf, &mrnsd_config).unwrap();
    let (mrnsd_second, mrnsd_second_report) =
        mrnsd_with(&degraded_image, &psf, &mrnsd_config).unwrap();
    assert!(arrays_equal_2d(
        &gray_to_array(&mrnsd_first.to_luma8()),
        &gray_to_array(&mrnsd_second.to_luma8())
    ));
    assert_eq!(
        mrnsd_first_report.objective_history,
        mrnsd_second_report.objective_history
    );
    assert_eq!(
        mrnsd_first_report.residual_history,
        mrnsd_second_report.residual_history
    );

    let cgls_config = Cgls::new()
        .iterations(18)
        .step_size(Some(1.0))
        .collect_history(true);
    let (cgls_first, cgls_first_report) = cgls_with(&degraded_image, &psf, &cgls_config).unwrap();
    let (cgls_second, cgls_second_report) = cgls_with(&degraded_image, &psf, &cgls_config).unwrap();
    assert!(arrays_equal_2d(
        &gray_to_array(&cgls_first.to_luma8()),
        &gray_to_array(&cgls_second.to_luma8())
    ));
    assert_eq!(
        cgls_first_report.objective_history,
        cgls_second_report.objective_history
    );
    assert_eq!(
        cgls_first_report.residual_history,
        cgls_second_report.residual_history
    );
}

#[test]
fn mrnsd_and_cgls_default_paths_are_finite() {
    let sharp = checkerboard_2d((44, 44), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.2).unwrap();
    let blurred = blur(&sharp, &psf).unwrap();
    let degraded = add_poisson_noise(&blurred, 20.0, 8888).unwrap();
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded).unwrap());

    let (mrnsd_restored, mrnsd_report) = mrnsd(&degraded_image, &psf).unwrap();
    let (cgls_restored, cgls_report) = cgls(&degraded_image, &psf).unwrap();

    let mrnsd_array = gray_to_array(&mrnsd_restored.to_luma8());
    let cgls_array = gray_to_array(&cgls_restored.to_luma8());
    assert!(is_finite_2d(&mrnsd_array));
    assert!(is_finite_2d(&cgls_array));
    assert!(mrnsd_array.iter().all(|value| *value >= 0.0));
    assert!(mrnsd_report.iterations >= 1);
    assert!(cgls_report.iterations >= 1);
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

fn objective_stabilizes(history: &[f32]) -> bool {
    if history.len() < 2 {
        return false;
    }
    for pair in history.windows(2) {
        if pair[1] > pair[0] * 1.001 + 1e-3 {
            return false;
        }
    }
    history[history.len() - 1] <= history[0] * 1.001 + 1e-3
}
