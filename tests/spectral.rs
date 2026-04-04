mod common;

use deconvolution::otf::psf2otf;
use deconvolution::psf::gaussian2d;
use deconvolution::simulate::{
    add_gaussian_noise, add_poisson_noise, add_readout_noise, blur, blur_otf, checkerboard_2d,
    degrade, gaussian_blob_2d, phantom_3d, rgb_edges_2d,
};

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
