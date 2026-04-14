use deconvolution::{
    otf::{otf2psf, otf2psf_3d, psf2otf, psf2otf_3d},
    prelude::{Kernel2D as PreludeKernel2D, Transfer2D as PreludeTransfer2D},
    psf::{
        born_wolf, box2d, box3d, center, center_3d, crop_to, crop_to_3d, defocus, delta2d, delta3d,
        disk, flip, flip_3d, from_support, gaussian2d, gaussian3d, gaussian_guess, gibson_lanni,
        motion_guess, motion_linear, normalize, normalize_3d, oriented_gaussian, pad_to, pad_to_3d,
        pillbox, richards_wolf, support_mask, support_mask_3d, uniform, validate, validate_3d,
        variable_ri_gibson_lanni, BornWolfParams, GibsonLanniParams, RichardsWolfParams,
        VariableRiGibsonLanniParams,
    },
    psf::{Blur2D, Blur3D},
    simulate::{blur, blur_otf, checkerboard_2d},
    Error, Kernel2D, Kernel3D, Transfer2D, Transfer3D,
};
use ndarray::{array, Array2};
use num_complex::Complex32;

#[test]
fn valid_construction_succeeds() {
    let kernel2 = Kernel2D::new(array![[0.25_f32, 0.25_f32], [0.25_f32, 0.25_f32]]).unwrap();
    let kernel3 = Kernel3D::new(array![[[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]]]).unwrap();
    let transfer2 = Transfer2D::new(array![
        [Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)],
        [Complex32::new(2.0, -1.0), Complex32::new(0.5, 0.5)]
    ])
    .unwrap();
    let transfer3 = Transfer3D::new(array![[[
        Complex32::new(1.0, 0.0),
        Complex32::new(0.5, -0.5)
    ]]])
    .unwrap();

    assert_eq!(kernel2.dims(), (2, 2));
    assert_eq!(kernel3.dims(), (1, 2, 2));
    assert_eq!(transfer2.dims(), (2, 2));
    assert_eq!(transfer3.dims(), (1, 1, 2));

    let _prelude_kernel: PreludeKernel2D = kernel2;
    let _prelude_transfer: PreludeTransfer2D = transfer2;
}

#[test]
fn non_finite_input_is_rejected() {
    let kernel2_err = Kernel2D::new(array![[1.0_f32, f32::NAN]]).unwrap_err();
    let kernel3_err = Kernel3D::new(array![[[1.0_f32, f32::INFINITY]]]).unwrap_err();
    let transfer2_err = Transfer2D::new(array![[
        Complex32::new(f32::NAN, 0.0),
        Complex32::new(1.0, 0.0)
    ]])
    .unwrap_err();
    let transfer3_err =
        Transfer3D::new(array![[[Complex32::new(0.0, f32::INFINITY)]]]).unwrap_err();

    assert_eq!(kernel2_err, Error::NonFiniteInput);
    assert_eq!(kernel3_err, Error::NonFiniteInput);
    assert_eq!(transfer2_err, Error::NonFiniteInput);
    assert_eq!(transfer3_err, Error::NonFiniteInput);
}

#[test]
fn sums_dims_and_normalization_are_reported() {
    let kernel2 = Kernel2D::new(array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]]).unwrap();
    let kernel3 = Kernel3D::new(array![[[1.0_f32, 1.0_f32], [2.0_f32, 2.0_f32]]]).unwrap();

    assert_eq!(kernel2.dims(), (2, 2));
    assert_eq!(kernel3.dims(), (1, 2, 2));
    assert!((kernel2.sum() - 10.0).abs() < 1e-6);
    assert!((kernel3.sum() - 6.0).abs() < 1e-6);
    assert!(kernel2.is_finite());
    assert!(kernel3.is_finite());

    let normalized_kernel2 = kernel2.normalized().unwrap();
    let normalized_kernel3 = kernel3.normalized().unwrap();
    assert!((normalized_kernel2.sum() - 1.0).abs() < 1e-6);
    assert!((normalized_kernel3.sum() - 1.0).abs() < 1e-6);

    let transfer2 = Transfer2D::new(array![
        [Complex32::new(2.0, 0.0), Complex32::new(1.0, -1.0)],
        [Complex32::new(0.0, 1.0), Complex32::new(1.0, 0.0)]
    ])
    .unwrap();
    let transfer3 = Transfer3D::new(array![[[
        Complex32::new(2.0, 0.0),
        Complex32::new(1.0, 1.0),
        Complex32::new(1.0, -1.0)
    ]]])
    .unwrap();

    assert_eq!(transfer2.dims(), (2, 2));
    assert_eq!(transfer3.dims(), (1, 1, 3));
    assert!(transfer2.is_finite());
    assert!(transfer3.is_finite());
    assert!((transfer2.sum() - Complex32::new(4.0, 0.0)).norm() < 1e-6);
    assert!((transfer3.sum() - Complex32::new(4.0, 0.0)).norm() < 1e-6);

    let normalized_transfer2 = transfer2.normalized().unwrap();
    let normalized_transfer3 = transfer3.normalized().unwrap();
    assert!((normalized_transfer2.sum() - Complex32::new(1.0, 0.0)).norm() < 1e-6);
    assert!((normalized_transfer3.sum() - Complex32::new(1.0, 0.0)).norm() < 1e-6);
}

#[test]
fn blur_wrappers_are_lightweight_references() {
    let kernel2 = Kernel2D::new(array![[1.0_f32]]).unwrap();
    let transfer2 = Transfer2D::new(array![[Complex32::new(1.0, 0.0)]]).unwrap();
    let kernel3 = Kernel3D::new(array![[[1.0_f32]]]).unwrap();
    let transfer3 = Transfer3D::new(array![[[Complex32::new(1.0, 0.0)]]]).unwrap();

    let blur2_psf = Blur2D::Psf(&kernel2);
    let blur2_otf = Blur2D::Otf(&transfer2);
    let blur3_psf = Blur3D::Psf(&kernel3);
    let blur3_otf = Blur3D::Otf(&transfer3);

    assert_eq!(blur2_psf.dims(), kernel2.dims());
    assert_eq!(blur2_otf.dims(), transfer2.dims());
    assert_eq!(blur3_psf.dims(), kernel3.dims());
    assert_eq!(blur3_otf.dims(), transfer3.dims());

    match blur2_psf {
        Blur2D::Psf(psf) => assert!(std::ptr::eq(psf, &kernel2)),
        Blur2D::Otf(_) => panic!("expected psf variant"),
    }
    match blur2_otf {
        Blur2D::Otf(otf) => assert!(std::ptr::eq(otf, &transfer2)),
        Blur2D::Psf(_) => panic!("expected otf variant"),
    }
    match blur3_psf {
        Blur3D::Psf(psf) => assert!(std::ptr::eq(psf, &kernel3)),
        Blur3D::Otf(_) => panic!("expected psf variant"),
    }
    match blur3_otf {
        Blur3D::Otf(otf) => assert!(std::ptr::eq(otf, &transfer3)),
        Blur3D::Psf(_) => panic!("expected otf variant"),
    }
}

#[test]
fn psf2otf_delta_is_all_ones() {
    let delta2 = Kernel2D::new(array![
        [0.0_f32, 0.0_f32, 0.0_f32],
        [0.0_f32, 1.0_f32, 0.0_f32],
        [0.0_f32, 0.0_f32, 0.0_f32]
    ])
    .unwrap();
    let otf2 = psf2otf(&delta2, delta2.dims()).unwrap();
    for value in otf2.as_array() {
        assert!((*value - Complex32::new(1.0, 0.0)).norm() < 1e-5);
    }

    let delta3 = Kernel3D::new(array![
        [
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32]
        ],
        [
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 1.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32]
        ],
        [
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32]
        ]
    ])
    .unwrap();
    let otf3 = psf2otf_3d(&delta3, delta3.dims()).unwrap();
    for value in otf3.as_array() {
        assert!((*value - Complex32::new(1.0, 0.0)).norm() < 1e-5);
    }
}

#[test]
fn otf_psf_roundtrip_within_tolerance() {
    let psf2 = Kernel2D::new(array![
        [0.0_f32, 0.1_f32, 0.0_f32],
        [0.1_f32, 0.6_f32, 0.1_f32],
        [0.0_f32, 0.1_f32, 0.0_f32]
    ])
    .unwrap();
    let psf2_norm = normalize(&psf2).unwrap();
    let otf2 = psf2otf(&psf2_norm, (7, 6)).unwrap();
    let restored2 = otf2psf(&otf2, psf2_norm.dims()).unwrap();
    for ((y, x), value) in psf2_norm.as_array().indexed_iter() {
        assert!((restored2.as_array()[[y, x]] - value).abs() < 1e-4);
    }

    let psf3 = Kernel3D::new(array![
        [
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.1_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32]
        ],
        [
            [0.0_f32, 0.1_f32, 0.0_f32],
            [0.1_f32, 0.4_f32, 0.1_f32],
            [0.0_f32, 0.1_f32, 0.0_f32]
        ],
        [
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.1_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32]
        ]
    ])
    .unwrap();
    let psf3_norm = normalize_3d(&psf3).unwrap();
    let otf3 = psf2otf_3d(&psf3_norm, (5, 6, 7)).unwrap();
    let restored3 = otf2psf_3d(&otf3, psf3_norm.dims()).unwrap();
    for ((d, y, x), value) in psf3_norm.as_array().indexed_iter() {
        assert!((restored3.as_array()[[d, y, x]] - value).abs() < 1e-4);
    }
}

#[test]
fn center_pad_crop_flip_and_validate_follow_conventions() {
    let psf = Kernel2D::new(array![
        [0.0_f32, 0.0_f32, 0.0_f32],
        [1.0_f32, 0.0_f32, 0.0_f32],
        [0.0_f32, 0.0_f32, 0.0_f32]
    ])
    .unwrap();
    let centered = center(&psf).unwrap();
    assert_eq!(centered.as_array()[[1, 1]], 1.0);

    let padded = pad_to(&centered, (5, 7)).unwrap();
    assert_eq!(padded.dims(), (5, 7));
    let cropped = crop_to(&padded, centered.dims()).unwrap();
    assert_eq!(cropped, centered);

    let flipped = flip(&centered).unwrap();
    assert_eq!(flipped.as_array()[[1, 1]], 1.0);
    validate(&centered).unwrap();

    let psf3 = Kernel3D::new(array![
        [
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32]
        ],
        [
            [0.0_f32, 0.0_f32, 0.0_f32],
            [1.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32]
        ],
        [
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32]
        ]
    ])
    .unwrap();
    let centered3 = center_3d(&psf3).unwrap();
    assert_eq!(centered3.as_array()[[1, 1, 1]], 1.0);

    let padded3 = pad_to_3d(&centered3, (5, 5, 5)).unwrap();
    let cropped3 = crop_to_3d(&padded3, centered3.dims()).unwrap();
    assert_eq!(cropped3, centered3);

    let flipped3 = flip_3d(&centered3).unwrap();
    assert_eq!(flipped3.as_array()[[1, 1, 1]], 1.0);
    validate_3d(&centered3).unwrap();
}

#[test]
fn support_mask_thresholding_is_stable() {
    let psf = Kernel2D::new(array![
        [1.0_f32, 0.5_f32, 0.1_f32],
        [0.0_f32, 0.0_f32, 0.0_f32],
        [0.0_f32, 0.0_f32, 0.0_f32]
    ])
    .unwrap();
    let mask = support_mask(&psf, 0.5).unwrap();
    assert!(mask[[0, 0]]);
    assert!(mask[[0, 1]]);
    assert!(!mask[[0, 2]]);

    let psf3 = Kernel3D::new(array![
        [[1.0_f32, 0.25_f32], [0.0_f32, 0.0_f32]],
        [[0.0_f32, 0.0_f32], [0.0_f32, 0.0_f32]]
    ])
    .unwrap();
    let mask3 = support_mask_3d(&psf3, 0.3).unwrap();
    assert!(mask3[[0, 0, 0]]);
    assert!(!mask3[[0, 0, 1]]);
}

#[test]
fn basic_generators_are_normalized_and_shape_conformant() {
    let d2 = delta2d((5, 7)).unwrap();
    assert_eq!(d2.dims(), (5, 7));
    assert!((d2.sum() - 1.0).abs() < 1e-6);
    assert_eq!(d2.as_array()[[2, 3]], 1.0);

    let d3 = delta3d((3, 5, 7)).unwrap();
    assert_eq!(d3.dims(), (3, 5, 7));
    assert!((d3.sum() - 1.0).abs() < 1e-6);
    assert_eq!(d3.as_array()[[1, 2, 3]], 1.0);

    let g2 = gaussian2d((7, 7), 1.2).unwrap();
    assert!((g2.sum() - 1.0).abs() < 1e-6);
    assert!((g2.as_array()[[3, 2]] - g2.as_array()[[3, 4]]).abs() < 1e-6);
    assert!((g2.as_array()[[2, 3]] - g2.as_array()[[4, 3]]).abs() < 1e-6);

    let g3 = gaussian3d((5, 5, 5), 1.0).unwrap();
    assert!((g3.sum() - 1.0).abs() < 1e-6);
    assert!((g3.as_array()[[1, 2, 2]] - g3.as_array()[[3, 2, 2]]).abs() < 1e-6);

    let b2 = box2d((3, 5)).unwrap();
    let b3 = box3d((3, 3, 3)).unwrap();
    assert!((b2.sum() - 1.0).abs() < 1e-6);
    assert!((b3.sum() - 1.0).abs() < 1e-6);
}

#[test]
fn disk_family_support_and_motion_size_behave_sensibly() {
    let radius = 3.0_f32;
    let k_disk = disk(radius).unwrap();
    let k_pill = pillbox(radius).unwrap();
    let k_defocus = defocus(radius).unwrap();

    assert_eq!(k_disk.dims(), (7, 7));
    assert_eq!(k_pill.dims(), (7, 7));
    assert_eq!(k_defocus.dims(), (7, 7));
    assert!((k_disk.sum() - 1.0).abs() < 1e-6);
    assert!((k_pill.sum() - 1.0).abs() < 1e-6);
    assert!((k_defocus.sum() - 1.0).abs() < 1e-6);

    let center = 3_i32;
    for y in 0..7_i32 {
        for x in 0..7_i32 {
            let dy = (y - center) as f32;
            let dx = (x - center) as f32;
            if dx * dx + dy * dy > radius * radius {
                assert_eq!(k_disk.as_array()[[y as usize, x as usize]], 0.0);
                assert_eq!(k_pill.as_array()[[y as usize, x as usize]], 0.0);
            }
        }
    }

    let m = motion_linear(8.0, 0.0).unwrap();
    let (mh, mw) = m.dims();
    assert_eq!(mh, mw);
    assert_eq!(mh % 2, 1);
    assert!(mh >= 9);
    assert!((m.sum() - 1.0).abs() < 1e-6);
}

#[test]
fn oriented_gaussian_and_init_helpers_produce_valid_psf() {
    let og = oriented_gaussian((9, 9), 2.0, 0.8, 30.0).unwrap();
    assert!((og.sum() - 1.0).abs() < 1e-6);
    validate(&og).unwrap();

    let u = uniform((5, 7)).unwrap();
    let gg = gaussian_guess((5, 7), 1.0).unwrap();
    let mg = motion_guess((5, 7), 9.0, 15.0).unwrap();
    validate(&u).unwrap();
    validate(&gg).unwrap();
    validate(&mg).unwrap();
    assert_eq!(u.dims(), (5, 7));
    assert_eq!(gg.dims(), (5, 7));
    assert_eq!(mg.dims(), (5, 7));

    let mut support = Array2::from_elem((5, 7), false);
    support[[2, 2]] = true;
    support[[2, 3]] = true;
    support[[2, 4]] = true;
    let masked = from_support(&support).unwrap();
    validate(&masked).unwrap();
    assert_eq!(masked.dims(), (5, 7));
    assert_eq!(masked.as_array()[[2, 2]], 1.0 / 3.0);
    assert_eq!(masked.as_array()[[2, 3]], 1.0 / 3.0);
    assert_eq!(masked.as_array()[[2, 4]], 1.0 / 3.0);
    assert_eq!(masked.as_array()[[0, 0]], 0.0);
}

#[test]
fn blur_matches_otf_blur_shape_and_finiteness() {
    let input = checkerboard_2d((28, 36), 4, 0.0, 1.0).unwrap();
    let psf = gaussian2d((7, 7), 1.1).unwrap();
    let otf = psf2otf(&psf, input.dim()).unwrap();

    let by_psf = blur(&input, &psf).unwrap();
    let by_otf = blur_otf(&input, &otf).unwrap();

    assert_eq!(by_psf.dim(), input.dim());
    assert_eq!(by_otf.dim(), input.dim());
    assert!(by_psf.iter().all(|value| value.is_finite()));
    assert!(by_otf.iter().all(|value| value.is_finite()));
}

#[test]
fn microscopy_models_produce_finite_normalized_kernels() {
    let born_wolf_kernel = born_wolf(&BornWolfParams::new().dims((17, 17, 17))).unwrap();
    let gibson_lanni_kernel = gibson_lanni(&GibsonLanniParams::new().dims((17, 17, 17))).unwrap();
    let variable_ri_kernel =
        variable_ri_gibson_lanni(&VariableRiGibsonLanniParams::new().dims((17, 17, 17))).unwrap();
    let richards_wolf_kernel =
        richards_wolf(&RichardsWolfParams::new().dims((17, 17, 17))).unwrap();

    let kernels = [
        born_wolf_kernel,
        gibson_lanni_kernel,
        variable_ri_kernel,
        richards_wolf_kernel,
    ];
    for kernel in &kernels {
        validate_3d(kernel).unwrap();
        assert_eq!(kernel.dims(), (17, 17, 17));
        assert!((kernel.sum() - 1.0).abs() < 1e-5);
        assert!(kernel.as_array().iter().all(|value| value.is_finite()));
        assert!(kernel.as_array().iter().all(|value| *value >= 0.0));
    }
}

#[test]
fn microscopy_centered_symmetric_cases_behave_sensibly() {
    let params = BornWolfParams::new()
        .dims((19, 19, 19))
        .wavelength_um(0.53)
        .numerical_aperture(1.2)
        .refractive_index(1.33)
        .axial_step_um(0.18);
    let kernel = born_wolf(&params).unwrap();
    let arr = kernel.as_array();
    let cz = 9_usize;
    let cy = 9_usize;
    let cx = 9_usize;

    assert!((arr[[cz, cy, cx - 2]] - arr[[cz, cy, cx + 2]]).abs() < 1e-6);
    assert!((arr[[cz, cy - 2, cx]] - arr[[cz, cy + 2, cx]]).abs() < 1e-6);
    assert!((arr[[cz - 2, cy, cx]] - arr[[cz + 2, cy, cx]]).abs() < 1e-6);
    assert!(arr[[cz, cy, cx]] >= arr[[cz, cy, cx + 1]]);
    assert!(arr[[cz, cy, cx]] >= arr[[cz + 1, cy, cx]]);
}

#[test]
fn variable_refractive_index_changes_output_meaningfully() {
    let uniform_index = VariableRiGibsonLanniParams::new()
        .dims((17, 17, 17))
        .refractive_index_start(1.36)
        .refractive_index_end(1.36);
    let varying_index = VariableRiGibsonLanniParams::new()
        .dims((17, 17, 17))
        .refractive_index_start(1.33)
        .refractive_index_end(1.46)
        .profile_exponent(1.4);

    let uniform_kernel = variable_ri_gibson_lanni(&uniform_index).unwrap();
    let varying_kernel = variable_ri_gibson_lanni(&varying_index).unwrap();

    let l1_distance = uniform_kernel
        .as_array()
        .iter()
        .zip(varying_kernel.as_array().iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>();
    assert!(l1_distance > 1e-3);
}

#[test]
fn microscopy_parameter_validation_rejects_invalid_sets() {
    let invalid_born_wolf = BornWolfParams::new()
        .dims((9, 9, 9))
        .refractive_index(1.33)
        .numerical_aperture(1.33);
    let invalid_richards_wolf = RichardsWolfParams::new()
        .dims((9, 9, 9))
        .polarization_weight(1.2);

    let err_bw = born_wolf(&invalid_born_wolf).unwrap_err();
    let err_rw = richards_wolf(&invalid_richards_wolf).unwrap_err();
    assert_eq!(err_bw, Error::InvalidParameter);
    assert_eq!(err_rw, Error::InvalidParameter);
}
