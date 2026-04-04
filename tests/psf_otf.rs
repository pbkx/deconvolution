use deconvolution::{
    otf::{otf2psf, otf2psf_3d, psf2otf, psf2otf_3d},
    prelude::{Kernel2D as PreludeKernel2D, Transfer2D as PreludeTransfer2D},
    psf::{
        center, center_3d, crop_to, crop_to_3d, flip, flip_3d, normalize, normalize_3d, pad_to,
        pad_to_3d, support_mask, support_mask_3d, validate, validate_3d,
    },
    psf::{Blur2D, Blur3D},
    Error, Kernel2D, Kernel3D, Transfer2D, Transfer3D,
};
use ndarray::array;
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
