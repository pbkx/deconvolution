use deconvolution::blind::{BlindOutput, BlindReport, ParametricPsf};
use deconvolution::psf::{apply_constraint, apply_constraints, PsfConstraint};
use deconvolution::{Error, Kernel2D, StopReason};
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
