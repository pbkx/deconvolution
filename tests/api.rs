use deconvolution::{
    prelude::{
        Boundary as PreludeBoundary, ChannelMode as PreludeChannelMode, Error as PreludeError,
        Padding as PreludePadding, RangePolicy as PreludeRangePolicy,
        RegOperator2D as PreludeRegOperator2D, Result as PreludeResult,
        SolveReport as PreludeSolveReport, StopReason as PreludeStopReason,
    },
    Boundary, ChannelMode, Error, Padding, RangePolicy, RegOperator2D, SolveReport, StopReason,
};
use ndarray::array;

#[test]
fn error_variants_format_non_empty_text() {
    let errors = [
        Error::DimensionMismatch,
        Error::InvalidPsf,
        Error::InvalidTransfer,
        Error::InvalidParameter,
        Error::UnsupportedPixelType,
        Error::NonFiniteInput,
        Error::ConvergenceFailure,
        Error::EmptyImage,
    ];

    for error in errors {
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn enums_derive_clone_and_compare() {
    assert_eq!(Boundary::Reflect.clone(), Boundary::Reflect);
    assert_eq!(Boundary::Periodic, Boundary::Periodic);
    assert_ne!(Boundary::Zero, Boundary::Replicate);

    assert_eq!(Padding::Explicit2(2, 3).clone(), Padding::Explicit2(2, 3));
    assert_eq!(Padding::Explicit3(1, 2, 3), Padding::Explicit3(1, 2, 3));
    assert_ne!(Padding::Same, Padding::Minimal);

    assert_eq!(ChannelMode::LumaOnly.clone(), ChannelMode::LumaOnly);
    assert_eq!(ChannelMode::IgnoreAlpha, ChannelMode::IgnoreAlpha);
    assert_ne!(ChannelMode::Independent, ChannelMode::PremultipliedAlpha);

    assert_eq!(RangePolicy::Clamp01.clone(), RangePolicy::Clamp01);
    assert_eq!(RangePolicy::Unbounded, RangePolicy::Unbounded);
    assert_ne!(RangePolicy::PreserveInput, RangePolicy::ClampNegPos1);
}

#[test]
fn prelude_reexports_compile() {
    let boundary = PreludeBoundary::Symmetric;
    let padding = PreludePadding::NextFastLen;
    let channel_mode = PreludeChannelMode::Independent;
    let range_policy = PreludeRangePolicy::Clamp01;
    let result: PreludeResult<()> = Err(PreludeError::EmptyImage);
    let stop_reason = PreludeStopReason::MaxIterations;
    let reg_op = PreludeRegOperator2D::Identity;
    let report = PreludeSolveReport {
        iterations: 0,
        stop_reason,
        objective_history: Vec::new(),
        residual_history: Vec::new(),
    };

    assert_eq!(boundary, PreludeBoundary::Symmetric);
    assert_eq!(padding, PreludePadding::NextFastLen);
    assert_eq!(channel_mode, PreludeChannelMode::Independent);
    assert_eq!(range_policy, PreludeRangePolicy::Clamp01);
    assert!(matches!(result, Err(PreludeError::EmptyImage)));
    assert!(matches!(reg_op, PreludeRegOperator2D::Identity));
    assert_eq!(report.iterations, 0);
}

#[test]
fn pr05_public_items_compile_and_behave() {
    let stop_reason = StopReason::RelativeUpdate;
    let report = SolveReport {
        iterations: 3,
        stop_reason,
        objective_history: vec![5.0_f32, 4.0_f32, 3.5_f32],
        residual_history: vec![2.0_f32, 1.0_f32, 0.8_f32],
    };

    assert_eq!(report.stop_reason, StopReason::RelativeUpdate);
    assert_eq!(report.objective_history.len(), 3);

    let input = array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]];
    let identity = RegOperator2D::Identity.apply(&input).unwrap();
    assert_eq!(identity, input);
}
