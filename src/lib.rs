#![forbid(unsafe_code)]

//! Rust image deconvolution and restoration library.
//!
//! Recovering images from blur depends on a point-spread function, stable
//! frequency-domain utilities, and careful regularization. `deconvolution`
//! provides known-PSF restoration, blind workflows, PSF/OTF conversion,
//! preprocessing helpers, simulation fixtures, and ndarray APIs.
//!
//! ## Overview
//!
//! - **Image API**: Top-level functions use [`image::DynamicImage`] and return
//!   images ready to save.
//! - **Known PSF methods**: Inverse filters, Wiener, Richardson-Lucy,
//!   constrained, proximal, Krylov, and MLE-style restoration.
//! - **Blind methods**: Blind Richardson-Lucy, blind maximum likelihood, and
//!   parametric PSF estimation.
//! - **PSF and OTF types**: [`Kernel2D`], [`Kernel3D`], [`Transfer2D`],
//!   [`Transfer3D`], and [`psf::Blur2D`]/[`psf::Blur3D`].
//! - **PSF tools**: Gaussian, motion, defocus, microscopy models, support
//!   utilities, and PSF/OTF conversion.
//! - **Preprocessing**: Edge tapering, apodization, range normalization, and NSR
//!   estimation.
//! - **Simulation**: Deterministic blur, noise, and synthetic fixture generation.
//! - **ndarray support**: 2D image arrays and 3D volume workflows.
//! - **Feature flags**: `rayon` by default; optional `f16` support.
//!
//! ## Quick Start
//!
//! ```no_run
//! use deconvolution::psf::basic::gaussian2d;
//! use deconvolution::spectral::{wiener_with, Wiener};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let input = image::open("before_deconvolution.png")?;
//!     let psf = gaussian2d((15, 15), 2.15)?;
//!
//!     let restored = wiener_with(&input, &psf, &Wiener::new().nsr(2.5e-4))?;
//!     restored.save("after_deconvolution.png")?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Image API
//!
//! Supported [`image::DynamicImage`] variants:
//!
//! - `ImageLuma8`
//! - `ImageLumaA8`
//! - `ImageRgb8`
//! - `ImageRgba8`
//! - `ImageLuma16`
//! - `ImageLumaA16`
//! - `ImageRgb16`
//! - `ImageRgba16`
//! - `ImageRgb32F`
//! - `ImageRgba32F`
//!
//! Configuration enums are shared across algorithm families:
//!
//! - [`Boundary`]: `Zero`, `Replicate`, `Reflect`, `Symmetric`, `Periodic`
//! - [`Padding`]: `None`, `Same`, `Minimal`, `NextFastLen`, `Explicit2`,
//!   `Explicit3`
//! - [`ChannelMode`]: `Independent`, `LumaOnly`, `IgnoreAlpha`,
//!   `PremultipliedAlpha`
//! - [`RangePolicy`]: `PreserveInput`, `Clamp01`, `ClampNegPos1`, `Unbounded`
//!
//! Use [`ChannelMode::Independent`] for per-channel color restoration,
//! [`ChannelMode::LumaOnly`] when the blur should primarily affect luminance,
//! and [`RangePolicy::PreserveInput`] when working in normal image sample
//! ranges.
//!
//! ## PSF and OTF API
//!
//! Kernel and transfer types:
//!
//! - [`Kernel2D`], [`Kernel3D`]
//! - [`Transfer2D`], [`Transfer3D`]
//! - [`psf::Blur2D`], [`psf::Blur3D`]
//!
//! Basic PSF generators:
//!
//! - [`psf::basic::delta2d`], [`psf::basic::delta3d`]
//! - [`psf::basic::gaussian2d`], [`psf::basic::gaussian3d`]
//! - [`psf::basic::motion_linear`]
//! - [`psf::basic::disk`], [`psf::basic::pillbox`],
//!   [`psf::basic::defocus`]
//! - [`psf::basic::box2d`], [`psf::basic::box3d`]
//! - [`psf::basic::oriented_gaussian`]
//!
//! Blind initialization helpers:
//!
//! - [`psf::init::uniform`]
//! - [`psf::init::gaussian_guess`]
//! - [`psf::init::motion_guess`]
//! - [`psf::init::from_support`]
//!
//! Support utilities:
//!
//! - [`psf::support::normalize`], [`psf::support::normalize_3d`]
//! - [`psf::support::center`], [`psf::support::center_3d`]
//! - [`psf::support::pad_to`], [`psf::support::pad_to_3d`]
//! - [`psf::support::crop_to`], [`psf::support::crop_to_3d`]
//! - [`psf::support::flip`], [`psf::support::flip_3d`]
//! - [`psf::support::validate`], [`psf::support::validate_3d`]
//! - [`psf::support::support_mask`], [`psf::support::support_mask_3d`]
//!
//! Transfer conversion utilities:
//!
//! - [`otf::convert::psf2otf`]
//! - [`otf::convert::psf2otf_3d`]
//! - [`otf::convert::otf2psf`]
//! - [`otf::convert::otf2psf_3d`]
//!
//! Optical and microscopy models:
//!
//! - [`psf::microscopy::BornWolfParams`] / [`psf::microscopy::born_wolf`]
//! - [`psf::microscopy::GibsonLanniParams`] /
//!   [`psf::microscopy::gibson_lanni`]
//! - [`psf::microscopy::VariableRiGibsonLanniParams`] /
//!   [`psf::microscopy::variable_ri_gibson_lanni`]
//! - [`psf::microscopy::RichardsWolfParams`] /
//!   [`psf::microscopy::richards_wolf`]
//! - [`psf::microscopy::lorentz2d`]
//! - [`psf::microscopy::astigmatic`]
//! - [`psf::microscopy::double_helix`]
//! - [`otf::spectra::koehler_otf`]
//! - [`otf::spectra::defocus_otf`]
//!
//! ## Algorithm Families
//!
//! Spectral and inverse filters:
//! Frequency-domain restoration.
//!
//! - [`spectral::naive_inverse_filter`]
//! - [`spectral::inverse_filter`]
//! - [`spectral::truncated_inverse_filter`]
//! - [`spectral::regularized_inverse_filter`]
//! - [`spectral::tikhonov_inverse_filter`]
//! - [`spectral::wiener`]
//! - [`spectral::unsupervised_wiener`]
//!
//! Richardson-Lucy and iterative restoration:
//! Poisson-style and residual-update restoration.
//!
//! - [`iterative::richardson_lucy`]
//! - [`iterative::damped_richardson_lucy`]
//! - [`iterative::richardson_lucy_tv`]
//! - [`iterative::landweber`]
//! - [`iterative::van_cittert`]
//! - [`iterative::tikhonov_miller`]
//! - [`iterative::ictm`]
//!
//! Constrained, proximal, Krylov, and MLE-style solvers:
//! Bound-aware, sparse, scientific-imaging, and microscopy-oriented
//! restoration.
//!
//! - [`optimization::nnls`], [`optimization::bvls`]
//! - [`optimization::ista`], [`optimization::fista`]
//! - [`optimization::mrnsd`], [`optimization::cgls`]
//! - [`optimization::wpl`], [`optimization::hybr`]
//! - [`optimization::cmle`], [`optimization::gmle`],
//!   [`optimization::qmle`]
//!
//! Custom configs: Use `_with` variants and configuration types such as
//! [`Wiener`], [`RichardsonLucy`], [`spectral::InverseFilter`],
//! [`iterative::Landweber`], [`optimization::Fista`], or
//! [`optimization::Qmle`].
//!
//! ## Blind Deconvolution
//!
//! Blind workflows estimate both the restored image and the PSF.
//! Image-facing blind workflows support Gray and GrayAlpha
//! [`image::DynamicImage`] variants for u8 and u16 samples.
//!
//! - [`blind::richardson_lucy`]
//! - [`blind::maximum_likelihood`]
//! - [`blind::parametric`]
//!
//! [`blind::maximum_likelihood`] shares the same Poisson EM restoration core as
//! blind Richardson-Lucy.
//!
//! Configuration and output types:
//!
//! - [`blind::BlindRichardsonLucy`]
//! - [`blind::BlindMaximumLikelihood`]
//! - [`blind::BlindParametric`]
//! - [`blind::BlindOutput`]
//! - [`blind::BlindReport`]
//! - [`blind::ParametricPsf`]
//! - [`psf::PsfConstraint`]
//!
//! Parametric PSF families:
//!
//! - `Gaussian { sigma }`
//! - `MotionLinear { length, angle_deg }`
//! - `Defocus { radius }`
//! - `OrientedGaussian { sigma_major, sigma_minor, angle_deg }`
//!
//! ## ndarray Workflows
//!
//! The public [`nd`] module exposes array-first workflows for users who already
//! work in ndarray or need 3D volumes. Enable the optional `f16` feature to
//! pass `half::f16` arrays into the 2D ndarray API while keeping computation in
//! `f32`.
//!
//! 2D known-PSF methods in [`nd::known_psf`]:
//!
//! - [`nd::known_psf::wiener`], [`nd::known_psf::unsupervised_wiener`]
//! - [`nd::known_psf::richardson_lucy`],
//!   [`nd::known_psf::richardson_lucy_tv`]
//! - [`nd::known_psf::landweber`], [`nd::known_psf::van_cittert`],
//!   [`nd::known_psf::tikhonov_miller`], [`nd::known_psf::ictm`]
//! - [`nd::known_psf::nnls`], [`nd::known_psf::bvls`]
//! - [`nd::known_psf::ista`], [`nd::known_psf::fista`]
//! - [`nd::known_psf::mrnsd`], [`nd::known_psf::cgls`],
//!   [`nd::known_psf::wpl`], [`nd::known_psf::hybr`]
//!
//! Blind methods in [`nd::blind`]:
//!
//! - [`nd::blind::richardson_lucy`]
//! - [`nd::blind::maximum_likelihood`]
//!
//! 3D and microscopy methods in [`nd::microscopy`]:
//!
//! - [`nd::microscopy::wiener`]
//! - [`nd::microscopy::richardson_lucy`]
//! - [`nd::microscopy::richardson_lucy_tv`]
//! - [`nd::microscopy::cmle`]
//! - [`nd::microscopy::gmle`]
//! - [`nd::microscopy::qmle`]
//!
//! ## Preprocessing
//!
//! Preprocessing utilities help reduce ringing and prepare numerical inputs.
//!
//! - [`preprocess::apodize()`]
//! - [`preprocess::apodize::window_edges`]
//! - [`preprocess::edgetaper()`]
//! - [`preprocess::estimate_nsr`]
//! - [`preprocess::normalize_range`]
//!
//! Use [`preprocess::edgetaper()`] or apodization before frequency-domain
//! deconvolution when strong edge discontinuities create ringing artifacts.
//!
//! ## Simulation and Fixtures
//!
//! Deterministic: Same input and seed produce the same simulated output.
//!
//! Fixtures: Synthetic images and volumes for tests, examples, and benchmarks.
//!
//! - [`simulate::blur::blur`]
//! - [`simulate::blur::blur_otf`]
//! - [`simulate::blur::blur_3d`]
//! - [`simulate::blur::blur_otf_3d`]
//! - [`simulate::blur::degrade`]
//! - [`simulate::noise::add_gaussian_noise`]
//! - [`simulate::noise::add_poisson_noise`]
//! - [`simulate::noise::add_readout_noise`]
//! - [`simulate::phantom::checkerboard_2d`]
//! - [`simulate::phantom::gaussian_blob_2d`]
//! - [`simulate::phantom::rgb_edges_2d`]
//! - [`simulate::phantom::phantom_3d`]
//!
//! ## Optional rayon Integration
//!
//! `rayon` is enabled by default. The optional `f16` feature adds `half::f16`
//! input/output support for the 2D ndarray API; computation remains in `f32`.
//!
//! ```toml
//! [features]
//! default = ["rayon"]
//! rayon = ["dep:rayon", "ndarray/rayon", "image/rayon"]
//! f16 = ["dep:half"]
//! ```
//!
//! Disable default features for serial builds:
//!
//! ```bash
//! cargo test --no-default-features
//! ```
//!
//! ## Reports, Errors, and Prelude
//!
//! Error handling:
//!
//! - [`Error`]
//! - [`Result`]
//!
//! Solver reports:
//!
//! - [`SolveReport`]
//! - [`StopReason`]
//!
//! Prelude:
//!
//! - [`prelude`]
//!
//! Runnable examples: See `examples/` and the repository README.

mod algorithms;
mod core;

pub mod blind;
pub mod error;
pub mod iterative;
pub mod nd;
pub mod optimization;
pub mod otf;
pub mod prelude;
pub mod preprocess;
pub mod psf;
pub mod simulate;
pub mod spectral;
pub mod traits;

pub use crate::core::diagnostics::SolveReport;
pub use crate::core::stopping::StopReason;
pub use crate::error::{Error, Result};
pub use crate::iterative::{RichardsonLucy, richardson_lucy};
pub use crate::otf::{Transfer2D, Transfer3D};
pub use crate::psf::{Kernel2D, Kernel3D};
pub use crate::spectral::{Wiener, wiener};
pub use crate::traits::{Boundary, ChannelMode, Padding, RangePolicy};
