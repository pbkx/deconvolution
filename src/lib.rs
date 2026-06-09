#![forbid(unsafe_code)]

//! Rust image deconvolution and restoration library.
//!
//! Recovering images from blur depends on a point-spread function, stable
//! frequency-domain utilities, and careful regularization. `deconvolution`
//! provides known-PSF restoration, blind workflows, PSF/OTF conversion,
//! preprocessing helpers, simulation fixtures, and ndarray-facing APIs for
//! array and volume workflows.
//!
//! ## Overview
//!
//! - **Image API**: Top-level functions operate on [`image::DynamicImage`] and
//!   return image buffers suitable for saving or further processing.
//! - **Known-PSF restoration**: Inverse filters, Wiener-family methods,
//!   Richardson-Lucy variants, least-squares solvers, constrained solvers,
//!   proximal methods, Krylov methods, and MLE-style solvers.
//! - **PSF/OTF utilities**: Owned [`Kernel2D`]/[`Kernel3D`] and
//!   [`Transfer2D`]/[`Transfer3D`] types, PSF generators, support utilities,
//!   and PSF/OTF conversions.
//! - **Blind deconvolution**: Blind Richardson-Lucy, blind maximum likelihood,
//!   and parametric blind workflows with PSF constraints.
//! - **Preprocessing and simulation**: Edge tapering, apodization, NSR
//!   estimation, deterministic blur/noise helpers, and synthetic fixtures.
//! - **ndarray workflows**: Public [`nd`] modules expose 2D image arrays and
//!   3D volume workflows for users who want to bypass `DynamicImage`.
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
//! ## Image API and Configuration
//!
//! Known-PSF image-facing algorithms accept [`image::DynamicImage`] values and
//! support these variants:
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
//! Shared configuration enums:
//!
//! - [`Boundary`]: `Zero`, `Replicate`, `Reflect`, `Symmetric`, `Periodic`
//! - [`Padding`]: `None`, `Same`, `Minimal`, `NextFastLen`, `Explicit2`,
//!   `Explicit3`
//! - [`ChannelMode`]: `Independent`, `LumaOnly`, `IgnoreAlpha`,
//!   `PremultipliedAlpha`
//! - [`RangePolicy`]: `PreserveInput`, `Clamp01`, `ClampNegPos1`, `Unbounded`
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
//!
//! - [`optimization::nnls`], [`optimization::bvls`]
//! - [`optimization::ista`], [`optimization::fista`]
//! - [`optimization::mrnsd`], [`optimization::cgls`]
//! - [`optimization::wpl`], [`optimization::hybr`]
//! - [`optimization::cmle`], [`optimization::gmle`],
//!   [`optimization::qmle`]
//!
//! Each configurable algorithm family exposes a `_with` variant and a
//! configuration type, such as [`Wiener`], [`RichardsonLucy`],
//! [`spectral::InverseFilter`], [`iterative::Landweber`],
//! [`optimization::Fista`], or [`optimization::Qmle`].
//!
//! ## Blind Deconvolution
//!
//! Blind workflows estimate both the restored image and the point-spread
//! function.
//!
//! - [`blind::richardson_lucy`]
//! - [`blind::maximum_likelihood`]
//! - [`blind::parametric`]
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
//! ## ndarray API
//!
//! The public [`nd`] module exposes array-first workflows for users who already
//! work in ndarray or need 3D volumes.
//! Enable the optional `f16` feature to pass `half::f16` arrays into the 2D
//! known-PSF ndarray API while keeping computation in `f32`.
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
//! ## Preprocessing and Simulation
//!
//! Preprocessing utilities:
//!
//! - [`preprocess::apodize()`]
//! - [`preprocess::apodize::window_edges`]
//! - [`preprocess::edgetaper()`]
//! - [`preprocess::estimate_nsr`]
//! - [`preprocess::normalize_range`]
//!
//! Simulation utilities:
//!
//! - [`simulate::blur::blur`]
//! - [`simulate::blur::blur_otf`]
//! - [`simulate::blur::degrade`]
//! - [`simulate::noise::add_gaussian_noise`]
//! - [`simulate::noise::add_poisson_noise`]
//! - [`simulate::noise::add_readout_noise`]
//! - [`simulate::phantom::checkerboard_2d`]
//! - [`simulate::phantom::gaussian_blob_2d`]
//! - [`simulate::phantom::rgb_edges_2d`]
//! - [`simulate::phantom::phantom_3d`]
//!
//! ## Reports, Errors, and Prelude
//!
//! Crate-level result and error types:
//!
//! - [`Error`]
//! - [`Result`]
//!
//! Solver reporting:
//!
//! - [`SolveReport`]
//! - [`StopReason`]
//!
//! Convenience imports:
//!
//! - [`prelude`]
//!
//! For complete workflows and runnable examples, see `examples/` and the
//! repository README.

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
pub use crate::iterative::{richardson_lucy, RichardsonLucy};
pub use crate::otf::{Transfer2D, Transfer3D};
pub use crate::psf::{Kernel2D, Kernel3D};
pub use crate::spectral::{wiener, Wiener};
pub use crate::traits::{Boundary, ChannelMode, Padding, RangePolicy};
