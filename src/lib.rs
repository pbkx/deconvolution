#![forbid(unsafe_code)]

//! Deconvolution toolkit with an image-first API and ndarray expert workflows.
//!
//! Module organization:
//! - [`spectral`]: frequency-domain inverse and Wiener-family methods.
//! - [`iterative`]: Richardson-Lucy and iterative restoration methods.
//! - [`optimization`]: constrained, proximal, Krylov, and MLE-related solvers.
//! - [`blind`]: blind deconvolution workflows.
//! - [`psf`]: kernels and PSF generation/utilities.
//! - [`otf`]: transfer-function conversions and spectra.
//! - [`preprocess`]: edge tapering, NSR estimation, and normalization.
//! - [`simulate`]: blur, noise, and phantom generation.
//! - [`nd`]: ndarray and N-dimensional workflows.

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
