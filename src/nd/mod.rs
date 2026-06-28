//! `ndarray` convenience APIs for 2D images and 3D microscopy volumes.
//!
//! 2D arrays use `(height, width)` order and 3D arrays use
//! `(depth, height, width)` order. Use [`known_psf`] for fixed-kernel solvers
//! and [`microscopy`] for volume deconvolution.

pub mod blind;
mod convert;
pub mod known_psf;
pub mod microscopy;

pub use convert::NdSample;
