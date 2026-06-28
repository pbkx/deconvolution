//! Optical transfer functions and PSF/OTF conversion utilities.
//!
//! Use [`Transfer2D`] or [`Transfer3D`] with spectral solvers, or build them
//! from point-spread functions through [`convert::psf2otf`].

pub mod convert;
pub mod spectra;
mod transfer;

pub use transfer::{Transfer2D, Transfer3D};
