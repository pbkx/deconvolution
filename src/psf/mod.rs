//! Point-spread function types, generators, constraints, and support helpers.
//!
//! Use [`Kernel2D`] or [`Kernel3D`] to wrap validated PSF arrays, [`basic`] for
//! common analytic kernels, and [`constraints`] for blind PSF projections.

pub mod basic;
pub mod constraints;
pub mod init;
mod kernel;
pub mod microscopy;
pub mod support;

pub use constraints::PsfConstraint;
pub use kernel::{Blur2D, Blur3D, Kernel2D, Kernel3D};
