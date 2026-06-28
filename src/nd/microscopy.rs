//! 3D `ndarray` deconvolution wrappers for microscopy volumes.
//!
//! Volume and PSF arrays use `(depth, height, width)` order and are restored
//! through the same configuration builders as the image APIs.

use ndarray::Array3;

use crate::algorithms;
use crate::iterative::{RichardsonLucy, RichardsonLucyTv};
use crate::optimization::{Cmle, Gmle, Qmle};
use crate::spectral::Wiener;
use crate::{Kernel3D, Result, SolveReport};

/// Restore a `(depth, height, width)` volume with Wiener filtering.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// Wiener settings, or non-finite restored values.
pub fn wiener(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<Array3<f32>> {
    wiener_with(volume, psf, &Wiener::new())
}

/// Restore a 3D volume with Wiener filtering and explicit settings.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// Wiener settings, or non-finite restored values.
pub fn wiener_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Wiener,
) -> Result<Array3<f32>> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::wiener_array3_with(volume, &kernel, config)
}

/// Restore a 3D volume with Richardson-Lucy Poisson EM.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// EM settings, or non-finite iterative updates.
pub fn richardson_lucy(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
) -> Result<(Array3<f32>, SolveReport)> {
    richardson_lucy_with(volume, psf, &RichardsonLucy::new())
}

/// Restore a 3D volume with Richardson-Lucy and explicit settings.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// EM settings, or non-finite iterative updates.
pub fn richardson_lucy_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &RichardsonLucy,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::richardson_lucy_array3_with(volume, &kernel, config)
}

/// Restore a 3D volume with Richardson-Lucy and total variation.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// TV/EM settings, or non-finite iterative updates.
pub fn richardson_lucy_tv(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
) -> Result<(Array3<f32>, SolveReport)> {
    richardson_lucy_tv_with(volume, psf, &RichardsonLucyTv::new())
}

/// Restore a 3D volume with RL-TV and explicit settings.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// TV/EM settings, or non-finite iterative updates.
pub fn richardson_lucy_tv_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &RichardsonLucyTv,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::richardson_lucy_tv_array3_with(volume, &kernel, config)
}

/// Restore a 3D volume with classical maximum-likelihood estimation.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// SNR, acuity, or EM settings, or non-finite iterative updates.
pub fn cmle(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<(Array3<f32>, SolveReport)> {
    cmle_with(volume, psf, &Cmle::new())
}

/// Restore a 3D volume with CMLE and explicit settings.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// SNR, acuity, or EM settings, or non-finite iterative updates.
pub fn cmle_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Cmle,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::cmle_array3_with(volume, &kernel, config)
}

/// Restore a 3D volume with Gaussian maximum-likelihood estimation.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// SNR, acuity, roughness, TV, or EM settings, or non-finite updates.
pub fn gmle(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<(Array3<f32>, SolveReport)> {
    gmle_with(volume, psf, &Gmle::new())
}

/// Restore a 3D volume with GMLE and explicit settings.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// SNR, acuity, roughness, TV, or EM settings, or non-finite updates.
pub fn gmle_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Gmle,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::gmle_array3_with(volume, &kernel, config)
}

/// Restore a 3D volume with quadratic maximum-likelihood estimation.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// SNR, acuity, or EM settings, or non-finite iterative updates.
pub fn qmle(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<(Array3<f32>, SolveReport)> {
    qmle_with(volume, psf, &Qmle::new())
}

/// Restore a 3D volume with QMLE and explicit settings.
///
/// # Errors
///
/// Returns an error for empty or non-finite volumes, invalid 3D PSFs, invalid
/// SNR, acuity, or EM settings, or non-finite iterative updates.
pub fn qmle_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Qmle,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::qmle_array3_with(volume, &kernel, config)
}

fn kernel3_from_array(psf: &Array3<f32>) -> Result<Kernel3D> {
    Kernel3D::new(psf.as_standard_layout().to_owned())
}
