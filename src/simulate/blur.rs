//! Synthetic blur and degradation helpers.
//!
//! Use [`blur`] and [`blur_3d`] to apply spatial PSFs, or [`blur_otf`] and
//! [`blur_otf_3d`] when the transfer function is already available.

use ndarray::{Array2, Array3};
use num_complex::Complex32;

use crate::core::fft::{
    fft2_forward_real, fft2_inverse_complex, fft3_forward_real, fft3_inverse_complex,
};
use crate::core::plan_cache::PlanCache;
use crate::otf::convert::{psf2otf, psf2otf_3d};
use crate::otf::{Transfer2D, Transfer3D};
use crate::psf::support::{validate, validate_3d};
use crate::psf::{Kernel2D, Kernel3D};
use crate::simulate::noise::{add_gaussian_noise, add_poisson_noise, add_readout_noise};
use crate::{Error, Result};

/// Blur a 2D image with a spatial PSF using FFT convolution.
///
/// `input` uses `(height, width)` order. The PSF is converted to an OTF with
/// the same dimensions as `input`.
///
/// # Errors
///
/// Returns an error when `input` is empty or non-finite, `psf` is invalid, PSF
/// to OTF conversion fails, or the frequency-domain product is non-finite.
pub fn blur(input: &Array2<f32>, psf: &Kernel2D) -> Result<Array2<f32>> {
    validate_input(input)?;
    validate(psf)?;

    let otf = psf2otf(psf, input.dim())?;
    blur_otf(input, &otf)
}

/// Blur a 2D image with a precomputed OTF.
///
/// `input.dim()` must match `otf.dims()`.
///
/// # Errors
///
/// Returns an error when `input` is empty or non-finite, dimensions do not
/// match, `otf` contains non-finite values, or FFT processing fails.
pub fn blur_otf(input: &Array2<f32>, otf: &Transfer2D) -> Result<Array2<f32>> {
    validate_input(input)?;
    if input.dim() != otf.dims() {
        return Err(Error::DimensionMismatch);
    }
    if !otf.is_finite() {
        return Err(Error::NonFiniteInput);
    }

    let mut cache = PlanCache::new();
    let mut spectrum = fft2_forward_real(input, &mut cache)?;
    multiply_in_place(&mut spectrum, otf.as_array())?;
    fft2_inverse_complex(&spectrum, &mut cache)
}

/// Blur a 3D volume with a spatial PSF using FFT convolution.
///
/// `input` uses `(depth, height, width)` order. The PSF is converted to an OTF
/// with the same dimensions as `input`.
///
/// # Errors
///
/// Returns an error when `input` is empty or non-finite, `psf` is invalid, PSF
/// to OTF conversion fails, or the frequency-domain product is non-finite.
pub fn blur_3d(input: &Array3<f32>, psf: &Kernel3D) -> Result<Array3<f32>> {
    validate_input_3d(input)?;
    validate_3d(psf)?;

    let otf = psf2otf_3d(psf, input.dim())?;
    blur_otf_3d(input, &otf)
}

/// Blur a 3D volume with a precomputed OTF.
///
/// `input.dim()` must match `otf.dims()` in `(depth, height, width)` order.
///
/// # Errors
///
/// Returns an error when `input` is empty or non-finite, dimensions do not
/// match, `otf` contains non-finite values, or FFT processing fails.
pub fn blur_otf_3d(input: &Array3<f32>, otf: &Transfer3D) -> Result<Array3<f32>> {
    validate_input_3d(input)?;
    if input.dim() != otf.dims() {
        return Err(Error::DimensionMismatch);
    }
    if !otf.is_finite() {
        return Err(Error::NonFiniteInput);
    }

    let mut cache = PlanCache::new();
    let mut spectrum = fft3_forward_real(input, &mut cache)?;
    multiply_in_place_3d(&mut spectrum, otf.as_array())?;
    fft3_inverse_complex(&spectrum, &mut cache)
}

/// Blur an image and add optional reproducible noise sources.
///
/// Gaussian and readout `sigma` values are additive intensity standard
/// deviations. `poisson_peak` scales normalized intensity into photon counts.
/// Each enabled noise source receives a deterministic seed derived from `seed`.
///
/// # Errors
///
/// Returns an error when the blur inputs are invalid, a noise parameter is
/// invalid, or any generated degraded value is non-finite.
pub fn degrade(
    input: &Array2<f32>,
    psf: &Kernel2D,
    gaussian_sigma: Option<f32>,
    poisson_peak: Option<f32>,
    readout_sigma: Option<f32>,
    seed: u64,
) -> Result<Array2<f32>> {
    validate_input(input)?;
    validate(psf)?;

    let mut degraded = blur(input, psf)?;

    if let Some(sigma) = gaussian_sigma {
        degraded = add_gaussian_noise(&degraded, sigma, derive_seed(seed, 1))?;
    }
    if let Some(peak) = poisson_peak {
        degraded = add_poisson_noise(&degraded, peak, derive_seed(seed, 2))?;
    }
    if let Some(sigma) = readout_sigma {
        degraded = add_readout_noise(&degraded, sigma, derive_seed(seed, 3))?;
    }

    Ok(degraded)
}

fn multiply_in_place(lhs: &mut Array2<Complex32>, rhs: &Array2<Complex32>) -> Result<()> {
    if lhs.dim() != rhs.dim() {
        return Err(Error::DimensionMismatch);
    }

    for ((y, x), value) in lhs.indexed_iter_mut() {
        let product = *value * rhs[[y, x]];
        if !product.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        *value = product;
    }

    Ok(())
}

fn multiply_in_place_3d(lhs: &mut Array3<Complex32>, rhs: &Array3<Complex32>) -> Result<()> {
    if lhs.dim() != rhs.dim() {
        return Err(Error::DimensionMismatch);
    }

    for ((z, y, x), value) in lhs.indexed_iter_mut() {
        let product = *value * rhs[[z, y, x]];
        if !product.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        *value = product;
    }

    Ok(())
}

fn validate_input(input: &Array2<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn validate_input_3d(input: &Array3<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn derive_seed(seed: u64, lane: u64) -> u64 {
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    seed.wrapping_add(GOLDEN.wrapping_mul(lane.saturating_add(1)))
}
