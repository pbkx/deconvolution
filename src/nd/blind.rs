//! Blind deconvolution wrappers for `ndarray` images.
//!
//! Inputs use `(height, width)` order and convert through the [`NdSample`]
//! `f32` compute path.

use ndarray::Array2;

use super::convert::{NdSample, array2_from_f32, array2_to_f32, kernel2_from_samples};
use crate::Result;
use crate::blind::{BlindMaximumLikelihood, BlindOutput, BlindRichardsonLucy};

/// Estimate a 2D ndarray and PSF with blind Richardson-Lucy.
///
/// # Errors
///
/// Returns an error when sample conversion fails, the input arrays are empty or
/// non-finite, the initial PSF is invalid, or blind updates become non-finite.
pub fn richardson_lucy<T: NdSample>(
    image: &Array2<T>,
    initial_psf: &Array2<T>,
) -> Result<BlindOutput<Array2<T>>> {
    richardson_lucy_with(image, initial_psf, &BlindRichardsonLucy::new())
}

/// Estimate a 2D ndarray and PSF with explicit blind Richardson-Lucy settings.
///
/// # Errors
///
/// Returns an error when sample conversion fails, the input arrays are empty or
/// non-finite, the initial PSF is invalid, or `config` has invalid constraints.
pub fn richardson_lucy_with<T: NdSample>(
    image: &Array2<T>,
    initial_psf: &Array2<T>,
    config: &BlindRichardsonLucy,
) -> Result<BlindOutput<Array2<T>>> {
    run_blind(image, initial_psf, |input, kernel| {
        crate::blind::richardson_lucy_array2_with(input, kernel, config)
    })
}

/// Estimate a 2D ndarray and PSF with blind maximum likelihood.
///
/// # Errors
///
/// Returns an error when sample conversion fails, the input arrays are empty or
/// non-finite, the initial PSF is invalid, or blind updates become non-finite.
pub fn maximum_likelihood<T: NdSample>(
    image: &Array2<T>,
    initial_psf: &Array2<T>,
) -> Result<BlindOutput<Array2<T>>> {
    maximum_likelihood_with(image, initial_psf, &BlindMaximumLikelihood::new())
}

/// Estimate a 2D ndarray and PSF with explicit blind ML settings.
///
/// # Errors
///
/// Returns an error when sample conversion fails, the input arrays are empty or
/// non-finite, the initial PSF is invalid, or `config` has invalid constraints.
pub fn maximum_likelihood_with<T: NdSample>(
    image: &Array2<T>,
    initial_psf: &Array2<T>,
    config: &BlindMaximumLikelihood,
) -> Result<BlindOutput<Array2<T>>> {
    run_blind(image, initial_psf, |input, kernel| {
        crate::blind::maximum_likelihood_array2_with(input, kernel, config)
    })
}

fn run_blind<T, F>(
    image: &Array2<T>,
    initial_psf: &Array2<T>,
    run: F,
) -> Result<BlindOutput<Array2<T>>>
where
    T: NdSample,
    F: FnOnce(&Array2<f32>, &crate::Kernel2D) -> Result<BlindOutput<Array2<f32>>>,
{
    let image = array2_to_f32(image)?;
    let initial_psf = kernel2_from_samples(initial_psf)?;
    let output = run(&image, &initial_psf)?;
    let image = array2_from_f32(&output.image)?;
    Ok(BlindOutput {
        image,
        psf: output.psf,
        report: output.report,
    })
}
