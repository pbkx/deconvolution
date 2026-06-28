//! Reproducible noise models for synthetic restoration tests.
//!
//! The noise helpers take a `seed` so examples and tests can produce stable
//! noisy inputs.

use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson};

use crate::{Error, Result};

/// Add zero-mean Gaussian noise to an `(height, width)` image.
///
/// `sigma` is the intensity standard deviation. `sigma == 0` returns a clone of
/// the input.
///
/// # Errors
///
/// Returns an error when `input` is empty or non-finite, `sigma` is negative or
/// non-finite, distribution construction fails, or a sampled value is non-finite.
pub fn add_gaussian_noise(input: &Array2<f32>, sigma: f32, seed: u64) -> Result<Array2<f32>> {
    validate_input(input)?;

    if !sigma.is_finite() || sigma < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if sigma <= f32::EPSILON {
        return Ok(input.clone());
    }

    let normal = Normal::new(0.0_f64, sigma as f64).map_err(|_| Error::InvalidParameter)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut output = Array2::zeros(input.dim());

    for ((y, x), value) in input.indexed_iter() {
        let sample = normal.sample(&mut rng) as f32;
        let noisy = *value + sample;
        if !noisy.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        output[[y, x]] = noisy;
    }

    Ok(output)
}

/// Add Poisson shot noise to a non-negative intensity model.
///
/// `peak` converts intensity `1.0` into expected photon counts. Negative input
/// samples are clipped to `0.0` before sampling.
///
/// # Errors
///
/// Returns an error when `input` is empty or non-finite, `peak` is not positive
/// and finite, distribution construction fails, or a sampled value is non-finite.
pub fn add_poisson_noise(input: &Array2<f32>, peak: f32, seed: u64) -> Result<Array2<f32>> {
    validate_input(input)?;

    if !peak.is_finite() || peak <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut output = Array2::zeros(input.dim());

    for ((y, x), value) in input.indexed_iter() {
        let nonnegative = value.max(0.0);
        let lambda = nonnegative * peak;

        let noisy = if lambda <= f32::EPSILON {
            0.0
        } else {
            let distribution = Poisson::new(lambda as f64).map_err(|_| Error::InvalidParameter)?;
            distribution.sample(&mut rng) as f32 / peak
        };

        if !noisy.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        output[[y, x]] = noisy;
    }

    Ok(output)
}

/// Add Gaussian readout noise to an `(height, width)` image.
///
/// This is an alias for [`add_gaussian_noise`] with detector-oriented naming.
///
/// # Errors
///
/// Returns the same errors as [`add_gaussian_noise`].
pub fn add_readout_noise(input: &Array2<f32>, sigma: f32, seed: u64) -> Result<Array2<f32>> {
    add_gaussian_noise(input, sigma, seed)
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
