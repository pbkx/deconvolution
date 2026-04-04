use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Poisson};

use crate::{Error, Result};

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
