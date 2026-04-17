use ndarray::Array2;
use num_complex::Complex32;

use crate::core::fft::{fft2_forward_real, fft2_inverse_complex};
use crate::core::plan_cache::PlanCache;
use crate::otf::convert::psf2otf;
use crate::otf::Transfer2D;
use crate::psf::support::validate;
use crate::psf::Kernel2D;
use crate::simulate::noise::{add_gaussian_noise, add_poisson_noise, add_readout_noise};
use crate::{Error, Result};

pub fn blur(input: &Array2<f32>, psf: &Kernel2D) -> Result<Array2<f32>> {
    validate_input(input)?;
    validate(psf)?;

    let otf = psf2otf(psf, input.dim())?;
    blur_otf(input, &otf)
}

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

fn validate_input(input: &Array2<f32>) -> Result<()> {
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
