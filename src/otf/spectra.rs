use std::f32::consts::PI;

use ndarray::Array2;
use num_complex::Complex32;

use crate::{Error, Result};

use super::Transfer2D;

pub fn koehler_otf(dims: (usize, usize), cutoff_frequency: f32) -> Result<Transfer2D> {
    validate_dims_2d(dims)?;
    validate_cutoff(cutoff_frequency)?;

    let (height, width) = dims;
    let mut spectrum = Array2::zeros((height, width));

    for y in 0..height {
        let fy = normalized_frequency(y, height)?;
        for x in 0..width {
            let fx = normalized_frequency(x, width)?;
            let rho = (fx * fx + fy * fy).sqrt();
            let value = incoherent_circular_otf(rho, cutoff_frequency)?;
            spectrum[[y, x]] = Complex32::new(value, 0.0);
        }
    }

    Transfer2D::new(spectrum)
}

pub fn defocus_otf(
    dims: (usize, usize),
    cutoff_frequency: f32,
    defocus_strength: f32,
) -> Result<Transfer2D> {
    validate_dims_2d(dims)?;
    validate_cutoff(cutoff_frequency)?;
    validate_nonnegative(defocus_strength)?;

    let (height, width) = dims;
    let mut spectrum = Array2::zeros((height, width));

    for y in 0..height {
        let fy = normalized_frequency(y, height)?;
        for x in 0..width {
            let fx = normalized_frequency(x, width)?;
            let rho = (fx * fx + fy * fy).sqrt();
            let base = incoherent_circular_otf(rho, cutoff_frequency)?;
            let phase = (PI * defocus_strength * rho * rho).cos();
            let attenuation = (-0.5 * defocus_strength * rho * rho).exp();
            let value = base * phase * attenuation;
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            spectrum[[y, x]] = Complex32::new(value, 0.0);
        }
    }

    Transfer2D::new(spectrum)
}

fn validate_dims_2d(dims: (usize, usize)) -> Result<()> {
    if dims.0 == 0 || dims.1 == 0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_cutoff(value: f32) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_nonnegative(value: f32) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn normalized_frequency(index: usize, size: usize) -> Result<f32> {
    if size == 0 {
        return Err(Error::InvalidParameter);
    }

    let index_isize = isize::try_from(index).map_err(|_| Error::DimensionMismatch)?;
    let size_isize = isize::try_from(size).map_err(|_| Error::DimensionMismatch)?;
    let half = size / 2;
    let signed = if index <= half {
        index_isize
    } else {
        index_isize - size_isize
    };

    let denom = (size as f32 * 0.5).max(1.0);
    let value = signed as f32 / denom;
    if !value.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(value)
}

fn incoherent_circular_otf(rho: f32, cutoff_frequency: f32) -> Result<f32> {
    if !rho.is_finite() || rho < 0.0 {
        return Err(Error::InvalidParameter);
    }
    validate_cutoff(cutoff_frequency)?;

    let nu = rho / cutoff_frequency;
    if nu >= 1.0 {
        return Ok(0.0);
    }

    let one_minus = (1.0 - nu * nu).max(0.0);
    let value = (2.0 / PI) * (nu.acos() - nu * one_minus.sqrt());
    if !value.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(value.max(0.0))
}
