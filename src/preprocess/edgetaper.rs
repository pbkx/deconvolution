use std::f32::consts::FRAC_PI_2;

use ndarray::Array2;

use crate::preprocess::padding::convolve_same_2d;
use crate::psf::support::{normalize, validate};
use crate::psf::Kernel2D;
use crate::{Boundary, Error, Result};

pub fn edgetaper(input: &Array2<f32>, psf: &Kernel2D) -> Result<Array2<f32>> {
    validate_input(input)?;
    validate(psf)?;

    let (height, width) = input.dim();
    let (psf_h, psf_w) = psf.dims();

    if psf_h > height || psf_w > width {
        return Err(Error::DimensionMismatch);
    }

    let normalized_psf = normalize(psf)?;
    let blurred = convolve_same_2d(input, normalized_psf.as_array(), Boundary::Periodic)?;

    let radius_y = ((psf_h - 1) / 2).max(1);
    let radius_x = ((psf_w - 1) / 2).max(1);
    let wy = taper_weights(height, radius_y)?;
    let wx = taper_weights(width, radius_x)?;

    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let alpha = wy[y] * wx[x];
            let value = alpha * input[[y, x]] + (1.0 - alpha) * blurred[[y, x]];
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }

    Ok(output)
}

fn taper_weights(length: usize, radius: usize) -> Result<Vec<f32>> {
    if length == 0 {
        return Err(Error::InvalidParameter);
    }

    if radius == 0 {
        return Ok(vec![1.0; length]);
    }

    let edge = radius.min(length / 2).max(1);
    let edge_f = edge as f32;
    let last = length - 1;
    let mut weights = vec![1.0_f32; length];

    for (index, weight) in weights.iter_mut().enumerate() {
        let distance = index.min(last - index);
        if distance >= edge {
            *weight = 1.0;
            continue;
        }

        let phase = (distance as f32) / edge_f;
        *weight = (phase * FRAC_PI_2).sin().powi(2);
    }

    Ok(weights)
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
