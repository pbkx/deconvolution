use ndarray::Array2;

use crate::core::validate::finite_real_2d;
use crate::{Error, Result};

pub(crate) fn tv_regularize_step_2d(
    input: &Array2<f32>,
    weight: f32,
    epsilon: f32,
) -> Result<Array2<f32>> {
    finite_real_2d(input)?;
    if !weight.is_finite() || weight < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if weight <= f32::EPSILON {
        return Ok(input.to_owned());
    }

    let (height, width) = input.dim();
    let mut px = Array2::zeros((height, width));
    let mut py = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let dx = if x + 1 < width {
                input[[y, x + 1]] - input[[y, x]]
            } else {
                0.0
            };
            let dy = if y + 1 < height {
                input[[y + 1, x]] - input[[y, x]]
            } else {
                0.0
            };
            let norm = (dx * dx + dy * dy + epsilon * epsilon).sqrt();
            if !norm.is_finite() || norm <= 0.0 {
                return Err(Error::NonFiniteInput);
            }
            px[[y, x]] = dx / norm;
            py[[y, x]] = dy / norm;
        }
    }

    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let div_x = px[[y, x]] - if x > 0 { px[[y, x - 1]] } else { 0.0 };
            let div_y = py[[y, x]] - if y > 0 { py[[y - 1, x]] } else { 0.0 };
            let value = input[[y, x]] + weight * (div_x + div_y);
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }

    Ok(output)
}
