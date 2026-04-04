use ndarray::{Array2, Array3};

use crate::{Error, Result};

use super::validate::{finite_real_2d, finite_real_3d};

pub(crate) fn project_nonnegative_2d(input: &Array2<f32>) -> Result<Array2<f32>> {
    finite_real_2d(input)?;
    Ok(input.mapv(|value| value.max(0.0)))
}

pub(crate) fn project_nonnegative_3d(input: &Array3<f32>) -> Result<Array3<f32>> {
    finite_real_3d(input)?;
    Ok(input.mapv(|value| value.max(0.0)))
}

pub(crate) fn project_bounds_2d(
    input: &Array2<f32>,
    lower: f32,
    upper: f32,
) -> Result<Array2<f32>> {
    finite_real_2d(input)?;
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(Error::InvalidParameter);
    }
    Ok(input.mapv(|value| value.clamp(lower, upper)))
}

pub(crate) fn project_bounds_3d(
    input: &Array3<f32>,
    lower: f32,
    upper: f32,
) -> Result<Array3<f32>> {
    finite_real_3d(input)?;
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(Error::InvalidParameter);
    }
    Ok(input.mapv(|value| value.clamp(lower, upper)))
}
