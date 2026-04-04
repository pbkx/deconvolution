use ndarray::{Array2, Array3};
use num_complex::Complex32;

use crate::{Error, Result};

pub(crate) fn finite_real_2d(input: &Array2<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::InvalidParameter);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn finite_real_3d(input: &Array3<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::InvalidParameter);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn finite_complex_2d(input: &Array2<Complex32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::InvalidParameter);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn finite_complex_3d(input: &Array3<Complex32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::InvalidParameter);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn same_dims_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> Result<()> {
    if lhs.dim() != rhs.dim() {
        return Err(Error::DimensionMismatch);
    }
    Ok(())
}

pub(crate) fn same_dims_3d(lhs: &Array3<f32>, rhs: &Array3<f32>) -> Result<()> {
    if lhs.dim() != rhs.dim() {
        return Err(Error::DimensionMismatch);
    }
    Ok(())
}
