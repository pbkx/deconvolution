use ndarray::{Array2, Array3, Axis};

use crate::{Error, Kernel2D, Result};

pub(crate) fn kernel2_from_array(input: &Array2<f32>) -> Result<Kernel2D> {
    validate_array2(input)?;
    Kernel2D::new(input.as_standard_layout().to_owned())
}

pub(crate) fn kernel3_to_projected_kernel2(input: &Array3<f32>) -> Result<Kernel2D> {
    validate_array3(input)?;
    let mut projected = input.sum_axis(Axis(0));
    let sum = projected.sum();
    if !sum.is_finite() || sum.abs() <= f32::EPSILON {
        return Err(Error::InvalidPsf);
    }
    for value in &mut projected {
        *value /= sum;
    }
    Kernel2D::new(projected)
}

pub(crate) fn validate_array2(input: &Array2<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn validate_array3(input: &Array3<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}
