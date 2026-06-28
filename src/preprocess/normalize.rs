//! Range normalization helpers for arrays.
//!
//! Use [`normalize_range`] to apply a [`crate::RangePolicy`] to an `(height,
//! width)` image before or after restoration.

use ndarray::{Array2, Array3};

use crate::{Error, RangePolicy, Result};

pub fn normalize_range(input: &Array2<f32>, policy: RangePolicy) -> Result<Array2<f32>> {
    validate_input(input)?;

    let output = match policy {
        RangePolicy::PreserveInput | RangePolicy::Unbounded => input.clone(),
        RangePolicy::Clamp01 => input.mapv(|value| value.clamp(0.0, 1.0)),
        RangePolicy::ClampNegPos1 => input.mapv(|value| value.clamp(-1.0, 1.0)),
    };

    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    Ok(output)
}

pub(crate) fn normalize_range_3d(input: &Array3<f32>, policy: RangePolicy) -> Result<Array3<f32>> {
    validate_input_3d(input)?;

    let output = match policy {
        RangePolicy::PreserveInput | RangePolicy::Unbounded => input.clone(),
        RangePolicy::Clamp01 => input.mapv(|value| value.clamp(0.0, 1.0)),
        RangePolicy::ClampNegPos1 => input.mapv(|value| value.clamp(-1.0, 1.0)),
    };

    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    Ok(output)
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

fn validate_input_3d(input: &Array3<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}
