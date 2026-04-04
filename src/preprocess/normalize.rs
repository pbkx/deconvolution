use ndarray::Array2;

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

fn validate_input(input: &Array2<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}
