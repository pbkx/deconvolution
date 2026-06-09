use ndarray::{Array2, Array3, Axis};

use crate::{Error, Kernel2D, Result};

pub trait NdSample: Copy {
    fn to_f32(self) -> Result<f32>;

    fn from_f32(value: f32) -> Result<Self>;
}

impl NdSample for f32 {
    fn to_f32(self) -> Result<f32> {
        if !self.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        Ok(self)
    }

    fn from_f32(value: f32) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        Ok(value)
    }
}

#[cfg(feature = "f16")]
impl NdSample for half::f16 {
    fn to_f32(self) -> Result<f32> {
        let value = self.to_f32();
        if !value.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        Ok(value)
    }

    fn from_f32(value: f32) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        let value = half::f16::from_f32(value);
        if !value.to_f32().is_finite() {
            return Err(Error::NonFiniteInput);
        }
        Ok(value)
    }
}

pub(crate) fn array2_to_f32<T: NdSample>(input: &Array2<T>) -> Result<Array2<f32>> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    let mut output = Array2::zeros(input.dim());
    for ((y, x), value) in input.indexed_iter() {
        output[[y, x]] = value.to_f32()?;
    }
    Ok(output.as_standard_layout().to_owned())
}

pub(crate) fn array2_from_f32<T: NdSample>(input: &Array2<f32>) -> Result<Array2<T>> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    let mut output = Array2::from_elem(input.dim(), T::from_f32(0.0)?);
    for ((y, x), value) in input.indexed_iter() {
        output[[y, x]] = T::from_f32(*value)?;
    }
    Ok(output.as_standard_layout().to_owned())
}

pub(crate) fn kernel2_from_samples<T: NdSample>(input: &Array2<T>) -> Result<Kernel2D> {
    let input = array2_to_f32(input)?;
    Kernel2D::new(input)
}

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
