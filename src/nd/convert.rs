use ndarray::Array2;

use crate::{Error, Kernel2D, Result};

mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}

    #[cfg(feature = "f16")]
    impl Sealed for half::f16 {}
}

/// Scalar type accepted by the ndarray convenience APIs.
///
/// Implementations convert through `f32`. The trait is sealed so downstream
/// crates cannot add sample types with different finite-value behavior.
pub trait NdSample: Copy + sealed::Sealed {
    /// Convert a sample into the crate's `f32` compute path.
    fn to_f32(self) -> Result<f32>;

    /// Convert a finite `f32` solver output back into the sample type.
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
