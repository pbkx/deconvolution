use ndarray::{Array2, Array3};
use num_complex::Complex32;

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
/// 2D optical transfer function stored as an `(height, width)` complex array.
///
/// Constructors copy data into standard layout and reject empty or non-finite
/// arrays. The dimensions should match the padded image used for spectral work.
pub struct Transfer2D {
    data: Array2<Complex32>,
}

impl Transfer2D {
    /// Build a 2D transfer from `(height, width)` complex samples.
    ///
    /// The data is copied into standard layout.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTransfer`] for empty arrays and
    /// [`Error::NonFiniteInput`] for non-finite real or imaginary parts.
    pub fn new(data: Array2<Complex32>) -> Result<Self> {
        validate_complex_2d(&data)?;
        Ok(Self {
            data: data.as_standard_layout().to_owned(),
        })
    }

    /// Borrow the underlying `(height, width)` complex array.
    pub fn as_array(&self) -> &Array2<Complex32> {
        &self.data
    }

    /// Consume the transfer and return its owned complex array.
    pub fn into_inner(self) -> Array2<Complex32> {
        self.data
    }

    /// Return `(height, width)`.
    pub fn dims(&self) -> (usize, usize) {
        self.data.dim()
    }

    /// Sum all complex transfer samples.
    pub fn sum(&self) -> Complex32 {
        self.data
            .iter()
            .copied()
            .fold(Complex32::new(0.0, 0.0), |acc, value| acc + value)
    }

    /// Return whether every real and imaginary part is finite.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }

    /// Normalize the transfer in place so its complex sum is `1 + 0i`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTransfer`] when the complex sum is non-finite or
    /// too close to zero.
    pub fn normalize(&mut self) -> Result<()> {
        let sum = self.sum();
        if !sum.is_finite() || sum.norm() <= f32::EPSILON {
            return Err(Error::InvalidTransfer);
        }
        self.data.iter_mut().for_each(|value| *value /= sum);
        Ok(())
    }

    /// Return a normalized copy of this transfer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTransfer`] when the complex sum is non-finite or
    /// too close to zero.
    pub fn normalized(&self) -> Result<Self> {
        let mut transfer = self.clone();
        transfer.normalize()?;
        Ok(transfer)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// 3D optical transfer function stored as a `(depth, height, width)` complex array.
///
/// Constructors copy data into standard layout and reject empty or non-finite
/// arrays. The dimensions should match the padded volume used for spectral work.
pub struct Transfer3D {
    data: Array3<Complex32>,
}

impl Transfer3D {
    /// Build a 3D transfer from `(depth, height, width)` complex samples.
    ///
    /// The data is copied into standard layout.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTransfer`] for empty arrays and
    /// [`Error::NonFiniteInput`] for non-finite real or imaginary parts.
    pub fn new(data: Array3<Complex32>) -> Result<Self> {
        validate_complex_3d(&data)?;
        Ok(Self {
            data: data.as_standard_layout().to_owned(),
        })
    }

    /// Borrow the underlying `(depth, height, width)` complex array.
    pub fn as_array(&self) -> &Array3<Complex32> {
        &self.data
    }

    /// Consume the transfer and return its owned complex array.
    pub fn into_inner(self) -> Array3<Complex32> {
        self.data
    }

    /// Return `(depth, height, width)`.
    pub fn dims(&self) -> (usize, usize, usize) {
        self.data.dim()
    }

    /// Sum all complex transfer samples.
    pub fn sum(&self) -> Complex32 {
        self.data
            .iter()
            .copied()
            .fold(Complex32::new(0.0, 0.0), |acc, value| acc + value)
    }

    /// Return whether every real and imaginary part is finite.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }

    /// Normalize the transfer in place so its complex sum is `1 + 0i`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTransfer`] when the complex sum is non-finite or
    /// too close to zero.
    pub fn normalize(&mut self) -> Result<()> {
        let sum = self.sum();
        if !sum.is_finite() || sum.norm() <= f32::EPSILON {
            return Err(Error::InvalidTransfer);
        }
        self.data.iter_mut().for_each(|value| *value /= sum);
        Ok(())
    }

    /// Return a normalized copy of this transfer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTransfer`] when the complex sum is non-finite or
    /// too close to zero.
    pub fn normalized(&self) -> Result<Self> {
        let mut transfer = self.clone();
        transfer.normalize()?;
        Ok(transfer)
    }
}

fn validate_complex_2d(data: &Array2<Complex32>) -> Result<()> {
    if data.is_empty() {
        return Err(Error::InvalidTransfer);
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn validate_complex_3d(data: &Array3<Complex32>) -> Result<()> {
    if data.is_empty() {
        return Err(Error::InvalidTransfer);
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}
