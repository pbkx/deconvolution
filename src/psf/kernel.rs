use ndarray::{Array2, Array3};

use crate::otf::{Transfer2D, Transfer3D};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
/// 2D point-spread function stored as an `(height, width)` `f32` array.
///
/// Constructors copy data into standard layout and reject empty or non-finite
/// arrays. Use [`Kernel2D::normalized`] when the PSF should sum to `1`.
pub struct Kernel2D {
    data: Array2<f32>,
}

impl Kernel2D {
    /// Build a 2D kernel from `(height, width)` samples.
    ///
    /// The data is copied into standard layout.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPsf`] for empty arrays and
    /// [`Error::NonFiniteInput`] for `NaN` or infinite samples.
    pub fn new(data: Array2<f32>) -> Result<Self> {
        validate_real_2d(&data)?;
        Ok(Self {
            data: data.as_standard_layout().to_owned(),
        })
    }

    /// Borrow the underlying `(height, width)` array.
    pub fn as_array(&self) -> &Array2<f32> {
        &self.data
    }

    /// Consume the kernel and return its owned array.
    pub fn into_inner(self) -> Array2<f32> {
        self.data
    }

    /// Return `(height, width)`.
    pub fn dims(&self) -> (usize, usize) {
        self.data.dim()
    }

    /// Sum all kernel samples.
    pub fn sum(&self) -> f32 {
        self.data.sum()
    }

    /// Return whether every sample is finite.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }

    /// Normalize the kernel in place so its sum is `1`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPsf`] when the kernel sum is non-finite or too
    /// close to zero.
    pub fn normalize(&mut self) -> Result<()> {
        normalize_real_2d(&mut self.data, Error::InvalidPsf)
    }

    /// Return a normalized copy of this kernel.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPsf`] when the kernel sum is non-finite or too
    /// close to zero.
    pub fn normalized(&self) -> Result<Self> {
        let mut kernel = self.clone();
        kernel.normalize()?;
        Ok(kernel)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// 3D point-spread function stored as a `(depth, height, width)` `f32` array.
///
/// Constructors copy data into standard layout and reject empty or non-finite
/// arrays. Use [`Kernel3D::normalized`] when the PSF should sum to `1`.
pub struct Kernel3D {
    data: Array3<f32>,
}

impl Kernel3D {
    /// Build a 3D kernel from `(depth, height, width)` samples.
    ///
    /// The data is copied into standard layout.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPsf`] for empty arrays and
    /// [`Error::NonFiniteInput`] for `NaN` or infinite samples.
    pub fn new(data: Array3<f32>) -> Result<Self> {
        validate_real_3d(&data)?;
        Ok(Self {
            data: data.as_standard_layout().to_owned(),
        })
    }

    /// Borrow the underlying `(depth, height, width)` array.
    pub fn as_array(&self) -> &Array3<f32> {
        &self.data
    }

    /// Consume the kernel and return its owned array.
    pub fn into_inner(self) -> Array3<f32> {
        self.data
    }

    /// Return `(depth, height, width)`.
    pub fn dims(&self) -> (usize, usize, usize) {
        self.data.dim()
    }

    /// Sum all kernel samples.
    pub fn sum(&self) -> f32 {
        self.data.sum()
    }

    /// Return whether every sample is finite.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }

    /// Normalize the kernel in place so its sum is `1`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPsf`] when the kernel sum is non-finite or too
    /// close to zero.
    pub fn normalize(&mut self) -> Result<()> {
        normalize_real_3d(&mut self.data, Error::InvalidPsf)
    }

    /// Return a normalized copy of this kernel.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidPsf`] when the kernel sum is non-finite or too
    /// close to zero.
    pub fn normalized(&self) -> Result<Self> {
        let mut kernel = self.clone();
        kernel.normalize()?;
        Ok(kernel)
    }
}

#[derive(Debug, Clone, Copy)]
/// Borrowed 2D blur representation accepted by convolution helpers.
pub enum Blur2D<'a> {
    /// Spatial-domain point-spread function.
    Psf(&'a Kernel2D),
    /// Frequency-domain optical transfer function.
    Otf(&'a Transfer2D),
}

impl Blur2D<'_> {
    /// Return the shape of the borrowed PSF or OTF.
    pub fn dims(&self) -> (usize, usize) {
        match self {
            Self::Psf(psf) => psf.dims(),
            Self::Otf(otf) => otf.dims(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Borrowed 3D blur representation accepted by volume convolution helpers.
pub enum Blur3D<'a> {
    /// Spatial-domain point-spread function.
    Psf(&'a Kernel3D),
    /// Frequency-domain optical transfer function.
    Otf(&'a Transfer3D),
}

impl Blur3D<'_> {
    /// Return the shape of the borrowed PSF or OTF.
    pub fn dims(&self) -> (usize, usize, usize) {
        match self {
            Self::Psf(psf) => psf.dims(),
            Self::Otf(otf) => otf.dims(),
        }
    }
}

fn validate_real_2d(data: &Array2<f32>) -> Result<()> {
    if data.is_empty() {
        return Err(Error::InvalidPsf);
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn validate_real_3d(data: &Array3<f32>) -> Result<()> {
    if data.is_empty() {
        return Err(Error::InvalidPsf);
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn normalize_real_2d(data: &mut Array2<f32>, error: Error) -> Result<()> {
    let sum = data.iter().copied().fold(0.0_f32, |acc, value| acc + value);
    if !sum.is_finite() || sum.abs() <= f32::EPSILON {
        return Err(error);
    }
    for value in data.iter_mut() {
        *value /= sum;
    }
    Ok(())
}

fn normalize_real_3d(data: &mut Array3<f32>, error: Error) -> Result<()> {
    let sum = data.iter().copied().fold(0.0_f32, |acc, value| acc + value);
    if !sum.is_finite() || sum.abs() <= f32::EPSILON {
        return Err(error);
    }
    for value in data.iter_mut() {
        *value /= sum;
    }
    Ok(())
}
