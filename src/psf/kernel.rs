use ndarray::{Array2, Array3};

use crate::otf::{Transfer2D, Transfer3D};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Kernel2D {
    data: Array2<f32>,
}

impl Kernel2D {
    pub fn new(data: Array2<f32>) -> Result<Self> {
        validate_real_2d(&data)?;
        Ok(Self {
            data: data.as_standard_layout().to_owned(),
        })
    }

    pub fn as_array(&self) -> &Array2<f32> {
        &self.data
    }

    pub fn into_inner(self) -> Array2<f32> {
        self.data
    }

    pub fn dims(&self) -> (usize, usize) {
        self.data.dim()
    }

    pub fn sum(&self) -> f32 {
        self.data.sum()
    }

    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }

    pub fn normalize(&mut self) -> Result<()> {
        normalize_real_2d(&mut self.data, Error::InvalidPsf)
    }

    pub fn normalized(&self) -> Result<Self> {
        let mut kernel = self.clone();
        kernel.normalize()?;
        Ok(kernel)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Kernel3D {
    data: Array3<f32>,
}

impl Kernel3D {
    pub fn new(data: Array3<f32>) -> Result<Self> {
        validate_real_3d(&data)?;
        Ok(Self {
            data: data.as_standard_layout().to_owned(),
        })
    }

    pub fn as_array(&self) -> &Array3<f32> {
        &self.data
    }

    pub fn into_inner(self) -> Array3<f32> {
        self.data
    }

    pub fn dims(&self) -> (usize, usize, usize) {
        self.data.dim()
    }

    pub fn sum(&self) -> f32 {
        self.data.sum()
    }

    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }

    pub fn normalize(&mut self) -> Result<()> {
        normalize_real_3d(&mut self.data, Error::InvalidPsf)
    }

    pub fn normalized(&self) -> Result<Self> {
        let mut kernel = self.clone();
        kernel.normalize()?;
        Ok(kernel)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Blur2D<'a> {
    Psf(&'a Kernel2D),
    Otf(&'a Transfer2D),
}

impl Blur2D<'_> {
    pub fn dims(&self) -> (usize, usize) {
        match self {
            Self::Psf(psf) => psf.dims(),
            Self::Otf(otf) => otf.dims(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Blur3D<'a> {
    Psf(&'a Kernel3D),
    Otf(&'a Transfer3D),
}

impl Blur3D<'_> {
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
