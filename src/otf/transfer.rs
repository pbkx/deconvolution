use ndarray::{Array2, Array3};
use num_complex::Complex32;

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Transfer2D {
    data: Array2<Complex32>,
}

impl Transfer2D {
    pub fn new(data: Array2<Complex32>) -> Result<Self> {
        validate_complex_2d(&data)?;
        Ok(Self {
            data: data.as_standard_layout().to_owned(),
        })
    }

    pub fn as_array(&self) -> &Array2<Complex32> {
        &self.data
    }

    pub fn into_inner(self) -> Array2<Complex32> {
        self.data
    }

    pub fn dims(&self) -> (usize, usize) {
        self.data.dim()
    }

    pub fn sum(&self) -> Complex32 {
        self.data
            .iter()
            .copied()
            .fold(Complex32::new(0.0, 0.0), |acc, value| acc + value)
    }

    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }

    pub fn normalize(&mut self) -> Result<()> {
        let sum = self.sum();
        if !sum.is_finite() || sum.norm() <= f32::EPSILON {
            return Err(Error::InvalidTransfer);
        }
        self.data.iter_mut().for_each(|value| *value /= sum);
        Ok(())
    }

    pub fn normalized(&self) -> Result<Self> {
        let mut transfer = self.clone();
        transfer.normalize()?;
        Ok(transfer)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transfer3D {
    data: Array3<Complex32>,
}

impl Transfer3D {
    pub fn new(data: Array3<Complex32>) -> Result<Self> {
        validate_complex_3d(&data)?;
        Ok(Self {
            data: data.as_standard_layout().to_owned(),
        })
    }

    pub fn as_array(&self) -> &Array3<Complex32> {
        &self.data
    }

    pub fn into_inner(self) -> Array3<Complex32> {
        self.data
    }

    pub fn dims(&self) -> (usize, usize, usize) {
        self.data.dim()
    }

    pub fn sum(&self) -> Complex32 {
        self.data
            .iter()
            .copied()
            .fold(Complex32::new(0.0, 0.0), |acc, value| acc + value)
    }

    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }

    pub fn normalize(&mut self) -> Result<()> {
        let sum = self.sum();
        if !sum.is_finite() || sum.norm() <= f32::EPSILON {
            return Err(Error::InvalidTransfer);
        }
        self.data.iter_mut().for_each(|value| *value /= sum);
        Ok(())
    }

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
