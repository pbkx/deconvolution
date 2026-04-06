use ndarray::Array2;

use crate::psf::{defocus, gaussian2d, motion_linear, oriented_gaussian};
use crate::{Error, Kernel2D, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParametricPsf {
    Gaussian {
        sigma: f32,
    },
    MotionLinear {
        length: f32,
        angle_deg: f32,
    },
    Defocus {
        radius: f32,
    },
    OrientedGaussian {
        sigma_major: f32,
        sigma_minor: f32,
        angle_deg: f32,
    },
}

impl ParametricPsf {
    pub fn realize(&self, dims: (usize, usize)) -> Result<Kernel2D> {
        validate_dims(dims)?;
        match *self {
            Self::Gaussian { sigma } => gaussian2d(dims, sigma),
            Self::MotionLinear { length, angle_deg } => {
                let base = motion_linear(length, angle_deg)?;
                fit_and_normalize(base.as_array(), dims)
            }
            Self::Defocus { radius } => {
                let base = defocus(radius)?;
                fit_and_normalize(base.as_array(), dims)
            }
            Self::OrientedGaussian {
                sigma_major,
                sigma_minor,
                angle_deg,
            } => oriented_gaussian(dims, sigma_major, sigma_minor, angle_deg),
        }
    }
}

fn fit_and_normalize(input: &Array2<f32>, dims: (usize, usize)) -> Result<Kernel2D> {
    let resized = fit_to_dims(input, dims)?;
    let mut kernel = Kernel2D::new(resized)?;
    kernel.normalize()?;
    Ok(kernel)
}

fn fit_to_dims(input: &Array2<f32>, dims: (usize, usize)) -> Result<Array2<f32>> {
    let (target_h, target_w) = dims;
    validate_dims(dims)?;

    let (source_h, source_w) = input.dim();
    if source_h == 0 || source_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let copy_h = source_h.min(target_h);
    let copy_w = source_w.min(target_w);
    let source_y = (source_h - copy_h) / 2;
    let source_x = (source_w - copy_w) / 2;
    let target_y = (target_h - copy_h) / 2;
    let target_x = (target_w - copy_w) / 2;

    let mut output = Array2::zeros((target_h, target_w));
    for y in 0..copy_h {
        for x in 0..copy_w {
            output[[target_y + y, target_x + x]] = input[[source_y + y, source_x + x]];
        }
    }

    Ok(output)
}

fn validate_dims(dims: (usize, usize)) -> Result<()> {
    if dims.0 == 0 || dims.1 == 0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}
