use ndarray::Array2;

use crate::psf::basic::{gaussian2d, motion_linear};
use crate::{Error, Kernel2D, Result};

pub fn uniform(dims: (usize, usize)) -> Result<Kernel2D> {
    let (height, width) = dims;
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut kernel = Kernel2D::new(Array2::ones((height, width)))?;
    kernel.normalize()?;
    Ok(kernel)
}

pub fn gaussian_guess(dims: (usize, usize), sigma: f32) -> Result<Kernel2D> {
    gaussian2d(dims, sigma)
}

pub fn motion_guess(dims: (usize, usize), length: f32, angle_deg: f32) -> Result<Kernel2D> {
    let base = motion_linear(length, angle_deg)?;
    let resized = fit_to_dims(base.as_array(), dims)?;
    let mut kernel = Kernel2D::new(resized)?;
    kernel.normalize()?;
    Ok(kernel)
}

pub fn from_support(support: &Array2<bool>) -> Result<Kernel2D> {
    if support.is_empty() {
        return Err(Error::InvalidParameter);
    }

    let true_count = support.iter().filter(|value| **value).count();
    if true_count == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut kernel = Array2::zeros(support.dim());
    for ((y, x), value) in support.indexed_iter() {
        kernel[[y, x]] = if *value { 1.0 } else { 0.0 };
    }

    let mut kernel = Kernel2D::new(kernel)?;
    kernel.normalize()?;
    Ok(kernel)
}

fn fit_to_dims(input: &Array2<f32>, dims: (usize, usize)) -> Result<Array2<f32>> {
    let (target_h, target_w) = dims;
    if target_h == 0 || target_w == 0 {
        return Err(Error::InvalidParameter);
    }

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
