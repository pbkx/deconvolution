use ndarray::{Array2, Array3};

use crate::{Error, Kernel2D, Kernel3D, Result};

pub fn normalize(kernel: &Kernel2D) -> Result<Kernel2D> {
    let mut normalized = kernel.clone();
    normalized.normalize()?;
    Ok(normalized)
}

pub fn normalize_3d(kernel: &Kernel3D) -> Result<Kernel3D> {
    let mut normalized = kernel.clone();
    normalized.normalize()?;
    Ok(normalized)
}

pub fn center(kernel: &Kernel2D) -> Result<Kernel2D> {
    validate(kernel)?;
    let shifted = center_array_2d(kernel.as_array())?;
    Kernel2D::new(shifted)
}

pub fn center_3d(kernel: &Kernel3D) -> Result<Kernel3D> {
    validate_3d(kernel)?;
    let shifted = center_array_3d(kernel.as_array())?;
    Kernel3D::new(shifted)
}

pub fn pad_to(kernel: &Kernel2D, dims: (usize, usize)) -> Result<Kernel2D> {
    validate(kernel)?;
    let (target_h, target_w) = dims;
    if target_h == 0 || target_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let (source_h, source_w) = kernel.dims();
    if source_h > target_h || source_w > target_w {
        return Err(Error::DimensionMismatch);
    }

    let offset_y = (target_h - source_h) / 2;
    let offset_x = (target_w - source_w) / 2;

    let mut padded = Array2::zeros((target_h, target_w));
    for y in 0..source_h {
        for x in 0..source_w {
            padded[[offset_y + y, offset_x + x]] = kernel.as_array()[[y, x]];
        }
    }

    Kernel2D::new(padded)
}

pub fn pad_to_3d(kernel: &Kernel3D, dims: (usize, usize, usize)) -> Result<Kernel3D> {
    validate_3d(kernel)?;
    let (target_d, target_h, target_w) = dims;
    if target_d == 0 || target_h == 0 || target_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let (source_d, source_h, source_w) = kernel.dims();
    if source_d > target_d || source_h > target_h || source_w > target_w {
        return Err(Error::DimensionMismatch);
    }

    let offset_d = (target_d - source_d) / 2;
    let offset_y = (target_h - source_h) / 2;
    let offset_x = (target_w - source_w) / 2;

    let mut padded = Array3::zeros((target_d, target_h, target_w));
    for d in 0..source_d {
        for y in 0..source_h {
            for x in 0..source_w {
                padded[[offset_d + d, offset_y + y, offset_x + x]] = kernel.as_array()[[d, y, x]];
            }
        }
    }

    Kernel3D::new(padded)
}

pub fn crop_to(kernel: &Kernel2D, dims: (usize, usize)) -> Result<Kernel2D> {
    validate(kernel)?;
    let (target_h, target_w) = dims;
    if target_h == 0 || target_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let (source_h, source_w) = kernel.dims();
    if target_h > source_h || target_w > source_w {
        return Err(Error::DimensionMismatch);
    }

    let offset_y = (source_h - target_h) / 2;
    let offset_x = (source_w - target_w) / 2;
    let mut cropped = Array2::zeros((target_h, target_w));
    for y in 0..target_h {
        for x in 0..target_w {
            cropped[[y, x]] = kernel.as_array()[[offset_y + y, offset_x + x]];
        }
    }

    Kernel2D::new(cropped)
}

pub fn crop_to_3d(kernel: &Kernel3D, dims: (usize, usize, usize)) -> Result<Kernel3D> {
    validate_3d(kernel)?;
    let (target_d, target_h, target_w) = dims;
    if target_d == 0 || target_h == 0 || target_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let (source_d, source_h, source_w) = kernel.dims();
    if target_d > source_d || target_h > source_h || target_w > source_w {
        return Err(Error::DimensionMismatch);
    }

    let offset_d = (source_d - target_d) / 2;
    let offset_y = (source_h - target_h) / 2;
    let offset_x = (source_w - target_w) / 2;
    let mut cropped = Array3::zeros((target_d, target_h, target_w));
    for d in 0..target_d {
        for y in 0..target_h {
            for x in 0..target_w {
                cropped[[d, y, x]] = kernel.as_array()[[offset_d + d, offset_y + y, offset_x + x]];
            }
        }
    }

    Kernel3D::new(cropped)
}

pub fn flip(kernel: &Kernel2D) -> Result<Kernel2D> {
    validate(kernel)?;
    let (height, width) = kernel.dims();
    let mut flipped = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            flipped[[height - 1 - y, width - 1 - x]] = kernel.as_array()[[y, x]];
        }
    }
    Kernel2D::new(flipped)
}

pub fn flip_3d(kernel: &Kernel3D) -> Result<Kernel3D> {
    validate_3d(kernel)?;
    let (depth, height, width) = kernel.dims();
    let mut flipped = Array3::zeros((depth, height, width));
    for d in 0..depth {
        for y in 0..height {
            for x in 0..width {
                flipped[[depth - 1 - d, height - 1 - y, width - 1 - x]] =
                    kernel.as_array()[[d, y, x]];
            }
        }
    }
    Kernel3D::new(flipped)
}

pub fn validate(kernel: &Kernel2D) -> Result<()> {
    if !kernel.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    let sum = kernel.sum();
    if !sum.is_finite() || sum.abs() <= f32::EPSILON {
        return Err(Error::InvalidPsf);
    }
    Ok(())
}

pub fn validate_3d(kernel: &Kernel3D) -> Result<()> {
    if !kernel.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    let sum = kernel.sum();
    if !sum.is_finite() || sum.abs() <= f32::EPSILON {
        return Err(Error::InvalidPsf);
    }
    Ok(())
}

pub fn support_mask(kernel: &Kernel2D, threshold: f32) -> Result<Array2<bool>> {
    validate(kernel)?;
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidParameter);
    }

    let max_abs = kernel
        .as_array()
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let cutoff = max_abs * threshold;

    let mut mask = Array2::from_elem(kernel.dims(), false);
    for ((y, x), value) in kernel.as_array().indexed_iter() {
        mask[[y, x]] = value.abs() >= cutoff;
    }
    Ok(mask)
}

pub fn support_mask_3d(kernel: &Kernel3D, threshold: f32) -> Result<Array3<bool>> {
    validate_3d(kernel)?;
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidParameter);
    }

    let max_abs = kernel
        .as_array()
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let cutoff = max_abs * threshold;

    let mut mask = Array3::from_elem(kernel.dims(), false);
    for ((d, y, x), value) in kernel.as_array().indexed_iter() {
        mask[[d, y, x]] = value.abs() >= cutoff;
    }
    Ok(mask)
}

fn center_array_2d(input: &Array2<f32>) -> Result<Array2<f32>> {
    let (height, width) = input.dim();
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut max_y = 0usize;
    let mut max_x = 0usize;
    let mut max_value = f32::NEG_INFINITY;

    for ((y, x), value) in input.indexed_iter() {
        let magnitude = value.abs();
        if magnitude > max_value {
            max_value = magnitude;
            max_y = y;
            max_x = x;
        }
    }

    let target_y = height / 2;
    let target_x = width / 2;
    let shift_y = to_isize(target_y)? - to_isize(max_y)?;
    let shift_x = to_isize(target_x)? - to_isize(max_x)?;

    circular_shift_2d(input, shift_y, shift_x)
}

fn center_array_3d(input: &Array3<f32>) -> Result<Array3<f32>> {
    let (depth, height, width) = input.dim();
    if depth == 0 || height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut max_d = 0usize;
    let mut max_y = 0usize;
    let mut max_x = 0usize;
    let mut max_value = f32::NEG_INFINITY;

    for ((d, y, x), value) in input.indexed_iter() {
        let magnitude = value.abs();
        if magnitude > max_value {
            max_value = magnitude;
            max_d = d;
            max_y = y;
            max_x = x;
        }
    }

    let target_d = depth / 2;
    let target_y = height / 2;
    let target_x = width / 2;
    let shift_d = to_isize(target_d)? - to_isize(max_d)?;
    let shift_y = to_isize(target_y)? - to_isize(max_y)?;
    let shift_x = to_isize(target_x)? - to_isize(max_x)?;

    circular_shift_3d(input, shift_d, shift_y, shift_x)
}

fn circular_shift_2d(input: &Array2<f32>, shift_y: isize, shift_x: isize) -> Result<Array2<f32>> {
    let (height, width) = input.dim();
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let height_isize = to_isize(height)?;
    let width_isize = to_isize(width)?;
    let mut shifted = Array2::zeros((height, width));
    for y in 0..height {
        let y_isize = to_isize(y)?;
        for x in 0..width {
            let x_isize = to_isize(x)?;
            let ny = (y_isize + shift_y).rem_euclid(height_isize);
            let nx = (x_isize + shift_x).rem_euclid(width_isize);
            shifted[[to_usize(ny)?, to_usize(nx)?]] = input[[y, x]];
        }
    }
    Ok(shifted)
}

fn circular_shift_3d(
    input: &Array3<f32>,
    shift_d: isize,
    shift_y: isize,
    shift_x: isize,
) -> Result<Array3<f32>> {
    let (depth, height, width) = input.dim();
    if depth == 0 || height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let depth_isize = to_isize(depth)?;
    let height_isize = to_isize(height)?;
    let width_isize = to_isize(width)?;
    let mut shifted = Array3::zeros((depth, height, width));
    for d in 0..depth {
        let d_isize = to_isize(d)?;
        for y in 0..height {
            let y_isize = to_isize(y)?;
            for x in 0..width {
                let x_isize = to_isize(x)?;
                let nd = (d_isize + shift_d).rem_euclid(depth_isize);
                let ny = (y_isize + shift_y).rem_euclid(height_isize);
                let nx = (x_isize + shift_x).rem_euclid(width_isize);
                shifted[[to_usize(nd)?, to_usize(ny)?, to_usize(nx)?]] = input[[d, y, x]];
            }
        }
    }
    Ok(shifted)
}

fn to_isize(value: usize) -> Result<isize> {
    isize::try_from(value).map_err(|_| Error::DimensionMismatch)
}

fn to_usize(value: isize) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::DimensionMismatch)
}
