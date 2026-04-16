#![allow(dead_code)]

use image::{GrayImage, Luma};
use ndarray::{Array2, Array3};

pub fn arrays_equal_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> bool {
    lhs.dim() == rhs.dim()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub fn arrays_equal_3d(lhs: &Array3<f32>, rhs: &Array3<f32>) -> bool {
    lhs.dim() == rhs.dim()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub fn arrays_differ_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> bool {
    if lhs.dim() != rhs.dim() {
        return true;
    }
    lhs.iter()
        .zip(rhs.iter())
        .any(|(left, right)| left.to_bits() != right.to_bits())
}

pub fn is_finite_2d(input: &Array2<f32>) -> bool {
    input.iter().all(|value| value.is_finite())
}

pub fn is_finite_3d(input: &Array3<f32>) -> bool {
    input.iter().all(|value| value.is_finite())
}

pub fn array_to_gray(input: &Array2<f32>) -> deconvolution::Result<GrayImage> {
    let (height, width) = input.dim();
    let width_u32 = u32::try_from(width).map_err(|_| deconvolution::Error::DimensionMismatch)?;
    let height_u32 = u32::try_from(height).map_err(|_| deconvolution::Error::DimensionMismatch)?;
    let mut image = GrayImage::new(width_u32, height_u32);

    for y in 0..height {
        let y_u32 = u32::try_from(y).map_err(|_| deconvolution::Error::DimensionMismatch)?;
        for x in 0..width {
            let x_u32 = u32::try_from(x).map_err(|_| deconvolution::Error::DimensionMismatch)?;
            let value = (input[[y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
            image.put_pixel(x_u32, y_u32, Luma([value]));
        }
    }

    Ok(image)
}

pub fn gray_to_array(input: &GrayImage) -> Array2<f32> {
    let width = input.width() as usize;
    let height = input.height() as usize;
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            output[[y, x]] = f32::from(input.get_pixel(x as u32, y as u32)[0]) / 255.0;
        }
    }
    output
}

pub fn mse_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> deconvolution::Result<f32> {
    if lhs.dim() != rhs.dim() {
        return Err(deconvolution::Error::DimensionMismatch);
    }
    if lhs.is_empty() {
        return Err(deconvolution::Error::InvalidParameter);
    }

    let mut sum = 0.0_f32;
    let count = lhs.len() as f32;
    for ((y, x), value) in lhs.indexed_iter() {
        let diff = *value - rhs[[y, x]];
        sum += diff * diff;
    }
    Ok(sum / count)
}

pub fn mse_3d(lhs: &Array3<f32>, rhs: &Array3<f32>) -> deconvolution::Result<f32> {
    if lhs.dim() != rhs.dim() {
        return Err(deconvolution::Error::DimensionMismatch);
    }
    if lhs.is_empty() {
        return Err(deconvolution::Error::InvalidParameter);
    }

    let mut sum = 0.0_f32;
    let count = lhs.len() as f32;
    for ((z, y, x), value) in lhs.indexed_iter() {
        let diff = *value - rhs[[z, y, x]];
        sum += diff * diff;
    }
    Ok(sum / count)
}

pub fn psnr_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> deconvolution::Result<f32> {
    let mse = mse_2d(lhs, rhs)?;
    if mse <= f32::EPSILON {
        return Ok(f32::INFINITY);
    }
    Ok(10.0 * (1.0 / mse).log10())
}

pub fn max_abs_diff_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> deconvolution::Result<f32> {
    if lhs.dim() != rhs.dim() {
        return Err(deconvolution::Error::DimensionMismatch);
    }
    if lhs.is_empty() {
        return Err(deconvolution::Error::InvalidParameter);
    }

    let mut max_diff = 0.0_f32;
    for ((y, x), value) in lhs.indexed_iter() {
        let diff = (*value - rhs[[y, x]]).abs();
        if !diff.is_finite() {
            return Err(deconvolution::Error::NonFiniteInput);
        }
        max_diff = max_diff.max(diff);
    }
    Ok(max_diff)
}
