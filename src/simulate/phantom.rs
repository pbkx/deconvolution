use image::{Rgb, RgbImage};
use ndarray::{Array2, Array3};

use crate::{Error, Result};

pub fn checkerboard_2d(
    dims: (usize, usize),
    tile_size: usize,
    low: f32,
    high: f32,
) -> Result<Array2<f32>> {
    let (height, width) = dims;
    if height == 0 || width == 0 || tile_size == 0 {
        return Err(Error::InvalidParameter);
    }
    if !low.is_finite() || !high.is_finite() {
        return Err(Error::InvalidParameter);
    }

    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let parity = (y / tile_size + x / tile_size) % 2;
            output[[y, x]] = if parity == 0 { low } else { high };
        }
    }

    Ok(output)
}

pub fn gaussian_blob_2d(dims: (usize, usize), sigma: f32) -> Result<Array2<f32>> {
    let (height, width) = dims;
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    let cy = center_coordinate(height)?;
    let cx = center_coordinate(width)?;
    let sigma2 = sigma * sigma;
    let mut output = Array2::zeros((height, width));
    let mut max_value = 0.0_f32;

    for y in 0..height {
        let dy = (y as f32) - cy;
        for x in 0..width {
            let dx = (x as f32) - cx;
            let value = (-0.5 * (dx * dx + dy * dy) / sigma2).exp();
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            max_value = max_value.max(value);
            output[[y, x]] = value;
        }
    }

    if max_value <= f32::EPSILON {
        return Err(Error::InvalidParameter);
    }
    for value in &mut output {
        *value /= max_value;
    }

    Ok(output)
}

pub fn rgb_edges_2d(dims: (usize, usize)) -> Result<RgbImage> {
    let (height, width) = dims;
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let width_u32 = u32::try_from(width).map_err(|_| Error::DimensionMismatch)?;
    let height_u32 = u32::try_from(height).map_err(|_| Error::DimensionMismatch)?;

    let mut image = RgbImage::new(width_u32, height_u32);
    let vertical_edge = width / 3;
    let horizontal_edge = height / 2;

    for y in 0..height {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;

            let r = if x < vertical_edge { 24 } else { 220 };
            let g = if y < horizontal_edge { 36 } else { 210 };
            let b = if x > y { 196 } else { 28 };

            image.put_pixel(x_u32, y_u32, Rgb([r, g, b]));
        }
    }

    Ok(image)
}

pub fn phantom_3d(dims: (usize, usize, usize)) -> Result<Array3<f32>> {
    let (depth, height, width) = dims;
    if depth == 0 || height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut volume = Array3::zeros((depth, height, width));
    let mut max_value = 0.0_f32;

    for d in 0..depth {
        let z = normalized_coord(d, depth)?;
        for y in 0..height {
            let v = normalized_coord(y, height)?;
            for x in 0..width {
                let u = normalized_coord(x, width)?;

                let ellipsoid_main =
                    (((u * 0.95).powi(2) + (v * 1.10).powi(2) + (z * 0.85).powi(2)) <= 1.0) as u8;
                let ellipsoid_offset = ((((u + 0.26) * 1.4).powi(2)
                    + ((v - 0.10) * 1.2).powi(2)
                    + ((z + 0.12) * 1.8).powi(2))
                    <= 1.0) as u8;

                let core = if ellipsoid_main == 1 { 0.9 } else { 0.0 };
                let cavity = if ellipsoid_offset == 1 { 0.35 } else { 0.0 };
                let lobe = 0.55
                    * (-6.0
                        * ((u - 0.35) * (u - 0.35)
                            + (v + 0.28) * (v + 0.28)
                            + (z - 0.18) * (z - 0.18)))
                        .exp();
                let ridge = 0.20 * (-20.0 * (v + 0.25).powi(2)).exp() * (-8.0 * z.powi(2)).exp();

                let value = (core - cavity + lobe + ridge).max(0.0);
                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                max_value = max_value.max(value);
                volume[[d, y, x]] = value;
            }
        }
    }

    if max_value <= f32::EPSILON {
        return Err(Error::InvalidParameter);
    }
    for value in &mut volume {
        *value /= max_value;
    }

    Ok(volume)
}

fn center_coordinate(length: usize) -> Result<f32> {
    if length == 0 {
        return Err(Error::InvalidParameter);
    }
    Ok((length as f32 - 1.0) * 0.5)
}

fn normalized_coord(index: usize, length: usize) -> Result<f32> {
    if length == 0 {
        return Err(Error::InvalidParameter);
    }
    if length == 1 {
        return Ok(0.0);
    }
    let denom = (length - 1) as f32;
    Ok(2.0 * (index as f32 / denom) - 1.0)
}
