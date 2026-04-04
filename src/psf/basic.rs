use ndarray::{Array2, Array3};

use crate::{Error, Kernel2D, Kernel3D, Result};

pub fn delta2d(dims: (usize, usize)) -> Result<Kernel2D> {
    let (height, width) = dims;
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut kernel = Array2::zeros((height, width));
    kernel[[height / 2, width / 2]] = 1.0;
    Kernel2D::new(kernel)
}

pub fn delta3d(dims: (usize, usize, usize)) -> Result<Kernel3D> {
    let (depth, height, width) = dims;
    if depth == 0 || height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut kernel = Array3::zeros((depth, height, width));
    kernel[[depth / 2, height / 2, width / 2]] = 1.0;
    Kernel3D::new(kernel)
}

pub fn gaussian2d(dims: (usize, usize), sigma: f32) -> Result<Kernel2D> {
    let (height, width) = dims;
    validate_dims_2d(height, width)?;
    validate_sigma(sigma)?;

    let cy = center_coordinate(height)?;
    let cx = center_coordinate(width)?;
    let sigma2 = sigma * sigma;

    let mut kernel = Array2::zeros((height, width));
    for y in 0..height {
        let dy = (y as f32) - cy;
        for x in 0..width {
            let dx = (x as f32) - cx;
            let exponent = -0.5 * (dx * dx + dy * dy) / sigma2;
            let value = exponent.exp();
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            kernel[[y, x]] = value;
        }
    }

    normalize_kernel2d(kernel)
}

pub fn gaussian3d(dims: (usize, usize, usize), sigma: f32) -> Result<Kernel3D> {
    let (depth, height, width) = dims;
    validate_dims_3d(depth, height, width)?;
    validate_sigma(sigma)?;

    let cd = center_coordinate(depth)?;
    let cy = center_coordinate(height)?;
    let cx = center_coordinate(width)?;
    let sigma2 = sigma * sigma;

    let mut kernel = Array3::zeros((depth, height, width));
    for d in 0..depth {
        let dd = (d as f32) - cd;
        for y in 0..height {
            let dy = (y as f32) - cy;
            for x in 0..width {
                let dx = (x as f32) - cx;
                let exponent = -0.5 * (dx * dx + dy * dy + dd * dd) / sigma2;
                let value = exponent.exp();
                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                kernel[[d, y, x]] = value;
            }
        }
    }

    normalize_kernel3d(kernel)
}

pub fn motion_linear(length: f32, angle_deg: f32) -> Result<Kernel2D> {
    if !length.is_finite() || length <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !angle_deg.is_finite() {
        return Err(Error::InvalidParameter);
    }

    let side = odd_size_from_length(length)?;
    let center = center_coordinate(side)?;
    let angle_rad = angle_deg.to_radians();
    let dir_x = angle_rad.cos();
    let dir_y = angle_rad.sin();
    let half = (length - 1.0).max(0.0) * 0.5;
    let start_x = -half * dir_x;
    let start_y = -half * dir_y;
    let end_x = half * dir_x;
    let end_y = half * dir_y;

    let mut kernel = Array2::zeros((side, side));
    for y in 0..side {
        let py = (y as f32) - center;
        for x in 0..side {
            let px = (x as f32) - center;
            let distance = point_segment_distance(px, py, start_x, start_y, end_x, end_y);
            let value = (1.0 - distance).max(0.0);
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            kernel[[y, x]] = value;
        }
    }

    if kernel.iter().all(|value| *value <= 0.0) {
        kernel[[side / 2, side / 2]] = 1.0;
    }

    normalize_kernel2d(kernel)
}

pub fn disk(radius: f32) -> Result<Kernel2D> {
    pillbox(radius)
}

pub fn pillbox(radius: f32) -> Result<Kernel2D> {
    let kernel = binary_disk(radius)?;
    normalize_kernel2d(kernel)
}

pub fn defocus(radius: f32) -> Result<Kernel2D> {
    validate_radius(radius)?;
    let side = odd_size_from_radius(radius)?;
    let center = center_coordinate(side)?;
    let samples = 8_usize;
    let sample_scale = samples as f32;
    let mut kernel = Array2::zeros((side, side));

    for y in 0..side {
        for x in 0..side {
            let mut inside = 0_usize;
            for sy in 0..samples {
                let oy = ((sy as f32) + 0.5) / sample_scale - 0.5;
                let py = (y as f32) - center + oy;
                for sx in 0..samples {
                    let ox = ((sx as f32) + 0.5) / sample_scale - 0.5;
                    let px = (x as f32) - center + ox;
                    if px * px + py * py <= radius * radius {
                        inside += 1;
                    }
                }
            }
            kernel[[y, x]] = (inside as f32) / ((samples * samples) as f32);
        }
    }

    if kernel.iter().all(|value| *value <= 0.0) {
        kernel[[side / 2, side / 2]] = 1.0;
    }

    normalize_kernel2d(kernel)
}

pub fn box2d(dims: (usize, usize)) -> Result<Kernel2D> {
    let (height, width) = dims;
    validate_dims_2d(height, width)?;
    let kernel = Array2::ones((height, width));
    normalize_kernel2d(kernel)
}

pub fn box3d(dims: (usize, usize, usize)) -> Result<Kernel3D> {
    let (depth, height, width) = dims;
    validate_dims_3d(depth, height, width)?;
    let kernel = Array3::ones((depth, height, width));
    normalize_kernel3d(kernel)
}

pub fn oriented_gaussian(
    dims: (usize, usize),
    sigma_major: f32,
    sigma_minor: f32,
    angle_deg: f32,
) -> Result<Kernel2D> {
    let (height, width) = dims;
    validate_dims_2d(height, width)?;
    validate_sigma(sigma_major)?;
    validate_sigma(sigma_minor)?;
    if !angle_deg.is_finite() {
        return Err(Error::InvalidParameter);
    }

    let cy = center_coordinate(height)?;
    let cx = center_coordinate(width)?;
    let theta = angle_deg.to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let major2 = sigma_major * sigma_major;
    let minor2 = sigma_minor * sigma_minor;

    let mut kernel = Array2::zeros((height, width));
    for y in 0..height {
        let dy = (y as f32) - cy;
        for x in 0..width {
            let dx = (x as f32) - cx;
            let xr = cos_t * dx + sin_t * dy;
            let yr = -sin_t * dx + cos_t * dy;
            let exponent = -0.5 * (xr * xr / major2 + yr * yr / minor2);
            let value = exponent.exp();
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            kernel[[y, x]] = value;
        }
    }

    normalize_kernel2d(kernel)
}

fn normalize_kernel2d(kernel: Array2<f32>) -> Result<Kernel2D> {
    let mut kernel = Kernel2D::new(kernel)?;
    kernel.normalize()?;
    Ok(kernel)
}

fn normalize_kernel3d(kernel: Array3<f32>) -> Result<Kernel3D> {
    let mut kernel = Kernel3D::new(kernel)?;
    kernel.normalize()?;
    Ok(kernel)
}

fn binary_disk(radius: f32) -> Result<Array2<f32>> {
    validate_radius(radius)?;
    let side = odd_size_from_radius(radius)?;
    let center = center_coordinate(side)?;
    let radius2 = radius * radius;

    let mut kernel = Array2::zeros((side, side));
    for y in 0..side {
        let dy = (y as f32) - center;
        for x in 0..side {
            let dx = (x as f32) - center;
            if dx * dx + dy * dy <= radius2 {
                kernel[[y, x]] = 1.0;
            }
        }
    }

    if kernel.iter().all(|value| *value <= 0.0) {
        kernel[[side / 2, side / 2]] = 1.0;
    }

    Ok(kernel)
}

fn validate_dims_2d(height: usize, width: usize) -> Result<()> {
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_dims_3d(depth: usize, height: usize, width: usize) -> Result<()> {
    if depth == 0 || height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_sigma(sigma: f32) -> Result<()> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_radius(radius: f32) -> Result<()> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn odd_size_from_length(length: f32) -> Result<usize> {
    let ceil_len = length.ceil();
    if !ceil_len.is_finite() || ceil_len <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    let base = ceil_len as usize;
    odd_size(base)
}

fn odd_size_from_radius(radius: f32) -> Result<usize> {
    let ceil_radius = radius.ceil();
    if !ceil_radius.is_finite() || ceil_radius <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    let r = ceil_radius as usize;
    let diameter = r
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or(Error::InvalidParameter)?;
    odd_size(diameter)
}

fn odd_size(value: usize) -> Result<usize> {
    if value == 0 {
        return Err(Error::InvalidParameter);
    }
    if value % 2 == 1 {
        return Ok(value);
    }
    value.checked_add(1).ok_or(Error::InvalidParameter)
}

fn center_coordinate(length: usize) -> Result<f32> {
    if length == 0 {
        return Err(Error::InvalidParameter);
    }
    Ok((length as f32 - 1.0) * 0.5)
}

fn point_segment_distance(px: f32, py: f32, sx0: f32, sy0: f32, sx1: f32, sy1: f32) -> f32 {
    let vx = sx1 - sx0;
    let vy = sy1 - sy0;
    let wx = px - sx0;
    let wy = py - sy0;
    let vv = vx * vx + vy * vy;
    if vv <= f32::EPSILON {
        return ((px - sx0) * (px - sx0) + (py - sy0) * (py - sy0)).sqrt();
    }

    let t = ((wx * vx + wy * vy) / vv).clamp(0.0, 1.0);
    let dx = px - (sx0 + t * vx);
    let dy = py - (sy0 + t * vy);
    (dx * dx + dy * dy).sqrt()
}
