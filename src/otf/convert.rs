use ndarray::{Array2, Array3};

use crate::core::fft::{
    fft2_forward_real, fft2_inverse_complex, fft3_forward_real, fft3_inverse_complex,
};
use crate::core::plan_cache::PlanCache;
use crate::otf::{Transfer2D, Transfer3D};
use crate::psf::{Kernel2D, Kernel3D};
use crate::{Error, Result};

pub fn psf2otf(psf: &Kernel2D, out_dims: (usize, usize)) -> Result<Transfer2D> {
    let (out_h, out_w) = out_dims;
    if out_h == 0 || out_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let (psf_h, psf_w) = psf.dims();
    if psf_h > out_h || psf_w > out_w {
        return Err(Error::DimensionMismatch);
    }

    let mut padded = Array2::zeros((out_h, out_w));
    for y in 0..psf_h {
        for x in 0..psf_w {
            padded[[y, x]] = psf.as_array()[[y, x]];
        }
    }

    let shift_y = -to_isize(psf_h / 2)?;
    let shift_x = -to_isize(psf_w / 2)?;
    let shifted = circular_shift_2d(&padded, shift_y, shift_x)?;

    let mut cache = PlanCache::new();
    let spectrum = fft2_forward_real(&shifted, &mut cache)?;
    Transfer2D::new(spectrum)
}

pub fn psf2otf_3d(psf: &Kernel3D, out_dims: (usize, usize, usize)) -> Result<Transfer3D> {
    let (out_d, out_h, out_w) = out_dims;
    if out_d == 0 || out_h == 0 || out_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let (psf_d, psf_h, psf_w) = psf.dims();
    if psf_d > out_d || psf_h > out_h || psf_w > out_w {
        return Err(Error::DimensionMismatch);
    }

    let mut padded = Array3::zeros((out_d, out_h, out_w));
    for d in 0..psf_d {
        for y in 0..psf_h {
            for x in 0..psf_w {
                padded[[d, y, x]] = psf.as_array()[[d, y, x]];
            }
        }
    }

    let shift_d = -to_isize(psf_d / 2)?;
    let shift_y = -to_isize(psf_h / 2)?;
    let shift_x = -to_isize(psf_w / 2)?;
    let shifted = circular_shift_3d(&padded, shift_d, shift_y, shift_x)?;

    let mut cache = PlanCache::new();
    let spectrum = fft3_forward_real(&shifted, &mut cache)?;
    Transfer3D::new(spectrum)
}

pub fn otf2psf(otf: &Transfer2D, psf_dims: (usize, usize)) -> Result<Kernel2D> {
    let (psf_h, psf_w) = psf_dims;
    if psf_h == 0 || psf_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let (otf_h, otf_w) = otf.dims();
    if psf_h > otf_h || psf_w > otf_w {
        return Err(Error::DimensionMismatch);
    }

    let mut cache = PlanCache::new();
    let spatial = fft2_inverse_complex(otf.as_array(), &mut cache)?;
    let shifted = circular_shift_2d(&spatial, to_isize(psf_h / 2)?, to_isize(psf_w / 2)?)?;
    let mut cropped = Array2::zeros((psf_h, psf_w));
    for y in 0..psf_h {
        for x in 0..psf_w {
            cropped[[y, x]] = shifted[[y, x]];
        }
    }
    Kernel2D::new(cropped)
}

pub fn otf2psf_3d(otf: &Transfer3D, psf_dims: (usize, usize, usize)) -> Result<Kernel3D> {
    let (psf_d, psf_h, psf_w) = psf_dims;
    if psf_d == 0 || psf_h == 0 || psf_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let (otf_d, otf_h, otf_w) = otf.dims();
    if psf_d > otf_d || psf_h > otf_h || psf_w > otf_w {
        return Err(Error::DimensionMismatch);
    }

    let mut cache = PlanCache::new();
    let spatial = fft3_inverse_complex(otf.as_array(), &mut cache)?;
    let shifted = circular_shift_3d(
        &spatial,
        to_isize(psf_d / 2)?,
        to_isize(psf_h / 2)?,
        to_isize(psf_w / 2)?,
    )?;
    let mut cropped = Array3::zeros((psf_d, psf_h, psf_w));
    for d in 0..psf_d {
        for y in 0..psf_h {
            for x in 0..psf_w {
                cropped[[d, y, x]] = shifted[[d, y, x]];
            }
        }
    }
    Kernel3D::new(cropped)
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
