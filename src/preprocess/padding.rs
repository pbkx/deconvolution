use ndarray::Array2;

use crate::{Boundary, Error, Result};

use super::boundary::map_index;

pub(crate) fn same_padding_2d(
    kernel_dims: (usize, usize),
) -> Result<((usize, usize), (usize, usize))> {
    let (kernel_h, kernel_w) = kernel_dims;
    if kernel_h == 0 || kernel_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let pad_top = kernel_h / 2;
    let pad_bottom = kernel_h - 1 - pad_top;
    let pad_left = kernel_w / 2;
    let pad_right = kernel_w - 1 - pad_left;

    Ok(((pad_top, pad_bottom), (pad_left, pad_right)))
}

pub(crate) fn pad_2d(
    input: &Array2<f32>,
    pad_y: (usize, usize),
    pad_x: (usize, usize),
    boundary: Boundary,
) -> Result<Array2<f32>> {
    validate_input(input)?;

    let (height, width) = input.dim();
    let out_h = height
        .checked_add(pad_y.0)
        .and_then(|value| value.checked_add(pad_y.1))
        .ok_or(Error::InvalidParameter)?;
    let out_w = width
        .checked_add(pad_x.0)
        .and_then(|value| value.checked_add(pad_x.1))
        .ok_or(Error::InvalidParameter)?;

    if out_h == 0 || out_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut output = Array2::zeros((out_h, out_w));
    let top = to_i64(pad_y.0)?;
    let left = to_i64(pad_x.0)?;

    for y in 0..out_h {
        let source_y = to_i64(y)? - top;
        let mapped_y = map_index(source_y, height, boundary)?;
        for x in 0..out_w {
            let source_x = to_i64(x)? - left;
            let mapped_x = map_index(source_x, width, boundary)?;

            output[[y, x]] = match (mapped_y, mapped_x) {
                (Some(my), Some(mx)) => input[[my, mx]],
                _ => 0.0,
            };
        }
    }

    Ok(output)
}

pub(crate) fn convolve_same_2d(
    input: &Array2<f32>,
    kernel: &Array2<f32>,
    boundary: Boundary,
) -> Result<Array2<f32>> {
    validate_input(input)?;
    validate_input(kernel)?;

    let (kernel_h, kernel_w) = kernel.dim();
    let (pad_y, pad_x) = same_padding_2d((kernel_h, kernel_w))?;
    let padded = pad_2d(input, pad_y, pad_x, boundary)?;
    convolve_valid_2d(&padded, kernel)
}

fn convolve_valid_2d(input: &Array2<f32>, kernel: &Array2<f32>) -> Result<Array2<f32>> {
    validate_input(input)?;
    validate_input(kernel)?;

    let (height, width) = input.dim();
    let (kernel_h, kernel_w) = kernel.dim();

    if kernel_h > height || kernel_w > width {
        return Err(Error::DimensionMismatch);
    }

    let out_h = height - kernel_h + 1;
    let out_w = width - kernel_w + 1;
    let mut output = Array2::zeros((out_h, out_w));

    for y in 0..out_h {
        for x in 0..out_w {
            let mut sum = 0.0_f32;
            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    sum += input[[y + ky, x + kx]] * kernel[[ky, kx]];
                }
            }
            if !sum.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = sum;
        }
    }

    Ok(output)
}

fn validate_input(input: &Array2<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn to_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::DimensionMismatch)
}
