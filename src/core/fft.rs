use ndarray::{Array2, Array3};
use num_complex::Complex32;

use super::plan_cache::{FftDirection, PlanCache};
use super::util::{checked_volume_2d, checked_volume_3d};
use crate::{Error, Result};

pub(crate) fn fft2_forward_real(
    input: &Array2<f32>,
    cache: &mut PlanCache,
) -> Result<Array2<Complex32>> {
    validate_real_2d(input)?;
    let mut spectrum = Array2::from_shape_fn(input.raw_dim(), |(row, col)| {
        Complex32::new(input[[row, col]], 0.0)
    });
    fft2_in_place(&mut spectrum, FftDirection::Forward, cache)?;
    Ok(spectrum)
}

pub(crate) fn fft2_inverse_complex(
    input: &Array2<Complex32>,
    cache: &mut PlanCache,
) -> Result<Array2<f32>> {
    validate_complex_2d(input)?;
    let mut spatial = input.as_standard_layout().to_owned();
    fft2_in_place(&mut spatial, FftDirection::Inverse, cache)?;

    let (height, width) = spatial.dim();
    let norm = 1.0_f32 / (checked_volume_2d(height, width)? as f32);
    let mut output = Array2::zeros((height, width));
    for row in 0..height {
        for col in 0..width {
            let value = spatial[[row, col]].re * norm;
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[row, col]] = value;
        }
    }
    Ok(output)
}

pub(crate) fn fft3_forward_real(
    input: &Array3<f32>,
    cache: &mut PlanCache,
) -> Result<Array3<Complex32>> {
    validate_real_3d(input)?;
    let mut spectrum = Array3::from_shape_fn(input.raw_dim(), |(depth, row, col)| {
        Complex32::new(input[[depth, row, col]], 0.0)
    });
    fft3_in_place(&mut spectrum, FftDirection::Forward, cache)?;
    Ok(spectrum)
}

pub(crate) fn fft3_inverse_complex(
    input: &Array3<Complex32>,
    cache: &mut PlanCache,
) -> Result<Array3<f32>> {
    validate_complex_3d(input)?;
    let mut spatial = input.as_standard_layout().to_owned();
    fft3_in_place(&mut spatial, FftDirection::Inverse, cache)?;

    let (depth, height, width) = spatial.dim();
    let norm = 1.0_f32 / (checked_volume_3d(depth, height, width)? as f32);
    let mut output = Array3::zeros((depth, height, width));
    for d in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let value = spatial[[d, y, x]].re * norm;
                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                output[[d, y, x]] = value;
            }
        }
    }
    Ok(output)
}

fn fft2_in_place(
    data: &mut Array2<Complex32>,
    direction: FftDirection,
    cache: &mut PlanCache,
) -> Result<()> {
    let (height, width) = data.dim();
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut row = vec![Complex32::new(0.0, 0.0); width];
    for y in 0..height {
        for x in 0..width {
            row[x] = data[[y, x]];
        }
        cache.process(width, direction, &mut row)?;
        for x in 0..width {
            data[[y, x]] = row[x];
        }
    }

    let mut col = vec![Complex32::new(0.0, 0.0); height];
    for x in 0..width {
        for y in 0..height {
            col[y] = data[[y, x]];
        }
        cache.process(height, direction, &mut col)?;
        for y in 0..height {
            data[[y, x]] = col[y];
        }
    }

    Ok(())
}

fn fft3_in_place(
    data: &mut Array3<Complex32>,
    direction: FftDirection,
    cache: &mut PlanCache,
) -> Result<()> {
    let (depth, height, width) = data.dim();
    if depth == 0 || height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut line_width = vec![Complex32::new(0.0, 0.0); width];
    for d in 0..depth {
        for y in 0..height {
            for x in 0..width {
                line_width[x] = data[[d, y, x]];
            }
            cache.process(width, direction, &mut line_width)?;
            for x in 0..width {
                data[[d, y, x]] = line_width[x];
            }
        }
    }

    let mut line_height = vec![Complex32::new(0.0, 0.0); height];
    for d in 0..depth {
        for x in 0..width {
            for y in 0..height {
                line_height[y] = data[[d, y, x]];
            }
            cache.process(height, direction, &mut line_height)?;
            for y in 0..height {
                data[[d, y, x]] = line_height[y];
            }
        }
    }

    let mut line_depth = vec![Complex32::new(0.0, 0.0); depth];
    for y in 0..height {
        for x in 0..width {
            for d in 0..depth {
                line_depth[d] = data[[d, y, x]];
            }
            cache.process(depth, direction, &mut line_depth)?;
            for d in 0..depth {
                data[[d, y, x]] = line_depth[d];
            }
        }
    }

    Ok(())
}

fn validate_real_2d(input: &Array2<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::InvalidParameter);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn validate_real_3d(input: &Array3<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::InvalidParameter);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn validate_complex_2d(input: &Array2<Complex32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::InvalidParameter);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn validate_complex_3d(input: &Array3<Complex32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::InvalidParameter);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use num_complex::Complex32;

    use super::{fft2_forward_real, fft2_inverse_complex, fft3_forward_real, fft3_inverse_complex};
    use crate::core::plan_cache::PlanCache;

    #[test]
    fn fft2_roundtrip_is_within_tolerance() {
        let input = array![
            [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32],
            [4.0_f32, 5.0_f32, 6.0_f32, 7.0_f32],
            [8.0_f32, 9.0_f32, 10.0_f32, 11.0_f32]
        ];

        let mut cache = PlanCache::new();
        let spectrum = fft2_forward_real(&input, &mut cache).unwrap();
        let restored = fft2_inverse_complex(&spectrum, &mut cache).unwrap();

        for ((row, col), original) in input.indexed_iter() {
            let error = (restored[[row, col]] - original).abs();
            assert!(error < 1e-4);
        }
    }

    #[test]
    fn fft3_roundtrip_is_within_tolerance() {
        let input = array![
            [[0.0_f32, 1.0_f32], [2.0_f32, 3.0_f32], [4.0_f32, 5.0_f32]],
            [[6.0_f32, 7.0_f32], [8.0_f32, 9.0_f32], [10.0_f32, 11.0_f32]]
        ];

        let mut cache = PlanCache::new();
        let spectrum = fft3_forward_real(&input, &mut cache).unwrap();
        let restored = fft3_inverse_complex(&spectrum, &mut cache).unwrap();

        for ((d, y, x), original) in input.indexed_iter() {
            let error = (restored[[d, y, x]] - original).abs();
            assert!(error < 1e-4);
        }
    }

    #[test]
    fn delta_transforms_to_constant_spectrum() {
        let input = array![
            [1.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32]
        ];

        let mut cache = PlanCache::new();
        let spectrum = fft2_forward_real(&input, &mut cache).unwrap();
        for value in &spectrum {
            let expected = Complex32::new(1.0, 0.0);
            assert!((*value - expected).norm() < 1e-5);
        }
    }
}
