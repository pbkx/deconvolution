use std::f32::consts::FRAC_PI_2;

use ndarray::Array2;

use crate::{Error, Result};

pub fn window_edges(input: &Array2<f32>, border: usize) -> Result<Array2<f32>> {
    validate_input(input)?;

    if border == 0 {
        return Ok(input.clone());
    }

    let (height, width) = input.dim();
    let max_border = (height / 2).min(width / 2);
    if max_border == 0 {
        return Ok(input.clone());
    }

    let edge = border.min(max_border);
    let edge_mean = edge_mean(input, edge)?;
    let wy = edge_weights(height, edge)?;
    let wx = edge_weights(width, edge)?;

    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let w = wy[y] * wx[x];
            let value = edge_mean + w * (input[[y, x]] - edge_mean);
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }

    Ok(output)
}

pub fn apodize(input: &Array2<f32>) -> Result<Array2<f32>> {
    validate_input(input)?;

    let (height, width) = input.dim();
    let min_dim = height.min(width);
    let max_border = min_dim / 2;
    if max_border == 0 {
        return Ok(input.clone());
    }

    let border = (min_dim / 16).max(1).min(max_border);
    window_edges(input, border)
}

fn edge_weights(length: usize, border: usize) -> Result<Vec<f32>> {
    if length == 0 {
        return Err(Error::InvalidParameter);
    }
    if border == 0 {
        return Ok(vec![1.0; length]);
    }

    let mut weights = vec![1.0_f32; length];
    let border_f = border as f32;
    let last = length - 1;

    for (index, weight) in weights.iter_mut().enumerate() {
        let distance = index.min(last - index);
        if distance >= border {
            *weight = 1.0;
            continue;
        }

        let phase = (distance as f32) / border_f;
        let value = (phase * FRAC_PI_2).sin().powi(2);
        *weight = value;
    }

    Ok(weights)
}

fn edge_mean(input: &Array2<f32>, border: usize) -> Result<f32> {
    let (height, width) = input.dim();
    if border == 0 || border * 2 > height || border * 2 > width {
        return mean(input);
    }

    let mut sum = 0.0_f32;
    let mut count = 0_usize;

    for y in 0..height {
        for x in 0..width {
            let on_edge = y < border || y >= height - border || x < border || x >= width - border;
            if on_edge {
                sum += input[[y, x]];
                count += 1;
            }
        }
    }

    if count == 0 {
        return mean(input);
    }

    let denom = count as f32;
    let value = sum / denom;
    if !value.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(value)
}

fn mean(input: &Array2<f32>) -> Result<f32> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    let sum: f32 = input.iter().copied().sum();
    let denom = input.len() as f32;
    let value = sum / denom;
    if !value.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(value)
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
