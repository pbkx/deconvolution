use ndarray::array;
use ndarray::Array2;

use crate::preprocess::padding::convolve_same_2d;
use crate::{Boundary, Error, Result};

pub fn estimate_nsr(input: &Array2<f32>) -> Result<f32> {
    validate_input(input)?;

    let lowpass_kernel = array![
        [1.0_f32 / 9.0_f32, 1.0_f32 / 9.0_f32, 1.0_f32 / 9.0_f32],
        [1.0_f32 / 9.0_f32, 1.0_f32 / 9.0_f32, 1.0_f32 / 9.0_f32],
        [1.0_f32 / 9.0_f32, 1.0_f32 / 9.0_f32, 1.0_f32 / 9.0_f32]
    ];
    let signal = convolve_same_2d(input, &lowpass_kernel, Boundary::Reflect)?;

    let mut noise_power = 0.0_f32;
    let mut signal_power = 0.0_f32;
    let count = input.len() as f32;

    for ((y, x), value) in input.indexed_iter() {
        let signal_value = signal[[y, x]];
        let noise = *value - signal_value;
        noise_power += noise * noise;
        signal_power += signal_value * signal_value;
    }

    let noise_power = noise_power / count;
    let signal_power = signal_power / count;
    if !noise_power.is_finite() || !signal_power.is_finite() {
        return Err(Error::NonFiniteInput);
    }

    let floor = f32::EPSILON;
    let ratio = (noise_power.max(floor)) / signal_power.max(floor);
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    Ok(ratio)
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
