use ndarray::Array2;

use super::convert::kernel2_from_array;
use crate::blind::{BlindMaximumLikelihood, BlindOutput, BlindRichardsonLucy};
use crate::Result;

pub fn richardson_lucy(
    image: &Array2<f32>,
    initial_psf: &Array2<f32>,
) -> Result<BlindOutput<Array2<f32>>> {
    richardson_lucy_with(image, initial_psf, &BlindRichardsonLucy::new())
}

pub fn richardson_lucy_with(
    image: &Array2<f32>,
    initial_psf: &Array2<f32>,
    config: &BlindRichardsonLucy,
) -> Result<BlindOutput<Array2<f32>>> {
    let initial_psf = kernel2_from_array(initial_psf)?;
    crate::blind::richardson_lucy_array2_with(image, &initial_psf, config)
}

pub fn maximum_likelihood(
    image: &Array2<f32>,
    initial_psf: &Array2<f32>,
) -> Result<BlindOutput<Array2<f32>>> {
    maximum_likelihood_with(image, initial_psf, &BlindMaximumLikelihood::new())
}

pub fn maximum_likelihood_with(
    image: &Array2<f32>,
    initial_psf: &Array2<f32>,
    config: &BlindMaximumLikelihood,
) -> Result<BlindOutput<Array2<f32>>> {
    let initial_psf = kernel2_from_array(initial_psf)?;
    crate::blind::maximum_likelihood_array2_with(image, &initial_psf, config)
}
