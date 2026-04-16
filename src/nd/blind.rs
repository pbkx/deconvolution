use ndarray::Array2;

use super::convert::{array2_to_dynamic, dynamic_to_array2, kernel2_from_array};
use crate::blind::{BlindMaximumLikelihood, BlindRichardsonLucy};
use crate::{BlindOutput, Result};

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
    let input = array2_to_dynamic(image)?;
    let initial_psf = kernel2_from_array(initial_psf)?;
    let output = crate::blind::richardson_lucy_with(&input, &initial_psf, config)?;
    let image = dynamic_to_array2(&output.image)?;
    Ok(BlindOutput {
        image,
        psf: output.psf,
        report: output.report,
    })
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
    let input = array2_to_dynamic(image)?;
    let initial_psf = kernel2_from_array(initial_psf)?;
    let output = crate::blind::maximum_likelihood_with(&input, &initial_psf, config)?;
    let image = dynamic_to_array2(&output.image)?;
    Ok(BlindOutput {
        image,
        psf: output.psf,
        report: output.report,
    })
}
