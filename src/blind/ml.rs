use image::DynamicImage;
use ndarray::Array2;

use crate::psf::{Kernel2D, PsfConstraint};
use crate::{Error, Result};

use super::rl::richardson_lucy_with;
use super::{BlindOutput, BlindRichardsonLucy};

#[derive(Debug, Clone, PartialEq)]
pub struct BlindMaximumLikelihood {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    filter_epsilon: f32,
    psf_constraints: Vec<PsfConstraint>,
    collect_history: bool,
}

impl Default for BlindMaximumLikelihood {
    fn default() -> Self {
        Self {
            iterations: 30,
            relative_update_tolerance: None,
            filter_epsilon: 1e-6,
            psf_constraints: vec![PsfConstraint::Nonnegative, PsfConstraint::NormalizeSum],
            collect_history: true,
        }
    }
}

impl BlindMaximumLikelihood {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn iterations(mut self, value: usize) -> Self {
        self.iterations = value;
        self
    }

    pub fn relative_update_tolerance(mut self, value: Option<f32>) -> Self {
        self.relative_update_tolerance = value;
        self
    }

    pub fn filter_epsilon(mut self, value: f32) -> Self {
        self.filter_epsilon = value;
        self
    }

    pub fn psf_constraints(mut self, value: Vec<PsfConstraint>) -> Self {
        self.psf_constraints = value;
        self
    }

    pub fn support_mask(mut self, mask: Array2<bool>) -> Self {
        if let Some(index) = self
            .psf_constraints
            .iter()
            .position(|constraint| matches!(constraint, PsfConstraint::NormalizeSum))
        {
            self.psf_constraints
                .insert(index, PsfConstraint::SupportMask(mask));
        } else {
            self.psf_constraints.push(PsfConstraint::SupportMask(mask));
        }
        self
    }

    pub fn collect_history(mut self, value: bool) -> Self {
        self.collect_history = value;
        self
    }
}

pub fn maximum_likelihood(
    image: &DynamicImage,
    initial_psf: &Kernel2D,
) -> Result<BlindOutput<DynamicImage>> {
    maximum_likelihood_with(image, initial_psf, &BlindMaximumLikelihood::new())
}

pub fn maximum_likelihood_with(
    image: &DynamicImage,
    initial_psf: &Kernel2D,
    config: &BlindMaximumLikelihood,
) -> Result<BlindOutput<DynamicImage>> {
    validate_config(config)?;

    let rl_config = BlindRichardsonLucy::new()
        .iterations(config.iterations)
        .relative_update_tolerance(config.relative_update_tolerance)
        .filter_epsilon(config.filter_epsilon)
        .psf_constraints(config.psf_constraints.clone())
        .collect_history(config.collect_history);

    richardson_lucy_with(image, initial_psf, &rl_config)
}

fn validate_config(config: &BlindMaximumLikelihood) -> Result<()> {
    if config.iterations == 0 {
        return Err(Error::InvalidParameter);
    }
    if let Some(tol) = config.relative_update_tolerance {
        if !tol.is_finite() || tol < 0.0 {
            return Err(Error::InvalidParameter);
        }
    }
    if !config.filter_epsilon.is_finite() || config.filter_epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if config.psf_constraints.is_empty() {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}
