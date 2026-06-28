use image::DynamicImage;
use ndarray::Array2;

use crate::psf::{Kernel2D, PsfConstraint};
use crate::{Error, Result};

use super::BlindOutput;
use super::rl::{BlindPoissonEm, restore_poisson_em_array2, restore_poisson_em_dynamic};

#[derive(Debug, Clone, PartialEq)]
/// Configuration for blind maximum-likelihood deconvolution.
///
/// This shares the same Poisson EM restoration core as blind Richardson-Lucy.
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
    /// Create a blind ML config with nonnegative normalized PSF constraints.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum alternating image/PSF iteration count.
    pub fn iterations(mut self, value: usize) -> Self {
        self.iterations = value;
        self
    }

    /// Set the relative update stopping tolerance.
    ///
    /// `None` disables this stopping criterion.
    pub fn relative_update_tolerance(mut self, value: Option<f32>) -> Self {
        self.relative_update_tolerance = value;
        self
    }

    /// Set the positive denominator floor used in multiplicative updates.
    pub fn filter_epsilon(mut self, value: f32) -> Self {
        self.filter_epsilon = value;
        self
    }

    /// Replace the PSF constraints applied after each PSF update.
    pub fn psf_constraints(mut self, value: Vec<PsfConstraint>) -> Self {
        self.psf_constraints = value;
        self
    }

    /// Add a 2D support mask before normalization.
    ///
    /// The mask shape must match the PSF shape used by the solver.
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

    /// Enable or disable objective and update histories in [`super::BlindReport`].
    pub fn collect_history(mut self, value: bool) -> Self {
        self.collect_history = value;
        self
    }
}

/// Estimate both a restored image and PSF with blind maximum likelihood.
///
/// # Errors
///
/// Returns an error for invalid image data, invalid initial PSFs, invalid
/// constraints or support masks, invalid solver parameters, or non-finite updates.
pub fn maximum_likelihood(
    image: &DynamicImage,
    initial_psf: &Kernel2D,
) -> Result<BlindOutput<DynamicImage>> {
    maximum_likelihood_with(image, initial_psf, &BlindMaximumLikelihood::new())
}

/// Estimate both a restored image and PSF with explicit blind ML settings.
///
/// # Errors
///
/// Returns an error for invalid image data, invalid initial PSFs, invalid
/// constraints or support masks, invalid solver parameters, or non-finite updates.
pub fn maximum_likelihood_with(
    image: &DynamicImage,
    initial_psf: &Kernel2D,
    config: &BlindMaximumLikelihood,
) -> Result<BlindOutput<DynamicImage>> {
    validate_config(config)?;
    let poisson = blind_poisson_em(config);
    restore_poisson_em_dynamic(image, initial_psf, &poisson)
}

pub(crate) fn maximum_likelihood_array2_with(
    image: &Array2<f32>,
    initial_psf: &Kernel2D,
    config: &BlindMaximumLikelihood,
) -> Result<BlindOutput<Array2<f32>>> {
    validate_config(config)?;
    let poisson = blind_poisson_em(config);
    restore_poisson_em_array2(image, initial_psf, &poisson)
}

fn blind_poisson_em(config: &BlindMaximumLikelihood) -> BlindPoissonEm {
    BlindPoissonEm {
        iterations: config.iterations,
        relative_update_tolerance: config.relative_update_tolerance,
        filter_epsilon: config.filter_epsilon,
        psf_constraints: config.psf_constraints.clone(),
        collect_history: config.collect_history,
    }
}

fn validate_config(config: &BlindMaximumLikelihood) -> Result<()> {
    if config.iterations == 0 {
        return Err(Error::InvalidParameter);
    }
    if let Some(tol) = config.relative_update_tolerance
        && (!tol.is_finite() || tol < 0.0)
    {
        return Err(Error::InvalidParameter);
    }
    if !config.filter_epsilon.is_finite() || config.filter_epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if config.psf_constraints.is_empty() {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}
