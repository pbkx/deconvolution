//! Constraints and projections for estimated point-spread functions.
//!
//! Use [`PsfConstraint`] with blind deconvolution builders to keep candidate
//! PSFs nonnegative, normalized, or within a support mask.

use ndarray::Array2;

use crate::{Error, Kernel2D, Result};

#[derive(Debug, Clone, PartialEq)]
/// Projection applied to a candidate point-spread function.
pub enum PsfConstraint {
    /// Clamp negative kernel samples to `0`.
    Nonnegative,
    /// Normalize the kernel so its sum is `1`.
    NormalizeSum,
    /// Zero samples where the `(height, width)` mask is `false`.
    SupportMask(Array2<bool>),
}

impl PsfConstraint {
    /// Apply this constraint to a 2D kernel and return the projected kernel.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when a support mask shape differs
    /// from the kernel, [`Error::InvalidParameter`] for an empty or all-false
    /// support mask, and [`Error::InvalidPsf`] when normalization cannot be
    /// performed.
    pub fn apply(&self, kernel: &Kernel2D) -> Result<Kernel2D> {
        match self {
            Self::Nonnegative => project_nonnegative(kernel),
            Self::NormalizeSum => project_normalize_sum(kernel),
            Self::SupportMask(mask) => project_support_mask(kernel, mask),
        }
    }
}

/// Apply one PSF constraint to a 2D kernel.
///
/// # Errors
///
/// Returns the same errors as [`PsfConstraint::apply`].
pub fn apply_constraint(kernel: &Kernel2D, constraint: &PsfConstraint) -> Result<Kernel2D> {
    constraint.apply(kernel)
}

/// Apply PSF constraints in slice order.
///
/// # Errors
///
/// Returns the first error produced by [`PsfConstraint::apply`].
pub fn apply_constraints(kernel: &Kernel2D, constraints: &[PsfConstraint]) -> Result<Kernel2D> {
    let mut projected = kernel.clone();
    for constraint in constraints {
        projected = constraint.apply(&projected)?;
    }
    Ok(projected)
}

fn project_nonnegative(kernel: &Kernel2D) -> Result<Kernel2D> {
    let (height, width) = kernel.dims();
    let mut projected = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            projected[[y, x]] = kernel.as_array()[[y, x]].max(0.0);
        }
    }
    Kernel2D::new(projected)
}

fn project_normalize_sum(kernel: &Kernel2D) -> Result<Kernel2D> {
    let mut projected = kernel.clone();
    projected.normalize()?;
    Ok(projected)
}

fn project_support_mask(kernel: &Kernel2D, mask: &Array2<bool>) -> Result<Kernel2D> {
    if mask.is_empty() {
        return Err(Error::InvalidParameter);
    }
    if !mask.iter().any(|value| *value) {
        return Err(Error::InvalidParameter);
    }
    if mask.dim() != kernel.dims() {
        return Err(Error::DimensionMismatch);
    }

    let (height, width) = kernel.dims();
    let mut projected = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            if mask[[y, x]] {
                projected[[y, x]] = kernel.as_array()[[y, x]];
            }
        }
    }
    Kernel2D::new(projected)
}
