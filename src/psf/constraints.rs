use ndarray::Array2;

use crate::{Error, Kernel2D, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum PsfConstraint {
    Nonnegative,
    NormalizeSum,
    SupportMask(Array2<bool>),
}

impl PsfConstraint {
    pub fn apply(&self, kernel: &Kernel2D) -> Result<Kernel2D> {
        match self {
            Self::Nonnegative => project_nonnegative(kernel),
            Self::NormalizeSum => project_normalize_sum(kernel),
            Self::SupportMask(mask) => project_support_mask(kernel, mask),
        }
    }
}

pub fn apply_constraint(kernel: &Kernel2D, constraint: &PsfConstraint) -> Result<Kernel2D> {
    constraint.apply(kernel)
}

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
