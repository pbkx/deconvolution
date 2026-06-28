use crate::{Kernel2D, StopReason};

#[derive(Debug, Clone, PartialEq)]
/// Output from a blind deconvolution run.
pub struct BlindOutput<I> {
    /// Restored image or array.
    pub image: I,
    /// Estimated 2D point-spread function.
    pub psf: Kernel2D,
    /// Iteration diagnostics for image and PSF updates.
    pub report: BlindReport,
}

#[derive(Debug, Clone, PartialEq)]
/// Iteration diagnostics returned by blind solvers.
pub struct BlindReport {
    /// Number of outer blind-deconvolution iterations completed.
    pub iterations: usize,
    /// Condition that stopped the outer loop.
    pub stop_reason: StopReason,
    /// Objective values recorded per iteration, empty when history collection is disabled.
    pub objective_history: Vec<f32>,
    /// Relative image-update norms recorded per iteration.
    pub image_update_history: Vec<f32>,
    /// Relative PSF-update norms recorded per iteration.
    pub psf_update_history: Vec<f32>,
}

impl Default for BlindReport {
    fn default() -> Self {
        Self {
            iterations: 0,
            stop_reason: StopReason::MaxIterations,
            objective_history: Vec::new(),
            image_update_history: Vec::new(),
            psf_update_history: Vec::new(),
        }
    }
}

impl BlindReport {
    /// Create an empty blind-solver report.
    pub fn new() -> Self {
        Self::default()
    }
}
