use crate::{Kernel2D, StopReason};

#[derive(Debug, Clone, PartialEq)]
pub struct BlindOutput<I> {
    pub image: I,
    pub psf: Kernel2D,
    pub report: BlindReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlindReport {
    pub iterations: usize,
    pub stop_reason: StopReason,
    pub objective_history: Vec<f32>,
    pub image_update_history: Vec<f32>,
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
    pub fn new() -> Self {
        Self::default()
    }
}
