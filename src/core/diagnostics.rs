use crate::{Error, Result};

use super::stopping::StopReason;

#[derive(Debug, Clone, PartialEq)]
pub struct SolveReport {
    pub iterations: usize,
    pub stop_reason: StopReason,
    pub objective_history: Vec<f32>,
    pub residual_history: Vec<f32>,
    pub estimated_nsr: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Diagnostics {
    objective_history: Vec<f32>,
    residual_history: Vec<f32>,
}

impl Diagnostics {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn record(&mut self, objective: f32, residual: f32) -> Result<()> {
        if !objective.is_finite() || !residual.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        self.objective_history.push(objective);
        self.residual_history.push(residual);
        Ok(())
    }

    pub(crate) fn objective_history(&self) -> &[f32] {
        &self.objective_history
    }

    pub(crate) fn residual_history(&self) -> &[f32] {
        &self.residual_history
    }

    pub(crate) fn finish(self, stop_reason: StopReason) -> SolveReport {
        let iterations = self.objective_history.len();
        SolveReport {
            iterations,
            stop_reason,
            objective_history: self.objective_history,
            residual_history: self.residual_history,
            estimated_nsr: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Diagnostics;
    use crate::core::stopping::StopReason;

    #[test]
    fn diagnostics_preserve_recording_order() {
        let mut diagnostics = Diagnostics::new();
        diagnostics.record(10.0_f32, 3.0_f32).unwrap();
        diagnostics.record(8.0_f32, 2.0_f32).unwrap();
        diagnostics.record(7.0_f32, 1.0_f32).unwrap();

        assert_eq!(
            diagnostics.objective_history(),
            [10.0_f32, 8.0_f32, 7.0_f32]
        );
        assert_eq!(diagnostics.residual_history(), [3.0_f32, 2.0_f32, 1.0_f32]);

        let report = diagnostics.finish(StopReason::RelativeUpdate);
        assert_eq!(report.iterations, 3);
        assert_eq!(report.stop_reason, StopReason::RelativeUpdate);
        assert_eq!(report.objective_history, vec![10.0_f32, 8.0_f32, 7.0_f32]);
        assert_eq!(report.residual_history, vec![3.0_f32, 2.0_f32, 1.0_f32]);
        assert_eq!(report.estimated_nsr, None);
    }
}
