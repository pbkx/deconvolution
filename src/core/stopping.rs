use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    MaxIterations,
    RelativeUpdate,
    ObjectivePlateau,
    Divergence,
}

#[derive(Debug, Clone)]
pub(crate) struct StopCriteria {
    pub(crate) max_iterations: usize,
    pub(crate) relative_update_tol: Option<f32>,
    pub(crate) objective_plateau_window: usize,
    pub(crate) objective_plateau_tol: f32,
    pub(crate) divergence_factor: f32,
}

impl Default for StopCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            relative_update_tol: Some(1e-4),
            objective_plateau_window: 4,
            objective_plateau_tol: 1e-6,
            divergence_factor: 1.5,
        }
    }
}

pub(crate) fn check_stop(
    criteria: &StopCriteria,
    iteration: usize,
    relative_update: Option<f32>,
    objective_history: &[f32],
) -> Result<Option<StopReason>> {
    validate_criteria(criteria)?;
    validate_objective_history(objective_history)?;

    if is_diverging(criteria, objective_history) {
        return Ok(Some(StopReason::Divergence));
    }

    if let Some(update) = relative_update {
        if !update.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        if let Some(tol) = criteria.relative_update_tol {
            if update <= tol {
                return Ok(Some(StopReason::RelativeUpdate));
            }
        }
    }

    if is_plateau(criteria, objective_history)? {
        return Ok(Some(StopReason::ObjectivePlateau));
    }

    if iteration >= criteria.max_iterations {
        return Ok(Some(StopReason::MaxIterations));
    }

    Ok(None)
}

fn validate_criteria(criteria: &StopCriteria) -> Result<()> {
    if criteria.max_iterations == 0 {
        return Err(Error::InvalidParameter);
    }
    if let Some(tol) = criteria.relative_update_tol {
        if !tol.is_finite() || tol < 0.0 {
            return Err(Error::InvalidParameter);
        }
    }
    if !criteria.objective_plateau_tol.is_finite() || criteria.objective_plateau_tol < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !criteria.divergence_factor.is_finite() || criteria.divergence_factor <= 1.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_objective_history(history: &[f32]) -> Result<()> {
    if history.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn is_plateau(criteria: &StopCriteria, history: &[f32]) -> Result<bool> {
    let window = criteria.objective_plateau_window;
    if window == 0 || history.len() < window + 1 {
        return Ok(false);
    }

    let recent = &history[history.len() - (window + 1)..];
    let (mut min_value, mut max_value) = (f32::INFINITY, f32::NEG_INFINITY);
    for value in recent {
        if *value < min_value {
            min_value = *value;
        }
        if *value > max_value {
            max_value = *value;
        }
    }
    if !min_value.is_finite() || !max_value.is_finite() {
        return Err(Error::NonFiniteInput);
    }

    Ok((max_value - min_value).abs() <= criteria.objective_plateau_tol)
}

fn is_diverging(criteria: &StopCriteria, history: &[f32]) -> bool {
    if history.len() < 2 {
        return false;
    }

    let current = history[history.len() - 1];
    let previous_best = history[..history.len() - 1]
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);

    if previous_best == 0.0 {
        return current > criteria.objective_plateau_tol;
    }

    current > previous_best * criteria.divergence_factor
}

#[cfg(test)]
mod tests {
    use super::{check_stop, StopCriteria, StopReason};

    #[test]
    fn max_iterations_trigger_is_detected() {
        let criteria = StopCriteria {
            max_iterations: 3,
            ..StopCriteria::default()
        };
        let reason = check_stop(&criteria, 3, None, &[3.0_f32, 2.0_f32, 1.0_f32]).unwrap();
        assert_eq!(reason, Some(StopReason::MaxIterations));
    }

    #[test]
    fn relative_update_trigger_is_detected() {
        let criteria = StopCriteria {
            relative_update_tol: Some(1e-3),
            ..StopCriteria::default()
        };
        let reason = check_stop(&criteria, 1, Some(1e-4), &[10.0_f32, 9.5_f32]).unwrap();
        assert_eq!(reason, Some(StopReason::RelativeUpdate));
    }

    #[test]
    fn objective_plateau_trigger_is_detected() {
        let criteria = StopCriteria {
            objective_plateau_window: 3,
            objective_plateau_tol: 1e-5,
            ..StopCriteria::default()
        };
        let reason = check_stop(
            &criteria,
            2,
            None,
            &[1.0_f32, 1.0_f32 + 1e-6, 1.0_f32 + 2e-6, 1.0_f32 + 1e-6],
        )
        .unwrap();
        assert_eq!(reason, Some(StopReason::ObjectivePlateau));
    }

    #[test]
    fn divergence_trigger_is_detected() {
        let criteria = StopCriteria {
            divergence_factor: 1.2,
            ..StopCriteria::default()
        };
        let reason = check_stop(&criteria, 1, None, &[1.0_f32, 1.3_f32]).unwrap();
        assert_eq!(reason, Some(StopReason::Divergence));
    }
}
