use std::collections::HashMap;
use std::sync::Arc;

use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum FftDirection {
    Forward,
    Inverse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlanKey {
    len: usize,
    direction: FftDirection,
}

pub(crate) struct PlanCache {
    planner: FftPlanner<f32>,
    plans: HashMap<PlanKey, Arc<dyn Fft<f32>>>,
}

impl PlanCache {
    pub(crate) fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            plans: HashMap::new(),
        }
    }

    pub(crate) fn plan(
        &mut self,
        len: usize,
        direction: FftDirection,
    ) -> Result<Arc<dyn Fft<f32>>> {
        if len == 0 {
            return Err(Error::InvalidParameter);
        }

        let key = PlanKey { len, direction };
        if let Some(plan) = self.plans.get(&key) {
            return Ok(Arc::clone(plan));
        }

        let plan = match direction {
            FftDirection::Forward => self.planner.plan_fft_forward(len),
            FftDirection::Inverse => self.planner.plan_fft_inverse(len),
        };
        self.plans.insert(key, Arc::clone(&plan));
        Ok(plan)
    }

    pub(crate) fn process(
        &mut self,
        len: usize,
        direction: FftDirection,
        buffer: &mut [Complex32],
    ) -> Result<()> {
        if buffer.len() != len {
            return Err(Error::DimensionMismatch);
        }
        if buffer.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFiniteInput);
        }

        let plan = self.plan(len, direction)?;
        plan.process(buffer);
        Ok(())
    }
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    use super::{FftDirection, PlanCache};

    #[test]
    fn cache_reuses_plans_for_same_size_and_direction() {
        let mut cache = PlanCache::new();
        let first = cache.plan(16, FftDirection::Forward).unwrap();
        let second = cache.plan(16, FftDirection::Forward).unwrap();
        assert!(std::sync::Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn cache_separates_directions() {
        let mut cache = PlanCache::new();
        let forward = cache.plan(16, FftDirection::Forward).unwrap();
        let inverse = cache.plan(16, FftDirection::Inverse).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&forward, &inverse));
    }

    #[test]
    fn process_rejects_non_finite_input() {
        let mut cache = PlanCache::new();
        let mut line = [Complex32::new(1.0, f32::NAN), Complex32::new(0.0, 0.0)];
        let error = cache
            .process(2, FftDirection::Forward, &mut line)
            .unwrap_err();
        assert_eq!(error.to_string(), "input contains non-finite values");
    }
}
