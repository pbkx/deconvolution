use ndarray::{Array2, Array3};

use crate::{Error, Result};

use super::validate::{finite_real_2d, finite_real_3d, same_dims_2d, same_dims_3d};

pub(crate) trait LinearOperator2D {
    fn apply(&self, input: &Array2<f32>) -> Result<Array2<f32>>;
    fn adjoint(&self, input: &Array2<f32>) -> Result<Array2<f32>>;
}

pub(crate) trait LinearOperator3D {
    fn apply(&self, input: &Array3<f32>) -> Result<Array3<f32>>;
    fn adjoint(&self, input: &Array3<f32>) -> Result<Array3<f32>>;
}

pub(crate) fn inner_product_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> Result<f32> {
    finite_real_2d(lhs)?;
    finite_real_2d(rhs)?;
    same_dims_2d(lhs, rhs)?;

    let mut sum = 0.0_f32;
    for (left, right) in lhs.iter().zip(rhs.iter()) {
        sum += left * right;
    }

    if !sum.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(sum)
}

pub(crate) fn inner_product_3d(lhs: &Array3<f32>, rhs: &Array3<f32>) -> Result<f32> {
    finite_real_3d(lhs)?;
    finite_real_3d(rhs)?;
    same_dims_3d(lhs, rhs)?;

    let mut sum = 0.0_f32;
    for (left, right) in lhs.iter().zip(rhs.iter()) {
        sum += left * right;
    }

    if !sum.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(sum)
}
