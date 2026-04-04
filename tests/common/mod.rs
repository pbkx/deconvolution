use ndarray::{Array2, Array3};

pub fn arrays_equal_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> bool {
    lhs.dim() == rhs.dim()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub fn arrays_equal_3d(lhs: &Array3<f32>, rhs: &Array3<f32>) -> bool {
    lhs.dim() == rhs.dim()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub fn arrays_differ_2d(lhs: &Array2<f32>, rhs: &Array2<f32>) -> bool {
    if lhs.dim() != rhs.dim() {
        return true;
    }
    lhs.iter()
        .zip(rhs.iter())
        .any(|(left, right)| left.to_bits() != right.to_bits())
}

pub fn is_finite_2d(input: &Array2<f32>) -> bool {
    input.iter().all(|value| value.is_finite())
}

pub fn is_finite_3d(input: &Array3<f32>) -> bool {
    input.iter().all(|value| value.is_finite())
}
