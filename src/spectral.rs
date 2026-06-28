//! Frequency-domain deconvolution algorithms.
//!
//! Use [`Wiener`] for configurable Wiener filtering, [`wiener`] for the default
//! path, and inverse-filter builders when you need direct spectral division.

pub use crate::algorithms::{
    InverseFilter, RegularizedInverseFilter, TikhonovInverseFilter, UnsupervisedWiener, Wiener,
    inverse_filter, inverse_filter_with, naive_inverse_filter, naive_inverse_filter_with,
    regularized_inverse_filter, regularized_inverse_filter_with, tikhonov_inverse_filter,
    tikhonov_inverse_filter_with, truncated_inverse_filter, truncated_inverse_filter_with,
    unsupervised_wiener, unsupervised_wiener_with, wiener, wiener_with,
};
pub use crate::core::regularizer::{RegOperator2D, RegOperator3D};
