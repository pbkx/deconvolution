#![forbid(unsafe_code)]

mod algorithms;
mod core;

pub mod error;
pub mod otf;
pub mod prelude;
pub mod preprocess;
pub mod psf;
pub mod simulate;
pub mod traits;

pub use crate::core::diagnostics::SolveReport;
pub use crate::core::regularizer::{RegOperator2D, RegOperator3D};
pub use crate::core::stopping::StopReason;
pub use crate::error::{Error, Result};
pub use crate::otf::{Transfer2D, Transfer3D};
pub use crate::psf::{Kernel2D, Kernel3D};
pub use crate::traits::{Boundary, ChannelMode, Padding, RangePolicy};
pub use algorithms::{
    damped_richardson_lucy, damped_richardson_lucy_with, inverse_filter, inverse_filter_with,
    naive_inverse_filter, naive_inverse_filter_with, regularized_inverse_filter,
    regularized_inverse_filter_with, richardson_lucy, richardson_lucy_with,
    tikhonov_inverse_filter, tikhonov_inverse_filter_with, truncated_inverse_filter,
    truncated_inverse_filter_with, unsupervised_wiener, unsupervised_wiener_with, wiener,
    wiener_with, InverseFilter, RegularizedInverseFilter, RichardsonLucy, TikhonovInverseFilter,
    UnsupervisedWiener, Wiener,
};
