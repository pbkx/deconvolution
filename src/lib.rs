mod core;

pub mod error;
pub mod otf;
pub mod prelude;
pub mod preprocess;
pub mod psf;
pub mod traits;

pub use crate::core::diagnostics::SolveReport;
pub use crate::core::regularizer::{RegOperator2D, RegOperator3D};
pub use crate::core::stopping::StopReason;
pub use crate::error::{Error, Result};
pub use crate::otf::{Transfer2D, Transfer3D};
pub use crate::psf::{Kernel2D, Kernel3D};
pub use crate::traits::{Boundary, ChannelMode, Padding, RangePolicy};
