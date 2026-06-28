//! Common imports for simple deconvolution workflows.
//!
//! This prelude gathers the primary algorithm configs, kernel/transfer types,
//! error types, and shared option enums used by most callers.

pub use crate::core::diagnostics::SolveReport;
pub use crate::core::stopping::StopReason;
pub use crate::error::{Error, Result};
pub use crate::iterative::RichardsonLucy;
pub use crate::otf::{Transfer2D, Transfer3D};
pub use crate::psf::{Blur2D, Blur3D, Kernel2D, Kernel3D, PsfConstraint};
pub use crate::spectral::Wiener;
pub use crate::traits::{Boundary, ChannelMode, Padding, RangePolicy};
