//! Crate error and result types.
//!
//! Most fallible APIs return [`Result`] and use [`Error`] for validation,
//! conversion, and solver failures.

/// Result alias used by fallible deconvolution APIs.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
/// Error categories returned by validation, conversion, and solver routines.
pub enum Error {
    /// Input, PSF, OTF, or output shapes are incompatible.
    #[error("dimension mismatch")]
    DimensionMismatch,
    /// A point-spread function is empty, non-normalizable, or otherwise invalid.
    #[error("invalid point spread function")]
    InvalidPsf,
    /// An optical transfer function is empty, non-normalizable, or otherwise invalid.
    #[error("invalid transfer function")]
    InvalidTransfer,
    /// A configuration value is outside its accepted range.
    #[error("invalid parameter")]
    InvalidParameter,
    /// The input image uses a pixel type this crate cannot restore.
    #[error("unsupported pixel type")]
    UnsupportedPixelType,
    /// Input data contains `NaN` or infinite values.
    #[error("input contains non-finite values")]
    NonFiniteInput,
    /// The configured solver stopped without satisfying its convergence checks.
    #[error("solver failed to converge")]
    ConvergenceFailure,
    /// The input image, array, kernel, or transfer has zero elements.
    #[error("input image is empty")]
    EmptyImage,
}
