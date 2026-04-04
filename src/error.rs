pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("dimension mismatch")]
    DimensionMismatch,
    #[error("invalid point spread function")]
    InvalidPsf,
    #[error("invalid transfer function")]
    InvalidTransfer,
    #[error("invalid parameter")]
    InvalidParameter,
    #[error("unsupported pixel type")]
    UnsupportedPixelType,
    #[error("input contains non-finite values")]
    NonFiniteInput,
    #[error("solver failed to converge")]
    ConvergenceFailure,
    #[error("input image is empty")]
    EmptyImage,
}
