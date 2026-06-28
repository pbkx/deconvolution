//! Shared configuration enums used across image and ndarray solvers.
//!
//! Use these options with builders such as [`crate::spectral::Wiener`] and
//! [`crate::iterative::RichardsonLucy`] to control boundaries, padding,
//! channels, and output ranges.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Boundary extension used when an algorithm samples outside the image.
pub enum Boundary {
    /// Treat samples outside the image as `0`.
    Zero,
    /// Repeat the nearest edge sample.
    Replicate,
    /// Mirror across the edge without repeating the edge sample.
    Reflect,
    /// Mirror across the edge and repeat the edge sample.
    Symmetric,
    /// Wrap coordinates modulo the image dimensions.
    Periodic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// FFT padding policy for spectral deconvolution.
pub enum Padding {
    /// Keep the original image or volume dimensions.
    None,
    /// Pad enough to return an output with the original dimensions.
    Same,
    /// Pad to the minimal linear-convolution extent.
    Minimal,
    /// Pad each axis to a fast FFT length at least as large as the minimal extent.
    NextFastLen,
    /// Pad a 2D operation to `(height, width)`.
    Explicit2(usize, usize),
    /// Pad a 3D operation to `(depth, height, width)`.
    Explicit3(usize, usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Channel handling for `image::DynamicImage` restoration.
pub enum ChannelMode {
    /// Restore each color channel independently.
    Independent,
    /// Restore only luminance and rebuild an image like the input.
    LumaOnly,
    /// Restore color channels while copying alpha unchanged.
    IgnoreAlpha,
    /// Restore premultiplied color and alpha together.
    PremultipliedAlpha,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Output range policy applied after restoration.
pub enum RangePolicy {
    /// Preserve the numeric range chosen by the solver.
    PreserveInput,
    /// Clamp output samples to `[0, 1]`.
    Clamp01,
    /// Clamp output samples to `[-1, 1]`.
    ClampNegPos1,
    /// Do not clamp output samples.
    Unbounded,
}
