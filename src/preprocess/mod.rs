//! Image preprocessing helpers used before restoration.
//!
//! Use [`normalize_range`] to map image values into a solver-friendly range,
//! [`edgetaper()`] to reduce boundary ringing, and [`estimate_nsr`] to estimate
//! a scalar noise-to-signal ratio.

pub mod apodize;
mod boundary;
pub mod edgetaper;
pub mod estimate;
pub mod normalize;
mod padding;

pub use apodize::apodize;
pub use edgetaper::edgetaper;
pub use estimate::estimate_nsr;
pub use normalize::normalize_range;
pub(crate) use normalize::normalize_range_3d;
