//! Synthetic blur, noise, and phantom generation utilities.
//!
//! Use [`blur`] to apply a PSF or OTF, [`noise`] to add reproducible noise, and
//! [`phantom`] to create small test images or volumes.

pub mod blur;
pub mod noise;
pub mod phantom;
