//! Iterative image deconvolution algorithms for known 2D PSFs.
//!
//! Use [`RichardsonLucy`] for Poisson noise, or builders such as [`Landweber`]
//! and [`TikhonovMiller`] when an additive-noise model is more appropriate.

pub use crate::algorithms::{
    Ictm, Landweber, RichardsonLucy, RichardsonLucyTv, TikhonovMiller, VanCittert,
    damped_richardson_lucy, damped_richardson_lucy_with, ictm, ictm_with, landweber,
    landweber_with, richardson_lucy, richardson_lucy_tv, richardson_lucy_tv_with,
    richardson_lucy_with, tikhonov_miller, tikhonov_miller_with, van_cittert, van_cittert_with,
};
