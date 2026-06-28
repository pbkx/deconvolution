//! Constrained, Krylov, MLE, and proximal restoration algorithms.
//!
//! Use this module when you need solver families beyond the common
//! [`crate::iterative`] and [`crate::spectral`] entry points, such as [`Nnls`]
//! or [`Fista`].

pub use crate::algorithms::{
    Bvls, Cgls, Cmle, Fista, Gmle, Hybr, Ista, Mrnsd, Nnls, Qmle, SparseBasis, Wpl, bvls,
    bvls_with, cgls, cgls_with, cmle, cmle_with, fista, fista_with, gmle, gmle_with, hybr,
    hybr_with, ista, ista_with, mrnsd, mrnsd_with, nnls, nnls_with, qmle, qmle_with, wpl, wpl_with,
};
