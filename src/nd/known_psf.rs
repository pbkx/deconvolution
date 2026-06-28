//! Known-PSF `ndarray` deconvolution wrappers.
//!
//! Inputs use `(height, width)` order. The `f16` feature enables half-precision
//! sample inputs, but computation still runs through `f32`.

use ndarray::Array2;

use super::convert::{NdSample, array2_from_f32, array2_to_f32, kernel2_from_samples};
use crate::algorithms;
use crate::iterative::{
    Ictm, Landweber, RichardsonLucy, RichardsonLucyTv, TikhonovMiller, VanCittert,
};
use crate::optimization::{Bvls, Cgls, Fista, Hybr, Ista, Mrnsd, Nnls, Wpl};
use crate::spectral::{UnsupervisedWiener, Wiener};
use crate::{Result, SolveReport};

pub fn wiener<T: NdSample>(image: &Array2<T>, psf: &Array2<T>) -> Result<Array2<T>> {
    wiener_with(image, psf, &Wiener::new())
}

pub fn wiener_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Wiener,
) -> Result<Array2<T>> {
    run_array_only(image, psf, |input, kernel| {
        algorithms::wiener_array2_with(input, kernel, config)
    })
}

pub fn unsupervised_wiener<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
) -> Result<(Array2<T>, SolveReport)> {
    unsupervised_wiener_with(image, psf, &UnsupervisedWiener::new())
}

pub fn unsupervised_wiener_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &UnsupervisedWiener,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::unsupervised_wiener_array2_with(input, kernel, config)
    })
}

pub fn richardson_lucy<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
) -> Result<(Array2<T>, SolveReport)> {
    richardson_lucy_with(image, psf, &RichardsonLucy::new())
}

pub fn richardson_lucy_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &RichardsonLucy,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::richardson_lucy_array2_with(input, kernel, config)
    })
}

pub fn richardson_lucy_tv<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
) -> Result<(Array2<T>, SolveReport)> {
    richardson_lucy_tv_with(image, psf, &RichardsonLucyTv::new())
}

pub fn richardson_lucy_tv_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &RichardsonLucyTv,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::richardson_lucy_tv_array2_with(input, kernel, config)
    })
}

pub fn landweber<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
) -> Result<(Array2<T>, SolveReport)> {
    landweber_with(image, psf, &Landweber::new())
}

pub fn landweber_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Landweber,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::landweber_array2_with(input, kernel, config)
    })
}

pub fn van_cittert<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
) -> Result<(Array2<T>, SolveReport)> {
    van_cittert_with(image, psf, &VanCittert::new())
}

pub fn van_cittert_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &VanCittert,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::van_cittert_array2_with(input, kernel, config)
    })
}

pub fn tikhonov_miller<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
) -> Result<(Array2<T>, SolveReport)> {
    tikhonov_miller_with(image, psf, &TikhonovMiller::new())
}

pub fn tikhonov_miller_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &TikhonovMiller,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::tikhonov_miller_array2_with(input, kernel, config)
    })
}

pub fn ictm<T: NdSample>(image: &Array2<T>, psf: &Array2<T>) -> Result<(Array2<T>, SolveReport)> {
    ictm_with(image, psf, &Ictm::new())
}

pub fn ictm_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Ictm,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::ictm_array2_with(input, kernel, config)
    })
}

pub fn nnls<T: NdSample>(image: &Array2<T>, psf: &Array2<T>) -> Result<(Array2<T>, SolveReport)> {
    nnls_with(image, psf, &Nnls::new())
}

pub fn nnls_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Nnls,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::nnls_array2_with(input, kernel, config)
    })
}

pub fn bvls<T: NdSample>(image: &Array2<T>, psf: &Array2<T>) -> Result<(Array2<T>, SolveReport)> {
    bvls_with(image, psf, &Bvls::new())
}

pub fn bvls_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Bvls,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::bvls_array2_with(input, kernel, config)
    })
}

pub fn ista<T: NdSample>(image: &Array2<T>, psf: &Array2<T>) -> Result<(Array2<T>, SolveReport)> {
    ista_with(image, psf, &Ista::new())
}

pub fn ista_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Ista,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::ista_array2_with(input, kernel, config)
    })
}

pub fn fista<T: NdSample>(image: &Array2<T>, psf: &Array2<T>) -> Result<(Array2<T>, SolveReport)> {
    fista_with(image, psf, &Fista::new())
}

pub fn fista_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Fista,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::fista_array2_with(input, kernel, config)
    })
}

pub fn mrnsd<T: NdSample>(image: &Array2<T>, psf: &Array2<T>) -> Result<(Array2<T>, SolveReport)> {
    mrnsd_with(image, psf, &Mrnsd::new())
}

pub fn mrnsd_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Mrnsd,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::mrnsd_array2_with(input, kernel, config)
    })
}

pub fn cgls<T: NdSample>(image: &Array2<T>, psf: &Array2<T>) -> Result<(Array2<T>, SolveReport)> {
    cgls_with(image, psf, &Cgls::new())
}

pub fn cgls_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Cgls,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::cgls_array2_with(input, kernel, config)
    })
}

pub fn wpl<T: NdSample>(image: &Array2<T>, psf: &Array2<T>) -> Result<(Array2<T>, SolveReport)> {
    wpl_with(image, psf, &Wpl::new())
}

pub fn wpl_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Wpl,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::wpl_array2_with(input, kernel, config)
    })
}

pub fn hybr<T: NdSample>(image: &Array2<T>, psf: &Array2<T>) -> Result<(Array2<T>, SolveReport)> {
    hybr_with(image, psf, &Hybr::new())
}

pub fn hybr_with<T: NdSample>(
    image: &Array2<T>,
    psf: &Array2<T>,
    config: &Hybr,
) -> Result<(Array2<T>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::hybr_array2_with(input, kernel, config)
    })
}

fn run_array_only<T, F>(image: &Array2<T>, psf: &Array2<T>, run: F) -> Result<Array2<T>>
where
    T: NdSample,
    F: FnOnce(&Array2<f32>, &crate::Kernel2D) -> Result<Array2<f32>>,
{
    let image = array2_to_f32(image)?;
    let kernel = kernel2_from_samples(psf)?;
    let restored = run(&image, &kernel)?;
    array2_from_f32(&restored)
}

fn run_array_report<T, F>(
    image: &Array2<T>,
    psf: &Array2<T>,
    run: F,
) -> Result<(Array2<T>, SolveReport)>
where
    T: NdSample,
    F: FnOnce(&Array2<f32>, &crate::Kernel2D) -> Result<(Array2<f32>, SolveReport)>,
{
    let image = array2_to_f32(image)?;
    let kernel = kernel2_from_samples(psf)?;
    let (restored, report) = run(&image, &kernel)?;
    let restored = array2_from_f32(&restored)?;
    Ok((restored, report))
}
