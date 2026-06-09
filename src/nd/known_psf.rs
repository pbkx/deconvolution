use ndarray::Array2;

use super::convert::kernel2_from_array;
use crate::algorithms;
use crate::iterative::{
    Ictm, Landweber, RichardsonLucy, RichardsonLucyTv, TikhonovMiller, VanCittert,
};
use crate::optimization::{Bvls, Cgls, Fista, Hybr, Ista, Mrnsd, Nnls, Wpl};
use crate::spectral::{UnsupervisedWiener, Wiener};
use crate::{Result, SolveReport};

pub fn wiener(image: &Array2<f32>, psf: &Array2<f32>) -> Result<Array2<f32>> {
    wiener_with(image, psf, &Wiener::new())
}

pub fn wiener_with(image: &Array2<f32>, psf: &Array2<f32>, config: &Wiener) -> Result<Array2<f32>> {
    run_array_only(image, psf, |input, kernel| {
        algorithms::wiener_array2_with(input, kernel, config)
    })
}

pub fn unsupervised_wiener(
    image: &Array2<f32>,
    psf: &Array2<f32>,
) -> Result<(Array2<f32>, SolveReport)> {
    unsupervised_wiener_with(image, psf, &UnsupervisedWiener::new())
}

pub fn unsupervised_wiener_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &UnsupervisedWiener,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::unsupervised_wiener_array2_with(input, kernel, config)
    })
}

pub fn richardson_lucy(
    image: &Array2<f32>,
    psf: &Array2<f32>,
) -> Result<(Array2<f32>, SolveReport)> {
    richardson_lucy_with(image, psf, &RichardsonLucy::new())
}

pub fn richardson_lucy_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &RichardsonLucy,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::richardson_lucy_array2_with(input, kernel, config)
    })
}

pub fn richardson_lucy_tv(
    image: &Array2<f32>,
    psf: &Array2<f32>,
) -> Result<(Array2<f32>, SolveReport)> {
    richardson_lucy_tv_with(image, psf, &RichardsonLucyTv::new())
}

pub fn richardson_lucy_tv_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &RichardsonLucyTv,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::richardson_lucy_tv_array2_with(input, kernel, config)
    })
}

pub fn landweber(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    landweber_with(image, psf, &Landweber::new())
}

pub fn landweber_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &Landweber,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::landweber_array2_with(input, kernel, config)
    })
}

pub fn van_cittert(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    van_cittert_with(image, psf, &VanCittert::new())
}

pub fn van_cittert_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &VanCittert,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::van_cittert_array2_with(input, kernel, config)
    })
}

pub fn tikhonov_miller(
    image: &Array2<f32>,
    psf: &Array2<f32>,
) -> Result<(Array2<f32>, SolveReport)> {
    tikhonov_miller_with(image, psf, &TikhonovMiller::new())
}

pub fn tikhonov_miller_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &TikhonovMiller,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::tikhonov_miller_array2_with(input, kernel, config)
    })
}

pub fn ictm(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    ictm_with(image, psf, &Ictm::new())
}

pub fn ictm_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &Ictm,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::ictm_array2_with(input, kernel, config)
    })
}

pub fn nnls(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    nnls_with(image, psf, &Nnls::new())
}

pub fn nnls_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &Nnls,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::nnls_array2_with(input, kernel, config)
    })
}

pub fn bvls(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    bvls_with(image, psf, &Bvls::new())
}

pub fn bvls_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &Bvls,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::bvls_array2_with(input, kernel, config)
    })
}

pub fn ista(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    ista_with(image, psf, &Ista::new())
}

pub fn ista_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &Ista,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::ista_array2_with(input, kernel, config)
    })
}

pub fn fista(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    fista_with(image, psf, &Fista::new())
}

pub fn fista_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &Fista,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::fista_array2_with(input, kernel, config)
    })
}

pub fn mrnsd(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    mrnsd_with(image, psf, &Mrnsd::new())
}

pub fn mrnsd_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &Mrnsd,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::mrnsd_array2_with(input, kernel, config)
    })
}

pub fn cgls(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    cgls_with(image, psf, &Cgls::new())
}

pub fn cgls_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &Cgls,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::cgls_array2_with(input, kernel, config)
    })
}

pub fn wpl(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    wpl_with(image, psf, &Wpl::new())
}

pub fn wpl_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &Wpl,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::wpl_array2_with(input, kernel, config)
    })
}

pub fn hybr(image: &Array2<f32>, psf: &Array2<f32>) -> Result<(Array2<f32>, SolveReport)> {
    hybr_with(image, psf, &Hybr::new())
}

pub fn hybr_with(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    config: &Hybr,
) -> Result<(Array2<f32>, SolveReport)> {
    run_array_report(image, psf, |input, kernel| {
        algorithms::hybr_array2_with(input, kernel, config)
    })
}

fn run_array_only<F>(image: &Array2<f32>, psf: &Array2<f32>, run: F) -> Result<Array2<f32>>
where
    F: FnOnce(&Array2<f32>, &crate::Kernel2D) -> Result<Array2<f32>>,
{
    let kernel = kernel2_from_array(psf)?;
    run(image, &kernel)
}

fn run_array_report<F>(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    run: F,
) -> Result<(Array2<f32>, SolveReport)>
where
    F: FnOnce(&Array2<f32>, &crate::Kernel2D) -> Result<(Array2<f32>, SolveReport)>,
{
    let kernel = kernel2_from_array(psf)?;
    run(image, &kernel)
}
