use image::DynamicImage;
use ndarray::Array2;

use super::convert::{array2_to_dynamic, dynamic_to_array2, kernel2_from_array};
use crate::iterative::{
    Ictm, Landweber, RichardsonLucy, RichardsonLucyTv, TikhonovMiller, VanCittert,
};
use crate::optimization::{Bvls, Cgls, Fista, Hybr, Ista, Mrnsd, Nnls, Wpl};
use crate::spectral::{UnsupervisedWiener, Wiener};
use crate::{iterative, optimization, spectral, Result, SolveReport};

pub fn wiener(image: &Array2<f32>, psf: &Array2<f32>) -> Result<Array2<f32>> {
    wiener_with(image, psf, &Wiener::new())
}

pub fn wiener_with(image: &Array2<f32>, psf: &Array2<f32>, config: &Wiener) -> Result<Array2<f32>> {
    run_image_only(image, psf, |input, kernel| {
        spectral::wiener_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        spectral::unsupervised_wiener_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        iterative::richardson_lucy_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        iterative::richardson_lucy_tv_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        iterative::landweber_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        iterative::van_cittert_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        iterative::tikhonov_miller_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        iterative::ictm_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        optimization::nnls_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        optimization::bvls_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        optimization::ista_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        optimization::fista_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        optimization::mrnsd_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        optimization::cgls_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        optimization::wpl_with(input, kernel, config)
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
    run_image_report(image, psf, |input, kernel| {
        optimization::hybr_with(input, kernel, config)
    })
}

fn run_image_only<F>(image: &Array2<f32>, psf: &Array2<f32>, run: F) -> Result<Array2<f32>>
where
    F: FnOnce(&DynamicImage, &crate::Kernel2D) -> Result<DynamicImage>,
{
    let dynamic = array2_to_dynamic(image)?;
    let kernel = kernel2_from_array(psf)?;
    let restored = run(&dynamic, &kernel)?;
    dynamic_to_array2(&restored)
}

fn run_image_report<F>(
    image: &Array2<f32>,
    psf: &Array2<f32>,
    run: F,
) -> Result<(Array2<f32>, SolveReport)>
where
    F: FnOnce(&DynamicImage, &crate::Kernel2D) -> Result<(DynamicImage, SolveReport)>,
{
    let dynamic = array2_to_dynamic(image)?;
    let kernel = kernel2_from_array(psf)?;
    let (restored, report) = run(&dynamic, &kernel)?;
    let restored = dynamic_to_array2(&restored)?;
    Ok((restored, report))
}
