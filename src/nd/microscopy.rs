use ndarray::Array3;

use crate::algorithms;
use crate::iterative::{RichardsonLucy, RichardsonLucyTv};
use crate::optimization::{Cmle, Gmle, Qmle};
use crate::spectral::Wiener;
use crate::{Kernel3D, Result, SolveReport};

pub fn wiener(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<Array3<f32>> {
    wiener_with(volume, psf, &Wiener::new())
}

pub fn wiener_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Wiener,
) -> Result<Array3<f32>> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::wiener_array3_with(volume, &kernel, config)
}

pub fn richardson_lucy(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
) -> Result<(Array3<f32>, SolveReport)> {
    richardson_lucy_with(volume, psf, &RichardsonLucy::new())
}

pub fn richardson_lucy_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &RichardsonLucy,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::richardson_lucy_array3_with(volume, &kernel, config)
}

pub fn richardson_lucy_tv(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
) -> Result<(Array3<f32>, SolveReport)> {
    richardson_lucy_tv_with(volume, psf, &RichardsonLucyTv::new())
}

pub fn richardson_lucy_tv_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &RichardsonLucyTv,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::richardson_lucy_tv_array3_with(volume, &kernel, config)
}

pub fn cmle(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<(Array3<f32>, SolveReport)> {
    cmle_with(volume, psf, &Cmle::new())
}

pub fn cmle_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Cmle,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::cmle_array3_with(volume, &kernel, config)
}

pub fn gmle(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<(Array3<f32>, SolveReport)> {
    gmle_with(volume, psf, &Gmle::new())
}

pub fn gmle_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Gmle,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::gmle_array3_with(volume, &kernel, config)
}

pub fn qmle(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<(Array3<f32>, SolveReport)> {
    qmle_with(volume, psf, &Qmle::new())
}

pub fn qmle_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Qmle,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_from_array(psf)?;
    algorithms::qmle_array3_with(volume, &kernel, config)
}

fn kernel3_from_array(psf: &Array3<f32>) -> Result<Kernel3D> {
    Kernel3D::new(psf.as_standard_layout().to_owned())
}
