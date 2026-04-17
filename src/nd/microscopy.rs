use image::DynamicImage;
use ndarray::{Array2, Array3, Axis};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::convert::{
    array2_to_dynamic, dynamic_to_array2, kernel3_to_projected_kernel2, validate_array3,
};
use crate::iterative::{self, RichardsonLucy, RichardsonLucyTv};
use crate::optimization::{self, Cmle, Gmle, Qmle};
use crate::spectral::{self, Wiener};
use crate::{Error, Result, SolveReport, StopReason};

pub fn wiener(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<Array3<f32>> {
    wiener_with(volume, psf, &Wiener::new())
}

pub fn wiener_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Wiener,
) -> Result<Array3<f32>> {
    let kernel = kernel3_to_projected_kernel2(psf)?;
    run_slicewise_image_only(volume, &kernel, |slice, psf| {
        spectral::wiener_with(slice, psf, config)
    })
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
    let kernel = kernel3_to_projected_kernel2(psf)?;
    run_slicewise_report(volume, &kernel, |slice, psf| {
        iterative::richardson_lucy_with(slice, psf, config)
    })
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
    let kernel = kernel3_to_projected_kernel2(psf)?;
    run_slicewise_report(volume, &kernel, |slice, psf| {
        iterative::richardson_lucy_tv_with(slice, psf, config)
    })
}

pub fn cmle(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<(Array3<f32>, SolveReport)> {
    cmle_with(volume, psf, &Cmle::new())
}

pub fn cmle_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Cmle,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_to_projected_kernel2(psf)?;
    run_slicewise_report(volume, &kernel, |slice, psf| {
        optimization::cmle_with(slice, psf, config)
    })
}

pub fn gmle(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<(Array3<f32>, SolveReport)> {
    gmle_with(volume, psf, &Gmle::new())
}

pub fn gmle_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Gmle,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_to_projected_kernel2(psf)?;
    run_slicewise_report(volume, &kernel, |slice, psf| {
        optimization::gmle_with(slice, psf, config)
    })
}

pub fn qmle(volume: &Array3<f32>, psf: &Array3<f32>) -> Result<(Array3<f32>, SolveReport)> {
    qmle_with(volume, psf, &Qmle::new())
}

pub fn qmle_with(
    volume: &Array3<f32>,
    psf: &Array3<f32>,
    config: &Qmle,
) -> Result<(Array3<f32>, SolveReport)> {
    let kernel = kernel3_to_projected_kernel2(psf)?;
    run_slicewise_report(volume, &kernel, |slice, psf| {
        optimization::qmle_with(slice, psf, config)
    })
}

fn run_slicewise_image_only<F>(
    volume: &Array3<f32>,
    psf: &crate::Kernel2D,
    run: F,
) -> Result<Array3<f32>>
where
    F: Fn(&DynamicImage, &crate::Kernel2D) -> Result<DynamicImage> + Sync,
{
    validate_array3(volume)?;
    let (depth, height, width) = volume.dim();

    #[cfg(feature = "rayon")]
    let slices: Vec<Result<(usize, Array2<f32>)>> = (0..depth)
        .into_par_iter()
        .map(|z| {
            let slice = volume.index_axis(Axis(0), z).to_owned();
            let input = array2_to_dynamic(&slice)?;
            let restored = run(&input, psf)?;
            let restored = dynamic_to_array2(&restored)?;
            if restored.dim() != (height, width) {
                return Err(Error::DimensionMismatch);
            }
            Ok((z, restored))
        })
        .collect();

    #[cfg(not(feature = "rayon"))]
    let slices: Vec<Result<(usize, Array2<f32>)>> = (0..depth)
        .map(|z| {
            let slice = volume.index_axis(Axis(0), z).to_owned();
            let input = array2_to_dynamic(&slice)?;
            let restored = run(&input, psf)?;
            let restored = dynamic_to_array2(&restored)?;
            if restored.dim() != (height, width) {
                return Err(Error::DimensionMismatch);
            }
            Ok((z, restored))
        })
        .collect();

    let mut output = Array3::zeros((depth, height, width));
    for item in slices {
        let (z, restored) = item?;
        output.index_axis_mut(Axis(0), z).assign(&restored);
    }

    Ok(output)
}

fn run_slicewise_report<F>(
    volume: &Array3<f32>,
    psf: &crate::Kernel2D,
    run: F,
) -> Result<(Array3<f32>, SolveReport)>
where
    F: Fn(&DynamicImage, &crate::Kernel2D) -> Result<(DynamicImage, SolveReport)> + Sync,
{
    validate_array3(volume)?;
    let (depth, height, width) = volume.dim();

    #[cfg(feature = "rayon")]
    let slices: Vec<Result<(usize, Array2<f32>, SolveReport)>> = (0..depth)
        .into_par_iter()
        .map(|z| {
            let slice = volume.index_axis(Axis(0), z).to_owned();
            let input = array2_to_dynamic(&slice)?;
            let (restored, report) = run(&input, psf)?;
            let restored = dynamic_to_array2(&restored)?;
            if restored.dim() != (height, width) {
                return Err(Error::DimensionMismatch);
            }
            Ok((z, restored, report))
        })
        .collect();

    #[cfg(not(feature = "rayon"))]
    let slices: Vec<Result<(usize, Array2<f32>, SolveReport)>> = (0..depth)
        .map(|z| {
            let slice = volume.index_axis(Axis(0), z).to_owned();
            let input = array2_to_dynamic(&slice)?;
            let (restored, report) = run(&input, psf)?;
            let restored = dynamic_to_array2(&restored)?;
            if restored.dim() != (height, width) {
                return Err(Error::DimensionMismatch);
            }
            Ok((z, restored, report))
        })
        .collect();

    let mut output = Array3::zeros((depth, height, width));
    let mut reports: Vec<Option<SolveReport>> = (0..depth).map(|_| None).collect();
    for item in slices {
        let (z, restored, report) = item?;
        output.index_axis_mut(Axis(0), z).assign(&restored);
        reports[z] = Some(report);
    }

    let reports: Vec<SolveReport> = reports
        .into_iter()
        .map(|report| report.ok_or(Error::InvalidParameter))
        .collect::<Result<Vec<_>>>()?;
    let report = combine_reports(&reports)?;
    Ok((output, report))
}

fn combine_reports(reports: &[SolveReport]) -> Result<SolveReport> {
    if reports.is_empty() {
        return Err(Error::InvalidParameter);
    }

    let iterations = reports
        .iter()
        .map(|report| report.iterations)
        .max()
        .ok_or(Error::InvalidParameter)?;
    let stop_reason = reports
        .last()
        .map(|report| report.stop_reason)
        .unwrap_or(StopReason::MaxIterations);
    let objective_history = average_histories(
        &reports
            .iter()
            .map(|report| report.objective_history.as_slice())
            .collect::<Vec<_>>(),
    )?;
    let residual_history = average_histories(
        &reports
            .iter()
            .map(|report| report.residual_history.as_slice())
            .collect::<Vec<_>>(),
    )?;
    let estimated_nsr = average_optional_nsr(reports)?;

    Ok(SolveReport {
        iterations,
        stop_reason,
        objective_history,
        residual_history,
        estimated_nsr,
    })
}

fn average_histories(histories: &[&[f32]]) -> Result<Vec<f32>> {
    let max_len = histories
        .iter()
        .map(|history| history.len())
        .max()
        .unwrap_or(0);
    let mut output = Vec::with_capacity(max_len);

    for index in 0..max_len {
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        for history in histories {
            if let Some(value) = history.get(index) {
                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                sum += *value;
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        let avg = sum / count as f32;
        if !avg.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        output.push(avg);
    }

    Ok(output)
}

fn average_optional_nsr(reports: &[SolveReport]) -> Result<Option<f32>> {
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for report in reports {
        if let Some(value) = report.estimated_nsr {
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            sum += value;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(None);
    }

    let avg = sum / count as f32;
    if !avg.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(Some(avg))
}
