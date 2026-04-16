use deconvolution::nd::microscopy::qmle_with;
use deconvolution::psf::gaussian3d;
use deconvolution::simulate::{blur, phantom_3d};
use deconvolution::Qmle;
use ndarray::{Array3, Axis};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let sharp = phantom_3d((9, 48, 48))?;
    let psf_3d = gaussian3d((7, 9, 9), 1.5)?;
    let blurred = blur_volume_slicewise(&sharp, psf_3d.as_array())?;

    let (restored, report) = qmle_with(
        &blurred,
        psf_3d.as_array(),
        &Qmle::new().iterations(9).snr(60.0).acuity(1.1),
    )?;

    println!(
        "microscopy volume output: {}x{}x{}, iterations={}",
        restored.dim().2,
        restored.dim().1,
        restored.dim().0,
        report.iterations
    );
    Ok(())
}

fn blur_volume_slicewise(
    volume: &Array3<f32>,
    psf_3d: &Array3<f32>,
) -> deconvolution::Result<Array3<f32>> {
    if volume.is_empty() {
        return Err(deconvolution::Error::EmptyImage);
    }
    if volume.iter().any(|value| !value.is_finite()) {
        return Err(deconvolution::Error::NonFiniteInput);
    }
    if psf_3d.is_empty() {
        return Err(deconvolution::Error::InvalidPsf);
    }
    if psf_3d.iter().any(|value| !value.is_finite()) {
        return Err(deconvolution::Error::NonFiniteInput);
    }

    let mut projected = psf_3d.sum_axis(Axis(0));
    let sum = projected.sum();
    if !sum.is_finite() || sum.abs() <= f32::EPSILON {
        return Err(deconvolution::Error::InvalidPsf);
    }
    for value in &mut projected {
        *value /= sum;
    }
    let projected = deconvolution::Kernel2D::new(projected)?;

    let (depth, height, width) = volume.dim();
    let mut output = Array3::zeros((depth, height, width));
    for z in 0..depth {
        let slice = volume.index_axis(Axis(0), z).to_owned();
        let blurred = blur(&slice, &projected)?;
        output.index_axis_mut(Axis(0), z).assign(&blurred);
    }
    Ok(output)
}
