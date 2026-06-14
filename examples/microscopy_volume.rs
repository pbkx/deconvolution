use deconvolution::nd::microscopy::qmle_with;
use deconvolution::optimization::Qmle;
use deconvolution::psf::basic::gaussian3d;
use deconvolution::simulate::blur::blur_3d;
use deconvolution::simulate::phantom::phantom_3d;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let sharp = phantom_3d((9, 48, 48))?;
    let psf_3d = gaussian3d((7, 9, 9), 1.5)?;
    let blurred = blur_3d(&sharp, &psf_3d)?;

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
