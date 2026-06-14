use criterion::{Criterion, criterion_group, criterion_main};
use deconvolution::nd;
use deconvolution::optimization::Qmle;
use deconvolution::psf::basic::gaussian3d;
use deconvolution::simulate::blur::blur_3d;
use deconvolution::simulate::phantom::phantom_3d;
use ndarray::Array3;
use std::hint::black_box;

fn bench_volume(c: &mut Criterion) {
    let (volume, psf) = degraded_volume_fixture().expect("fixture");
    let config = Qmle::new().iterations(8).snr(60.0).acuity(1.1);

    c.bench_function("volume_qmle", |b| {
        b.iter(|| {
            let _ =
                nd::microscopy::qmle_with(black_box(&volume), black_box(&psf), black_box(&config))
                    .expect("nd_qmle");
        });
    });
}

fn degraded_volume_fixture() -> deconvolution::Result<(Array3<f32>, Array3<f32>)> {
    let sharp = phantom_3d((7, 40, 40))?;
    let psf_3d = gaussian3d((7, 9, 9), 1.5)?;
    let blurred = blur_3d(&sharp, &psf_3d)?;
    Ok((blurred, psf_3d.as_array().to_owned()))
}

criterion_group!(benches, bench_volume);
criterion_main!(benches);
