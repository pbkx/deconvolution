use criterion::{black_box, criterion_group, criterion_main, Criterion};
use deconvolution::psf::basic::gaussian2d;
use deconvolution::simulate::blur::blur;
use deconvolution::simulate::noise::add_gaussian_noise;
use deconvolution::simulate::phantom::checkerboard_2d;
use deconvolution::spectral::{unsupervised_wiener_with, wiener_with, UnsupervisedWiener, Wiener};
use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;

fn bench_spectral(c: &mut Criterion) {
    let (image, psf) = degraded_fixture().expect("fixture");
    let config_wiener = Wiener::new().nsr(1e-2);
    let config_unsupervised = UnsupervisedWiener::new()
        .max_iterations(12)
        .min_iterations(2)
        .tolerance(1e-3);

    c.bench_function("spectral_wiener", |b| {
        b.iter(|| {
            let _ = wiener_with(
                black_box(&image),
                black_box(&psf),
                black_box(&config_wiener),
            )
            .expect("wiener");
        });
    });

    c.bench_function("spectral_unsupervised_wiener", |b| {
        b.iter(|| {
            let _ = unsupervised_wiener_with(
                black_box(&image),
                black_box(&psf),
                black_box(&config_unsupervised),
            )
            .expect("unsupervised_wiener");
        });
    });
}

fn degraded_fixture() -> deconvolution::Result<(DynamicImage, deconvolution::Kernel2D)> {
    let sharp = checkerboard_2d((96, 96), 8, 0.0, 1.0)?;
    let psf = gaussian2d((9, 9), 1.6)?;
    let blurred = blur(&sharp, &psf)?;
    let degraded = add_gaussian_noise(&blurred, 0.03, 1234)?;
    let image = DynamicImage::ImageLuma8(array_to_gray(&degraded)?);
    Ok((image, psf))
}

fn array_to_gray(input: &Array2<f32>) -> deconvolution::Result<GrayImage> {
    let (height, width) = input.dim();
    let width_u32 = u32::try_from(width).map_err(|_| deconvolution::Error::DimensionMismatch)?;
    let height_u32 = u32::try_from(height).map_err(|_| deconvolution::Error::DimensionMismatch)?;
    let mut image = GrayImage::new(width_u32, height_u32);

    for y in 0..height {
        let y_u32 = u32::try_from(y).map_err(|_| deconvolution::Error::DimensionMismatch)?;
        for x in 0..width {
            let x_u32 = u32::try_from(x).map_err(|_| deconvolution::Error::DimensionMismatch)?;
            let value = (input[[y, x]].clamp(0.0, 1.0) * 255.0).round() as u8;
            image.put_pixel(x_u32, y_u32, Luma([value]));
        }
    }

    Ok(image)
}

criterion_group!(benches, bench_spectral);
criterion_main!(benches);
