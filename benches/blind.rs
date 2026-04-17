use criterion::{black_box, criterion_group, criterion_main, Criterion};
use deconvolution::blind::{richardson_lucy_with, BlindRichardsonLucy};
use deconvolution::psf::basic::motion_linear;
use deconvolution::psf::init::uniform;
use deconvolution::simulate::blur::blur;
use deconvolution::simulate::noise::add_poisson_noise;
use deconvolution::simulate::phantom::checkerboard_2d;
use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;

fn bench_blind(c: &mut Criterion) {
    let (image, initial_psf) = degraded_fixture().expect("fixture");
    let config = BlindRichardsonLucy::new()
        .iterations(14)
        .filter_epsilon(1e-3)
        .collect_history(false);

    c.bench_function("blind_richardson_lucy", |b| {
        b.iter(|| {
            let _ = richardson_lucy_with(
                black_box(&image),
                black_box(&initial_psf),
                black_box(&config),
            )
            .expect("blind_richardson_lucy");
        });
    });
}

fn degraded_fixture() -> deconvolution::Result<(DynamicImage, deconvolution::Kernel2D)> {
    let sharp = checkerboard_2d((96, 96), 8, 0.0, 1.0)?;
    let true_psf = motion_linear(11.0, 25.0)?;
    let blurred = blur(&sharp, &true_psf)?;
    let degraded = add_poisson_noise(&blurred, 42.0, 2101)?;
    let image = DynamicImage::ImageLuma8(array_to_gray(&degraded)?);
    let initial_psf = uniform(true_psf.dims())?;
    Ok((image, initial_psf))
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

criterion_group!(benches, bench_blind);
criterion_main!(benches);
