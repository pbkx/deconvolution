use criterion::{black_box, criterion_group, criterion_main, Criterion};
use deconvolution::iterative::{
    richardson_lucy_tv_with, richardson_lucy_with, RichardsonLucy, RichardsonLucyTv,
};
use deconvolution::psf::basic::gaussian2d;
use deconvolution::simulate::blur::blur;
use deconvolution::simulate::noise::add_poisson_noise;
use deconvolution::simulate::phantom::checkerboard_2d;
use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;

fn bench_rl(c: &mut Criterion) {
    let (image, psf) = degraded_fixture().expect("fixture");
    let rl = RichardsonLucy::new().iterations(16).filter_epsilon(1e-3);
    let rl_tv = RichardsonLucyTv::new()
        .iterations(16)
        .filter_epsilon(1e-3)
        .tv_weight(1.5e-2);

    c.bench_function("iterative_richardson_lucy", |b| {
        b.iter(|| {
            let _ = richardson_lucy_with(black_box(&image), black_box(&psf), black_box(&rl))
                .expect("richardson_lucy");
        });
    });

    c.bench_function("iterative_richardson_lucy_tv", |b| {
        b.iter(|| {
            let _ = richardson_lucy_tv_with(black_box(&image), black_box(&psf), black_box(&rl_tv))
                .expect("richardson_lucy_tv");
        });
    });
}

fn degraded_fixture() -> deconvolution::Result<(DynamicImage, deconvolution::Kernel2D)> {
    let sharp = checkerboard_2d((96, 96), 8, 0.0, 1.0)?;
    let psf = gaussian2d((9, 9), 1.6)?;
    let blurred = blur(&sharp, &psf)?;
    let degraded = add_poisson_noise(&blurred, 30.0, 4321)?;
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

criterion_group!(benches, bench_rl);
criterion_main!(benches);
