use deconvolution::psf::gaussian2d;
use deconvolution::simulate::{add_poisson_noise, blur, checkerboard_2d};
use deconvolution::{richardson_lucy_with, RichardsonLucy};
use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let sharp = checkerboard_2d((128, 128), 8, 0.0, 1.0)?;
    let psf = gaussian2d((9, 9), 1.6)?;
    let blurred = blur(&sharp, &psf)?;
    let degraded = add_poisson_noise(&blurred, 24.0, 2026)?;
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded)?);

    let (restored, report) = richardson_lucy_with(
        &degraded_image,
        &psf,
        &RichardsonLucy::new()
            .iterations(20)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )?;

    println!(
        "richardson_lucy output: {}x{}, iterations={}",
        restored.width(),
        restored.height(),
        report.iterations
    );
    Ok(())
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
