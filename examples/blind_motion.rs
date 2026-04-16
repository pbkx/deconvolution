use deconvolution::blind::{richardson_lucy_with, BlindRichardsonLucy};
use deconvolution::psf::{motion_linear, uniform};
use deconvolution::simulate::{add_poisson_noise, blur, checkerboard_2d};
use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let sharp = checkerboard_2d((128, 128), 8, 0.0, 1.0)?;
    let true_psf = motion_linear(13.0, 30.0)?;
    let blurred = blur(&sharp, &true_psf)?;
    let degraded = add_poisson_noise(&blurred, 40.0, 404)?;
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded)?);
    let initial_psf = uniform(true_psf.dims())?;

    let output = richardson_lucy_with(
        &degraded_image,
        &initial_psf,
        &BlindRichardsonLucy::new()
            .iterations(20)
            .filter_epsilon(1e-3)
            .collect_history(true),
    )?;

    println!(
        "blind output: {}x{}, psf={}x{}, iterations={}",
        output.image.width(),
        output.image.height(),
        output.psf.dims().1,
        output.psf.dims().0,
        output.report.iterations
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
