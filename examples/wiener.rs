use std::env;

use deconvolution::psf::basic::gaussian2d;
use deconvolution::simulate::blur::blur;
use deconvolution::simulate::phantom::checkerboard_2d;
use deconvolution::spectral::{wiener_with, Wiener};
use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let input = if let Some(path) = args.get(1) {
        image::open(path)?
    } else {
        default_input()?
    };

    let psf = gaussian2d((9, 9), 1.6)?;
    let restored = wiener_with(
        &input,
        &psf,
        &Wiener::new().nsr(1e-2).collect_history(false),
    )?;

    if let Some(path) = args.get(2) {
        restored.save(path)?;
    }

    let output = restored.to_luma8();
    println!("wiener output: {}x{}", output.width(), output.height());
    Ok(())
}

fn default_input() -> deconvolution::Result<DynamicImage> {
    let sharp = checkerboard_2d((128, 128), 8, 0.0, 1.0)?;
    let psf = gaussian2d((9, 9), 1.6)?;
    let blurred = blur(&sharp, &psf)?;
    Ok(DynamicImage::ImageLuma8(array_to_gray(&blurred)?))
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
