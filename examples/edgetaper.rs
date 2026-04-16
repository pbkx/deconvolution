use std::env;

use deconvolution::preprocess::edgetaper;
use deconvolution::psf::gaussian2d;
use deconvolution::simulate::checkerboard_2d;
use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array2;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let input = if let Some(path) = args.get(1) {
        image::open(path)?.to_luma8()
    } else {
        array_to_gray(&checkerboard_2d((128, 128), 8, 0.0, 1.0)?)?
    };

    let psf = gaussian2d((9, 9), 1.6)?;
    let tapered = edgetaper(&gray_to_array(&input), &psf)?;
    let tapered_image = array_to_gray(&tapered)?;

    if let Some(path) = args.get(2) {
        DynamicImage::ImageLuma8(tapered_image.clone()).save(path)?;
    }

    println!(
        "edgetaper output: {}x{}",
        tapered_image.width(),
        tapered_image.height()
    );
    Ok(())
}

fn gray_to_array(input: &GrayImage) -> Array2<f32> {
    let width = input.width() as usize;
    let height = input.height() as usize;
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            output[[y, x]] = f32::from(input.get_pixel(x as u32, y as u32)[0]) / 255.0;
        }
    }
    output
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
