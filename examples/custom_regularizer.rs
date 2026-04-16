use deconvolution::psf::gaussian2d;
use deconvolution::simulate::{add_gaussian_noise, blur, checkerboard_2d};
use deconvolution::{regularized_inverse_filter_with, RegOperator2D, RegularizedInverseFilter};
use image::{DynamicImage, GrayImage, Luma};
use ndarray::{array, Array2};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let sharp = checkerboard_2d((128, 128), 8, 0.0, 1.0)?;
    let psf = gaussian2d((9, 9), 1.5)?;
    let blurred = blur(&sharp, &psf)?;
    let degraded = add_gaussian_noise(&blurred, 0.03, 777)?;
    let degraded_image = DynamicImage::ImageLuma8(array_to_gray(&degraded)?);

    let reg_kernel = deconvolution::Kernel2D::new(array![
        [0.0_f32, -1.0_f32, 0.0_f32],
        [-1.0_f32, 4.0_f32, -1.0_f32],
        [0.0_f32, -1.0_f32, 0.0_f32]
    ])?;
    let restored = regularized_inverse_filter_with(
        &degraded_image,
        &psf,
        &RegularizedInverseFilter::new()
            .lambda(0.015)
            .regularizer(RegOperator2D::CustomKernel(&reg_kernel))
            .stabilization_floor(1e-3),
    )?;

    println!(
        "custom regularizer output: {}x{}",
        restored.width(),
        restored.height()
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
