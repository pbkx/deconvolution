use image::{DynamicImage, GrayImage, Luma};
use ndarray::{Array2, Array3, Axis};

use crate::core::color::{sample_from_f32, sample_to_f32};
use crate::{Error, Kernel2D, Result};

pub(crate) fn array2_to_dynamic(input: &Array2<f32>) -> Result<DynamicImage> {
    let gray = array2_to_gray(input)?;
    Ok(DynamicImage::ImageLuma8(gray))
}

pub(crate) fn dynamic_to_array2(image: &DynamicImage) -> Result<Array2<f32>> {
    match image {
        DynamicImage::ImageLuma8(gray) => gray_to_array2(gray),
        DynamicImage::ImageLumaA8(gray_alpha) => {
            let width =
                usize::try_from(gray_alpha.width()).map_err(|_| Error::DimensionMismatch)?;
            let height =
                usize::try_from(gray_alpha.height()).map_err(|_| Error::DimensionMismatch)?;
            if width == 0 || height == 0 {
                return Err(Error::EmptyImage);
            }

            let mut output = Array2::zeros((height, width));
            for y in 0..height {
                let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
                for x in 0..width {
                    let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
                    output[[y, x]] = sample_to_f32(gray_alpha.get_pixel(x_u32, y_u32)[0]) / 255.0;
                }
            }
            Ok(output)
        }
        _ => Err(Error::UnsupportedPixelType),
    }
}

pub(crate) fn gray_to_array2(image: &GrayImage) -> Result<Array2<f32>> {
    let width = usize::try_from(image.width()).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(image.height()).map_err(|_| Error::DimensionMismatch)?;
    if width == 0 || height == 0 {
        return Err(Error::EmptyImage);
    }

    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
            output[[y, x]] = sample_to_f32(image.get_pixel(x_u32, y_u32)[0]) / 255.0;
        }
    }
    Ok(output)
}

pub(crate) fn array2_to_gray(input: &Array2<f32>) -> Result<GrayImage> {
    validate_array2(input)?;
    let (height, width) = input.dim();
    let width_u32 = u32::try_from(width).map_err(|_| Error::DimensionMismatch)?;
    let height_u32 = u32::try_from(height).map_err(|_| Error::DimensionMismatch)?;
    let mut image = GrayImage::new(width_u32, height_u32);

    for y in 0..height {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
            let scaled = input[[y, x]].clamp(0.0, 1.0) * 255.0;
            let value = sample_from_f32(scaled)?;
            image.put_pixel(x_u32, y_u32, Luma([value]));
        }
    }

    Ok(image)
}

pub(crate) fn kernel2_from_array(input: &Array2<f32>) -> Result<Kernel2D> {
    validate_array2(input)?;
    Kernel2D::new(input.as_standard_layout().to_owned())
}

pub(crate) fn kernel3_to_projected_kernel2(input: &Array3<f32>) -> Result<Kernel2D> {
    validate_array3(input)?;
    let mut projected = input.sum_axis(Axis(0));
    let sum = projected.sum();
    if !sum.is_finite() || sum.abs() <= f32::EPSILON {
        return Err(Error::InvalidPsf);
    }
    for value in &mut projected {
        *value /= sum;
    }
    Kernel2D::new(projected)
}

pub(crate) fn validate_array2(input: &Array2<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn validate_array3(input: &Array3<f32>) -> Result<()> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}
