use image::{
    DynamicImage, GenericImageView, GrayAlphaImage, GrayImage, Luma, LumaA, Rgb, RgbImage, Rgba,
    RgbaImage,
};
use ndarray::{Array2, Array3};

use super::color::{sample_from_f32, sample_to_f32, PixelLayout};
use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlanarImage {
    width: u32,
    height: u32,
    layout: PixelLayout,
    color: Array3<f32>,
    alpha: Option<Array2<f32>>,
}

impl PlanarImage {
    pub(crate) fn from_dynamic(image: &DynamicImage) -> Result<Self> {
        match image {
            DynamicImage::ImageLuma8(gray) => Self::from_gray_view(gray),
            DynamicImage::ImageLumaA8(gray_alpha) => Self::from_gray_alpha_view(gray_alpha),
            DynamicImage::ImageRgb8(rgb) => Self::from_rgb_view(rgb),
            DynamicImage::ImageRgba8(rgba) => Self::from_rgba_view(rgba),
            _ => Err(Error::UnsupportedPixelType),
        }
    }

    pub(crate) fn from_gray_view<I>(image: &I) -> Result<Self>
    where
        I: GenericImageView<Pixel = Luma<u8>>,
    {
        let (width, height) = image.dimensions();
        let mut color = new_color(PixelLayout::Gray, width, height)?;

        for (x, y, pixel) in image.pixels() {
            let x = to_usize(x)?;
            let y = to_usize(y)?;
            color[[0, y, x]] = sample_to_f32(pixel[0]);
        }

        Self::new(width, height, PixelLayout::Gray, color, None)
    }

    pub(crate) fn from_gray_alpha_view<I>(image: &I) -> Result<Self>
    where
        I: GenericImageView<Pixel = LumaA<u8>>,
    {
        let (width, height) = image.dimensions();
        let mut color = new_color(PixelLayout::GrayAlpha, width, height)?;
        let mut alpha = new_alpha(width, height)?;

        for (x, y, pixel) in image.pixels() {
            let x = to_usize(x)?;
            let y = to_usize(y)?;
            color[[0, y, x]] = sample_to_f32(pixel[0]);
            alpha[[y, x]] = sample_to_f32(pixel[1]);
        }

        Self::new(width, height, PixelLayout::GrayAlpha, color, Some(alpha))
    }

    pub(crate) fn from_rgb_view<I>(image: &I) -> Result<Self>
    where
        I: GenericImageView<Pixel = Rgb<u8>>,
    {
        let (width, height) = image.dimensions();
        let mut color = new_color(PixelLayout::Rgb, width, height)?;

        for (x, y, pixel) in image.pixels() {
            let x = to_usize(x)?;
            let y = to_usize(y)?;
            color[[0, y, x]] = sample_to_f32(pixel[0]);
            color[[1, y, x]] = sample_to_f32(pixel[1]);
            color[[2, y, x]] = sample_to_f32(pixel[2]);
        }

        Self::new(width, height, PixelLayout::Rgb, color, None)
    }

    pub(crate) fn from_rgba_view<I>(image: &I) -> Result<Self>
    where
        I: GenericImageView<Pixel = Rgba<u8>>,
    {
        let (width, height) = image.dimensions();
        let mut color = new_color(PixelLayout::Rgba, width, height)?;
        let mut alpha = new_alpha(width, height)?;

        for (x, y, pixel) in image.pixels() {
            let x = to_usize(x)?;
            let y = to_usize(y)?;
            color[[0, y, x]] = sample_to_f32(pixel[0]);
            color[[1, y, x]] = sample_to_f32(pixel[1]);
            color[[2, y, x]] = sample_to_f32(pixel[2]);
            alpha[[y, x]] = sample_to_f32(pixel[3]);
        }

        Self::new(width, height, PixelLayout::Rgba, color, Some(alpha))
    }

    pub(crate) fn to_gray_like<I>(&self, like: &I) -> Result<GrayImage>
    where
        I: GenericImageView<Pixel = Luma<u8>>,
    {
        if self.layout != PixelLayout::Gray {
            return Err(Error::UnsupportedPixelType);
        }

        let mut output = like.buffer_with_dimensions(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let x_usize = to_usize(x)?;
                let y_usize = to_usize(y)?;
                let value = sample_from_f32(self.color[[0, y_usize, x_usize]])?;
                output.put_pixel(x, y, Luma([value]));
            }
        }
        Ok(output)
    }

    pub(crate) fn to_gray_alpha_like<I>(&self, like: &I) -> Result<GrayAlphaImage>
    where
        I: GenericImageView<Pixel = LumaA<u8>>,
    {
        if self.layout != PixelLayout::GrayAlpha {
            return Err(Error::UnsupportedPixelType);
        }

        let alpha = self.alpha.as_ref().ok_or(Error::DimensionMismatch)?;
        let mut output = like.buffer_with_dimensions(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let x_usize = to_usize(x)?;
                let y_usize = to_usize(y)?;
                let luma = sample_from_f32(self.color[[0, y_usize, x_usize]])?;
                let alpha = sample_from_f32(alpha[[y_usize, x_usize]])?;
                output.put_pixel(x, y, LumaA([luma, alpha]));
            }
        }
        Ok(output)
    }

    pub(crate) fn to_rgb_like<I>(&self, like: &I) -> Result<RgbImage>
    where
        I: GenericImageView<Pixel = Rgb<u8>>,
    {
        if self.layout != PixelLayout::Rgb {
            return Err(Error::UnsupportedPixelType);
        }

        let mut output = like.buffer_with_dimensions(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let x_usize = to_usize(x)?;
                let y_usize = to_usize(y)?;
                let r = sample_from_f32(self.color[[0, y_usize, x_usize]])?;
                let g = sample_from_f32(self.color[[1, y_usize, x_usize]])?;
                let b = sample_from_f32(self.color[[2, y_usize, x_usize]])?;
                output.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        Ok(output)
    }

    pub(crate) fn to_rgba_like<I>(&self, like: &I) -> Result<RgbaImage>
    where
        I: GenericImageView<Pixel = Rgba<u8>>,
    {
        if self.layout != PixelLayout::Rgba {
            return Err(Error::UnsupportedPixelType);
        }

        let alpha = self.alpha.as_ref().ok_or(Error::DimensionMismatch)?;
        let mut output = like.buffer_with_dimensions(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let x_usize = to_usize(x)?;
                let y_usize = to_usize(y)?;
                let r = sample_from_f32(self.color[[0, y_usize, x_usize]])?;
                let g = sample_from_f32(self.color[[1, y_usize, x_usize]])?;
                let b = sample_from_f32(self.color[[2, y_usize, x_usize]])?;
                let a = sample_from_f32(alpha[[y_usize, x_usize]])?;
                output.put_pixel(x, y, Rgba([r, g, b, a]));
            }
        }
        Ok(output)
    }

    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub(crate) fn color(&self) -> &Array3<f32> {
        &self.color
    }

    pub(crate) fn alpha(&self) -> Option<&Array2<f32>> {
        self.alpha.as_ref()
    }

    pub(crate) fn layout(&self) -> PixelLayout {
        self.layout
    }

    fn new(
        width: u32,
        height: u32,
        layout: PixelLayout,
        color: Array3<f32>,
        alpha: Option<Array2<f32>>,
    ) -> Result<Self> {
        let width_usize = to_usize(width)?;
        let height_usize = to_usize(height)?;

        if color.shape() != [layout.color_channels(), height_usize, width_usize] {
            return Err(Error::DimensionMismatch);
        }
        if color.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFiniteInput);
        }

        if layout.has_alpha() {
            let alpha = alpha.ok_or(Error::DimensionMismatch)?;
            if alpha.shape() != [height_usize, width_usize] {
                return Err(Error::DimensionMismatch);
            }
            if alpha.iter().any(|value| !value.is_finite()) {
                return Err(Error::NonFiniteInput);
            }

            Ok(Self {
                width,
                height,
                layout,
                color: color.as_standard_layout().to_owned(),
                alpha: Some(alpha.as_standard_layout().to_owned()),
            })
        } else {
            if alpha.is_some() {
                return Err(Error::DimensionMismatch);
            }
            Ok(Self {
                width,
                height,
                layout,
                color: color.as_standard_layout().to_owned(),
                alpha: None,
            })
        }
    }
}

pub(crate) fn rebuild_dynamic_like(
    source: &DynamicImage,
    color: &Array3<f32>,
) -> Result<DynamicImage> {
    match source {
        DynamicImage::ImageLuma8(luma) => {
            let restored = rebuild_luma(luma.width(), luma.height(), color)?;
            Ok(DynamicImage::ImageLuma8(restored))
        }
        DynamicImage::ImageLumaA8(luma_alpha) => {
            let restored =
                rebuild_luma_alpha(luma_alpha.width(), luma_alpha.height(), color, luma_alpha)?;
            Ok(DynamicImage::ImageLumaA8(restored))
        }
        DynamicImage::ImageRgb8(rgb) => {
            let restored = rebuild_rgb(rgb.width(), rgb.height(), color)?;
            Ok(DynamicImage::ImageRgb8(restored))
        }
        DynamicImage::ImageRgba8(rgba) => {
            let restored = rebuild_rgba(rgba.width(), rgba.height(), color, rgba)?;
            Ok(DynamicImage::ImageRgba8(restored))
        }
        _ => Err(Error::UnsupportedPixelType),
    }
}

fn rebuild_luma(width: u32, height: u32, color: &Array3<f32>) -> Result<GrayImage> {
    verify_color_shape(color, 1, width, height)?;
    let width_usize = to_usize(width)?;
    let height_usize = to_usize(height)?;

    let mut output = GrayImage::new(width, height);
    for y in 0..height_usize {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width_usize {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
            let l = sample_from_f32(color[[0, y, x]])?;
            output.put_pixel(x_u32, y_u32, Luma([l]));
        }
    }
    Ok(output)
}

fn rebuild_luma_alpha(
    width: u32,
    height: u32,
    color: &Array3<f32>,
    source: &GrayAlphaImage,
) -> Result<GrayAlphaImage> {
    verify_color_shape(color, 1, width, height)?;
    let width_usize = to_usize(width)?;
    let height_usize = to_usize(height)?;

    let mut output = GrayAlphaImage::new(width, height);
    for y in 0..height_usize {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width_usize {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
            let l = sample_from_f32(color[[0, y, x]])?;
            let a = source.get_pixel(x_u32, y_u32)[1];
            output.put_pixel(x_u32, y_u32, LumaA([l, a]));
        }
    }
    Ok(output)
}

fn rebuild_rgb(width: u32, height: u32, color: &Array3<f32>) -> Result<RgbImage> {
    verify_color_shape(color, 3, width, height)?;
    let width_usize = to_usize(width)?;
    let height_usize = to_usize(height)?;

    let mut output = RgbImage::new(width, height);
    for y in 0..height_usize {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width_usize {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
            let r = sample_from_f32(color[[0, y, x]])?;
            let g = sample_from_f32(color[[1, y, x]])?;
            let b = sample_from_f32(color[[2, y, x]])?;
            output.put_pixel(x_u32, y_u32, Rgb([r, g, b]));
        }
    }
    Ok(output)
}

fn rebuild_rgba(
    width: u32,
    height: u32,
    color: &Array3<f32>,
    source: &RgbaImage,
) -> Result<RgbaImage> {
    verify_color_shape(color, 3, width, height)?;
    let width_usize = to_usize(width)?;
    let height_usize = to_usize(height)?;

    let mut output = RgbaImage::new(width, height);
    for y in 0..height_usize {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width_usize {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
            let r = sample_from_f32(color[[0, y, x]])?;
            let g = sample_from_f32(color[[1, y, x]])?;
            let b = sample_from_f32(color[[2, y, x]])?;
            let a = source.get_pixel(x_u32, y_u32)[3];
            output.put_pixel(x_u32, y_u32, Rgba([r, g, b, a]));
        }
    }
    Ok(output)
}

fn verify_color_shape(color: &Array3<f32>, channels: usize, width: u32, height: u32) -> Result<()> {
    let width = to_usize(width)?;
    let height = to_usize(height)?;
    if color.shape() != [channels, height, width] {
        return Err(Error::DimensionMismatch);
    }
    if color.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn to_usize(value: u32) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::DimensionMismatch)
}

fn new_color(layout: PixelLayout, width: u32, height: u32) -> Result<Array3<f32>> {
    let width = to_usize(width)?;
    let height = to_usize(height)?;
    Ok(Array3::zeros((layout.color_channels(), height, width)))
}

fn new_alpha(width: u32, height: u32) -> Result<Array2<f32>> {
    let width = to_usize(width)?;
    let height = to_usize(height)?;
    Ok(Array2::zeros((height, width)))
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, GenericImageView, GrayAlphaImage, GrayImage, RgbImage, RgbaImage};

    use super::{rebuild_dynamic_like, PlanarImage};
    use crate::Error;

    #[test]
    fn grayscale_roundtrip_preserves_values() {
        let image = GrayImage::from_raw(3, 2, vec![0, 64, 128, 255, 32, 16]).unwrap();
        let planar = PlanarImage::from_gray_view(&image).unwrap();
        let rebuilt = planar.to_gray_like(&image).unwrap();
        assert_eq!(rebuilt, image);
    }

    #[test]
    fn rgb_roundtrip_preserves_values() {
        let image =
            RgbImage::from_raw(2, 2, vec![1, 2, 3, 4, 5, 6, 250, 240, 230, 0, 1, 2]).unwrap();
        let planar = PlanarImage::from_rgb_view(&image).unwrap();
        let rebuilt = planar.to_rgb_like(&image).unwrap();
        assert_eq!(rebuilt, image);
    }

    #[test]
    fn rgba_roundtrip_preserves_alpha_exactly() {
        let image = RgbaImage::from_raw(
            2,
            2,
            vec![10, 20, 30, 40, 1, 2, 3, 255, 90, 91, 92, 0, 7, 8, 9, 128],
        )
        .unwrap();
        let planar = PlanarImage::from_rgba_view(&image).unwrap();
        let rebuilt = planar.to_rgba_like(&image).unwrap();
        assert_eq!(rebuilt, image);
    }

    #[test]
    fn gray_alpha_roundtrip_preserves_alpha_exactly() {
        let image = GrayAlphaImage::from_raw(2, 2, vec![10, 40, 1, 255, 90, 0, 7, 128]).unwrap();
        let planar = PlanarImage::from_gray_alpha_view(&image).unwrap();
        let rebuilt = planar.to_gray_alpha_like(&image).unwrap();
        assert_eq!(rebuilt, image);
    }

    #[test]
    fn dynamic_rebuild_preserves_u8_variants() {
        let luma = GrayImage::from_raw(2, 2, vec![0, 64, 128, 255]).unwrap();
        let source = DynamicImage::ImageLuma8(luma.clone());
        let planar = PlanarImage::from_dynamic(&source).unwrap();
        match rebuild_dynamic_like(&source, planar.color()).unwrap() {
            DynamicImage::ImageLuma8(output) => assert_eq!(output, luma),
            _ => panic!("expected luma8"),
        }

        let luma_alpha =
            GrayAlphaImage::from_raw(2, 2, vec![0, 255, 64, 128, 128, 64, 255, 0]).unwrap();
        let source = DynamicImage::ImageLumaA8(luma_alpha.clone());
        let planar = PlanarImage::from_dynamic(&source).unwrap();
        match rebuild_dynamic_like(&source, planar.color()).unwrap() {
            DynamicImage::ImageLumaA8(output) => assert_eq!(output, luma_alpha),
            _ => panic!("expected lumaA8"),
        }

        let rgb = RgbImage::from_raw(2, 2, vec![1, 2, 3, 4, 5, 6, 250, 240, 230, 0, 1, 2]).unwrap();
        let source = DynamicImage::ImageRgb8(rgb.clone());
        let planar = PlanarImage::from_dynamic(&source).unwrap();
        match rebuild_dynamic_like(&source, planar.color()).unwrap() {
            DynamicImage::ImageRgb8(output) => assert_eq!(output, rgb),
            _ => panic!("expected rgb8"),
        }

        let rgba = RgbaImage::from_raw(
            2,
            2,
            vec![10, 20, 30, 40, 1, 2, 3, 255, 90, 91, 92, 0, 7, 8, 9, 128],
        )
        .unwrap();
        let source = DynamicImage::ImageRgba8(rgba.clone());
        let planar = PlanarImage::from_dynamic(&source).unwrap();
        match rebuild_dynamic_like(&source, planar.color()).unwrap() {
            DynamicImage::ImageRgba8(output) => assert_eq!(output, rgba),
            _ => panic!("expected rgba8"),
        }
    }

    #[test]
    fn dimensions_are_preserved() {
        let image = RgbaImage::new(7, 5);
        let planar = PlanarImage::from_rgba_view(&image).unwrap();
        assert_eq!(planar.dimensions(), (7, 5));
        assert_eq!(planar.color().shape(), [3, 5, 7]);
        assert_eq!(planar.alpha().unwrap().shape(), [5, 7]);
    }

    #[test]
    fn subimage_views_roundtrip() {
        let mut image = RgbaImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                image.put_pixel(x, y, image::Rgba([x as u8, y as u8, (x + y) as u8, 200]));
            }
        }

        let view = image.view(1, 1, 2, 3);
        let planar = PlanarImage::from_rgba_view(&*view).unwrap();
        let rebuilt = planar.to_rgba_like(&*view).unwrap();

        assert_eq!(rebuilt.dimensions(), (2, 3));
        for y in 0..3 {
            for x in 0..2 {
                assert_eq!(*rebuilt.get_pixel(x, y), view.get_pixel(x, y));
            }
        }
    }

    #[test]
    fn unsupported_dynamic_layout_is_rejected() {
        let image =
            DynamicImage::ImageLuma16(image::ImageBuffer::from_raw(1, 1, vec![1024_u16]).unwrap());
        let error = PlanarImage::from_dynamic(&image).unwrap_err();
        assert_eq!(error, Error::UnsupportedPixelType);
    }
}
