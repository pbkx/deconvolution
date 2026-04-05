use image::{DynamicImage, GrayAlphaImage, GrayImage, Luma, LumaA, Rgb, RgbImage, Rgba, RgbaImage};
use ndarray::{Array2, Array3, Axis};
use num_complex::Complex32;

use crate::core::color::sample_from_f32;
use crate::core::convert::PlanarImage;
use crate::core::fft::{fft2_forward_real, fft2_inverse_complex};
use crate::core::plan_cache::PlanCache;
use crate::core::regularizer::spectral_response_2d;
use crate::core::util::next_fast_len;
use crate::otf::{psf2otf, Transfer2D};
use crate::preprocess::normalize_range;
use crate::psf::{validate, Kernel2D};
use crate::{Error, Padding, RangePolicy, RegOperator2D, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct InverseFilter {
    stabilization_floor: f32,
    truncation_cutoff: f32,
    padding: Padding,
    range_policy: RangePolicy,
}

impl Default for InverseFilter {
    fn default() -> Self {
        Self {
            stabilization_floor: 1e-3,
            truncation_cutoff: 1e-2,
            padding: Padding::None,
            range_policy: RangePolicy::PreserveInput,
        }
    }
}

impl InverseFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stabilization_floor(mut self, value: f32) -> Self {
        self.stabilization_floor = value;
        self
    }

    pub fn truncation_cutoff(mut self, value: f32) -> Self {
        self.truncation_cutoff = value;
        self
    }

    pub fn padding(mut self, value: Padding) -> Self {
        self.padding = value;
        self
    }

    pub fn range_policy(mut self, value: RangePolicy) -> Self {
        self.range_policy = value;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RegularizedInverseFilter<'a> {
    lambda: f32,
    stabilization_floor: f32,
    padding: Padding,
    range_policy: RangePolicy,
    regularizer: Option<RegOperator2D<'a>>,
}

impl<'a> Default for RegularizedInverseFilter<'a> {
    fn default() -> Self {
        Self {
            lambda: 1e-2,
            stabilization_floor: 1e-3,
            padding: Padding::None,
            range_policy: RangePolicy::PreserveInput,
            regularizer: None,
        }
    }
}

impl<'a> RegularizedInverseFilter<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lambda(mut self, value: f32) -> Self {
        self.lambda = value;
        self
    }

    pub fn stabilization_floor(mut self, value: f32) -> Self {
        self.stabilization_floor = value;
        self
    }

    pub fn padding(mut self, value: Padding) -> Self {
        self.padding = value;
        self
    }

    pub fn range_policy(mut self, value: RangePolicy) -> Self {
        self.range_policy = value;
        self
    }

    pub fn regularizer(mut self, value: RegOperator2D<'a>) -> Self {
        self.regularizer = Some(value);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TikhonovInverseFilter {
    lambda: f32,
    stabilization_floor: f32,
    padding: Padding,
    range_policy: RangePolicy,
}

impl Default for TikhonovInverseFilter {
    fn default() -> Self {
        Self {
            lambda: 1e-2,
            stabilization_floor: 1e-3,
            padding: Padding::None,
            range_policy: RangePolicy::PreserveInput,
        }
    }
}

impl TikhonovInverseFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lambda(mut self, value: f32) -> Self {
        self.lambda = value;
        self
    }

    pub fn stabilization_floor(mut self, value: f32) -> Self {
        self.stabilization_floor = value;
        self
    }

    pub fn padding(mut self, value: Padding) -> Self {
        self.padding = value;
        self
    }

    pub fn range_policy(mut self, value: RangePolicy) -> Self {
        self.range_policy = value;
        self
    }
}

pub fn naive_inverse_filter(image: &DynamicImage, psf: &Kernel2D) -> Result<DynamicImage> {
    naive_inverse_filter_with(image, psf, &InverseFilter::new())
}

pub fn naive_inverse_filter_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &InverseFilter,
) -> Result<DynamicImage> {
    restore(image, psf, config, Mode::Naive)
}

pub fn inverse_filter(image: &DynamicImage, psf: &Kernel2D) -> Result<DynamicImage> {
    inverse_filter_with(image, psf, &InverseFilter::new())
}

pub fn inverse_filter_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &InverseFilter,
) -> Result<DynamicImage> {
    restore(image, psf, config, Mode::Stabilized)
}

pub fn truncated_inverse_filter(image: &DynamicImage, psf: &Kernel2D) -> Result<DynamicImage> {
    truncated_inverse_filter_with(image, psf, &InverseFilter::new())
}

pub fn truncated_inverse_filter_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &InverseFilter,
) -> Result<DynamicImage> {
    restore(image, psf, config, Mode::Truncated)
}

pub fn regularized_inverse_filter(image: &DynamicImage, psf: &Kernel2D) -> Result<DynamicImage> {
    regularized_inverse_filter_with(image, psf, &RegularizedInverseFilter::new())
}

pub fn regularized_inverse_filter_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &RegularizedInverseFilter<'_>,
) -> Result<DynamicImage> {
    restore_regularized(image, psf, config)
}

pub fn tikhonov_inverse_filter(image: &DynamicImage, psf: &Kernel2D) -> Result<DynamicImage> {
    tikhonov_inverse_filter_with(image, psf, &TikhonovInverseFilter::new())
}

pub fn tikhonov_inverse_filter_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &TikhonovInverseFilter,
) -> Result<DynamicImage> {
    restore_tikhonov(image, psf, config)
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Naive,
    Stabilized,
    Truncated,
}

fn restore(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &InverseFilter,
    mode: Mode,
) -> Result<DynamicImage> {
    validate(psf)?;
    validate_config(config)?;

    let planar = PlanarImage::from_dynamic(image)?;
    let (width_u32, height_u32) = planar.dimensions();
    let width = usize::try_from(width_u32).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(height_u32).map_err(|_| Error::DimensionMismatch)?;
    if width == 0 || height == 0 {
        return Err(Error::EmptyImage);
    }

    let restored_color = restore_color(planar.color(), psf, config, mode)?;
    rebuild_image(image, &restored_color)
}

fn restore_regularized(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &RegularizedInverseFilter<'_>,
) -> Result<DynamicImage> {
    validate(psf)?;
    validate_regularized_config(config)?;

    let planar = PlanarImage::from_dynamic(image)?;
    let (width_u32, height_u32) = planar.dimensions();
    let width = usize::try_from(width_u32).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(height_u32).map_err(|_| Error::DimensionMismatch)?;
    if width == 0 || height == 0 {
        return Err(Error::EmptyImage);
    }

    let restored_color = restore_color_regularized(planar.color(), psf, config)?;
    rebuild_image(image, &restored_color)
}

fn restore_tikhonov(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &TikhonovInverseFilter,
) -> Result<DynamicImage> {
    validate_tikhonov_config(config)?;

    let regularized = RegularizedInverseFilter::new()
        .lambda(config.lambda)
        .stabilization_floor(config.stabilization_floor)
        .padding(config.padding)
        .range_policy(config.range_policy)
        .regularizer(RegOperator2D::Identity);

    restore_regularized(image, psf, &regularized)
}

fn restore_color(
    color: &Array3<f32>,
    psf: &Kernel2D,
    config: &InverseFilter,
    mode: Mode,
) -> Result<Array3<f32>> {
    let shape = color.shape();
    if shape.len() != 3 {
        return Err(Error::DimensionMismatch);
    }
    let channels = shape[0];
    let height = shape[1];
    let width = shape[2];
    if channels == 0 || height == 0 || width == 0 {
        return Err(Error::EmptyImage);
    }

    let fft_dims = resolve_fft_dims((height, width), psf.dims(), config.padding)?;
    let otf = psf2otf(psf, fft_dims)?;

    let mut output = Array3::zeros((channels, height, width));
    for channel_idx in 0..channels {
        let channel = color.index_axis(Axis(0), channel_idx).to_owned();
        let restored = restore_channel(&channel, &otf, config, mode)?;
        for y in 0..height {
            for x in 0..width {
                output[[channel_idx, y, x]] = restored[[y, x]];
            }
        }
    }

    Ok(output)
}

fn restore_color_regularized(
    color: &Array3<f32>,
    psf: &Kernel2D,
    config: &RegularizedInverseFilter<'_>,
) -> Result<Array3<f32>> {
    let shape = color.shape();
    if shape.len() != 3 {
        return Err(Error::DimensionMismatch);
    }
    let channels = shape[0];
    let height = shape[1];
    let width = shape[2];
    if channels == 0 || height == 0 || width == 0 {
        return Err(Error::EmptyImage);
    }

    let fft_dims = resolve_fft_dims((height, width), psf.dims(), config.padding)?;
    let otf = psf2otf(psf, fft_dims)?;
    let regularizer = config.regularizer.unwrap_or(RegOperator2D::Laplacian);
    let regularizer_transfer = spectral_response_2d(regularizer, fft_dims)?;

    let mut output = Array3::zeros((channels, height, width));
    for channel_idx in 0..channels {
        let channel = color.index_axis(Axis(0), channel_idx).to_owned();
        let restored = restore_channel_regularized(
            &channel,
            otf.as_array(),
            &regularizer_transfer,
            config.lambda,
            config.stabilization_floor,
            config.range_policy,
        )?;
        for y in 0..height {
            for x in 0..width {
                output[[channel_idx, y, x]] = restored[[y, x]];
            }
        }
    }

    Ok(output)
}

fn restore_channel(
    input: &Array2<f32>,
    otf: &Transfer2D,
    config: &InverseFilter,
    mode: Mode,
) -> Result<Array2<f32>> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let (height, width) = input.dim();
    let padded = pad_to_dims(input, otf.dims())?;
    let mut cache = PlanCache::new();
    let mut spectrum = fft2_forward_real(&padded, &mut cache)?;
    invert_spectrum(&mut spectrum, otf.as_array(), config, mode)?;
    let restored = fft2_inverse_complex(&spectrum, &mut cache)?;
    let mut cropped = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            cropped[[y, x]] = restored[[y, x]];
        }
    }
    normalize_range(&cropped, config.range_policy)
}

fn restore_channel_regularized(
    input: &Array2<f32>,
    blur_transfer: &Array2<Complex32>,
    regularizer_transfer: &Array2<Complex32>,
    lambda: f32,
    stabilization_floor: f32,
    range_policy: RangePolicy,
) -> Result<Array2<f32>> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let (height, width) = input.dim();
    let padded = pad_to_dims(input, blur_transfer.dim())?;
    let mut cache = PlanCache::new();
    let mut spectrum = fft2_forward_real(&padded, &mut cache)?;
    apply_regularized_inverse(
        &mut spectrum,
        blur_transfer,
        regularizer_transfer,
        lambda,
        stabilization_floor,
    )?;
    let restored = fft2_inverse_complex(&spectrum, &mut cache)?;

    let mut cropped = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            cropped[[y, x]] = restored[[y, x]];
        }
    }

    normalize_range(&cropped, range_policy)
}

fn invert_spectrum(
    spectrum: &mut Array2<Complex32>,
    transfer: &Array2<Complex32>,
    config: &InverseFilter,
    mode: Mode,
) -> Result<()> {
    if spectrum.dim() != transfer.dim() {
        return Err(Error::DimensionMismatch);
    }

    for ((y, x), value) in spectrum.indexed_iter_mut() {
        let h = transfer[[y, x]];
        let restored = match mode {
            Mode::Naive => divide_naive(*value, h)?,
            Mode::Stabilized => divide_stabilized(*value, h, config.stabilization_floor)?,
            Mode::Truncated => divide_truncated(
                *value,
                h,
                config.stabilization_floor,
                config.truncation_cutoff,
            )?,
        };

        if !restored.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        *value = restored;
    }

    Ok(())
}

fn apply_regularized_inverse(
    spectrum: &mut Array2<Complex32>,
    blur_transfer: &Array2<Complex32>,
    regularizer_transfer: &Array2<Complex32>,
    lambda: f32,
    stabilization_floor: f32,
) -> Result<()> {
    if spectrum.dim() != blur_transfer.dim() || spectrum.dim() != regularizer_transfer.dim() {
        return Err(Error::DimensionMismatch);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !stabilization_floor.is_finite() || stabilization_floor <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    let denom_floor = stabilization_floor * stabilization_floor;
    for ((y, x), value) in spectrum.indexed_iter_mut() {
        let h = blur_transfer[[y, x]];
        let l = regularizer_transfer[[y, x]];
        let denom = (h.norm_sqr() + lambda * l.norm_sqr()).max(denom_floor);
        if !denom.is_finite() || denom <= 0.0 {
            return Err(Error::InvalidParameter);
        }

        let restored = h.conj() * *value / denom;
        if !restored.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        *value = restored;
    }

    Ok(())
}

fn divide_naive(value: Complex32, transfer: Complex32) -> Result<Complex32> {
    if transfer.norm() <= f32::EPSILON {
        return Err(Error::ConvergenceFailure);
    }
    Ok(value / transfer)
}

fn divide_stabilized(value: Complex32, transfer: Complex32, floor: f32) -> Result<Complex32> {
    if !floor.is_finite() || floor <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    let magnitude = transfer.norm();
    if magnitude <= floor {
        return Ok(value / clamped_transfer(transfer, floor));
    }
    Ok(value / transfer)
}

fn divide_truncated(
    value: Complex32,
    transfer: Complex32,
    floor: f32,
    cutoff: f32,
) -> Result<Complex32> {
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(Error::InvalidParameter);
    }

    let magnitude = transfer.norm();
    if magnitude < cutoff {
        return Ok(Complex32::new(0.0, 0.0));
    }

    if magnitude <= floor {
        return Ok(value / clamped_transfer(transfer, floor));
    }

    Ok(value / transfer)
}

fn clamped_transfer(transfer: Complex32, floor: f32) -> Complex32 {
    let magnitude = transfer.norm();
    if magnitude <= f32::EPSILON {
        return Complex32::new(floor, 0.0);
    }
    transfer * (floor / magnitude)
}

fn pad_to_dims(input: &Array2<f32>, dims: (usize, usize)) -> Result<Array2<f32>> {
    let (height, width) = input.dim();
    let (target_h, target_w) = dims;
    if target_h < height || target_w < width {
        return Err(Error::DimensionMismatch);
    }

    let mut padded = Array2::zeros((target_h, target_w));
    for y in 0..height {
        for x in 0..width {
            padded[[y, x]] = input[[y, x]];
        }
    }
    Ok(padded)
}

fn resolve_fft_dims(
    image_dims: (usize, usize),
    psf_dims: (usize, usize),
    padding: Padding,
) -> Result<(usize, usize)> {
    let (image_h, image_w) = image_dims;
    let (psf_h, psf_w) = psf_dims;
    if image_h == 0 || image_w == 0 || psf_h == 0 || psf_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let minimal_h = image_h
        .checked_add(psf_h)
        .and_then(|value| value.checked_sub(1))
        .ok_or(Error::InvalidParameter)?;
    let minimal_w = image_w
        .checked_add(psf_w)
        .and_then(|value| value.checked_sub(1))
        .ok_or(Error::InvalidParameter)?;

    match padding {
        Padding::None | Padding::Same => Ok((image_h, image_w)),
        Padding::Minimal => Ok((minimal_h, minimal_w)),
        Padding::NextFastLen => Ok((next_fast_len(minimal_h), next_fast_len(minimal_w))),
        Padding::Explicit2(height, width) => {
            if height == 0 || width == 0 {
                return Err(Error::InvalidParameter);
            }
            if height < image_h || width < image_w || height < psf_h || width < psf_w {
                return Err(Error::DimensionMismatch);
            }
            Ok((height, width))
        }
        Padding::Explicit3(_, _, _) => Err(Error::InvalidParameter),
    }
}

fn rebuild_image(source: &DynamicImage, color: &Array3<f32>) -> Result<DynamicImage> {
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
    let width_usize = usize::try_from(width).map_err(|_| Error::DimensionMismatch)?;
    let height_usize = usize::try_from(height).map_err(|_| Error::DimensionMismatch)?;

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
    let width_usize = usize::try_from(width).map_err(|_| Error::DimensionMismatch)?;
    let height_usize = usize::try_from(height).map_err(|_| Error::DimensionMismatch)?;

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
    let width_usize = usize::try_from(width).map_err(|_| Error::DimensionMismatch)?;
    let height_usize = usize::try_from(height).map_err(|_| Error::DimensionMismatch)?;

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
    let width_usize = usize::try_from(width).map_err(|_| Error::DimensionMismatch)?;
    let height_usize = usize::try_from(height).map_err(|_| Error::DimensionMismatch)?;

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
    let width = usize::try_from(width).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(height).map_err(|_| Error::DimensionMismatch)?;
    if color.shape() != [channels, height, width] {
        return Err(Error::DimensionMismatch);
    }
    if color.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(())
}

fn validate_config(config: &InverseFilter) -> Result<()> {
    if !config.stabilization_floor.is_finite() || config.stabilization_floor <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !config.truncation_cutoff.is_finite() || config.truncation_cutoff < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if let Padding::Explicit3(_, _, _) = config.padding {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_regularized_config(config: &RegularizedInverseFilter<'_>) -> Result<()> {
    if !config.lambda.is_finite() || config.lambda < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !config.stabilization_floor.is_finite() || config.stabilization_floor <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if let Padding::Explicit3(_, _, _) = config.padding {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_tikhonov_config(config: &TikhonovInverseFilter) -> Result<()> {
    if !config.lambda.is_finite() || config.lambda < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !config.stabilization_floor.is_finite() || config.stabilization_floor <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if let Padding::Explicit3(_, _, _) = config.padding {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}
