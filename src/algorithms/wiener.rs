use image::{DynamicImage, GrayAlphaImage, GrayImage, Luma, LumaA, Rgb, RgbImage, Rgba, RgbaImage};
use ndarray::{Array2, Array3, Axis};
use num_complex::Complex32;

use crate::core::color::sample_from_f32;
use crate::core::convert::PlanarImage;
use crate::core::fft::{fft2_forward_real, fft2_inverse_complex};
use crate::core::plan_cache::PlanCache;
use crate::core::util::next_fast_len;
use crate::otf::{psf2otf, Transfer2D};
use crate::preprocess::normalize_range;
use crate::psf::{validate, Kernel2D};
use crate::{Boundary, ChannelMode, Error, Padding, RangePolicy, Result, SolveReport, StopReason};

#[derive(Debug, Clone, PartialEq)]
pub struct Wiener {
    nsr: f32,
    noise_autocorr: Option<Transfer2D>,
    image_autocorr: Option<Transfer2D>,
    boundary: Boundary,
    padding: Padding,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for Wiener {
    fn default() -> Self {
        Self {
            nsr: 1e-2,
            noise_autocorr: None,
            image_autocorr: None,
            boundary: Boundary::Reflect,
            padding: Padding::None,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: false,
        }
    }
}

impl Wiener {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn nsr(mut self, value: f32) -> Self {
        self.nsr = value;
        self
    }

    pub fn noise_autocorr(mut self, value: Transfer2D) -> Self {
        self.noise_autocorr = Some(value);
        self
    }

    pub fn image_autocorr(mut self, value: Transfer2D) -> Self {
        self.image_autocorr = Some(value);
        self
    }

    pub fn boundary(mut self, value: Boundary) -> Self {
        self.boundary = value;
        self
    }

    pub fn padding(mut self, value: Padding) -> Self {
        self.padding = value;
        self
    }

    pub fn channel_mode(mut self, value: ChannelMode) -> Self {
        self.channel_mode = value;
        self
    }

    pub fn range_policy(mut self, value: RangePolicy) -> Self {
        self.range_policy = value;
        self
    }

    pub fn collect_history(mut self, value: bool) -> Self {
        self.collect_history = value;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnsupervisedWiener {
    initial_nsr: f32,
    min_nsr: f32,
    max_iterations: usize,
    min_iterations: usize,
    tolerance: f32,
    boundary: Boundary,
    padding: Padding,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for UnsupervisedWiener {
    fn default() -> Self {
        Self {
            initial_nsr: 1e-2,
            min_nsr: 1e-8,
            max_iterations: 30,
            min_iterations: 3,
            tolerance: 1e-3,
            boundary: Boundary::Reflect,
            padding: Padding::None,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: false,
        }
    }
}

impl UnsupervisedWiener {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn initial_nsr(mut self, value: f32) -> Self {
        self.initial_nsr = value;
        self
    }

    pub fn min_nsr(mut self, value: f32) -> Self {
        self.min_nsr = value;
        self
    }

    pub fn max_iterations(mut self, value: usize) -> Self {
        self.max_iterations = value;
        self
    }

    pub fn min_iterations(mut self, value: usize) -> Self {
        self.min_iterations = value;
        self
    }

    pub fn tolerance(mut self, value: f32) -> Self {
        self.tolerance = value;
        self
    }

    pub fn boundary(mut self, value: Boundary) -> Self {
        self.boundary = value;
        self
    }

    pub fn padding(mut self, value: Padding) -> Self {
        self.padding = value;
        self
    }

    pub fn channel_mode(mut self, value: ChannelMode) -> Self {
        self.channel_mode = value;
        self
    }

    pub fn range_policy(mut self, value: RangePolicy) -> Self {
        self.range_policy = value;
        self
    }

    pub fn collect_history(mut self, value: bool) -> Self {
        self.collect_history = value;
        self
    }
}

pub fn wiener(image: &DynamicImage, psf: &Kernel2D) -> Result<DynamicImage> {
    wiener_with(image, psf, &Wiener::new())
}

pub fn wiener_with(image: &DynamicImage, psf: &Kernel2D, config: &Wiener) -> Result<DynamicImage> {
    validate(psf)?;
    validate_config(config)?;

    let planar = PlanarImage::from_dynamic(image)?;
    let (width_u32, height_u32) = planar.dimensions();
    let width = usize::try_from(width_u32).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(height_u32).map_err(|_| Error::DimensionMismatch)?;
    if width == 0 || height == 0 {
        return Err(Error::EmptyImage);
    }

    let fft_dims = resolve_fft_dims((height, width), psf.dims(), config.padding)?;
    let blur_transfer = psf2otf(psf, fft_dims)?.into_inner();
    let correlation = resolve_correlation_form(config, fft_dims)?;

    let (restored_color, stats) = restore_color(
        planar.color(),
        planar.alpha(),
        &blur_transfer,
        correlation,
        config,
    )?;
    if config.collect_history {
        validate_stats(&stats)?;
    }

    rebuild_image(image, &restored_color)
}

pub fn unsupervised_wiener(
    image: &DynamicImage,
    psf: &Kernel2D,
) -> Result<(DynamicImage, SolveReport)> {
    unsupervised_wiener_with(image, psf, &UnsupervisedWiener::new())
}

pub fn unsupervised_wiener_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &UnsupervisedWiener,
) -> Result<(DynamicImage, SolveReport)> {
    validate(psf)?;
    validate_unsupervised_config(config)?;

    let planar = PlanarImage::from_dynamic(image)?;
    let (width_u32, height_u32) = planar.dimensions();
    let width = usize::try_from(width_u32).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(height_u32).map_err(|_| Error::DimensionMismatch)?;
    if width == 0 || height == 0 {
        return Err(Error::EmptyImage);
    }

    let fft_dims = resolve_fft_dims((height, width), psf.dims(), config.padding)?;
    let blur_transfer = psf2otf(psf, fft_dims)?.into_inner();

    let mut nsr = config.initial_nsr.max(config.min_nsr);
    let mut objective_history = Vec::with_capacity(config.max_iterations);
    let mut residual_history = Vec::with_capacity(config.max_iterations);
    let mut stop_reason = StopReason::MaxIterations;
    let mut iterations = 0usize;

    for iteration in 0..config.max_iterations {
        iterations = iteration + 1;
        let iteration_config = unsupervised_iteration_config(config, nsr, true);
        let correlation = resolve_correlation_form(&iteration_config, fft_dims)?;
        let (_, stats) = restore_color(
            planar.color(),
            planar.alpha(),
            &blur_transfer,
            correlation,
            &iteration_config,
        )?;
        validate_stats(&stats)?;

        let estimated_nsr = estimate_nsr_from_stats(&stats, config.min_nsr)?;
        let relative_update = relative_change(nsr, estimated_nsr)?;
        objective_history.push(estimated_nsr);
        residual_history.push(relative_update);
        nsr = estimated_nsr;

        let step = iteration + 1;
        if step >= config.min_iterations && relative_update <= config.tolerance {
            stop_reason = StopReason::RelativeUpdate;
            break;
        }
    }

    let final_config = unsupervised_iteration_config(config, nsr, config.collect_history);
    let correlation = resolve_correlation_form(&final_config, fft_dims)?;
    let (restored_color, stats) = restore_color(
        planar.color(),
        planar.alpha(),
        &blur_transfer,
        correlation,
        &final_config,
    )?;
    if config.collect_history {
        validate_stats(&stats)?;
    }

    let objective_history = if config.collect_history {
        objective_history
    } else {
        Vec::new()
    };
    let residual_history = if config.collect_history {
        residual_history
    } else {
        Vec::new()
    };
    let report = SolveReport {
        iterations,
        stop_reason,
        objective_history,
        residual_history,
        estimated_nsr: Some(nsr),
    };

    let restored = rebuild_image(image, &restored_color)?;
    Ok((restored, report))
}

#[derive(Debug, Clone, Copy)]
enum CorrelationForm<'a> {
    ScalarNsr(f32),
    Ratio {
        noise: &'a Array2<Complex32>,
        image: &'a Array2<Complex32>,
    },
}

#[derive(Debug, Clone, Copy)]
struct ChannelStats {
    mean_denom: f32,
    mean_filter_norm: f32,
    signal_power: f32,
    noise_power: f32,
}

fn resolve_correlation_form<'a>(
    config: &'a Wiener,
    dims: (usize, usize),
) -> Result<CorrelationForm<'a>> {
    match (&config.noise_autocorr, &config.image_autocorr) {
        (None, None) => {
            if !config.nsr.is_finite() || config.nsr < 0.0 {
                return Err(Error::InvalidParameter);
            }
            Ok(CorrelationForm::ScalarNsr(config.nsr))
        }
        (Some(noise), Some(image)) => {
            if noise.dims() != dims || image.dims() != dims {
                return Err(Error::DimensionMismatch);
            }
            Ok(CorrelationForm::Ratio {
                noise: noise.as_array(),
                image: image.as_array(),
            })
        }
        _ => Err(Error::InvalidParameter),
    }
}

fn restore_color(
    color: &Array3<f32>,
    alpha: Option<&Array2<f32>>,
    blur_transfer: &Array2<Complex32>,
    correlation: CorrelationForm<'_>,
    config: &Wiener,
) -> Result<(Array3<f32>, Vec<ChannelStats>)> {
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

    let mut stats = Vec::new();
    let mut cache = PlanCache::new();
    let restored = match config.channel_mode {
        ChannelMode::Independent | ChannelMode::IgnoreAlpha => restore_independent(
            color,
            blur_transfer,
            correlation,
            config,
            &mut stats,
            &mut cache,
        )?,
        ChannelMode::LumaOnly => restore_luma_only(
            color,
            blur_transfer,
            correlation,
            config,
            &mut stats,
            &mut cache,
        )?,
        ChannelMode::PremultipliedAlpha => restore_premultiplied(
            color,
            alpha,
            blur_transfer,
            correlation,
            config,
            &mut stats,
            &mut cache,
        )?,
    };

    Ok((restored, stats))
}

fn restore_independent(
    color: &Array3<f32>,
    blur_transfer: &Array2<Complex32>,
    correlation: CorrelationForm<'_>,
    config: &Wiener,
    stats: &mut Vec<ChannelStats>,
    cache: &mut PlanCache,
) -> Result<Array3<f32>> {
    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    let mut output = Array3::zeros((channels, height, width));

    for channel_idx in 0..channels {
        let channel = color.index_axis(Axis(0), channel_idx).to_owned();
        let (restored, channel_stats) =
            restore_channel(&channel, blur_transfer, correlation, config, cache)?;
        if config.collect_history {
            stats.push(channel_stats);
        }
        output
            .index_axis_mut(Axis(0), channel_idx)
            .assign(&restored);
    }

    Ok(output)
}

fn restore_luma_only(
    color: &Array3<f32>,
    blur_transfer: &Array2<Complex32>,
    correlation: CorrelationForm<'_>,
    config: &Wiener,
    stats: &mut Vec<ChannelStats>,
    cache: &mut PlanCache,
) -> Result<Array3<f32>> {
    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    if channels == 1 {
        return restore_independent(color, blur_transfer, correlation, config, stats, cache);
    }

    let mut luma = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let r = color[[0, y, x]];
            let g = color[[1, y, x]];
            let b = color[[2, y, x]];
            luma[[y, x]] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    }

    let (restored_luma, channel_stats) =
        restore_channel(&luma, blur_transfer, correlation, config, cache)?;
    if config.collect_history {
        stats.push(channel_stats);
    }

    let mut output = Array3::zeros((channels, height, width));
    for y in 0..height {
        for x in 0..width {
            let delta = restored_luma[[y, x]] - luma[[y, x]];
            for c in 0..channels {
                output[[c, y, x]] = color[[c, y, x]] + delta;
            }
        }
    }

    for c in 0..channels {
        let channel = output.index_axis(Axis(0), c).to_owned();
        let normalized = normalize_range(&channel, config.range_policy)?;
        output.index_axis_mut(Axis(0), c).assign(&normalized);
    }

    Ok(output)
}

fn restore_premultiplied(
    color: &Array3<f32>,
    alpha: Option<&Array2<f32>>,
    blur_transfer: &Array2<Complex32>,
    correlation: CorrelationForm<'_>,
    config: &Wiener,
    stats: &mut Vec<ChannelStats>,
    cache: &mut PlanCache,
) -> Result<Array3<f32>> {
    let Some(alpha) = alpha else {
        return restore_independent(color, blur_transfer, correlation, config, stats, cache);
    };

    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    if channels != 3 || alpha.dim() != (height, width) {
        return restore_independent(color, blur_transfer, correlation, config, stats, cache);
    }

    let mut premultiplied = Array3::zeros((channels, height, width));
    for y in 0..height {
        for x in 0..width {
            let a = (alpha[[y, x]] / 255.0).clamp(0.0, 1.0);
            for c in 0..channels {
                premultiplied[[c, y, x]] = color[[c, y, x]] * a;
            }
        }
    }

    let restored = restore_independent(
        &premultiplied,
        blur_transfer,
        correlation,
        config,
        stats,
        cache,
    )?;
    let mut output = Array3::zeros((channels, height, width));
    for y in 0..height {
        for x in 0..width {
            let a = (alpha[[y, x]] / 255.0).clamp(0.0, 1.0);
            for c in 0..channels {
                output[[c, y, x]] = if a > f32::EPSILON {
                    restored[[c, y, x]] / a
                } else {
                    0.0
                };
            }
        }
    }

    for c in 0..channels {
        let channel = output.index_axis(Axis(0), c).to_owned();
        let normalized = normalize_range(&channel, config.range_policy)?;
        output.index_axis_mut(Axis(0), c).assign(&normalized);
    }

    Ok(output)
}

fn restore_channel(
    input: &Array2<f32>,
    blur_transfer: &Array2<Complex32>,
    correlation: CorrelationForm<'_>,
    config: &Wiener,
    cache: &mut PlanCache,
) -> Result<(Array2<f32>, ChannelStats)> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let (height, width) = input.dim();
    let padded = pad_to_dims(input, blur_transfer.dim(), config.boundary)?;
    let mut spectrum = fft2_forward_real(&padded, cache)?;
    let channel_stats = apply_wiener(&mut spectrum, blur_transfer, correlation)?;
    let restored = fft2_inverse_complex(&spectrum, cache)?;

    let mut cropped = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            cropped[[y, x]] = restored[[y, x]];
        }
    }
    let normalized = normalize_range(&cropped, config.range_policy)?;
    Ok((normalized, channel_stats))
}

fn apply_wiener(
    spectrum: &mut Array2<Complex32>,
    blur_transfer: &Array2<Complex32>,
    correlation: CorrelationForm<'_>,
) -> Result<ChannelStats> {
    if spectrum.dim() != blur_transfer.dim() {
        return Err(Error::DimensionMismatch);
    }

    let mut denom_sum = 0.0_f32;
    let mut filter_sum = 0.0_f32;
    let mut signal_sum = 0.0_f32;
    let mut noise_sum = 0.0_f32;
    let mut count = 0usize;

    for ((y, x), value) in spectrum.indexed_iter_mut() {
        let h = blur_transfer[[y, x]];
        let ratio = match correlation {
            CorrelationForm::ScalarNsr(nsr) => nsr,
            CorrelationForm::Ratio { noise, image } => {
                let sn = noise[[y, x]].norm();
                let sf = image[[y, x]].norm().max(1e-8);
                sn / sf
            }
        };

        if !ratio.is_finite() || ratio < 0.0 {
            return Err(Error::InvalidParameter);
        }

        let denom = (h.norm_sqr() + ratio).max(1e-8);
        let filter = h.conj() / denom;
        let observed = *value;
        let restored = filter * observed;
        if !restored.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        let residual = observed - h * restored;
        if !residual.is_finite() {
            return Err(Error::NonFiniteInput);
        }

        *value = restored;
        denom_sum += denom;
        filter_sum += filter.norm();
        signal_sum += restored.norm_sqr();
        noise_sum += residual.norm_sqr();
        count += 1;
    }

    if count == 0 {
        return Err(Error::InvalidParameter);
    }
    let inv = 1.0 / (count as f32);
    Ok(ChannelStats {
        mean_denom: denom_sum * inv,
        mean_filter_norm: filter_sum * inv,
        signal_power: signal_sum * inv,
        noise_power: noise_sum * inv,
    })
}

fn pad_to_dims(
    input: &Array2<f32>,
    dims: (usize, usize),
    boundary: Boundary,
) -> Result<Array2<f32>> {
    let (height, width) = input.dim();
    let (target_h, target_w) = dims;
    if target_h < height || target_w < width {
        return Err(Error::DimensionMismatch);
    }

    let mut padded = Array2::zeros((target_h, target_w));
    for y in 0..target_h {
        let source_y = map_boundary_index(y as i64, height, boundary)?;
        for x in 0..target_w {
            let source_x = map_boundary_index(x as i64, width, boundary)?;
            padded[[y, x]] = match (source_y, source_x) {
                (Some(sy), Some(sx)) => input[[sy, sx]],
                _ => 0.0,
            };
        }
    }
    Ok(padded)
}

fn map_boundary_index(index: i64, len: usize, boundary: Boundary) -> Result<Option<usize>> {
    if len == 0 {
        return Err(Error::InvalidParameter);
    }
    let len_i64 = i64::try_from(len).map_err(|_| Error::DimensionMismatch)?;
    match boundary {
        Boundary::Zero => {
            if index < 0 || index >= len_i64 {
                Ok(None)
            } else {
                Ok(Some(index as usize))
            }
        }
        Boundary::Replicate => {
            let mapped = if index < 0 {
                0
            } else if index >= len_i64 {
                len_i64 - 1
            } else {
                index
            };
            Ok(Some(mapped as usize))
        }
        Boundary::Periodic => {
            let mapped = index.rem_euclid(len_i64);
            Ok(Some(mapped as usize))
        }
        Boundary::Reflect => {
            if len == 1 {
                return Ok(Some(0));
            }
            let period = 2 * (len_i64 - 1);
            let folded = index.rem_euclid(period);
            let mapped = if folded < len_i64 {
                folded
            } else {
                period - folded
            };
            Ok(Some(mapped as usize))
        }
        Boundary::Symmetric => {
            let period = 2 * len_i64;
            let folded = index.rem_euclid(period);
            let mapped = if folded < len_i64 {
                folded
            } else {
                period - 1 - folded
            };
            Ok(Some(mapped as usize))
        }
    }
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

fn validate_stats(stats: &[ChannelStats]) -> Result<()> {
    for stat in stats {
        if !stat.mean_denom.is_finite()
            || !stat.mean_filter_norm.is_finite()
            || !stat.signal_power.is_finite()
            || !stat.noise_power.is_finite()
        {
            return Err(Error::NonFiniteInput);
        }
    }
    Ok(())
}

fn estimate_nsr_from_stats(stats: &[ChannelStats], min_nsr: f32) -> Result<f32> {
    if stats.is_empty() {
        return Err(Error::InvalidParameter);
    }
    if !min_nsr.is_finite() || min_nsr <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    let mut signal = 0.0_f32;
    let mut noise = 0.0_f32;
    for stat in stats {
        if stat.signal_power < 0.0 || stat.noise_power < 0.0 {
            return Err(Error::InvalidParameter);
        }
        signal += stat.signal_power;
        noise += stat.noise_power;
    }

    if !signal.is_finite() || !noise.is_finite() {
        return Err(Error::NonFiniteInput);
    }

    let estimated = if signal <= f32::EPSILON {
        min_nsr
    } else {
        (noise / signal).max(min_nsr)
    };
    if !estimated.is_finite() || estimated <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(estimated)
}

fn relative_change(previous: f32, current: f32) -> Result<f32> {
    if !previous.is_finite() || !current.is_finite() || previous < 0.0 || current < 0.0 {
        return Err(Error::InvalidParameter);
    }

    let scale = previous.max(1e-8);
    let value = (current - previous).abs() / scale;
    if !value.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(value)
}

fn validate_config(config: &Wiener) -> Result<()> {
    if !config.nsr.is_finite() || config.nsr < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if let Padding::Explicit3(_, _, _) = config.padding {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_unsupervised_config(config: &UnsupervisedWiener) -> Result<()> {
    if !config.initial_nsr.is_finite() || config.initial_nsr < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !config.min_nsr.is_finite() || config.min_nsr <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if config.max_iterations == 0 {
        return Err(Error::InvalidParameter);
    }
    if config.min_iterations == 0 || config.min_iterations > config.max_iterations {
        return Err(Error::InvalidParameter);
    }
    if !config.tolerance.is_finite() || config.tolerance < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if let Padding::Explicit3(_, _, _) = config.padding {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn unsupervised_iteration_config(
    config: &UnsupervisedWiener,
    nsr: f32,
    collect_history: bool,
) -> Wiener {
    Wiener::new()
        .nsr(nsr)
        .boundary(config.boundary)
        .padding(config.padding)
        .channel_mode(config.channel_mode)
        .range_policy(config.range_policy)
        .collect_history(collect_history)
}
