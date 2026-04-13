use image::{DynamicImage, GrayAlphaImage, GrayImage, Luma, LumaA, Rgb, RgbImage, Rgba, RgbaImage};
use ndarray::{Array2, Array3, Axis};

use crate::core::color::sample_from_f32;
use crate::core::conv::Convolution2D;
use crate::core::convert::PlanarImage;
use crate::core::diagnostics::Diagnostics;
use crate::core::operator::{inner_product_2d, LinearOperator2D};
use crate::core::projections::project_nonnegative_2d;
use crate::core::stopping::{check_stop, StopCriteria};
use crate::preprocess::normalize_range;
use crate::psf::{validate, Kernel2D};
use crate::{ChannelMode, Error, RangePolicy, Result, SolveReport, StopReason};

#[derive(Debug, Clone, PartialEq)]
pub struct Mrnsd {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    step_size: Option<f32>,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for Mrnsd {
    fn default() -> Self {
        Self {
            iterations: 40,
            relative_update_tolerance: None,
            step_size: None,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: true,
        }
    }
}

impl Mrnsd {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn iterations(mut self, value: usize) -> Self {
        self.iterations = value;
        self
    }

    pub fn relative_update_tolerance(mut self, value: Option<f32>) -> Self {
        self.relative_update_tolerance = value;
        self
    }

    pub fn step_size(mut self, value: Option<f32>) -> Self {
        self.step_size = value;
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
pub struct Cgls {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    step_size: Option<f32>,
    positivity: bool,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for Cgls {
    fn default() -> Self {
        Self {
            iterations: 40,
            relative_update_tolerance: None,
            step_size: None,
            positivity: false,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: true,
        }
    }
}

impl Cgls {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn iterations(mut self, value: usize) -> Self {
        self.iterations = value;
        self
    }

    pub fn relative_update_tolerance(mut self, value: Option<f32>) -> Self {
        self.relative_update_tolerance = value;
        self
    }

    pub fn step_size(mut self, value: Option<f32>) -> Self {
        self.step_size = value;
        self
    }

    pub fn positivity(mut self, value: bool) -> Self {
        self.positivity = value;
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum KrylovMethod {
    Mrnsd,
    Cgls { positivity: bool },
}

#[derive(Debug, Clone, Copy)]
struct KrylovConfig {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    step_size: Option<f32>,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

pub fn mrnsd(image: &DynamicImage, psf: &Kernel2D) -> Result<(DynamicImage, SolveReport)> {
    mrnsd_with(image, psf, &Mrnsd::new())
}

pub fn mrnsd_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &Mrnsd,
) -> Result<(DynamicImage, SolveReport)> {
    run_krylov(
        image,
        psf,
        KrylovConfig {
            iterations: config.iterations,
            relative_update_tolerance: config.relative_update_tolerance,
            step_size: config.step_size,
            channel_mode: config.channel_mode,
            range_policy: config.range_policy,
            collect_history: config.collect_history,
        },
        KrylovMethod::Mrnsd,
    )
}

pub fn cgls(image: &DynamicImage, psf: &Kernel2D) -> Result<(DynamicImage, SolveReport)> {
    cgls_with(image, psf, &Cgls::new())
}

pub fn cgls_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &Cgls,
) -> Result<(DynamicImage, SolveReport)> {
    run_krylov(
        image,
        psf,
        KrylovConfig {
            iterations: config.iterations,
            relative_update_tolerance: config.relative_update_tolerance,
            step_size: config.step_size,
            channel_mode: config.channel_mode,
            range_policy: config.range_policy,
            collect_history: config.collect_history,
        },
        KrylovMethod::Cgls {
            positivity: config.positivity,
        },
    )
}

fn run_krylov(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: KrylovConfig,
    method: KrylovMethod,
) -> Result<(DynamicImage, SolveReport)> {
    validate(psf)?;
    validate_config(config)?;

    let normalized_psf = psf.normalized()?;
    let operator = Convolution2D::new(&normalized_psf)?;
    let planar = PlanarImage::from_dynamic(image)?;
    let (width_u32, height_u32) = planar.dimensions();
    let width = usize::try_from(width_u32).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(height_u32).map_err(|_| Error::DimensionMismatch)?;
    if width == 0 || height == 0 {
        return Err(Error::EmptyImage);
    }

    let step_size = resolve_step_size(config.step_size)?;
    let (restored_color, report) = restore_color(
        planar.color(),
        planar.alpha(),
        &operator,
        config,
        method,
        step_size,
    )?;
    let restored = rebuild_image(image, &restored_color)?;
    Ok((restored, report))
}

fn restore_color(
    color: &Array3<f32>,
    alpha: Option<&Array2<f32>>,
    operator: &Convolution2D,
    config: KrylovConfig,
    method: KrylovMethod,
    step_size: f32,
) -> Result<(Array3<f32>, SolveReport)> {
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

    match config.channel_mode {
        ChannelMode::Independent | ChannelMode::IgnoreAlpha => {
            restore_independent(color, operator, config, method, step_size)
        }
        ChannelMode::LumaOnly => restore_luma_only(color, operator, config, method, step_size),
        ChannelMode::PremultipliedAlpha => {
            restore_premultiplied(color, alpha, operator, config, method, step_size)
        }
    }
}

fn restore_independent(
    color: &Array3<f32>,
    operator: &Convolution2D,
    config: KrylovConfig,
    method: KrylovMethod,
    step_size: f32,
) -> Result<(Array3<f32>, SolveReport)> {
    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    let mut output = Array3::zeros((channels, height, width));
    let mut reports = Vec::with_capacity(channels);

    for channel_idx in 0..channels {
        let channel = color.index_axis(Axis(0), channel_idx).to_owned();
        let (restored, report) = restore_channel(&channel, operator, config, method, step_size)?;
        reports.push(report);
        for y in 0..height {
            for x in 0..width {
                output[[channel_idx, y, x]] = restored[[y, x]];
            }
        }
    }

    let report = merge_reports(&reports, config.collect_history)?;
    Ok((output, report))
}

fn restore_luma_only(
    color: &Array3<f32>,
    operator: &Convolution2D,
    config: KrylovConfig,
    method: KrylovMethod,
    step_size: f32,
) -> Result<(Array3<f32>, SolveReport)> {
    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    if channels == 1 {
        return restore_independent(color, operator, config, method, step_size);
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

    let (restored_luma, report) = restore_channel(&luma, operator, config, method, step_size)?;
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
        let projected = apply_output_projection(&normalized, method)?;
        for y in 0..height {
            for x in 0..width {
                output[[c, y, x]] = projected[[y, x]];
            }
        }
    }

    Ok((output, report))
}

fn restore_premultiplied(
    color: &Array3<f32>,
    alpha: Option<&Array2<f32>>,
    operator: &Convolution2D,
    config: KrylovConfig,
    method: KrylovMethod,
    step_size: f32,
) -> Result<(Array3<f32>, SolveReport)> {
    let Some(alpha) = alpha else {
        return restore_independent(color, operator, config, method, step_size);
    };

    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    if channels != 3 || alpha.dim() != (height, width) {
        return restore_independent(color, operator, config, method, step_size);
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

    let (restored, report) =
        restore_independent(&premultiplied, operator, config, method, step_size)?;
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
        let projected = apply_output_projection(&normalized, method)?;
        for y in 0..height {
            for x in 0..width {
                output[[c, y, x]] = projected[[y, x]];
            }
        }
    }

    Ok((output, report))
}

fn restore_channel(
    input: &Array2<f32>,
    operator: &Convolution2D,
    config: KrylovConfig,
    method: KrylovMethod,
    step_size: f32,
) -> Result<(Array2<f32>, SolveReport)> {
    match method {
        KrylovMethod::Mrnsd => restore_channel_mrnsd(input, operator, config, step_size),
        KrylovMethod::Cgls { positivity } => {
            restore_channel_cgls(input, operator, config, positivity, step_size)
        }
    }
}

fn restore_channel_mrnsd(
    input: &Array2<f32>,
    operator: &Convolution2D,
    config: KrylovConfig,
    step_size: f32,
) -> Result<(Array2<f32>, SolveReport)> {
    validate_step_size(step_size)?;
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let mut estimate = project_nonnegative_2d(input)?;
    let mut diagnostics = Diagnostics::new();
    let criteria = StopCriteria {
        max_iterations: config.iterations,
        relative_update_tol: config.relative_update_tolerance,
        objective_plateau_window: 0,
        objective_plateau_tol: 0.0,
        divergence_factor: f32::MAX,
    };
    let mut stop_reason = StopReason::MaxIterations;

    for iteration in 0..config.iterations {
        let predicted = operator.apply(&estimate)?;
        let residual_vec = residual(input, &predicted)?;
        let gradient = operator.adjoint(&residual_vec)?;
        let direction = elementwise_mul(&estimate, &gradient)?;
        let search = operator.apply(&direction)?;

        let denom = squared_l2_norm(&search)?;
        let numer = inner_product_2d(&gradient, &direction)?;
        let alpha = if denom <= 1e-12 || numer <= 0.0 {
            0.0_f32
        } else {
            let value = step_size * (numer / denom);
            if !value.is_finite() || value < 0.0 {
                return Err(Error::NonFiniteInput);
            }
            value
        };

        let mut next = axpy(&estimate, &direction, alpha)?;
        next = project_nonnegative_2d(&next)?;

        let predicted_next = operator.apply(&next)?;
        let residual_next = residual(input, &predicted_next)?;
        let objective = 0.5 * squared_l2_norm(&residual_next)?;
        let residual_update = relative_update_norm(&next, &estimate)?;
        diagnostics.record(objective, residual_update)?;

        estimate = next;
        if let Some(reason) = check_stop(
            &criteria,
            iteration + 1,
            Some(residual_update),
            diagnostics.objective_history(),
        )? {
            stop_reason = reason;
            break;
        }
    }

    let mut report = diagnostics.finish(stop_reason);
    if !config.collect_history {
        report.objective_history.clear();
        report.residual_history.clear();
    }
    report.estimated_nsr = None;

    let normalized = normalize_range(&estimate, config.range_policy)?;
    let projected = project_nonnegative_2d(&normalized)?;
    Ok((projected, report))
}

fn restore_channel_cgls(
    input: &Array2<f32>,
    operator: &Convolution2D,
    config: KrylovConfig,
    positivity: bool,
    step_size: f32,
) -> Result<(Array2<f32>, SolveReport)> {
    validate_step_size(step_size)?;
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let mut estimate = input.to_owned();
    if positivity {
        estimate = project_nonnegative_2d(&estimate)?;
    }

    let predicted = operator.apply(&estimate)?;
    let residual_vec = residual(input, &predicted)?;
    let s = operator.adjoint(&residual_vec)?;
    let mut p = s.to_owned();
    let mut gamma = inner_product_2d(&s, &s)?;
    if !gamma.is_finite() || gamma < 0.0 {
        return Err(Error::NonFiniteInput);
    }

    let mut diagnostics = Diagnostics::new();
    let criteria = StopCriteria {
        max_iterations: config.iterations,
        relative_update_tol: config.relative_update_tolerance,
        objective_plateau_window: 0,
        objective_plateau_tol: 0.0,
        divergence_factor: f32::MAX,
    };
    let mut stop_reason = StopReason::MaxIterations;

    for iteration in 0..config.iterations {
        let q = operator.apply(&p)?;
        let delta = squared_l2_norm(&q)?;
        let alpha = if delta <= 1e-12 || gamma <= 1e-12 {
            0.0_f32
        } else {
            let value = step_size * (gamma / delta);
            if !value.is_finite() || value < 0.0 {
                return Err(Error::NonFiniteInput);
            }
            value
        };

        let mut next = axpy(&estimate, &p, alpha)?;
        if positivity {
            next = project_nonnegative_2d(&next)?;
        }

        let predicted_next = operator.apply(&next)?;
        let residual_next = residual(input, &predicted_next)?;
        let s_next = operator.adjoint(&residual_next)?;
        let gamma_next = inner_product_2d(&s_next, &s_next)?;
        if !gamma_next.is_finite() || gamma_next < 0.0 {
            return Err(Error::NonFiniteInput);
        }
        let beta = if gamma <= 1e-12 {
            0.0_f32
        } else {
            let value = gamma_next / gamma;
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            value
        };
        let p_next = axpy(&s_next, &p, beta)?;

        let objective = 0.5 * squared_l2_norm(&residual_next)?;
        let residual_update = relative_update_norm(&next, &estimate)?;
        diagnostics.record(objective, residual_update)?;

        estimate = next;
        p = p_next;
        gamma = gamma_next;
        if let Some(reason) = check_stop(
            &criteria,
            iteration + 1,
            Some(residual_update),
            diagnostics.objective_history(),
        )? {
            stop_reason = reason;
            break;
        }
    }

    let mut report = diagnostics.finish(stop_reason);
    if !config.collect_history {
        report.objective_history.clear();
        report.residual_history.clear();
    }
    report.estimated_nsr = None;

    let normalized = normalize_range(&estimate, config.range_policy)?;
    let projected = if positivity {
        project_nonnegative_2d(&normalized)?
    } else {
        normalized
    };
    Ok((projected, report))
}

fn resolve_step_size(configured: Option<f32>) -> Result<f32> {
    if let Some(step_size) = configured {
        return validate_step_size(step_size);
    }
    Ok(1.0)
}

fn apply_output_projection(input: &Array2<f32>, method: KrylovMethod) -> Result<Array2<f32>> {
    match method {
        KrylovMethod::Mrnsd => project_nonnegative_2d(input),
        KrylovMethod::Cgls { positivity } => {
            if positivity {
                project_nonnegative_2d(input)
            } else {
                Ok(input.to_owned())
            }
        }
    }
}

fn elementwise_mul(lhs: &Array2<f32>, rhs: &Array2<f32>) -> Result<Array2<f32>> {
    if lhs.dim() != rhs.dim() {
        return Err(Error::DimensionMismatch);
    }

    let (height, width) = lhs.dim();
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let value = lhs[[y, x]] * rhs[[y, x]];
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }
    Ok(output)
}

fn residual(observed: &Array2<f32>, predicted: &Array2<f32>) -> Result<Array2<f32>> {
    if observed.dim() != predicted.dim() {
        return Err(Error::DimensionMismatch);
    }

    let (height, width) = observed.dim();
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let value = observed[[y, x]] - predicted[[y, x]];
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }
    Ok(output)
}

fn axpy(base: &Array2<f32>, direction: &Array2<f32>, scale: f32) -> Result<Array2<f32>> {
    if base.dim() != direction.dim() {
        return Err(Error::DimensionMismatch);
    }
    if !scale.is_finite() {
        return Err(Error::InvalidParameter);
    }

    let (height, width) = base.dim();
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let value = base[[y, x]] + scale * direction[[y, x]];
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }
    Ok(output)
}

fn squared_l2_norm(input: &Array2<f32>) -> Result<f32> {
    let mut sum = 0.0_f32;
    for value in input {
        sum += value * value;
    }
    if !sum.is_finite() || sum < 0.0 {
        return Err(Error::NonFiniteInput);
    }
    Ok(sum)
}

fn relative_update_norm(next: &Array2<f32>, prev: &Array2<f32>) -> Result<f32> {
    if next.dim() != prev.dim() {
        return Err(Error::DimensionMismatch);
    }

    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for ((y, x), value) in next.indexed_iter() {
        let delta = *value - prev[[y, x]];
        num += delta * delta;
        den += prev[[y, x]] * prev[[y, x]];
    }

    let residual = num.sqrt() / den.max(1e-12).sqrt();
    if !residual.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(residual)
}

fn validate_step_size(step_size: f32) -> Result<f32> {
    if !step_size.is_finite() || step_size <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(step_size)
}

fn merge_reports(reports: &[SolveReport], collect_history: bool) -> Result<SolveReport> {
    if reports.is_empty() {
        return Err(Error::InvalidParameter);
    }

    let iterations = reports
        .iter()
        .map(|report| report.iterations)
        .max()
        .unwrap_or(0);
    let stop_reason = select_stop_reason(reports);

    if !collect_history {
        return Ok(SolveReport {
            iterations,
            stop_reason,
            objective_history: Vec::new(),
            residual_history: Vec::new(),
            estimated_nsr: None,
        });
    }

    let max_len = reports
        .iter()
        .map(|report| report.objective_history.len())
        .max()
        .unwrap_or(0);
    let mut objective_history = Vec::with_capacity(max_len);
    let mut residual_history = Vec::with_capacity(max_len);

    for idx in 0..max_len {
        let mut objective_sum = 0.0_f32;
        let mut objective_count = 0usize;
        let mut residual_sum = 0.0_f32;
        let mut residual_count = 0usize;

        for report in reports {
            if let Some(value) = report.objective_history.get(idx) {
                objective_sum += *value;
                objective_count += 1;
            }
            if let Some(value) = report.residual_history.get(idx) {
                residual_sum += *value;
                residual_count += 1;
            }
        }

        if objective_count == 0 || residual_count == 0 {
            return Err(Error::InvalidParameter);
        }

        let objective = objective_sum / objective_count as f32;
        let residual = residual_sum / residual_count as f32;
        if !objective.is_finite() || !residual.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        objective_history.push(objective);
        residual_history.push(residual);
    }

    Ok(SolveReport {
        iterations,
        stop_reason,
        objective_history,
        residual_history,
        estimated_nsr: None,
    })
}

fn select_stop_reason(reports: &[SolveReport]) -> StopReason {
    if reports
        .iter()
        .any(|report| report.stop_reason == StopReason::Divergence)
    {
        return StopReason::Divergence;
    }
    if reports
        .iter()
        .any(|report| report.stop_reason == StopReason::ObjectivePlateau)
    {
        return StopReason::ObjectivePlateau;
    }
    if reports
        .iter()
        .any(|report| report.stop_reason == StopReason::RelativeUpdate)
    {
        return StopReason::RelativeUpdate;
    }
    StopReason::MaxIterations
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

fn validate_config(config: KrylovConfig) -> Result<()> {
    if config.iterations == 0 {
        return Err(Error::InvalidParameter);
    }
    if let Some(tol) = config.relative_update_tolerance {
        if !tol.is_finite() || tol < 0.0 {
            return Err(Error::InvalidParameter);
        }
    }
    if let Some(step_size) = config.step_size {
        validate_step_size(step_size)?;
    }
    Ok(())
}
