use image::{DynamicImage, GrayAlphaImage, GrayImage, Luma, LumaA, Rgb, RgbImage, Rgba, RgbaImage};
use ndarray::{Array2, Array3, Axis};

use crate::core::color::sample_from_f32;
use crate::core::conv::Convolution2D;
use crate::core::convert::PlanarImage;
use crate::core::diagnostics::Diagnostics;
use crate::core::operator::{inner_product_2d, LinearOperator2D};
use crate::core::projections::project_nonnegative_2d;
use crate::core::stopping::{check_stop, StopCriteria};
use crate::core::validate::finite_real_2d;
use crate::preprocess::normalize_range;
use crate::psf::support::validate;
use crate::psf::Kernel2D;
use crate::{ChannelMode, Error, RangePolicy, Result, SolveReport, StopReason};

pub(crate) fn tv_regularize_step_2d(
    input: &Array2<f32>,
    weight: f32,
    epsilon: f32,
) -> Result<Array2<f32>> {
    finite_real_2d(input)?;
    if !weight.is_finite() || weight < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if weight <= f32::EPSILON {
        return Ok(input.to_owned());
    }

    let (height, width) = input.dim();
    let mut px = Array2::zeros((height, width));
    let mut py = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let dx = if x + 1 < width {
                input[[y, x + 1]] - input[[y, x]]
            } else {
                0.0
            };
            let dy = if y + 1 < height {
                input[[y + 1, x]] - input[[y, x]]
            } else {
                0.0
            };
            let norm = (dx * dx + dy * dy + epsilon * epsilon).sqrt();
            if !norm.is_finite() || norm <= 0.0 {
                return Err(Error::NonFiniteInput);
            }
            px[[y, x]] = dx / norm;
            py[[y, x]] = dy / norm;
        }
    }

    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let div_x = px[[y, x]] - if x > 0 { px[[y, x - 1]] } else { 0.0 };
            let div_y = py[[y, x]] - if y > 0 { py[[y - 1, x]] } else { 0.0 };
            let value = input[[y, x]] + weight * (div_x + div_y);
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }

    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseBasis {
    Pixel,
    Haar,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ista {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    step_size: Option<f32>,
    lambda: f32,
    basis: SparseBasis,
    positivity: bool,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for Ista {
    fn default() -> Self {
        Self {
            iterations: 40,
            relative_update_tolerance: None,
            step_size: None,
            lambda: 0.35,
            basis: SparseBasis::Pixel,
            positivity: true,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: true,
        }
    }
}

impl Ista {
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

    pub fn lambda(mut self, value: f32) -> Self {
        self.lambda = value;
        self
    }

    pub fn basis(mut self, value: SparseBasis) -> Self {
        self.basis = value;
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

#[derive(Debug, Clone, PartialEq)]
pub struct Fista {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    step_size: Option<f32>,
    lambda: f32,
    basis: SparseBasis,
    positivity: bool,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for Fista {
    fn default() -> Self {
        Self {
            iterations: 40,
            relative_update_tolerance: None,
            step_size: None,
            lambda: 0.35,
            basis: SparseBasis::Pixel,
            positivity: true,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: true,
        }
    }
}

impl Fista {
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

    pub fn lambda(mut self, value: f32) -> Self {
        self.lambda = value;
        self
    }

    pub fn basis(mut self, value: SparseBasis) -> Self {
        self.basis = value;
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
enum ProximalMethod {
    Ista,
    Fista,
}

#[derive(Debug, Clone, Copy)]
struct ProximalConfig {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    step_size: Option<f32>,
    lambda: f32,
    basis: SparseBasis,
    positivity: bool,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

pub fn ista(image: &DynamicImage, psf: &Kernel2D) -> Result<(DynamicImage, SolveReport)> {
    ista_with(image, psf, &Ista::new())
}

pub fn ista_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &Ista,
) -> Result<(DynamicImage, SolveReport)> {
    run_proximal(
        image,
        psf,
        ProximalConfig {
            iterations: config.iterations,
            relative_update_tolerance: config.relative_update_tolerance,
            step_size: config.step_size,
            lambda: config.lambda,
            basis: config.basis,
            positivity: config.positivity,
            channel_mode: config.channel_mode,
            range_policy: config.range_policy,
            collect_history: config.collect_history,
        },
        ProximalMethod::Ista,
    )
}

pub fn fista(image: &DynamicImage, psf: &Kernel2D) -> Result<(DynamicImage, SolveReport)> {
    fista_with(image, psf, &Fista::new())
}

pub fn fista_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &Fista,
) -> Result<(DynamicImage, SolveReport)> {
    run_proximal(
        image,
        psf,
        ProximalConfig {
            iterations: config.iterations,
            relative_update_tolerance: config.relative_update_tolerance,
            step_size: config.step_size,
            lambda: config.lambda,
            basis: config.basis,
            positivity: config.positivity,
            channel_mode: config.channel_mode,
            range_policy: config.range_policy,
            collect_history: config.collect_history,
        },
        ProximalMethod::Fista,
    )
}

fn run_proximal(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: ProximalConfig,
    method: ProximalMethod,
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

    let step_size = resolve_step_size(config.step_size, &operator, (height, width))?;
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
    config: ProximalConfig,
    method: ProximalMethod,
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
    config: ProximalConfig,
    method: ProximalMethod,
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
    config: ProximalConfig,
    method: ProximalMethod,
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
        let projected = apply_output_projection(&normalized, config.positivity)?;
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
    config: ProximalConfig,
    method: ProximalMethod,
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
        let projected = apply_output_projection(&normalized, config.positivity)?;
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
    config: ProximalConfig,
    method: ProximalMethod,
    step_size: f32,
) -> Result<(Array2<f32>, SolveReport)> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let mut estimate = input.to_owned();
    if config.positivity {
        estimate = project_nonnegative_2d(&estimate)?;
    }
    let mut momentum = estimate.to_owned();
    let mut momentum_factor = 1.0_f32;
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
        let gradient_source = match method {
            ProximalMethod::Ista => &estimate,
            ProximalMethod::Fista => &momentum,
        };

        let gradient = gradient_data_term(gradient_source, input, operator)?;
        let unconstrained = subtract_scaled(gradient_source, &gradient, step_size)?;
        let threshold = validated_threshold(step_size, config.lambda)?;
        let mut next = prox_sparse(&unconstrained, threshold, config.basis)?;
        if config.positivity {
            next = project_nonnegative_2d(&next)?;
        }

        let (next_momentum, next_factor) = match method {
            ProximalMethod::Ista => (next.to_owned(), 1.0_f32),
            ProximalMethod::Fista => {
                let updated_factor = next_fista_factor(momentum_factor)?;
                let beta = (momentum_factor - 1.0) / updated_factor;
                let mut accelerated = fista_accelerate(&next, &estimate, beta)?;
                if config.positivity {
                    accelerated = project_nonnegative_2d(&accelerated)?;
                }
                (accelerated, updated_factor)
            }
        };

        let objective = proximal_objective(input, &next, operator, config.lambda, config.basis)?;
        let residual_update = relative_update_norm(&next, &estimate)?;
        diagnostics.record(objective, residual_update)?;

        estimate = next;
        momentum = next_momentum;
        momentum_factor = next_factor;
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
    let projected = apply_output_projection(&normalized, config.positivity)?;
    Ok((projected, report))
}

fn apply_output_projection(input: &Array2<f32>, positivity: bool) -> Result<Array2<f32>> {
    if positivity {
        return project_nonnegative_2d(input);
    }
    Ok(input.to_owned())
}

fn gradient_data_term(
    estimate: &Array2<f32>,
    observed: &Array2<f32>,
    operator: &Convolution2D,
) -> Result<Array2<f32>> {
    let predicted = operator.apply(estimate)?;
    let residual = residual(&predicted, observed)?;
    operator.adjoint(&residual)
}

fn prox_sparse(input: &Array2<f32>, threshold: f32, basis: SparseBasis) -> Result<Array2<f32>> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidParameter);
    }

    match basis {
        SparseBasis::Pixel => soft_threshold_2d(input, threshold),
        SparseBasis::Haar => soft_threshold_haar_2d(input, threshold),
    }
}

fn soft_threshold_2d(input: &Array2<f32>, threshold: f32) -> Result<Array2<f32>> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidParameter);
    }

    let (height, width) = input.dim();
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            output[[y, x]] = soft_threshold_scalar(input[[y, x]], threshold)?;
        }
    }
    Ok(output)
}

fn soft_threshold_haar_2d(input: &Array2<f32>, threshold: f32) -> Result<Array2<f32>> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidParameter);
    }

    let mut coeffs = haar_forward_2d(input)?;
    soft_threshold_haar_details_in_place(&mut coeffs, threshold)?;
    haar_inverse_2d(&coeffs)
}

fn soft_threshold_haar_details_in_place(coeffs: &mut Array2<f32>, threshold: f32) -> Result<()> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidParameter);
    }

    let (height, width) = coeffs.dim();
    let approx_h = height / 2;
    let approx_w = width / 2;

    for y in 0..height {
        for x in 0..width {
            if y < approx_h && x < approx_w {
                continue;
            }
            coeffs[[y, x]] = soft_threshold_scalar(coeffs[[y, x]], threshold)?;
        }
    }

    Ok(())
}

fn soft_threshold_scalar(value: f32, threshold: f32) -> Result<f32> {
    if !value.is_finite() || !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidParameter);
    }

    let magnitude = (value.abs() - threshold).max(0.0);
    let output = value.signum() * magnitude;
    if !output.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(output)
}

fn proximal_objective(
    observed: &Array2<f32>,
    estimate: &Array2<f32>,
    operator: &Convolution2D,
    lambda: f32,
    basis: SparseBasis,
) -> Result<f32> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(Error::InvalidParameter);
    }

    let predicted = operator.apply(estimate)?;
    let residual = residual(observed, &predicted)?;
    let data_term = 0.5 * squared_l2_norm(&residual)?;
    let sparse_term = sparse_l1_penalty(estimate, basis)?;
    let objective = data_term + lambda * sparse_term;
    if !objective.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(objective)
}

fn sparse_l1_penalty(input: &Array2<f32>, basis: SparseBasis) -> Result<f32> {
    match basis {
        SparseBasis::Pixel => {
            let mut penalty = 0.0_f32;
            for value in input {
                penalty += value.abs();
            }
            if !penalty.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            Ok(penalty)
        }
        SparseBasis::Haar => {
            let coeffs = haar_forward_2d(input)?;
            let (height, width) = coeffs.dim();
            let approx_h = height / 2;
            let approx_w = width / 2;
            let mut penalty = 0.0_f32;
            for y in 0..height {
                for x in 0..width {
                    if y < approx_h && x < approx_w {
                        continue;
                    }
                    penalty += coeffs[[y, x]].abs();
                }
            }
            if !penalty.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            Ok(penalty)
        }
    }
}

fn haar_forward_2d(input: &Array2<f32>) -> Result<Array2<f32>> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let (height, width) = input.dim();
    let pairs_w = width / 2;
    let pairs_h = height / 2;

    let mut row_transformed = Array2::zeros((height, width));
    for y in 0..height {
        for pair in 0..pairs_w {
            let left = input[[y, 2 * pair]];
            let right = input[[y, 2 * pair + 1]];
            row_transformed[[y, pair]] = (left + right) * std::f32::consts::FRAC_1_SQRT_2;
            row_transformed[[y, pairs_w + pair]] = (left - right) * std::f32::consts::FRAC_1_SQRT_2;
        }
        if width % 2 == 1 {
            row_transformed[[y, width - 1]] = input[[y, width - 1]];
        }
    }

    let mut output = Array2::zeros((height, width));
    for x in 0..width {
        for pair in 0..pairs_h {
            let top = row_transformed[[2 * pair, x]];
            let bottom = row_transformed[[2 * pair + 1, x]];
            output[[pair, x]] = (top + bottom) * std::f32::consts::FRAC_1_SQRT_2;
            output[[pairs_h + pair, x]] = (top - bottom) * std::f32::consts::FRAC_1_SQRT_2;
        }
        if height % 2 == 1 {
            output[[height - 1, x]] = row_transformed[[height - 1, x]];
        }
    }

    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(output)
}

fn haar_inverse_2d(coeffs: &Array2<f32>) -> Result<Array2<f32>> {
    if coeffs.is_empty() {
        return Err(Error::EmptyImage);
    }
    if coeffs.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let (height, width) = coeffs.dim();
    let pairs_w = width / 2;
    let pairs_h = height / 2;

    let mut col_reconstructed = Array2::zeros((height, width));
    for x in 0..width {
        for pair in 0..pairs_h {
            let average = coeffs[[pair, x]];
            let detail = coeffs[[pairs_h + pair, x]];
            col_reconstructed[[2 * pair, x]] = (average + detail) * std::f32::consts::FRAC_1_SQRT_2;
            col_reconstructed[[2 * pair + 1, x]] =
                (average - detail) * std::f32::consts::FRAC_1_SQRT_2;
        }
        if height % 2 == 1 {
            col_reconstructed[[height - 1, x]] = coeffs[[height - 1, x]];
        }
    }

    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for pair in 0..pairs_w {
            let average = col_reconstructed[[y, pair]];
            let detail = col_reconstructed[[y, pairs_w + pair]];
            output[[y, 2 * pair]] = (average + detail) * std::f32::consts::FRAC_1_SQRT_2;
            output[[y, 2 * pair + 1]] = (average - detail) * std::f32::consts::FRAC_1_SQRT_2;
        }
        if width % 2 == 1 {
            output[[y, width - 1]] = col_reconstructed[[y, width - 1]];
        }
    }

    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    Ok(output)
}

fn resolve_step_size(
    configured: Option<f32>,
    operator: &Convolution2D,
    dims: (usize, usize),
) -> Result<f32> {
    if let Some(step_size) = configured {
        return validate_step_size(step_size);
    }

    let norm_squared = estimate_operator_norm_squared(operator, dims)?;
    let step_size = 0.9 / norm_squared.max(1.0);
    validate_step_size(step_size)
}

fn estimate_operator_norm_squared(operator: &Convolution2D, dims: (usize, usize)) -> Result<f32> {
    let (height, width) = dims;
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let mut vector = Array2::from_elem((height, width), 1.0_f32);
    normalize_in_place(&mut vector)?;

    for _ in 0..8 {
        let applied = operator.apply(&vector)?;
        let mut next = operator.adjoint(&applied)?;
        let norm = l2_norm(&next)?;
        if norm <= 1e-6 {
            return Ok(1.0);
        }
        scale_in_place(&mut next, 1.0 / norm)?;
        vector = next;
    }

    let applied = operator.apply(&vector)?;
    let ata_vector = operator.adjoint(&applied)?;
    let estimate = inner_product_2d(&vector, &ata_vector)?;
    if !estimate.is_finite() || estimate <= 0.0 {
        return Err(Error::ConvergenceFailure);
    }
    Ok(estimate)
}

fn validated_threshold(step_size: f32, lambda: f32) -> Result<f32> {
    let step_size = validate_step_size(step_size)?;
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(Error::InvalidParameter);
    }
    let threshold = step_size * lambda;
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(threshold)
}

fn next_fista_factor(current: f32) -> Result<f32> {
    if !current.is_finite() || current <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    let next = 0.5 * (1.0 + (1.0 + 4.0 * current * current).sqrt());
    if !next.is_finite() || next <= 0.0 {
        return Err(Error::NonFiniteInput);
    }
    Ok(next)
}

fn fista_accelerate(next: &Array2<f32>, prev: &Array2<f32>, beta: f32) -> Result<Array2<f32>> {
    if next.dim() != prev.dim() {
        return Err(Error::DimensionMismatch);
    }
    if !beta.is_finite() {
        return Err(Error::InvalidParameter);
    }

    let (height, width) = next.dim();
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let value = next[[y, x]] + beta * (next[[y, x]] - prev[[y, x]]);
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }
    Ok(output)
}

fn normalize_in_place(input: &mut Array2<f32>) -> Result<()> {
    let norm = l2_norm(input)?;
    if norm <= f32::EPSILON {
        return Err(Error::InvalidParameter);
    }
    scale_in_place(input, 1.0 / norm)
}

fn scale_in_place(input: &mut Array2<f32>, scale: f32) -> Result<()> {
    if !scale.is_finite() {
        return Err(Error::InvalidParameter);
    }
    for value in input.iter_mut() {
        let scaled = *value * scale;
        if !scaled.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        *value = scaled;
    }
    Ok(())
}

fn l2_norm(input: &Array2<f32>) -> Result<f32> {
    let norm_squared = squared_l2_norm(input)?;
    let norm = norm_squared.sqrt();
    if !norm.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(norm)
}

fn validate_step_size(step_size: f32) -> Result<f32> {
    if !step_size.is_finite() || step_size <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(step_size)
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

fn subtract_scaled(
    base: &Array2<f32>,
    direction: &Array2<f32>,
    step_size: f32,
) -> Result<Array2<f32>> {
    if base.dim() != direction.dim() {
        return Err(Error::DimensionMismatch);
    }

    let step_size = validate_step_size(step_size)?;
    let (height, width) = base.dim();
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let value = base[[y, x]] - step_size * direction[[y, x]];
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

fn validate_config(config: ProximalConfig) -> Result<()> {
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
    if !config.lambda.is_finite() || config.lambda < 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}
