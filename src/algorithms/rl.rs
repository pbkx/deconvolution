use image::DynamicImage;
use ndarray::{Array2, Array3, Axis};
use num_complex::Complex32;

use super::proximal::{tv_regularize_step_2d, tv_regularize_step_3d};
use crate::core::conv::Convolution2D;
use crate::core::convert::{PlanarImage, rebuild_dynamic_like};
use crate::core::diagnostics::Diagnostics;
use crate::core::fft::{fft3_forward_real, fft3_inverse_complex};
use crate::core::operator::LinearOperator2D;
use crate::core::plan_cache::PlanCache;
use crate::core::projections::{project_nonnegative_2d, project_nonnegative_3d};
use crate::core::stopping::{StopCriteria, check_stop};
use crate::otf::convert::psf2otf_3d;
use crate::preprocess::{normalize_range, normalize_range_3d};
use crate::psf::support::{validate, validate_3d};
use crate::psf::{Kernel2D, Kernel3D};
use crate::{ChannelMode, Error, RangePolicy, Result, SolveReport, StopReason};

#[derive(Debug, Clone, PartialEq)]
/// Configuration for Richardson-Lucy deconvolution with a known 2D PSF.
///
/// Defaults run 30 Poisson EM iterations, keep history, and enforce
/// nonnegative estimates.
pub struct RichardsonLucy {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    filter_epsilon: f32,
    damping: Option<f32>,
    weights: Option<Array2<f32>>,
    readout_noise: f32,
    positivity: bool,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for RichardsonLucy {
    fn default() -> Self {
        Self {
            iterations: 30,
            relative_update_tolerance: None,
            filter_epsilon: 1e-6,
            damping: None,
            weights: None,
            readout_noise: 0.0,
            positivity: true,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: true,
        }
    }
}

impl RichardsonLucy {
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

    pub fn filter_epsilon(mut self, value: f32) -> Self {
        self.filter_epsilon = value;
        self
    }

    pub fn damping(mut self, value: Option<f32>) -> Self {
        self.damping = value;
        self
    }

    pub fn weights(mut self, value: Array2<f32>) -> Self {
        self.weights = Some(value);
        self
    }

    pub fn clear_weights(mut self) -> Self {
        self.weights = None;
        self
    }

    pub fn readout_noise(mut self, value: f32) -> Self {
        self.readout_noise = value;
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
/// Richardson-Lucy configuration with total-variation regularization.
///
/// The base Richardson-Lucy options are preserved, with additional TV weight
/// and epsilon controls.
pub struct RichardsonLucyTv {
    base: RichardsonLucy,
    tv_weight: f32,
    tv_epsilon: f32,
}

impl Default for RichardsonLucyTv {
    fn default() -> Self {
        Self {
            base: RichardsonLucy::default(),
            tv_weight: 1e-2,
            tv_epsilon: 1e-3,
        }
    }
}

impl RichardsonLucyTv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn iterations(mut self, value: usize) -> Self {
        self.base = self.base.iterations(value);
        self
    }

    pub fn relative_update_tolerance(mut self, value: Option<f32>) -> Self {
        self.base = self.base.relative_update_tolerance(value);
        self
    }

    pub fn filter_epsilon(mut self, value: f32) -> Self {
        self.base = self.base.filter_epsilon(value);
        self
    }

    pub fn damping(mut self, value: Option<f32>) -> Self {
        self.base = self.base.damping(value);
        self
    }

    pub fn weights(mut self, value: Array2<f32>) -> Self {
        self.base = self.base.weights(value);
        self
    }

    pub fn clear_weights(mut self) -> Self {
        self.base = self.base.clear_weights();
        self
    }

    pub fn readout_noise(mut self, value: f32) -> Self {
        self.base = self.base.readout_noise(value);
        self
    }

    pub fn positivity(mut self, value: bool) -> Self {
        self.base = self.base.positivity(value);
        self
    }

    pub fn channel_mode(mut self, value: ChannelMode) -> Self {
        self.base = self.base.channel_mode(value);
        self
    }

    pub fn range_policy(mut self, value: RangePolicy) -> Self {
        self.base = self.base.range_policy(value);
        self
    }

    pub fn collect_history(mut self, value: bool) -> Self {
        self.base = self.base.collect_history(value);
        self
    }

    pub fn tv_weight(mut self, value: f32) -> Self {
        self.tv_weight = value;
        self
    }

    pub fn tv_epsilon(mut self, value: f32) -> Self {
        self.tv_epsilon = value;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PoissonRegularization {
    None,
    Tv { weight: f32, epsilon: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PoissonEm {
    pub(crate) iterations: usize,
    pub(crate) relative_update_tolerance: Option<f32>,
    pub(crate) filter_epsilon: f32,
    pub(crate) damping: Option<f32>,
    pub(crate) weights: Option<Array2<f32>>,
    pub(crate) readout_noise: f32,
    pub(crate) positivity: bool,
    pub(crate) channel_mode: ChannelMode,
    pub(crate) range_policy: RangePolicy,
    pub(crate) collect_history: bool,
}

impl PoissonEm {
    fn from_richardson_lucy(config: &RichardsonLucy) -> Self {
        Self {
            iterations: config.iterations,
            relative_update_tolerance: config.relative_update_tolerance,
            filter_epsilon: config.filter_epsilon,
            damping: config.damping,
            weights: config.weights.clone(),
            readout_noise: config.readout_noise,
            positivity: config.positivity,
            channel_mode: config.channel_mode,
            range_policy: config.range_policy,
            collect_history: config.collect_history,
        }
    }
}

pub fn richardson_lucy(
    image: &DynamicImage,
    psf: &Kernel2D,
) -> Result<(DynamicImage, SolveReport)> {
    richardson_lucy_with(image, psf, &RichardsonLucy::new())
}

pub fn richardson_lucy_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &RichardsonLucy,
) -> Result<(DynamicImage, SolveReport)> {
    run_richardson_lucy(image, psf, config, false, PoissonRegularization::None)
}

pub(crate) fn richardson_lucy_array2_with(
    image: &Array2<f32>,
    psf: &Kernel2D,
    config: &RichardsonLucy,
) -> Result<(Array2<f32>, SolveReport)> {
    run_richardson_lucy_array2(image, psf, config, false, PoissonRegularization::None)
}

pub(crate) fn richardson_lucy_array3_with(
    volume: &Array3<f32>,
    psf: &Kernel3D,
    config: &RichardsonLucy,
) -> Result<(Array3<f32>, SolveReport)> {
    run_richardson_lucy_array3(volume, psf, config, false, PoissonRegularization::None)
}

pub fn damped_richardson_lucy(
    image: &DynamicImage,
    psf: &Kernel2D,
) -> Result<(DynamicImage, SolveReport)> {
    damped_richardson_lucy_with(image, psf, &RichardsonLucy::new())
}

pub fn damped_richardson_lucy_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &RichardsonLucy,
) -> Result<(DynamicImage, SolveReport)> {
    run_richardson_lucy(image, psf, config, true, PoissonRegularization::None)
}

pub fn richardson_lucy_tv(
    image: &DynamicImage,
    psf: &Kernel2D,
) -> Result<(DynamicImage, SolveReport)> {
    richardson_lucy_tv_with(image, psf, &RichardsonLucyTv::new())
}

pub fn richardson_lucy_tv_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &RichardsonLucyTv,
) -> Result<(DynamicImage, SolveReport)> {
    validate_tv_config(config)?;
    let regularization = PoissonRegularization::Tv {
        weight: config.tv_weight,
        epsilon: config.tv_epsilon,
    };
    run_richardson_lucy(image, psf, &config.base, false, regularization)
}

pub(crate) fn richardson_lucy_tv_array2_with(
    image: &Array2<f32>,
    psf: &Kernel2D,
    config: &RichardsonLucyTv,
) -> Result<(Array2<f32>, SolveReport)> {
    validate_tv_config(config)?;
    let regularization = PoissonRegularization::Tv {
        weight: config.tv_weight,
        epsilon: config.tv_epsilon,
    };
    run_richardson_lucy_array2(image, psf, &config.base, false, regularization)
}

pub(crate) fn richardson_lucy_tv_array3_with(
    volume: &Array3<f32>,
    psf: &Kernel3D,
    config: &RichardsonLucyTv,
) -> Result<(Array3<f32>, SolveReport)> {
    validate_tv_config(config)?;
    let regularization = PoissonRegularization::Tv {
        weight: config.tv_weight,
        epsilon: config.tv_epsilon,
    };
    run_richardson_lucy_array3(volume, psf, &config.base, false, regularization)
}

fn run_richardson_lucy(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &RichardsonLucy,
    force_damping: bool,
    regularization: PoissonRegularization,
) -> Result<(DynamicImage, SolveReport)> {
    let effective_config = resolve_effective_config(config, force_damping);
    validate_config(&effective_config)?;
    let poisson_config = PoissonEm::from_richardson_lucy(&effective_config);
    run_poisson_em(image, psf, &poisson_config, regularization)
}

pub(crate) fn run_poisson_em(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &PoissonEm,
    regularization: PoissonRegularization,
) -> Result<(DynamicImage, SolveReport)> {
    validate(psf)?;
    validate_poisson_em_config(config)?;
    validate_regularization(regularization)?;
    let normalized_psf = psf.normalized()?;
    let op = Convolution2D::new(&normalized_psf)?;
    let planar = PlanarImage::from_dynamic(image)?;
    let (width_u32, height_u32) = planar.dimensions();
    let width = usize::try_from(width_u32).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(height_u32).map_err(|_| Error::DimensionMismatch)?;
    if width == 0 || height == 0 {
        return Err(Error::EmptyImage);
    }

    let (restored_color, report) = restore_color(
        planar.color(),
        planar.alpha(),
        planar.alpha_denominator(),
        &op,
        config,
        regularization,
    )?;
    let restored = rebuild_dynamic_like(image, &restored_color)?;
    Ok((restored, report))
}

fn run_richardson_lucy_array2(
    image: &Array2<f32>,
    psf: &Kernel2D,
    config: &RichardsonLucy,
    force_damping: bool,
    regularization: PoissonRegularization,
) -> Result<(Array2<f32>, SolveReport)> {
    let effective_config = resolve_effective_config(config, force_damping);
    validate_config(&effective_config)?;
    let poisson_config = PoissonEm::from_richardson_lucy(&effective_config);
    run_poisson_em_array2(image, psf, &poisson_config, regularization)
}

fn run_richardson_lucy_array3(
    volume: &Array3<f32>,
    psf: &Kernel3D,
    config: &RichardsonLucy,
    force_damping: bool,
    regularization: PoissonRegularization,
) -> Result<(Array3<f32>, SolveReport)> {
    let effective_config = resolve_effective_config(config, force_damping);
    validate_config(&effective_config)?;
    let poisson_config = PoissonEm::from_richardson_lucy(&effective_config);
    run_poisson_em_array3(volume, psf, &poisson_config, regularization)
}

pub(crate) fn run_poisson_em_array2(
    image: &Array2<f32>,
    psf: &Kernel2D,
    config: &PoissonEm,
    regularization: PoissonRegularization,
) -> Result<(Array2<f32>, SolveReport)> {
    validate(psf)?;
    validate_poisson_em_config(config)?;
    validate_regularization(regularization)?;
    let normalized_psf = psf.normalized()?;
    let op = Convolution2D::new(&normalized_psf)?;
    let planar = PlanarImage::from_array2(image)?;
    let (width_u32, height_u32) = planar.dimensions();
    let width = usize::try_from(width_u32).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(height_u32).map_err(|_| Error::DimensionMismatch)?;
    if width == 0 || height == 0 {
        return Err(Error::EmptyImage);
    }

    let (restored_color, report) = restore_color(
        planar.color(),
        planar.alpha(),
        planar.alpha_denominator(),
        &op,
        config,
        regularization,
    )?;
    let restored = PlanarImage::to_array2_gray(&restored_color)?;
    Ok((restored, report))
}

pub(crate) fn run_poisson_em_array3(
    volume: &Array3<f32>,
    psf: &Kernel3D,
    config: &PoissonEm,
    regularization: PoissonRegularization,
) -> Result<(Array3<f32>, SolveReport)> {
    validate_3d(psf)?;
    validate_poisson_em_config(config)?;
    validate_regularization(regularization)?;
    if volume.is_empty() {
        return Err(Error::EmptyImage);
    }
    if volume.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let normalized_psf = psf.normalized()?;
    let op = VolumeOperator3D::new(&normalized_psf, volume.dim())?;
    restore_volume(volume, &op, config, regularization)
}

fn restore_color(
    color: &Array3<f32>,
    alpha: Option<&Array2<f32>>,
    alpha_denominator: f32,
    operator: &Convolution2D,
    config: &PoissonEm,
    regularization: PoissonRegularization,
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
            restore_independent(color, operator, config, regularization)
        }
        ChannelMode::LumaOnly => restore_luma_only(color, operator, config, regularization),
        ChannelMode::PremultipliedAlpha => restore_premultiplied(
            color,
            alpha,
            alpha_denominator,
            operator,
            config,
            regularization,
        ),
    }
}

fn restore_independent(
    color: &Array3<f32>,
    operator: &Convolution2D,
    config: &PoissonEm,
    regularization: PoissonRegularization,
) -> Result<(Array3<f32>, SolveReport)> {
    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    let mut output = Array3::zeros((channels, height, width));
    let mut reports = Vec::with_capacity(channels);

    for channel_idx in 0..channels {
        let channel = color.index_axis(Axis(0), channel_idx).to_owned();
        let (restored, report) = restore_channel(&channel, operator, config, regularization)?;
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
    config: &PoissonEm,
    regularization: PoissonRegularization,
) -> Result<(Array3<f32>, SolveReport)> {
    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    if channels == 1 {
        return restore_independent(color, operator, config, regularization);
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

    let (restored_luma, report) = restore_channel(&luma, operator, config, regularization)?;
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
        for y in 0..height {
            for x in 0..width {
                output[[c, y, x]] = normalized[[y, x]];
            }
        }
    }

    Ok((output, report))
}

fn restore_premultiplied(
    color: &Array3<f32>,
    alpha: Option<&Array2<f32>>,
    alpha_denominator: f32,
    operator: &Convolution2D,
    config: &PoissonEm,
    regularization: PoissonRegularization,
) -> Result<(Array3<f32>, SolveReport)> {
    let Some(alpha) = alpha else {
        return restore_independent(color, operator, config, regularization);
    };

    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    if channels != 3 || alpha.dim() != (height, width) {
        return restore_independent(color, operator, config, regularization);
    }

    let mut premultiplied = Array3::zeros((channels, height, width));
    for y in 0..height {
        for x in 0..width {
            let a = (alpha[[y, x]] / alpha_denominator).clamp(0.0, 1.0);
            for c in 0..channels {
                premultiplied[[c, y, x]] = color[[c, y, x]] * a;
            }
        }
    }

    let (restored, report) = restore_independent(&premultiplied, operator, config, regularization)?;
    let mut output = Array3::zeros((channels, height, width));
    for y in 0..height {
        for x in 0..width {
            let a = (alpha[[y, x]] / alpha_denominator).clamp(0.0, 1.0);
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
        for y in 0..height {
            for x in 0..width {
                output[[c, y, x]] = normalized[[y, x]];
            }
        }
    }

    Ok((output, report))
}

fn restore_channel(
    input: &Array2<f32>,
    operator: &Convolution2D,
    config: &PoissonEm,
    regularization: PoissonRegularization,
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
    let weights = resolve_weights(config.weights.as_ref(), input.dim())?;

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
        let blurred = operator.apply(&estimate)?;
        let ratio =
            multiplicative_ratio(input, &blurred, config.filter_epsilon, config.readout_noise)?;
        let mut correction = operator.adjoint(&ratio)?;
        correction = apply_update_weights(&correction, weights)?;
        correction = apply_damping(&correction, config.damping)?;
        let mut next = elementwise_mul(&estimate, &correction)?;
        if config.positivity {
            next = project_nonnegative_2d(&next)?;
        }
        next = apply_regularization(&next, regularization, config.positivity)?;

        let objective =
            poisson_objective(input, &blurred, config.filter_epsilon, config.readout_noise)?;
        let residual = relative_update_norm(&next, &estimate)?;
        diagnostics.record(objective, residual)?;

        estimate = next;
        if let Some(reason) = check_stop(
            &criteria,
            iteration + 1,
            Some(residual),
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
    Ok((normalized, report))
}

struct VolumeOperator3D {
    transfer: Array3<Complex32>,
}

impl VolumeOperator3D {
    fn new(psf: &Kernel3D, dims: (usize, usize, usize)) -> Result<Self> {
        Ok(Self {
            transfer: psf2otf_3d(psf, dims)?.into_inner(),
        })
    }

    fn apply(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        apply_volume_transfer(input, &self.transfer, false)
    }

    fn adjoint(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        apply_volume_transfer(input, &self.transfer, true)
    }
}

fn apply_volume_transfer(
    input: &Array3<f32>,
    transfer: &Array3<Complex32>,
    adjoint: bool,
) -> Result<Array3<f32>> {
    if input.dim() != transfer.dim() {
        return Err(Error::DimensionMismatch);
    }

    let mut cache = PlanCache::new();
    let mut spectrum = fft3_forward_real(input, &mut cache)?;
    for ((z, y, x), value) in spectrum.indexed_iter_mut() {
        let h = transfer[[z, y, x]];
        let product = if adjoint {
            *value * h.conj()
        } else {
            *value * h
        };
        if !product.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        *value = product;
    }
    fft3_inverse_complex(&spectrum, &mut cache)
}

fn restore_volume(
    input: &Array3<f32>,
    operator: &VolumeOperator3D,
    config: &PoissonEm,
    regularization: PoissonRegularization,
) -> Result<(Array3<f32>, SolveReport)> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let mut estimate = input.to_owned();
    if config.positivity {
        estimate = project_nonnegative_3d(&estimate)?;
    }
    let (_, height, width) = input.dim();
    let weights = resolve_weights(config.weights.as_ref(), (height, width))?;
    let sensitivity = operator.adjoint(&Array3::from_elem(input.dim(), 1.0))?;

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
        let blurred = operator.apply(&estimate)?;
        let ratio =
            multiplicative_ratio_3d(input, &blurred, config.filter_epsilon, config.readout_noise)?;
        let mut correction = operator.adjoint(&ratio)?;
        correction = apply_sensitivity_3d(&correction, &sensitivity, config.filter_epsilon)?;
        correction = apply_update_weights_3d(&correction, weights)?;
        correction = apply_damping_3d(&correction, config.damping)?;
        let mut next = elementwise_mul_3d(&estimate, &correction)?;
        if config.positivity {
            next = project_nonnegative_3d(&next)?;
        }
        next = apply_regularization_3d(&next, regularization, config.positivity)?;

        let objective =
            poisson_objective_3d(input, &blurred, config.filter_epsilon, config.readout_noise)?;
        let residual = relative_update_norm_3d(&next, &estimate)?;
        diagnostics.record(objective, residual)?;

        estimate = next;
        if let Some(reason) = check_stop(
            &criteria,
            iteration + 1,
            Some(residual),
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

    let normalized = normalize_range_3d(&estimate, config.range_policy)?;
    Ok((normalized, report))
}

fn multiplicative_ratio(
    observed: &Array2<f32>,
    predicted: &Array2<f32>,
    epsilon: f32,
    readout_noise: f32,
) -> Result<Array2<f32>> {
    if observed.dim() != predicted.dim() {
        return Err(Error::DimensionMismatch);
    }
    let (height, width) = observed.dim();
    let mut ratio = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let pred = (predicted[[y, x]] + readout_noise).max(epsilon);
            let value = observed[[y, x]] / pred;
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            ratio[[y, x]] = value;
        }
    }
    Ok(ratio)
}

fn multiplicative_ratio_3d(
    observed: &Array3<f32>,
    predicted: &Array3<f32>,
    epsilon: f32,
    readout_noise: f32,
) -> Result<Array3<f32>> {
    if observed.dim() != predicted.dim() {
        return Err(Error::DimensionMismatch);
    }
    let (depth, height, width) = observed.dim();
    let mut ratio = Array3::zeros((depth, height, width));
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let pred = (predicted[[z, y, x]] + readout_noise).max(epsilon);
                let value = observed[[z, y, x]] / pred;
                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                ratio[[z, y, x]] = value;
            }
        }
    }
    Ok(ratio)
}

fn apply_update_weights(
    correction: &Array2<f32>,
    weights: Option<&Array2<f32>>,
) -> Result<Array2<f32>> {
    let Some(weights) = weights else {
        return Ok(correction.to_owned());
    };

    if correction.dim() != weights.dim() {
        return Err(Error::DimensionMismatch);
    }

    let (height, width) = correction.dim();
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let w = weights[[y, x]];
            let value = 1.0 + w * (correction[[y, x]] - 1.0);
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }

    Ok(output)
}

fn apply_update_weights_3d(
    correction: &Array3<f32>,
    weights: Option<&Array2<f32>>,
) -> Result<Array3<f32>> {
    let Some(weights) = weights else {
        return Ok(correction.to_owned());
    };

    let (depth, height, width) = correction.dim();
    if weights.dim() != (height, width) {
        return Err(Error::DimensionMismatch);
    }

    let mut output = Array3::zeros((depth, height, width));
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let value = 1.0 + weights[[y, x]] * (correction[[z, y, x]] - 1.0);
                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                output[[z, y, x]] = value;
            }
        }
    }

    Ok(output)
}

fn apply_sensitivity_3d(
    correction: &Array3<f32>,
    sensitivity: &Array3<f32>,
    epsilon: f32,
) -> Result<Array3<f32>> {
    if correction.dim() != sensitivity.dim() {
        return Err(Error::DimensionMismatch);
    }
    let (depth, height, width) = correction.dim();
    let mut output = Array3::zeros((depth, height, width));
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let denom = sensitivity[[z, y, x]].max(epsilon);
                let value = correction[[z, y, x]] / denom;
                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                output[[z, y, x]] = value;
            }
        }
    }
    Ok(output)
}

fn apply_damping(correction: &Array2<f32>, damping: Option<f32>) -> Result<Array2<f32>> {
    let Some(damping) = damping else {
        return Ok(correction.to_owned());
    };

    let lower = 1.0 / (1.0 + damping);
    let upper = 1.0 + damping;
    let (height, width) = correction.dim();
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let value = correction[[y, x]].clamp(lower, upper);
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }
    Ok(output)
}

fn apply_damping_3d(correction: &Array3<f32>, damping: Option<f32>) -> Result<Array3<f32>> {
    let Some(damping) = damping else {
        return Ok(correction.to_owned());
    };

    let lower = 1.0 / (1.0 + damping);
    let upper = 1.0 + damping;
    let (depth, height, width) = correction.dim();
    let mut output = Array3::zeros((depth, height, width));
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let value = correction[[z, y, x]].clamp(lower, upper);
                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                output[[z, y, x]] = value;
            }
        }
    }
    Ok(output)
}

fn apply_regularization(
    input: &Array2<f32>,
    regularization: PoissonRegularization,
    positivity: bool,
) -> Result<Array2<f32>> {
    match regularization {
        PoissonRegularization::None => Ok(input.to_owned()),
        PoissonRegularization::Tv { weight, epsilon } => {
            let mut output = tv_regularize_step_2d(input, weight, epsilon)?;
            if positivity {
                output = project_nonnegative_2d(&output)?;
            }
            Ok(output)
        }
    }
}

fn apply_regularization_3d(
    input: &Array3<f32>,
    regularization: PoissonRegularization,
    positivity: bool,
) -> Result<Array3<f32>> {
    match regularization {
        PoissonRegularization::None => Ok(input.to_owned()),
        PoissonRegularization::Tv { weight, epsilon } => {
            let mut output = tv_regularize_step_3d(input, weight, epsilon)?;
            if positivity {
                output = project_nonnegative_3d(&output)?;
            }
            Ok(output)
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

fn elementwise_mul_3d(lhs: &Array3<f32>, rhs: &Array3<f32>) -> Result<Array3<f32>> {
    if lhs.dim() != rhs.dim() {
        return Err(Error::DimensionMismatch);
    }
    let (depth, height, width) = lhs.dim();
    let mut output = Array3::zeros((depth, height, width));
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let value = lhs[[z, y, x]] * rhs[[z, y, x]];
                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                output[[z, y, x]] = value;
            }
        }
    }
    Ok(output)
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

fn relative_update_norm_3d(next: &Array3<f32>, prev: &Array3<f32>) -> Result<f32> {
    if next.dim() != prev.dim() {
        return Err(Error::DimensionMismatch);
    }

    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for ((z, y, x), value) in next.indexed_iter() {
        let delta = *value - prev[[z, y, x]];
        num += delta * delta;
        den += prev[[z, y, x]] * prev[[z, y, x]];
    }

    let residual = num.sqrt() / den.max(1e-12).sqrt();
    if !residual.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(residual)
}

fn poisson_objective(
    observed: &Array2<f32>,
    predicted: &Array2<f32>,
    epsilon: f32,
    readout_noise: f32,
) -> Result<f32> {
    if observed.dim() != predicted.dim() {
        return Err(Error::DimensionMismatch);
    }

    let mut objective = 0.0_f32;
    for ((y, x), value) in observed.indexed_iter() {
        let pred = (predicted[[y, x]] + readout_noise).max(epsilon);
        let term = pred - *value * pred.ln();
        if !term.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        objective += term;
    }

    if !objective.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(objective)
}

fn poisson_objective_3d(
    observed: &Array3<f32>,
    predicted: &Array3<f32>,
    epsilon: f32,
    readout_noise: f32,
) -> Result<f32> {
    if observed.dim() != predicted.dim() {
        return Err(Error::DimensionMismatch);
    }

    let mut objective = 0.0_f32;
    for ((z, y, x), value) in observed.indexed_iter() {
        let pred = (predicted[[z, y, x]] + readout_noise).max(epsilon);
        let term = pred - *value * pred.ln();
        if !term.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        objective += term;
    }

    if !objective.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(objective)
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

fn resolve_effective_config(config: &RichardsonLucy, force_damping: bool) -> RichardsonLucy {
    let mut effective = config.clone();
    if force_damping && effective.damping.is_none() {
        effective.damping = Some(0.1);
    }
    effective
}

fn resolve_weights(
    weights: Option<&Array2<f32>>,
    dims: (usize, usize),
) -> Result<Option<&Array2<f32>>> {
    let Some(weights) = weights else {
        return Ok(None);
    };

    if weights.dim() != dims {
        return Err(Error::DimensionMismatch);
    }
    if weights
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
    {
        return Err(Error::InvalidParameter);
    }

    Ok(Some(weights))
}

fn validate_tv_config(config: &RichardsonLucyTv) -> Result<()> {
    validate_config(&config.base)?;
    if !config.tv_weight.is_finite() || config.tv_weight < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !config.tv_epsilon.is_finite() || config.tv_epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_regularization(regularization: PoissonRegularization) -> Result<()> {
    match regularization {
        PoissonRegularization::None => Ok(()),
        PoissonRegularization::Tv { weight, epsilon } => {
            if !weight.is_finite() || weight < 0.0 {
                return Err(Error::InvalidParameter);
            }
            if !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(Error::InvalidParameter);
            }
            Ok(())
        }
    }
}

pub(crate) fn validate_poisson_em_config(config: &PoissonEm) -> Result<()> {
    if config.iterations == 0 {
        return Err(Error::InvalidParameter);
    }
    if let Some(tol) = config.relative_update_tolerance
        && (!tol.is_finite() || tol < 0.0)
    {
        return Err(Error::InvalidParameter);
    }
    if !config.filter_epsilon.is_finite() || config.filter_epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if let Some(damping) = config.damping
        && (!damping.is_finite() || damping < 0.0)
    {
        return Err(Error::InvalidParameter);
    }
    if !config.readout_noise.is_finite() || config.readout_noise < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if let Some(weights) = config.weights.as_ref()
        && weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
    {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_config(config: &RichardsonLucy) -> Result<()> {
    validate_poisson_em_config(&PoissonEm::from_richardson_lucy(config))
}
