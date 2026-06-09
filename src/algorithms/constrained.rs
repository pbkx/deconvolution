use image::DynamicImage;
use ndarray::{Array2, Array3, Axis};

use crate::core::conv::Convolution2D;
use crate::core::convert::{rebuild_dynamic_like, PlanarImage};
use crate::core::diagnostics::Diagnostics;
use crate::core::operator::{inner_product_2d, LinearOperator2D};
use crate::core::projections::{project_bounds_2d, project_nonnegative_2d};
use crate::core::stopping::{check_stop, StopCriteria};
use crate::preprocess::normalize_range;
use crate::psf::support::validate;
use crate::psf::Kernel2D;
use crate::{ChannelMode, Error, RangePolicy, Result, SolveReport, StopReason};

#[derive(Debug, Clone, PartialEq)]
pub struct Nnls {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    step_size: Option<f32>,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for Nnls {
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

impl Nnls {
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
pub struct Bvls {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    step_size: Option<f32>,
    lower_bound: f32,
    upper_bound: f32,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for Bvls {
    fn default() -> Self {
        Self {
            iterations: 40,
            relative_update_tolerance: None,
            step_size: None,
            lower_bound: 0.0,
            upper_bound: 255.0,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: true,
        }
    }
}

impl Bvls {
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

    pub fn lower_bound(mut self, value: f32) -> Self {
        self.lower_bound = value;
        self
    }

    pub fn upper_bound(mut self, value: f32) -> Self {
        self.upper_bound = value;
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
enum Constraint {
    Nonnegative,
    Bounds { lower_bound: f32, upper_bound: f32 },
}

#[derive(Debug, Clone, Copy)]
struct ConstrainedConfig {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    step_size: Option<f32>,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

pub fn nnls(image: &DynamicImage, psf: &Kernel2D) -> Result<(DynamicImage, SolveReport)> {
    nnls_with(image, psf, &Nnls::new())
}

pub fn nnls_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &Nnls,
) -> Result<(DynamicImage, SolveReport)> {
    run_constrained(
        image,
        psf,
        ConstrainedConfig {
            iterations: config.iterations,
            relative_update_tolerance: config.relative_update_tolerance,
            step_size: config.step_size,
            channel_mode: config.channel_mode,
            range_policy: config.range_policy,
            collect_history: config.collect_history,
        },
        Constraint::Nonnegative,
    )
}

pub fn bvls(image: &DynamicImage, psf: &Kernel2D) -> Result<(DynamicImage, SolveReport)> {
    bvls_with(image, psf, &Bvls::new())
}

pub fn bvls_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &Bvls,
) -> Result<(DynamicImage, SolveReport)> {
    run_constrained(
        image,
        psf,
        ConstrainedConfig {
            iterations: config.iterations,
            relative_update_tolerance: config.relative_update_tolerance,
            step_size: config.step_size,
            channel_mode: config.channel_mode,
            range_policy: config.range_policy,
            collect_history: config.collect_history,
        },
        Constraint::Bounds {
            lower_bound: config.lower_bound,
            upper_bound: config.upper_bound,
        },
    )
}

fn run_constrained(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: ConstrainedConfig,
    constraint: Constraint,
) -> Result<(DynamicImage, SolveReport)> {
    validate(psf)?;
    validate_config(config, constraint)?;

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
        constraint,
        step_size,
    )?;
    let restored = rebuild_dynamic_like(image, &restored_color)?;
    Ok((restored, report))
}

fn restore_color(
    color: &Array3<f32>,
    alpha: Option<&Array2<f32>>,
    operator: &Convolution2D,
    config: ConstrainedConfig,
    constraint: Constraint,
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
            restore_independent(color, operator, config, constraint, step_size)
        }
        ChannelMode::LumaOnly => restore_luma_only(color, operator, config, constraint, step_size),
        ChannelMode::PremultipliedAlpha => {
            restore_premultiplied(color, alpha, operator, config, constraint, step_size)
        }
    }
}

fn restore_independent(
    color: &Array3<f32>,
    operator: &Convolution2D,
    config: ConstrainedConfig,
    constraint: Constraint,
    step_size: f32,
) -> Result<(Array3<f32>, SolveReport)> {
    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    let mut output = Array3::zeros((channels, height, width));
    let mut reports = Vec::with_capacity(channels);

    for channel_idx in 0..channels {
        let channel = color.index_axis(Axis(0), channel_idx).to_owned();
        let (restored, report) =
            restore_channel(&channel, operator, config, constraint, step_size)?;
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
    config: ConstrainedConfig,
    constraint: Constraint,
    step_size: f32,
) -> Result<(Array3<f32>, SolveReport)> {
    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    if channels == 1 {
        return restore_independent(color, operator, config, constraint, step_size);
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

    let (restored_luma, report) = restore_channel(&luma, operator, config, constraint, step_size)?;
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
        let projected = project_constraint(&normalized, constraint)?;
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
    config: ConstrainedConfig,
    constraint: Constraint,
    step_size: f32,
) -> Result<(Array3<f32>, SolveReport)> {
    let Some(alpha) = alpha else {
        return restore_independent(color, operator, config, constraint, step_size);
    };

    let channels = color.shape()[0];
    let height = color.shape()[1];
    let width = color.shape()[2];
    if channels != 3 || alpha.dim() != (height, width) {
        return restore_independent(color, operator, config, constraint, step_size);
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
        restore_independent(&premultiplied, operator, config, constraint, step_size)?;
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
        let projected = project_constraint(&normalized, constraint)?;
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
    config: ConstrainedConfig,
    constraint: Constraint,
    step_size: f32,
) -> Result<(Array2<f32>, SolveReport)> {
    if input.is_empty() {
        return Err(Error::EmptyImage);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteInput);
    }

    let mut estimate = project_constraint(input, constraint)?;
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
        let residual = residual(input, &predicted)?;
        let direction = operator.adjoint(&residual)?;
        let unconstrained = add_scaled(&estimate, &direction, step_size)?;
        let next = project_constraint(&unconstrained, constraint)?;

        let objective = 0.5 * squared_l2_norm(&residual)?;
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
    let projected = project_constraint(&normalized, constraint)?;
    Ok((projected, report))
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

fn project_constraint(input: &Array2<f32>, constraint: Constraint) -> Result<Array2<f32>> {
    match constraint {
        Constraint::Nonnegative => project_nonnegative_2d(input),
        Constraint::Bounds {
            lower_bound,
            upper_bound,
        } => project_bounds_2d(input, lower_bound, upper_bound),
    }
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

fn add_scaled(base: &Array2<f32>, direction: &Array2<f32>, step_size: f32) -> Result<Array2<f32>> {
    if base.dim() != direction.dim() {
        return Err(Error::DimensionMismatch);
    }

    let step_size = validate_step_size(step_size)?;
    let (height, width) = base.dim();
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let value = base[[y, x]] + step_size * direction[[y, x]];
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

fn validate_config(config: ConstrainedConfig, constraint: Constraint) -> Result<()> {
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
    if let Constraint::Bounds {
        lower_bound,
        upper_bound,
    } = constraint
    {
        if !lower_bound.is_finite() || !upper_bound.is_finite() || lower_bound > upper_bound {
            return Err(Error::InvalidParameter);
        }
    }
    Ok(())
}
