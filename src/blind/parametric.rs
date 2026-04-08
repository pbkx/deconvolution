use image::{DynamicImage, GrayAlphaImage, GrayImage, Luma, LumaA};
use ndarray::Array2;

use crate::core::color::{sample_from_f32, sample_to_f32};
use crate::core::projections::project_nonnegative_2d;
use crate::core::stopping::{check_stop, StopCriteria};
use crate::core::validate::finite_real_2d;
use crate::psf::{
    apply_constraints, defocus, flip, gaussian2d, motion_linear, oriented_gaussian, validate,
    Kernel2D, PsfConstraint,
};
use crate::simulate::blur;
use crate::{Error, Result, StopReason};

use super::{BlindOutput, BlindReport};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParametricPsf {
    Gaussian {
        sigma: f32,
    },
    MotionLinear {
        length: f32,
        angle_deg: f32,
    },
    Defocus {
        radius: f32,
    },
    OrientedGaussian {
        sigma_major: f32,
        sigma_minor: f32,
        angle_deg: f32,
    },
}

impl ParametricPsf {
    pub fn realize(&self, dims: (usize, usize)) -> Result<Kernel2D> {
        validate_dims(dims)?;
        match *self {
            Self::Gaussian { sigma } => gaussian2d(dims, sigma),
            Self::MotionLinear { length, angle_deg } => {
                let base = motion_linear(length, angle_deg)?;
                fit_and_normalize(base.as_array(), dims)
            }
            Self::Defocus { radius } => {
                let base = defocus(radius)?;
                fit_and_normalize(base.as_array(), dims)
            }
            Self::OrientedGaussian {
                sigma_major,
                sigma_minor,
                angle_deg,
            } => oriented_gaussian(dims, sigma_major, sigma_minor, angle_deg),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlindParametric {
    iterations: usize,
    image_iterations: usize,
    relative_update_tolerance: Option<f32>,
    filter_epsilon: f32,
    initial_step_scale: f32,
    min_step_scale: f32,
    psf_constraints: Vec<PsfConstraint>,
    collect_history: bool,
}

impl Default for BlindParametric {
    fn default() -> Self {
        Self {
            iterations: 16,
            image_iterations: 18,
            relative_update_tolerance: None,
            filter_epsilon: 1e-6,
            initial_step_scale: 0.35,
            min_step_scale: 0.01,
            psf_constraints: vec![PsfConstraint::Nonnegative, PsfConstraint::NormalizeSum],
            collect_history: true,
        }
    }
}

impl BlindParametric {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn iterations(mut self, value: usize) -> Self {
        self.iterations = value;
        self
    }

    pub fn image_iterations(mut self, value: usize) -> Self {
        self.image_iterations = value;
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

    pub fn initial_step_scale(mut self, value: f32) -> Self {
        self.initial_step_scale = value;
        self
    }

    pub fn min_step_scale(mut self, value: f32) -> Self {
        self.min_step_scale = value;
        self
    }

    pub fn psf_constraints(mut self, value: Vec<PsfConstraint>) -> Self {
        self.psf_constraints = value;
        self
    }

    pub fn support_mask(mut self, mask: Array2<bool>) -> Self {
        if let Some(index) = self
            .psf_constraints
            .iter()
            .position(|constraint| matches!(constraint, PsfConstraint::NormalizeSum))
        {
            self.psf_constraints
                .insert(index, PsfConstraint::SupportMask(mask));
        } else {
            self.psf_constraints.push(PsfConstraint::SupportMask(mask));
        }
        self
    }

    pub fn collect_history(mut self, value: bool) -> Self {
        self.collect_history = value;
        self
    }
}

pub fn parametric(
    image: &DynamicImage,
    initial_model: ParametricPsf,
    psf_dims: (usize, usize),
) -> Result<BlindOutput<DynamicImage>> {
    parametric_with(image, initial_model, psf_dims, &BlindParametric::new())
}

pub fn parametric_with(
    image: &DynamicImage,
    initial_model: ParametricPsf,
    psf_dims: (usize, usize),
    config: &BlindParametric,
) -> Result<BlindOutput<DynamicImage>> {
    validate_config(config)?;
    validate_dims(psf_dims)?;

    let mut observed = image_to_luma_array(image)?;
    observed = project_nonnegative_2d(&observed)?;
    finite_real_2d(&observed)?;

    let mut current = evaluate_model(
        &observed,
        clamp_model(initial_model, psf_dims)?,
        psf_dims,
        config.image_iterations,
        config.filter_epsilon,
        &config.psf_constraints,
    )?;
    let mut previous_image = current.image.clone();
    let mut previous_psf = current.psf.clone();
    let mut step_scale = config.initial_step_scale;
    let criteria = StopCriteria {
        max_iterations: config.iterations,
        relative_update_tol: config.relative_update_tolerance,
        objective_plateau_window: 0,
        objective_plateau_tol: 0.0,
        divergence_factor: f32::MAX,
    };
    let mut stop_reason = StopReason::MaxIterations;

    let mut objective_history = Vec::with_capacity(config.iterations);
    let mut image_update_history = Vec::with_capacity(config.iterations);
    let mut psf_update_history = Vec::with_capacity(config.iterations);

    for iteration in 0..config.iterations {
        let candidates = generate_candidates(current.model, psf_dims, step_scale)?;
        let mut best = current.clone();
        let mut improved = false;

        for candidate_model in candidates {
            let candidate = evaluate_model(
                &observed,
                candidate_model,
                psf_dims,
                config.image_iterations,
                config.filter_epsilon,
                &config.psf_constraints,
            )?;
            if candidate.objective < best.objective {
                best = candidate;
                improved = true;
            }
        }

        if improved {
            current = best;
        } else {
            step_scale = (step_scale * 0.5).max(config.min_step_scale);
        }

        let image_update = relative_update_norm(&current.image, &previous_image)?;
        let psf_update = relative_update_norm(current.psf.as_array(), previous_psf.as_array())?;
        let combined_update = image_update.max(psf_update);

        objective_history.push(current.objective);
        image_update_history.push(image_update);
        psf_update_history.push(psf_update);

        previous_image = current.image.clone();
        previous_psf = current.psf.clone();

        if let Some(reason) = check_stop(
            &criteria,
            iteration + 1,
            Some(combined_update),
            &objective_history,
        )? {
            stop_reason = reason;
            break;
        }
    }

    let mut report = BlindReport {
        iterations: objective_history.len(),
        stop_reason,
        objective_history,
        image_update_history,
        psf_update_history,
    };
    if !config.collect_history {
        report.objective_history.clear();
        report.image_update_history.clear();
        report.psf_update_history.clear();
    }

    let restored = luma_array_to_image(&current.image, image)?;
    Ok(BlindOutput {
        image: restored,
        psf: current.psf,
        report,
    })
}

#[derive(Debug, Clone)]
struct ParametricState {
    model: ParametricPsf,
    image: Array2<f32>,
    psf: Kernel2D,
    objective: f32,
}

fn evaluate_model(
    observed: &Array2<f32>,
    model: ParametricPsf,
    psf_dims: (usize, usize),
    image_iterations: usize,
    epsilon: f32,
    constraints: &[PsfConstraint],
) -> Result<ParametricState> {
    let mut psf = model.realize(psf_dims)?;
    psf = apply_constraints(&psf, constraints)?;
    validate(&psf)?;

    let image = estimate_image_ml(observed, &psf, image_iterations, epsilon)?;
    let predicted = blur(&image, &psf)?;
    let objective = poisson_objective(observed, &predicted, epsilon)?;

    Ok(ParametricState {
        model,
        image,
        psf,
        objective,
    })
}

fn estimate_image_ml(
    observed: &Array2<f32>,
    psf: &Kernel2D,
    iterations: usize,
    epsilon: f32,
) -> Result<Array2<f32>> {
    if iterations == 0 {
        return Err(Error::InvalidParameter);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    let mut estimate = observed.to_owned();
    estimate = project_nonnegative_2d(&estimate)?;
    let psf_flipped = flip(psf)?;

    for _ in 0..iterations {
        let predicted = blur(&estimate, psf)?;
        let ratio = multiplicative_ratio(observed, &predicted, epsilon)?;
        let correction = blur(&ratio, &psf_flipped)?;
        estimate = elementwise_mul(&estimate, &correction)?;
        estimate = project_nonnegative_2d(&estimate)?;
    }

    Ok(estimate)
}

fn generate_candidates(
    model: ParametricPsf,
    psf_dims: (usize, usize),
    step_scale: f32,
) -> Result<Vec<ParametricPsf>> {
    if !step_scale.is_finite() || step_scale <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    let mut candidates = Vec::new();
    let count = parameter_count(model);
    for index in 0..count {
        let current = parameter(model, index)?;
        let (min_value, max_value) = parameter_bounds(model, index, psf_dims)?;
        let delta = parameter_delta(model, index, current, step_scale);

        let low_value = (current - delta).clamp(min_value, max_value);
        let high_value = (current + delta).clamp(min_value, max_value);

        let low_model = with_parameter(model, index, low_value)?;
        let high_model = with_parameter(model, index, high_value)?;

        if low_model != model && !candidates.contains(&low_model) {
            candidates.push(low_model);
        }
        if high_model != model && !candidates.contains(&high_model) {
            candidates.push(high_model);
        }
    }

    Ok(candidates)
}

fn clamp_model(model: ParametricPsf, psf_dims: (usize, usize)) -> Result<ParametricPsf> {
    let mut output = model;
    for index in 0..parameter_count(model) {
        let value = parameter(output, index)?;
        let (min_value, max_value) = parameter_bounds(output, index, psf_dims)?;
        output = with_parameter(output, index, value.clamp(min_value, max_value))?;
    }
    Ok(output)
}

fn parameter_count(model: ParametricPsf) -> usize {
    match model {
        ParametricPsf::Gaussian { .. } => 1,
        ParametricPsf::MotionLinear { .. } => 2,
        ParametricPsf::Defocus { .. } => 1,
        ParametricPsf::OrientedGaussian { .. } => 3,
    }
}

fn parameter(model: ParametricPsf, index: usize) -> Result<f32> {
    match model {
        ParametricPsf::Gaussian { sigma } => match index {
            0 => Ok(sigma),
            _ => Err(Error::InvalidParameter),
        },
        ParametricPsf::MotionLinear { length, angle_deg } => match index {
            0 => Ok(length),
            1 => Ok(angle_deg),
            _ => Err(Error::InvalidParameter),
        },
        ParametricPsf::Defocus { radius } => match index {
            0 => Ok(radius),
            _ => Err(Error::InvalidParameter),
        },
        ParametricPsf::OrientedGaussian {
            sigma_major,
            sigma_minor,
            angle_deg,
        } => match index {
            0 => Ok(sigma_major),
            1 => Ok(sigma_minor),
            2 => Ok(angle_deg),
            _ => Err(Error::InvalidParameter),
        },
    }
}

fn with_parameter(model: ParametricPsf, index: usize, value: f32) -> Result<ParametricPsf> {
    match model {
        ParametricPsf::Gaussian { .. } => match index {
            0 => Ok(ParametricPsf::Gaussian { sigma: value }),
            _ => Err(Error::InvalidParameter),
        },
        ParametricPsf::MotionLinear { length, angle_deg } => match index {
            0 => Ok(ParametricPsf::MotionLinear {
                length: value,
                angle_deg,
            }),
            1 => Ok(ParametricPsf::MotionLinear {
                length,
                angle_deg: value,
            }),
            _ => Err(Error::InvalidParameter),
        },
        ParametricPsf::Defocus { radius } => match index {
            0 => Ok(ParametricPsf::Defocus { radius: value }),
            _ => Err(Error::InvalidParameter),
        },
        ParametricPsf::OrientedGaussian {
            sigma_major,
            sigma_minor,
            angle_deg,
        } => match index {
            0 => Ok(ParametricPsf::OrientedGaussian {
                sigma_major: value,
                sigma_minor,
                angle_deg,
            }),
            1 => Ok(ParametricPsf::OrientedGaussian {
                sigma_major,
                sigma_minor: value,
                angle_deg,
            }),
            2 => Ok(ParametricPsf::OrientedGaussian {
                sigma_major,
                sigma_minor,
                angle_deg: value,
            }),
            _ => Err(Error::InvalidParameter),
        },
    }
}

fn parameter_bounds(
    model: ParametricPsf,
    index: usize,
    psf_dims: (usize, usize),
) -> Result<(f32, f32)> {
    let max_dim = psf_dims.0.max(psf_dims.1) as f32;
    let max_scale = (max_dim * 2.0).max(8.0);

    match model {
        ParametricPsf::Gaussian { .. } => match index {
            0 => Ok((0.2, max_scale)),
            _ => Err(Error::InvalidParameter),
        },
        ParametricPsf::MotionLinear { .. } => match index {
            0 => Ok((1.0, max_scale)),
            1 => Ok((-180.0, 180.0)),
            _ => Err(Error::InvalidParameter),
        },
        ParametricPsf::Defocus { .. } => match index {
            0 => Ok((0.5, max_scale)),
            _ => Err(Error::InvalidParameter),
        },
        ParametricPsf::OrientedGaussian { .. } => match index {
            0 | 1 => Ok((0.2, max_scale)),
            2 => Ok((-180.0, 180.0)),
            _ => Err(Error::InvalidParameter),
        },
    }
}

fn parameter_delta(model: ParametricPsf, index: usize, value: f32, step_scale: f32) -> f32 {
    let base = (value.abs().max(1.0) * step_scale).max(0.05);
    match model {
        ParametricPsf::MotionLinear { .. } | ParametricPsf::OrientedGaussian { .. } => {
            if is_angle_parameter(model, index) {
                (12.0 * step_scale).max(0.5)
            } else {
                base
            }
        }
        _ => base,
    }
}

fn is_angle_parameter(model: ParametricPsf, index: usize) -> bool {
    matches!(model, ParametricPsf::MotionLinear { .. } if index == 1)
        || matches!(model, ParametricPsf::OrientedGaussian { .. } if index == 2)
}

fn fit_and_normalize(input: &Array2<f32>, dims: (usize, usize)) -> Result<Kernel2D> {
    let resized = fit_to_dims(input, dims)?;
    let mut kernel = Kernel2D::new(resized)?;
    kernel.normalize()?;
    Ok(kernel)
}

fn image_to_luma_array(image: &DynamicImage) -> Result<Array2<f32>> {
    match image {
        DynamicImage::ImageLuma8(gray) => gray_to_array(gray),
        DynamicImage::ImageLumaA8(gray_alpha) => gray_alpha_to_array(gray_alpha),
        _ => Err(Error::UnsupportedPixelType),
    }
}

fn luma_array_to_image(input: &Array2<f32>, source: &DynamicImage) -> Result<DynamicImage> {
    match source {
        DynamicImage::ImageLuma8(gray) => {
            let rebuilt = array_to_gray(input, gray.width(), gray.height())?;
            Ok(DynamicImage::ImageLuma8(rebuilt))
        }
        DynamicImage::ImageLumaA8(gray_alpha) => {
            let rebuilt =
                array_to_gray_alpha(input, gray_alpha.width(), gray_alpha.height(), gray_alpha)?;
            Ok(DynamicImage::ImageLumaA8(rebuilt))
        }
        _ => Err(Error::UnsupportedPixelType),
    }
}

fn gray_to_array(image: &GrayImage) -> Result<Array2<f32>> {
    let width = usize::try_from(image.width()).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(image.height()).map_err(|_| Error::DimensionMismatch)?;
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
            output[[y, x]] = sample_to_f32(image.get_pixel(x_u32, y_u32)[0]);
        }
    }
    Ok(output)
}

fn gray_alpha_to_array(image: &GrayAlphaImage) -> Result<Array2<f32>> {
    let width = usize::try_from(image.width()).map_err(|_| Error::DimensionMismatch)?;
    let height = usize::try_from(image.height()).map_err(|_| Error::DimensionMismatch)?;
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
            output[[y, x]] = sample_to_f32(image.get_pixel(x_u32, y_u32)[0]);
        }
    }
    Ok(output)
}

fn array_to_gray(input: &Array2<f32>, width: u32, height: u32) -> Result<GrayImage> {
    let width_usize = usize::try_from(width).map_err(|_| Error::DimensionMismatch)?;
    let height_usize = usize::try_from(height).map_err(|_| Error::DimensionMismatch)?;
    if input.dim() != (height_usize, width_usize) {
        return Err(Error::DimensionMismatch);
    }

    let mut output = GrayImage::new(width, height);
    for y in 0..height_usize {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width_usize {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
            let luma = sample_from_f32(input[[y, x]])?;
            output.put_pixel(x_u32, y_u32, Luma([luma]));
        }
    }

    Ok(output)
}

fn array_to_gray_alpha(
    input: &Array2<f32>,
    width: u32,
    height: u32,
    source: &GrayAlphaImage,
) -> Result<GrayAlphaImage> {
    let width_usize = usize::try_from(width).map_err(|_| Error::DimensionMismatch)?;
    let height_usize = usize::try_from(height).map_err(|_| Error::DimensionMismatch)?;
    if input.dim() != (height_usize, width_usize) {
        return Err(Error::DimensionMismatch);
    }

    let mut output = GrayAlphaImage::new(width, height);
    for y in 0..height_usize {
        let y_u32 = u32::try_from(y).map_err(|_| Error::DimensionMismatch)?;
        for x in 0..width_usize {
            let x_u32 = u32::try_from(x).map_err(|_| Error::DimensionMismatch)?;
            let luma = sample_from_f32(input[[y, x]])?;
            let alpha = source.get_pixel(x_u32, y_u32)[1];
            output.put_pixel(x_u32, y_u32, LumaA([luma, alpha]));
        }
    }

    Ok(output)
}

fn multiplicative_ratio(
    observed: &Array2<f32>,
    predicted: &Array2<f32>,
    epsilon: f32,
) -> Result<Array2<f32>> {
    if observed.dim() != predicted.dim() {
        return Err(Error::DimensionMismatch);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    let (height, width) = observed.dim();
    let mut ratio = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let denominator = predicted[[y, x]].max(epsilon);
            let value = observed[[y, x]] / denominator;
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            ratio[[y, x]] = value;
        }
    }

    Ok(ratio)
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

fn poisson_objective(observed: &Array2<f32>, predicted: &Array2<f32>, epsilon: f32) -> Result<f32> {
    if observed.dim() != predicted.dim() {
        return Err(Error::DimensionMismatch);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }

    let mut objective = 0.0_f32;
    for ((y, x), value) in observed.indexed_iter() {
        let pred = predicted[[y, x]].max(epsilon);
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

    let update = num.sqrt() / den.max(1e-12).sqrt();
    if !update.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(update)
}

fn fit_to_dims(input: &Array2<f32>, dims: (usize, usize)) -> Result<Array2<f32>> {
    let (target_h, target_w) = dims;
    validate_dims(dims)?;

    let (source_h, source_w) = input.dim();
    if source_h == 0 || source_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let copy_h = source_h.min(target_h);
    let copy_w = source_w.min(target_w);
    let source_y = (source_h - copy_h) / 2;
    let source_x = (source_w - copy_w) / 2;
    let target_y = (target_h - copy_h) / 2;
    let target_x = (target_w - copy_w) / 2;

    let mut output = Array2::zeros((target_h, target_w));
    for y in 0..copy_h {
        for x in 0..copy_w {
            output[[target_y + y, target_x + x]] = input[[source_y + y, source_x + x]];
        }
    }

    Ok(output)
}

fn validate_dims(dims: (usize, usize)) -> Result<()> {
    if dims.0 == 0 || dims.1 == 0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_config(config: &BlindParametric) -> Result<()> {
    if config.iterations == 0 || config.image_iterations == 0 {
        return Err(Error::InvalidParameter);
    }
    if let Some(tol) = config.relative_update_tolerance {
        if !tol.is_finite() || tol < 0.0 {
            return Err(Error::InvalidParameter);
        }
    }
    if !config.filter_epsilon.is_finite() || config.filter_epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !config.initial_step_scale.is_finite() || config.initial_step_scale <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !config.min_step_scale.is_finite()
        || config.min_step_scale <= 0.0
        || config.min_step_scale > config.initial_step_scale
    {
        return Err(Error::InvalidParameter);
    }
    if config.psf_constraints.is_empty() {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}
