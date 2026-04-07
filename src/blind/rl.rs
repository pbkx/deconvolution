use image::{DynamicImage, GrayAlphaImage, GrayImage, Luma, LumaA};
use ndarray::Array2;

use crate::core::color::{sample_from_f32, sample_to_f32};
use crate::core::projections::project_nonnegative_2d;
use crate::core::stopping::{check_stop, StopCriteria};
use crate::core::validate::finite_real_2d;
use crate::psf::{apply_constraints, flip, validate, Kernel2D, PsfConstraint};
use crate::simulate::blur;
use crate::{Error, Result, StopReason};

use super::{BlindOutput, BlindReport};

#[derive(Debug, Clone, PartialEq)]
pub struct BlindRichardsonLucy {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    filter_epsilon: f32,
    psf_constraints: Vec<PsfConstraint>,
    collect_history: bool,
}

impl Default for BlindRichardsonLucy {
    fn default() -> Self {
        Self {
            iterations: 30,
            relative_update_tolerance: None,
            filter_epsilon: 1e-6,
            psf_constraints: vec![PsfConstraint::Nonnegative, PsfConstraint::NormalizeSum],
            collect_history: true,
        }
    }
}

impl BlindRichardsonLucy {
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

pub fn richardson_lucy(
    image: &DynamicImage,
    initial_psf: &Kernel2D,
) -> Result<BlindOutput<DynamicImage>> {
    richardson_lucy_with(image, initial_psf, &BlindRichardsonLucy::new())
}

pub fn richardson_lucy_with(
    image: &DynamicImage,
    initial_psf: &Kernel2D,
    config: &BlindRichardsonLucy,
) -> Result<BlindOutput<DynamicImage>> {
    validate(initial_psf)?;
    validate_config(config)?;

    let mut observed = image_to_luma_array(image)?;
    observed = project_nonnegative_2d(&observed)?;
    finite_real_2d(&observed)?;

    let mut image_estimate = observed.to_owned();
    let mut psf_estimate = initialize_psf(initial_psf, observed.dim(), &config.psf_constraints)?;
    let criteria = StopCriteria {
        max_iterations: config.iterations,
        relative_update_tol: config.relative_update_tolerance,
        objective_plateau_window: 0,
        objective_plateau_tol: 0.0,
        divergence_factor: f32::MAX,
    };

    let mut objective_history = Vec::with_capacity(config.iterations);
    let mut image_update_history = Vec::with_capacity(config.iterations);
    let mut psf_update_history = Vec::with_capacity(config.iterations);
    let mut stop_reason = StopReason::MaxIterations;

    for iteration in 0..config.iterations {
        let predicted_for_image = blur(&image_estimate, &psf_estimate)?;
        let ratio_for_image =
            multiplicative_ratio(&observed, &predicted_for_image, config.filter_epsilon)?;
        let psf_flipped = flip(&psf_estimate)?;
        let image_correction = blur(&ratio_for_image, &psf_flipped)?;
        let mut next_image = elementwise_mul(&image_estimate, &image_correction)?;
        next_image = project_nonnegative_2d(&next_image)?;

        let predicted_for_psf = blur(&next_image, &psf_estimate)?;
        let ratio_for_psf =
            multiplicative_ratio(&observed, &predicted_for_psf, config.filter_epsilon)?;
        let psf_correction =
            psf_update_correction(&ratio_for_psf, &next_image, psf_estimate.dims())?;
        let mut next_psf = update_psf(&psf_estimate, &psf_correction, config.filter_epsilon)?;
        next_psf = apply_constraints(&next_psf, &config.psf_constraints)?;

        let predicted = blur(&next_image, &next_psf)?;
        let objective = poisson_objective(&observed, &predicted, config.filter_epsilon)?;
        let image_update = relative_update_norm(&next_image, &image_estimate)?;
        let psf_update = relative_update_norm(next_psf.as_array(), psf_estimate.as_array())?;
        let combined_update = image_update.max(psf_update);

        objective_history.push(objective);
        image_update_history.push(image_update);
        psf_update_history.push(psf_update);

        image_estimate = next_image;
        psf_estimate = next_psf;

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

    let restored = luma_array_to_image(&image_estimate, image)?;
    Ok(BlindOutput {
        image: restored,
        psf: psf_estimate,
        report,
    })
}

fn initialize_psf(
    initial_psf: &Kernel2D,
    image_dims: (usize, usize),
    constraints: &[PsfConstraint],
) -> Result<Kernel2D> {
    let (image_h, image_w) = image_dims;
    let (psf_h, psf_w) = initial_psf.dims();
    if psf_h > image_h || psf_w > image_w {
        return Err(Error::DimensionMismatch);
    }

    let mut psf = initial_psf.normalized()?;
    psf = apply_constraints(&psf, constraints)?;
    validate(&psf)?;
    Ok(psf)
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

fn psf_update_correction(
    ratio: &Array2<f32>,
    image: &Array2<f32>,
    psf_dims: (usize, usize),
) -> Result<Array2<f32>> {
    if ratio.dim() != image.dim() {
        return Err(Error::DimensionMismatch);
    }
    finite_real_2d(ratio)?;
    finite_real_2d(image)?;

    let (psf_h, psf_w) = psf_dims;
    if psf_h == 0 || psf_w == 0 {
        return Err(Error::InvalidParameter);
    }

    let (height, width) = ratio.dim();
    if psf_h > height || psf_w > width {
        return Err(Error::DimensionMismatch);
    }

    let center_y = to_i64(psf_h / 2)?;
    let center_x = to_i64(psf_w / 2)?;
    let mut correction = Array2::zeros((psf_h, psf_w));
    for ky in 0..psf_h {
        let ky_i64 = to_i64(ky)?;
        for kx in 0..psf_w {
            let kx_i64 = to_i64(kx)?;
            let mut acc = 0.0_f32;
            for y in 0..height {
                let y_i64 = to_i64(y)?;
                let iy = wrap_index(y_i64 - ky_i64 + center_y, height)?;
                for x in 0..width {
                    let x_i64 = to_i64(x)?;
                    let ix = wrap_index(x_i64 - kx_i64 + center_x, width)?;
                    acc += ratio[[y, x]] * image[[iy, ix]];
                }
            }
            if !acc.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            correction[[ky, kx]] = acc.max(0.0);
        }
    }

    Ok(correction)
}

fn update_psf(current: &Kernel2D, correction: &Array2<f32>, epsilon: f32) -> Result<Kernel2D> {
    if current.dims() != correction.dim() {
        return Err(Error::DimensionMismatch);
    }

    let (height, width) = correction.dim();
    let mut updated = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let value = current.as_array()[[y, x]] * correction[[y, x]];
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            updated[[y, x]] = value.max(0.0);
        }
    }

    let sum = updated.sum();
    if !sum.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    if sum <= epsilon {
        return Ok(current.clone());
    }

    Kernel2D::new(updated)
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

fn to_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::DimensionMismatch)
}

fn wrap_index(value: i64, size: usize) -> Result<usize> {
    let size_i64 = to_i64(size)?;
    let wrapped = value.rem_euclid(size_i64);
    usize::try_from(wrapped).map_err(|_| Error::DimensionMismatch)
}

fn validate_config(config: &BlindRichardsonLucy) -> Result<()> {
    if config.iterations == 0 {
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
    if config.psf_constraints.is_empty() {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}
