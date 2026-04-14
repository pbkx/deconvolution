use std::f32::consts::PI;

use ndarray::{Array2, Array3};

use crate::{Error, Kernel3D, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BornWolfParams {
    dims: (usize, usize, usize),
    wavelength_um: f32,
    numerical_aperture: f32,
    refractive_index: f32,
    axial_step_um: f32,
}

impl Default for BornWolfParams {
    fn default() -> Self {
        Self {
            dims: (33, 33, 33),
            wavelength_um: 0.55,
            numerical_aperture: 1.2,
            refractive_index: 1.33,
            axial_step_um: 0.2,
        }
    }
}

impl BornWolfParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dims(mut self, value: (usize, usize, usize)) -> Self {
        self.dims = value;
        self
    }

    pub fn wavelength_um(mut self, value: f32) -> Self {
        self.wavelength_um = value;
        self
    }

    pub fn numerical_aperture(mut self, value: f32) -> Self {
        self.numerical_aperture = value;
        self
    }

    pub fn refractive_index(mut self, value: f32) -> Self {
        self.refractive_index = value;
        self
    }

    pub fn axial_step_um(mut self, value: f32) -> Self {
        self.axial_step_um = value;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GibsonLanniParams {
    dims: (usize, usize, usize),
    wavelength_um: f32,
    numerical_aperture: f32,
    immersion_index: f32,
    specimen_index: f32,
    coverslip_index: f32,
    design_coverslip_index: f32,
    coverslip_thickness_um: f32,
    design_coverslip_thickness_um: f32,
    axial_step_um: f32,
}

impl Default for GibsonLanniParams {
    fn default() -> Self {
        Self {
            dims: (33, 33, 33),
            wavelength_um: 0.55,
            numerical_aperture: 1.3,
            immersion_index: 1.515,
            specimen_index: 1.33,
            coverslip_index: 1.52,
            design_coverslip_index: 1.52,
            coverslip_thickness_um: 170.0,
            design_coverslip_thickness_um: 170.0,
            axial_step_um: 0.2,
        }
    }
}

impl GibsonLanniParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dims(mut self, value: (usize, usize, usize)) -> Self {
        self.dims = value;
        self
    }

    pub fn wavelength_um(mut self, value: f32) -> Self {
        self.wavelength_um = value;
        self
    }

    pub fn numerical_aperture(mut self, value: f32) -> Self {
        self.numerical_aperture = value;
        self
    }

    pub fn immersion_index(mut self, value: f32) -> Self {
        self.immersion_index = value;
        self
    }

    pub fn specimen_index(mut self, value: f32) -> Self {
        self.specimen_index = value;
        self
    }

    pub fn coverslip_index(mut self, value: f32) -> Self {
        self.coverslip_index = value;
        self
    }

    pub fn design_coverslip_index(mut self, value: f32) -> Self {
        self.design_coverslip_index = value;
        self
    }

    pub fn coverslip_thickness_um(mut self, value: f32) -> Self {
        self.coverslip_thickness_um = value;
        self
    }

    pub fn design_coverslip_thickness_um(mut self, value: f32) -> Self {
        self.design_coverslip_thickness_um = value;
        self
    }

    pub fn axial_step_um(mut self, value: f32) -> Self {
        self.axial_step_um = value;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariableRiGibsonLanniParams {
    dims: (usize, usize, usize),
    wavelength_um: f32,
    numerical_aperture: f32,
    immersion_index: f32,
    refractive_index_start: f32,
    refractive_index_end: f32,
    profile_exponent: f32,
    coverslip_index: f32,
    design_coverslip_index: f32,
    coverslip_thickness_um: f32,
    design_coverslip_thickness_um: f32,
    axial_step_um: f32,
}

impl Default for VariableRiGibsonLanniParams {
    fn default() -> Self {
        Self {
            dims: (33, 33, 33),
            wavelength_um: 0.55,
            numerical_aperture: 1.3,
            immersion_index: 1.515,
            refractive_index_start: 1.33,
            refractive_index_end: 1.40,
            profile_exponent: 1.0,
            coverslip_index: 1.52,
            design_coverslip_index: 1.52,
            coverslip_thickness_um: 170.0,
            design_coverslip_thickness_um: 170.0,
            axial_step_um: 0.2,
        }
    }
}

impl VariableRiGibsonLanniParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dims(mut self, value: (usize, usize, usize)) -> Self {
        self.dims = value;
        self
    }

    pub fn wavelength_um(mut self, value: f32) -> Self {
        self.wavelength_um = value;
        self
    }

    pub fn numerical_aperture(mut self, value: f32) -> Self {
        self.numerical_aperture = value;
        self
    }

    pub fn immersion_index(mut self, value: f32) -> Self {
        self.immersion_index = value;
        self
    }

    pub fn refractive_index_start(mut self, value: f32) -> Self {
        self.refractive_index_start = value;
        self
    }

    pub fn refractive_index_end(mut self, value: f32) -> Self {
        self.refractive_index_end = value;
        self
    }

    pub fn profile_exponent(mut self, value: f32) -> Self {
        self.profile_exponent = value;
        self
    }

    pub fn coverslip_index(mut self, value: f32) -> Self {
        self.coverslip_index = value;
        self
    }

    pub fn design_coverslip_index(mut self, value: f32) -> Self {
        self.design_coverslip_index = value;
        self
    }

    pub fn coverslip_thickness_um(mut self, value: f32) -> Self {
        self.coverslip_thickness_um = value;
        self
    }

    pub fn design_coverslip_thickness_um(mut self, value: f32) -> Self {
        self.design_coverslip_thickness_um = value;
        self
    }

    pub fn axial_step_um(mut self, value: f32) -> Self {
        self.axial_step_um = value;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RichardsWolfParams {
    dims: (usize, usize, usize),
    wavelength_um: f32,
    numerical_aperture: f32,
    immersion_index: f32,
    specimen_index: f32,
    polarization_weight: f32,
    axial_step_um: f32,
}

impl Default for RichardsWolfParams {
    fn default() -> Self {
        Self {
            dims: (33, 33, 33),
            wavelength_um: 0.55,
            numerical_aperture: 1.3,
            immersion_index: 1.515,
            specimen_index: 1.33,
            polarization_weight: 0.5,
            axial_step_um: 0.2,
        }
    }
}

impl RichardsWolfParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dims(mut self, value: (usize, usize, usize)) -> Self {
        self.dims = value;
        self
    }

    pub fn wavelength_um(mut self, value: f32) -> Self {
        self.wavelength_um = value;
        self
    }

    pub fn numerical_aperture(mut self, value: f32) -> Self {
        self.numerical_aperture = value;
        self
    }

    pub fn immersion_index(mut self, value: f32) -> Self {
        self.immersion_index = value;
        self
    }

    pub fn specimen_index(mut self, value: f32) -> Self {
        self.specimen_index = value;
        self
    }

    pub fn polarization_weight(mut self, value: f32) -> Self {
        self.polarization_weight = value;
        self
    }

    pub fn axial_step_um(mut self, value: f32) -> Self {
        self.axial_step_um = value;
        self
    }
}

pub fn born_wolf(params: &BornWolfParams) -> Result<Kernel3D> {
    validate_born_wolf_params(params)?;
    let (depth, height, width) = params.dims;
    let (lateral_scale, axial_scale, lateral_step) = diffraction_scales(
        params.wavelength_um,
        params.numerical_aperture,
        params.refractive_index,
    )?;
    let radial = radial_map((height, width), lateral_step)?;
    let center_z = center_coordinate(depth)?;

    let mut psf = Array3::zeros((depth, height, width));
    for z in 0..depth {
        let z_um = (z as f32 - center_z) * params.axial_step_um;
        let axial = gaussian(z_um, 0.0, axial_scale)?;
        for y in 0..height {
            for x in 0..width {
                let r_um = radial[[y, x]];
                let lateral = airy_like(
                    r_um,
                    lateral_scale,
                    params.wavelength_um,
                    params.numerical_aperture,
                )?;
                let value = lateral * axial;
                if !value.is_finite() || value < 0.0 {
                    return Err(Error::NonFiniteInput);
                }
                psf[[z, y, x]] = value;
            }
        }
    }

    to_normalized_kernel(psf)
}

pub fn gibson_lanni(params: &GibsonLanniParams) -> Result<Kernel3D> {
    validate_gibson_lanni_params(params)?;
    let (depth, height, width) = params.dims;
    let (lateral_scale, axial_scale_base, lateral_step) = diffraction_scales(
        params.wavelength_um,
        params.numerical_aperture,
        params.specimen_index,
    )?;
    let radial = radial_map((height, width), lateral_step)?;
    let center_z = center_coordinate(depth)?;
    let thickness_ratio = normalized_thickness_mismatch(
        params.coverslip_thickness_um,
        params.design_coverslip_thickness_um,
    )?;
    let index_mismatch = (params.immersion_index - params.specimen_index).abs()
        + (params.coverslip_index - params.design_coverslip_index).abs();

    let mut psf = Array3::zeros((depth, height, width));
    for z in 0..depth {
        let z_um = (z as f32 - center_z) * params.axial_step_um;
        for y in 0..height {
            for x in 0..width {
                let r_um = radial[[y, x]];
                let normalized_r = r_um / lateral_scale.max(1e-6);
                let aberration = index_mismatch * normalized_r * normalized_r
                    + thickness_ratio * normalized_r.abs();
                let axial_scale = axial_scale_base * (1.0 + 0.45 * aberration.abs());
                let axial_shift = axial_scale_base
                    * 0.18
                    * (params.immersion_index - params.specimen_index)
                    * normalized_r;
                let lateral = airy_like(
                    r_um,
                    lateral_scale,
                    params.wavelength_um,
                    params.numerical_aperture,
                )?;
                let axial = gaussian(z_um, axial_shift, axial_scale)?;
                let apodization = (-0.5 * aberration * aberration).exp();
                let value = lateral * axial * apodization;
                if !value.is_finite() || value < 0.0 {
                    return Err(Error::NonFiniteInput);
                }
                psf[[z, y, x]] = value;
            }
        }
    }

    to_normalized_kernel(psf)
}

pub fn variable_ri_gibson_lanni(params: &VariableRiGibsonLanniParams) -> Result<Kernel3D> {
    validate_variable_ri_gibson_lanni_params(params)?;
    let (depth, height, width) = params.dims;
    let (lateral_scale, _, lateral_step) = diffraction_scales(
        params.wavelength_um,
        params.numerical_aperture,
        params
            .refractive_index_start
            .max(params.refractive_index_end),
    )?;
    let radial = radial_map((height, width), lateral_step)?;
    let center_z = center_coordinate(depth)?;
    let thickness_ratio = normalized_thickness_mismatch(
        params.coverslip_thickness_um,
        params.design_coverslip_thickness_um,
    )?;
    let depth_den = (depth.saturating_sub(1)).max(1) as f32;

    let mut psf = Array3::zeros((depth, height, width));
    for z in 0..depth {
        let z_um = (z as f32 - center_z) * params.axial_step_um;
        let t = (z as f32 / depth_den).clamp(0.0, 1.0);
        let local_index = params.refractive_index_start
            + (params.refractive_index_end - params.refractive_index_start)
                * t.powf(params.profile_exponent);
        let (_, local_axial_scale, _) =
            diffraction_scales(params.wavelength_um, params.numerical_aperture, local_index)?;
        let index_mismatch = (params.immersion_index - local_index).abs()
            + (params.coverslip_index - params.design_coverslip_index).abs();

        for y in 0..height {
            for x in 0..width {
                let r_um = radial[[y, x]];
                let normalized_r = r_um / lateral_scale.max(1e-6);
                let aberration = index_mismatch * normalized_r * normalized_r
                    + thickness_ratio * normalized_r.abs();
                let axial_scale = local_axial_scale * (1.0 + 0.4 * aberration.abs());
                let axial_shift = local_axial_scale * 0.12 * (params.immersion_index - local_index);
                let lateral = airy_like(
                    r_um,
                    lateral_scale,
                    params.wavelength_um,
                    params.numerical_aperture,
                )?;
                let axial = gaussian(z_um, axial_shift, axial_scale)?;
                let apodization = (-0.5 * aberration * aberration).exp();
                let value = lateral * axial * apodization;
                if !value.is_finite() || value < 0.0 {
                    return Err(Error::NonFiniteInput);
                }
                psf[[z, y, x]] = value;
            }
        }
    }

    to_normalized_kernel(psf)
}

pub fn richards_wolf(params: &RichardsWolfParams) -> Result<Kernel3D> {
    validate_richards_wolf_params(params)?;
    let (depth, height, width) = params.dims;
    let (lateral_scale, axial_scale, lateral_step) = diffraction_scales(
        params.wavelength_um,
        params.numerical_aperture,
        params.specimen_index,
    )?;
    let radial = radial_map((height, width), lateral_step)?;
    let center_z = center_coordinate(depth)?;
    let alpha = (params.numerical_aperture / params.immersion_index).clamp(0.0, 0.999_999);

    let mut psf = Array3::zeros((depth, height, width));
    for z in 0..depth {
        let z_um = (z as f32 - center_z) * params.axial_step_um;
        let axial = gaussian(z_um, 0.0, axial_scale)?;
        for y in 0..height {
            for x in 0..width {
                let r_um = radial[[y, x]];
                let normalized_r = (r_um / (3.0 * lateral_scale).max(1e-6)).clamp(0.0, 1.0);
                let cos_theta = (1.0 - normalized_r * normalized_r).sqrt();
                let vector_term =
                    ((1.0 - params.polarization_weight) * 0.5 * (1.0 + cos_theta * cos_theta)
                        + params.polarization_weight * cos_theta)
                        .max(0.0);
                let apodization = (1.0 - alpha * normalized_r * normalized_r).max(0.0).sqrt();
                let lateral = airy_like(
                    r_um,
                    lateral_scale,
                    params.wavelength_um,
                    params.numerical_aperture,
                )?;
                let value = lateral * axial * vector_term * apodization;
                if !value.is_finite() || value < 0.0 {
                    return Err(Error::NonFiniteInput);
                }
                psf[[z, y, x]] = value;
            }
        }
    }

    to_normalized_kernel(psf)
}

fn validate_born_wolf_params(params: &BornWolfParams) -> Result<()> {
    validate_dims_3d(params.dims)?;
    validate_positive(params.wavelength_um)?;
    validate_positive(params.numerical_aperture)?;
    validate_positive(params.refractive_index)?;
    validate_positive(params.axial_step_um)?;
    if params.numerical_aperture >= params.refractive_index {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_gibson_lanni_params(params: &GibsonLanniParams) -> Result<()> {
    validate_dims_3d(params.dims)?;
    validate_positive(params.wavelength_um)?;
    validate_positive(params.numerical_aperture)?;
    validate_positive(params.immersion_index)?;
    validate_positive(params.specimen_index)?;
    validate_positive(params.coverslip_index)?;
    validate_positive(params.design_coverslip_index)?;
    validate_positive(params.coverslip_thickness_um)?;
    validate_positive(params.design_coverslip_thickness_um)?;
    validate_positive(params.axial_step_um)?;
    let na_limit = params
        .immersion_index
        .min(params.specimen_index)
        .min(params.coverslip_index)
        .min(params.design_coverslip_index);
    if params.numerical_aperture >= na_limit {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_variable_ri_gibson_lanni_params(params: &VariableRiGibsonLanniParams) -> Result<()> {
    validate_dims_3d(params.dims)?;
    validate_positive(params.wavelength_um)?;
    validate_positive(params.numerical_aperture)?;
    validate_positive(params.immersion_index)?;
    validate_positive(params.refractive_index_start)?;
    validate_positive(params.refractive_index_end)?;
    validate_positive(params.profile_exponent)?;
    validate_positive(params.coverslip_index)?;
    validate_positive(params.design_coverslip_index)?;
    validate_positive(params.coverslip_thickness_um)?;
    validate_positive(params.design_coverslip_thickness_um)?;
    validate_positive(params.axial_step_um)?;
    let min_sample_index = params
        .refractive_index_start
        .min(params.refractive_index_end);
    let na_limit = params
        .immersion_index
        .min(min_sample_index)
        .min(params.coverslip_index)
        .min(params.design_coverslip_index);
    if params.numerical_aperture >= na_limit {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_richards_wolf_params(params: &RichardsWolfParams) -> Result<()> {
    validate_dims_3d(params.dims)?;
    validate_positive(params.wavelength_um)?;
    validate_positive(params.numerical_aperture)?;
    validate_positive(params.immersion_index)?;
    validate_positive(params.specimen_index)?;
    validate_positive(params.axial_step_um)?;
    if !params.polarization_weight.is_finite()
        || params.polarization_weight < 0.0
        || params.polarization_weight > 1.0
    {
        return Err(Error::InvalidParameter);
    }
    let na_limit = params.immersion_index.min(params.specimen_index);
    if params.numerical_aperture >= na_limit {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_dims_3d(dims: (usize, usize, usize)) -> Result<()> {
    if dims.0 == 0 || dims.1 == 0 || dims.2 == 0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn validate_positive(value: f32) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    Ok(())
}

fn normalized_thickness_mismatch(actual: f32, design: f32) -> Result<f32> {
    validate_positive(actual)?;
    validate_positive(design)?;
    Ok((actual - design).abs() / design.max(1.0))
}

fn center_coordinate(length: usize) -> Result<f32> {
    if length == 0 {
        return Err(Error::InvalidParameter);
    }
    Ok((length as f32 - 1.0) * 0.5)
}

fn radial_map(dims: (usize, usize), lateral_step_um: f32) -> Result<Array2<f32>> {
    validate_positive(lateral_step_um)?;
    let (height, width) = dims;
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    let cy = center_coordinate(height)?;
    let cx = center_coordinate(width)?;
    let mut radial = Array2::zeros((height, width));
    for y in 0..height {
        let dy = (y as f32 - cy) * lateral_step_um;
        for x in 0..width {
            let dx = (x as f32 - cx) * lateral_step_um;
            let value = (dx * dx + dy * dy).sqrt();
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            radial[[y, x]] = value;
        }
    }
    Ok(radial)
}

fn diffraction_scales(
    wavelength_um: f32,
    numerical_aperture: f32,
    refractive_index: f32,
) -> Result<(f32, f32, f32)> {
    validate_positive(wavelength_um)?;
    validate_positive(numerical_aperture)?;
    validate_positive(refractive_index)?;
    if numerical_aperture >= refractive_index {
        return Err(Error::InvalidParameter);
    }

    let lateral_scale = 0.61 * wavelength_um / numerical_aperture.max(1e-6);
    let axial_scale = 2.0 * refractive_index * wavelength_um
        / (numerical_aperture * numerical_aperture).max(1e-6);
    let lateral_step = 0.5 * lateral_scale;
    if !lateral_scale.is_finite()
        || !axial_scale.is_finite()
        || !lateral_step.is_finite()
        || lateral_scale <= 0.0
        || axial_scale <= 0.0
        || lateral_step <= 0.0
    {
        return Err(Error::NonFiniteInput);
    }
    Ok((lateral_scale, axial_scale, lateral_step))
}

fn gaussian(z_um: f32, center_um: f32, sigma_um: f32) -> Result<f32> {
    validate_positive(sigma_um)?;
    if !z_um.is_finite() || !center_um.is_finite() {
        return Err(Error::InvalidParameter);
    }
    let normalized = (z_um - center_um) / sigma_um;
    let value = (-0.5 * normalized * normalized).exp();
    if !value.is_finite() || value < 0.0 {
        return Err(Error::NonFiniteInput);
    }
    Ok(value)
}

fn airy_like(
    radius_um: f32,
    lateral_scale_um: f32,
    wavelength_um: f32,
    numerical_aperture: f32,
) -> Result<f32> {
    validate_positive(lateral_scale_um)?;
    validate_positive(wavelength_um)?;
    validate_positive(numerical_aperture)?;
    if !radius_um.is_finite() || radius_um < 0.0 {
        return Err(Error::InvalidParameter);
    }

    let rho = 2.0 * PI * numerical_aperture * radius_um / wavelength_um.max(1e-6);
    let value = if rho.abs() <= 1e-4 {
        1.0
    } else {
        let j1 = bessel_j1(rho);
        let normalized = 2.0 * j1 / rho;
        normalized * normalized
    };
    if !value.is_finite() || value < 0.0 {
        return Err(Error::NonFiniteInput);
    }
    Ok(value)
}

fn to_normalized_kernel(psf: Array3<f32>) -> Result<Kernel3D> {
    if psf.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(Error::NonFiniteInput);
    }
    let sum = psf.sum();
    if !sum.is_finite() || sum <= f32::EPSILON {
        return Err(Error::InvalidPsf);
    }

    let mut kernel = Kernel3D::new(psf)?;
    kernel.normalize()?;
    Ok(kernel)
}

fn bessel_j1(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let numerator = x
            * (72_362_610_000.0
                + y * (-7_895_059_000.0
                    + y * (242_396_860.0
                        + y * (-2_972_611.5 + y * (15_704.483 + y * -30.160366)))));
        let denominator = 144_725_230_000.0
            + y * (2_300_535_300.0 + y * (18_583_304.0 + y * (99_447.44 + y * (376.99915 + y))));
        numerator / denominator
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_5;
        let ans1 = 1.0
            + y * (0.001_831_05
                + y * (-0.000_035_163_966 + y * (0.000_002_457_520_2 + y * -0.000_000_240_337)));
        let ans2 = 0.046_875
            + y * (-0.000_200_269_09
                + y * (0.000_008_449_199 + y * (-0.000_000_882_289_9 + y * 0.000_000_105_787_41)));
        let value = (0.636_619_75 / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2);
        if x < 0.0 {
            -value
        } else {
            value
        }
    }
}
