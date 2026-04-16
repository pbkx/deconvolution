use image::DynamicImage;

use super::rl::{richardson_lucy_tv_with, richardson_lucy_with, RichardsonLucy, RichardsonLucyTv};
use crate::{ChannelMode, Error, Kernel2D, RangePolicy, Result, SolveReport};

#[derive(Debug, Clone, PartialEq)]
pub struct Cmle {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    filter_epsilon: f32,
    snr: f32,
    acuity: f32,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for Cmle {
    fn default() -> Self {
        Self {
            iterations: 30,
            relative_update_tolerance: None,
            filter_epsilon: 1e-6,
            snr: 40.0,
            acuity: 1.0,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: true,
        }
    }
}

impl Cmle {
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

    pub fn snr(mut self, value: f32) -> Self {
        self.snr = value;
        self
    }

    pub fn acuity(mut self, value: f32) -> Self {
        self.acuity = value;
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
pub struct Gmle {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    filter_epsilon: f32,
    snr: f32,
    acuity: f32,
    roughness: f32,
    tv_epsilon: f32,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for Gmle {
    fn default() -> Self {
        Self {
            iterations: 20,
            relative_update_tolerance: Some(1e-4),
            filter_epsilon: 1e-6,
            snr: 24.0,
            acuity: 0.85,
            roughness: 1.0,
            tv_epsilon: 1e-3,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: true,
        }
    }
}

impl Gmle {
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

    pub fn snr(mut self, value: f32) -> Self {
        self.snr = value;
        self
    }

    pub fn acuity(mut self, value: f32) -> Self {
        self.acuity = value;
        self
    }

    pub fn roughness(mut self, value: f32) -> Self {
        self.roughness = value;
        self
    }

    pub fn tv_epsilon(mut self, value: f32) -> Self {
        self.tv_epsilon = value;
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
pub struct Qmle {
    iterations: usize,
    relative_update_tolerance: Option<f32>,
    filter_epsilon: f32,
    snr: f32,
    acuity: f32,
    channel_mode: ChannelMode,
    range_policy: RangePolicy,
    collect_history: bool,
}

impl Default for Qmle {
    fn default() -> Self {
        Self {
            iterations: 10,
            relative_update_tolerance: Some(2e-4),
            filter_epsilon: 1e-6,
            snr: 60.0,
            acuity: 1.1,
            channel_mode: ChannelMode::Independent,
            range_policy: RangePolicy::PreserveInput,
            collect_history: true,
        }
    }
}

impl Qmle {
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

    pub fn snr(mut self, value: f32) -> Self {
        self.snr = value;
        self
    }

    pub fn acuity(mut self, value: f32) -> Self {
        self.acuity = value;
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

pub fn cmle(image: &DynamicImage, psf: &Kernel2D) -> Result<(DynamicImage, SolveReport)> {
    cmle_with(image, psf, &Cmle::new())
}

pub fn cmle_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &Cmle,
) -> Result<(DynamicImage, SolveReport)> {
    validate_cmle(config)?;
    let rl = RichardsonLucy::new()
        .iterations(config.iterations)
        .relative_update_tolerance(config.relative_update_tolerance)
        .filter_epsilon(config.filter_epsilon)
        .damping(acuity_to_damping(config.acuity)?)
        .readout_noise(snr_to_readout_noise(config.snr)?)
        .positivity(true)
        .channel_mode(config.channel_mode)
        .range_policy(config.range_policy)
        .collect_history(config.collect_history);
    richardson_lucy_with(image, psf, &rl)
}

pub fn gmle(image: &DynamicImage, psf: &Kernel2D) -> Result<(DynamicImage, SolveReport)> {
    gmle_with(image, psf, &Gmle::new())
}

pub fn gmle_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &Gmle,
) -> Result<(DynamicImage, SolveReport)> {
    validate_gmle(config)?;
    let snr_noise = snr_to_readout_noise(config.snr)?;
    let noise_damping = (2.0 / config.snr.max(1.0)).sqrt().clamp(0.0, 1.0);
    let damping = acuity_to_damping(config.acuity)?
        .unwrap_or(0.0)
        .max(noise_damping);
    let tv_weight = ((config.roughness / config.snr.max(1.0)) * 0.8).clamp(0.0, 0.15);
    let rl_tv = RichardsonLucyTv::new()
        .iterations(config.iterations)
        .relative_update_tolerance(config.relative_update_tolerance)
        .filter_epsilon(config.filter_epsilon)
        .damping(Some(damping))
        .readout_noise(snr_noise)
        .positivity(true)
        .channel_mode(config.channel_mode)
        .range_policy(config.range_policy)
        .collect_history(config.collect_history)
        .tv_weight(tv_weight)
        .tv_epsilon(config.tv_epsilon);
    richardson_lucy_tv_with(image, psf, &rl_tv)
}

pub fn qmle(image: &DynamicImage, psf: &Kernel2D) -> Result<(DynamicImage, SolveReport)> {
    qmle_with(image, psf, &Qmle::new())
}

pub fn qmle_with(
    image: &DynamicImage,
    psf: &Kernel2D,
    config: &Qmle,
) -> Result<(DynamicImage, SolveReport)> {
    validate_qmle(config)?;
    let rl = RichardsonLucy::new()
        .iterations(config.iterations)
        .relative_update_tolerance(config.relative_update_tolerance)
        .filter_epsilon(config.filter_epsilon)
        .damping(acuity_to_damping(config.acuity)?)
        .readout_noise(snr_to_readout_noise(config.snr)? * 0.5)
        .positivity(true)
        .channel_mode(config.channel_mode)
        .range_policy(config.range_policy)
        .collect_history(config.collect_history);
    richardson_lucy_with(image, psf, &rl)
}

fn snr_to_readout_noise(snr: f32) -> Result<f32> {
    if !snr.is_finite() || snr <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    let noise = (1.0 / snr).clamp(0.0, 0.25);
    if !noise.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(noise)
}

fn acuity_to_damping(acuity: f32) -> Result<Option<f32>> {
    if !acuity.is_finite() || acuity <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    if acuity >= 1.0 {
        return Ok(None);
    }
    let damping = (1.0 / acuity - 1.0).clamp(0.0, 1.0);
    if !damping.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(Some(damping))
}

fn validate_cmle(config: &Cmle) -> Result<()> {
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
    snr_to_readout_noise(config.snr)?;
    let _ = acuity_to_damping(config.acuity)?;
    Ok(())
}

fn validate_gmle(config: &Gmle) -> Result<()> {
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
    if !config.roughness.is_finite() || config.roughness < 0.0 {
        return Err(Error::InvalidParameter);
    }
    if !config.tv_epsilon.is_finite() || config.tv_epsilon <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    snr_to_readout_noise(config.snr)?;
    let _ = acuity_to_damping(config.acuity)?;
    Ok(())
}

fn validate_qmle(config: &Qmle) -> Result<()> {
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
    snr_to_readout_noise(config.snr)?;
    let _ = acuity_to_damping(config.acuity)?;
    Ok(())
}
