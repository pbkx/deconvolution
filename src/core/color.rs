use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PixelLayout {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
}

impl PixelLayout {
    pub(crate) fn color_channels(self) -> usize {
        match self {
            Self::Gray | Self::GrayAlpha => 1,
            Self::Rgb | Self::Rgba => 3,
        }
    }

    pub(crate) fn has_alpha(self) -> bool {
        matches!(self, Self::GrayAlpha | Self::Rgba)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SampleKind {
    U8,
    U16,
    F32,
}

impl SampleKind {
    pub(crate) fn alpha_denominator(self) -> f32 {
        match self {
            Self::U8 => 255.0,
            Self::U16 => 65_535.0,
            Self::F32 => 1.0,
        }
    }
}

pub(crate) trait PixelSample: Copy {
    const KIND: SampleKind;

    fn to_f32(self) -> Result<f32>;

    fn from_f32(value: f32) -> Result<Self>;
}

impl PixelSample for u8 {
    const KIND: SampleKind = SampleKind::U8;

    fn to_f32(self) -> Result<f32> {
        Ok(f32::from(self))
    }

    fn from_f32(value: f32) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::NonFiniteInput);
        }

        let rounded = value.round().clamp(0.0, 255.0);
        Ok(rounded as u8)
    }
}

impl PixelSample for u16 {
    const KIND: SampleKind = SampleKind::U16;

    fn to_f32(self) -> Result<f32> {
        Ok(f32::from(self))
    }

    fn from_f32(value: f32) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::NonFiniteInput);
        }

        let rounded = value.round().clamp(0.0, 65_535.0);
        Ok(rounded as u16)
    }
}

impl PixelSample for f32 {
    const KIND: SampleKind = SampleKind::F32;

    fn to_f32(self) -> Result<f32> {
        if !self.is_finite() {
            return Err(Error::NonFiniteInput);
        }

        Ok(self)
    }

    fn from_f32(value: f32) -> Result<Self> {
        if !value.is_finite() {
            return Err(Error::NonFiniteInput);
        }

        Ok(value)
    }
}

pub(crate) fn sample_to_f32<T: PixelSample>(value: T) -> Result<f32> {
    value.to_f32()
}

pub(crate) fn sample_from_f32<T: PixelSample>(value: f32) -> Result<T> {
    T::from_f32(value)
}
