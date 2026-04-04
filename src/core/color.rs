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

pub(crate) fn sample_to_f32(value: u8) -> f32 {
    f32::from(value)
}

pub(crate) fn sample_from_f32(value: f32) -> Result<u8> {
    if !value.is_finite() {
        return Err(Error::NonFiniteInput);
    }

    let rounded = value.round().clamp(0.0, 255.0);
    Ok(rounded as u8)
}
