use ndarray::{Array2, Array3};

use crate::core::conv::{Convolution2D, Convolution3D};
use crate::core::fft::{
    fft2_forward_real, fft2_inverse_complex, fft3_forward_real, fft3_inverse_complex,
};
use crate::core::operator::{LinearOperator2D, LinearOperator3D};
use crate::core::plan_cache::PlanCache;
use crate::core::validate::{finite_real_2d, finite_real_3d};
use crate::otf::convert::psf2otf;
use crate::otf::{Transfer2D, Transfer3D};
use crate::psf::{Kernel2D, Kernel3D};
use crate::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub enum RegOperator2D<'a> {
    Identity,
    Laplacian,
    Gradient,
    CustomKernel(&'a Kernel2D),
    CustomTransfer(&'a Transfer2D),
}

impl RegOperator2D<'_> {
    pub fn apply(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        finite_real_2d(input)?;

        match self {
            Self::Identity => Ok(input.clone()),
            Self::Laplacian => laplacian_2d(input),
            Self::Gradient => gradient_apply_2d(input),
            Self::CustomKernel(kernel) => Convolution2D::new(kernel)?.apply(input),
            Self::CustomTransfer(transfer) => transfer_apply_2d(input, transfer, false),
        }
    }

    pub fn adjoint(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        finite_real_2d(input)?;

        match self {
            Self::Identity => Ok(input.clone()),
            Self::Laplacian => laplacian_2d(input),
            Self::Gradient => gradient_adjoint_2d(input),
            Self::CustomKernel(kernel) => Convolution2D::new(kernel)?.adjoint(input),
            Self::CustomTransfer(transfer) => transfer_apply_2d(input, transfer, true),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RegOperator3D<'a> {
    Identity,
    Laplacian,
    Gradient,
    CustomKernel(&'a Kernel3D),
    CustomTransfer(&'a Transfer3D),
}

impl RegOperator3D<'_> {
    pub fn apply(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        finite_real_3d(input)?;

        match self {
            Self::Identity => Ok(input.clone()),
            Self::Laplacian => laplacian_3d(input),
            Self::Gradient => gradient_apply_3d(input),
            Self::CustomKernel(kernel) => Convolution3D::new(kernel)?.apply(input),
            Self::CustomTransfer(transfer) => transfer_apply_3d(input, transfer, false),
        }
    }

    pub fn adjoint(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        finite_real_3d(input)?;

        match self {
            Self::Identity => Ok(input.clone()),
            Self::Laplacian => laplacian_3d(input),
            Self::Gradient => gradient_adjoint_3d(input),
            Self::CustomKernel(kernel) => Convolution3D::new(kernel)?.adjoint(input),
            Self::CustomTransfer(transfer) => transfer_apply_3d(input, transfer, true),
        }
    }
}

pub(crate) fn spectral_response_2d(
    operator: RegOperator2D<'_>,
    dims: (usize, usize),
) -> Result<Array2<num_complex::Complex32>> {
    let (height, width) = dims;
    if height == 0 || width == 0 {
        return Err(Error::InvalidParameter);
    }

    match operator {
        RegOperator2D::CustomTransfer(transfer) => {
            if transfer.dims() != dims {
                return Err(Error::DimensionMismatch);
            }
            Ok(transfer.as_array().as_standard_layout().to_owned())
        }
        RegOperator2D::CustomKernel(kernel) => {
            let (kh, kw) = kernel.dims();
            if kh > height || kw > width {
                return Err(Error::DimensionMismatch);
            }
            Ok(psf2otf(kernel, dims)?.into_inner())
        }
        _ => {
            let mut impulse = Array2::zeros((height, width));
            impulse[[0, 0]] = 1.0;
            let response = operator.apply(&impulse)?;
            let mut cache = PlanCache::new();
            fft2_forward_real(&response, &mut cache)
        }
    }
}

fn transfer_apply_2d(
    input: &Array2<f32>,
    transfer: &Transfer2D,
    adjoint: bool,
) -> Result<Array2<f32>> {
    if input.dim() != transfer.dims() {
        return Err(Error::DimensionMismatch);
    }

    let mut cache = PlanCache::new();
    let mut spectrum = fft2_forward_real(input, &mut cache)?;
    for (value, h) in spectrum.iter_mut().zip(transfer.as_array().iter()) {
        let weight = if adjoint { h.conj() } else { *h };
        *value *= weight;
    }
    fft2_inverse_complex(&spectrum, &mut cache)
}

fn transfer_apply_3d(
    input: &Array3<f32>,
    transfer: &Transfer3D,
    adjoint: bool,
) -> Result<Array3<f32>> {
    if input.dim() != transfer.dims() {
        return Err(Error::DimensionMismatch);
    }

    let mut cache = PlanCache::new();
    let mut spectrum = fft3_forward_real(input, &mut cache)?;
    for (value, h) in spectrum.iter_mut().zip(transfer.as_array().iter()) {
        let weight = if adjoint { h.conj() } else { *h };
        *value *= weight;
    }
    fft3_inverse_complex(&spectrum, &mut cache)
}

fn laplacian_2d(input: &Array2<f32>) -> Result<Array2<f32>> {
    let gradient = gradient_apply_2d(input)?;
    gradient_adjoint_2d(&gradient)
}

fn laplacian_3d(input: &Array3<f32>) -> Result<Array3<f32>> {
    let gradient = gradient_apply_3d(input)?;
    gradient_adjoint_3d(&gradient)
}

fn gradient_apply_2d(input: &Array2<f32>) -> Result<Array2<f32>> {
    finite_real_2d(input)?;
    let (height, width) = input.dim();
    let mut output = Array2::zeros((height, width));

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
            let value = dx + dy;
            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }

    Ok(output)
}

fn gradient_adjoint_2d(input: &Array2<f32>) -> Result<Array2<f32>> {
    finite_real_2d(input)?;
    let (height, width) = input.dim();
    let mut output = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let left = if x > 0 { input[[y, x - 1]] } else { 0.0 };
            let up = if y > 0 { input[[y - 1, x]] } else { 0.0 };
            let mut value = left + up;
            if x + 1 < width {
                value -= input[[y, x]];
            }
            if y + 1 < height {
                value -= input[[y, x]];
            }

            if !value.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = value;
        }
    }

    Ok(output)
}

fn gradient_apply_3d(input: &Array3<f32>) -> Result<Array3<f32>> {
    finite_real_3d(input)?;
    let (depth, height, width) = input.dim();
    let mut output = Array3::zeros((depth, height, width));

    for d in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let dx = if x + 1 < width {
                    input[[d, y, x + 1]] - input[[d, y, x]]
                } else {
                    0.0
                };
                let dy = if y + 1 < height {
                    input[[d, y + 1, x]] - input[[d, y, x]]
                } else {
                    0.0
                };
                let dz = if d + 1 < depth {
                    input[[d + 1, y, x]] - input[[d, y, x]]
                } else {
                    0.0
                };
                let value = dx + dy + dz;
                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                output[[d, y, x]] = value;
            }
        }
    }

    Ok(output)
}

fn gradient_adjoint_3d(input: &Array3<f32>) -> Result<Array3<f32>> {
    finite_real_3d(input)?;
    let (depth, height, width) = input.dim();
    let mut output = Array3::zeros((depth, height, width));

    for d in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let left = if x > 0 { input[[d, y, x - 1]] } else { 0.0 };
                let up = if y > 0 { input[[d, y - 1, x]] } else { 0.0 };
                let back = if d > 0 { input[[d - 1, y, x]] } else { 0.0 };
                let mut value = left + up + back;
                if x + 1 < width {
                    value -= input[[d, y, x]];
                }
                if y + 1 < height {
                    value -= input[[d, y, x]];
                }
                if d + 1 < depth {
                    value -= input[[d, y, x]];
                }

                if !value.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                output[[d, y, x]] = value;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use num_complex::Complex32;

    use super::{RegOperator2D, RegOperator3D};
    use crate::{Kernel2D, Kernel3D, Transfer2D, Transfer3D};

    #[test]
    fn regularizer_2d_behaviors_are_sane() {
        let constant = array![[2.0_f32, 2.0_f32, 2.0_f32], [2.0_f32, 2.0_f32, 2.0_f32]];

        let identity = RegOperator2D::Identity.apply(&constant).unwrap();
        assert_eq!(identity, constant);

        let laplacian = RegOperator2D::Laplacian.apply(&constant).unwrap();
        assert!(laplacian.iter().all(|value| value.abs() < 1e-6));

        let gradient = RegOperator2D::Gradient.apply(&constant).unwrap();
        assert!(gradient.iter().all(|value| value.abs() < 1e-6));

        let transfer = Transfer2D::new(array![
            [
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0)
            ],
            [
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(1.0, 0.0)
            ]
        ])
        .unwrap();
        let transfer_out = RegOperator2D::CustomTransfer(&transfer)
            .apply(&constant)
            .unwrap();
        assert!(transfer_out.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn regularizer_3d_behaviors_are_sane() {
        let constant = array![
            [[1.0_f32, 1.0_f32], [1.0_f32, 1.0_f32]],
            [[1.0_f32, 1.0_f32], [1.0_f32, 1.0_f32]]
        ];

        let identity = RegOperator3D::Identity.apply(&constant).unwrap();
        assert_eq!(identity, constant);

        let laplacian = RegOperator3D::Laplacian.apply(&constant).unwrap();
        assert!(laplacian.iter().all(|value| value.abs() < 1e-6));

        let gradient = RegOperator3D::Gradient.apply(&constant).unwrap();
        assert!(gradient.iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn custom_kernel_and_transfer_variants_execute() {
        let image2 = array![[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]];
        let kernel2 = Kernel2D::new(array![[0.0_f32, 1.0_f32], [1.0_f32, 0.0_f32]]).unwrap();
        let out2 = RegOperator2D::CustomKernel(&kernel2)
            .apply(&image2)
            .unwrap();
        assert_eq!(out2.dim(), image2.dim());

        let transfer2 = Transfer2D::new(array![
            [Complex32::new(1.0, 0.0), Complex32::new(0.5, 0.0)],
            [Complex32::new(0.25, 0.0), Complex32::new(1.0, 0.0)]
        ])
        .unwrap();
        let out2_transfer = RegOperator2D::CustomTransfer(&transfer2)
            .adjoint(&image2)
            .unwrap();
        assert_eq!(out2_transfer.dim(), image2.dim());

        let image3 = array![[[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]]];
        let kernel3 = Kernel3D::new(array![[[1.0_f32]]]).unwrap();
        let out3 = RegOperator3D::CustomKernel(&kernel3)
            .apply(&image3)
            .unwrap();
        assert_eq!(out3.dim(), image3.dim());

        let transfer3 = Transfer3D::new(array![[
            [Complex32::new(1.0, 0.0), Complex32::new(0.5, 0.0)],
            [Complex32::new(0.25, 0.0), Complex32::new(1.0, 0.0)]
        ]])
        .unwrap();
        let out3_transfer = RegOperator3D::CustomTransfer(&transfer3)
            .adjoint(&image3)
            .unwrap();
        assert_eq!(out3_transfer.dim(), image3.dim());
    }
}
