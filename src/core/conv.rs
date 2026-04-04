use ndarray::{Array2, Array3};

use crate::psf::{Kernel2D, Kernel3D};
use crate::{Error, Result};

use super::operator::{LinearOperator2D, LinearOperator3D};
use super::validate::{finite_real_2d, finite_real_3d};

pub(crate) struct Convolution2D {
    kernel: Array2<f32>,
    adjoint_kernel: Array2<f32>,
}

impl Convolution2D {
    pub(crate) fn new(kernel: &Kernel2D) -> Result<Self> {
        let kernel = kernel.as_array().as_standard_layout().to_owned();
        finite_real_2d(&kernel)?;
        let adjoint_kernel = flip_2d(&kernel);

        Ok(Self {
            kernel,
            adjoint_kernel,
        })
    }
}

impl LinearOperator2D for Convolution2D {
    fn apply(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        convolve_2d_same(input, &self.kernel)
    }

    fn adjoint(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        convolve_2d_same(input, &self.adjoint_kernel)
    }
}

pub(crate) struct Convolution3D {
    kernel: Array3<f32>,
    adjoint_kernel: Array3<f32>,
}

impl Convolution3D {
    pub(crate) fn new(kernel: &Kernel3D) -> Result<Self> {
        let kernel = kernel.as_array().as_standard_layout().to_owned();
        finite_real_3d(&kernel)?;
        let adjoint_kernel = flip_3d(&kernel);

        Ok(Self {
            kernel,
            adjoint_kernel,
        })
    }
}

impl LinearOperator3D for Convolution3D {
    fn apply(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        convolve_3d_same(input, &self.kernel)
    }

    fn adjoint(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        convolve_3d_same(input, &self.adjoint_kernel)
    }
}

fn convolve_2d_same(input: &Array2<f32>, kernel: &Array2<f32>) -> Result<Array2<f32>> {
    finite_real_2d(input)?;
    finite_real_2d(kernel)?;

    let (height, width) = input.dim();
    let (kernel_h, kernel_w) = kernel.dim();
    let center_y = to_i64(kernel_h / 2)?;
    let center_x = to_i64(kernel_w / 2)?;
    let height_i64 = to_i64(height)?;
    let width_i64 = to_i64(width)?;

    let mut output = Array2::zeros((height, width));

    for y in 0..height {
        let y_i64 = to_i64(y)?;
        for x in 0..width {
            let x_i64 = to_i64(x)?;
            let mut acc = 0.0_f32;
            for ky in 0..kernel_h {
                let ky_i64 = to_i64(ky)?;
                let iy = y_i64 + ky_i64 - center_y;
                if iy < 0 || iy >= height_i64 {
                    continue;
                }
                for kx in 0..kernel_w {
                    let kx_i64 = to_i64(kx)?;
                    let ix = x_i64 + kx_i64 - center_x;
                    if ix < 0 || ix >= width_i64 {
                        continue;
                    }

                    let iy_usize = to_usize(iy)?;
                    let ix_usize = to_usize(ix)?;
                    acc += input[[iy_usize, ix_usize]] * kernel[[ky, kx]];
                }
            }
            if !acc.is_finite() {
                return Err(Error::NonFiniteInput);
            }
            output[[y, x]] = acc;
        }
    }

    Ok(output)
}

fn convolve_3d_same(input: &Array3<f32>, kernel: &Array3<f32>) -> Result<Array3<f32>> {
    finite_real_3d(input)?;
    finite_real_3d(kernel)?;

    let (depth, height, width) = input.dim();
    let (kernel_d, kernel_h, kernel_w) = kernel.dim();
    let center_d = to_i64(kernel_d / 2)?;
    let center_y = to_i64(kernel_h / 2)?;
    let center_x = to_i64(kernel_w / 2)?;
    let depth_i64 = to_i64(depth)?;
    let height_i64 = to_i64(height)?;
    let width_i64 = to_i64(width)?;

    let mut output = Array3::zeros((depth, height, width));

    for d in 0..depth {
        let d_i64 = to_i64(d)?;
        for y in 0..height {
            let y_i64 = to_i64(y)?;
            for x in 0..width {
                let x_i64 = to_i64(x)?;
                let mut acc = 0.0_f32;
                for kd in 0..kernel_d {
                    let kd_i64 = to_i64(kd)?;
                    let id = d_i64 + kd_i64 - center_d;
                    if id < 0 || id >= depth_i64 {
                        continue;
                    }
                    for ky in 0..kernel_h {
                        let ky_i64 = to_i64(ky)?;
                        let iy = y_i64 + ky_i64 - center_y;
                        if iy < 0 || iy >= height_i64 {
                            continue;
                        }
                        for kx in 0..kernel_w {
                            let kx_i64 = to_i64(kx)?;
                            let ix = x_i64 + kx_i64 - center_x;
                            if ix < 0 || ix >= width_i64 {
                                continue;
                            }

                            let id_usize = to_usize(id)?;
                            let iy_usize = to_usize(iy)?;
                            let ix_usize = to_usize(ix)?;
                            acc += input[[id_usize, iy_usize, ix_usize]] * kernel[[kd, ky, kx]];
                        }
                    }
                }
                if !acc.is_finite() {
                    return Err(Error::NonFiniteInput);
                }
                output[[d, y, x]] = acc;
            }
        }
    }

    Ok(output)
}

fn flip_2d(kernel: &Array2<f32>) -> Array2<f32> {
    let (height, width) = kernel.dim();
    let mut flipped = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            flipped[[height - 1 - y, width - 1 - x]] = kernel[[y, x]];
        }
    }

    flipped
}

fn flip_3d(kernel: &Array3<f32>) -> Array3<f32> {
    let (depth, height, width) = kernel.dim();
    let mut flipped = Array3::zeros((depth, height, width));

    for d in 0..depth {
        for y in 0..height {
            for x in 0..width {
                flipped[[depth - 1 - d, height - 1 - y, width - 1 - x]] = kernel[[d, y, x]];
            }
        }
    }

    flipped
}

fn to_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::DimensionMismatch)
}

fn to_usize(value: i64) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::DimensionMismatch)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2, Array3};

    use super::{Convolution2D, Convolution3D};
    use crate::core::operator::{
        inner_product_2d, inner_product_3d, LinearOperator2D, LinearOperator3D,
    };
    use crate::Kernel2D;
    use crate::Kernel3D;

    #[test]
    fn convolution_2d_passes_adjoint_consistency() {
        let kernel = Kernel2D::new(array![
            [0.0_f32, 1.0_f32, 0.0_f32],
            [1.0_f32, 2.0_f32, 1.0_f32],
            [0.0_f32, 1.0_f32, 0.0_f32]
        ])
        .unwrap();
        let op = Convolution2D::new(&kernel).unwrap();

        let x: Array2<f32> = array![[1.0_f32, 2.0_f32, 3.0_f32], [4.0_f32, 5.0_f32, 6.0_f32]];
        let y: Array2<f32> = array![[0.5_f32, 1.5_f32, 0.5_f32], [2.0_f32, 1.0_f32, 0.0_f32]];

        let ax = op.apply(&x).unwrap();
        let aty = op.adjoint(&y).unwrap();

        let lhs = inner_product_2d(&ax, &y).unwrap();
        let rhs = inner_product_2d(&x, &aty).unwrap();
        assert!((lhs - rhs).abs() < 1e-4);
    }

    #[test]
    fn convolution_3d_passes_adjoint_consistency() {
        let kernel = Kernel3D::new(array![[
            [0.0_f32, 0.0_f32, 0.0_f32],
            [0.0_f32, 1.0_f32, 0.0_f32],
            [0.0_f32, 0.0_f32, 0.0_f32]
        ]])
        .unwrap();
        let op = Convolution3D::new(&kernel).unwrap();

        let x: Array3<f32> = array![
            [[1.0_f32, 2.0_f32], [3.0_f32, 4.0_f32]],
            [[5.0_f32, 6.0_f32], [7.0_f32, 8.0_f32]]
        ];
        let y: Array3<f32> = array![
            [[0.5_f32, 0.0_f32], [1.0_f32, 1.5_f32]],
            [[2.0_f32, 0.5_f32], [1.0_f32, 0.0_f32]]
        ];

        let ax = op.apply(&x).unwrap();
        let aty = op.adjoint(&y).unwrap();

        let lhs = inner_product_3d(&ax, &y).unwrap();
        let rhs = inner_product_3d(&x, &aty).unwrap();
        assert!((lhs - rhs).abs() < 1e-4);
    }
}
