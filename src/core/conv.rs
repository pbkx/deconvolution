use ndarray::{Array2, Array3};

use crate::psf::{Kernel2D, Kernel3D};
use crate::{Error, Result};

use super::operator::{LinearOperator2D, LinearOperator3D};
use super::validate::{finite_real_2d, finite_real_3d};

pub(crate) struct Convolution2D {
    kernel: Array2<f32>,
}

impl Convolution2D {
    pub(crate) fn new(kernel: &Kernel2D) -> Result<Self> {
        let kernel = kernel.as_array().as_standard_layout().to_owned();
        finite_real_2d(&kernel)?;

        Ok(Self { kernel })
    }
}

impl LinearOperator2D for Convolution2D {
    fn apply(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        convolve_2d_same(input, &self.kernel)
    }

    fn adjoint(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        correlate_2d_same(input, &self.kernel)
    }
}

pub(crate) struct Convolution3D {
    kernel: Array3<f32>,
}

impl Convolution3D {
    pub(crate) fn new(kernel: &Kernel3D) -> Result<Self> {
        let kernel = kernel.as_array().as_standard_layout().to_owned();
        finite_real_3d(&kernel)?;

        Ok(Self { kernel })
    }
}

impl LinearOperator3D for Convolution3D {
    fn apply(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        convolve_3d_same(input, &self.kernel)
    }

    fn adjoint(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        correlate_3d_same(input, &self.kernel)
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
        let y_i64 = y as i64;
        for x in 0..width {
            let x_i64 = x as i64;
            let mut acc = 0.0_f32;
            for ky in 0..kernel_h {
                let ky_i64 = ky as i64;
                let iy = y_i64 + center_y - ky_i64;
                if iy < 0 || iy >= height_i64 {
                    continue;
                }
                let iy_usize = iy as usize;
                for kx in 0..kernel_w {
                    let kx_i64 = kx as i64;
                    let ix = x_i64 + center_x - kx_i64;
                    if ix < 0 || ix >= width_i64 {
                        continue;
                    }

                    let ix_usize = ix as usize;
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
        let d_i64 = d as i64;
        for y in 0..height {
            let y_i64 = y as i64;
            for x in 0..width {
                let x_i64 = x as i64;
                let mut acc = 0.0_f32;
                for kd in 0..kernel_d {
                    let kd_i64 = kd as i64;
                    let id = d_i64 + center_d - kd_i64;
                    if id < 0 || id >= depth_i64 {
                        continue;
                    }
                    let id_usize = id as usize;
                    for ky in 0..kernel_h {
                        let ky_i64 = ky as i64;
                        let iy = y_i64 + center_y - ky_i64;
                        if iy < 0 || iy >= height_i64 {
                            continue;
                        }
                        let iy_usize = iy as usize;
                        for kx in 0..kernel_w {
                            let kx_i64 = kx as i64;
                            let ix = x_i64 + center_x - kx_i64;
                            if ix < 0 || ix >= width_i64 {
                                continue;
                            }

                            let ix_usize = ix as usize;
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

fn correlate_2d_same(input: &Array2<f32>, kernel: &Array2<f32>) -> Result<Array2<f32>> {
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
        let y_i64 = y as i64;
        for x in 0..width {
            let x_i64 = x as i64;
            let mut acc = 0.0_f32;
            for ky in 0..kernel_h {
                let ky_i64 = ky as i64;
                let iy = y_i64 + ky_i64 - center_y;
                if iy < 0 || iy >= height_i64 {
                    continue;
                }
                let iy_usize = iy as usize;
                for kx in 0..kernel_w {
                    let kx_i64 = kx as i64;
                    let ix = x_i64 + kx_i64 - center_x;
                    if ix < 0 || ix >= width_i64 {
                        continue;
                    }

                    let ix_usize = ix as usize;
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

fn correlate_3d_same(input: &Array3<f32>, kernel: &Array3<f32>) -> Result<Array3<f32>> {
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
        let d_i64 = d as i64;
        for y in 0..height {
            let y_i64 = y as i64;
            for x in 0..width {
                let x_i64 = x as i64;
                let mut acc = 0.0_f32;
                for kd in 0..kernel_d {
                    let kd_i64 = kd as i64;
                    let id = d_i64 + kd_i64 - center_d;
                    if id < 0 || id >= depth_i64 {
                        continue;
                    }
                    let id_usize = id as usize;
                    for ky in 0..kernel_h {
                        let ky_i64 = ky as i64;
                        let iy = y_i64 + ky_i64 - center_y;
                        if iy < 0 || iy >= height_i64 {
                            continue;
                        }
                        let iy_usize = iy as usize;
                        for kx in 0..kernel_w {
                            let kx_i64 = kx as i64;
                            let ix = x_i64 + kx_i64 - center_x;
                            if ix < 0 || ix >= width_i64 {
                                continue;
                            }

                            let ix_usize = ix as usize;
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

fn to_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::DimensionMismatch)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3, array};

    use super::{Convolution2D, Convolution3D};
    use crate::Kernel2D;
    use crate::Kernel3D;
    use crate::core::operator::{
        LinearOperator2D, LinearOperator3D, inner_product_2d, inner_product_3d,
    };
    use crate::simulate::blur::blur;

    #[test]
    fn convolution_2d_matches_fft_blur_for_asymmetric_kernel() {
        let kernel = Kernel2D::new(array![
            [0.01_f32, 0.02_f32, 0.03_f32],
            [0.04_f32, 0.25_f32, 0.06_f32],
            [0.07_f32, 0.08_f32, 0.09_f32]
        ])
        .unwrap();
        let op = Convolution2D::new(&kernel).unwrap();
        let mut input = Array2::zeros((5, 5));
        input[[2, 2]] = 1.0;

        let direct = op.apply(&input).unwrap();
        let fft = blur(&input, &kernel).unwrap();

        assert_max_abs_diff_2d(&direct, &fft, 1e-5);
    }

    #[test]
    fn convolution_2d_preserves_asymmetric_kernel_orientation_on_impulse() {
        let expected = array![
            [1.0_f32, 2.0_f32, 3.0_f32],
            [4.0_f32, 5.0_f32, 6.0_f32],
            [7.0_f32, 8.0_f32, 9.0_f32]
        ];
        let kernel = Kernel2D::new(expected.clone()).unwrap();
        let op = Convolution2D::new(&kernel).unwrap();
        let mut input = Array2::zeros((3, 3));
        input[[1, 1]] = 1.0;

        let output = op.apply(&input).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn convolution_2d_passes_adjoint_consistency() {
        let kernel = Kernel2D::new(array![
            [0.1_f32, 0.2_f32, 0.3_f32],
            [0.4_f32, 0.5_f32, 0.6_f32],
            [0.7_f32, 0.8_f32, 0.9_f32]
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
    fn convolution_2d_even_kernel_passes_adjoint_consistency() {
        let kernel = Kernel2D::new(array![
            [0.1_f32, 0.2_f32, 0.3_f32],
            [0.4_f32, 0.5_f32, 0.6_f32]
        ])
        .unwrap();
        let op = Convolution2D::new(&kernel).unwrap();

        let x: Array2<f32> = array![
            [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32],
            [5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32],
            [9.0_f32, 10.0_f32, 11.0_f32, 12.0_f32]
        ];
        let y: Array2<f32> = array![
            [0.5_f32, 1.5_f32, 0.5_f32, 2.0_f32],
            [2.0_f32, 1.0_f32, 0.0_f32, 1.0_f32],
            [1.5_f32, 0.5_f32, 2.5_f32, 3.0_f32]
        ];

        let ax = op.apply(&x).unwrap();
        let aty = op.adjoint(&y).unwrap();

        let lhs = inner_product_2d(&ax, &y).unwrap();
        let rhs = inner_product_2d(&x, &aty).unwrap();
        assert!((lhs - rhs).abs() < 1e-4);
    }

    #[test]
    fn convolution_3d_preserves_asymmetric_kernel_orientation_on_impulse() {
        let expected = array![
            [
                [1.0_f32, 2.0_f32, 3.0_f32],
                [4.0_f32, 5.0_f32, 6.0_f32],
                [7.0_f32, 8.0_f32, 9.0_f32]
            ],
            [
                [10.0_f32, 11.0_f32, 12.0_f32],
                [13.0_f32, 14.0_f32, 15.0_f32],
                [16.0_f32, 17.0_f32, 18.0_f32]
            ],
            [
                [19.0_f32, 20.0_f32, 21.0_f32],
                [22.0_f32, 23.0_f32, 24.0_f32],
                [25.0_f32, 26.0_f32, 27.0_f32]
            ]
        ];
        let kernel = Kernel3D::new(expected.clone()).unwrap();
        let op = Convolution3D::new(&kernel).unwrap();
        let mut input = Array3::zeros((3, 3, 3));
        input[[1, 1, 1]] = 1.0;

        let output = op.apply(&input).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn convolution_3d_passes_adjoint_consistency() {
        let kernel = Kernel3D::new(array![
            [
                [0.1_f32, 0.2_f32, 0.3_f32],
                [0.4_f32, 0.5_f32, 0.6_f32],
                [0.7_f32, 0.8_f32, 0.9_f32]
            ],
            [
                [1.0_f32, 1.1_f32, 1.2_f32],
                [1.3_f32, 1.4_f32, 1.5_f32],
                [1.6_f32, 1.7_f32, 1.8_f32]
            ],
            [
                [1.9_f32, 2.0_f32, 2.1_f32],
                [2.2_f32, 2.3_f32, 2.4_f32],
                [2.5_f32, 2.6_f32, 2.7_f32]
            ]
        ])
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

    fn assert_max_abs_diff_2d(lhs: &Array2<f32>, rhs: &Array2<f32>, tolerance: f32) {
        assert_eq!(lhs.dim(), rhs.dim());
        let mut max_diff = 0.0_f32;
        for ((y, x), lhs_value) in lhs.indexed_iter() {
            let diff = (*lhs_value - rhs[[y, x]]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(max_diff <= tolerance, "max_diff={max_diff}");
    }
}
