use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::vec::vec4f64::Vec4f64;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Mat4f64 {
    /// Values in row major order
    pub cols: [Vec4f64; 4],
}

impl Mat4f64 {
    pub fn new_row_major(row1: Vec4f64, row2: Vec4f64, row3: Vec4f64, row4: Vec4f64) -> Self {
        Self {
            cols: [
                Vec4f64::new(row1.x, row2.x, row3.x, row4.x),
                Vec4f64::new(row1.y, row2.y, row3.y, row4.y),
                Vec4f64::new(row1.z, row2.z, row3.z, row4.z),
                Vec4f64::new(row1.w, row2.w, row3.w, row4.w),
            ],
        }
    }

    pub fn new_column_major(col1: Vec4f64, col2: Vec4f64, col3: Vec4f64, col4: Vec4f64) -> Self {
        Self {
            cols: [col1, col2, col3, col4],
        }
    }

    pub fn zero() -> Self {
        Self {
            cols: [Vec4f64::zero(); 4],
        }
    }

    pub fn identity() -> Self {
        Self {
            cols: [
                Vec4f64::new(1.0, 0.0, 0.0, 0.0),
                Vec4f64::new(0.0, 1.0, 0.0, 0.0),
                Vec4f64::new(0.0, 0.0, 1.0, 0.0),
                Vec4f64::new(0.0, 0.0, 0.0, 1.0),
            ],
        }
    }

    pub fn transpose(&mut self) {
        self.cols = self.transposed().cols;
    }

    pub fn transposed(&self) -> Self {
        Self::new_row_major(self.cols[0], self.cols[1], self.cols[2], self.cols[3])
    }
}

impl Add<Mat4f64> for Mat4f64 {
    type Output = Mat4f64;
    fn add(mut self, rhs: Mat4f64) -> Self::Output {
        self.cols[0] += rhs.cols[0];
        self.cols[1] += rhs.cols[1];
        self.cols[2] += rhs.cols[2];
        self.cols[3] += rhs.cols[3];
        self
    }
}

impl AddAssign<Mat4f64> for Mat4f64 {
    fn add_assign(&mut self, rhs: Mat4f64) {
        self.cols[0] += rhs.cols[0];
        self.cols[1] += rhs.cols[1];
        self.cols[2] += rhs.cols[2];
        self.cols[3] += rhs.cols[3];
    }
}

impl Sub<Mat4f64> for Mat4f64 {
    type Output = Mat4f64;
    fn sub(mut self, rhs: Mat4f64) -> Self::Output {
        self.cols[0] -= rhs.cols[0];
        self.cols[1] -= rhs.cols[1];
        self.cols[2] -= rhs.cols[2];
        self.cols[3] -= rhs.cols[3];
        self
    }
}

impl SubAssign<Mat4f64> for Mat4f64 {
    fn sub_assign(&mut self, rhs: Mat4f64) {
        self.cols[0] -= rhs.cols[0];
        self.cols[1] -= rhs.cols[1];
        self.cols[2] -= rhs.cols[2];
        self.cols[3] -= rhs.cols[3];
    }
}

impl Mul<Mat4f64> for Mat4f64 {
    type Output = Mat4f64;
    fn mul(self, rhs: Mat4f64) -> Self::Output {
        let a11 = self.cols[0].x;
        let a21 = self.cols[0].y;
        let a31 = self.cols[0].z;
        let a41 = self.cols[0].w;
        let a12 = self.cols[1].x;
        let a22 = self.cols[1].y;
        let a32 = self.cols[1].z;
        let a42 = self.cols[1].w;
        let a13 = self.cols[2].x;
        let a23 = self.cols[2].y;
        let a33 = self.cols[2].z;
        let a43 = self.cols[2].w;
        let a14 = self.cols[3].x;
        let a24 = self.cols[3].y;
        let a34 = self.cols[3].z;
        let a44 = self.cols[3].w;

        let b11 = rhs.cols[0].x;
        let b21 = rhs.cols[0].y;
        let b31 = rhs.cols[0].z;
        let b41 = rhs.cols[0].w;
        let b12 = rhs.cols[1].x;
        let b22 = rhs.cols[1].y;
        let b32 = rhs.cols[1].z;
        let b42 = rhs.cols[1].w;
        let b13 = rhs.cols[2].x;
        let b23 = rhs.cols[2].y;
        let b33 = rhs.cols[2].z;
        let b43 = rhs.cols[2].w;
        let b14 = rhs.cols[3].x;
        let b24 = rhs.cols[3].y;
        let b34 = rhs.cols[3].z;
        let b44 = rhs.cols[3].w;

        Mat4f64 {
            cols: [
                Vec4f64::new(
                    a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41,
                    a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41,
                    a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41,
                    a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41,
                ),
                Vec4f64::new(
                    a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42,
                    a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42,
                    a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42,
                    a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42,
                ),
                Vec4f64::new(
                    a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43,
                    a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43,
                    a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43,
                    a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43,
                ),
                Vec4f64::new(
                    a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44,
                    a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44,
                    a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44,
                    a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44,
                ),
            ],
        }
    }
}

impl MulAssign<Mat4f64> for Mat4f64 {
    fn mul_assign(&mut self, rhs: Mat4f64) {
        let result = *self * rhs;
        self.cols = result.cols;
    }
}

impl Mul<f64> for Mat4f64 {
    type Output = Mat4f64;
    fn mul(mut self, scalar: f64) -> Self::Output {
        self.cols[0] *= scalar;
        self.cols[1] *= scalar;
        self.cols[2] *= scalar;
        self.cols[3] *= scalar;
        self
    }
}

impl Mul<Mat4f64> for f64 {
    type Output = Mat4f64;
    fn mul(self, mut mat: Mat4f64) -> Self::Output {
        mat.cols[0] *= self;
        mat.cols[1] *= self;
        mat.cols[2] *= self;
        mat.cols[3] *= self;
        mat
    }
}

impl MulAssign<f64> for Mat4f64 {
    fn mul_assign(&mut self, scalar: f64) {
        self.cols[0] *= scalar;
        self.cols[1] *= scalar;
        self.cols[2] *= scalar;
        self.cols[3] *= scalar;
    }
}

impl Mul<Vec4f64> for Mat4f64 {
    type Output = Vec4f64;
    fn mul(self, rhs: Vec4f64) -> Self::Output {
        Vec4f64::new(
            self.cols[0].x * rhs.x
                + self.cols[1].x * rhs.y
                + self.cols[2].x * rhs.z
                + self.cols[3].x * rhs.w,
            self.cols[0].y * rhs.x
                + self.cols[1].y * rhs.y
                + self.cols[2].y * rhs.z
                + self.cols[3].y * rhs.w,
            self.cols[0].z * rhs.x
                + self.cols[1].z * rhs.y
                + self.cols[2].z * rhs.z
                + self.cols[3].z * rhs.w,
            self.cols[0].w * rhs.x
                + self.cols[1].w * rhs.y
                + self.cols[2].w * rhs.z
                + self.cols[3].w * rhs.w,
        )
    }
}

impl Div<f64> for Mat4f64 {
    type Output = Mat4f64;
    fn div(mut self, scalar: f64) -> Self::Output {
        self.cols[0] /= scalar;
        self.cols[1] /= scalar;
        self.cols[2] /= scalar;
        self.cols[3] /= scalar;
        self
    }
}

impl DivAssign<f64> for Mat4f64 {
    fn div_assign(&mut self, scalar: f64) {
        self.cols[0] /= scalar;
        self.cols[1] /= scalar;
        self.cols[2] /= scalar;
        self.cols[3] /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::Mat4f64;
    use crate::vec::vec4f64::Vec4f64;

    #[test]
    fn matrix_creation() {
        let m = Mat4f64::new_column_major(
            Vec4f64::new(1.0, 2.0, 3.0, 4.0),
            Vec4f64::new(5.0, 6.0, 7.0, 8.0),
            Vec4f64::new(9.0, 10.0, 11.0, 12.0),
            Vec4f64::new(13.0, 14.0, 15.0, 16.0),
        );
        assert_eq!(m.cols[0].x, 1.0);
        assert_eq!(m.cols[0].y, 2.0);
        assert_eq!(m.cols[0].z, 3.0);
        assert_eq!(m.cols[0].w, 4.0);
        assert_eq!(m.cols[1].x, 5.0);
        assert_eq!(m.cols[1].y, 6.0);
        assert_eq!(m.cols[1].z, 7.0);
        assert_eq!(m.cols[1].w, 8.0);
        assert_eq!(m.cols[2].x, 9.0);
        assert_eq!(m.cols[2].y, 10.0);
        assert_eq!(m.cols[2].z, 11.0);
        assert_eq!(m.cols[2].w, 12.0);
        assert_eq!(m.cols[3].x, 13.0);
        assert_eq!(m.cols[3].y, 14.0);
        assert_eq!(m.cols[3].z, 15.0);
        assert_eq!(m.cols[3].w, 16.0);

        let m = Mat4f64::new_row_major(
            Vec4f64::new(1.0, 5.0, 9.0, 13.0),
            Vec4f64::new(2.0, 6.0, 10.0, 14.0),
            Vec4f64::new(3.0, 7.0, 11.0, 15.0),
            Vec4f64::new(4.0, 8.0, 12.0, 16.0),
        );
        assert_eq!(m.cols[0].x, 1.0);
        assert_eq!(m.cols[0].y, 2.0);
        assert_eq!(m.cols[0].z, 3.0);
        assert_eq!(m.cols[0].w, 4.0);
        assert_eq!(m.cols[1].x, 5.0);
        assert_eq!(m.cols[1].y, 6.0);
        assert_eq!(m.cols[1].z, 7.0);
        assert_eq!(m.cols[1].w, 8.0);
        assert_eq!(m.cols[2].x, 9.0);
        assert_eq!(m.cols[2].y, 10.0);
        assert_eq!(m.cols[2].z, 11.0);
        assert_eq!(m.cols[2].w, 12.0);
        assert_eq!(m.cols[3].x, 13.0);
        assert_eq!(m.cols[3].y, 14.0);
        assert_eq!(m.cols[3].z, 15.0);
        assert_eq!(m.cols[3].w, 16.0);
    }

    #[test]
    fn zero() {
        let zero = Mat4f64::zero();
        assert_eq!(zero.cols[0].x, 0.0);
        assert_eq!(zero.cols[0].y, 0.0);
        assert_eq!(zero.cols[0].z, 0.0);
        assert_eq!(zero.cols[0].w, 0.0);
        assert_eq!(zero.cols[1].x, 0.0);
        assert_eq!(zero.cols[1].y, 0.0);
        assert_eq!(zero.cols[1].z, 0.0);
        assert_eq!(zero.cols[1].w, 0.0);
        assert_eq!(zero.cols[2].x, 0.0);
        assert_eq!(zero.cols[2].y, 0.0);
        assert_eq!(zero.cols[2].z, 0.0);
        assert_eq!(zero.cols[2].w, 0.0);
        assert_eq!(zero.cols[3].x, 0.0);
        assert_eq!(zero.cols[3].y, 0.0);
        assert_eq!(zero.cols[3].z, 0.0);
        assert_eq!(zero.cols[3].w, 0.0);
    }

    #[test]
    fn identity() {
        let id = Mat4f64::identity();
        assert_eq!(id.cols[0].x, 1.0);
        assert_eq!(id.cols[0].y, 0.0);
        assert_eq!(id.cols[0].z, 0.0);
        assert_eq!(id.cols[0].w, 0.0);
        assert_eq!(id.cols[1].x, 0.0);
        assert_eq!(id.cols[1].y, 1.0);
        assert_eq!(id.cols[1].z, 0.0);
        assert_eq!(id.cols[1].w, 0.0);
        assert_eq!(id.cols[2].x, 0.0);
        assert_eq!(id.cols[2].y, 0.0);
        assert_eq!(id.cols[2].z, 1.0);
        assert_eq!(id.cols[2].w, 0.0);
        assert_eq!(id.cols[3].x, 0.0);
        assert_eq!(id.cols[3].y, 0.0);
        assert_eq!(id.cols[3].z, 0.0);
        assert_eq!(id.cols[3].w, 1.0);
    }

    #[test]
    fn transpose() {
        let mut m = Mat4f64::new_column_major(
            Vec4f64::new(1.0, 2.0, 3.0, 4.0),
            Vec4f64::new(5.0, 6.0, 7.0, 8.0),
            Vec4f64::new(9.0, 10.0, 11.0, 12.0),
            Vec4f64::new(13.0, 14.0, 15.0, 16.0),
        );
        m.transpose();
        assert_eq!(m.cols[0].x, 1.0);
        assert_eq!(m.cols[0].y, 5.0);
        assert_eq!(m.cols[0].z, 9.0);
        assert_eq!(m.cols[0].w, 13.0);
        assert_eq!(m.cols[1].x, 2.0);
        assert_eq!(m.cols[1].y, 6.0);
        assert_eq!(m.cols[1].z, 10.0);
        assert_eq!(m.cols[1].w, 14.0);
        assert_eq!(m.cols[2].x, 3.0);
        assert_eq!(m.cols[2].y, 7.0);
        assert_eq!(m.cols[2].z, 11.0);
        assert_eq!(m.cols[2].w, 15.0);
        assert_eq!(m.cols[3].x, 4.0);
        assert_eq!(m.cols[3].y, 8.0);
        assert_eq!(m.cols[3].z, 12.0);
        assert_eq!(m.cols[3].w, 16.0);
    }

    #[test]
    fn addition() {
        let mut one = Mat4f64::new_column_major(
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
        );
        let two = one + one;
        assert_eq!(two.cols[0].x, 2.0);
        assert_eq!(two.cols[0].y, 2.0);
        assert_eq!(two.cols[0].z, 2.0);
        assert_eq!(two.cols[0].w, 2.0);
        assert_eq!(two.cols[1].x, 2.0);
        assert_eq!(two.cols[1].y, 2.0);
        assert_eq!(two.cols[1].z, 2.0);
        assert_eq!(two.cols[1].w, 2.0);
        assert_eq!(two.cols[2].x, 2.0);
        assert_eq!(two.cols[2].y, 2.0);
        assert_eq!(two.cols[2].z, 2.0);
        assert_eq!(two.cols[2].w, 2.0);
        assert_eq!(two.cols[3].x, 2.0);
        assert_eq!(two.cols[3].y, 2.0);
        assert_eq!(two.cols[3].z, 2.0);
        assert_eq!(two.cols[3].w, 2.0);

        one += one;
        assert_eq!(one.cols[0].x, 2.0);
        assert_eq!(one.cols[0].y, 2.0);
        assert_eq!(one.cols[0].z, 2.0);
        assert_eq!(one.cols[0].w, 2.0);
        assert_eq!(one.cols[1].x, 2.0);
        assert_eq!(one.cols[1].y, 2.0);
        assert_eq!(one.cols[1].z, 2.0);
        assert_eq!(one.cols[1].w, 2.0);
        assert_eq!(one.cols[2].x, 2.0);
        assert_eq!(one.cols[2].y, 2.0);
        assert_eq!(one.cols[2].z, 2.0);
        assert_eq!(one.cols[2].w, 2.0);
        assert_eq!(one.cols[3].x, 2.0);
        assert_eq!(one.cols[3].y, 2.0);
        assert_eq!(one.cols[3].z, 2.0);
        assert_eq!(one.cols[3].w, 2.0);
    }

    #[test]
    fn subtraction() {
        let one = Mat4f64::new_column_major(
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
        );
        let mut zero = one - one;
        assert_eq!(zero.cols[0].x, 0.0);
        assert_eq!(zero.cols[0].y, 0.0);
        assert_eq!(zero.cols[0].z, 0.0);
        assert_eq!(zero.cols[0].w, 0.0);
        assert_eq!(zero.cols[1].x, 0.0);
        assert_eq!(zero.cols[1].y, 0.0);
        assert_eq!(zero.cols[1].z, 0.0);
        assert_eq!(zero.cols[1].w, 0.0);
        assert_eq!(zero.cols[2].x, 0.0);
        assert_eq!(zero.cols[2].y, 0.0);
        assert_eq!(zero.cols[2].z, 0.0);
        assert_eq!(zero.cols[2].w, 0.0);
        assert_eq!(zero.cols[3].x, 0.0);
        assert_eq!(zero.cols[3].y, 0.0);
        assert_eq!(zero.cols[3].z, 0.0);
        assert_eq!(zero.cols[3].w, 0.0);

        zero -= one;
        assert_eq!(zero.cols[0].x, -1.0);
        assert_eq!(zero.cols[0].y, -1.0);
        assert_eq!(zero.cols[0].z, -1.0);
        assert_eq!(zero.cols[0].w, -1.0);
        assert_eq!(zero.cols[1].x, -1.0);
        assert_eq!(zero.cols[1].y, -1.0);
        assert_eq!(zero.cols[1].z, -1.0);
        assert_eq!(zero.cols[1].w, -1.0);
        assert_eq!(zero.cols[2].x, -1.0);
        assert_eq!(zero.cols[2].y, -1.0);
        assert_eq!(zero.cols[2].z, -1.0);
        assert_eq!(zero.cols[2].w, -1.0);
        assert_eq!(zero.cols[3].x, -1.0);
        assert_eq!(zero.cols[3].y, -1.0);
        assert_eq!(zero.cols[3].z, -1.0);
        assert_eq!(zero.cols[3].w, -1.0);
    }

    #[test]
    fn multiplication() {
        let mut a = Mat4f64::new_column_major(
            Vec4f64::new(1.0, 2.0, 3.0, 4.0),
            Vec4f64::new(5.0, 6.0, 7.0, 8.0),
            Vec4f64::new(9.0, 10.0, 11.0, 12.0),
            Vec4f64::new(13.0, 14.0, 15.0, 16.0),
        );
        let b = Mat4f64::new_column_major(
            Vec4f64::new(17.0, 18.0, 19.0, 20.0),
            Vec4f64::new(21.0, 22.0, 23.0, 24.0),
            Vec4f64::new(25.0, 26.0, 27.0, 28.0),
            Vec4f64::new(29.0, 30.0, 31.0, 32.0),
        );
        let c = a * b;
        assert_eq!(c.cols[0].x, 538.0);
        assert_eq!(c.cols[0].y, 612.0);
        assert_eq!(c.cols[0].z, 686.0);
        assert_eq!(c.cols[0].w, 760.0);
        assert_eq!(c.cols[1].x, 650.0);
        assert_eq!(c.cols[1].y, 740.0);
        assert_eq!(c.cols[1].z, 830.0);
        assert_eq!(c.cols[1].w, 920.0);
        assert_eq!(c.cols[2].x, 762.0);
        assert_eq!(c.cols[2].y, 868.0);
        assert_eq!(c.cols[2].z, 974.0);
        assert_eq!(c.cols[2].w, 1080.0);
        assert_eq!(c.cols[3].x, 874.0);
        assert_eq!(c.cols[3].y, 996.0);
        assert_eq!(c.cols[3].z, 1118.0);
        assert_eq!(c.cols[3].w, 1240.0);
        a *= b;
        assert_eq!(a.cols[0].x, 538.0);
        assert_eq!(a.cols[0].y, 612.0);
        assert_eq!(a.cols[0].z, 686.0);
        assert_eq!(a.cols[0].w, 760.0);
        assert_eq!(a.cols[1].x, 650.0);
        assert_eq!(a.cols[1].y, 740.0);
        assert_eq!(a.cols[1].z, 830.0);
        assert_eq!(a.cols[1].w, 920.0);
        assert_eq!(a.cols[2].x, 762.0);
        assert_eq!(a.cols[2].y, 868.0);
        assert_eq!(a.cols[2].z, 974.0);
        assert_eq!(a.cols[2].w, 1080.0);
        assert_eq!(a.cols[3].x, 874.0);
        assert_eq!(a.cols[3].y, 996.0);
        assert_eq!(a.cols[3].z, 1118.0);
        assert_eq!(a.cols[3].w, 1240.0);
    }

    #[test]
    fn scalar_multiplication() {
        let one = Mat4f64::new_column_major(
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
        );
        let mut four: Mat4f64 = 2.0 * one * 2.0;
        assert_eq!(four.cols[0].x, 4.0);
        assert_eq!(four.cols[0].y, 4.0);
        assert_eq!(four.cols[0].z, 4.0);
        assert_eq!(four.cols[0].w, 4.0);
        assert_eq!(four.cols[1].x, 4.0);
        assert_eq!(four.cols[1].y, 4.0);
        assert_eq!(four.cols[1].z, 4.0);
        assert_eq!(four.cols[1].w, 4.0);
        assert_eq!(four.cols[2].x, 4.0);
        assert_eq!(four.cols[2].y, 4.0);
        assert_eq!(four.cols[2].z, 4.0);
        assert_eq!(four.cols[2].w, 4.0);
        assert_eq!(four.cols[3].x, 4.0);
        assert_eq!(four.cols[3].y, 4.0);
        assert_eq!(four.cols[3].z, 4.0);
        assert_eq!(four.cols[3].w, 4.0);

        four *= 2.0;
        assert_eq!(four.cols[0].x, 8.0);
        assert_eq!(four.cols[0].y, 8.0);
        assert_eq!(four.cols[0].z, 8.0);
        assert_eq!(four.cols[0].w, 8.0);
        assert_eq!(four.cols[1].x, 8.0);
        assert_eq!(four.cols[1].y, 8.0);
        assert_eq!(four.cols[1].z, 8.0);
        assert_eq!(four.cols[1].w, 8.0);
        assert_eq!(four.cols[2].x, 8.0);
        assert_eq!(four.cols[2].y, 8.0);
        assert_eq!(four.cols[2].z, 8.0);
        assert_eq!(four.cols[2].w, 8.0);
        assert_eq!(four.cols[3].x, 8.0);
        assert_eq!(four.cols[3].y, 8.0);
        assert_eq!(four.cols[3].z, 8.0);
        assert_eq!(four.cols[3].w, 8.0);
    }

    #[test]
    fn scalar_division() {
        let one = Mat4f64::new_column_major(
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
            Vec4f64::new(1.0, 1.0, 1.0, 1.0),
        );
        let mut half: Mat4f64 = one / 2.0;
        assert_eq!(half.cols[0].x, 0.5);
        assert_eq!(half.cols[0].y, 0.5);
        assert_eq!(half.cols[0].z, 0.5);
        assert_eq!(half.cols[0].w, 0.5);
        assert_eq!(half.cols[1].x, 0.5);
        assert_eq!(half.cols[1].y, 0.5);
        assert_eq!(half.cols[1].z, 0.5);
        assert_eq!(half.cols[1].w, 0.5);
        assert_eq!(half.cols[2].x, 0.5);
        assert_eq!(half.cols[2].y, 0.5);
        assert_eq!(half.cols[2].z, 0.5);
        assert_eq!(half.cols[2].w, 0.5);
        assert_eq!(half.cols[3].x, 0.5);
        assert_eq!(half.cols[3].y, 0.5);
        assert_eq!(half.cols[3].z, 0.5);
        assert_eq!(half.cols[3].w, 0.5);

        half /= 2.0;
        assert_eq!(half.cols[0].x, 0.25);
        assert_eq!(half.cols[0].y, 0.25);
        assert_eq!(half.cols[0].z, 0.25);
        assert_eq!(half.cols[0].w, 0.25);
        assert_eq!(half.cols[1].x, 0.25);
        assert_eq!(half.cols[1].y, 0.25);
        assert_eq!(half.cols[1].z, 0.25);
        assert_eq!(half.cols[1].w, 0.25);
        assert_eq!(half.cols[2].x, 0.25);
        assert_eq!(half.cols[2].y, 0.25);
        assert_eq!(half.cols[2].z, 0.25);
        assert_eq!(half.cols[2].w, 0.25);
        assert_eq!(half.cols[3].x, 0.25);
        assert_eq!(half.cols[3].y, 0.25);
        assert_eq!(half.cols[3].z, 0.25);
        assert_eq!(half.cols[3].w, 0.25);
    }

    #[test]
    fn multiply_vector() {
        let a = Mat4f64::new_column_major(
            Vec4f64::new(1.0, 5.0, 9.0, 13.0),
            Vec4f64::new(2.0, 6.0, 10.0, 14.0),
            Vec4f64::new(3.0, 7.0, 11.0, 15.0),
            Vec4f64::new(4.0, 8.0, 12.0, 16.0),
        );
        let vec = Vec4f64::new(2.0, 3.0, 4.0, 5.0);
        let c = a * vec;
        assert_eq!(c.x, 40.0);
        assert_eq!(c.y, 96.0);
        assert_eq!(c.z, 152.0);
        assert_eq!(c.w, 208.0);
    }
}
