use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::vec::vec3f64::Vec3f64;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Mat3f64 {
    /// Values in row major order
    pub cols: [Vec3f64; 3],
}

impl Mat3f64 {
    pub fn new_row_major(row1: Vec3f64, row2: Vec3f64, row3: Vec3f64) -> Self {
        Self {
            cols: [
                Vec3f64::new(row1.x, row2.x, row3.x),
                Vec3f64::new(row1.y, row2.y, row3.y),
                Vec3f64::new(row1.z, row2.z, row3.z),
            ],
        }
    }

    pub fn new_column_major(col1: Vec3f64, col2: Vec3f64, col3: Vec3f64) -> Self {
        Self {
            cols: [col1, col2, col3],
        }
    }

    pub fn zero() -> Self {
        Self {
            cols: [Vec3f64::zero(); 3],
        }
    }

    pub fn identity() -> Self {
        Self {
            cols: [
                Vec3f64::new(1.0, 0.0, 0.0),
                Vec3f64::new(0.0, 1.0, 0.0),
                Vec3f64::new(0.0, 0.0, 1.0),
            ],
        }
    }

    pub fn transpose(&mut self) {
        self.cols = self.transposed().cols;
    }

    pub fn transposed(&self) -> Self {
        Self::new_row_major(self.cols[0], self.cols[1], self.cols[2])
    }
}

impl Add<Mat3f64> for Mat3f64 {
    type Output = Mat3f64;
    fn add(mut self, rhs: Mat3f64) -> Self::Output {
        self.cols[0] += rhs.cols[0];
        self.cols[1] += rhs.cols[1];
        self.cols[2] += rhs.cols[2];
        self
    }
}

impl AddAssign<Mat3f64> for Mat3f64 {
    fn add_assign(&mut self, rhs: Mat3f64) {
        self.cols[0] += rhs.cols[0];
        self.cols[1] += rhs.cols[1];
        self.cols[2] += rhs.cols[2];
    }
}

impl Sub<Mat3f64> for Mat3f64 {
    type Output = Mat3f64;
    fn sub(mut self, rhs: Mat3f64) -> Self::Output {
        self.cols[0] -= rhs.cols[0];
        self.cols[1] -= rhs.cols[1];
        self.cols[2] -= rhs.cols[2];
        self
    }
}

impl SubAssign<Mat3f64> for Mat3f64 {
    fn sub_assign(&mut self, rhs: Mat3f64) {
        self.cols[0] -= rhs.cols[0];
        self.cols[1] -= rhs.cols[1];
        self.cols[2] -= rhs.cols[2];
    }
}

impl Mul<Mat3f64> for Mat3f64 {
    type Output = Mat3f64;
    fn mul(self, rhs: Mat3f64) -> Self::Output {
        let a11 = self.cols[0].x;
        let a21 = self.cols[0].y;
        let a31 = self.cols[0].z;
        let a12 = self.cols[1].x;
        let a22 = self.cols[1].y;
        let a32 = self.cols[1].z;
        let a13 = self.cols[2].x;
        let a23 = self.cols[2].y;
        let a33 = self.cols[2].z;

        let b11 = rhs.cols[0].x;
        let b21 = rhs.cols[0].y;
        let b31 = rhs.cols[0].z;
        let b12 = rhs.cols[1].x;
        let b22 = rhs.cols[1].y;
        let b32 = rhs.cols[1].z;
        let b13 = rhs.cols[2].x;
        let b23 = rhs.cols[2].y;
        let b33 = rhs.cols[2].z;

        Mat3f64 {
            cols: [
                Vec3f64::new(
                    a11 * b11 + a12 * b21 + a13 * b31,
                    a21 * b11 + a22 * b21 + a23 * b31,
                    a31 * b11 + a32 * b21 + a33 * b31,
                ),
                Vec3f64::new(
                    a11 * b12 + a12 * b22 + a13 * b32,
                    a21 * b12 + a22 * b22 + a23 * b32,
                    a31 * b12 + a32 * b22 + a33 * b32,
                ),
                Vec3f64::new(
                    a11 * b13 + a12 * b23 + a13 * b33,
                    a21 * b13 + a22 * b23 + a23 * b33,
                    a31 * b13 + a32 * b23 + a33 * b33,
                ),
            ],
        }
    }
}

impl MulAssign<Mat3f64> for Mat3f64 {
    fn mul_assign(&mut self, rhs: Mat3f64) {
        let result = *self * rhs;
        self.cols = result.cols;
    }
}

impl Mul<f64> for Mat3f64 {
    type Output = Mat3f64;
    fn mul(mut self, scalar: f64) -> Self::Output {
        self.cols[0] *= scalar;
        self.cols[1] *= scalar;
        self.cols[2] *= scalar;
        self
    }
}

impl Mul<Mat3f64> for f64 {
    type Output = Mat3f64;
    fn mul(self, mut mat: Mat3f64) -> Self::Output {
        mat.cols[0] *= self;
        mat.cols[1] *= self;
        mat.cols[2] *= self;
        mat
    }
}

impl MulAssign<f64> for Mat3f64 {
    fn mul_assign(&mut self, scalar: f64) {
        self.cols[0] *= scalar;
        self.cols[1] *= scalar;
        self.cols[2] *= scalar;
    }
}

impl Mul<Vec3f64> for Mat3f64 {
    type Output = Vec3f64;
    fn mul(self, rhs: Vec3f64) -> Self::Output {
        Vec3f64::new(
            self.cols[0].x * rhs.x + self.cols[1].x * rhs.y + self.cols[2].x * rhs.z,
            self.cols[0].y * rhs.x + self.cols[1].y * rhs.y + self.cols[2].y * rhs.z,
            self.cols[0].z * rhs.x + self.cols[1].z * rhs.y + self.cols[2].z * rhs.z,
        )
    }
}

impl Div<f64> for Mat3f64 {
    type Output = Mat3f64;
    fn div(mut self, scalar: f64) -> Self::Output {
        self.cols[0] /= scalar;
        self.cols[1] /= scalar;
        self.cols[2] /= scalar;
        self
    }
}

impl DivAssign<f64> for Mat3f64 {
    fn div_assign(&mut self, scalar: f64) {
        self.cols[0] /= scalar;
        self.cols[1] /= scalar;
        self.cols[2] /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::Mat3f64;
    use crate::vec::vec3f64::Vec3f64;

    #[test]
    fn matrix_creation() {
        let m = Mat3f64::new_column_major(
            Vec3f64::new(1.0, 2.0, 3.0),
            Vec3f64::new(4.0, 5.0, 6.0),
            Vec3f64::new(7.0, 8.0, 9.0),
        );
        assert_eq!(m.cols[0].x, 1.0);
        assert_eq!(m.cols[0].y, 2.0);
        assert_eq!(m.cols[0].z, 3.0);
        assert_eq!(m.cols[1].x, 4.0);
        assert_eq!(m.cols[1].y, 5.0);
        assert_eq!(m.cols[1].z, 6.0);
        assert_eq!(m.cols[2].x, 7.0);
        assert_eq!(m.cols[2].y, 8.0);
        assert_eq!(m.cols[2].z, 9.0);

        let m = Mat3f64::new_row_major(
            Vec3f64::new(1.0, 4.0, 7.0),
            Vec3f64::new(2.0, 5.0, 8.0),
            Vec3f64::new(3.0, 6.0, 9.0),
        );
        assert_eq!(m.cols[0].x, 1.0);
        assert_eq!(m.cols[0].y, 2.0);
        assert_eq!(m.cols[0].z, 3.0);
        assert_eq!(m.cols[1].x, 4.0);
        assert_eq!(m.cols[1].y, 5.0);
        assert_eq!(m.cols[1].z, 6.0);
        assert_eq!(m.cols[2].x, 7.0);
        assert_eq!(m.cols[2].y, 8.0);
        assert_eq!(m.cols[2].z, 9.0);
    }

    #[test]
    fn zero() {
        let zero = Mat3f64::zero();
        assert_eq!(zero.cols[0].x, 0.0);
        assert_eq!(zero.cols[0].y, 0.0);
        assert_eq!(zero.cols[0].z, 0.0);
        assert_eq!(zero.cols[1].x, 0.0);
        assert_eq!(zero.cols[1].y, 0.0);
        assert_eq!(zero.cols[1].z, 0.0);
        assert_eq!(zero.cols[2].x, 0.0);
        assert_eq!(zero.cols[2].y, 0.0);
        assert_eq!(zero.cols[2].z, 0.0);
    }

    #[test]
    fn identity() {
        let id = Mat3f64::identity();
        assert_eq!(id.cols[0].x, 1.0);
        assert_eq!(id.cols[0].y, 0.0);
        assert_eq!(id.cols[0].z, 0.0);
        assert_eq!(id.cols[1].x, 0.0);
        assert_eq!(id.cols[1].y, 1.0);
        assert_eq!(id.cols[1].z, 0.0);
        assert_eq!(id.cols[2].x, 0.0);
        assert_eq!(id.cols[2].y, 0.0);
        assert_eq!(id.cols[2].z, 1.0);
    }

    #[test]
    fn transpose() {
        let mut m = Mat3f64::new_column_major(
            Vec3f64::new(1.0, 2.0, 3.0),
            Vec3f64::new(4.0, 5.0, 6.0),
            Vec3f64::new(7.0, 8.0, 9.0),
        );
        m.transpose();
        assert_eq!(m.cols[0].x, 1.0);
        assert_eq!(m.cols[0].y, 4.0);
        assert_eq!(m.cols[0].z, 7.0);
        assert_eq!(m.cols[1].x, 2.0);
        assert_eq!(m.cols[1].y, 5.0);
        assert_eq!(m.cols[1].z, 8.0);
        assert_eq!(m.cols[2].x, 3.0);
        assert_eq!(m.cols[2].y, 6.0);
        assert_eq!(m.cols[2].z, 9.0);
    }

    #[test]
    fn addition() {
        let mut one = Mat3f64::new_column_major(
            Vec3f64::new(1.0, 1.0, 1.0),
            Vec3f64::new(1.0, 1.0, 1.0),
            Vec3f64::new(1.0, 1.0, 1.0),
        );
        let two = one + one;
        assert_eq!(two.cols[0].x, 2.0);
        assert_eq!(two.cols[0].y, 2.0);
        assert_eq!(two.cols[0].z, 2.0);
        assert_eq!(two.cols[1].x, 2.0);
        assert_eq!(two.cols[1].y, 2.0);
        assert_eq!(two.cols[1].z, 2.0);
        assert_eq!(two.cols[2].x, 2.0);
        assert_eq!(two.cols[2].y, 2.0);
        assert_eq!(two.cols[2].z, 2.0);

        one += one;
        assert_eq!(one.cols[0].x, 2.0);
        assert_eq!(one.cols[0].y, 2.0);
        assert_eq!(one.cols[0].z, 2.0);
        assert_eq!(one.cols[1].x, 2.0);
        assert_eq!(one.cols[1].y, 2.0);
        assert_eq!(one.cols[1].z, 2.0);
        assert_eq!(one.cols[2].x, 2.0);
        assert_eq!(one.cols[2].y, 2.0);
        assert_eq!(one.cols[2].z, 2.0);
    }

    #[test]
    fn subtraction() {
        let one = Mat3f64::new_column_major(
            Vec3f64::new(1.0, 1.0, 1.0),
            Vec3f64::new(1.0, 1.0, 1.0),
            Vec3f64::new(1.0, 1.0, 1.0),
        );
        let mut zero = one - one;
        assert_eq!(zero.cols[0].x, 0.0);
        assert_eq!(zero.cols[0].y, 0.0);
        assert_eq!(zero.cols[0].z, 0.0);
        assert_eq!(zero.cols[1].x, 0.0);
        assert_eq!(zero.cols[1].y, 0.0);
        assert_eq!(zero.cols[1].z, 0.0);
        assert_eq!(zero.cols[2].x, 0.0);
        assert_eq!(zero.cols[2].y, 0.0);
        assert_eq!(zero.cols[2].z, 0.0);

        zero -= one;
        assert_eq!(zero.cols[0].x, -1.0);
        assert_eq!(zero.cols[0].y, -1.0);
        assert_eq!(zero.cols[0].z, -1.0);
        assert_eq!(zero.cols[1].x, -1.0);
        assert_eq!(zero.cols[1].y, -1.0);
        assert_eq!(zero.cols[1].z, -1.0);
        assert_eq!(zero.cols[2].x, -1.0);
        assert_eq!(zero.cols[2].y, -1.0);
        assert_eq!(zero.cols[2].z, -1.0);
    }

    #[test]
    fn multiplication() {
        let mut a = Mat3f64::new_column_major(
            Vec3f64::new(1.0, 2.0, 3.0),
            Vec3f64::new(4.0, 5.0, 6.0),
            Vec3f64::new(7.0, 8.0, 9.0),
        );
        let b = Mat3f64::new_column_major(
            Vec3f64::new(10.0, 11.0, 12.0),
            Vec3f64::new(13.0, 14.0, 15.0),
            Vec3f64::new(16.0, 17.0, 18.0),
        );
        let c = a * b;
        assert_eq!(c.cols[0].x, 138.0);
        assert_eq!(c.cols[0].y, 171.0);
        assert_eq!(c.cols[0].z, 204.0);
        assert_eq!(c.cols[1].x, 174.0);
        assert_eq!(c.cols[1].y, 216.0);
        assert_eq!(c.cols[1].z, 258.0);
        assert_eq!(c.cols[2].x, 210.0);
        assert_eq!(c.cols[2].y, 261.0);
        assert_eq!(c.cols[2].z, 312.0);
        a *= b;
        assert_eq!(a.cols[0].x, 138.0);
        assert_eq!(a.cols[0].y, 171.0);
        assert_eq!(a.cols[0].z, 204.0);
        assert_eq!(a.cols[1].x, 174.0);
        assert_eq!(a.cols[1].y, 216.0);
        assert_eq!(a.cols[1].z, 258.0);
        assert_eq!(a.cols[2].x, 210.0);
        assert_eq!(a.cols[2].y, 261.0);
        assert_eq!(a.cols[2].z, 312.0);
    }

    #[test]
    fn scalar_multiplication() {
        let one = Mat3f64::new_column_major(
            Vec3f64::new(1.0, 1.0, 1.0),
            Vec3f64::new(1.0, 1.0, 1.0),
            Vec3f64::new(1.0, 1.0, 1.0),
        );
        let mut four: Mat3f64 = 2.0 * one * 2.0;
        assert_eq!(four.cols[0].x, 4.0);
        assert_eq!(four.cols[0].y, 4.0);
        assert_eq!(four.cols[0].z, 4.0);
        assert_eq!(four.cols[1].x, 4.0);
        assert_eq!(four.cols[1].y, 4.0);
        assert_eq!(four.cols[1].z, 4.0);
        assert_eq!(four.cols[2].x, 4.0);
        assert_eq!(four.cols[2].y, 4.0);
        assert_eq!(four.cols[2].z, 4.0);

        four *= 2.0;
        assert_eq!(four.cols[0].x, 8.0);
        assert_eq!(four.cols[0].y, 8.0);
        assert_eq!(four.cols[0].z, 8.0);
        assert_eq!(four.cols[1].x, 8.0);
        assert_eq!(four.cols[1].y, 8.0);
        assert_eq!(four.cols[1].z, 8.0);
        assert_eq!(four.cols[2].x, 8.0);
        assert_eq!(four.cols[2].y, 8.0);
        assert_eq!(four.cols[2].z, 8.0);
    }

    #[test]
    fn scalar_division() {
        let one = Mat3f64::new_column_major(
            Vec3f64::new(1.0, 1.0, 1.0),
            Vec3f64::new(1.0, 1.0, 1.0),
            Vec3f64::new(1.0, 1.0, 1.0),
        );
        let mut half: Mat3f64 = one / 2.0;
        assert_eq!(half.cols[0].x, 0.5);
        assert_eq!(half.cols[0].y, 0.5);
        assert_eq!(half.cols[0].z, 0.5);
        assert_eq!(half.cols[1].x, 0.5);
        assert_eq!(half.cols[1].y, 0.5);
        assert_eq!(half.cols[1].z, 0.5);
        assert_eq!(half.cols[2].x, 0.5);
        assert_eq!(half.cols[2].y, 0.5);
        assert_eq!(half.cols[2].z, 0.5);

        half /= 2.0;
        assert_eq!(half.cols[0].x, 0.25);
        assert_eq!(half.cols[0].y, 0.25);
        assert_eq!(half.cols[0].z, 0.25);
        assert_eq!(half.cols[1].x, 0.25);
        assert_eq!(half.cols[1].y, 0.25);
        assert_eq!(half.cols[1].z, 0.25);
        assert_eq!(half.cols[2].x, 0.25);
        assert_eq!(half.cols[2].y, 0.25);
        assert_eq!(half.cols[2].z, 0.25);
    }

    #[test]
    fn multiply_vector() {
        let a = Mat3f64::new_column_major(
            Vec3f64::new(1.0, 4.0, 7.0),
            Vec3f64::new(2.0, 5.0, 8.0),
            Vec3f64::new(3.0, 6.0, 9.0),
        );
        let vec = Vec3f64::new(2.0, 3.0, 4.0);
        let c = a * vec;
        assert_eq!(c.x, 20.0);
        assert_eq!(c.y, 47.0);
        assert_eq!(c.z, 74.0);
    }
}
