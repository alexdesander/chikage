use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::vec::vec2f64::Vec2f64;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Mat2f64 {
    /// Values in row major order
    pub cols: [Vec2f64; 2],
}

impl Mat2f64 {
    pub fn new_row_major(row1: Vec2f64, row2: Vec2f64) -> Self {
        Self {
            cols: [Vec2f64::new(row1.x, row2.x), Vec2f64::new(row1.y, row2.y)],
        }
    }

    pub fn new_column_major(col1: Vec2f64, col2: Vec2f64) -> Self {
        Self { cols: [col1, col2] }
    }

    pub fn zero() -> Self {
        Self {
            cols: [Vec2f64::zero(); 2],
        }
    }

    pub fn identity() -> Self {
        Self {
            cols: [Vec2f64::new(1.0, 0.0), Vec2f64::new(0.0, 1.0)],
        }
    }

    pub fn transpose(&mut self) {
        self.cols = self.transposed().cols;
    }

    pub fn transposed(&self) -> Self {
        Self::new_row_major(self.cols[0], self.cols[1])
    }
}

impl Add<Mat2f64> for Mat2f64 {
    type Output = Mat2f64;
    fn add(mut self, rhs: Mat2f64) -> Self::Output {
        self.cols[0] += rhs.cols[0];
        self.cols[1] += rhs.cols[1];
        self
    }
}

impl AddAssign<Mat2f64> for Mat2f64 {
    fn add_assign(&mut self, rhs: Mat2f64) {
        self.cols[0] += rhs.cols[0];
        self.cols[1] += rhs.cols[1];
    }
}

impl Sub<Mat2f64> for Mat2f64 {
    type Output = Mat2f64;
    fn sub(mut self, rhs: Mat2f64) -> Self::Output {
        self.cols[0] -= rhs.cols[0];
        self.cols[1] -= rhs.cols[1];
        self
    }
}

impl SubAssign<Mat2f64> for Mat2f64 {
    fn sub_assign(&mut self, rhs: Mat2f64) {
        self.cols[0] -= rhs.cols[0];
        self.cols[1] -= rhs.cols[1];
    }
}

impl Mul<Mat2f64> for Mat2f64 {
    type Output = Mat2f64;
    fn mul(self, rhs: Mat2f64) -> Self::Output {
        let a = self.cols;
        let b = rhs.cols;
        Mat2f64 {
            cols: [
                Vec2f64::new(
                    a[0].x * b[0].x + a[1].x * b[0].y,
                    a[0].y * b[0].x + a[1].y * b[0].y,
                ),
                Vec2f64::new(
                    a[0].x * b[1].x + a[1].x * b[1].y,
                    a[0].y * b[1].x + a[1].y * b[1].y,
                ),
            ],
        }
    }
}

impl MulAssign<Mat2f64> for Mat2f64 {
    fn mul_assign(&mut self, rhs: Mat2f64) {
        let result = *self * rhs;
        self.cols = result.cols;
    }
}

impl Mul<f64> for Mat2f64 {
    type Output = Mat2f64;
    fn mul(mut self, scalar: f64) -> Self::Output {
        self.cols[0] *= scalar;
        self.cols[1] *= scalar;
        self
    }
}

impl Mul<Mat2f64> for f64 {
    type Output = Mat2f64;
    fn mul(self, mut mat: Mat2f64) -> Self::Output {
        mat.cols[0] *= self;
        mat.cols[1] *= self;
        mat
    }
}

impl MulAssign<f64> for Mat2f64 {
    fn mul_assign(&mut self, scalar: f64) {
        self.cols[0] *= scalar;
        self.cols[1] *= scalar;
    }
}

impl Mul<Vec2f64> for Mat2f64 {
    type Output = Vec2f64;
    fn mul(self, rhs: Vec2f64) -> Self::Output {
        Vec2f64::new(
            self.cols[0].x * rhs.x + self.cols[1].x * rhs.y,
            self.cols[0].y * rhs.x + self.cols[1].y * rhs.y,
        )
    }
}

impl Div<f64> for Mat2f64 {
    type Output = Mat2f64;
    fn div(mut self, scalar: f64) -> Self::Output {
        self.cols[0] /= scalar;
        self.cols[1] /= scalar;
        self
    }
}

impl DivAssign<f64> for Mat2f64 {
    fn div_assign(&mut self, scalar: f64) {
        self.cols[0] /= scalar;
        self.cols[1] /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::Mat2f64;
    use crate::vec::vec2f64::Vec2f64;

    #[test]
    fn matrix_creation() {
        let m = Mat2f64::new_column_major(Vec2f64::new(1.0, 2.0), Vec2f64::new(3.0, 4.0));
        assert_eq!(m.cols[0].x, 1.0);
        assert_eq!(m.cols[0].y, 2.0);
        assert_eq!(m.cols[1].x, 3.0);
        assert_eq!(m.cols[1].y, 4.0);

        let m = Mat2f64::new_row_major(Vec2f64::new(1.0, 3.0), Vec2f64::new(2.0, 4.0));
        assert_eq!(m.cols[0].x, 1.0);
        assert_eq!(m.cols[0].y, 2.0);
        assert_eq!(m.cols[1].x, 3.0);
        assert_eq!(m.cols[1].y, 4.0);
    }

    #[test]
    fn zero() {
        let zero = Mat2f64::zero();
        assert_eq!(zero.cols[0].x, 0.0);
        assert_eq!(zero.cols[0].y, 0.0);
        assert_eq!(zero.cols[1].x, 0.0);
        assert_eq!(zero.cols[1].y, 0.0);
    }

    #[test]
    fn identity() {
        let id = Mat2f64::identity();
        assert_eq!(id.cols[0].x, 1.0);
        assert_eq!(id.cols[0].y, 0.0);
        assert_eq!(id.cols[1].x, 0.0);
        assert_eq!(id.cols[1].y, 1.0);
    }

    #[test]
    fn transpose() {
        let mut m = Mat2f64::new_column_major(Vec2f64::new(1.0, 2.0), Vec2f64::new(3.0, 4.0));
        m.transpose();
        assert_eq!(m.cols[0].x, 1.0);
        assert_eq!(m.cols[0].y, 3.0);
        assert_eq!(m.cols[1].x, 2.0);
        assert_eq!(m.cols[1].y, 4.0);
    }

    #[test]
    fn addition() {
        let mut one = Mat2f64::new_column_major(Vec2f64::new(1.0, 1.0), Vec2f64::new(1.0, 1.0));
        let two = one + one;
        assert_eq!(two.cols[0].x, 2.0);
        assert_eq!(two.cols[0].y, 2.0);
        assert_eq!(two.cols[1].x, 2.0);
        assert_eq!(two.cols[1].y, 2.0);

        one += one;
        assert_eq!(one.cols[0].x, 2.0);
        assert_eq!(one.cols[0].y, 2.0);
        assert_eq!(one.cols[1].x, 2.0);
        assert_eq!(one.cols[1].y, 2.0);
    }

    #[test]
    fn subtraction() {
        let one = Mat2f64::new_column_major(Vec2f64::new(1.0, 1.0), Vec2f64::new(1.0, 1.0));
        let mut zero = one - one;
        assert_eq!(zero.cols[0].x, 0.0);
        assert_eq!(zero.cols[0].y, 0.0);
        assert_eq!(zero.cols[1].x, 0.0);
        assert_eq!(zero.cols[1].y, 0.0);

        zero -= one;
        assert_eq!(zero.cols[0].x, -1.0);
        assert_eq!(zero.cols[0].y, -1.0);
        assert_eq!(zero.cols[1].x, -1.0);
        assert_eq!(zero.cols[1].y, -1.0);
    }

    #[test]
    fn multiplication() {
        let mut a = Mat2f64::new_column_major(Vec2f64::new(1.0, 3.0), Vec2f64::new(2.0, 4.0));
        let b = Mat2f64::new_column_major(Vec2f64::new(5.0, 7.0), Vec2f64::new(6.0, 8.0));
        let c = a * b;
        assert_eq!(c.cols[0].x, 19.0);
        assert_eq!(c.cols[0].y, 43.0);
        assert_eq!(c.cols[1].x, 22.0);
        assert_eq!(c.cols[1].y, 50.0);
        a *= b;
        assert_eq!(a.cols[0].x, 19.0);
        assert_eq!(a.cols[0].y, 43.0);
        assert_eq!(a.cols[1].x, 22.0);
        assert_eq!(a.cols[1].y, 50.0);
    }

    #[test]
    fn scalar_multiplication() {
        let one = Mat2f64::new_column_major(Vec2f64::new(1.0, 1.0), Vec2f64::new(1.0, 1.0));
        let mut four: Mat2f64 = 2.0 * one * 2.0;
        assert_eq!(four.cols[0].x, 4.0);
        assert_eq!(four.cols[0].y, 4.0);
        assert_eq!(four.cols[1].x, 4.0);
        assert_eq!(four.cols[1].y, 4.0);

        four *= 2.0;
        assert_eq!(four.cols[0].x, 8.0);
        assert_eq!(four.cols[0].y, 8.0);
        assert_eq!(four.cols[1].x, 8.0);
        assert_eq!(four.cols[1].y, 8.0);
    }

    #[test]
    fn scalar_division() {
        let one = Mat2f64::new_column_major(Vec2f64::new(1.0, 1.0), Vec2f64::new(1.0, 1.0));
        let mut half: Mat2f64 = one / 2.0;
        assert_eq!(half.cols[0].x, 0.5);
        assert_eq!(half.cols[0].y, 0.5);
        assert_eq!(half.cols[1].x, 0.5);
        assert_eq!(half.cols[1].y, 0.5);

        half /= 2.0;
        assert_eq!(half.cols[0].x, 0.25);
        assert_eq!(half.cols[0].y, 0.25);
        assert_eq!(half.cols[1].x, 0.25);
        assert_eq!(half.cols[1].y, 0.25);
    }

    #[test]
    fn multiply_vector() {
        let a = Mat2f64::new_column_major(Vec2f64::new(1.0, 3.0), Vec2f64::new(2.0, 4.0));
        let vec = Vec2f64::new(10.0, 5.0);
        let c = a * vec;
        assert_eq!(c.x, 20.0);
        assert_eq!(c.y, 50.0);
    }
}
