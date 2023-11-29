use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::vec::vec2f64::Vec2f64;

/// A 2x2 floating point matrix.
/// Indexing follows row major order, like in most mathematical texts.
#[derive(Debug, Clone, Copy)]
pub struct Mat2f64 {
    pub rows: [[f64; 2]; 2],
}

impl Mat2f64 {
    /// Creates a new matrix with user defined elements.
    /// The user defined elements are in row major order.
    pub fn new(rows: [[f64; 2]; 2]) -> Self {
        Self { rows }
    }

    /// Creates a new matrix with user defined elements.
    /// The user defined elements are in column major order.
    pub fn new_from_cols(cols: [[f64; 2]; 2]) -> Self {
        Self::new(cols).transposed()
    }

    /// Creates a new matrix with all elements equal to 0.0.
    pub fn zero() -> Self {
        Self {
            rows: [[0.0, 0.0], [0.0, 0.0]],
        }
    }

    /// Creates a new identity matrix.
    pub fn identity() -> Self {
        Self {
            rows: [[1.0, 0.0], [0.0, 1.0]],
        }
    }

    /// Returns self but transposed
    /// (Rows are now columns and columns are now rows).
    pub fn transposed(&self) -> Self {
        Self {
            rows: [[self[0][0], self[1][0]], [self[0][1], self[1][1]]],
        }
    }

    /// Transposes self
    /// (Rows are now columns and columns are now rows).
    pub fn transpose(&mut self) {
        *self = self.transposed()
    }

    /// Returns the matrix rows as arrays in row major order.
    pub fn as_row_major(&self) -> [[f64; 2]; 2] {
        self.rows
    }

    /// Returns the matrix columns as arrays in column major order.
    pub fn as_col_major(&self) -> [[f64; 2]; 2] {
        self.transposed().rows
    }
}

impl Index<usize> for Mat2f64 {
    type Output = [f64; 2];
    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index]
    }
}

impl IndexMut<usize> for Mat2f64 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

impl Add<Mat2f64> for Mat2f64 {
    type Output = Mat2f64;
    fn add(mut self, rhs: Mat2f64) -> Self::Output {
        self[0][0] += rhs[0][0];
        self[0][1] += rhs[0][1];
        self[1][0] += rhs[1][0];
        self[1][1] += rhs[1][1];
        self
    }
}

impl AddAssign<Mat2f64> for Mat2f64 {
    fn add_assign(&mut self, rhs: Mat2f64) {
        *self = *self + rhs;
    }
}

impl Sub<Mat2f64> for Mat2f64 {
    type Output = Mat2f64;
    fn sub(mut self, rhs: Mat2f64) -> Self::Output {
        self[0][0] -= rhs[0][0];
        self[0][1] -= rhs[0][1];
        self[1][0] -= rhs[1][0];
        self[1][1] -= rhs[1][1];
        self
    }
}

impl SubAssign<Mat2f64> for Mat2f64 {
    fn sub_assign(&mut self, rhs: Mat2f64) {
        *self = *self - rhs;
    }
}

impl Mul<f64> for Mat2f64 {
    type Output = Mat2f64;
    fn mul(mut self, scalar: f64) -> Self::Output {
        self[0][0] *= scalar;
        self[0][1] *= scalar;
        self[1][0] *= scalar;
        self[1][1] *= scalar;
        self
    }
}

impl Mul<Mat2f64> for f64 {
    type Output = Mat2f64;
    fn mul(self, m: Mat2f64) -> Self::Output {
        m * self
    }
}

impl MulAssign<f64> for Mat2f64 {
    fn mul_assign(&mut self, scalar: f64) {
        *self = *self * scalar;
    }
}

impl Div<f64> for Mat2f64 {
    type Output = Mat2f64;
    fn div(mut self, scalar: f64) -> Self::Output {
        self[0][0] /= scalar;
        self[0][1] /= scalar;
        self[1][0] /= scalar;
        self[1][1] /= scalar;
        self
    }
}

impl DivAssign<f64> for Mat2f64 {
    fn div_assign(&mut self, scalar: f64) {
        *self = *self / scalar;
    }
}

impl Mul<Mat2f64> for Mat2f64 {
    type Output = Mat2f64;
    fn mul(self, rhs: Mat2f64) -> Self::Output {
        Self::new([
            [
                self[0][0] * rhs[0][0] + self[0][1] * rhs[1][0],
                self[0][0] * rhs[0][1] + self[0][1] * rhs[1][1],
            ],
            [
                self[1][0] * rhs[0][0] + self[1][1] * rhs[1][0],
                self[1][0] * rhs[0][1] + self[1][1] * rhs[1][1],
            ],
        ])
    }
}

impl MulAssign<Mat2f64> for Mat2f64 {
    fn mul_assign(&mut self, rhs: Mat2f64) {
        *self = *self * rhs;
    }
}

impl Mul<Vec2f64> for Mat2f64 {
    type Output = Vec2f64;
    fn mul(self, v: Vec2f64) -> Self::Output {
        Vec2f64::new([
            self[0][0] * v[0] + self[0][1] * v[1],
            self[1][0] * v[0] + self[1][1] * v[1],
        ])
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::vec2f64::Vec2f64;

    use super::Mat2f64;

    #[test]
    fn matrix_creation() {
        let zero = Mat2f64::zero();
        let id = Mat2f64::identity();
        let m = Mat2f64::new([[1.0, 2.0], [3.0, 4.0]]);
        let t = Mat2f64::new_from_cols([[1.0, 3.0], [2.0, 4.0]]);

        assert_eq!(zero[0][0], 0.0);
        assert_eq!(zero[0][1], 0.0);
        assert_eq!(zero[1][0], 0.0);
        assert_eq!(zero[1][1], 0.0);

        assert_eq!(id[0][0], 1.0);
        assert_eq!(id[0][1], 0.0);
        assert_eq!(id[1][0], 0.0);
        assert_eq!(id[1][1], 1.0);

        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[0][1], 2.0);
        assert_eq!(m[1][0], 3.0);
        assert_eq!(m[1][1], 4.0);

        assert_eq!(t[0][0], 1.0);
        assert_eq!(t[0][1], 2.0);
        assert_eq!(t[1][0], 3.0);
        assert_eq!(t[1][1], 4.0);
    }

    #[test]
    fn transpose() {
        let mut m = Mat2f64::new([[1.0, 2.0], [3.0, 4.0]]);
        let t = m.transposed();
        m.transpose();

        assert_eq!(t[0][0], 1.0);
        assert_eq!(t[0][1], 3.0);
        assert_eq!(t[1][0], 2.0);
        assert_eq!(t[1][1], 4.0);

        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[0][1], 3.0);
        assert_eq!(m[1][0], 2.0);
        assert_eq!(m[1][1], 4.0);
    }

    #[test]
    fn as_row_and_col_major() {
        let m = Mat2f64::new([[1.0, 2.0], [3.0, 4.0]]);
        let cols = m.as_col_major();
        let rows = m.as_row_major();

        assert_eq!(cols[0][0], 1.0);
        assert_eq!(cols[0][1], 3.0);
        assert_eq!(cols[1][0], 2.0);
        assert_eq!(cols[1][1], 4.0);

        assert_eq!(rows[0][0], 1.0);
        assert_eq!(rows[0][1], 2.0);
        assert_eq!(rows[1][0], 3.0);
        assert_eq!(rows[1][1], 4.0);
    }

    #[test]
    fn matrix_addition() {
        let m = Mat2f64::new([[1.0, 2.0], [3.0, 4.0]]);
        let mut n = Mat2f64::new([[5.0, 6.0], [7.0, 8.0]]);
        let r = n + m;
        n += m;
        assert_eq!(r[0][0], 6.0);
        assert_eq!(r[0][1], 8.0);
        assert_eq!(r[1][0], 10.0);
        assert_eq!(r[1][1], 12.0);
        assert_eq!(n[0][0], 6.0);
        assert_eq!(n[0][1], 8.0);
        assert_eq!(n[1][0], 10.0);
        assert_eq!(n[1][1], 12.0);
    }

    #[test]
    fn matrix_subtraction() {
        let m = Mat2f64::new([[1.0, 2.0], [3.0, 4.0]]);
        let mut n = Mat2f64::new([[5.0, 6.0], [7.0, 8.0]]);
        let r = m - n;
        n -= m;
        assert_eq!(r[0][0], -4.0);
        assert_eq!(r[0][1], -4.0);
        assert_eq!(r[1][0], -4.0);
        assert_eq!(r[1][1], -4.0);
        assert_eq!(n[0][0], 4.0);
        assert_eq!(n[0][1], 4.0);
        assert_eq!(n[1][0], 4.0);
        assert_eq!(n[1][1], 4.0);
    }

    #[test]
    fn scalar_multiplication() {
        let mut m = Mat2f64::new([[1.0, 2.0], [3.0, 4.0]]);
        let n = m * 3.0;
        assert_eq!(n[0][0], 3.0);
        assert_eq!(n[0][1], 6.0);
        assert_eq!(n[1][0], 9.0);
        assert_eq!(n[1][1], 12.0);

        let n = 3.0 * m;
        assert_eq!(n[0][0], 3.0);
        assert_eq!(n[0][1], 6.0);
        assert_eq!(n[1][0], 9.0);
        assert_eq!(n[1][1], 12.0);

        m *= 3.0;
        assert_eq!(m[0][0], 3.0);
        assert_eq!(m[0][1], 6.0);
        assert_eq!(m[1][0], 9.0);
        assert_eq!(m[1][1], 12.0);
    }

    #[test]
    fn scalar_division() {
        let mut m = Mat2f64::new([[1.0, 2.0], [3.0, 4.0]]);
        let n = m / 3.0;
        assert_eq!(n[0][0], 1.0 / 3.0);
        assert_eq!(n[0][1], 2.0 / 3.0);
        assert_eq!(n[1][0], 3.0 / 3.0);
        assert_eq!(n[1][1], 4.0 / 3.0);

        m /= 3.0;
        assert_eq!(m[0][0], 1.0 / 3.0);
        assert_eq!(m[0][1], 2.0 / 3.0);
        assert_eq!(m[1][0], 3.0 / 3.0);
        assert_eq!(m[1][1], 4.0 / 3.0);
    }

    #[test]
    fn matrix_multiplication() {
        let mut m = Mat2f64::new([[1.0, 2.0], [3.0, 4.0]]);
        let n = Mat2f64::new([[5.0, 6.0], [7.0, 8.0]]);
        let r = m * n;
        m *= n;

        assert_eq!(r[0][0], 19.0);
        assert_eq!(r[0][1], 22.0);
        assert_eq!(r[1][0], 43.0);
        assert_eq!(r[1][1], 50.0);

        assert_eq!(m[0][0], 19.0);
        assert_eq!(m[0][1], 22.0);
        assert_eq!(m[1][0], 43.0);
        assert_eq!(m[1][1], 50.0);
    }

    #[test]
    fn vector_multiplication() {
        let m = Mat2f64::new([[1.0, 2.0], [3.0, 4.0]]);
        let v = Vec2f64::new([2.0, 3.0]);
        let w = m * v;

        assert_eq!(w[0], 8.0);
        assert_eq!(w[1], 18.0);
    }
}
