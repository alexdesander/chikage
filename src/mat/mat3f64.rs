use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::vec::vec3f64::Vec3f64;

/// A 3x3 floating point matrix.
/// Indexing follows row major order, like in most mathematical texts.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
pub struct Mat3f64 {
    pub rows: [[f64; 3]; 3],
}

impl Mat3f64 {
    /// Creates a new matrix with user defined elements.
    /// The user defined elements are in row major order.
    pub fn new(rows: [[f64; 3]; 3]) -> Self {
        Self { rows }
    }

    /// Creates a new matrix with user defined elements.
    /// The user defined elements are in column major order.
    pub fn new_from_cols(cols: [[f64; 3]; 3]) -> Self {
        Self::new(cols).transposed()
    }

    /// Creates a new matrix with all elements equal to 0.0.
    pub fn zero() -> Self {
        Self {
            rows: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        }
    }

    /// Creates a new identity matrix.
    pub fn identity() -> Self {
        Self {
            rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Returns self but transposed
    /// (Rows are now columns and columns are now rows).
    pub fn transposed(&self) -> Self {
        Self {
            rows: [
                [self[0][0], self[1][0], self[2][0]],
                [self[0][1], self[1][1], self[2][1]],
                [self[0][2], self[1][2], self[2][2]],
            ],
        }
    }

    /// Transposes self
    /// (Rows are now columns and columns are now rows).
    pub fn transpose(&mut self) {
        *self = self.transposed()
    }

    /// Returns the matrix rows as arrays in row major order.
    pub fn as_row_major(&self) -> [[f64; 3]; 3] {
        self.rows
    }

    /// Returns the matrix columns as arrays in column major order.
    pub fn as_col_major(&self) -> [[f64; 3]; 3] {
        self.transposed().rows
    }
}

impl Index<usize> for Mat3f64 {
    type Output = [f64; 3];
    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index]
    }
}

impl IndexMut<usize> for Mat3f64 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

impl Add<Mat3f64> for Mat3f64 {
    type Output = Mat3f64;
    fn add(mut self, rhs: Mat3f64) -> Self::Output {
        self[0][0] += rhs[0][0];
        self[0][1] += rhs[0][1];
        self[0][2] += rhs[0][2];
        self[1][0] += rhs[1][0];
        self[1][1] += rhs[1][1];
        self[1][2] += rhs[1][2];
        self[2][0] += rhs[2][0];
        self[2][1] += rhs[2][1];
        self[2][2] += rhs[2][2];
        self
    }
}

impl AddAssign<Mat3f64> for Mat3f64 {
    fn add_assign(&mut self, rhs: Mat3f64) {
        *self = *self + rhs;
    }
}

impl Sub<Mat3f64> for Mat3f64 {
    type Output = Mat3f64;
    fn sub(mut self, rhs: Mat3f64) -> Self::Output {
        self[0][0] -= rhs[0][0];
        self[0][1] -= rhs[0][1];
        self[0][2] -= rhs[0][2];
        self[1][0] -= rhs[1][0];
        self[1][1] -= rhs[1][1];
        self[1][2] -= rhs[1][2];
        self[2][0] -= rhs[2][0];
        self[2][1] -= rhs[2][1];
        self[2][2] -= rhs[2][2];
        self
    }
}

impl SubAssign<Mat3f64> for Mat3f64 {
    fn sub_assign(&mut self, rhs: Mat3f64) {
        *self = *self - rhs;
    }
}

impl Mul<f64> for Mat3f64 {
    type Output = Mat3f64;
    fn mul(mut self, scalar: f64) -> Self::Output {
        self[0][0] *= scalar;
        self[0][1] *= scalar;
        self[0][2] *= scalar;
        self[1][0] *= scalar;
        self[1][1] *= scalar;
        self[1][2] *= scalar;
        self[2][0] *= scalar;
        self[2][1] *= scalar;
        self[2][2] *= scalar;
        self
    }
}

impl Mul<Mat3f64> for f64 {
    type Output = Mat3f64;
    fn mul(self, m: Mat3f64) -> Self::Output {
        m * self
    }
}

impl MulAssign<f64> for Mat3f64 {
    fn mul_assign(&mut self, scalar: f64) {
        *self = *self * scalar;
    }
}

impl Div<f64> for Mat3f64 {
    type Output = Mat3f64;
    fn div(mut self, scalar: f64) -> Self::Output {
        self[0][0] /= scalar;
        self[0][1] /= scalar;
        self[0][2] /= scalar;
        self[1][0] /= scalar;
        self[1][1] /= scalar;
        self[1][2] /= scalar;
        self[2][0] /= scalar;
        self[2][1] /= scalar;
        self[2][2] /= scalar;
        self
    }
}

impl DivAssign<f64> for Mat3f64 {
    fn div_assign(&mut self, scalar: f64) {
        *self = *self / scalar;
    }
}

impl Mul<Mat3f64> for Mat3f64 {
    type Output = Mat3f64;
    fn mul(self, b: Mat3f64) -> Self::Output {
        let a = self;
        Self::new([
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
                a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
                a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
            ],
            [
                a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
                a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
                a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
            ],
        ])
    }
}

impl MulAssign<Mat3f64> for Mat3f64 {
    fn mul_assign(&mut self, rhs: Mat3f64) {
        *self = *self * rhs;
    }
}

impl Mul<Vec3f64> for Mat3f64 {
    type Output = Vec3f64;
    fn mul(self, v: Vec3f64) -> Self::Output {
        Vec3f64::new([
            self[0][0] * v[0] + self[0][1] * v[1] + self[0][2] * v[2],
            self[1][0] * v[0] + self[1][1] * v[1] + self[1][2] * v[2],
            self[2][0] * v[0] + self[2][1] * v[1] + self[2][2] * v[2],
        ])
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::vec3f64::Vec3f64;

    use super::Mat3f64;

    #[test]
    fn matrix_creation() {
        let zero = Mat3f64::zero();
        let id = Mat3f64::identity();
        let m = Mat3f64::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let t = Mat3f64::new_from_cols([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]);

        assert_eq!(zero[0][0], 0.0);
        assert_eq!(zero[0][1], 0.0);
        assert_eq!(zero[0][2], 0.0);
        assert_eq!(zero[1][0], 0.0);
        assert_eq!(zero[1][1], 0.0);
        assert_eq!(zero[1][2], 0.0);
        assert_eq!(zero[2][0], 0.0);
        assert_eq!(zero[2][1], 0.0);
        assert_eq!(zero[2][2], 0.0);

        assert_eq!(id[0][0], 1.0);
        assert_eq!(id[0][1], 0.0);
        assert_eq!(id[0][2], 0.0);
        assert_eq!(id[1][0], 0.0);
        assert_eq!(id[1][1], 1.0);
        assert_eq!(id[1][2], 0.0);
        assert_eq!(id[2][0], 0.0);
        assert_eq!(id[2][1], 0.0);
        assert_eq!(id[2][2], 1.0);

        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[0][1], 2.0);
        assert_eq!(m[0][2], 3.0);
        assert_eq!(m[1][0], 4.0);
        assert_eq!(m[1][1], 5.0);
        assert_eq!(m[1][2], 6.0);
        assert_eq!(m[2][0], 7.0);
        assert_eq!(m[2][1], 8.0);
        assert_eq!(m[2][2], 9.0);

        assert_eq!(t[0][0], 1.0);
        assert_eq!(t[0][1], 2.0);
        assert_eq!(t[0][2], 3.0);
        assert_eq!(t[1][0], 4.0);
        assert_eq!(t[1][1], 5.0);
        assert_eq!(t[1][2], 6.0);
        assert_eq!(t[2][0], 7.0);
        assert_eq!(t[2][1], 8.0);
        assert_eq!(t[2][2], 9.0);
    }

    #[test]
    fn transpose() {
        let mut m = Mat3f64::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let t = m.transposed();
        m.transpose();

        assert_eq!(t[0][0], 1.0);
        assert_eq!(t[0][1], 4.0);
        assert_eq!(t[0][2], 7.0);
        assert_eq!(t[1][0], 2.0);
        assert_eq!(t[1][1], 5.0);
        assert_eq!(t[1][2], 8.0);
        assert_eq!(t[2][0], 3.0);
        assert_eq!(t[2][1], 6.0);
        assert_eq!(t[2][2], 9.0);

        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[0][1], 4.0);
        assert_eq!(m[0][2], 7.0);
        assert_eq!(m[1][0], 2.0);
        assert_eq!(m[1][1], 5.0);
        assert_eq!(m[1][2], 8.0);
        assert_eq!(m[2][0], 3.0);
        assert_eq!(m[2][1], 6.0);
        assert_eq!(m[2][2], 9.0);
    }

    #[test]
    fn as_row_and_col_major() {
        let m = Mat3f64::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let cols = m.as_col_major();
        let rows = m.as_row_major();

        assert_eq!(cols[0][0], 1.0);
        assert_eq!(cols[0][1], 4.0);
        assert_eq!(cols[0][2], 7.0);
        assert_eq!(cols[1][0], 2.0);
        assert_eq!(cols[1][1], 5.0);
        assert_eq!(cols[1][2], 8.0);
        assert_eq!(cols[2][0], 3.0);
        assert_eq!(cols[2][1], 6.0);
        assert_eq!(cols[2][2], 9.0);

        assert_eq!(rows[0][0], 1.0);
        assert_eq!(rows[0][1], 2.0);
        assert_eq!(rows[0][2], 3.0);
        assert_eq!(rows[1][0], 4.0);
        assert_eq!(rows[1][1], 5.0);
        assert_eq!(rows[1][2], 6.0);
        assert_eq!(rows[2][0], 7.0);
        assert_eq!(rows[2][1], 8.0);
        assert_eq!(rows[2][2], 9.0);
    }

    #[test]
    fn matrix_addition() {
        let m = Mat3f64::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let mut n = Mat3f64::new([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]);
        let r = n + m;
        n += m;

        assert_eq!(r[0][0], 11.0);
        assert_eq!(r[0][1], 13.0);
        assert_eq!(r[0][2], 15.0);
        assert_eq!(r[1][0], 17.0);
        assert_eq!(r[1][1], 19.0);
        assert_eq!(r[1][2], 21.0);
        assert_eq!(r[2][0], 23.0);
        assert_eq!(r[2][1], 25.0);
        assert_eq!(r[2][2], 27.0);

        assert_eq!(n[0][0], 11.0);
        assert_eq!(n[0][1], 13.0);
        assert_eq!(n[0][2], 15.0);
        assert_eq!(n[1][0], 17.0);
        assert_eq!(n[1][1], 19.0);
        assert_eq!(n[1][2], 21.0);
        assert_eq!(n[2][0], 23.0);
        assert_eq!(n[2][1], 25.0);
        assert_eq!(n[2][2], 27.0);
    }

    #[test]
    fn matrix_subtraction() {
        let m = Mat3f64::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let mut n = Mat3f64::new([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]);
        let r = m - n;
        n -= m;

        assert_eq!(r[0][0], -9.0);
        assert_eq!(r[0][1], -9.0);
        assert_eq!(r[0][2], -9.0);
        assert_eq!(r[1][0], -9.0);
        assert_eq!(r[1][1], -9.0);
        assert_eq!(r[1][2], -9.0);
        assert_eq!(r[2][0], -9.0);
        assert_eq!(r[2][1], -9.0);
        assert_eq!(r[2][2], -9.0);

        assert_eq!(n[0][0], 9.0);
        assert_eq!(n[0][1], 9.0);
        assert_eq!(n[0][2], 9.0);
        assert_eq!(n[1][0], 9.0);
        assert_eq!(n[1][1], 9.0);
        assert_eq!(n[1][2], 9.0);
        assert_eq!(n[2][0], 9.0);
        assert_eq!(n[2][1], 9.0);
        assert_eq!(n[2][2], 9.0);
    }

    #[test]
    fn scalar_multiplication() {
        let mut m = Mat3f64::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let n = m * 3.0;
        assert_eq!(n[0][0], 3.0);
        assert_eq!(n[0][1], 6.0);
        assert_eq!(n[0][2], 9.0);
        assert_eq!(n[1][0], 12.0);
        assert_eq!(n[1][1], 15.0);
        assert_eq!(n[1][2], 18.0);
        assert_eq!(n[2][0], 21.0);
        assert_eq!(n[2][1], 24.0);
        assert_eq!(n[2][2], 27.0);

        let n = 3.0 * m;
        assert_eq!(n[0][0], 3.0);
        assert_eq!(n[0][1], 6.0);
        assert_eq!(n[0][2], 9.0);
        assert_eq!(n[1][0], 12.0);
        assert_eq!(n[1][1], 15.0);
        assert_eq!(n[1][2], 18.0);
        assert_eq!(n[2][0], 21.0);
        assert_eq!(n[2][1], 24.0);
        assert_eq!(n[2][2], 27.0);

        m *= 3.0;
        assert_eq!(m[0][0], 3.0);
        assert_eq!(m[0][1], 6.0);
        assert_eq!(m[0][2], 9.0);
        assert_eq!(m[1][0], 12.0);
        assert_eq!(m[1][1], 15.0);
        assert_eq!(m[1][2], 18.0);
        assert_eq!(m[2][0], 21.0);
        assert_eq!(m[2][1], 24.0);
        assert_eq!(m[2][2], 27.0);
    }

    #[test]
    fn scalar_division() {
        let mut m = Mat3f64::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let n = m / 3.0;
        assert_eq!(n[0][0], 1.0 / 3.0);
        assert_eq!(n[0][1], 2.0 / 3.0);
        assert_eq!(n[0][2], 3.0 / 3.0);
        assert_eq!(n[1][0], 4.0 / 3.0);
        assert_eq!(n[1][1], 5.0 / 3.0);
        assert_eq!(n[1][2], 6.0 / 3.0);
        assert_eq!(n[2][0], 7.0 / 3.0);
        assert_eq!(n[2][1], 8.0 / 3.0);
        assert_eq!(n[2][2], 9.0 / 3.0);

        m /= 3.0;
        assert_eq!(m[0][0], 1.0 / 3.0);
        assert_eq!(m[0][1], 2.0 / 3.0);
        assert_eq!(m[0][2], 3.0 / 3.0);
        assert_eq!(m[1][0], 4.0 / 3.0);
        assert_eq!(m[1][1], 5.0 / 3.0);
        assert_eq!(m[1][2], 6.0 / 3.0);
        assert_eq!(m[2][0], 7.0 / 3.0);
        assert_eq!(m[2][1], 8.0 / 3.0);
        assert_eq!(m[2][2], 9.0 / 3.0);
    }

    #[test]
    fn matrix_multiplication() {
        let mut m = Mat3f64::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let n = Mat3f64::new([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]);
        let r = m * n;
        m *= n;

        assert_eq!(r[0][0], 84.0);
        assert_eq!(r[0][1], 90.0);
        assert_eq!(r[0][2], 96.0);
        assert_eq!(r[1][0], 201.0);
        assert_eq!(r[1][1], 216.0);
        assert_eq!(r[1][2], 231.0);
        assert_eq!(r[2][0], 318.0);
        assert_eq!(r[2][1], 342.0);
        assert_eq!(r[2][2], 366.0);

        assert_eq!(m[0][0], 84.0);
        assert_eq!(m[0][1], 90.0);
        assert_eq!(m[0][2], 96.0);
        assert_eq!(m[1][0], 201.0);
        assert_eq!(m[1][1], 216.0);
        assert_eq!(m[1][2], 231.0);
        assert_eq!(m[2][0], 318.0);
        assert_eq!(m[2][1], 342.0);
        assert_eq!(m[2][2], 366.0);
    }

    #[test]
    fn vector_multiplication() {
        let m = Mat3f64::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let v = Vec3f64::new([2.0, 3.0, 4.0]);
        let w = m * v;

        assert_eq!(w[0], 20.0);
        assert_eq!(w[1], 47.0);
        assert_eq!(w[2], 74.0);
    }
}
