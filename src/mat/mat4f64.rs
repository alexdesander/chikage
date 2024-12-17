use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::vec::vec4f64::Vec4f64;

/// A 4x4 floating point matrix.
/// Indexing follows row major order, like in most mathematical texts.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bitcode", derive(bitcode::Encode, bitcode::Decode))]
pub struct Mat4f64 {
    pub rows: [[f64; 4]; 4],
}

impl Mat4f64 {
    /// Creates a new matrix with user defined elements.
    /// The user defined elements are in row major order.
    pub fn new(rows: [[f64; 4]; 4]) -> Self {
        Self { rows }
    }

    /// Creates a new matrix with user defined elements.
    /// The user defined elements are in column major order.
    pub fn new_from_cols(cols: [[f64; 4]; 4]) -> Self {
        Self::new(cols).transposed()
    }

    /// Creates a new matrix with all elements equal to 0.0.
    pub fn zero() -> Self {
        Self {
            rows: [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        }
    }

    /// Creates a new identity matrix.
    pub fn identity() -> Self {
        Self {
            rows: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Returns self but transposed
    /// (Rows are now columns and columns are now rows).
    pub fn transposed(&self) -> Self {
        Self {
            rows: [
                [self[0][0], self[1][0], self[2][0], self[3][0]],
                [self[0][1], self[1][1], self[2][1], self[3][1]],
                [self[0][2], self[1][2], self[2][2], self[3][2]],
                [self[0][3], self[1][3], self[2][3], self[3][3]],
            ],
        }
    }

    /// Transposes self
    /// (Rows are now columns and columns are now rows).
    pub fn transpose(&mut self) {
        *self = self.transposed()
    }

    /// Returns the matrix rows as arrays in row major order.
    pub fn as_row_major(&self) -> [[f64; 4]; 4] {
        self.rows
    }

    /// Returns the matrix columns as arrays in column major order.
    pub fn as_col_major(&self) -> [[f64; 4]; 4] {
        self.transposed().rows
    }
}

impl Index<usize> for Mat4f64 {
    type Output = [f64; 4];
    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index]
    }
}

impl IndexMut<usize> for Mat4f64 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

impl Add<Mat4f64> for Mat4f64 {
    type Output = Mat4f64;
    fn add(mut self, rhs: Mat4f64) -> Self::Output {
        self[0][0] += rhs[0][0];
        self[0][1] += rhs[0][1];
        self[0][2] += rhs[0][2];
        self[0][3] += rhs[0][3];
        self[1][0] += rhs[1][0];
        self[1][1] += rhs[1][1];
        self[1][2] += rhs[1][2];
        self[1][3] += rhs[1][3];
        self[2][0] += rhs[2][0];
        self[2][1] += rhs[2][1];
        self[2][2] += rhs[2][2];
        self[2][3] += rhs[2][3];
        self[3][0] += rhs[3][0];
        self[3][1] += rhs[3][1];
        self[3][2] += rhs[3][2];
        self[3][3] += rhs[3][3];
        self
    }
}

impl AddAssign<Mat4f64> for Mat4f64 {
    fn add_assign(&mut self, rhs: Mat4f64) {
        *self = *self + rhs;
    }
}

impl Sub<Mat4f64> for Mat4f64 {
    type Output = Mat4f64;
    fn sub(mut self, rhs: Mat4f64) -> Self::Output {
        self[0][0] -= rhs[0][0];
        self[0][1] -= rhs[0][1];
        self[0][2] -= rhs[0][2];
        self[0][3] -= rhs[0][3];
        self[1][0] -= rhs[1][0];
        self[1][1] -= rhs[1][1];
        self[1][2] -= rhs[1][2];
        self[1][3] -= rhs[1][3];
        self[2][0] -= rhs[2][0];
        self[2][1] -= rhs[2][1];
        self[2][2] -= rhs[2][2];
        self[2][3] -= rhs[2][3];
        self[3][0] -= rhs[3][0];
        self[3][1] -= rhs[3][1];
        self[3][2] -= rhs[3][2];
        self[3][3] -= rhs[3][3];
        self
    }
}

impl SubAssign<Mat4f64> for Mat4f64 {
    fn sub_assign(&mut self, rhs: Mat4f64) {
        *self = *self - rhs;
    }
}

impl Mul<f64> for Mat4f64 {
    type Output = Mat4f64;
    fn mul(mut self, scalar: f64) -> Self::Output {
        self[0][0] *= scalar;
        self[0][1] *= scalar;
        self[0][2] *= scalar;
        self[0][3] *= scalar;
        self[1][0] *= scalar;
        self[1][1] *= scalar;
        self[1][2] *= scalar;
        self[1][3] *= scalar;
        self[2][0] *= scalar;
        self[2][1] *= scalar;
        self[2][2] *= scalar;
        self[2][3] *= scalar;
        self[3][0] *= scalar;
        self[3][1] *= scalar;
        self[3][2] *= scalar;
        self[3][3] *= scalar;
        self
    }
}

impl Mul<Mat4f64> for f64 {
    type Output = Mat4f64;
    fn mul(self, m: Mat4f64) -> Self::Output {
        m * self
    }
}

impl MulAssign<f64> for Mat4f64 {
    fn mul_assign(&mut self, scalar: f64) {
        *self = *self * scalar;
    }
}

impl Div<f64> for Mat4f64 {
    type Output = Mat4f64;
    fn div(mut self, scalar: f64) -> Self::Output {
        self[0][0] /= scalar;
        self[0][1] /= scalar;
        self[0][2] /= scalar;
        self[0][3] /= scalar;
        self[1][0] /= scalar;
        self[1][1] /= scalar;
        self[1][2] /= scalar;
        self[1][3] /= scalar;
        self[2][0] /= scalar;
        self[2][1] /= scalar;
        self[2][2] /= scalar;
        self[2][3] /= scalar;
        self[3][0] /= scalar;
        self[3][1] /= scalar;
        self[3][2] /= scalar;
        self[3][3] /= scalar;
        self
    }
}

impl DivAssign<f64> for Mat4f64 {
    fn div_assign(&mut self, scalar: f64) {
        *self = *self / scalar;
    }
}

impl Mul<Mat4f64> for Mat4f64 {
    type Output = Mat4f64;
    fn mul(self, b: Mat4f64) -> Self::Output {
        let a = self;
        Self::new([
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0] + a[0][3] * b[3][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1] + a[0][3] * b[3][1],
                a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2] + a[0][3] * b[3][2],
                a[0][0] * b[0][3] + a[0][1] * b[1][3] + a[0][2] * b[2][3] + a[0][3] * b[3][3],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0] + a[1][3] * b[3][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1] + a[1][3] * b[3][1],
                a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2] + a[1][3] * b[3][2],
                a[1][0] * b[0][3] + a[1][1] * b[1][3] + a[1][2] * b[2][3] + a[1][3] * b[3][3],
            ],
            [
                a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0] + a[2][3] * b[3][0],
                a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1] + a[2][3] * b[3][1],
                a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2] + a[2][3] * b[3][2],
                a[2][0] * b[0][3] + a[2][1] * b[1][3] + a[2][2] * b[2][3] + a[2][3] * b[3][3],
            ],
            [
                a[3][0] * b[0][0] + a[3][1] * b[1][0] + a[3][2] * b[2][0] + a[3][3] * b[3][0],
                a[3][0] * b[0][1] + a[3][1] * b[1][1] + a[3][2] * b[2][1] + a[3][3] * b[3][1],
                a[3][0] * b[0][2] + a[3][1] * b[1][2] + a[3][2] * b[2][2] + a[3][3] * b[3][2],
                a[3][0] * b[0][3] + a[3][1] * b[1][3] + a[3][2] * b[2][3] + a[3][3] * b[3][3],
            ],
        ])
    }
}

impl MulAssign<Mat4f64> for Mat4f64 {
    fn mul_assign(&mut self, rhs: Mat4f64) {
        *self = *self * rhs;
    }
}

impl Mul<Vec4f64> for Mat4f64 {
    type Output = Vec4f64;
    fn mul(self, v: Vec4f64) -> Self::Output {
        Vec4f64::new([
            self[0][0] * v[0] + self[0][1] * v[1] + self[0][2] * v[2] + self[0][3] * v[3],
            self[1][0] * v[0] + self[1][1] * v[1] + self[1][2] * v[2] + self[1][3] * v[3],
            self[2][0] * v[0] + self[2][1] * v[1] + self[2][2] * v[2] + self[2][3] * v[3],
            self[3][0] * v[0] + self[3][1] * v[1] + self[3][2] * v[2] + self[3][3] * v[3],
        ])
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::vec4f64::Vec4f64;

    use super::Mat4f64;

    #[test]
    fn matrix_creation() {
        let zero = Mat4f64::zero();
        let id = Mat4f64::identity();
        let m = Mat4f64::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let t = Mat4f64::new_from_cols([
            [1.0, 5.0, 9.0, 13.0],
            [2.0, 6.0, 10.0, 14.0],
            [3.0, 7.0, 11.0, 15.0],
            [4.0, 8.0, 12.0, 16.0],
        ]);

        assert_eq!(zero[0][0], 0.0);
        assert_eq!(zero[0][1], 0.0);
        assert_eq!(zero[0][2], 0.0);
        assert_eq!(zero[0][3], 0.0);
        assert_eq!(zero[1][0], 0.0);
        assert_eq!(zero[1][1], 0.0);
        assert_eq!(zero[1][2], 0.0);
        assert_eq!(zero[1][3], 0.0);
        assert_eq!(zero[2][0], 0.0);
        assert_eq!(zero[2][1], 0.0);
        assert_eq!(zero[2][2], 0.0);
        assert_eq!(zero[2][3], 0.0);
        assert_eq!(zero[3][0], 0.0);
        assert_eq!(zero[3][1], 0.0);
        assert_eq!(zero[3][2], 0.0);
        assert_eq!(zero[3][3], 0.0);

        assert_eq!(id[0][0], 1.0);
        assert_eq!(id[0][1], 0.0);
        assert_eq!(id[0][2], 0.0);
        assert_eq!(id[0][3], 0.0);
        assert_eq!(id[1][0], 0.0);
        assert_eq!(id[1][1], 1.0);
        assert_eq!(id[1][2], 0.0);
        assert_eq!(id[1][3], 0.0);
        assert_eq!(id[2][0], 0.0);
        assert_eq!(id[2][1], 0.0);
        assert_eq!(id[2][2], 1.0);
        assert_eq!(id[2][3], 0.0);
        assert_eq!(id[3][0], 0.0);
        assert_eq!(id[3][1], 0.0);
        assert_eq!(id[3][2], 0.0);
        assert_eq!(id[3][3], 1.0);

        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[0][1], 2.0);
        assert_eq!(m[0][2], 3.0);
        assert_eq!(m[0][3], 4.0);
        assert_eq!(m[1][0], 5.0);
        assert_eq!(m[1][1], 6.0);
        assert_eq!(m[1][2], 7.0);
        assert_eq!(m[1][3], 8.0);
        assert_eq!(m[2][0], 9.0);
        assert_eq!(m[2][1], 10.0);
        assert_eq!(m[2][2], 11.0);
        assert_eq!(m[2][3], 12.0);
        assert_eq!(m[3][0], 13.0);
        assert_eq!(m[3][1], 14.0);
        assert_eq!(m[3][2], 15.0);
        assert_eq!(m[3][3], 16.0);

        assert_eq!(t[0][0], 1.0);
        assert_eq!(t[0][1], 2.0);
        assert_eq!(t[0][2], 3.0);
        assert_eq!(t[0][3], 4.0);
        assert_eq!(t[1][0], 5.0);
        assert_eq!(t[1][1], 6.0);
        assert_eq!(t[1][2], 7.0);
        assert_eq!(t[1][3], 8.0);
        assert_eq!(t[2][0], 9.0);
        assert_eq!(t[2][1], 10.0);
        assert_eq!(t[2][2], 11.0);
        assert_eq!(t[2][3], 12.0);
        assert_eq!(t[3][0], 13.0);
        assert_eq!(t[3][1], 14.0);
        assert_eq!(t[3][2], 15.0);
        assert_eq!(t[3][3], 16.0);
    }

    #[test]
    fn transpose() {
        let mut m = Mat4f64::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let t = m.transposed();
        m.transpose();

        assert_eq!(t[0][0], 1.0);
        assert_eq!(t[0][1], 5.0);
        assert_eq!(t[0][2], 9.0);
        assert_eq!(t[0][3], 13.0);
        assert_eq!(t[1][0], 2.0);
        assert_eq!(t[1][1], 6.0);
        assert_eq!(t[1][2], 10.0);
        assert_eq!(t[1][3], 14.0);
        assert_eq!(t[2][0], 3.0);
        assert_eq!(t[2][1], 7.0);
        assert_eq!(t[2][2], 11.0);
        assert_eq!(t[2][3], 15.0);
        assert_eq!(t[3][0], 4.0);
        assert_eq!(t[3][1], 8.0);
        assert_eq!(t[3][2], 12.0);
        assert_eq!(t[3][3], 16.0);

        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[0][1], 5.0);
        assert_eq!(m[0][2], 9.0);
        assert_eq!(m[0][3], 13.0);
        assert_eq!(m[1][0], 2.0);
        assert_eq!(m[1][1], 6.0);
        assert_eq!(m[1][2], 10.0);
        assert_eq!(m[1][3], 14.0);
        assert_eq!(m[2][0], 3.0);
        assert_eq!(m[2][1], 7.0);
        assert_eq!(m[2][2], 11.0);
        assert_eq!(m[2][3], 15.0);
        assert_eq!(m[3][0], 4.0);
        assert_eq!(m[3][1], 8.0);
        assert_eq!(m[3][2], 12.0);
        assert_eq!(m[3][3], 16.0);
    }

    #[test]
    fn as_row_and_col_major() {
        let m = Mat4f64::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let cols = m.as_col_major();
        let rows = m.as_row_major();

        assert_eq!(cols[0][0], 1.0);
        assert_eq!(cols[0][1], 5.0);
        assert_eq!(cols[0][2], 9.0);
        assert_eq!(cols[0][3], 13.0);
        assert_eq!(cols[1][0], 2.0);
        assert_eq!(cols[1][1], 6.0);
        assert_eq!(cols[1][2], 10.0);
        assert_eq!(cols[1][3], 14.0);
        assert_eq!(cols[2][0], 3.0);
        assert_eq!(cols[2][1], 7.0);
        assert_eq!(cols[2][2], 11.0);
        assert_eq!(cols[2][3], 15.0);
        assert_eq!(cols[3][0], 4.0);
        assert_eq!(cols[3][1], 8.0);
        assert_eq!(cols[3][2], 12.0);
        assert_eq!(cols[3][3], 16.0);

        assert_eq!(rows[0][0], 1.0);
        assert_eq!(rows[0][1], 2.0);
        assert_eq!(rows[0][2], 3.0);
        assert_eq!(rows[0][3], 4.0);
        assert_eq!(rows[1][0], 5.0);
        assert_eq!(rows[1][1], 6.0);
        assert_eq!(rows[1][2], 7.0);
        assert_eq!(rows[1][3], 8.0);
        assert_eq!(rows[2][0], 9.0);
        assert_eq!(rows[2][1], 10.0);
        assert_eq!(rows[2][2], 11.0);
        assert_eq!(rows[2][3], 12.0);
        assert_eq!(rows[3][0], 13.0);
        assert_eq!(rows[3][1], 14.0);
        assert_eq!(rows[3][2], 15.0);
        assert_eq!(rows[3][3], 16.0);
    }

    #[test]
    fn matrix_addition() {
        let m = Mat4f64::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let mut n = Mat4f64::new([
            [17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0],
            [29.0, 30.0, 31.0, 32.0],
        ]);
        let r = n + m;
        n += m;

        assert_eq!(r[0][0], 18.0);
        assert_eq!(r[0][1], 20.0);
        assert_eq!(r[0][2], 22.0);
        assert_eq!(r[0][3], 24.0);
        assert_eq!(r[1][0], 26.0);
        assert_eq!(r[1][1], 28.0);
        assert_eq!(r[1][2], 30.0);
        assert_eq!(r[1][3], 32.0);
        assert_eq!(r[2][0], 34.0);
        assert_eq!(r[2][1], 36.0);
        assert_eq!(r[2][2], 38.0);
        assert_eq!(r[2][3], 40.0);
        assert_eq!(r[3][0], 42.0);
        assert_eq!(r[3][1], 44.0);
        assert_eq!(r[3][2], 46.0);
        assert_eq!(r[3][3], 48.0);

        assert_eq!(n[0][0], 18.0);
        assert_eq!(n[0][1], 20.0);
        assert_eq!(n[0][2], 22.0);
        assert_eq!(n[0][3], 24.0);
        assert_eq!(n[1][0], 26.0);
        assert_eq!(n[1][1], 28.0);
        assert_eq!(n[1][2], 30.0);
        assert_eq!(n[1][3], 32.0);
        assert_eq!(n[2][0], 34.0);
        assert_eq!(n[2][1], 36.0);
        assert_eq!(n[2][2], 38.0);
        assert_eq!(n[2][3], 40.0);
        assert_eq!(n[3][0], 42.0);
        assert_eq!(n[3][1], 44.0);
        assert_eq!(n[3][2], 46.0);
        assert_eq!(n[3][3], 48.0);
    }

    #[test]
    fn matrix_subtraction() {
        let m = Mat4f64::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let mut n = Mat4f64::new([
            [17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0],
            [29.0, 30.0, 31.0, 32.0],
        ]);
        let r = m - n;
        n -= m;

        assert_eq!(r[0][0], -16.0);
        assert_eq!(r[0][1], -16.0);
        assert_eq!(r[0][2], -16.0);
        assert_eq!(r[0][3], -16.0);
        assert_eq!(r[1][0], -16.0);
        assert_eq!(r[1][1], -16.0);
        assert_eq!(r[1][2], -16.0);
        assert_eq!(r[1][3], -16.0);
        assert_eq!(r[2][0], -16.0);
        assert_eq!(r[2][1], -16.0);
        assert_eq!(r[2][2], -16.0);
        assert_eq!(r[2][3], -16.0);
        assert_eq!(r[3][0], -16.0);
        assert_eq!(r[3][1], -16.0);
        assert_eq!(r[3][2], -16.0);
        assert_eq!(r[3][3], -16.0);

        assert_eq!(n[0][0], 16.0);
        assert_eq!(n[0][1], 16.0);
        assert_eq!(n[0][2], 16.0);
        assert_eq!(n[0][3], 16.0);
        assert_eq!(n[1][0], 16.0);
        assert_eq!(n[1][1], 16.0);
        assert_eq!(n[1][2], 16.0);
        assert_eq!(n[1][3], 16.0);
        assert_eq!(n[2][0], 16.0);
        assert_eq!(n[2][1], 16.0);
        assert_eq!(n[2][2], 16.0);
        assert_eq!(n[2][3], 16.0);
        assert_eq!(n[3][0], 16.0);
        assert_eq!(n[3][1], 16.0);
        assert_eq!(n[3][2], 16.0);
        assert_eq!(n[3][3], 16.0);
    }

    #[test]
    fn scalar_multiplication() {
        let mut m = Mat4f64::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let n = m * 3.0;
        assert_eq!(n[0][0], 3.0);
        assert_eq!(n[0][1], 6.0);
        assert_eq!(n[0][2], 9.0);
        assert_eq!(n[0][3], 12.0);
        assert_eq!(n[1][0], 15.0);
        assert_eq!(n[1][1], 18.0);
        assert_eq!(n[1][2], 21.0);
        assert_eq!(n[1][3], 24.0);
        assert_eq!(n[2][0], 27.0);
        assert_eq!(n[2][1], 30.0);
        assert_eq!(n[2][2], 33.0);
        assert_eq!(n[2][3], 36.0);
        assert_eq!(n[3][0], 39.0);
        assert_eq!(n[3][1], 42.0);
        assert_eq!(n[3][2], 45.0);
        assert_eq!(n[3][3], 48.0);

        let n = 3.0 * m;
        assert_eq!(n[0][0], 3.0);
        assert_eq!(n[0][1], 6.0);
        assert_eq!(n[0][2], 9.0);
        assert_eq!(n[0][3], 12.0);
        assert_eq!(n[1][0], 15.0);
        assert_eq!(n[1][1], 18.0);
        assert_eq!(n[1][2], 21.0);
        assert_eq!(n[1][3], 24.0);
        assert_eq!(n[2][0], 27.0);
        assert_eq!(n[2][1], 30.0);
        assert_eq!(n[2][2], 33.0);
        assert_eq!(n[2][3], 36.0);
        assert_eq!(n[3][0], 39.0);
        assert_eq!(n[3][1], 42.0);
        assert_eq!(n[3][2], 45.0);
        assert_eq!(n[3][3], 48.0);

        m *= 3.0;
        assert_eq!(m[0][0], 3.0);
        assert_eq!(m[0][1], 6.0);
        assert_eq!(m[0][2], 9.0);
        assert_eq!(m[0][3], 12.0);
        assert_eq!(m[1][0], 15.0);
        assert_eq!(m[1][1], 18.0);
        assert_eq!(m[1][2], 21.0);
        assert_eq!(m[1][3], 24.0);
        assert_eq!(m[2][0], 27.0);
        assert_eq!(m[2][1], 30.0);
        assert_eq!(m[2][2], 33.0);
        assert_eq!(m[2][3], 36.0);
        assert_eq!(m[3][0], 39.0);
        assert_eq!(m[3][1], 42.0);
        assert_eq!(m[3][2], 45.0);
        assert_eq!(m[3][3], 48.0);
    }

    #[test]
    fn scalar_division() {
        let mut m = Mat4f64::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let n = m / 3.0;
        assert_eq!(n[0][0], 1.0 / 3.0);
        assert_eq!(n[0][1], 2.0 / 3.0);
        assert_eq!(n[0][2], 3.0 / 3.0);
        assert_eq!(n[0][3], 4.0 / 3.0);
        assert_eq!(n[1][0], 5.0 / 3.0);
        assert_eq!(n[1][1], 6.0 / 3.0);
        assert_eq!(n[1][2], 7.0 / 3.0);
        assert_eq!(n[1][3], 8.0 / 3.0);
        assert_eq!(n[2][0], 9.0 / 3.0);
        assert_eq!(n[2][1], 10.0 / 3.0);
        assert_eq!(n[2][2], 11.0 / 3.0);
        assert_eq!(n[2][3], 12.0 / 3.0);
        assert_eq!(n[3][0], 13.0 / 3.0);
        assert_eq!(n[3][1], 14.0 / 3.0);
        assert_eq!(n[3][2], 15.0 / 3.0);
        assert_eq!(n[3][3], 16.0 / 3.0);

        m /= 3.0;
        assert_eq!(m[0][0], 1.0 / 3.0);
        assert_eq!(m[0][1], 2.0 / 3.0);
        assert_eq!(m[0][2], 3.0 / 3.0);
        assert_eq!(m[0][3], 4.0 / 3.0);
        assert_eq!(m[1][0], 5.0 / 3.0);
        assert_eq!(m[1][1], 6.0 / 3.0);
        assert_eq!(m[1][2], 7.0 / 3.0);
        assert_eq!(m[1][3], 8.0 / 3.0);
        assert_eq!(m[2][0], 9.0 / 3.0);
        assert_eq!(m[2][1], 10.0 / 3.0);
        assert_eq!(m[2][2], 11.0 / 3.0);
        assert_eq!(m[2][3], 12.0 / 3.0);
        assert_eq!(m[3][0], 13.0 / 3.0);
        assert_eq!(m[3][1], 14.0 / 3.0);
        assert_eq!(m[3][2], 15.0 / 3.0);
        assert_eq!(m[3][3], 16.0 / 3.0);
    }

    #[test]
    fn matrix_multiplication() {
        let mut m = Mat4f64::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let n = Mat4f64::new([
            [17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0],
            [29.0, 30.0, 31.0, 32.0],
        ]);
        let r = m * n;
        m *= n;

        assert_eq!(r[0][0], 250.0);
        assert_eq!(r[0][1], 260.0);
        assert_eq!(r[0][2], 270.0);
        assert_eq!(r[0][3], 280.0);
        assert_eq!(r[1][0], 618.0);
        assert_eq!(r[1][1], 644.0);
        assert_eq!(r[1][2], 670.0);
        assert_eq!(r[1][3], 696.0);
        assert_eq!(r[2][0], 986.0);
        assert_eq!(r[2][1], 1028.0);
        assert_eq!(r[2][2], 1070.0);
        assert_eq!(r[2][3], 1112.0);
        assert_eq!(r[3][0], 1354.0);
        assert_eq!(r[3][1], 1412.0);
        assert_eq!(r[3][2], 1470.0);
        assert_eq!(r[3][3], 1528.0);

        assert_eq!(m[0][0], 250.0);
        assert_eq!(m[0][1], 260.0);
        assert_eq!(m[0][2], 270.0);
        assert_eq!(m[0][3], 280.0);
        assert_eq!(m[1][0], 618.0);
        assert_eq!(m[1][1], 644.0);
        assert_eq!(m[1][2], 670.0);
        assert_eq!(m[1][3], 696.0);
        assert_eq!(m[2][0], 986.0);
        assert_eq!(m[2][1], 1028.0);
        assert_eq!(m[2][2], 1070.0);
        assert_eq!(m[2][3], 1112.0);
        assert_eq!(m[3][0], 1354.0);
        assert_eq!(m[3][1], 1412.0);
        assert_eq!(m[3][2], 1470.0);
        assert_eq!(m[3][3], 1528.0);
    }

    #[test]
    fn vector_multiplication() {
        let m = Mat4f64::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let v = Vec4f64::new([2.0, 3.0, 4.0, 5.0]);
        let w = m * v;

        assert_eq!(w[0], 40.0);
        assert_eq!(w[1], 96.0);
        assert_eq!(w[2], 152.0);
        assert_eq!(w[3], 208.0);
    }
}
