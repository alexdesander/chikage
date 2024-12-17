use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

/// A four dimensional vector.
#[derive(Debug, Clone, Copy)]
pub struct Vec4f64 {
    pub coords: [f64; 4],
}

impl Vec4f64 {
    /// Create a new vector with user defined components.
    pub fn new(coords: [f64; 4]) -> Self {
        Self { coords }
    }

    /// Create a new vector with all components equal to 0.0.
    pub fn zero() -> Self {
        Self::new([0.0, 0.0, 0.0, 0.0])
    }

    /// Create a new vector with all components equal to 1.0.
    pub fn ones() -> Self {
        Self::new([1.0, 1.0, 1.0, 1.0])
    }

    /// The magnitude of the vector (also known as length).
    pub fn mag(&self) -> f64 {
        self.mag_squared().sqrt()
    }

    /// The magnitude of the vector (also known as length), but squared.
    /// This is faster to compute than mag() and useful in some situations.
    pub fn mag_squared(&self) -> f64 {
        self[0] * self[0] + self[1] * self[1] + self[2] * self[2] + self[3] * self[3]
    }

    /// Normalizes self
    /// This makes the vector a unit vector.
    pub fn norm(&mut self) {
        let mag = self.mag();
        *self /= mag;
    }

    /// Return self but as a normalized vector.
    /// This returns a unit vector.
    pub fn normed(&self) -> Self {
        let mag = self.mag();
        *self / mag
    }

    /// Calculate the dot product between self and other.
    pub fn dot(&self, other: Self) -> f64 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2] + self[3] * other[3]
    }
}

impl Index<usize> for Vec4f64 {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl IndexMut<usize> for Vec4f64 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl Add<Vec4f64> for Vec4f64 {
    type Output = Vec4f64;
    fn add(mut self, rhs: Vec4f64) -> Self::Output {
        self[0] += rhs[0];
        self[1] += rhs[1];
        self[2] += rhs[2];
        self[3] += rhs[3];
        self
    }
}

impl AddAssign<Vec4f64> for Vec4f64 {
    fn add_assign(&mut self, rhs: Vec4f64) {
        *self = *self + rhs;
    }
}

impl Sub<Vec4f64> for Vec4f64 {
    type Output = Vec4f64;
    fn sub(mut self, rhs: Vec4f64) -> Self::Output {
        self[0] -= rhs[0];
        self[1] -= rhs[1];
        self[2] -= rhs[2];
        self[3] -= rhs[3];
        self
    }
}

impl SubAssign<Vec4f64> for Vec4f64 {
    fn sub_assign(&mut self, rhs: Vec4f64) {
        *self = *self - rhs;
    }
}

impl Mul<Vec4f64> for f64 {
    type Output = Vec4f64;

    fn mul(self, mut v: Vec4f64) -> Self::Output {
        v[0] *= self;
        v[1] *= self;
        v[2] *= self;
        v[3] *= self;
        v
    }
}

impl Mul<f64> for Vec4f64 {
    type Output = Vec4f64;

    fn mul(self, scalar: f64) -> Self::Output {
        scalar * self
    }
}

impl MulAssign<f64> for Vec4f64 {
    fn mul_assign(&mut self, scalar: f64) {
        self[0] *= scalar;
        self[1] *= scalar;
        self[2] *= scalar;
        self[3] *= scalar;
    }
}

impl Div<f64> for Vec4f64 {
    type Output = Vec4f64;

    fn div(mut self, scalar: f64) -> Self::Output {
        self[0] /= scalar;
        self[1] /= scalar;
        self[2] /= scalar;
        self[3] /= scalar;
        self
    }
}

impl DivAssign<f64> for Vec4f64 {
    fn div_assign(&mut self, scalar: f64) {
        self[0] /= scalar;
        self[1] /= scalar;
        self[2] /= scalar;
        self[3] /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::vec4f64::Vec4f64;

    #[test]
    fn vector_creation() {
        let zero = Vec4f64::zero();
        let ones = Vec4f64::ones();
        let v = Vec4f64::new([2.0, 3.0, 4.0, 5.0]);

        assert_eq!(zero[0], 0.0);
        assert_eq!(zero[1], 0.0);
        assert_eq!(zero[2], 0.0);
        assert_eq!(zero[3], 0.0);
        assert_eq!(ones[0], 1.0);
        assert_eq!(ones[1], 1.0);
        assert_eq!(ones[2], 1.0);
        assert_eq!(ones[3], 1.0);
        assert_eq!(v[0], 2.0);
        assert_eq!(v[1], 3.0);
        assert_eq!(v[2], 4.0);
        assert_eq!(v[3], 5.0);
    }

    #[test]
    fn mag_and_mag_squared() {
        let zero = Vec4f64::zero();
        let ones = Vec4f64::ones();
        let v = Vec4f64::new([4.0, 7.0, 5.0, 2.0]);

        assert_eq!(zero.mag(), 0.0);
        assert_eq!(ones.mag(), 4.0f64.sqrt());
        assert_eq!(
            v.mag(),
            (4.0 * 4.0 + 7.0 * 7.0 + 5.0 * 5.0 + 2.0 * 2.0f64).sqrt()
        );

        assert_eq!(zero.mag_squared(), 0.0);
        assert_eq!(ones.mag_squared(), 4.0f64);
        assert_eq!(
            v.mag_squared(),
            4.0 * 4.0 + 7.0 * 7.0f64 + 5.0 * 5.0 + 2.0 * 2.0f64
        );
    }

    #[test]
    fn norm_and_normed() {
        // Zero vec
        let mut zero = Vec4f64::zero();
        let normed = zero.normed();
        zero.norm();
        assert!(normed[0].is_nan());
        assert!(normed[1].is_nan());
        assert!(normed[2].is_nan());
        assert!(normed[3].is_nan());
        assert!(zero[0].is_nan());
        assert!(zero[1].is_nan());
        assert!(zero[2].is_nan());
        assert!(zero[3].is_nan());

        // Nonzero vecs
        let mut ones = Vec4f64::ones();
        let normed = ones.normed();
        ones.norm();
        assert!((0.99999..1.000001).contains(&ones.mag()));
        assert!((0.99999..1.000001).contains(&normed.mag()));

        let mut v = Vec4f64::new([4.0, 7.0, 5.0, 2.0]);
        let normed = v.normed();
        v.norm();
        assert!((0.99999..1.000001).contains(&v.mag()));
        assert!((0.99999..1.000001).contains(&normed.mag()));
    }

    #[test]
    fn dot() {
        let zero = Vec4f64::zero();
        let ones = Vec4f64::ones();
        let v = Vec4f64::new([4.0, 7.0, 5.0, 2.0]);

        let dot = zero.dot(v);
        assert_eq!(dot, 0.0);

        let dot = v.dot(zero);
        assert_eq!(dot, 0.0);

        let dot = ones.dot(v);
        assert_eq!(dot, 18.0);

        let dot = v.dot(ones);
        assert_eq!(dot, 18.0);

        let dot = v.dot(v);
        assert_eq!(dot, 16.0 + 49.0 + 25.0 + 4.0);
    }

    #[test]
    fn scalar_multiplication() {
        let zero = Vec4f64::zero();
        let ones = Vec4f64::ones();
        let mut v = Vec4f64::new([4.0, 7.0, 5.0, 2.0]);

        let w = 3.0 * zero;
        assert_eq!(w[0], 0.0);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);
        assert_eq!(w[3], 0.0);

        let w = zero * 3.0;
        assert_eq!(w[0], 0.0);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);
        assert_eq!(w[3], 0.0);

        let w = 3.0 * ones;
        assert_eq!(w[0], 3.0);
        assert_eq!(w[1], 3.0);
        assert_eq!(w[2], 3.0);
        assert_eq!(w[3], 3.0);

        let w = ones * 3.0;
        assert_eq!(w[0], 3.0);
        assert_eq!(w[1], 3.0);
        assert_eq!(w[2], 3.0);
        assert_eq!(w[3], 3.0);

        let w = 3.0 * v;
        assert_eq!(w[0], 12.0);
        assert_eq!(w[1], 21.0);
        assert_eq!(w[2], 15.0);
        assert_eq!(w[3], 6.0);

        let w = v * 3.0;
        assert_eq!(w[0], 12.0);
        assert_eq!(w[1], 21.0);
        assert_eq!(w[2], 15.0);
        assert_eq!(w[3], 6.0);

        v *= 3.0;
        assert_eq!(v[0], 12.0);
        assert_eq!(v[1], 21.0);
        assert_eq!(v[2], 15.0);
        assert_eq!(v[3], 6.0);
    }

    #[test]
    fn scalar_division() {
        let zero = Vec4f64::zero();
        let ones = Vec4f64::ones();
        let mut v = Vec4f64::new([4.0, 7.0, 5.0, 2.0]);

        let w = zero / 3.0;
        assert_eq!(w[0], 0.0);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);
        assert_eq!(w[3], 0.0);

        let w = ones / 3.0;
        assert_eq!(w[0], 1.0 / 3.0);
        assert_eq!(w[1], 1.0 / 3.0);
        assert_eq!(w[2], 1.0 / 3.0);
        assert_eq!(w[3], 1.0 / 3.0);

        let w = v / 3.0;
        assert_eq!(w[0], 4.0 / 3.0);
        assert_eq!(w[1], 7.0 / 3.0);
        assert_eq!(w[2], 5.0 / 3.0);
        assert_eq!(w[3], 2.0 / 3.0);

        v /= 3.0;
        assert_eq!(v[0], 4.0 / 3.0);
        assert_eq!(v[1], 7.0 / 3.0);
        assert_eq!(v[2], 5.0 / 3.0);
        assert_eq!(v[3], 2.0 / 3.0);
    }

    #[test]
    fn vector_addition() {
        let mut v = Vec4f64::new([4.0, 7.0, 5.0, 2.0]);
        let w = Vec4f64::new([-2.0, 10.0, 2.0, 3.0]);
        let r = v + w;
        assert_eq!(r[0], 2.0);
        assert_eq!(r[1], 17.0);
        assert_eq!(r[2], 7.0);
        assert_eq!(r[3], 5.0);
        v += w;
        assert_eq!(v[0], 2.0);
        assert_eq!(v[1], 17.0);
        assert_eq!(v[2], 7.0);
        assert_eq!(v[3], 5.0);
    }

    #[test]
    fn vector_subtraction() {
        let mut v = Vec4f64::new([4.0, 7.0, 5.0, 2.0]);
        let w = Vec4f64::new([-2.0, 10.0, 2.0, 3.0]);
        let r = v - w;
        assert_eq!(r[0], 6.0);
        assert_eq!(r[1], -3.0);
        assert_eq!(r[2], 3.0);
        assert_eq!(r[3], -1.0);
        v -= w;
        assert_eq!(v[0], 6.0);
        assert_eq!(v[1], -3.0);
        assert_eq!(v[2], 3.0);
        assert_eq!(v[3], -1.0);
    }
}
