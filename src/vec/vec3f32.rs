use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

/// A three dimensional vector.
#[derive(Debug, Clone, Copy)]
pub struct Vec3f32 {
    pub coords: [f32; 3],
}

impl Vec3f32 {
    /// Create a new vector with user defined components.
    pub fn new(coords: [f32; 3]) -> Self {
        Self { coords }
    }

    /// Create a new vector with all components equal to 0.0.
    pub fn zero() -> Self {
        Self::new([0.0, 0.0, 0.0])
    }

    /// Create a new vector with all components equal to 1.0.
    pub fn ones() -> Self {
        Self::new([1.0, 1.0, 1.0])
    }

    /// The magnitude of the vector (also known as length).
    pub fn mag(&self) -> f32 {
        self.mag_squared().sqrt()
    }

    /// The magnitude of the vector (also known as length), but squared.
    /// This is faster to compute than mag() and useful in some situations.
    pub fn mag_squared(&self) -> f32 {
        self[0] * self[0] + self[1] * self[1] + self[2] * self[2]
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
    pub fn dot(&self, other: Self) -> f32 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2]
    }

    /// Calculates the crossproduct between self and rhs
    pub fn cross(&self, rhs: Self) -> Self {
        Self::new([
            self[1] * rhs[2] - self[2] * rhs[1],
            self[2] * rhs[0] - self[0] * rhs[2],
            self[0] * rhs[1] - self[1] * rhs[0],
        ])
    }
}

impl Index<usize> for Vec3f32 {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl IndexMut<usize> for Vec3f32 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl Add<Vec3f32> for Vec3f32 {
    type Output = Vec3f32;
    fn add(mut self, rhs: Vec3f32) -> Self::Output {
        self[0] += rhs[0];
        self[1] += rhs[1];
        self[2] += rhs[2];
        self
    }
}

impl AddAssign<Vec3f32> for Vec3f32 {
    fn add_assign(&mut self, rhs: Vec3f32) {
        *self = *self + rhs;
    }
}

impl Sub<Vec3f32> for Vec3f32 {
    type Output = Vec3f32;
    fn sub(mut self, rhs: Vec3f32) -> Self::Output {
        self[0] -= rhs[0];
        self[1] -= rhs[1];
        self[2] -= rhs[2];
        self
    }
}

impl SubAssign<Vec3f32> for Vec3f32 {
    fn sub_assign(&mut self, rhs: Vec3f32) {
        *self = *self - rhs;
    }
}

impl Mul<Vec3f32> for f32 {
    type Output = Vec3f32;

    fn mul(self, mut v: Vec3f32) -> Self::Output {
        v[0] *= self;
        v[1] *= self;
        v[2] *= self;
        v
    }
}

impl Mul<f32> for Vec3f32 {
    type Output = Vec3f32;

    fn mul(self, scalar: f32) -> Self::Output {
        scalar * self
    }
}

impl MulAssign<f32> for Vec3f32 {
    fn mul_assign(&mut self, scalar: f32) {
        self[0] *= scalar;
        self[1] *= scalar;
        self[2] *= scalar;
    }
}

impl Div<f32> for Vec3f32 {
    type Output = Vec3f32;

    fn div(mut self, scalar: f32) -> Self::Output {
        self[0] /= scalar;
        self[1] /= scalar;
        self[2] /= scalar;
        self
    }
}

impl DivAssign<f32> for Vec3f32 {
    fn div_assign(&mut self, scalar: f32) {
        self[0] /= scalar;
        self[1] /= scalar;
        self[2] /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::vec3f32::Vec3f32;

    #[test]
    fn vector_creation() {
        let zero = Vec3f32::zero();
        let ones = Vec3f32::ones();
        let v = Vec3f32::new([2.0, 3.0, 4.0]);

        assert_eq!(zero[0], 0.0);
        assert_eq!(zero[1], 0.0);
        assert_eq!(zero[2], 0.0);
        assert_eq!(ones[0], 1.0);
        assert_eq!(ones[1], 1.0);
        assert_eq!(ones[2], 1.0);
        assert_eq!(v[0], 2.0);
        assert_eq!(v[1], 3.0);
        assert_eq!(v[2], 4.0);
    }

    #[test]
    fn mag_and_mag_squared() {
        let zero = Vec3f32::zero();
        let ones = Vec3f32::ones();
        let v = Vec3f32::new([4.0, 7.0, 5.0]);

        assert_eq!(zero.mag(), 0.0);
        assert_eq!(ones.mag(), 3.0f32.sqrt());
        assert_eq!(v.mag(), (4.0 * 4.0 + 7.0 * 7.0 + 5.0 * 5.0f32).sqrt());

        assert_eq!(zero.mag_squared(), 0.0);
        assert_eq!(ones.mag_squared(), 3.0f32);
        assert_eq!(v.mag_squared(), 4.0 * 4.0 + 7.0 * 7.0f32 + 5.0 * 5.0f32);
    }

    #[test]
    fn norm_and_normed() {
        // Zero vec
        let mut zero = Vec3f32::zero();
        let normed = zero.normed();
        zero.norm();
        assert!(normed[0].is_nan());
        assert!(normed[1].is_nan());
        assert!(normed[2].is_nan());
        assert!(zero[0].is_nan());
        assert!(zero[1].is_nan());
        assert!(zero[2].is_nan());

        // Nonzero vecs
        let mut ones = Vec3f32::ones();
        let normed = ones.normed();
        ones.norm();
        assert!((0.99999..1.000001).contains(&ones.mag()));
        assert!((0.99999..1.000001).contains(&normed.mag()));

        let mut v = Vec3f32::new([4.0, 7.0, 5.0]);
        let normed = v.normed();
        v.norm();
        assert!((0.99999..1.000001).contains(&v.mag()));
        assert!((0.99999..1.000001).contains(&normed.mag()));
    }

    #[test]
    fn dot() {
        let zero = Vec3f32::zero();
        let ones = Vec3f32::ones();
        let v = Vec3f32::new([4.0, 7.0, 5.0]);

        let dot = zero.dot(v);
        assert_eq!(dot, 0.0);

        let dot = v.dot(zero);
        assert_eq!(dot, 0.0);

        let dot = ones.dot(v);
        assert_eq!(dot, 16.0);

        let dot = v.dot(ones);
        assert_eq!(dot, 16.0);

        let dot = v.dot(v);
        assert_eq!(dot, 16.0 + 49.0 + 25.0);
    }

    #[test]
    fn scalar_multiplication() {
        let zero = Vec3f32::zero();
        let ones = Vec3f32::ones();
        let mut v = Vec3f32::new([4.0, 7.0, 5.0]);

        let w = 3.0 * zero;
        assert_eq!(w[0], 0.0);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);

        let w = zero * 3.0;
        assert_eq!(w[0], 0.0);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);

        let w = 3.0 * ones;
        assert_eq!(w[0], 3.0);
        assert_eq!(w[1], 3.0);
        assert_eq!(w[2], 3.0);

        let w = ones * 3.0;
        assert_eq!(w[0], 3.0);
        assert_eq!(w[1], 3.0);
        assert_eq!(w[2], 3.0);

        let w = 3.0 * v;
        assert_eq!(w[0], 12.0);
        assert_eq!(w[1], 21.0);
        assert_eq!(w[2], 15.0);

        let w = v * 3.0;
        assert_eq!(w[0], 12.0);
        assert_eq!(w[1], 21.0);
        assert_eq!(w[2], 15.0);

        v *= 3.0;
        assert_eq!(v[0], 12.0);
        assert_eq!(v[1], 21.0);
        assert_eq!(v[2], 15.0);
    }

    #[test]
    fn scalar_division() {
        let zero = Vec3f32::zero();
        let ones = Vec3f32::ones();
        let mut v = Vec3f32::new([4.0, 7.0, 5.0]);

        let w = zero / 3.0;
        assert_eq!(w[0], 0.0);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);

        let w = ones / 3.0;
        assert_eq!(w[0], 1.0 / 3.0);
        assert_eq!(w[1], 1.0 / 3.0);
        assert_eq!(w[2], 1.0 / 3.0);

        let w = v / 3.0;
        assert_eq!(w[0], 4.0 / 3.0);
        assert_eq!(w[1], 7.0 / 3.0);
        assert_eq!(w[2], 5.0 / 3.0);

        v /= 3.0;
        assert_eq!(v[0], 4.0 / 3.0);
        assert_eq!(v[1], 7.0 / 3.0);
        assert_eq!(v[2], 5.0 / 3.0);
    }

    #[test]
    fn vector_addition() {
        let mut v = Vec3f32::new([4.0, 7.0, 5.0]);
        let w = Vec3f32::new([-2.0, 10.0, 2.0]);
        let r = v + w;
        assert_eq!(r[0], 2.0);
        assert_eq!(r[1], 17.0);
        assert_eq!(r[2], 7.0);
        v += w;
        assert_eq!(v[0], 2.0);
        assert_eq!(v[1], 17.0);
        assert_eq!(v[2], 7.0);
    }

    #[test]
    fn vector_subtraction() {
        let mut v = Vec3f32::new([4.0, 7.0, 5.0]);
        let w = Vec3f32::new([-2.0, 10.0, 2.0]);
        let r = v - w;
        assert_eq!(r[0], 6.0);
        assert_eq!(r[1], -3.0);
        assert_eq!(r[2], 3.0);
        v -= w;
        assert_eq!(v[0], 6.0);
        assert_eq!(v[1], -3.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn cross_product() {
        let v = Vec3f32::new([4.0, 7.0, 5.0]);
        let w = Vec3f32::new([2.0, -4.0, 3.0]);
        let cross = v.cross(w);
        assert_eq!(cross[0], 41.0);
        assert_eq!(cross[1], -2.0);
        assert_eq!(cross[2], -30.0);
    }
}
