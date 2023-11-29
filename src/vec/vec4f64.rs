use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vec4f64 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4f64 {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }

    pub fn one() -> Self {
        Self {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            w: 1.0,
        }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        self.x /= mag;
        self.y /= mag;
        self.z /= mag;
        self.w /= mag;
    }

    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        Self {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
            w: self.w / mag,
        }
    }

    pub fn dot(&self, other: Vec4f64) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl Add<Vec4f64> for Vec4f64 {
    type Output = Vec4f64;
    fn add(mut self, rhs: Vec4f64) -> Self::Output {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
        self
    }
}

impl AddAssign<Vec4f64> for Vec4f64 {
    fn add_assign(&mut self, rhs: Vec4f64) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl Sub<Vec4f64> for Vec4f64 {
    type Output = Vec4f64;
    fn sub(mut self, rhs: Vec4f64) -> Self::Output {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
        self
    }
}

impl SubAssign<Vec4f64> for Vec4f64 {
    fn sub_assign(&mut self, rhs: Vec4f64) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl Mul<f64> for Vec4f64 {
    type Output = Vec4f64;
    fn mul(mut self, scalar: f64) -> Self::Output {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        self.w *= scalar;
        self
    }
}

impl Mul<Vec4f64> for f64 {
    type Output = Vec4f64;
    fn mul(self, mut vec: Vec4f64) -> Self::Output {
        vec.x *= self;
        vec.y *= self;
        vec.z *= self;
        vec.w *= self;
        vec
    }
}

impl MulAssign<f64> for Vec4f64 {
    fn mul_assign(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        self.w *= scalar;
    }
}

impl Div<f64> for Vec4f64 {
    type Output = Vec4f64;
    fn div(mut self, scalar: f64) -> Self::Output {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
        self.w /= scalar;
        self
    }
}

impl DivAssign<f64> for Vec4f64 {
    fn div_assign(&mut self, scalar: f64) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
        self.w /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::vec4f64::Vec4f64;

    #[test]
    fn vector_addition() {
        let two = Vec4f64::one() + Vec4f64::one();
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
        assert_eq!(two.z, 2.0);
        assert_eq!(two.w, 2.0);
    }

    #[test]
    fn vector_subtraction() {
        let zero = Vec4f64::one() - Vec4f64::one();
        assert_eq!(zero.x, 0.0);
        assert_eq!(zero.y, 0.0);
        assert_eq!(zero.z, 0.0);
        assert_eq!(zero.w, 0.0);
    }

    #[test]
    fn scalar_multiplication() {
        let one = Vec4f64::one();
        let two = one * 2.0;
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
        assert_eq!(two.z, 2.0);
        assert_eq!(two.w, 2.0);
        assert_eq!(one.x, 1.0);
        assert_eq!(one.y, 1.0);
        assert_eq!(one.z, 1.0);
        assert_eq!(one.w, 1.0);

        let one = Vec4f64::one();
        let two = 2.0 * one;
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
        assert_eq!(two.z, 2.0);
        assert_eq!(two.w, 2.0);
        assert_eq!(one.x, 1.0);
        assert_eq!(one.y, 1.0);
        assert_eq!(one.z, 1.0);
        assert_eq!(one.w, 1.0);
    }

    #[test]
    fn scalar_division() {
        let one = Vec4f64::one();
        let half = one / 2.0;
        assert_eq!(half.x, 0.5);
        assert_eq!(half.y, 0.5);
        assert_eq!(half.z, 0.5);
        assert_eq!(half.w, 0.5);
        assert_eq!(one.x, 1.0);
        assert_eq!(one.y, 1.0);
        assert_eq!(one.z, 1.0);
        assert_eq!(one.w, 1.0);
    }
}
