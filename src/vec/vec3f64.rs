use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vec3f64 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3f64 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn one() -> Self {
        Self {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        self.x /= mag;
        self.y /= mag;
        self.z /= mag;
    }

    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        Self {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }

    pub fn dot(&self, other: Vec3f64) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&self, right: Vec3f64) -> Vec3f64 {
        Vec3f64 {
            x: self.y * right.z - self.z * right.y,
            y: self.z * right.x - self.x * right.z,
            z: self.x * right.y - self.y * right.x,
        }
    }
}

impl Add<Vec3f64> for Vec3f64 {
    type Output = Vec3f64;
    fn add(mut self, rhs: Vec3f64) -> Self::Output {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self
    }
}

impl AddAssign<Vec3f64> for Vec3f64 {
    fn add_assign(&mut self, rhs: Vec3f64) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub<Vec3f64> for Vec3f64 {
    type Output = Vec3f64;
    fn sub(mut self, rhs: Vec3f64) -> Self::Output {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self
    }
}

impl SubAssign<Vec3f64> for Vec3f64 {
    fn sub_assign(&mut self, rhs: Vec3f64) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<f64> for Vec3f64 {
    type Output = Vec3f64;
    fn mul(mut self, scalar: f64) -> Self::Output {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        self
    }
}

impl Mul<Vec3f64> for f64 {
    type Output = Vec3f64;
    fn mul(self, mut vec: Vec3f64) -> Self::Output {
        vec.x *= self;
        vec.y *= self;
        vec.z *= self;
        vec
    }
}

impl MulAssign<f64> for Vec3f64 {
    fn mul_assign(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl Div<f64> for Vec3f64 {
    type Output = Vec3f64;
    fn div(mut self, scalar: f64) -> Self::Output {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
        self
    }
}

impl DivAssign<f64> for Vec3f64 {
    fn div_assign(&mut self, scalar: f64) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::vec3f64::Vec3f64;

    #[test]
    fn vector_addition() {
        let two = Vec3f64::one() + Vec3f64::one();
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
        assert_eq!(two.z, 2.0);
    }

    #[test]
    fn vector_subtraction() {
        let zero = Vec3f64::one() - Vec3f64::one();
        assert_eq!(zero.x, 0.0);
        assert_eq!(zero.y, 0.0);
        assert_eq!(zero.z, 0.0);
    }

    #[test]
    fn scalar_multiplication() {
        let one = Vec3f64::one();
        let two = one * 2.0;
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
        assert_eq!(two.z, 2.0);
        assert_eq!(one.x, 1.0);
        assert_eq!(one.y, 1.0);
        assert_eq!(one.z, 1.0);

        let one = Vec3f64::one();
        let two = 2.0 * one;
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
        assert_eq!(two.z, 2.0);
        assert_eq!(one.x, 1.0);
        assert_eq!(one.y, 1.0);
        assert_eq!(one.z, 1.0);
    }

    #[test]
    fn scalar_division() {
        let one = Vec3f64::one();
        let half = one / 2.0;
        assert_eq!(half.x, 0.5);
        assert_eq!(half.y, 0.5);
        assert_eq!(half.z, 0.5);
        assert_eq!(one.x, 1.0);
        assert_eq!(one.y, 1.0);
        assert_eq!(one.z, 1.0);
    }

    #[test]
    fn cross() {
        let left = Vec3f64::new(2.5, 1.5, 0.5);
        let right = Vec3f64::new(3.2, 2.2, 1.1);
        let cross = left.cross(right);
        assert!((0.54..0.56).contains(&cross.x));
        assert!((-1.16..-1.14).contains(&cross.y));
        assert!((0.69..0.71).contains(&cross.z));
    }
}
