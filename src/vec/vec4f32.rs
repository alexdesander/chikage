use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vec4f32 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4f32 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
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

    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn magnitude_squared(&self) -> f32 {
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

    pub fn dot(&self, other: Vec4f32) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl Add<Vec4f32> for Vec4f32 {
    type Output = Vec4f32;
    fn add(mut self, rhs: Vec4f32) -> Self::Output {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
        self
    }
}

impl AddAssign<Vec4f32> for Vec4f32 {
    fn add_assign(&mut self, rhs: Vec4f32) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl Sub<Vec4f32> for Vec4f32 {
    type Output = Vec4f32;
    fn sub(mut self, rhs: Vec4f32) -> Self::Output {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
        self
    }
}

impl SubAssign<Vec4f32> for Vec4f32 {
    fn sub_assign(&mut self, rhs: Vec4f32) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl Mul<f32> for Vec4f32 {
    type Output = Vec4f32;
    fn mul(mut self, scalar: f32) -> Self::Output {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        self.w *= scalar;
        self
    }
}

impl Mul<Vec4f32> for f32 {
    type Output = Vec4f32;
    fn mul(self, mut vec: Vec4f32) -> Self::Output {
        vec.x *= self;
        vec.y *= self;
        vec.z *= self;
        vec.w *= self;
        vec
    }
}

impl MulAssign<f32> for Vec4f32 {
    fn mul_assign(&mut self, scalar: f32) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
        self.w *= scalar;
    }
}

impl Div<f32> for Vec4f32 {
    type Output = Vec4f32;
    fn div(mut self, scalar: f32) -> Self::Output {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
        self.w /= scalar;
        self
    }
}

impl DivAssign<f32> for Vec4f32 {
    fn div_assign(&mut self, scalar: f32) {
        self.x /= scalar;
        self.y /= scalar;
        self.z /= scalar;
        self.w /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::vec4f32::Vec4f32;

    #[test]
    fn vector_addition() {
        let two = Vec4f32::one() + Vec4f32::one();
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
        assert_eq!(two.z, 2.0);
        assert_eq!(two.w, 2.0);
    }

    #[test]
    fn vector_subtraction() {
        let zero = Vec4f32::one() - Vec4f32::one();
        assert_eq!(zero.x, 0.0);
        assert_eq!(zero.y, 0.0);
        assert_eq!(zero.z, 0.0);
        assert_eq!(zero.w, 0.0);
    }

    #[test]
    fn scalar_multiplication() {
        let one = Vec4f32::one();
        let two = one * 2.0;
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
        assert_eq!(two.z, 2.0);
        assert_eq!(two.w, 2.0);
        assert_eq!(one.x, 1.0);
        assert_eq!(one.y, 1.0);
        assert_eq!(one.z, 1.0);
        assert_eq!(one.w, 1.0);

        let one = Vec4f32::one();
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
        let one = Vec4f32::one();
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
