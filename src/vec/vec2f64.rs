use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vec2f64 {
    pub x: f64,
    pub y: f64,
}

impl Vec2f64 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn one() -> Self {
        Self { x: 1.0, y: 1.0 }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        self.x /= mag;
        self.y /= mag;
    }

    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        Self {
            x: self.x / mag,
            y: self.y / mag,
        }
    }

    pub fn dot(&self, other: Vec2f64) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

impl Add<Vec2f64> for Vec2f64 {
    type Output = Vec2f64;
    fn add(mut self, rhs: Vec2f64) -> Self::Output {
        self.x += rhs.x;
        self.y += rhs.y;
        self
    }
}

impl AddAssign<Vec2f64> for Vec2f64 {
    fn add_assign(&mut self, rhs: Vec2f64) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub<Vec2f64> for Vec2f64 {
    type Output = Vec2f64;
    fn sub(mut self, rhs: Vec2f64) -> Self::Output {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self
    }
}

impl SubAssign<Vec2f64> for Vec2f64 {
    fn sub_assign(&mut self, rhs: Vec2f64) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<f64> for Vec2f64 {
    type Output = Vec2f64;
    fn mul(mut self, scalar: f64) -> Self::Output {
        self.x *= scalar;
        self.y *= scalar;
        self
    }
}

impl Mul<Vec2f64> for f64 {
    type Output = Vec2f64;
    fn mul(self, mut vec: Vec2f64) -> Self::Output {
        vec.x *= self;
        vec.y *= self;
        vec
    }
}

impl MulAssign<f64> for Vec2f64 {
    fn mul_assign(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
    }
}

impl Div<f64> for Vec2f64 {
    type Output = Vec2f64;
    fn div(mut self, scalar: f64) -> Self::Output {
        self.x /= scalar;
        self.y /= scalar;
        self
    }
}

impl DivAssign<f64> for Vec2f64 {
    fn div_assign(&mut self, scalar: f64) {
        self.x /= scalar;
        self.y /= scalar;
    }
}

#[cfg(test)]
mod tests {
    use crate::vec::vec2f64::Vec2f64;

    #[test]
    fn vector_addition() {
        let two = Vec2f64::one() + Vec2f64::one();
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
    }

    #[test]
    fn vector_subtraction() {
        let zero = Vec2f64::one() - Vec2f64::one();
        assert_eq!(zero.x, 0.0);
        assert_eq!(zero.y, 0.0);
    }

    #[test]
    fn scalar_multiplication() {
        let one = Vec2f64::one();
        let two = one * 2.0;
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
        assert_eq!(one.x, 1.0);
        assert_eq!(one.y, 1.0);

        let one = Vec2f64::one();
        let two = 2.0 * one;
        assert_eq!(two.x, 2.0);
        assert_eq!(two.y, 2.0);
        assert_eq!(one.x, 1.0);
        assert_eq!(one.y, 1.0);
    }

    #[test]
    fn scalar_division() {
        let one = Vec2f64::one();
        let half = one / 2.0;
        assert_eq!(half.x, 0.5);
        assert_eq!(half.y, 0.5);
        assert_eq!(one.x, 1.0);
        assert_eq!(one.y, 1.0);
    }
}
