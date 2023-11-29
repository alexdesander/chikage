use crate::{vec::{vec3f64::Vec3f64, vec4f64::Vec4f64}, mat::mat4f64::Mat4f64};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Rot3f64 {
    // Scalar
    pub s: f64,
    // Bivector
    pub xy: f64,
    pub yz: f64,
    pub zx: f64,
}

impl Rot3f64 {
    /// Returns the identity of a rotor (basically no rotation)
    pub fn identity() -> Self {
        Rot3f64 {
            s: 1.0,
            xy: 0.0,
            yz: 0.0,
            zx: 0.0,
        }
    }

    /// Construct a new Rotor that rotates vectors by DOUBLE the angle and
    /// direction between vector a and b.
    /// Make sure a and b are normalized vectors and not 0.
    pub fn new(a: Vec3f64, b: Vec3f64) -> Self {
        if cfg!(debug_assertions) {
            debug_assert!(
                (0.999..1.001).contains(&a.magnitude())
                    && (0.999..1.001).contains(&b.magnitude()),
                "Construction of a rotor requires normalized vectors!"
            );
        }

        Rot3f64 {
            s: b.x * a.x + b.y * a.y + b.z * a.z,
            xy: b.x * a.y - b.y * a.x,
            yz: b.y * a.z - b.z * a.y,
            zx: b.z * a.x - b.x * a.z,
        }
    }

    /// Construct a new Rotor that rotates vectors by the angle and
    /// direction between vector a and b.
    /// Make sure a and b are normalized vectors and not 0.
    pub fn new_exact(a: Vec3f64, mut b: Vec3f64) -> Self {
        if cfg!(debug_assertions) {
            debug_assert!(
                (0.9999..1.0001).contains(&a.magnitude())
                    && (0.9999..1.0001).contains(&b.magnitude()),
                "Construction of a rotor requires normalized vectors!"
            );
        }

        // Check if a ~= -b using the dot product
        if (-1.00001..-0.99999).contains(&a.dot(b)) {
            b = a.perpendicular();
        } else {
            let a_plus_b = a + b;
            b = a_plus_b / a_plus_b.magnitude();
        }

        Self::new(a, b)
    }

    /// Returns self but inverted (reverse rotation)
    pub fn inverted(&self) -> Self {
        let mut result = *self;
        result.invert();
        result
    }

    pub fn invert(&mut self) {
        self.xy = -self.xy;
        self.yz = -self.yz;
        self.zx = -self.zx;
    }

    /// Returns the resulting vector after rotating v (this is RvR^(-1))
    pub fn rotated_vec(&self, mut v: Vec3f64) -> Vec3f64 {
        self.rotate_vec(&mut v);
        v
    }

    /// Rotates v (this is RvR^(-1))
    pub fn rotate_vec(&self, v: &mut Vec3f64) {
        let tx = self.s * v.x + self.xy * v.y - self.zx * v.z;
        let ty = self.s * v.y - self.xy * v.x + self.yz * v.z;
        let tz = self.s * v.z - self.yz * v.y + self.zx * v.x;
        let txyz = self.xy * v.z + self.yz * v.x + self.zx * v.y;

        v.x = tx * self.s + ty * self.xy - tz * self.zx + txyz * self.yz;
        v.y = ty * self.s - tx * self.xy + tz * self.yz + txyz * self.zx;
        v.z = tz * self.s + tx * self.zx - ty * self.yz + txyz * self.xy;
    }

    /// Creates a new rotor which is the combination of self and r
    /// (First self then r)
    pub fn appended(&self, r: Rot3f64) -> Self {
        let mut result = *self;
        result.append(r);
        result
    }

    /// Appends a rotor to this rotor
    /// The new rotation is the combination of both
    pub fn append(&mut self, r: Rot3f64) {
        let s = self.s * r.s - self.xy * r.xy - self.yz * r.yz - self.zx * r.zx;
        let xy = self.s * r.xy + self.xy * r.s - self.yz * r.zx + self.zx * r.yz;
        let yz = self.s * r.yz + self.yz * r.s + self.xy * r.zx - self.zx * r.xy;
        let zx = self.s * r.zx + self.zx * r.s - self.xy * r.yz + self.yz * r.xy;

        self.s = s;
        self.xy = xy;
        self.yz = yz;
        self.zx = zx;
    }

    /// Normalizes the rotor, doing this is pretty important
    pub fn normalize(&mut self) {
        let mag_sqrd = self.s * self.s + self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
        let mag = mag_sqrd.sqrt();
        self.s /= mag;
        self.xy /= mag;
        self.yz /= mag;
        self.zx /= mag;
    }

    /// Creates a 4x4 rotation matrix (3x3 and padded to make it homogenous)
    #[rustfmt::skip]
    pub fn rotation_mat(&self) -> Mat4f64 {
        let new_x = self.rotated_vec(Vec3f64 { x: 1.0, y: 0.0, z: 0.0 });
        let new_y = self.rotated_vec(Vec3f64 { x: 0.0, y: 1.0, z: 0.0 });
        let new_z = self.rotated_vec(Vec3f64 { x: 0.0, y: 0.0, z: 1.0 });

        Mat4f64::new_row_major(
            Vec4f64::new(new_x.x, new_y.x, new_z.x, 0.0),
            Vec4f64::new(new_x.y, new_y.y, new_z.y, 0.0),
            Vec4f64::new(new_x.z, new_y.z, new_z.z, 0.0),
            Vec4f64::new(0.0, 0.0, 0.0, 1.0)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_rotation() {
        let a = Vec3f64::new(1.0, 0.0, 0.0);
        let b = Vec3f64::new(1.0, 0.0, 0.0);
        let rotor = Rot3f64::new(a, b);

        let mut v = Vec3f64::new(1.0, 1.0, 1.0);
        rotor.rotate_vec(&mut v);
        assert!((0.9999..1.0001).contains(&v.x));
        assert!((0.9999..1.0001).contains(&v.y));
        assert!((0.9999..1.0001).contains(&v.z));
    }

    #[test]
    fn test_double_rotation_fix() {
        let a = Vec3f64::new(1.0, 0.0, 0.0);
        let b = Vec3f64::new(0.0, 1.0, 0.0);
        let rotor = Rot3f64::new_exact(a, b);

        let mut v = Vec3f64::new(1.0, 0.0, 0.0);
        rotor.rotate_vec(&mut v);
        assert!((-0.0001..0.0001).contains(&v.x));
        assert!((0.9999..1.0001).contains(&v.y));
        assert!((-0.0001..0.0001).contains(&v.z));
    }

    #[test]
    fn test_double_rotation_fix_180() {
        let a = Vec3f64::new(1.0, 0.0, 0.0);
        let b = Vec3f64::new(-1.0, 0.0, 0.0);
        let rotor = Rot3f64::new_exact(a, b);

        let mut v = Vec3f64::new(1.0, 0.0, 0.0);
        rotor.rotate_vec(&mut v);
        assert!((-1.0001..-0.9999).contains(&v.x));
        assert!((-0.0001..0.0001).contains(&v.y));
        assert!((-0.0001..0.0001).contains(&v.z));
    }

    #[test]
    fn test_append() {
        let a = Vec3f64::new(1.0, 0.0, 0.0);
        let mut b = Vec3f64::new(0.0, 1.0, 0.0);
        b.normalize();
        let mut rotor = Rot3f64::new(a, b);
        rotor.append(rotor);
        let mut v = Vec3f64::new(1.0, 0.0, 0.0);
        rotor.rotate_vec(&mut v);
        assert!((0.99999..1.00001).contains(&v.x));
        assert!((-0.00001..0.00001).contains(&v.y));
        assert!((-0.00001..0.00001).contains(&v.z));
    }
}
