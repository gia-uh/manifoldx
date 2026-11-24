//! Small fixed-function math for the base renderer.
//!
//! Row-major `Mat4` matches the Python camera matrix layout; `to_columns`
//! produces the column-major array WGSL expects for a uniform.

#![allow(clippy::needless_range_loop)]

/// A 3-component vector.
pub type Vec3 = [f32; 3];

/// A 4-component quaternion in `(x, y, z, w)` order, as the Transform row stores it.
pub type Quat = [f32; 4];

/// A 4x4 row-major matrix, indexed `m[row * 4 + column]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4(pub [f32; 16]);

impl Mat4 {
    /// The identity matrix.
    pub const IDENTITY: Self = Self([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);

    /// The zero matrix.
    pub const ZERO: Self = Self([0.0; 16]);

    /// Element at `row, column`.
    pub fn at(&self, row: usize, column: usize) -> f32 {
        self.0[row * 4 + column]
    }

    /// Multiply two row-major matrices: `self * rhs`.
    pub fn multiply(&self, rhs: &Self) -> Self {
        let mut out = Self::ZERO;
        for row in 0..4 {
            for column in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.at(row, k) * rhs.at(k, column);
                }
                out.0[row * 4 + column] = sum;
            }
        }
        out
    }

    /// Transform a point, returning homogeneous `(x, y, z, w)`.
    pub fn transform_point(&self, point: Vec3) -> [f32; 4] {
        let mut out = [0.0; 4];
        for row in 0..4 {
            out[row] = self.at(row, 0) * point[0]
                + self.at(row, 1) * point[1]
                + self.at(row, 2) * point[2]
                + self.at(row, 3);
        }
        out
    }

    /// Column-major order for a WGSL `mat4x4<f32>` uniform.
    pub fn to_columns(&self) -> [f32; 16] {
        let mut out = [0.0; 16];
        for row in 0..4 {
            for column in 0..4 {
                out[column * 4 + row] = self.at(row, column);
            }
        }
        out
    }
}

/// Normalize a quaternion, returning identity for a zero or non-finite value.
pub fn normalize_quat(quat: Quat) -> Quat {
    let norm_sq = quat.iter().map(|v| v * v).sum::<f32>();
    if !norm_sq.is_finite() || norm_sq < 1e-20 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let inv = norm_sq.sqrt().recip();
    [quat[0] * inv, quat[1] * inv, quat[2] * inv, quat[3] * inv]
}

/// Rotation matrix rows for a unit quaternion.
pub fn quat_to_mat3(quat: Quat) -> [f32; 9] {
    let [x, y, z, w] = normalize_quat(quat);
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    [
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy - wz),
        2.0 * (xz + wy),
        2.0 * (xy + wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz - wx),
        2.0 * (xz - wy),
        2.0 * (yz + wx),
        1.0 - 2.0 * (xx + yy),
    ]
}

/// Build the model matrix `T * R * S` from one Transform row's components.
pub fn trs(position: Vec3, rotation: Quat, scale: Vec3) -> Mat4 {
    let r = quat_to_mat3(rotation);
    let mut out = Mat4::IDENTITY;
    for row in 0..3 {
        for column in 0..3 {
            out.0[row * 4 + column] = r[row * 3 + column] * scale[column];
        }
        out.0[row * 4 + 3] = position[row];
    }
    out
}

/// Decode one 10-float Transform row into a model matrix.
///
/// Layout is the existing engine schema: position `0..3`, quaternion
/// `3..7` in `(x, y, z, w)` order, scale `7..10`.
pub fn model_from_transform_row(row: &[f32]) -> Mat4 {
    debug_assert!(row.len() >= 10);
    trs(
        [row[0], row[1], row[2]],
        [row[3], row[4], row[5], row[6]],
        [row[7], row[8], row[9]],
    )
}

/// Transform a direction by the model's linear part, ignoring translation.
pub fn transform_direction(model: &Mat4, direction: Vec3) -> Vec3 {
    let mut out = [0.0; 3];
    for (row, value) in out.iter_mut().enumerate() {
        *value = model.at(row, 0) * direction[0]
            + model.at(row, 1) * direction[1]
            + model.at(row, 2) * direction[2];
    }
    out
}

/// The unit normal after the inverse-transpose normal transform.
///
/// [`trs`] builds `T * R * S`, whose upper-left 3x3 columns are `s_j * R e_j`.
/// For that family `M^-T n = sum_j (n_j / |column_j|^2) * column_j`, so no
/// explicit inverse is needed. A zero or non-finite axis is dropped; when
/// every axis is degenerate the normalized input normal is returned, and a
/// non-finite input falls back to `+Z`. The shader in `renderer.rs` mirrors
/// this function.
pub fn transform_normal(model: &Mat4, normal: Vec3) -> Vec3 {
    let mut out = [0.0f32; 3];
    for column in 0..3 {
        let axis = [
            model.at(0, column),
            model.at(1, column),
            model.at(2, column),
        ];
        let len_sq = axis.iter().map(|value| value * value).sum::<f32>();
        if len_sq.is_finite() && len_sq > 1e-20 && len_sq < 1e30 {
            let factor = normal[column] / len_sq;
            for (row, value) in out.iter_mut().enumerate() {
                *value += factor * axis[row];
            }
        }
    }
    unit_or(out).unwrap_or_else(|| unit_or(normal).unwrap_or([0.0, 0.0, 1.0]))
}

fn unit_or(value: Vec3) -> Option<Vec3> {
    let len_sq = value
        .iter()
        .map(|component| component * component)
        .sum::<f32>();
    if len_sq.is_finite() && len_sq > 1e-20 && len_sq < 1e30 {
        let inverse = len_sq.sqrt().recip();
        Some([value[0] * inverse, value[1] * inverse, value[2] * inverse])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    #[test]
    fn identity_leaves_a_point_alone() {
        let p = Mat4::IDENTITY.transform_point([1.0, 2.0, 3.0]);
        assert_eq!(p, [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn trs_translates_and_scales() {
        let m = trs([2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 2.0]);
        let p = m.transform_point([1.0, 0.0, 0.0]);
        assert_eq!(&p[..3], &[4.0, 3.0, 4.0]);
    }

    #[test]
    fn quaternion_turns_x_towards_negative_z() {
        // +90 degrees about Y.
        let half = std::f32::consts::FRAC_PI_4;
        let m = trs([0.0; 3], [0.0, half.sin(), 0.0, half.cos()], [1.0; 3]);
        let p = m.transform_point([1.0, 0.0, 0.0]);
        close(p[0], 0.0);
        close(p[1], 0.0);
        close(p[2], -1.0);
    }

    #[test]
    fn zero_quaternion_falls_back_to_identity() {
        let m = trs([0.0; 3], [0.0, 0.0, 0.0, 0.0], [1.0; 3]);
        assert_eq!(m, Mat4::IDENTITY);
    }

    #[test]
    fn columns_are_the_transpose() {
        let m = Mat4([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let c = m.to_columns();
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], 5.0);
        assert_eq!(c[3], 13.0);
        assert_eq!(c[4], 2.0);
        assert_eq!(c[12], 4.0);
    }

    fn unit(v: Vec3) -> Vec3 {
        let len = dot(v, v).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }

    fn dot(a: Vec3, b: Vec3) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[test]
    fn inverse_transpose_keeps_an_oblique_normal_perpendicular() {
        // The reviewed failure case: scale(2, 1, 1) with an oblique normal
        // (1, 1, 0)/sqrt(2) and the in-plane tangent (1, -1, 0)/sqrt(2).
        // A yaw rotation and a translation must not change the result.
        let half = 0.6f32 * 0.5;
        let model = trs(
            [1.0, 2.0, 3.0],
            [0.0, half.sin(), 0.0, half.cos()],
            [2.0, 1.0, 1.0],
        );
        let normal = unit([1.0, 1.0, 0.0]);
        let tangent = unit([1.0, -1.0, 0.0]);
        let scaled_tangent = unit(transform_direction(&model, tangent));

        // The old formula transformed a normal like a tangent. Its dot product
        // with the scaled tangent is 0.6, so the normal stops being perpendicular.
        let legacy = unit(transform_direction(&model, normal));
        let legacy_dot = dot(legacy, scaled_tangent);
        assert!(
            (legacy_dot - 0.6).abs() < 1e-5,
            "old formula dot {legacy_dot}, expected 0.6"
        );

        let correct = transform_normal(&model, normal);
        let correct_dot = dot(correct, scaled_tangent);
        assert!(
            correct_dot.abs() < 1e-5,
            "inverse-transpose dot {correct_dot}, expected 0"
        );
    }

    #[test]
    fn degenerate_scale_keeps_the_normal_finite() {
        let model = trs([0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0]);
        for normal in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]] {
            let out = transform_normal(&model, normal);
            assert!(out.iter().all(|value| value.is_finite()), "{out:?}");
            let len = dot(out, out).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "{out:?} has length {len}");
        }

        let collapsed = trs([0.0; 3], [0.0, 0.0, 0.0, 1.0], [0.0; 3]);
        assert_eq!(
            transform_normal(&collapsed, [0.0, 0.0, 1.0]),
            [0.0, 0.0, 1.0]
        );
    }
}
