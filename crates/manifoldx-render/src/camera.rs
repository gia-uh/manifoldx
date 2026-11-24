//! Camera and projection math.
//!
//! The conventions match the existing engine camera: right-handed view space
//! looking down `-Z`, WebGPU NDC with `z` in `[0, 1]`, and reversed depth
//! (near maps to 1, far to 0) so a float depth buffer keeps precision far
//! away. The render pass clears depth to 0 and compares with `Greater`.

use crate::math::Mat4;

/// A perspective camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// World-space eye position.
    pub position: [f32; 3],
    /// World-space point the camera looks at.
    pub target: [f32; 3],
    /// World up hint.
    pub up: [f32; 3],
    /// Vertical field of view in degrees.
    pub fov_y_degrees: f32,
    /// Near clip distance, positive.
    pub near: f32,
    /// Far clip distance, greater than `near`.
    pub far: f32,
}

impl Camera {
    /// The engine defaults, with explicit values for a native caller.
    pub fn new(position: [f32; 3], target: [f32; 3]) -> Self {
        Self {
            position,
            target,
            up: [0.0, 1.0, 0.0],
            fov_y_degrees: 60.0,
            near: 0.1,
            far: 100.0,
        }
    }

    /// The look-at view matrix in row-major order.
    ///
    /// This is the same construction as the Python camera, including the
    /// top-view fallback when the forward axis is parallel to `up`.
    pub fn view(&self) -> Mat4 {
        let [px, py, pz] = self.position;
        let mut fx = self.target[0] - px;
        let mut fy = self.target[1] - py;
        let mut fz = self.target[2] - pz;
        let length = (fx * fx + fy * fy + fz * fz).sqrt();
        if length < 1e-9 {
            return Mat4::IDENTITY;
        }
        fx /= length;
        fy /= length;
        fz /= length;

        let [mut ux, mut uy, mut uz] = self.up;
        let (mut rx, mut ry, mut rz) = (fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux);
        let mut n = (rx * rx + ry * ry + rz * rz).sqrt();
        if n < 1e-9 {
            // Looking along up: pick world -Z (or +Y when nearly vertical),
            // then recompute right from the corrected up.
            (ux, uy, uz) = if fz.abs() < 0.9 {
                (0.0, 0.0, -1.0)
            } else {
                (0.0, 1.0, 0.0)
            };
            (rx, ry, rz) = (fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux);
            n = (rx * rx + ry * ry + rz * rz).sqrt();
            if n < 1e-9 {
                return Mat4::IDENTITY;
            }
        }
        rx /= n;
        ry /= n;
        rz /= n;

        let (tx, ty, tz) = (ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx);

        Mat4([
            rx,
            ry,
            rz,
            -(rx * px + ry * py + rz * pz),
            tx,
            ty,
            tz,
            -(tx * px + ty * py + tz * pz),
            -fx,
            -fy,
            -fz,
            fx * px + fy * py + fz * pz,
            0.0,
            0.0,
            0.0,
            1.0,
        ])
    }

    /// The reversed-depth perspective projection matrix.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        let f = 1.0 / (self.fov_y_degrees.to_radians() * 0.5).tan();
        let near = self.near;
        let far = self.far;
        let mut out = Mat4::ZERO;
        out.0[0] = f / aspect;
        out.0[5] = f;
        out.0[10] = near / (far - near);
        out.0[11] = (far * near) / (far - near);
        out.0[14] = -1.0;
        out
    }

    /// `projection * view`, the matrix the vertex shader receives.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect).multiply(&self.view())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ndc_depth(camera: &Camera, z_view: f32) -> f32 {
        let clip = camera.projection(1.0).transform_point([0.0, 0.0, z_view]);
        clip[2] / clip[3]
    }

    #[test]
    fn reversed_depth_maps_near_to_one_and_far_to_zero() {
        let camera = Camera::new([0.0, 0.0, 5.0], [0.0; 3]);
        assert!((ndc_depth(&camera, -camera.near) - 1.0).abs() < 1e-5);
        assert!(ndc_depth(&camera, -camera.far).abs() < 1e-5);
    }

    #[test]
    fn view_puts_the_eye_at_the_origin_and_target_on_negative_z() {
        let camera = Camera::new([0.0, 0.0, 5.0], [0.0; 3]);
        let eye = camera.view().transform_point(camera.position);
        assert!(eye[0].abs() < 1e-5 && eye[1].abs() < 1e-5 && eye[2].abs() < 1e-5);
        let target = camera.view().transform_point(camera.target);
        assert!(target[2] < 0.0);
        assert!(target[0].abs() < 1e-5 && target[1].abs() < 1e-5);
    }

    #[test]
    fn a_point_behind_the_camera_is_clipped_by_a_negative_w() {
        let camera = Camera::new([0.0, 0.0, 5.0], [0.0; 3]);
        let clip = camera.view_projection(1.0).transform_point([0.0, 0.0, 6.0]);
        assert!(clip[3] < 0.0);
    }
}
