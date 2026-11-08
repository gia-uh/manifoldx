"""Camera class for 3D rendering."""

import math

import numpy as np


class Camera:
    """Camera with position, target, FOV and projection."""

    def __init__(
        self, position=(0, 1, 2), target=(0, 0, 0), fov=60, up=(0, 1, 0), near=0.1, far=1000.0
    ):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.fov = fov
        self.up = np.array(up, dtype=np.float32)
        self.near = near
        self.far = far
        # Render layers this camera draws: a set of Layer indices, or None for all.
        self.layers: set[int] | None = None
        self._compute_spherical()

    def _compute_spherical(self):
        direction = self.position - self.target
        self._distance = np.linalg.norm(direction)
        direction_normalized = direction / self._distance
        self._azimuth = float(
            np.degrees(np.arctan2(direction_normalized[0], direction_normalized[2]))
        )
        self._elevation = float(np.degrees(np.arcsin(direction_normalized[1])))

    def get_view_matrix(self):
        """Compute view matrix using look-at formula.

        Convention: camera looks down -Z in view space (standard OpenGL/WebGPU).
        """
        # Plain float arithmetic: np.cross on 3-vectors costs more than the math,
        # and the renderer asks for this matrix several times a frame.
        px, py, pz = (float(v) for v in self.position)
        fx, fy, fz = (float(t) - p for t, p in zip(self.target, (px, py, pz)))
        n = math.sqrt(fx * fx + fy * fy + fz * fz)
        fx, fy, fz = fx / n, fy / n, fz / n  # forward, camera toward target
        ux, uy, uz = (float(v) for v in self.up)
        rx, ry, rz = fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux  # forward x up
        n = math.sqrt(rx * rx + ry * ry + rz * rz)
        if n < 1e-9:  # looking along up (a top view): world -z, north, is up on screen
            ux, uy, uz = (0.0, 0.0, -1.0) if abs(fz) < 0.9 else (0.0, 1.0, 0.0)
            rx, ry, rz = fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux
            n = math.sqrt(rx * rx + ry * ry + rz * rz)
        rx, ry, rz = rx / n, ry / n, rz / n
        tx, ty, tz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx  # right x forward

        # Rotation rows right, up, -forward (camera looks down -Z), then the translation.
        view = np.array([
            [rx, ry, rz, -(rx * px + ry * py + rz * pz)],
            [tx, ty, tz, -(tx * px + ty * py + tz * pz)],
            [-fx, -fy, -fz, fx * px + fy * py + fz * pz],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return view

    def get_forward(self):
        direction = self.target - self.position
        return direction / np.linalg.norm(direction)

    def get_right(self):
        return self.get_view_matrix()[0, :3]  # the view's first row, top views included

    def get_up(self):
        return self.get_view_matrix()[1, :3]

    def get_view_projection_matrix(self, aspect):
        view = self.get_view_matrix()
        proj = self.get_projection_matrix(aspect, near=self.near, far=self.far)
        return (proj @ view).T

    def get_azimuth_elevation(self):
        return (self._azimuth, self._elevation)

    def get_projection_matrix(self, aspect, near=0.1, far=100.0):
        """Compute perspective projection matrix for WebGPU.

        WebGPU NDC: x,y in [-1,1], z in [0,1].
        Uses the same formula as pylinalg mat_perspective with depth_range=(0,1).
        """
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        # WebGPU depth range [0, 1]:
        # c = -(far*1 - near*0) / (far - near) = -far / (far - near)
        # d = -(far * near * (1 - 0)) / (far - near) = -far*near / (far - near)
        proj[2, 2] = -far / (far - near)
        proj[2, 3] = -(far * near) / (far - near)
        proj[3, 2] = -1.0

        return proj

    def project(self, points, width, height):
        """World points to pixel coordinates (origin top-left) and camera depth.

        Uses the same view and projection matrices as the renderer, so a point
        lands on the pixel the GPU draws it at. Depth is the distance along the
        viewing direction.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        m = (
            self.get_projection_matrix(width / height, near=self.near, far=self.far)
            @ self.get_view_matrix()
        ).astype(np.float64)
        clip = np.c_[pts, np.ones(len(pts))] @ m.T
        w = clip[:, 3]
        xy = np.empty((len(pts), 2))
        xy[:, 0] = (clip[:, 0] / w + 1) / 2 * width
        xy[:, 1] = (1 - clip[:, 1] / w) / 2 * height
        return xy, w

    def move_to(self, position):
        """Set camera world position."""
        self.position = np.array(position, dtype=np.float32)
        self._compute_spherical()

    def move_by(self, delta):
        """Move camera by delta in world space."""
        self.position += np.array(delta, dtype=np.float32)
        self._compute_spherical()

    def look_at(self, target):
        """Set the point the camera looks at."""
        self.target = np.array(target, dtype=np.float32)
        self._compute_spherical()

    def set_pose(self, position, target):
        """Set both position and target at once."""
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self._compute_spherical()

    def orbit(self, d_azimuth=0.0, d_elevation=0.0):
        """Rotate camera around target by delta angles.

        Args:
            d_azimuth: Delta azimuth angle in degrees (horizontal rotation)
            d_elevation: Delta elevation angle in degrees (vertical rotation)

        Elevation clamped to ±89° to avoid gimbal lock.
        """
        self._azimuth += d_azimuth
        self._elevation += d_elevation
        self._elevation = np.clip(self._elevation, -89, 89)

        az_rad = np.radians(self._azimuth)
        el_rad = np.radians(self._elevation)

        direction = np.array(
            [np.cos(az_rad) * np.cos(el_rad), np.sin(el_rad), np.sin(az_rad) * np.cos(el_rad)],
            dtype=np.float32,
        )

        self.position = self.target + self._distance * direction

    def pan(self, dx, dy, relative_to_viewport=True):
        """Pan the camera perpendicular to view direction.

        Args:
            dx: Horizontal displacement
            dy: Vertical displacement
            relative_to_viewport: If True (default), dx/dy are in viewport units (0-1)
                                  scaled by distance. If False, dx/dy are in world units.
        """
        right = self.get_right()
        up = self.get_up()

        if relative_to_viewport:
            scale = self._distance * np.tan(np.radians(self.fov / 2))
        else:
            scale = 1.0

        displacement = (right * dx + up * dy) * scale

        self.target += displacement
        self.position += displacement

    def get_distance(self) -> float:
        """Get current distance from camera to target."""
        return float(self._distance)

    def zoom(self, factor):
        """Zoom by dividing distance to target by factor.

        Args:
            factor: Divisor for distance. >1 zooms in, <1 zooms out.

        Example: zoom(2.0) halves the distance to target (zooms in).
        """
        self._distance /= factor
        self._distance = max(0.1, self._distance)

        forward = self.get_forward()
        self.position = self.target - forward * self._distance

    def dolly(self, distance):
        """Move camera toward/away from target by distance.

        Args:
            distance: World units to move toward (positive) or away (negative).
        """
        self._distance -= distance
        self._distance = max(0.1, self._distance)

        forward = self.get_forward()
        self.position = self.target - forward * self._distance

    def fit(self, radius, center=(0, 0, 0), margin=0.8, azimuth=45, elevation=30):
        """Position camera to frame a sphere.

        Args:
            radius: Radius of sphere to fit in view
            center: Center of sphere in world space
            margin: Fraction of viewport height sphere should occupy (0.8 = 80%)
            azimuth: Camera azimuth angle in degrees
            elevation: Camera elevation angle in degrees
        """
        self.target = np.array(center, dtype=np.float32)
        self._azimuth = azimuth
        self._elevation = elevation

        fov_rad = np.radians(self.fov)
        self._distance = radius / np.sin((margin * fov_rad) / 2)

        az_rad = np.radians(self._azimuth)
        el_rad = np.radians(self._elevation)

        direction = np.array(
            [np.cos(az_rad) * np.cos(el_rad), np.sin(el_rad), np.sin(az_rad) * np.cos(el_rad)],
            dtype=np.float32,
        )

        self.position = self.target + self._distance * direction

    def fit_bounds(self, center, extent, margin=0.8, azimuth=45, elevation=30):
        """Position camera to frame an axis-aligned bounding box.

        Args:
            center: Center of bounding box
            extent: Half-extents (half-widths) of bounding box
            margin: Fraction of viewport to leave as padding
            azimuth: Camera azimuth angle in degrees
            elevation: Camera elevation angle in degrees

        extent can be:
          - float: cube with given half-extent
          - tuple(n,): uniform box (nx, ny, nz) all equal
          - tuple(3,): non-uniform box (hx, hy, hz)
        """
        if isinstance(extent, (int, float)):
            half_extents = np.array([extent, extent, extent], dtype=np.float32)
        elif len(extent) == 1:
            half_extents = np.array([extent[0], extent[0], extent[0]], dtype=np.float32)
        else:
            half_extents = np.array(extent[:3], dtype=np.float32)

        radius = np.linalg.norm(half_extents)

        self.fit(radius=radius, center=center, margin=margin, azimuth=azimuth, elevation=elevation)


__all__ = ["Camera"]
