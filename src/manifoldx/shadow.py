"""Light-space matrices for shadow mapping (pure numpy, no GPU)."""

import numpy as np


def _look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(np.dot(f, up)) > 0.999:  # degenerate: sun straight down/up
        up = np.array([0.0, 0.0, 1.0])
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float64)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def _ortho(extent, near, far):
    """Symmetric ortho box [-extent, extent]^2, depth near..far, wgpu z in [0,1]."""
    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = 1.0 / extent
    M[1, 1] = 1.0 / extent
    M[2, 2] = -1.0 / (far - near)
    M[2, 3] = -near / (far - near)
    M[3, 3] = 1.0
    return M


def _perspective(fov_y, aspect, near, far):
    """Perspective projection, wgpu convention (z in [0,1], right-handed)."""
    f = 1.0 / np.tan(fov_y * 0.5)
    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = far / (near - far)
    M[2, 3] = (near * far) / (near - far)
    M[3, 2] = -1.0
    return M


def compute_spot_light_view_proj(position, direction, outer_angle, near, far):
    """Perspective light-space view-projection for a spot light.

    The spot's cone maps to a square perspective frustum with vertical FOV =
    2·outer_angle. Returns a (4, 4) float32 matrix; M @ [x, y, z, 1] -> clip
    coords (wgpu: x, y in [-1, 1], z in [0, 1]).
    """
    position = np.asarray(position, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    view = _look_at(position, position + direction, up=np.array([0.0, 1.0, 0.0]))
    proj = _perspective(2.0 * outer_angle, 1.0, max(near, 1e-3), far)
    return (proj @ view).astype(np.float32)


def compute_light_view_proj(direction, target, extent, near, far, back_distance=None):
    """Ortho light-space view-projection for a directional sun.

    Returns a (4, 4) float32 matrix M with ``M @ [x, y, z, 1]`` -> clip coords
    (wgpu convention: x, y in [-1, 1], z in [0, 1]).
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    target = np.asarray(target, dtype=np.float64)
    if back_distance is None:
        back_distance = far * 0.5
    eye = target - direction * back_distance
    view = _look_at(eye, target, up=np.array([0.0, 1.0, 0.0]))
    proj = _ortho(extent, near, far)
    return (proj @ view).astype(np.float32)


MAX_CASCADES = 4


def cascade_splits(near, far, count, lam=0.75):
    """The view-space far distance of each of `count` cascades from `near` to `far`: the practical
    scheme, `lam` of the way from uniform to logarithmic splits."""
    i = np.arange(1, count + 1, dtype=np.float64) / count
    near = max(float(near), 1e-3)
    return lam * near * (far / near) ** i + (1.0 - lam) * (near + (far - near) * i)


def fit_cascade(inv_view, tan_half_fov, aspect, d0, d1, direction, resolution, reach):
    """An ortho light view-projection around the bounding sphere of the camera's view slice from depth
    `d0` to `d1`, its origin snapped to whole texels so shadows hold still while the camera moves.

    The sphere depends only on the slice, not on where the camera looks, so the box keeps its size when
    the camera turns. `reach` metres before the sphere toward the sun still cast into it. `inv_view` is
    the camera's view-to-world matrix (column vectors). Returns (matrix (4, 4) float32, the world size of
    a texel, the box's depth in metres)."""
    a2 = tan_half_fov**2 * (1.0 + aspect**2)  # squared corner offset per metre of depth
    c = min(0.5 * (d0 + d1) * (1.0 + a2), d1)  # where the near and far corners are equally far
    r = float(np.sqrt(max((c - d0) ** 2 + a2 * d0**2, (d1 - c) ** 2 + a2 * d1**2)))
    inv_view = np.asarray(inv_view, dtype=np.float64)
    centre = inv_view[:3, :3] @ np.array([0.0, 0.0, -c]) + inv_view[:3, 3]
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    basis = _look_at(np.zeros(3), direction, up=np.array([0.0, 1.0, 0.0]))[:2, :3]  # light right, up
    texel = 2.0 * r / resolution
    along = basis @ centre
    centre = centre + basis.T @ (np.round(along / texel) * texel - along)
    back = r + reach
    lvp = compute_light_view_proj(direction, centre, r, 0.0, back + r, back_distance=back)
    return lvp, texel, back + r
