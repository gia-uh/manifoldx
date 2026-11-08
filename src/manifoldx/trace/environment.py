"""Light from infinitely far away: an equirectangular radiance map and an optional sun disk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUN_ANGULAR_RADIUS = np.radians(0.2665)


@dataclass
class Environment:
    """radiance: (H, W, 3) linear, row 0 at the zenith, column u = atan2(z, x) / 2π + 1/2, as
    manifoldx.ibl. sun_direction points towards the sun; sun_illuminance is what the sun
    alone delivers to a surface facing it (in the same units as radiance × sr)."""

    radiance: np.ndarray
    intensity: float = 1.0
    sun_direction: tuple | None = None
    sun_color: tuple = (1.0, 1.0, 1.0)
    sun_illuminance: float = 0.0
    sun_angular_radius: float = SUN_ANGULAR_RADIUS

    @classmethod
    def uniform(cls, rgb=(1.0, 1.0, 1.0)) -> "Environment":
        return cls(np.full((8, 16, 3), rgb, np.float32))

    @classmethod
    def from_map(cls, env) -> "Environment":
        """From a manifoldx.ibl.EnvironmentMap."""
        return cls(np.asarray(env.data, np.float32), float(env.intensity))

    @classmethod
    def from_hdr(cls, path, intensity=1.0) -> "Environment":
        from manifoldx.ibl import _decode_rgbe

        return cls(_decode_rgbe(str(path)), intensity)

    @classmethod
    def sky(cls, sun_direction=(0.4, 0.7, 0.3), *, turbidity: float = 3.0, sun_ratio: float = 5.0,
            ground=(0.25, 0.23, 0.2), width: int = 512, height: int = 256) -> "Environment":
        """A clear sky (Preetham et al. 1999) with zenith luminance 1, and a sun whose
        illuminance is sun_ratio times the sky's on a horizontal surface. Below the horizon
        the map holds a ground of the given reflectance lit by sun and sky."""
        s = np.asarray(sun_direction, np.float64)
        s /= np.linalg.norm(s)
        dirs = directions(width, height)
        rad = _preetham(dirs, s, turbidity)
        up = dirs[..., 1]
        sky_e = _horizontal_illuminance(rad, height)
        sun_e = sun_ratio * sky_e
        sun_rgb = _sun_tint(max(s[1], 0.0), turbidity)
        e_ground = sky_e + sun_e * max(s[1], 0.0)
        ground_rad = np.asarray(ground, np.float64) * e_ground / np.pi
        horizon = rad[height // 2 - 1: height // 2].mean(axis=(0, 1))
        blend = np.clip(-up / 0.05, 0.0, 1.0)[..., None]  # soften the horizon line
        rad = np.where(up[..., None] >= 0, rad, horizon * (1 - blend) + ground_rad * blend)
        return cls(rad.astype(np.float32), 1.0, tuple(s), tuple(sun_rgb), float(sun_e) if s[1] > -0.02 else 0.0)

    # ------------------------------------------------------------ sampling tables

    def tables(self):
        """(radiance (H*W, 4), marginal cdf (H,), conditional cdf (H*W,), total weight).

        Weights are luminance × sin θ per texel, so sampling a texel then a point in it
        gives pdf(ω) = w / total × H W / (2π² sin θ)."""
        rad = np.asarray(self.radiance, np.float64) * self.intensity
        H, W = rad.shape[:2]
        theta = (np.arange(H) + 0.5) / H * np.pi
        lum = rad @ [0.2126, 0.7152, 0.0722]
        w = (lum + 1e-6 * max(lum.mean(), 1e-12)) * np.sin(theta)[:, None]
        rows = w.sum(axis=1)
        total = rows.sum()
        marginal = np.cumsum(rows) / total
        cond = np.cumsum(w, axis=1) / rows[:, None]
        marginal[-1] = 1.0
        cond[:, -1] = 1.0
        out = np.zeros((H * W, 4), np.float32)
        out[:, :3] = rad.reshape(-1, 3)
        out[:, 3] = (w / total * H * W).reshape(-1)  # pdf over the unit square, per texel
        return out, marginal.astype(np.float32), cond.reshape(-1).astype(np.float32), float(total)


def directions(width, height):
    """Unit direction at each texel centre of an equirect map, (H, W, 3)."""
    v = (np.arange(height) + 0.5) / height
    u = (np.arange(width) + 0.5) / width
    lat = (0.5 - v) * np.pi
    phi = (u - 0.5) * 2 * np.pi
    lat, phi = np.meshgrid(lat, phi, indexing="ij")
    return np.stack([np.cos(lat) * np.cos(phi), np.sin(lat), np.cos(lat) * np.sin(phi)], axis=-1)


def _horizontal_illuminance(rad, height):
    H, W = rad.shape[:2]
    lat = (0.5 - (np.arange(H) + 0.5) / H) * np.pi
    up = np.sin(lat)
    solid = np.cos(lat) * (np.pi / H) * (2 * np.pi / W)  # solid angle of a texel
    lum = rad @ [0.2126, 0.7152, 0.0722]
    return float((lum * (np.maximum(up, 0) * solid)[:, None]).sum())


def _perez(theta, gamma, c):
    A, B, C, D, E = c
    return (1 + A * np.exp(B / np.maximum(np.cos(theta), 0.01))) * \
        (1 + C * np.exp(D * gamma) + E * np.cos(gamma) ** 2)


def _preetham(dirs, sun, T):
    theta_s = np.arccos(np.clip(sun[1], -1, 1))
    theta_s = min(theta_s, np.radians(89.5))
    Y = [0.1787 * T - 1.4630, -0.3554 * T + 0.4275, -0.0227 * T + 5.3251, 0.1206 * T - 2.5771,
         -0.0670 * T + 0.3703]
    x = [-0.0193 * T - 0.2592, -0.0665 * T + 0.0008, -0.0004 * T + 0.2125, -0.0641 * T - 0.8989,
         -0.0033 * T + 0.0452]
    y = [-0.0167 * T - 0.2608, -0.0950 * T + 0.0092, -0.0079 * T + 0.2102, -0.0441 * T - 1.6537,
         -0.0109 * T + 0.0529]
    t, t2, t3 = theta_s, theta_s ** 2, theta_s ** 3
    zx = (0.00166 * t3 - 0.00375 * t2 + 0.00209 * t) * T * T + (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * t
                                                               + 0.00394) * T + (0.11693 * t3 - 0.21196 * t2
                                                                                 + 0.06052 * t + 0.25886)
    zy = (0.00275 * t3 - 0.00610 * t2 + 0.00317 * t) * T * T + (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * t
                                                               + 0.00516) * T + (0.15346 * t3 - 0.26756 * t2
                                                                                 + 0.06670 * t + 0.26688)
    up = np.maximum(dirs[..., 1], 0.001)
    theta = np.arccos(up)
    gamma = np.arccos(np.clip(dirs @ sun, -1, 1))
    lum = _perez(theta, gamma, Y) / _perez(0.0, theta_s, Y)
    cx = zx * _perez(theta, gamma, x) / _perez(0.0, theta_s, x)
    cy = zy * _perez(theta, gamma, y) / _perez(0.0, theta_s, y)
    X = cx / cy * lum
    Z = (1 - cx - cy) / cy * lum
    rgb = np.stack([3.2406 * X - 1.5372 * lum - 0.4986 * Z,
                    -0.9689 * X + 1.8758 * lum + 0.0415 * Z,
                    0.0557 * X - 0.2040 * lum + 1.0570 * Z], axis=-1)
    return np.maximum(rgb, 0.0)


def _sun_tint(elevation_sin, T):
    """Colour of direct sunlight, warmer as the sun nears the horizon; luminance 1."""
    m = 1.0 / max(elevation_sin + 0.15 * (elevation_sin * 90 + 3.885) ** -1.253, 0.02)  # air mass
    beta = 0.04608 * T - 0.04586
    lam = np.array([0.68, 0.55, 0.44])  # µm for R, G, B
    tau = np.exp(-m * (0.008735 * lam ** -4.08 + beta * lam ** -1.3))
    return tau / (tau @ [0.2126, 0.7152, 0.0722])
