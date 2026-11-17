"""Image-Based Lighting: EnvironmentMap class, builders, and CPU precomputation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from manifoldx import _deferred

# The precompute's shape. PRECOMPUTE_VERSION goes into the disk cache key with the rest,
# so a change to the algorithm or to any of these numbers never reads a stale entry.
PRECOMPUTE_VERSION = 2
CUBE_SIZE = 128
IRRADIANCE_SIZE, IRRADIANCE_SAMPLES = 64, 256
PREFILTER_MIPS, PREFILTER_SAMPLES = 8, 512


def _cache_dir() -> Path:
    """Where precomputed environments are kept: $MANIFOLDX_CACHE_DIR, else ~/.cache/manifoldx."""
    root = os.environ.get("MANIFOLDX_CACHE_DIR") or Path.home() / ".cache" / "manifoldx"
    return Path(root) / "ibl"


class EnvironmentMap(_deferred._DeferredWrites):
    """Equirectangular float32 radiance data (H, W, 3), linear colour space.

    Build via class methods; pass to engine.set_environment().
    Precomputed GPU resources are cached on the object after first upload.
    """

    def __init__(self, data: np.ndarray, intensity: float = 1.0, show_skybox: bool = False):
        assert data.ndim == 3 and data.shape[2] == 3, "data must be (H, W, 3)"
        self.data = data.astype(np.float32)
        self.intensity = intensity
        self.show_skybox = show_skybox

        # Lazily populated by _precompute()
        self._irradiance: Optional[np.ndarray] = None   # (6, 64, 64, 4) float16
        self._prefiltered: Optional[list] = None         # list of 8 arrays (6, S, S, 4) float16
        self._computed: bool = False

    @classmethod
    def from_color(cls, rgb: Tuple[float, float, float]) -> "EnvironmentMap":
        """Uniform ambient — same radiance from all directions."""
        data = np.full((32, 64, 3), rgb, dtype=np.float32)
        return cls(data=data)

    @classmethod
    def from_sky(
        cls,
        zenith: Tuple[float, float, float],
        horizon: Tuple[float, float, float],
        ground: Tuple[float, float, float],
        height: int = 64,
        width: int = 128,
    ) -> "EnvironmentMap":
        """Procedural sky gradient: zenith (top) → horizon (middle) → ground (bottom)."""
        lats = np.linspace(np.pi / 2, -np.pi / 2, height)
        data = np.zeros((height, width, 3), dtype=np.float32)
        z = np.array(zenith, dtype=np.float32)
        h = np.array(horizon, dtype=np.float32)
        g = np.array(ground, dtype=np.float32)
        for i, lat in enumerate(lats):
            t = float(np.sin(lat))  # 1 at zenith, 0 at horizon, -1 at ground
            if t >= 0:
                data[i] = z * t + h * (1.0 - t)
            else:
                data[i] = h * (1.0 + t) + g * (-t)
        return cls(data=data)

    @classmethod
    def from_image(cls, path: str, exposure: float = 1.0) -> "EnvironmentMap":
        """Load any Pillow-supported image (JPEG/PNG) as equirectangular env map.

        Input is treated as sRGB and linearised before use.
        """
        from PIL import Image

        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        # sRGB → linear
        arr = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
        arr *= exposure
        return cls(data=arr)

    @classmethod
    def from_hdr(cls, path: str) -> "EnvironmentMap":
        """Load a Radiance RGBE (.hdr) file and return linear-float32 data."""
        data = _decode_rgbe(path)
        return cls(data=data)

    def _cache_key(self) -> str:
        h = hashlib.sha256()
        h.update(repr((PRECOMPUTE_VERSION, CUBE_SIZE, IRRADIANCE_SIZE, IRRADIANCE_SAMPLES, PREFILTER_MIPS,
                       PREFILTER_SAMPLES, self.data.shape)).encode())
        h.update(np.ascontiguousarray(self.data).tobytes())
        return h.hexdigest()

    def _precompute(self):
        """Compute irradiance cubemap + prefiltered mip chain. Cached on the object after
        the first call, and on disk (see _cache_dir) across runs."""
        if self._computed:
            return
        path = _cache_dir() / f"{self._cache_key()}.npz"
        try:
            with np.load(path) as f:
                self._irradiance = f["irradiance"]
                self._prefiltered = [f[f"mip{m}"] for m in range(PREFILTER_MIPS)]
            self._computed = True
            return
        except (OSError, KeyError, ValueError):  # no entry, or an unreadable one: compute it
            pass
        cube = _equirect_to_cubemap(self.data, face_size=CUBE_SIZE)
        self._irradiance = _compute_irradiance(cube, out_size=IRRADIANCE_SIZE, samples=IRRADIANCE_SAMPLES)
        # Mip 0 is roughness 0, a mirror: its prefiltered map is the cube itself. Sampling it
        # cost about 70% of the precompute (manifoldx#25).
        self._prefiltered = [_with_alpha(cube).astype(np.float16)]
        for mip in range(1, PREFILTER_MIPS):
            roughness = mip / (PREFILTER_MIPS - 1)
            size = max(1, CUBE_SIZE >> mip)
            self._prefiltered.append(
                _compute_prefiltered(cube, roughness=roughness, out_size=size, samples=PREFILTER_SAMPLES)
            )
        self._computed = True
        try:  # write then rename, so a reader never sees half a file
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
            np.savez(tmp, irradiance=self._irradiance, **{f"mip{m}": a for m, a in enumerate(self._prefiltered)})
            tmp.replace(path)
        except OSError:  # a read-only home only costs the next run the precompute
            pass


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

PRESETS = {
    "studio": lambda: EnvironmentMap.from_sky(
        zenith=(0.6, 0.6, 0.65),
        horizon=(0.5, 0.5, 0.55),
        ground=(0.1, 0.1, 0.1),
    ),
    "sky": lambda: EnvironmentMap.from_sky(
        zenith=(0.05, 0.15, 0.6),
        horizon=(0.5, 0.65, 0.9),
        ground=(0.05, 0.05, 0.05),
    ),
    "neutral": lambda: EnvironmentMap.from_color((0.3, 0.3, 0.3)),
    "dark": lambda: EnvironmentMap.from_color((0.02, 0.02, 0.02)),
}


# ---------------------------------------------------------------------------
# RGBE decoder
# ---------------------------------------------------------------------------

def _decode_rgbe(path: str) -> np.ndarray:
    """Decode a Radiance RGBE (.hdr) file to float32 (H, W, 3), linear."""
    with open(path, "rb") as f:
        raw = f.read()

    # Parse header: skip to blank line
    pos = 0
    while True:
        nl = raw.index(b"\n", pos)
        line = raw[pos:nl]
        pos = nl + 1
        if line == b"":
            break

    # Size line: -Y H +X W
    nl = raw.index(b"\n", pos)
    size_line = raw[pos:nl].decode("ascii")
    pos = nl + 1
    parts = size_line.split()
    H, W = int(parts[1]), int(parts[3])

    # Decode scanlines
    result = np.zeros((H, W, 4), dtype=np.uint8)
    for row in range(H):
        marker = raw[pos: pos + 4]
        pos += 4
        if marker[0] == 2 and marker[1] == 2:
            # New RLE: each channel encoded separately
            scanline_w = (marker[2] << 8) | marker[3]
            line = bytearray()  # the four channels one after another
            for ch in range(4):
                end = len(line) + scanline_w
                while len(line) < end:
                    code = raw[pos]
                    if code > 128:
                        line += raw[pos + 1: pos + 2] * (code - 128)
                        pos += 2
                    else:
                        line += raw[pos + 1: pos + 1 + code]
                        pos += 1 + code
                del line[end:]  # a run past the end of the scanline is cut there
            w = min(W, scanline_w)
            result[row, :w] = np.frombuffer(line, dtype=np.uint8).reshape(4, scanline_w)[:, :w].T
        else:
            # Uncompressed / old RLE
            chunk = np.frombuffer(raw[pos - 4: pos - 4 + W * 4], dtype=np.uint8)
            result[row] = chunk.reshape(W, 4)
            pos += (W - 1) * 4

    # RGBE → float32 RGB
    # 2^(E - 136) is a power of two, so the float32 product is exact: no rounding
    # beyond what a float64 product cast to float32 does.
    e = np.arange(256)
    scale = np.where(e > 0, np.ldexp(1.0, e - 128 - 8), 0.0).astype(np.float32)
    return result[..., :3] * scale[result[..., 3]][..., None]


# ---------------------------------------------------------------------------
# Cubemap utilities
# ---------------------------------------------------------------------------

def _face_directions(face_size: int):
    """Return list of 6 (face_size, face_size, 3) world-direction arrays."""
    coords = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
    u, v = np.meshgrid(coords, coords)
    return [
        np.stack([ np.ones_like(u), -v, -u], axis=-1),  # +X
        np.stack([-np.ones_like(u), -v,  u], axis=-1),  # -X
        np.stack([ u,  np.ones_like(v),  v], axis=-1),  # +Y
        np.stack([ u, -np.ones_like(v), -v], axis=-1),  # -Y
        np.stack([ u, -v,  np.ones_like(u)], axis=-1),  # +Z
        np.stack([-u, -v, -np.ones_like(u)], axis=-1),  # -Z
    ]


def _equirect_to_cubemap(equirect: np.ndarray, face_size: int) -> np.ndarray:
    """Sample equirectangular image into 6-face cubemap.

    equirect: (H, W, 3) float32
    Returns:  (6, face_size, face_size, 3) float32
    """
    H, W = equirect.shape[:2]
    cube = np.zeros((6, face_size, face_size, 3), dtype=np.float32)
    for fi, dirs in enumerate(_face_directions(face_size)):
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-8)
        d = dirs / norms
        phi = np.arctan2(d[..., 2], d[..., 0])
        theta = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
        eu = (phi / (2 * np.pi) + 0.5) * (W - 1)
        ev = (0.5 - theta / np.pi) * (H - 1)
        eu0 = np.clip(np.floor(eu).astype(int), 0, W - 1)
        eu1 = np.clip(eu0 + 1, 0, W - 1)
        ev0 = np.clip(np.floor(ev).astype(int), 0, H - 1)
        ev1 = np.clip(ev0 + 1, 0, H - 1)
        wu, wv = (eu - eu0)[..., None], (ev - ev0)[..., None]
        cube[fi] = (
            equirect[ev0, eu0] * (1 - wu) * (1 - wv)
            + equirect[ev0, eu1] * wu * (1 - wv)
            + equirect[ev1, eu0] * (1 - wu) * wv
            + equirect[ev1, eu1] * wu * wv
        )
    return cube


def _with_alpha(rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([rgb, np.ones((*rgb.shape[:-1], 1), dtype=rgb.dtype)], axis=-1)


def _sample_cube(cube: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Sample (6, H, W, C) cubemap at N directions (N, 3). Returns (N, C)."""
    ax = np.abs(dirs)
    dom = np.argmax(ax, axis=-1)
    sign = np.sign(dirs[np.arange(len(dirs)), dom])
    face_map = np.where(
        dom == 0,
        np.where(sign > 0, 0, 1),
        np.where(dom == 1, np.where(sign > 0, 2, 3), np.where(sign > 0, 4, 5)),
    )
    dom_val = ax[np.arange(len(dirs)), dom].clip(min=1e-8)
    u_sel = np.where(
        dom == 0,
        np.where(sign > 0, -dirs[:, 2], dirs[:, 2]),
        np.where(dom == 1, dirs[:, 0], np.where(sign > 0, dirs[:, 0], -dirs[:, 0])),
    ) / dom_val
    v_sel = np.where(
        dom == 0,
        -dirs[:, 1],
        np.where(dom == 1, np.where(sign > 0, dirs[:, 2], -dirs[:, 2]), -dirs[:, 1]),
    ) / dom_val
    S = cube.shape[1]
    pu = np.clip(((u_sel + 1) / 2) * (S - 1), 0, S - 1).astype(int)
    pv = np.clip(((v_sel + 1) / 2) * (S - 1), 0, S - 1).astype(int)
    return cube[face_map, pv, pu]


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

def _compute_irradiance(cube: np.ndarray, out_size: int = 64, samples: int = 256) -> np.ndarray:
    """Cosine-weighted hemisphere integration for diffuse irradiance.

    cube: (6, S, S, 3) float32
    Returns: (6, out_size, out_size, 4) float16
    """
    irr = np.zeros((6, out_size, out_size, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    phi_s = rng.uniform(0.0, 2 * np.pi, samples).astype(np.float32)
    cos_t = rng.uniform(0.0, 1.0, samples).astype(np.float32)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    ts = np.stack([sin_t * np.cos(phi_s), sin_t * np.sin(phi_s), cos_t], axis=-1)

    for fi, dirs in enumerate(_face_directions(out_size)):
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-8)
        N_all = (dirs / norms).reshape(-1, 3)
        is_up = np.abs(N_all[:, 1]) > 0.99
        ref = np.where(is_up[:, None], np.array([[1., 0., 0.]]), np.array([[0., 1., 0.]]))
        right = np.cross(N_all, ref)
        right /= np.linalg.norm(right, axis=-1, keepdims=True).clip(min=1e-8)
        up2 = np.cross(N_all, right)
        world = (
            ts[None, :, 0:1] * right[:, None, :]
            + ts[None, :, 1:2] * up2[:, None, :]
            + ts[None, :, 2:3] * N_all[:, None, :]
        ).reshape(-1, 3)
        colors = _sample_cube(cube, world).reshape(-1, samples, 3)
        irr[fi] = colors.mean(axis=1).reshape(out_size, out_size, 3) * np.pi

    out = np.concatenate([irr, np.ones((*irr.shape[:3], 1), dtype=np.float32)], axis=-1)
    return out.astype(np.float16)


def _ggx_importance_sample(roughness: float, samples: int, rng) -> np.ndarray:
    """GGX-importance-sampled half-vectors in tangent space. Returns (samples, 3)."""
    a = roughness * roughness
    xi1 = rng.random(samples).astype(np.float32)
    xi2 = rng.random(samples).astype(np.float32)
    phi = 2 * np.pi * xi1
    cos_theta = np.sqrt((1 - xi2) / np.maximum(1 - (1 - a ** 2) * xi2, 1e-8))
    sin_theta = np.sqrt(np.maximum(1 - cos_theta ** 2, 0))
    return np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=-1).astype(np.float32)


def _compute_prefiltered(cube: np.ndarray, roughness: float, out_size: int, samples: int = 512) -> np.ndarray:
    """GGX-importance-sampled prefiltered env map at given roughness.

    cube: (6, S, S, 3) float32
    Returns: (6, out_size, out_size, 4) float16
    """
    pf = np.zeros((6, out_size, out_size, 3), dtype=np.float32)
    rng = np.random.default_rng(42)
    H_ts = _ggx_importance_sample(max(roughness, 0.01), samples, rng)

    for fi, dirs in enumerate(_face_directions(out_size)):
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-8)
        N_all = (dirs / norms).reshape(-1, 3)
        is_up = np.abs(N_all[:, 1]) > 0.99
        ref = np.where(is_up[:, None], np.array([[1., 0., 0.]]), np.array([[0., 1., 0.]]))
        right = np.cross(N_all, ref)
        right /= np.linalg.norm(right, axis=-1, keepdims=True).clip(min=1e-8)
        up2 = np.cross(N_all, right)
        H_world = (
            H_ts[None, :, 0:1] * right[:, None, :]
            + H_ts[None, :, 1:2] * up2[:, None, :]
            + H_ts[None, :, 2:3] * N_all[:, None, :]
        )
        N_exp = N_all[:, None, :]
        NdotH = np.clip((H_world * N_exp).sum(axis=-1, keepdims=True), 0, 1)
        L = 2 * NdotH * H_world - N_exp
        L_flat = L.reshape(-1, 3)
        colors = _sample_cube(cube, L_flat).reshape(-1, samples, 3)
        NdotL = np.clip((L * N_exp).sum(axis=-1, keepdims=True), 0, 1)
        weighted = (colors * NdotL).sum(axis=1) / NdotL.sum(axis=1).clip(min=1e-4)
        pf[fi] = weighted.reshape(out_size, out_size, 3)

    out = np.concatenate([pf, np.ones((*pf.shape[:3], 1), dtype=np.float32)], axis=-1)
    return out.astype(np.float16)


# ---------------------------------------------------------------------------
# BRDF LUT loader
# ---------------------------------------------------------------------------

def load_brdf_lut() -> np.ndarray:
    """Load pre-baked Smith-GGX BRDF LUT. Returns (512, 512, 2) float32."""
    from pathlib import Path
    path = Path(__file__).parent / "assets" / "ibl" / "brdf_lut.npy"
    return np.load(path).astype(np.float32)
