"""The path tracer: BVH build time by triangle count, and samples per second by image and mesh size."""

import numpy as np

from benchmarks.harness import case, measure

_TRACERS = {}  # one device for every case: a new one per size would time device creation


def _terrain(tris):
    """A wavy heightfield of about `tris` triangles over 20 x 20 m, under a sky."""
    from manifoldx.trace import Camera, Environment, Material, Scene

    k = max(2, int(np.sqrt(tris / 2)))
    x, z = np.meshgrid(np.linspace(-10, 10, k + 1), np.linspace(-10, 10, k + 1))
    y = 0.6 * np.sin(x * 1.3) * np.cos(z * 0.9) + 0.2 * np.sin(x * 4.1 + z * 3.7)
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    i = np.arange(k)[:, None] * (k + 1) + np.arange(k)[None, :]
    quads = np.stack([i, i + k + 1, i + k + 2, i, i + k + 2, i + 1], axis=-1).reshape(-1, 3)
    geo = {"positions": pos, "indices": quads.astype(np.uint32).ravel()}
    cam = Camera(position=(0.0, 6.0, 14.0), target=(0.0, 0.0, 0.0), fov=50.0)
    return Scene.from_meshes([(geo, Material("ground", base_color=(0.5, 0.45, 0.35), roughness=0.7))],
                             camera=cam, environment=Environment.sky((0.4, 0.6, 0.3)))


def _device():
    if "device" not in _TRACERS:
        from manifoldx.trace.tracer import _device as make

        _TRACERS["device"] = make()
    return _TRACERS["device"]


def _spp(scene, w, h):
    from manifoldx.trace import Tracer

    t = Tracer(scene, w, h, device=_device())
    t.step(2)  # compile and warm up
    ms = measure(lambda: t.step(4), repeat=3) / 4
    return {"ms": ms, "spp_s": round(1e3 / ms, 1), "Msamples_s": round(w * h / ms / 1e3, 1)}


@case("trace", "bvh_build", sizes=[10_000, 100_000, 1_000_000], smoke=500)
def bvh_build(n):
    """Binned SAH build over a heightfield of n triangles (numpy, CPU)."""
    from manifoldx.trace.bvh import build

    p = _terrain(n).positions
    return measure(lambda: build(p), repeat=1, number=1)


@case("trace", "cornell_spp", sizes=[256, 512, 1024], smoke=16)
def cornell_spp(n):
    """One sample per pixel of the Cornell box at n x n; ms per sample per pixel."""
    from manifoldx.trace.scenes import cornell_box

    return _spp(cornell_box(), n, n)


@case("trace", "mesh_spp", sizes=[10_000, 100_000, 1_000_000], smoke=200)
def mesh_spp(n):
    """One sample per pixel at 1024 x 640 of a sunlit heightfield of n triangles."""
    w, h = (1024, 640) if n > 1000 else (32, 20)
    return _spp(_terrain(n), w, h)
