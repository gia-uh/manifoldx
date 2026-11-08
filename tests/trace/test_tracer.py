"""The kernels against analytic answers and stored references (64x64, fixed seed)."""
from pathlib import Path

import numpy as np
import pytest

from manifoldx.trace import scenes
from manifoldx.trace.tracer import Tracer

REFS = Path(__file__).parent / "refs"
SIZE = 48


def test_primary_rays_see_a_plane_at_its_distance(device):
    t = Tracer(scenes.lit_plane(), 16, 16, device=device)
    a = t.aovs()
    # camera 3 m above the plane, looking straight down
    assert abs(a["depth"][8, 8] - 3.0) < 0.05
    assert np.allclose(a["normal"][8, 8], (0, 1, 0), atol=1e-4)
    assert np.allclose(a["albedo"][8, 8], 0.5, atol=1e-6)


def test_white_furnace_conserves_energy(device):
    t = Tracer(scenes.furnace(1.0), 16, 16, max_bounces=16, diffuse_only=True, device=device)
    t.step(64)
    assert abs(t.radiance()[6:10, 6:10].mean() - 1.0) < 0.01


def test_lambert_plane_under_a_directional_light(device):
    t = Tracer(scenes.lit_plane(60, 0.5), 8, 8, diffuse_only=True, device=device)
    t.step(4)
    want = 0.5 * np.sin(np.radians(60)) / np.pi
    assert np.allclose(t.radiance(), want, rtol=1e-3)


def test_the_same_seed_renders_the_same_image(device):
    a = Tracer(scenes.cornell_box(), 16, 16, seed=3, device=device)
    b = Tracer(scenes.cornell_box(), 16, 16, seed=3, device=device)
    a.step(2), b.step(2)
    assert np.array_equal(a.radiance(), b.radiance())


def error(img, ref, block=6):
    """Mean relative error of block averages: noise averages out, a bias does not."""
    k = SIZE // block
    a = img.reshape(k, block, k, block, 3).mean(axis=(1, 3))
    b = ref.reshape(k, block, k, block, 3).mean(axis=(1, 3))
    return float(np.abs(a - b).mean() / b.mean())


def render(name, spp, seed=1, device=None):
    t = Tracer(getattr(scenes, name)(), SIZE, SIZE, seed=seed, device=device)
    t.step(spp)
    return t.radiance()


@pytest.mark.parametrize("name", ["cornell_box", "glass_sphere", "window_room"])
def test_reference_scene(name, device):
    ref = np.load(REFS / f"{name}.npy").astype(np.float32)
    img = render(name, 64, device=device)
    err = error(img, ref)
    limit = float((REFS / f"{name}.limit").read_text())
    assert err < limit, f"{name}: mean error {err:.4f} over {limit:.4f}"


@pytest.mark.parametrize("name", ["window_room"])  # at 48 px the Cornell box is too small for 16 px steps
def test_the_denoiser_lowers_the_error(name, device):
    ref = np.load(REFS / f"{name}.npy").astype(np.float32)
    t = Tracer(getattr(scenes, name)(), SIZE, SIZE, seed=5, device=device)
    t.step(16)
    raw = np.abs(t.radiance() - ref).mean()
    den = np.abs(t.denoised() - ref).mean()
    assert den < 0.6 * raw, f"denoised {den:.4f}, raw {raw:.4f}"


def test_every_kernel_fits_webgpus_default_limits(device):
    """A device with no raised limits has 8 storage buffers per stage, as a browser's does."""
    adapter = device.adapter
    plain = adapter.request_device_sync(required_limits={"max-storage-buffers-per-shader-stage": 8})
    assert plain.limits["max-storage-buffers-per-shader-stage"] == 8
    t = Tracer(scenes.window_room(), 16, 16, device=plain)
    t.aovs()
    t.step(2)
    assert np.isfinite(t.denoised()).all()


def test_the_denoiser_prep_matches_numpy(device):
    t = Tracer(scenes.cornell_box(), 24, 16, device=device)
    t.step(4)
    src, guide = t.denoise_inputs()
    rad, g = t.radiance(), t.guides()
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    albedo = np.maximum(g["albedo"], 0.02)
    lum_a = np.maximum(albedo @ w, 0.02)
    var = np.maximum(g["moment2"] - (rad @ w) ** 2, 0.0) / g["samples"] / lum_a ** 2
    p = np.pad(var, 1, mode="edge")
    k = (1.0, 2.0, 1.0)
    rows = sum(k[i] * p[i: i + var.shape[0], :] for i in range(3)) / 4
    var = sum(k[j] * rows[:, j: j + var.shape[1]] for j in range(3)) / 4
    normal = g["normal"] / np.maximum(np.linalg.norm(g["normal"], axis=2, keepdims=True), 1e-6)
    assert np.allclose(src[..., :3], rad / albedo, rtol=1e-5, atol=1e-6)
    assert np.allclose(src[..., 3], var, rtol=1e-4, atol=1e-7)
    assert np.allclose(guide[..., :3], normal, atol=1e-5)
    assert np.allclose(guide[..., 3], g["depth"], rtol=1e-5)
