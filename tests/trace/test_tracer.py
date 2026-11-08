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


def render(name, spp, seed=1, device=None):
    t = Tracer(getattr(scenes, name)(), SIZE, SIZE, seed=seed, device=device)
    t.step(spp)
    return t.radiance()


@pytest.mark.parametrize("name", ["cornell_box", "glass_sphere", "window_room"])
def test_reference_scene(name, device):
    ref = np.load(REFS / f"{name}.npy").astype(np.float32)
    img = render(name, 64, device=device)
    err = np.abs(img - ref).mean() / ref.mean()
    limit = float((REFS / f"{name}.limit").read_text())
    assert err < limit, f"{name}: mean error {err:.4f} over {limit:.4f}"
