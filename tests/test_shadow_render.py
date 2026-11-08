"""GPU-gated render tests for directional sun + shadow mapping.

Gate on the offscreen wgpu backend (skip if unavailable) so the suite stays
runnable headless. Pattern mirrors tests/test_textured_material.py.
"""

import numpy as np
import pytest

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import DirectionalLight, SpotLight, StandardMaterial, sphere, plane


def _make_offscreen_engine(name="shadow-test", w=128, h=128):
    try:
        from manifoldx.backends import get_offscreen_canvas

        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx

    engine = mx.Engine(name, width=w, height=h)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def _render_array(engine):
    engine._draw_frame()
    return engine._render_canvas.draw()  # HxWx4 uint8


# --------------------------------------------------------------------------
# Task 1 — directional sun
# --------------------------------------------------------------------------


def test_set_sun_stores_light():
    eng = _make_offscreen_engine()
    sun = DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.5, -1.0, -0.3))
    eng.set_sun(sun)
    assert eng._sun is sun


def test_globals_buffer_is_560_bytes():
    eng = _make_offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, -1, 0)))
    eng.spawn(Mesh(sphere(1.0, 24)), Material(StandardMaterial(color="#ffffff", roughness=0.5)))
    _render_array(eng)
    assert eng._render_pipeline._globals_buffer.size == 560


def test_sun_lights_sphere_with_gradient():
    eng = _make_offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-1, -1, -1)))
    eng.spawn(
        Mesh(sphere(1.0, 32)),
        Material(StandardMaterial(color="#ffffff", roughness=0.5)),
        Transform(pos=(0, 0, 0)),
    )
    eng.camera.fit(radius=3.0, center=(0, 0, 0), azimuth=0, elevation=10)
    img = _render_array(eng)
    lum = img[..., :3].mean(axis=2)
    lit = lum[lum > 12]  # ignore dark-blue background
    assert lit.size > 0
    assert lit.max() - lit.min() > 20  # a Lambert gradient, not flat


# --------------------------------------------------------------------------
# Task 3 — shadow map resources
# --------------------------------------------------------------------------


def test_enable_shadows_stores_config():
    eng = _make_offscreen_engine()
    eng.enable_shadows(target=(0, 0, 0), extent=8.0, resolution=1024, bias=0.004)
    cfg = eng._shadow_config
    assert cfg["extent"] == 8.0 and cfg["resolution"] == 1024 and cfg["bias"] == 0.004


def test_shadow_map_allocated_when_enabled():
    eng = _make_offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, -1, 0)))
    eng.enable_shadows(target=(0, 0, 0), extent=10.0, resolution=512)
    eng.spawn(Mesh(plane(12, 12)), Material(StandardMaterial(color="#ffffff", roughness=0.8)))
    eng.spawn(
        Mesh(sphere(1.0, 16)), Material(StandardMaterial(color="#ffffff")), Transform(pos=(0, 2, 0))
    )
    _render_array(eng)
    tex = eng._render_pipeline._shadow_map
    assert tex is not None
    assert tex.size[0] == 512 and tex.size[1] == 512


# --------------------------------------------------------------------------
# Task 4 — cast shadow onto the ground
# --------------------------------------------------------------------------

# -90deg about X (quaternion x,y,z,w): turns plane()'s +Z normal into +Y (a floor).
_FLOOR_ROT = (-0.70710678, 0.0, 0.0, 0.70710678)


def _plane_sphere_scene(shadows, pcf=1):
    eng = _make_offscreen_engine(w=192, h=192)
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.3)))
    if shadows:
        eng.enable_shadows(
            target=(0, 0, 0), extent=6.0, resolution=1024, bias=0.004, pcf_radius=pcf
        )
    eng.spawn(
        Mesh(plane(12, 12)),
        Material(StandardMaterial(color="#ffffff", roughness=0.9)),
        Transform(pos=(0, 0, 0), rot=_FLOOR_ROT),
    )
    eng.spawn(
        Mesh(sphere(1.0, 32)),
        Material(StandardMaterial(color="#ffffff", roughness=0.5)),
        Transform(pos=(0, 2, 0)),
    )
    eng.camera.fit(radius=6.0, center=(0, 0, 0), azimuth=25, elevation=55)
    return _render_array(eng)


def test_sphere_casts_shadow_on_plane():
    lit = _plane_sphere_scene(shadows=False)[..., :3].mean(axis=2).astype(np.float64)
    shadowed = _plane_sphere_scene(shadows=True)[..., :3].mean(axis=2).astype(np.float64)
    # A clearly darkened region (the cast shadow) must appear when shadows are on.
    darkening = lit - shadowed
    assert darkening.max() > 20, f"no shadow appeared (max darkening {darkening.max():.1f})"


# --------------------------------------------------------------------------
# VS2 — PCF soft shadows
# --------------------------------------------------------------------------


def test_enable_shadows_stores_pcf_radius():
    eng = _make_offscreen_engine()
    eng.enable_shadows(target=(0, 0, 0), extent=6.0, pcf_radius=3)
    assert eng._shadow_config["pcf_radius"] == 3
    # Default is a 3x3 soft kernel.
    eng.enable_shadows(target=(0, 0, 0), extent=6.0)
    assert eng._shadow_config["pcf_radius"] == 1


def _penumbra_count(lit, shadowed):
    """Pixels in the soft transition band (partial darkening) of the shadow."""
    d = lit - shadowed
    mx = d.max()
    if mx < 20:
        return 0
    return int(((d > 0.2 * mx) & (d < 0.8 * mx)).sum())


def test_pcf_widens_the_penumbra():
    lit = _plane_sphere_scene(shadows=False)[..., :3].mean(axis=2).astype(np.float64)
    hard = _plane_sphere_scene(shadows=True, pcf=0)[..., :3].mean(axis=2).astype(np.float64)
    soft = _plane_sphere_scene(shadows=True, pcf=4)[..., :3].mean(axis=2).astype(np.float64)
    hard_band = _penumbra_count(lit, hard)
    soft_band = _penumbra_count(lit, soft)
    # A wider PCF kernel must produce a wider partial-shadow (penumbra) band.
    assert soft_band > hard_band, f"pcf did not soften edges (hard={hard_band}, soft={soft_band})"


# --------------------------------------------------------------------------
# Block 1 A — auto-fit frustum
# --------------------------------------------------------------------------


def test_enable_shadows_stores_auto_fit():
    eng = _make_offscreen_engine()
    eng.enable_shadows()
    assert eng._shadow_config["auto_fit"] is True
    eng.enable_shadows(auto_fit=False)
    assert eng._shadow_config["auto_fit"] is False


def test_scene_bounds_matches_known_scene():
    eng = _make_offscreen_engine()
    eng.spawn(
        Mesh(sphere(1.0, 16)), Material(StandardMaterial(color="#fff")), Transform(pos=(-3, 0, 0))
    )
    eng.spawn(
        Mesh(sphere(1.0, 16)), Material(StandardMaterial(color="#fff")), Transform(pos=(3, 0, 0))
    )
    center, radius = eng._render_pipeline._scene_bounds(eng)
    np.testing.assert_allclose(center, [0, 0, 0], atol=1e-4)
    # World AABB x in [-4, 4], y,z in [-1, 1] -> half-diagonal = sqrt(64+4+4)/2.
    assert abs(radius - (np.sqrt(72.0) / 2)) < 0.1


def test_set_geometry_moves_the_auto_fit_bounds():
    eng = _make_offscreen_engine()
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.3)))
    eng.enable_shadows(auto_fit=True)
    geo = sphere(1.0, 16)
    h = eng.spawn(Mesh(geo), Material(StandardMaterial(color="#ffffff")), Transform())
    _render_array(eng)  # the vertex buffer exists, so set_geometry rewrites it
    eng._render_pipeline._scene_bounds(eng)  # warm the per-geometry local-bounds cache

    moved = {**geo, "positions": np.asarray(geo["positions"], np.float32) + np.float32([8, 0, 0])}
    h.set_geometry(moved)
    center, radius = eng._render_pipeline._scene_bounds(eng)

    np.testing.assert_allclose(center, [8, 0, 0], atol=0.2)
    dist = np.linalg.norm(moved["positions"] - center, axis=1)
    assert dist.max() <= radius + 1e-4  # the fitted sphere contains the new vertices


def test_auto_fit_casts_shadow_without_manual_extent():
    # No target/extent given — auto-fit must produce a working frustum on its own.
    eng = _make_offscreen_engine(w=192, h=192)
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.3)))
    eng.enable_shadows()  # auto_fit=True, no extent/target
    eng.spawn(
        Mesh(plane(12, 12)),
        Material(StandardMaterial(color="#ffffff", roughness=0.9)),
        Transform(pos=(0, 0, 0), rot=_FLOOR_ROT),
    )
    eng.spawn(
        Mesh(sphere(1.0, 32)),
        Material(StandardMaterial(color="#ffffff", roughness=0.5)),
        Transform(pos=(0, 2, 0)),
    )
    eng.camera.fit(radius=6.0, center=(0, 0, 0), azimuth=25, elevation=55)
    lit = eng._render_pipeline
    off = _make_offscreen_engine(w=192, h=192)
    off.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.3)))
    off.spawn(
        Mesh(plane(12, 12)),
        Material(StandardMaterial(color="#ffffff", roughness=0.9)),
        Transform(pos=(0, 0, 0), rot=_FLOOR_ROT),
    )
    off.spawn(
        Mesh(sphere(1.0, 32)),
        Material(StandardMaterial(color="#ffffff", roughness=0.5)),
        Transform(pos=(0, 2, 0)),
    )
    off.camera.fit(radius=6.0, center=(0, 0, 0), azimuth=25, elevation=55)
    on_img = _render_array(eng)[..., :3].mean(axis=2).astype(np.float64)
    off_img = _render_array(off)[..., :3].mean(axis=2).astype(np.float64)
    assert (off_img - on_img).max() > 20  # a cast shadow appears under auto-fit


# --------------------------------------------------------------------------
# Block 1 B — slope-scaled bias (no self-shadow acne)
# --------------------------------------------------------------------------


def _lone_floor(shadows):
    eng = _make_offscreen_engine(w=160, h=160)
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.8, -0.5, -0.3)))
    if shadows:
        eng.enable_shadows(bias=0.002)  # small bias — slope-scaling must carry it
    eng.spawn(
        Mesh(plane(14, 14)),
        Material(StandardMaterial(color="#ffffff", roughness=0.9)),
        Transform(pos=(0, 0, 0), rot=_FLOOR_ROT),
    )
    eng.camera.fit(radius=7.0, center=(0, 0, 0), azimuth=20, elevation=35)
    return _render_array(eng)[..., :3].mean(axis=2).astype(np.float64)


def test_grazing_floor_has_no_self_shadow_acne():
    # A lone floor under a raking sun must not self-shadow: with slope-scaled
    # bias the shadowed render matches the unshadowed one across the floor.
    off = _lone_floor(shadows=False)
    on = _lone_floor(shadows=True)
    mask = off > 12  # floor pixels (ignore background)
    darkening = (off - on)[mask]
    # No meaningful acne: the floor stays lit (allow a couple of edge texels).
    assert darkening.mean() < 3.0, (
        f"self-shadow acne on floor (mean darkening {darkening.mean():.1f})"
    )


# --------------------------------------------------------------------------
# Block 1 D — spot light (flashlight) with shadow
# --------------------------------------------------------------------------


def test_set_spot_stores_light():
    eng = _make_offscreen_engine()
    spot = SpotLight(
        color="#ffffff",
        intensity=100.0,
        position=(0, 5, 0),
        direction=(0, -1, 0),
        inner_angle=0.3,
        outer_angle=0.5,
    )
    eng.set_spot(spot)
    assert eng._spot is spot


def test_spot_cone_center_brighter_than_outside():
    # A spot straight down over a floor: the cone centre is lit, outside is dark.
    eng = _make_offscreen_engine(w=192, h=192)
    eng.set_spot(
        SpotLight(
            color="#ffffff",
            intensity=140.0,
            position=(0, 6, 0),
            direction=(0, -1, 0),
            inner_angle=0.22,
            outer_angle=0.34,
        )
    )
    eng.spawn(
        Mesh(plane(20, 20)),
        Material(StandardMaterial(color="#ffffff", roughness=0.95)),
        Transform(pos=(0, 0, 0), rot=_FLOOR_ROT),
    )
    eng.camera.fit(radius=8.0, center=(0, 0, 0), azimuth=0, elevation=80)
    img = _render_array(eng)[..., :3].mean(axis=2).astype(np.float64)
    h, w = img.shape
    center = img[h // 2 - 8 : h // 2 + 8, w // 2 - 8 : w // 2 + 8].mean()
    corner = img[:16, :16].mean()  # outside the cone
    assert center > corner + 40, f"no flashlight cone (center {center:.0f}, corner {corner:.0f})"


def test_spot_casts_shadow():
    def scene(shadows):
        eng = _make_offscreen_engine(w=192, h=192)
        eng.set_spot(
            SpotLight(
                color="#ffffff",
                intensity=160.0,
                position=(2.5, 5.0, 2.5),
                direction=(-0.4, -1.0, -0.4),
                inner_angle=0.5,
                outer_angle=0.7,
            )
        )
        if shadows:
            eng.enable_shadows(resolution=2048, near=0.2, bias=0.003)
        eng.spawn(
            Mesh(plane(20, 20)),
            Material(StandardMaterial(color="#ffffff", roughness=0.95)),
            Transform(pos=(0, 0, 0), rot=_FLOOR_ROT),
        )
        eng.spawn(
            Mesh(sphere(1.0, 32)),
            Material(StandardMaterial(color="#ffffff", roughness=0.5)),
            Transform(pos=(0, 1.2, 0)),
        )
        eng.camera.fit(radius=8.0, center=(0, 0, 0), azimuth=20, elevation=45)
        return _render_array(eng)[..., :3].mean(axis=2).astype(np.float64)

    lit = scene(shadows=False)
    shadowed = scene(shadows=True)
    assert (lit - shadowed).max() > 20  # the spot casts a visible shadow


def test_shadows_fade_out_before_the_edge_of_the_map():
    """A roof shadows the whole floor; inside the map's outer 15% the shadow thins out,
    so the map's edge never shows as a line on screen."""
    from manifoldx.resources import cube

    eng = _make_offscreen_engine(w=160, h=160)
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(0, -1, 0)))
    eng.enable_shadows(target=(0, 0, 0), extent=4.0, resolution=1024, pcf_radius=0, auto_fit=False, far=30.0)
    eng.spawn(Mesh(plane(30, 30)), Material(StandardMaterial(color="#ffffff", roughness=1.0)),
              Transform(pos=(0, 0, 0), rot=Transform.rotation(x=-np.pi / 2, y=0, z=0)))
    eng.spawn(Mesh(cube(30, 0.1, 30)), Material(StandardMaterial(color="#ffffff")), Transform(pos=(0, 3, 0)))
    eng.camera.set_pose(position=(0, 2.5, 0.001), target=(0, 0, 0))   # under the roof, looking down
    eng.camera.fov = 140
    lum = _render_array(eng)[..., :3].mean(axis=2)
    pts = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [6.0, 0.0, 0.0]])
    px, _ = eng.camera.project(pts, eng.w, eng.h)
    centre, rim, out = (lum[int(round(y)), int(round(x))] for x, y in px)
    assert out > centre + 30           # outside the map: unshadowed
    assert rim > centre + 0.5 * (out - centre)   # 95% of the way out: mostly faded
