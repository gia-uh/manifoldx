"""Procedural detail: StandardMaterial(detail=True) paints asphalt and ground surfaces from world-space noise."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import (
    DETAIL_ASPHALT,
    DETAIL_GROUND,
    DirectionalLight,
    StandardMaterial,
    detail_shader,
    fade_shader,
    tint_shader,
)

GRASS = (0.30, 0.44, 0.20)
ROAD = (0.30, 0.30, 0.32)  # lighter than a real road, so a test reads its tones
FIELD = dict(eye=(0, 2, 0), target=(0, 0, -8), side=128)  # down a field, from a driver's height


def _ground(sid, size=12.0, color=GRASS):
    h = size / 2
    return {
        "positions": np.array([[-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h]], np.float32),
        "normals": np.tile([0, 1, 0], (4, 1)).astype(np.float32),
        "colors": np.tile(color, (4, 1)).astype(np.float32),
        "indices": np.array([0, 2, 1, 0, 3, 2], np.uint32),
        "surfaces": np.full(4, sid, np.int32),
    }


def _engine(side=96):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=side, height=side)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=side, height=side)
    eng._init_canvas(canvas)
    eng._running = True
    eng.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.3, -1.0, -0.4)))
    return eng, canvas


def _render(geo, detail=True, eye=(0, 3, 4), target=(0, 0, 0), at=(0.0, 0.0), origin=None, side=96):
    eng, canvas = _engine(side)
    mat = StandardMaterial("#ffffff", vertex_colors=True, detail=detail)
    eng.spawn(Mesh(geo), Material(mat), Transform(pos=(at[0], 0, at[1])))
    eng.camera.set_pose(position=(at[0] + eye[0], eye[1], at[1] + eye[2]),
                        target=(at[0] + target[0], target[1], at[1] + target[2]))
    if origin is not None:
        eng.set_origin(*origin)
    eng._draw_frame()
    return np.asarray(canvas.draw())[..., :3].astype(int), eng


def test_detail_is_its_own_pipeline_subtype():
    assert StandardMaterial("#fff", vertex_colors=True, detail=True).pipeline_subtype == "vcolor-detail"
    assert StandardMaterial("#fff", vertex_colors=True, tinted=True, detail=True,
                            faded=True).pipeline_subtype == "vcolor-tint-detail-fade"
    assert StandardMaterial("#fff", vertex_colors=True).pipeline_subtype == "vcolor"


def test_detail_needs_vertex_colours():
    with pytest.raises(ValueError):
        StandardMaterial("#ffffff", detail=True)


def test_the_shader_is_unchanged_without_detail():
    plain = StandardMaterial._compile(vertex_colors=True)
    assert "detail_surface" not in plain and "origin_tile" not in plain
    _, eng = _render(_ground(DETAIL_GROUND), detail=False)
    sources = list(eng._render_pipeline._shader_sources.values())
    assert plain in sources
    assert not any("detail_surface" in s for s in sources)


@pytest.mark.parametrize("variant", ["vcolor", "facade", "tint", "fade", "tint-fade"])
def test_the_detail_shader_compiles(variant):
    eng, _ = _engine(16)
    src = StandardMaterial._compile(vertex_colors=True, facade=variant == "facade")
    if "tint" in variant:
        src = tint_shader(src)
    src = detail_shader(src)
    if "fade" in variant:
        src = fade_shader(src)
    assert "detail_surface(vertex_albedo" in src and "detail_rough" in src
    eng._device.create_shader_module(code=src)


def test_without_detail_its_surface_ids_draw_flat():
    a, _ = _render(_ground(DETAIL_GROUND), detail=False)
    b, _ = _render(_ground(0), detail=False)
    assert np.abs(a - b).max() <= 1


@pytest.mark.parametrize("sid,color", [(DETAIL_GROUND, GRASS), (DETAIL_ASPHALT, ROAD)])
def test_detail_varies_the_colour_round_the_vertex_colour(sid, color):
    flat, _ = _render(_ground(0, size=200.0, color=color), **FIELD)
    rich, _ = _render(_ground(sid, size=200.0, color=color), **FIELD)
    flat, rich = flat[70:], rich[70:]
    assert flat.std(axis=(0, 1)).max() < 3
    assert np.abs(rich - flat).mean() > 2  # blotches and grain
    assert np.abs(flat.mean(axis=(0, 1)) - rich.mean(axis=(0, 1))).max() < 30


def test_only_the_detail_ids_change():
    a, _ = _render(_ground(0))
    b, _ = _render(_ground(0), detail=False)
    assert np.abs(a - b).max() <= 1


def test_far_away_the_detail_is_gone():
    far = dict(eye=(0, 60, 700))  # the plane 700 m off: past every fade
    rich, _ = _render(_ground(DETAIL_GROUND, size=600.0), **far)
    flat, _ = _render(_ground(0, size=600.0), **far)
    assert np.abs(rich - flat).max() <= 2


def test_the_noise_stays_put_when_the_origin_moves():
    """The same world spot looks the same whatever the floating origin, so the noise has no seams or repeats."""
    at = (5000.3, -3100.7)
    near, _ = _render(_ground(DETAIL_GROUND, size=200.0), at=at, **FIELD)
    moved, _ = _render(_ground(DETAIL_GROUND, size=200.0), at=at, origin=(4900.0, -3000.0), **FIELD)
    flat, _ = _render(_ground(0, size=200.0), at=at, **FIELD)
    assert np.abs(near - flat)[70:].mean() > 2
    assert np.abs(near - moved).max() <= 3
