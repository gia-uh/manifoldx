"""Baked images: a StandardMaterial(baked=) draws the SURFACE_BAKED vertices of each entity from its Baked
layer of a texture array; QuadStreamer gives each node it draws with an image a layer of its own."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Baked, Fade, Material, Mesh, Transform
from manifoldx.resources import SURFACE_BAKED, StandardMaterial, baked_columns, pack_vertices
from manifoldx.streaming import MeshData, QuadStreamer
from manifoldx.textures import layer_mips, texture_array, write_layer


def _offscreen(w=64, h=64):
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=w, height=h)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    eng = mx.Engine("t", width=w, height=h)
    eng._init_canvas(canvas)
    eng._running = True
    eng.camera.set_pose(position=(0.5, 0.5, 3), target=(0.5, 0.5, 0))
    eng.camera.fov = 30
    return eng, canvas


def _square(surface=SURFACE_BAKED, colour=(0.0, 1.0, 0.0)):
    """A unit square facing the camera, its uv its (x, y)."""
    p = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], np.float32)
    return {"positions": p, "normals": np.tile((0, 0, 1), (4, 1)).astype(np.float32),
            "colors": np.tile(np.float32(colour), (4, 1)), "surfaces": np.full(4, surface, np.int32),
            "baked_uv": p[:, :2], "indices": np.array([0, 1, 2, 0, 2, 3], np.uint32)}


def _halves():
    """A 16x16 image: red on the left half (u < 0.5), blue on the right."""
    img = np.zeros((16, 16, 3), np.uint8)
    img[:, :8, 0] = 255
    img[:, 8:, 2] = 255
    return img


def test_baked_vertices_carry_their_uv_and_their_own_colour():
    g = _square(colour=(1.0, 0.5, 0.0))
    g["surfaces"] = np.array([SURFACE_BAKED, SURFACE_BAKED, 0, 0], np.int32)
    data, info = pack_vertices(g)
    rows = np.frombuffer(data, np.float32).reshape(4, -1)
    assert np.array_equal(rows[:2, 6:8], g["baked_uv"][:2])
    assert np.array_equal(rows[:2, 8], [255 * 65536 + 128 * 256, 255 * 65536 + 128 * 256])  # sRGB, packed
    plain, _ = pack_vertices({k: v for k, v in g.items() if k != "baked_uv"})
    assert np.array_equal(rows[2:], np.frombuffer(plain, np.float32).reshape(4, -1)[2:])  # the rest as before
    assert baked_columns(np.ones((1, 3)), [[0.25, 0.75]])[0, 2] == 0xFFFFFF


def test_layer_mips_halve_down_to_one_pixel():
    levels = layer_mips(_halves())
    assert [lv.shape for lv in levels] == [(16, 16, 4), (8, 8, 4), (4, 4, 4), (2, 2, 4), (1, 1, 4)]
    with pytest.raises(ValueError):
        layer_mips(np.zeros((4, 8, 3), np.uint8))


def test_baked_needs_vertex_colours_and_an_array():
    arr = SimpleNamespace(layers=4)
    with pytest.raises(ValueError):
        StandardMaterial("#fff", baked=arr)
    with pytest.raises(ValueError):
        StandardMaterial("#fff", vertex_colors=True, tinted=True, baked=arr)
    with pytest.raises(ValueError):
        StandardMaterial("#fff", vertex_colors=True, baked=SimpleNamespace(layers=0))
    m = StandardMaterial("#fff", vertex_colors=True, baked=arr, detail=True, faded=True)
    assert m.pipeline_subtype == "vcolor-baked-detail-fade"


def _colours(canvas):
    f = np.asarray(canvas.draw()).astype(int)[:, :, :3]
    return f[32, 16], f[32, 48]  # left and right of the square's middle row


@pytest.mark.parametrize("kw", [{}, {"detail": True}, {"faded": True}, {"detail": True, "faded": True}])
def test_an_entity_draws_its_layer_and_without_one_its_own_colour(kw):
    eng, canvas = _offscreen()
    arr = texture_array(eng, layers=2, size=16)
    write_layer(eng, arr, 1, layer_mips(_halves()))
    mat = StandardMaterial("#ffffff", roughness=1.0, vertex_colors=True, baked=arr, **kw)
    h = eng.spawn(Mesh(_square()), Material(mat), Transform(), Baked(layer=1.0),
                  *([Fade(value=1.0)] if kw.get("faded") else []))
    eng._draw_frame()
    left, right = _colours(canvas)
    assert left[0] > 3 * max(left[1], left[2]) and right[2] > 3 * max(right[0], right[1])  # red | blue
    eng.store._components["Baked"][h.index] = -1.0
    eng._draw_frame()
    left, right = _colours(canvas)
    assert left[1] > 3 * max(left[0], left[2]) and right[1] > 3 * max(right[0], right[2])  # its own green


def test_other_surfaces_ignore_the_image():
    eng, canvas = _offscreen()
    arr = texture_array(eng, layers=1, size=16)
    write_layer(eng, arr, 0, layer_mips(_halves()))
    mat = StandardMaterial("#ffffff", roughness=1.0, vertex_colors=True, baked=arr)
    eng.spawn(Mesh(_square(surface=0)), Material(mat), Transform(), Baked(layer=0.0))
    eng._draw_frame()
    left, right = _colours(canvas)
    assert left[1] > 3 * max(left[0], left[2]) and right[1] > 3 * max(right[0], right[2])


CELL = 200.0


def _node(L, a, b):
    pos = np.array([[0, 0, 0], [CELL, 0, 0], [0, 0, CELL]], np.float32) * 2**L
    return MeshData(pos, np.tile([0, 1, 0], (3, 1)).astype(np.float32), np.full((3, 3), 0.5, np.float32),
                    np.array([0, 2, 1], np.uint32), surfaces=np.full(3, SURFACE_BAKED, np.int32),
                    albedo=np.zeros((8, 8, 3), np.uint8) if L == 1 else None)


def test_the_streamer_gives_each_drawn_node_with_an_image_a_layer_and_takes_it_back(monkeypatch):
    written = []
    monkeypatch.setattr("manifoldx.textures.write_layer", lambda eng, h, layer, levels: written.append(layer))
    arr = SimpleNamespace(layers=1)
    mat = StandardMaterial("#ffffff", vertex_colors=True, faded=True, baked=arr)
    engine = mx.Engine("quad", width=64, height=600)
    engine.camera.far = 15000.0
    s = QuadStreamer(engine, CELL, (0, 0), 2, lambda L, a, b: 0 <= a < 1 << (2 - L) and 0 <= b < 1 << (2 - L),
                     _node, lambda L: CELL * 2**L / 16, refine_ms=None, crossfade=0.05, material=mat,
                     workers="thread")

    def layers():
        return {n: engine.store._components["Baked"][h.index][0] for n, h in s._drawn.items() if h is not None}

    def settle(h):
        for _ in range(400):
            s.focus(200.0, 200.0, h)
            if s.settled:
                return
            time.sleep(0.005)

    settle(2600.0)  # far enough for level 1: four nodes with images, one layer
    got = layers()
    assert {n[0] for n in got} == {1} and sorted(got.values()) == [-1.0, -1.0, -1.0, 0.0]
    assert written == [0] and s.stats()["unbaked"] == 3
    settle(8000.0)  # the root, with no image: the level-1 nodes fade out and give the layer back
    assert set(layers()) == {(2, 0, 0)} and layers()[(2, 0, 0)] == -1.0
    assert s._free_layers == [0] and not s._layers
    settle(2600.0)
    assert sorted(layers().values()) == [-1.0, -1.0, -1.0, 0.0] and written == [0, 0]
    s.close()
