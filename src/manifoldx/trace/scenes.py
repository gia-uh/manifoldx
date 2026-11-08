"""Reference scenes built in code: the Cornell box, a glass sphere, a room with a window."""

from __future__ import annotations

import numpy as np

from manifoldx.trace.environment import Environment
from manifoldx.trace.scene import Camera, Light, Material, Scene

WHITE = Material("white", base_color=(0.73, 0.73, 0.73))
RED = Material("red", base_color=(0.63, 0.065, 0.05))
GREEN = Material("green", base_color=(0.14, 0.45, 0.091))
LAMP = Material("lamp", base_color=(0.78, 0.78, 0.78), emission=(17.0, 12.0, 4.0))
GLASS = Material("glass", base_color=(1.0, 1.0, 1.0), roughness=0.0, transmission=1.0, ior=1.5, thin=False)
BLACK = Environment(np.zeros((4, 8, 3), np.float32))


def quad(a, b, c, d):
    """Two triangles a-b-c, a-c-d; the normal follows the right-hand rule."""
    return {"positions": np.array([a, b, c, d], np.float32), "indices": np.array([0, 1, 2, 0, 2, 3], np.uint32)}


def box(center, size, angle=0.0):
    """An axis-aligned box turned by angle radians about +y, outward normals, flat faces."""
    sx, sy, sz = np.asarray(size, np.float64) / 2
    corners = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    c, s = np.cos(angle), np.sin(angle)
    corners = corners @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]) + center
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    pos = np.concatenate([corners[list(f)] for f in faces]).astype(np.float32)
    idx = np.concatenate([[4 * i, 4 * i + 1, 4 * i + 2, 4 * i, 4 * i + 2, 4 * i + 3] for i in range(6)])
    return {"positions": pos, "indices": idx.astype(np.uint32)}


def sphere(center, radius, subdivisions=4):
    from manifoldx.modeling.primitives import icosphere

    geo = icosphere(subdivisions, radius).to_geometry()
    pos = np.asarray(geo["positions"], np.float32)
    return {"positions": pos + np.asarray(center, np.float32), "normals": pos / radius,
            "indices": np.asarray(geo["indices"], np.uint32)}


def _shell(red=RED, green=GREEN, white=WHITE):
    return [
        (quad((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)), white, "floor"),
        (quad((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)), white, "ceiling"),
        (quad((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)), white, "back"),
        (quad((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)), red, "left"),
        (quad((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)), green, "right"),
    ]


def _camera():
    return Camera(position=(0.5, 0.5, -1.35), target=(0.5, 0.5, 0.0), fov=40.0)


def cornell_box() -> Scene:
    """The unit Cornell box with its two blocks and a ceiling lamp; no light from outside."""
    lamp = quad((0.37, 0.999, 0.4), (0.63, 0.999, 0.4), (0.63, 0.999, 0.62), (0.37, 0.999, 0.62))
    meshes = _shell() + [
        (lamp, LAMP, "lamp"),
        (box((0.33, 0.3, 0.62), (0.29, 0.6, 0.29), np.radians(-17)), WHITE, "tall"),
        (box((0.67, 0.15, 0.36), (0.29, 0.3, 0.29), np.radians(18)), WHITE, "short"),
    ]
    return Scene.from_meshes(meshes, camera=_camera(), environment=BLACK)


def glass_sphere() -> Scene:
    """A solid glass sphere and a rough gold one in the Cornell box."""
    lamp = quad((0.37, 0.999, 0.4), (0.63, 0.999, 0.4), (0.63, 0.999, 0.62), (0.37, 0.999, 0.62))
    gold = Material("gold", base_color=(1.0, 0.71, 0.29), metallic=1.0, roughness=0.3)
    meshes = _shell() + [
        (lamp, LAMP, "lamp"),
        (sphere((0.3, 0.2, 0.55), 0.2), GLASS, "glass"),
        (sphere((0.72, 0.16, 0.4), 0.16), gold, "gold"),
    ]
    return Scene.from_meshes(meshes, camera=_camera(), environment=BLACK)


def window_room(sun_direction=(0.5, 0.55, -0.65)) -> Scene:
    """A closed room, 4 × 2.6 × 4 m, with a glazed window in the wall at z = 0, lit only by
    sky and sun through the glass. The camera looks at the window from inside."""
    W, H, D = 4.0, 2.6, 4.0
    wx0, wx1, wy0, wy1 = 1.2, 2.8, 0.9, 2.1  # window opening
    t = 0.15
    plaster = Material("plaster", base_color=(0.8, 0.78, 0.74))
    floor = Material("floor", base_color=(0.45, 0.3, 0.18), roughness=0.4)
    pane = Material("pane", base_color=(0.95, 0.97, 0.96), roughness=0.0, transmission=1.0, ior=1.5)
    wall = [  # the window wall, as four slabs around the opening
        box(((wx0) / 2, H / 2, -t / 2), (wx0, H, t)),
        box(((wx1 + W) / 2, H / 2, -t / 2), (W - wx1, H, t)),
        box((W / 2, wy0 / 2, -t / 2), (wx1 - wx0, wy0, t)),
        box((W / 2, (wy1 + H) / 2, -t / 2), (wx1 - wx0, H - wy1, t)),
    ]
    meshes = [
        (box((W / 2, -t / 2, D / 2), (W + 2 * t, t, D + 2 * t)), floor, "floor"),
        (box((W / 2, H + t / 2, D / 2), (W + 2 * t, t, D + 2 * t)), plaster, "ceiling"),
        (box((W / 2, H / 2, D + t / 2), (W, H, t)), plaster, "back"),
        (box((-t / 2, H / 2, D / 2), (t, H, D + 2 * t)), plaster, "left"),
        (box((W + t / 2, H / 2, D / 2), (t, H, D + 2 * t)), plaster, "right"),
        *[(g, plaster, f"window-wall-{i}") for i, g in enumerate(wall)],
        (quad((wx0, wy0, -t / 2), (wx0, wy1, -t / 2), (wx1, wy1, -t / 2), (wx1, wy0, -t / 2)), pane, "pane"),
        (box((1.3, 0.375, 2.6), (1.2, 0.75, 0.8)), Material("table", base_color=(0.5, 0.35, 0.22), roughness=0.5),
         "table"),
    ]
    cam = Camera(position=(3.4, 1.5, 3.6), target=(1.6, 1.2, 0.0), fov=65.0)
    return Scene.from_meshes(meshes, camera=cam, environment=Environment.sky(sun_direction))


def furnace(albedo=0.8) -> Scene:
    """A diffuse sphere in a uniform white environment: with albedo 1 it vanishes."""
    m = Material("sphere", base_color=(albedo,) * 3)
    cam = Camera(position=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), fov=40.0)
    return Scene.from_meshes([(sphere((0, 0, 0), 1.0, 5), m, "sphere")], camera=cam,
                             environment=Environment.uniform((1.0, 1.0, 1.0)))


def lit_plane(sun_elevation_deg=60.0, albedo=0.5) -> Scene:
    """A large Lambert plane lit by a directional light, no environment: radiance albedo·E·cosθ/π."""
    m = Material("plane", base_color=(albedo,) * 3)
    e = np.radians(sun_elevation_deg)
    light = Light("directional", intensity=1.0, direction=(-np.cos(e), -np.sin(e), 0.0))
    cam = Camera(position=(0.0, 3.0, 0.0), target=(0.0, 0.0, 0.0), up=(0, 0, -1), fov=30.0)
    return Scene.from_meshes([(quad((-50, 0, -50), (-50, 0, 50), (50, 0, 50), (50, 0, -50)), m, "plane")],
                             camera=cam, lights=[light], environment=BLACK)
