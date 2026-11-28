"""Sun + Earth + Moon orbital demo.

Not-to-scale (orbital radii and rotation rates picked for visual rhythm,
not astronomical realism):

- Sun at origin, unlit BasicMaterial with a procedural surface texture.
  A point light at origin illuminates Earth and Moon from the sun's
  direction.
- Earth orbits the sun on the XZ plane, axial tilt 23.5° around Z, spins
  on that tilted axis ~12× faster than its orbit so the rotation reads.
- Moon orbits Earth on a slightly inclined plane, ~12× faster than
  Earth's year.

Run interactively:
    uv run python examples/sun_earth_moon_demo.py

Or render to video:
    uv run python examples/sun_earth_moon_demo.py --render --duration 12 \
        --fps 30 --output /tmp/solar.mp4
"""

from pathlib import Path

import numpy as np

import manifoldx as mx
from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import StandardMaterial, BasicMaterial, PointLight, sphere
from manifoldx.textures import load_texture


# Geometry: one sphere reused for all three bodies; per-entity scale handled
# via the Transform scale field.
ASSETS = Path(__file__).parent / "assets" / "sun_earth_moon"

# Visual scale (not physical):
SUN_RADIUS = 1.5
EARTH_RADIUS = 0.45
MOON_RADIUS = 0.18

EARTH_ORBIT_RADIUS = 6.0
MOON_ORBIT_RADIUS = 1.1

# Periods in seconds — fast enough that you actually see them move.
EARTH_YEAR = 14.0
EARTH_DAY = EARTH_YEAR / 12.0   # 12 "days" per "year"
MOON_MONTH = EARTH_YEAR / 12.0

EARTH_TILT = np.deg2rad(23.5)
MOON_INCLINATION = np.deg2rad(8.0)


engine = mx.Engine("Sun · Earth · Moon", width=900, height=900)

# Strong key light at the sun's center; lights Earth/Moon from the sun's
# direction the way it should.
engine.add_light(PointLight(position=(0, 0, 0), color="#ffffff", intensity=200.0))
# Soft cool fill so the night side isn't pitch black.
engine.add_light(PointLight(position=(0, 8, 8), color="#7799cc", intensity=12.0))


# Module-level handles to the three entities — populated in startup.
bodies = {"sun": None, "earth": None, "moon": None}


def _quat_axis_angle(axis_xyz, angle):
    """Build a (x, y, z, w) quaternion for a rotation of `angle` around `axis`."""
    ax = np.asarray(axis_xyz, dtype=np.float32)
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, np.cos(half)], dtype=np.float32)


def _quat_multiply(q1, q2):
    """Hamilton product q1 * q2 (both in (x, y, z, w) order)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)


@engine.on("startup")
def setup(_payload):
    # All three bodies share the same UV sphere; per-body scale via Transform.
    geo = sphere(1.0, segments=64)

    earth_tex = load_texture(engine, ASSETS / "earth.jpg")
    moon_tex = load_texture(engine, ASSETS / "moon.jpg")

    # Sun: unlit yellow ball. BasicMaterial doesn't sample textures (that's
    # follow-up work), but a point light at the sun's center can't illuminate
    # the sun's own surface anyway — every face normal points away. An unlit
    # flat-yellow ball reads as "sun" in the scene; planets do the heavy
    # lifting for the textured-PBR demo.
    sun_mat = BasicMaterial(color="#ffdd55")

    earth_mat = StandardMaterial(
        color="#ffffff", roughness=0.55, metallic=0.0, ao=1.0,
        albedo_map=earth_tex,
    )
    moon_mat = StandardMaterial(
        color="#ffffff", roughness=0.85, metallic=0.0, ao=1.0,
        albedo_map=moon_tex,
    )

    # Spawn in order and keep a handle on each: the frame handler poses them
    # through the public transform surface, not the engine's internal store.
    bodies["sun"] = engine.spawn(Mesh(geo), Material(sun_mat),
                                 Transform(pos=(0, 0, 0), scale=(SUN_RADIUS,) * 3))
    bodies["earth"] = engine.spawn(Mesh(geo), Material(earth_mat),
                                   Transform(pos=(EARTH_ORBIT_RADIUS, 0, 0),
                                             scale=(EARTH_RADIUS,) * 3))
    bodies["moon"] = engine.spawn(Mesh(geo), Material(moon_mat),
                                  Transform(pos=(EARTH_ORBIT_RADIUS + MOON_ORBIT_RADIUS, 0, 0),
                                            scale=(MOON_RADIUS,) * 3))

    # Camera: pulled back, tilted down, looking at the sun. Use set_pose so
    # the spherical (distance, azimuth, elevation) cache stays consistent —
    # otherwise orbit() reverts to the constructor's default distance.
    engine.camera.set_pose(position=(0, 5.0, 14.0), target=(0, 0, 0))


@engine.on("frame")
def orbit_bodies(payload):
    if bodies["sun"] is None:
        return

    t = payload["elapsed"]
    earth = bodies["earth"].transform
    moon = bodies["moon"].transform
    sun = bodies["sun"].transform

    # Earth orbits the sun.
    earth_theta = (t / EARTH_YEAR) * 2 * np.pi
    ex = EARTH_ORBIT_RADIUS * np.cos(earth_theta)
    ez = EARTH_ORBIT_RADIUS * np.sin(earth_theta)
    earth.pos = (ex, 0.0, ez)

    # Earth spin on its tilted axis.
    earth_spin = (t / EARTH_DAY) * 2 * np.pi
    tilt_q = _quat_axis_angle((0, 0, 1), EARTH_TILT)
    spin_q = _quat_axis_angle((0, 1, 0), earth_spin)
    earth_rot = _quat_multiply(tilt_q, spin_q)
    earth.rot = earth_rot

    # Moon orbits Earth on an inclined plane.
    moon_theta = (t / MOON_MONTH) * 2 * np.pi
    mx_ = MOON_ORBIT_RADIUS * np.cos(moon_theta)
    mz_ = MOON_ORBIT_RADIUS * np.sin(moon_theta) * np.cos(MOON_INCLINATION)
    my_ = MOON_ORBIT_RADIUS * np.sin(moon_theta) * np.sin(MOON_INCLINATION)
    moon.pos = (ex + mx_, my_, ez + mz_)

    # Moon spin (slow — tidally locked toward Earth would be one rotation per
    # orbit; we leave it independent for visual interest).
    moon_spin = (t / (MOON_MONTH * 0.5)) * 2 * np.pi
    moon_rot = _quat_axis_angle((0, 1, 0), moon_spin)
    moon.rot = moon_rot

    # Sun spin (the texture rotates so you can see it's not a flat ball).
    sun_spin = (t / 30.0) * 2 * np.pi
    sun_rot = _quat_axis_angle((0, 1, 0), sun_spin)
    sun.rot = sun_rot


@engine.system
def slow_camera_drift(query, dt: float):
    """Slow camera orbit so the perspective keeps changing during long runs."""
    engine.camera.orbit(4 * dt, 0)


if __name__ == "__main__":
    engine.cli(fps=30, duration=12, output="sun_earth_moon.mp4")
