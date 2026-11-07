"""Point cloud demo — sci-viz primitives v1 (Plans 1 + 2).

A protoplanetary-disk-style demo: ~10000 dust particles orbiting a central
star under Keplerian gravity. Each particle is rendered as a camera-facing
point sprite with sphere-imposter shading and colormapped by orbital speed
(inferno LUT — slow → black, fast → yellow). The central star is a real
PBR sphere lit by a bright point light at the origin. Two HUD text labels
above the star show running FPS and the disk's average orbital speed.

Demonstrates the full Plan 1 + Plan 2 surface end-to-end in one scene:
- Plan 1: PointCloud + ColormapMaterial + ScalarValue + Radius (sprite path)
- Plan 1: Mesh + StandardMaterial + Transform (PBR mesh path) coexisting
- Plan 2: TextLabel + LabelMaterial + LabelTextureAtlas (label pass)
- Per-frame mutation of ScalarValue and TextLabel propagating to GPU
- 10000 sprites + 1 mesh + 2 labels = 4 draw calls per frame
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import PointLight, StandardMaterial, sphere
from manifoldx.viz import (
    ColormapMaterial,
    LabelMaterial,
    PointCloud,
    Radius,
    ScalarValue,
    TextLabel,
)

# Simulation parameters -------------------------------------------------------
NUM_PARTICLES = 10000
GM = 60.0           # gravitational parameter of the central star
SOFTENING = 0.4     # avoids singularity / wild accelerations near the star
DISK_INNER = 2.0    # inner edge of the disk
DISK_OUTER = 8.0    # outer edge of the disk
DISK_THICK = 0.18   # z-axis half-width of the disk
ECCENTRICITY = 0.06  # fractional perturbation on circular orbits
STAR_RADIUS = 1.0   # central PBR sphere
EXTENT = DISK_OUTER * 1.2

engine = mx.Engine("Protoplanetary disk")
engine.camera.fit(EXTENT)

# No `engine.store.register_component(...)` boilerplate needed — the viz
# components inherit from `Component` and auto-register on first spawn.

# Central star: real 3D PBR sphere lit by a bright point light at the origin.
# StandardMaterial parameters tuned for a warm, diffuse star surface.
engine.spawn(
    Mesh(sphere(STAR_RADIUS, 32)),
    Material(StandardMaterial("#ffd45a", roughness=0.85, metallic=0.0)),
    Transform(pos=(0.0, 0.0, 0.0)),
    n=1,
)
engine.set_lights(
    [
        PointLight(color="#ffe9a8", intensity=80.0, position=(0.0, 0.0, 0.0)),
        PointLight(color="#ffffff", intensity=8.0, position=(6.0, 8.0, 6.0)),
    ]
)

# Disk particles --------------------------------------------------------------
# The star occupies entity index 0; particles occupy indices 1..NUM_PARTICLES.
# The gravity system's query names ScalarValue, which the star lacks, so it sees
# the particles and the HUD labels only: the star never moves.
positions = mx.random.positions_in_disk(
    NUM_PARTICLES, inner=DISK_INNER, outer=DISK_OUTER,
    thickness=DISK_THICK, axis="y", rng=7,
)

# Keplerian circular orbit around the central GM, plus small eccentric
# perturbation so orbits aren't perfectly circular.
particle_velocities = mx.random.velocities_orbit(positions, GM=GM, axis="y")
v_circ_mag = np.linalg.norm(particle_velocities, axis=1, keepdims=True)
particle_velocities += (
    mx.random.velocities_gaussian(NUM_PARTICLES, sigma=ECCENTRICITY, rng=8)
    * v_circ_mag
)

# Per-particle radius: brighter dust closer in, dustier farther out.
radii = mx.random.scalars_uniform(NUM_PARTICLES, low=0.04, high=0.10, rng=9)

initial_speeds = np.linalg.norm(particle_velocities, axis=1).astype(np.float32)

engine.spawn(
    PointCloud(),
    Material(ColormapMaterial(cmap="inferno", vmin=2.5, vmax=6.5)),
    Transform(pos=positions),
    ScalarValue(value=initial_speeds),
    Radius(radius=radii),
    n=NUM_PARTICLES,
)

# HUD: two world-space text labels above the disk. The label render pass
# is camera-facing, so world position is what matters; orientation is
# automatic. Stack FPS and SPD vertically; sit them above the disk plane
# so dust particles don't occlude the glyphs.
HUD_X = 0.0
HUD_Z = 0.0
HUD_BASE_Y = DISK_OUTER * 0.7
HUD_LINE = 1.4
NUM_HUD = 2

hud_positions = np.array(
    [
        [HUD_X, HUD_BASE_Y + HUD_LINE, HUD_Z],  # FPS
        [HUD_X, HUD_BASE_Y, HUD_Z],             # SPD
    ],
    dtype=np.float32,
)

# Pre-warm the atlas so slot 0 isn't a stale zero-texture on the first frame.
atlas = engine.get_label_atlas()
INITIAL_FPS_SLOT = atlas.get_or_create("FPS: --")
INITIAL_SPD_SLOT = atlas.get_or_create("SPD: --")

engine.spawn(
    Material(LabelMaterial(pixel_width=400, pixel_height=96)),
    Transform(pos=hud_positions),
    TextLabel(index=np.array([INITIAL_FPS_SLOT, INITIAL_SPD_SLOT], dtype=np.int64)),
    ScalarValue(value=np.zeros(NUM_HUD, dtype=np.float32)),
    Radius(radius=np.zeros(NUM_HUD, dtype=np.float32)),
    n=NUM_HUD,
)

# Velocities live outside the ECS, one row per entity the gravity query sees:
# rows 0..NUM_PARTICLES-1 are the dust particles, the last NUM_HUD the HUD labels
# (kept stationary by the gravity system). The labels' store slots follow the
# star and the particles.
N_TOTAL = NUM_PARTICLES + 1
HUD_FPS_IDX = N_TOTAL
HUD_SPD_IDX = N_TOTAL + 1
velocities = np.zeros((NUM_PARTICLES + NUM_HUD, 3), dtype=np.float32)
velocities[:NUM_PARTICLES] = particle_velocities

# HUD update state (boxed so the closure can mutate them).
_hud_state = {"fps_smooth": 60.0, "clock": 0.0}
HUD_INTERVAL = 0.25  # update HUD strings 4× per second


@engine.system
def gravity(query: mx.Query[Transform, ScalarValue], dt: float):
    """Soft gravity from the central star + speed-driven colormap + HUD."""
    global velocities

    pos = query[Transform].pos.data  # (NUM_PARTICLES + NUM_HUD, 3) snapshot

    # Soft Keplerian gravity toward the central star.
    accel = mx.physics.central_gravity(pos, GM=GM, softening=SOFTENING)

    # HUD labels are static — keep them anchored regardless of gravity.
    accel[NUM_PARTICLES:] = 0.0

    velocities[:] = velocities + accel * dt
    velocities[NUM_PARTICLES:] = 0.0

    speeds = np.linalg.norm(velocities, axis=1)

    query[Transform].pos += velocities * dt
    query[ScalarValue].value = speeds.reshape(-1, 1)

    # ----- HUD text update (throttled, bucketed to bound atlas usage) -----
    _hud_state["fps_smooth"] = (
        0.9 * _hud_state["fps_smooth"] + 0.1 * (1.0 / max(dt, 1e-4))
    )
    _hud_state["clock"] += dt
    if _hud_state["clock"] >= HUD_INTERVAL:
        _hud_state["clock"] = 0.0
        # Bucketing keeps the unique-string count well under the 256-slot cap.
        fps_bucket = int(round(_hud_state["fps_smooth"] / 5.0)) * 5
        avg_speed = float(speeds[:NUM_PARTICLES].mean())
        spd_bucket = round(avg_speed * 2.0) / 2.0

        atlas = engine.get_label_atlas()
        fps_idx = atlas.get_or_create(f"FPS: {fps_bucket}")
        spd_idx = atlas.get_or_create(f"SPD: {spd_bucket:.1f}")

        text_label_data = engine.store._components["TextLabel"]
        text_label_data[HUD_FPS_IDX, 0] = float(fps_idx)
        text_label_data[HUD_SPD_IDX, 0] = float(spd_idx)


if __name__ == "__main__":
    engine.cli()
