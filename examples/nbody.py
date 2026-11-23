"""N-Body gravitational simulation demo.

Pure-numpy vectorized all-pairs gravity. No Python loops in the hot path.
See examples/gas.py for a collision-based simulation.
"""

import os

import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.native import load as load_native
from manifoldx.resources import sphere, PhongMaterial

NATIVE = load_native()  # the native step, when the extension is built; MANIFOLDX_NUMPY=1 forces numpy
NUM_BODIES = int(os.environ.get("NBODY_N", "500"))
G = 20.0  # gravitational constant
SOFTENING = 0.05  # prevents singularities at close range
MAX_SPEED = 20.0  # velocity clamp
SPHERE_RADIUS = 0.5  # base mesh radius
SIZE = 5 * NUM_BODIES ** (1 / 3)

engine = mx.Engine("N-Body Simulation")
engine.camera.fit(SIZE)

# Random initial positions spread uniformly + random masses (cube-root so
# volume ∝ mass for the visual sphere scale).
positions = mx.random.positions_in_box(NUM_BODIES, half_size=SIZE, rng=7)
masses = mx.random.scalars_uniform(NUM_BODIES, low=0.5, high=3.0, rng=7)
scales = (masses ** (1 / 3)).reshape(-1, 1)  # (N, 1) → broadcasts to (N, 3)

# Spawn all bodies at once (instanced rendering)
engine.spawn(
    Mesh(sphere(SPHERE_RADIUS, 12)),
    Material(PhongMaterial("#ffaa44")),
    Transform(pos=positions, scale=scales),
    n=NUM_BODIES,
)

# Velocities (not part of ECS — pure numpy)
velocities = np.zeros((NUM_BODIES, 3), dtype=np.float32)


@engine.system
def nbody_gravity(query: mx.Query[Transform], dt: float):
    global velocities
    pos = query[Transform].pos.data
    if NATIVE is not None:
        NATIVE.nbody_step(pos, velocities, masses, dt, G, SOFTENING)
        query[Transform].pos += pos - query[Transform].pos.data
    else:
        accel = mx.physics.gravity(pos, masses=masses, G=G, softening=SOFTENING)
        velocities += accel * dt
        query[Transform].pos += velocities * dt


if __name__ == "__main__":
    engine.cli()
