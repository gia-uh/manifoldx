"""Boids flocking simulation demo.

300 boids with separation, alignment, cohesion, and a soft boundary.
3 predators wander randomly — boids flee from them on sight.
Pure-numpy vectorized — no Python loops in the hot path.
"""

import os

import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.native import load as load_native
from manifoldx.resources import sphere, PhongMaterial

NATIVE = load_native()  # the native step, when the extension is built; MANIFOLDX_NUMPY=1 forces numpy

# ── Simulation parameters ───────────────────────────────────────
NUM_BOIDS = int(os.environ.get("BOIDS_N", "300"))
NUM_PREDATORS = 3
PERCEPTION_SQ = 4.0**2  # squared boid perception radius
FEAR_RADIUS_SQ = 10.0**2  # predator detection range (very wide)
SEP_W = 1.5  # separation weight
ALI_W = 1.0  # alignment weight
COH_W = 0.8  # cohesion weight
FEAR_W = 20.0  # predator avoidance (dominates all other forces)
MAX_SPEED = 10.0  # boid speed cap
PANIC_SPEED = 15.0  # fleeing boids get a speed boost
MIN_SPEED = 4.0  # prevents hovering
BOUND_RADIUS = 12.0  # soft boundary
PULL_STRENGTH = 0.02  # gentle central attraction

# Predator parameters
PRED_SPEED = 3.0  # predators are slower than boids
PRED_TURN_RATE = 1.0  # how fast predators change direction

# ── Engine setup ────────────────────────────────────────────────
engine = mx.Engine("Boids & Predators")
engine.camera.fit(BOUND_RADIUS * 1.2)

# ── Boid initial conditions ─────────────────────────────────────
# Gaussian cloud of positions; tangential swirl velocity around a tilted axis.
boid_pos = mx.random.positions_gaussian(NUM_BOIDS, sigma=5.0, rng=7)
boid_vel = mx.random.velocities_tangent(boid_pos, axis=(0.2, 1.0, 0.3), speed=6.0)

# ── Predator initial conditions ─────────────────────────────────
pred_pos = mx.random.positions_in_box(
    NUM_PREDATORS, half_size=BOUND_RADIUS * 0.5, rng=8
)
pred_dir = mx.random.velocities_on_sphere(NUM_PREDATORS, speed=1.0, rng=9)

# ── Spawn boids (instanced — single draw call) ─────────────────
engine.spawn(
    Mesh(sphere(0.15, 6)),
    Material(PhongMaterial("#33ddaa")),
    Transform(pos=boid_pos),
    n=NUM_BOIDS,
)

# ── Spawn predators (separate batch — different mesh/material) ──
engine.spawn(
    Mesh(sphere(0.8, 12)),
    Material(PhongMaterial("#ff3333")),
    Transform(pos=pred_pos),
    n=NUM_PREDATORS,
)


@engine.system
def boids_physics(query: mx.Query[Transform], dt: float):
    global boid_vel, pred_pos, pred_dir

    # Read all positions: first NUM_BOIDS are boids, last NUM_PREDATORS are predators
    all_pos = query[Transform].pos.data  # (N+P, 3)
    pos = all_pos[:NUM_BOIDS]  # (N, 3)
    pred_pos = all_pos[NUM_BOIDS:]  # (P, 3)

    if NATIVE is not None:
        NATIVE.boids_step(pos, boid_vel, pred_pos, pred_dir, dt)
        query[Transform].pos += all_pos - query[Transform].pos.data
        return

    # ── Boid-boid interactions ──────────────────────────────────
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # (N, N, 3)
    dist_sq = (diff * diff).sum(axis=2)  # (N, N)

    neighbors = (dist_sq < PERCEPTION_SQ) & (dist_sq > 1e-6)
    n_count = neighbors.sum(axis=1, keepdims=True).astype(np.float32)
    safe_count = np.maximum(n_count, 1.0)

    # Separation
    inv_dsq = np.zeros_like(dist_sq)
    inv_dsq[neighbors] = 1.0 / dist_sq[neighbors]
    sep = (-diff * (neighbors[:, :, np.newaxis] * inv_dsq[:, :, np.newaxis])).sum(axis=1)

    # Alignment
    avg_vel = (boid_vel[np.newaxis, :, :] * neighbors[:, :, np.newaxis]).sum(axis=1)
    avg_vel /= safe_count
    ali = avg_vel - boid_vel

    # Cohesion
    center = (pos[np.newaxis, :, :] * neighbors[:, :, np.newaxis]).sum(axis=1)
    center /= safe_count
    coh = center - pos

    # ── Predator avoidance ──────────────────────────────────────
    # diff from each boid to each predator: (N, P, 3)
    pred_diff = pos[:, np.newaxis, :] - pred_pos[np.newaxis, :, :]  # away from predator
    pred_dist = np.linalg.norm(pred_diff, axis=2)  # (N, P) — actual distance (not squared)

    # Boids within fear radius of any predator
    scared = pred_dist < np.sqrt(FEAR_RADIUS_SQ)  # (N, P)
    scared_any = scared.any(axis=1)  # (N,) — which boids are scared

    # Flee force: use 1/dist (linear) instead of 1/dist² — wider, stronger
    inv_dist = np.zeros_like(pred_dist)
    inv_dist[scared] = 1.0 / pred_dist[scared]
    fear = (pred_diff * (scared[:, :, np.newaxis] * inv_dist[:, :, np.newaxis])).sum(axis=1)

    # ── Central pull (soft boundary handled below via mx.physics) ──────
    pull = -pos * PULL_STRENGTH

    # ── Combine and integrate boid velocities ───────────────────
    accel = SEP_W * sep + ALI_W * ali + COH_W * coh + FEAR_W * fear + pull
    boid_vel += accel * dt
    # Continuous inward pull when boids stray outside BOUND_RADIUS.
    mx.physics.sphere_boundary(
        pos, boid_vel, radius=BOUND_RADIUS, mode="soft", strength=3.0, dt=dt
    )

    # Speed clamping with panic boost
    speed = np.linalg.norm(boid_vel, axis=1, keepdims=True)
    # Fleeing boids can go faster
    max_s = np.where(scared_any[:, np.newaxis], PANIC_SPEED, MAX_SPEED)
    min_s = np.where(scared_any[:, np.newaxis], MIN_SPEED * 1.5, MIN_SPEED)
    boid_vel = np.where(speed > max_s, boid_vel * max_s / speed, boid_vel)
    boid_vel = np.where(
        (speed < min_s) & (speed > 1e-6),
        boid_vel * min_s / speed,
        boid_vel,
    )

    # ── Predator movement (random wandering inside boundary) ────
    # Slowly rotate direction with random perturbation
    pred_dir_local = pred_dir.copy()
    pred_dir_local += np.random.randn(NUM_PREDATORS, 3).astype(np.float32) * PRED_TURN_RATE * dt
    norms = np.linalg.norm(pred_dir_local, axis=1, keepdims=True)
    pred_dir_local = pred_dir_local / np.maximum(norms, 1e-6)
    pred_dir[:] = pred_dir_local

    # Predator boundary: steer inward when outside
    pred_dist = np.linalg.norm(pred_pos, axis=1, keepdims=True)
    pred_over = np.maximum(pred_dist - BOUND_RADIUS * 0.8, 0.0)
    pred_inward = -pred_pos / np.maximum(pred_dist, 1e-6) * pred_over * 0.5
    pred_vel = pred_dir * PRED_SPEED + pred_inward

    # ── Write back all positions ────────────────────────────────
    # Boid displacement
    boid_delta = boid_vel * dt  # (N, 3)
    # Predator displacement
    pred_delta = pred_vel * dt  # (P, 3)
    # Combined
    delta = np.concatenate([boid_delta, pred_delta], axis=0)  # (N+P, 3)
    query[Transform].pos += delta


if __name__ == "__main__":
    engine.cli()
