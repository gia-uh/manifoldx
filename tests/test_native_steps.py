"""The native steps, when the extension is built on this branch.

The test skips on a normal install: the native module is a development build from
``crates/spike-nbody-boids``, not a wheel dependency yet.
"""

import numpy as np
import pytest

from manifoldx.native import load

native = load()
pytestmark = pytest.mark.skipif(native is None, reason="native extension not built")


def test_nbody_step_matches_numpy():
    import manifoldx as mx

    rng = np.random.default_rng(3)
    n = 64
    pos = (rng.random((n, 3), dtype=np.float32) - 0.5) * 8.0
    vel = np.zeros((n, 3), np.float32)
    masses = rng.uniform(0.5, 3.0, n).astype(np.float32)
    dt, g, softening = 1 / 60, 20.0, 0.05

    want_pos, want_vel = pos.copy(), vel.copy()
    accel = mx.physics.gravity(want_pos, masses=masses, G=g, softening=softening)
    want_vel += accel * dt
    want_pos += want_vel * dt

    got_pos, got_vel = pos.copy(), vel.copy()
    native.nbody_step(got_pos, got_vel, masses, dt, g, softening)
    assert np.allclose(got_pos, want_pos, rtol=1e-4, atol=1e-4)
    assert np.allclose(got_vel, want_vel, rtol=1e-4, atol=1e-4)


def test_boids_step_runs_and_moves():
    rng = np.random.default_rng(3)
    n, p = 128, 3
    pos = (rng.standard_normal((n, 3)) * 5.0).astype(np.float32)
    vel = (rng.standard_normal((n, 3)) * 6.0).astype(np.float32)
    pred = np.full((p, 3), 1000.0, np.float32)
    direction = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (p, 1))

    before = pos.copy()
    native.boids_step(pos, vel, pred, direction, 1 / 60)
    assert np.isfinite(pos).all() and np.isfinite(vel).all()
    assert not np.array_equal(pos, before)
