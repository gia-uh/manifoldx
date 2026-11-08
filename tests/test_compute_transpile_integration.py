"""Integration: transpiled GravityKernel must produce the same numeric
output as the Phase-1 hand-written WGSL on identical inputs.

The strongest possible signal of transpiler correctness — if both
kernels diverge by more than a few ULPs after one frame, something in
the codegen is wrong.
"""
import numpy as np
import pytest


def _make_offscreen_engine():
    try:
        from manifoldx.backends import get_offscreen_canvas
        canvas = get_offscreen_canvas(width=64, height=64)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    import manifoldx as mx
    engine = mx.Engine("test", width=64, height=64)
    engine._init_canvas(canvas)
    engine._running = True
    return engine


def test_transpiled_gravity_matches_hand_written_wgsl_after_one_frame():
    """Same initial state → run one frame each → arrays agree to rtol=1e-5."""
    from manifoldx.compute import Compute, Reads, ReadsWrites, Uniform
    from manifoldx.compute.shader import vec3, dot, sqrt, u32
    from manifoldx.components import Component, Material, Mesh, Transform
    from manifoldx.types import Float, Vector3
    from manifoldx.resources import BasicMaterial, sphere

    class Velocity(Component):
        vector: Vector3

    class Mass(Component):
        value: Float = 1.0

    HAND_WGSL = """
    struct Uniforms { G: f32, softening: f32, dt: f32, n: f32, };
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var<storage, read> masses: array<f32>;
    @group(0) @binding(2) var<storage, read_write> transforms: array<f32>;
    @group(0) @binding(3) var<storage, read_write> velocities: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        let n = u32(uniforms.n);
        if (i >= n) { return; }
        let pos_i = vec3<f32>(transforms[i*10u + 0u], transforms[i*10u + 1u], transforms[i*10u + 2u]);
        var accel = vec3<f32>(0.0);
        for (var j = 0u; j < n; j = j + 1u) {
            if (i == j) { continue; }
            let pos_j = vec3<f32>(transforms[j*10u + 0u], transforms[j*10u + 1u], transforms[j*10u + 2u]);
            let diff = pos_j - pos_i;
            let r2 = dot(diff, diff) + uniforms.softening * uniforms.softening;
            let inv_r3 = 1.0 / (r2 * sqrt(r2));
            accel = accel + uniforms.G * masses[j] * diff * inv_r3;
        }
        var vel = vec3<f32>(velocities[i*3u + 0u], velocities[i*3u + 1u], velocities[i*3u + 2u]);
        vel = vel + accel * uniforms.dt;
        velocities[i*3u + 0u] = vel.x;
        velocities[i*3u + 1u] = vel.y;
        velocities[i*3u + 2u] = vel.z;
        transforms[i*10u + 0u] = pos_i.x + vel.x * uniforms.dt;
        transforms[i*10u + 1u] = pos_i.y + vel.y * uniforms.dt;
        transforms[i*10u + 2u] = pos_i.z + vel.z * uniforms.dt;
    }
    """.strip()

    class HandWritten(Compute):
        transforms: ReadsWrites[Transform]
        masses:     Reads[Mass]
        velocities: ReadsWrites[Velocity]
        G:         Uniform[float] = 20.0
        softening: Uniform[float] = 0.05
        dt:        Uniform[float] = "frame_dt"
        n:         Uniform[float] = "entity_count"
        workgroup_size = 64
        dispatch = "entity_count"
        def compile(self) -> str:
            return HAND_WGSL

    class Transpiled(Compute):
        transforms: ReadsWrites[Transform]
        masses:     Reads[Mass]
        velocities: ReadsWrites[Velocity]
        G:         Uniform[float] = 20.0
        softening: Uniform[float] = 0.05
        dt:        Uniform[float] = "frame_dt"
        n:         Uniform[float] = "entity_count"
        workgroup_size = 64
        dispatch = "entity_count"
        def pair_accel(self, pos_i: vec3, pos_j: vec3, m_j: float) -> vec3:
            diff: vec3 = pos_j - pos_i
            r2: float = dot(diff, diff) + self.softening * self.softening
            inv_r3: float = 1.0 / (r2 * sqrt(r2))
            return self.G * m_j * diff * inv_r3
        def main(self, i: int):
            if i >= u32(self.n):
                return
            pos_i: vec3 = self.transforms[i].pos
            accel: vec3 = vec3(0.0, 0.0, 0.0)
            for j in range(u32(self.n)):
                if i == j:
                    continue
                accel += self.pair_accel(pos_i, self.transforms[j].pos, self.masses[j].value)
            self.velocities[i].vector += accel * self.dt
            self.transforms[i].pos    += self.velocities[i].vector * self.dt

    rng = np.random.default_rng(7)
    N = 16
    positions = rng.uniform(-2.0, 2.0, size=(N, 3)).astype(np.float32)
    mass_vals = rng.uniform(0.5, 3.0, size=N).astype(np.float32)
    initial_vel = np.zeros((N, 3), dtype=np.float32)

    def run(kernel_cls):
        engine = _make_offscreen_engine()
        # Pin dt to 1/60 — _draw_frame's wall-clock measurement makes
        # auto-bound `frame_dt` differ between the two engine runs and
        # would dominate any numerical comparison.
        engine._use_fixed_dt = True
        engine._fixed_dt_value = 1 / 60
        engine.spawn(
            Mesh(sphere(0.1, 8)),
            Material(BasicMaterial("#ffffff")),
            Transform(pos=positions),
            Velocity(vector=initial_vel),
            Mass(value=mass_vals),
            n=N,
        )
        engine.compute(kernel_cls)
        engine._draw_frame()
        t_data = engine.store._components["Transform"][:N].copy()
        v_data = engine.store._components["Velocity"][:N].copy()
        return t_data, v_data

    t_hand, v_hand = run(HandWritten)
    t_trans, v_trans = run(Transpiled)

    np.testing.assert_allclose(t_hand[:, :3], t_trans[:, :3], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v_hand,         v_trans,        rtol=1e-5, atol=1e-5)
