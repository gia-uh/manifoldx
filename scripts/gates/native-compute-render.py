"""Exercise Compute WGSL and its dependency on the next generated kernel."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.compute import Compute, Reads, ReadsWrites, Uniform
from manifoldx.compute.shader import u32, vec3
from manifoldx.resources import FlatMaterial, plane
from PIL import Image


class Raw(Compute):
    transforms: ReadsWrites[Transform]
    count: Uniform[float] = "entity_count"
    amount: Uniform[float] = 0.25

    def compile(self):
        return """
struct Uniforms { count: f32, amount: f32 };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> transforms: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(uniforms.count) { return; }
    let row = id.x * 10u;
    transforms[row] += uniforms.amount;
}
"""


class Generated(Compute):
    transforms: ReadsWrites[Transform]
    count: Uniform[float] = "entity_count"
    multiplier: Uniform[float] = 2.0

    def main(self, i: int):
        if i >= u32(self.count):
            return
        position: vec3 = self.transforms[i].pos
        self.transforms[i].pos = vec3(position.x * self.multiplier, position.y, position.z)



class Ticks(Component, gpu_only=True):
    value: float


class SparseRaw(Compute):
    transforms: ReadsWrites[Transform]
    ticks: ReadsWrites[Ticks]
    count: Uniform[float] = "entity_count"

    def compile(self):
        return """
@group(0) @binding(0) var<uniform> count: f32;
@group(0) @binding(1) var<storage, read_write> transforms: array<f32>;
@group(0) @binding(2) var<storage, read_write> ticks: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(count) { return; }
    ticks[id.x] += 1.0;
    transforms[id.x * 10u] += 0.25;
}
"""


class SparseGenerated(Compute):
    transforms: ReadsWrites[Transform]
    ticks: Reads[Ticks]
    count: Uniform[float] = "entity_count"

    def main(self, i: int):
        if i >= u32(self.count):
            return
        p: vec3 = self.transforms[i].pos
        self.transforms[i].pos = vec3(p.x + 0.125 * self.ticks[i].value, p.y, p.z)


class SparseSlots(Compute):
    transforms: ReadsWrites[Transform]
    ticks: Reads[Ticks]
    count: Uniform[float] = "entity_count"
    dispatch = "max_entities"

    def main(self, i: int):
        if i == 64:
            p: vec3 = self.transforms[i].pos
            self.transforms[i].pos = vec3(p.x + 0.125 * self.ticks[i].value, p.y, p.z)


class SparseExplicit(Compute):
    transforms: ReadsWrites[Transform]
    dispatch = 1

    def main(self, i: int):
        # The unguarded padded invocations must not publish changes to other slots.
        p: vec3 = self.transforms[i].pos
        self.transforms[i].pos = vec3(p.x + 0.125, p.y, p.z)


class SparseCallable(Compute):
    transforms: ReadsWrites[Transform]
    dispatch = staticmethod(lambda engine: 2)

    def main(self, i: int):
        if i == 1:
            p: vec3 = self.transforms[i].pos
            self.transforms[i].pos = vec3(p.x + 0.125, p.y, p.z)


def sparse(output, mode):
    engine = mx.Engine("Offscreen sparse native compute", width=160, height=112, max_entities=67)
    assert type(engine).__name__ == "_NativeEngine"
    engine.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    geometry = Mesh(plane(0.4, 0.5))
    material = Material(FlatMaterial("#cd4034"))
    engine.spawn(geometry, material, Transform(pos=(-2.8, 0.9, 0)))
    engine.spawn(geometry, material, Transform(pos=(-2.0, 0, 0)))
    # A live non-member must keep its dense row without gaining a Transform.
    engine.world.spawn(count=62)
    engine.spawn(geometry, material, Transform(pos=(0.1, -0.9, 0)))
    initial = engine.world.query("Transform").read()
    old = initial.entities[1]
    payload = {
        "Mesh": engine.world.query("Mesh").read().numpy()[1:2].copy(),
        "Material": engine.world.query("Material").read().numpy()[1:2].copy(),
        "Transform": Transform(pos=(-2.0, 0, 0)).get_data(1),
    }
    command = None

    @engine.world.system(structural=True)
    def edit(snapshot, commands):
        if command == "destroy":
            commands.destroy((old,))
        elif command == "recycle":
            commands.spawn(payload=payload)

    for cls in (SparseRaw, SparseGenerated):
        engine.compute(cls)
    if mode == "mixed":
        for cls in (SparseSlots, SparseExplicit, SparseCallable):
            engine.compute(cls)

    results = []
    for frame, command in enumerate((None, "destroy", "recycle")):
        image = engine.render_frame(supersample=1)
        Image.fromarray(image).save(output / f"sparse-{frame}.png")
        stats = dict(engine.last_frame)
        transfer_probe = getattr(engine._graphics, "compute_transfers", None)
        transfers = list(transfer_probe()) if transfer_probe else None
        rows = engine.world.query("Transform").read()
        slots = [entity.slot for entity in rows.entities]
        positions = rows.numpy()[:, :3]
        assert stats == engine.last_frame
        hits = np.all(image == (0xCD, 0x40, 0x34), axis=-1)
        pixels = {}
        for slot, (x, y, z) in zip(slots, positions):
            # Derive the expected image point from the ECS result using the public camera.
            projected, _ = engine.camera.project((x, y, z), 160, 112)
            px, py = projected[0]
            ix, iy = round(px), round(py)
            pixels[slot] = int(hits[iy - 2:iy + 3, ix - 2:ix + 3].sum())
        results.append({
            "slots": slots,
            "generations": [entity.generation for entity in rows.entities],
            "x": positions[:, 0].tolist(),
            "pixels": pixels,
            "middle_roi_pixels": int(hits[51:61].sum()),
            "transfers": transfers,
        })

    (output / "sparse.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
    assert [row["slots"] for row in results] == [[0, 1, 64], [0, 64], [0, 1, 64]], "Compute changed membership"
    assert results[2]["generations"][1] > old.generation, "recycle did not create a new generation"
    assert all(min(row["pixels"].values()) >= 9 for row in results), "compute-driven mesh missed its ROI"
    assert results[1]["middle_roi_pixels"] == 0, "retired mesh still drew"
    expected = [
        [-2.425, -1.625, 0.475],
        [-1.925, 0.975],
        [-1.3, -1.625, 1.6],
    ] if mode == "dense" else [
        [-2.3, -1.5, 0.6],
        [-1.675, 1.35],
        [-0.925, -1.5, 2.35],
    ]
    for frame, (row, values) in enumerate(zip(results, expected)):
        np.testing.assert_allclose(row["x"], values, atol=5e-7, rtol=0,
                                   err_msg=f"sparse {mode} frame {frame}: wrong identity or generation")
    expected_transfers = [2 if mode == "dense" else 5, 1, 1]
    assert all(row["transfers"] == expected_transfers for row in results), "unexpected Compute transfers"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--disable", choices=("raw", "generated", "dependency"))
    parser.add_argument("--sparse", choices=("dense", "mixed"))
    args = parser.parse_args()
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.sparse:
        sparse(args.output, args.sparse)
        return
    engine = mx.Engine("Offscreen native compute", width=112, height=72, max_entities=8)
    native = os.environ["MANIFOLDX_CORE"] == "rust"
    if native:
        assert type(engine).__name__ == "_NativeEngine"
    engine.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    material = FlatMaterial("#cd4034")
    engine.spawn(Mesh(plane(0.5, 0.6)), Material(material),
                 Transform(pos=[(-1.0, 0.3, 0), (0.125, -0.3, 0)]), n=2)
    classes = [Generated, Raw] if args.disable == "dependency" else [Raw, Generated]
    for cls in classes:
        if args.disable != ("raw" if cls is Raw else "generated"):
            assert engine.compute(cls) is cls
    results = []
    image_centers = []
    for frame in range(2):
        image = engine.render_frame(supersample=1)
        Image.fromarray(image).save(args.output / f"frame-{frame}.png")
        hits = np.all(image[..., :3] == (0xCD, 0x40, 0x34), axis=-1)
        image_centers.append([
            float(xs.mean()) if xs.size else None
            for xs in (np.nonzero(half)[1] for half in np.split(hits, 2))
        ])
        if native:
            positions = engine.world.query("Transform").read().numpy()[:, :3].tolist()
        else:
            positions = engine.store.get_component_data("Transform", [0, 1])[:, :3].tolist()
        results.append([position[0] for position in positions])
    (args.output / "compute.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
    if native:
        before = dict(engine.last_frame)
        timings = engine._graphics.compute_timings()
        assert before == engine.last_frame
        if all(value is not None for value in timings):
            assert all(value > 0 for value in timings), timings
        (args.output / "timestamps.json").write_text(json.dumps({
            "compute_passes_ns": timings,
            "frame_sample_unchanged": before == engine.last_frame,
        }, indent=2) + "\n")
    (args.output / "image-centers.json").write_text(json.dumps(image_centers) + "\n")
    assert results == [[-1.5, 0.75], [-2.5, 2.0]], "compute dispatch or dependency produced wrong coordinates"
    assert all(x is not None for centers in image_centers for x in centers), "missing compute-driven mesh ROI"
    np.testing.assert_allclose(image_centers, [[40, 63], [29.5, 76]], atol=1.0, rtol=0)


if __name__ == "__main__":
    main()
