"""Exercise Compute WGSL and its dependency on the next generated kernel."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.compute import Compute, ReadsWrites, Uniform
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--disable", choices=("raw", "generated", "dependency"))
    args = parser.parse_args()
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
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
