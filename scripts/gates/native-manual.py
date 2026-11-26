"""Exercise custom WGSL via the legacy Material methods and native 2D textures."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import Material as MaterialBase
from manifoldx.resources import plane
from PIL import Image

WGSL = """
struct Globals { vp: mat4x4<f32> };
struct Tint { color: vec4<f32> };
@group(0) @binding(0) var<uniform> world: Globals;
@group(0) @binding(1) var<storage, read> models: array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> tint: Tint;
@group(0) @binding(3) var<uniform> light_data: array<vec4<f32>, 8>;
@group(0) @binding(4) var sampler_at_four: sampler;
@group(0) @binding(5) var view_at_five: texture_2d<f32>;
struct Out { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs_main(@location(0) position: vec3<f32>, @location(2) uv: vec2<f32>,
                   @builtin(instance_index) index: u32) -> Out {
    return Out(world.vp * models[index] * vec4<f32>(position, 1.0), uv);
}
@fragment fn fs_main(input: Out) -> @location(0) vec4<f32> {
    return textureSample(view_at_five, sampler_at_four, input.uv) * tint.color;
}
"""


class Manual(MaterialBase):
    pipeline_subtype = "textured"

    def __init__(self, texture, gain):
        self.albedo_map = texture
        self.gain = gain

    @classmethod
    def _compile(cls, *, textured=True):
        return WGSL

    @classmethod
    def uniform_type(cls):
        return {"color": "vec4<f32>"}

    def get_texture_bindings(self):
        return {4: self.albedo_map}

    def get_data(self, n, registry):
        return np.tile(np.array([self.gain, self.gain, self.gain, 1], dtype="f4"), (n, 1))


class WrongBinding(Manual):
    @classmethod
    def _compile(cls, *, textured=True):
        return WGSL.replace("@binding(4)", "@binding(5)").replace(
            "@binding(5) var view", "@binding(4) var view"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--swap", action="store_true")
    args = parser.parse_args()
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    e = mx.Engine("Manual WGSL texture", width=96, height=80, max_entities=16)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(e).__name__ == "_NativeEngine"
    e.camera.set_pose(position=(0, 0, 4), target=(0, 0, 0))
    if os.environ["MANIFOLDX_CORE"] == "python":
        e.render_frame(supersample=1)
    pixels = np.tile(np.array([79, 198, 91, 255], dtype="u1"), (2, 2, 1))
    tex = mx.texture_from_array(e, pixels)
    cls = WrongBinding if args.swap else Manual
    mat = cls(tex, 1.0)
    mesh = plane(1.25, 1.35)
    e.spawn(Mesh(mesh), Material(mat), Transform(pos=[(-0.9, 0, 0), (0.9, 0, 0)]), n=2)
    regions = [np.s_[35:45, 28:38], np.s_[35:45, 58:68]]
    report = {}
    for name, gain in (("original", 1.0), ("mutated", 0.25)):
        mat.gain = gain
        rgb = e.render_frame(supersample=1)
        Image.fromarray(rgb).save(args.output / f"{name}.png")
        report[name] = [rgb[roi].mean(axis=(0, 1)).tolist() for roi in regions]
        if args.reference:
            reference = np.asarray(Image.open(args.reference / f"{name}.png"))
            delta = np.abs(rgb.astype("i2") - reference)
            Image.fromarray(delta.astype("u1")).save(args.output / f"{name}-diff.png")
            assert max(int(delta[roi].max()) for roi in regions) <= 1
    assert report["original"] == [[79, 198, 91], [79, 198, 91]], report
    assert np.linalg.norm(np.array(report["original"]) - report["mutated"]) > 100, report
    (args.output / "manual.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
