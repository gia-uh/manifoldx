"""Exercise custom WGSL via the legacy Material methods and native 2D textures."""

import argparse
import json
import os
import sys
from pathlib import Path
from weakref import ref

import manifoldx as mx
import numpy as np
from manifoldx.camera import Camera
from manifoldx.components import Layer, Material, Mesh, Transform
from manifoldx.resources import Material as MaterialBase
from manifoldx.resources import DirectionalLight, FlatMaterial, StandardMaterial, plane
from manifoldx.systems import Query
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


def view_retire(output):
    engine = mx.Engine("Manual WGSL view retirement", width=160, height=120, max_entities=16)
    assert type(engine).__name__ == "_NativeEngine"
    engine.camera.set_pose((0, 0, 4), (0, 0, 0))
    engine.camera.layers = {2}
    engine.set_sun(DirectionalLight("#ffffff", 3.0, (0, 0, -1)))
    camera = Camera()
    camera.set_pose((0, 0, 4), (0, 0, 0))
    camera.layers = {1}
    geometry = plane(1.5, 1.5)
    unrelated = engine.spawn(
        Mesh(plane(3, 3)), Material(FlatMaterial("#e83728")),
        Transform(pos=(0.25, 0, 0)), Layer(layer=1),
    )
    remove = set()

    @engine.system
    def destroy_consumers(query: Query[Transform], dt):
        rows = query.get_component_data("Transform")
        engine.destroy(np.flatnonzero(np.isin(rows[:, 0], tuple(remove))))
        remove.clear()

    engine.render_frame()
    baseline = engine.last_frame["view_resources"]
    assert baseline == [0, 0]
    texture = engine.render_to_texture(camera, (40, 40), live=False)
    material = Manual(texture, 1.0)
    texture_ref, material_ref = ref(texture), ref(material)
    engine.spawn(Mesh(geometry), Material(material), Transform(pos=(-1, 0, 0)), Layer(layer=2))
    engine.spawn(
        Mesh(geometry), Material(StandardMaterial("#ffffff", albedo_map=texture)),
        Transform(pos=(1, 0, 0)), Layer(layer=2),
    )
    first = engine.render_frame()
    assert engine.last_frame["frame_order"][0] == "view"
    manual = np.s_[55:65, 52:62]
    standard = np.s_[55:65, 102:112]
    for roi in (manual, standard):
        red, green, blue = np.median(first[roi], axis=(0, 1))
        assert red > max(green, blue) + 40, (red, green, blue)
    Image.fromarray(first).save(output / "shared.png")

    engine.drop_texture(texture)
    del texture, material
    camera.set_pose((20, 0, 4), (20, 0, 0))
    remove.add(1.0)
    last = engine.render_frame()
    assert np.array_equal(last[manual], first[manual])
    assert not np.array_equal(last[standard], first[standard])
    assert "view" not in engine.last_frame["frame_order"]
    assert engine.last_frame["view_resources"] == [0, 1]
    Image.fromarray(last).save(output / "manual-last.png")

    read = engine.read_texture(texture_ref())
    assert not read.done()
    remove.add(-1.0)
    engine.render_frame()
    red, green, blue = read.result(timeout=1)[20, 20]
    assert int(red) > max(int(green), int(blue)) + 40
    for _ in range(3):
        engine.render_frame()
    assert np.array_equal(unrelated.transform.pos, [0.25, 0, 0])
    report = {
        "baseline": baseline,
        "drained": engine.last_frame["view_resources"],
        "material_released": material_ref() is None,
        "handle_released": texture_ref() is None,
        "shared_standard": True,
        "pending_read_completed": True,
        "unrelated_entity_alive": True,
    }
    print(json.dumps(report), flush=True)
    assert report["drained"] == baseline, f"manual view texture retained after its last consumer: {report}"
    assert report["material_released"] and report["handle_released"], report
    engine.close()
    (output / "manual-view.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--view-retire", action="store_true")
    args = parser.parse_args()
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.view_retire:
        view_retire(args.output)
        return
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
