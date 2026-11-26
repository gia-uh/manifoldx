"""Check baked Standard layers, fallback color and an isolated per-layer write."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from PIL import Image

from manifoldx.components import Baked, Material, Mesh, Transform
from manifoldx.resources import SURFACE_BAKED, DirectionalLight, StandardMaterial, plane
from manifoldx.textures import layer_mips, texture_array, write_layer


def tile(color):
    return layer_mips(np.broadcast_to(np.asarray(color, dtype="u1"), (8, 8, 3)))


def capture(disable):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    engine = mx.Engine("Offscreen baked array", width=160, height=120, max_entities=8)
    native = os.environ["MANIFOLDX_CORE"] == "rust"
    if native:
        assert type(engine).__name__ == "_NativeEngine"
    else:
        engine._ensure_offscreen(1)
    engine.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    engine.set_sun(DirectionalLight("#ffffff", 2.8, (0, 0, -1)))
    geometry = plane(1.15, 1.4)
    geometry["baked_uv"] = geometry.pop("uvs")
    geometry["colors"] = np.tile(np.asarray((0.42, 0.68, 0.36), dtype="f4"), (4, 1))
    geometry["surfaces"] = np.full(4, SURFACE_BAKED, dtype="f4")
    handle = texture_array(engine, 2, 8)
    write_layer(engine, handle, 0, tile((220, 43, 62)))
    write_layer(engine, handle, 1, tile((54, 73, 235)))
    material = StandardMaterial("#ffffff", roughness=0.75, vertex_colors=True, baked=handle)
    engine.spawn(
        Mesh(geometry), Material(material), Transform(pos=[(-1.5, 0, 0), (0, 0, 0)]),
        Baked(layer=[0, 0 if disable == "layer" else 1]), n=2,
    )
    engine.spawn(Mesh(geometry), Material(material), Transform(pos=(1.5, 0, 0)))
    if native:
        other = mx.Engine("Offscreen foreign texture owner", max_entities=1)
        wrong_mip = [tile((1, 2, 3))[0], np.zeros((3, 3, 4), dtype="u1")]
        for owner, layer, levels, message in (
            (other, 0, tile((1, 2, 3)), "different graphics owner"),
            (engine, 2, tile((1, 2, 3)), "outside the array"),
            (engine, 0, wrong_mip, "must be a 4x4 RGBA"),
        ):
            try:
                write_layer(owner, handle, layer, levels)
            except ValueError as error:
                assert message in str(error), str(error)
            else:
                raise AssertionError(f"write_layer accepted {message}")
    first = engine.render_frame(supersample=1)
    if disable != "update":
        write_layer(engine, handle, 0, tile((225, 184, 38)))
    return first, engine.render_frame(supersample=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable", choices=("layer", "update"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    first, second = capture(args.disable)
    regions = [np.s_[55:65, x - 5:x + 5] for x in (54, 80, 106)]
    result = {"changed_pixels": [int(np.any(first[r] != second[r], axis=2).sum()) for r in regions]}
    for name, frame in (("first", first), ("second", second)):
        Image.fromarray(frame).save(args.output / f"{name}.png")
        result[f"{name}_rgb"] = [np.median(frame[r], axis=(0, 1)).tolist() for r in regions]
        if args.reference:
            expected = np.asarray(Image.open(args.reference / f"{name}.png"))
            delta = np.abs(frame.astype("i2") - expected)
            Image.fromarray(delta.astype("u1")).save(args.output / f"{name}-diff.png")
            result[f"{name}_error"] = [int(delta[r].max()) for r in regions]
    (args.output / "baked.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    red, blue, green = result["first_rgb"]
    assert red[0] > red[2] + 30, "the first layer did not sample red"
    assert blue[2] > blue[0] + 30, "the selected second layer did not sample blue"
    assert green[1] > green[0] + 30, "the missing Baked layer did not draw its vertex color"
    assert result["changed_pixels"] == [100, 0, 0], "layer write changed the wrong ROIs"
    if args.reference:
        assert max(result["first_error"] + result["second_error"]) <= 2, result


if __name__ == "__main__":
    main()
