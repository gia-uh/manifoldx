"""Verify per-instance StandardMaterial palette slots against frozen sRGB pixels."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from PIL import Image

from manifoldx.components import Material, Mesh, Palette, Transform
from manifoldx.resources import PALETTE_SURFACE, DirectionalLight, StandardMaterial, plane


def capture(disable):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    engine = mx.Engine("Offscreen palette", width=160, height=120, max_entities=8)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine"
    engine.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    engine.set_sun(DirectionalLight("#ffffff", 2.8, (0, 0, -1)))
    geometry = plane(1.15, 1.4)
    geometry.pop("uvs")
    geometry["colors"] = np.tile(np.asarray((0.68, 0.73, 0.82), dtype="f4"), (4, 1))
    geometry["surfaces"] = np.full(4, PALETTE_SURFACE + (0 if disable == "slot" else 7), dtype="f4")
    material = StandardMaterial("#bacde0", vertex_colors=True, paletted=True, roughness=0.75)
    red = (0.95, 0.22, 0.29)
    blue = (0.21, 0.32, 0.94)
    engine.spawn(
        Mesh(geometry), Material(material), Transform(pos=[(-1.5, 0, 0), (0, 0, 0)]),
        Palette(c7=[red, red if disable == "instance" else blue]), n=2,
    )
    engine.spawn(Mesh(geometry), Material(material), Transform(pos=(1.5, 0, 0)))
    return engine.render_frame(supersample=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable", choices=("slot", "instance"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image = capture(args.disable)
    Image.fromarray(image).save(args.output / "palette.png")
    regions = [np.s_[55:65, x - 5:x + 5] for x in (47, 80, 113)]
    medians = [np.median(image[region], axis=(0, 1)).tolist() for region in regions]
    result = {"median_rgb": medians}
    if args.reference:
        expected = np.asarray(Image.open(args.reference / "palette.png"))
        delta = np.abs(image.astype("i2") - expected)
        Image.fromarray(delta.astype("u1")).save(args.output / "diff.png")
        result["max_channel_error"] = [int(delta[region].max()) for region in regions]
    (args.output / "palette.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert medians[0][0] > medians[0][2] + 30, "palette slot did not tint the first instance red"
    assert medians[1][2] > medians[1][0] + 30, "palette slot did not tint the second instance blue"
    assert medians[2][0] > medians[1][0] + 30, "missing Palette did not use white"
    if args.reference:
        assert max(result["max_channel_error"]) <= 2, result


if __name__ == "__main__":
    main()
