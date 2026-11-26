"""Verify LineList axes with world and screen anchors."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.viz import AxisFrame, AxisMaterial
from manifoldx.viz.geometry import AXIS_LINE_X, AXIS_LINE_Y, AXIS_LINE_Z
from PIL import Image


def frame(screen, disabled):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    engine = mx.Engine("Offscreen axis", width=128, height=128, max_entities=8)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine"
    if screen:
        engine.camera.set_pose(position=(20, 15, 9), target=(2, 2, 1))
        if not disabled:
            engine.spawn(
                Mesh(AXIS_LINE_X), Material(AxisMaterial(color="#ffff00", anchor_mode="screen")),
                Transform(pos=(0, -0.7, 0), scale=(0.4, 1, 1)), AxisFrame(),
            )
    else:
        engine.camera.set_pose(position=(6, 4, 6), target=(0, 0, 0))
        for axis, geometry in enumerate((AXIS_LINE_X, AXIS_LINE_Y, AXIS_LINE_Z)):
            if axis == 1 and disabled:
                continue
            scale = tuple(2.8 if i == axis else 1.0 for i in range(3))
            color = ("#ff0000", "#00ff00", "#0000ff")[axis]
            engine.spawn(
                Mesh(geometry), Material(AxisMaterial(color=color)),
                Transform(scale=scale), AxisFrame(extent=2.8),
            )
    return engine.render_frame(supersample=1)


def mask(image, color):
    return np.all(image == color, axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable", choices=("world", "screen"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = {"world": [(255,0,0), (0,255,0), (0,0,255)], "screen": [(255,255,0)]}
    result = {}
    for name in ("world", "screen"):
        image = frame(name == "screen", args.disable == name)
        Image.fromarray(image).save(args.output / f"{name}.png")
        if args.reference:
            expected = np.asarray(Image.open(args.reference / f"{name}.png"))
            delta = np.abs(image.astype("i2") - expected)
            Image.fromarray(delta.astype("u1")).save(args.output / f"{name}-diff.png")
        records = []
        for color in samples[name]:
            pixels = mask(image, color)
            y, x = np.nonzero(pixels)
            assert len(y) >= 20, f"{name} axis {color} missing from its ROI"
            sample = {"count": len(y), "y": [int(y.min()), int(y.max())],
                      "x": [int(x.min()), int(x.max())]}
            if args.reference:
                ref = mask(expected, color)
                sample["iou"] = float((ref & pixels).sum() / (ref | pixels).sum())
                assert sample["iou"] >= 0.90, (name, color, sample)
            records.append(sample)
        if name == "screen":
            assert records[0]["y"] == [108, 108]
        result[name] = records
    (args.output / "axis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
