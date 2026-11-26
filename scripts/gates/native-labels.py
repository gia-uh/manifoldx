"""Check world/screen TextLabel anchors and native atlas layers."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Transform
from manifoldx.viz import LabelMaterial, TextLabel
from PIL import Image

CASES = ("world", "screen", "legend", "empty")


def frame(case):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    e = mx.Engine("Offscreen TextLabel", width=160, height=112, max_entities=8)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(e).__name__ == "_NativeEngine"
    e.background_color = (0.014, 0.014, 0.014)
    e.camera.set_pose(position=(0, 0, 4), target=(0, 0, 0))
    atlas = e.get_label_atlas()
    index = 0
    if case != "empty":
        index = atlas.get_or_create("E5 labels", font_size=36)
    if case == "legend":
        index = atlas.register_colormap_legend("viridis")
    screen = case in ("screen", "legend")
    e.spawn(
        TextLabel(index=index),
        Material(LabelMaterial(pixel_width=128, pixel_height=32, anchor_mode="screen" if screen else "world")),
        Transform(pos=(0, 0.4, 0) if screen else (0, 0, 0)),
    )
    return e.render_frame(supersample=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable", choices=("world", "screen", "legend"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in CASES:
        image = frame("empty" if args.disable == name else name)
        Image.fromarray(image).save(args.output / f"{name}.png")
        background = image[0, 0]
        changed = np.max(np.abs(image.astype("i2") - background), axis=2) > 12
        y, x = np.nonzero(changed)
        record = {"changed": len(y)}
        if len(y):
            record["bbox"] = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
        if args.reference:
            expected = np.asarray(Image.open(args.reference / f"{name}.png"))
            delta = np.abs(image.astype("i2") - expected)
            Image.fromarray(delta.astype("u1")).save(args.output / f"{name}-diff.png")
            record["roi_mae"] = float(delta[16:76, 10:150].mean())
        result[name] = record
    (args.output / "labels.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert result["world"]["changed"] > 120, "world glyph ROI is empty"
    assert result["screen"]["changed"] > 120, "screen glyph ROI is empty"
    assert result["world"]["bbox"][1] - result["screen"]["bbox"][1] >= 19
    assert result["legend"]["changed"] > 3500, "legend sampled the wrong atlas layer"
    assert result["empty"]["changed"] == 0
    if args.reference:
        assert max(record["roi_mae"] for record in result.values()) <= 1.5, result


if __name__ == "__main__":
    main()
