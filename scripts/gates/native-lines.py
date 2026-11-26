"""Verify native physical-width lines and reverse-depth occlusion."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import FlatMaterial, plane
from PIL import Image


def scene(size, distance, occluder=False):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen lines", width=size, height=size, max_entities=8)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine", type(engine)
    engine.camera.set_pose(position=(0, 0, distance), target=(0, 0, 0))
    engine.spawn(Mesh(plane(400, 400)), Material(FlatMaterial("#22cc22")), Transform())
    engine.lines.set("road", [(-50, 0, 0)], [(50, 0, 0)], color=(1, 0, 0, 1), width=6)
    if occluder:
        engine.spawn(Mesh(plane(2, 2)), Material(FlatMaterial("#2222cc")), Transform(pos=(0, 0, 3)))
    return engine


def reds(rgb, column):
    return int(((rgb[:, column, 0] > 200) & (rgb[:, column, 1] < 120)).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--skip-line", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    near = scene(128, 10)
    near.lines.visible = not args.skip_line
    images = {"near": near.render_frame(supersample=1)}
    far = scene(256, 60)
    images["far"] = far.render_frame(supersample=1)
    hidden = scene(128, 10, occluder=True)
    images["occluded"] = hidden.render_frame(supersample=1)
    near.lines.look("road", scale=2.0)
    images["scaled"] = near.render_frame(supersample=1)
    near.lines.look("road", alpha=0.0)
    images["transparent"] = near.render_frame(supersample=1)
    near.lines.remove("road")
    images["removed"] = near.render_frame(supersample=1)
    report = {
        "core_widths": {k: reds(v, v.shape[1] // 2) for k, v in images.items()},
        "occluded_side_width": reds(images["occluded"], 5),
    }
    hashes = {}
    for name, rgb in images.items():
        path = args.output / f"{name}.png"
        Image.fromarray(rgb).save(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if args.reference:
            ref = np.asarray(Image.open(args.reference / path.name))
            error = np.abs(rgb.astype("i2") - ref)
            col = rgb.shape[1] // 2
            report[f"{name}_max_channel_error"] = int(error[:, col - 5 : col + 5].max())
            Image.fromarray(error.astype("u1")).save(args.output / f"{name}-diff.png")
    (args.output / "lines.json").write_text(
        json.dumps({"measurement": report, "sha256": hashes}, indent=2) + "\n"
    )
    print(json.dumps(report))
    assert report["core_widths"] == {
        "near": 6,
        "far": 6,
        "occluded": 0,
        "scaled": 12,
        "transparent": 0,
        "removed": 0,
    }, "physical line width or depth differs"
    assert report["occluded_side_width"] == 6
    if args.reference:
        assert max(v for k, v in report.items() if k.endswith("_max_channel_error")) <= 2


if __name__ == "__main__":
    main()
