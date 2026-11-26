"""Verify the physical Rect path before the GUI runtime is ported."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.gui.material import RectMaterial
from manifoldx.gui.painter import RectOp
from manifoldx.gui.layout import LayoutBox
from manifoldx.gui.style import parse_color
from manifoldx.post import Grade
from manifoldx.resources import FlatMaterial, plane
from PIL import Image

WIDTH, HEIGHT = 160, 104
BOX = (10, 12, 52, 31)
FILL, BORDER = "#e45125", "#ffdd76"
RADIUS, BORDER_WIDTH = 6, 3


def frame(post, early):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen Rect", width=WIDTH, height=HEIGHT, max_entities=6)
    engine.background_color = (0.05, 0.20, 0.32)
    engine.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    engine.spawn(Mesh(plane(2, 1.6)), Material(FlatMaterial("#1289de")), Transform())
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine", type(engine)
        op = RectOp(LayoutBox(*BOX), parse_color(FILL), parse_color(BORDER), BORDER_WIDTH, RADIUS)
        engine._graphics.fixture_rects(
            RectMaterial.pack_instances([op], exact=True).tolist(), early
        )
    else:
        from manifoldx.gui import Panel

        engine.gui.exact_colors = True
        engine.gui.append(
            Panel(
                children=[],
                anchor="top-left",
                offset=BOX[:2],
                style_overrides={
                    "width": BOX[2],
                    "height": BOX[3],
                    "bg": FILL,
                    "radius": RADIUS,
                    "border": BORDER_WIDTH,
                    "border_color": BORDER,
                },
            )
        )
    if post:
        engine.set_post(grade=Grade(saturation=0.0), fxaa=True)
    pixels = engine.render_frame(supersample=1)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        expected = (
            ["rect", "composite", "fxaa"]
            if early and post
            else (["composite", "fxaa", "rect"] if post else ["rect"])
        )
        assert engine.last_frame["post_order"] == expected, engine.last_frame
    return pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--before-post", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    off, on = frame(False, args.before_post), frame(True, args.before_post)
    rect = np.s_[20:35, 20:50]
    border = np.s_[22:33, 11:13]
    edge = np.s_[22:33, 10:11]
    scene = np.s_[48:60, 72:83]
    outside = np.s_[14:28, 90:114]
    report = {
        "fill_off": off[26, 25].tolist(),
        "fill_on": on[26, 25].tolist(),
        "border_off": off[24, 11].tolist(),
        "border_on": on[24, 11].tolist(),
        "scene_delta": float(np.abs(off[scene].astype("i2") - on[scene]).mean()),
        "outside_delta": float(np.abs(off[outside].astype("i2") - on[outside]).mean()),
    }
    hashes = {}
    for name, rgb in [("off", off), ("on", on)]:
        path = args.output / f"{name}.png"
        Image.fromarray(rgb).save(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if args.reference:
            reference = np.asarray(Image.open(args.reference / f"{name}.png"))
            error = np.abs(rgb.astype("i2") - reference)
            Image.fromarray(error.astype("u1")).save(args.output / f"{name}-diff.png")
            report[f"{name}_max_roi_error"] = [
                int(error[region].max()) for region in (rect, border, edge, scene, outside)
            ]
    (args.output / "rect.json").write_text(
        json.dumps({"measurement": report, "sha256": hashes}, indent=2) + "\n"
    )
    print(json.dumps(report))
    assert np.array_equal(off[rect], on[rect]) and np.array_equal(off[border], on[border]), (
        "post touched the Rect"
    )
    assert report["fill_on"] == [228, 81, 37] and report["border_on"] == [255, 221, 118]
    assert not np.array_equal(off[edge], on[edge]), (
        "partially covered edge ignored the underlying scene"
    )
    assert report["scene_delta"] > 20 and report["outside_delta"] > 20
    if args.reference:
        assert max(report["off_max_roi_error"] + report["on_max_roi_error"]) <= 2


if __name__ == "__main__":
    main()
