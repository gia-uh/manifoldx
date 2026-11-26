"""Check colormap sprites, depth, layer filtering, and daylight star fading."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Layer, Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, FlatMaterial, plane
from manifoldx.sky import StarfieldMaterial
from manifoldx.viz import ColormapMaterial, PointCloud, Radius, ScalarValue
from PIL import Image

CASES = ("first", "second", "lit", "occluded", "hidden", "star-night", "star-day")
ROI = np.s_[49:79, 49:79]


def capture(case):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    engine = mx.Engine("Offscreen sprite", width=128, height=128, max_entities=8)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine"
    engine.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    engine.background_color = (0.02, 0.024, 0.036)
    if case.startswith("star"):
        engine.set_sun(DirectionalLight("#ffffff", 2, (0, 0, -1 if case == "star-day" else 1)))
        material = StarfieldMaterial(ground_radius=4.8, atmo_top=1.0)
    else:
        material = ColormapMaterial("viridis", 0, 1, lit=case == "lit")
    if case == "hidden":
        engine.camera.layers = {0}
    engine.spawn(
        PointCloud(), Material(material), ScalarValue(value=0 if case == "first" else 1),
        Radius(radius=0.9), Transform(), Layer(layer=1),
    )
    if case == "occluded":
        engine.spawn(Mesh(plane(2, 2)), Material(FlatMaterial("#315c98")), Transform(pos=(0, 0, 1)))
    return engine.render_frame(supersample=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable", choices=("scalar", "lighting", "depth", "layers", "star-fade"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    substitutions = {
        "scalar": ("second", "first"), "lighting": ("lit", "second"), "depth": ("occluded", "second"),
        "layers": ("hidden", "second"), "star-fade": ("star-day", "star-night"),
    }
    substitute = dict([substitutions[args.disable]]) if args.disable else {}
    images = {case: capture(substitute.get(case, case)) for case in CASES}
    report = {}
    for name, image in images.items():
        path = args.output / f"{name}.png"
        Image.fromarray(image).save(path)
        info = {"center": image[64, 64].tolist()}
        if args.reference:
            ref = np.asarray(Image.open(args.reference / path.name))
            delta = np.abs(image.astype("i2") - ref)
            Image.fromarray(delta.astype("u1")).save(args.output / f"{name}-diff.png")
            info["roi_mae"] = float(delta[ROI].mean())
        report[name] = info
    (args.output / "sprites.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert report["first"]["center"] == [68, 1, 84], "first scalar missed its LUT texel"
    assert report["second"]["center"] == [253, 231, 36], "second scalar missed its LUT texel"
    assert report["lit"]["center"] != report["second"]["center"], "sprite lighting was ignored"
    assert report["occluded"]["center"] == [49, 92, 152], "opaque mesh failed to occlude the sprite"
    background = images["hidden"][0, 0].tolist()
    assert report["hidden"]["center"] == background, "sprite bypassed the view layer filter"
    assert report["star-night"]["center"] == [255, 255, 255], "night star disappeared"
    assert report["star-day"]["center"] == background, "day star failed to fade"
    if args.reference:
        assert max(item["roi_mae"] for item in report.values()) <= 2, report


if __name__ == "__main__":
    main()
