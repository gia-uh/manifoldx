"""Compare the near, transitioning, and fully fogged RGB regions."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, FlatMaterial, StandardMaterial, plane
from PIL import Image

REGIONS = ((47, 31), (47, 59), (47, 88), (71, 59))


def capture(enabled):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    e = mx.Engine("Distance fog", width=120, height=100, max_entities=16)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(e).__name__ == "_NativeEngine"
    e.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    e.set_sun(DirectionalLight("#fffefe", 3, (0, 0, -1)))
    if enabled:
        e.enable_fog(3.0, 7.0, color=(0.04, 0.32, 0.71))
    material = StandardMaterial("#cf562f")
    for x, z, size in [(-0.93, 3.2, 0.65), (0, 1.2, 1.10), (2.55, -1.6, 1.75)]:
        e.spawn(Mesh(plane(size, size)), Material(material), Transform(pos=(x, 0.15, z)))
    e.spawn(Mesh(plane(0.6, 0.6)), Material(FlatMaterial("#d0874e")), Transform(pos=(0, -1.2, 1.2)))
    return e.render_frame(supersample=1)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("output", type=Path)
    args.add_argument("--reference", type=Path)
    args.add_argument("--disable", action="store_true")
    opt = args.parse_args()
    opt.output.mkdir(parents=True, exist_ok=True)
    outputs = {name: capture(active) for name, active in (("off", False), ("on", not opt.disable))}
    rois = [np.s_[y-2:y+3, x-2:x+3] for y, x in REGIONS]
    report = {}
    for name, pixels in outputs.items():
        Image.fromarray(pixels).save(opt.output / f"{name}.png")
        report[name] = [pixels[roi].mean(axis=(0, 1)).round(2).tolist() for roi in rois]
        if opt.reference:
            expected = np.asarray(Image.open(opt.reference / f"{name}.png"))
            delta = np.abs(pixels.astype("i2") - expected)
            Image.fromarray(delta.astype("u1")).save(opt.output / f"{name}-diff.png")
            report[f"{name}_max"] = [int(delta[roi].max()) for roi in rois]
    difference = np.abs(outputs["on"].astype("i2") - outputs["off"])
    Image.fromarray(difference.astype("u1")).save(opt.output / "delta.png")
    report["effect_max"] = [int(difference[roi].max()) for roi in rois]
    (opt.output / "fog.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    near, transition, far, flat = report["effect_max"]
    assert near == flat == 0 and 25 < transition < far, "distance fog missed its ROI"
    if opt.reference:
        assert max(report["off_max"] + report["on_max"]) <= 2, report


if __name__ == "__main__":
    main()
