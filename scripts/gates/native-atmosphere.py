"""Compare the native water, sea, glow, scattering and cloud mesh routes."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import (
    AtmosphereMaterial,
    AtmosphereScatteringMaterial,
    CloudMaterial,
    DirectionalLight,
    FlatMaterial,
    SeaMaterial,
    WaterMaterial,
    plane,
    sphere,
)
from PIL import Image

CASES = ("water", "sea", "glow", "scattering", "cloud")
ROIS = {
    "water": np.s_[28:92, 37:123],
    "sea": np.s_[70:118, 42:118],
    "glow": np.s_[25:95, 37:123],
    "scattering": np.s_[10:110, 27:133],
    "cloud": np.s_[28:92, 37:123],
}


def capture(case, enabled):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    e = mx.Engine("Offscreen atmospheric material", width=160, height=120, max_entities=16)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(e).__name__ == "_NativeEngine"
    e.background_color = (0.012, 0.024, 0.046)
    e.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    e.set_sun(DirectionalLight("#ffffff", 1.5, (-0.2, -0.32, -1)))
    if case == "water":
        m = WaterMaterial("#174584" if enabled else "#521b15", fresnel_power=3)
        e.spawn(Mesh(sphere(1.5, 48)), Material(m), Transform())
    elif case == "sea":
        e.camera.far = 20_000
        e.camera.set_pose(position=(0, 4, 0), target=(0, 3, -100))
        e.set_sun(DirectionalLight("#ffffff", 1.5, (0, -0.08, 1)))
        m = SeaMaterial(waves=(0.02, 6, 1, 1.0 if enabled else 0.0))
        quarter_turn = (-math.sqrt(0.5), 0, 0, math.sqrt(0.5))
        e.spawn(Mesh(plane(20_000, 20_000)), Material(m), Transform(rot=quarter_turn))
    elif case == "scattering":
        e.camera.set_pose(position=(0, 0, 11.1), target=(0, 0.1, 0))
        e.set_sun(DirectionalLight("#ffffff", 1.5, (0.18, 0.08, -1)))
        m = AtmosphereScatteringMaterial(9, 10.8, intensity=24 if enabled else 0)
        e.spawn(Mesh(sphere(9, 48)), Material(FlatMaterial("#181d22")), Transform())
        e.spawn(Mesh(sphere(10.8, 64)), Material(m), Transform())
    else:
        e.spawn(Mesh(sphere(1.36, 48)), Material(FlatMaterial("#24486b")), Transform())
        if case == "cloud":
            m = CloudMaterial(coverage=0.44, opacity=0.85 if enabled else 0)
        else:
            m = AtmosphereMaterial("#70baff", intensity=0.72 if enabled else 0)
        e.spawn(Mesh(sphere(1.50, 60)), Material(m), Transform())
    return e.render_frame(supersample=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--disable", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in (args.case,) if args.case else CASES:
        off, on = capture(case, False), capture(case, not args.disable)
        delta = np.abs(on.astype("i2") - off)
        roi = ROIS[case]
        result = {"pixels_changed_gt_12": int((delta[roi].max(axis=2) > 12).sum())}
        for variant, img in (("off", off), ("on", on), ("delta", delta.astype("u1"))):
            name = f"{case}-{variant}.png"
            Image.fromarray(img).save(args.output / name)
            if args.reference and variant != "delta":
                ref = np.asarray(Image.open(args.reference / name))
                error = np.abs(img.astype("i2") - ref)
                Image.fromarray(error.astype("u1")).save(args.output / f"{case}-{variant}-diff.png")
                result[f"{variant}_mae"] = float(error[roi].mean())
        results[case] = result
    (args.output / "atmosphere.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
    for case, result in results.items():
        assert result["pixels_changed_gt_12"] > 30, f"{case} failed to change its ROI"
        if args.reference:
            assert result["on_mae"] <= 2 and result["off_mae"] <= 2, (case, result)


if __name__ == "__main__":
    main()
