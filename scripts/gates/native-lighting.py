"""Capture direct Standard GGX lighting offscreen with frozen pixel regions."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, PointLight, SpotLight, StandardMaterial, plane
from PIL import Image

WIDTH, HEIGHT = 192, 128
ROIS = {"center": (92, 60, 100, 68), "penumbra": (113, 60, 117, 68), "outside": (129, 60, 135, 68)}
CASES = ("off", "point-near", "point-far", "sun", "sun-behind", "spot", "spot-beyond", "combined")
MAX_ERROR = 2


def capture(case, disabled):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen Standard lighting", width=WIDTH, height=HEIGHT, max_entities=8)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine", type(engine)
    engine.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    engine.camera.fov = 45
    engine.background_color = (0, 0, 0)
    engine.spawn(
        Mesh(plane(3.5, 2)), Material(StandardMaterial("#8b9db1", 0.55, 0.15, ao=0.8)), Transform()
    )
    if case in ("point-near", "point-far", "combined") and disabled != "point":
        engine.set_lights([PointLight("#ffc684", 8, (0, 0, 2 if case != "point-far" else 5))])
    if case in ("sun", "sun-behind", "combined") and disabled != "sun":
        engine.set_sun(
            DirectionalLight("#77c1ff", 2.2, (0.1, 0.12, 1 if case == "sun-behind" else -1))
        )
    if case in ("spot", "spot-beyond", "combined") and disabled != "spot":
        engine.set_spot(
            SpotLight(
                "#ffaf8a",
                12,
                (0, 0, 2.2),
                (0, 0, -1),
                0.18,
                0.36,
                distance=1.8 if case == "spot-beyond" else 5,
            )
        )
    return engine.render_frame(supersample=1)


def roi(image, region):
    x0, y0, x1, y1 = region
    return image[y0:y1, x0:x1].astype(np.int16)


def contrast(first, second, region):
    return float(np.mean(roi(first, ROIS[region]) - roi(second, ROIS[region])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable", choices=("point", "sun", "spot"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = {name: capture(name, args.disable) for name in CASES}
    for name, image in frames.items():
        Image.fromarray(image).save(args.output / f"{name}.png")
    differences = {
        "point_near_minus_far": contrast(frames["point-near"], frames["point-far"], "center"),
        "sun_front_minus_back": contrast(frames["sun"], frames["sun-behind"], "center"),
        "spot_center_minus_off": contrast(frames["spot"], frames["off"], "center"),
        "spot_penumbra_minus_off": contrast(frames["spot"], frames["off"], "penumbra"),
        "spot_outside_minus_off": contrast(frames["spot"], frames["off"], "outside"),
        "spot_beyond_minus_off": contrast(frames["spot-beyond"], frames["off"], "center"),
    }
    errors = {}
    if args.reference is not None:
        for name, image in frames.items():
            expected = np.asarray(Image.open(args.reference / f"{name}.png"))
            diff = np.abs(image.astype(np.int16) - expected)
            Image.fromarray(diff.astype(np.uint8)).save(args.output / f"diff-{name}.png")
            errors[name] = {
                tag: roi(diff, bounds).max(axis=(0, 1)).tolist() for tag, bounds in ROIS.items()
            }
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.glob("*.png"))
    }
    data = {"roi": ROIS, "differences": differences, "roi_errors": errors, "sha256": hashes}
    (args.output / "lighting.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data))
    assert differences["point_near_minus_far"] > 20, "point attenuation absent"
    assert differences["sun_front_minus_back"] > 20, "sun direction response absent"
    assert differences["spot_center_minus_off"] > 20, "spot center absent"
    assert 3 < differences["spot_penumbra_minus_off"] < differences["spot_center_minus_off"] - 5, (
        "spot cone profile absent"
    )
    assert abs(differences["spot_outside_minus_off"]) <= 2, "spot leaks outside its cone"
    assert abs(differences["spot_beyond_minus_off"]) <= 2, "spot exceeds its range"
    assert all(max(error) <= MAX_ERROR for row in errors.values() for error in row.values()), errors


if __name__ == "__main__":
    main()
