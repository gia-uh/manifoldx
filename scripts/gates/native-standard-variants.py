"""Check Standard material variants against offscreen reference pixel regions."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, FlatMaterial, StandardMaterial, plane
from PIL import Image

REGIONS = {
    "vertex-color": ((73, 56, 89, 71), (103, 56, 119, 71)),
    "cut": ((91, 55, 99, 73),),
    "mirror": ((84, 81, 109, 98),),
    "opacity": ((83, 54, 103, 74),),
}
WIDTH, HEIGHT = 192, 128


def frame(case, on):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen Standard variants", width=WIDTH, height=HEIGHT, max_entities=12)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine", type(engine)
    engine.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    engine.camera.fov = 45
    engine.background_color = (0.025, 0.045, 0.060)
    engine.set_sun(DirectionalLight("#ffffff", 3.2, (-0.14, -0.17, -1)))

    def mesh(geometry, material, position=(0, 0, 0)):
        engine.spawn(Mesh(geometry), Material(material), Transform(pos=position))

    if case == "vertex-color":
        geometry = plane(2.1, 1.3)
        geometry.pop("uvs")
        geometry["colors"] = np.array(
            [[0.9, 0.16, 0.14], [0.1, 0.25, 0.95], [0.1, 0.25, 0.95], [0.9, 0.16, 0.14]], dtype="f4"
        )
        mesh(geometry, StandardMaterial("#808080", roughness=0.6, vertex_colors=on))
    elif case == "cut":
        mesh(plane(2.7, 1.7), FlatMaterial("#1dbf22"), (0, 0, -0.25))
        material = StandardMaterial("#df5630")
        mesh(plane(2.4, 1.5), material)
        if on:
            engine.render_frame(supersample=1)
            material.cut = (-0.25, -0.2, 0.25, 0.2)
    elif case == "mirror":
        mesh(
            plane(1.25, 0.74), StandardMaterial("#ecd518", mirror=0.0 if on else None), (0, 0.85, 0)
        )
    elif case == "opacity":
        # Spawn the closer translucent pane first. It blends without writing depth,
        # so the next pane still contributes in the prescribed stable batch order.
        mesh(
            plane(1.6, 1.4),
            StandardMaterial("#ec5326", opacity=0.34 if on else 1.0),
            (-0.13, 0, 0.35),
        )
        mesh(
            plane(1.6, 1.4),
            StandardMaterial("#2258f2", opacity=0.52 if on else 1.0),
            (0.13, 0, 0.23),
        )
        mesh(plane(3, 2), FlatMaterial("#317752"), (0, 0, -0.4))
    return engine.render_frame(supersample=1)


def region(image, bounds):
    x0, y0, x1, y1 = bounds
    return image[y0:y1, x0:x1].astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--case", choices=REGIONS)
    parser.add_argument("--disable", choices=REGIONS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outcomes = {}
    for name, bounds in REGIONS.items():
        if args.case and name != args.case:
            continue
        off, on = frame(name, False), frame(name, args.disable != name)
        delta = np.abs(on.astype(np.int16) - off)
        measurement = {
            "roi": bounds,
            "mean_delta": [float(region(delta, b).mean()) for b in bounds],
        }
        for variant, frame_rgb in [("off", off), ("on", on), ("delta", delta.astype("u1"))]:
            filename = f"{name}-{variant}.png"
            Image.fromarray(frame_rgb).save(args.output / filename)
            if args.reference is not None and variant != "delta":
                ref = np.asarray(Image.open(args.reference / filename))
                error = np.abs(frame_rgb.astype(np.int16) - ref)
                measurement[f"{variant}_roi_max"] = [
                    region(error, b).max(axis=(0, 1)).tolist() for b in bounds
                ]
                Image.fromarray(error.astype("u1")).save(
                    args.output / f"{name}-{variant}-error.png"
                )
        outcomes[name] = measurement
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.glob("*.png"))
    }
    (args.output / "standard-variants.json").write_text(
        json.dumps({"cases": outcomes, "sha256": hashes}, indent=2) + "\n"
    )
    print(json.dumps(outcomes))
    for name, result in outcomes.items():
        assert min(result["mean_delta"]) > 10, f"{name} ROI did not respond"
        if args.reference is not None:
            for variant in ("on", "off"):
                assert all(max(error) <= 2 for error in result[f"{variant}_roi_max"]), (
                    name,
                    variant,
                    result,
                )


if __name__ == "__main__":
    main()
