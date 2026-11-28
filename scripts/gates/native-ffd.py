"""Verify a modeling FFD reaches the rendered mesh through set_geometry."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Mesh as GeoMesh
from manifoldx.resources import DirectionalLight, StandardMaterial
from manifoldx.systems import Query
from PIL import Image

WIDTH = HEIGHT = 256
ROI = np.s_[0:220, 20:236]


def scene(moved):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen FFD swap", width=WIDTH, height=HEIGHT, max_entities=8)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine", type(engine)
    engine.set_sun(DirectionalLight("#fff4e0", 3.3, (-0.7, -0.75, -0.4)))
    engine.background_color = (0.1, 0.1, 0.2)

    base = GeoMesh.icosphere(subdivisions=3)
    ffd = base.ffd(resolution=(3, 3, 3))
    lattice = ffd.points.copy()
    if moved:
        points = lattice.copy()
        points[..., 0] += 0.35 * np.sin(1.0 + lattice[..., 1] * 2.2)
        points[..., 1] += 0.30 * np.sin(1.0 + lattice[..., 0] * 2.0)
        points[..., 2] += 0.35 * np.cos(1.0 + lattice[..., 1] * 2.2)
        ffd.points[:] = points

    handle = engine.spawn(
        Mesh(base.to_geometry()),
        Material(StandardMaterial("#8a5cd0", roughness=0.35)),
        Transform(),
    )
    swapped = {"done": False}

    @engine.system
    def swap(query: Query[Transform], dt: float):
        if swapped["done"]:
            return
        swapped["done"] = True
        handle.set_geometry(ffd.apply())  # a modeling Mesh, not a baked dict

    engine.camera.fit(radius=3.0, center=(0, 0, 0), azimuth=20, elevation=10)
    return engine.render_frame(supersample=1)


def mismatched_count_raises():
    engine = mx.Engine("Offscreen FFD count", width=64, height=64, max_entities=4)
    base = GeoMesh.icosphere(subdivisions=2)
    handle = engine.spawn(
        Mesh(base.to_geometry()),
        Material(StandardMaterial("#ffffff")),
        Transform(),
    )
    result = {}

    @engine.system
    def swap(query: Query[Transform], dt: float):
        if result:
            return
        handle.set_geometry(base.with_positions(base.positions + 0.1))
        try:
            handle.set_geometry(GeoMesh.icosphere(subdivisions=3))
        except ValueError as error:
            result["error"] = str(error)
            return
        result["error"] = None

    engine.render_frame(supersample=1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    moved_before = scene(False)
    moved_after = scene(True)
    moved_again = scene(True)
    Image.fromarray(moved_before).save(args.output / "unmoved.png")
    Image.fromarray(moved_after).save(args.output / "moved.png")
    difference = np.abs(moved_before.astype("i2") - moved_after)
    repeat = np.abs(moved_after.astype("i2") - moved_again)
    result = {
        "core": os.environ["MANIFOLDX_CORE"],
        "changed_roi_pixels": int((difference[ROI].max(axis=2) > 8).sum()),
        "changed_roi_mean": round(float(difference[ROI].mean()), 4),
        "repeat_roi_pixels": int((repeat[ROI].max(axis=2) > 8).sum()),
        "mismatch": mismatched_count_raises(),
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.output / "unmoved.png", args.output / "moved.png")
        },
    }
    if args.reference:
        for name in ("unmoved.png", "moved.png"):
            reference = np.asarray(Image.open(args.reference / name))
            ours = np.asarray(Image.open(args.output / name))
            result[f"{name}-roi-mae"] = round(
                float(np.abs(reference.astype("i2") - ours).astype("i2")[ROI].mean()), 4
            )
        Image.fromarray(difference.astype("u1")).save(args.output / "difference.png")
    (args.output / "ffd.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert result["changed_roi_pixels"] > 300, "moving the lattice did not move the mesh"
    assert result["repeat_roi_pixels"] == 0, "the same scene rendered differently"
    assert result["mismatch"]["error"], "a mismatched vertex count was accepted"
    assert result["mismatch"]["error"].startswith("set_geometry requires equal vertex count")
    if args.reference:
        assert result["unmoved.png-roi-mae"] <= 1.0 and result["moved.png-roi-mae"] <= 1.0


if __name__ == "__main__":
    main()
