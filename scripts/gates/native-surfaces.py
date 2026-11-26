"""Compare Standard surface patterns and procedural roughness with frozen RGB."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import SURFACE_PATTERNS, DirectionalLight, StandardMaterial, plane
from PIL import Image

CASES = ("patterns", "override", "far", "detail")
PATTERNS = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def capture(case, active, *, enlarge_cells=False):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    e = mx.Engine("Offscreen surface patterns", width=240, height=240, max_entities=16)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(e).__name__ == "_NativeEngine"
    e.background_color = (0.02, 0.024, 0.03)
    e.set_sun(DirectionalLight("#ffffff", 2.8, (-0.1, -0.4, -1)))
    if case == "detail":
        e.camera.set_pose(position=(0, 3.5, 3), target=(0, 0, 0))
        mat = StandardMaterial("#ffffff", roughness=0.54, vertex_colors=True, detail=active)
        for i, sid in enumerate((10, 11)):
            mesh = plane(2.65, 5.0)
            mesh.pop("uvs")
            mesh["surfaces"] = np.full(4, sid, dtype="f4")
            mesh["colors"] = np.tile(np.array((0.46, 0.52, 0.59), dtype="f4"), (4, 1))
            rotation = (-math.sqrt(0.5), 0, 0, math.sqrt(0.5))
            e.spawn(Mesh(mesh), Material(mat), Transform(pos=(-1.4 + i * 2.8, 0, 0), rot=rotation))
    else:
        distance = 135 if case == "far" else 6
        e.camera.far = 350
        e.camera.set_pose(position=(0, 0, distance), target=(0, 0, 0))
        size = 2.05 * distance / 6
        # Keep the screen geometry fixed while varying projected cell size.
        # An override changes the first pattern family in the same material.
        settings = {1: (0.73, 0.55, 2, 0.09, 0.82)} if case == "override" and active else None
        if enlarge_cells:
            settings = {i: (values[0] * 22.5, *values[1:]) for i, values in SURFACE_PATTERNS.items()}
        mat = StandardMaterial("#ffffff", roughness=0.58, vertex_colors=True, surface_patterns=settings)
        for i, sid in enumerate(PATTERNS):
            mesh = plane(size, size)
            mesh.pop("uvs")
            surface = sid if active or case == "override" else 0
            mesh["surfaces"] = np.full(4, surface, dtype="f4")
            mesh["colors"] = np.tile(np.array((0.57, 0.60, 0.66), dtype="f4"), (4, 1))
            center = ((i % 3 - 1) * (size + 0.12), (1 - i // 3) * (size + 0.12), 0)
            e.spawn(Mesh(mesh), Material(mat), Transform(pos=center))
    return e.render_frame(supersample=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable", choices=CASES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for case in CASES:
        force_larger_cells = case == args.disable == "far"
        off = capture(case, False)
        on = capture(case, args.disable != case or force_larger_cells, enlarge_cells=force_larger_cells)
        delta = np.abs(on.astype("i2") - off)
        zones = [np.s_[88:170, 35:110], np.s_[88:170, 130:205]] if case == "detail" else [
            np.s_[y-22:y+23, x-22:x+23] for y in (45, 120, 195) for x in (45, 120, 195)
        ]
        info = {"changed": [int((delta[roi].max(axis=2) > 3).sum()) for roi in zones]}
        for key, image in (("off", off), ("on", on), ("delta", delta.astype("u1"))):
            name = f"{case}-{key}.png"
            Image.fromarray(image).save(args.output / name)
            if args.reference and key != "delta":
                expected = np.asarray(Image.open(args.reference / name))
                difference = np.abs(image.astype("i2") - expected)
                Image.fromarray(difference.astype("u1")).save(args.output / f"{case}-{key}-diff.png")
                info[f"{key}_mae"] = [float(difference[roi].mean()) for roi in zones]
        report[case] = info
    (args.output / "surfaces.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert min(report["patterns"]["changed"]) > 35, "a surface pattern failed its ROI"
    assert report["override"]["changed"][0] > 35 and max(report["override"]["changed"][1:]) == 0
    assert max(report["far"]["changed"]) < 10, "distant patterns did not fade"
    assert min(report["detail"]["changed"]) > 60, "procedural detail failed its ROI"
    if args.reference:
        assert all(max(info["off_mae"] + info["on_mae"]) <= 2 for info in report.values()), report


if __name__ == "__main__":
    main()
