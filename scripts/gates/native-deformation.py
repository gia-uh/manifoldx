"""Verify active deformation and screen-door fades offscreen."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Fade, Layer, Material, Mesh, Tint, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane
from PIL import Image

WIDTH = HEIGHT = 128
CASES = ("sway", "ripple", "bender", "fade-tint", "fade-shadow")


def scene(case, enabled):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    e = mx.Engine("Offscreen material deformation", width=WIDTH, height=HEIGHT, max_entities=16)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(e).__name__ == "_NativeEngine", type(e)
    e.set_sun(DirectionalLight("#ffffff", 3, (-0.4, -1, -0.5)))
    e.background_color = (0.11, 0.14, 0.22)
    e.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    if case in ("sway", "bender"):
        if case == "sway":
            pole = cube(0.24, 3.0, 0.22)
            pole["positions"] += np.array([0, 1.5, 0], dtype="f4")
            e.camera.set_pose(position=(0, 1.5, 5), target=(0, 1.5, 0))
            mat = StandardMaterial("#cf4b30", sway=0.20 if enabled else 0.0)
        else:
            pole = cube(0.10, 0.7, 0.08)
            pole["positions"] += np.array([0, 0.35, 0], dtype="f4")
            e.camera.set_pose(position=(0, 0.4, 2), target=(0, 0.4, 0))
            mat = StandardMaterial("#080808", sway=1e-4, sway_reach=0.6)
            if enabled:
                e.benders = [(-0.3, 0.0, 0.8, 1.2)]
        e.spawn(Mesh(pole), Material(mat), Transform())
        e.render_frame(supersample=1)
        e.elapsed = 2.5 if enabled and case == "sway" else 1.0
    elif case == "ripple":
        e.camera.set_pose(position=(0, 6, 3), target=(0, 0, 0))
        mesh = {
            "positions": np.array([(-4, 0, -4), (4, 0, -4), (4, 0, 4), (-4, 0, 4)], dtype="f4"),
            "normals": np.tile(np.array([0, 1, 0], dtype="f4"), (4, 1)),
            "indices": np.array([0, 2, 1, 0, 3, 2], dtype="u4"),
        }
        e.spawn(
            Mesh(mesh),
            Material(StandardMaterial("#3f97d6", roughness=0.75, ripple=float(enabled))),
            Transform(),
        )
        e.render_frame(supersample=1)
        e.elapsed = 1.0
    elif case == "fade-tint":
        mesh = plane(1.05, 1.8)
        mesh.pop("uvs")
        mesh["colors"] = np.ones((4, 3), dtype="f4")
        material = StandardMaterial(
            "#ffffff", vertex_colors=True, tinted=enabled, faded=enabled, roughness=0.85
        )
        e.spawn(
            Mesh(mesh),
            Material(material),
            Transform(pos=[(-0.75, 0, 0), (0.75, 0, 0)]),
            Tint(rgb=[(1, 0.14, 0.14), (0.12, 0.12, 1)]),
            Fade(value=[1, 0.5]),
            n=2,
        )
    else:
        e.camera.set_pose(position=(3.4, 3.3, 5.7), target=(0, 0, 0))
        e.camera.layers = {0}
        e.enable_shadows(extent=5.0, resolution=512, auto_fit=False)
        e.spawn(
            Mesh(cube(1.0, 1.4, 1.0)),
            Material(StandardMaterial("#ffffff", faded=True)),
            Transform(pos=(0, 0.10, 0)),
            Fade(value=0.0 if enabled else 1.0),
            Layer(layer=1),
        )
        e.spawn(
            Mesh(cube(7, 0.2, 7)),
            Material(StandardMaterial("#ccbbaa")),
            Transform(pos=(0, -0.65, 0)),
            Layer(layer=0),
        )
    return e.render_frame(supersample=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    roi = np.s_[16:116, 20:108]
    for case in (args.case,) if args.case else CASES:
        before, after = scene(case, False), scene(case, not args.freeze)
        difference = np.abs(before.astype(np.int16) - after)
        result = {"changed_roi_pixels": int((difference[roi].max(axis=2) > 15).sum())}
        for tag, rgb in [("off", before), ("on", after), ("delta", difference.astype("u1"))]:
            filename = f"{case}-{tag}.png"
            Image.fromarray(rgb).save(args.output / filename)
            if args.reference and tag != "delta":
                ref = np.asarray(Image.open(args.reference / filename))
                error = np.abs(ref.astype("i2") - rgb)
                Image.fromarray(error.astype("u1")).save(args.output / f"{case}-{tag}-diff.png")
                result[f"{tag}_roi_mae"] = float(error[roi].mean())
        results[case] = result
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.glob("*.png"))
    }
    (args.output / "deformation.json").write_text(
        json.dumps({"cases": results, "sha256": hashes}, indent=2) + "\n"
    )
    print(json.dumps(results))
    for case, result in results.items():
        assert result["changed_roi_pixels"] > 30, f"{case} did not change its ROI"
        if args.reference:
            assert result["off_roi_mae"] <= 1.0 and result["on_roi_mae"] <= 1.0, (case, result)


if __name__ == "__main__":
    main()
