"""Verify offscreen direct-light shadows and cascades in fixed receiver regions."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, SpotLight, StandardMaterial, cube
from PIL import Image

WIDTH, HEIGHT = 192, 144
CASES = ("sun", "spot", "cascades")
ROI = (45, 67, 151, 115)


def frame(case, shadows):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen shadow", width=WIDTH, height=HEIGHT, max_entities=20)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine", type(engine)
    engine.camera.set_pose(position=(3.4, 3.3, 5.7), target=(0, 0, 0))
    engine.camera.fov, engine.camera.near, engine.camera.far = 45, 0.15, 50
    engine.background_color = (0.04, 0.04, 0.06)
    engine.spawn(
        Mesh(cube(7.5, 0.2, 7.5)),
        Material(StandardMaterial("#c1b7a0")),
        Transform(pos=(0, -0.74, 0)),
    )
    engine.spawn(
        Mesh(cube(1.05, 1.5, 1.05)),
        Material(StandardMaterial("#734e3b")),
        Transform(pos=(0, 0.1, 0)),
    )
    engine.spawn(
        Mesh(cube(0.6, 1, 0.6)),
        Material(StandardMaterial("#40527a")),
        Transform(pos=(-1.65, -0.13, -0.7)),
    )
    if case == "spot":
        engine.set_spot(
            SpotLight("#fff1d0", 80, (-2, 4.6, -2), (2, -4.6, 2), 0.57, 0.78, distance=14)
        )
    else:
        engine.set_sun(DirectionalLight("#fff1d0", 4.5, (0.65, -1, 0.43)))
    if shadows:
        engine.enable_shadows(
            extent=5.5,
            resolution=512,
            near=0.1,
            far=40,
            pcf_radius=1,
            bias=0.00016 if case == "spot" else 0.005,
            auto_fit=False,
            cascades=3 if case == "cascades" else 1,
            shadow_distance=18,
        )
    pixels = engine.render_frame(supersample=1)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        expected = (3 if case == "cascades" else 1) if shadows else 0
        assert engine.last_frame["shadow_layers"] == expected, engine.last_frame
    return pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--disable", choices=CASES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = ROI
    results = {}
    for case in (args.case,) if args.case else CASES:
        off, on = frame(case, False), frame(case, args.disable != case)
        delta = off.astype(np.int16) - on
        roi = delta[y0:y1, x0:x1]
        result = {
            "roi": ROI,
            "mean_darkening": float(roi.mean()),
            "darkened_pixels": int((roi.mean(axis=2) > 12).sum()),
        }
        for tag, pixels in [("off", off), ("on", on), ("delta", np.abs(delta).astype("u1"))]:
            Image.fromarray(pixels).save(args.output / f"{case}-{tag}.png")
        if args.reference:
            historical = np.asarray(Image.open(args.reference / f"{case}-on.png"))
            diff = np.abs(on.astype(np.int16) - historical)
            Image.fromarray(diff.astype("u1")).save(args.output / f"{case}-diff.png")
            result["roi_error_mean"] = float(diff[y0:y1, x0:x1].mean())
            mask = roi.mean(axis=2) > 12
            ref_mask = (off.astype(np.int16) - historical)[y0:y1, x0:x1].mean(axis=2) > 12
            union = np.count_nonzero(mask | ref_mask)
            result["shadow_mask_iou"] = float(np.count_nonzero(mask & ref_mask) / max(1, union))
        results[case] = result
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.glob("*.png"))
    }
    (args.output / "shadow.json").write_text(
        json.dumps({"cases": results, "sha256": hashes}, indent=2) + "\n"
    )
    print(json.dumps(results))
    for case, result in results.items():
        assert result["mean_darkening"] > 1 and result["darkened_pixels"] > 40, (
            f"{case} shadow ROI did not darken"
        )
        if args.reference:
            assert result["roi_error_mean"] <= 4 and result["shadow_mask_iou"] >= 0.9, (
                case,
                result,
            )


if __name__ == "__main__":
    main()
