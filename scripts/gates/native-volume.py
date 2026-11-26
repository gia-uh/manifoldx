"""Check DVR color, opacity, local bounds, and same-shape voxel updates."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import FlatMaterial, material_rgb, plane
from manifoldx.viz import Volume, VolumeMaterial
from PIL import Image

ROI = np.s_[28:68, 43:96]


def capture(disabled):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    e = mx.Engine("Offscreen DVR", width=144, height=96, max_entities=8)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(e).__name__ == "_NativeEngine"
    e.background_color = (0.018, 0.028, 0.044)
    e.camera.set_pose(position=(0.25, 0.2, 4), target=(0, 0, 0))
    z, y, x = np.mgrid[-1:1:16j, -1:1:20j, -1:1:24j]
    base = np.exp(-7 * (x*x + y*y + z*z)).astype("f4")
    updated = np.exp(-7 * ((x-0.42)**2 + y*y + z*z)).astype("f4")
    handle = e.register_volume(base, name="gaussian")
    density = 0 if disabled == "opacity" else 5
    material = VolumeMaterial(
        "magma", density_scale=density, opacity_stops=[(0.0, 0.0), (0.25, 0.1), (1.0, 0.95)]
    )
    e.spawn(Volume(volume_id=handle), Material(material), Transform(scale=(1.6, 1.5, 1.25)))
    frames = {"original": e.render_frame(supersample=1)}
    if disabled != "update":
        e.update_volume(handle, updated)
    frames["updated"] = e.render_frame(supersample=1)
    return frames



SIZES = ((160, 112), (213, 131))
OCCLUDER = "#4567ab"
BOUNDS = {"behind": (-0.9, 1), "through": (0, 2), "segment": (0.5, 1), "front": (0.9, 1)}
REFERENCES = {
    "opaque": (None, True, False),
    "clear": (None, False, False),
    "behind-uncovered": ("behind", False, False),
    "isolated": ("through", False, False),
    "segment": ("segment", True, False),
    "front-uncovered": ("front", False, False),
    "front-over-solid-color": ("front", False, True),
}
CASES = {"behind": ("behind", True, False), "through": ("through", True, False),
         "front": ("front", True, False), "isolated": REFERENCES["isolated"]}


def depth_scene(volume, opaque, matching_clear):
    engine = mx.Engine("Offscreen volume depth", width=160, height=112, max_entities=5)
    assert type(engine).__name__ == "_NativeEngine"
    engine.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    engine.camera.near, engine.camera.far, engine.camera.fov = 0.27, 70.0, 45.0
    engine.background_color = material_rgb(OCCLUDER).tolist() if matching_clear else (0.02, 0.033, 0.05)
    engine.world.register(Volume)
    if opaque:
        # Leave part of the volume uncovered at x > 0.07.
        engine.spawn(Mesh(plane(1.10, 3.2)), Material(FlatMaterial(OCCLUDER)),
                     Transform(pos=(-0.48, 0, 0)))
    if volume:
        center, depth = BOUNDS[volume]
        handle = engine.register_volume(np.full((16, 12, 16), 0.8, dtype="f4"), name="uniform")
        mat = VolumeMaterial("magma", density_scale=0.85, step_size=0.025,
                             opacity_stops=[(0, 0), (1, 0.8)])
        engine.spawn(Volume(volume_id=handle), Material(mat),
                     Transform(pos=(0, 0, center), scale=(3.2, 2.5, depth)))
    return engine


def depth_frames(configurations, output):
    frames = {}
    for name, config in configurations.items():
        engine = depth_scene(*config)
        for width, height in SIZES:
            engine.w, engine.h = width, height
            image = engine.render_frame(supersample=1)
            assert image.shape == (height, width, 3)
            key = f"{name}-{width}x{height}"
            frames[key] = image
            Image.fromarray(image).save(output / f"{key}.png")
    return frames


def depth_regions(width, height):
    engine = depth_scene(None, False, False)
    points, _ = engine.camera.project([(-0.48, 0, 0), (0.59, 0, 0)], width, height)
    regions = []
    for x, y in np.rint(points).astype(int):
        regions.append(np.s_[y-7:y+8, x-6:x+7])
    return regions


def verify_occlusion(output, reference):
    frames = depth_frames(CASES, output)
    report = {}
    for width, height in SIZES:
        suffix = f"{width}x{height}"
        cover, opening = depth_regions(width, height)

        def frozen(name):
            return np.asarray(Image.open(reference / f"{name}-{suffix}.png"))

        def measure(case, oracle, selection, label):
            image = frames[f"{case}-{suffix}"]
            delta = np.abs(image.astype("i2") - oracle)
            Image.fromarray(delta.astype("u1")).save(output / f"{label}-{suffix}-diff.png")
            return float(np.max(delta[selection]))

        # Behind is compared to the solid alone; a crossing volume is compared
        # to a box containing only the segment in front of the opaque plane.
        row = {
            "behind": measure("behind", frozen("opaque"), cover, "behind"),
            "crossing": measure("through", frozen("segment"), cover, "crossing"),
            "front": measure("front", frozen("front-over-solid-color"), cover, "front"),
            "isolated": measure("isolated", frozen("isolated"), np.s_[:, :], "isolated"),
        }
        for case, oracle in (("behind", "behind-uncovered"), ("through", "isolated"),
                             ("front", "front-uncovered")):
            row[f"uncovered_{case}"] = measure(case, frozen(oracle), opening, f"uncovered-{case}")
        row["crossing_signal"] = float(np.abs(
            frames[f"through-{suffix}"][cover].astype("i2") - frozen("opaque")[cover]
        ).mean())
        row["front_signal"] = float(np.abs(
            frames[f"front-{suffix}"][cover].astype("i2") - frozen("opaque")[cover]
        ).mean())
        row["background_ray_signal"] = float(np.abs(
            frames[f"through-{suffix}"][opening].astype("i2") - frozen("clear")[opening]
        ).mean())
        report[suffix] = row
    (output / "occlusion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))

    for size, row in report.items():
        assert row["behind"] == 0, f"{size}: volume behind opaque geometry leaked"
        assert row["crossing"] <= 3 and row["crossing_signal"] > 10, f"{size}: incorrect visible ray segment"
        assert row["front"] <= 1 and row["front_signal"] > 10, f"{size}: volume in front was clipped"
        assert row["isolated"] <= 1, f"{size}: isolated volume changed"
        assert all(row[f"uncovered_{case}"] <= 1 for case in ("behind", "through", "front")), (
            f"{size}: background depth occluded an uncovered ray"
        )
        assert row["background_ray_signal"] > 10, f"{size}: uncovered region did not draw"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable", choices=("update", "opacity"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--occlusion", action="store_true")
    modes.add_argument("--freeze-occlusion-reference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.freeze_occlusion_reference or args.occlusion:
        assert sys.flags.isolated and Path(mx.__file__).resolve().is_relative_to(sys.prefix)
        if args.freeze_occlusion_reference:
            depth_frames(REFERENCES, args.output)
        else:
            assert args.reference is not None, "--occlusion requires a frozen reference"
            verify_occlusion(args.output, args.reference)
        return
    frames = capture(args.disable)
    result = {}
    for name, img in frames.items():
        Image.fromarray(img).save(args.output / f"{name}.png")
        change = np.abs(img.astype("i2") - img[0, 0])
        result[name] = {"bounds_pixels": int((change[ROI].max(axis=-1) > 12).sum())}
        if args.reference:
            expected = np.asarray(Image.open(args.reference / f"{name}.png"))
            delta = np.abs(img.astype("i2") - expected)
            Image.fromarray(delta.astype("u1")).save(args.output / f"{name}-diff.png")
            result[name]["roi_mae"] = float(delta[ROI].mean())
    movement = np.abs(frames["updated"].astype("i2") - frames["original"])
    result["moved"] = int((movement[ROI].max(axis=-1) > 12).sum())
    Image.fromarray(movement.astype("u1")).save(args.output / "update-delta.png")
    (args.output / "volume.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert result["original"]["bounds_pixels"] > 120, "opacity or volume data did not draw"
    assert result["moved"] > 100, "voxel replacement did not change its ROI"
    if args.reference:
        assert max(result[k]["roi_mae"] for k in frames) <= 1.0, result


if __name__ == "__main__":
    main()
