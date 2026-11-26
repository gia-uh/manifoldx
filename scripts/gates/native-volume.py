"""Check DVR color, opacity, local bounds, and same-shape voxel updates."""

import argparse
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Transform
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable", choices=("update", "opacity"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
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
