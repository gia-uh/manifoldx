"""Check rigid and blended skinning, clip interpolation and exact ID masks."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import BasicMaterial, plane
from PIL import Image

WIDTH, HEIGHT = 168, 112
CASES = ("rest", "rigid", "weighted", "clip")
ROI = (46, 20, 123, 93)


def frame(case, freeze):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen skinning", width=WIDTH, height=HEIGHT, max_entities=8)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine", type(engine)
    engine.camera.set_pose(position=(0, 0, 4), target=(0, 0, 0))
    engine.camera.fov = 45
    engine.background_color = (0, 0, 0)
    geometry = plane(1.0, 0.8)
    if case in ("weighted", "clip"):
        geometry["joints"] = np.tile([0, 1], (4, 1))
        geometry["weights"] = np.array([0.25, 0.75, 0.75, 0.25], dtype=np.float32)
    else:
        geometry["joints"] = np.array([0, 1, 1, 0])
    mat = BasicMaterial("#e29f59")
    left = engine.spawn(Mesh(geometry), Material(mat), Transform(pos=(-0.57, 0, 0)))
    right = engine.spawn(Mesh(geometry), Material(mat), Transform(pos=(0.65, 0, 0)))
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    poses[0, 1, 3] = 0.52
    poses[1, 1, 3] = -0.26
    poses[1, 2, 0] = 0.20
    poses[1, 0, 0] = 0.78
    if case in ("rigid", "weighted") and not freeze:
        left.set_joints(poses)
    elif case == "clip":
        rest = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        clip = engine.bake_poses(np.stack([rest, poses]))
        engine.set_pose_frames([left.index, right.index], clip, [0, 0] if freeze else [0.5, 0.75])
    beauty = engine.render_frame(supersample=1)
    ids = engine.render_frame(supersample=1, pass_="ids", groups=[[left.index], [right.index]])
    return beauty, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = ROI
    results = {}
    rest_ids = None
    for case in CASES:
        rgb, ids = frame(case, args.freeze)
        if rest_ids is None:
            rest_ids = ids
        roi = ids[y0:y1, x0:x1]
        shifted = np.count_nonzero(roi != rest_ids[y0:y1, x0:x1])
        result = {
            "roi": ROI,
            "changed_label_pixels": int(shifted),
            "pixels_per_label": [int(np.sum(ids == n)) for n in (1, 2)],
        }
        Image.fromarray(rgb).save(args.output / f"{case}.png")
        Image.fromarray(ids.astype("u1") * 100).save(args.output / f"{case}-ids.png")
        np.save(args.output / f"{case}-ids.npy", ids, allow_pickle=False)
        if args.reference:
            ref_ids = np.load(args.reference / f"{case}-ids.npy", allow_pickle=False)
            ref_rgb = np.asarray(Image.open(args.reference / f"{case}.png"))
            diff = np.abs(rgb.astype(np.int16) - ref_rgb)
            Image.fromarray(diff.astype("u1")).save(args.output / f"{case}-diff.png")
            result["id_difference"] = int(np.count_nonzero(ids != ref_ids))
            interior = (ids == ref_ids) & (ids != 0)
            result["interior_error"] = diff[interior].max(axis=0).tolist()
        results[case] = result
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.iterdir())
        if path.suffix in (".png", ".npy")
    }
    (args.output / "skinning.json").write_text(
        json.dumps({"cases": results, "sha256": hashes}, indent=2) + "\n"
    )
    print(json.dumps(results))
    for case, result in results.items():
        if case != "rest":
            assert result["changed_label_pixels"] > 45, f"{case} pose did not move the ID mask"
        assert min(result["pixels_per_label"]) > 150
        if args.reference:
            assert result["id_difference"] == 0, (case, result)
            assert max(result["interior_error"]) <= 2, (case, result)


if __name__ == "__main__":
    main()
