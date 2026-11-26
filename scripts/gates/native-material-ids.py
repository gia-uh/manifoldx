"""Check fixed material colours, group labels and reverse depth offscreen."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import BasicMaterial, FlatMaterial, PointLight, plane
from PIL import Image

WIDTH, HEIGHT = 160, 100
ROIS = ((67, 37), (93, 37), (67, 63), (93, 63))
EXPECTED_RGB = ((109, 136, 164), (72, 91, 111), (128, 160, 192), (128, 160, 192))
LABELS = (1, 0x0102)


def scene():
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen Basic, Flat and IDs", width=WIDTH, height=HEIGHT, max_entities=16)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine", type(engine)
    engine.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    engine.camera.fov = 45
    engine.background_color = (0.08, 0.02, 0.12)
    entities = []
    for y, material in [(0.55, BasicMaterial), (-0.55, FlatMaterial)]:
        for x, normal in [(-0.55, (0, 0, 1)), (0.55, (0, 0, -1))]:
            mesh = plane(0.8, 0.8)
            mesh["normals"][:] = normal
            entities.append(
                engine.spawn(
                    Mesh(mesh), Material(material("#80a0c0")), Transform(pos=(x, y, 0))
                ).index
            )
    # A closer, unlabelled mesh still writes reverse depth over part of group 1.
    entities.append(
        engine.spawn(
            Mesh(plane(0.2, 0.2)),
            Material(FlatMaterial("#ee4477")),
            Transform(pos=(-0.76, 0.55, 0.4)),
        ).index
    )
    return engine, entities


def roi_values(pixels):
    return [pixels[y, x].tolist() for x, y in ROIS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["positive", "wrong-id", "no-occluder"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--supersample", default=1, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    engine, (upper_left, upper_right, lower_left, lower_right, occluder) = scene()
    sample = args.supersample
    before = engine.render_frame(supersample=sample)
    engine.set_lights([PointLight("#ff5522", 100, (0, 0, 3))])
    with_light = engine.render_frame(supersample=sample)
    elapsed = engine.elapsed
    groups = [[] for _ in range(LABELS[1])]
    groups[0] = [upper_left, lower_left]
    groups[-1] = [upper_right, lower_right]
    if args.mode == "wrong-id":
        groups[0], groups[-1] = groups[-1], groups[0]
    if args.mode == "no-occluder":
        groups[0].append(occluder)
    ids = engine.render_frame(supersample=sample, pass_="ids", groups=groups)
    assert engine.elapsed == elapsed, "ID pass advanced simulation"
    after = engine.render_frame(supersample=sample)
    assert ids.shape == (HEIGHT, WIDTH) and ids.dtype == np.int32
    assert np.array_equal(before, with_light), "scene lights changed Basic or Flat"
    assert np.array_equal(with_light, after), "ID pass altered beauty state"
    id_rgb = np.stack([ids & 255, (ids >> 8) & 255, (ids >> 16) & 255], axis=-1).astype("u1")
    for name, image in [("beauty", before), ("with-light", with_light), ("ids", id_rgb)]:
        Image.fromarray(image).save(args.output / f"{name}.png")
    np.save(args.output / "ids.npy", ids, allow_pickle=False)
    report = {
        "mode": args.mode,
        "supersample": sample,
        "roi_rgb": roi_values(before),
        "roi_ids": roi_values(ids),
        "roi_id_bytes": roi_values(id_rgb),
        "occluded_id": int(ids[36, 60]),
        "background_id": int(ids[0, 0]),
        "observed_ids": np.unique(ids).tolist(),
        "unchanged_by_light_or_ids": True,
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output.iterdir())
            if p.suffix in (".png", ".npy")
        },
    }
    (args.output / "material-ids.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert report["roi_rgb"] == [list(rgb) for rgb in EXPECTED_RGB], report["roi_rgb"]
    assert report["roi_ids"] == list(LABELS) * 2, "ID bytes differ in labelled regions"
    assert report["roi_id_bytes"] == [[1, 0, 0], [2, 1, 0]] * 2, "ID RGB encoding differs"
    assert report["occluded_id"] == 0, "unlabelled reverse-depth occluder did not mask group 1"
    assert report["background_id"] == 0 and report["observed_ids"] == [0, *LABELS]


if __name__ == "__main__":
    main()
