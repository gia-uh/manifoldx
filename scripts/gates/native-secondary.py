"""Check a secondary camera's texture is sampled in the current main frame."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.camera import Camera
from manifoldx.components import Layer, Material, Mesh, Transform
from manifoldx.resources import (
    DirectionalLight,
    FlatMaterial,
    StandardMaterial,
    material_rgb,
    plane,
)
from manifoldx.textures import TextureHandle
from PIL import Image

WIDTH, HEIGHT = 160, 104
VIEW_SIZE = (64, 64)
ROI = (72, 44, 88, 60)
VIEW_COLOR = "#3059a7"


def capture(mode):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen secondary view", width=WIDTH, height=HEIGHT, max_entities=8)
    rust = os.environ["MANIFOLDX_CORE"] == "rust"
    if rust:
        assert type(engine).__name__ == "_NativeEngine", type(engine)
    engine.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    engine.camera.layers = {0}
    camera = Camera(position=(8, 0, 3.5), target=(8, 0, 0), fov=45)
    camera.layers = {1}
    engine.set_sun(DirectionalLight("#ffffff", 2.5, (0.05, -0.1, -1)))
    engine.enable_shadows(auto_fit=False, resolution=128, extent=4.0)
    subject = engine.spawn(
        Mesh(plane(1.6, 1.6)),
        Material(FlatMaterial("#db441a")),
        Transform(pos=(8, 0, 0)),
        Layer(layer=1),
    )

    @engine.on("startup")
    def configure(_):
        if rust:
            # This sparse, fixed fixture does not expose the E6 view/texture API.
            engine._graphics.fixture_view(
                1,
                (
                    camera.position.tolist(),
                    camera.target.tolist(),
                    camera.up.tolist(),
                    camera.fov,
                    camera.near,
                    camera.far,
                ),
                list(camera.layers),
                VIEW_SIZE,
                material_rgb(VIEW_COLOR).tolist(),
                mode,
            )
            handle = TextureHandle(1, None, None, None, VIEW_SIZE)
            handle._native_owner = engine
        else:
            handle = engine.render_to_texture(camera, VIEW_SIZE, background=VIEW_COLOR)
        engine.spawn(
            Mesh(plane(2.2, 1.8)),
            Material(StandardMaterial("#ffffff", albedo_map=handle)),
            Transform(),
            Layer(layer=0),
        )

    images = [engine.render_frame(supersample=1)]
    trace = [engine.last_frame["frame_order"]] if rust else None
    subject.transform.pos = (9.55, 0, 0)
    images.append(engine.render_frame(supersample=1))
    if rust:
        trace.append(engine.last_frame["frame_order"])
    return images, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mode", choices=["normal", "omit", "late"], default="normal")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images, trace = capture(args.mode)
    x0, y0, x1, y1 = ROI
    region = np.s_[y0:y1, x0:x1]
    report = {
        "roi": ROI,
        "rgb": [im[52, 80].tolist() for im in images],
        "roi_frame_delta": float(np.abs(images[1][region].astype("i2") - images[0][region]).mean()),
    }
    if trace is not None:
        report["trace"] = trace
    hashes = {}
    for i, pixels in enumerate(images):
        path = args.output / f"{i}.png"
        Image.fromarray(pixels).save(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if args.reference:
            ref = np.asarray(Image.open(args.reference / path.name))
            error = np.abs(ref.astype("i2") - pixels)
            report[f"{i}_roi_max_error"] = int(error[region].max())
            Image.fromarray(error.astype("u1")).save(args.output / f"{i}-diff.png")
    (args.output / "secondary.json").write_text(
        json.dumps({"measurement": report, "sha256": hashes}, indent=2) + "\n"
    )
    print(json.dumps(report))
    assert report["rgb"][0][0] > report["rgb"][0][2] + 40, (
        "main frame did not sample the current red view"
    )
    assert report["rgb"][1][2] > report["rgb"][1][0] + 40, "main frame sampled a stale view"
    assert report["roi_frame_delta"] > 40
    if trace is not None:
        assert all(order[:3] == ["view", "shadow", "main"] for order in trace), trace
    if args.reference:
        assert max(report["0_roi_max_error"], report["1_roi_max_error"]) <= 2, report


if __name__ == "__main__":
    main()
