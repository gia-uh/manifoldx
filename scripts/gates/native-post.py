"""Capture post effects and compare their regions with frozen image pairs."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.post import SSAO, Bloom, Grade, LightShafts, Sky
from manifoldx.resources import DirectionalLight, FlatMaterial, cube, plane
from PIL import Image

WIDTH, HEIGHT = 192, 128
CASES = {
    "ssao": {"ssao": SSAO(radius=1.2, samples=24, power=2)},
    "bloom": {"bloom": Bloom(threshold=0.75, knee=0.08, intensity=1.0)},
    "shafts": {"shafts": LightShafts(intensity=1.8, samples=48)},
    "saturation": {"grade": Grade(saturation=0.0)},
    "contrast": {"grade": Grade(contrast=1.45)},
    "vignette": {"grade": Grade(vignette=0.8)},
    "sky": {"sky": Sky(zenith=(0.12, 0.24, 0.8), horizon=(0.42, 0.28, 0.16), sun_size=2)},
    "fxaa": {"fxaa": True},
}
CASES["combined"] = {
    key: value
    for name, settings in CASES.items()
    if name not in ("saturation", "contrast", "vignette")
    for key, value in settings.items()
}
CASES["combined"]["grade"] = Grade(0.72, 1.15, 0.3)
REGIONS = {
    "ssao": (54, 58, 145, 108),
    "bloom": (113, 50, 143, 77),
    "shafts": (76, 16, 117, 63),
    "saturation": (66, 48, 84, 79),
    "contrast": (102, 48, 123, 79),
    "vignette": (10, 10, 37, 37),
    "sky": (76, 18, 117, 50),
    "fxaa": (51, 28, 142, 102),
    "combined": (8, 10, 184, 119),
}
MAX_ERROR = 3


def frame(name, enabled, disabled, *, flat_depth=False):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen Post", width=WIDTH, height=HEIGHT, max_entities=16)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine", type(engine)
    engine.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    engine.camera.fov = 45
    engine.background_color = (0.06, 0.09, 0.12)

    def mesh(geometry, color, position=(0, 0, 0), rotation=None):
        args = {"pos": position}
        if rotation is not None:
            args["rot"] = rotation
        engine.spawn(Mesh(geometry), Material(FlatMaterial(color)), Transform(**args))

    if name in ("ssao", "combined"):
        engine.camera.set_pose(position=(2.8, 2.6, 5), target=(0, 0, 0))
        mesh(cube(5, 0.24, 4), "#93989e", (0, -0.75, 0))
        mesh(cube(1.0, 1.3, 1.0), "#afbbc4", (0, 0.02, 0.1))
        mesh(cube(1.8, 1.6, 0.25), "#d3be99", (-0.4, 0.17, -0.55))
        mesh(cube(0.4, 0.65, 0.4), "#ffffff", (1.1, -0.3, 0))
    elif name == "bloom":
        mesh(plane(0.95, 0.7), "#ffffff")
    elif name in ("saturation", "contrast"):
        mesh(plane(0.83, 1.5), "#ad3942", (-0.5, 0, 0))
        mesh(plane(0.83, 1.5), "#626262", (0.5, 0, 0))
    elif name == "fxaa":
        theta = 0.37
        mesh(plane(2.1, 1.0), "#ffffff", rotation=(0, 0, np.sin(theta / 2), np.cos(theta / 2)))
    elif name in ("shafts", "sky"):
        mesh(plane(0.55, 0.6), "#222222", (0.52, 1.13, 0))
    if name in ("shafts", "sky", "combined"):
        engine.set_sun(DirectionalLight("#fff6df", 2.0, (0, -0.30, 1)))
    if enabled and name != disabled:
        engine.set_post(**CASES[name])
    if flat_depth:
        engine._graphics.fixture_render_fault("flat-post-depth")
    rendered = engine.render_frame(supersample=1)
    order = None
    if os.environ["MANIFOLDX_CORE"] == "rust":
        order = engine.last_frame["post_order"]
        expected = {
            "ssao": ["ssao", "ssao-blur", "composite"],
            "bloom": ["bloom-bright", *(["bloom-down"] * 3), *(["bloom-up"] * 3), "composite"],
            "shafts": ["shafts", "composite"],
            "fxaa": ["composite", "fxaa"],
        }
        if not enabled or name == disabled:
            assert order == [], order
        elif name == "combined":
            assert order[:2] == expected["ssao"][:2], order
            assert order[2:9] == expected["bloom"][:7], order
            assert order[-2:] == expected["fxaa"], order
        else:
            assert order == expected.get(name, ["composite"]), order
    return rendered, order


def resize_and_ids():
    """Change the physical target, then verify post never touches the ID readback."""
    engine = mx.Engine("Offscreen Post resize", width=80, height=60, max_entities=4)
    engine.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    entity = engine.spawn(Mesh(plane(2, 2)), Material(FlatMaterial("#994422")), Transform())
    engine.set_post(grade=Grade(0.0, 1.3, 0.5), fxaa=True)
    first = engine.render_frame()
    generation = engine.last_frame["post_target_generation"]
    assert generation == 1
    assert np.array_equal(first, engine.render_frame())
    assert engine.last_frame["post_target_generation"] == generation
    engine.w, engine.h = 101, 67
    resized = engine.render_frame()
    assert resized.shape == (67, 101, 3)
    assert np.array_equal(resized, engine.render_frame())
    assert engine.last_frame["post_order"] == ["composite", "fxaa"]
    assert engine.last_frame["post_target_generation"] == generation + 1
    labels = engine.render_frame(pass_="ids", groups=[[entity.index]])
    assert set(np.unique(labels)) == {0, 1}
    assert labels[33, 50] == 1 and labels[0, 0] == 0
    assert engine.last_frame["post_order"] == []
    again = engine.render_frame()
    assert engine.last_frame["post_target_generation"] == generation + 1
    assert np.array_equal(resized, again)
    return {
        "before": list(first.shape),
        "after": list(resized.shape),
        "generations": [generation, generation + 1],
    }


def region(frame, rectangle):
    x0, y0, x1, y1 = rectangle
    return frame[y0:y1, x0:x1].astype(np.int16)



def post_targets(output, reference, control):
    expected = np.asarray(Image.open(reference / "ssao-on.png"))
    frame_depth, order = frame("ssao", True, None, flat_depth=control == "flat-depth")
    depth_error = np.abs(frame_depth.astype("i2") - expected)
    Image.fromarray(frame_depth).save(output / "ssao-depth.png")
    Image.fromarray(depth_error.astype("u1")).save(output / "ssao-depth-diff.png")
    depth_p99 = float(np.percentile(region(depth_error, REGIONS["ssao"]), 99))

    # Read back the real native target. Use the returned byte count before any
    # wrapper can reshape it according to the requested dimensions.
    engine = mx.Engine("Offscreen post target control", width=80, height=60, max_entities=4)
    engine.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    engine.spawn(Mesh(plane(2, 2)), Material(FlatMaterial("#b34d35")), Transform())
    engine.set_post(grade=Grade(0.65, 1.1, 0), fxaa=True)
    before = engine.render_frame()
    before_generation = engine.last_frame["post_target_generation"]
    width, height = 111, 72
    if control == "stale":
        engine._graphics.fixture_render_fault("stale-offscreen-target")
    captured = engine._graphics.capture(engine.world._native, width, height)
    size_bytes = len(captured["rgba"])
    shape = (height, width) if size_bytes == width * height * 4 else before.shape[:2]
    image = np.frombuffer(captured["rgba"], dtype="u1").reshape(*shape, 4)[..., :3]
    Image.fromarray(before).save(output / "before-resize.png")
    Image.fromarray(image).save(output / "after-resize.png")
    roi = np.s_[height//2-4:height//2+4, width//2-4:width//2+4]
    error = None
    if image.shape == (height, width, 3) and (reference / "after-resize.png").is_file():
        frozen = np.asarray(Image.open(reference / "after-resize.png"))
        error = int(np.abs(image[roi].astype("i2") - frozen[roi]).max())
    center = image[roi]
    report = {
        "ssao_roi_error_p99": depth_p99, "ssao_order": order,
        "frame_bytes": size_bytes, "shape": list(image.shape), "expected_shape": [height, width, 3],
        "generation": [before_generation, captured["post_target_generation"]],
        "resize_roi_error": error, "resize_roi": np.median(center, axis=(0, 1)).tolist(),
    }
    (output / "post-targets.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert depth_p99 <= MAX_ERROR, "flat depth changed SSAO ROI"
    assert image.shape == (height, width, 3), "stale framebuffer has the wrong physical dimensions"
    assert captured["post_target_generation"] == before_generation + 1, "post targets did not resize"
    assert error is None or error <= MAX_ERROR, "resized framebuffer missed its ROI"
    assert np.median(center[..., 0]) > np.median(center[..., 2]) + 30, "resized mesh missed its ROI"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--disable", choices=CASES)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--targets", action="store_true")
    parser.add_argument("--target-control", choices=("flat-depth", "stale"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.targets:
        assert args.reference is not None, "--targets requires the frozen SSAO reference"
        post_targets(args.output, args.reference, args.target_control)
        return
    cases = [args.case] if args.case else CASES
    result = {}
    for name in cases:
        before, before_order = frame(name, False, args.disable)
        after, after_order = frame(name, True, args.disable)
        bounds = REGIONS[name]
        pair = region(after, bounds) - region(before, bounds)
        frames = {
            "off": before,
            "on": after,
            "delta": np.abs(after.astype(np.int16) - before).astype("u1"),
        }
        result[name] = {
            "roi": bounds,
            "mean_change": float(np.abs(pair).mean()),
            "mean_signed_delta": float(pair.mean()),
            "affected_pixels": int(np.any(pair, axis=2).sum()),
        }
        if before_order is not None:
            result[name]["off_order"] = before_order
            result[name]["on_order"] = after_order
        for variant, image in frames.items():
            Image.fromarray(image).save(args.output / f"{name}-{variant}.png")
            if args.reference is not None and variant != "delta":
                ref = np.asarray(Image.open(args.reference / f"{name}-{variant}.png"))
                error = np.abs(image.astype(np.int16) - ref)
                Image.fromarray(error.astype("u1")).save(
                    args.output / f"{name}-{variant}-error.png"
                )
                result[name][f"{variant}_roi_error_p99"] = np.percentile(
                    region(error, bounds), 99, axis=(0, 1)
                ).tolist()
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.glob("*.png"))
    }
    report = {"cases": result, "sha256": hashes}
    if os.environ["MANIFOLDX_CORE"] == "rust" and not (args.case or args.disable):
        report["resize_and_ids"] = resize_and_ids()
    (args.output / "post.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result))
    for name, metrics in result.items():
        assert metrics["mean_change"] > 0.35 and metrics["affected_pixels"] > 20, (
            f"{name} ROI did not respond"
        )
        if args.reference is not None:
            for variant in ("off", "on"):
                assert max(metrics[f"{variant}_roi_error_p99"]) <= MAX_ERROR, (
                    name,
                    variant,
                    metrics,
                )


if __name__ == "__main__":
    main()
