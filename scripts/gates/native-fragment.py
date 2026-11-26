"""Verify generated fragment color, AST diagnostics and material bindings."""

import argparse
import json
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import plane
from manifoldx.shader import ShaderCompileError, Texture1D, Vec2, Vec4, clamp, f32, sample
from PIL import Image

FIRST = (221, 39, 51)
SECOND = (33, 161, 227)
OTHER = (79, 198, 91)
ROI = np.s_[28:36, 36:44]


@mx.shader.fragment
def heat(uv: Vec2, value: f32, cmap: Texture1D) -> Vec4:
    return sample(cmap, clamp(value, 0.0, 1.0))


@mx.shader.fragment
def gradient(uv: Vec2, cmap: Texture1D) -> Vec4:
    return sample(cmap, uv.x)


def capture(material):
    engine = mx.Engine("Offscreen custom material", width=80, height=64, max_entities=4)
    assert type(engine).__name__ == "_NativeEngine"
    engine.camera.set_pose(position=(0, 0, 2), target=(0, 0, 0))
    engine.spawn(Mesh(plane(2, 2)), Material(material), Transform())
    return engine.render_frame(supersample=1)


def palette(a, b):
    return np.array([(*a, 255), (*b, 255)], dtype="u1")


def verify_rejections(original):
    diagnostics = {}
    good = {"value": 0.25, "cmap": original}
    cases = {
        "missing": ({"cmap": original}, TypeError, "missing=['value']"),
        "extra": (dict(good, uv=(0, 0)), TypeError, "extra=['uv']"),
        "scalar_type": (dict(good, value="0.25"), TypeError, "requires an f32 scalar"),
        "finite": (dict(good, value=float("inf")), ValueError, "requires a finite f32 scalar"),
        "dtype": (dict(good, cmap=original.astype("f4")), TypeError, "requires uint8 texels"),
        "shape": (dict(good, cmap=original[:, :3]), ValueError, "requires (width, 4)"),
    }
    for name, (bindings, exception, expected) in cases.items():
        try:
            mx.ShaderMaterial(heat, **bindings)
        except exception as error:
            diagnostics[name] = str(error)
            assert expected in str(error), error
        else:
            raise AssertionError(f"{name} binding should be rejected")
    try:

        @mx.shader.fragment
        def invalid(uv: Vec2) -> Vec4:
            return np.asarray(uv)
    except ShaderCompileError as error:
        diagnostics["ast"] = {
            "file": Path(error.filename).name,
            "line": error.lineno,
            "column": error.offset,
            "message": error.msg,
        }
        assert Path(error.filename) == Path(__file__)
        assert error.lineno > 0 and error.offset > 0
        assert "unsupported fragment expression" in error.msg
    else:
        raise AssertionError("invalid fragment should fail during declaration")
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--disable", choices=("value", "palette"))
    args = parser.parse_args()
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    original = palette(FIRST, SECOND)
    alternate = palette(OTHER, SECOND)
    images = {
        "first": capture(mx.ShaderMaterial(heat, value=0.25, cmap=original)),
        "second": capture(
            mx.ShaderMaterial(heat, cmap=original, value=0.25 if args.disable == "value" else 0.75)
        ),
        "other": capture(
            mx.ShaderMaterial(
                heat, value=0.25, cmap=original if args.disable == "palette" else alternate
            )
        ),
        "uv": capture(mx.ShaderMaterial(gradient, cmap=original)),
    }
    results = {"binding_rejections": verify_rejections(original)}
    for name, img in images.items():
        Image.fromarray(img).save(args.output / f"{name}.png")
        results[name] = img[ROI].mean(axis=(0, 1)).tolist()
    (args.output / "fragment.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
    for name, expected in (("first", FIRST), ("second", SECOND), ("other", OTHER)):
        assert np.abs(images[name][ROI].astype("i2") - expected).max() <= 1, (
            f"{name} ROI missed its texel"
        )
    assert np.abs(images["first"][ROI].astype("i2") - images["second"][ROI]).max() > 25
    assert np.abs(images["first"][ROI].astype("i2") - images["other"][ROI]).max() > 25
    assert int(images["uv"][32, 18, 0]) > int(images["uv"][32, 62, 0]) + 80


if __name__ == "__main__":
    main()
