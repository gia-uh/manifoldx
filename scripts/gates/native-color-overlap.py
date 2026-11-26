"""Compare analytic sRGB ramps and the order of two translucent mesh layers."""

import argparse
import json
import runpy
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import FlatMaterial, StandardMaterial, plane
from PIL import Image

LEVELS = np.array([22, 58, 101, 153, 204, 238], dtype="u1")
SOLID = np.array([60, 91, 155], dtype="u1")
FRONT = np.array([50, 125, 225], dtype="u1")


def decode(rgb):
    x = np.asarray(rgb, dtype=float) / 255
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def encode(x):
    return np.rint(255 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x**(1/2.4) - 0.055)).astype("u1")


def sample(engine, image, point):
    xy, _ = engine.camera.project(point, engine.w, engine.h)
    x, y = np.rint(xy[0]).astype(int)
    return np.median(image[y-2:y+3, x-2:x+3], axis=(0, 1))


def setup(name):
    engine = mx.Engine(name, width=192, height=120, max_entities=24)
    engine.camera.set_pose(position=(0, 0, 5), target=(0, 0, 0))
    engine.background_color = (0, 0, 0)
    return engine


def ramps(control):
    e = setup("Offscreen color ramps")
    recipe = runpy.run_path(str(Path(__file__).with_name("native-manual.py")))
    base = recipe["Manual"]
    shader = recipe["WGSL"]

    class DoubleDecode(base):
        @classmethod
        def _compile(cls, *, textured=True):
            return shader.replace(
                "return textureSample(view_at_five, sampler_at_four, input.uv) * tint.color;",
                """let tex = textureSample(view_at_five, sampler_at_four, input.uv);
    let rgb = select(pow((tex.rgb + 0.055) / 1.055, vec3<f32>(2.4)),
                     tex.rgb / 12.92, tex.rgb <= vec3<f32>(0.04045));
    return vec4<f32>(rgb, tex.a) * tint.color;""",
            )

    if type(e).__name__ != "_NativeEngine":
        e._ensure_offscreen(1)
    positions = np.linspace(-1.85, 1.85, len(LEVELS))
    for x, level in zip(positions, LEVELS):
        rgb = np.array([level, level, level], dtype="u1")
        color = (np.rint(255*decode(rgb)).astype("u1") if control == "gamma" else rgb)
        hex_color = "#" + color.tobytes().hex()
        e.spawn(Mesh(plane(0.55, 0.6)), Material(FlatMaterial(hex_color)), Transform(pos=(x, 0.55, 0)))
        pixels = np.tile(np.r_[rgb, np.uint8(255)], (8, 8, 1))
        texture = mx.texture_from_array(e, pixels)
        material = (DoubleDecode if control == "gamma" else base)(texture, 1.0)
        e.spawn(Mesh(plane(0.55, 0.6)), Material(material), Transform(pos=(x, -0.55, 0)))
    pixels = e.render_frame(supersample=1)
    medians = {label: [sample(e, pixels, (x, y, 0)).tolist() for x in positions]
               for label, y in (("solid", 0.55), ("texture", -0.55))}
    return pixels, medians


def overlap(control):
    e = setup("Offscreen transparent overlap")

    def draw(geo, mat, pos):
        e.spawn(Mesh(geo), Material(mat), Transform(pos=pos))

    draw(plane(3.8, 2.2), FlatMaterial("#" + SOLID.tobytes().hex()), (0, 0, -0.6))
    draw(plane(0.55, 0.55), FlatMaterial("#" + FRONT.tobytes().hex()), (-0.65, 0, 0.65))
    layers = [
        (StandardMaterial("#ff0000", opacity=0.56), (0, 0, 0.0)),
        (StandardMaterial("#00ff00", opacity=0.72), (0, 0, 0.3)),
    ]
    if control == "blend":
        layers.reverse()
    for mat, pos in layers:
        draw(plane(2.5, 1.6), mat, pos)
    pixels = e.render_frame(supersample=1)
    result = {name: sample(e, pixels, point).tolist() for name, point in (
        ("overlap", (0, 0, 0)), ("foreground", (-0.65, 0, 0.65)), ("solid", (1.62, 0, -0.6)),
    )}
    return pixels, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--control", choices=("gamma", "blend"))
    args = parser.parse_args()
    assert sys.flags.isolated and Path(mx.__file__).resolve().is_relative_to(sys.prefix)
    args.output.mkdir(parents=True, exist_ok=True)
    ramp, values = ramps(args.control)
    layers, colors = overlap(args.control)
    expected_ramp = np.repeat(LEVELS[:, None], 3, axis=1)
    # With no lamps, Standard contributes 0.03 times its linear albedo.
    red = np.array([0.03, 0, 0])
    green = np.array([0, 0.03, 0])
    expected_layers = {
        "overlap": encode(0.72*green + 0.28*(0.56*red + 0.44*decode(SOLID))).tolist(),
        "foreground": FRONT.tolist(),
        "solid": SOLID.tolist(),
    }
    report = {"ramp": values, "overlap": colors, "expected_overlap": expected_layers}
    for name, image in (("ramps", ramp), ("overlap", layers)):
        Image.fromarray(image).save(args.output / f"{name}.png")
        if args.reference:
            prior = np.asarray(Image.open(args.reference / f"{name}.png"))
            Image.fromarray(np.abs(image.astype("i2") - prior).astype("u1")).save(args.output / f"{name}-diff.png")
    (args.output / "color.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    for name, rows in values.items():
        np.testing.assert_allclose(rows, expected_ramp, atol=1, rtol=0, err_msg=f"{name} gamma ROI")
    for name, rgb in colors.items():
        np.testing.assert_allclose(rgb, expected_layers[name], atol=2, rtol=0, err_msg=f"{name} blend/depth ROI")


if __name__ == "__main__":
    main()
