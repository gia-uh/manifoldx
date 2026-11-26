"""Verify Phong highlights against an analytic front-facing plane."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 160, 120
FOV = 45
CAMERA_Z, LIGHT_Z = 5.0, 2.4
INTENSITY = 2.0
EXPONENTS = (8, 64)
COLOR = "#59738c"
# Half-open x0, y0, x1, y1 in physical pixels. Avoid silhouettes and clipping.
REGIONS = {"shoulder": (95, 56, 101, 64), "center": (76, 56, 84, 64)}
MAX_CHANNEL_ERROR = 2
MIN_SHOULDER_DELTA = 20


def analytic(exponent):
    """Intersect each pixel ray with z=0 and project the reflected light vector."""
    rows, cols = np.indices((HEIGHT, WIDTH), dtype=np.float64)
    focal = HEIGHT / (2 * np.tan(np.deg2rad(FOV / 2)))
    x = (cols + 0.5 - WIDTH / 2) * CAMERA_Z / focal
    y = (rows + 0.5 - HEIGHT / 2) * CAMERA_Z / focal
    radius_sq = x**2 + y**2
    light_distance = np.sqrt(radius_sq + LIGHT_Z**2)
    view_distance = np.sqrt(radius_sq + CAMERA_Z**2)
    cosine = LIGHT_Z / light_distance
    reflected = (LIGHT_Z * CAMERA_Z - radius_sq) / (light_distance * view_distance)
    srgb = np.frombuffer(bytes.fromhex(COLOR[1:]), dtype=np.uint8) / 255
    albedo = ((srgb + 0.055) / 1.055) ** 2.4
    linear = 0.03 * albedo + (INTENSITY / light_distance**2)[..., None] * (
        cosine[..., None] * albedo + np.maximum(0.0, reflected[..., None]) ** exponent
    )
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    rgb = np.rint(255 * encoded).astype(np.uint8)
    rgb[(abs(x) > 1.0) | (abs(y) > 0.8)] = 0
    return rgb


def capture(exponent):
    import manifoldx as mx
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import PointLight, phong

    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    engine = mx.Engine("Offscreen Phong", width=WIDTH, height=HEIGHT, max_entities=8)
    if os.environ["MANIFOLDX_CORE"] == "rust":
        assert type(engine).__name__ == "_NativeEngine", type(engine)
    engine.camera.set_pose(position=(0, 0, CAMERA_Z), target=(0, 0, 0))
    engine.camera.fov = FOV
    engine.background_color = (0, 0, 0)
    geometry = {
        "positions": np.array([[-1, -0.8, 0], [1, -0.8, 0], [-1, 0.8, 0], [1, 0.8, 0]], dtype="f4"),
        "normals": np.tile(np.array([0, 0, 1], dtype="f4"), (4, 1)),
        "indices": np.array([0, 1, 2, 2, 1, 3], dtype="u4"),
    }
    engine.set_lights([PointLight("#ffffff", INTENSITY, (0, 0, LIGHT_Z))])
    engine.spawn(Mesh(geometry), Material(phong(COLOR, shininess=exponent)), Transform())
    pixels = engine.render_frame(supersample=1, dt=1 / 30)
    assert pixels.shape == (HEIGHT, WIDTH, 3) and pixels.dtype == np.uint8
    return pixels


def region(image, rectangle):
    x0, y0, x1, y1 = rectangle
    return image[y0:y1, x0:x1].astype(np.int16)


def compare(observed, expected):
    return {
        name: [
            int(v) for v in np.abs(region(observed, roi) - region(expected, roi)).max(axis=(0, 1))
        ]
        for name, roi in REGIONS.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["historical", "candidate", "same-shininess"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    actual, expected, hashes = {}, {}, {}
    for exponent in EXPONENTS:
        expected[exponent] = analytic(exponent)
        actual[exponent] = capture(EXPONENTS[0] if args.mode == "same-shininess" else exponent)
        images = {
            f"oracle-{exponent}": expected[exponent],
            f"{args.mode}-{exponent}": actual[exponent],
        }
        images[f"diff-{exponent}"] = np.abs(
            actual[exponent].astype(np.int16) - expected[exponent]
        ).astype(np.uint8)
        for name, rgb in images.items():
            path = args.output / f"{name}.png"
            Image.fromarray(rgb).save(path)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    shoulder = REGIONS["shoulder"]
    delta = float(np.mean(region(actual[8], shoulder) - region(actual[64], shoulder)))
    oracle_delta = float(np.mean(region(expected[8], shoulder) - region(expected[64], shoulder)))
    errors = {n: compare(actual[n], expected[n]) for n in EXPONENTS}
    report = {
        "mode": args.mode,
        "roi": REGIONS,
        "max_channel_error": MAX_CHANNEL_ERROR,
        "minimum_shoulder_delta": MIN_SHOULDER_DELTA,
        "shoulder_delta": delta,
        "oracle_shoulder_delta": oracle_delta,
        "roi_max_errors": errors,
        "sha256": hashes,
    }
    (args.output / "phong.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert oracle_delta >= MIN_SHOULDER_DELTA
    if args.mode == "historical":
        assert np.array_equal(actual[8], actual[64]), (
            "historical Phong unexpectedly responds to shininess"
        )
    else:
        assert delta >= MIN_SHOULDER_DELTA, f"Phong shoulder delta {delta:.2f} is too small"
        assert all(
            max(error) <= MAX_CHANNEL_ERROR for row in errors.values() for error in row.values()
        ), errors


if __name__ == "__main__":
    main()
