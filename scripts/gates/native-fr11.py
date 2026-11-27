"""Run the installed FR-11 demo and require effective settings and visible changes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx._fr11 import DESIGN, GROUPS, POST, VERSION
from PIL import Image

BASE_PASSES = ["view", "view", "shadow", "skybox", "main", "sprite", "volume", "label", "axis", "lines"]
POST_PASSES = [
    "ssao", "ssao-blur", "bloom-bright", *["bloom-down"] * 3, *["bloom-up"] * 3,
    "shafts", "composite", "fxaa", "gui",
]
POST_EFFECTS = (*POST, "saturation", "contrast", "vignette")
HUD = np.s_[18:185, 18:210]
ENV_SKY = np.s_[157:186, 516:621]
POST_SKY = np.s_[5:45, 233:491]
MINIMAP = np.s_[161:206, 515:619]
SNAPSHOT = np.s_[267:335, 516:622]


def run(output, *args, must_fail=None, frames=2):
    command = [
        sys.executable, "-I", "-m", "manifoldx._fr11", "--offscreen",
        "--output", str(output), "--frames", str(frames), *args,
    ]
    output.mkdir(parents=True, exist_ok=True)
    execution = subprocess.run(command, capture_output=True, text=True, check=False)
    (output / "invocation.json").write_text(json.dumps({
        "argv": command, "exit": execution.returncode, "core": os.environ.get("MANIFOLDX_CORE"),
    }, indent=2) + "\n")
    (output / "stdout.log").write_text(execution.stdout)
    (output / "stderr.log").write_text(execution.stderr)
    if must_fail:
        assert execution.returncode != 0 and must_fail in execution.stderr, execution.stderr
        return None
    assert execution.returncode == 0, execution.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    images = []
    for frame in manifest:
        path = output / f"{frame['label']}.png"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frame["image_sha256"]
        image = np.asarray(Image.open(path), dtype=np.int16)
        assert image.shape == (*DESIGN[::-1], 3)
        images.append(image)
    return manifest, images


def change(before, after, roi=np.s_[:, :]):
    return int(np.count_nonzero(np.max(np.abs(before[roi] - after[roi]), axis=2) > 3))


def validate(frame):
    assert frame["version"] == VERSION and frame["provenance"]["core"] == "rust", frame
    assert frame["format"] == "rgba8unorm-srgb", frame
    assert frame["passes"] == BASE_PASSES + POST_PASSES, frame["passes"]
    settings = frame["settings"]
    expected = {
        "ssao": (POST["ssao"].radius, POST["ssao"].intensity, POST["ssao"].samples,
                 POST["ssao"].bias, POST["ssao"].power),
        "bloom": (POST["bloom"].threshold, POST["bloom"].knee, POST["bloom"].intensity),
        "shafts": (POST["shafts"].intensity, POST["shafts"].decay,
                   POST["shafts"].density, POST["shafts"].samples),
        "grade": (POST["grade"].saturation, POST["grade"].contrast, POST["grade"].vignette),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(settings["post"][name], values, atol=1e-6)
    zenith, horizon, size, intensity, glow = settings["post"]["sky"]
    np.testing.assert_allclose([zenith, horizon], [POST["sky"].zenith, POST["sky"].horizon], atol=1e-6)
    np.testing.assert_allclose([size, intensity, glow],
                               [POST["sky"].sun_size, POST["sky"].sun_intensity, POST["sky"].glow], atol=1e-6)
    assert settings["post"]["fxaa"] is True and settings["environment"][1] is True
    assert settings["lights"] == [True, 1, False] and settings["fog"] is not None
    assert set(settings["post_formats"]) == {"rgba8-srgb", "depth32float", "r8unorm", "rgba16float"}
    counts = frame["counts"]
    assert counts["shadow_layers"] >= 3 and counts["instances"] >= 20
    assert counts["view_resources"] == [2, 2] and counts["gui_getter_calls"] == 1
    assert counts["gui_quads"][2] == 2 and min(counts["gui_quads"]) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--post-only", action="store_true", help="omit the individual scene-family runs")
    args = parser.parse_args()
    assert sys.flags.isolated and Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix))
    output = args.output.resolve()
    manifest, images = run(output / "combined", frames=49)
    first = manifest[0]
    baseline = images[0]
    validate(first)
    assert manifest[-1]["timeline"]["photo"] == 1
    assert manifest[-1]["timeline"]["shadow_caster"] == "spot"
    assert manifest[-1]["settings"]["lights"] == [True, 1, True]
    assert manifest[-1]["counts"]["shadow_layers"] == 1
    assert any(frame["passes"].count("view") == 1 for frame in manifest[1:])
    assert any(frame["passes"].count("view") == 2 for frame in manifest[1:])
    assert any(frame["counts"]["streamed_cells"] > 0 for frame in manifest[1:])
    assert change(baseline, images[-1], MINIMAP) >= 12, "live secondary view remained static"
    assert change(baseline, images[-1], SNAPSHOT) >= 12, "photo did not reflect its refresh"

    changed = {}
    for name in (*POST_EFFECTS, *(() if args.post_only else GROUPS)):
        can_move = name in {"stream", "compute", "spot"}
        other, captures = run(output / f"without-{name}", "--omit", name, frames=49 if can_move else 2)
        reference, frame = (images[-1], captures[-1]) if can_move else (baseline, captures[0])
        roi = {"sky": POST_SKY, "environment": ENV_SKY, "gui": HUD}.get(name, np.s_[:, :])
        pixels = change(reference, frame, roi)
        assert pixels >= 8, f"no visible change with {name} disabled ({pixels} pixels)"
        if name in POST:
            expected = False if name == "fxaa" else None
            assert other[0]["settings"]["post"][name] == expected
        changed[name] = pixels

    reordered, reordered_images = run(output / "gui-before-post", "--fault", "hud-order")
    assert reordered[0]["passes"].index("gui") < reordered[0]["passes"].index("composite")
    try:
        validate(reordered[0])
    except AssertionError:
        pass
    else:
        raise AssertionError("strict manifest accepted GUI under post")
    changed["hud-order"] = change(baseline, reordered_images[0], HUD)
    assert changed["hud-order"] >= 80, "GUI under post did not alter the HUD ROI"
    run(output / "false-provenance", "--fault", "provenance", must_fail="requires the native desktop Engine")
    plain, _ = run(output / "plain", "--no-post")
    assert plain[0]["passes"] == BASE_PASSES + ["gui"], plain[0]["passes"]
    (output / "fr11.json").write_text(json.dumps({"changed_pixels": changed}, indent=2) + "\n")
    print(json.dumps(changed))


if __name__ == "__main__":
    main()
