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

# Coordinates are fixed in the versioned 640×360 composition. Sample the
# contribution itself, away from the HUD and secondary effects where possible.
REGIONS = {
    "ssao": np.s_[194:222, 311:374],          # contact with the sample shelf
    "bloom": np.s_[245:265, 219:277],        # halo around the bright water glint
    "shafts": np.s_[10:75, 521:608],         # light through the sky
    "grade": np.s_[151:191, 241:316],
    "sky": POST_SKY,
    "fxaa": np.s_[64:91, 381:429],           # pennant silhouette
    "saturation": np.s_[145:192, 323:406],    # purple material samples
    "contrast": np.s_[192:229, 295:354],
    "vignette": np.s_[306:345, 50:124],       # near the image edge
    "basic": np.s_[158:198, 190:236],
    "flat": np.s_[156:201, 231:270],
    "phong": np.s_[152:197, 275:314],
    "pbr": np.s_[151:198, 319:403],
    "patterns": np.s_[205:235, 191:490],
    "palette": np.s_[123:144, 232:268],
    "baked": np.s_[117:146, 274:321],
    "fade": np.s_[154:190, 442:486],
    "cut": np.s_[155:196, 403:441],
    "mirror": np.s_[294:328, 130:175],
    "alpha": np.s_[220:253, 125:176],
    "textured": np.s_[96:118, 352:382],
    "skin": np.s_[62:91, 384:428],
    "fragment": np.s_[91:116, 246:282],
    "water": np.s_[243:270, 160:256],
    "sea": np.s_[291:337, 193:395],
    "glow": np.s_[91:119, 412:459],
    "scattering": SNAPSHOT,
    "cloud": np.s_[89:113, 290:346],
    "sprite-lit": np.s_[126:151, 324:353],
    "sprite-unlit": np.s_[126:151, 366:392],
    "volume": np.s_[126:147, 402:427],
    "label": np.s_[94:110, 252:388],
    "axis": np.s_[242:251, 380:482],
    "lines": np.s_[240:264, 223:365],
    "stream": np.s_[187:231, 463:510],       # streamed strip at frame 48
    "compute": np.s_[214:245, 425:487],      # buoy travel at frame 48
    "view": MINIMAP,
    "snapshot": SNAPSHOT,
    "gui": HUD,
    "environment": ENV_SKY,
    "fog": np.s_[115:152, 437:482],          # far sample-chart edge
    "shadow": np.s_[212:235, 315:362],       # receiving shelf
    "point": np.s_[155:188, 272:311],
    "spot": np.s_[124:147, 272:319],
}
MESH_INSTANCES = dict.fromkeys((
    "basic", "flat", "phong", "palette", "baked", "fade", "cut", "mirror",
    "alpha", "textured", "skin", "fragment", "water", "sea", "glow", "cloud",
), 1) | {"pbr": 2, "patterns": 8, "axis": 1}
REMOVED_PASSES = {
    "ssao": {"ssao", "ssao-blur"},
    "bloom": {"bloom-bright", "bloom-down", "bloom-up"},
    **{name: {name} for name in ("shafts", "fxaa", "volume", "label", "axis", "lines", "shadow", "gui")},
    "environment": {"skybox"},
}
GRADE_NEUTRAL = {"saturation": (0, 1), "contrast": (1, 1), "vignette": (2, 0)}


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


def change(before, after, roi):
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
    assert counts["compute_dispatches"] == 1


def visible_change(name, before, after):
    pixels = change(before, after, REGIONS[name])
    assert pixels >= 8, f"{name}: only {pixels} changed pixels in its expected region"
    return pixels


def omission_signals(name, active, omitted):
    passes = [pass_ for pass_ in active["passes"] if pass_ not in REMOVED_PASSES.get(name, ())]
    if name in {"view", "snapshot", "gui"}:
        passes.remove("view")
    assert omitted["passes"] == passes, (name, omitted["passes"])
    settings = omitted["settings"]
    if name in POST:
        expected = False if name == "fxaa" else None
        assert settings["post"][name] is expected, name
    if name in GRADE_NEUTRAL:
        channel, neutral = GRADE_NEUTRAL[name]
        expected = active["settings"]["post"]["grade"].copy()
        expected[channel] = neutral
        assert settings["post"]["grade"] == expected, name
    if name in {"environment", "fog"}:
        assert settings[name] is None, name
    before, after = active["counts"], omitted["counts"]
    if name in MESH_INSTANCES:
        assert before["instances"] - after["instances"] == MESH_INSTANCES[name], name
    if name in {"sprite-lit", "sprite-unlit", "volume", "label"}:
        assert before["draw_calls"] - after["draw_calls"] == 1, name
    if name in {"view", "snapshot"}:
        assert after["view_resources"] == [1, 1] and after["gui_quads"][2] == 1, name
    if name == "gui":
        assert after["gui_quads"] == [0, 0, 0] and after["gui_getter_calls"] == 0
    if name == "point":
        assert settings["lights"] == [True, 0, False]
    if name == "spot":
        assert settings["lights"] == [True, 1, False]
        assert omitted["timeline"]["shadow_caster"] == "sun" and after["shadow_layers"] == 3
    if name == "stream":
        assert not omitted["timeline"]["cells"] and after["stream_resources"] == [0, 0]
    if name in {"shadow", "stream", "compute"}:
        counter = {"shadow": "shadow_layers", "stream": "streamed_cells", "compute": "compute_dispatches"}[name]
        assert before[counter] > 0 and after[counter] == 0, name


def reject_unrelated_changes(name, before, after):
    # Restore the designated region while keeping actual changes elsewhere.
    # If the omission has no spill, inject a change in a disjoint corner.
    region = REGIONS[name]
    decoy = after.copy()
    decoy[region] = before[region]
    corner = np.s_[-8:, -8:]
    if change(before, decoy, np.s_[:, :]) < 8:
        decoy[corner] = 255 - before[corner]
    assert change(before, decoy, np.s_[:, :]) >= 8
    try:
        visible_change(name, before, decoy)
    except AssertionError:
        return
    raise AssertionError(f"{name}: accepted unrelated image changes")


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
    by_label = {frame["label"]: (frame, image) for frame, image in zip(manifest, images, strict=True)}
    before_refresh, before_photo = by_label["frame-030"]
    after_refresh, after_photo = by_label["frame-042"]
    last = manifest[-1]
    assert first["timeline"]["shadow_caster"] == "sun" and first["counts"]["shadow_layers"] == 3
    assert before_refresh["timeline"]["shadow_caster"] == "spot"
    assert before_refresh["settings"]["lights"] == [True, 1, True]
    assert before_refresh["counts"]["shadow_layers"] == 1
    assert last["timeline"]["photo"] == 1 and last["passes"].count("view") == 1
    assert before_refresh["timeline"]["photo"] == 0 and before_refresh["passes"].count("view") == 1
    assert after_refresh["timeline"]["photo"] == 1 and after_refresh["passes"].count("view") == 2
    assert not first["timeline"]["cells"] and last["timeline"]["cells"]
    assert first["counts"]["streamed_cells"] == 0
    assert change(baseline, before_photo, MINIMAP) >= 12, "live secondary view remained static"
    assert change(baseline, before_photo, SNAPSHOT) == 0, "on-demand view redrew without refresh"
    assert change(before_photo, after_photo, SNAPSHOT) >= 12, "photo did not reflect its refresh"
    assert change(after_photo, images[-1], SNAPSHOT) == 0, "photo redrew after refresh"

    assert set(REGIONS) == set(POST_EFFECTS) | set(GROUPS)
    changed = {}
    for name in (*POST_EFFECTS, *(() if args.post_only else GROUPS)):
        can_move = name in {"stream", "compute", "spot"}
        other, captures = run(output / f"without-{name}", "--omit", name, frames=49 if can_move else 2)
        index = -1 if can_move else 0
        assert manifest[index]["label"] == other[index]["label"], name
        omission_signals(name, manifest[index], other[index])
        changed[name] = visible_change(name, images[index], captures[index])
        reject_unrelated_changes(name, images[index], captures[index])

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
    regions = REGIONS | {"hud-order": HUD}
    report = {
        "changed_pixels": changed,
        "regions_yx": {name: [[axis.start, axis.stop] for axis in regions[name]] for name in changed},
        "unrelated_change_controls": len(changed) - 1,
    }
    (output / "fr11.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(changed))


if __name__ == "__main__":
    main()
