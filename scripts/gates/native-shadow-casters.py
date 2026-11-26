"""Check moving joints, Fade and caster exclusions against a fixed receiver."""

import argparse
import json
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Fade, Layer, Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, plane
from PIL import Image

ROI = np.s_[30:108, 20:156]
FRAMES = (("left", -1.0, 1.0), ("middle-full", 0.0, 1.0), ("middle", 0.0, 0.5), ("right", 1.0, 1.0), ("faded", 1.0, 0))


def capture(output, control):
    engine = mx.Engine("Offscreen shadow casters", width=176, height=136, max_entities=8)
    assert type(engine).__name__ == "_NativeEngine"
    engine.camera.set_pose(position=(0, 0, 6), target=(0, 0, 0))
    engine.camera.layers = {0}
    engine.world.register(Fade)
    engine.set_sun(DirectionalLight("#fff4dc", 3.0, (0.45, -0.3, -1)))
    engine.enable_shadows(extent=5.0, resolution=256, auto_fit=False)
    engine.spawn(Mesh(plane(6, 4.4)), Material(StandardMaterial("#c1aa83")),
                 Transform(pos=(0, 0, -1)), Layer(layer=0))
    empty = engine.render_frame(supersample=1)
    mesh = plane(0.8, 0.8)
    mesh["joints"] = np.zeros(len(mesh["positions"]), dtype="u4")
    regular = StandardMaterial("#cc6655", faded=True)
    caster = engine.spawn(Mesh(mesh), Material(regular), Transform(pos=(0, 0.36, 0.5)),
                          Layer(layer=1), Fade(value=1.0))
    identity = engine.world.query(Fade).read().entities[0]
    frames = {"empty": empty}
    for name, offset, fade in FRAMES:
        pose = np.eye(4, dtype="f4")[None]
        pose[0, 0, 3] = 0.0 if control == "joints" else offset
        caster.set_joints(pose)
        engine.world.replace(Fade, (identity,), Fade(value=1.0 if control == "fade" else fade).get_data(1))
        frames[name] = engine.render_frame(supersample=1)
    for kind, effect in (("mirror", {"mirror": 0.0}), ("translucent", {"opacity": 0.42})):
        obj = engine.spawn(Mesh(mesh), Material(StandardMaterial("#cc6655", **(
            {} if kind == control else effect))), Transform(pos=(0, 0.36, 0.5)), Layer(layer=1))
        frames[kind] = engine.render_frame(supersample=1)
        # Keep the fixture alive across the sequence and move each probe outside the receiver.
        obj.transform.pos = (18, 0, 0.5)
    masks = {}
    darkness = {}
    for name, frame in frames.items():
        Image.fromarray(frame).save(output / f"{name}.png")
        signal = (empty.astype("i2") - frame)[ROI].mean(axis=2).clip(min=0)
        darkness[name] = float(signal.sum())
        mask = signal > 10
        masks[name] = mask
        Image.fromarray(mask.astype("u1") * 255).save(output / f"{name}-mask.png")
    Image.fromarray(frames["left"]).save(
        output / "joints-and-fade.gif", save_all=True,
        append_images=[Image.fromarray(frames[k]) for k, _, _ in FRAMES[1:]], duration=180, loop=0,
    )
    return masks, frames, darkness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--control", choices=("joints", "fade", "mirror", "translucent"))
    args = parser.parse_args()
    assert sys.flags.isolated and Path(mx.__file__).resolve().is_relative_to(sys.prefix)
    args.output.mkdir(parents=True, exist_ok=True)
    masks, frames, darkness = capture(args.output, args.control)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    centers = {name: float(np.nonzero(mask)[1].mean()) for name, mask in masks.items() if mask.any()}
    report = {"darkened_pixels": counts, "centers_x": centers, "darkness": darkness}
    (args.output / "casters.json").write_text(json.dumps(report, indent=2) + "\n")
    assert min(counts["left"], counts["right"]) > 35, "opaque skinned caster did not cast"
    assert centers["right"] - centers["left"] > 20, "moving joints did not move the shadow ROI"
    assert 0.1 < darkness["middle"] / darkness["middle-full"] < 0.75, "Fade did not attenuate its shadow"
    assert counts["faded"] == 0, "fully faded caster left a shadow"
    for name in ("mirror", "translucent"):
        assert np.array_equal(frames[name][ROI], frames["empty"][ROI]), f"{name} cast a shadow"
    if args.reference:
        for name, image in frames.items():
            reference = np.asarray(Image.open(args.reference / f"{name}.png"))
            assert np.abs(image[ROI].astype("i2") - reference[ROI]).mean() <= 1, f"{name} ROI differs"
    print(json.dumps(report))


if __name__ == "__main__":
    main()
