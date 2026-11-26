"""Exercise the E5 encoder dependencies in one installed-wheel scene."""

import argparse
import json
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.camera import Camera
from manifoldx.components import Layer, Material, Mesh, Transform
from manifoldx.gui.layout import LayoutBox
from manifoldx.gui.material import RectMaterial
from manifoldx.gui.painter import RectOp
from manifoldx.post import Grade
from manifoldx.resources import DirectionalLight, FlatMaterial, StandardMaterial, cube, plane
from manifoldx.textures import TextureHandle
from manifoldx.viz import (
    AxisFrame, AxisMaterial, ColormapMaterial, LabelMaterial, PointCloud, Radius,
    ScalarValue, TextLabel, Volume, VolumeMaterial,
)
from manifoldx.viz.geometry import AXIS_LINE_X
from PIL import Image

WIDTH, HEIGHT = 216, 144
SCENE = ("view", "shadow", "skybox", "opaque", "mirror", "translucent",
         "sprite", "volume", "label", "axis", "lines", "rect")
ORDER = ["view", "shadow", "skybox", "main", "sprite", "volume", "label", "axis", "lines"]
REGIONS = {
    "view": np.s_[54:78, 52:88], "shadow": np.s_[61:82, 84:120],
    "skybox": np.s_[2:30, 172:208], "opaque": np.s_[59:82, 90:120],
    "mirror": np.s_[90:105, 58:73], "translucent": np.s_[93:106, 141:156],
    "sprite": np.s_[39:55, 141:157], "volume": np.s_[58:71, 143:158],
    "label": np.s_[29:43, 81:94], "axis": np.s_[76:86, 122:162],
    "lines": np.s_[98:109, 89:129], "rect": np.s_[7:25, 7:41],
}


def capture(post, omit=None, late=False):
    engine = mx.Engine("Offscreen pass order", width=WIDTH, height=HEIGHT, max_entities=32)
    assert type(engine).__name__ == "_NativeEngine"
    engine.background_color = (0.022, 0.025, 0.036)
    engine.camera.set_pose(position=(0, 0, 7), target=(0, 0, 0))
    engine.camera.layers = {0}
    for component in (Layer, PointCloud, ScalarValue, Radius, Volume, TextLabel, AxisFrame):
        engine.world.register(component)
    engine.set_sun(DirectionalLight("#fff1e1", 3.0, (0.65, -0.35, -1)))
    engine.set_environment("sky")
    engine.environment.show_skybox = omit != "skybox"
    if omit != "shadow":
        engine.enable_shadows(extent=6.0, resolution=256, auto_fit=False)

    def mesh(geometry, material, pos, layer=0):
        return engine.spawn(Mesh(geometry), Material(material), Transform(pos=pos), Layer(layer=layer))

    mesh(plane(3.2, 2.6), StandardMaterial("#b7a891"), (0.12, 0, -0.9))
    if omit != "opaque":
        mesh(cube(0.58, 0.58, 0.58), StandardMaterial("#a34b2f"), (-0.55, 0.25, 0.2))
    if omit != "mirror":
        mesh(plane(0.7, 0.6), StandardMaterial("#a534ed", mirror=0), (-2.35, 1.45, 0))
    if omit != "translucent":
        mesh(plane(0.65, 0.5), StandardMaterial("#e4d351", opacity=0.65), (2.28, -1.55, 0))
    if omit != "view":
        view = Camera(position=(8, 0, 4), target=(8, 0, 0))
        engine._graphics.fixture_view(
            1, (view.position.tolist(), view.target.tolist(), view.up.tolist(),
                view.fov, view.near, view.far), [1], (40, 40), (0.05, 0.1, 0.2), "normal",
        )
        mesh(plane(2.4, 2.4), FlatMaterial("#e54519"), (8, 0, 0), layer=1)
        handle = TextureHandle(1, None, None, None, (40, 40))
        handle._native_owner = engine
        mesh(plane(1.08, 0.85), StandardMaterial("#ffffff", albedo_map=handle), (-2.5, 0.5, 0))
    if omit != "sprite":
        engine.spawn(PointCloud(), Material(ColormapMaterial("viridis", 0, 1, lit=False)),
                     ScalarValue(value=1), Radius(radius=0.36), Transform(pos=(2.3, 1.4, 0)))
    if omit != "volume":
        z, y, x = np.mgrid[-1:1:16j, -1:1:16j, -1:1:16j]
        resource = engine.register_volume(np.exp(-6 * (x*x+y*y+z*z)).astype("f4"), name="blob")
        material = VolumeMaterial("magma", density_scale=5, opacity_stops=[(0, 0), (1, 0.95)])
        engine.spawn(Volume(volume_id=resource), Material(material),
                     Transform(pos=(2.42, 0.42, 0), scale=(0.8, 0.8, 0.8)))
    if omit != "label":
        index = engine.get_label_atlas().get_or_create("E5", font_size=36)
        engine.spawn(TextLabel(index=index), Material(LabelMaterial(pixel_width=44, pixel_height=22)),
                     Transform(pos=(-0.18, 2.02, 0)))
    if omit != "axis":
        engine.spawn(Mesh(AXIS_LINE_X), Material(AxisMaterial(color="#00ff99")),
                     Transform(pos=(1.9, -0.55, 0), scale=(1.0, 1.0, 1.0)), AxisFrame())
    if omit != "lines":
        engine.lines.set("line", [(-0.75, -1.75, 0)], [(0.88, -1.75, 0)],
                         color=(0.05, 0.4, 0.95, 1), width=5)
    if omit != "rect":
        rect = RectOp(LayoutBox(8, 8, 32, 16), (0.8, 0.24, 0.05, 1), (0, 0, 0, 0), 0, 0)
        engine._graphics.fixture_rects(RectMaterial.pack_instances([rect], exact=True).tolist(), False)
    if post:
        engine.set_post(grade=Grade(saturation=0.45), fxaa=True)
    if late:
        engine._graphics.fixture_render_fault("late-shadow")
    image = engine.render_frame(supersample=1)
    return image, engine.last_frame["frame_order"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--late-shadow", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert sys.flags.isolated and Path(mx.__file__).resolve().is_relative_to(sys.prefix)
    report = {}
    for name, post in (("plain", False), ("post", True)):
        image, order = capture(post, late=args.late_shadow)
        Image.fromarray(image).save(args.output / f"{name}.png")
        report[name] = {"order": order, "changed_pixels": {}}
        if not args.late_shadow:
            for stage in SCENE:
                muted, _ = capture(post, omit=stage)
                Image.fromarray(muted).save(args.output / f"{name}-without-{stage}.png")
                delta = np.max(np.abs(image.astype("i2") - muted), axis=2)
                report[name]["changed_pixels"][stage] = int((delta[REGIONS[stage]] > 4).sum())
        if args.reference:
            expected = np.asarray(Image.open(args.reference / f"{name}.png"))
            delta = np.abs(image.astype("i2") - expected)
            Image.fromarray(delta.astype("u1")).save(args.output / f"{name}-diff.png")
            report[name]["shadow_roi_error"] = float(delta[REGIONS["shadow"]].mean())
            report[name]["scene_p99_error"] = float(np.percentile(delta, 99))
    (args.output / "composition.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.reference:
        assert max(case["shadow_roi_error"] for case in report.values()) <= 1, "shadow dependency changed receiver ROI"
    if args.reference:
        assert max(case["scene_p99_error"] for case in report.values()) <= 2, "combined frame differs"
    for name, post in (("plain", False), ("post", True)):
        expected = ORDER + (["composite", "fxaa"] if post else []) + ["rect"]
        assert report[name]["order"] == expected, report[name]
        if not args.late_shadow:
            assert min(report[name]["changed_pixels"].values()) >= 8, report[name]
    print(json.dumps(report))


if __name__ == "__main__":
    main()
