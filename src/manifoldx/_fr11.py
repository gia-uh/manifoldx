"""FR-11 coast scene, for an installed native wheel or the Python demo."""

import argparse
import hashlib
import importlib.machinery
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image as PillowImage

import manifoldx as mx
from manifoldx.camera import Camera
from manifoldx.components import Baked, Fade, Layer, Material, Mesh, Palette, Tint, Transform
from manifoldx.compute import Compute, Reads, ReadsWrites, Uniform
from manifoldx.gui import Button, Image, Joystick, Panel, Slider, Text, Toggle, ValueDisplay
from manifoldx.modeling.primitives import cylinder, icosphere, torus
from manifoldx.post import SSAO, Bloom, Grade, LightShafts, Sky
from manifoldx.resources import (
    PALETTE_SURFACE,
    SURFACE_BAKED,
    AtmosphereMaterial,
    AtmosphereScatteringMaterial,
    BasicMaterial,
    CloudMaterial,
    DirectionalLight,
    FlatMaterial,
    PhongMaterial,
    PointLight,
    SeaMaterial,
    SpotLight,
    StandardMaterial,
    WaterMaterial,
    cube,
    plane,
    sphere,
)
from manifoldx.shader import Texture1D, Vec2, Vec4, sample
from manifoldx.streaming import MeshData, Streamer
from manifoldx.textures import layer_mips, texture_array, write_layer
from manifoldx.viz import (
    AxisFrame,
    AxisMaterial,
    ColormapMaterial,
    LabelMaterial,
    PointCloud,
    Radius,
    ScalarValue,
    TextLabel,
    Volume,
    VolumeMaterial,
)
from manifoldx.viz.geometry import AXIS_LINE_X

VERSION = 1
DESIGN = (640, 360)
GROUPS = (
    "basic", "flat", "phong", "pbr", "patterns", "palette", "baked", "fade", "cut",
    "mirror", "alpha", "textured", "skin", "fragment", "water", "sea", "glow", "scattering", "cloud", "sprite-lit",
    "sprite-unlit", "volume", "label", "axis", "lines", "stream", "compute",
    "view", "snapshot", "gui", "environment", "fog", "shadow", "point", "spot",
)
POST = {
    "ssao": SSAO(radius=1.0, intensity=1.35, samples=24),
    "bloom": Bloom(threshold=0.54, knee=0.1, intensity=0.7),
    "shafts": LightShafts(intensity=0.8, samples=32),
    "grade": Grade(saturation=0.78, contrast=1.18, vignette=0.3),
    "sky": Sky(zenith=(0.07, 0.26, 0.52), horizon=(0.9, 0.58, 0.3),
               sun_size=1.3, sun_intensity=1.0),
    "fxaa": True,
}
_OMISSIONS = (*GROUPS, *POST, "saturation", "contrast", "vignette")
_WATER_ROTATION = (math.sqrt(0.5), 0, 0, -math.sqrt(0.5))


class _Drift(Compute):
    # Reserve layer 3 for an observable compute-fed buoy. Layer is stored as f32.
    transforms: ReadsWrites[Transform]
    layers: Reads[Layer]
    count: Uniform[float] = "entity_count"
    frame_index: Uniform[float] = "frame_index"

    def compile(self):
        return """
struct Uniforms { count: f32, frame_index: f32 };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> layers: array<f32>;
@group(0) @binding(2) var<storage, read_write> transforms: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i < u32(u.count) && layers[i] == 3.0 {
        transforms[i * 10u] = 2.85 + 0.75 * sin(u.frame_index * 0.08);
    }
}
"""



@mx.shader.fragment
def _signal(uv: Vec2, colors: Texture1D) -> Vec4:
    return sample(colors, uv.x)


def _tile(i, j):
    # A contiguous strip of tessellated sand behind the material samples.
    if j != -1:
        return None
    xz = np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype="f4")
    heights = np.array([0.02, 0.14, 0.02, 0.14], dtype="f4")
    positions = np.column_stack((xz[:, 0], heights, xz[:, 1]))
    colors = np.tile((0.5 + 0.08 * (i % 2), 0.34, 0.12), (4, 1)).astype("f4")
    return MeshData(positions, np.tile((0, 1, 0), (4, 1)).astype("f4"), colors,
                    np.array([0, 1, 2, 2, 1, 3], dtype="u4"))


def _native(engine, false=False):
    import manifoldx._core_native as extension

    if (false or type(engine).__module__ != "manifoldx._desktop"
            or not isinstance(extension.__loader__, importlib.machinery.ExtensionFileLoader)):
        raise RuntimeError("FR-11 requires the native desktop Engine from a compiled wheel")
    return {"core": "rust", "renderer": "wgpu", "extension": Path(extension.__file__).name}


def _setup(args):
    width, height = DESIGN if args.offscreen else (1920, 1080)
    engine = mx.Engine("ManifoldX FR-11 Coast", width=width, height=height, max_entities=256,
                       fullscreen=args.fullscreen, monitor=args.monitor)
    provenance = _native(engine, args.fault == "provenance")
    omit = args.omit
    engine.gui.design_height = DESIGN[1]
    engine.background_color = (0.07, 0.13, 0.16)
    engine.camera.set_pose((0, 3.15, 10.5), (0, 0.9, 0))
    engine.camera.fov = 44
    engine.camera.layers = {0, 3}
    engine.set_sun(DirectionalLight("#fff4df", 2.9, (0.26, -0.26, 0.93)))
    spot = SpotLight("#cfe8ff", 16, (0, 4, 4), (0.05, -0.55, -1), 0.18, 0.42, 15)
    spot_active = False
    if omit != "point":
        engine.add_light(PointLight(position=(-3, 3, 3), color="#ffcf91", intensity=25))
    if omit != "shadow":
        engine.enable_shadows(extent=8, resolution=512, cascades=3, auto_fit=False, shadow_distance=20)
    if omit != "environment":
        engine.set_environment("sky")
        engine.environment.show_skybox = True
    if omit != "fog":
        engine.enable_fog(9, 30, (0.18, 0.26, 0.30))
    settings = {} if args.no_post else {key: value for key, value in POST.items() if key != omit}
    neutral = {"saturation": 1.0, "contrast": 1.0, "vignette": 0.0}
    if omit in neutral and settings:
        settings["grade"] = replace(POST["grade"], **{omit: neutral[omit]})
    engine.set_post(**settings)
    for component in (Layer, Tint, Palette, Baked, Fade, PointCloud, ScalarValue,
                      Radius, Volume, TextLabel, AxisFrame):
        engine.world.register(component)

    id_groups = []

    def mesh(group, geometry, material, position, *components, layer=0, **transform):
        if group is None or omit != group:
            entity = engine.spawn(Mesh(geometry), Material(material), Transform(pos=position, **transform),
                                  Layer(layer=layer), *components)
            if args.ids and group is not None and layer == 0:
                id_groups.append([int(entity.index)])
            return entity
        return None

    sand = StandardMaterial("#ccad6b", roughness=0.86)
    mesh(None, cube(11, 0.2, 3.5), sand, (0, -0.2, 0))
    mesh("sea", plane(15, 13), SeaMaterial(), (0, -0.28, -3.0), rot=_WATER_ROTATION)
    mesh("water", plane(5, 2.2), WaterMaterial(), (-2.4, -0.04, 2.0), rot=_WATER_ROTATION)
    mesh(None, cube(1.15, 2.4, 0.65), sand, (-3.3, 1.1, -1.1))
    mesh(None, cube(0.9, 0.18, 0.9), BasicMaterial("#ffffff"), (-3.3, 2.34, -1.1))
    mesh("mirror", plane(1.0, 1.2), StandardMaterial("#bd53df", mirror=-0.08), (-3.2, 1.0, 2.4))
    mesh("alpha", plane(1.1, 0.72), StandardMaterial("#318ee3", opacity=0.47, ripple=0.6), (-3.2, 0.34, 2.55))
    mesh("glow", sphere(0.71, 24), AtmosphereMaterial("#3aaeff", 1.3), (2.8, 2.15, -1.7))
    mesh("scattering", sphere(0.64, 24), AtmosphereScatteringMaterial(0.55, 0.64), (3.95, 2.0, -1.8))
    mesh(None, sphere(0.54, 24), StandardMaterial("#4f715d"), (3.95, 2, -1.8))
    mesh("cloud", sphere(1.0, 32), CloudMaterial(coverage=0.4, opacity=0.8), (0.3, 3.14, -2.5),
         scale=(1.6, 0.4, 0.75))

    balloon = cube(0.61, 0.61, 0.61)
    sample = sphere(0.42, 24)
    mesh("basic", balloon, BasicMaterial("#df5d4f"), (-2.58, 1.2, 1.0))
    mesh("flat", icosphere(0, 0.44).to_geometry(), FlatMaterial("#e6a452"), (-1.54, 1.2, 1.0))
    mesh("phong", cylinder(0.33, 0.74, 20).to_geometry(), PhongMaterial("#74ded6", shininess=22),
         (-0.52, 1.2, 1.0))
    for column, (roughness, metallic, ao) in enumerate(((0.25, 0.8, 1), (0.62, 0.15, 0.7))):
        mesh("pbr", sample, StandardMaterial("#a988fd", roughness=roughness, metallic=metallic, ao=ao),
             (0.52 + column, 1.2, 1.0))
    mesh("cut", balloon, StandardMaterial("#e7dc51", cut=(2.35, 0.7, 2.72, 1.31)),
         (2.34, 1.2, 1.0))
    mesh("fade", balloon, StandardMaterial("#ec704f", faded=True), (3.28, 1.2, 1.0), Fade(value=0.56))

    def painted(geometry, surface):
        return {**geometry, "surfaces": np.full(len(geometry["positions"]), surface, dtype="i4"),
                "colors": np.ones_like(geometry["positions"], dtype="f4")}

    pattern = StandardMaterial("#b6ae99", vertex_colors=True, detail=True, tinted=True, sway=0.25)
    tile = cube(0.44, 0.44, 0.44)
    for number in range(1, 9):
        mesh("patterns", painted(tile, number), pattern, (-2.57 + (number - 1) * 0.77, 0.42, 1.35),
             Tint(rgb=(0.5, 0.75, 0.9)))
    mesh("palette", painted(torus(0.26, 0.10, 20, 12).to_geometry(), PALETTE_SURFACE),
         StandardMaterial("#ffffff", vertex_colors=True, paletted=True), (-1.5, 2.07, 0.8),
         Palette(c0=(0.96, 0.32, 0.22)))
    atlas = texture_array(engine, layers=2, size=16)
    swatch = np.empty((16, 16, 3), dtype="u1")
    swatch[:, :8] = (60, 190, 198)
    swatch[:, 8:] = (242, 151, 70)
    write_layer(engine, atlas, 1, layer_mips(swatch))
    baked = painted(plane(0.75, 0.62), SURFACE_BAKED)
    baked["baked_uv"] = baked["uvs"]
    mesh("baked", baked, StandardMaterial("#ffffff", vertex_colors=True, baked=atlas),
         (-0.51, 2.07, 0.85), Baked(layer=1))

    image = np.full((16, 16, 4), (235, 154, 45, 255), dtype="u1")
    image[::2, ::2] = (34, 143, 187, 180)
    texture = engine._upload_native_albedo(PillowImage.fromarray(image), None)
    mesh("textured", plane(0.5, 0.48), StandardMaterial("#ffffff", albedo_map=texture, opacity=0.88),
         (1.16, 2.77, 0.15))
    mesh("fragment", plane(0.53, 0.35), mx.ShaderMaterial(
        _signal, colors=np.array([[96, 208, 154, 255], [238, 98, 182, 255]], dtype="u1"),
    ), (-1.15, 2.8, 0.34))
    flag = plane(0.72, 0.46)
    flag["joints"] = np.tile([0, 1], (4, 1))
    flag["weights"] = np.array([0.15, 0.85, 0.85, 0.15], dtype="f4")
    pennant = mesh("skin", flag, FlatMaterial("#f27c6a"), (-3.3, 2.72, -1.0))
    clip_id = None
    if pennant is not None:
        rest = np.repeat(np.eye(4, dtype="f4")[None], 2, axis=0)
        wave = rest.copy()
        wave[1, 1, 3] = 0.25
        wave[1, 2, 0] = 0.45
        clip_id = engine.bake_poses(np.stack([rest, wave]))

    for x, lit in ((0.43, True), (1.28, False)):
        if omit != ("sprite-lit" if lit else "sprite-unlit"):
            engine.spawn(PointCloud(), Material(ColormapMaterial("viridis", 0, 1, lit=lit)),
                         Radius(radius=0.27), ScalarValue(value=0.82), Transform(pos=(x, 2.04, 1)))
    if omit != "volume":
        z, y, x = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
        volume = engine.register_volume(np.exp(-5 * (x*x + y*y + z*z)).astype("f4"), name="mist")
        engine.spawn(Volume(volume_id=volume),
                     Material(VolumeMaterial("magma", density_scale=7, opacity_stops=[(0, 0), (1, 0.95)])),
                     Transform(pos=(2.06, 2.04, 1.0), scale=(0.75, 0.75, 0.75)))
    if omit != "label":
        title = engine.get_label_atlas().get_or_create("Material samples", font_size=36)
        engine.spawn(TextLabel(index=title), Material(LabelMaterial(pixel_width=138, pixel_height=19)),
                     Transform(pos=(0, 2.82, 1.3)))
    if omit != "axis":
        engine.spawn(Mesh(AXIS_LINE_X), Material(AxisMaterial(color="#4ced93")),
                     Transform(pos=(2.25, 0.12, 2.42)), AxisFrame())
    if omit != "lines":
        engine.lines.set("shore", [(-4.8, 0.03, 1.85)], [(4.1, 0.03, 1.85)],
                         color=(0.9, 0.6, 0.16, 1.0), width=2)
    mesh(None, sphere(0.21, 16), BasicMaterial("#fff1bf"), (2.85, 0.5, 2.34), layer=3)
    if omit != "compute":
        engine.compute(_Drift)

    side = Camera()
    side.layers = {1}
    side.set_pose((0, 0, 5), (0, 0, 0))
    mesh(None, sphere(0.88, 32), PhongMaterial("#df6436"), (0, 0, 0), layer=1)
    mesh(None, cube(1, 1, 1), BasicMaterial("#5078de"), (3, 0, 0), layer=1)
    live = None if omit == "view" else engine.render_to_texture(side, (160, 120), fog=False)
    photo_camera = Camera()
    photo_camera.layers = {1}
    photo_camera.set_pose((0, 0, 5), (0, 0, 0))
    photo = None if omit == "snapshot" else engine.render_to_texture(photo_camera, (160, 120), live=False)

    status = Text("Drag past the panel", style_overrides={"fg": "#f5e7c4"})
    slider = Slider(name="tide", min=0, max=1, value=0.25)
    refresh = Button(name="photo", label="Refresh photo")
    scale = Toggle(name="large", value=False, label="Large HUD")
    timeline = {
        "state": "start", "value": 0.25, "photo": 0, "events": [], "cells": [],
        "shadow_caster": "sun" if omit != "shadow" else "none",
    }

    def record(name, fields):
        if args.output is not None:
            timeline["events"].append({"name": name, "frame_time": engine.elapsed, **fields})

    def refresh_photo():
        if photo is not None:
            engine.refresh_texture(photo)
            timeline["photo"] += 1

    @engine.on("gui:button:photo:click")
    def _photo(_):
        refresh_photo()
        record("button:photo", {"refreshes": timeline["photo"]})

    @engine.on("gui:toggle:large:change")
    def _scale(payload):
        engine.gui.design_height = 285 if payload["value"] else DESIGN[1]
        record("toggle:large", payload)

    @engine.on("gui:slider:tide:change")
    def _tide(payload):
        timeline["value"] = payload["value"]
        status.text = f"Tide {payload['value']:.2f}"
        record("slider:tide:change", payload)

    @engine.on("gui:slider:tide:commit")
    def _commit(payload):
        record("slider:tide:commit", payload)

    if args.output is not None and not args.offscreen:
        def observe(event, payload, consumed, target):
            record(event, {"input": asdict(payload), "consumed": consumed, "target": target})

        engine._input_probe = observe
        engine.on("resize")(lambda event: record("resize", {"input": asdict(event)}))

    if omit != "gui":
        engine.gui.append(Panel([
            Text("FR-11 Coast / v1", style_overrides={"font_size": 16, "fg": "#ffefca"}),
            status, slider, refresh, scale,
            ValueDisplay(lambda: f"{timeline['state']} | photo {timeline['photo']}", min_width=120),
            Joystick(name="map", style_overrides={"height": 36}),
        ], offset=(10, 10), style_overrides={"width": 210, "height": 190, "bg": "#1b354be8",
                                            "padding": 8, "border": 1, "radius": 8}))
        views = []
        if live is not None:
            views.extend([Text("Live minimap"), Image(live, style_overrides={"width": 112, "radius": 35})])
        if photo is not None:
            views.extend([Text("Snapshot"), Image(photo, style_overrides={"width": 112})])
        engine.gui.append(Panel(views, anchor="bottom-right", offset=(10, 10),
                                style_overrides={"bg": "#1b354be8", "width": 128, "padding": 8, "radius": 6}))

    streamer = None if omit == "stream" else Streamer(
        engine, 2, _tile, radius=1, ready_radius=1, material=StandardMaterial("#ffffff", vertex_colors=True),
    )

    @engine.system
    def animate(query, dt):
        nonlocal spot_active
        t = engine.elapsed
        # The desktop renderer uses one shadow caster. Keep the cascades
        # visible in the initial composition, then show the spot's map.
        if t >= 2.5 and not spot_active and omit != "spot":
            engine.set_spot(spot)
            spot_active = True
            if omit != "shadow":
                timeline["shadow_caster"] = "spot"
        if clip_id is not None:
            engine.set_pose_frames([pennant.index], clip_id, [0.5 - 0.5 * math.cos(t * 5)])
        side_x = 1.5 - 1.5 * math.cos(t * 1.5) + 0.28 * timeline["value"]
        side.set_pose((side_x, 0.25, 5), (side_x, 0, 0))
        photo_camera.set_pose(side.position, side.target)
        if streamer is not None:
            streamer.focus(-2 + ((t * 3) % 7), -1)
            timeline["cells"] = sorted(streamer.drawn())
        timeline["state"] = "static" if t < 0.5 else "animation" if t < 2 else "streaming"
        if t >= 3.5 and timeline["photo"] == 0:
            refresh_photo()

    if args.fault == "hud-order":
        engine._graphics.fixture_gui_before_post(True)
    elif args.fault == "capture":
        engine._graphics.fixture_gui_omit_capture(True)
    return engine, timeline, provenance, id_groups


def _frame_record(engine, timeline, provenance, label):
    report = engine.last_frame
    return {
        "version": VERSION,
        "label": label,
        "timeline": {key: value for key, value in timeline.items() if key != "events"},
        "provenance": provenance,
        "format": report["target"],
        "passes": report["frame_order"],
        "settings": report["settings"],
        "counts": {key: report[key] for key in (
            "instances", "draw_calls", "triangles", "shadow_layers", "gui_quads",
            "view_resources", "stream_resources", "streamed_cells", "gui_getter_calls",
        )},
    }


def _offscreen(engine, timeline, provenance, id_groups, args):
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    previous_photo = 0
    previous_caster = timeline["shadow_caster"]
    for step in range(args.frames):
        image = engine.render_frame(dt=1 / 12)
        photo_count = timeline["photo"]
        caster = timeline["shadow_caster"]
        transition = photo_count != previous_photo or caster != previous_caster
        if step in {0, 1, args.frames // 2, args.frames - 1} or transition:
            previous_photo = photo_count
            previous_caster = caster
            name = f"frame-{step:03d}"
            path = args.output / f"{name}.png"
            PillowImage.fromarray(image).save(path)
            frame = _frame_record(engine, timeline, provenance, name)
            frame["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            frames.append(frame)
    if args.ids:
        image = engine.render_frame(pass_="ids", groups=id_groups)
        np.save(args.output / "ids.npy", image)
    (args.output / "manifest.json").write_text(json.dumps(frames, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--monitor")
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument("--omit", choices=_OMISSIONS)
    parser.add_argument("--fault", choices=("hud-order", "capture", "provenance"))
    parser.add_argument("--ids", action="store_true", help="save diagnostic IDs separately from beauty")
    args = parser.parse_args(argv)
    if args.offscreen and (args.output is None or not 1 <= args.frames <= 900):
        parser.error("--offscreen requires --output and 1 to 900 frames")
    if args.ids and not args.offscreen:
        parser.error("--ids requires --offscreen")
    engine, timeline, provenance, id_groups = _setup(args)
    try:
        if args.offscreen:
            _offscreen(engine, timeline, provenance, id_groups, args)
        else:
            engine.run()
            if args.output is not None:
                args.output.mkdir(parents=True, exist_ok=True)
                report = {"version": VERSION, "provenance": provenance,
                          "desktop": engine.desktop_report, "timeline": timeline}
                (args.output / "desktop.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
