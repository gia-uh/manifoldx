"""The plaza at dusk, end to end: paving, hall, fountain, crowd, people and panel."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.gui import Panel, Slider, Text, Toggle, ValueDisplay, style
from manifoldx.gui.layout import LayoutBox, anchored_slot, compute_layout
from manifoldx.humanoid import Proportions, facing, solve, spawn_mannequin
from manifoldx.ibl import EnvironmentMap
from manifoldx.input import PointerEvent
from manifoldx.modeling import Gradient, Mesh as GeoMesh, fields
from manifoldx.modeling.primitives import cylinder
from manifoldx.physics import all_pairs
from manifoldx.resources import (
    DirectionalLight,
    FlatMaterial,
    PointLight,
    StandardMaterial,
    WaterMaterial,
    cube,
    plane,
    sphere,
)
from manifoldx.trace import Camera, Environment, Scene, Tracer
from manifoldx.types import Float, Vector3

W, H = 480, 270
N = 200
FLOOR = Transform.rotation(x=-np.pi / 2)
engine = mx.Engine("plaza", width=W, height=H)
engine.background_color = (0.06, 0.07, 0.13)
engine.camera.fov = 44
engine.camera.set_pose(position=(10.0, 5.5, 12.0), target=(0, 3.2, -14))
engine.set_sun(DirectionalLight(color="#ffab63", intensity=2.8, direction=(0.55, -0.28, -0.45)))
engine.enable_shadows(
    resolution=1024, bias=0.004, pcf_radius=2, auto_fit=False, target=(0, 0, -8), extent=26, far=80
)

dusk = EnvironmentMap.from_sky(
    zenith=(0.08, 0.12, 0.30),
    horizon=(0.62, 0.34, 0.20),
    ground=(0.09, 0.08, 0.07),
    height=32,
    width=64,
)
dusk.intensity = 0.08
engine.set_environment(dusk)

# The street lamp of chapter 4.
LAMP = (-4.8, 1.4)
pole = StandardMaterial(color="#2a2c30", roughness=0.4, metallic=0.6)
engine.spawn(Mesh(cube(0.14, 4.2, 0.14)), Material(pole), Transform(pos=(LAMP[0], 2.1, LAMP[1])))
engine.spawn(
    Mesh(cube(0.9, 0.1, 0.1)), Material(pole), Transform(pos=(LAMP[0] + 0.4, 4.2, LAMP[1]))
)
engine.spawn(
    Mesh(sphere(0.2, 20)),
    Material(FlatMaterial("#ffe2b0")),
    Transform(pos=(LAMP[0] + 0.8, 4.02, LAMP[1])),
)
lamp = PointLight(color="#ffb464", intensity=110.0, position=(LAMP[0] + 0.8, 4.02, LAMP[1]))
engine.set_lights([lamp])

# The paving of chapter 6, from stone cells and two noise fields.
settle = fields.fbm(seed=4, freq=0.12, octaves=3) * 0.05
wear = fields.fbm(seed=11, freq=0.9, octaves=3) * 0.03
paving = GeoMesh.plane(width=40, depth=40, segments=120).displace(settle + wear, amount=1.0)
stones = fields.worley(seed=3, freq=1.1, feature="f2f1")
paving = paving.color_by(
    stones,
    Gradient([(0.00, "#6a6253"), (0.10, "#948b78"), (0.30, "#c3b7a0"), (0.85, "#ddd2ba")]),
)
engine.spawn(
    Mesh(paving.to_geometry()),
    Material(StandardMaterial(color="#ffffff", roughness=0.95, vertex_colors=True)),
    Transform(),
)
engine.spawn(
    Mesh(plane(120, 120)),
    Material(StandardMaterial(color="#33342e", roughness=1.0)),
    Transform(pos=(0, -0.08, 0), rot=FLOOR),
)
frame = engine.render_frame(supersample=1)
print(frame.shape, frame.dtype)
Image.fromarray(frame)
# The town hall of chapter 6.
def along_z(mesh):
    """Turn a mesh a quarter turn about x, so a y-axis cylinder runs along z."""
    x, y, z = mesh.positions.T
    return mesh.with_positions(np.stack([x, -z, y], axis=1))


def build_hall(engine):
    """A stone facade with two arched doors and a round window, a body and a dome."""
    wall = GeoMesh.box(16, 10, 0.6).translate((0, 5, 0))
    top = wall.positions[wall.faces].mean(axis=1)[:, 1] > 9.99
    cornice = wall.extrude(top, 0.7)

    door = GeoMesh.box(1.9, 2.7, 2.0).translate((-3.6, 1.35, 0))
    arch = along_z(GeoMesh.cylinder(radius=0.95, height=2.0, segments=24))
    doorway = door.union(arch.translate((-3.6, 2.7, 0)))
    window = along_z(GeoMesh.cylinder(radius=0.95, height=2.0, segments=24))
    cutter = doorway.union(doorway.translate((7.2, 0, 0))).union(window.translate((0, 7.1, 0)))
    facade = cornice.difference(cutter).translate((0, 0, -20.0))
    engine.spawn(
        Mesh(facade.to_geometry()),
        Material(StandardMaterial(color="#cfc3a6", roughness=0.9)),
        Transform(),
    )

    body = GeoMesh.box(16, 10, 8).translate((0, 5, -25.0))
    engine.spawn(
        Mesh(body.to_geometry()),
        Material(StandardMaterial(color="#c9bda0", roughness=0.9)),
        Transform(),
    )
    drum = GeoMesh.cylinder(radius=4.05, height=3.0, segments=40).translate((0, 11.5, -24.5))
    engine.spawn(
        Mesh(drum.to_geometry()),
        Material(StandardMaterial(color="#c9bda0", roughness=0.9)),
        Transform(),
    )

    ball = GeoMesh.icosphere(subdivisions=1, radius=4.2).subdivide(iterations=1, scheme="loop")
    ball = ball.inflate(center=(3.0, 2.4, 1.6), radius=2.0, strength=-0.35)
    ball = ball.smooth(iterations=3, strength=0.5, center=(3.0, 2.4, 1.6), radius=2.8)
    dome = ball.intersection(GeoMesh.box(20, 20, 20).translate((0, 10, 0)))
    dome = dome.decimate(grid=128)
    lattice = dome.ffd(resolution=(3, 3, 3))
    for i in range(4):
        for k in range(4):
            lattice.move(i, 3, k, (0.0, 0.25, 0.0))
    dome = lattice.apply().translate((0, 13.0, -24.5))
    engine.spawn(
        Mesh(dome.to_geometry()),
        Material(StandardMaterial(color="#7d9b8c", roughness=0.4, metallic=0.25)),
        Transform(),
    )


build_hall(engine)
frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
# The fountain of chapter 5 and the benches of chapter 12.
def build_fountain(engine):
    """Four stone walls on a bottom, bronze rims, a water plane and a small jet."""
    stone = StandardMaterial(color="#9a958c", roughness=0.8)
    bronze = StandardMaterial(color="#c08a4a", metallic=1.0, roughness=0.2)
    engine.spawn(Mesh(cube(4, 0.2, 4)), Material(stone), Transform(pos=(0, 0.1, 0)))
    for z in (-1.9, 1.9):
        engine.spawn(Mesh(cube(4, 0.9, 0.2)), Material(stone), Transform(pos=(0, 0.55, z)))
        engine.spawn(Mesh(cube(4.2, 0.12, 0.3)), Material(bronze), Transform(pos=(0, 1.02, z)))
    for x in (-1.9, 1.9):
        engine.spawn(Mesh(cube(0.2, 0.9, 3.6)), Material(stone), Transform(pos=(x, 0.55, 0)))
        engine.spawn(Mesh(cube(0.3, 0.12, 3.6)), Material(bronze), Transform(pos=(x, 1.02, 0)))
    pool = engine.spawn(
        Mesh(plane(3.6, 3.6)),
        Material(WaterMaterial(color="#0a2352", fresnel_power=4.0)),
        Transform(pos=(0, 0.88, 0), rot=FLOOR),
    )
    engine.spawn(Mesh(cube(0.6, 1.2, 0.6)), Material(stone), Transform(pos=(0, 0.7, 0)))
    engine.spawn(Mesh(sphere(0.34, 32)), Material(bronze), Transform(pos=(0, 1.45, 0)))
    jet = engine.spawn(
        Mesh(cube(0.14, 2.0, 0.14)),
        Material(WaterMaterial(color="#7fc4e8", fresnel_power=2.0)),
        Transform(pos=(0, 2.4, 0)),
    )
    return pool, jet


def build_bench(engine, at, yaw, wood, stone):
    """A seat, a back and two legs, turned to face the fountain."""
    x, z = at
    rot = Transform.rotation(y=yaw)
    along_x, along_z = np.cos(yaw), -np.sin(yaw)
    out_x, out_z = np.sin(yaw), np.cos(yaw)
    engine.spawn(Mesh(cube(1.9, 0.1, 0.55)), Material(wood), Transform(pos=(x, 0.46, z), rot=rot))
    engine.spawn(
        Mesh(cube(1.9, 0.55, 0.09)),
        Material(wood),
        Transform(pos=(x + 0.24 * out_x, 0.74, z + 0.24 * out_z), rot=rot),
    )
    for side in (-0.8, 0.8):
        engine.spawn(
            Mesh(cube(0.12, 0.46, 0.5)),
            Material(stone),
            Transform(pos=(x + side * along_x, 0.23, z + side * along_z), rot=rot),
        )


pool, jet = build_fountain(engine)
wood = StandardMaterial(color="#6b4a2f", roughness=0.8)
seat_stone = StandardMaterial(color="#8f8b82", roughness=0.85)
BENCHES = [(4.7 * np.sin(a), 4.7 * np.cos(a), a) for a in (0.8 + k * np.pi / 2 for k in range(4))]
for x, z, yaw in BENCHES:
    build_bench(engine, (x, z), yaw, wood, seat_stone)

frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
# The crowd of chapter 2.
class Walker(Component):
    """What one person knows: where they are going and how fast."""

    velocity: Vector3
    destination: Vector3
    pace: Float = 1.2


rng = np.random.default_rng(11)
start = np.column_stack(
    [
        rng.uniform(-19.0, 18.0, N),
        np.full(N, 0.85),
        rng.uniform(-19.0, 5.0, N),
    ]
).astype(np.float32)
# Nobody starts inside the basin: push the few who do out to the rim.
flat = start[:, [0, 2]]
rad = np.linalg.norm(flat, axis=1, keepdims=True)
inside = rad[:, 0] < 3.2
start[inside, 0] = flat[inside, 0] / rad[inside, 0] * 3.2
start[inside, 2] = flat[inside, 1] / rad[inside, 0] * 3.2
destinations = start * np.array([-0.8, 1.0, -0.8], np.float32)
pace = rng.uniform(0.9, 1.6, N).astype(np.float32)
HEIGHTS = rng.uniform(0.92, 1.08, (N, 3)).astype(np.float32)
COATS = ["#c8553d", "#f2c14e", "#5b8e7d", "#e8e2d6"]
LIVE = [N]


def spawn_walkers(engine):
    """Two hundred people in four coats, crossing the square."""
    body = Mesh(cylinder(0.24, 1.7, 10).to_geometry())
    for i, coat in enumerate(COATS):
        group = slice(i * 50, (i + 1) * 50)
        engine.spawn(
            body,
            Material(StandardMaterial(color=coat, roughness=0.8)),
            Transform(pos=start[group]),
            Walker(
                velocity=np.zeros((50, 3), np.float32),
                destination=destinations[group],
                pace=pace[group],
            ),
            n=50,
        )


spawn_walkers(engine)


@engine.system
def walk(query: mx.Query[Transform, Walker], dt: float):
    pos = query[Transform].pos.data
    vel = query[Walker].velocity.data
    dest = query[Walker].destination.data
    pace = query[Walker].pace.data[:, 0]

    to = dest - pos
    dist = np.linalg.norm(to, axis=1, keepdims=True)
    wanted = to / np.maximum(dist, 1e-6) * pace[:, None]
    wanted[dist[:, 0] < 0.6] = 0.0

    # The basin is solid: anyone who comes close is steered around it.
    flat = pos[:, [0, 2]]
    rad = np.linalg.norm(flat, axis=1, keepdims=True)
    near = rad[:, 0] < 3.0
    wanted[near, 0] = flat[near, 0] / rad[near, 0] * 1.5
    wanted[near, 2] = flat[near, 1] / rad[near, 0] * 1.5

    pairs = all_pairs(pos, softening=0.01)
    overlap = np.maximum(0.9 - pairs.dist, 0.0)
    np.fill_diagonal(overlap, 0.0)
    push = (-pairs.diff / pairs.dist_safe[..., None] * overlap[..., None]).sum(axis=1)
    wanted += push * 1.5

    new_vel = vel + (wanted - vel) * np.minimum(1.0, 4.0 * dt)
    query[Walker].velocity += new_vel - vel
    query[Transform].pos += new_vel * dt

    scale = np.zeros((N, 3), np.float32)
    scale[: LIVE[0]] = HEIGHTS[: LIVE[0]]
    query[Transform].scale = scale


@engine.system
def face(query: mx.Query[Transform, Walker], dt: float):
    to = query[Walker].destination.data - query[Transform].pos.data
    query[Transform].rot = Transform.rotation(y=np.arctan2(to[:, 0], to[:, 2]))


engine.set_fixed_timestep(0.05)
for _ in range(24):
    frame = engine.render_frame(supersample=1)
print("people on the square:", LIVE[0])
Image.fromarray(frame)
# The posed people of chapter 7.
woman = Proportions.measured("female")
man = Proportions.measured("male")


def seat_people(engine):
    """Two people on the benches and one at the rim, all facing the water."""
    for k, (x, z, _) in enumerate((BENCHES[0], BENCHES[3])):
        figure = solve(
            "seated",
            woman if k == 0 else man,
            height=1.65 if k == 0 else 1.78,
            yaw=facing((x, z), (0, 0)),
        )
        spawn_mannequin(engine, figure, at=(x, 0, z), color="#d1495b" if k == 0 else "#2a6f97")
    stand = (0.0, 0, -3.1)
    figure = solve(
        {"base": "stand", "neck": {"lean": 6}}, man, height=1.78, yaw=facing(stand, (0.6, 4.0))
    )
    spawn_mannequin(engine, figure, at=stand, color="#e9a23b")


seat_people(engine)
frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
# The panel of chapter 9.
style.set_theme({"font_size": 11, "fg": "#e8e2d6", "gap": 5})
style.define(
    "panel",
    {
        "bg": "#12161ce8",
        "radius": 10,
        "padding": 9,
        "gap": 5,
        "width": 150,
        "border": 1,
        "border_color": "#3a4552",
    },
)
style.define("title", {"font_size": 13, "fg": "#f2c14e"})
style.define("readout", {"font_size": 10, "fg": "#9fb3c8"})


def frame_ms() -> str:
    split = engine.frame_split
    return f"frame {1000 * (split['update'] + split['acquire'] + split['render']):.0f} ms"


slider = Slider(name="people", min=0, max=N, value=N)
toggle = Toggle(name="jet", value=True, label="Jet")
readout = ValueDisplay(getter=frame_ms, style="readout", min_width=120)
panel = Panel(
    children=[Text("Plaza at dusk", style="title"), slider, toggle, readout],
    style="panel",
    anchor="bottom-right",
    offset=(10, 10),
)
engine.gui.append(panel)


@engine.on("gui:slider:people:change")
def on_people(payload):
    LIVE[0] = int(round(payload["value"]))


@engine.on("gui:toggle:jet:change")
def on_jet(payload):
    jet.transform.scale = (1, 1, 1) if payload["value"] else (0, 0, 0)


def box_of(widget):
    """Where the panel's layout puts one of its children, in UI pixels."""
    spec = panel.build_layout_spec()
    scale = engine.gui.effective_scale(engine.h)
    viewport = LayoutBox(0, 0, engine.w / scale, engine.h / scale)
    slot = anchored_slot(panel.anchor, panel.offset, spec, viewport)
    boxes = compute_layout(spec, viewport=slot)
    for child, child_spec in zip(panel.children, spec["children"]):
        if child is widget:
            return boxes[id(child_spec)]
    raise ValueError("not a child of the panel")


def pointer(phase_name, x, y):
    engine.emit(
        f"pointer_{phase_name}",
        PointerEvent(
            x=x, y=y, dx=0.0, dy=0.0, button=1, buttons=(1,), modifiers=(), phase=phase_name
        ),
    )


engine.render_frame(supersample=1)
engine.render_frame(supersample=1)
box = box_of(slider)
pointer("down", box.x + box.w * 0.6, box.y + box.h / 2)
pointer("up", box.x + box.w * 0.6, box.y + box.h / 2)
engine.render_frame(supersample=1)
frame = engine.render_frame(supersample=1)
print("people after the click:", LIVE[0])
Image.fromarray(frame)
# The path-traced still of chapter 12.
LIVE[0] = N
engine.render_frame(supersample=1)
mirror = StandardMaterial(color="#123a56", roughness=0.04, metallic=1.0)
spray = StandardMaterial(color="#9fd0e8", roughness=0.1, metallic=1.0)
pool.set_material(mirror)
jet.set_material(spray)

path = Path(tempfile.gettempdir()) / "plaza-end-to-end.glb"
report = engine.export_gltf(path)
print(len(report), "things left out")
scene = Scene.from_glb(path)
print(scene.triangle_count, "triangles")

sun = scene.lights[0]
sun.color = (1.0, 0.70, 0.44)
sun.intensity = 7.0
sky = Environment.from_map(dusk)
sky.intensity = 1.1
camera = Camera(position=(10.5, 5.8, 13.0), target=(0, 3.4, -9.0), fov=46.0)
tracer = Tracer(scene, 320, 180, camera=camera, environment=sky)
seconds = tracer.step(6)
print(f"{tracer.samples} samples per pixel in {seconds:.1f} s")
Image.fromarray(tracer.image())
