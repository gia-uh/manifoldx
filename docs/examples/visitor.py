import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.commands import Command, CommandType
from manifoldx.components import Material, Mesh, Transform
from manifoldx.input import KeyEvent, PointerEvent, ResizeEvent, WheelEvent
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane
from manifoldx.systems import Query

engine = mx.Engine("plaza", width=480, height=270)
FLOOR = (-0.70710678, 0.0, 0.0, 0.70710678)  # a quarter turn about x

STONE = StandardMaterial(color="#c8b89a", roughness=0.9)
BASIN = StandardMaterial(color="#8a9bb0", roughness=0.6)
WOOD = StandardMaterial(color="#7a5230", roughness=0.8)
GOLD = StandardMaterial(color="#ffd166", roughness=0.35)
WATER = StandardMaterial(color="#7fbfe0", roughness=0.15)

engine.spawn(Mesh(plane(40, 40)), Material(STONE), Transform(pos=(0, 0, 0), rot=FLOOR))
engine.spawn(Mesh(cube(4, 1, 4)), Material(BASIN), Transform(pos=(0, 0.5, 0)))

benches = []
for x, z in ((4.5, 0.0), (-4.5, 0.0), (0.0, 4.5)):
    parts = [
        engine.spawn(Mesh(cube(2.2, 0.14, 0.65)), Material(WOOD), Transform(pos=(x, 0.45, z))),
        engine.spawn(Mesh(cube(1.9, 0.38, 0.5)), Material(WOOD), Transform(pos=(x, 0.19, z))),
    ]
    benches.append({"pos": np.array([x, 0.45, z]), "parts": parts})

jet = engine.spawn(
    Mesh(cube(0.4, 1.0, 0.4)),
    Material(WATER),
    Transform(pos=(0, 0.99, 0), scale=(1, 0.01, 1)),
)

engine.camera.set_pose(position=(8, 4.5, 8), target=(0, 0.7, 0))
engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.4, -1, -0.3)))
print("the plaza holds", len(engine.store.get_component_view([])), "entities")
journal = []


@engine.on("startup")
def ready(_payload):
    journal.append(("startup", round(engine.elapsed, 2)))


@engine.on("frame")
def every(payload):
    journal.append(("frame", payload["frame"], round(payload["elapsed"], 2)))


@engine.on("knock")
def knock(payload):
    journal.append(("knock", payload["n"]))
    engine.emit("answered", payload["n"])


@engine.on("answered")
def answered(n):
    journal.append(("answered", n))


engine.emit("knock", {"n": 1})
engine.render_frame(supersample=1)
print(journal)
frame = engine.render_frame(supersample=1)
print(journal)

engine.off("frame", every)
journal.clear()
engine.render_frame(supersample=1)
print("after off:", journal)
Image.fromarray(engine.render_frame(supersample=1))


@engine.on("inspect")
def inspect(_payload, view: Query[Transform]):
    print("handler sees", len(view), "transforms")
    print("the first bench is at", view[Transform].pos.data[2:4].round(2))
    try:
        view[Transform].pos = np.zeros((len(view), 3))
    except RuntimeError as exc:
        print("blocked:", exc)


engine.emit("inspect", None)
engine.render_frame(supersample=1)


held: set[str] = set()
selected: int | None = None


@engine.on("key_down")
def press(ev: KeyEvent):
    held.add(ev.key)


@engine.on("key_up")
def release(ev: KeyEvent):
    held.discard(ev.key)


@engine.on("frame")
def walk(payload):
    step = 4.0 * payload["dt"]
    forward = engine.camera.get_forward()
    forward[1] = 0.0
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0, 1, 0))
    move = np.zeros(3)
    if "w" in held:
        move += forward * step
    if "s" in held:
        move -= forward * step
    if "d" in held:
        move += right * step
    if "a" in held:
        move -= right * step
    if np.any(move):
        engine.camera.set_pose(engine.camera.position + move, engine.camera.target + move)


@engine.on("pointer_down")
def pick(ev: PointerEvent):
    global selected
    if ev.button != 1:
        return
    centres = np.array([bench["pos"] for bench in benches])
    pixels, _ = engine.camera.project(centres, engine.w, engine.h)
    distance = np.linalg.norm(pixels - (ev.x, ev.y), axis=1)
    hit = int(np.argmin(distance))
    if distance[hit] > 40:
        hit = None
    if hit == selected:
        return
    if selected is not None:
        for part in benches[selected]["parts"]:
            part.set_material(WOOD)
    selected = hit
    if selected is not None:
        for part in benches[selected]["parts"]:
            part.set_material(GOLD)
    print(f"click at ({ev.x:.0f}, {ev.y:.0f}) selected bench {selected}")


@engine.on("wheel")
def zoom(ev: WheelEvent):
    offset = engine.camera.position - engine.camera.target
    distance = np.linalg.norm(offset)
    wanted = np.clip(distance * (1.0 - ev.dy * 0.001), 3.0, 30.0)
    engine.camera.set_pose(engine.camera.target + offset / distance * wanted, engine.camera.target)


@engine.on("resize")
def resized(ev: ResizeEvent):
    print(f"window now {ev.width}x{ev.height} at {ev.pixel_ratio}x")


start = engine.camera.position.copy()
engine.emit("key_down", KeyEvent("w", (), True))
engine.render_frame(supersample=1)  # the press arrives at the end of this frame
for _ in range(4):
    engine.render_frame(supersample=1)
engine.emit("key_up", KeyEvent("w", (), False))
print(
    "walked",
    np.linalg.norm(engine.camera.position - start).round(2),
    "m to",
    engine.camera.position.round(2),
)

pixels, _ = engine.camera.project(benches[0]["pos"], engine.w, engine.h)
engine.emit(
    "pointer_down",
    PointerEvent(float(pixels[0, 0]), float(pixels[0, 1]), 0.0, 0.0, 1, (1,), (), "down"),
)
engine.render_frame(supersample=1)

before = float(np.linalg.norm(engine.camera.position - engine.camera.target))
engine.emit("wheel", WheelEvent(0.0, 100.0, 0.0, 0.0, (), ()))
engine.render_frame(supersample=1)
after = float(np.linalg.norm(engine.camera.position - engine.camera.target))
print("camera distance", round(before, 2), "->", round(after, 2))

engine.emit("resize", ResizeEvent(800, 450, 1.0))
engine.render_frame(supersample=1)
print("the canvas is still", engine.w, "x", engine.h)

engine.emit("key_down", KeyEvent("w", (), True))
engine.render_frame(supersample=1)
print("polling sees", engine.input.pressed_keys, engine.input.mouse_pos)
print("capture offscreen:", engine.input.capture_pointer(True))
engine.emit("key_up", KeyEvent("w", (), False))
Image.fromarray(engine.render_frame(supersample=1))


def alive():
    return len(engine.store.get_component_view([]))


crate = engine.spawn(Mesh(cube(0.6, 0.6, 0.6)), Material(WOOD), Transform(pos=(2.5, 0.3, 2.5)))
print("crate in:", alive(), "entities,", len(engine.commands), "commands")
crate.destroy()
print("after destroy():", alive(), "entities,", len(engine.commands), "command")
engine.render_frame(supersample=1)
print("after the flush:", alive(), "entities,", len(engine.commands), "commands")

crate = engine.spawn(Mesh(cube(0.6, 0.6, 0.6)), Material(WOOD), Transform(pos=(-2.5, 0.3, 2.5)))
engine.commands.append(Command(CommandType.DESTROY, {"indices": np.array([crate.index])}))
print("by hand:", len(engine.commands), "command,", alive(), "entities")
engine.render_frame(supersample=1)
print("after the flush:", len(engine.commands), "commands,", alive(), "entities")


def set_level(value):
    jet.transform.scale = (1.0, value, 1.0)
    jet.transform.pos = (0.0, 1.0 + 0.5 * value, 0.0)


level = 0.0
busy = False


@engine.on("key_down")
def space_starts(ev: KeyEvent):
    if ev.key == "Space":
        engine.emit("fountain")


@engine.on("fountain")
async def raise_jet(_payload):
    global level, busy
    if busy:
        return
    busy = True
    while level < 1.2:
        level = min(level + 0.3, 1.2)
        set_level(level)
        await engine.tick()
    await engine.delay(0.2)
    busy = False
    print(f"fountain on at {engine.elapsed:.2f} s, level {level:.2f}")


print("before the press: level", round(level, 2), "busy", busy)
engine.emit("key_down", KeyEvent("Space", (), True))
engine.render_frame(supersample=1)
for _ in range(12):
    engine.render_frame(supersample=1)
engine.emit("key_up", KeyEvent("Space", (), False))
print("after: level", round(level, 2), "busy", busy, "elapsed", round(engine.elapsed, 2))
Image.fromarray(engine.render_frame(supersample=1))
