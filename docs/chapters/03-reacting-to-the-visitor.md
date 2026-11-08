# Reacting to the visitor

The plaza stands, but the person at the screen is still a spectator. They
should walk the paving with the keys, click a bench and see which one they chose,
and press space to start the fountain. The plaza also has work that happens once,
when it opens, and work that happens on every frame. None of that is geometry, so
none of it is answered by what the earlier chapters taught.

## Something to react to

The scene is the one from the first chapter, with a few additions: a floor, the
basin, three wooden benches, and a fountain jet. The jet is a cube scaled to a
hundredth of its height and set just under the rim, so it is out of sight inside
the basin until someone turns it on. `engine.spawn` returns an `EntityHandle`; we
keep the ones for the benches and the jet, because they are how a later handler
will move the water or repaint a bench.

```python {run export=examples/visitor.py}
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
```

## One place where things happen

`render_frame` draws one frame, and drawing a frame is not only drawing: the
engine updates the clock, delivers what was queued for that frame, runs the
systems and applies pending changes before it renders. The plaza needs a way to
say what to do at those moments, and that is the event bus.

`engine.on` registers a function for an event name and `engine.emit` posts one.
Under the name, `engine.on` wraps your function in a `Handler` — the callable,
whether it is a coroutine, and the components it asked for — and the engine's
`EventBus` keeps one list of handlers per event name and a queue of pending
events. At the start of a frame the bus drains the queue and calls the handlers
in the order they were registered. `engine.off` removes one again.

```python {run continue export=examples/visitor.py}
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
```

Three things are worth reading there. `startup` fired first, at `0.0`, when the
first `render_frame` created the offscreen canvas, not from the `Engine(...)`
call above it: a handler registered after the first frame has been drawn never
sees it. `frame` arrived with the frame number and the clock, once per frame. And
`knock` was delivered after `frame` in the first frame, but the `answered` it emitted only
arrived in the second: `emit` never calls anyone, it appends to the bus's pending
list and the list is drained at the start of the next frame. The engine puts its
own `frame` event at the head of that queue, so a frame handler always runs
before the events queued for the same frame. `engine.off("frame", every)` removed
the handler after the second frame, and the third adds nothing to the journal.
`shutdown` is the other engine phase: `quit` fires it.

## What a handler may read

A handler that needs the world asks for components with `Query`, exactly as a
system does, as its second parameter. What arrives is not the writable view a
system gets but a `ReadOnlyView`: reads pass through, and a write raises. The
reason is timing. Handlers run in the middle of a frame, between the point where
the previous frame's systems left the arrays and the point where this frame's
systems and the renderer will read them, and a write there would leave half the
frame looking at old data and half at new.

```python {run continue export=examples/visitor.py}


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
```

The view held all nine entities, because every one of them has a `Transform`, and
the two rows of the first bench came back as they were spawned. The assignment
was refused with a message that names the three doors out:
`engine.commands.append(...)`, `engine.spawn(...)` and `engine.destroy(...)`.

## The visitor's hands

With a window, this chapter's next question is already answered: an input bridge
inside the engine turns each keyboard, pointer, wheel and resize callback into
one of four frozen dataclasses, emits it on the bus, and records the same thing
in `engine.input`. A `KeyEvent` carries the key, the modifiers held and whether
it went down; a `PointerEvent` the position in logical pixels from the top-left
corner, the movement since the previous event, the button and the buttons held,
the modifiers, the phase and the pointer id (the left button is 1); a
`WheelEvent` the scroll amounts (about 100 per notch) and the pointer position; a
`ResizeEvent` the new width, height and pixel ratio.

This guide has no window, so we emit those dataclasses ourselves and render a
frame to have them delivered. The handlers cannot tell the difference. What is
missing is the bridge's other half: `engine.input` is an `InputState` that only
the bridge fills, so an emitted event reaches the handlers and leaves the polling
state alone. Both halves are in the last block of this section.

Walking is the one job that is genuinely per-frame, so the key handlers only
record which keys are held and a `frame` handler moves the camera while one of
them is in the set. The camera keeps its height and looks where it looked; the
step is four metres per second of `dt`. `render_frame` gives every call the same
fixed `dt` of `1/30 s`, so the walk is the same in every run.

```python {run continue export=examples/visitor.py}


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
```

The walk took four steps of `4 / 30 = 0.133 m`, because the first frame ran its
`frame` handler before delivering the key, and the press only counts from the
next one. A bench was picked by projecting the three centres and keeping the
nearest within forty pixels — enough for three benches in plain sight, though a
bench behind the fountain still owns pixels on screen and only a depth test
could refuse it. One notch of the wheel moved the camera from `11.93 m` to
`10.74 m` from its target. The resize handler printed, but `engine.w` and
`engine.h` stayed at `480 x 270`: the bridge is what resizes the engine, and an
emitted event is not the bridge. And the polling line is the honest one: an
emitted `key_down` reaches the handler and leaves `engine.input` empty, and
`capture_pointer` returns `False` because there is no window to lock the cursor
into. With a real window both halves happen together, and `held` could be
replaced by `engine.input.is_pressed("w")` in the same frame handler, with
`just_pressed` for the edge of a single press. The frame above is the plaza after
the click and the scroll: the chosen bench is gold and the camera is closer.

## Changes that take effect between frames

The click could repaint a bench because `set_material` on a handle writes the
material column itself, outside the array a handler reads. It could not remove
one, so destruction needs a queue that the frame loop drains at a safe moment.

`engine.commands` is a `CommandBuffer`: a list of `Command` objects, each with a
`type` from `CommandType` and a `data` dictionary. The frame loop executes it
after the systems and before the draw, then clears it. `engine.destroy` and
`EntityHandle.destroy` append a `DESTROY` command, which is why the entity is
still alive right after the call and gone at the flush. `engine.spawn` is the
exception: it creates the row at once and returns the handle, even though
`CommandType` has a `SPAWN` type for commands written by hand. A system's `+=`
on a component is queued as `UPDATE_COMPONENT` and applied the same way. `NOP`
does nothing, for a command that has to occupy a slot.

A crate left on the paving is the smallest thing that can disappear, so we clear
two of them, one through the handle and one through a hand-written command:

```python {run continue export=examples/visitor.py}


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
```

The spawn is visible at once: ten entities on the line after it. The destroy is
not: the crate is still there, and the buffer holds one command. A frame later
the buffer is empty and the crate is gone. The second crate is removed by the
command `destroy` appends — the same `type` and the same `data` dictionary — and
it goes in the same buffer.

## Taking the time it takes

The fountain cannot start in one frame. The jet has to rise over a handful of
frames and the plaza should not accept another press while it moves, which in a
synchronous handler would mean a state machine with a counter and a phase.

An async handler does not need one. `engine.on` noticed that the function was a
coroutine and marked its `Handler` async, so the bus schedules it as a task
instead of calling it, and `await engine.tick()` gives a future that resolves at
the next frame boundary. `engine.delay(seconds)` waits that much of
`engine.elapsed`, and `engine.elapsed_at(target)` waits until the clock reaches a
value. All three are futures in a `FrameWaiters`, which the engine resolves at
the start of every frame; a loop asks again on each pass, since one
`await engine.tick()` means one frame and no more.

```python {run continue export=examples/visitor.py}


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
```

The press queued `fountain`, the bus started the coroutine on the following
frame, and it raised the jet by `0.3` on each of four frames, waiting for each
one. Then it waited `0.2 s` of clock, released the `busy` flag and reported
itself on at `0.97 s`. A second press during the rise would have reached the same
handler, seen `busy` and returned. The last frame is the plaza with the water up
and the clicked bench still gold.

## The finished program

Those blocks are the whole program, tangled into `docs/examples/visitor.py`. Run
it as a script and it drives the same events a window would — the presses, the
click, the scroll, the resize — and prints what each one did. To hand the plaza
to a person instead, delete the synthetic `engine.emit` calls and end with
`engine.run()`, which opens a window, delivers the events from the canvas bridge
and keeps both the handlers and `engine.input` up to date until the window is
closed or `quit` is called.

```python
engine.run()
```

The plaza now answers one visitor. The next chapter gives it light: materials
with more than a colour, shadows, and a sun that can move.
