# Working with an agent

The plaza has reached the size where one look is never enough: every bench
position, every colour and every camera height is a guess you check by running
the program, and checking the next guess means stopping it and walking the view
back to where it was. Much of the editing is done by a coding agent, which
cannot see the window at all. We want the plaza to stay up while the files
change, a way for the agent to look at the frame and to make the app do things
from a shell, and, at the end, the same plaza in a browser tab.

## Save, and keep your place

`manifoldx run` is `python` for an app that draws. It runs the program until
`engine.run()`, takes the engine, and every quarter of a second compares the
modification times of the app's own source files. When one has changed it
forgets the app's modules, runs the program again from the top, and hands the
new engine the same window and the same GPU device. The old engine is detached
first: its `shutdown` handlers fire, its async tasks and frame waiters stop, its
input unhooks from the window and its workers shut down. The scene is rebuilt
from the new code, every time.

```bash
uv run manifoldx run agent_plaza.py                # a script
uv run manifoldx run -m plaza.app                  # a module, like python -m
uv run manifoldx run agent_plaza.py -- --level 3   # after --, the app's own argv
```

Every app in this loop ends with the line below. Under `manifoldx run` the
reloader takes the engine there instead of opening a window, which is what lets
it replace the app on the next save.

```python
engine.run()
```

Because the program starts over, module-level code runs on every save: a file
written there, a thread started there, a banner printed there happens once per
edit. Move that work into a `startup` handler, and undo it in `shutdown`. And
the only thing that crosses a reload unchanged is `engine.state`.

`State` is a dict whose keys also read and write as attributes, so `cam` and
`["cam"]` are the same entry.

```python {run}
from manifoldx.hot import State

state = State()
state["cam"] = {"yaw": 40.0, "dist": 18.0}
state.benches = 2
state.setdefault("visits", 0)
print(state.cam, state.benches, "visits" in state)
```

`engine.state` is one of these. Under `manifoldx run` every engine the program
makes shares one, so a `setdefault` at startup reads what an earlier run left;
under plain `python` each engine gets a fresh one, and the same code works both
ways. Store what the scene can be rebuilt from: numbers, strings, tuples,
lists, dicts, numpy arrays. An instance of one of the app's own classes does
not survive, because after a save that class is a different class, and a GPU
object — a mesh, a material, an entity handle — belongs to the old engine's
buffers. Keep the numbers; build the objects again.

To see the machinery without a window, drive the reloader yourself, as the
library's tests do. A small version of the plaza goes into a temporary file; it
counts its own builds in the state and takes the number of benches from a
constant the agent is about to edit.

```python {run}
import os
import tempfile
import time
from pathlib import Path

from manifoldx.backends import get_offscreen_canvas
from manifoldx.hot import HotReloader

APP = """\
import math

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane

engine = mx.Engine("plaza", width=320, height=180)
engine.state.setdefault("visits", 0)
engine.state.setdefault("builds", 0)
engine.state.builds += 1
BENCHES = %d

@engine.on("startup")
def build(payload):
    engine.spawn(Mesh(plane(40, 40)), Material(StandardMaterial(color="#c8b89a", roughness=0.9)),
                 Transform(rot=Transform.rotation(x=-math.pi / 2)))
    engine.spawn(Mesh(cube(4, 1, 4)), Material(StandardMaterial(color="#8a9bb0", roughness=0.6)),
                 Transform(pos=(0, 0.5, 0)))
    for i in range(BENCHES):
        engine.spawn(Mesh(cube(2.4, 0.45, 0.7)), Material(StandardMaterial(color="#7a5a3c")),
                     Transform(pos=(-4 + 3 * i, 0.22, 3)))
    engine.camera.set_pose(position=(12, 6, 14), target=(0, 1, 0))
    engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.4, -1, -0.3)))
    engine.log(f"built {BENCHES} benches, build {engine.state.builds}")

@engine.on("visit")
def visit(payload):
    '''A person crosses the plaza: {"name": "Ana"}.'''
    engine.state.visits += 1
    engine.log(f"{payload['name']} crossed: {engine.state.visits} visits")

engine.run()
"""

root = Path(tempfile.mkdtemp(prefix="plaza-dev-"))
app = root / "app.py"
saves = [0]


def save(benches):
    """An editor's save: the new text, and a modification time the watcher sees."""
    saves[0] += 1
    stamp = time.time() + 2 * saves[0]
    app.write_text(APP % benches)
    os.utime(app, (stamp, stamp))


save(2)
hot = HotReloader(
    script=str(app),
    canvas_factory=lambda engine: get_offscreen_canvas(engine.w, engine.h),
    poll=0,  # check on every step instead of every 0.25 s
    channel=False,  # no dev server yet
)
hot.start()
for _ in range(2):
    hot.step()
print(dict(hot.state))
```

The first `hot.step()` has already run the file to `engine.run()`, which the
reloader replaced with a capture, so nothing blocks. `hot.state` is the state
its engine was given: one build, no visits. Now the agent edits `BENCHES` and
saves.

```python {run continue}
first = hot.engine
save(4)  # two more benches
hot.step()  # one frame: notice the save, reload, draw
print(hot.engine is not first, dict(hot.state))
```

The engine is a new object, built by the four-bench loop, and the state is the
same object as before: `builds` says the file ran twice, and `visits` is still
zero because nobody has crossed. The window and the GPU device are the old
ones. A save that does not load — a syntax error, a bad import, an exception at
module level — never gets as far as detaching: the running version stays on
screen, the traceback is printed, and the next save tries again. An exception
in a frame pauses the app until the next edit, even an unrelated one.

## The dev channel

`manifoldx run` also serves a page and an HTTP API on 127.0.0.1 (port 8770, or
the next free one) while the app runs, and `manifoldx` has a command for each
thing an agent needs to ask it.

```bash
uv run manifoldx run --debug agent_plaza.py
# manifoldx run: dev page and API at http://127.0.0.1:8770
uv run manifoldx look                          # the frame on screen, saved to .manifoldx/look.png
uv run manifoldx log -n 20                     # what the app said, and the keys pressed
uv run manifoldx state cam                     # part of engine.state, as JSON
uv run manifoldx set '{"cam": {"yaw": 120}}'   # a JSON merge patch: objects merge, null deletes
uv run manifoldx events                        # what the app can be asked to do
uv run manifoldx emit visit '{"name": "Ana"}'
uv run manifoldx log --match 'Ana crossed' --wait 60
```

`events` lists each event next to the first line of its handler's docstring;
the page has a button for each one. `emit` answers before the handlers run, so
an agent reads the state after them or waits for their log line. Events are
worth having even when the state could be patched: an event goes through the
app's own code and keeps the world consistent, while `set` only changes the
values the app reads. The log also gets the runner's own lines — every reload,
a failed reload with its error, and under `--debug` the keys, slow frames and a
metrics line every ten seconds.

The same channel is a Python object, and having it in hand shows what the
commands are talking to. `Channel` is what `manifoldx run` builds: it owns the
log, the server and a queue of jobs. A request never touches the engine
directly; `on_engine` puts the work in the queue and waits, and `run_jobs`,
called by the reload loop on every frame, paused or not, runs it on the
engine's thread between frames. That is why `manifoldx state` can read while a
system writes without a race.

```python {run continue}
import threading

from manifoldx.devchannel import Channel

channel = Channel(root, hot, port=0)
answer = {}


def ask():
    answer["state"] = channel.on_engine(lambda engine: dict(engine.state))


worker = threading.Thread(target=ask)
worker.start()
while worker.is_alive():  # what the reload loop does on every frame
    channel.run_jobs()
print(channel.url, answer["state"])
```

`channel.lines` is a `Lines`: the last five thousand lines the app logged,
numbered from 1, with one query behind every form of `manifoldx log`.
`read(after=..., match=..., wait=..., last=...)` returns the lines and the
sequence number to pass as `after` next time; `wait` is how `--match --wait`
blocks, and a `match` with no line left after the wait is what makes the
command exit 1.

```python {run continue}
hot.engine.log("entered the plaza")
print(channel.lines.read(last=3)["lines"])
```

`engine.log` writes here and to `.manifoldx/log.txt` under `manifoldx run`;
outside it the line goes nowhere, so log what a person or an agent would need
to know, once per change, never every frame. An event fired from the shell
arrives as the same call the app makes itself.

```python {run continue}
hot.engine.emit("visit", {"name": "Ana"})
hot.step()  # the next frame: the handler runs, then the log line appears
print(hot.state.visits, [line["text"] for line in channel.lines.read(last=1)["lines"]])
```

`manifoldx look` comes back as a PNG of the next frame presented. Behind it a
`Readback` copies the frame's texture on one frame, maps that copy on the next
and encodes the PNG on a thread, which is why a look costs a frame or two, and
why it waits five seconds and gives up when the app is paused and no frame
comes.

```python {run continue}
import io

from PIL import Image

shot = {}


def grab():
    shot["png"] = channel.look()


worker = threading.Thread(target=grab)
worker.start()
for _ in range(4):
    hot.step()  # draw, and copy the frame for the look
    channel.run_jobs()  # hand the request to the engine's thread
worker.join(timeout=5)
png = shot["png"]
channel.close()
Image.open(io.BytesIO(png))
```

The four benches are the two the first build placed and the two the save
added. Under `--debug` the channel also keeps a `Meter`, which accounts for
each second's frames, their mean and worst time, the resident memory and the
garbage collections. `frame(now, split)` is
fed after every frame with `engine.frame_split`, whose update, acquire and
render parts leave `idle` as the rest of the frame, and `history` keeps the last
two minutes. This is what `manifoldx metrics` prints and what its slow-frame and
slow-collection log lines come from.

```python {run}
from manifoldx.devchannel import Meter

meter = Meter()
for i in range(61):  # a minute of made-up frame times, to show the accounting
    meter.frame(i / 60.0, {"update": 0.004, "render": 0.003, "acquire": 0.009})
row = meter.history[-1]
print(row["fps"], row["frame_ms"], row["split_ms"])
meter.close()
```

`Meter` here is the dev channel's, not a `manifoldx.viz` chart, and `Channel` is
the dev channel's too, not the viz one the guide used for the data chart.

## The plaza in a tab

`manifoldx export --web` runs the app the way the export does — until
`engine.run()`, with `startup` fired on a headless device — and writes a folder
that a browser runs instead of a window: a page, the app's modules transpiled
to JavaScript, the scene as `scene.glb`, the state as `state.json`, and
`replay.mjs` for Node.

```bash
uv run manifoldx export --web agent_plaza.py out/
cd out && python -m http.server 8000          # then open http://localhost:8000/
```

Setup stays in Python: module-level code and the `startup` handler build the
scene, and the export snapshots it. What the page runs is the frame code — the
systems and the event handlers — and everything they read, which is why
`engine.state` must already hold plain data. It is also why a handler's payload
needs a type the transpiler can see, `payload: dict[str, float]` and
`x: float = payload["x"]`, rather than `float(payload["x"])`: the browser has
no live object to learn the type from. When the export meets something it
cannot carry or type it exits 1, writes nothing, and names the file, the line
and the function that reached it.

```bash
uv run manifoldx run --web agent_plaza.py
# manifoldx run --web: open http://127.0.0.1:8770/ (dev page at /dev)
```

`manifoldx run --web` is the same loop as before, with the app in a tab instead
of a window: it exports, serves the page, and on every save asks the tab for its
state, freezes it, reruns the setup in Python, merges the state back and pushes
the new version into the open tab without a page reload. `engine.state` comes
back as plain data, so numbers, strings, lists, tuples and dicts survive; a key
holding an app object, a handle or a function keeps Python's value and the log
says so once per key. The pointer lock and the keys held across the save
survive too, the frame log says `reloaded v4 in ...` with its phases, and the
dev page moves to `/dev` while the same `look`, `state`, `set`, `emit` and
`log` commands work against the tab. `--debug` is refused there, because under
`--web` the tab runs the frames, not the Python process.

## The finished program

The app from here on is the one the agent drives. Its state is the benches, the
visits and the camera angle; the scene is built from that state at startup and
rebuilt on every save; two events give the agent a way in; and the camera
reads the state every frame, so a `set` patch moves the view without an event.

```python {run export=examples/agent_plaza.py}
import math

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane
from PIL import Image

engine = mx.Engine("plaza", width=480, height=270)

# The only things that survive an edit, and all of it plain data.
engine.state.setdefault("benches", [[-4.0, -3.0, 0.0], [4.0, -5.0, 90.0]])
engine.state.setdefault("visits", 0)
engine.state.setdefault("cam", {"yaw": 40.0, "dist": 18.0, "height": 7.0})

FLOOR = Transform(rot=Transform.rotation(x=-math.pi / 2))
PAVING = StandardMaterial(color="#c8b89a", roughness=0.9)
STONE = StandardMaterial(color="#8a9bb0", roughness=0.6)
WOOD = StandardMaterial(color="#7a5a3c", roughness=0.7)
BENCH = cube(2.4, 0.45, 0.7)


def bench(x, z, yaw):
    """One bench at (x, z) metres, turned yaw degrees about y."""
    turn = Transform.rotation(y=math.radians(yaw))
    return engine.spawn(Mesh(BENCH), Material(WOOD), Transform(pos=(x, 0.22, z), rot=turn))

```

`setdefault` is the pattern: a key that is already there keeps the value the
last save left, and a key that is missing gets its first value, so the same
code works on a cold start and after an edit. The geometries and materials are
built once at module level and used again by the helpers; the state holds the
numbers they are placed with.

```python {run continue export=examples/agent_plaza.py}

@engine.on("startup")
def build(payload):
    """Stand up the plaza from the state."""
    engine.spawn(Mesh(plane(40, 40)), Material(PAVING), FLOOR)
    engine.spawn(Mesh(cube(4, 1, 4)), Material(STONE), Transform(pos=(0, 0.5, 0)))
    for x, z, yaw in engine.state.benches:
        bench(x, z, yaw)
    engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.4, -1, -0.3)))
    engine.log(f"plaza built: {len(engine.state.benches)} benches, {engine.state.visits} visits")


@engine.on("add_bench")
def add_bench(payload: dict[str, float]):
    """Place a bench: {"x": metres, "z": metres, "yaw": degrees}."""
    x: float = payload["x"]
    z: float = payload["z"]
    yaw: float = payload.get("yaw", 0.0)
    placed: list[list[float]] = engine.state.benches
    placed.append([x, z, yaw])
    bench(x, z, yaw)
    engine.log(f"bench {len(placed)} at {x:.1f}, {z:.1f}")


@engine.on("visit")
def visit(payload: dict[str, str]):
    """A person crosses the plaza: {"name": "Ana"}."""
    name: str = payload["name"]
    n: int = engine.state.visits
    engine.state.visits = n + 1
    engine.log(f"{name} crossed: {n + 1} visits")

```

The event is the agent's way in, and it must also be the app's own path: the
handler records the bench in the state *and* spawns it, so a reload after the
event finds the bench in the list and rebuilds it. The docstring's first line
is what `manifoldx events` and the page show, so it says what the payload
holds. The annotations are for the browser: `dict[str, float]` and the typed
locals are the annotations the export asked for, and they cost nothing on the
desktop.

```python {run continue export=examples/agent_plaza.py}

@engine.on("frame")
def look_around(payload):
    """Put the camera where the state says, every frame, so a patch moves the view."""
    yaw: float = engine.state.cam["yaw"]
    dist: float = engine.state.cam["dist"]
    height: float = engine.state.cam["height"]
    a = math.radians(yaw)
    engine.camera.position = np.array([math.sin(a) * dist, height, math.cos(a) * dist], np.float32)
    engine.camera.target = np.array([0.0, 1.0, 0.0], np.float32)


frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
```

The camera is placed from the state on every frame rather than at startup, so
`manifoldx set '{"cam": {"yaw": 120}}'` moves the view on the next frame; a
camera copied from the state at startup would keep the old angle until the next
save. `camera.position` and `camera.target` are written as float32 vectors
because that is what the page applies; on the desktop `set_pose` would do the
same work.

To develop this program, the last two lines become `engine.run()`, the line
from the start of the chapter, and the file is then the app under
`manifoldx run`, `manifoldx export --web` and `manifoldx run --web`. The guide
keeps the still instead, because it has no window to draw into.
