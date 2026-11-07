---
when: running an app with `manifoldx run`, keeping state across reloads with `engine.state`, or making an app reload fast
triggers: ["hot reload", "manifoldx run", "engine.state", "reload", "live edit", "keep state", "watch files"]
---

# Hot reload with `manifoldx run` and `engine.state`

## When to use this

Reach for this when you develop an app or game on manifoldx and want edits to show up in the open window without losing your place, or when you write app code that should behave well under reload. The code is `src/manifoldx/hot.py`; the tests in `tests/test_hot_reload.py` show every behaviour below.

## Running

```bash
manifoldx run app.py                      # a script
manifoldx run -m mygame.main              # a module, like python -m (run from the folder that holds the package)
manifoldx run app.py -- --level 3         # everything after -- goes to the app's sys.argv
```

Run it with `uv run` inside a project, like every other Python command here.

## What a reload does

Every 0.25 s the reloader compares the modification times of the app's source files. When one changed:

1. It drops the app's modules from `sys.modules`. The app's modules are those loaded from the script's folder (the working directory for `-m`), excluding anything under `site-packages` and manifoldx itself.
2. It reruns the app from the top, until it calls `engine.run()` (or `engine.cli()`, which calls `run()`). That call does not block under `manifoldx run`: the reloader takes the engine.
3. If that worked, it detaches the old engine: fires its `shutdown` handlers, cancels its async tasks and frame waiters, unhooks its input from the window and shuts down its `engine.background` / `engine.submit_process` workers. The new engine takes over the same window and the same GPU device, then fires `startup` on its first frame.

So the whole scene is rebuilt from the new code each time. The only thing that carries over is `engine.state`.

## Errors

- An edit that fails to load (syntax error, import error, exception at module level) leaves the running version on screen and prints the traceback. Fix and save again.
- An exception in a frame (a handler, a system, startup) prints the traceback and pauses the app: no more frames run until the next save. Saving reloads it even if the edit is unrelated.

## `engine.state`

A dict whose keys also read and write as attributes. Under `manifoldx run` every engine the app creates shares one; under plain `python app.py` each engine gets a fresh one, so the same code runs both ways.

The pattern: give each value a default with `setdefault` at startup, read it when you build the scene, and keep it current while running.

```python
engine.state.setdefault("cam", {"yaw": 0.0, "pitch": 0.3, "dist": 12.0})
engine.state.setdefault("level", 1)

@engine.on("startup")
def build(_):
    load_level(engine.state.level)          # rebuild where the player was
    place_camera(**engine.state.cam)

@engine.system
def orbit(query: Query[Transform], dt: float):
    engine.state.cam["yaw"] += engine.input.mouse_delta[0] * 0.01   # the state is the source of truth
    place_camera(**engine.state.cam)
```

Rules:

- **Store plain data**: numbers, strings, tuples, lists, dicts, numpy arrays. Instances of classes defined in the app's own modules go stale: after a reload the module is new, so `isinstance` against the new class fails and methods are the old code.
- **Never store manifoldx GPU objects** (geometries, materials, textures, entity handles). They belong to the old engine's registries and buffers. Store the numbers you build them from.
- **Keep the value in state, not a copy.** If a system keeps the camera angle in a local variable and writes it to state only sometimes, a reload loses whatever was not written yet.
- **Do not count on `engine.camera` surviving.** The camera is per engine; save what you need of it in state.

## Gotchas

- **Module-level code runs again on every save.** Anything with side effects there (writing files, opening sockets, starting threads, printing a banner) happens once per reload. Move it into `startup`, and undo it in a `shutdown` handler.
- **Threads and processes you start yourself are not stopped.** Only the engine's own workers are. Stop yours in `@engine.on("shutdown")`, which fires on every reload as well as on exit.
- **Engine constructor arguments do not apply on reload.** The window stays, so a new `title`, `width`, `height`, `fullscreen` or `monitor` needs a restart.
- **Only the app's own files reload.** Editing manifoldx or any other installed package (even an editable install from another repo) needs a restart. With `-m`, run from the project root so the package's files count as the app's.
- **A file the app never imported is not watched**, except the one it starts from (the script, or the `-m` module). A module that failed to import on its first load is watched only once something imported it.
- **`-- --render` does not work**: `engine.cli()` then renders a video instead of calling `run()`, and `manifoldx run` exits saying the app never called `engine.run()`.
- **Two saves of one module within the same second that leave it the same size** can load the old cached bytecode (Python checks `.pyc` by whole-second mtime and size). Save again.

## The dev server

`manifoldx run` also serves a dev page and an HTTP API on 127.0.0.1 (the frame on screen, the log, `engine.state`, the app's events), and the commands `manifoldx look`, `state`, `set`, `emit`, `events` and `log` that use them. `know-how/agent-loop.md` says how to work with them and how to write an app an agent can drive.

## In the browser: `run --web`

`manifoldx run --web app.py` (or `-m mygame.main`) runs the app in a browser tab instead of a window: each save reruns setup in Python, exports again and swaps the new version into the open tab without a page reload. `engine.state` survives as here, through the tab: what the tab holds comes back as plain data, and keys holding app objects, handles or functions keep Python's value. The tab freezes while the reload builds, as the window blocks here. `know-how/web-export.md` ("While you edit: `run --web`") says what a save does, what survives, the status line's texts and what `.manifoldx/web/` holds; `know-how/agent-loop.md` the commands against the tab and the reload's timing line.

## Performance

Watching is cheap: one `stat` per source file every 0.25 s (about 0.4 ms for 100 files). A reload costs what the app's startup costs, minus opening the window and the GPU device; a one-cube app reloads in 25-30 ms, against about 3 s for a cold start. To keep reloads fast:

- **Cache expensive CPU work in `engine.state`**: generated terrain heightmaps, decoded images, parsed level files, vertex arrays. Key the cache by the inputs, so an edit that changes them rebuilds:
  ```python
  key = ("terrain", SEED, SIZE)
  if key not in engine.state:
      engine.state[key] = generate_heightmap(SEED, SIZE)   # numpy array
  heights = engine.state[key]
  ```
- **Load heavy models once.** Objects from installed libraries (an onnxruntime session, a TTS model) can live in state, because their classes never reload: `engine.state.setdefault("tts", Kokoro(model_path, voices_path))`. Never load them at module level.
- **Shader pipelines are recompiled on every reload**, because each engine starts with an empty pipeline cache. An app with many materials pays for each one per save. Reuse material types rather than creating many variants.
- **Expensive async startup** (`@engine.on("startup")` coroutines with `engine.loading(...)`) runs again per reload; check state first and skip what is already there.
- GPU memory of the old engine is freed when nothing references it any more. That has not been measured across many reloads; if memory grows, look for app code that keeps a reference to an old engine (a global outside the app's folder, a thread).

## Testing reload behaviour

Tests drive the reloader without a window: pass an offscreen canvas and call `step()` yourself.

```python
from manifoldx.backends import get_offscreen_canvas
from manifoldx.hot import HotReloader

hot = HotReloader(script="app.py", canvas_factory=lambda e: get_offscreen_canvas(e.w, e.h), poll=0)
hot.start()
hot.step()          # one frame; with poll=0 it checks for changes on every step
# edit app.py, then:
hot.step()          # reloads, then draws a frame of the new code
assert hot.state.level == 1
```

To see a real window on a machine without a display (the VPS), start a virtual one and grab frames from it:

```bash
Xvfb :97 -screen 0 800x600x24 &
DISPLAY=:97 uv run manifoldx run app.py &
ffmpeg -f x11grab -video_size 800x600 -i :97 -frames:v 1 shot.png
```
