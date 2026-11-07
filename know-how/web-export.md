---
when: exporting an app with `manifoldx export --web`, fixing an error it reports, annotating app code so it transpiles, serving or debugging the browser page it writes, re-measuring its calibration, or checking with a replay that the browser runs an app as Python does
triggers: ["export --web", "web export", "transpile", "transpiler", "browser", "manifoldx.web", "replay", "replay_node", "outside the subset", "cannot cross to the browser", "annotate", "index.html", "three.js", "pointer lock", "CALIBRATION", "playwright"]
---

# Exporting an app to the web with `manifoldx export --web`

## When to use this

Reach for this when you export an app for the browser, when the export refuses an app and you have to change its code, when the page draws or behaves differently from the desktop, or when you want proof that the exported app behaves as the Python one does. The code is `src/manifoldx/web/` (the page's modules in `js/`, three.js in `vendor/three/`); the tests in `tests/web/` show every behaviour below, and `examples/web_walk.py` is an app that exports, replays and runs in the page.

The export writes a folder with a page: the transpiled app runs its systems every animation frame on the browser's input, three.js draws the exported scene, and the GUI is HTML. The same folder replays in Node. `run --web` is not there yet. Spec: `.knowledge/analysis/2026-11-06-web-target-design.html`. Plans: `.knowledge/plans/2026-11-06-web-target-plan-1-transpiler-and-replay.html` (transpiler, snapshot, replay) and `.knowledge/plans/2026-11-06-web-target-plan-2-browser-page.html` (the page); each plan's "where this plan departs" section explains the choices that differ from the spec.

## Running it

```bash
uv run manifoldx export --web app.py out/                  # a script
uv run manifoldx export --web -m mygame.main out/          # a module, from the folder that holds the package
uv run manifoldx export --web app.py out/ -- --level 3     # everything after -- goes to the app's sys.argv
```

The app runs as `python app.py` would, until it calls `engine.run()` (or `engine.cli()`), which returns at once. If the engine has startup handlers, the export opens a headless wgpu device and fires them (an `async def` startup handler is awaited), so `load_texture` and other device work runs. That needs a wgpu adapter (lavapipe, `mesa-vulkan-drivers`, is enough on a machine with no GPU), not the `[offline]` extra: only video renders need imageio-ffmpeg. Everything up to here is setup and stays in Python. Then the export finds the roots, types and reaches the code, snapshots the state and transpiles.

Exit codes: 0 written; 1 the app's code, state or GUI cannot cross (the error names the file and line, the path to the value, or the widget, and **nothing is written**); 2 the app never called `engine.run()`. `--strict` belongs to the `.glb` export and is refused with `--web`. The export does not need Node; the replay and the tests do.

## What the folder holds

```text
out/
  index.html                 the page; carries the title and scene.glb's size and gzipped size for the progress line
  browser.js                 the page's bootstrap: loads the folder, runs the frames, renders; window.__mx for tests
  input.js                   DOM events to one replay frame per animation frame, and pointer lock
  scene.js                   the glTF in three.js: extras, CALIBRATION, the basic material, entity and material handles
  gui.js                     engine.gui as HTML
  sourcemap.js               a JS error's stack as Python file:line
  progress.js                the progress line and the frame clock
  vendor/three/              three.js 0.186.1, unminified, as its SOURCE file says (2.3 MB, about 450 KB gzipped)
  fonts/DejaVuSansMono.ttf   the GUI's font, as the desktop draws it
  app/<module>.js, .js.map   one ES module per app module, with a source map to the .py; the script (or -m module) is app/__main__.js
  pyrt.js                    the Python runtime the generated code imports as $rt
  manifoldx.js               the engine core the page and Node share: state loader, input, camera, set_material, step, trace
  replay.mjs                 loads the folder in Node
  state.json                 the snapshot: reached objects, engine.state, the engine's size and camera, the GUI
  scene.glb                  the scene, as export_gltf writes it, plus scene.export-report.json
  report.json                roots, transpiled functions, setup functions, reached classes and globals, state left out
```

The page's own modules and the runtime are about 23 KB gzipped together; three.js is the bulk of the download, then `scene.glb`. Nothing is minified or precompressed: a server that gzips gets the sizes above.

`node out/replay.mjs` prints the roots' labels, `engine.state` and what each ValueDisplay shows, as JSON: the quickest check that the folder loads.

The app's modules are the script (or `-m` module) and every module loaded from its folder (the working directory for `-m`), outside `site-packages` and manifoldx: the files `manifoldx run` watches.

## The page

### Serving it

```bash
cd out && python -m http.server 8000          # then open http://localhost:8000/
```

Serve the folder over HTTP: opened as a `file://` URL the page cannot load its ES modules, and its loading text says to serve the folder over HTTP (python -m http.server). Any static server works; one that gzips `scene.glb` keeps the progress line right, since `index.html` carries both sizes. The page needs WebGL2 and says so when the browser has none.

### What it does each frame

`browser.js` loads `state.json`, imports the app's modules, streams `scene.glb` behind a progress bar (`#mx-progress`, hidden once loaded) and builds the scene and the GUI. Then, every animation frame, it takes one frame from the DOM's input, sets `engine.gui.pointer_over_gui`, runs the systems with the frame's real dt, applies the widget fields they wrote, updates the ValueDisplays, follows the canvas's size, syncs the three.js camera to `engine.camera` and renders. The canvas fills the window; `engine.w` and `engine.h` are its CSS size, and the drawing buffer is that times the device pixel ratio, capped at 2.

### Input

`input.js` turns DOM events into the replay's frame form, so the page and a replay feed the engine the same way:

- Keys are named as the desktop names them (a letter lowercase, the space bar `Space`, `Esc` as `Escape`), tracked by their code, so Shift released before a key still releases what it pressed. Tab and Space are kept from the page; browser shortcuts are left alone. Losing focus releases everything.
- Buttons map as on the desktop (left 1, right 2, middle 3); `pos` is in CSS pixels from the canvas's top-left; pixel wheel deltas are divided by the device pixel ratio, lines and pages scaled as rendercanvas does. The context menu is suppressed.
- `engine.input.capture_pointer(True)` asks for pointer lock. A browser grants it only during a user gesture: with an active one it is requested at once, otherwise at the next click on the canvas. It returns False where the canvas has no pointer lock or the last pointer was a touch, as the desktop does with no window.
- When the lock ends without the app asking (the user pressed Escape, which the browser keeps from the page), the next frame's `just` holds `Escape` and `pointer_captured` turns False, as the desktop's Escape would. An app that releases on Escape needs no other code.
- Touch arrives as pointer events: a tap is a left click with its `pos`. With no lock on touch, an app that picks at the crosshair should pick at `mouse_pos` through `camera.fov` when `capture_pointer` returns False.

### The GUI

`gui.js` draws `engine.gui` as HTML over the canvas from `state.json`'s `gui` (written by `src/manifoldx/web/gui.py`): `Panel`, `Text` and `ValueDisplay`, with their resolved styles, the GUI's font and its scale (rescaled with the window when the GUI has a `design_height`). Root panels take all nine anchors (`top-left` to `bottom-right`), an offset measured inward from the anchored edges; the desktop still paints every anchor top-left (manifoldx#13), so only top-left panels match today. `Button`, `Slider`, `Toggle` and `Image` fail the export with exit 1 and the widget's place:

```text
manifoldx export --web: the GUI widget Button is not supported by export --web yet (panel 0, child 2)
```

Reached code may write three widget fields, and the page applies them every frame as the desktop does: a Panel's `offset` (`hud.panel.offset = (x, y)` moves it), a Text's `text`, and any widget's `style_overrides` (`panel.style_overrides["width"] = 120`, or a new dict), resolved over the widget's theme and class. Each frame `gui.js`'s `sync` compares the widget objects (`engine.gui.widgets`, one per widget in `state.json`'s `gui`, by its `widget` index) with what it last applied, and writes only the CSS properties that changed. A write of any other field of a widget or of `engine.gui` would move the desktop and not the page, so it fails the export:

```text
app.py:21: writing `hud.panel.visible`, a field of a Panel the page does not apply, is outside the subset (reached from system tick)
```

The check needs the receiver's type: a widget held in an app object's field or a global types from the live object, and a write through a value of unknown type fails the export, naming what to annotate (see Annotating).

### Errors

When the app's code throws, the page stops its loop and shows the error panel (`#mx-error`, also `__mx.error` and the console) with the Python file and line from the source maps, then the Python callers:

```text
raises.py:26 in tick: TypeError: Cannot read properties of null (reading 'x')
```

Nothing runs after it; reload to start again. A load error (a missing file, no WebGL2) shows in the same panel.

### The scene and its calibration

`scene.js` builds the three.js scene from `scene.glb` and the glTF's manifoldx extras: background, fog, the sky environment, the sun and shadows. `StandardMaterial` is PBR, its `opacity` below 1 written as BLEND with alpha = colour alpha × opacity. `BasicMaterial` and `PhongMaterial` get the desktop's fixed world light (0.3 ambient + 0.7 Lambert from (1, 1, 1)), which a glTF cannot carry, and only `StandardMaterial` is fogged, as on the desktop. `engine.set_material` swaps the materials of an entity's meshes; every material id in `mx_materials` is built at load, used or not.

three.js and the desktop light the same scene differently, so `CALIBRATION` at the top of `src/manifoldx/web/js/scene.js` scales them: `environment` is 1.45 and `sun` 1.0, with the measurements in the comment beside it. To re-measure after a change to the desktop's lighting or a three.js upgrade:

1. Run `uv run pytest tests/web/browser/test_visual.py -v -s` on a machine with a wgpu adapter (lavapipe is enough). It prints, for `web_walk` and `tests/web/apps/lit_yard.py` at 320×200, the desktop's and the page's mean colour per channel and their ratio.
2. If `lit_yard` misses, sweep the factor in the page: in a test, after the page is ready, call `__mx.scene.setCalibration({environment: f})` and take the mean again, for f from 1.30 to 1.70 in steps of 0.05 (each step moves the means about 0.5%).
3. Keep the factor whose worst channel is smallest, write it and the measured means in the comment beside `CALIBRATION`, and set `THRESHOLD` in `test_visual.py` to the worst case plus 2 points (today 0.3% + 2 = 2.3%; the floor of 4.0 in 0-255 is the bound for scenes near 130).

The breaking tests (`test_break_sky_uncalibrated`, `test_break_plain_standard_for_basic`) must still exceed the threshold; if one passes, the threshold is too loose.

### Upgrading three.js

`src/manifoldx/web/vendor/three/SOURCE` lists the files taken from the npm tarball and the upgrade steps: pack the version, check its integrity, copy the files, set `VERSION` and `manifoldx.web.export.THREE_VERSION`, then `uv run pytest tests/web/test_vendor.py` and re-measure the calibration.

## What reaches the browser

**Roots** are the functions the engine calls (`find_roots` in `reach.py`):

- each system, in registration order (`system <name>` in labels);
- each event handler except `startup` handlers (they ran at export) and manifoldx's own (`on("<event>") <name>`);
- each `ValueDisplay`'s getter, walking `engine.gui`'s panels and their children (`ValueDisplay.getter <name>`).

A root must be a plain `def` or a lambda of an app module. A bound method root (`engine.system(viewer.tick)`) brings its object into the snapshot; a closure root brings its cells and defaults.

**Reach** follows calls and attribute reads through types, from the roots. It stops at a reached global: the global's value comes from the snapshot, so its initializer is setup. `X0, X1 = grid(LEVELS.values())` does not bring `grid` into the JS, and only the globals reached code reads go in the snapshot, not those an initializer read. The app classes and functions a reached global holds are reached from its type. A method is reached when reached code calls it on a value whose type allows its class, never because its name matches. Of each reached class, only the fields reached code reads are written. Every function not reached is setup: it ran in Python and is not in the JS. `report.json` lists both (`transpiled`, `setup`), so read it when something you expected in the browser is missing.

**The snapshot** (`state.json`) starts from engine.state, the roots (bound methods' objects, closures' cells and defaults) and the module globals reached code reads, and writes what setup left there. A shared object stays one object. These cross: `None`, bools, numbers (an int beyond 2^53 does not), strings, lists, tuples, sets, dicts with str keys, 1-d float32 or float64 numpy arrays (not views: copy them at setup), objects of reached app classes, the engine and its `input`, `camera` and `gui`, materials (registered so `scene.glb` writes them), entity handles, GUI widgets in the engine's GUI, reached functions, bound methods and closures.

What does not cross:

- In a value reached code reads, it fails the export: `Viewer.names: a dict with the int key 1 cannot cross to the browser`. Other forms: `a <type>, a library object,`, `an object of <Class>, a class reached code does not use,`, `a numpy array of shape (3, 3) and dtype float64`, `a numpy view of another array (copy it at setup)`, `the function <name>, which is not reached code,`.
- In a key of `engine.state`, the key is left out, the export goes on, prints `left out of the state: engine.state['lock']: a _thread.lock, a library object, cannot cross to the browser` and lists it under `state_left_out` in `report.json`. Reached code that reads that key will not find it in the browser.

## The subset and its errors

Every construct outside the subset that reached code uses is an error with its file, line and the root that reached it; nothing is dropped:

```text
bad.py:12: try/except is outside the subset (reached from system walk)
app.py:9: global is outside the subset (reached from system tick)
app.py:6: `Path('x').exists` on a pathlib.Path, a library object the browser does not have, is outside the subset (reached from system tick)
app.py:6: `query`, the system's component query (component queries are not supported in the browser yet), is outside the subset (reached from system tick)
```

What is in: functions with positional and keyword arguments and defaults, nested functions and lambdas (closures), classes (dataclasses with `field(default_factory=...)`, `@property`, `__call__`, a value that may be one of several app classes), `if`/`for`/`while`/`break`/`continue`, comprehensions, f-strings with or without a format spec, `%` with Python's sign, `and`/`or` on any values, lists, tuples, dicts with str keys, sets, `math` (`sin cos tan asin acos atan atan2 sqrt exp log floor ceil fabs hypot radians degrees`, `pi e tau inf nan`), numpy 1-d float vectors (`np.array`, `np.zeros`, `np.arange`, `np.linalg.norm`, `+ - * /` and in-place `+=` that aliases see; float32 stays float32), `textwrap.wrap(s, width)`, the set methods `add`, `discard`, `remove` (KeyError when the item is missing), `clear` and `update` (any number of iterables), each returning None, and the builtins `len str bool int float abs min max any all range list tuple set next hasattr isinstance`. A JS Set compares items by identity, so `add`, `discard`, `remove` or `update` of a tuple, a list or an object fails the export: keep set items to strings, numbers, bools and None. Strings count code points, as Python does.

What is out, among others: `try`, `raise`, `with`, `global`, `nonlocal`, `del`, `assert`, `async`/`await`, `yield`, `match`, `:=`, `for`/`else` and `while`/`else`, imports inside functions, `*args`/`**kwargs`, slices with a step or assigned to, dict comprehensions, imports of libraries other than `math`, `numpy` and `textwrap`, and a system that uses its query (component queries in the browser are not done yet). A builtin or library function not listed above fails as `app.py:5: enumerate() is outside the subset (reached from system tick)`.

There is no `global`: state that frame code reassigns lives in an object's attribute (`rig.mode = "fly"`) or in `engine.state` (`engine.state["selected"] = box.name`), never in a module-level name. Module globals are fine to read and to mutate in place.

`engine.state` works in both of its desktop forms: subscripts (`engine.state["selected"]`, as `examples/web_walk.py` does) and attributes (`engine.state.level`, `engine.state.cam = {...}`, `engine.state.cam["yaw"]`), plus `in`, `not in`, `.get(key[, default])`, `.setdefault(key[, default])`, `.keys()`, `.values()` and `.items()`. The transpiler types `engine.state` as a State and turns its attributes into reads and writes of the runtime's Map; a missing key throws in the browser as it raises in Python. A value read from it is still of unknown type, so `engine.state.n += 1` fails as `engine.state['n'] += 1` does: read it into an annotated local (`n: int = engine.state.n`), then write `engine.state.n = n + 1`. These fail the export with the file and line:

```text
app.py:7: `engine.state.i` as a loop or unpacking target is outside the subset (reached from system tick)
app.py:7: unpacking into `engine.state.a` is outside the subset (reached from system tick)
app.py:7: `engine.state.keys`, a dict method used as a value, is outside the subset (reached from system tick)
app.py:7: hasattr() of a dict[str, unknown] is outside the subset (reached from system tick)
```

As in Python, a dict's own attribute wins over a key of the same name: keep keys like `keys` or `get` behind subscripts.

## Annotating

The transpiler types values from annotations, from the live objects setup left (each app class's fields, a union over its instances), and from call sites, solved to a fixed point. When an operator needs a type it could not find, the export stops and names what to annotate:

```text
app.py:6: `a + b` with an operand of unknown type (annotate `a`) is outside the subset (reached from system tick -> __main__.add)
```

An attribute read on a value the transpiler could not type fails the same way, since reach cannot tell which class's field to snapshot:

```text
app.py:8: `p.x`, an attribute of a value of unknown type (annotate `p`), is outside the subset (reached from system tick)
```

So does a write, `x.attr = v` or `x.attr += v`, since the export cannot tell whether `x` is a widget, the engine or an app object (a material or an entity handle types as unknown too):

```text
app.py:8: writing `x.attr`, a field of a value of unknown type (annotate `x`), is outside the subset (reached from system tick)
```

The operations that need a type are attribute reads and writes, `+`, `*`, `%` (and their `+=` forms), `==` and `!=` unless the other side is a known number, bool or None, f-string fields with no format spec and `str()` (JS cannot tell `3.0` from `3`), method calls, slicing, `len`/`abs`/`int`/`float`/`hasattr` and keyword arguments. Annotate the name it gives, and nothing else:

- a parameter: `def add(a: float, b: float)`;
- a local, typically a read of `engine.state`, which always types as unknown: `name: str | None = engine.state["selected"]`;
- a field, where `__init__` sets it: `self.parts: list[Part] = []`;
- a return, when callers need it before the body is typed: `def corners(self) -> list[float]:`. A return annotation wins over the inferred return.

The annotations understood are `int`, `float`, `bool`, `str`, `None`, app classes, `X | Y`, `Optional[X]`, `list[X]`, `set[X]`, `dict[K, V]`, `tuple[X, Y, ...]` (one type per item), `tuple[X, ...]` (any length, every item X) and `np.ndarray` (float64; build float32 vectors with `np.array(x, np.float32)`). An annotation the transpiler reads wins over a live type. One it cannot read (a bare `tuple`, a library type) gives way to the live type, the type of what setup left in the field; with no live value either, it is unknown. Unpacking a call result (`a, b = f()`) takes `f`'s return type, and a starred item in a display (`[*xs, *ys]`) keeps its items' types.

## The engine from transpiled code

Reached code may use exactly what `RUNTIME_SURFACE` in `src/manifoldx/web/types.py` lists; anything else of a manifoldx object types as unknown:

- `engine.input`, `engine.camera`, `engine.gui.pointer_over_gui` (whether the pointer is over a root panel, so a click there is the GUI's), `engine.w`, `engine.h`, `engine.set_material(entity, material)`, and `engine.state` (subscripts or attributes, see above);
- `engine.input`: `pressed_keys`, `modifiers`, `mouse_pos`, `mouse_delta`, `wheel_delta`, `pressed_buttons`, `viewport_size`, `pointer_captured`, `is_pressed(key)`, `just_pressed(key)`, `just_released(key)`, `is_mouse_pressed(button)`, `just_mouse_pressed(button)`, `just_mouse_released(button)`, `capture_pointer(on)`;
- `engine.camera`: `position` and `target` (float32 vectors), `fov` (vertical, in degrees: with `mouse_pos` and `engine.w`/`engine.h` it turns a tap into a ray), `near`, `far`, `get_forward()`;
- the GUI's widgets (see The GUI): a Panel's `offset`, a Text's `text`, every widget's `style_overrides`, to read and to write.

Reached code may write only the fields the page applies each frame: the camera's `position`, `target`, `fov`, `near` and `far`, which `scene.js`'s `sync` copies to the three.js camera before each render, the widget fields above, and `engine.state`'s keys. `position` and `target` take a float32 vector (`np.array(x, np.float32)`, or float32 arithmetic such as `cam.position + forward`; `cam.position += v` and `cam.position[i] = x` write in place) and `fov`, `near` and `far` a number: numpy would keep a float64 array or a tuple where the runtime rounds to float32. A write of any other field of the engine, its `input`, `camera` or `gui`, or another manifoldx object moves the desktop and not the page, so it fails the export:

```text
app.py:9: writing `engine.camera.up`, a field of the camera the page does not apply, is outside the subset (reached from system tick)
app.py:9: writing a vec[f64] to `engine.camera.position`, a float32 vector (build it with np.array(x, np.float32)), is outside the subset (reached from system tick)
```

Fields of app objects are written freely, through a global, a field or a typed local (`r: Rig = engine.state["rig"]`, then `r.x = 2.0`).

Spawning, destroying, component queries, transforms and events are not there yet. Event handlers are transpiled but neither the page nor a replay fires them: read input by polling `engine.input` in a system. Do the scene's building in a `startup` handler or at module level: that is setup.

## Replaying: proof that both sides agree

`manifoldx.web.replay` runs one script of input frames through the app in Python and through its export in Node, traces the same values on both sides, and reports the first frame where they differ. Use it for every app you export; it is how `tests/web/test_replay.py` checks `examples/web_walk.py`.

```python
from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

frames = [{"keys": ["w"], "just": [], "d": [4.0, 0.0], "buttons": [], "clicks": [], "vp": [800, 500]}] * 60
paths = ["rig.pos", "rig.yaw", "selected"]
assert export_web(None, "app.py", out, ["--level", "3"]) == 0
python = replay_python(None, "app.py", frames, paths, app_args=["--level", "3"])
mismatch = compare(python, replay_node(out, frames, paths))
assert mismatch is None, str(mismatch)          # "frame 41, paths.rig.yaw: 0.12 != 0.13"
```

Give `replay_python` the `app_args` the export got: the app starts with them as its `sys.argv[1:]`, so it takes the branch the export took. A path can read a widget's fields (`hud.panel.offset`) as any object's.

A frame is what `engine.input` reports for one frame: `keys` held, `just` pressed this frame, `d` the mouse motion, `buttons` held, `clicks` pressed this frame, `vp` the viewport size (a new size resizes the engine). Five more fields are optional and read as empty when absent: `up` and `released` (keys and buttons released this frame), `pos` (the pointer in CSS pixels from the canvas's top-left), `wheel` and `mods` (held modifiers among Alt, Control, Meta, Shift, sorted). The page builds every field; `src/manifoldx/web/replay.py`'s docstring is the reference. In a replay `capture_pointer` always succeeds, as in a headless browser.

Each side reads a frame, runs the systems once with dt 1/60, and traces the camera's position and target, what each ValueDisplay shows, the `["material", entity, material_id]` and `["capture", on]` calls of that frame, all of `engine.state`, and the values of `paths`.

A path starts at a global of the app's main module or, when it has no global by that name, at a key of `engine.state`, then goes through attributes and dict keys (`rig.pos`, `selected`). The global must be one reached code reads: the JS module has only those.

`compare` matches numbers by value (JSON does not tell `1` from `1.0`), never a number against a bool, and floats within 2^-30 of the larger one (2^-40 near zero). The tolerance is there because Python's `math` calls the C library and Node has its own: V8's `sin` and `cos` differ from glibc's in the last bit (62 of 2226 values in the web_walk replay). 2^-30 is 64 times finer than a float32 step, so a value that skipped float32 rounding still mismatches. `compare(a, b, rel=0)` compares bit for bit.

### Writing a script

- Generate frames from a seeded `random.Random` in the test file, and keep the frames as a JSON fixture so a change to the generator shows in the diff. `tests/web/test_replay.py` does this; `uv run pytest tests/web/test_replay.py --regen` rewrites its fixture. `--regen` is defined in `tests/web/conftest.py`, so pytest knows it only when you give it `tests/web` or a file in it; `uv run pytest --regen` from the root fails with `unrecognized arguments: --regen`.
- Exercise every branch you care about: clicks, keys that toggle modes, a resize, a drag. Then assert the trace shows it (web_walk's test asserts at least three distinct picks and a material change), or a script that never picks proves nothing about picking.
- Add breaking tests: damage the export one way (patch the generated JS, drop an object from `state.json`) and assert `compare` returns a mismatch. A replay that passes with a broken export is not checking anything.
- Without Node on the PATH, everything under `tests/web` skips with "node not found".

## Tests

```bash
uv run pytest tests/web -q --ignore=tests/web/browser    # parity, reach, snapshot, export, replay, the JS suites; needs node
uv run pytest tests/web/test_replay.py -q                # the web_walk replay alone
uv sync --group browser                                   # Playwright, for the browser tests
uv run pytest tests/web/browser -v -s                     # the page in headless Chromium
```

Each construct of the subset has a parity test that runs a `main()` in Python and its transpiled form in Node (`tests/web/parity.py`, `run_both`) and compares the results. Add one when you widen the subset. The page's modules have Node suites in `tests/web/js/*.test.mjs`, run by `tests/web/test_node_suites.py`.

`tests/web/browser/` serves an export with `python -m http.server` and opens it in headless Chromium through Playwright. Without the `browser` group it skips with "playwright not installed", without a Chromium Playwright can launch with "chromium not installed". On a shared machine, reuse the Chromium in `~/.cache/ms-playwright` by keeping playwright at the version `uv.lock` pins (1.63.0) rather than downloading another.

- `test_page.py` drives `examples/web_walk.py` through Playwright's real keyboard, mouse, wheel and viewport, one `__mx.tick(1/60)` after each step, then replays the frames the page built in Python and asserts `compare` is None. It also checks the error panel, `file://`, a device scale of 2, the progress count and the lock lost on Escape.
- `test_visual.py` compares the page's first frame with the desktop's `Engine.render_frame` by mean colour per channel (see the calibration above).

**The fake pointer lock.** Headless Chromium grants the lock, but under it each synthetic mouse move arrives as two cancelling movements, so mouse look nets zero. The session probes the lock once (granted, and does a 10 px move net 10 px?) and prints `pointer lock probe: ...` at the end of the run. When the probe fails, as it does in headless Chromium, every page gets `FAKE_LOCK` from `tests/web/browser/conftest.py`: an init script that keeps the lock in a field and records whether each request came with a user activation, which the replay test asserts.

**Frame checks.** The Python side replays the frames the page built, so a defect in turning DOM events into frames (a key's case, a missing `just`) changes both traces alike and `compare` cannot see it. The test asserts the frames themselves: the tick after `keyboard.down("w")` has `w` in `just` and `keys`, Shift+W gives `just == ["w"]`, the tick after the resize has `vp == [640, 480]`. Keep such checks for every input path you add. The material check likewise finds the meshes under the glTF node whose extras name the entity, rather than trusting the page's own map.

**CI** (`.github/workflows/test.yml`) runs three jobs: `test` (the whole suite on the offscreen backend, every GPU request skipped), `web` (`tests/web` outside `browser/`, with lavapipe so the startup device exists) and `browser` (lavapipe, the `browser` group and Playwright's Chromium, `tests/web/browser`). `web` and `browser` fail on any skip, so a missing tool shows as a red job, not a quiet pass. Read the `browser` job's log for the lock-probe line.
