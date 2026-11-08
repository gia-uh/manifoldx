---
when: exporting an app with `manifoldx export --web` or running it in a tab with `manifoldx run --web`, fixing an error it reports, annotating app code so it transpiles, serving or debugging the browser page it writes, re-measuring its calibration, or checking with a replay that the browser runs an app as Python does
triggers: ["export --web", "run --web", "web export", "transpile", "transpiler", "browser", "manifoldx.web", "replay", "replay_node", "outside the subset", "cannot cross to the browser", "annotate", "index.html", "three.js", "pointer lock", "CALIBRATION", "playwright"]
---

# Exporting an app to the web with `manifoldx export --web`

## When to use this

Reach for this when you export an app for the browser, when the export refuses an app and you have to change its code, when the page draws or behaves differently from the desktop, or when you want proof that the exported app behaves as the Python one does. The code is `src/manifoldx/web/` (the page's modules in `js/`, three.js in `vendor/three/`); the tests in `tests/web/` show every behaviour below, and `examples/web_walk.py` and `examples/web_crowd.py` are apps that export, replay and run in the page (web_crowd's systems are queries over the entity store).

The export writes a folder with a page: the transpiled app runs its systems every animation frame on the browser's input, three.js draws the exported scene, and the GUI is HTML. The same folder replays in Node. `manifoldx run --web` serves the same page and swaps each save into the open tab (see "While you edit: `run --web`"). Spec: `.knowledge/analysis/2026-11-06-web-target-design.html`. Plans: `.knowledge/plans/2026-11-06-web-target-plan-1-transpiler-and-replay.html` (transpiler, snapshot, replay), `.knowledge/plans/2026-11-06-web-target-plan-2-browser-page.html` (the page), `.knowledge/plans/2026-11-07-web-target-plan-3-run-web.html` (`run --web`) and `.knowledge/plans/2026-11-07-web-target-plan-5a-store-and-batches.html` (the entity store, queries, spawn and destroy, batches; spec `.knowledge/analysis/2026-11-07-web-target-slice5-design.html`) and `.knowledge/plans/2026-11-07-web-target-plan-5b-events-and-widgets.html` (events, input events, the clock, `random`, the widgets, the host page; same spec) and `.knowledge/plans/2026-11-07-web-target-plan-5c-skinning-and-second-consumer.html` (skinned entities, matrices, `camera.project`; same spec); each plan's "where this plan departs" section explains the choices that differ from the spec.

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
  scene.js                   the glTF in three.js: extras, CALIBRATION, the basic material, the geometry library, one InstancedMesh per batch, one SkinnedMesh per skinned entity
  gui.js                     engine.gui as HTML
  sourcemap.js               a JS error's stack as Python file:line
  progress.js                the progress line and the frame clock
  vendor/three/              three.js 0.186.1, unminified, as its SOURCE file says (2.3 MB, about 450 KB gzipped)
  fonts/DejaVuSansMono.ttf   the GUI's font, as the desktop draws it
  app/<module>.js, .js.map   one ES module per app module, with a source map to the .py; the script (or -m module) is app/__main__.js
  pyrt.js                    the Python runtime the generated code imports as $rt
  manifoldx.js               the engine core the page and Node share: state loader, the entity Store, queries, handles, spawn and destroy, input, camera, step, trace
  replay.mjs                 loads the folder in Node
  state.json                 the snapshot: reached objects, engine.state, the engine's size, origin and camera, the GUI, the store's layouts ("ecs")
  ecs.bin                    the entity store's arrays below the high-water mark: alive, membership, each component's rows
  scene.glb                  a library of geometries and materials by id, with no entity nodes (export_engine(library=True)), plus scene.export-report.json
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

`browser.js` loads `state.json`, imports the app's modules, streams `scene.glb` behind a progress bar (`#mx-progress`, hidden once loaded) and builds the scene and the GUI. Then, every animation frame, it takes one frame from the DOM's input, delivers the frame's events (see Events in the page), runs the systems with the frame's real dt, applies the widget fields they wrote, updates the ValueDisplays, follows the canvas's size, syncs the three.js camera to `engine.camera` and renders. The canvas fills the window; `engine.w` and `engine.h` are its CSS size, and the drawing buffer is that times the device pixel ratio, capped at 2.

### Input

`input.js` turns DOM events into the replay's frame form, so the page and a replay feed the engine the same way:

- Keys are named as the desktop names them (a letter lowercase, the space bar `Space`, `Esc` as `Escape`), tracked by their code, so Shift released before a key still releases what it pressed. Tab and Space are kept from the page; browser shortcuts are left alone. Losing focus releases everything.
- Buttons map as on the desktop (left 1, right 2, middle 3); `pos` is in CSS pixels from the canvas's top-left; pixel wheel deltas are divided by the device pixel ratio, lines and pages scaled as rendercanvas does. The context menu is suppressed.
- `engine.input.capture_pointer(True)` asks for pointer lock. A browser grants it only during a user gesture: with an active one it is requested at once, otherwise at the next click on the canvas. It returns False where the canvas has no pointer lock or the last pointer was a touch, as the desktop does with no window.
- When the lock ends without the app asking (the user pressed Escape, which the browser keeps from the page), the next frame's `just` holds `Escape` and `pointer_captured` turns False, as the desktop's Escape would. An app that releases on Escape needs no other code.
- Touch arrives as pointer events: a tap is a left click with its `pos`. With no lock on touch, an app that picks at the crosshair should pick at `mouse_pos` through `camera.fov` when `capture_pointer` returns False.

### The GUI

`gui.js` draws `engine.gui` as HTML over the canvas from `state.json`'s `gui` (written by `src/manifoldx/web/gui.py`): `Panel`, `Text`, `ValueDisplay`, `Button`, `Slider`, `Toggle`, `Joystick` and `Image`, with their resolved styles (the keys the desktop's painter reads), the GUI's font and its scale (rescaled with the window when the GUI has a `design_height`). Root panels take all nine anchors (`top-left` to `bottom-right`), an offset measured inward from the anchored edges (on a centred axis, rightward or downward), and the desktop paints and hit-tests them by the same rule (`gui.layout.anchored_slot`). A panel without a width or height fits its content in the page; on the desktop it does too, except a top-left one, which fills the window, so give a top-left panel a size for the two to agree. A widget of another class fails the export with exit 1 and its place:

```text
manifoldx export --web: the GUI widget Gauge is not supported by export --web yet (panel 0, child 2)
```

The widgets, what they emit, and what they hold:

| Widget | Fields | Event and payload |
| --- | --- | --- |
| `Button(name=, label=)` | `label`, `was_clicked()` (True once per click, the latch consumed on read) | `gui:button:<name>:click`, `{}`, on the pointer's down |
| `Slider(name=, min=, max=, value=, label=)` | `value`, `min`, `max`, `label` | `gui:slider:<name>:change` `{"value": v}` on down and on each move of the drag; `:commit` on the up |
| `Toggle(name=, value=, label=)` | `value` (bool), `label` | `gui:toggle:<name>:change` `{"value": b}` on the down |
| `Joystick(name=)` | `value`, `(x, y)` in the unit disc, y up, `(0.0, 0.0)` at rest | `gui:joystick:<name>:change` `{"value": (x, y)}` on down, move and up (the up gives `(0.0, 0.0)`) |
| `Image(svg)` (a path, SVG markup or bytes) | (none) | (none); an Image of a texture is left blank in the page |

The widget code that runs is the desktop's own, in both replays, and the page runs a line-for-line port of it, so a value is set before its event is queued and the event reaches its handler the frame after the pointer (see Events in the page). A Slider and a Joystick keep the pointer that went down on them until it lifts, one capture per pointer, so a thumb on the Joystick and another finger on a Button work at once. After a pointer goes down on a Slider, the arrow keys step it by a twentieth of its range (Left and Down lower it, Right and Up raise it), emitting change and commit, until a pointer goes down outside every widget; this works on the desktop too. An Image is shown from its SVG text as a `data:image/svg+xml` URL, which runs no script and needs no request; measuring the SVG at export needs the `svg` extra (`uv sync --extra svg`). Each interactive element has `touch-action: none` and its ARIA role (the Slider `role="slider"` with `aria-valuenow`, the Toggle `role="switch"` with `aria-checked`, the Button a `<button>`), and every widget's element carries `data-mx-widget` with its index in `engine.gui.widgets`.

A pointer that goes down on the GUI is also a click in the frame (`buttons`, `clicks`, `pos`), as the desktop's input layer records every pointer event before the GUI sees it. An app that picks on click checks `engine.gui.pointer_over_gui` first, as `examples/web_crowd.py`'s `pick` does.

Reached code may write these widget fields, and the page applies them every frame as the desktop does: a Panel's `offset` (`hud.panel.offset = (x, y)` moves it), a Text's `text`, a Button's `label`, a Slider's `value`, `min` and `max`, a Toggle's `value`, a Joystick's `value`, and any widget's `style_overrides` (`panel.style_overrides["width"] = 120`, or a new dict), resolved over the widget's theme and class. A write sets the field and emits nothing, as on the desktop. Each frame `gui.js`'s `sync` compares the widget objects (`engine.gui.widgets`, one per widget in `state.json`'s `gui`, by its `widget` index) with what it last applied, and writes only the CSS properties that changed. A write of any other field of a widget or of `engine.gui` would move the desktop and not the page, so it fails the export:

```text
app.py:21: writing `hud.panel.visible`, a field of a Panel the page does not apply, is outside the subset (reached from system tick)
```

The check needs the receiver's type: a widget held in an app object's field or a global types from the live object, and a write through a value of unknown type fails the export, naming what to annotate (see Annotating).

### Errors

When the app's code throws, the page stops its loop and shows the error panel (`#mx-error`, also `__mx.error` and the console) with the Python file and line from the source maps, then the Python callers:

```text
raises.py:26 in tick: TypeError: Cannot read properties of null (reading 'x')
```

Nothing runs after it; reload to start again (under `run --web`, the next save resumes it). A load error (a missing file, no WebGL2) shows in the same panel.

### The scene and its calibration

`scene.js` builds the three.js scene from `scene.glb` and the glTF's manifoldx extras: background, fog, the sky environment, the sun and shadows. `StandardMaterial` is PBR, its `opacity` below 1 written as BLEND with alpha = colour alpha × opacity. `BasicMaterial` and `PhongMaterial` get the desktop's fixed world light (0.3 ambient + 0.7 Lambert from (1, 1, 1)), which a glTF cannot carry, and only `StandardMaterial` is fogged, as on the desktop. Every geometry id in `mx_geometries` and material id in `mx_materials` is built at load, used or not, and the entities are drawn from the store, one batch per (geometry, material) (see Entities in the page).

three.js and the desktop light the same scene differently, so `CALIBRATION` at the top of `src/manifoldx/web/js/scene.js` scales them: `environment` is 1.45 and `sun` 1.0, with the measurements in the comment beside it. To re-measure after a change to the desktop's lighting or a three.js upgrade:

1. Run `uv run pytest tests/web/browser/test_visual.py -v -s` on a machine with a wgpu adapter (lavapipe is enough). It prints, for `web_walk` and `tests/web/apps/lit_yard.py` at 320×200, the desktop's and the page's mean colour per channel and their ratio.
2. If `lit_yard` misses, sweep the factor in the page: in a test, after the page is ready, call `__mx.scene.setCalibration({environment: f})` and take the mean again, for f from 1.30 to 1.70 in steps of 0.05 (each step moves the means about 0.5%).
3. Keep the factor whose worst channel is smallest, write it and the measured means in the comment beside `CALIBRATION`, and set `THRESHOLD` in `test_visual.py` to the worst case plus 2 points (today 0.3% + 2 = 2.3%; the floor of 4.0 in 0-255 is the bound for scenes near 130).

The breaking tests (`test_break_sky_uncalibrated`, `test_break_plain_standard_for_basic`) must still exceed the threshold; if one passes, the threshold is too loose.

### While you edit: `run --web`

```bash
uv run manifoldx run --web app.py                  # a script
uv run manifoldx run --web -m mygame.main          # a module
uv run manifoldx run --web app.py --port 8800 -- --level 3
```

It exports the app, serves it on 127.0.0.1 and prints `manifoldx run --web: open http://127.0.0.1:8770/ (dev page at /dev)`. `/` is the app's page, `/dev` the dev page that `/` is under plain `run` (`know-how/agent-loop.md`). The same reloader watches the app's files as `manifoldx run` does (`know-how/hot-reload.md`), and a save:

1. asks the tab for its `engine.state`; the tab freezes (no systems run, the last frame stays) until the new version, a failure, or 30 s;
2. merges what the tab sent into Python's `engine.state`, reruns the app's setup until `engine.run()` and its startup handlers on the same headless device, and builds the export again;
3. writes it as a new version and pushes it to every open tab, which loads the new modules and snapshot in place, without a page reload, and resumes.

What survives a save:

- **`engine.state`, as plain data.** The tab sends its state in the snapshot's forms; numbers, strings, lists, tuples, dicts, sets and float vectors come back and replace Python's values, a whole number comes back a float where Python had a float, and a list a tuple where Python had a tuple. A key whose value holds an object of an app class, a handle or a function, and a key that never crossed (`state_left_out`), keeps Python's value; the log says `kept from Python: ...` once per key. This is the desktop's rule: store plain data.
- **The three.js scene, when the scene's hash did not change.** The batches rebuild from the new version's store at the next frame, so materials the app set at runtime (`set_material`) are gone, as a rebuilt scene would have them; the new version sets them again. A changed scene is fetched, built and swapped in, then the old one is freed.
- **The pointer lock and held keys.** The input listeners live for the page's life, so a key held across the save is still held, and the new `engine.input` learns that the lock is on.

The GUI is rebuilt from the new snapshot every time. A tab that opens later, or an F5, gets a rebuild with the last state any tab sent (on `pagehide` and with every reply), so the camera stays where it was. Several tabs all receive every version.

The status line at the bottom right says where the reload is:

| status line | when | error panel |
|---|---|---|
| `v3 · live` | running | hidden |
| `v4 · reloading` | from the state request to the swap | unchanged |
| `v3 · reload failed, running v3` | an edit failed to load or transpile | the error, e.g. `dev_orbit.py:12: SyntaxError: '(' was never closed`, then `Fix it and save: the tab keeps running v3.` |
| `v3 · paused until the next edit` | an exception in a frame | the Python file, line and callers, as in Errors above |
| `v3 · dev server gone, retrying` | the server stopped; the stream reconnects by itself | unchanged |

An exception in a frame pauses the tab until the next save, even an unrelated one, as on the desktop. A version that fails to load in the tab (a module that throws on import) shows its error with `The tab keeps running v3.` and the running version goes on.

`.manifoldx/web/` holds what the server serves: the page's modules, the runtime, `vendor/three/` and the font once at the root, `v{n}/` per version (`app/`, `state.json`, `report.json`, the last three kept), and `scenes/<sha>.glb`, written only when the hash is new. Under `--web` the glTF has no camera node, so moving the view does not change the scene's hash. `--debug` is refused (`--debug measures the window's frames; under --web the tab runs them`, exit 2).

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
```

What is in: functions with positional and keyword arguments and defaults, nested functions and lambdas (closures), classes (dataclasses with `field(default_factory=...)`, `@property`, `__call__`, a value that may be one of several app classes), `if`/`for`/`while`/`break`/`continue`, comprehensions, f-strings with or without a format spec, `%` with Python's sign, `and`/`or` on any values, lists, tuples, dicts with str keys, sets, `math` (`sin cos tan asin acos atan atan2 sqrt exp log floor ceil fabs hypot radians degrees`, `pi e tau inf nan`), numpy 1-d float vectors (`np.array`, `np.zeros`, `np.arange`, `np.linalg.norm`, `+ - * /` and in-place `+=` that aliases see; float32 stays float32) and the array subset of Entities in the page, `textwrap.wrap(s, width)`, the set methods `add`, `discard`, `remove` (KeyError when the item is missing), `clear` and `update` (any number of iterables), each returning None, and the builtins `len str bool int float abs min max any all range list tuple set next hasattr isinstance`. A JS Set compares items by identity, so `add`, `discard`, `remove` or `update` of a tuple, a list or an object fails the export: keep set items to strings, numbers, bools and None. Strings count code points, as Python does.

What is out, among others: `try`, `raise`, `with`, `global`, `nonlocal`, `del`, `assert`, `async`/`await`, `yield`, `match`, `:=`, `for`/`else` and `while`/`else`, imports inside functions, `*args`/`**kwargs`, slices with a step or assigned to, dict comprehensions, and imports of libraries other than `math`, `numpy`, `textwrap`, `random` and `time` (see Events in the page for the last two). A builtin or library function not listed above fails as `app.py:5: enumerate() is outside the subset (reached from system tick)`.

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

Reached code may use exactly what `RUNTIME_SURFACE` in `src/manifoldx/web/types.py` lists; any other member of one of those objects fails the export, whether it is read, called as a statement or used in an expression:

- `engine.input`, `engine.camera`, `engine.gui.pointer_over_gui` (whether a pointer event of this frame went to an interactive widget, so a click there is the GUI's), `engine.w`, `engine.h`, `engine.set_material(entity, material)`, `engine.spawn(...)`, `engine.destroy(...)` and `engine.set_joints(indices, matrices)` (see Entities in the page), `engine.emit(name, payload)` and `engine.elapsed` (see Events in the page), `engine.log(text)` (under `run --web` the line goes to the dev server's log; an exported page drops it, as the desktop does outside `run`, and a replay in Node collects it in `engine.lines`), and `engine.state` (subscripts or attributes, see above);
- `engine.input`: `pressed_keys`, `modifiers`, `mouse_pos`, `mouse_delta`, `wheel_delta`, `pressed_buttons`, `viewport_size`, `pointer_captured`, `is_pressed(key)`, `just_pressed(key)`, `just_released(key)`, `is_mouse_pressed(button)`, `just_mouse_pressed(button)`, `just_mouse_released(button)`, `capture_pointer(on)`;
- `engine.camera`: `position`, `target` and `up` (float32 vectors), `fov` (vertical, in degrees: with `mouse_pos` and `engine.w`/`engine.h` it turns a tap into a ray), `near`, `far`, `get_forward()`, `project(points, width, height)` (an `(n, 3)` array of world points to their `(n, 2)` pixels, origin top-left, and their `(n,)` depths, both float64, as the desktop's: put a DOM label where an entity is);
- the GUI's widgets (see The GUI): a Panel's `offset`, a Text's `text`, the fields of `Button`, `Slider`, `Toggle` and `Joystick` in its table, every widget's `style_overrides`, to read and to write.

The error names the line:

```text
app.py:8: `engine.enable_shadows`, a member of the engine outside the page's runtime surface, is outside the subset (reached from system tick)
```

Reached code may write only the fields the page applies each frame: the camera's `position`, `target`, `up`, `fov`, `near` and `far`, which `scene.js`'s `sync` copies to the three.js camera before each render, the widget fields above, and `engine.state`'s keys. `position`, `target` and `up` take a float32 vector (`np.array(x, np.float32)`, or float32 arithmetic such as `cam.position + forward`; `cam.position += v` and `cam.position[i] = x` write in place) and `fov`, `near` and `far` a number: numpy would keep a float64 array or a tuple where the runtime rounds to float32. A write of any other field of the engine, its `input`, `camera` or `gui`, or another manifoldx object moves the desktop and not the page, so it fails the export:

```text
app.py:9: writing `engine.camera.zoom`, a field of the camera the page does not apply, is outside the subset (reached from system tick)
app.py:9: writing a vec[f64] to `engine.camera.position`, a float32 vector (build it with np.array(x, np.float32)), is outside the subset (reached from system tick)
```

Fields of app objects are written freely, through a global, a field or a typed local (`r: Rig = engine.state["rig"]`, then `r.x = 2.0`).

Handlers and systems registered with the decorators (`@engine.on("bump")`, `@engine.system` on a module function, `engine` a module global) export as the call forms `engine.on("bump")(bump)` and `engine.system(orbit)` do; any other decorator fails the export. Do the scene's building in a `startup` handler or at module level: that is setup.

The runtime's record of the engine calls a frame made (`["material", ...]`, `["capture", ...]`) is `engine.calls`; the trace's JSON keeps the key `log` for it, so fixtures and `replay.py` read it as before.

## Events in the page

The page has the desktop's event bus. `engine.emit(name, payload)` and every `@engine.on(name)` handler work in transpiled code, and each frame runs as `Engine._draw_frame` does:

1. the queue gets, in order, the `frame` event, what was emitted since the last delivery, and this frame's own events (input, host page, agent) last;
2. the queue is delivered, each event to its handlers in registration order; manifoldx's own GUI handlers come first, so a widget sees a pointer before the app does;
3. the systems run, then the flush.

Delivery swaps the queue before it starts, so an event emitted by a handler or a system waits for the next frame's delivery: a Button's click reaches its handler the frame after the pointer went down, and an event the click handler emits one frame after that. Write handlers that expect it.

**Input events.** The seven input events of the desktop reach handlers with payloads of the desktop's classes (`manifoldx.input`), built from the frame's events:

| Event | Payload fields |
| --- | --- |
| `key_down`, `key_up` | `key`, `modifiers` (a tuple of str), `is_down` |
| `pointer_down`, `pointer_up`, `pointer_move` | `x`, `y` (CSS pixels from the canvas's top-left), `dx`, `dy`, `button`, `buttons` (held after the event), `modifiers`, `phase`, `pointer_id` (0 for the mouse and the first finger, the DOM's `pointerId` for the others; always 0 on the desktop) |
| `wheel` | `dx`, `dy`, `x`, `y`, `buttons`, `modifiers` |
| `resize` | `width`, `height`, `pixel_ratio` |

Their fields are read only. Polling `engine.input` in a system still works and is often simpler.

**The frame event and `engine.elapsed`.** `frame` is first in every frame's queue, with `{"dt": dt, "elapsed": engine.elapsed, "frame": n}`. `engine.elapsed` is 0.0 on the first frame and adds each later frame's dt, as the desktop does in wall-clock mode and, one `dt` per simulated step, under `set_fixed_timestep`; it is read only.

**A handler's payload.** The transpiler types the payload of the events above, and of the widget events, from a table (`EVENT_PAYLOADS` in `src/manifoldx/web/types.py`). An app's own event has no class, so annotate its handler's payload, or the export stops:

```text
app.py:5: the payload of on("ping") on_ping, with no type (annotate it: def on_ping(p: dict)), is outside the subset
```

**The clock.** `time.time()` in frame code returns the frame's clock: the page reads `Date.now() / 1000` once per frame and never lets it run backwards, and a replay frame carries it as `clock`. Any other function of `time`, and `from time import time`, fail the export with a hint to use `time.time()` or `engine.elapsed`. Setup runs in Python and sees the real time.

**`random.Random`.** Make a generator at setup (`RNG = random.Random(1107)`, a global or a field) and call `random()`, `uniform(a, b)`, `randrange(...)`, `randint(a, b)`, `choice(seq)` and `getrandbits(k)` (k up to 53) on it in frame code: the page runs CPython's MT19937 from the state the generator had at export, so both draw the same numbers bit for bit. The module functions `random.random`, `random.uniform`, `random.randint`, `random.randrange` and `random.choice` use the module's generator, whose state the export writes too. Seed every generator at setup (`random.seed(n)` for the module's): an unseeded one differs between any two runs, so no replay can match. Making or reseeding a generator in frame code fails the export (`a reseed in frame code (seed it at setup)`), as do its other methods and `from random import ...`.

## A host page around the export

A page that frames the export in an `<iframe>` talks to the app through events, with `postMessage`:

- **Into the app.** The host posts `{mx: "emit", name, payload}` to the frame's window, the payload plain JSON. The page takes it from `window.parent` only, records it in the next frame's events, and the app's handlers get it as from `engine.emit` (a JSON object is a dict). A message posted while the page loads waits for its first frame.
- **Out of the app.** Every event the app or its widgets emit is posted to `window.parent` as `{mx: "emit", name, payload}` when the queue delivers it, and dispatched on the page's own window as the `CustomEvent` `"mx:emit"` with `detail: {name, payload}`. The payload is as `state.json` writes values, so a dict arrives as `{"__dict": [[key, value], ...]}`. Input events and the `frame` event are not sent.

```html
<iframe id="app" src="out/index.html"></iframe>
<script>
  const app = document.getElementById("app");
  addEventListener("message", (ev) => {
    if (ev.source === app.contentWindow && ev.data?.mx === "emit") console.log(ev.data.name, ev.data.payload);
  });
  function hello() { app.contentWindow.postMessage({ mx: "emit", name: "host:hello", payload: { n: 2 } }, "*"); }
</script>
```

The app handles `host:hello` with `@engine.on("host:hello") def hello(p: dict)` and answers with `engine.emit("app:ack", {"n": p["n"]})`. Since the host's events are in the frames, a replay of the page's frames runs them too (`tests/web/browser/test_host_page.py`).

## Entities in the page

The page holds manifoldx's entity store. The export writes the store's arrays below the high-water mark into `ecs.bin` (float32 and uint32 only: a component of another dtype fails the export with `the component Flags: dtype int64; ecs.bin carries float32 and uint32 and it cannot cross to the browser`) and their layouts under `state.json`'s `"ecs"`. `manifoldx.js`'s `Store` keeps them as typed arrays with the desktop's slot rules, and the transpiled systems read and write them. `examples/web_crowd.py` uses all of what follows.

**Queries filter by component, on the desktop too.** A system `def drift(q: mx.Query[Transform, Velocity], dt: float)` sees the alive entities that have every component it names, in slot order, and nothing else; before this, the desktop's query returned every alive entity. A query naming a component no entity has is empty, and one naming a component never registered fails on its first field read as the desktop's does (`ValueError: Component Life not registered`), so register a component the queries read before its first spawn (`Velocity.register(engine.store)` at setup). The parameter's type comes from the system's registration, so `Query[...]`, `mx.Query[...]` and the desktop's other spellings all work. An event handler registered with a query gets a read-only one: a write through it fails the export.

**Fields keep the desktop's semantics.** `q[Velocity].v` is a field view, not an array, as `_FieldView` is on the desktop:

- `len(q)` and `for i in q` (entity indices) work; `q[X].f.data` is the `(k, c)` array, to use with numpy.
- Arithmetic casts the other operand to float32 first, so a float64 operand does not promote; a number may stand on the left (`2.0 * f`), an array may not.
- `f += x`, `f -= x`, `f *= x`, `f /= x` and `q[X].f = x` queue a write that lands at the flush after every system.
- `f[:, j]` is a one-column view (new on the desktop too). A comparison takes a one-column field and gives the entity indices where it holds, so `engine.destroy(q[Transform].pos[:, 1] < -10.0)` destroys the entities below -10. A write into a column fails, as the desktop raises.
- There is no unary minus on a field, and a field inside a numpy function or a builtin fails the export, naming `.data`. Mesh and Material fields fail the export: read them through handles.

**What queues and what is immediate** follows `Engine._draw_frame`: query writes and `engine.destroy` queue and apply at the flush after every system, so a later system of the same frame still sees the old rows and the destroyed entities; `engine.spawn`, handle writes and `set_material` apply at once.

**Spawn and destroy.** `engine.spawn(Mesh(CUBE), Material(RED), Transform(pos=pos), Velocity(v=vel), n=24)` spawns `n` entities, each component's values a number, a tuple of numbers or an array of one row per entity; with no `n` it returns the entity's handle. A geometry must be one the export saw (a global built at setup): `Mesh()` of a geometry built in frame code fails the export, and `Material()` takes a material made at setup. Slots are taken as `EntityStore.spawn` takes them: the free list from its end, then the high-water mark; past the capacity the page loaded the arrays double, and past `max_entities` it throws the desktop's message. A Component subclass first spawned in frame code is registered at that spawn, as on the desktop. `engine.destroy` takes an array of entity indices, a bool mask, a list of ints or None, dedupes and skips dead slots.

**Handles.** `engine.spawn(...)` returns an entity handle, and a handle kept in an app object or a global crosses: `h.index`, `h.transform.pos` (world coordinates, float64, so the floating origin of `engine.set_origin` is subtracted), `.rot` and `.scale` (float32), read and written at once (`h.transform.pos = (x, y, z)`, `h.transform.pos += v`), `h.set_material(m)`, `h.set_joints(matrices)` (see Skinned entities) and `h.destroy()`. `Transform.rotation(x, y, z)` with numbers builds a quaternion. Annotate a field that holds one (`ball: EntityHandle | None = None`, at class level).

**The array subset.** Arrays of rank 1 and 2 of float32, float64, int32 and bool, and stacks of matrices of any rank for `@`, `np.stack` and `set_joints`, with numpy's rules for dtypes, broadcasting and rounding (each operation writes a new array of the result's dtype, no fusion):

- built by `np.array` (of numbers, or nested: a list of equal-length rows of numbers or rank-1 arrays gives a matrix), `np.zeros`, `np.ones`, `np.full` (a float fill), `np.arange`, `np.eye(n, dtype=...)`, `np.stack` (of equal-shape arrays of any rank; `axis=1` of rank-1 arrays) and `np.column_stack` (of rank-1 arrays), or read from a field's `.data`;
- `+ - * /`, `@` (see below), comparisons, `&`, `|`, `**` with the exponents 2, 0.5 and -1 only (another fails with `` `a ** 3`, a power numpy does not compute exactly in the page (2, 0.5 and -1 are in; use a * a * a) ``), indexing by an int, `[:, j]`, a bool mask or an index array (no other slices) (past the end it throws numpy's `IndexError`), `.shape`, `.tolist()`;
- `np.sin cos exp sqrt floor clip minimum maximum where sum mean min max argmin argmax` and `np.linalg.norm` (`axis` by keyword); reductions take `axis` None, 0 or 1.

Its limits: float32 `sin`, `cos` and `exp` may differ from numpy's in the last float32 step (exp in two), since numpy runs its own float32 kernels; keep trigonometry the replay must match in float64, as web_crowd does. Integer arrays are int32 in the page where numpy makes int64, which differs only past 2^31. `a += b` on an array fails the export (numpy writes in place and keeps the dtype): build a new array. For random numbers use `random.Random` (see Events in the page).

**`@` on matrices and stacks.** `a @ b` takes `(..., n, k) @ (..., k, m)` and broadcasts the leading dimensions by numpy's rule (a dimension of 1 stretches; anything else throws numpy's `ValueError`). float64 sums each product in order. float32 adds each exact product to a float64 accumulator and rounds to float32 after each step, k in order: the FMA chain numpy's float32 matmul computes, so the page matches it bit for bit. int32 wraps each step; bool is out. A vector in `@` fails the export, because numpy multiplies a matrix by a vector through another kernel whose rounding the page cannot match: write the vector as a column (`m @ np.array([[x], [y], [z]])`, then read `[:, 0]` of the result) or as a row (`np.array([[x, y, z]]) @ m`).

```text
app.py:12: `m @ v`, a matrix times a vector, which numpy rounds in an order the page cannot match (write the vector as a column: m @ np.array([[x], [y], [z]])) is outside the subset (reached from system walk)
```

Build a matrix from rows, not by writing into one: a write through an index is out, so the desktop's `m[:3, :3] = r` has no page form. A helper that returns a matrix leaves its return unannotated, since `-> np.ndarray` types as a float64 vector; `about()` in `examples/web_crowd.py` builds a rotation about a pivot as `translation(p) @ r @ translation(-p)` from nested `np.array(..., np.float32)`. Indexing by `None` (`np.eye(4)[None]`) is out: `np.stack([np.eye(4, dtype=np.float32)])` gives the `(1, 4, 4)` stack. A `(4, 4)` global does not cross the snapshot, which carries 1-d arrays only: build the matrix in frame code.

**Skinned entities.** A geometry with joints (`Mesh(joints=...)` of `manifoldx.modeling`, one joint per vertex) spawns skinned entities. Pose one with `h.set_joints(np.stack([...]))`, a `(J, 4, 4)` stack, or many in one call with `engine.set_joints(np.array([h.index for h in hs]), np.stack(poses))`, an `(N, J, 4, 4)` stack: each matrix moves its joint's vertices in model space, before the Transform, and joints past J keep what an earlier call gave them, identity until then. Both check what the desktop checks, with its messages: an entity whose geometry has no joints, or a shape other than these, throws. The poses travel in `ecs.bin` as whole 32-joint blocks, and the trace's `joints` holds each posed entity's pose (its geometry's joint count of row-major matrices), or a digest past 8 posed entities; a hand-built trace entry needs `traceJoints`. `examples/web_crowd.py`'s figure walks a circle by its handle and is posed each frame from matrices built in frame code.

The page draws each skinned entity as a three.js `SkinnedMesh` of its own (the geometry shared, a skeleton per entity), outside the batches: `__mx.scene.skinned` maps the entity to its mesh, whose `skeleton.boneMatrices` is the pose transposed (three is column-major), current after each `sync`. Each binds with `DetachedBindMode` and identity inverse binds, so a vertex lands at TRS · pose[j] · v. The basic light reads the skinned normal. The glTF the desktop's `export_gltf` writes carries the same skin: `JOINTS_0`, `WEIGHTS_0` of 1, joint nodes under `<name>_skeleton` holding the Transform.

**Batches and draw calls.** `scene.glb` under `--web` is a library: each geometry and material by id, mapped in the scene's extras (`mx_geometries`, `mx_materials`), with no entity nodes. `scene.js` keeps one `InstancedMesh` per (geometry, material) of the alive entities with a Mesh, in slot order; an entity with no Material is drawn white with the basic light. When the store's version moves (a spawn, a destroy, a Mesh or Material write) or the store is another one (a new version under `run --web`) the batches are rebuilt; otherwise only the dirty Transform rows are written. A batch that empties draws nothing. One frame draws one call per non-empty batch and one per skinned entity (`__mx.renderer.info.render.calls`), plus the shadow pass when the scene has shadows. `__mx.scene.batches` maps `"gid:mid"` to the InstancedMesh, and `__mx.scene.materialOf(entity)` reads the store's Material column as the last sync saw it (null before the first frame).

**Ids.** Geometry and material ids are labels: the export registers every material and geometry its snapshot meets, in walk order, and the Python replay does the same after setup, so both sides name them alike. The desktop app outside a replay registers them at first use.

`__mx.frameMs` holds `{work, render}` of the last frame in ms: `work` from the frame's start to the end of the batches' sync, `render` the `renderer.render` call. `uv run python -m benchmarks web` reads it (`know-how/benchmarking.md`).

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

A frame is what `engine.input` reports for one frame: `keys` held, `just` pressed this frame, `d` the mouse motion, `buttons` held, `clicks` pressed this frame, `vp` the viewport size (a new size resizes the engine). Five more fields are optional and read as empty when absent: `up` and `released` (keys and buttons released this frame), `pos` (the pointer in CSS pixels from the canvas's top-left), `wheel` and `mods` (held modifiers among Alt, Control, Meta, Shift, sorted), `events` and `clock`. `events` is the frame's events in the order they happened, each `[name, payload]` with a JSON payload: the input events with the fields of Events in the page (lists for the tuples), a host page's or an agent's events as they came. A pointer event routed to a widget carries `"gui": [widget, fx, fy]`, the widget's index in `engine.gui.widgets` and the pointer's place in that widget's box as fractions of its width and height: the page decides which widget a pointer is on, and the Python replay hands that widget to the desktop's GUI bridge (it replaces only `_GuiBridge._locate`), so the widget code runs as on the desktop. A pointer event with no `gui` is on no widget. `clock` is what `time.time()` returns in that frame, 0.0 when absent. The page builds every field; `src/manifoldx/web/replay.py`'s docstring is the reference. In a replay `capture_pointer` always succeeds, as in a headless browser.

Each side reads a frame, delivers its events and runs the systems once with dt 1/60, and traces `events`, the names of the events delivered that frame in order (payloads are left out: keep the values you care about in `engine.state` from a handler), the camera's position and target, what each ValueDisplay shows, the `["material", entity, material_id]` and `["capture", on]` calls of that frame, all of `engine.state`, the values of `paths`, and the entity store after the frame's flush: `store` holds `alive` (the alive slots), `free` (the free list), `high_water` and `rows`, each component's rows of the alive slots, or past 1,000 alive entities `{"digest", "n"}`, the weighted sum of its uint32 words mod 2^32. NaN and infinities are written `"nan"`, `"inf"` and `"-inf"`, since JSON has neither. The Python replay flushes after the systems as `Engine._draw_frame` does, and registers materials and geometries in the export's order, so ids agree.

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
- `test_run_web.py` runs `manifoldx run --web` of `tests/web/apps/dev_orbit.py` in a temporary copy (the `run_web` fixture, which needs the `manifoldx` script beside `sys.executable`) and edits it: a save keeps the state, a colour edit rebuilds the scene, a broken edit and a frame error show their lines, the lock and a reopened tab's camera survive, and the agent commands answer from the tab. Two breaking tests damage the tab (an empty state answer, batches that rebuild on a new version number but not on a new store) and assert that the check then fails.
- `test_web_crowd_page.py` drives `examples/web_crowd.py` with `tests/web/test_web_crowd.py`'s script (each frame's events made with Playwright's keyboard, mouse and wheel, a routed one at its place in the widget's real box), replays the page's frames in Python, and checks that `renderer.info.render.calls` equals the number of non-empty batches plus one for the figure, that the figure's bones equal the pose the Python replay gave it on each of 30 frames, and that a picked box is drawn by the batch of its material. It uses each widget under the mouse, a Joystick and a Button under two fingers at once (a touch context, `touch(page, points)` in `conftest.py` sending `Input.dispatchTouchEvent`), and checks the widgets' ARIA roles; a breaking test serves a copy whose GUI bridge keeps one capture for all pointers and sees the Joystick stop following its finger.
- `test_host_page.py` frames an export in a host page and sends events both ways.
- `test_visual.py` compares the page's first frame with the desktop's `Engine.render_frame` by mean colour per channel (see the calibration above).

**The fake pointer lock.** Headless Chromium grants the lock, but under it each synthetic mouse move arrives as two cancelling movements, so mouse look nets zero. The session probes the lock once (granted, and does a 10 px move net 10 px?) and prints `pointer lock probe: ...` at the end of the run. When the probe fails, as it does in headless Chromium, every page gets `FAKE_LOCK` from `tests/web/browser/conftest.py`: an init script that keeps the lock in a field and records whether each request came with a user activation, which the replay test asserts.

**Frame checks.** The Python side replays the frames the page built, so a defect in turning DOM events into frames (a key's case, a missing `just`) changes both traces alike and `compare` cannot see it. The test asserts the frames themselves: the tick after `keyboard.down("w")` has `w` in `just` and `keys`, Shift+W gives `just == ["w"]`, the tick after the resize has `vp == [640, 480]`. In `test_web_crowd_page.py`, the Button's press is a `pointer_down` routed to the Button and a click, its click event is delivered the frame after and the throw it emits the frame after that, each move of a Slider's drag is routed to it with its fraction rising past 1.0, the clock never runs backwards, and an arrow key's change and commit come the frame after the key. Keep such checks for every input path you add. The material check likewise finds the batch whose drawn instance sits at the entity's Transform row (`BATCHES_OF` in `tests/web/browser/conftest.py`), rather than trusting the store's Material column. The scene has no meshes before the first tick: batches are built at the first sync.

**CI** (`.github/workflows/test.yml`) runs three jobs: `test` (the whole suite on the offscreen backend, with lavapipe so the render tests run instead of skipping, and failing on a no-adapter skip), `web` (`tests/web` outside `browser/`, with lavapipe so the startup device exists) and `browser` (lavapipe, the `browser` group and Playwright's Chromium, `tests/web/browser`). `web` and `browser` fail on any skip, so a missing tool shows as a red job, not a quiet pass. Read the `browser` job's log for the lock-probe line.
