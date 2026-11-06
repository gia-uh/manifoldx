---
when: exporting an app with `manifoldx export --web`, fixing an error it reports, annotating app code so it transpiles, or checking with a replay that the browser runs an app as Python does
triggers: ["export --web", "web export", "transpile", "transpiler", "browser", "manifoldx.web", "replay", "replay_node", "outside the subset", "cannot cross to the browser", "annotate"]
---

# Exporting an app to the web with `manifoldx export --web`

## When to use this

Reach for this when you export an app for the browser, when the export refuses an app and you have to change its code, or when you want proof that the exported app behaves as the Python one does. The code is `src/manifoldx/web/`; the tests in `tests/web/` show every behaviour below, and `examples/web_walk.py` is an app that exports and replays.

What exists today (plan 1 of the web target) is the transpiler, the state snapshot, a headless JavaScript runtime and a replay harness that runs the export in Node. **There is no browser page yet**: nothing renders the scene or the GUI, and nothing feeds browser input to the app. The export folder loads in Node only. Rendering, the HTML GUI and browser input are plan 2; `run --web` comes after. Spec: `.knowledge/analysis/2026-11-06-web-target-design.html`. Plan: `.knowledge/plans/2026-11-06-web-target-plan-1-transpiler-and-replay.html`.

## Running it

```bash
uv run manifoldx export --web app.py out/                  # a script
uv run manifoldx export --web -m mygame.main out/          # a module, from the folder that holds the package
uv run manifoldx export --web app.py out/ -- --level 3     # everything after -- goes to the app's sys.argv
```

The app runs as `python app.py` would, until it calls `engine.run()` (or `engine.cli()`), which returns at once. If the engine has startup handlers, the export opens an offscreen device and fires them (an `async def` startup handler is awaited), so `load_texture` and other device work runs. Everything up to here is setup and stays in Python. Then the export finds the roots, types and reaches the code, snapshots the state and transpiles.

Exit codes: 0 written; 1 the app's code or state cannot cross (the error names the file and line, or the path to the value, and **nothing is written**); 2 the app never called `engine.run()`. `--strict` belongs to the `.glb` export and is refused with `--web`. The export does not need Node; the replay and the tests do.

## What the folder holds

```text
out/
  app/<module>.js, .js.map   one ES module per app module, with a source map to the .py; the script (or -m module) is app/__main__.js
  pyrt.js                    the Python runtime the generated code imports as $rt
  manifoldx.js               the headless engine: state loader, input, camera, set_material, step
  replay.mjs                 loads the folder in Node
  state.json                 the snapshot: reached objects, engine.state, the engine's size and camera
  scene.glb                  the scene, as export_gltf writes it, plus scene.export-report.json
  report.json                roots, transpiled functions, setup functions, reached classes and globals, state left out
```

`node out/replay.mjs` prints the roots' labels, `engine.state` and what each ValueDisplay shows, as JSON: the quickest check that the folder loads.

The app's modules are the script (or `-m` module) and every module loaded from its folder (the working directory for `-m`), outside `site-packages` and manifoldx: the files `manifoldx run` watches.

## What reaches the browser

**Roots** are the functions the engine calls (`find_roots` in `reach.py`):

- each system, in registration order (`system <name>` in labels);
- each event handler except `startup` handlers (they ran at export) and manifoldx's own (`on("<event>") <name>`);
- each `ValueDisplay`'s getter, walking `engine.gui`'s panels and their children (`ValueDisplay.getter <name>`).

A root must be a plain `def` or a lambda of an app module. A bound method root (`engine.system(viewer.tick)`) brings its object into the snapshot; a closure root brings its cells and defaults.

**Reach** follows calls and attribute reads through types, from the roots. A method is reached when reached code calls it on a value whose type allows its class, never because its name matches. Of each reached class, only the fields reached code reads are written. Every function not reached is setup: it ran in Python and is not in the JS. `report.json` lists both (`transpiled`, `setup`), so read it when something you expected in the browser is missing.

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

What is in: functions with positional and keyword arguments and defaults, nested functions and lambdas (closures), classes (dataclasses with `field(default_factory=...)`, `@property`, `__call__`, a value that may be one of several app classes), `if`/`for`/`while`/`break`/`continue`, comprehensions, f-strings with or without a format spec, `%` with Python's sign, `and`/`or` on any values, lists, tuples, dicts with str keys, sets, `math` (`sin cos tan asin acos atan atan2 sqrt exp log floor ceil fabs hypot radians degrees`, `pi e tau inf nan`), numpy 1-d float vectors (`np.array`, `np.zeros`, `np.arange`, `np.linalg.norm`, `+ - * /` and in-place `+=` that aliases see; float32 stays float32), `textwrap.wrap(s, width)`, and the builtins `len str bool int float abs min max any all range list tuple set next hasattr isinstance`. Strings count code points, as Python does.

What is out, among others: `try`, `raise`, `with`, `global`, `nonlocal`, `del`, `assert`, `async`/`await`, `yield`, `match`, `:=`, `for`/`else` and `while`/`else`, imports inside functions, `*args`/`**kwargs`, slices with a step or assigned to, dict comprehensions, imports of libraries other than `math`, `numpy` and `textwrap`, and a system that uses its query (component queries in the browser are not done yet). A builtin or library function not listed above fails as `app.py:5: enumerate() is outside the subset (reached from system tick)`.

There is no `global`: state that frame code reassigns lives in an object's attribute (`rig.mode = "fly"`) or in `engine.state` (`engine.state["selected"] = box.name`), never in a module-level name. Module globals are fine to read and to mutate in place.

Read and write `engine.state` with subscripts, as `examples/web_walk.py` does. `engine.state.key` exports without an error but does not work in the browser: the runtime's engine.state is a JS Map, and the attribute reads nothing.

## Annotating

The transpiler types values from annotations, from the live objects setup left (each app class's fields, a union over its instances), and from call sites, solved to a fixed point. When an operator needs a type it could not find, the export stops and names what to annotate:

```text
app.py:6: `a + b` with an operand of unknown type (annotate `a`) is outside the subset (reached from system tick -> __main__.add)
```

The operations that need a type are `+`, `*`, `%` (and their `+=` forms), `==` and `!=` unless the other side is a known number, bool or None, f-string fields with no format spec and `str()` (JS cannot tell `3.0` from `3`), method calls, slicing, `len`/`abs`/`int`/`float`/`hasattr` and keyword arguments. Annotate the name it gives, and nothing else:

- a parameter: `def add(a: float, b: float)`;
- a local, typically a read of `engine.state`, which always types as unknown: `name: str | None = engine.state["selected"]`;
- a field, where `__init__` sets it: `self.parts: list[Part] = []`.

The annotations understood are `int`, `float`, `bool`, `str`, `None`, app classes, `X | Y`, `Optional[X]`, `list[X]`, `set[X]`, `dict[K, V]`, `tuple[...]` and `np.ndarray` (float64; build float32 vectors with `np.array(x, np.float32)`). An annotation wins over a live type. Other annotations count as unknown.

## The engine from transpiled code

Reached code may use exactly what `RUNTIME_SURFACE` in `src/manifoldx/web/types.py` lists; anything else of a manifoldx object types as unknown:

- `engine.input`, `engine.camera`, `engine.w`, `engine.h`, `engine.set_material(entity, material)`, and `engine.state` (subscripts, see above);
- `engine.input`: `pressed_keys`, `modifiers`, `mouse_pos`, `mouse_delta`, `wheel_delta`, `pressed_buttons`, `viewport_size`, `pointer_captured`, `is_pressed(key)`, `just_pressed(key)`, `just_released(key)`, `is_mouse_pressed(button)`, `just_mouse_pressed(button)`, `just_mouse_released(button)`, `capture_pointer(on)`;
- `engine.camera`: `position` and `target` (float32 vectors), `near`, `far`, `get_forward()`.

Spawning, destroying, component queries, transforms and events are not there yet. Do the scene's building in a `startup` handler or at module level: that is setup.

## Replaying: proof that both sides agree

`manifoldx.web.replay` runs one script of input frames through the app in Python and through its export in Node, traces the same values on both sides, and reports the first frame where they differ. Use it for every app you export; it is how `tests/web/test_replay.py` checks `examples/web_walk.py`.

```python
from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

frames = [{"keys": ["w"], "just": [], "d": [4.0, 0.0], "buttons": [], "clicks": [], "vp": [800, 500]}] * 60
paths = ["rig.pos", "rig.yaw", "selected"]
assert export_web(None, "app.py", out, []) == 0
mismatch = compare(replay_python(None, "app.py", frames, paths), replay_node(out, frames, paths))
assert mismatch is None, str(mismatch)          # "frame 41, paths.rig.yaw: 0.12 != 0.13"
```

A frame is what `engine.input` reports for one frame: `keys` held, `just` pressed this frame, `d` the mouse motion, `buttons` held, `clicks` pressed this frame, `vp` the viewport size (a new size resizes the engine). Releases, modifiers, pointer position and wheel always read as nothing. `capture_pointer` always succeeds, as in a headless browser.

Each side reads a frame, runs the systems once with dt 1/60, and traces the camera's position and target, what each ValueDisplay shows, the `["material", entity, material_id]` and `["capture", on]` calls of that frame, all of `engine.state`, and the values of `paths`. Event handlers are transpiled but neither side fires them in a replay.

A path starts at a global of the app's main module or, when it has no global by that name, at a key of `engine.state`, then goes through attributes and dict keys (`rig.pos`, `selected`). The global must be one reached code reads: the JS module has only those.

`compare` matches numbers by value (JSON does not tell `1` from `1.0`), never a number against a bool, and floats within 2^-30 of the larger one (2^-40 near zero). The tolerance is there because Python's `math` calls the C library and Node has its own: V8's `sin` and `cos` differ from glibc's in the last bit (62 of 2226 values in the web_walk replay). 2^-30 is 64 times finer than a float32 step, so a value that skipped float32 rounding still mismatches. `compare(a, b, rel=0)` compares bit for bit.

### Writing a script

- Generate frames from a seeded `random.Random` in the test file, and keep the frames as a JSON fixture so a change to the generator shows in the diff. `tests/web/test_replay.py` does this; `uv run pytest tests/web/test_replay.py --regen` rewrites its fixture. `--regen` is defined in `tests/web/conftest.py`, so pytest knows it only when you give it `tests/web` or a file in it; `uv run pytest --regen` from the root fails with `unrecognized arguments: --regen`.
- Exercise every branch you care about: clicks, keys that toggle modes, a resize, a drag. Then assert the trace shows it (web_walk's test asserts at least three distinct picks and a material change), or a script that never picks proves nothing about picking.
- Add breaking tests: damage the export one way (patch the generated JS, drop an object from `state.json`) and assert `compare` returns a mismatch. A replay that passes with a broken export is not checking anything.
- Without Node on the PATH, everything under `tests/web` skips with "node not found".

## Tests

```bash
uv run pytest tests/web -q                       # parity, reach, snapshot, export, replay; needs node
uv run pytest tests/web/test_replay.py -q        # the web_walk replay alone
```

Each construct of the subset has a parity test that runs a `main()` in Python and its transpiled form in Node (`tests/web/parity.py`, `run_both`) and compares the results. Add one when you widen the subset.
