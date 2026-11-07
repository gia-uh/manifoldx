// The engine surface the transpiled modules call, the state loader and the frame step.

import { callable_, str_, vnorm32 } from "./pyrt.js";

// ---------------------------------------------------------------- state

// The class "module.Class" names among the transpiled modules ({name: namespace}).
function classOf(modules, qualname) {
  for (let i = qualname.lastIndexOf("."); i > 0; i = qualname.lastIndexOf(".", i - 1)) {
    const mod = modules[qualname.slice(0, i)];
    if (mod !== undefined) {
      const cls = mod[qualname.slice(i + 1)];
      if (typeof cls !== "function") throw new Error(`the state names ${qualname}, which no module exports`);
      return cls;
    }
  }
  throw new Error(`the state names ${qualname}, whose module was not given`);
}

function moduleOf(modules, name) {
  const mod = modules[name];
  if (mod === undefined) throw new Error(`the state names the module ${name}, which was not given`);
  return mod;
}

const FLOATS = { nan: NaN, inf: Infinity, "-inf": -Infinity };

// loadState(state, modules, resolve): build the objects snapshot() wrote and return its
// roots, [{kind, label, event, fn}]. modules maps each module name to its namespace;
// resolve(kind, value) turns a handle into the runtime's object: kind "engine" (value
// "Engine", "input", "camera" or "gui"), "material" (an id), "entity" (an index) or
// "widget" (an index). An object is built without its constructor, from its class's
// prototype; an object of a class with __call__ is callable; a shared object stays one.
export function loadState(state, modules, resolve) {
  return load(state, modules, resolve).roots;
}

// loadState's work: {roots, state}, state being engine.state as a Map (undefined when the
// snapshot has none).
function load(state, modules, resolve) {
  const table = state.table;
  const built = table.map((e) => {
    if (Array.isArray(e)) return [];
    if ("__dict" in e) return new Map();
    if ("__set" in e) return new Set();
    if ("__f32" in e) return Float32Array.from(e.__f32, value);
    if ("__f64" in e) return Float64Array.from(e.__f64, value);
    if ("__type" in e) {
      const proto = classOf(modules, e.__type).prototype;
      const obj = Object.create(proto);
      return typeof proto.__call__ === "function" ? callable_(obj) : obj;
    }
    throw new Error(`a state entry of unknown form: ${JSON.stringify(e).slice(0, 80)}`);
  });

  function value(v) {
    if (v === null || typeof v !== "object") return v;
    if (Array.isArray(v)) return v.map(value);
    if ("__ref" in v) return built[v.__ref];
    if ("__float" in v) return FLOATS[v.__float];
    if ("__engine" in v) return resolve("engine", v.__engine);
    if ("__material" in v) return resolve("material", v.__material);
    if ("__entity" in v) return resolve("entity", v.__entity);
    if ("__widget" in v) return resolve("widget", v.__widget);
    if ("__function" in v) {
      const at = v.__function.lastIndexOf(":");
      const f = moduleOf(modules, v.__function.slice(0, at))[v.__function.slice(at + 1)];
      if (typeof f !== "function") throw new Error(`the state names ${v.__function}, which is not exported`);
      return f;
    }
    if ("__bound" in v) {
      const obj = value(v.__bound);
      return obj[v.name].bind(obj);
    }
    if ("__closure" in v) {
      const [col, line, ...rest] = v.__closure.split(":").reverse();
      const name = rest.reverse().join(":");
      const factory = moduleOf(modules, name)[`__closure_${line}_${col}`];
      if (typeof factory !== "function") throw new Error(`the state names the closure ${v.__closure}, which is not exported`);
      const cells = Object.create(null);
      for (const [k, x] of Object.entries(v.cells)) cells[k] = value(x);
      return factory(cells, v.defaults.map(value));
    }
    throw new Error(`a state value of unknown form: ${JSON.stringify(v).slice(0, 80)}`);
  }

  // The shells exist first, so a reference to an object later in the table finds it.
  table.forEach((e, i) => {
    const obj = built[i];
    if (Array.isArray(e)) for (const x of e) obj.push(value(x));
    else if ("__dict" in e) for (const [k, x] of e.__dict) obj.set(k, value(x));
    else if ("__set" in e) for (const x of e.__set) obj.add(value(x));
    else if ("__type" in e) {
      for (const [k, x] of Object.entries(e.fields)) {
        Object.defineProperty(obj, k, { value: value(x), writable: true, enumerable: true, configurable: true });
      }
    }
  });

  for (const [name, globals] of Object.entries(state.globals ?? {})) {
    const values = Object.create(null);
    for (const [k, x] of Object.entries(globals)) values[k] = value(x);
    moduleOf(modules, name).__set_globals(values);
  }
  return {
    roots: state.roots.map((r) => ({ kind: r.kind, label: r.label, event: r.event, fn: value(r.fn) })),
    state: state.state === undefined ? undefined : value(state.state),
  };
}

// loadApp(state, modules, options): the app as an export left it, {engine, roots}, ready for
// step(). The engine is createEngine(state.engine) with options over it (options.hooks are the
// browser's scene hooks, see createEngine); engine.state is the
// snapshot's, set once the state is loaded, because its handles resolve through the engine.
export function loadApp(state, modules, options = {}) {
  const engine = createEngine({ ...state.engine, gui: state.gui ?? null, ...options });
  const loaded = load(state, modules, engine.resolve);
  if (loaded.state !== undefined) engine.state = loaded.state;
  return { engine, roots: loaded.roots };
}

// ---------------------------------------------------------------- the engine, headless

// One name per physical key, as manifoldx.input._key_name: a letter lowercase, the space
// bar "Space".
const keyName = (k) => (k === " " ? "Space" : k.length === 1 ? k.toLowerCase() : k);

// engine.input, fed one frame at a time: setFrame({keys, just, d: [dx, dy], buttons,
// clicks, vp: [w, h], up, released, pos: [x, y], wheel: [dx, dy], mods}) replaces what the
// last frame said. keys and just are key names (pressed, and pressed this frame), buttons and
// clicks mouse buttons the same way. The last five are optional and read as empty when
// absent: up and released are the keys and buttons released this frame, pos the pointer
// position in CSS pixels from the canvas's top-left, wheel the wheel's motion, mods the held
// modifiers among Alt, Control, Meta and Shift, sorted.
export class ScriptInput {
  constructor(onResize = null) {
    this._keys = new Set();
    this._just = new Set();
    this._buttons = new Set();
    this._clicks = new Set();
    this._delta = [0, 0];
    this._up = new Set();
    this._released = new Set();
    this._pos = [0, 0];
    this._wheel = [0, 0];
    this._mods = [];
    this._viewport = [0, 0];
    this._captured = false;
    this._onResize = onResize;
    this._log = null;
    this._capture = () => true;
  }

  setFrame(frame) {
    this._keys = new Set(frame.keys.map(keyName));
    this._just = new Set(frame.just.map(keyName));
    this._buttons = new Set(frame.buttons);
    this._clicks = new Set(frame.clicks);
    this._delta = [frame.d[0], frame.d[1]];
    this._up = new Set((frame.up ?? []).map(keyName));
    this._released = new Set(frame.released ?? []);
    const pos = frame.pos ?? [0, 0], wheel = frame.wheel ?? [0, 0];
    this._pos = [pos[0], pos[1]];
    this._wheel = [wheel[0], wheel[1]];
    this._mods = [...(frame.mods ?? [])];
    const [w, h] = frame.vp;
    if (w !== this._viewport[0] || h !== this._viewport[1]) {
      this._viewport = [w, h];
      if (this._onResize) this._onResize(w, h);
    }
  }

  get pressed_keys() { return new Set(this._keys); }
  is_pressed(key) { return this._keys.has(keyName(key)); }
  just_pressed(key) { return this._just.has(keyName(key)); }
  get mouse_delta() { return this._delta; }
  is_mouse_pressed(button) { return this._buttons.has(button); }
  just_mouse_pressed(button) { return this._clicks.has(button); }
  get pressed_buttons() { return [...this._buttons].sort((a, b) => a - b); }
  just_released(key) { return this._up.has(keyName(key)); }
  just_mouse_released(button) { return this._released.has(button); }
  get modifiers() { return this._mods; }
  get mouse_pos() { return this._pos; }
  get wheel_delta() { return this._wheel; }
  get viewport_size() { return this._viewport; }
  get pointer_captured() { return this._captured; }

  // The capture hook decides (headless, it always takes). When it takes, the flag follows
  // and the log records it; when it refuses, nothing changes.
  capture_pointer(on) {
    if (!this._capture(Boolean(on))) return false;
    this._captured = Boolean(on);
    if (this._log) this._log.push(["capture", this._captured]);
    return true;
  }

  // The browser ended the capture without the app asking (Escape, a refused lock):
  // pointer_captured reads false from now.
  lockLost() {
    this._captured = false;
  }
}

// engine.camera: position, target and up are float32, as manifoldx.camera.Camera holds
// them; fov is the vertical field of view in degrees.
export class Camera {
  constructor({ position, target, fov = 60, up = [0, 1, 0], near = 0.1, far = 1000.0 }) {
    this.position = Float32Array.from(position);
    this.target = Float32Array.from(target);
    this.fov = fov;
    this.up = Float32Array.from(up);
    this.near = near;
    this.far = far;
  }

  // numpy's (target - position) / norm(...) in float32: subtract, norm and divide each
  // rounded to float32.
  get_forward() {
    const d = new Float32Array(3);
    for (let i = 0; i < 3; i++) d[i] = this.target[i] - this.position[i];
    const n = vnorm32(d);
    return d.map((x) => Math.fround(x / n));
  }
}

// createEngine({w, h, camera: {position, target, fov, up, near, far}, gui, materials, log,
// state, hooks}): the engine object transpiled code reaches. gui is state.json's "gui": each
// widget it names becomes engine.gui.widgets[n] with the fields reached code may write (a
// Panel's offset, a Text's text, every widget's style_overrides), which the page applies. materials maps a material id to the
// renderer's material (none, headless); log receives ["material", entity, material_id] and
// ["capture", on] as reached code asks for them, the trace's record; state is engine.state, a
// Map as every dict. hooks are where the browser acts, both optional:
// setMaterial(entityIndex, materialId) after the log, and capture(on) -> boolean, whether the
// capture took (headless: a no-op and () => true). resolve(kind, value) is loadState's
// resolver for this engine.
export function createEngine({ w, h, camera, gui = null, materials = {}, log = [], state = new Map(), hooks = {} }) {
  const { setMaterial = () => {}, capture = () => true } = hooks;
  const engine = { w, h, state, log };
  engine.input = new ScriptInput((nw, nh) => { engine.w = nw; engine.h = nh; });
  engine.input._log = log;
  engine.input._capture = capture;
  engine.input._viewport = [w, h];
  engine.camera = new Camera(camera);
  engine.gui = { pointer_over_gui: false, widgets: gui === null ? [] : guiWidgets(gui) };

  const entities = new Map(), mats = new Map();
  engine.set_material = (entity, material) => {
    const index = typeof entity === "number" ? entity : entity.index;
    log.push(["material", index, material.id]);
    setMaterial(index, material.id);
  };
  engine.resolve = (kind, value) => {
    if (kind === "engine") {
      if (value === "Engine") return engine;
      if (value in engine) return engine[value];
      throw new Error(`the state names engine.${value}, which the runtime does not have`);
    }
    if (kind === "entity") {
      if (!entities.has(value)) entities.set(value, { index: value });
      return entities.get(value);
    }
    if (kind === "material") {
      if (!mats.has(value)) mats.set(value, { id: value, material: materials[value] ?? null });
      return mats.get(value);
    }
    if (kind === "widget") {
      engine.gui.widgets[value] ??= { index: value };
      return engine.gui.widgets[value];
    }
    throw new Error(`a state handle of unknown kind: ${kind}`);
  };
  return engine;
}

// The widget objects of a gui spec, by their index in the snapshot's walk.
function guiWidgets(spec) {
  const widgets = [];
  const walk = (node) => {
    const w = { index: node.widget, style_overrides: new Map(Object.entries(node.overrides)) };
    if (node.kind === "Panel") w.offset = [...node.offset];
    if (node.kind === "Text") w.text = node.text;
    widgets[node.widget] = w;
    for (const c of node.children ?? []) walk(c);
  };
  spec.panels.forEach(walk);
  return widgets;
}

// ---------------------------------------------------------------- frames

// app: {engine, roots}, roots as loadState returns them. One frame: every system, in the
// order the app registered them, called as the engine calls them, with (query, dt).
export function step(app, dt) {
  for (const r of app.roots) if (r.kind === "system") r.fn(null, dt);
}

// What each ValueDisplay shows: its getter's value, as str() makes it.
export function guiValues(app) {
  return app.roots
    .filter((r) => r.kind === "getter")
    .map((r) => {
      const v = r.fn();
      return typeof v === "string" ? v : str_(v);
    });
}

// ---------------------------------------------------------------- the trace

// JSON as Python's json writes the same values: a Map (a dict) as an object, typed arrays and
// Sets as arrays.
export function toJSON(value) {
  return JSON.stringify(value, (k, v) => {
    if (v instanceof Map) return Object.fromEntries(v);
    if (ArrayBuffer.isView(v)) return Array.from(v);
    if (v instanceof Set) return [...v];
    return v;
  });
}

// path's value in app ({engine, roots, modules}): its first name is a global of the app's
// main module or, when it has none by that name, a key of engine.state; then attributes and
// dict keys.
export function pathValue(app, path) {
  const [head, ...rest] = path.split(".");
  const main = app.modules.__main__;
  let v;
  if (main !== undefined && head in main) v = main[head];
  else if (app.engine.state.has(head)) v = app.engine.state.get(head);
  else throw new Error(`the path ${path} names nothing at ${head}`);
  for (const name of rest) {
    if (v instanceof Map ? !v.has(name) : v === null || typeof v !== "object" || !(name in v)) {
      throw new Error(`the path ${path} names nothing at ${name}`);
    }
    v = v instanceof Map ? v.get(name) : v[name];
  }
  return v;
}

// The trace entry of the frame just run, as plain JSON values (the form is
// manifoldx.web.replay's): {camera, gui, log, state, paths}, log being what the engine's log
// gained since logStart.
export function traceEntry(app, paths, logStart) {
  const { engine } = app;
  const entry = {
    camera: { position: engine.camera.position, target: engine.camera.target },
    gui: guiValues(app),
    log: engine.log.slice(logStart),
    state: engine.state,
    paths: Object.fromEntries(paths.map((p) => [p, pathValue(app, p)])),
  };
  return JSON.parse(toJSON(entry));
}

// ---------------------------------------------------------------- the state, both ways

// A number as the snapshot writes it: NaN and the infinities as {"__float": ...}.
const number = (x) =>
  Number.isNaN(x) ? { __float: "nan" } : x === Infinity ? { __float: "inf" } : x === -Infinity ? { __float: "-inf" } : x;

// A handle's name, as the desktop's repr fallback shows it; null when v is not a handle.
function handleName(v) {
  if (v instanceof Camera) return "<camera>";
  if (v instanceof ScriptInput) return "<input>";
  const proto = Object.getPrototypeOf(v);
  if (proto !== Object.prototype && proto !== null) return null;
  if ("resolve" in v) return "<engine>";
  if ("pointer_over_gui" in v) return "<gui>";
  if ("style_overrides" in v) return `<widget ${v.index}>`;
  if ("id" in v) return `<material ${v.id}>`;
  if ("index" in v) return `<entity ${v.index}>`;
  return "<an object>";
}

// encodeState(value): engine.state (a Map) as the snapshot's value forms, in plain JSON, for
// Python to decode: {"__dict": [[k, v], ...]}, {"__set": [...]}, {"__f32": [...]},
// {"__f64": [...]}, {"__float": ...}, arrays as arrays. What is not plain data (an object of a
// transpiled class, a handle, a function, a cycle) is {"__skip": what}.
export function encodeState(value) {
  const path = new Set(); // the containers above the value being written
  const enc = (v) => {
    if (typeof v === "number") return number(v);
    if (v === null || v === undefined) return null;
    if (typeof v === "function") return { __skip: "a function" };
    if (typeof v !== "object") return v;
    if (path.has(v)) return { __skip: "a cycle" };
    if (v instanceof Float32Array) return { __f32: Array.from(v, number) };
    if (v instanceof Float64Array) return { __f64: Array.from(v, number) };
    const inside = (f) => {
      path.add(v);
      try {
        return f();
      } finally {
        path.delete(v);
      }
    };
    if (v instanceof Map) return inside(() => ({ __dict: [...v].map(([k, x]) => [k, enc(x)]) }));
    if (v instanceof Set) return inside(() => ({ __set: [...v].map(enc) }));
    if (Array.isArray(v)) return inside(() => v.map(enc));
    const handle = handleName(v);
    if (handle !== null) return { __skip: handle };
    return { __skip: `an object of ${Object.getPrototypeOf(v).constructor.name}` };
  };
  return enc(value);
}

// fromJSON(value): a JSON value (an agent's patch or event payload) as runtime values: every
// object a Map, recursively.
export function fromJSON(value) {
  if (Array.isArray(value)) return value.map(fromJSON);
  if (value === null || typeof value !== "object") return value;
  return new Map(Object.entries(value).map(([k, v]) => [k, fromJSON(v)]));
}

// mergePatch(target, patch): RFC 7386 merge patch on Maps, the patch through fromJSON first: a
// Map merges key by key (in place when the target is a Map), null deletes, anything else
// replaces. Returns the merged value.
export function mergePatch(target, patch) {
  if (!(patch instanceof Map)) return patch;
  const out = target instanceof Map ? target : new Map();
  for (const [k, v] of patch) {
    if (v === null) out.delete(k);
    else out.set(k, mergePatch(out.get(k), v));
  }
  return out;
}

// stateView(value): engine.state as GET /api/state shows it, plain JSON: a Map as an object,
// typed arrays and Sets as arrays, what encodeState skips as its what string.
export function stateView(value) {
  const view = (v) => {
    if (v === null || typeof v !== "object") return v;
    if ("__skip" in v) return v.__skip;
    if ("__float" in v) return FLOATS[v.__float];
    if ("__dict" in v) return Object.fromEntries(v.__dict.map(([k, x]) => [k, view(x)]));
    if ("__set" in v) return v.__set.map(view);
    if ("__f32" in v) return v.__f32.map(view);
    if ("__f64" in v) return v.__f64.map(view);
    return v.map(view);
  };
  return view(encodeState(value));
}
