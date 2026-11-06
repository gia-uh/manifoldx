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
// step(). The engine is createEngine(state.engine) with options over it; engine.state is the
// snapshot's, set once the state is loaded, because its handles resolve through the engine.
export function loadApp(state, modules, options = {}) {
  const engine = createEngine({ ...state.engine, ...options });
  const loaded = load(state, modules, engine.resolve);
  if (loaded.state !== undefined) engine.state = loaded.state;
  return { engine, roots: loaded.roots };
}

// ---------------------------------------------------------------- the engine, headless

// One name per physical key, as manifoldx.input._key_name: a letter lowercase, the space
// bar "Space".
const keyName = (k) => (k === " " ? "Space" : k.length === 1 ? k.toLowerCase() : k);

// engine.input, fed one frame at a time: setFrame({keys, just, d: [dx, dy], buttons,
// clicks, vp: [w, h]}) replaces what the last frame said. keys and just are key names
// (pressed, and pressed this frame), buttons and clicks mouse buttons the same way.
export class ScriptInput {
  constructor(onResize = null) {
    this._keys = new Set();
    this._just = new Set();
    this._buttons = new Set();
    this._clicks = new Set();
    this._delta = [0, 0];
    this._viewport = [0, 0];
    this._captured = false;
    this._onResize = onResize;
    this._log = null;
  }

  setFrame(frame) {
    this._keys = new Set(frame.keys.map(keyName));
    this._just = new Set(frame.just.map(keyName));
    this._buttons = new Set(frame.buttons);
    this._clicks = new Set(frame.clicks);
    this._delta = [frame.d[0], frame.d[1]];
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
  get viewport_size() { return this._viewport; }
  get pointer_captured() { return this._captured; }

  // Headless, the capture always takes: the flag follows and the log records it.
  capture_pointer(on) {
    this._captured = Boolean(on);
    if (this._log) this._log.push(["capture", this._captured]);
    return true;
  }
}

// engine.camera: position and target are float32, as manifoldx.camera.Camera holds them.
export class Camera {
  constructor({ position, target, near = 0.1, far = 1000.0 }) {
    this.position = Float32Array.from(position);
    this.target = Float32Array.from(target);
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

// createEngine({w, h, camera: {position, target, near, far}, materials, log, state}):
// the engine object transpiled code reaches. materials maps a material id to the
// renderer's material (none, headless); log receives ["material", entity, material_id] and
// ["capture", on] as reached code asks for them; state is engine.state, a Map as every dict.
// resolve(kind, value) is loadState's resolver for this engine.
export function createEngine({ w, h, camera, materials = {}, log = [], state = new Map() }) {
  const engine = { w, h, state, log };
  engine.input = new ScriptInput((nw, nh) => { engine.w = nw; engine.h = nh; });
  engine.input._log = log;
  engine.input._viewport = [w, h];
  engine.camera = new Camera(camera);
  engine.gui = { pointer_over_gui: false, widgets: [] };

  const entities = new Map(), mats = new Map();
  engine.set_material = (entity, material) => {
    log.push(["material", typeof entity === "number" ? entity : entity.index, material.id]);
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
