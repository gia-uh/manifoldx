// The engine surface the transpiled modules call, the state loader and the frame step.

import { callable_ } from "./pyrt.js";

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
  return state.roots.map((r) => ({ kind: r.kind, label: r.label, event: r.event, fn: value(r.fn) }));
}
