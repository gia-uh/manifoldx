// The engine surface the transpiled modules call, the state loader and the frame step.

import { NdArray, Random, callable_, nd, setRandom, str_, vnorm32 } from "./pyrt.js";

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
// "Engine", "input", "camera" or "gui"), "material" (an id), "entity" (an index),
// "widget" (an index) or "geometry" (an id in the engine's geometry registry). An object is built without its constructor, from its class's
// prototype; an object of a class with __call__ is callable; a shared object stays one. A
// random.Random is pyrt's, and the state's "random" becomes the random module's ($rt.random).
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
    if ("__random" in e) return Random.fromState(e.__random);
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
    if ("__geometry" in v) return resolve("geometry", v.__geometry);
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

  if (state.random !== undefined) setRandom(Random.fromState(state.random));
  for (const [name, globals] of Object.entries(state.globals ?? {})) {
    const values = Object.create(null);
    for (const [k, x] of Object.entries(globals)) values[k] = value(x);
    moduleOf(modules, name).__set_globals(values);
  }
  return {
    roots: state.roots.map((r) => ({
      kind: r.kind,
      label: r.label,
      event: r.event,
      components: r.components ?? null,
      fn: value(r.fn),
    })),
    state: state.state === undefined ? undefined : value(state.state),
  };
}

// loadApp(state, modules, options): the app as an export left it, {engine, roots}, ready for
// step(). The engine is createEngine(state.engine) with options over it (options.hooks are the
// browser's scene hooks, see createEngine; options.ecs is ecs.bin's ArrayBuffer); engine.store
// is the Store state.ecs and options.ecs describe (null for a state without one); engine.state
// is the snapshot's, set once the state is loaded, because its handles resolve through the engine.
export function loadApp(state, modules, options = {}) {
  const { ecs, ...rest } = options;
  const engine = createEngine({ ...state.engine, gui: state.gui ?? null, ...rest });
  engine.store = state.ecs === undefined ? null : Store.load(state.ecs, ecs);
  const loaded = load(state, modules, engine.resolve);
  if (loaded.state !== undefined) engine.state = loaded.state;
  return { engine, roots: loaded.roots };
}

// ---------------------------------------------------------------- the entity store

const TYPED = { f4: Float32Array, u4: Uint32Array };

// The page's entity store, as manifoldx.ecs.EntityStore holds it: per component a row-major
// typed array of `cols` columns and a member bit per slot, an alive bit per slot and the free
// list. The arrays hold `capacity` slots and grow by doubling up to maxEntities; a slot at or
// past capacity is dead. Slots, membership and the free list follow EntityStore's rules, so
// the same spawns and destroys give the same indices. version moves on every spawn and
// destroy and on a write to Mesh or Material; dirty is [lo, hi), the slots whose Transform
// rows changed since takeDirty(), or null.
export class Store {
  // header: state.json's "ecs"; buffer: ecs.bin, laid out by it.
  static load(header, buffer) {
    const store = new Store(header.max_entities, Math.max(header.high_water, 1));
    const hw = header.high_water;
    store.highWater = hw;
    store.free = [...header.free];
    store.alive.set(new Uint8Array(buffer, header.alive[0], hw));
    for (const c of header.components) {
      const Typed = TYPED[c.dtype];
      if (Typed === undefined) throw new Error(`the component ${c.name} has the dtype ${c.dtype}, which the page does not hold`);
      store._layouts.set(c.name, { dtype: c.dtype, cols: c.cols, fields: c.fields, defaults: c.defaults ?? {} });
      const rows = new Typed(store.capacity * c.cols);
      rows.set(new Typed(buffer.slice(c.rows[0], c.rows[0] + c.rows[1])));
      store._columns.set(c.name, rows);
      const members = new Uint8Array(store.capacity);
      members.set(new Uint8Array(buffer, c.members[0], hw));
      store._members.set(c.name, members);
    }
    for (const c of header.spawnable ?? []) store._spawnable.set(c.name, c);
    return store;
  }

  constructor(maxEntities, capacity) {
    this.maxEntities = maxEntities;
    this.capacity = Math.min(capacity, maxEntities);
    this.highWater = 0;
    this.alive = new Uint8Array(this.capacity);
    this.free = [];
    this.version = 0;
    this.dirty = null;
    this._layouts = new Map();
    this._columns = new Map();
    this._members = new Map();
    this._spawnable = new Map(); // layouts registered at their first spawn
    this._queue = [];
  }

  layout(name) { return this._layouts.get(name); }

  // A component reached code spawns that setup never registered gets its columns and member
  // bits now, all zero, as Component.register gives them on the desktop's first spawn.
  register(name) {
    if (this._layouts.has(name)) return;
    const c = this._spawnable.get(name);
    if (c === undefined) throw new Error(`ValueError: Component ${name} not registered`);
    this._layouts.set(name, { dtype: c.dtype, cols: c.cols, fields: c.fields, defaults: c.defaults ?? {} });
    this._columns.set(name, new TYPED[c.dtype](this.capacity * c.cols));
    this._members.set(name, new Uint8Array(this.capacity));
  }
  column(name) { return this._columns.get(name); }
  member(name) { return this._members.get(name); }

  // Room for slot `index`: double the arrays (at most maxEntities), keeping rows and bits.
  _grow(index) {
    if (index < this.capacity) return;
    let cap = this.capacity;
    while (cap <= index) cap = Math.min(cap * 2, this.maxEntities);
    const widen = (a, n) => {
      const b = new a.constructor(n);
      b.set(a);
      return b;
    };
    this.alive = widen(this.alive, cap);
    for (const [name, rows] of this._columns) this._columns.set(name, widen(rows, cap * this._layouts.get(name).cols));
    for (const [name, bits] of this._members) this._members.set(name, widen(bits, cap));
    this.capacity = cap;
  }

  _touch(name, lo, hi) {
    if (name === "Transform") this.dirty = this.dirty === null ? [lo, hi] : [Math.min(this.dirty[0], lo), Math.max(this.dirty[1], hi)];
    else if (name === "Mesh" || name === "Material") this.version++;
  }

  // n entities, the free list first (popped from its end), then the lowest dead slots; rows
  // maps a component name to its n × cols values (or one value for all). The slots get the
  // components given and Transform as members, and lose the others.
  spawn(n, rows = new Map()) {
    const k = Math.min(n, this.free.length);
    const reused = this.free.slice(this.free.length - k).reverse();
    let alive = 0;
    for (let i = 0; i < this.capacity; i++) alive += this.alive[i];
    const dead = this.maxEntities - alive;
    if (dead < n) {
      throw new Error(
        `RuntimeError: Max entities ${this.maxEntities} reached: spawning ${n} with ${dead} free slots; raise it with Engine(max_entities=...)`,
      );
    }
    for (const name of rows.keys()) {
      if (!this._layouts.has(name)) throw new Error(`ValueError: Component ${name} not registered`);
    }
    const indices = new Int32Array(n);
    indices.set(reused);
    const taken = new Set(reused);
    for (let i = 0, j = k; j < n; i++) {
      if (i >= this.capacity || (!this.alive[i] && !taken.has(i))) indices[j++] = i;
    }
    this.free.length -= k;
    if (!n) return indices;
    let lo = Infinity, hi = -1;
    for (const i of indices) {
      lo = Math.min(lo, i);
      hi = Math.max(hi, i + 1);
    }
    this._grow(hi - 1);
    for (const i of indices) this.alive[i] = 1;
    this.highWater = Math.max(this.highWater, hi);
    for (const [name, bits] of this._members) {
      const on = rows.has(name) || name === "Transform" ? 1 : 0;
      for (const i of indices) bits[i] = on;
    }
    for (const [name, values] of rows) {
      const { cols } = this._layouts.get(name), col = this._columns.get(name);
      if (values.length !== 1 && values.length !== n * cols) {
        throw new Error(`ValueError: Component ${name}: expected ${n} rows, got ${values.length / cols}`);
      }
      indices.forEach((i, r) => {
        if (values.length === 1) col.fill(values[0], i * cols, (i + 1) * cols);
        else for (let c = 0; c < cols; c++) col[i * cols + c] = values[r * cols + c];
      });
      this._touch(name, lo, hi);
    }
    this.version++;
    return indices;
  }

  // Each index once and only if alive, then to the free list, ascending.
  destroy(indices) {
    const gone = [...new Set(Array.from(indices, Number))]
      .sort((a, b) => a - b)
      .filter((i) => i < this.capacity && this.alive[i]);
    if (!gone.length) return;
    for (const i of gone) this.alive[i] = 0;
    this.free.push(...gone);
    this.version++;
  }

  // The alive slots below highWater that have every named component, ascending.
  query(names) {
    const bits = names.map((n) => this._members.get(n));
    if (bits.includes(undefined)) return new Int32Array(0);
    const out = [];
    for (let i = 0; i < this.highWater; i++) {
      if (this.alive[i] && bits.every((b) => b[i])) out.push(i);
    }
    return Int32Array.from(out);
  }

  // values into columns [col, col + values.length) of slot index's row, now.
  write(name, index, col, values) {
    const { cols } = this._layouts.get(name);
    this._columns.get(name).set(values, index * cols + col);
    this._touch(name, index, index + 1);
  }

  // At the flush: data (one row of colEnd - colStart values per index, or one row for all)
  // into those columns of each index's row, as CommandBuffer's UPDATE_COMPONENT writes.
  queueUpdate(name, indices, colStart, colEnd, data) {
    this._queue.push(["update", name, Array.from(indices), colStart, colEnd, data]);
  }

  queueDestroy(indices) {
    this._queue.push(["destroy", Array.from(indices)]);
  }

  // The queue applied in order, then cleared, as the engine executes its commands after the
  // systems.
  flush() {
    const queue = this._queue;
    this._queue = [];
    for (const [op, ...args] of queue) {
      if (op === "destroy") {
        this.destroy(args[0]);
        continue;
      }
      const [name, indices, colStart, colEnd, data] = args;
      const { cols } = this._layouts.get(name), col = this._columns.get(name);
      const w = colEnd - colStart, row = data.length === w;
      let lo = Infinity, hi = -1;
      indices.forEach((i, r) => {
        if (i >= this.capacity) return; // a dead slot
        for (let c = 0; c < w; c++) col[i * cols + colStart + c] = data.length === 1 ? data[0] : data[(row ? 0 : r * w) + c];
        lo = Math.min(lo, i);
        hi = Math.max(hi, i + 1);
      });
      if (hi > 0) this._touch(name, lo, hi);
    }
  }

  takeDirty() {
    const d = this.dirty;
    this.dirty = null;
    return d;
  }
}

// ---------------------------------------------------------------- queries

// A value as float32 values, as np.asarray(v, np.float32) makes them: a number, a typed array,
// an NdArray or nested arrays (a tuple).
function toF32(v) {
  if (typeof v === "number") return Float32Array.of(v);
  if (v instanceof NdArray) return Float32Array.from(v.data);
  if (ArrayBuffer.isView(v)) return Float32Array.from(v);
  if (Array.isArray(v)) return Float32Array.from(v.flat(Infinity));
  throw new Error(`TypeError: a ${typeof v} cannot be a component's float32 values`);
}

// A system's query, as the engine's ComponentView: the alive entities that have every named
// component when the system runs, ascending. q.get(name) is q[name]; len(q) is q.length, and
// iterating it gives the entity indices.
export class Query {
  constructor(store, names) {
    this.store = store;
    this.indices = store.query(names);
    this.length = this.indices.length;
  }

  get(name) {
    return new Accessor(this.store, name, this.indices);
  }

  [Symbol.iterator]() {
    return Array.from(this.indices)[Symbol.iterator]();
  }
}

// q[X], as ComponentAccessor: .field(name) is q[X].name, a FieldView.
export class Accessor {
  constructor(store, component, indices) {
    this.store = store;
    this.component = component;
    this.indices = indices;
  }

  field(name) {
    const layout = this.store.layout(this.component);
    if (layout === undefined) throw new Error(`ValueError: Component ${this.component} not registered`);
    const f = layout.fields[name];
    if (f === undefined) throw new Error(`AttributeError: No field '${name}' in component '${this.component}'`);
    return new FieldView(this.store, this.component, name, this.indices, f[0], f[0] + f[1]);
  }
}

const F = Math.fround;

// Columns [c0, c1) of a component's rows for a query's entities, as manifoldx.ecs._FieldView:
// data reads the rows now, as a float32 (k, c1 - c0) array; arithmetic casts the other side
// to float32 first and gives a float32 array; a comparison gives the entity indices whose
// (one) column holds; set and the in-place forms queue the new rows for the flush, so every
// system this frame reads the old ones. Transform's rotation += composes quaternions.
export class FieldView {
  constructor(store, component, name, indices, c0, c1) {
    this.store = store;
    this.component = component;
    this.name = name;
    this.indices = indices;
    this.c0 = c0;
    this.c1 = c1;
  }

  get data() {
    const w = this.c1 - this.c0, k = this.indices.length;
    const { cols } = this.store.layout(this.component), col = this.store.column(this.component);
    const out = new Float32Array(k * w);
    this.indices.forEach((i, r) => {
      for (let c = 0; c < w; c++) out[r * w + c] = col[i * cols + this.c0 + c];
    });
    return new NdArray(out, [k, w], "f32");
  }

  col(j) {
    return new FieldView(this.store, this.component, this.name, this.indices, this.c0 + j, this.c0 + j + 1);
  }

  _other(v) {
    return v instanceof FieldView ? v.data : v;
  }

  add(v) { return nd.add(this.data, this._other(v), "f32"); }
  sub(v) { return nd.sub(this.data, this._other(v), "f32"); }
  mul(v) { return nd.mul(this.data, this._other(v), "f32"); }
  div(v) { return nd.div(this.data, this._other(v), "f32"); }
  radd(v) { return nd.add(v, this.data, "f32"); }
  rsub(v) { return nd.sub(v, this.data, "f32"); }
  rtruediv(v) { return nd.div(v, this.data, "f32"); }

  _where(mask) {
    const out = [];
    mask.data.forEach((m, r) => { if (m) out.push(this.indices[r]); });
    return new NdArray(Int32Array.from(out), [out.length], "i32");
  }

  lt(v, dtype) { return this._where(nd.lt(this.data, this._other(v), dtype)); }
  le(v, dtype) { return this._where(nd.le(this.data, this._other(v), dtype)); }
  gt(v, dtype) { return this._where(nd.gt(this.data, this._other(v), dtype)); }
  ge(v, dtype) { return this._where(nd.ge(this.data, this._other(v), dtype)); }

  _write(data) {
    this.store.queueUpdate(this.component, this.indices, this.c0, this.c1, data);
  }

  set(v) {
    this._write(toF32(v));
    return this;
  }

  iadd(v) {
    if (this.component === "Transform" && (this.name === "rot" || this.name === "rotation")) {
      this._write(quatCompose(toF32(v), this.data.data));
    } else {
      this._write(nd.add(this.data, v, "f32").data);
    }
    return this;
  }

  isub(v) { this._write(nd.sub(this.data, v, "f32").data); return this; }
  imul(v) { this._write(nd.mul(this.data, v, "f32").data); return this; }
  idiv(v) { return this.set(this.div(v)); }
}

// manifoldx.ecs._quat_multiply(q1, q2) in float32, each product and sum rounded as numpy
// rounds them, left to right; then each row normalised, a zero norm kept as 1. q1 is one
// quaternion or one per row of q2.
function quatCompose(q1, q2) {
  const k = q2.length / 4, s = q1.length === 4 ? 0 : 4, out = new Float32Array(q2.length);
  for (let i = 0; i < k; i++) {
    const [x1, y1, z1, w1] = q1.subarray(s * i, s * i + 4), [x2, y2, z2, w2] = q2.subarray(4 * i, 4 * i + 4);
    const x = F(F(F(F(w1 * x2) + F(x1 * w2)) + F(y1 * z2)) - F(z1 * y2));
    const y = F(F(F(F(w1 * y2) - F(x1 * z2)) + F(y1 * w2)) + F(z1 * x2));
    const z = F(F(F(F(w1 * z2) + F(x1 * y2)) - F(y1 * x2)) + F(z1 * w2));
    const w = F(F(F(F(w1 * w2) - F(x1 * x2)) - F(y1 * y2)) - F(z1 * z2));
    let n = F(Math.sqrt(F(F(F(F(x * x) + F(y * y)) + F(z * z)) + F(w * w))));
    if (n === 0) n = 1;
    out.set([F(x / n), F(y / n), F(z / n), F(w / n)], 4 * i);
  }
  return out;
}

// Past this many alive entities the trace carries a digest of each component's rows.
const DIGEST_PAST = 1000;

// A float as the trace writes it: NaN and the infinities as strings, which JSON can hold.
const traceFloat = (x) => (Number.isNaN(x) ? "nan" : x === Infinity ? "inf" : x === -Infinity ? "-inf" : x);

// The store as the trace writes it (manifoldx.web.replay.store_trace's form): {alive: the
// alive slots, free, high_water, rows: {component: the alive slots' rows}}, a component's
// rows being {digest, n} past 1,000 alive entities: the rows' uint32 words w[i], row-major,
// d = Σ w[i]·(i + 1) mod 2^32.
export function traceStore(store) {
  const alive = [];
  for (let i = 0; i < store.highWater; i++) if (store.alive[i]) alive.push(i);
  const rows = {};
  for (const [name, col] of store._columns) {
    const { cols } = store._layouts.get(name);
    if (alive.length > DIGEST_PAST) {
      const words = new Uint32Array(col.buffer, col.byteOffset, col.length);
      let d = 0, w = 0;
      for (const i of alive) for (let c = 0; c < cols; c++) d = (d + Math.imul(words[i * cols + c], ++w)) >>> 0;
      rows[name] = { digest: d, n: alive.length };
    } else {
      rows[name] = alive.map((i) => Array.from(col.subarray(i * cols, (i + 1) * cols), traceFloat));
    }
  }
  return { alive, free: [...store.free], high_water: store.highWater, rows };
}

// ---------------------------------------------------------------- entity handles

// A value as float64 values, as np.asarray(v, np.float64) makes them.
const toF64 = (v) => (v instanceof NdArray ? Float64Array.from(v.data) : Float64Array.from(Array.isArray(v) ? v.flat(Infinity) : v));

// An entity's Transform row, as manifoldx.engine._TransformProxy: pos is world coordinates in
// float64, the row plus the engine's origin [ox, 0, oz]; setting it stores v - origin rounded to
// float32. rot and scale read float32 copies. Every read is a new array (writing into it
// changes nothing) and every set writes the row at once.
export class TransformProxy {
  constructor(engine, index) {
    Object.defineProperty(this, "_engine", { value: engine });
    Object.defineProperty(this, "_index", { value: index });
  }

  _row() {
    return this._engine.store.column("Transform").subarray(this._index * 10, this._index * 10 + 10);
  }

  get pos() {
    const r = this._row(), [ox, oz] = this._engine.origin;
    return Float64Array.of(r[0] + ox, r[1] + 0, r[2] + oz);
  }

  set pos(v) {
    const a = toF64(v), [ox, oz] = this._engine.origin;
    this._engine.store.write("Transform", this._index, 0, Float32Array.of(a[0] - ox, a[1] - 0, a[2] - oz));
  }

  get rot() { return Float32Array.from(this._row().subarray(3, 7)); }
  set rot(v) { this._engine.store.write("Transform", this._index, 3, toF32(v)); }
  get scale() { return Float32Array.from(this._row().subarray(7, 10)); }
  set scale(v) { this._engine.store.write("Transform", this._index, 7, toF32(v)); }
}

// manifoldx.engine.EntityHandle: set_material writes the entity's Material column at once, as
// the handle's does (no "material" record: that is engine.set_material's); destroy queues it
// for the flush and frees nothing, since the geometry stays in the library for later spawns.
export class EntityHandle {
  constructor(engine, index) {
    Object.defineProperty(this, "_engine", { value: engine });
    this.index = index;
    this.transform = new TransformProxy(engine, index);
  }

  set_material(material) {
    this._engine.store.write("Material", this.index, 0, [material.id]);
    return null;
  }

  destroy(free_geometry = true) {
    this._engine.store.queueDestroy([this.index]);
    return null;
  }
}

// ---------------------------------------------------------------- spawning

// A value's float64 numbers and numpy shape, as np.asarray sees it: a number (shape []), a
// typed array, an NdArray, or nested arrays (a tuple of numbers, a list of rows).
function asValues(v) {
  if (typeof v === "number") return { data: Float64Array.of(v), shape: [] };
  if (v instanceof NdArray) return { data: Float64Array.from(v.data), shape: [...v.shape] };
  if (ArrayBuffer.isView(v)) return { data: Float64Array.from(v), shape: [v.length] };
  const shape = [];
  for (let x = v; Array.isArray(x); x = x[0]) shape.push(x.length);
  return { data: Float64Array.from(v.flat(Infinity)), shape };
}

const shapeOf = (s) => `(${s.join(", ")}${s.length === 1 ? "," : ""})`;

// rows[:, c0:c0+size] = v for n rows of `cols` columns, numpy's broadcast of one value, one
// row or n rows; `shift` (optional) is subtracted in float64 before the float32 store.
function assignColumns(rows, n, cols, c0, size, v, shift = null) {
  const { data, shape } = asValues(v);
  const per = data.length === n * size && shape[0] === n && n !== 1;
  if (!(data.length === 1 || data.length === size || per)) {
    throw new Error(`ValueError: could not broadcast input array from shape ${shapeOf(shape)} into shape (${n},${size})`);
  }
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < size; c++) {
      const x = data.length === 1 ? data[0] : data[(per ? r * size : 0) + c];
      rows[r * cols + c0 + c] = shift === null ? x : x - shift[c];
    }
  }
}

// One Component subclass's rows, as Component.get_data packs them: each field's value (or the
// class default when the call left it out; zeros without one) broadcast from a number, one
// field-sized row, n numbers for a one-column field, or n rows.
function componentRows(name, layout, values, n) {
  const rows = new Float32Array(n * layout.cols);
  for (const [field, [c0, size]] of Object.entries(layout.fields)) {
    const v = values.has(field) && values.get(field) !== null ? values.get(field) : layout.defaults[field];
    if (v === undefined || v === null) continue;
    const { data, shape } = asValues(v);
    const one = shape.length === 0 || data.length === 1 && shape.length <= 1 && !(shape[0] === n && size === 1 && n > 1);
    if (one) {
      for (let r = 0; r < n; r++) rows.fill(data[0], r * layout.cols + c0, r * layout.cols + c0 + size);
    } else if (shape.length === 1 && shape[0] === n && size === 1) {
      for (let r = 0; r < n; r++) rows[r * layout.cols + c0] = data[r];
    } else if (shape.length === 1 && shape[0] === size) {
      for (let r = 0; r < n; r++) rows.set(data, r * layout.cols + c0);
    } else if (shape[0] === n) {
      const w = data.length / n;
      for (let r = 0; r < n; r++) for (let c = 0; c < size; c++) rows[r * layout.cols + c0 + c] = data[r * w + c];
    } else {
      throw new Error(`ValueError: ${name}: field '${field}' value shape ${shapeOf(shape)} incompatible with n=${n}, field size=${size}`);
    }
  }
  return rows;
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

  // The pointer is still locked from the version before this one (manifoldx run --web swapped
  // versions under the lock): pointer_captured reads true, as it did.
  lockHeld() {
    this._captured = true;
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

// createEngine({w, h, camera: {position, target, fov, up, near, far}, origin: [x, z], gui, materials, calls,
// state, hooks}): the engine object transpiled code reaches. gui is state.json's "gui": each
// widget it names becomes engine.gui.widgets[n] with the fields reached code may write (a
// Panel's offset, a Text's text, every widget's style_overrides), which the page applies. materials maps a material id to the
// renderer's material (none, headless); calls receives ["material", entity, material_id] and
// ["capture", on] as reached code asks for them, the trace's record (its "log"); state is
// engine.state, a Map as every dict. hooks are where the browser acts, all optional:
// setMaterial(entityIndex, materialId) after the record, capture(on) -> boolean, whether the
// capture took (headless: a no-op and () => true), and log(text), where engine.log(text)
// sends the app's lines (headless: they collect in engine.lines). resolve(kind, value) is loadState's
// resolver for this engine.
export function createEngine({ w, h, camera, origin = [0, 0], gui = null, materials = {}, calls = [], state = new Map(), hooks = {} }) {
  const { setMaterial = () => {}, capture = () => true, log = null } = hooks;
  const engine = { w, h, origin, state, calls, lines: [] };
  engine.log = (text) => {
    const line = typeof text === "string" ? text : str_(text);
    if (log) log(line);
    else engine.lines.push(line);
  };
  engine.input = new ScriptInput((nw, nh) => { engine.w = nw; engine.h = nh; });
  engine.input._log = calls;
  engine.input._capture = capture;
  engine.input._viewport = [w, h];
  engine.camera = new Camera(camera);
  engine.gui = { pointer_over_gui: false, widgets: gui === null ? [] : guiWidgets(gui) };

  const entities = new Map(), mats = new Map(), geometries = new Map();

  // engine.spawn(...components, {n}), as Engine.spawn: each component's rows as its get_data
  // makes them (the last of a kind wins), a Component subclass registered on its first spawn,
  // Transform's pos taken as world coordinates (the origin subtracted in float64), the slots
  // filled at once. The handle of the first entity, or null for n 0.
  engine.spawn = (...args) => {
    const last = args[args.length - 1];
    const opts = last && !("component" in last) ? args.pop() : {};
    const n = opts.n ?? 1, store = engine.store;
    const given = new Map(args.map((c) => [c.component, c.values]));
    for (const name of given.keys()) store.register(name);
    const rows = new Map();
    for (const [name, values] of given) {
      if (name === "Mesh") rows.set(name, Uint32Array.of(values.get("geometry").geometry_id));
      else if (name === "Material") rows.set(name, Uint32Array.of(values.get("material").id));
      else if (name !== "Transform") rows.set(name, componentRows(name, store.layout(name), values, n));
      else {
        const t = new Float32Array(n * 10);
        for (let r = 0; r < n; r++) t.set([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], r * 10);
        const [ox, oz] = engine.origin;
        const at = (k, c0, size, shift) => values.has(k) && values.get(k) !== null && assignColumns(t, n, 10, c0, size, values.get(k), shift);
        at("pos", 0, 3, ox === 0 && oz === 0 ? null : [ox, 0, oz]);
        at("rot", 3, 4, null);
        at("scale", 7, 3, null);
        rows.set(name, t);
      }
    }
    if (!rows.has("Transform")) store.register("Transform");
    if (n <= 0) return null;
    const indices = store.spawn(n, rows);
    return engine.resolve("entity", indices[0]);
  };

  // engine.destroy(x), as Engine.destroy: null does nothing; a bool mask gives its true
  // positions, an index array its values; the entities go at the flush.
  engine.destroy = (x) => {
    if (x === null || x === undefined) return null;
    let ix;
    if (x instanceof NdArray && x.dtype === "bool") {
      ix = [];
      x.data.forEach((m, i) => { if (m) ix.push(i); });
    } else ix = Array.from(x instanceof NdArray ? x.data : x);
    if (ix.length) engine.store.queueDestroy(ix);
    return null;
  };
  engine.set_material = (entity, material) => {
    const index = typeof entity === "number" ? entity : entity.index;
    const store = engine.store;
    if (store && !(index >= 0 && index < store.capacity && store.alive[index])) {
      throw new Error(`ValueError: set_material: entity ${index} is not alive`);
    }
    calls.push(["material", index, material.id]);
    engine.store?.write("Material", index, 0, [material.id]); // the column, now, as the desktop's
    setMaterial(index, material.id);
  };
  engine.resolve = (kind, value) => {
    if (kind === "engine") {
      if (value === "Engine") return engine;
      if (value in engine) return engine[value];
      throw new Error(`the state names engine.${value}, which the runtime does not have`);
    }
    if (kind === "entity") {
      if (!entities.has(value)) entities.set(value, new EntityHandle(engine, value));
      return entities.get(value);
    }
    if (kind === "material") {
      if (!mats.has(value)) mats.set(value, { id: value, material: materials[value] ?? null });
      return mats.get(value);
    }
    if (kind === "geometry") {
      if (!geometries.has(value)) geometries.set(value, Object.freeze({ geometry_id: value }));
      return geometries.get(value);
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
// order the app registered them, called as the engine calls them, with (query, dt), the query
// of the components it registered (null without a store); then the
// store's queued writes and destroys land, as the engine executes its commands.
export function step(app, dt) {
  const store = app.engine.store ?? null;
  for (const r of app.roots) {
    if (r.kind === "system") r.fn(store === null ? null : new Query(store, r.components ?? []), dt);
  }
  store?.flush();
}

// An event, as engine.emit delivers it: every handler root of that event, in the order the
// app registered them, called with the payload (runtime values: a dict is a Map).
export function fire(app, event, payload) {
  const store = app.engine.store ?? null;
  for (const r of app.roots) {
    if (r.kind !== "handler" || r.event !== event) continue;
    if (r.components && store !== null) r.fn(payload, new Query(store, r.components));
    else r.fn(payload);
  }
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
// manifoldx.web.replay's): {camera, gui, log, state, paths, store}, log being what
// engine.calls gained since logStart, store traceStore's (null without a store).
export function traceEntry(app, paths, logStart) {
  const { engine } = app;
  const entry = {
    camera: { position: engine.camera.position, target: engine.camera.target },
    gui: guiValues(app),
    log: engine.calls.slice(logStart),
    state: engine.state,
    paths: Object.fromEntries(paths.map((p) => [p, pathValue(app, p)])),
    store: engine.store === null || engine.store === undefined ? null : traceStore(engine.store),
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
  if (v instanceof EntityHandle) return `<entity ${v.index}>`;
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
