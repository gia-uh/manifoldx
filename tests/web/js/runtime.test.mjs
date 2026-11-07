// The headless runtime surface: what transpiled code calls on the engine, in Node.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import test from "node:test";

import {
  advance,
  createEngine,
  encodeState,
  fromJSON,
  guiValues,
  loadApp,
  mergePatch,
  pathValue,
  stateView,
  step,
  toJSON,
  traceEntry,
  traceStore,
} from "../../../src/manifoldx/web/js/manifoldx.js";

const CORE = new URL("../../../src/manifoldx/web/js/manifoldx.js", import.meta.url);
const FORWARD = JSON.parse(readFileSync(new URL("./forward.json", import.meta.url)));
// The folder export --web wrote for examples/web_walk.py, set by tests/web/test_node_suites.py.
const WEB_WALK = process.env.MX_WEB_WALK_EXPORT;

const engine = (opts = {}) =>
  createEngine({ w: 640, h: 480, camera: { position: [0, 1, 2], target: [0, 0, 0], near: 0.1, far: 1000 }, ...opts });

test("camera.get_forward equals numpy's float32 result", () => {
  assert.equal(FORWARD.length, 5);
  for (const { position, target, forward } of FORWARD) {
    const e = engine({ camera: { position, target, near: 0.1, far: 100 } });
    const f = e.camera.get_forward();
    assert.ok(f instanceof Float32Array);
    assert.deepEqual(Array.from(f), forward);
  }
});

test("the camera holds float32 arrays, near and far", () => {
  const e = engine();
  assert.ok(e.camera.position instanceof Float32Array && e.camera.target instanceof Float32Array);
  assert.deepEqual(Array.from(e.camera.position), [0, 1, 2]);
  assert.equal(e.camera.near, 0.1);
  assert.equal(e.camera.far, 1000);
  assert.equal(e.w, 640);
  assert.equal(e.h, 480);
});

test("setFrame drives every input getter", () => {
  const e = engine();
  const input = e.input;
  assert.equal(input.pressed_keys.size, 0);
  assert.deepEqual(input.mouse_delta, [0, 0]);
  input.setFrame({ keys: ["w", "Shift"], just: ["Tab", "w"], d: [3.5, -2], buttons: [0, 2], clicks: [2], vp: [800, 600] });
  assert.deepEqual([...input.pressed_keys].sort(), ["Shift", "w"]);
  assert.equal(input.is_pressed("W"), true);
  assert.equal(input.just_pressed("Tab"), true);
  assert.equal(input.just_pressed("W"), true);
  assert.equal(input.just_pressed("Escape"), false);
  assert.deepEqual(input.mouse_delta, [3.5, -2]);
  assert.equal(input.is_mouse_pressed(0), true);
  assert.equal(input.is_mouse_pressed(1), false);
  assert.equal(input.just_mouse_pressed(2), true);
  assert.equal(input.just_mouse_pressed(0), false);
  assert.deepEqual(input.viewport_size, [800, 600]);
  assert.deepEqual([e.w, e.h], [800, 600]);
  // The next frame replaces the last one.
  input.setFrame({ keys: [], just: [], d: [0, 0], buttons: [], clicks: [], vp: [800, 600] });
  assert.equal(input.just_pressed("Tab"), false);
  assert.equal(input.is_mouse_pressed(0), false);
  assert.deepEqual(input.mouse_delta, [0, 0]);
});

test("what a frame does not hold answers as nothing happened", () => {
  const input = engine().input;
  input.setFrame({ keys: ["w"], just: ["w"], d: [1, 1], buttons: [2, 1], clicks: [1], vp: [800, 600] });
  assert.deepEqual(input.pressed_buttons, [1, 2]);
  assert.equal(input.just_released("w"), false);
  assert.equal(input.just_mouse_released(1), false);
  assert.deepEqual(input.modifiers, []);
  assert.deepEqual(input.mouse_pos, [0, 0]);
  assert.deepEqual(input.wheel_delta, [0, 0]);
});

const RELEASE = { keys: [], just: [], d: [0, 0], buttons: [], clicks: [], vp: [800, 600] };

test("up drives just_released", () => {
  const input = engine().input;
  input.setFrame({ ...RELEASE, up: ["W", " "] });
  assert.equal(input.just_released("w"), true);
  assert.equal(input.just_released("Space"), true);
  assert.equal(input.just_released("a"), false);
});

test("released drives just_mouse_released", () => {
  const input = engine().input;
  input.setFrame({ ...RELEASE, released: [2] });
  assert.equal(input.just_mouse_released(2), true);
  assert.equal(input.just_mouse_released(1), false);
});

test("pos drives mouse_pos", () => {
  const input = engine().input;
  input.setFrame({ ...RELEASE, pos: [12.5, 30] });
  assert.deepEqual(input.mouse_pos, [12.5, 30]);
});

test("wheel drives wheel_delta", () => {
  const input = engine().input;
  input.setFrame({ ...RELEASE, wheel: [0, -3] });
  assert.deepEqual(input.wheel_delta, [0, -3]);
});

test("mods drives modifiers", () => {
  const input = engine().input;
  input.setFrame({ ...RELEASE, mods: ["Control", "Shift"] });
  assert.deepEqual(input.modifiers, ["Control", "Shift"]);
  input.setFrame(RELEASE);
  assert.deepEqual(input.modifiers, []);
});

test("the camera holds fov and a float32 up", () => {
  const e = engine({ camera: { position: [0, 1, 2], target: [0, 0, 0], fov: 50, up: [0, 0, 1] } });
  assert.equal(e.camera.fov, 50);
  assert.ok(e.camera.up instanceof Float32Array);
  assert.deepEqual(Array.from(e.camera.up), [0, 0, 1]);
  const d = engine().camera;
  assert.equal(d.fov, 60);
  assert.deepEqual(Array.from(d.up), [0, 1, 0]);
});

test("engine.gui.pointer_over_gui is the flag the state's gui handle resolves to", () => {
  const e = engine();
  assert.equal(e.resolve("engine", "gui").pointer_over_gui, false);
  e.gui.pointer_over_gui = true;
  assert.equal(e.resolve("engine", "gui").pointer_over_gui, true);
});

test("set_material and capture_pointer append to the log", () => {
  const log = [];
  const e = engine({ calls: log });
  const box = e.resolve("entity", 4);
  const red = e.resolve("material", 7);
  assert.equal(e.resolve("entity", 4), box);
  e.set_material(box, red);
  e.set_material(5, e.resolve("material", 2));
  assert.equal(e.input.pointer_captured, false);
  assert.equal(e.input.capture_pointer(true), true);
  assert.equal(e.input.pointer_captured, true);
  e.input.capture_pointer(false);
  assert.deepEqual(log, [["material", 4, 7], ["material", 5, 2], ["capture", true], ["capture", false]]);
  assert.equal(e.calls, log);
  assert.equal(e.gui.pointer_over_gui, false);
  assert.equal(e.resolve("engine", "Engine"), e);
  assert.equal(e.resolve("engine", "input"), e.input);
  assert.equal(e.resolve("engine", "camera"), e.camera);
  assert.equal(e.resolve("engine", "gui"), e.gui);
});

test("step calls the systems in registration order with (null, dt)", () => {
  const e = engine();
  const calls = [];
  const app = {
    engine: e,
    roots: [
      { kind: "system", label: "a", fn: (q, dt) => calls.push(["a", q, dt]) },
      { kind: "handler", label: "h", event: "key_down", fn: () => calls.push(["h"]) },
      { kind: "getter", label: "g", fn: () => "x" },
      { kind: "system", label: "b", fn: (q, dt) => calls.push(["b", q, dt]) },
    ],
  };
  step(app, 0.25);
  assert.deepEqual(calls, [["a", null, 0.25], ["b", null, 0.25]]);
});

test("guiValues returns each getter's text in order", () => {
  const app = {
    engine: engine(),
    roots: [
      { kind: "getter", label: "g1", fn: () => "fps 60" },
      { kind: "system", label: "s", fn: () => null },
      { kind: "getter", label: "g2", fn: () => "Matemática" },
    ],
  };
  assert.deepEqual(guiValues(app), ["fps 60", "Matemática"]);
});

// ---------------------------------------------------------------- scene hooks

test("set_material calls the setMaterial hook with the entity's index and the material's id", () => {
  const calls = [];
  const log = [];
  const e = engine({ calls: log, hooks: { setMaterial: (entity, id) => calls.push([entity, id]) } });
  e.set_material(e.resolve("entity", 4), e.resolve("material", 7));
  e.set_material(5, e.resolve("material", 2));
  assert.deepEqual(calls, [[4, 7], [5, 2]]);
  assert.deepEqual(log, [["material", 4, 7], ["material", 5, 2]]);
});

test("a capture hook that refuses leaves the flag false and the log unchanged", () => {
  const asked = [];
  const log = [];
  const e = engine({ calls: log, hooks: { capture: (on) => (asked.push(on), false) } });
  assert.equal(e.input.capture_pointer(true), false);
  assert.equal(e.input.pointer_captured, false);
  assert.deepEqual(asked, [true]);
  assert.deepEqual(log, []);
});

test("a capture hook that takes sets the flag and logs, as headless does", () => {
  const log = [];
  const e = engine({ calls: log, hooks: { capture: () => true } });
  assert.equal(e.input.capture_pointer(true), true);
  assert.equal(e.input.pointer_captured, true);
  assert.equal(e.input.capture_pointer(false), true);
  assert.equal(e.input.pointer_captured, false);
  assert.deepEqual(log, [["capture", true], ["capture", false]]);
});

test("lockLost clears pointer_captured", () => {
  const e = engine();
  e.input.capture_pointer(true);
  e.input.lockLost();
  assert.equal(e.input.pointer_captured, false);
});

test("loadApp passes options.hooks to the engine", () => {
  const calls = [];
  const state = {
    table: [],
    roots: [],
    engine: { w: 10, h: 10, camera: { position: [0, 0, 1], target: [0, 0, 0] } },
  };
  const { engine: e } = loadApp(state, {}, { hooks: { setMaterial: (i, m) => calls.push([i, m]) } });
  e.set_material(3, e.resolve("material", 1));
  assert.deepEqual(calls, [[3, 1]]);
});

test("engine.log collects the app's lines in engine.lines, or hands them to hooks.log", () => {
  const e = engine();
  e.log("yaw 1.0");
  e.log("yaw 2.0");
  assert.deepEqual(e.lines, ["yaw 1.0", "yaw 2.0"]);
  assert.deepEqual(e.calls, []);
  const said = [];
  const t = engine({ hooks: { log: (text) => said.push(text) } });
  t.log("hello");
  assert.deepEqual(said, ["hello"]);
  assert.deepEqual(t.lines, []);
});

// An app whose module holds one system and two handlers, as loadApp builds it.
function handlerApp() {
  const seen = [];
  const mod = {
    tick: () => seen.push("tick"),
    bump: (payload) => seen.push(["bump", payload]),
    bump2: (payload) => seen.push(["bump2", payload]),
    __set_globals: () => {},
  };
  const fn = (name) => ({ __function: `__main__:${name}` });
  const state = {
    table: [],
    roots: [
      { kind: "system", label: "system tick", fn: fn("tick") },
      { kind: "handler", label: 'on("bump") bump', event: "bump", fn: fn("bump") },
      { kind: "handler", label: 'on("other") bump2', event: "other", fn: fn("bump2") },
    ],
    engine: { w: 10, h: 10, camera: { position: [0, 0, 1], target: [0, 0, 0] } },
  };
  return { app: loadApp(state, { __main__: mod }), seen };
}

const IDLE = { keys: [], just: [], d: [0, 0], buttons: [], clicks: [], vp: [10, 10] };

test("advance delivers a frame's event to every handler root of it once, with the payload as a Map, before the systems", () => {
  const { app, seen } = handlerApp();
  advance(app, { ...IDLE, events: [["bump", fromJSON({ by: 10 })]] }, 1 / 60);
  assert.equal(seen.length, 2);
  const [name, payload] = seen[0];
  assert.equal(name, "bump");
  assert.ok(payload instanceof Map);
  assert.equal(payload.get("by"), 10);
  assert.equal(seen[1], "tick");
  assert.deepEqual(app.engine.bus.delivered, ["frame", "bump"]);
});

test("an event no root handles is delivered to nobody", () => {
  const { app, seen } = handlerApp();
  advance(app, { ...IDLE, events: [["nobody", null]] }, 1 / 60);
  assert.deepEqual(seen, ["tick"]);
});

test("an emit during a frame waits for the next frame, after the frame event", () => {
  const { app, seen } = handlerApp();
  app.engine.emit("bump", null);
  advance(app, IDLE, 0.5);
  advance(app, IDLE, 0.5);
  assert.deepEqual(seen, [["bump", null], "tick", "tick"]);
  assert.deepEqual(app.engine.bus.delivered, ["frame", "bump", "frame"]);
  assert.equal(app.engine.elapsed, 0.5);
  assert.equal(app.engine.frame, 2);
});

// ---------------------------------------------------------------- the trace helpers

test("traceEntry is the entry replay.mjs built, for a frame of the web_walk export", { skip: !WEB_WALK && "MX_WEB_WALK_EXPORT not set" }, async () => {
  const folder = pathToFileURL(WEB_WALK + "/");
  const { load } = await import(new URL("replay.mjs", folder).href);
  const app = await load(folder);
  const frames = JSON.parse(readFileSync(new URL("../fixtures/web_walk_script.json", import.meta.url)));
  const paths = ["rig.pos", "rig.yaw", "rig.mode", "selected"];
  // Frames up to the first click, which captures the pointer and logs.
  const upto = frames.findIndex((f) => f.clicks.length) + 1;
  let entry;
  for (const frame of frames.slice(0, upto)) {
    const start = app.engine.calls.length;
    advance(app, frame, 1 / 60);
    const events = [...app.engine.bus.delivered];
    entry = traceEntry(app, paths, start);
    // What replay() built before the helpers moved into the core.
    const { engine: e } = app;
    const before = {
      camera: { position: e.camera.position, target: e.camera.target },
      gui: guiValues(app),
      log: e.calls.slice(start),
      state: e.state,
      paths: Object.fromEntries(paths.map((p) => [p, pathValue(app, p)])),
      store: traceStore(e.store),
      events,
    };
    assert.deepEqual(entry, JSON.parse(toJSON(before)));
  }
  assert.deepEqual(entry.log, [["capture", true]]);
  assert.equal(entry.camera.position.length, 3);
  assert.ok(!(entry.camera.position instanceof Float32Array));
});

test("toJSON writes Maps as objects, typed arrays and Sets as arrays", () => {
  const v = new Map([["a", new Float32Array([1, 2])], ["b", new Set(["x"])]]);
  assert.equal(toJSON(v), '{"a":[1,2],"b":["x"]}');
});

test("pathValue reads a main global first, then a key of engine.state", () => {
  const e = engine({ state: new Map([["selected", "b2"], ["rig", "shadowed"]]) });
  const app = { engine: e, modules: { __main__: { rig: { pos: [1, 2, 3], cfg: new Map([["k", 4]]) } } } };
  assert.deepEqual(pathValue(app, "rig.pos"), [1, 2, 3]);
  assert.equal(pathValue(app, "rig.cfg.k"), 4);
  assert.equal(pathValue(app, "selected"), "b2");
  assert.throws(() => pathValue(app, "rig.nope"), /names nothing at nope/);
});

test("the core imports only pyrt.js and names no DOM or three.js", () => {
  const src = readFileSync(CORE, "utf8");
  const specifiers = [...src.matchAll(/^\s*import\b[^;]*?["']([^"']+)["']/gm)].map((m) => m[1]);
  assert.deepEqual(specifiers, ["./pyrt.js"]);
  assert.doesNotMatch(src, /\bimport\s*\(/);
  assert.doesNotMatch(src, /\b(document|window|three|THREE)\b/);
});

// ---------------------------------------------------------------- the state, both ways

test("encodeState writes the snapshot's value forms and skips what is not plain data", () => {
  class Viewer {}
  const loop = new Map();
  loop.set("self", loop);
  const state = new Map([
    ["cam", new Map([["yaw", 0.5]])],
    ["v", Float32Array.from([1.5, 2])],
    ["tags", new Set(["a"])],
    ["bad", NaN],
    ["pos", [1, -Infinity]],
    ["viewer", Object.create(Viewer.prototype)],
    ["loop", loop],
  ]);
  assert.deepEqual(encodeState(state), {
    __dict: [
      ["cam", { __dict: [["yaw", 0.5]] }],
      ["v", { __f32: [1.5, 2] }],
      ["tags", { __set: ["a"] }],
      ["bad", { __float: "nan" }],
      ["pos", [1, { __float: "-inf" }]],
      ["viewer", { __skip: "an object of Viewer" }],
      ["loop", { __dict: [["self", { __skip: "a cycle" }]] }],
    ],
  });
  assert.deepEqual(encodeState(new Map([["d", Float64Array.from([0.25])]])), { __dict: [["d", { __f64: [0.25] }]] });
  assert.deepEqual(encodeState(new Map([["f", () => 1]])), { __dict: [["f", { __skip: "a function" }]] });
});

test("a value two keys share is not a cycle", () => {
  const shared = [1, 2];
  assert.deepEqual(encodeState(new Map([["a", shared], ["b", shared]])), { __dict: [["a", [1, 2]], ["b", [1, 2]]] });
});

test("fromJSON makes every object a Map", () => {
  const v = fromJSON({ cam: { yaw: 1 }, list: [{ a: 2 }], n: null });
  assert.ok(v instanceof Map && v.get("cam") instanceof Map);
  assert.equal(v.get("cam").get("yaw"), 1);
  assert.ok(v.get("list")[0] instanceof Map);
  assert.equal(v.get("n"), null);
});

test("mergePatch merges nested Maps, deletes on null and replaces arrays whole", () => {
  const state = new Map([
    ["cam", new Map([["yaw", 1], ["pitch", 2]])],
    ["gone", 3],
    ["path", [1, 2, 3]],
  ]);
  const out = mergePatch(state, fromJSON({ cam: { yaw: 0 }, gone: null, path: [9], extra: { a: 1 } }));
  assert.equal(out, state); // in place: engine.state stays one object
  assert.deepEqual(toJSON(state), JSON.stringify({ cam: { yaw: 0, pitch: 2 }, path: [9], extra: { a: 1 } }));
  assert.equal(mergePatch(state, 5), 5);
});

test("stateView is plain JSON, a handle as the desktop's repr would name it", () => {
  const e = engine();
  const state = new Map([
    ["cam", new Map([["yaw", 1]])],
    ["v", Float32Array.from([0.5])],
    ["tags", new Set(["a"])],
    ["m", e.resolve("material", 3)],
    ["box", e.resolve("entity", 7)],
  ]);
  assert.deepEqual(stateView(state), {
    cam: { yaw: 1 },
    v: [0.5],
    tags: ["a"],
    m: "<material 3>",
    box: "<entity 7>",
  });
});
