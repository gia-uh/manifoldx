// The headless runtime surface: what transpiled code calls on the engine, in Node.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { createEngine, guiValues, step } from "../../../src/manifoldx/web/js/manifoldx.js";

const FORWARD = JSON.parse(readFileSync(new URL("./forward.json", import.meta.url)));

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

test("set_material and capture_pointer append to the log", () => {
  const log = [];
  const e = engine({ log });
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
  assert.equal(e.log, log);
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
