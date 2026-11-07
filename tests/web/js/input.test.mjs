// Browser input without a browser: DOM-like events to Frames, pointer lock over fakes.

import assert from "node:assert/strict";
import test from "node:test";

import { createEngine } from "../../../src/manifoldx/web/js/manifoldx.js";
import { FrameBuilder, PointerLock, attachDom } from "../../../src/manifoldx/web/js/input.js";

const vp = [800, 500];
const pick = (frame, ...names) => Object.fromEntries(names.map((n) => [n, frame[n]]));

test("a frame with no events is the empty frame, every field present", () => {
  const b = new FrameBuilder();
  assert.deepEqual(b.take(vp), {
    keys: [], just: [], d: [0, 0], buttons: [], clicks: [], vp: [800, 500],
    up: [], released: [], pos: [0, 0], wheel: [0, 0], mods: [],
  });
});

test("press and release inside one frame: just has it, keys does not, up has it", () => {
  const b = new FrameBuilder();
  b.keyDown({ key: "w", code: "KeyW" });
  b.keyUp({ code: "KeyW" });
  assert.deepEqual(pick(b.take(vp), "keys", "just", "up"), { keys: [], just: ["w"], up: ["w"] });
});

test("a held key stays in keys and leaves just after its frame", () => {
  const b = new FrameBuilder();
  b.keyDown({ key: "a", code: "KeyA" });
  assert.deepEqual(pick(b.take(vp), "keys", "just"), { keys: ["a"], just: ["a"] });
  assert.deepEqual(pick(b.take(vp), "keys", "just", "up"), { keys: ["a"], just: [], up: [] });
  b.keyUp({ code: "KeyA" });
  assert.deepEqual(pick(b.take(vp), "keys", "just", "up"), { keys: [], just: [], up: ["a"] });
});

test("Shift released first: the key is tracked by code and named at keydown", () => {
  const b = new FrameBuilder();
  b.keyDown({ key: "Shift", code: "ShiftLeft", shiftKey: true });
  b.keyDown({ key: "@", code: "Digit2", shiftKey: true });
  assert.deepEqual(b.take(vp).keys, ["@", "Shift"]);
  b.keyUp({ code: "ShiftLeft", shiftKey: false });
  b.keyUp({ code: "Digit2" });
  const f = b.take(vp);
  assert.deepEqual(f.keys, []);
  assert.deepEqual(f.up, ["@", "Shift"]);
});

test("key names are manifoldx's: letters lowercase, Space, Escape, Control, Delete", () => {
  const b = new FrameBuilder();
  const named = [["W", "KeyW"], [" ", "Space"], ["Esc", "Escape"], ["Ctrl", "ControlLeft"], ["Del", "Delete"], ["Tab", "Tab"], ["ArrowUp", "ArrowUp"]];
  for (const [key, code] of named) b.keyDown({ key, code });
  assert.deepEqual(b.take(vp).just, ["ArrowUp", "Control", "Delete", "Escape", "Space", "Tab", "w"]);
});

test("repeat keydowns add nothing to just", () => {
  const b = new FrameBuilder();
  b.keyDown({ key: "w", code: "KeyW" });
  b.take(vp);
  b.keyDown({ key: "w", code: "KeyW", repeat: true });
  b.keyDown({ key: "w", code: "KeyW", repeat: true });
  assert.deepEqual(pick(b.take(vp), "keys", "just"), { keys: ["w"], just: [] });
});

test("blur releases every held key and button, listed in the next frame's up and released", () => {
  const b = new FrameBuilder();
  b.keyDown({ key: "w", code: "KeyW", shiftKey: true });
  b.keyDown({ key: "d", code: "KeyD", shiftKey: true });
  b.pointerDown({ button: 2, x: 5, y: 6, pointerType: "mouse" });
  b.take(vp);
  b.blur();
  const f = b.take(vp);
  assert.deepEqual(pick(f, "keys", "up", "buttons", "released", "mods"), { keys: [], up: ["d", "w"], buttons: [], released: [2], mods: [] });
  assert.deepEqual(pick(b.take(vp), "up", "released"), { up: [], released: [] });
});

test("buttons map as rendercanvas's MOUSE_BUTTON_MAP", () => {
  const b = new FrameBuilder();
  for (const button of [0, 1, 2, 3, 4]) b.pointerDown({ button, x: 0, y: 0, pointerType: "mouse" });
  const f = b.take(vp);
  assert.deepEqual(pick(f, "buttons", "clicks"), { buttons: [1, 2, 3, 4, 5], clicks: [1, 2, 3, 4, 5] });
  b.pointerUp({ button: 1, x: 0, y: 0 });
  b.pointerUp({ button: 0, x: 0, y: 0 });
  assert.deepEqual(pick(b.take(vp), "buttons", "clicks", "released"), { buttons: [2, 4, 5], clicks: [], released: [1, 3] });
});

test("a click inside one frame: clicks and released have it, buttons does not", () => {
  const b = new FrameBuilder();
  b.pointerDown({ button: 0, x: 10, y: 20, pointerType: "mouse" });
  b.pointerUp({ button: 0, x: 11, y: 21 });
  assert.deepEqual(pick(b.take(vp), "buttons", "clicks", "released", "pos"), { buttons: [], clicks: [1], released: [1], pos: [11, 21] });
});

test("movement sums over the frame into d; pos is the last pointer position", () => {
  const b = new FrameBuilder();
  b.pointerMove({ movementX: 3, movementY: -1, x: 100, y: 50 });
  b.pointerMove({ movementX: 2.5, movementY: 4, x: 102.5, y: 54 });
  assert.deepEqual(pick(b.take(vp), "d", "pos"), { d: [5.5, 3], pos: [102.5, 54] });
  assert.deepEqual(pick(b.take(vp), "d", "pos"), { d: [0, 0], pos: [102.5, 54] });
});

test("wheel: pixels scale by 1/devicePixelRatio, lines by 16, pages by 600, summed", () => {
  const b = new FrameBuilder({ devicePixelRatio: 2 });
  b.wheel({ deltaX: 0, deltaY: 100, deltaMode: 0 });
  b.wheel({ deltaX: 1, deltaY: 3, deltaMode: 1 });
  b.wheel({ deltaX: 0, deltaY: -1, deltaMode: 2 });
  assert.deepEqual(b.take(vp).wheel, [16, 50 + 48 - 600]);
  assert.deepEqual(b.take(vp).wheel, [0, 0]);
});

test("mods: sorted names of the held modifier flags of the latest event", () => {
  const b = new FrameBuilder();
  b.keyDown({ key: "Shift", code: "ShiftLeft", shiftKey: true, altKey: true });
  assert.deepEqual(b.take(vp).mods, ["Alt", "Shift"]);
  assert.deepEqual(b.take(vp).mods, ["Alt", "Shift"]);
  b.pointerMove({ movementX: 0, movementY: 0, x: 0, y: 0, ctrlKey: true, metaKey: true });
  assert.deepEqual(b.take(vp).mods, ["Control", "Meta"]);
  // An event that carries no flags leaves them.
  b.wheel({ deltaX: 0, deltaY: 1, deltaMode: 1 });
  assert.deepEqual(b.take(vp).mods, ["Control", "Meta"]);
});

test("vp is the viewport take() is given", () => {
  const b = new FrameBuilder();
  assert.deepEqual(b.take([640, 480]).vp, [640, 480]);
});

test("lockEnded: the next frame's just holds Escape, once", () => {
  const b = new FrameBuilder();
  b.lockEnded();
  assert.deepEqual(pick(b.take(vp), "keys", "just"), { keys: [], just: ["Escape"] });
  assert.deepEqual(b.take(vp).just, []);
});

test("lockEnded after an Escape keydown in the same frame adds nothing", () => {
  const b = new FrameBuilder();
  b.keyDown({ key: "Escape", code: "Escape" });
  b.lockEnded();
  assert.deepEqual(pick(b.take(vp), "keys", "just"), { keys: ["Escape"], just: ["Escape"] });
});

// ---------------------------------------------------------------- pointer lock

// A canvas and document with just what PointerLock touches; dispatch(type) fires listeners.
function fakes({ active = true, lockable = true } = {}) {
  const listeners = (target) => {
    target.on = {};
    target.addEventListener = (type, fn, opts) => { (target.on[type] ??= []).push({ fn, once: opts?.once }); };
    target.removeEventListener = (type, fn) => { target.on[type] = (target.on[type] ?? []).filter((l) => l.fn !== fn); };
    target.dispatch = (type, ev = {}) => {
      for (const l of [...(target.on[type] ?? [])]) {
        if (l.once) target.removeEventListener(type, l.fn);
        l.fn(ev);
      }
    };
    target.count = (type) => (target.on[type] ?? []).length;
  };
  const doc = { userActivation: { isActive: active }, pointerLockElement: null, exits: 0 };
  doc.exitPointerLock = () => { doc.exits++; doc.pointerLockElement = null; doc.dispatch("pointerlockchange"); };
  const canvas = { requests: 0 };
  if (lockable) canvas.requestPointerLock = () => { canvas.requests++; doc.pointerLockElement = canvas; doc.dispatch("pointerlockchange"); };
  listeners(doc);
  listeners(canvas);
  return { doc, canvas };
}

const lockedEngine = () => createEngine({ w: 800, h: 500, camera: { position: [0, 1, 2], target: [0, 0, 0] } });

test("request(true) with a user activation locks at once and returns true", () => {
  const { doc, canvas } = fakes();
  const lock = new PointerLock(canvas, doc, new FrameBuilder());
  assert.equal(lock.request(true), true);
  assert.equal(canvas.requests, 1);
  assert.equal(lock.locked, true);
});

test("request(true) with no activation arms a one-shot pointerdown that locks", () => {
  const { doc, canvas } = fakes({ active: false });
  const lock = new PointerLock(canvas, doc, new FrameBuilder());
  assert.equal(lock.request(true), true);
  assert.equal(canvas.requests, 0);
  assert.equal(lock.locked, false);
  canvas.dispatch("pointerdown", { pointerType: "mouse" });
  assert.equal(canvas.requests, 1);
  assert.equal(lock.locked, true);
  doc.exitPointerLock();
  canvas.dispatch("pointerdown", { pointerType: "mouse" });
  assert.equal(canvas.requests, 1);
});

test("request(true) returns false where the canvas has no pointer lock or the last pointer was a touch", () => {
  const plain = fakes({ lockable: false });
  assert.equal(new PointerLock(plain.canvas, plain.doc, new FrameBuilder()).request(true), false);
  const { doc, canvas } = fakes();
  const b = new FrameBuilder();
  b.pointerDown({ button: 0, x: 1, y: 1, pointerType: "touch" });
  const lock = new PointerLock(canvas, doc, b);
  assert.equal(lock.request(true), false);
  assert.equal(canvas.requests, 0);
  b.pointerDown({ button: 0, x: 1, y: 1, pointerType: "mouse" });
  assert.equal(lock.request(true), true);
});

test("the engine's capture_pointer goes through request: refused, the flag stays false", () => {
  const { doc, canvas } = fakes({ lockable: false });
  const lock = new PointerLock(canvas, doc, new FrameBuilder());
  const engine = createEngine({ w: 800, h: 500, camera: { position: [0, 1, 2], target: [0, 0, 0] }, hooks: { capture: (on) => lock.request(on) } });
  assert.equal(engine.input.capture_pointer(true), false);
  assert.equal(engine.input.pointer_captured, false);
  assert.deepEqual(engine.log, []);
});

test("a lock the app did not end: Escape in the next frame and pointer_captured false", () => {
  const { doc, canvas } = fakes();
  const b = new FrameBuilder();
  const lock = new PointerLock(canvas, doc, b);
  const engine = lockedEngine();
  lock.input = engine.input;
  engine.input._capture = (on) => lock.request(on);
  assert.equal(engine.input.capture_pointer(true), true);
  assert.equal(engine.input.pointer_captured, true);
  // The browser releases it (Escape, which Chromium keeps from the page).
  doc.pointerLockElement = null;
  doc.dispatch("pointerlockchange");
  assert.equal(lock.locked, false);
  assert.equal(engine.input.pointer_captured, false);
  assert.deepEqual(b.take(vp).just, ["Escape"]);
});

test("a pointerlockerror is a lock lost too", () => {
  const { doc, canvas } = fakes();
  const b = new FrameBuilder();
  const lock = new PointerLock(canvas, doc, b);
  const engine = lockedEngine();
  lock.input = engine.input;
  engine.input._capture = (on) => lock.request(on);
  // The refusal arrives after the request returns, as the browser fires it.
  canvas.requestPointerLock = () => { canvas.requests++; };
  assert.equal(engine.input.capture_pointer(true), true);
  assert.equal(engine.input.pointer_captured, true);
  doc.dispatch("pointerlockerror");
  assert.equal(engine.input.pointer_captured, false);
  assert.deepEqual(b.take(vp).just, ["Escape"]);
});

test("request(false) exits only when locked, and the release it asked for is no lock lost", () => {
  const { doc, canvas } = fakes();
  const b = new FrameBuilder();
  const lock = new PointerLock(canvas, doc, b);
  const engine = lockedEngine();
  lock.input = engine.input;
  engine.input._capture = (on) => lock.request(on);
  assert.equal(engine.input.capture_pointer(false), true);
  assert.equal(doc.exits, 0);
  engine.input.capture_pointer(true);
  b.take(vp);
  assert.equal(engine.input.capture_pointer(false), true);
  assert.equal(doc.exits, 1);
  assert.equal(lock.locked, false);
  assert.deepEqual(b.take(vp).just, []);
  assert.deepEqual(engine.log, [["capture", false], ["capture", true], ["capture", false]]);
});

test("request(false) disarms a pending gesture request", () => {
  const { doc, canvas } = fakes({ active: false });
  const lock = new PointerLock(canvas, doc, new FrameBuilder());
  lock.request(true);
  lock.request(false);
  canvas.dispatch("pointerdown", { pointerType: "mouse" });
  assert.equal(canvas.requests, 0);
  assert.equal(canvas.count("pointerdown"), 0);
});

// ---------------------------------------------------------------- attachDom

function domFakes() {
  const mk = () => {
    const t = { on: {} };
    t.addEventListener = (type, fn, opts) => { t.on[type] = { fn, opts }; };
    t.removeEventListener = (type, fn) => { if (t.on[type]?.fn === fn) delete t.on[type]; };
    t.fire = (type, ev) => {
      ev.prevented = false;
      ev.preventDefault = () => { ev.prevented = true; };
      t.on[type].fn(ev);
      return ev;
    };
    return t;
  };
  const win = mk();
  win.devicePixelRatio = 1;
  const canvas = mk();
  canvas.ownerDocument = { defaultView: win };
  canvas.getBoundingClientRect = () => ({ left: 10, top: 40, width: 800, height: 500 });
  canvas.captured = [];
  canvas.setPointerCapture = (id) => canvas.captured.push(id);
  return { win, canvas };
}

test("attachDom feeds the builder, positions relative to the canvas", () => {
  const { win, canvas } = domFakes();
  const b = new FrameBuilder();
  attachDom(canvas, b, { devicePixelRatio: 2 });
  win.fire("keydown", { key: "w", code: "KeyW" });
  canvas.fire("pointerdown", { button: 0, clientX: 110, clientY: 90, pointerType: "mouse", pointerId: 7 });
  win.fire("pointermove", { movementX: 4, movementY: 2, clientX: 114, clientY: 92 });
  win.fire("pointerup", { button: 0, clientX: 114, clientY: 92 });
  const wheel = canvas.fire("wheel", { deltaX: 0, deltaY: 100, deltaMode: 0 });
  const f = b.take(vp);
  assert.deepEqual(pick(f, "keys", "clicks", "released", "d", "pos", "wheel"), { keys: ["w"], clicks: [1], released: [1], d: [4, 2], pos: [104, 52], wheel: [0, 50] });
  assert.deepEqual(canvas.captured, [7]);
  assert.equal(wheel.prevented, true);
  assert.equal(canvas.on.wheel.opts.passive, false);
  win.fire("blur", {});
  assert.deepEqual(b.take(vp).up, ["w"]);
});

test("attachDom keeps Tab and Space from the page, leaves browser shortcuts and stops the context menu", () => {
  const { win, canvas } = domFakes();
  attachDom(canvas, new FrameBuilder());
  assert.equal(win.fire("keydown", { key: "Tab", code: "Tab" }).prevented, true);
  assert.equal(win.fire("keydown", { key: " ", code: "Space" }).prevented, true);
  assert.equal(win.fire("keydown", { key: "w", code: "KeyW", ctrlKey: true }).prevented, false);
  assert.equal(win.fire("keyup", { key: "w", code: "KeyW" }).prevented, false);
  assert.equal(canvas.fire("contextmenu", {}).prevented, true);
});

test("detach removes every listener", () => {
  const { win, canvas } = domFakes();
  const detach = attachDom(canvas, new FrameBuilder());
  assert.ok(Object.keys(win.on).length && Object.keys(canvas.on).length);
  detach();
  assert.deepEqual([Object.keys(win.on), Object.keys(canvas.on)], [[], []]);
});

test("a click while the pointer is locked reaches the builder; no capture is tried", () => {
  // Chromium throws InvalidStateError from setPointerCapture while the pointer is locked.
  const { win, canvas } = domFakes();
  canvas.ownerDocument.pointerLockElement = canvas;
  canvas.setPointerCapture = () => { throw new Error("InvalidStateError"); };
  const b = new FrameBuilder();
  attachDom(canvas, b);
  canvas.fire("pointerdown", { button: 0, clientX: 410, clientY: 290, pointerType: "mouse", pointerId: 1 });
  assert.deepEqual(b.take(vp).clicks, [1]);
});
