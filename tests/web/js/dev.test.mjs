// The dev loop's status in the tab under manifoldx run --web: status(model, message) as a pure
// function over the stream's messages.

import assert from "node:assert/strict";
import test from "node:test";

import { INITIAL, status } from "../../../src/manifoldx/web/js/dev.js";

const run = (...messages) => messages.reduce(status, INITIAL);
const v = (version) => ({ type: "version", version });

test("the first version is live", () => {
  assert.deepEqual(run(v(1)), { version: 1, phase: "live", text: null });
});

test("reloading, then version 4, is live on 4", () => {
  const m = run(v(3), { type: "reloading" });
  assert.deepEqual(m, { version: 3, phase: "reloading", text: null });
  assert.deepEqual(status(m, v(4)), { version: 4, phase: "live", text: null });
});

test("version 3 after version 4 is ignored: two saves close together", () => {
  const m = run(v(3), v(4));
  assert.equal(status(m, v(3)), m);
  assert.equal(status(m, v(4)), m);
});

test("failed keeps the running version, with its text", () => {
  const text = "dev_orbit.py:15: SyntaxError: '(' was never closed";
  const m = run(v(3), { type: "reloading" }, { type: "failed", running: 3, text });
  assert.deepEqual(m, { version: 3, phase: "failed", text });
  assert.deepEqual(status(m, v(4)), { version: 4, phase: "live", text: null });
});

test("a stream error is offline; the next version clears it, the same one too", () => {
  const m = run(v(3), { type: "offline" });
  assert.deepEqual(m, { version: 3, phase: "offline", text: null });
  assert.deepEqual(status(m, v(3)), { version: 3, phase: "live", text: null }); // the stream came back
  assert.deepEqual(status(m, v(5)), { version: 5, phase: "live", text: null });
});
