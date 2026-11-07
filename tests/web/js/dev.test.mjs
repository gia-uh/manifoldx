// The dev loop's status in the tab under manifoldx run --web: status(model, message) as a pure
// function over the stream's messages.

import assert from "node:assert/strict";
import test from "node:test";

import { INITIAL, overlayText, status } from "../../../src/manifoldx/web/js/dev.js";

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

// overlayText(model) -> {status, panel}: the status line, and the error panel's text (null:
// hidden, undefined: left as it is).
test("running: the version is live and the panel hidden", () => {
  assert.deepEqual(overlayText(run(v(3))), { status: "v3 · live", panel: null });
});

test("from the state request to the swap: the next version is reloading, the panel unchanged", () => {
  assert.deepEqual(overlayText(run(v(3), { type: "reloading" })), { status: "v4 · reloading", panel: undefined });
});

test("an edit that failed: the running version, the failure's line and what to do", () => {
  const text = "dev_orbit.py:12: SyntaxError: '(' was never closed";
  const m = run(v(3), { type: "reloading" }, { type: "failed", running: 3, text });
  assert.deepEqual(overlayText(m), {
    status: "v3 · reload failed, running v3",
    panel: "dev_orbit.py:12: SyntaxError: '(' was never closed\nFix it and save: the tab keeps running v3.",
  });
});

test("an exception in a frame: paused, the panel shows describe's text", () => {
  const text = "raises.py:26 in tick: TypeError: x is null\n  called from raises.py:30 in walk";
  assert.deepEqual(overlayText(run(v(3), { type: "paused", text })), {
    status: "v3 · paused until the next edit",
    panel: text,
  });
});

test("the stream is down: the server is gone, the panel unchanged", () => {
  assert.deepEqual(overlayText(run(v(3), { type: "offline" })), {
    status: "v3 · dev server gone, retrying",
    panel: undefined,
  });
});
