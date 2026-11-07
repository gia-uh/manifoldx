// A JS error in the app's transpiled code, named by its Python file and line through the
// export's source maps. The export comes from tests/web/test_node_suites.py's fixture.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { describe, lineTable, pythonFrames } from "../../../src/manifoldx/web/js/sourcemap.js";

const EXPORT = process.env.MX_SOURCEMAP_EXPORT;
const skip = !EXPORT && "MX_SOURCEMAP_EXPORT not set";

// {"app/<name>.js": {source, lines}} for the export's modules, as browser.js builds it.
function maps(folder) {
  const state = JSON.parse(readFileSync(new URL("state.json", folder), "utf8"));
  const out = {};
  for (const name of state.modules) {
    const map = JSON.parse(readFileSync(new URL(`app/${name}.js.map`, folder), "utf8"));
    out[`app/${name}.js`] = { source: map.sources[0], lines: lineTable(map) };
  }
  return out;
}

test("lineTable decodes each generated line's first segment, 0 where unmapped", () => {
  // Line 0: nothing; line 1: source line 3 (0-based 2); line 2: back one to 2; line 3: two segments.
  const table = lineTable({ mappings: ";AAEA;AADA;AAAA,IAEA" });
  assert.deepEqual([...table], [0, 3, 2, 2]);
  assert.ok(table instanceof Int32Array);
});

test("a system's error names the helper's line, then the system's call", { skip }, async () => {
  const folder = pathToFileURL(EXPORT + "/");
  const expect = JSON.parse(readFileSync(new URL("../expect.json", folder), "utf8"));
  const { load } = await import(new URL("replay.mjs", folder).href);
  const app = await load(folder);
  const system = app.roots.find((r) => r.kind === "system");
  let err;
  try {
    system.fn(null, 1 / 60);
  } catch (e) {
    err = e;
  }
  assert.ok(err, "the system did not throw");
  const m = maps(folder);
  const text = describe(err, m);
  assert.ok(text.startsWith(`mxsm_helper.py:${expect.helper} in far: TypeError: `), text);
  assert.equal(text.split("\n")[1], `  called from failing.py:${expect.call} in tick`);
  const frames = pythonFrames(err.stack, m);
  assert.deepEqual(frames.slice(0, 2), [
    { file: "mxsm_helper.py", line: expect.helper, fn: "far" },
    { file: "failing.py", line: expect.call, fn: "tick" },
  ]);
});

test("Firefox's stack form parses too, and frames outside app/ are dropped", () => {
  const m = {
    "app/__main__.js": { source: "app.py", lines: Int32Array.from([0, 5, 6, 7]) },
    "app/lib.js": { source: "lib.py", lines: Int32Array.from([0, 0, 12]) },
  };
  const stack = [
    "helper@http://127.0.0.1:8000/app/lib.js:3:9",
    "Thing.walk@http://127.0.0.1:8000/app/__main__.js:2:4",
    "step@http://127.0.0.1:8000/manifoldx.js:278:5",
    "@http://127.0.0.1:8000/app/__main__.js:4:1",
  ].join("\n");
  assert.deepEqual(pythonFrames(stack, m), [
    { file: "lib.py", line: 12, fn: "helper" },
    { file: "app.py", line: 5, fn: "walk" },
    { file: "app.py", line: 7, fn: "<lambda>" },
  ]);
});

test("V8's form, with an async frame and no function name", () => {
  const m = { "app/__main__.js": { source: "app.py", lines: Int32Array.from([0, 5, 6, 7]) } };
  const stack = [
    "TypeError: boom",
    "    at Rig.move (http://h/app/__main__.js:3:10)",
    "    at async http://h/app/__main__.js:2:1",
    "    at step (http://h/manifoldx.js:278:5)",
  ].join("\n");
  assert.deepEqual(pythonFrames(stack, m), [
    { file: "app.py", line: 6, fn: "move" },
    { file: "app.py", line: 5, fn: "<lambda>" },
  ]);
});

test("an error with no Python frame keeps its own message", () => {
  const err = new RangeError("too far");
  err.stack = "RangeError: too far\n    at step (http://h/manifoldx.js:1:1)";
  assert.equal(describe(err, {}), "RangeError: too far");
});
