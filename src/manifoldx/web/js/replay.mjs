// Load a folder `manifoldx export --web` wrote, in Node: the app's modules (app/*.js), the
// state it starts from (state.json) and a headless engine. Run as a script, it prints the
// roots' labels, engine.state and what the GUI shows, as JSON; with --replay it reads
// {frames, paths, dt} from stdin and prints the trace replay() makes, as JSON (the form is
// manifoldx.web.replay's).

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { guiValues, loadApp, step } from "./manifoldx.js";

// The folder's app, {engine, roots, modules}; options go to createEngine over the exported ones.
export async function load(folder = new URL(".", import.meta.url), options = {}) {
  const state = JSON.parse(readFileSync(new URL("state.json", folder), "utf8"));
  const modules = {};
  for (const name of state.modules) modules[name] = await import(new URL(`app/${name}.js`, folder).href);
  return { ...loadApp(state, modules, options), modules };
}

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

// path's value: its first name is a global of the app's main module or, when it has none by
// that name, a key of engine.state; then attributes and dict keys.
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

// One trace entry per frame: the frame goes to engine.input, the systems run once with dt,
// then {camera, gui, log, state, paths} are read, as plain JSON values.
export function replay(app, frames, paths, dt = 1 / 60) {
  const { engine } = app;
  const trace = [];
  for (const frame of frames) {
    const start = engine.log.length;
    engine.input.setFrame(frame);
    step(app, dt);
    const entry = {
      camera: { position: engine.camera.position, target: engine.camera.target },
      gui: guiValues(app),
      log: engine.log.slice(start),
      state: engine.state,
      paths: Object.fromEntries(paths.map((p) => [p, pathValue(app, p)])),
    };
    trace.push(JSON.parse(toJSON(entry)));
  }
  return trace;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const app = await load();
  if (process.argv.includes("--replay")) {
    const { frames, paths, dt } = JSON.parse(readFileSync(0, "utf8"));
    console.log(toJSON(replay(app, frames, paths, dt)));
  } else {
    console.log(toJSON({ roots: app.roots.map((r) => r.label), state: app.engine.state, gui: guiValues(app) }));
  }
}
