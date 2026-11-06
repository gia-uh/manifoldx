// Load a folder `manifoldx export --web` wrote, in Node: the app's modules (app/*.js), the
// state it starts from (state.json) and a headless engine. Run as a script, it prints the
// roots' labels, engine.state and what the GUI shows, as JSON.

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { guiValues, loadApp } from "./manifoldx.js";

// The folder's app, {engine, roots}; options go to createEngine over the exported ones.
export async function load(folder = new URL(".", import.meta.url), options = {}) {
  const state = JSON.parse(readFileSync(new URL("state.json", folder), "utf8"));
  const modules = {};
  for (const name of state.modules) modules[name] = await import(new URL(`app/${name}.js`, folder).href);
  return loadApp(state, modules, options);
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

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const app = await load();
  console.log(toJSON({ roots: app.roots.map((r) => r.label), state: app.engine.state, gui: guiValues(app) }));
}
