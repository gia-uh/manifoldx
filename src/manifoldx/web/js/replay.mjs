// Load a folder `manifoldx export --web` wrote, in Node: the app's modules (app/*.js), the
// state it starts from (state.json) and a headless engine. Run as a script, it prints the
// roots' labels, engine.state and what the GUI shows, as JSON; with --replay it reads
// {frames, paths, dt} from stdin and prints the trace replay() makes, as JSON (the form is
// manifoldx.web.replay's).

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { guiValues, loadApp, step, toJSON, traceEntry } from "./manifoldx.js";

// The folder's app, {engine, roots, modules}; options go to createEngine over the exported ones.
export async function load(folder = new URL(".", import.meta.url), options = {}) {
  const state = JSON.parse(readFileSync(new URL("state.json", folder), "utf8"));
  const modules = {};
  for (const name of state.modules) modules[name] = await import(new URL(`app/${name}.js`, folder).href);
  return { ...loadApp(state, modules, options), modules };
}

// One trace entry per frame: the frame goes to engine.input, the systems run once with dt,
// then traceEntry reads it.
export function replay(app, frames, paths, dt = 1 / 60) {
  const { engine } = app;
  const trace = [];
  for (const frame of frames) {
    const start = engine.log.length;
    engine.input.setFrame(frame);
    step(app, dt);
    trace.push(traceEntry(app, paths, start));
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
