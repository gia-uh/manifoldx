// The page's bootstrap: load the export (state, the app's modules, the scene with a progress
// line), draw it with three.js, and run the app's systems every animation frame on the input
// the DOM gave, through the same Frame the replay feeds. window.__mx is how tests drive it:
// {ready, app, scene, gui, frames, trace, error, paths, manual(on), tick(dt), loadStats}.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import { buildGui } from "./gui.js";
import { FrameBuilder, PointerLock, attachDom } from "./input.js";
import { guiValues, loadApp, step, traceEntry } from "./manifoldx.js";
import { RateMeter, frameDt, progressLine } from "./progress.js";
import { buildScene } from "./scene.js";
import { describe, lineTable } from "./sourcemap.js";

const HINT = "serve this folder over HTTP (python -m http.server)";
const $ = (id) => document.getElementById(id);
const meta = (name) => document.querySelector(`meta[name="${name}"]`)?.content;

let resolveReady, rejectReady;
const mx = {
  ready: new Promise((res, rej) => { resolveReady = res; rejectReady = rej; }),
  app: null,
  scene: null,
  gui: null,
  frames: [],
  trace: [],
  error: null,
  paths: [],
  loadStats: { bytes: 0 },
  manual: () => {},
  tick: () => null,
};
mx.ready.catch(() => {}); // a load error is shown on the page; a test reads the rejection
window.__mx = mx;

// The error panel: the message, also in __mx.error and the console.
function showError(text) {
  mx.error = text;
  console.error(text);
  const panel = $("mx-error");
  panel.textContent = text;
  panel.hidden = false;
}

async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status} ${r.statusText}`);
  return r;
}

// scene.glb as an ArrayBuffer, the progress line following it. The page knows the scene's
// size and its gzipped size: with gzip on the wire, the line counts the wire's bytes.
async function fetchScene() {
  const bytes = Number(meta("scene-bytes")) || 0;
  const gzipBytes = Number(meta("scene-gzip-bytes")) || 0;
  const r = await fetchOk("scene.glb");
  const gzip = /gzip/.test(r.headers.get("content-encoding") ?? "");
  const download = bytes ? (gzip && gzipBytes ? gzipBytes : bytes) : null;
  const bar = document.querySelector("#mx-progress .bar");
  const text = document.querySelector("#mx-progress .text");
  if (!r.body) {
    const buf = await r.arrayBuffer();
    mx.loadStats.bytes = buf.byteLength;
    return buf;
  }
  const reader = r.body.getReader();
  const meter = new RateMeter();
  const t0 = performance.now();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const fraction = bytes ? Math.min(received / bytes, 1) : null;
    meter.add((performance.now() - t0) / 1000, fraction === null ? received : fraction * download);
    bar.style.width = fraction === null ? "100%" : `${(fraction * 100).toFixed(1)}%`;
    text.textContent = progressLine({ fraction, download, rate: meter.rate(), received });
  }
  mx.loadStats.bytes = received;
  const out = new Uint8Array(received);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out.buffer;
}

async function start() {
  const canvas = $("mx-canvas");
  const gl = canvas.getContext("webgl2", { antialias: true });
  if (!gl) {
    const err = new Error("this browser has no WebGL2, which the page needs to draw the scene");
    err.noHint = true;
    throw err;
  }

  const state = await (await fetchOk("state.json")).json();
  const modules = {}, maps = {};
  for (const name of state.modules) {
    modules[name] = await import(new URL(`app/${name}.js`, import.meta.url).href);
    const map = await (await fetchOk(`app/${name}.js.map`)).json();
    maps[`app/${name}.js`] = { source: map.sources[0], lines: lineTable(map) };
  }
  const gltf = await new GLTFLoader().parseAsync(await fetchScene(), "./");

  const renderer = new THREE.WebGLRenderer({ canvas, context: gl, antialias: true });
  let ratio = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(ratio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

  const scene = await buildScene(THREE, gltf, renderer);
  const gui = buildGui($("mx-gui"), state.gui);
  const builder = new FrameBuilder({ devicePixelRatio: window.devicePixelRatio || 1 });
  const lock = new PointerLock(canvas, document, builder);
  const app = {
    ...loadApp(state, modules, {
      hooks: { setMaterial: (entity, id) => scene.setMaterial(entity, id), capture: (on) => lock.request(on) },
    }),
    modules,
  };
  lock.input = app.engine.input;
  attachDom(canvas, builder);
  Object.assign(mx, { app, scene, gui });

  const { engine } = app;
  let size = [0, 0], guiH = null;

  // One frame as the loop runs it: the frame the DOM built, the systems, the GUI, the camera,
  // the render. Returns {frame, start} (start: where this frame's log begins), or null when the
  // app's code threw, which pauses the page.
  function runFrame(dt) {
    if (mx.error !== null) return null;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const frame = builder.take([w, h]);
    engine.log.length = 0; // the log is the trace's record of one frame
    try {
      engine.gui.pointer_over_gui = gui.pointerOver;
      engine.input.setFrame(frame);
      step(app, dt);
      gui.sync(engine.gui.widgets);
      gui.update(guiValues(app));
    } catch (err) {
      renderer.setAnimationLoop(null);
      showError(describe(err, maps));
      return null;
    }
    if (engine.h !== guiH) gui.rescale((guiH = engine.h));
    const r = Math.min(window.devicePixelRatio || 1, 2);
    if (r !== ratio || w !== size[0] || h !== size[1]) {
      renderer.setPixelRatio((ratio = r));
      renderer.setSize(w, h, false);
      size = [w, h];
    }
    scene.sync(engine);
    renderer.render(scene.scene, scene.camera);
    return { frame, start: 0 };
  }

  let last = null;
  const loop = (now) => {
    const dt = last === null ? 1 / 60 : frameDt(now, last);
    last = now;
    runFrame(dt);
  };
  mx.manual = (on) => {
    last = null;
    renderer.setAnimationLoop(on ? null : loop);
  };
  mx.tick = (dt) => {
    const ran = runFrame(dt);
    if (ran === null) return null;
    const entry = traceEntry(app, mx.paths, ran.start);
    mx.frames.push(ran.frame);
    mx.trace.push(entry);
    return { frame: ran.frame, entry };
  };

  $("mx-progress").hidden = true;
  renderer.setAnimationLoop(loop);
}

start().then(
  () => resolveReady(mx),
  (err) => {
    $("mx-progress").hidden = true;
    showError(err.noHint ? err.message : `${err.message}\nThe page could not load. If it was opened as a file, ${HINT}.`);
    rejectReady(err);
  },
);
