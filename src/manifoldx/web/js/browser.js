// The page's bootstrap: load the export (state, the app's modules, the scene with a progress
// line), draw it with three.js, and run the app's systems every animation frame on the input
// the DOM gave, through the same Frame the replay feeds. window.__mx is how tests drive it:
// {ready, app, scene, gui, frames, trace, error, paths, manual(on), tick(dt), loadStats}, and
// under manifoldx run --web (<meta name="mx-dev">) also {version, swaps, dev, status, swap(info)}.
//
// The renderer, the canvas, the input (FrameBuilder, PointerLock, the DOM listeners) live as
// long as the page; a Version (the state, the app's modules and their maps, the app) is what a
// save replaces, in place, without reloading the page.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import { buildGui } from "./gui.js";
import { FrameBuilder, PointerLock, attachDom } from "./input.js";
import { encodeState, fire, fromJSON, guiValues, loadApp, mergePatch, stateView, step, traceEntry } from "./manifoldx.js";
import { RateMeter, frameDt, progressLine } from "./progress.js";
import { buildScene } from "./scene.js";
import { describe, lineTable } from "./sourcemap.js";

const HINT = "serve this folder over HTTP (python -m http.server)";
const $ = (id) => document.getElementById(id);
const meta = (name) => document.querySelector(`meta[name="${name}"]`)?.content;
const DEV = document.querySelector('meta[name="mx-dev"]') !== null;
const FREEZE_MAX = 30000; // ms a reload may hold the tab frozen without a version or a failure

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
  version: null,
  swaps: [],
  swap: async () => false,
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

// The scene as an ArrayBuffer, the progress line following it. The page knows the scene's
// size and its gzipped size: with gzip on the wire, the line counts the wire's bytes.
async function fetchScene(url, bytes, gzipBytes) {
  const r = await fetchOk(url);
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

// A version of the app: its state, modules and their maps, from base ("" for an export,
// "v4/" under run --web). The app is made when the version runs (makeApp).
export async function loadVersion(base, info = null) {
  const state = await (await fetchOk(`${base}state.json`)).json();
  const ecs = await (await fetchOk(`${base}ecs.bin`)).arrayBuffer();
  const modules = {}, maps = {};
  for (const name of state.modules) {
    modules[name] = await import(new URL(`${base}app/${name}.js`, import.meta.url).href);
    const map = await (await fetchOk(`${base}app/${name}.js.map`)).json();
    maps[`app/${name}.js`] = { source: map.sources[0], lines: lineTable(map) };
  }
  return { state, ecs, modules, maps, app: null, info };
}

async function start() {
  const canvas = $("mx-canvas");
  const gl = canvas.getContext("webgl2", { antialias: true });
  if (!gl) {
    const err = new Error("this browser has no WebGL2, which the page needs to draw the scene");
    err.noHint = true;
    throw err;
  }

  // Under run --web the dev server names the version and its scene; an export is one folder.
  const info = DEV ? await (await fetchOk("api/web/current")).json() : null;
  let running = await loadVersion(info ? info.base : "", info);
  const sceneBytes = info ? info.scene_bytes : Number(meta("scene-bytes")) || 0;
  const sceneGzip = info ? info.scene_gzip_bytes : Number(meta("scene-gzip-bytes")) || 0;
  const loader = new GLTFLoader();
  const gltf = await loader.parseAsync(await fetchScene(info ? info.scene : "scene.glb", sceneBytes, sceneGzip), "./");

  const renderer = new THREE.WebGLRenderer({ canvas, context: gl, antialias: true });
  let ratio = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(ratio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

  let scene = await buildScene(THREE, gltf, renderer);
  let gui = buildGui($("mx-gui"), running.state.gui);
  const builder = new FrameBuilder({ devicePixelRatio: window.devicePixelRatio || 1 });
  const lock = new PointerLock(canvas, document, builder);
  const hooks = {
    setMaterial: (entity, id) => scene.setMaterial(entity, id),
    capture: (on) => lock.request(on),
    log: (text) => mx.dev?.appLog(text), // engine.log: to the dev server's log, nowhere in an export
  };
  const makeApp = (version) => ({ ...loadApp(version.state, version.modules, { hooks, ecs: version.ecs }), modules: version.modules });
  running.app = makeApp(running);
  lock.input = running.app.engine.input;
  attachDom(canvas, builder);
  Object.assign(mx, { app: running.app, scene, gui, version: info ? info.version : null });

  let size = [0, 0], guiH = null;
  let frozen = null; // while a reload builds (Q2): the timer that unfreezes after FREEZE_MAX
  let manual = false;
  const events = []; // [name, payload] an agent emitted, fired at the start of the next frame
  let onPause = () => {}; // dev mode: an exception in a frame pauses until the next edit

  // One frame as the loop runs it: the frame the DOM built, the systems, the GUI, the camera,
  // the render. Returns {frame, start} (start: where this frame's log begins), or null when the
  // app's code threw, which pauses the page, or while a reload holds the tab frozen.
  function runFrame(dt) {
    if (mx.error !== null || frozen !== null) return null;
    const { app } = running, { engine } = app;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const frame = builder.take([w, h]);
    engine.calls.length = 0; // the trace's record of one frame
    try {
      engine.gui.pointer_over_gui = gui.pointerOver;
      engine.input.setFrame(frame);
      for (const [name, payload] of events.splice(0)) fire(app, name, payload);
      step(app, dt);
      gui.sync(engine.gui.widgets);
      gui.update(guiValues(app));
    } catch (err) {
      renderer.setAnimationLoop(null);
      const text = describe(err, running.maps);
      showError(text);
      onPause(text);
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
    manual = on;
    renderer.setAnimationLoop(on ? null : loop);
  };
  mx.tick = (dt) => {
    const ran = runFrame(dt);
    if (ran === null) return null;
    const entry = traceEntry(running.app, mx.paths, ran.start);
    mx.frames.push(ran.frame);
    mx.trace.push(entry);
    return { frame: ran.frame, entry };
  };

  const freeze = () => {
    clearTimeout(frozen);
    frozen = setTimeout(thaw, FREEZE_MAX);
  };
  function thaw() {
    clearTimeout(frozen);
    frozen = null;
    last = null; // the frozen time is not one long frame
  }

  // Swap to the version info names, in place: everything new is loaded and built before the
  // running version is touched, so a version that fails to load leaves the running one as it
  // was. Returns whether it swapped.
  mx.swap = async (next) => {
    if (mx.version !== null && next.version <= mx.version) return false;
    const t0 = performance.now();
    const kept = next.scene_sha === running.info?.scene_sha;
    let version, built = null;
    try {
      version = await loadVersion(next.base, next);
      if (!kept) {
        const buf = await (await fetchOk(next.scene)).arrayBuffer();
        built = await buildScene(THREE, await loader.parseAsync(buf, "./"), renderer);
      }
      version.app = makeApp(version);
    } catch (err) {
      showSwapError(`${err.message}\nThe tab keeps running v${mx.version}.`);
      thaw();
      return false;
    }
    if (kept) scene.resetMaterials();
    else {
      scene.dispose();
      scene = built;
    }
    gui.destroy();
    gui = buildGui($("mx-gui"), version.state.gui);
    guiH = null;
    lock.input = version.app.engine.input;
    if (lock.locked) version.app.engine.input.lockHeld();
    running = version;
    Object.assign(mx, { app: version.app, scene, gui, version: next.version, error: null });
    $("mx-error").hidden = true;
    thaw();
    if (!manual) renderer.setAnimationLoop(loop);
    const ms = Math.round(performance.now() - t0);
    mx.swaps.push({ version: next.version, ms, scene: kept ? "kept" : "new" });
    mx.dev?.log(`tab: v${next.version} in ${ms} ms (scene ${kept ? "kept" : "new"})`);
    return true;
  };

  // A swap that failed: the error panel shows it as a load error does, the running version
  // goes on.
  function showSwapError(text) {
    console.error(text);
    const panel = $("mx-error");
    panel.textContent = text;
    panel.hidden = false;
  }

  if (DEV) {
    const dev = await import("./dev.js");
    mx.dev = dev;
    dev.forwardConsole(window);
    const overlay = $("mx-dev");
    overlay.hidden = false;
    // The status line and the error panel follow the model (overlayText's rows).
    const say = (message) => {
      mx.status = dev.status(mx.status, message);
      const { status, panel } = dev.overlayText(mx.status);
      overlay.textContent = status;
      if (panel === null) $("mx-error").hidden = true;
      else if (panel !== undefined) {
        $("mx-error").textContent = panel;
        $("mx-error").hidden = false;
      }
    };
    mx.status = dev.INITIAL;
    say({ type: "version", version: mx.version });
    onPause = (text) => {
      say({ type: "paused", text });
      dev.log(`paused until the next edit: ${text.split("\n")[0]}`);
    };
    // A swap waits for the one before it, so versions apply in order.
    let swapping = Promise.resolve();
    dev.connect(mx, {
      onVersion: (msg) => {
        swapping = swapping.then(() => mx.swap(msg)).then((swapped) => {
          if (swapped || msg.version === mx.version) say(msg); // the same one: the stream came back
        });
      },
      onFailed: (msg) => {
        thaw();
        say(msg);
      },
      onOffline: () => say({ type: "offline" }),
      // The agent's commands. Every reply carries the tab's state (Q4); a look's is the PNG.
      onCall: (kind, body) => {
        const engine = running.app.engine;
        const reply = (value) => ({ ok: true, value, state: encodeState(engine.state) });
        switch (kind) {
          case "reload_state":
            freeze();
            say({ type: "reloading" });
            return reply(null);
          case "state":
            return reply(stateView(engine.state));
          case "patch":
            mergePatch(engine.state, fromJSON(body));
            return reply(stateView(engine.state));
          case "emit":
            events.push([body.name, fromJSON(body.payload)]);
            return reply(null);
          case "look":
            return dev.lookPng(canvas, $("mx-gui"), () => {
              scene.sync(engine);
              renderer.render(scene.scene, scene.camera);
            });
          default:
            return { ok: false, error: `the tab does not answer ${kind}` };
        }
      },
    });
    // The state a tab leaves with: a tab opened next starts from it (Q4).
    window.addEventListener("pagehide", () => {
      navigator.sendBeacon("api/web/state", JSON.stringify(encodeState(running.app.engine.state)));
    });
  }

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
