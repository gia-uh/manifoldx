// The dev loop's side of the page under manifoldx run --web: the stream of versions from the
// dev server, the reload's status and the overlay's text, the calls the server makes to the
// tab, the lines the tab sends to the server's log, and the look composite. browser.js imports
// it only when the page has <meta name="mx-dev">; status and overlayText are pure, for Node's
// tests.

// The reload's status: the running version, its phase ("live", "reloading", "failed",
// "paused", "offline") and the text that goes with it (a failure's line, a pause's error).
export const INITIAL = Object.freeze({ version: null, phase: "live", text: null });

// status(model, message) -> model, over the stream's messages and the tab's own:
// {type: "version", version}, {type: "reloading"}, {type: "failed", running, text},
// {type: "paused", text}, {type: "offline"}. A version below the running one is ignored (two
// saves close together); the running one again only clears offline (the stream came back).
export function status(model, message) {
  switch (message.type) {
    case "version": {
      const n = message.version;
      if (model.version !== null && n < model.version) return model;
      if (n === model.version) return model.phase === "offline" ? { ...model, phase: "live", text: null } : model;
      return { version: n, phase: "live", text: null };
    }
    case "reloading":
      return { ...model, phase: "reloading", text: null };
    case "failed":
      return { ...model, phase: "failed", text: message.text };
    case "paused":
      return { ...model, phase: "paused", text: message.text };
    case "offline":
      return { ...model, phase: "offline" };
    default:
      return model;
  }
}

// overlayText(model) -> {status, panel}: the overlay's status line (#mx-dev) and the error
// panel's text (#mx-error): null hides the panel, undefined leaves it as it is.
export function overlayText(model) {
  const v = model.version;
  switch (model.phase) {
    case "reloading":
      return { status: `v${v + 1} · reloading`, panel: undefined };
    case "failed":
      return { status: `v${v} · reload failed, running v${v}`, panel: `${model.text}\nFix it and save: the tab keeps running v${v}.` };
    case "paused":
      return { status: `v${v} · paused until the next edit`, panel: model.text };
    case "offline":
      return { status: `v${v} · dev server gone, retrying`, panel: undefined };
    default:
      return { status: `v${v} · live`, panel: null };
  }
}

// POST a JSON body to the dev server; errors are dropped (the server may be gone).
export function post(path, body) {
  return fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    .catch(() => {});
}

// A line in the dev server's log, as `manifoldx log` shows it.
export function log(text) {
  return post("api/web/log", [{ level: "info", text }]);
}

// The tab's lines for the server's log, sent together every 250 ms: the app's engine.log
// lines as they are, the console's warnings and errors with the prefix "tab: ".
const LOG_EVERY = 250;
let pending = [], flushing = null;
function queueLine(level, text) {
  pending.push({ level, text });
  flushing ??= setTimeout(() => {
    flushing = null;
    const lines = pending;
    pending = [];
    post("api/web/log", lines);
  }, LOG_EVERY);
}

// engine.log(text) from the app's code, in the tab.
export function appLog(text) {
  queueLine("info", text);
}

// console.warn, console.error, uncaught errors and unhandled rejections, to the server's log.
export function forwardConsole(win = globalThis) {
  const text = (x) => (x instanceof Error ? `${x.name}: ${x.message}` : typeof x === "string" ? x : String(x));
  for (const level of ["warn", "error"]) {
    const original = console[level].bind(console);
    console[level] = (...args) => {
      original(...args);
      queueLine(level, `tab: ${args.map(text).join(" ")}`);
    };
  }
  win.addEventListener("error", (e) => queueLine("error", `tab: ${e.message} (${e.filename}:${e.lineno})`));
  win.addEventListener("unhandledrejection", (e) => queueLine("error", `tab: unhandled rejection: ${text(e.reason)}`));
}

// lookPng(canvas, guiLayer, render) -> Promise<Blob>: what manifoldx look shows, a PNG of the
// canvas's drawing buffer with the HTML GUI painted over it (Q5). render() draws the current
// frame, and the canvas is copied in the same task, before the WebGL buffer is cleared. Each
// visible GUI element is painted from its box and computed style: its background, its border,
// then each line of its own text in its font and colour, scaled by the drawing buffer's ratio
// to CSS pixels. The text and the layout match the HTML; the pixels need not.
export function lookPng(canvas, guiLayer, render) {
  render();
  const out = document.createElement("canvas");
  out.width = canvas.width;
  out.height = canvas.height;
  const ctx = out.getContext("2d");
  ctx.drawImage(canvas, 0, 0);
  const box = canvas.getBoundingClientRect();
  const k = canvas.width / (box.width || canvas.width);
  for (const el of guiLayer.querySelectorAll("*")) {
    const r = el.getBoundingClientRect(), css = getComputedStyle(el);
    if (!r.width || !r.height || css.display === "none" || css.visibility === "hidden") continue;
    const x = (r.left - box.left) * k, y = (r.top - box.top) * k, w = r.width * k, h = r.height * k;
    const radius = parseFloat(css.borderTopLeftRadius) * k || 0;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, radius);
    ctx.fillStyle = css.backgroundColor;
    ctx.fill();
    const border = parseFloat(css.borderTopWidth) * k || 0;
    if (border) {
      ctx.lineWidth = border;
      ctx.strokeStyle = css.borderTopColor;
      ctx.beginPath();
      ctx.roundRect(x + border / 2, y + border / 2, w - border, h - border, Math.max(0, radius - border / 2));
      ctx.stroke();
    }
    ctx.font = `${css.fontStyle} ${css.fontWeight} ${parseFloat(css.fontSize) * k}px ${css.fontFamily}`;
    ctx.fillStyle = css.color;
    ctx.textBaseline = "middle";
    for (const node of el.childNodes) {
      if (node.nodeType !== Node.TEXT_NODE || !node.textContent) continue;
      const range = document.createRange();
      range.selectNodeContents(node);
      const lines = node.textContent.split("\n"), rects = [...range.getClientRects()];
      lines.forEach((line, i) => {
        const at = rects[i];
        if (at) ctx.fillText(line, (at.left - box.left) * k, (at.top - box.top + at.height / 2) * k);
      });
    }
  }
  return new Promise((resolve, reject) => out.toBlob((b) => (b ? resolve(b) : reject(new Error("the look could not be encoded"))), "image/png"));
}

// connect(mx, {url, onVersion, onFailed, onCall, onOffline}): follow the dev server's stream.
// Each version and failure goes to its handler; each call to onCall(kind, body), whose result
// (or promise of one) is the reply: {ok, value, state}, or a Blob posted as it is (a look).
// EventSource reconnects by itself when the server comes back. Returns the EventSource.
export function connect(mx, { url = "api/web/stream", onVersion, onFailed, onCall, onOffline }) {
  const id = (globalThis.crypto?.randomUUID?.() ?? String(Math.random()).slice(2));
  const source = new EventSource(`${url}?tab=${encodeURIComponent(id)}`);
  source.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "version") onVersion(msg);
    else if (msg.type === "failed") onFailed(msg);
    else if (msg.type === "call") answer(msg);
  };
  source.onerror = () => onOffline?.();
  async function answer(msg) {
    let reply;
    try {
      reply = await onCall(msg.kind, msg.body);
    } catch (err) {
      reply = { ok: false, error: String(err?.message ?? err) };
    }
    const path = `api/web/reply/${msg.id}`;
    if (reply instanceof Blob) fetch(path, { method: "POST", headers: { "Content-Type": "image/png" }, body: reply }).catch(() => {});
    else post(path, reply);
  }
  mx.tabId = id;
  return source;
}
