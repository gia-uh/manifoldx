// The dev loop's side of the page under manifoldx run --web: the stream of versions from the
// dev server, the reload's status, and the calls the server makes to the tab. browser.js
// imports it only when the page has <meta name="mx-dev">; status is pure, for Node's tests.

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
