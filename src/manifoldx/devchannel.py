"""The dev channel of `manifoldx run`: a page and an HTTP API beside the running app, for a person and an agent.

`manifoldx run` serves, on 127.0.0.1 only (default port 8770, the next free one if taken):

- `GET /`: a page with the frame on screen, the log as it grows, engine.state, and the app's events.
- `GET /api/state`: engine.state as JSON (`repr` for what JSON cannot hold).
- `PATCH /api/state`: a JSON merge patch (RFC 7386) applied to engine.state: objects merge, `null` deletes.
- `GET /api/events`: the events the app handles, each with its handlers' docstrings.
- `POST /api/events/<name>`: emit the event with the JSON body as its payload; it runs on the next frame.
- `GET /api/log?after=N&match=RE&wait=S&last=K`: log lines after sequence number N (or the last K), those
  matching the regex RE; with `wait`, block up to S seconds until one arrives.
- `GET /api/look`: the next frame presented, GUI included, as a PNG.
- `GET /api/metrics` (under `manifoldx run --debug`): each second's frame rate, frame times, memory and
  garbage collections, the last METRICS_KEEP seconds. Debug also logs the keys pressed, a frame slower than
  SLOW_FRAME, a collection slower than SLOW_GC, and the metrics every METRICS_LOG seconds.

Under `manifoldx run --web` the app runs in a browser tab (manifoldx.web.dev): `/` and every other path
serve the web root, the dev page moves to `/dev`, `/api/web/*` is the tab's stream and its answers, and
state, set, emit and look are forwarded to the tab.

Everything that reads or changes the app runs on the engine's thread between frames, so a request
never races the app. The log is also written to `.manifoldx/log.txt` (rotated at 256 KB, one old file
kept), and `.manifoldx/server.json` holds the URL the `manifoldx` commands use.
"""

import collections
import gc
import json
import logging
import os
import queue
import re
import select
import socket
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import sys

import numpy as np

FOLDER = ".manifoldx"
PORT = 8770
MAX_BYTES = 256_000
KEEP_LINES = 5000   # log lines the API keeps in memory
WAIT_MAX = 300.0    # seconds a log request may wait
LOOK_WAIT = 5.0     # seconds a look waits for a frame (none come while the app is paused)
SLOW_FRAME = 0.1    # seconds: under --debug a frame this long is logged at once
SLOW_GC = 0.02      # seconds: under --debug a garbage collection this long is logged at once
METRICS_LOG = 10.0  # seconds between metrics lines in the log under --debug
METRICS_KEEP = 120  # seconds of metrics /api/metrics keeps

# engine.log writes here. Only manifoldx run gives it somewhere to go; otherwise its lines are dropped.
LOG = logging.getLogger("manifoldx.app")
LOG.addHandler(logging.NullHandler())
LOG.setLevel(logging.INFO)
LOG.propagate = False


class Lines(logging.Handler):
    """The last KEEP_LINES log lines, numbered from 1, and a way to wait for the next."""

    def __init__(self):
        super().__init__()
        self.lines = collections.deque(maxlen=KEEP_LINES)
        self.seq = 0
        self.cv = threading.Condition()

    def emit(self, record):
        with self.cv:
            self.seq += 1
            self.lines.append({"seq": self.seq, "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                               "text": record.getMessage()})
            self.cv.notify_all()

    def read(self, after=None, match=None, wait=0.0, last=None):
        """Lines after `after` (or the last `last`), only those matching `match`, waiting up to `wait` for one."""
        pattern = re.compile(match) if match else None
        end = time.monotonic() + min(wait, WAIT_MAX)
        with self.cv:
            while True:
                if after is None:
                    found = list(self.lines)[-last:] if last else []
                    after = self.seq
                else:
                    found = [line for line in self.lines if line["seq"] > after]
                found = [line for line in found if pattern is None or pattern.search(line["text"])]
                left = end - time.monotonic()
                if found or left <= 0:
                    return {"lines": found, "next": self.seq}
                self.cv.wait(left)


def merge_patch(target, patch):
    """Apply a JSON merge patch (RFC 7386) to the dict `target` in place."""
    for key, value in patch.items():
        if value is None:
            target.pop(key, None)
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_patch(target[key], value)
        else:
            target[key] = value


def _rss_mb():
    """The process's resident memory in MB (Linux /proc; elsewhere the peak, from resource)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError):
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 1024


class Meter:
    """Under --debug: each second's frame rate, frame times, memory and garbage collections."""

    def __init__(self):
        self.history = collections.deque(maxlen=METRICS_KEEP)
        self.start = self.last = None
        self.frames, self.total, self.worst = 0, 0.0, 0.0
        self.split = {"update": 0.0, "acquire": 0.0, "render": 0.0}  # summed over the second
        self.gc_runs, self.gc_time, self.gc_at = [0, 0, 0], 0.0, None
        self.logged = 0.0
        gc.callbacks.append(self._gc)

    def close(self):
        gc.callbacks.remove(self._gc)

    def _gc(self, phase, info):
        if phase == "start":
            self.gc_at = time.perf_counter()
        elif self.gc_at is not None:
            took, self.gc_at = time.perf_counter() - self.gc_at, None
            self.gc_runs[info["generation"]] += 1
            self.gc_time += took
            if took >= SLOW_GC:
                LOG.info("slow gc: generation %d took %.0f ms, %d collected", info["generation"], 1000 * took,
                         info["collected"])

    def frame(self, now, split=None):
        """After each frame, with the time it ended and engine.frame_split (seconds per part)."""
        if self.last is not None:
            dt = now - self.last
            self.frames += 1
            self.total += dt
            for part, seconds in (split or {}).items():
                self.split[part] += seconds
            self.worst = max(self.worst, dt)
            if dt >= SLOW_FRAME:
                LOG.info("slow frame %.0f ms", 1000 * dt)
        self.last = now
        if self.start is None:
            self.start = now
        elif now - self.start >= 1.0:
            m = {"time": time.strftime("%H:%M:%S"), "fps": round(self.frames / (now - self.start), 1),
                 "frame_ms": round(1000 * self.total / max(self.frames, 1), 1), "worst_ms": round(1000 * self.worst, 1),
                 "rss_mb": round(_rss_mb(), 1), "gc": {"runs": list(self.gc_runs), "ms": round(1000 * self.gc_time, 1)}}
            n = max(self.frames, 1)
            parts = {part: round(1000 * seconds / n, 1) for part, seconds in self.split.items()}
            parts["idle"] = round(max(0.0, m["frame_ms"] - sum(parts.values())), 1)  # between frames: the loop
            m["split_ms"] = parts
            self.history.append(m)
            if now - self.logged >= METRICS_LOG:
                self.logged = now
                LOG.info("fps %.0f, frame %.1f ms (worst %.0f; update %.1f, render %.1f, acquire %.1f, idle %.1f), "
                         "%.0f MB, gc %s in %.0f ms", m["fps"], m["frame_ms"], m["worst_ms"], parts["update"],
                         parts["render"], parts["acquire"], parts["idle"], m["rss_mb"],
                         "/".join(map(str, self.gc_runs)), m["gc"]["ms"])
            self.start, self.frames, self.total, self.worst = now, 0, 0.0, 0.0
            self.split = dict.fromkeys(self.split, 0.0)
            self.gc_runs, self.gc_time = [0, 0, 0], 0.0


class Channel:
    """The dev channel of one `manifoldx run`: the log, the server, and the jobs the engine's thread runs."""

    def __init__(self, root, hot, port=PORT, debug=False, web=None):
        self.web = web  # under --web, the WebReloader: its tab runs the app
        self.dir = Path(root) / FOLDER
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hot = hot  # the HotReloader: its engine is the one running now, across reloads
        self.file = RotatingFileHandler(self.dir / "log.txt", maxBytes=MAX_BYTES, backupCount=1, encoding="utf-8")
        self.file.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        self.lines = Lines()
        LOG.addHandler(self.file)
        LOG.addHandler(self.lines)
        self.jobs = queue.SimpleQueue()
        self.keys = frozenset()
        self.meter = Meter() if debug else None  # with it, the keys pressed are logged too
        self.server = _bind(port, self)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        (self.dir / "server.json").write_text(json.dumps({"url": self.url, "pid": os.getpid()}))
        threading.Thread(target=self.server.serve_forever, name="manifoldx-dev", daemon=True).start()

    def close(self):
        if self.meter is not None:
            self.meter.close()
        if self.web is not None:
            self.web.tab.close()
        self.server.shutdown()
        self.server.server_close()
        (self.dir / "server.json").unlink(missing_ok=True)
        LOG.removeHandler(self.file)
        LOG.removeHandler(self.lines)
        self.file.close()

    # -- the engine's thread --

    def run_jobs(self):
        """Run what requests asked of the app; the reloader calls it every frame, paused or not."""
        while True:
            try:
                fn, future = self.jobs.get_nowait()
            except queue.Empty:
                return
            try:
                future.set_result(fn(self.hot.engine))
            except Exception as e:  # noqa: BLE001  the request gets the error, the app goes on
                future.set_exception(e)

    def frame(self, engine):
        """After a frame, under --debug: measure it, and log the keys pressed since the last one."""
        if self.meter is None:
            return
        self.meter.frame(time.perf_counter(), getattr(engine, "frame_split", None))
        keys = engine.input.pressed_keys
        if down := sorted(keys - self.keys):
            LOG.info("keys %s", " ".join(down))
        self.keys = keys

    # -- the server's threads --

    def on_engine(self, fn, timeout=LOOK_WAIT):
        """Run fn(engine) on the engine's thread and return its result."""
        future = Future()
        self.jobs.put((fn, future))
        return future.result(timeout)

    def look(self):
        """PNG bytes of the next frame presented."""
        shot = Future()
        self.on_engine(lambda engine: engine._looks.append(shot))
        return shot.result(LOOK_WAIT)


def events(engine):
    """{event: [first docstring line of each handler]} for the events the app handles."""
    return {name: [((h.func.__doc__ or "").strip().splitlines() or [""])[0] for h in handlers]
            for name, handlers in sorted(engine._event_bus._handlers.items()) if handlers}


def tab_events(engine):
    """The events a tab can be asked to fire, as events() lists them: the app's own handlers, but
    not startup (it ran in Python) or shutdown, and not manifoldx's (the GUI's pointer handlers),
    which the tab does not run."""
    return {name: [((h.func.__doc__ or "").strip().splitlines() or [""])[0] for h in app]
            for name, handlers in sorted(engine._event_bus._handlers.items())
            if name not in ("startup", "shutdown")
            and (app := [h for h in handlers if not getattr(h.func, "__module__", "").startswith("manifoldx.")])}


TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
         ".json": "application/json", ".map": "application/json", ".glb": "model/gltf-binary",
         ".ttf": "font/ttf"}
KEEPALIVE = 15.0  # seconds between the comment lines that keep a tab's stream open


class _NoTabOpen(Exception):
    """Under --web, a request the tab answers, with no tab open."""


class _TabError(Exception):
    """Under --web, the tab's answer was an error."""


def _closed(conn) -> bool:
    """Whether the peer of a socket that sends nothing has hung up."""
    try:
        readable, _, _ = select.select([conn], [], [], 0)
        return bool(readable) and conn.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


class _Handler(BaseHTTPRequestHandler):
    channel = None  # set per server

    def log_message(self, *args):  # requests stay out of the app's terminal
        pass

    def _send(self, code, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=repr, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"null") if n else None

    def _answer(self, fn):
        try:
            self._send(*fn())
        except _NoTabOpen as e:
            self._send(503, {"error": str(e)})
        except _TabError as e:
            self._send(500, {"error": str(e)})
        except (FutureTimeout, TimeoutError):
            who = "the tab did not answer" if self.channel.web else "the app did not answer: is it paused on an error?"
            self._send(504, {"error": who})
        except json.JSONDecodeError as e:
            self._send(400, {"error": f"the body is not JSON: {e}"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        ch = self.channel
        if ch.web is not None:
            return self._web_get(url, q)
        if url.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self._answer(lambda: (200, json.loads(ch.on_engine(
                lambda e: json.dumps(e.state, default=repr)))))
        if url.path == "/api/events":
            return self._answer(lambda: (200, ch.on_engine(events)))
        if url.path == "/api/log":
            return self._answer(lambda: (200, ch.lines.read(
                after=int(q["after"]) if "after" in q else None, match=q.get("match"),
                wait=float(q.get("wait", 0)), last=int(q.get("last", 50)))))
        if url.path == "/api/look":
            return self._answer(lambda: (200, ch.look(), "image/png"))
        if url.path == "/api/metrics":
            if ch.meter is None:
                return self._send(404, {"error": "metrics need manifoldx run --debug"})
            return self._send(200, list(ch.meter.history))
        self._send(404, {"error": f"no {url.path}"})

    # -- under --web: the tab runs the app --

    def _web_get(self, url, q):
        ch, path = self.channel, url.path
        if path == "/dev":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/web/current":
            return self._send(200, ch.web.current)
        if path == "/api/web/stream":
            return self._stream(q.get("tab") or f"anon-{id(self)}")
        if path == "/api/state":
            return self._answer(lambda: (200, self._tab("state")))
        if path == "/api/events":
            return self._answer(lambda: (200, ch.on_engine(tab_events)))
        if path == "/api/look":
            return self._answer(lambda: (200, self._tab("look"), "image/png"))
        if path == "/api/metrics":
            return self._send(404, {"error": "metrics are not measured under --web"})
        if path == "/api/log":
            return self._answer(lambda: (200, ch.lines.read(
                after=int(q["after"]) if "after" in q else None, match=q.get("match"),
                wait=float(q.get("wait", 0)), last=int(q.get("last", 50)))))
        self._file(unquote(path))

    def _file(self, path):
        root = ch_root = self.channel.web.web_root.resolve()
        f = (root / path.lstrip("/")).resolve()
        if f.is_dir():
            f = f / "index.html"
        if not f.is_relative_to(ch_root) or not f.is_file():
            return self._send(404, {"error": f"no {path}"})
        self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

    def _tab(self, kind, body=None):
        """Ask the answering tab and return its value; raises _NoTabOpen without a tab."""
        from manifoldx.web.dev import NoTab

        try:
            reply = self.channel.web.tab.call(kind, body)
        except NoTab:
            raise _NoTabOpen(f"no tab is open: open {self.channel.url}/") from None
        if isinstance(reply, bytes):
            return reply
        if not reply.get("ok"):
            raise _TabError(reply.get("error") or "the tab failed")
        return reply.get("value")

    def _stream(self, tab_id):
        """Server-Sent Events to one tab: the current version, then every push and every call to it."""
        tab = self.channel.web.tab
        q = tab.connect(tab_id)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = time.monotonic()
        try:
            while True:
                try:
                    msg = q.get(timeout=0.25)
                except queue.Empty:
                    if _closed(self.connection):
                        return
                    if time.monotonic() - last < KEEPALIVE:
                        continue
                    data = b": keep\n\n"
                else:
                    if msg is None:  # the server closes
                        return
                    data = b"data: " + json.dumps(msg, allow_nan=False).encode() + b"\n\n"
                self.wfile.write(data)
                self.wfile.flush()
                last = time.monotonic()
        except OSError:
            pass
        finally:
            tab.disconnect(tab_id, q)

    def _web_post(self, path):
        tab = self.channel.web.tab
        if path.startswith("/api/web/reply/"):
            n = int(path.removeprefix("/api/web/reply/"))
            if self.headers.get("Content-Type", "").startswith("image/png"):
                payload = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            else:
                payload = self._body()
            return self._send(200 if tab.reply(n, payload) else 404, {})
        if path == "/api/web/state":
            return self._answer(lambda: (tab.beacon(self._body()), (200, {}))[1])
        if path == "/api/web/log":
            def log():
                for line in self._body() or []:
                    LOG.info("%s", line["text"])
                return 200, {}
            return self._answer(log)
        return self._send(404, {"error": f"no {path}"})

    def do_PATCH(self):
        if urlparse(self.path).path != "/api/state":
            return self._send(404, {"error": f"no {self.path}"})
        if self.channel.web is not None:
            def forward():
                body = self._body()
                if not isinstance(body, dict):
                    return 400, {"error": "a merge patch is a JSON object"}
                value = self._tab("patch", body)
                LOG.info("state patched: %s", json.dumps(body)[:200])
                return 200, value
            return self._answer(forward)

        def patch():
            body = self._body()
            if not isinstance(body, dict):
                return 400, {"error": "a merge patch is a JSON object"}

            def apply(engine):
                merge_patch(engine.state, body)
                LOG.info("state patched: %s", json.dumps(body, default=repr)[:200])
                return json.loads(json.dumps(engine.state, default=repr))
            return 200, self.channel.on_engine(apply)
        self._answer(patch)

    def do_POST(self):
        path = urlparse(self.path).path
        if self.channel.web is not None and path.startswith("/api/web/"):
            return self._web_post(path)
        if not path.startswith("/api/events/"):
            return self._send(404, {"error": f"no {path}"})
        name = unquote(path.removeprefix("/api/events/"))

        def emit():
            payload = self._body()
            if self.channel.web is not None:
                known = self.channel.on_engine(tab_events)
                if name not in known:
                    return 404, {"error": f"no handler for {name!r}", "events": sorted(known)}
                self._tab("emit", {"name": name, "payload": payload})
                LOG.info("event %s %s", name, json.dumps(payload)[:200])
                return 200, {"emitted": name, "handlers": len(known[name])}

            def fire(engine):
                known = events(engine)
                if name not in known:
                    return 404, {"error": f"no handler for {name!r}", "events": sorted(known)}
                LOG.info("event %s %s", name, json.dumps(payload, default=repr)[:200])
                engine.emit(name, payload)
                return 200, {"emitted": name, "handlers": len(known[name])}
            return self.channel.on_engine(fire)
        self._answer(emit)


def _bind(port, channel):
    handler = type("Handler", (_Handler,), {"channel": channel})
    for p in [port + i for i in range(10)] + [0] if port else [0]:
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), handler)
        except OSError:
            continue
    raise OSError("no free port for the dev server")


def server_url(root="."):
    """The URL of the dev server of the app running under manifoldx run in `root`, or None."""
    try:
        return json.loads((Path(root) / FOLDER / "server.json").read_text())["url"]
    except (OSError, ValueError, KeyError):
        return None


class Readback:
    """A presented frame on its way to PNG bytes for the looks waiting, kept off the frames.

    The frame only queues a copy of its texture into a buffer. The next frame maps the buffer,
    which the GPU has filled by then, and a thread encodes the PNG and answers the looks."""

    def __init__(self, device, texture, texture_format, looks):
        import wgpu

        self.looks, self.bgra = looks, str(texture_format).startswith("bgra")
        self.w, self.h = texture.size[0], texture.size[1]
        self.row = (self.w * 4 + 255) // 256 * 256  # a buffer copy's rows are 256-byte aligned
        try:
            self.buffer = device.create_buffer(size=self.row * self.h,
                                               usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
            encoder = device.create_command_encoder()
            encoder.copy_texture_to_buffer({"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
                                           {"buffer": self.buffer, "offset": 0, "bytes_per_row": self.row,
                                            "rows_per_image": self.h}, (self.w, self.h, 1))
            device.queue.submit([encoder.finish()])
        except Exception as e:  # noqa: BLE001  a surface without COPY_SRC: the looks say so, drawing goes on
            self.fail(e)

    def fail(self, e):
        for look in self.looks:
            look.set_exception(e)
        self.looks = []

    def finish(self):
        if not self.looks:
            return
        import wgpu

        try:
            self.buffer.map_sync(wgpu.MapMode.READ)
            data = bytes(self.buffer.read_mapped())
            self.buffer.unmap()
        except Exception as e:  # noqa: BLE001
            return self.fail(e)
        threading.Thread(target=self._encode, args=(data,), name="manifoldx-look", daemon=True).start()

    def _encode(self, data):
        try:
            pixels = np.frombuffer(data, np.uint8).reshape(self.h, self.row)[:, :self.w * 4].reshape(self.h, self.w, 4)
            shot = png(pixels[..., [2, 1, 0]] if self.bgra else pixels[..., :3])
        except Exception as e:  # noqa: BLE001
            return self.fail(e)
        for look in self.looks:
            look.set_result(shot)


def png(rgb):
    """PNG bytes of an (h, w, 3) uint8 image."""
    from manifoldx.image import _encode_png

    return _encode_png(rgb)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>manifoldx run</title>
<style>
body{margin:0;font:14px/1.4 system-ui,sans-serif;background:#16161c;color:#e8e6f0}
header{padding:10px 16px;background:#1f1e27;display:flex;gap:16px;align-items:center}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:12px;padding:12px}
section{background:#1f1e27;border-radius:8px;padding:10px;min-width:0}
h2{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#a9a4c0}
img{width:100%;border-radius:4px;background:#000;min-height:120px}
pre,textarea{font:12px/1.35 ui-monospace,monospace;background:#121217;color:#e8e6f0;border:0;border-radius:4px;
  padding:8px;width:100%;box-sizing:border-box;margin:0}
#log{height:46vh;overflow:auto;white-space:pre-wrap}#state{max-height:40vh;overflow:auto}
button{background:#5b4fd6;color:#fff;border:0;border-radius:4px;padding:5px 10px;cursor:pointer;margin:4px 4px 0 0}
.ev{margin:6px 0}.ev small{color:#a9a4c0;display:block}.err{color:#ff8a8a}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><b>manifoldx run</b><span id="status"></span>
<label><input type="checkbox" id="auto"> live frame</label></header>
<main><div>
<section><h2>Frame</h2><img id="frame" alt="the frame on screen"><button onclick="look()">refresh</button></section>
<section><h2>Log</h2><pre id="log"></pre></section>
</div><div>
<section id="msec" hidden><h2>Metrics (--debug)</h2><pre id="metrics"></pre></section>
<section><h2>State</h2><pre id="state"></pre>
<textarea id="patch" rows="4" placeholder='merge patch, e.g. {"cam": {"yaw": 0.5}}'></textarea>
<button onclick="patch()">patch state</button><span id="perr" class="err"></span></section>
<section><h2>Events</h2><textarea id="payload" rows="3" placeholder="payload as JSON (optional)"></textarea>
<div id="events"></div></section>
</div></main>
<script>
const $ = id => document.getElementById(id);
let after = null;
async function get(u, o) { const r = await fetch(u, o); return r.json(); }
async function look() {
  const r = await fetch('/api/look');
  if (r.ok) { const old = $('frame').src; $('frame').src = URL.createObjectURL(await r.blob());
    if (old) URL.revokeObjectURL(old); $('status').textContent = ''; }
  else $('status').textContent = (await r.json()).error;
}
async function logLoop() {
  for (;;) {
    try {
      const q = after === null ? 'last=200' : 'after=' + after + '&wait=25';
      const d = await get('/api/log?' + q);
      after = d.next;
      const box = $('log'), end = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
      for (const l of d.lines) box.textContent += l.seq + ' ' + l.time + ' ' + l.text + '\\n';
      if (end) box.scrollTop = box.scrollHeight;
    } catch (e) { $('status').textContent = 'the app is gone'; await new Promise(r => setTimeout(r, 2000)); }
  }
}
async function state() {
  try { $('state').textContent = JSON.stringify(await get('/api/state'), null, 1); } catch (e) {}
}
async function patch() {
  $('perr').textContent = '';
  try { const d = await get('/api/state', {method: 'PATCH', body: $('patch').value});
    if (d.error) $('perr').textContent = d.error; else state(); }
  catch (e) { $('perr').textContent = String(e); }
}
async function events() {
  const d = await get('/api/events'), box = $('events'); box.textContent = '';
  for (const [name, docs] of Object.entries(d)) {
    const div = document.createElement('div'), b = document.createElement('button'), s = document.createElement('small');
    div.className = 'ev'; b.textContent = name; s.textContent = docs.filter(Boolean).join(' / ');
    b.onclick = async () => { const p = $('payload').value.trim();
      const r = await get('/api/events/' + encodeURIComponent(name), {method: 'POST', body: p || 'null'});
      if (r.error) alert(r.error); };
    div.append(b, s); box.append(div);
  }
}
async function metrics() {
  const r = await fetch('/api/metrics');
  if (!r.ok) return;
  const h = await r.json(), m = h[h.length - 1];
  $('msec').hidden = false;
  if (!m) return;
  const fps = h.slice(-60).map(x => x.fps), bars = '▁▂▃▄▅▆▇█', top = Math.max(...fps, 1);
  const s = m.split_ms || {};
  $('metrics').textContent = `fps ${m.fps}   frame ${m.frame_ms} ms   worst ${m.worst_ms} ms\n` +
    `update ${s.update} ms   render ${s.render} ms   acquire ${s.acquire} ms   idle ${s.idle} ms\n` +
    `memory ${m.rss_mb} MB   gc ${m.gc.runs.join('/')} in ${m.gc.ms} ms\n` +
    fps.map(f => bars[Math.min(7, Math.floor(8 * f / top))]).join('') + '  fps, last minute';
}
logLoop(); state(); events(); look(); metrics();
setInterval(state, 1000); setInterval(events, 5000); setInterval(metrics, 1000);
setInterval(() => { if ($('auto').checked) look(); }, 1500);
</script></body></html>
"""
