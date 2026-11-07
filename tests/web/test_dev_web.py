"""manifoldx run --web: the WebReloader rebuilds the export into version folders on every save,
taking engine.state from the tab; here the tab is scripted (FakeTab), then a Python client of
the dev server's stream plays it (TabClient)."""

import http.client
import json
import os
import re
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from manifoldx.web.dev import NoTab, WebReloader

DEV_ORBIT = Path(__file__).parent / "apps" / "dev_orbit.py"

_clock = [time.time()]


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new))
    # Whole seconds later, so the watcher sees it (as tests/test_hot_reload.py does).
    _clock[0] += 2
    os.utime(path, (_clock[0], _clock[0]))


class FakeTab:
    """Records pushes; answers reload_state with `answer`, or raises NoTab when it is None.
    on_call runs inside the call, for edits that land during a reload."""

    def __init__(self, answer=None):
        self.pushed = []
        self.calls = []
        self.answer = answer
        self.on_call = None
        self.last_state = None

    def push(self, message):
        self.pushed.append(message)

    def call(self, kind, body=None, timeout=5.0):
        self.calls.append((kind, timeout))
        if self.on_call is not None:
            self.on_call()
        if self.answer is None:
            raise NoTab
        return {"ok": True, "value": None, "state": self.answer}


def cam_state(yaw, pitch=0.3, dist=6.0):
    return {"__dict": [["cam", {"__dict": [["yaw", yaw], ["pitch", pitch], ["dist", dist]]}]]}


@pytest.fixture
def dev_app(tmp_path):
    app = tmp_path / "dev_orbit.py"
    shutil.copy(DEV_ORBIT, app)
    return app


def started(app, answer=None) -> WebReloader:
    w = WebReloader(script=app, channel=False, poll=0)
    w.tab = FakeTab(answer)
    w.start()
    return w


def snapshot_state(w, version) -> dict:
    """engine.state as the version's state.json holds it, {key: value} of plain JSON."""
    s = json.loads((w.web_root / f"v{version}" / "state.json").read_text())
    table = s["table"]

    def plain(v):
        if isinstance(v, dict) and "__ref" in v:
            return plain(table[v["__ref"]])
        if isinstance(v, dict) and "__dict" in v:
            return {k: plain(x) for k, x in v["__dict"]}
        return v

    return plain(s["state"])


def test_start_writes_v1_and_static(dev_app):
    w = WebReloader(script=dev_app, channel=False, poll=0)
    w.tab = FakeTab()
    w.start()
    root = w.web_root
    assert root == dev_app.parent / ".manifoldx" / "web"
    assert (root / "v1" / "state.json").exists() and (root / "browser.js").exists()
    assert (root / "dev.js").exists() and '<meta name="mx-dev">' in (
        root / "index.html"
    ).read_text()
    assert (root / w.current["scene"]).exists()
    assert w.current["base"] == "v1/"
    assert w.current["type"] == "version" and w.current["version"] == w.version == 1
    assert w.current["title"] == "Dev orbit"
    assert w.current["scene_bytes"] == (root / w.current["scene"]).stat().st_size
    assert 0 < w.current["scene_gzip_bytes"] < w.current["scene_bytes"]
    assert 'from "../../pyrt.js"' in (root / "v1" / "app" / "__main__.js").read_text()


def test_code_edit_keeps_the_scene_and_takes_the_tab_state(dev_app):
    w = started(dev_app, answer=cam_state(1.25))
    v1 = dict(w.current)
    edit(dev_app, "SPEED = 1.0", "SPEED = 2.0")
    w.step()
    assert w.tab.calls == [("reload_state", 0.5)]
    assert w.current["version"] == 2 and w.current["scene_sha"] == v1["scene_sha"]
    assert w.tab.pushed[-1] == w.current and w.current["base"] == "v2/"
    assert w.state["cam"]["yaw"] == 1.25
    assert w.state[("cache", 1)].shape == (4, 4)  # never crossed, never touched
    assert snapshot_state(w, 2)["cam"] == {"yaw": 1.25, "pitch": 0.3, "dist": 6.0}
    assert (
        json.loads((w.web_root / "v2" / "state.json").read_text())["globals"]["__main__"]["SPEED"]
        == 2.0
    )


def test_colour_edit_writes_a_new_scene(dev_app):
    w = started(dev_app, answer=cam_state(0.0))
    v1 = dict(w.current)
    edit(dev_app, 'BOX = "#4080c0"', 'BOX = "#c04040"')
    w.step()
    assert w.current["version"] == 2 and w.current["scene_sha"] != v1["scene_sha"]
    assert (w.web_root / w.current["scene"]).exists() and (w.web_root / v1["scene"]).exists()


def test_broken_edit_pushes_failed_with_the_line(dev_app):
    w = started(dev_app, answer=cam_state(0.0))
    line = dev_app.read_text().splitlines().index("SPEED = 1.0  # radians per second") + 1
    edit(dev_app, "SPEED = 1.0", "SPEED = (")
    w.step()
    assert w.tab.pushed[-1] == {
        "type": "failed",
        "running": 1,
        "text": f"dev_orbit.py:{line}: SyntaxError: '(' was never closed",
    }
    assert w.version == 1 and w.current["version"] == 1
    assert not (w.web_root / "v2").exists()
    edit(dev_app, "SPEED = (", "SPEED = 3.0")  # fixing it reloads
    w.step()
    assert w.current["version"] == 2


def test_transpile_error_pushes_failed(dev_app):
    w = started(dev_app, answer=cam_state(0.0))
    engine = w.engine
    line = dev_app.read_text().splitlines().index("    place_camera(engine.camera)") + 2
    edit(
        dev_app,
        "    place_camera(engine.camera)\n",
        "    place_camera(engine.camera)\n    getattr(engine, 'w')\n",
    )
    w.step()
    assert w.tab.pushed[-1] == {
        "type": "failed",
        "running": 1,
        "text": f"dev_orbit.py:{line}: this call to getattr() is outside the subset (reached from system orbit)",
    }
    assert w.engine is engine and w.version == 1


def test_no_tab_uses_the_last_state(dev_app):
    w = started(dev_app, answer=None)
    w.state["cam"]["yaw"] = 2.0  # what the last reply or beacon left
    edit(dev_app, "SPEED = 1.0", "SPEED = 2.0")
    w.step()
    assert w.current["version"] == 2
    assert snapshot_state(w, 2)["cam"]["yaw"] == 2.0


def test_old_versions_are_pruned(dev_app):
    w = started(dev_app, answer=None)
    colours = ["#4080c0", "#100000", "#200000", "#300000", "#400000"]
    for old, new in zip(colours, colours[1:]):
        edit(dev_app, f'BOX = "{old}"', f'BOX = "{new}"')
        w.step()
    assert w.current["version"] == 5
    root = w.web_root
    assert sorted(p.name for p in root.iterdir() if re.fullmatch(r"v\d+", p.name)) == [
        "v3",
        "v4",
        "v5",
    ]
    shas = {w._scenes[n] for n in (3, 4, 5)}
    assert len(shas) == 3
    assert {p.name for p in (root / "scenes").glob("*.glb")} == {f"{s}.glb" for s in shas}
    assert len(list((root / "scenes").glob("*.export-report.json"))) == 3


def test_a_save_during_the_build_reloads_again(dev_app):
    w = started(dev_app, answer=cam_state(0.0))
    edit(dev_app, "SPEED = 1.0", "SPEED = 2.0")
    w.tab.on_call = lambda: (
        edit(dev_app, "SPEED = 2.0", "SPEED = 3.0"),
        setattr(w.tab, "on_call", None),
    )
    w.step()
    assert w.current["version"] == 2
    w.step()
    assert w.current["version"] == 3
    w.step()
    assert w.current["version"] == 3  # nothing changed since


# -- the dev server and the tab (task 4) --


class TabClient:
    """A tab, played in Python: reads /api/web/stream and answers each call with
    answers[kind](body), or not at all when its kind has no answer."""

    def __init__(self, url, tab_id, answers=None):
        self.messages, self.calls = [], []
        self.url, self.answers = url, dict(answers or {})
        host, port = url.removeprefix("http://").split(":")
        self.conn = http.client.HTTPConnection(host, int(port), timeout=30)
        self.conn.request("GET", f"/api/web/stream?tab={tab_id}")
        self.sock = self.conn.sock
        self.resp = self.conn.getresponse()
        assert self.resp.getheader("Content-Type") == "text/event-stream"
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            while line := self.resp.readline():
                line = line.decode().rstrip("\n")
                if not line.startswith("data: "):
                    continue
                msg = json.loads(line.removeprefix("data: "))
                self.messages.append(msg)
                if msg["type"] == "call":
                    self.calls.append(msg)
                    answer = self.answers.get(msg["kind"])
                    if answer is not None:
                        post(self.url, f"/api/web/reply/{msg['id']}", answer(msg["body"]))
        except (OSError, ValueError):
            pass

    def wait_for(self, test, timeout=10.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            found = [m for m in self.messages if test(m)]
            if found:
                return found[0]
            time.sleep(0.02)
        raise AssertionError(f"no such message in {self.messages}")

    def close(self):
        self.sock.shutdown(socket.SHUT_RDWR)  # the reader sees the end of the stream and stops


def request(url, path, method="GET", body=None):
    """(status, content type, body as JSON or bytes) of a request to the dev server."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            status, kind, raw = r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        status, kind, raw = e.code, e.headers.get("Content-Type"), e.read()
    return status, kind, json.loads(raw) if kind == "application/json" else raw


def post(url, path, body):
    return request(url, path, "POST", body)


def raw_get(url, path) -> int:
    """The status of GET path sent as written, without the client normalizing it."""
    host, port = url.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    conn.request("GET", path)
    status = conn.getresponse().status
    conn.close()
    return status


def state_reply(yaw):
    return lambda body: {"ok": True, "value": {"cam": {"yaw": yaw}}, "state": cam_state(yaw)}


@pytest.fixture
def served(dev_app):
    """A WebReloader serving on a free port, stepped by a thread as run() would."""
    w = WebReloader(script=dev_app, poll=0.05, port=0)
    w.start()
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            w.step()
            time.sleep(0.02)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    yield w
    stop.set()
    thread.join(10)
    w.channel.close()


def test_stream_starts_with_the_current_version(served):
    tab = TabClient(served.channel.url, "a")
    assert tab.wait_for(lambda m: True) == served.current
    assert request(served.channel.url, "/api/web/current")[2] == served.current


def test_state_is_the_tabs(served):
    url = served.channel.url
    tab = TabClient(url, "a", {"state": state_reply(0.5)})
    tab.wait_for(lambda m: m["type"] == "version")
    assert request(url, "/api/state") == (200, "application/json", {"cam": {"yaw": 0.5}})
    assert served.tab.last_state == cam_state(0.5)  # every reply carries the tab's state


def test_set_and_emit_reach_the_tab(served):
    url = served.channel.url
    answers = {
        "patch": lambda body: {"ok": True, "value": {"cam": {"yaw": 0}}, "state": cam_state(0.0)},
        "emit": lambda body: {"ok": True, "value": None, "state": cam_state(0.0)},
    }
    tab = TabClient(url, "a", answers)
    tab.wait_for(lambda m: m["type"] == "version")
    assert request(url, "/api/state", "PATCH", {"cam": {"yaw": 0}})[::2] == (
        200,
        {"cam": {"yaw": 0}},
    )
    assert tab.calls[-1]["kind"] == "patch" and tab.calls[-1]["body"] == {"cam": {"yaw": 0}}
    assert post(url, "/api/events/bump", {"by": 10})[::2] == (
        200,
        {"emitted": "bump", "handlers": 1},
    )
    assert tab.calls[-1]["kind"] == "emit"
    assert tab.calls[-1]["body"] == {"name": "bump", "payload": {"by": 10}}
    status, _, body = post(url, "/api/events/nope", None)
    assert status == 404 and body["events"] == ["bump"] and len(tab.calls) == 2
    assert request(url, "/api/events")[2] == {"bump": ['Turn the camera: {"by": degrees}.']}
    log = request(url, "/api/log?last=50")[2]["lines"]
    texts = [line["text"] for line in log]
    assert 'state patched: {"cam": {"yaw": 0}}' in texts and 'event bump {"by": 10}' in texts


def test_no_tab_is_503(served):
    url = served.channel.url
    for path in ("/api/state", "/api/look"):
        assert request(url, path)[::2] == (503, {"error": f"no tab is open: open {url}/"})
    assert request(url, "/api/metrics")[::2] == (
        404,
        {"error": "metrics are not measured under --web"},
    )


def test_the_last_tab_answers(served):
    url = served.channel.url
    first = TabClient(url, "a", {"state": state_reply(1.0)})
    first.wait_for(lambda m: m["type"] == "version")
    second = TabClient(url, "b", {"state": state_reply(2.0)})
    second.wait_for(lambda m: m["type"] == "version")
    assert request(url, "/api/state")[2] == {"cam": {"yaw": 2.0}}
    second.close()
    end = time.monotonic() + 5
    while served.tab.connected() != ["a"] and time.monotonic() < end:
        time.sleep(0.02)
    assert request(url, "/api/state")[2] == {"cam": {"yaw": 1.0}}
    assert not first.calls[1:] and len(first.calls) == 1


def test_static_files_and_types(served):
    url = served.channel.url
    status, kind, body = request(url, "/")
    assert status == 200 and kind.startswith("text/html") and b'<meta name="mx-dev">' in body
    assert request(url, "/browser.js")[:2] == (200, "text/javascript")
    assert request(url, "/v1/app/__main__.js")[:2] == (200, "text/javascript")
    assert request(url, "/v1/state.json")[1] == "application/json"
    assert request(url, "/" + served.current["scene"])[1] == "model/gltf-binary"
    assert request(url, "/fonts/DejaVuSansMono.ttf")[1] == "font/ttf"
    assert b"manifoldx run" in request(url, "/dev")[2]
    assert raw_get(url, "/../pyproject.toml") == 404
    assert raw_get(url, "/v1/../../x") == 404
    # files that exist, one and two folders above the web root
    assert (served.web_root.parent / "server.json").exists()
    assert raw_get(url, "/../server.json") == 404
    assert raw_get(url, "/v1/../../log.txt") == 404
    assert raw_get(url, "/%2e%2e/server.json") == 404
    assert raw_get(url, "/v9/state.json") == 404


def test_a_tab_with_a_newer_state_triggers_a_rebuild(served):
    url = served.channel.url
    post(url, "/api/web/state", cam_state(2.5))  # the pagehide beacon of a tab that moved on
    tab = TabClient(url, "f5")
    assert tab.wait_for(lambda m: True)["version"] == 1
    assert tab.wait_for(lambda m: m.get("version") == 2)["type"] == "version"
    assert snapshot_state(served, 2)["cam"]["yaw"] == 2.5
    texts = [line["text"] for line in request(url, "/api/log?last=50")[2]["lines"]]
    assert any(t.startswith("reloaded v2 ") and t.endswith("reason tab") for t in texts)
    assert not tab.calls  # the rebuild took the beacon's state, not the new tab's


def test_tab_gone_during_reload(served, dev_app):
    url = served.channel.url
    tab = TabClient(url, "a", {"state": state_reply(0.9)})  # never answers reload_state
    tab.wait_for(lambda m: m["type"] == "version")
    request(url, "/api/state")  # its reply leaves yaw 0.9 as the last state
    began = time.monotonic()
    edit(dev_app, "SPEED = 1.0", "SPEED = 2.0")
    tab.wait_for(lambda m: m.get("version") == 2)
    assert time.monotonic() - began >= 0.5
    assert [c["kind"] for c in tab.calls] == ["state", "reload_state"]
    assert snapshot_state(served, 2)["cam"]["yaw"] == 0.9
