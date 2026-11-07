"""A host page around the export: an iframe of index.html that talks to it through events
only. The host posts {mx: "emit", name, payload} to the frame, which records it in the next
frame it takes; the export posts what its app emits back to window.parent and dispatches
mx:emit on its own window. A message from any window but the parent is ignored."""

import json
import shutil

import pytest

from manifoldx.web.replay import compare, replay_python

from .conftest import APPS, _export

EVENTS = APPS / "events.py"
PATHS = ["hellos", "pings"]

HOST = """<!doctype html>
<html><body>
<iframe id="app" src="index.html" width="320" height="200"></iframe>
<iframe id="other" srcdoc="<p>another window</p>"></iframe>
<script>
  window.got = [];
  const app = () => document.getElementById("app").contentWindow;
  addEventListener("message", (e) => { if (e.source === app()) got.push(e.data); });
  window.hello = () => app().postMessage({ mx: "emit", name: "host:hello", payload: { n: 2 } }, "*");
</script>
</body></html>
"""

# In the export's frame: a promise that resolves once the next message has been handled.
SEEN = """() => {
  window.__seen = new Promise((r) => addEventListener("message", () => setTimeout(r, 0), { once: true }));
}"""


@pytest.fixture(scope="module")
def events_export(tmp_path_factory):
    return _export(EVENTS, tmp_path_factory.mktemp("events"))


@pytest.fixture(scope="module")
def host_copy(tmp_path_factory, events_export):
    folder = tmp_path_factory.mktemp("host")
    shutil.copytree(events_export, folder, dirs_exist_ok=True)
    (folder / "host.html").write_text(HOST)
    return folder


def open_host(open_page, served, host_copy):
    """The host page and the export's frame, past __mx.ready, its paths set."""
    page = open_page(served(host_copy) + "host.html", ready=False)
    frame = page.wait_for_selector("#app").content_frame()
    frame.wait_for_function("window.__mx !== undefined")
    frame.evaluate("__mx.ready.then(() => true)")
    frame.evaluate(f"() => {{ __mx.paths = {json.dumps(PATHS)}; }}")
    return page, frame


def post(page, frame):
    frame.evaluate(SEEN)
    page.evaluate("hello()")
    frame.evaluate("__seen.then(() => true)")


def tick(frame, n: int = 1):
    for _ in range(n):
        frame.evaluate("__mx.tick(1 / 60)")


def test_a_host_page_emits_into_the_export_and_hears_back(open_page, served, host_copy):
    page, frame = open_host(open_page, served, host_copy)
    tick(frame, 2)
    post(page, frame)
    tick(frame, 3)
    frames, trace = frame.evaluate("__mx.frames"), frame.evaluate("__mx.trace")
    assert frames[2]["events"] == [["host:hello", {"n": 2}]]
    assert "host:hello" in trace[2]["events"]
    assert "app:ack" not in trace[2]["events"]
    assert "app:ack" in trace[3]["events"]  # emitted during delivery: the next frame
    assert trace[3]["paths"]["hellos"] == 1
    page.wait_for_function("got.some((m) => m.name === 'app:ack')")
    got = page.evaluate("got")
    assert {"mx": "emit", "name": "app:ack", "payload": {"__dict": [["n", 2]]}} in got
    assert all(m["name"] not in ("frame", "pointer_move", "key_down") for m in got)


def test_the_page_replays_with_the_host_events(open_page, served, host_copy):
    page, frame = open_host(open_page, served, host_copy)
    tick(frame, 3)
    post(page, frame)
    tick(frame, 4)
    post(page, frame)
    tick(frame, 3)
    frames, trace = frame.evaluate("__mx.frames"), frame.evaluate("__mx.trace")
    python = replay_python(None, str(EVENTS), frames, PATHS)
    assert python[-1]["paths"]["hellos"] == 2
    mismatch = compare(python, trace)
    assert mismatch is None, str(mismatch)


def test_a_message_from_another_window_is_ignored(open_page, served, host_copy):
    page, frame = open_host(open_page, served, host_copy)
    other = page.wait_for_selector("#other").content_frame()
    frame.evaluate(SEEN)
    other.evaluate(
        """parent.document.getElementById("app").contentWindow
             .postMessage({ mx: "emit", name: "host:hello", payload: { n: 2 } }, "*")"""
    )
    frame.evaluate("__seen.then(() => true)")
    tick(frame, 2)
    frames = frame.evaluate("__mx.frames")
    assert all(e[0] != "host:hello" for f in frames for e in f["events"])


def test_a_message_before_the_first_tick_waits_for_it(open_page, served, host_copy):
    page, frame = open_host(open_page, served, host_copy)
    post(page, frame)
    assert frame.evaluate("__mx.frames.length") == 0
    tick(frame)
    assert frame.evaluate("__mx.frames[0].events") == [["host:hello", {"n": 2}]]


def test_the_export_dispatches_mx_emit_on_its_window(open_page, served, events_export):
    page = open_page(served(events_export))
    page.evaluate(
        """() => { window.__emits = []; addEventListener("mx:emit", (e) => __emits.push(e.detail)); }"""
    )
    for _ in range(6):
        page.evaluate("__mx.tick(1 / 60)")
    emits = page.evaluate("__emits")
    assert {"name": "ping", "payload": {"__dict": [["n", 3]]}} in emits
    assert all(e["name"] != "frame" for e in emits)
