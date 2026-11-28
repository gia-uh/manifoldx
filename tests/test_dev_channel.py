import json
import struct
import textwrap
import threading
import urllib.error
import urllib.request
import zlib

import numpy as np
import pytest

from manifoldx import cli, devchannel
from manifoldx.backends import get_offscreen_canvas
from manifoldx.hot import HotReloader

APP = """
import manifoldx as mx

engine = mx.Engine("channel", width=100, height=60)
engine.background_color = (1.0, 0.0, 0.0)
engine.state.setdefault("loads", 0)
engine.state.loads += 1
engine.state.setdefault("where", {"place": "yard", "x": 2.5})
engine.state.setdefault("hits", [])

@engine.on("frame")
def frame(payload):
    engine.log("at the yard")

@engine.on("hit")
def hit(payload):
    \"\"\"Hit something: payload {"what": name}.\"\"\"
    engine.state.hits.append(payload["what"])
    engine.log(f"hit {payload['what']}")

engine.run()
"""


def unpng(data):
    """(h, w, 3) uint8 from the PNG bytes devchannel.png writes (filter 0 on every row)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    at, idat = 8, b""
    while at < len(data):
        n, kind = struct.unpack(">I4s", data[at:at + 8])
        if kind == b"IDAT":
            idat += data[at + 8:at + 8 + n]
        at += 12 + n
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * 3)
    assert (rows[:, 0] == 0).all()
    return rows[:, 1:].reshape(h, w, 3)


def test_png_round_trips_an_image():
    rgb = np.random.default_rng(1).integers(0, 256, (7, 13, 3), dtype=np.uint8)
    assert (unpng(devchannel.png(rgb)) == rgb).all()


def test_merge_patch_merges_objects_and_deletes_nulls():
    state = {"a": {"b": 1, "c": 2}, "d": 3}
    devchannel.merge_patch(state, {"a": {"b": 5, "c": None}, "d": [1], "e": {"f": 1}})
    assert state == {"a": {"b": 5}, "d": [1], "e": {"f": 1}}


@pytest.fixture
def hot(tmp_path, monkeypatch):
    try:
        import wgpu
        wgpu.gpu.request_adapter_sync()
    except Exception as e:  # noqa: BLE001  no GPU and no software Vulkan
        pytest.skip(f"offscreen canvas unavailable: {e}")
    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent(APP))
    h = HotReloader(script=str(app), canvas_factory=lambda e: get_offscreen_canvas(e.w, e.h), poll=0, port=0)
    h.start()
    h.step()
    monkeypatch.chdir(tmp_path)
    yield h
    h.channel.close()


def ask(hot, path, method="GET", body=None):
    """(status, bytes) of a request to the dev server, stepping the app until it is answered."""
    out = {}

    def go():
        request = urllib.request.Request(hot.channel.url + path, method=method,
                                         data=None if body is None else body.encode())
        try:
            with urllib.request.urlopen(request, timeout=10) as r:
                out["got"] = r.status, r.read()
        except urllib.error.HTTPError as e:
            out["got"] = e.code, e.read()

    t = threading.Thread(target=go)
    t.start()
    while t.is_alive():
        hot.step()
        t.join(0.01)
    return out["got"]


def test_the_page_is_served(hot):
    status, body = ask(hot, "/")
    assert status == 200 and b"manifoldx run" in body and b"/api/log" in body


def test_look_is_the_presented_frame(hot):
    status, data = ask(hot, "/api/look")
    image = unpng(data)
    assert status == 200 and image.shape == (60, 100, 3)  # 100 px is 400 bytes a row: not a multiple of 256
    assert (np.abs(image[30, 50].astype(int) - (255, 0, 0)) <= 2).all()


def test_state_reads_and_a_merge_patch_changes_it(hot):
    status, body = ask(hot, "/api/state")
    assert status == 200 and json.loads(body)["where"] == {"place": "yard", "x": 2.5}
    status, body = ask(hot, "/api/state", "PATCH", '{"where": {"x": 7, "place": null}, "new": 1}')
    assert status == 200 and json.loads(body)["where"] == {"x": 7}
    assert hot.engine.state.where == {"x": 7} and hot.engine.state.new == 1
    assert ask(hot, "/api/state", "PATCH", "[1]")[0] == 400


def test_events_are_listed_and_emitted_into_the_app(hot):
    status, body = ask(hot, "/api/events")
    assert status == 200 and json.loads(body)["hit"] == ['Hit something: payload {"what": name}.']
    status, body = ask(hot, "/api/events/hit", "POST", '{"what": "the wall"}')
    assert status == 200 and json.loads(body) == {"emitted": "hit", "handlers": 1}
    hot.step()  # emitted events run at the start of the next frame
    assert hot.engine.state.hits == ["the wall"]
    status, body = ask(hot, "/api/events/nothing", "POST", "null")
    assert status == 404 and "hit" in json.loads(body)["events"]


def test_log_waits_for_a_matching_line(hot):
    status, body = ask(hot, "/api/log?last=1")
    after = json.loads(body)["next"]
    hot.engine.emit("hit", {"what": "a tree"})
    status, body = ask(hot, f"/api/log?after={after}&match=hit%20a&wait=5")
    assert status == 200 and [line["text"] for line in json.loads(body)["lines"]] == ["hit a tree"]
    status, body = ask(hot, f"/api/log?after={json.loads(body)['next']}&match=never&wait=0.2")
    assert json.loads(body)["lines"] == []


def test_the_log_holds_the_run_the_reloads_and_the_apps_lines(hot, tmp_path):
    app = tmp_path / "app.py"
    app.write_text(app.read_text() + "\n")
    hot.reload()
    app.write_text("this is not python(")
    hot.reload()
    hot.step()
    log = (tmp_path / ".manifoldx" / "log.txt").read_text()
    assert "started" in log and "at the yard" in log
    assert "reloaded" in log and "reload failed, kept the running version" in log
    assert ask(hot, "/api/state")[0] == 200  # the server follows the new engine
    assert json.loads(ask(hot, "/api/state")[1])["loads"] == 2


def test_keys_are_logged_once_when_pressed():
    lines = devchannel.Lines()
    devchannel.LOG.addHandler(lines)
    channel = object.__new__(devchannel.Channel)
    channel.keys, channel.meter = frozenset(), devchannel.Meter()  # keys are logged under --debug

    class Engine:
        class input:
            pressed_keys = frozenset()
    try:
        for keys in ({"w"}, {"w"}, {"w", "Space"}, set(), {"w"}):
            Engine.input.pressed_keys = frozenset(keys)
            channel.frame(Engine)
    finally:
        devchannel.LOG.removeHandler(lines)
        channel.meter.close()
    assert [line["text"] for line in lines.lines if line["text"].startswith("keys")] == [
        "keys w", "keys Space", "keys w"]


def test_cli_talks_to_the_running_app(hot, capsys):
    def run(*argv):
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("rc", cli.main(list(argv))))
        t.start()
        while t.is_alive():
            hot.step()
            t.join(0.01)
        return out["rc"], capsys.readouterr().out

    assert run("state", "where.place") == (0, '"yard"\n')
    assert run("state", "nope")[0] == 1
    assert run("set", '{"where": {"place": "lake"}}')[0] == 0 and hot.engine.state.where["place"] == "lake"
    assert run("emit", "hit", '{"what": "a rock"}')[0] == 0
    rc, out = run("log", "--match", "hit a rock", "--wait", "5")
    assert rc == 0 and out.endswith("hit a rock\n")
    assert run("log", "--match", "never", "--wait", "0.2")[0] == 1
    rc, out = run("look")
    assert rc == 0 and unpng(open(out.strip(), "rb").read()).shape == (60, 100, 3)
    assert "hit\n    Hit something" in run("events")[1]


def test_cli_says_when_no_app_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["state"]) == 1


def test_the_meter_counts_each_second_and_logs_slow_frames():
    lines = devchannel.Lines()
    devchannel.LOG.addHandler(lines)
    meter = devchannel.Meter()
    try:
        t = 100.0
        split = {"update": 0.01, "acquire": 0.02, "render": 0.005}
        for dt in [0.0] + [0.05] * 18 + [0.15]:  # 19 frames in 1.05 s, the last one slow
            t += dt
            meter.frame(t, split)
        import gc
        gc.collect()
        t += 1.0
        meter.frame(t)
    finally:
        devchannel.LOG.removeHandler(lines)
        meter.close()
    first, second = meter.history
    assert first["fps"] == pytest.approx(19 / 1.05, abs=0.1) and first["worst_ms"] == 150.0
    assert first["frame_ms"] == pytest.approx(1050 / 19, abs=0.1) and first["rss_mb"] > 0
    assert first["split_ms"] == {"update": 10.0, "acquire": 20.0, "render": 5.0, "idle": 20.3}
    assert second["gc"]["runs"][2] >= 1
    texts = [line["text"] for line in lines.lines]
    assert "slow frame 150 ms" in texts and "slow frame 1000 ms" in texts
    assert any(text.startswith("fps 18, frame 55.3 ms (worst 150; update 10.0, render 5.0, acquire 20.0, idle 20.3)")
               for text in texts)


def test_metrics_need_debug(hot):
    assert ask(hot, "/api/metrics")[0] == 404


def test_metrics_under_debug(tmp_path, monkeypatch):
    try:
        import wgpu
        wgpu.gpu.request_adapter_sync()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"offscreen canvas unavailable: {e}")
    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent(APP))
    h = HotReloader(script=str(app), canvas_factory=lambda e: get_offscreen_canvas(e.w, e.h), poll=0, port=0,
                    debug=True)
    h.start()
    try:
        import time
        end = time.monotonic() + 1.3
        while time.monotonic() < end:
            h.step()
        status, body = ask(h, "/api/metrics")
        got = json.loads(body)
        assert status == 200 and got and {"fps", "frame_ms", "worst_ms", "rss_mb", "gc", "split_ms"} <= set(got[-1])
        assert got[-1]["split_ms"]["render"] > 0
    finally:
        h.channel.close()


def test_the_meter_does_not_count_overlap_as_frame_time():
    # overlap is time update and render already hold; idle must not lose it twice
    meter = devchannel.Meter()
    try:
        t = 100.0
        split = {"update": 0.01, "acquire": 0.0, "render": 0.01, "overlap": 0.008}
        for dt in [0.0] + [0.05] * 21:
            t += dt
            meter.frame(t, split)
    finally:
        meter.close()
    parts = meter.history[0]["split_ms"]
    assert parts["overlap"] == 8.0 and parts["idle"] == 30.0


def test_the_meter_takes_the_cores_measured_interval_and_over_budget():
    meter = devchannel.Meter()
    try:
        t = 100.0
        split = {"update": 0.002, "acquire": 0.004, "render": 0.003, "submit": 0.001, "present": 0.001}
        for _ in range(61):
            t += 1.0
            meter.frame(t, split, {"step": 0.002}, over_budget={"step": 0.002}, interval=0.016)
    finally:
        meter.close()
    entry = meter.history[0]
    assert entry["frame_ms"] == pytest.approx(16.0, abs=0.1)
    assert entry["split_ms"]["submit"] == 1.0 and entry["split_ms"]["present"] == 1.0
    assert entry["over_budget_ms"] == {"step": 2.0}
