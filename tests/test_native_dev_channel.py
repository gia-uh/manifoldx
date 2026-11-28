"""The dev channel of `manifoldx run` under the Rust core, at its frame barrier."""

import json
import socket
import textwrap
import threading
import urllib.error
import urllib.request

import pytest

from manifoldx import cli, devchannel
from manifoldx._native_reload import _DevChannelHost


@pytest.fixture
def native_engine(monkeypatch):
    native = pytest.importorskip("manifoldx._core_native")
    if not hasattr(native, "_NativeGraphics"):
        pytest.skip("native renderer is unavailable")
    monkeypatch.setenv("MANIFOLDX_CORE", "rust")
    from manifoldx import Engine

    engine = Engine("e8 channel", width=32, height=24)
    try:
        yield engine
    finally:
        engine.close()


def step(engine):
    """One native frame barrier, without opening a window."""
    return engine._desktop_begin_frame((32, 24), 1.0, engine.elapsed, [], True)


def drive(engine, fn):
    """Run fn on the channel's thread, stepping the frame until it answers."""
    out = {}

    def go():
        try:
            out["value"] = fn()
        except BaseException as e:  # noqa: BLE001  the test reports it
            out["error"] = e

    worker = threading.Thread(target=go)
    worker.start()
    while worker.is_alive():
        step(engine)
        worker.join(0.01)
    if "error" in out:
        raise out["error"]
    return out["value"]


def ask(engine, channel, path, method="GET", body=None):
    """(status, bytes) of a request to the channel, stepping frames until it is answered."""

    def request():
        req = urllib.request.Request(
            channel.url + path, method=method, data=None if body is None else body.encode()
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    return drive(engine, request)


def test_native_engine_exposes_the_game_facing_state_and_log(native_engine):
    lines = devchannel.Lines()
    devchannel.LOG.addHandler(lines)
    try:
        native_engine.state.setdefault("hp", 10)
        native_engine.state.hp += 1
        native_engine.log("the yard")
    finally:
        devchannel.LOG.removeHandler(lines)
    assert native_engine.state.hp == 11
    assert [line["text"] for line in lines.lines] == ["the yard"]


def test_the_http_contract_runs_at_the_frame_barrier(native_engine, tmp_path):
    channel = devchannel.Channel(tmp_path, _DevChannelHost(native_engine), port=0)
    native_engine._dev_channel = channel
    native_engine.state["where"] = {"place": "yard", "x": 2.5}
    native_engine.state["hits"] = []

    @native_engine.on("hit")
    def hit(payload):
        """Hit something: payload {"what": name}."""
        native_engine.state["hits"].append(payload["what"])
        native_engine.log(f"hit {payload['what']}")

    try:
        status, body = ask(native_engine, channel, "/api/state")
        assert status == 200 and json.loads(body)["where"] == {"place": "yard", "x": 2.5}

        status, body = ask(
            native_engine,
            channel,
            "/api/state",
            "PATCH",
            '{"where": {"x": 7, "place": null}, "new": 1}',
        )
        assert status == 200 and json.loads(body)["where"] == {"x": 7}
        assert native_engine.state.where == {"x": 7} and native_engine.state.new == 1

        status, body = ask(native_engine, channel, "/api/events")
        assert status == 200 and json.loads(body)["hit"] == [
            'Hit something: payload {"what": name}.'
        ]

        status, body = ask(
            native_engine, channel, "/api/events/hit", "POST", '{"what": "the wall"}'
        )
        assert status == 200 and json.loads(body) == {"emitted": "hit", "handlers": 1}
        assert native_engine.state["hits"] == ["the wall"]

        status, body = ask(native_engine, channel, "/api/log?match=hit%20the")
        assert status == 200 and [line["text"] for line in json.loads(body)["lines"]] == [
            "hit the wall"
        ]

        status, body = ask(native_engine, channel, "/api/events/nothing", "POST", "null")
        assert status == 404 and "hit" in json.loads(body)["events"]
    finally:
        native_engine._dev_channel = None
        channel.close()


def test_native_look_answers_that_it_is_unavailable(native_engine, tmp_path):
    channel = devchannel.Channel(tmp_path, _DevChannelHost(native_engine), port=0)
    try:
        status, body = ask(native_engine, channel, "/api/look")
        assert status == 501 and "Rust core" in json.loads(body)["error"]
    finally:
        channel.close()


def test_native_run_serves_the_port_and_releases_it_on_close(tmp_path, monkeypatch, capsys):
    native = pytest.importorskip("manifoldx._core_native")
    if not hasattr(native, "_NativeGraphics"):
        pytest.skip("native renderer is unavailable")
    from manifoldx._desktop import _NativeEngine

    monkeypatch.setenv("MANIFOLDX_CORE", "rust")
    app = tmp_path / "app.py"
    app.write_text(
        textwrap.dedent("""\
        from manifoldx import Engine

        engine = Engine("channel", width=16, height=16)
        engine.state.setdefault("n", 1)
        engine.run()
    """)
    )
    seen = {}

    def run_offscreen(engine):
        port = int(engine._dev_channel.url.rsplit(":", 1)[1])
        seen["port"], seen["state"] = port, dict(engine.state)
        seen["server"] = (tmp_path / ".manifoldx" / "server.json").exists()
        probe = socket.socket()
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            with pytest.raises(OSError):
                probe.bind(("127.0.0.1", port))
        finally:
            probe.close()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_NativeEngine, "run", run_offscreen)
    assert cli.main(["run", str(app), "--port", "0"]) == 0

    assert seen["server"] and seen["state"] == {"n": 1}
    assert "dev page and API at" in capsys.readouterr().err
    assert not (tmp_path / ".manifoldx" / "server.json").exists()
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", seen["port"]))
    finally:
        probe.close()


def test_native_debug_still_needs_the_python_core(tmp_path, monkeypatch, capsys):
    native = pytest.importorskip("manifoldx._core_native")
    if not hasattr(native, "_NativeGraphics"):
        pytest.skip("native renderer is unavailable")
    monkeypatch.setenv("MANIFOLDX_CORE", "rust")
    app = tmp_path / "app.py"
    app.write_text("from manifoldx import Engine\nEngine('c', width=16, height=16).run()\n")
    with pytest.raises(SystemExit) as raised:
        cli.main(["run", str(app), "--debug"])
    assert raised.value.code == 2
    assert "--debug requires the Python desktop dev channel" in capsys.readouterr().err
