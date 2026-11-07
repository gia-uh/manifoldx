"""manifoldx run --web: the WebReloader rebuilds the export into version folders on every save,
taking engine.state from the tab; here the tab is scripted (FakeTab)."""

import json
import os
import re
import shutil
import time
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
