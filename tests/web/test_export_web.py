"""manifoldx export --web: the app runs its setup in Python, then the folder holds the
transpiled code, the state it starts from, the scene and a report, and loads in Node."""

import gzip
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import manifoldx as mx
from manifoldx.cli import main
from manifoldx.web.export import VENDOR, export_web
from manifoldx.web.replay import compare

WEB_WALK = Path(__file__).parents[2] / "examples" / "web_walk.py"

RIG = """
class Rig:
    def __init__(self):
        self.x = 0.0
        self.speed = 2.0
        self.notes = "setup only"

    def step(self, dt: float):
        self.x = self.x + self.speed * dt

    def describe(self):
        return self.notes
"""

APP = """
import manifoldx as mx
from manifoldx.gui import Panel, ValueDisplay
from webx_rig import Rig

engine = mx.Engine("web", width=320, height=200)
engine.state.setdefault("level", 2)
rig = Rig()


def walk(query, dt):
    rig.step(dt)


def label():
    return f"x {rig.x:.1f}"


def setup_only():
    return "never in the browser"


@engine.on("startup")
def build(_):
    engine.spawn(
        mx.components.Mesh(mx.resources.cube(1, 1, 1)),
        mx.components.Material(mx.resources.BasicMaterial("#ff0000")),
        mx.components.Transform(pos=(0, 1, 0)),
    )
    engine.state["cam"] = {"yaw": 0.5, "dist": 12.0}
    engine.state["names"] = ["a", "b"]
    rig.x = 1.0


engine.system(walk)
engine.gui.append(Panel(children=[ValueDisplay(label)]))

if __name__ == "__main__":
    engine.run()
"""

BARE = """
import manifoldx as mx

engine = mx.Engine("bare", width=64, height=64)
engine.spawn(
    mx.components.Mesh(mx.resources.cube(1, 1, 1)),
    mx.components.Material(mx.resources.BasicMaterial("#ffffff")),
    mx.components.Transform(pos=(0, 0, 0)),
)
engine.run()
"""

ASYNC = """
import asyncio
import manifoldx as mx

engine = mx.Engine("async", width=64, height=64)


@engine.on("startup")
async def load(_):
    await asyncio.sleep(0)
    engine.state["loaded"] = "yes"


engine.run()
"""

BAD = """
import manifoldx as mx

engine = mx.Engine("bad", width=64, height=64)


def walk(query, dt):
    try:
        return 1
    except ValueError:
        return 2


engine.system(walk)
engine.run()
"""

LEFT_OUT = """
import numpy as np
import manifoldx as mx

engine = mx.Engine("left", width=64, height=64)
engine.state["kept"] = [1, 2.5, "three"]
engine.state[("terrain", 7)] = 1.0
engine.state["grid"] = np.zeros((2, 2))
engine.run()
"""

# Typed arrays and Sets become arrays, a Map (a dict) an object, as JSON.stringify sees them.
EXPECTED_FILES = {
    "app/__main__.js",
    "app/__main__.js.map",
    "app/webx_rig.js",
    "app/webx_rig.js.map",
    "pyrt.js",
    "manifoldx.js",
    "replay.mjs",
    "state.json",
    "scene.glb",
    "scene.export-report.json",
    "report.json",
    # The browser page.
    "index.html",
    "browser.js",
    "input.js",
    "scene.js",
    "gui.js",
    "sourcemap.js",
    "progress.js",
    "fonts/DejaVuSansMono.ttf",
    *(f"vendor/three/{p.relative_to(VENDOR)}" for p in VENDOR.rglob("*") if p.is_file()),
}


@pytest.fixture
def app_dir(tmp_path):
    """A folder for the app's files; the modules it imports are forgotten afterwards."""
    folder = tmp_path / "src"
    folder.mkdir()
    yield folder
    for name, m in list(sys.modules.items()):
        if str(getattr(m, "__file__", None) or "").startswith(str(folder)):
            del sys.modules[name]


def _export(folder: Path, out: Path, src: str, name: str = "app.py", **modules) -> int:
    for module, text in modules.items():
        (folder / f"{module}.py").write_text(text)
    (folder / name).write_text(src)
    return main(["export", "--web", str(folder / name), str(out)])


def _node(out: Path) -> dict:
    done = subprocess.run(
        ["node", str(out / "replay.mjs")], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _files(out: Path) -> set[str]:
    return {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}


def test_writes_every_file(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, APP, webx_rig=RIG) == 0
    assert _files(out) == EXPECTED_FILES
    js = (out / "app" / "__main__.js").read_text()
    assert js.rstrip().endswith("//# sourceMappingURL=__main__.js.map")
    assert json.loads((out / "app" / "__main__.js.map").read_text())["sources"] == ["app.py"]
    assert 'import * as $rt from "../pyrt.js";' in js  # one runtime, beside manifoldx.js


def test_the_report_lists_roots_transpiled_and_setup(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, APP, webx_rig=RIG) == 0
    report = json.loads((out / "report.json").read_text())
    assert [r["label"] for r in report["roots"]] == [
        "system walk",
        "ValueDisplay.getter label",
    ]
    assert {"__main__.walk", "__main__.label", "webx_rig.Rig.step"} <= set(report["transpiled"])
    assert {"__main__.setup_only", "__main__.build", "webx_rig.Rig.describe"} <= set(
        report["setup"]
    )
    assert not set(report["transpiled"]) & set(report["setup"])
    assert report["modules"] == {"__main__": "app/__main__.js", "webx_rig": "app/webx_rig.js"}
    assert report["state_left_out"] == []


def test_the_folder_loads_in_node_with_the_state_python_left(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, APP, webx_rig=RIG) == 0
    state = json.loads((out / "state.json").read_text())
    assert state["engine"] == {
        "w": 320,
        "h": 200,
        "camera": {
            "position": [0.0, 1.0, 2.0],
            "target": [0.0, 0.0, 0.0],
            "fov": 60.0,
            "up": [0.0, 1.0, 0.0],
            "near": 0.1,
            "far": 1000.0,
        },
    }
    assert state["modules"] == ["__main__", "webx_rig"]
    got = _node(out)
    assert got["roots"] == ["system walk", "ValueDisplay.getter label"]
    # engine.state as Python had it after startup, in Python's key order.
    assert got["state"] == {"level": 2, "cam": {"yaw": 0.5, "dist": 12.0}, "names": ["a", "b"]}
    assert list(got["state"]) == ["level", "cam", "names"]
    assert got["gui"] == ["x 1.0"]


def test_an_app_with_no_systems_and_no_gui(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, BARE) == 0
    assert json.loads((out / "report.json").read_text())["roots"] == []
    assert _node(out) == {"roots": [], "state": {}, "gui": []}


def test_an_async_startup_handler_runs_before_the_snapshot(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, ASYNC) == 0
    assert _node(out)["state"] == {"loaded": "yes"}


def test_state_that_cannot_cross_is_left_out_and_reported(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, LEFT_OUT) == 0
    assert _node(out)["state"] == {"kept": [1, 2.5, "three"]}
    left = json.loads((out / "report.json").read_text())["state_left_out"]
    assert [x["path"] for x in left] == ["engine.state[('terrain', 7)]", "engine.state['grid']"]


def test_a_transpile_error_exits_1_names_the_line_and_writes_nothing(app_dir, tmp_path, capsys):
    out = tmp_path / "out"
    assert _export(app_dir, out, BAD, name="bad.py") == 1
    line = BAD.splitlines().index("    try:") + 1
    assert f"bad.py:{line}: try/except is outside the subset" in capsys.readouterr().err
    assert not out.exists()


def test_the_module_form_names_the_module_file(app_dir, tmp_path, capsys, monkeypatch):
    pkg = app_dir / "webx_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(BAD)
    monkeypatch.chdir(app_dir)
    out = tmp_path / "out"
    assert main(["export", "--web", "-m", "webx_pkg.main", str(out)]) == 1
    line = BAD.splitlines().index("    try:") + 1
    assert f"webx_pkg/main.py:{line}:" in capsys.readouterr().err
    assert not out.exists()


def test_the_capture_restores_engine_run_and_main(app_dir, tmp_path):
    before_run, before_main = mx.Engine.run, sys.modules["__main__"]
    _export(app_dir, tmp_path / "out", APP, webx_rig=RIG)
    assert mx.Engine.run is before_run
    assert sys.modules["__main__"] is before_main


def test_the_scene_report_does_not_call_transpiled_code_dropped(app_dir, tmp_path, capsys):
    assert _export(app_dir, tmp_path / "out", APP, webx_rig=RIG) == 0
    err = capsys.readouterr().err
    assert "custom Python that runs every frame" not in err
    assert "ValueDisplay.getter = label" not in err  # transpiled too


# XML namespaces three.js passes to createElementNS: names, not requests.
NAMESPACES = {"http://www.w3.org/1999/xhtml"}


def _urls_outside_comments(text: str) -> list[str]:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"(^|[^:\\])//.*$", r"\1", text, flags=re.M)
    found = re.findall(r"https?://[^\s'\"`)]*", text)
    return [u for u in found if u not in NAMESPACES]


@pytest.fixture(scope="module")
def walk_page(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("walk_page") / "out"
    assert export_web(None, str(WEB_WALK), out, []) == 0
    return out


def test_the_page_has_every_file(walk_page):
    files = _files(walk_page)
    assert {"index.html", "browser.js", "input.js", "scene.js", "gui.js"} <= files
    assert {"sourcemap.js", "progress.js", "fonts/DejaVuSansMono.ttf"} <= files
    vendored = {f"vendor/three/{p.relative_to(VENDOR)}" for p in VENDOR.rglob("*") if p.is_file()}
    assert len(vendored) == 8 and vendored <= files


def test_index_html_names_the_title_and_the_scene_sizes(walk_page):
    page = (walk_page / "index.html").read_text()
    assert "<title>Web walk</title>" in page
    glb = (walk_page / "scene.glb").read_bytes()
    meta = dict(re.findall(r'<meta name="(scene-[a-z-]+)" content="(\d+)">', page))
    assert int(meta["scene-bytes"]) == len(glb)
    assert int(meta["scene-gzip-bytes"]) == len(gzip.compress(glb, 9))
    imports = json.loads(
        re.search(r'<script type="importmap">(.*?)</script>', page, re.S).group(1)
    )["imports"]
    assert imports == {
        "three": "./vendor/three/three.module.js",
        "three/addons/": "./vendor/three/addons/",
    }
    assert "serve this folder over HTTP (python -m http.server)" in page
    assert '<script type="module" src="browser.js"></script>' in page


def test_the_title_is_escaped(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, BARE.replace('"bare"', '"a <b> & c"')) == 0
    page = (out / "index.html").read_text()
    assert f"<title>{html.escape('a <b> & c')}</title>" in page


def test_no_file_requests_anything_outside_the_folder(walk_page):
    for f in walk_page.rglob("*"):
        if f.suffix not in {".js", ".mjs", ".html", ".json"}:
            continue  # the license, the font, the glb and the maps hold no code
        assert _urls_outside_comments(f.read_text()) == [], f


def test_the_url_check_bites():
    assert _urls_outside_comments('fetch("https://cdn.example/three.js") // https://ok') == [
        "https://cdn.example/three.js"
    ]
    assert _urls_outside_comments("/* https://ok */ <!-- http://ok -->") == []


FRAME = {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [320, 200]}


def _replay_both(folder: Path, out: Path, src: str, paths=(), **modules):
    """Export the app, then replay one frame in Python and in Node: (python, node)."""
    from manifoldx.web.replay import replay_node, replay_python

    assert _export(folder, out, src, **modules) == 0
    return replay_python(None, str(folder / "app.py"), [FRAME], list(paths)), replay_node(
        out, [FRAME], list(paths)
    )


SHELF = """
from dataclasses import dataclass
import manifoldx as mx

engine = mx.Engine("shelf", width=320, height=200)


@dataclass
class Item:
    name: str
    size: float


@dataclass
class Shelf:
    items: tuple


shelf = Shelf((Item("a", 1.0), Item("b", 2.0)))


def tick(query, dt: float):
    engine.state["names"] = [it.name for it in shelf.items]


engine.system(tick)
engine.run()
"""


def test_items_of_an_unannotated_tuple_field_type_from_live_values(app_dir, tmp_path):
    """gia-uh/manifoldx#142: a bare `tuple` annotation gives way to the live items' types, so
    the field Item.name is in the snapshot and the browser reads what Python reads."""
    py, node = _replay_both(app_dir, tmp_path / "out", SHELF)
    assert py[0]["state"]["names"] == ["a", "b"]
    assert compare(py, node) is None, str(compare(py, node))


UNKNOWN_READ = """
import manifoldx as mx

engine = mx.Engine("unknown", width=320, height=200)


class Item:
    def __init__(self, name):
        self.name = name


engine.state["item"] = Item("a")


def tick(query, dt: float):
    it = engine.state["item"]
    engine.state["name"] = it.name


engine.system(tick)
engine.run()
"""


def test_an_attribute_read_on_an_unknown_type_fails_naming_what_to_annotate(
    app_dir, tmp_path, capsys
):
    """gia-uh/manifoldx#142: a read whose receiver has no type cannot know which field to
    snapshot, so the export stops at its line instead of leaving the field out."""
    out = tmp_path / "out"
    assert _export(app_dir, out, UNKNOWN_READ) == 1
    line = UNKNOWN_READ.splitlines().index('    engine.state["name"] = it.name') + 1
    err = capsys.readouterr().err
    assert f"app.py:{line}: `it.name`" in err
    assert "annotate `it`" in err
    assert not out.exists()


GRID = """
import manifoldx as mx

engine = mx.Engine("grid", width=320, height=200)
LEVELS = {"a": [1.0, 2.0], "b": [3.0]}


def grid(levels):
    xs = [x for level in levels for x in level]
    return min(xs), max(xs)


X0, X1 = grid(LEVELS.values())


def tick(query, dt: float):
    engine.state["span"] = X1 - X0


engine.system(tick)
engine.run()
"""

LEVELS = """
from pathlib import Path

DATA = Path(__file__).parent
LEVELS = {"a": {"w": 3.0}, "root": {"w": float(len(str(DATA)))}}
"""

ROOMS = """
import manifoldx as mx
from webx_levels import LEVELS

engine = mx.Engine("rooms", width=320, height=200)
ROOMS = {"first": LEVELS["a"]}


def tick(query, dt: float):
    w: float = ROOMS["first"]["w"]
    engine.state["w"] = w * 2.0


engine.system(tick)
engine.run()
"""


def test_a_global_set_by_a_call_does_not_reach_the_call(app_dir, tmp_path):
    """A global's value comes from the snapshot: its initializer ran at setup, so the function
    it calls stays setup and the globals it reads stay out of the browser."""
    out = tmp_path / "out"
    py, node = _replay_both(app_dir, out, GRID)
    assert py[0]["state"]["span"] == 2.0
    assert compare(py, node) is None, str(compare(py, node))
    report = json.loads((out / "report.json").read_text())
    assert "__main__.grid" in report["setup"]
    assert report["globals"] == ["__main__.X0", "__main__.X1", "__main__.engine"]


def test_a_global_built_from_another_does_not_snapshot_that_one(app_dir, tmp_path):
    """ROOMS reads LEVELS at setup; reached code reads only ROOMS, so neither LEVELS nor the
    Path LEVELS' initializer read goes in the snapshot."""
    out = tmp_path / "out"
    py, node = _replay_both(app_dir, out, ROOMS, webx_levels=LEVELS)
    assert py[0]["state"]["w"] == 6.0
    assert compare(py, node) is None, str(compare(py, node))
    report = json.loads((out / "report.json").read_text())
    assert report["globals"] == ["__main__.ROOMS", "__main__.engine"]
    assert set(json.loads((out / "state.json").read_text())["globals"]) == {"__main__"}


HELD = """
import manifoldx as mx

engine = mx.Engine("held", width=320, height=200)


class Item:
    def __init__(self, name):
        self.name = name


PICK = Item("a")


def tick(query, dt: float):
    engine.state["p"] = PICK


engine.system(tick)
engine.run()
"""


def test_a_global_s_object_crosses_when_reached_code_only_passes_it(app_dir, tmp_path):
    """With no initializer to walk, the class of what a reached global holds is reached from
    the global's type, so the snapshot writes it."""
    out = tmp_path / "out"
    assert _export(app_dir, out, HELD) == 0
    assert "__main__.Item" in json.loads((out / "report.json").read_text())["classes"]


def test_startup_handlers_run_without_the_offline_extra(app_dir, tmp_path, monkeypatch):
    """The export needs a wgpu device for startup, not video: plain manifold-gfx, without
    imageio-ffmpeg, exports an app with a startup handler."""
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # `import imageio_ffmpeg` fails
    out = tmp_path / "out"
    assert _export(app_dir, out, ASYNC) == 0
    assert _node(out)["state"] == {"loaded": "yes"}


FIRST = """
import manifoldx as mx

engine = mx.Engine("first", width=320, height=200)


class P:
    def __init__(self):
        self.pt: tuple[float, float] = (1.0, 2.0)

    def first(self) -> tuple[float, float]:
        return self.pt


p = P()


def tick(query, dt: float):
    ax, ay = p.first()
    engine.state["x"] = ax * 2.0


engine.system(tick)
engine.run()
"""


def test_unpacking_a_call_result_types_from_the_return_annotation(app_dir, tmp_path):
    """gia-uh/manifoldx#143: the names unpacked from a call take the callee's return type."""
    py, node = _replay_both(app_dir, tmp_path / "out", FIRST)
    assert py[0]["state"]["x"] == 2.0
    assert compare(py, node) is None, str(compare(py, node))


READ = """
import manifoldx as mx

engine = mx.Engine("read", width=320, height=200)
engine.state["a"] = 1.5
engine.state["b"] = 2.0


def read() -> tuple[float, float]:
    return engine.state["a"], engine.state["b"]


def tick(query, dt: float):
    a, b = read()
    engine.state["s"] = a + b


engine.system(tick)
engine.run()
"""


def test_a_return_annotation_wins_over_an_unknown_inferred_return(app_dir, tmp_path):
    """gia-uh/manifoldx#143: reads of engine.state are unknown; the annotation types them."""
    py, node = _replay_both(app_dir, tmp_path / "out", READ)
    assert py[0]["state"]["s"] == 3.5
    assert compare(py, node) is None, str(compare(py, node))


MOVING = """
import manifoldx as mx
from manifoldx.gui import Panel, Text

engine = mx.Engine("moving", width=320, height=200)


class Hud:
    def __init__(self):
        self.label = Text("frame 0")
        self.panel = Panel(children=[self.label], offset=(10, 20))
        self.n = 0


hud = Hud()
engine.gui.append(hud.panel)


def tick(query, dt: float):
    hud.n += 1
    x, y = hud.panel.offset
    hud.panel.offset = (x + 3, y)
    hud.label.text = f"frame {hud.n}"
    hud.panel.style_overrides["width"] = 100 + hud.n


engine.system(tick)
engine.run()
"""


def test_widget_writes_replay_in_node_as_in_python(app_dir, tmp_path):
    """gia-uh/manifoldx#147: a panel's offset, a Text's text and a style override written by
    a system are fields of the widget objects the page reads, the same values as in Python."""
    out = tmp_path / "out"
    assert _export(app_dir, out, MOVING) == 0
    from manifoldx.web.replay import replay_node, replay_python

    frame = {"keys": [], "just": [], "d": [0, 0], "buttons": [], "clicks": [], "vp": [320, 200]}
    paths = ["hud.panel.offset", "hud.label.text", "hud.panel.style_overrides"]
    py = replay_python(None, str(app_dir / "app.py"), [frame] * 4, paths)
    assert py[-1]["paths"] == {
        "hud.panel.offset": [22, 20],
        "hud.label.text": "frame 4",
        "hud.panel.style_overrides": {"width": 104},
    }
    mismatch = compare(py, replay_node(out, [frame] * 4, paths))
    assert mismatch is None, str(mismatch)


def test_the_gui_spec_names_each_widget_and_its_base_style(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, MOVING) == 0
    panel = json.loads((out / "state.json").read_text())["gui"]["panels"][0]
    assert (panel["widget"], panel["overrides"]) == (0, {})
    assert panel["children"][0]["widget"] == 1
    assert panel["base"] == panel["style"]


UNAPPLIED = """
import manifoldx as mx
from manifoldx.gui import Panel, Text

engine = mx.Engine("unapplied", width=320, height=200)


class Hud:
    def __init__(self):
        self.panel = Panel(children=[Text("hi")])


hud = Hud()
engine.gui.append(hud.panel)


def tick(query, dt: float):
    {write}


engine.system(tick)
engine.run()
"""


@pytest.mark.parametrize(
    "write, what",
    [
        ("hud.panel.visible = False", "`hud.panel.visible`, a field of a Panel"),
        ('hud.panel.anchor = "center"', "`hud.panel.anchor`, a field of a Panel"),
        ('hud.panel.style = "hud"', "`hud.panel.style`, a field of a Panel"),
        ("engine.gui.scale = 2.0", "`engine.gui.scale`, a field of the GUI"),
    ],
)
def test_a_widget_write_the_page_does_not_apply_fails_the_export(
    app_dir, tmp_path, capsys, write, what
):
    """gia-uh/manifoldx#147: the desktop applies the write and the page would not, so the
    export stops at its line rather than differ in silence."""
    out = tmp_path / "out"
    src = UNAPPLIED.replace("{write}", write)
    assert _export(app_dir, out, src) == 1
    line = src.splitlines().index(f"    {write}") + 1
    err = capsys.readouterr().err
    assert f"app.py:{line}: writing {what} the page does not apply, is outside the subset" in err
    assert "(reached from system tick)" in err
    assert not out.exists()
