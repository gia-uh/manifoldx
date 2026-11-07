"""Camera.project in frame code: the page's projection is the desktop's, its matrices
computed in float32 as get_projection_matrix @ get_view_matrix are, the rest in float64."""

from pathlib import Path

from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

PROJECTING = Path(__file__).parent / "apps" / "projecting.py"
IDLE = {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [320, 200]}
FRAMES = [IDLE] * 60 + [{**IDLE, "vp": [640, 480]}] * 60


def _replay(tmp_path, patch=None):
    out = tmp_path / "out"
    assert export_web(None, str(PROJECTING), out, []) == 0
    if patch is not None:
        js = out / "manifoldx.js"
        src = js.read_text()
        assert patch[0] in src
        js.write_text(src.replace(patch[0], patch[1]))
    return replay_python(None, str(PROJECTING), FRAMES, []), replay_node(out, FRAMES, [])


def test_project_replays(tmp_path):
    python, node = _replay(tmp_path)
    assert compare(python, node) is None
    xy = [e["state"]["xy"] for e in python]
    assert len(xy[0]) == 5 and len(xy[0][0]) == 2
    assert xy[10] != xy[11]  # the camera moves
    # The origin, the camera's target, lands at the viewport's centre, before and after the resize.
    for e, (w, h) in ((python[30], (320, 200)), (python[90], (640, 480))):
        x, y = e["state"]["xy"][0]
        assert abs(x - w / 2) < 1e-3 and abs(y - h / 2) < 1e-3


def test_a_point_behind_the_camera_has_negative_w(tmp_path):
    python, node = _replay(tmp_path)
    for trace in (python, node):
        assert all(e["state"]["w"][4] < 0 for e in trace)
        assert all(w > 0 for e in trace for w in e["state"]["w"][:4])


def test_the_matrices_are_the_desktops_float32(tmp_path):
    marker = 'nd.matmul(P, V, "f32")'
    python, node = _replay(tmp_path, (marker, 'nd.matmul(P, V, "f64")'))
    mismatch = compare(python, node)
    assert mismatch is not None and mismatch.key.startswith("state.xy"), str(mismatch)
