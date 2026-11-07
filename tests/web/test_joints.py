"""set_joints in frame code: poses cross the export in ecs.bin, the page's store keeps them by
the desktop's rules (joints past J keep their matrix, a destroyed or respawned slot is back at
rest) and the trace carries each posed entity's joints."""

from pathlib import Path

import numpy as np
import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

SKINNED = Path(__file__).parent / "apps" / "skinned.py"
IDLE = {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [320, 200]}
FRAMES = [IDLE] * 90


def _export(tmp_path, args=()):
    out = tmp_path / "out"
    assert export_web(None, str(SKINNED), out, list(args)) == 0
    return out


def _replay(tmp_path, args=()):
    out = _export(tmp_path, args)
    python = replay_python(None, str(SKINNED), FRAMES, [], app_args=list(args))
    return python, replay_node(out, FRAMES, [])


def _rows(matrices):
    """(J, 4, 4) row-major matrices as the trace writes them: J lists of 16 floats."""
    return np.asarray(matrices, np.float32).reshape(len(matrices), 16).tolist()


def test_skinned_replays(tmp_path):
    python, node = _replay(tmp_path)
    assert compare(python, node) is None
    for trace in (python, node):
        assert all("joints" in e for e in trace)
        assert sorted(trace[20]["joints"]) == ["0", "1", "2"]
        assert all(len(m) == 3 and len(m[0]) == 16 for m in trace[20]["joints"].values())
    assert python[20]["joints"]["0"] != python[21]["joints"]["0"]  # posed every frame


def test_a_setup_pose_crosses_the_export(tmp_path):
    from manifoldx.web.export import capture_engine

    engine, _ = capture_engine(None, str(SKINNED), [])
    expected = _rows(engine._render_pipeline._joints.pose(0, 3))
    assert expected != _rows(np.tile(np.eye(4), (3, 1, 1)))
    python, node = _replay(tmp_path)
    for trace in (python, node):
        assert trace[0]["joints"] == {"0": expected}  # frame 0 poses nothing


def test_fewer_joints_leave_the_rest_where_they_were(tmp_path):
    python, node = _replay(tmp_path)
    for trace in (python, node):
        nine = trace[9]["joints"]["1"]
        for e in trace[10:30]:
            lone = e["joints"]["1"]
            assert lone[2] == nine[2]
            assert lone[2] != _rows([np.eye(4)])[0]
            assert lone[0] != nine[0]


def test_a_respawned_slot_starts_at_rest(tmp_path):
    python, node = _replay(tmp_path)
    for trace in (python, node):
        assert "2" in trace[29]["joints"]
        assert 2 in trace[31]["store"]["alive"]
        for e in trace[30:40]:
            assert "2" not in e["joints"]
        assert "2" in trace[40]["joints"]


def test_keeping_a_pose_across_destroy_is_a_mismatch(tmp_path):
    out = _export(tmp_path)
    js = out / "manifoldx.js"
    src = js.read_text()
    marker = "for (const i of gone) this.joints.delete(i);"
    assert marker in src
    spawn_marker = "for (const i of indices) this.joints.delete(i);"
    assert spawn_marker in src
    js.write_text(src.replace(marker, "").replace(spawn_marker, ""))
    python = replay_python(None, str(SKINNED), FRAMES, [])
    mismatch = compare(python, replay_node(out, FRAMES, []))
    assert mismatch is not None and mismatch.frame == 31, str(mismatch)


def test_set_joints_on_an_unskinned_entity_raises_as_the_desktop(tmp_path):
    out = _export(tmp_path, ["--box"])
    message = "set_joints needs an entity whose geometry has joints (Mesh(joints=...))"
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        replay_python(None, str(SKINNED), FRAMES, [], app_args=["--box"])
    with pytest.raises(
        RuntimeError, match=r"ValueError: set_joints needs an entity whose geometry"
    ):
        replay_node(out, FRAMES, [])


def test_set_joints_shapes_raise_as_the_desktop(tmp_path):
    out = _export(tmp_path, ["--shapes"])
    message = (
        "set_joints expects indices of shape (N,) and matrices of shape (N, J, 4, 4) "
        "with 1 <= J <= 32, got (3,) and (3, 3, 3, 3)"
    )
    with pytest.raises(ValueError) as python_error:
        replay_python(None, str(SKINNED), FRAMES, [], app_args=["--shapes"])
    assert str(python_error.value) == message
    with pytest.raises(RuntimeError) as node_error:
        replay_node(out, FRAMES, [])
    assert f"ValueError: {message}" in str(node_error.value)


def test_past_eight_posed_the_trace_is_a_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("SKINNED_N", "12")
    python, node = _replay(tmp_path)
    assert compare(python, node) is None
    for trace in (python, node):
        assert set(trace[20]["joints"]) == {"digest", "n"}
        assert trace[20]["joints"]["n"] == 12
    assert python[20]["joints"]["digest"] != python[21]["joints"]["digest"]
