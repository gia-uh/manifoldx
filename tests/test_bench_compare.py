"""bench compare: run folders, sides and sets into a table with spread, ratios and notes, and sheets."""

import json
import shutil

import numpy as np

from manifoldx.bench import compare
from manifoldx.bench.cli import main
from manifoldx.bench.folder import write_run
from manifoldx.bench.runner import Run
from manifoldx.image import read_png


def _run(tmp_path, p50, load=0.1, cpus=8, shots=None, wall=None):
    phase = {"frames": 10, "p50_ms": p50, "p95_ms": p50 * 1.5, "draw_p50_ms": p50 / 2, "cpu_p50_ms": p50,
             "split_ms": {}, "systems_ms": {"a.spin": p50 / 3}, "spans_ms": {},
             "counters": {"draws": {"mean": 4.0, "max": 4}}, "app": {}, "rss_max_mb": 100.0}
    summary = {"schema": 3, "scenario": "s", "params": {}, "phases": {"main": phase}, "phase_order": ["main"],
               "host": {"cpus": cpus, "load_max": load}, "app": {}, "memory": {}, "error": None, "label": None}
    frames = {"wall_ms": np.full(10, p50) if wall is None else np.asarray(wall, float),
              "phase": np.zeros(10, np.int32)}
    return write_run(Run(summary, frames, [], shots or {}), tmp_path / "runs", None)


def _row(rows, key):
    return next(r for r in rows if r.key == key)


def test_ratios_are_against_the_first_side(tmp_path):
    sides = compare.load_sides([_run(tmp_path, 20.0), _run(tmp_path, 10.0)])
    p50 = _row(compare.rows(sides), "p50_ms")
    assert p50.values == [20.0, 10.0] and p50.ratios == [1.0, 0.5] and p50.note == ""


def test_close_ratios_and_loaded_runs_get_a_note(tmp_path):
    sides = compare.load_sides([_run(tmp_path, 20.0), _run(tmp_path, 19.0)])
    assert "needs a quiet re-run" in _row(compare.rows(sides), "p50_ms").note
    assert _row(compare.rows(sides), "draws").note == ""  # a count is exact, not noisy
    sides = compare.load_sides([_run(tmp_path, 20.0, load=6.0), _run(tmp_path, 10.0)])
    assert "taken under load (6.0 of 8 cpus)" in _row(compare.rows(sides), "p50_ms").note


def test_one_run_a_side_spreads_over_its_frames(tmp_path):
    a = _run(tmp_path, 20.0, wall=[10, 15, 18, 19, 20, 20, 21, 22, 25, 30])
    sides = compare.load_sides([a, _run(tmp_path, 10.0)])
    lo, hi = _row(compare.rows(sides), "p50_ms").spreads[0]
    assert (lo, hi) == (np.percentile([10, 15, 18, 19, 20, 20, 21, 22, 25, 30], 25),
                        np.percentile([10, 15, 18, 19, 20, 20, 21, 22, 25, 30], 75))


def test_a_set_folder_is_its_sides(tmp_path):
    s = tmp_path / "set"
    runs = []
    for k, p50s in enumerate(([20.0, 22.0], [10.0, 12.0])):
        (s / f"{k}-side{k}").mkdir(parents=True)
        for r, p50 in enumerate(p50s):
            made = _run(tmp_path, p50)
            shutil.move(made, s / f"{k}-side{k}" / made.name)
            runs.append({"side": k, "round": r, "folder": f"{k}-side{k}/{made.name}", "pid": 1, "exit": 0,
                         "error": None})
    (s / "set.json").write_text(json.dumps({"scenario": "s", "repeat": 2, "runs": runs, "sides": [
        {"label": "a", "root": ".", "params": {}, "manifoldx": None, "other": False},
        {"label": "b", "root": ".", "params": {}, "manifoldx": None, "other": False}]}))
    sides = compare.load_sides([s])
    assert [x.label for x in sides] == ["a", "b"] and [len(x.runs) for x in sides] == [2, 2]
    p50 = _row(compare.rows(sides), "p50_ms")
    assert p50.values == [21.0, 11.0] and p50.spreads == [(20.0, 22.0), (10.0, 12.0)]


def test_shots_with_the_same_name_share_a_sheet(tmp_path):
    red = np.zeros((8, 12, 3), np.uint8)
    red[..., 0] = 255
    blue = np.zeros((8, 12, 3), np.uint8)
    blue[..., 2] = 255
    sides = compare.load_sides([_run(tmp_path, 20.0, shots={"street": red}),
                                _run(tmp_path, 10.0, shots={"street": blue, "only_b": blue})])
    out = compare.sheets(sides, tmp_path / "out")
    img = read_png(out["street"])
    assert img.shape[1] > 24 and img[4, 2, 0] == 255 and img[4, -3, 2] == 255
    assert "only_b" not in out  # a shot one side lacks gets no sheet


def test_compare_cli_writes_the_table_and_sheets(tmp_path, capsys):
    a, b = _run(tmp_path, 20.0), _run(tmp_path, 10.0)
    assert main(["compare", str(a), str(b), "--out", str(tmp_path / "cmp")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| phase | probe |") and (tmp_path / "cmp" / "compare.md").exists()


def test_compare_needs_two_sides(tmp_path, capsys):
    assert main(["compare", str(_run(tmp_path, 20.0))]) == 2
    assert "two" in capsys.readouterr().err


def test_a_side_with_no_runs_is_left_out_not_the_table(tmp_path):
    sides = compare.load_sides([_run(tmp_path, 20.0), _run(tmp_path, 10.0)])
    sides.insert(0, compare.SideRuns("crashed"))  # every run of the baseline failed
    out = compare.write(sides, tmp_path / "cmp")
    assert any(r["key"] == "p50_ms" for r in out["rows"])
    assert "crashed: no run finished, left out" in out["table"]
