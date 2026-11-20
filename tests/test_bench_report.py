"""report.html: one self-contained page per run folder."""

import json

import numpy as np

from manifoldx import bench
from manifoldx.bench import checks
from manifoldx.bench.cli import main
from manifoldx.bench.folder import write_run
from manifoldx.bench.runner import run_scenario
from test_bench_runner import make


def _folder(tmp_path, budgets=None):
    sc, _ = make([bench.mark("a"), bench.hold(frames=4), bench.shot("s")])
    run = run_scenario(sc, size=(64, 48), budgets=budgets)
    return write_run(run, tmp_path / "runs", None), run


def test_every_run_has_a_report(tmp_path):
    folder, run = _folder(tmp_path)
    page = (folder / "report.html").read_text()
    p50 = run.summary["phases"]["a"]["p50_ms"]
    assert "tiny" in page and ">a<" in page and f"{p50:.1f}" in page and page.count("<svg") >= 1


def test_the_report_is_self_contained(tmp_path):
    folder, _ = _folder(tmp_path)
    page = (folder / "report.html").read_text()
    assert not any(bad in page for bad in ('src="http', "href=\"http", "<script", "@import", "url("))
    assert 'src="data:image/png;base64,' in page


def test_a_failed_check_is_in_the_report(tmp_path):
    folder, run = _folder(tmp_path, budgets=[checks.Check("fast", {"p95_ms": 0.001})])
    page = (folder / "report.html").read_text()
    (c,) = run.summary["checks"]
    assert "fail" in page and c["message"].replace(">", "&gt;") in page


def test_strings_from_the_run_are_escaped(tmp_path):
    sc, _ = make([bench.mark("<b>x</b>"), bench.hold(frames=2)])
    folder = write_run(run_scenario(sc, size=(64, 48), label="<i>l</i>"), tmp_path / "runs", None)
    page = (folder / "report.html").read_text()
    assert "<b>x</b>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page and "<i>l</i>" not in page


def test_report_rewrites_an_old_run(tmp_path, capsys):
    folder, _ = _folder(tmp_path)
    summary = json.loads((folder / "run.json").read_text())
    summary["schema"] = 2
    summary.pop("checks")
    (folder / "run.json").write_text(json.dumps(summary))
    frames = dict(np.load(folder / "frames.npz"))
    frames.pop("bench_ms")
    np.savez(folder / "frames.npz", **frames)
    (folder / "report.html").unlink()
    assert main(["report", str(folder)]) == 0
    assert str(folder / "report.html") in capsys.readouterr().out and (folder / "report.html").exists()


def test_a_report_that_fails_leaves_the_run(tmp_path, monkeypatch, capsys):
    from manifoldx.bench import report

    def boom(folder):
        raise RuntimeError("no page")

    monkeypatch.setattr(report, "write_report", boom)
    folder, _ = _folder(tmp_path)
    assert (folder / "run.json").exists() and not (folder / "report.html").exists()
    assert "no page" in capsys.readouterr().err


def test_names_in_headers_and_charts_are_escaped(tmp_path):
    from manifoldx.bench.report import write_report

    folder, _ = _folder(tmp_path)
    summary = json.loads((folder / "run.json").read_text())
    summary["phases"]["a"]["split_ms"] = {"<u>part</u>": 1.0}
    (folder / "run.json").write_text(json.dumps(summary))
    with open(folder / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": 0.1, "kind": "node", "ev": "<u>ev</u>", "level": 0}) + "\n")
    page = write_report(folder).read_text()
    assert "<u>" not in page and "&lt;u&gt;part&lt;/u&gt;" in page and "&lt;u&gt;ev&lt;/u&gt;" in page
