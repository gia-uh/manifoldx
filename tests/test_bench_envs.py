"""--against and --manifoldx run the other side through uv; a fake uv on the PATH stands in for it,
so no environment is built."""

import json
import os
import shutil
import sys
import textwrap

import pytest

from manifoldx.bench import envs
from manifoldx.bench.cli import main
from test_bench_cli import project  # noqa: F401

FAKE_UV = textwrap.dedent('''\
    #!{python}
    import json, os, subprocess, sys
    args = sys.argv[1:]
    with open({log!r}, "a") as f:
        f.write(json.dumps({{"args": args, "cwd": os.getcwd(),
                             "env": os.environ.get("UV_PROJECT_ENVIRONMENT")}}) + "\\n")
    assert args[0] == "run"
    i = 1
    while args[i].startswith("--"):
        i += 2  # --project P, --with S, --with-editable S
    if args[i] == "manifoldx":
        from manifoldx.cli import main
        sys.argv = args[i:]
        sys.exit(main(args[i + 1:]))
    sys.exit(subprocess.call(args[i:]))
''')


@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "uv.log"
    (bin_ / "uv").write_text(FAKE_UV.format(python=sys.executable, log=str(log)))
    (bin_ / "uv").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("MANIFOLDX_BENCH_CACHE", str(tmp_path / "cache"))
    return log


def _calls(log):
    return [json.loads(x) for x in log.read_text().splitlines()]


def _other(root, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(root, other, ignore=shutil.ignore_patterns("bench", "bin", "cache", "other", "__pycache__",
                                                                  "uv.log"))
    return other


def test_against_runs_the_other_copy_through_uv(project, fake_uv, tmp_path, capsys):  # noqa: F811
    other = _other(project, tmp_path)
    assert main(["run", "cli_demo", "--against", str(other), "--smoke", "--json"]) == 0
    (call,) = _calls(fake_uv)
    assert call["args"][:3] == ["run", "--project", str(other)] and call["cwd"] == str(other)
    assert call["env"].startswith(str(tmp_path / "cache")) and not (other / ".venv").exists()
    out = json.loads(capsys.readouterr().out)
    assert [s["label"] for s in out["sides"]] == ["other", "this copy"]


def test_manifoldx_spec_goes_to_uv_with(project, fake_uv, capsys):  # noqa: F811
    assert main(["run", "cli_demo", "--manifoldx", "manifold-gfx==9.9.9", "--smoke", "--json"]) == 0
    (call,) = _calls(fake_uv)
    assert call["args"][3:5] == ["--with", "manifold-gfx==9.9.9"]
    out = json.loads(capsys.readouterr().out)
    assert [s["label"] for s in out["sides"]] == ["this copy", "manifold-gfx==9.9.9"]


def test_a_directory_spec_is_editable(project, fake_uv, tmp_path, capsys):  # noqa: F811
    spec = tmp_path / "mx"
    spec.mkdir()
    assert main(["run", "cli_demo", "--manifoldx", str(spec), "--smoke", "--json"]) == 0
    (call,) = _calls(fake_uv)
    assert call["args"][3:5] == ["--with-editable", str(spec)]


def test_a_sweep_crosses_with_the_copies(project, fake_uv, tmp_path, capsys):  # noqa: F811
    other = _other(project, tmp_path)
    assert main(["run", "cli_demo", "--against", str(other), "--param", "frames=2,3", "--smoke", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [s["label"] for s in out["sides"]] == ["other frames=2", "other frames=3",
                                                  "this copy frames=2", "this copy frames=3"]
    assert len(_calls(fake_uv)) == 2


def test_against_without_uv_is_refused(project, monkeypatch, tmp_path, capsys):  # noqa: F811
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert main(["run", "cli_demo", "--against", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "uv" in err and "bench compare" in err


def test_against_a_directory_without_a_project_is_refused(project, fake_uv, tmp_path, capsys):  # noqa: F811
    (tmp_path / "empty").mkdir()
    assert main(["run", "cli_demo", "--against", str(tmp_path / "empty")]) == 2
    assert "pyproject.toml" in capsys.readouterr().err


def test_against_and_manifoldx_together_are_refused(project, fake_uv, tmp_path, capsys):  # noqa: F811
    assert main(["run", "cli_demo", "--against", str(tmp_path), "--manifoldx", "x"]) == 2
    assert "one at a time" in capsys.readouterr().err


def test_the_key_depends_on_project_and_spec(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert envs.key(a, None) != envs.key(b, None) != envs.key(b, "x")
    assert envs.key(a, "x") == envs.key(tmp_path / "a" / ".." / "a", "x")


# --- the cache -----------------------------------------------------------------------------------


def _env(cache, name, kb, used):
    d = cache / name
    (d / "venv").mkdir(parents=True)
    (d / "venv" / "blob").write_bytes(b"\0" * kb * 1024)
    (d / "used").write_text("")
    os.utime(d / "used", (used, used))


def test_the_cache_drops_the_oldest_past_its_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("MANIFOLDX_BENCH_CACHE", str(tmp_path))
    for name, used in (("old", 1), ("mid", 2), ("new", 3)):
        _env(tmp_path, name, 400, used)
    dropped = envs.trim(cap=0.9, keep={"new"})
    assert dropped == ["old"] and sorted(p.name for p in tmp_path.iterdir()) == ["mid", "new"]


def test_the_cache_never_drops_what_this_run_uses(tmp_path, monkeypatch):
    monkeypatch.setenv("MANIFOLDX_BENCH_CACHE", str(tmp_path))
    _env(tmp_path, "only", 2048, 1)
    assert envs.trim(cap=1, keep={"only"}) == []


def test_the_cap_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("MANIFOLDX_BENCH_CACHE_GB", "0.5")
    assert envs.cap_mb() == 512


def test_clean_empties_the_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MANIFOLDX_BENCH_CACHE", str(tmp_path / "c"))
    _env(tmp_path / "c", "a", 512, 1)
    assert main(["clean"]) == 0
    assert not (tmp_path / "c").exists() and "0.5 MB" in capsys.readouterr().out
    assert main(["clean"]) == 0  # nothing left: still fine


def test_a_set_says_how_much_the_cache_grew(project, fake_uv, tmp_path, capsys):  # noqa: F811
    other = _other(project, tmp_path)
    assert main(["run", "cli_demo", "--against", str(other), "--smoke", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cache_added_mb"] >= 0 and "cache_mb" in out


def test_a_file_hardlinked_from_elsewhere_is_not_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("MANIFOLDX_BENCH_CACHE", str(tmp_path / "c"))
    _env(tmp_path / "c", "a", 512, 1)
    shared = tmp_path / "uv-cache-file"
    shared.write_bytes(b"\0" * 1024 * 1024)
    os.link(shared, tmp_path / "c" / "a" / "venv" / "linked")  # as uv links a package from its own cache
    assert envs.size_mb() == 0.5  # removing the cache would free only its own half megabyte
