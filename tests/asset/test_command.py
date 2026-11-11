"""The `manifoldx asset` command: lint's exit codes and JSON, a missing drawing, --again and render's blender.

Everything runs from `main([...])`; the examples carry their blueprints, so lint needs no network.
"""

import json
from pathlib import Path

import manifoldx.asset.blender as blender
from manifoldx.cli import main

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "models"


def test_lint_on_the_hydrant_exits_zero(capsys):
    assert main(["asset", str(EXAMPLES / "hydrant.py"), "lint"]) == 0
    assert "A-SEEN warn" in capsys.readouterr().out


def test_lint_on_the_armchair_exits_one_and_json_parses(capsys):
    assert main(["asset", str(EXAMPLES / "armchair.py"), "lint"]) == 1
    assert "A-SEEN error" in capsys.readouterr().out

    assert main(["asset", str(EXAMPLES / "armchair.py"), "lint", "--json"]) == 1
    rows = json.loads(capsys.readouterr().out)
    assert [row["code"] for row in rows][0] == "A-SEEN"
    assert any(row["level"] == "error" for row in rows)


def test_a_missing_cached_image_says_to_run_draw(tmp_path, capsys):
    asset = tmp_path / "needs.py"
    asset.write_text(
        "from manifoldx.asset import Asset\n"
        "\n"
        "class Needs(Asset):\n"
        "    name = 'needs'\n"
        "\n"
        "    def build(self, c):\n"
        "        c.load('bp-front')\n"
        "        return c.scene()\n",
        encoding="utf-8")
    assert main(["asset", str(asset), "build"]) == 2
    error = capsys.readouterr().err
    assert "bp-front" in error and "draw" in error


def test_again_deletes_only_the_named_images(tmp_path):
    asset = tmp_path / "quiet.py"
    asset.write_text(
        "from manifoldx.asset import Asset\n"
        "\n"
        "class Quiet(Asset):\n"
        "    name = 'quiet'\n"
        "\n"
        "    def build(self, c):\n"
        "        return c.scene()\n",
        encoding="utf-8")
    images = tmp_path / "quiet.asset" / "images"
    images.mkdir(parents=True)
    for name in ("bp-front", "bp-left"):
        (images / f"{name}.png").write_bytes(b"x")
        (images / f"{name}.json").write_text('{"key": "x"}', encoding="utf-8")

    assert main(["asset", str(asset), "build", "--again", "bp-front"]) == 0
    assert not (images / "bp-front.png").exists()
    assert not (images / "bp-front.json").exists()
    assert (images / "bp-left.png").exists() and (images / "bp-left.json").exists()


def test_render_exits_two_when_blender_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(blender.shutil, "which", lambda name: None)
    assert main(["asset", str(EXAMPLES / "hydrant.py"), "render"]) == 2
    error = capsys.readouterr().err
    assert "blender" in error and "PATH" in error
