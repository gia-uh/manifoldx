"""Layer 2 end to end: a Scene, a Report, and a tiny Asset that builds two touching boxes."""

import json
from pathlib import Path

import numpy as np
import pytest

from manifoldx.asset import Asset, Context, Report, Result, Scene
from manifoldx.modeling.lint import Finding
from manifoldx.modeling.views import extrude

SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def box(size=1.0):
    m = extrude(SQUARE * size / 2, size)
    return m


def test_scene_names_its_meshes_in_order_and_places_them():
    scene = Scene()
    scene.add("base", box(), at=(0.0, -0.5, 0.0))
    scene.add("cap", box(0.5), at=(0.0, 0.25, 0.0))
    assert list(scene.meshes) == ["base", "cap"] and len(scene) == 2
    assert scene["base"].positions[:, 1].max() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="base"):
        scene.add("base", box())


def test_scene_writes_a_glb(tmp_path):
    scene = Scene()
    scene.add("base", box(), at=(0.0, -0.5, 0.0))
    scene.add("cap", box(0.5), at=(0.0, 0.25, 0.0))
    path = tmp_path / "model.glb"
    scene.to_glb(path)
    assert path.read_bytes()[:4] == b"glTF"


def test_report_counts_levels_and_formats_each_finding():
    report = Report([
        Finding("A-FLOAT", "error", part="cap", value=1.2, message="floats; nearest part base",
                hint="attach to the nearest part"),
        Finding("A-SEEN", "warn", part="door", view="front", value=0.66, message="differs",
                hint="another builder"),
    ])
    assert len(report) == 2
    assert [f.code for f in report.errors] == ["A-FLOAT"]
    assert [f.code for f in report.warnings] == ["A-SEEN"]
    assert report.exit_code == 1
    assert report.lines() == [
        "A-FLOAT error cap: floats; nearest part base (1.200) — attach to the nearest part",
        "A-SEEN warn door front: differs (0.660) — another builder",
    ]
    rows = json.loads(report.to_json())
    assert rows[0]["code"] == "A-FLOAT" and rows[0]["value"] == 1.2


def test_a_report_without_errors_exits_zero():
    assert Report([Finding("A-TRIS", "warn", part="ball")]).exit_code == 0
    assert Report().exit_code == 0


class TwoBoxes(Asset):
    """Two boxes of size 1 that share a face, so nothing floats."""

    height = 2.0

    def build(self, c):
        scene = c.scene()
        scene.add("base", box(), at=(0.0, -0.5, 0.0))
        scene.add("cap", box(), at=(0.0, 0.5, 0.0))
        return scene


def test_an_asset_runs_end_to_end_and_finds_nothing_wrong(tmp_path):
    result = TwoBoxes().run(cache=tmp_path / "two-boxes.asset")
    assert isinstance(result, Result) and isinstance(result.scene, Scene)
    assert list(result.scene.meshes) == ["base", "cap"]
    assert result.textures == {}
    assert result.report.findings == [] and result.exit_code == 0


def test_the_default_cache_dir_sits_next_to_the_file_that_defines_the_class():
    assert TwoBoxes().cache_dir() == Path(__file__).resolve().parent / "twoboxes.asset"


def test_the_name_defaults_to_the_class_name_lowercased():
    class GardenGnome(Asset):
        pass

    assert GardenGnome.name == "gardengnome"


def test_an_asset_without_build_says_so(tmp_path):
    class Empty(Asset):
        pass

    with pytest.raises(NotImplementedError, match="Empty"):
        Empty().run(cache=tmp_path / "empty.asset", stages=("build",))


def test_lint_alone_builds_first(tmp_path):
    class Counted(Asset):
        built = 0

        def build(self, c):
            Counted.built += 1
            return c.scene()

    result = Counted().run(cache=tmp_path / "counted.asset", stages=("lint",))
    assert Counted.built == 1 and result.report.findings == []


def test_a_floating_part_is_an_error_through_the_pipeline(tmp_path):
    class Floats(Asset):
        def build(self, c):
            scene = c.scene()
            scene.add("base", box(), at=(0.0, -0.5, 0.0))
            scene.add("cap", box(), at=(0.0, 3.0, 0.0))
            return scene

    result = Floats().run(cache=tmp_path / "floats.asset", stages=("build", "lint"))
    assert [f.code for f in result.report.errors] == ["A-FLOAT"] and result.exit_code == 1


def test_the_assets_own_check_adds_to_the_standard_lint(tmp_path):
    class Opinionated(Asset):
        def build(self, c):
            return c.scene()

        def check(self, c, scene):
            return [Finding("A-CODE", "warn", part="door", message="still in its code colour")]

    result = Opinionated().run(cache=tmp_path / "opinionated.asset", stages=("build", "lint"))
    assert result.report.lines() == ["A-CODE warn door: still in its code colour"]


def test_an_expectation_compares_the_camera_with_the_drawing(tmp_path):
    class Mirrored(Asset):
        def build(self, c):
            scene = c.scene()
            scene.add("box", box())
            mask = np.zeros((24, 24), bool)
            mask[2:8, 2:8] = True
            c.expect("front", mask, names=["box"])
            return scene

    result = Mirrored().run(cache=tmp_path / "mirrored.asset", stages=("build", "lint"))
    assert [f.code for f in result.report.findings] == ["A-SEEN"]
    assert result.report.findings[0].view == "front"


def test_an_expectation_naming_a_missing_part_says_so(tmp_path):
    class Typo(Asset):
        def build(self, c):
            scene = c.scene()
            scene.add("box", box())
            c.expect("front", np.ones((8, 8), bool), names=["bxo"])
            return scene

    with pytest.raises(ValueError, match="bxo"):
        Typo().run(cache=tmp_path / "typo.asset", stages=("build", "lint"))


def test_an_asset_may_build_another_asset(tmp_path):
    class Child(Asset):
        def build(self, c):
            scene = c.scene()
            scene.add("peg", box(0.2))
            return scene

    class Parent(Asset):
        def build(self, c):
            scene = c.scene()
            scene.add("child", next(iter(c.asset(Child).meshes.values())))
            return scene

    result = Parent().run(cache=tmp_path / "parent.asset", stages=("build", "lint"))
    assert list(result.scene.meshes) == ["child"] and result.report.findings == []


def test_an_unknown_stage_is_refused(tmp_path):
    with pytest.raises(ValueError, match="render"):
        TwoBoxes().run(cache=tmp_path / "two-boxes.asset", stages=("build", "render"))
