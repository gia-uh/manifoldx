"""Painting a Parts scene with a fake image model: the views, the outline gate, A-CODE/A-OUTLINE, and saving.

No network: the fake model answers the template the paint call sends, either recoloured (a good answer) or
shrunk (an answer that moves the object), and the cache makes a second run free.
"""

import numpy as np

from manifoldx.asset import CODE_COLOURS, Parts, save
from manifoldx.asset.parts import two_view
from manifoldx.image import read_png, write_png

SIZE = (180, 180)


def disc(centre, radius):
    yy, xx = np.indices(SIZE)
    return (xx + 0.5 - centre[0]) ** 2 + (yy + 0.5 - centre[1]) ** 2 <= radius * radius


def rect(rows, cols):
    mask = np.zeros(SIZE, bool)
    mask[rows[0]:rows[1], cols[0]:cols[1]] = True
    return mask


def stamp(**masks):
    image = np.full(SIZE + (3,), 255, np.uint8)
    for colour, mask in masks.items():
        image[np.asarray(mask, bool)] = CODE_COLOURS[colour]
    return image


class Toy(Parts):
    desc, height, core = "a toy with a round body and a cap", 1.0, "body"
    atlas_size = 512

    def parts(self, p):
        p.add("body", "orange", "the round body", build=two_view(views=("front", "left"), rings=16))
        p.add("cap", "blue", "the cap on top", build=two_view(views=("front", "left"), rings=12))


def cache_with_blueprints(tmp_path):
    body, cap = disc((90, 100), 55), rect((10, 35), (65, 115))
    front = stamp(orange=body, blue=cap)
    left = stamp(orange=disc((90, 100), 55), blue=rect((10, 35), (65, 115)))
    folder = tmp_path / "toy.asset" / "images"
    folder.mkdir(parents=True)
    for view, picture in (("front", front), ("left", left), ("back", front)):
        write_png(folder / f"bp-{view}.png", picture)
    return tmp_path / "toy.asset"


class Recolour:
    """The template with every coloured pixel repainted in one colour no code colour is near."""

    name, cost = "fake-recolour", 0.0

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, images=(), aspect="1:1", size="1K"):
        self.calls += 1
        picture = np.array(images[0])
        picture[(picture < 250).any(axis=2)] = (100, 0, 100)
        return picture


class Shrunk:
    """An answer that keeps only a small patch in the middle: its outline no longer matches the template."""

    name, cost = "fake-shrunk", 0.0

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, images=(), aspect="1:1", size="1K"):
        self.calls += 1
        picture = np.full_like(np.asarray(images[0]), 255)
        height, width = picture.shape[:2]
        picture[height // 2 - 8:height // 2 + 8, width // 2 - 8:width // 2 + 8] = (100, 0, 100)
        return picture


def test_paint_makes_one_call_per_view_and_returns_the_maps(tmp_path):
    cache = cache_with_blueprints(tmp_path)
    model = Recolour()
    result = Toy().run(cache=cache, stages=("build", "paint"), model=model)

    assert model.calls == 3
    assert result.textures["albedo"].shape == (512, 512, 3)
    assert result.textures["normal"].shape == (512, 512, 3)
    assert set(result.textures["uvs"]) == set(result.scene.meshes)
    assert set(result.textures["meshes"]) == set(result.scene.meshes)
    assert [finding.code for finding in result.report] == []
    assert (cache / "images" / "albedo-front.png").exists()


def test_paint_asks_three_times_then_reports_a_kept_outline(tmp_path):
    cache = cache_with_blueprints(tmp_path)
    model = Shrunk()
    result = Toy().run(cache=cache, stages=("build", "paint"), model=model)

    assert model.calls == 9                                  # three views, three attempts each
    outlines = [finding for finding in result.report if finding.code == "A-OUTLINE"]
    assert [finding.view for finding in outlines] == ["front", "back", "left"]
    assert all(finding.level == "warn" and 0.0 <= finding.value < 0.85 for finding in outlines)
    for view in ("front", "back", "left"):
        assert (cache / "images" / f"albedo-{view}-3.png").exists()
        assert not (cache / "images" / f"albedo-{view}-4.png").exists()


def test_a_second_paint_run_asks_the_model_nothing(tmp_path):
    cache = cache_with_blueprints(tmp_path)
    first = Toy().run(cache=cache, stages=("build", "paint"), model=Shrunk())
    model = Shrunk()
    second = Toy().run(cache=cache, stages=("build", "paint"), model=model)

    assert model.calls == 0
    assert [finding.code for finding in second.report] == [finding.code for finding in first.report]
    assert np.array_equal(first.textures["albedo"], second.textures["albedo"])


def test_save_writes_a_textured_glb_the_textures_and_the_report(tmp_path):
    cache = cache_with_blueprints(tmp_path)
    result = Toy().run(cache=cache, stages=("build", "paint"), model=Recolour())
    written = save(result, tmp_path / "out")

    names = {path.name for path in written}
    assert names == {"model.glb", "albedo.png", "normal.png", "report.json"}
    assert (tmp_path / "out" / "model.glb").read_bytes()[:4] == b"glTF"
    assert read_png(tmp_path / "out" / "albedo.png").shape == (512, 512, 3)
    assert (tmp_path / "out" / "report.json").read_text().strip() == "[]"
