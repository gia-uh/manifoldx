"""The Parts builders and their options, and the lint codes only Parts raises (A-ROWS, A-STRAY, A-KEEP, A-SHEET).

Each builder is called on synthetic masks first, then through a small Parts subclass whose blueprints are drawn
here, so a test needs no image model and no committed drawing.
"""

import numpy as np
import pytest

from manifoldx.asset import CODE_COLOURS, Parts, lathe, plate, two_view, tube
from manifoldx.asset.parts import Built, Frame
from manifoldx.image import write_png
from manifoldx.modeling.lint import mesh_report
from manifoldx.modeling.views import extrude

COLOUR = CODE_COLOURS


def disc(size, centre, radius):
    """A filled disc as a bool mask of ``size=(h, w)`` with pixel centres."""
    yy, xx = np.indices(size)
    return (xx + 0.5 - centre[0]) ** 2 + (yy + 0.5 - centre[1]) ** 2 <= radius * radius


def rect(size, rows, cols):
    mask = np.zeros(size, bool)
    mask[rows[0]:rows[1], cols[0]:cols[1]] = True
    return mask


def frame(rows=100, scale=0.01, top=1.0, x=50.0, z=50.0, n=100):
    return Frame(rows=rows, scale=scale, top=top, x=x, z=z, n=n)


def _volume(mesh):
    p = np.asarray(mesh.positions, float)
    f = np.asarray(mesh.faces, np.intp)
    v0, v1, v2 = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


# --- the builders, on synthetic masks ---------------------------------------------------------------------


def test_two_view_of_two_discs_is_a_roughly_spherical_mesh():
    front = disc((100, 100), (50, 50), 45)
    masks = {"front": front, "left": front.copy(), "back": front.copy()}
    built = two_view(power=2.0, rings=24, segments=16).build(masks, frame())
    assert isinstance(built, Built) and len(built.meshes) == 1
    mesh = built.meshes[0]
    p = np.asarray(mesh.positions, float)
    spans = p.max(axis=0) - p.min(axis=0)
    assert spans[0] == pytest.approx(spans[2], abs=0.05)          # as wide as it is deep
    assert spans[1] == pytest.approx(spans[0], abs=0.12)          # and about as tall: a ball, not a slab
    assert mesh_report(mesh)["open_edges"] == 0 and _volume(mesh) > 0


def test_two_view_power_makes_a_part_boxier():
    front = disc((100, 100), (50, 50), 45)
    masks = {"front": front, "left": front, "back": front}

    round_volume = _volume(two_view(power=2.0, rings=24).build(masks, frame()).meshes[0])
    boxy_volume = _volume(two_view(power=8.0, rings=24).build(masks, frame()).meshes[0])
    assert boxy_volume > round_volume * 1.15          # a superellipse of power 8 fills more of its rectangle


def test_two_view_fill_copy_takes_the_other_views_span_and_off_leaves_it():
    front = rect((100, 100), (10, 30), (40, 60))
    side = rect((100, 100), (10, 70), (30, 70))
    narrow = two_view().build({"front": front, "left": side}, frame())
    off = two_view(fill="off").build({"front": front, "left": side}, frame())
    assert narrow.filled > 0 and off.filled == 0
    assert np.asarray(narrow.meshes[0].positions)[:, 1].min() < np.asarray(off.meshes[0].positions)[:, 1].min()
    copied = two_view(fill="copy").build({"front": front, "left": side}, frame())
    assert copied.filled == narrow.filled


def test_lathe_of_a_rectangle_profile_is_a_closed_cylinder():
    front = rect((100, 100), (30, 70), (40, 60))
    built = lathe(rings=20, segments=16).build({"front": front}, frame())
    (mesh,) = built.meshes
    p = np.asarray(mesh.positions, float)
    assert np.hypot(p[:, 0], p[:, 2]).max() == pytest.approx(0.1, abs=1e-6)
    report = mesh_report(mesh)
    assert report["open_edges"] == 0 and report["volume"] > 0


def test_plate_extrudes_the_side_outline_along_x():
    side = rect((100, 100), (30, 70), (40, 60))
    front = rect((100, 100), (30, 70), (30, 70))
    (mesh,) = plate(thickness=0.05).build({"left": side, "front": front}, frame()).meshes
    p = np.asarray(mesh.positions, float)
    assert p[:, 0].max() - p[:, 0].min() == pytest.approx(0.05, abs=1e-6)
    assert p[:, 2].max() - p[:, 2].min() == pytest.approx(0.2, abs=1e-6)
    assert mesh_report(mesh)["open_edges"] == 0


def test_plate_lean_bends_it_toward_the_front_views_centre():
    side = rect((100, 100), (30, 70), (40, 60))
    front = np.zeros((100, 100), bool)
    for row in range(30, 70):
        shift = int((row - 30) / 4)
        front[row, 30 + shift:70 + shift] = True
    (flat,) = plate(thickness=0.05).build({"left": side, "front": front}, frame()).meshes
    (leaning,) = plate(thickness=0.05, lean="front").build({"left": side, "front": front}, frame()).meshes
    flat_x = np.asarray(flat.positions, float)[:, 0]
    lean_x = np.asarray(leaning.positions, float)[:, 0]
    assert np.ptp(lean_x) > np.ptp(flat_x) + 0.02
    assert mesh_report(leaning)["open_edges"] == 0


def test_tube_follows_a_rod_that_reaches_across_and_forward():
    front = rect((100, 100), (48, 53), (50, 90))
    side = rect((100, 100), (48, 53), (10, 50))
    (mesh,) = tube(samples=16).build({"front": front, "left": side}, frame(), anchor=(0.0, 0.5, 0.0)).meshes
    p = np.asarray(mesh.positions, float)
    assert np.abs(p[:, 0] - p[:, 2]).max() < 0.15


def test_tube_radius_mean_or_a_number_widens_or_narrows_it():
    front = rect((100, 100), (48, 53), (50, 90))
    side = rect((100, 100), (48, 58), (10, 50))

    def girth(builder):
        (mesh,) = builder.build({"front": front, "left": side}, frame(), anchor=(0, 0.5, 0)).meshes
        p = np.asarray(mesh.positions, float)
        return float(np.linalg.norm(p - p.mean(axis=0), axis=1).max())

    assert girth(tube(radius="min")) < girth(tube(radius="mean"))
    assert girth(tube(radius=0.005)) < girth(tube(radius="mean"))
    with pytest.raises(ValueError, match="radius"):
        tube(radius="wide")


def test_keep_drops_a_small_blob_largest_keeps_only_the_big_one_and_all_keeps_both():
    size = (100, 100)
    front = disc(size, (40, 50), 20) | rect(size, (80, 82), (90, 92))   # a ball and a speck of 4 px
    side = disc(size, (40, 50), 20)
    default = two_view(power=2.0, rings=12).build({"front": front, "left": side}, frame())
    assert len(default.meshes) == 1 and len(default.dropped) == 1
    assert default.dropped[0] == pytest.approx(4 / 10000)
    everything = two_view(keep="all", rings=12).build({"front": front, "left": side}, frame())
    assert len(everything.meshes) == 2 and everything.dropped == ()
    largest = two_view(keep="largest", rings=12).build({"front": front, "left": side}, frame())
    assert len(largest.meshes) == 1 and len(largest.dropped) == 1


def test_grid_pairs_two_front_instances_with_each_side_instance():
    front = np.zeros((100, 100), bool)
    for left in (10, 30, 60, 80):                 # the outer pair is further from the centre
        front[20:40, left:left + 8] = True
    side = np.zeros((100, 100), bool)
    side[20:40, 20:28] = True                     # the near leg, at the front of the side view
    side[20:40, 70:78] = True                     # the far leg
    built = two_view(grid=True, rings=8).build({"front": front, "left": side}, frame())
    assert len(built.meshes) == 4
    centres = sorted(float(np.mean(np.asarray(mesh.positions)[:, 2])) for mesh in built.meshes)
    assert len(set(np.round(centres, 3))) == 2     # two front legs, two back legs


def test_mirror_reflects_the_instance_the_side_view_shows_near_the_core():
    front = np.zeros((100, 100), bool)
    front[40:60, 70:90] = True                    # one arm only, to the right of the core (x > 0)
    side = np.zeros((100, 100), bool)
    side[40:60, 40:60] = True
    built = two_view(mirror=True, rings=8).build({"front": front, "left": side}, frame())
    assert len(built.meshes) == 2
    spans = sorted((float(np.asarray(mesh.positions)[:, 0].min()),
                    float(np.asarray(mesh.positions)[:, 0].max())) for mesh in built.meshes)
    assert spans[0][0] == pytest.approx(-0.40, abs=0.02) and spans[1][1] == pytest.approx(0.40, abs=0.02)


def test_a_builder_refuses_a_view_it_needs_and_a_bad_option():
    with pytest.raises(ValueError, match="front"):
        two_view(views=("left",))
    with pytest.raises(ValueError, match="fill"):
        two_view(fill="stretch")
    with pytest.raises(ValueError, match="keep"):
        two_view(keep="some")
    with pytest.raises(ValueError, match="power"):
        two_view(power=0)


# --- a Parts subclass whose drawings are made here --------------------------------------------------------


def paint(size, **masks):
    """A white RGB picture with each mask painted in a code colour."""
    image = np.full((size[0], size[1], 3), 255, np.uint8)
    for colour, mask in masks.items():
        image[np.asarray(mask, bool)] = COLOUR[colour]
    return image


def write_blueprints(cache, **views):
    folder = cache / "images"
    folder.mkdir(parents=True, exist_ok=True)
    for view, image in views.items():
        write_png(folder / f"bp-{view}.png", image)


def test_keep_drops_a_small_blob_and_a_stray_reports_its_size(tmp_path):
    size = (160, 160)
    body = disc(size, (80, 80), 40)
    speck = rect(size, (130, 135), (150, 155))     # five pixels wide, so open_mask(2) leaves it
    stamp = paint(size, orange=body | speck)

    class One(Parts):
        desc, height, core = "a ball", 1.0, "body"

        def parts(self, p):
            p.add("body", "orange", "the ball",
                  build=two_view(views=("front", "left"), rings=12, keep=0.01))

    cache = tmp_path / "one.asset"
    write_blueprints(cache, front=stamp, left=stamp)
    result = One().run(cache=cache, stages=("build", "lint"))
    assert [f.code for f in result.report.findings if f.code == "A-STRAY"] == ["A-STRAY"]
    (stray,) = [f for f in result.report.findings if f.code == "A-STRAY"]
    assert stray.level == "info" and stray.part == "body" and stray.value == 1
    assert "%" in stray.message and "keep" in stray.hint


def test_keep_all_keeps_the_speck_and_the_lint_is_silent_about_it(tmp_path):
    size = (160, 160)
    body = disc(size, (80, 80), 40)
    speck = rect(size, (130, 134), (150, 154))
    stamp = paint(size, orange=body | speck)

    class One(Parts):
        desc, height, core = "a ball", 1.0, "body"

        def parts(self, p):
            p.add("body", "orange", "the ball", build=two_view(keep="all", views=("front", "left"), rings=12))

    cache = tmp_path / "one.asset"
    write_blueprints(cache, front=stamp, left=stamp)
    result = One().run(cache=cache, stages=("build", "lint"))
    assert [f.code for f in result.report.findings if f.code == "A-STRAY"] == []


def test_a_rows_warns_when_one_view_draws_a_part_over_more_rows_than_the_other(tmp_path):
    size = (200, 200)
    core = rect(size, (10, 190), (40, 80))
    arm_front = rect(size, (10, 120), (100, 140))  # the arm sits high in the front view...
    arm_side = rect(size, (100, 190), (100, 140))  # ...and low in the side one: the two disagree by rows
    front = paint(size, orange=core, blue=arm_front)
    left = paint(size, orange=core, blue=arm_side)

    class Row(Parts):
        desc, height, core = "a post with an arm", 1.0, "core"

        def parts(self, p):
            p.add("core", "orange", "the post", build=two_view(views=("front", "left"), rings=12))
            p.add("arm", "blue", "the arm", build=two_view(views=("front", "left"), rings=12))

    cache = tmp_path / "row.asset"
    write_blueprints(cache, front=front, left=left)
    result = Row().run(cache=cache, stages=("build", "lint"))
    (finding,) = [f for f in result.report.findings if f.code == "A-ROWS"]
    assert finding.level == "warn" and finding.part == "arm" and finding.value > 96
    assert "filled from the other view" in finding.message and "fill" in finding.hint


def test_a_keep_warns_when_a_view_drawing_is_cached_and_lost_its_outline(tmp_path):
    size = (160, 160)
    stamp = paint(size, orange=disc(size, (80, 80), 40))

    class One(Parts):
        desc, height, core = "a ball", 1.0, "body"

        def parts(self, p):
            p.add("body", "orange", "the ball", build=two_view(views=("front", "left"), rings=12))

    cache = tmp_path / "one.asset"
    write_blueprints(cache, front=stamp, left=stamp)
    folder = cache / "images"
    write_png(folder / "view-front.png", paint(size, orange=rect(size, (35, 125), (35, 125))))
    result = One().run(cache=cache, stages=("build", "lint"))
    (finding,) = [f for f in result.report.findings if f.code == "A-KEEP"]
    assert finding.level == "warn" and finding.view == "front" and finding.value < 0.85
    assert "outline" in finding.message


def test_a_keep_is_silent_when_no_view_drawing_is_cached(tmp_path):
    size = (160, 160)
    stamp = paint(size, orange=disc(size, (80, 80), 40))

    class One(Parts):
        desc, height, core = "a ball", 1.0, "body"

        def parts(self, p):
            p.add("body", "orange", "the ball", build=two_view(views=("front", "left"), rings=12))

    cache = tmp_path / "one.asset"
    write_blueprints(cache, front=stamp, left=stamp)
    result = One().run(cache=cache, stages=("build", "lint"))
    assert [f.code for f in result.report.findings if f.code == "A-KEEP"] == []


def test_a_sheet_error_when_a_drawing_holds_two_figures_of_comparable_size(tmp_path):
    size = (200, 200)
    one = paint(size, orange=disc(size, (100, 100), 80))
    two = paint(size, orange=disc(size, (50, 100), 40) | disc(size, (150, 100), 40))

    class One(Parts):
        desc, height, core = "a ball", 1.0, "body"

        def parts(self, p):
            p.add("body", "orange", "the ball", build=two_view(views=("front", "left"), rings=12))

    cache = tmp_path / "one.asset"
    write_blueprints(cache, front=one, left=one)
    folder = cache / "images"
    write_png(folder / "view-front.png", two)
    result = One().run(cache=cache, stages=("build", "lint"))
    (finding,) = [f for f in result.report.findings if f.code == "A-SHEET"]
    assert finding.level == "error" and finding.view == "front" and finding.value == 2
    assert "comparable size" in finding.message


def test_a_sheet_is_silent_on_one_figure_and_on_disconnected_small_parts(tmp_path):
    size = (200, 200)
    body = disc(size, (80, 100), 45)
    legs = rect(size, (140, 160), (40, 55)) | rect(size, (140, 160), (105, 120))
    stamp = paint(size, orange=body, blue=legs)

    class Chair(Parts):
        desc, height, core = "a chair", 1.0, "body"

        def parts(self, p):
            p.add("body", "orange", "the seat", build=two_view(views=("front", "left"), rings=12))
            p.add("legs", "blue", "the legs", build=two_view(views=("front", "left"), rings=12, segments=8))

    cache = tmp_path / "chair.asset"
    write_blueprints(cache, front=stamp, left=stamp)
    result = Chair().run(cache=cache, stages=("build", "lint"))
    assert [f.code for f in result.report.findings if f.code == "A-SHEET"] == []


def test_attach_moves_a_part_the_shortest_way_until_it_touches(tmp_path):
    size = (160, 160)
    below = rect(size, (90, 120), (50, 110))
    above = rect(size, (60, 80), (50, 110))        # a 10 px gap between them
    stamp = paint(size, orange=below, blue=above)

    class Stack(Parts):
        desc, height, core = "two blocks", 1.0, "below"

        def parts(self, p):
            p.add("below", "orange", "the lower block", build=two_view(views=("front", "left"), rings=12))
            p.add("above", "blue", "the upper block",
                  build=two_view(views=("front", "left"), rings=12, attach="below"))

    cache = tmp_path / "stack.asset"
    write_blueprints(cache, front=stamp, left=stamp)
    result = Stack().run(cache=cache, stages=("build", "lint"))
    assert [f.code for f in result.report.findings if f.code == "A-FLOAT"] == []
    (moved,) = [f for f in result.report.findings if f.code == "A-ATTACH"]
    assert moved.level == "info" and moved.part == "above" and moved.value > 0
    assert "touch below" in moved.message


def test_parts_refuses_a_duplicate_colour_and_a_bad_core(tmp_path):
    size = (80, 80)
    stamp = paint(size, orange=rect(size, (10, 70), (20, 60)))

    class Clash(Parts):
        desc, height = "two parts", 1.0

        def parts(self, p):
            p.add("a", "orange", "one")
            p.add("b", "orange", "the other")

    class WrongCore(Parts):
        desc, height, core = "a part", 1.0, "nope"

        def parts(self, p):
            p.add("a", "orange", "one")

    cache = tmp_path / "clash.asset"
    write_blueprints(cache, front=stamp, left=stamp)
    with pytest.raises(ValueError, match="code colour"):
        Clash().run(cache=cache, stages=("build",))
    with pytest.raises(ValueError, match="nope"):
        WrongCore().run(cache=cache, stages=("build",))


def test_a_plate_keeps_its_own_name_while_instances_are_numbered():
    from manifoldx.asset.parts import _Added, _instance_names

    assert _instance_names(_Added("tail", "cyan", "a tail", plate()), (None,)) == ["tail"]
    assert _instance_names(_Added("arms", "blue", "an arm", two_view()), (None, None)) == ["arms1", "arms2"]


def test_draw_asks_for_a_reference_then_each_view_and_its_blueprint(tmp_path):
    class FakeImage:
        name = "fake/parts"
        cost = 0.0

        def __init__(self):
            self.calls = []

        def generate(self, prompt, images=(), aspect="1:1", size="1K"):
            self.calls.append({"prompt": prompt, "images": list(images), "aspect": aspect})
            return np.full((16, 16, 3), 255, np.uint8)

    class Bucket(Parts):
        desc, height, core = "a red bucket", 1.0, "body"
        notes = {"front": "the handle points at the viewer"}

        def parts(self, p):
            p.add("body", "orange", "the round body")
            p.add("rim", "yellow", "the rim on top")

    model = FakeImage()
    Bucket().run(cache=tmp_path / "bucket.asset", stages=("draw",), model=model)
    assert model.calls[0]["prompt"].startswith("Reference for a 3D model: a red bucket, the whole object")
    assert len(model.calls) == 7                       # reference, then a view and a blueprint for three views
    assert len(model.calls[0]["images"]) == 0
    assert len(model.calls[1]["images"]) == 1 and len(model.calls[2]["images"]) == 1
    blueprint = model.calls[2]["prompt"]
    assert "- orange (255, 140, 0) = the round body;" in blueprint
    assert "- yellow (255, 220, 0) = the rim on top;" in blueprint
    assert "the handle points at the viewer" in blueprint
    assert "BLUEPRINT" in blueprint and "pure white" in blueprint
