"""The paid step, cached: a second draw sends nothing, a changed prompt re-requests only its image."""

import json

import numpy as np
import pytest

from manifoldx.asset import Asset, Context
from manifoldx.imagegen import BudgetExceeded

RED = (220, 40, 40)


class FakeImage:
    """An image model that counts its calls and draws a red square on white."""

    name = "fake/solid"
    cost = 0.25

    def __init__(self, cost=0.25):
        self.cost = cost
        self.calls = []

    def generate(self, prompt, images=(), aspect="1:1", size="1K"):
        self.calls.append({"prompt": prompt, "images": list(images), "aspect": aspect})
        image = np.full((20, 20, 3), 255, np.uint8)
        image[5:15, 5:15] = RED
        return image


def test_a_second_draw_sends_nothing(tmp_path):
    model = FakeImage()
    c = Context(tmp_path / "thing.asset", model=model)
    first = c.image("plan", "a plan with walls in black")
    assert len(model.calls) == 1 and first.shape == (20, 20, 3)
    again = Context(tmp_path / "thing.asset", model=model).image("plan", "a plan with walls in black")
    assert len(model.calls) == 1
    assert np.array_equal(first, again)


def test_the_sidecar_records_what_shaped_the_answer(tmp_path):
    model = FakeImage()
    c = Context(tmp_path / "thing.asset", model=model)
    c.image("plan", "a plan", refs=(), aspect="4:3")
    meta = json.loads((tmp_path / "thing.asset/images/plan.json").read_text())
    assert meta["prompt"] == "a plan" and meta["model"] == "fake/solid" and meta["aspect"] == "4:3"
    assert meta["refs"] == [] and len(meta["key"]) == 64


def test_changing_one_prompt_re_requests_only_that_image(tmp_path):
    class Drawing(Asset):
        prompts = {"plan": "a plan", "door": "a door"}

        def draw(self, c):
            for name, prompt in self.prompts.items():
                c.image(name, prompt)

    model = FakeImage()
    Drawing().run(cache=tmp_path / "drawing.asset", stages=("draw",), model=model)
    assert [call["prompt"] for call in model.calls] == ["a plan", "a door"]

    class Changed(Drawing):
        prompts = {"plan": "a plan", "door": "a taller door"}

    Changed().run(cache=tmp_path / "drawing.asset", stages=("draw",), model=model)
    assert [call["prompt"] for call in model.calls] == ["a plan", "a door", "a taller door"]


def test_a_reference_is_part_of_the_key(tmp_path):
    model = FakeImage()
    c = Context(tmp_path / "thing.asset", model=model)
    c.image("plan", "a plan")
    c.image("facade", "the facade of that plan", refs=["plan"])
    assert len(model.calls) == 2 and len(model.calls[1]["images"]) == 1
    Context(tmp_path / "thing.asset", model=model).image("facade", "the facade of that plan", refs=["plan"])
    assert len(model.calls) == 2


def test_load_never_asks_the_model(tmp_path):
    model = FakeImage()
    c = Context(tmp_path / "thing.asset", model=model)
    drawn = c.image("plan", "a plan")
    assert len(model.calls) == 1
    assert np.array_equal(c.load("plan"), drawn)
    assert len(model.calls) == 1
    with pytest.raises(FileNotFoundError, match="plan"):
        c.load("plan-does-not-exist")


def test_no_model_is_an_error_naming_the_image(tmp_path):
    c = Context(tmp_path / "thing.asset")
    with pytest.raises(RuntimeError, match="door"):
        c.image("door", "a door")


def test_a_cached_image_is_served_with_no_model_configured(tmp_path):
    drawn = Context(tmp_path / "thing.asset", model=FakeImage()).image("plan", "a plan")
    assert np.array_equal(Context(tmp_path / "thing.asset").image("plan", "a plan"), drawn)


def test_the_budget_caps_dollars_when_the_model_reports_a_cost(tmp_path):
    model = FakeImage(cost=0.6)
    c = Context(tmp_path / "thing.asset", model=model, budget=1.0)
    c.image("a", "one")
    c.image("b", "two")
    assert c.spent == pytest.approx(1.2)
    with pytest.raises(BudgetExceeded, match="budget"):
        c.image("c", "three")
    assert len(model.calls) == 2


def test_the_budget_caps_calls_when_the_model_reports_no_cost(tmp_path):
    model = FakeImage(cost=None)
    c = Context(tmp_path / "thing.asset", model=model, budget=2)
    c.image("a", "one")
    c.image("b", "two")
    with pytest.raises(BudgetExceeded):
        c.image("c", "three")
    assert c.calls == 2


def test_a_named_code_colour_is_one_mask(tmp_path):
    c = Context(tmp_path / "thing.asset", model=FakeImage())
    c.image("plan", "a plan")
    red = c.mask("plan", "red")
    assert red.dtype == bool and red.sum() == 100 and red[7, 7] and not red[0, 0]
    assert np.array_equal(red, c.mask("plan", RED))
    assert c.mask("plan", "white").sum() == 300
    with pytest.raises(ValueError, match="pink"):
        c.mask("plan", "pink")


def test_metres_scales_pixels_by_the_asset_height(tmp_path):
    c = Context(tmp_path / "thing.asset", height=2.4)
    assert c.metres(480) == pytest.approx(2.4)
    assert c.metres([0, 480]).tolist() == pytest.approx([0.0, 2.4])
    assert c.metres(10, scale=0.5) == pytest.approx(5.0)
