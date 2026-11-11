"""Inspect writes the overlays, the groups picture and the summary for the hydrant example, with no network."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from manifoldx.asset import inspect
from manifoldx.image import read_png

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "models"


def _load(name):
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_inspect_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, name.capitalize())


def test_inspect_writes_an_overlay_per_view_the_groups_and_a_summary(tmp_path):
    asset = _load("hydrant")()
    result = asset.run(stages=("build", "lint"))
    files = inspect(asset, result.context, result.scene, tmp_path, report=result.report)
    names = [path.name for path in files]

    assert names == ["overlay-front.png", "overlay-back.png", "overlay-left.png", "groups.png", "summary.txt"]
    for name in names[:-1]:
        image = read_png(tmp_path / name)
        assert image.shape[2] == 3 and image.dtype == np.uint8
        assert (image < 250).any()                      # something was drawn, not a white page
    summary = (tmp_path / "summary.txt").read_text().splitlines()
    assert summary[0] == "6 parts, 5760 triangles"
    assert summary[1].startswith("1 group(s) of touching parts:")
    assert "IoU barrel front: 0.736" in summary
    assert any(line.startswith("A-SEEN ") for line in summary)
