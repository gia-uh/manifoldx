"""The web tests run generated JavaScript in Node; without Node they skip."""

import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    if shutil.which("node") is not None:
        return
    skip = pytest.mark.skip(reason="node not found")
    for item in items:
        if item.path.is_relative_to(HERE):
            item.add_marker(skip)
