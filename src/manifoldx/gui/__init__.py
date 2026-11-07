"""ManifoldX in-engine GUI layer.

Public API:
- `Panel`, `Text`, `ValueDisplay`, `Image` — non-interactive widgets.
- `Button`, `Slider`, `Toggle`, `Joystick` — interactive widgets.
- `style` — theme + named classes + per-widget overrides.
"""

from manifoldx.gui import style  # noqa: F401
from manifoldx.gui.button import Button  # noqa: F401
from manifoldx.gui.joystick import Joystick  # noqa: F401
from manifoldx.gui.slider import Slider  # noqa: F401
from manifoldx.gui.toggle import Toggle  # noqa: F401
from manifoldx.gui.value_display import ValueDisplay  # noqa: F401
from manifoldx.gui.widgets import Image, Panel, Text  # noqa: F401

__all__ = [
    "Button",
    "Image",
    "Joystick",
    "Panel",
    "Slider",
    "Text",
    "Toggle",
    "ValueDisplay",
    "style",
]
