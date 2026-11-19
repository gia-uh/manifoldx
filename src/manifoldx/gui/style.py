"""Style resolution for the GUI layer.

Three layers, merged at paint time in this order (later wins):

    theme  →  named class  →  per-widget overrides

The style vocabulary is fixed (see DEFAULT_THEME keys). Unknown keys are
ignored at resolve time but preserved if you set them in a class — they
just never affect rendering.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from manifoldx import _deferred

DEFAULT_THEME: dict[str, Any] = {
    "bg": "#222",
    "fg": "#ddd",
    "font_size": 14,
    "padding": 4,
    "gap": 4,
    "border": 0,
    "border_color": "#444",
    "radius": 0,
    "width": None,
    "height": None,
    "flex": None,
    "direction": "v",
}

_theme: dict[str, Any] = deepcopy(DEFAULT_THEME)
_classes: dict[str, dict[str, Any]] = {}
version = 0  # grows at every change of the theme or a class, so a cached layout knows it is stale


def reset() -> None:
    """Reset theme to defaults and clear named classes. For tests."""
    global _theme, _classes, version
    _theme = deepcopy(DEFAULT_THEME)
    _classes = {}
    version += 1


@_deferred.deferred_in_pipeline
def set_theme(theme: dict[str, Any]) -> None:
    """Merge `theme` over the current theme. Keys absent from `theme`
    keep their previous values."""
    global version
    _theme.update(theme)
    version += 1


@_deferred.deferred_in_pipeline
def define(class_name: str, props: dict[str, Any]) -> None:
    """Register or replace a named style class."""
    global version
    _classes[class_name] = dict(props)
    version += 1


def resolve(
    class_name: str | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return effective style: theme < class < overrides."""
    out = dict(_theme)
    if class_name is not None:
        if class_name not in _classes:
            raise KeyError(f"unknown gui style class: {class_name!r}")
        out.update(_classes[class_name])
    if overrides is not None:
        out.update(overrides)
    return out


def parse_color(hex_str: str) -> tuple[float, float, float, float]:
    """Parse `#rgb`, `#rrggbb`, or `#rrggbbaa` to (r, g, b, a) floats in [0,1].

    Linear sRGB pass-through — the renderer handles gamma if needed.
    """
    if not isinstance(hex_str, str) or not hex_str.startswith("#"):
        raise ValueError(f"not a hex color: {hex_str!r}")
    body = hex_str[1:]
    if len(body) == 3:
        # Expand each digit: #abc -> #aabbcc.
        body = "".join(c + c for c in body) + "ff"
    elif len(body) == 6:
        body = body + "ff"
    elif len(body) == 8:
        pass
    else:
        raise ValueError(f"invalid hex length: {hex_str!r}")
    try:
        r = int(body[0:2], 16) / 255.0
        g = int(body[2:4], 16) / 255.0
        b = int(body[4:6], 16) / 255.0
        a = int(body[6:8], 16) / 255.0
    except ValueError as e:
        raise ValueError(f"invalid hex digits: {hex_str!r}") from e
    return (r, g, b, a)


def parse_padding(value: int | str) -> tuple[int, int, int, int]:
    """Return (top, right, bottom, left) in pixels.

    Accepts either an int (all four equal) or a string of 1, 2, or 4
    space-separated ints. CSS-style 2-value shorthand: "v h" -> top=bottom=v,
    right=left=h. 3-value is rejected (rare in style sheets, error-prone).
    """
    if isinstance(value, int):
        return (value, value, value, value)
    if not isinstance(value, str):
        raise ValueError(f"padding must be int or str, got {type(value).__name__}")
    parts = value.split()
    nums = [int(p) for p in parts]
    if len(nums) == 1:
        v = nums[0]
        return (v, v, v, v)
    if len(nums) == 2:
        v, h = nums
        return (v, h, v, h)
    if len(nums) == 4:
        t, r, b, left = nums
        return (t, r, b, left)
    raise ValueError(f"padding string must have 1, 2, or 4 values; got {len(nums)}")
