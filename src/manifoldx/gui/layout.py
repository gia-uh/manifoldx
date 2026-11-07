"""Stack + flex layout algorithm for the GUI layer.

Operates on plain spec dicts so it stays decoupled from widget classes.
A spec is::

    {
        "direction": "v" | "h",
        "padding":   (top, right, bottom, left),
        "gap":       int,
        "width":     int | None,        # explicit size; overrides intrinsic
        "height":    int | None,
        "flex":      int | None,        # share of free space (containers only)
        "intrinsic": (w, h),            # default size if no explicit/flex
        "children":  [spec, ...],
    }

`compute_layout(root, viewport)` returns ``{id(spec): LayoutBox}`` for every
spec in the tree, keyed by Python object id (callers retain the spec
references themselves).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LayoutBox:
    x: float
    y: float
    w: float
    h: float


def compute_layout(
    root: dict[str, Any],
    viewport: LayoutBox,
) -> dict[int, LayoutBox]:
    """Compute boxes for every node in the tree rooted at `root`."""
    out: dict[int, LayoutBox] = {}
    _layout_node(root, viewport, out)
    return out


def _layout_node(
    node: dict[str, Any],
    box: LayoutBox,
    out: dict[int, LayoutBox],
) -> None:
    out[id(node)] = box
    children = node.get("children") or []
    if not children:
        return

    top, right, bottom, left = node["padding"]
    inner = LayoutBox(
        box.x + left,
        box.y + top,
        max(0.0, box.w - left - right),
        max(0.0, box.h - top - bottom),
    )
    direction = node["direction"]
    gap = node["gap"]
    main_axis = "h" if direction == "v" else "w"
    main_size = getattr(inner, main_axis)

    total_gap = gap * max(0, len(children) - 1)
    free = main_size - total_gap

    # First pass: subtract fixed + intrinsic-without-flex from `free`.
    main_sizes: list[float] = []
    total_flex = 0
    for child in children:
        flex = child.get("flex")
        explicit = child.get("height") if direction == "v" else child.get("width")
        if flex is not None:
            total_flex += flex
            main_sizes.append(0.0)  # filled in second pass.
            continue
        if explicit is not None:
            size = float(explicit)
        else:
            iw, ih = child["intrinsic"]
            size = float(ih if direction == "v" else iw)
        main_sizes.append(size)
        free -= size

    # Second pass: distribute remaining `free` among flex children.
    if total_flex > 0 and free > 0:
        for i, child in enumerate(children):
            flex = child.get("flex")
            if flex is None:
                continue
            main_sizes[i] = free * (flex / total_flex)

    # Place children sequentially along the main axis.
    cursor = 0.0
    for child, size in zip(children, main_sizes):
        if direction == "v":
            child_box = LayoutBox(inner.x, inner.y + cursor, inner.w, size)
        else:
            child_box = LayoutBox(inner.x + cursor, inner.y, size, inner.h)
        _layout_node(child, child_box, out)
        cursor += size + gap


# Where an anchor puts a root panel, per axis: "start", "end" or "center" (the page's AXES
# in web/js/gui.js).
AXES = {
    "top-left": ("start", "start"),
    "top-center": ("center", "start"),
    "top-right": ("end", "start"),
    "center-left": ("start", "center"),
    "center": ("center", "center"),
    "center-right": ("end", "center"),
    "bottom-left": ("start", "end"),
    "bottom-center": ("center", "end"),
    "bottom-right": ("end", "end"),
}


def anchored_slot(
    anchor: str,
    offset: tuple[float, float],
    root: dict[str, Any],
    viewport: LayoutBox,
) -> LayoutBox:
    """The slot of a root panel's spec: placed by its anchor, the offset measured inward
    from the anchored edges and, on a centred axis, rightward or downward, as the page's
    panelCss places it. An explicit width or height sizes the slot. Without one, a top-left
    panel fills the viewport, as it always has, and any other fits its content, as CSS sizes
    an absolute box, so the panel touches its edges."""
    content = content_size(root)
    fill = anchor == "top-left"
    slot = []
    for i, (place, start, extent, off) in enumerate(
        zip(AXES[anchor], (viewport.x, viewport.y), (viewport.w, viewport.h), offset)
    ):
        explicit = root.get("width" if i == 0 else "height")
        if explicit is not None:
            size = float(explicit)
        else:
            size = extent if fill else content[i]
        if place == "start":
            pos = start + off
        elif place == "end":
            pos = start + extent - size - off
        else:
            pos = start + (extent - size) / 2 + off
        slot.append((pos, size))
    (x, w), (y, h) = slot
    return LayoutBox(x, y, w, h)


def content_size(node: dict[str, Any]) -> tuple[float, float]:
    """The (w, h) a spec needs for its content: explicit sizes, leaves' intrinsic sizes, and
    a container's children stacked with its gap and padding. Flex children count at their
    own content size."""
    children = node.get("children") or []
    if children:
        top, right, bottom, left = node["padding"]
        sizes = [content_size(c) for c in children]
        gaps = node["gap"] * (len(children) - 1)
        if node["direction"] == "v":
            w = max(s[0] for s in sizes) + left + right
            h = sum(s[1] for s in sizes) + gaps + top + bottom
        else:
            w = sum(s[0] for s in sizes) + gaps + left + right
            h = max(s[1] for s in sizes) + top + bottom
    else:
        w, h = node["intrinsic"]
    return (
        float(node["width"]) if node.get("width") is not None else float(w),
        float(node["height"]) if node.get("height") is not None else float(h),
    )
