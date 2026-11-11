"""What a built model looks like from each view, for an agent that cannot see it.

:func:`inspect` writes three kinds of file next to the asset:

- ``overlay-<view>.png``: the scene's silhouette over the blueprint, green where both agree, grey where the
  drawing shows something the build did not make, red where the build made something the drawing does not show;
- ``groups.png``: the front, left and top silhouettes, each group of touching parts in its own colour;
- ``summary.txt``: one fact per line — parts and triangles, the groups, each part's IoU per view, where each
  drawn part spans in metres (to place a skeleton's joints), the asset's marks, every finding.

An asset's :meth:`~manifoldx.asset.Asset.marks` (a skeleton's bones) are drawn on every overlay in dark blue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manifoldx.asset.base import _frame, standard_findings
from manifoldx.image import write_png
from manifoldx.modeling.lint import part_groups
from manifoldx.modeling.trace import iou
from manifoldx.modeling.views import Atlas, Island, _VIEWS, silhouette, template

_DRAWN, _BUILT, _BOTH = (180, 180, 180), (220, 60, 60), (60, 140, 60)
_TILE = 400
_GAP = 10
_MARK = (20, 30, 120)
_HUES = [(60, 150, 60), (220, 50, 50), (40, 90, 220), (230, 150, 0), (150, 50, 200)]


def inspect(asset, c, scene, out_dir, report=None) -> list[Path]:
    """Write the overlays, ``groups.png`` and ``summary.txt`` for ``scene``; returns the files written.

    ``asset`` supplies the height and the triangle limit, ``c`` the expectations each drawing left behind, and
    ``scene`` the built meshes. ``report`` is the run's findings; without one, the standard lint and the asset's
    own checks are run here, so a Python caller gets the same summary as the command.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for expectation in c.expectations:
        files.append(_overlay(scene, expectation, out / f"overlay-{expectation.view}.png", asset.marks()))
    tol = max(0.01 * float(asset.height), 1e-6)
    groups = part_groups(scene.meshes, tol=tol)["groups"]
    files.append(_groups(scene, groups, out / "groups.png"))
    if report is None:
        findings = standard_findings(c, scene, tol=tol, max_tris=asset.max_tris)
        findings.extend(asset.check(c, scene))
        findings.extend(c.findings)
        finding_lines = [_line(finding) for finding in findings]
    else:
        finding_lines = report.lines()
    summary = out / "summary.txt"
    lines = _summary(c, scene, groups, finding_lines, asset.marks())
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.append(summary)
    return files


def _overlay(scene, expectation, path: Path, marks=()) -> Path:
    """One view's silhouette over its blueprint: green both, grey only drawn, red only built, marks in blue."""
    shape = next(iter(expectation.masks.values())).shape
    frame = expectation.frame if expectation.frame is not None else _frame(scene.meshes, expectation.view)
    drawn = np.zeros(shape, bool)
    for mask in expectation.masks.values():
        drawn |= np.asarray(mask, bool)
    built = np.zeros(shape, bool)
    for mesh in scene.meshes.values():
        built |= silhouette(mesh, expectation.view, (shape[1], shape[0]), frame)
    if expectation.view == "back":                  # the drawing is not mirrored; the projection is
        built = built[:, ::-1]
    image = np.full(shape + (3,), 255, np.uint8)
    image[drawn & ~built] = _DRAWN
    image[drawn & built] = _BOTH
    image[~drawn & built] = _BUILT
    for _, start, end in marks:
        _line_on(image, _pixel(expectation.view, frame, shape, start), _pixel(expectation.view, frame, shape, end))
    write_png(path, image)
    return path


def _pixel(view, frame, shape, point):
    """Where a point in metres falls on a view's drawing, as (column, row)."""
    _, right, up = _VIEWS[view]
    u, v = np.asarray(point, float) @ np.asarray(right), np.asarray(point, float) @ np.asarray(up)
    u0, u1, v0, v1 = frame
    column = (u - u0) / (u1 - u0) * shape[1]
    if view == "back":                              # the overlay shows the back drawing as drawn
        column = shape[1] - column
    return column, (v1 - v) / (v1 - v0) * shape[0]


def _line_on(image, a, b):
    """A three-pixel dark blue line from ``a`` to ``b`` with a dot at each end."""
    steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1))
    for t in np.linspace(0.0, 1.0, steps + 1):
        x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        image[max(0, int(y) - 1):int(y) + 2, max(0, int(x) - 1):int(x) + 2] = _MARK
    for x, y in (a, b):
        image[max(0, int(y) - 4):int(y) + 5, max(0, int(x) - 4):int(x) + 5] = _MARK


def _groups(scene, groups, path: Path) -> Path:
    """Front, left and top silhouettes, each group of touching parts in its own colour."""
    tiles = []
    positions = [np.asarray(mesh.positions, float) for mesh in scene.meshes.values()]
    points = np.vstack(positions) if positions else np.zeros((1, 3))
    for view in ("front", "left", "top"):
        _, right, up = _VIEWS[view]
        u, v = points @ np.asarray(right), points @ np.asarray(up)
        pad = 0.05 * max(float(u.max() - u.min()), float(v.max() - v.min()), 1e-6)
        frame = (float(u.min()) - pad, float(u.max()) + pad, float(v.min()) - pad, float(v.max()) + pad)
        width = int(_TILE * (frame[1] - frame[0]) / (frame[3] - frame[2]))
        image = np.full((_TILE, max(width, _GAP + 1), 3), 255, np.uint8)
        for index, group in enumerate(groups):
            colour = _HUES[min(index, len(_HUES) - 1)]
            for name in group:
                image[silhouette(scene.meshes[name], view, (image.shape[1], _TILE), frame)] = colour
        tiles.extend([image, np.full((_TILE, _GAP, 3), 255, np.uint8)])
    write_png(path, np.concatenate(tiles[:-1], axis=1))
    return path


def _summary(c, scene, groups, finding_lines, marks=()) -> list[str]:
    """One fact per line: parts and triangles, the groups, IoUs, extents, marks, every finding."""
    triangles = sum(int(len(mesh.faces)) for mesh in scene.meshes.values())
    lines = [f"{len(scene)} parts, {triangles} triangles",
             f"{len(groups)} group(s) of touching parts: " + " | ".join(", ".join(group) for group in groups)]
    for expectation in c.expectations:
        lines.extend(_ious(scene, expectation))
    for expectation in c.expectations:
        lines.extend(_extents(scene, expectation))
    lines.extend(f"mark {label}: ({', '.join(f'{v:.2f}' for v in start)}) to ({', '.join(f'{v:.2f}' for v in end)})"
                 for label, start, end in marks)
    lines.extend(finding_lines)
    return lines


def _ious(scene, expectation) -> list[str]:
    """Each part's IoU against its mask in one view, instances of a part counted together."""
    shape = next(iter(expectation.masks.values())).shape
    frame = expectation.frame if expectation.frame is not None else _frame(scene.meshes, expectation.view)
    height, width = shape
    island = Island(view=expectation.view, rect=(0, 0, width, height), frame=frame)
    atlas = Atlas(size=(width, height), islands=[island], uvs={}, meshes={},
                  island_of={name: np.zeros(len(mesh.faces), np.int32)
                             for name, mesh in scene.meshes.items()}, texel=1.0)
    _, regions = template(scene.meshes, atlas, {name: (0, 0, 0) for name in scene.meshes}, 0, whole=True)
    if expectation.view == "back":
        regions = regions[:, ::-1]
    order = list(scene.meshes)
    base_of = {name: name.rstrip("0123456789") or name for name in order}
    lines = []
    for part, mask in expectation.masks.items():
        wanted = [index + 1 for index, name in enumerate(order) if base_of[name] == part]
        drawn = np.asarray(mask, bool)
        if not wanted or not drawn.any():
            continue
        seen = np.isin(regions, wanted)
        lines.append(f"IoU {part} {expectation.view}: {iou(seen, drawn):.3f}")
    return lines


def _extents(scene, expectation) -> list[str]:
    """Where each drawn part spans in one view, in metres along that view's axes."""
    frame = expectation.frame if expectation.frame is not None else _frame(scene.meshes, expectation.view)
    u0, u1, v0, v1 = frame
    axis = {"front": ("x", 1), "back": ("x", -1), "left": ("z", -1)}.get(expectation.view)
    if axis is None:
        return []
    name, sign = axis
    lines = []
    for part, mask in expectation.masks.items():
        rows, cols = np.nonzero(np.asarray(mask, bool))
        if not rows.size:
            continue
        height, width = np.asarray(mask).shape
        u = u0 + (u1 - u0) * (np.array([cols.min(), cols.max() + 1])) / width
        if expectation.view == "back":
            u = u0 + (u1 - u0) * (width - np.array([cols.max() + 1, cols.min()])) / width
        v = v1 - (v1 - v0) * np.array([rows.max() + 1, rows.min()]) / height
        lo, hi = sorted(sign * u)
        lines.append(f"extent {part} {expectation.view}: {name} {lo:.2f}..{hi:.2f}, y {v[0]:.2f}..{v[1]:.2f}")
    return lines


def _line(finding) -> str:
    """One finding, in the report's own line format."""
    from manifoldx.asset.report import Report

    return Report([finding]).lines()[0]
