"""Axes demo — sci-viz primitives v1, Plan 3 part 1.

A small demo that exercises everything Plan 3 part 1 introduced:

- Three colored axis lines (X red, Y green, Z blue) via the new
  AxisFrame + AxisMaterial + LineList rendering path.
- World-anchored end-cap labels at each axis tip — TextLabel +
  LabelMaterial in the default world-anchored mode (Plan 2 with the
  Plan 3 viewport-aware sizing fix).
- A screen-anchored HUD overlay showing the camera spec — exercises
  the new LabelMaterial(anchor_mode="screen") path that bypasses
  view/proj entirely.
- A small reference sphere at the origin so the axes have depth context
  and labels at the +X/+Y/+Z tips occlude correctly behind/in-front of
  the geometry.
"""

import manifoldx as mx
import numpy as np

from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import PointLight, StandardMaterial, sphere
from manifoldx.viz import (
    AxisFrame,
    AxisMaterial,
    LabelMaterial,
    TextLabel,
)
from manifoldx.viz.geometry import AXIS_LINE_X, AXIS_LINE_Y, AXIS_LINE_Z

# Scene parameters ------------------------------------------------------------
EXTENT = 3.0          # half-length of each axis line in world units
SPHERE_RADIUS = 0.4   # central reference sphere

engine = mx.Engine("Axes demo")
engine.camera.fit(EXTENT * 1.6, azimuth=35, elevation=25)

# Reference sphere ------------------------------------------------------------
engine.spawn(
    Mesh(sphere(SPHERE_RADIUS, 24)),
    Material(StandardMaterial("#cccccc", roughness=0.4, metallic=0.1)),
    Transform(pos=(0.0, 0.0, 0.0)),
    n=1,
)
engine.set_lights(
    [
        PointLight(color="#ffffff", intensity=8.0, position=(4.0, 4.0, 4.0)),
        PointLight(color="#88aaff", intensity=3.0, position=(-3.0, -2.0, -3.0)),
    ]
)

# Three axes ------------------------------------------------------------------
# Each axis is one entity carrying its own AXIS_LINE_* geometry, an
# AxisMaterial with the per-batch color, and a Transform whose `scale`
# stretches the unit-line geometry to ±EXTENT on the right axis direction.
for geom, color, scale in [
    (AXIS_LINE_X, "#e64545", (EXTENT, 1.0, 1.0)),
    (AXIS_LINE_Y, "#5fbf5f", (1.0, EXTENT, 1.0)),
    (AXIS_LINE_Z, "#5588ff", (1.0, 1.0, EXTENT)),
]:
    engine.spawn(
        Mesh(geom),
        Material(AxisMaterial(color=color)),
        Transform(pos=(0.0, 0.0, 0.0), scale=scale),
        AxisFrame(extent=EXTENT),
        n=1,
    )

# World-anchored end-cap labels ----------------------------------------------
# Pre-rasterize each axis name into the atlas; each cap entity carries the
# slice index via TextLabel. The LabelMaterial defaults to world-anchored,
# so the entity's Transform.pos is the world position of the label center
# and the billboard always faces the camera. Label aspect ratio matches the
# atlas tile (256:64 = 4:1) so the text fills the rendered quad cleanly.
atlas = engine.get_label_atlas()
slot_x = atlas.get_or_create("+X")
slot_y = atlas.get_or_create("+Y")
slot_z = atlas.get_or_create("+Z")

LABEL_OFFSET = EXTENT + 0.4   # sit just past the +axis tip
for pos, slot in [
    ((LABEL_OFFSET, 0.0, 0.0), slot_x),
    ((0.0, LABEL_OFFSET, 0.0), slot_y),
    ((0.0, 0.0, LABEL_OFFSET), slot_z),
]:
    engine.spawn(
        Material(LabelMaterial(pixel_width=160, pixel_height=40)),
        Transform(pos=pos),
        TextLabel(index=slot),
        n=1,
    )

# Screen-anchored HUD ---------------------------------------------------------
# Top-left corner, independent of the camera. Uses the new
# anchor_mode="screen" path from Plan 3 Task 4. NDC anchor pulled in to
# (-0.55, +0.85) so a 300×40 label on an 800×600 viewport (ndc half-width
# = 300/800 = 0.375) sits comfortably inside the visible region.
hud_slot = atlas.get_or_create(f"manifoldx axes  EXTENT={EXTENT}")
engine.spawn(
    Material(LabelMaterial(pixel_width=300, pixel_height=40, anchor_mode="screen")),
    Transform(pos=(-0.55, 0.85, 0.0)),
    TextLabel(index=hud_slot),
    n=1,
)

# Screen-anchored colormap legend --------------------------------------------
# A viridis colormap legend at the bottom-right corner. The LUT is rasterized
# into a single label-atlas slice via the new register_colormap_legend()
# method and rendered via the existing screen-anchored LabelMaterial path —
# no new material or pipeline. Plan 4's `colormap_legend(...)` shim will
# package this with min/max tick labels.
legend_slot = atlas.register_colormap_legend("viridis")
engine.spawn(
    Material(LabelMaterial(pixel_width=240, pixel_height=24, anchor_mode="screen")),
    Transform(pos=(0.55, -0.85, 0.0)),
    TextLabel(index=legend_slot),
    n=1,
)
legend_caption_slot = atlas.get_or_create("viridis: 0.0 → 1.0")
engine.spawn(
    Material(LabelMaterial(pixel_width=200, pixel_height=24, anchor_mode="screen")),
    Transform(pos=(0.55, -0.93, 0.0)),
    TextLabel(index=legend_caption_slot),
    n=1,
)

# Screen-anchored scale-bar ---------------------------------------------------
# A simple "scale bar" overlay built from Plan 3 part 2 primitives:
# a screen-anchored AxisMaterial line (the bar) plus a screen-anchored
# LabelMaterial (the bar's caption). Both stay pinned to the bottom-left
# corner regardless of camera rotation.
SCALE_BAR_NDC_HALF = 0.18      # 36% of viewport-width across
SCALE_BAR_Y = -0.85
engine.spawn(
    Mesh(AXIS_LINE_X),
    Material(AxisMaterial(color="#ffffff", anchor_mode="screen")),
    Transform(pos=(-0.55, SCALE_BAR_Y, 0.0), scale=(SCALE_BAR_NDC_HALF, 1.0, 1.0)),
    AxisFrame(extent=1.0),
    n=1,
)
scale_label_slot = atlas.get_or_create(f"= {EXTENT * 2:.1f} units")
engine.spawn(
    Material(LabelMaterial(pixel_width=160, pixel_height=30, anchor_mode="screen")),
    Transform(pos=(-0.55, SCALE_BAR_Y - 0.08, 0.0)),
    TextLabel(index=scale_label_slot),
    n=1,
)


# Camera orbit ----------------------------------------------------------------
# Rotate the camera around the +Y axis so the axes spin in view. The Query
# is unused (the system needs *some* component query to register; Transform
# is always present) — the work is on the camera, not the entities.
ORBIT_DEG_PER_SEC = 30.0


@engine.system
def orbit_camera(query: mx.Query[Transform], dt: float):
    engine.camera.orbit(d_azimuth=ORBIT_DEG_PER_SEC * dt)


if __name__ == "__main__":
    engine.cli()
