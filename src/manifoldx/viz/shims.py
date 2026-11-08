"""Altair-style declarative shim layer for sci-viz primitives v1, plan 4.

Wraps the imperative ECS API in a grammar-of-graphics shape so the 80% case
fits in 10 lines:

    chart = (
        mxv.points(positions=p, color=mxv.color(s, cmap="viridis"), size=r)
        + mxv.axes(extent=10)
        + mxv.legend("Speed")
    )

    @chart.simulate
    def step(dt):
        p[:] += v * dt

    chart.cli()

Design: `.knowledge/analysis/2026-05-06-sci-viz-primitives-v1-plan-4-altair-shim-design.md`.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np


# =============================================================================
# Channel objects
# =============================================================================


class Channel:
    """Base for declarative channel wrappers (`mxv.color(...)`, etc.)."""

    data: np.ndarray

    def __init__(self, data):
        self.data = np.asarray(data)


class ColorChannel(Channel):
    """Color encoding: per-instance scalar mapped through a colormap LUT."""

    def __init__(
        self,
        data,
        *,
        cmap: str = "viridis",
        domain: Optional[tuple[float, float]] = None,
        title: str = "",
    ):
        super().__init__(data)
        self.cmap = cmap
        self.domain = domain
        self.title = title


class SizeChannel(Channel):
    """Size encoding: per-instance world-space radius for sprites."""

    def __init__(self, data, *, units: str = "world"):
        super().__init__(data)
        self.units = units


class PositionChannel(Channel):
    """Position encoding: (N, 3) world-space positions."""

    def __init__(self, data):
        super().__init__(data)


def _as_channel(value, cls: type, **defaults) -> Channel:
    """Auto-wrap a bare array in the given channel class with default config."""
    if isinstance(value, Channel):
        return value
    if value is None:
        return None
    return cls(value, **defaults)


# Public factory aliases — `mxv.color(...)`, `mxv.size(...)`, `mxv.position(...)`.
def color(data, **kwargs) -> ColorChannel:
    return ColorChannel(data, **kwargs)


def size(data, **kwargs) -> SizeChannel:
    return SizeChannel(data, **kwargs)


def position(data, **kwargs) -> PositionChannel:
    return PositionChannel(data, **kwargs)


def _infer_domain(arr: np.ndarray) -> tuple[float, float]:
    """Auto-derive (vmin, vmax) from data when the user didn't pin one."""
    if arr.size == 0:
        return (0.0, 1.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        hi = lo + 1.0  # avoid degenerate domain when all values are equal
    return (lo, hi)


# =============================================================================
# Mark base
# =============================================================================


class Mark:
    """Base class for declarative marks. Each mark spawns its entities into
    a target Engine when `apply(engine)` is called by `Chart.build()`."""

    def apply(self, engine) -> None:
        raise NotImplementedError

    def __add__(self, other: "Mark | Chart") -> "Chart":
        chart = Chart()
        chart.marks.append(self)
        if isinstance(other, Chart):
            chart.marks.extend(other.marks)
            chart.simulate_callbacks.extend(other.simulate_callbacks)
        else:
            chart.marks.append(other)
        return chart


# =============================================================================
# Concrete marks
# =============================================================================


class PointsMark(Mark):
    """Sprite point cloud — wraps Plan 1's PointCloud + ColormapMaterial path.

    Per-instance position, color (via colormap), and size (radius). Color
    accepts a bare numpy array (auto-wrapped with cmap='viridis' and a
    domain inferred from the data) or a `mxv.color(...)` channel object
    that pins cmap and domain explicitly.

    The user's arrays are referenced, not copied; per-frame mutation in
    `@chart.simulate` propagates to the GPU because a sync system copies
    the live arrays into ECS storage each frame.
    """

    def __init__(
        self,
        *,
        positions,
        color=None,
        size=None,
    ):
        # Normalize each channel — bare array → wrapped Channel; None stays None.
        self.positions = _as_channel(positions, PositionChannel)
        self.color = _as_channel(color, ColorChannel)
        self.size = _as_channel(size, SizeChannel)

        if self.positions is None:
            raise ValueError("points(...) requires a `positions` channel")

        pos_arr = self.positions.data
        if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
            raise ValueError(
                f"positions must be (N, 3); got shape {pos_arr.shape}"
            )
        self._n = pos_arr.shape[0]

    def apply(self, engine) -> None:
        from manifoldx.components import Material, Transform
        from manifoldx.viz import (
            ColormapMaterial,
            PointCloud,
            Radius,
            ScalarValue,
        )

        n = self._n
        pos_arr = np.asarray(self.positions.data, dtype=np.float32)

        # Color → ColormapMaterial. Domain auto-inferred from data when None.
        if self.color is not None:
            color_arr = np.asarray(self.color.data, dtype=np.float32)
            domain = self.color.domain or _infer_domain(color_arr)
            material = ColormapMaterial(
                cmap=self.color.cmap, vmin=domain[0], vmax=domain[1]
            )
        else:
            color_arr = np.zeros(n, dtype=np.float32)
            material = ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)

        # Size → Radius. Defaults to 1.0 when no size channel.
        if self.size is not None:
            size_arr = np.asarray(self.size.data, dtype=np.float32)
        else:
            size_arr = np.full(n, 1.0, dtype=np.float32)

        alive_before = engine.store._alive.copy()
        engine.spawn(
            PointCloud(),
            Material(material),
            Transform(pos=pos_arr),
            ScalarValue(value=color_arr),
            Radius(radius=size_arr),
            n=n,
        )

        # Live-data sync: each frame, copy the user's arrays into ECS
        # component storage so per-frame mutations propagate to the GPU.
        # We capture the arrays + the slice of entity indices via closure.
        # The entities this spawn added, wherever they sit in the store: a mark
        # spawned before this one (a mesh) holds the first rows.
        entity_slice = np.flatnonzero(engine.store._alive & ~alive_before)

        positions_ref = self.positions.data  # alias the user's array
        color_ref = self.color.data if self.color is not None else None
        size_ref = self.size.data if self.size is not None else None

        store = engine.store

        from manifoldx.systems import Query

        def _sync(query: Query[Transform], dt: float):
            # Copy mutable user arrays into the corresponding component slots.
            store._components["Transform"][entity_slice, 0:3] = positions_ref
            if color_ref is not None:
                store._components["ScalarValue"][entity_slice, 0] = color_ref
            if size_ref is not None:
                store._components["Radius"][entity_slice, 0] = size_ref

        _sync.__name__ = f"_points_sync_{id(self)}"
        engine.system(_sync)


def points(*, positions, color=None, size=None) -> "Chart":
    """Declarative point cloud mark. Returns a Chart with one mark so
    callers can immediately access `.engine`, `.simulate`, `.run()`, etc.
    or compose with `+`."""
    return _wrap(PointsMark(positions=positions, color=color, size=size))


# --- mesh -------------------------------------------------------------------


class MeshMark(Mark):
    """Single 3D mesh entity (PBR or unlit)."""

    def __init__(self, *, geometry, material, position=(0.0, 0.0, 0.0), scale=None):
        self.geometry = geometry
        self.material = material
        self.position = position
        self.scale = scale

    def apply(self, engine) -> None:
        from manifoldx.components import Material, Mesh, Transform

        kwargs = {"pos": self.position}
        if self.scale is not None:
            kwargs["scale"] = self.scale
        engine.spawn(
            Mesh(self.geometry),
            Material(self.material),
            Transform(**kwargs),
            n=1,
        )


def mesh(*, geometry, material, position=(0.0, 0.0, 0.0), scale=None) -> "Chart":
    """Single 3D mesh — sphere, cube, etc. + a PBR/unlit material."""
    return _wrap(
        MeshMark(geometry=geometry, material=material, position=position, scale=scale)
    )


# --- axes -------------------------------------------------------------------


class AxesMark(Mark):
    """Three colored axes (X red, Y green, Z blue) + optional end-cap labels."""

    DEFAULT_COLORS = {"x": "#e64545", "y": "#5fbf5f", "z": "#5588ff"}

    def __init__(
        self,
        *,
        extent: float = 1.0,
        labels: bool = True,
        colors: Optional[dict] = None,
    ):
        self.extent = float(extent)
        self.labels = labels
        self.colors = {**self.DEFAULT_COLORS, **(colors or {})}

    def apply(self, engine) -> None:
        from manifoldx.components import Material, Mesh, Transform
        from manifoldx.viz import AxisFrame, AxisMaterial, LabelMaterial, TextLabel

        e = self.extent
        for geom_name, axis_key, scale in [
            ("axis_line_x", "x", (e, 1.0, 1.0)),
            ("axis_line_y", "y", (1.0, e, 1.0)),
            ("axis_line_z", "z", (1.0, 1.0, e)),
        ]:
            geom = engine._geometry_registry.get_by_name(geom_name)
            engine.spawn(
                Mesh(geom),
                Material(AxisMaterial(color=self.colors[axis_key])),
                Transform(pos=(0.0, 0.0, 0.0), scale=scale),
                AxisFrame(extent=e),
                n=1,
            )

        if self.labels:
            atlas = engine.get_label_atlas()
            slot_x = atlas.get_or_create("+X")
            slot_y = atlas.get_or_create("+Y")
            slot_z = atlas.get_or_create("+Z")
            offset = e + 0.4 * e  # 40% of extent past the tip
            for pos, slot in [
                ((offset, 0.0, 0.0), slot_x),
                ((0.0, offset, 0.0), slot_y),
                ((0.0, 0.0, offset), slot_z),
            ]:
                engine.spawn(
                    Material(LabelMaterial(pixel_width=160, pixel_height=40)),
                    Transform(pos=pos),
                    TextLabel(index=slot),
                    n=1,
                )


def axes(*, extent: float = 1.0, labels: bool = True, colors=None) -> "Chart":
    """Three colored 3D axes + optional end-cap labels."""
    return _wrap(AxesMark(extent=extent, labels=labels, colors=colors))


# --- legend -----------------------------------------------------------------


_NDC_POSITIONS = {
    "bottom-left": (-0.55, -0.85),
    "bottom-right": (0.55, -0.85),
    "top-left": (-0.55, 0.85),
    "top-right": (0.55, 0.85),
}


class LegendMark(Mark):
    """Screen-anchored colormap legend.

    Either pass a ColorChannel directly (preserves cmap + domain + title)
    or a `cmap=` string + `title=` for a standalone legend.
    """

    def __init__(
        self,
        color_channel: Optional[ColorChannel] = None,
        *,
        cmap: Optional[str] = None,
        title: str = "",
        position: str = "bottom-right",
    ):
        if color_channel is None and cmap is None:
            raise ValueError(
                "legend(...) needs either a ColorChannel or an explicit cmap=..."
            )
        if color_channel is not None:
            self.cmap = color_channel.cmap
            self.title = title or color_channel.title
            self.domain = color_channel.domain
        else:
            self.cmap = cmap
            self.title = title
            self.domain = None
        if position not in _NDC_POSITIONS:
            raise ValueError(
                f"position must be one of {list(_NDC_POSITIONS)}; got {position!r}"
            )
        self.position = position

    def apply(self, engine) -> None:
        from manifoldx.components import Material, Transform
        from manifoldx.viz import LabelMaterial, TextLabel

        atlas = engine.get_label_atlas()
        legend_slot = atlas.register_colormap_legend(self.cmap)

        anchor_x, anchor_y = _NDC_POSITIONS[self.position]
        # Bar at the anchor.
        engine.spawn(
            Material(LabelMaterial(pixel_width=240, pixel_height=24, anchor_mode="screen")),
            Transform(pos=(anchor_x, anchor_y, 0.0)),
            TextLabel(index=legend_slot),
            n=1,
        )

        # Caption below the bar (when title is non-empty).
        if self.title:
            domain_str = (
                f"  ({self.domain[0]:.2g} → {self.domain[1]:.2g})"
                if self.domain is not None
                else ""
            )
            caption_slot = atlas.get_or_create(f"{self.title}{domain_str}")
            engine.spawn(
                Material(LabelMaterial(pixel_width=220, pixel_height=24, anchor_mode="screen")),
                Transform(pos=(anchor_x, anchor_y - 0.08, 0.0)),
                TextLabel(index=caption_slot),
                n=1,
            )


def legend(
    color_channel: Optional[ColorChannel] = None,
    *,
    cmap: Optional[str] = None,
    title: str = "",
    position: str = "bottom-right",
) -> "Chart":
    """Screen-anchored colormap legend. Pass a ColorChannel from a points
    mark to auto-sync cmap + domain, or pass `cmap=` + `title=` for a
    standalone legend."""
    return _wrap(
        LegendMark(color_channel, cmap=cmap, title=title, position=position)
    )


# --- scale_bar --------------------------------------------------------------


class ScaleBarMark(Mark):
    """Screen-anchored scale bar — a horizontal line + caption label."""

    def __init__(
        self,
        *,
        ndc_length: float = 0.3,
        label: str = "",
        position: str = "bottom-left",
        color: str = "#ffffff",
    ):
        if position not in _NDC_POSITIONS:
            raise ValueError(
                f"position must be one of {list(_NDC_POSITIONS)}; got {position!r}"
            )
        self.ndc_length = float(ndc_length)
        self.label = label
        self.position = position
        self.color = color

    def apply(self, engine) -> None:
        from manifoldx.components import Material, Mesh, Transform
        from manifoldx.viz import AxisFrame, AxisMaterial, LabelMaterial, TextLabel

        anchor_x, anchor_y = _NDC_POSITIONS[self.position]
        half_len = self.ndc_length * 0.5

        # The bar — screen-anchored AxisMaterial.
        geom = engine._geometry_registry.get_by_name("axis_line_x")
        engine.spawn(
            Mesh(geom),
            Material(AxisMaterial(color=self.color, anchor_mode="screen")),
            Transform(pos=(anchor_x, anchor_y, 0.0), scale=(half_len, 1.0, 1.0)),
            AxisFrame(extent=1.0),
            n=1,
        )

        # The caption — screen-anchored LabelMaterial.
        atlas = engine.get_label_atlas()
        caption_slot = atlas.get_or_create(self.label or " ")
        engine.spawn(
            Material(LabelMaterial(pixel_width=160, pixel_height=24, anchor_mode="screen")),
            Transform(pos=(anchor_x, anchor_y - 0.08, 0.0)),
            TextLabel(index=caption_slot),
            n=1,
        )


def scale_bar(
    *,
    ndc_length: float = 0.3,
    label: str = "",
    position: str = "bottom-left",
    color: str = "#ffffff",
) -> "Chart":
    """Screen-anchored scale bar — a horizontal line + caption label."""
    return _wrap(
        ScaleBarMark(
            ndc_length=ndc_length, label=label, position=position, color=color
        )
    )


# --- lights -----------------------------------------------------------------


class LightsMark(Mark):
    """Declarative light-list — thin wrapper over engine.set_lights."""

    def __init__(self, lights):
        self.lights = list(lights)

    def apply(self, engine) -> None:
        engine.set_lights(self.lights)


def lights(specs) -> "Chart":
    """Set the engine's light list — a thin wrapper over engine.set_lights."""
    return _wrap(LightsMark(specs))


def _wrap(mark: Mark) -> "Chart":
    """Promote a single mark into a Chart so factory callers can immediately
    access Chart methods or compose with `+`."""
    chart = Chart()
    chart.marks.append(mark)
    return chart


# =============================================================================
# Chart
# =============================================================================


class Chart:
    """A composed scene of marks + simulation callbacks. Lazy-builds an Engine
    on first terminal call (`run`, `render`, `cli`, or `.engine`)."""

    def __init__(self):
        self.marks: list[Mark] = []
        self.simulate_callbacks: list[Callable] = []
        self._engine = None
        self._title: str = "manifoldx"
        self._width: int = 800
        self._height: int = 600

    # --- composition ---------------------------------------------------------

    def __add__(self, other: "Mark | Chart") -> "Chart":
        out = Chart()
        out.marks.extend(self.marks)
        out.simulate_callbacks.extend(self.simulate_callbacks)
        if isinstance(other, Chart):
            out.marks.extend(other.marks)
            out.simulate_callbacks.extend(other.simulate_callbacks)
        else:
            out.marks.append(other)
        out._title = self._title
        out._width = self._width
        out._height = self._height
        return out

    # --- live data -----------------------------------------------------------

    def simulate(self, func: Callable) -> Callable:
        """Decorator: register a per-frame simulation callback.

        The callback is invoked once per frame with `dt` as its sole argument.
        Multiple callbacks run in registration order before any channel-sync
        systems and before the renderer reads component data.
        """
        self.simulate_callbacks.append(func)
        # If the engine is already built, attach immediately.
        if self._engine is not None:
            self._attach_simulate(func)
        return func

    # --- build / terminal ----------------------------------------------------

    @property
    def engine(self):
        """Lazily build the Engine and return it."""
        if self._engine is None:
            self.build()
        return self._engine

    def build(self):
        """Construct an Engine, apply each mark, wire simulate callbacks."""
        if self._engine is not None:
            return self._engine

        from manifoldx.engine import Engine
        from manifoldx.components import Transform

        engine = Engine(self._title, width=self._width, height=self._height)
        self._engine = engine

        for mark in self.marks:
            mark.apply(engine)

        for cb in self.simulate_callbacks:
            self._attach_simulate(cb)

        return engine

    def _attach_simulate(self, func: Callable) -> None:
        """Wire one user-provided simulate callback into the engine as a system."""
        from manifoldx.systems import Query
        from manifoldx.components import Transform

        # Compile the user's `def step(dt): ...` into an
        # `@engine.system def _wrapper(query, dt): step(dt)` shape so it
        # ticks every frame against the existing system runner.
        engine = self._engine

        def _wrapper(query: Query[Transform], dt: float):
            func(dt)

        # Preserve the original function name for any introspection.
        _wrapper.__name__ = f"_chart_simulate_{func.__name__}"
        engine.system(_wrapper)

    def run(self) -> None:
        """Open an interactive window and start the render loop."""
        self.build()
        self._engine.run()

    def render(
        self,
        output: str = "scene.mp4",
        *,
        duration: float = 60.0,
        fps: int = 30,
        quality: str = "high",
    ) -> None:
        """Render to a video file (offline, no window)."""
        self.build()
        self._engine.render(
            output=output, duration=duration, fps=fps, quality=quality
        )

    def cli(self) -> None:
        """Parse argv and dispatch to run() or render() — same shape as
        the existing `engine.cli()` so demos can drop in this method
        verbatim."""
        self.build()
        self._engine.cli()
