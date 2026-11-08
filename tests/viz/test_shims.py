"""Unit tests for the Plan 4 declarative shim layer."""
import numpy as np
import pytest


def test_chart_construction_no_marks():
    """A Chart can be constructed with no marks; build() returns an Engine."""
    import manifoldx.viz as mxv

    chart = mxv.Chart()
    engine = chart.engine
    assert engine is not None


def test_chart_addition_returns_chart_with_combined_marks():
    """`mark + mark` returns a Chart whose mark list is the concatenation."""
    import manifoldx.viz as mxv
    from manifoldx.viz.shims import Mark

    class _DummyMark(Mark):
        def apply(self, engine):
            pass

    a = _DummyMark()
    b = _DummyMark()
    chart = a + b
    assert isinstance(chart, mxv.Chart)
    assert len(chart.marks) == 2
    assert chart.marks[0] is a and chart.marks[1] is b


def test_chart_addition_associative():
    """Three marks combined in different groupings yield the same mark order."""
    import manifoldx.viz as mxv
    from manifoldx.viz.shims import Mark

    class _DummyMark(Mark):
        def __init__(self, tag):
            self.tag = tag

        def apply(self, engine):
            pass

    a, b, c = _DummyMark("a"), _DummyMark("b"), _DummyMark("c")
    left = (a + b) + c
    right = a + (b + c)
    assert [m.tag for m in left.marks] == ["a", "b", "c"]
    assert [m.tag for m in right.marks] == ["a", "b", "c"]


def test_color_channel_wraps_bare_array():
    """A bare numpy array passed to mxv.color is held as the data field."""
    import manifoldx.viz as mxv

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ch = mxv.color(arr, cmap="viridis", domain=(0, 5))
    assert ch.data is arr
    assert ch.cmap == "viridis"
    assert ch.domain == (0, 5)


def test_color_channel_defaults():
    """Defaults: cmap='viridis', domain inferred from data."""
    import manifoldx.viz as mxv

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ch = mxv.color(arr)
    assert ch.cmap == "viridis"
    assert ch.domain is None  # None means "infer at build time"


def test_points_mark_spawns_entities_on_build():
    """mxv.points(...) materializes a sprite point cloud when the chart builds."""
    import manifoldx.viz as mxv

    n = 10
    positions = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
    speeds = np.random.uniform(0, 5, n).astype(np.float32)
    radii = np.full(n, 0.05, dtype=np.float32)

    chart = mxv.points(positions=positions, color=speeds, size=radii)
    engine = chart.engine

    # The Plan-1 components for the sprite path must be registered.
    assert "PointCloud" in engine.store._components
    assert "ScalarValue" in engine.store._components
    assert "Radius" in engine.store._components
    # n alive entities.
    assert int(np.sum(engine.store._alive)) == n


def test_points_mark_with_explicit_color_channel():
    """A wrapped color channel sets cmap and domain on the underlying material."""
    import manifoldx.viz as mxv
    from manifoldx.viz import ColormapMaterial

    n = 8
    positions = np.zeros((n, 3), dtype=np.float32)
    speeds = np.full(n, 2.5, dtype=np.float32)

    chart = mxv.points(
        positions=positions,
        color=mxv.color(speeds, cmap="inferno", domain=(0.0, 5.0)),
    )
    engine = chart.engine

    # Find the ColormapMaterial registered by the points mark.
    found = [
        m for m in engine._material_registry._materials.values()
        if isinstance(m, ColormapMaterial)
    ]
    assert len(found) == 1
    mat = found[0]
    assert mat.cmap == "inferno"
    assert mat.vmin == 0.0
    assert mat.vmax == 5.0


def test_chart_simulate_decorator_registers_callback():
    """@chart.simulate appends to simulate_callbacks before build."""
    import manifoldx.viz as mxv

    chart = mxv.points(
        positions=np.zeros((3, 3), dtype=np.float32),
        color=np.zeros(3, dtype=np.float32),
        size=np.full(3, 0.05, dtype=np.float32),
    )

    @chart.simulate
    def step(dt):
        pass

    assert step in chart.simulate_callbacks


def test_mesh_mark_spawns_mesh_entity():
    """mxv.mesh(...) materializes one mesh entity with Material + Transform."""
    import manifoldx.viz as mxv
    from manifoldx.resources import StandardMaterial, sphere

    chart = mxv.mesh(
        geometry=sphere(0.5, 16),
        material=StandardMaterial("#ff8800"),
        position=(1.0, 2.0, 3.0),
    )
    engine = chart.engine
    assert "Mesh" in engine.store._components
    assert int(np.sum(engine.store._alive)) == 1


def test_axes_mark_spawns_three_axis_entities():
    """mxv.axes(labels=False) spawns three AxisFrame entities (X, Y, Z)."""
    import manifoldx.viz as mxv

    chart = mxv.axes(extent=5.0, labels=False)
    engine = chart.engine
    assert "AxisFrame" in engine.store._components
    assert int(np.sum(engine.store._alive)) == 3


def test_axes_mark_with_labels_spawns_six_entities():
    """labels=True (default) adds three end-cap TextLabel entities."""
    import manifoldx.viz as mxv

    chart = mxv.axes(extent=5.0, labels=True)
    engine = chart.engine
    assert "AxisFrame" in engine.store._components
    assert "TextLabel" in engine.store._components
    # 3 axes + 3 end-cap labels.
    assert int(np.sum(engine.store._alive)) == 6


def test_legend_mark_spawns_screen_anchored_label():
    """mxv.legend(...) spawns one screen-anchored TextLabel entity carrying
    the colormap LUT atlas slot."""
    import manifoldx.viz as mxv

    chart = mxv.legend(cmap="viridis", title="Speed")
    engine = chart.engine
    assert "TextLabel" in engine.store._components
    # legend = 1 atlas-LUT label + (optional) caption label
    assert int(np.sum(engine.store._alive)) >= 1


def test_scale_bar_mark_spawns_screen_anchored_line_and_label():
    """mxv.scale_bar(...) spawns a screen-anchored AxisFrame + label entity."""
    import manifoldx.viz as mxv

    chart = mxv.scale_bar(ndc_length=0.3, label="2 m")
    engine = chart.engine
    # bar (1 AxisFrame) + caption label (1 TextLabel) = 2 entities
    assert int(np.sum(engine.store._alive)) == 2


def test_lights_mark_attaches_lights_to_engine():
    """mxv.lights([...]) is a thin wrapper over engine.set_lights."""
    import manifoldx.viz as mxv
    from manifoldx.resources import PointLight

    chart = mxv.lights([PointLight(color="#ffffff", intensity=10.0, position=(5.0, 5.0, 5.0))])
    engine = chart.engine
    assert len(engine._lights) == 1


def test_chart_full_compose():
    """Compose every Tier 1 + Tier 2 mark into one chart and confirm build."""
    import manifoldx.viz as mxv
    from manifoldx.resources import PointLight, StandardMaterial, sphere

    n = 20
    positions = np.zeros((n, 3), dtype=np.float32)
    speeds = np.linspace(0, 5, n, dtype=np.float32)
    radii = np.full(n, 0.05, dtype=np.float32)

    chart = (
        mxv.points(
            positions=positions,
            color=mxv.color(speeds, cmap="viridis", domain=(0, 5)),
            size=radii,
        )
        + mxv.mesh(geometry=sphere(0.4, 16), material=StandardMaterial("#cccccc"))
        + mxv.axes(extent=5.0)
        + mxv.legend(cmap="viridis", title="Speed")
        + mxv.scale_bar(ndc_length=0.3, label="2 m")
        + mxv.lights([PointLight(color="#ffffff", intensity=10.0, position=(3.0, 3.0, 3.0))])
    )
    engine = chart.engine
    # Component sanity: every Plan 1-3 component path is registered.
    for comp in ("Transform", "Mesh", "Material", "PointCloud", "AxisFrame", "TextLabel"):
        assert comp in engine.store._components, f"missing component: {comp}"


def test_points_after_a_mesh_mark_sync_their_own_rows():
    """The points' per-frame sync wrote into the first live rows of the store,
    which belong to a mesh mark spawned before them."""
    import manifoldx.viz as mxv
    from manifoldx.resources import cube, standard

    try:
        from manifoldx.backends import get_offscreen_canvas
        get_offscreen_canvas(width=8, height=8)
    except Exception as e:
        pytest.skip(f"offscreen canvas unavailable: {e}")
    positions = np.array([[11.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=np.float32)
    chart = mxv.mesh(geometry=cube(1, 1, 1), material=standard("#888888"), position=(0, 5, 0))
    chart = chart + mxv.points(positions=positions)
    engine = chart.engine
    engine.render_frame(supersample=1)
    t = engine.store._components["Transform"]
    alive = np.flatnonzero(engine.store._alive)
    rows = {tuple(np.round(t[i, 0:3], 3)) for i in alive}
    assert (0.0, 5.0, 0.0) in rows, rows
    assert {(11.0, 0.0, 0.0), (12.0, 0.0, 0.0)} <= rows, rows
