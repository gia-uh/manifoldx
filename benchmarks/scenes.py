"""Scene builders for the suites: each returns a populated headless engine, not yet started."""

from __future__ import annotations

import numpy as np

from benchmarks.harness import headless_engine


def _positions(n: int, extent: float = 50.0) -> np.ndarray:
    return np.random.default_rng(0).uniform(-extent, extent, (n, 3)).astype(np.float32)


def sprites(n: int):
    """n point sprites (PointCloud + ColormapMaterial + ScalarValue + Radius)."""
    from manifoldx.components import Material, Transform
    from manifoldx.viz import ColormapMaterial, PointCloud, Radius, ScalarValue

    engine = headless_engine(n)
    engine.camera.fit(60)
    engine.spawn(
        PointCloud(),
        Material(ColormapMaterial(cmap="viridis", vmin=0, vmax=1)),
        Transform(pos=_positions(n)),
        ScalarValue(value=np.random.default_rng(1).uniform(0, 1, n).astype(np.float32)),
        Radius(radius=np.full(n, 0.1, np.float32)),
        n=n,
    )
    return engine


def sprites_mixed(n: int):
    """n point sprites of one material, of which the last max(1, n // 100) move every frame (a starfield
    with rain)."""
    from manifoldx.components import Material, Transform
    from manifoldx.viz import ColormapMaterial, PointCloud, Radius, ScalarValue

    engine = headless_engine(n)
    engine.camera.fit(60)
    engine.spawn(PointCloud(), Material(ColormapMaterial(cmap="viridis", vmin=0, vmax=1)),
                 Transform(pos=_positions(n)),
                 ScalarValue(value=np.random.default_rng(1).uniform(0, 1, n).astype(np.float32)),
                 Radius(radius=np.full(n, 0.1, np.float32)), n=n)
    moving = slice(n - max(1, n // 100), n)

    @engine.system
    def fall(q, dt: float):  # writes the store's rows: a Query would match every sprite
        engine.store._components["Transform"][moving, 1] -= np.float32(0.01)

    return engine


def meshes(n: int, *, shadows: bool = False, moving: bool = False, vertex_colors: bool = False,
           tinted: bool = False):
    """n instances of one cube geometry with one StandardMaterial; tinted gives each its own Tint."""
    import manifoldx as mx
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import DirectionalLight, StandardMaterial, cube

    engine = headless_engine(n)
    engine.camera.fit(60)
    geo = cube(0.5, 0.5, 0.5)
    if vertex_colors or tinted:
        geo["colors"] = np.ones((len(geo["positions"]), 3), np.float32)
    mat = StandardMaterial("#88aaff", vertex_colors=vertex_colors or tinted, tinted=tinted)
    if tinted:
        from manifoldx.components import Tint

        rgb = np.random.default_rng(1).uniform(0.2, 1.0, (n, 3))
        engine.spawn(Mesh(geo), Material(mat), Transform(pos=_positions(n)), Tint(rgb=rgb), n=n)
    else:
        engine.spawn(Mesh(geo), Material(mat), Transform(pos=_positions(n)), n=n)
    if shadows:
        engine.set_sun(DirectionalLight("#ffffff", 2.0, (-1, -2, -1)))
        engine.enable_shadows()
    if moving:

        @engine.system
        def drift(q: mx.Query[Transform], dt: float):
            q[Transform].position += np.float32(0.01)

    return engine


def unique_meshes(n: int):
    """n entities, each with its own geometry (one draw batch each), one shared material."""
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial, cube

    engine = headless_engine(n)
    engine.camera.fit(60)
    mat = StandardMaterial("#88aaff")
    for p in _positions(n):
        engine.spawn(Mesh(cube(0.5, 0.5, 0.5)), Material(mat), Transform(pos=tuple(p)), n=1)
    return engine


def skinned(n: int):
    """n entities of one skinned cube (every vertex on joint 0); returns (engine, first handle)."""
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial, cube

    engine = headless_engine(n)
    geo = cube(0.5, 0.5, 0.5)
    geo["joints"] = np.zeros(len(geo["positions"]), np.uint32)
    first = engine.spawn(
        Mesh(geo), Material(StandardMaterial("#cccccc")), Transform(pos=_positions(n)), n=n
    )
    return engine, first


# One component class per independent column, built on first use so k can be any
# size. The names are stable: a second scene of the same k reuses the classes.
_COLUMNS: dict = {}


def _column(i: int):
    from manifoldx.components import Component
    from manifoldx.types import Vector3

    cls = _COLUMNS.get(i)
    if cls is None:
        name = f"ParallelColumn{i}"
        cls = type(name, (Component,), {"__annotations__": {"value": Vector3}})
        _COLUMNS[i] = cls
    return cls


def independent_systems(k: int, *, parallel: bool = True):
    """k independent systems, each moving 2,000 entities of a component of its own.

    One group when the engine has parallel systems; on a build without the
    `parallel` argument the same systems register serially, so `--against main`
    times the work rather than failing on the new keyword.
    """
    import inspect

    import manifoldx as mx
    from manifoldx.types import Vector3  # noqa: F401  (the annotation below uses it)

    engine = headless_engine(2_000 * k)
    supports_parallel = "parallel" in inspect.signature(mx.Engine.system).parameters
    for i in range(k):
        column = _column(i)
        column.register(engine.store)
        engine.store.spawn(
            2_000, **{column.__name__: np.random.default_rng(i).random((2_000, 3), np.float32)}
        )

        def step(q, dt: float, _column=column):
            # ~2.5 ms of small numpy calls, the shape of ftscale.py's system: a
            # few passes over this system's own (2,000, 3) block.
            v = q[_column].value.data
            for _ in range(96):
                v = np.sin(v) * 0.5 + np.sqrt(v * v + 0.25)
                v = np.tanh(v * 0.5)
            q[_column].value = v

        step.__name__ = f"step{i}"
        step.__qualname__ = f"independent_systems.<locals>.step{i}"
        step.__annotations__ = {"q": mx.Query[column], "dt": float}
        if supports_parallel:
            engine.system(step, parallel=parallel)
        else:
            engine.system(step)

    return engine


def city(n: int, moving: int = 32, stream: bool = False):
    """A streamed city: n static mesh instances (a tenth of them cells with a geometry each, the rest
    props of 4 shared geometries) over 8 materials, and `moving` cars driven by a system every frame.
    With `stream`, every frame the oldest cell goes and a new one, of a new geometry, comes in its place."""
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial, cube, sphere

    engine = headless_engine(n + moving + 8)
    engine.camera.fit(60)
    mats = [StandardMaterial(f"#{0x404040 + 0x101010 * k:06x}") for k in range(8)]
    pos = _positions(n)
    cells = max(1, n // 10)
    live = [engine.spawn(Mesh(cube(1.0, 0.2 + 0.01 * (i % 50), 1.0)), Material(mats[i % 8]),
                         Transform(pos=tuple(pos[i])), n=1) for i in range(cells)]
    props = [cube(0.3, 0.3, 0.3), sphere(0.3), cube(0.1, 1.0, 0.1), sphere(0.5)]
    for k, start in enumerate(range(cells, n, -(-(n - cells) // 4))):
        stop = min(n, start - (-(n - cells) // 4))
        engine.spawn(Mesh(props[k]), Material(mats[k]), Transform(pos=pos[start:stop]), n=stop - start)
    first = engine.store._high_water
    engine.spawn(Mesh(cube(0.4, 0.3, 0.8)), Material(mats[7]), Transform(pos=_positions(moving, 40.0)), n=moving)
    cars = slice(first, first + moving)

    @engine.system
    def drive(q, dt: float):  # writes the store's rows: a Query would match every entity
        engine.store._components["Transform"][cars, 0] += np.float32(0.05)

    if stream:
        made = [cells]

        @engine.system
        def arrive(q, dt: float):
            i = made[0] % cells
            engine.destroy([live[i].index])
            live[i] = engine.spawn(Mesh(cube(1.0, 0.7 + 0.001 * made[0], 1.0)), Material(mats[i % 8]),
                                   Transform(pos=tuple(pos[i])), n=1)
            made[0] += 1

    return engine


def gui_hud(n: int, size: int = 640):
    """An empty scene under n HUD panels, each a background, two font lines (drawn through the image path,
    like SVG icons) and an atlas line, laid out at the four corners and scaled from a 360-pixel design
    height. Returns (engine, texts)."""
    from importlib.resources import files

    import manifoldx as mx
    from manifoldx.gui import Panel, Text

    font = str(files("manifoldx.viz.assets").joinpath("DejaVuSansMono.ttf"))
    engine = mx.Engine("gui", width=size, height=size)
    engine.gui.design_height = 360
    corners = ("top-left", "top-right", "bottom-left", "bottom-right")
    texts = []
    for i in range(n):
        text = Text(f"{i:03d}", style_overrides={"font": font, "font_size": 18, "fg": "#ffffff"})
        texts.append(text)
        engine.gui.append(Panel(
            [Text("N", style_overrides={"font": font, "font_size": 12, "fg": "#f0c020"}), text, Text("km/h", style_overrides={"font_size": 10})],
            anchor=corners[i % 4], offset=(8 + 40 * (i // 4 % 8), 8 + 30 * (i // 32)),
            style_overrides={"direction": "h", "gap": 4, "padding": 4, "bg": "#202020c0"},
        ))
    return engine, texts
