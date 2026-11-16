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
    """n point sprites of one material, of which max(1, n // 100) move every frame (a starfield with rain)."""
    import manifoldx as mx
    from manifoldx.components import Component, Material, Transform
    from manifoldx.types import Float
    from manifoldx.viz import ColormapMaterial, PointCloud, Radius, ScalarValue

    Mover = type("Mover", (Component,), {"__annotations__": {"k": Float}})  # this module's annotations are strings

    engine = headless_engine(n)
    engine.camera.fit(60)
    mat = ColormapMaterial(cmap="viridis", vmin=0, vmax=1)
    still, moving = n - max(1, n // 100), max(1, n // 100)
    for count, extra in ((still, {}), (moving, {"Mover": Mover(k=np.zeros(moving))})):
        engine.spawn(PointCloud(), Material(mat), Transform(pos=_positions(count)),
                     ScalarValue(value=np.random.default_rng(1).uniform(0, 1, count).astype(np.float32)),
                     Radius(radius=np.full(count, 0.1, np.float32)), n=count, **extra)

    @engine.system
    def fall(q: mx.Query[Transform, Mover], dt: float):
        q[Transform].position -= np.float32(0.01)

    return engine


def meshes(n: int, *, shadows: bool = False, moving: bool = False, vertex_colors: bool = False):
    """n instances of one cube geometry with one StandardMaterial."""
    import manifoldx as mx
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import DirectionalLight, StandardMaterial, cube

    engine = headless_engine(n)
    engine.camera.fit(60)
    geo = cube(0.5, 0.5, 0.5)
    if vertex_colors:
        geo["colors"] = np.ones((len(geo["positions"]), 3), np.float32)
    mat = StandardMaterial("#88aaff", vertex_colors=vertex_colors)
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
