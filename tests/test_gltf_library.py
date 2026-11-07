"""export_engine(..., library=True): every geometry once by its registry id, every material,
no entity nodes. export --web writes its scene this way; the page draws from the store."""

import numpy as np
import pytest

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.gltf.engine_export import export_engine
from manifoldx.resources import PhongMaterial, StandardMaterial, cube, sphere

from gltf_helpers import load


@pytest.fixture
def engine():
    e = mx.Engine("library", width=64, height=48)
    box = cube(1.0, 1.0, 1.0)
    painted = cube(0.5, 0.5, 0.5)
    painted["colors"] = np.linspace(
        0.0, 1.0, len(painted["positions"]) * 4, dtype=np.float32
    ).reshape(-1, 4)
    e.spawn(Mesh(box), Material(PhongMaterial(color="#557755")), Transform(scale=(20.0, 0.1, 20.0)))
    e.spawn(
        Mesh(box), Material(PhongMaterial(color="#aa3333")), Transform(pos=(0.0, 1.0, 0.0)), n=2
    )
    e.spawn(Mesh(painted), Material(StandardMaterial("#ffffff", vertex_colors=True)), Transform())
    e.state["ball"] = sphere(1.0, 12)
    e._geometry_registry.register(e.state["ball"])  # registered, drawn by no entity
    e.state["painted"] = painted
    return e


def _extras(g):
    return g.scenes[0].extras["manifoldx"]


def test_library_writes_each_geometry_once_and_no_entity_nodes(engine, tmp_path):
    export_engine(engine, tmp_path / "s.glb", library=True)
    g = load(tmp_path / "s.glb")
    extras = _extras(g)
    reg = engine._geometry_registry
    builtin = set(reg._name_to_id.values())
    assert set(extras["mx_geometries"]) == {str(i) for i in reg._geometries if i not in builtin}
    assert len(extras["mx_geometries"]) == 3
    assert not any("mx" in (n.extras or {}) for n in g.nodes)
    for mesh in extras["mx_geometries"].values():
        prim = g.meshes[mesh].primitives[0]
        assert prim.attributes.POSITION is not None and prim.material is None
    assert len(g.meshes) == 3


def test_the_unused_geometry_is_in_the_library(engine, tmp_path):
    export_engine(engine, tmp_path / "s.glb", library=True)
    g = load(tmp_path / "s.glb")
    gid = engine._geometry_registry.register(engine.state["ball"])
    prim = g.meshes[_extras(g)["mx_geometries"][str(gid)]].primitives[0]
    assert g.accessors[prim.attributes.POSITION].count == len(engine.state["ball"]["positions"])


def test_colours_ride_on_the_geometry_and_the_material_says_whether_to_use_them(engine, tmp_path):
    export_engine(engine, tmp_path / "s.glb", library=True)
    g = load(tmp_path / "s.glb")
    library = _extras(g)["mx_geometries"]
    painted = engine._geometry_registry.register(engine.state["painted"])
    for gid, mesh in library.items():
        has_colours = g.meshes[mesh].primitives[0].attributes.COLOR_0 is not None
        assert has_colours == (gid == str(painted))
    flags = {m.name: m.extras["mx"]["vertex_colors"] for m in g.materials}
    assert flags == {"StandardMaterial": True, "PhongMaterial": False}


def test_mx_materials_still_maps_every_material(engine, tmp_path):
    export_engine(engine, tmp_path / "s.glb", library=True)
    g = load(tmp_path / "s.glb")
    ids = _extras(g)["mx_materials"]
    assert set(ids) == {str(i) for i in engine._material_registry._materials}
    assert len(g.materials) == 3


def test_an_entity_drawn_without_a_mesh_is_still_reported(engine, tmp_path):
    from manifoldx.viz import ColormapMaterial, PointCloud

    engine.spawn(
        PointCloud(),
        Material(ColormapMaterial(cmap="viridis", vmin=0.0, vmax=1.0)),
        Transform(),
        n=4,
    )
    report = export_engine(engine, tmp_path / "s.glb", library=True)
    assert sum(e["count"] for e in report.to_dict()["entries"] if e["kind"] == "entity") == 4


def test_without_library_nothing_changes(engine, tmp_path):
    export_engine(engine, tmp_path / "a.glb")
    g = load(tmp_path / "a.glb")
    assert "mx_geometries" not in _extras(g)
    assert sum("mx" in (n.extras or {}) for n in g.nodes) == 4


def test_a_drawn_material_is_reported_as_without_library(engine, tmp_path):
    def entries(**kw):
        report = export_engine(engine, tmp_path / "s.glb", **kw).to_dict()["entries"]
        return [(e["name"], e["effect"], e["count"]) for e in report if e["kind"] == "material"]

    assert entries(library=True) == entries() == [("PhongMaterial", "approximated", 3)]
