"""Public engine APIs the web target's transpiled code calls instead of private state."""

import numpy as np
import pytest

from manifoldx.components import Material, Mesh, Transform
from manifoldx.engine import Engine
from manifoldx.resources import BasicMaterial, cube


def _engine_with_box():
    engine = Engine(
        "t", width=64, height=64
    )  # no device needed: spawn and the registry are CPU-side
    h = engine.spawn(
        Mesh(cube(1, 1, 1)), Material(BasicMaterial("#ff0000")), Transform(pos=(0, 0, 0))
    )
    return engine, h


def test_set_material_writes_the_registered_id():
    engine, h = _engine_with_box()
    green = BasicMaterial("#00ff00")
    engine.set_material(h, green)
    assert engine.store._components["Material"][h.index, 0] == engine._material_registry.register(
        green
    )


def test_set_material_takes_an_entity_index():
    engine, h = _engine_with_box()
    green = BasicMaterial("#00ff00")
    engine.set_material(h.index, green)
    reg_id = engine.store._components["Material"][h.index, 0]
    assert engine._material_registry.get(int(reg_id)) is green


def test_set_material_on_a_dead_entity_raises():
    engine, h = _engine_with_box()
    engine.destroy(np.array([h.index]))
    engine.commands.execute(engine.store)
    with pytest.raises(ValueError):
        engine.set_material(h, BasicMaterial("#00ff00"))
    with pytest.raises(ValueError):
        engine.set_material(h.index + 1, BasicMaterial("#00ff00"))  # never spawned
