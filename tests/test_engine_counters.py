"""The engine counts its work for manifoldx bench: draws, bundles, material reads, matrices, bytes, entities."""

import re
from pathlib import Path

import numpy as np
import pytest

from manifoldx import counters
from manifoldx.bench import micro
from manifoldx.engine import _TransformProxy
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import StandardMaterial, cube

SRC = Path(__file__).parents[1] / "src" / "manifoldx"


@pytest.fixture
def scene():
    engine = micro.headless_engine(128)
    pos = np.random.default_rng(0).uniform(-5, 5, (50, 3)).astype(np.float32)
    cubes = engine.spawn(Mesh(cube(0.5, 0.5, 0.5)), Material(StandardMaterial("#88aaff")), Transform(pos=pos), n=50)
    engine.camera.fit(20)
    micro.start(engine)
    counters.start(None)
    yield engine, cubes
    counters.stop()
    engine.close()


def frame(engine):
    engine._draw_frame()
    engine._render_canvas.draw()
    return counters.take().counts


def test_every_upload_goes_through_the_counter():
    direct = [f"{p.relative_to(SRC)}:{n}" for p in SRC.rglob("*.py") if p.name != "counters.py"
              for n, line in enumerate(p.read_text().splitlines(), 1)
              if re.search(r"\bqueue\.write_buffer\(", line)]
    assert direct == []


def test_a_frame_counts_draws_uploads_and_matrices(scene):
    engine, _ = scene
    c = frame(engine)
    assert c["draws"] >= 1 and c["material_reads"] >= 1 and c["bytes_uploaded"] > 0
    assert "spawned" not in c  # spawned before the run started


def test_a_still_frame_recomputes_no_matrix_and_records_no_bundle(scene):
    engine, _ = scene
    frame(engine)
    frame(engine)
    c = frame(engine)
    assert "matrices" not in c and "bundles_recorded" not in c and "material_uploads" not in c
    assert c["draws"] >= 1 and c["material_reads"] >= 1


def test_moving_one_cube_recomputes_one_matrix(scene):
    engine, cubes = scene
    frame(engine)
    frame(engine)
    _TransformProxy(engine.store, int(cubes.indices[3]), engine).pos = (1.0, 2.0, 3.0)
    assert frame(engine).get("matrices") == 1


def test_spawn_and_destroy_are_counted(scene):
    engine, _ = scene
    more = engine.spawn(Mesh(cube(0.5, 0.5, 0.5)), Material(StandardMaterial("#ffffff")),
                        Transform(pos=np.zeros((10, 3), np.float32)), n=10)
    engine.destroy(more.indices[:4])  # goes at the frame's command flush
    c = frame(engine)
    assert c["spawned"] == 10 and c["destroyed"] == 4
