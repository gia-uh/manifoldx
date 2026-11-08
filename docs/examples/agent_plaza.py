import math

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane
from PIL import Image

engine = mx.Engine("plaza", width=480, height=270)

# The only things that survive an edit, and all of it plain data.
engine.state.setdefault("benches", [[-4.0, -3.0, 0.0], [4.0, -5.0, 90.0]])
engine.state.setdefault("visits", 0)
engine.state.setdefault("cam", {"yaw": 40.0, "dist": 18.0, "height": 7.0})

FLOOR = Transform(rot=Transform.rotation(x=-math.pi / 2))
PAVING = StandardMaterial(color="#c8b89a", roughness=0.9)
STONE = StandardMaterial(color="#8a9bb0", roughness=0.6)
WOOD = StandardMaterial(color="#7a5a3c", roughness=0.7)
BENCH = cube(2.4, 0.45, 0.7)


def bench(x, z, yaw):
    """One bench at (x, z) metres, turned yaw degrees about y."""
    turn = Transform.rotation(y=math.radians(yaw))
    return engine.spawn(Mesh(BENCH), Material(WOOD), Transform(pos=(x, 0.22, z), rot=turn))


@engine.on("startup")
def build(payload):
    """Stand up the plaza from the state."""
    engine.spawn(Mesh(plane(40, 40)), Material(PAVING), FLOOR)
    engine.spawn(Mesh(cube(4, 1, 4)), Material(STONE), Transform(pos=(0, 0.5, 0)))
    for x, z, yaw in engine.state.benches:
        bench(x, z, yaw)
    engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.4, -1, -0.3)))
    engine.log(f"plaza built: {len(engine.state.benches)} benches, {engine.state.visits} visits")


@engine.on("add_bench")
def add_bench(payload: dict[str, float]):
    """Place a bench: {"x": metres, "z": metres, "yaw": degrees}."""
    x: float = payload["x"]
    z: float = payload["z"]
    yaw: float = payload.get("yaw", 0.0)
    placed: list[list[float]] = engine.state.benches
    placed.append([x, z, yaw])
    bench(x, z, yaw)
    engine.log(f"bench {len(placed)} at {x:.1f}, {z:.1f}")


@engine.on("visit")
def visit(payload: dict[str, str]):
    """A person crosses the plaza: {"name": "Ana"}."""
    name: str = payload["name"]
    n: int = engine.state.visits
    engine.state.visits = n + 1
    engine.log(f"{name} crossed: {n + 1} visits")


@engine.on("frame")
def look_around(payload):
    """Put the camera where the state says, every frame, so a patch moves the view."""
    yaw: float = engine.state.cam["yaw"]
    dist: float = engine.state.cam["dist"]
    height: float = engine.state.cam["height"]
    a = math.radians(yaw)
    engine.camera.position = np.array([math.sin(a) * dist, height, math.cos(a) * dist], np.float32)
    engine.camera.target = np.array([0.0, 1.0, 0.0], np.float32)


frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
