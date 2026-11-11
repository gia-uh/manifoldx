import importlib.util
import sys
from pathlib import Path

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Mesh as GeoMesh
from manifoldx.resources import DirectionalLight, StandardMaterial
from PIL import Image

ROOT = next(path for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (path / "pyproject.toml").exists())


def example(path, name):
    """Load an example from its own file, so its cache stays beside it."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hydrant = example("examples/models/hydrant.py", "hydrant_example").Hydrant()
armchair = example("examples/models/armchair.py", "armchair_example").Armchair()
house = example("examples/models/house.py", "house_example").House()

# Where each object stands, and the colour of its clay until it is painted.
placed = [(hydrant, (-2.0, 0.0, 2.8), "#b03028"),
          (armchair, (-0.4, 0.0, -1.2), "#c4962c"),
          (house, (5.2, 0.0, -7.6), "#ded2b8")]

engine = mx.Engine("plaza", width=640, height=360)
engine.background_color = (0.62, 0.72, 0.84)
engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.2, direction=(-0.45, -0.85, -0.3)))
engine.enable_shadows(resolution=1024, bias=0.004, pcf_radius=2)

engine.spawn(Mesh(GeoMesh.plane(width=40, depth=40).to_geometry()),
             Material(StandardMaterial(color="#b5a98f", roughness=0.95)), Transform())

counts = []
for asset, at, colour in placed:
    scene = asset.run(stages=("build",)).scene      # the lint was read in this chapter
    counts.append((len(scene), asset.name))
    material = StandardMaterial(color=colour, roughness=0.85)
    for mesh in scene.meshes.values():
        engine.spawn(Mesh(mesh.to_geometry()), Material(material), Transform(pos=at))

engine.camera.set_pose(position=(0.5, 1.6, 6.4), target=(1.0, 0.9, -3.0))
frame = engine.render_frame(supersample=1)
print(counts)
Image.fromarray(frame)
