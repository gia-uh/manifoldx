"""Render the model with each photo's measured camera, for the colour comparison in register.py.

Run: blender -b box-blender.blend --python render_matched.py -- REPORT.json OUT_DIR

The cameras come from register.py's report (fixed or registered). Light: Blender's bundled studio HDRI, the
nearest thing at hand to the catalogue's grey studio; the colour numbers are only as fair as that match.
"""

import json
import os
import sys
from math import radians

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(__file__))
import calib  # noqa: E402

report, out = sys.argv[sys.argv.index("--") + 1 :][:2]
os.makedirs(out, exist_ok=True)
views = json.load(open(report))["views"]
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 1400, 933
# Overrides for the colour search: VIEW names a view transform, PAINT is "#rrggbb,metallic,roughness".
# Khronos PBR Neutral keeps base colours (AgX desaturated the paint's highlights from chroma 24 to 17), and +1.1
# exposure brings the paint's lightness to the photos'; both were measured, see the colour table in the report.
scene.view_settings.view_transform = os.environ.get("VIEW", "Khronos PBR Neutral")
scene.view_settings.exposure = float(os.environ.get("EXPOSURE", "1.1"))
if os.environ.get("PAINT"):
    hex_, metal, rough = os.environ["PAINT"].split(",")
    lin = [int(hex_[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in lin]
    bsdf = bpy.data.materials["Paint"].node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*lin, 1)
    bsdf.inputs["Metallic"].default_value = float(metal)
    bsdf.inputs["Roughness"].default_value = float(rough)
world = bpy.data.worlds.new("W")
scene.world = world
nt = world.node_tree
env = nt.nodes.new("ShaderNodeTexEnvironment")
env.image = bpy.data.images.load(os.path.join(bpy.utils.system_resource("DATAFILES", path="studiolights/world"), "studio.exr"))
nt.links.new(env.outputs["Color"], nt.nodes["Background"].inputs["Color"])
bpy.ops.mesh.primitive_plane_add(size=60)
floor = bpy.context.active_object
m = bpy.data.materials.new("floor")
m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.75, 0.75, 0.76, 1)
floor.data.materials.append(m)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.sensor_fit = "HORIZONTAL"
cam.data.sensor_width = 36.0
for photo, v in views.items():
    cx, cy, cz, yaw, pitch, roll, f = v["params"]
    right, up, fwd = calib.basis(yaw, pitch, roll)
    rot = Matrix([list(right), list(up), list(-np.array(fwd))]).transposed()  # camera looks down its -Z
    cam.matrix_world = Matrix.Translation((cx, cy, cz)) @ rot.to_4x4()
    cam.data.lens = f * 36.0 / calib.W
    scene.render.filepath = f"{out}/matched-{photo[:2]}.png"
    bpy.ops.render.render(write_still=True)
print("RESULT matched", len(views))
