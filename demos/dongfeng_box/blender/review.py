"""Render review stills of box-blender.blend: blender -b box-blender.blend --python review.py -- OUT_DIR [eevee]

Workbench with material colours by default (shape and regions); 'eevee' gives a lit look-dev still.
"""

import sys
from math import radians

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1 :]
OUT = args[0]
EEVEE = "eevee" in args
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = 1200, 800
if EEVEE:
    scene.render.engine = "BLENDER_EEVEE"
    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.57, 0.6, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.5
    sun.rotation_euler = (radians(50), 0, radians(-35))
    scene.collection.objects.link(sun)
    for loc, e in (((-4, -4, 4), 1500), ((4, 3, 3), 900), ((0, 0, 6), 800)):
        area = bpy.data.objects.new("a", bpy.data.lights.new("a", "AREA"))
        area.data.energy, area.data.size = e, 4
        area.location = loc
        area.rotation_euler = (Vector((0, 0, 0.8)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(area)
    bpy.ops.mesh.primitive_plane_add(size=40)
    floor = bpy.context.active_object
    m = bpy.data.materials.new("floor")
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.7, 0.7, 0.72, 1)
    floor.data.materials.append(m)
    scene.view_settings.view_transform = "AgX"
else:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_cavity = True
    scene.display.shading.show_object_outline = True
    scene.display.shading.show_specular_highlight = True
    for m in bpy.data.materials:
        if m.node_tree and "Principled BSDF" in m.node_tree.nodes:
            m.diffuse_color = m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
target = Vector((0, 0, 0.75))
VIEWS = {
    "threequarter": ((-5.2, -4.6, 2.4), 35),
    "side": ((0, -9, 0.8), 35),
    "front": ((-9, 0, 0.9), 35),
    "rear34": ((5.0, 4.4, 2.2), 35),
}
for name, (loc, lens) in VIEWS.items():
    cam.location = loc
    cam.data.lens = lens
    cam.rotation_euler = (target - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{OUT}/{'eevee' if EEVEE else 'wb'}-{name}.png"
    bpy.ops.render.render(write_still=True)
print("RESULT views", len(VIEWS))
