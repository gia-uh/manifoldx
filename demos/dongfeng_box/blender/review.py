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
    import os
    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    hdri = os.path.join(bpy.utils.system_resource("DATAFILES", path="studiolights/world"), "interior.exr")
    env.image = bpy.data.images.load(hdri)
    nt.links.new(env.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = 1.0
    for loc, e in (((-3, -3, 4.5), 500), ((3.5, 2.5, 4), 300)):
        area = bpy.data.objects.new("a", bpy.data.lights.new("a", "AREA"))
        area.data.energy, area.data.size = e, 5
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
    "front": ((-6.5, -0.6, 1.1), 50),
    "rear34": ((5.0, 4.4, 2.2), 35),
}
for name, (loc, lens) in VIEWS.items():
    cam.location = loc
    cam.data.lens = lens
    cam.rotation_euler = (target - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{OUT}/{'eevee' if EEVEE else 'wb'}-{name}.png"
    bpy.ops.render.render(write_still=True)
print("RESULT views", len(VIEWS))
