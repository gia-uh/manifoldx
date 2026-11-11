"""blender -b --python real.py -- MODEL.glb ALBEDO.png|- OUT_DIR [GROUND]: photo-like stills of a model.

Cycles on the CPU, a physical sky with a low sun, and a glossy sea plane at the model's base (or a matte studio
floor with ``GROUND=floor``). Same five shots as toon.py.
"""
import math
import sys

import bpy
from mathutils import Vector

model, albedo, out, *rest = sys.argv[sys.argv.index("--") + 1:]
ground = rest[0] if rest else "sea"
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 40
scene.cycles.use_denoising = True
scene.render.resolution_x, scene.render.resolution_y = 1024, 1280
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.view_settings.exposure = -0.6

world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
background = nodes["Background"]
sky = nodes.new("ShaderNodeTexSky")
sky.sun_elevation = math.radians(22)
sky.sun_rotation = math.radians(200)
world.node_tree.links.new(sky.outputs[0], background.inputs[0])
background.inputs[1].default_value = 0.18

if ground == "floor":  # a studio: neutral grey light, no tone mapping that shifts colours
    world.node_tree.links.remove(background.inputs[0].links[0])
    background.inputs[0].default_value = (0.8, 0.8, 0.8, 1)
    background.inputs[1].default_value = 0.9
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0

bpy.ops.import_scene.gltf(filepath=model)
parts = [o for o in bpy.context.selected_objects if o.type == "MESH"]
if not parts:
    parts = [o for o in bpy.data.objects if o.type == "MESH"]
mat = bpy.data.materials.new("albedo")
mat.use_nodes = True
tree = mat.node_tree
bsdf = tree.nodes["Principled BSDF"]
bsdf.inputs["Roughness"].default_value = 0.35
if albedo != "-":
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(albedo)
    tree.links.new(tex.outputs[0], bsdf.inputs["Base Color"])
else:
    bsdf.inputs["Base Color"].default_value = (0.78, 0.76, 0.72, 1)
for o in parts:
    o.data.materials.clear()
    o.data.materials.append(mat)
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
bpy.ops.object.shade_smooth()

low = Vector((1e9,) * 3)
high = Vector((-1e9,) * 3)
for o in parts:
    for corner in o.bound_box:
        moved = o.matrix_world @ Vector(corner)
        low = Vector(map(min, low, moved))
        high = Vector(map(max, high, moved))
centre, size = (low + high) / 2, max(high - low)
bpy.ops.mesh.primitive_plane_add(size=size * 40,
                                 location=(centre.x, centre.y, low.z + (size * 0.01 if ground == "sea" else 0.0)))
floor = bpy.context.object
sm = bpy.data.materials.new("ground")
sm.use_nodes = True
sb = sm.node_tree.nodes["Principled BSDF"]
sb.inputs["Base Color"].default_value = (0.005, 0.03, 0.05, 1)
sb.inputs["Roughness"].default_value = 0.08
if ground == "floor":
    sb.inputs["Base Color"].default_value = (0.55, 0.52, 0.48, 1)
    sb.inputs["Roughness"].default_value = 0.7
floor.data.materials.append(sm)

sun = bpy.data.lights.new("sun", "SUN")
sun.energy = 2.2
sun.angle = math.radians(1.0)
so = bpy.data.objects.new("sun", sun)
scene.collection.objects.link(so)
so.rotation_euler = (math.radians(62), 0, math.radians(200))

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 50
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam_data.clip_end = size * 100
SHOTS = {"three-quarter": (-35, 6), "front": (0, 3), "side": (-90, 3), "back": (180, 3), "above": (-30, 30)}
for name, (yaw, pitch) in SHOTS.items():
    a, p = math.radians(yaw), math.radians(pitch)
    direction = Vector((math.sin(a) * math.cos(p), -math.cos(a) * math.cos(p), math.sin(p)))
    cam.location = centre + direction * size * 2.6
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{out}/{name}.png"
    bpy.ops.render.render(write_still=True)
