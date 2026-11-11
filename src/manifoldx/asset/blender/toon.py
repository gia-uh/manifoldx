"""blender -b --python toon.py -- MODEL.glb ALBEDO.png|- OUT_DIR: toon-shaded stills of a textured model.

Two-band cel shading (a constant colour ramp on the light) times the albedo, emitted so the bands stay flat,
and a black outline from an inverted hull (solidify with flipped normals, back faces culled). Stills: front
three-quarter, front, side, back, and a three-quarter from above. The model stands on y = 0 and faces +z in
manifoldx axes; glTF import maps (x, y, z) to Blender (x, -z, y), so it faces -y in Blender.
"""
import math
import sys

import bpy
from mathutils import Vector

model, albedo, out = sys.argv[sys.argv.index("--") + 1:][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = scene.render.resolution_y = 1024
scene.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("w")
scene.world = world
scene.render.film_transparent = True

bpy.ops.import_scene.gltf(filepath=model)
parts = [o for o in bpy.context.selected_objects if o.type == "MESH"]
if not parts:
    parts = [o for o in bpy.data.objects if o.type == "MESH"]

toon = bpy.data.materials.new("toon")
toon.use_nodes = True
nt = toon.node_tree
nt.nodes.clear()
tex = nt.nodes.new("ShaderNodeTexImage")
tex.interpolation = "Closest"
if albedo != "-":
    tex.image = bpy.data.images.load(albedo)
diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
s2r = nt.nodes.new("ShaderNodeShaderToRGB")
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.interpolation = "CONSTANT"
ramp.color_ramp.elements[0].position = 0.0
ramp.color_ramp.elements[0].color = (0.62, 0.62, 0.68, 1)
ramp.color_ramp.elements[1].position = 0.12
ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
mix = nt.nodes.new("ShaderNodeMix")
mix.data_type = "RGBA"
mix.blend_type = "MULTIPLY"
mix.inputs[0].default_value = 1.0
emit = nt.nodes.new("ShaderNodeEmission")
outn = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(diff.outputs[0], s2r.inputs[0])
nt.links.new(s2r.outputs[0], ramp.inputs[0])
if albedo != "-":
    nt.links.new(tex.outputs[0], mix.inputs[6])
else:
    mix.inputs[6].default_value = (0.78, 0.76, 0.72, 1)
nt.links.new(ramp.outputs[0], mix.inputs[7])
nt.links.new(mix.outputs[2], emit.inputs[0])
nt.links.new(emit.outputs[0], outn.inputs[0])

line = bpy.data.materials.new("outline")
line.use_nodes = True
line.use_backface_culling = True
ln = line.node_tree
ln.nodes.clear()
le = ln.nodes.new("ShaderNodeEmission")
le.inputs[0].default_value = (0.05, 0.03, 0.02, 1)
lo = ln.nodes.new("ShaderNodeOutputMaterial")
ln.links.new(le.outputs[0], lo.inputs[0])

for o in parts:
    o.data.materials.clear()
    o.data.materials.append(toon)
    o.data.materials.append(line)
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    bpy.ops.object.shade_smooth()
    hull = o.modifiers.new("hull", "SOLIDIFY")
    hull.thickness = -0.004
    hull.use_flip_normals = True
    hull.material_offset = 1
    hull.use_rim = False

sun = bpy.data.lights.new("sun", "SUN")
sun.energy = 3.0
so = bpy.data.objects.new("sun", sun)
scene.collection.objects.link(so)
so.rotation_euler = (math.radians(50), 0, math.radians(-35))

low = Vector((1e9,) * 3)
high = Vector((-1e9,) * 3)
for o in parts:
    for corner in o.bound_box:
        moved = o.matrix_world @ Vector(corner)
        low = Vector(map(min, low, moved))
        high = Vector(map(max, high, moved))
centre, size = (low + high) / 2, max(high - low)
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = size * 1.15
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
SHOTS = {"three-quarter": (-35, 8), "front": (0, 0), "side": (-90, 0), "back": (180, 0), "above": (-30, 30)}
for name, (yaw, pitch) in SHOTS.items():
    a, p = math.radians(yaw), math.radians(pitch)
    direction = Vector((math.sin(a) * math.cos(p), -math.cos(a) * math.cos(p), math.sin(p)))
    cam.location = centre + direction * size * 3
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    so.rotation_euler = (math.radians(50), 0, a + math.radians(-35))
    scene.render.filepath = f"{out}/{name}.png"
    bpy.ops.render.render(write_still=True)
