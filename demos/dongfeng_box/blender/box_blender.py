"""The Dongfeng BOX modelled in Blender, headless: a spike to compare against the manifoldx loft in box.py.

Run: blender -b --python box_blender.py -- OUT_DIR
Writes OUT_DIR/box-blender.glb and OUT_DIR/box-blender.blend.

Axes: metres, X along the car with the front at -X, Z up, Y across. The glTF exporter turns Z up into Y up,
which gives the game's axes (x along the car, front at -x, y up).
"""

import sys
from math import pi, radians

import bmesh
import bpy
import numpy as np
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "/tmp"
MM = 0.001
AXLES = (-1.302, 1.358)  # wheel centres along X, wheelbase 2660 mm
TYRE_R = 0.3184  # 205/50 R17
TYRE_W = 0.205
TRACK_HALF = 0.775  # tyre centre plane

# --- the control cage ---------------------------------------------------------------------------------------
# One row per station, front to back, in mm: x, top of the section, bottom, half-width at the shoulder,
# belt height, roof half-width. The cage is the control net of a subdivision surface, so it sits a little
# outside the final skin and its rows are where the surface is pulled, not points on it.
STATIONS = [
    # x      top   bottom  half  belt  roof
    (-2030, 700, 330, 560, 680, 360),
    (-2010, 900, 260, 760, 880, 560),
    (-1940, 985, 240, 860, 960, 700),
    (-1750, 1040, 235, 905, 1015, 790),
    (-1450, 1080, 240, 915, 1055, 800),
    (-1150, 1125, 235, 915, 1100, 790),
    (-980, 1165, 230, 915, 1130, 770),
    (-650, 1360, 225, 915, 1140, 700),
    (-300, 1545, 225, 915, 1145, 660),
    (150, 1585, 225, 915, 1155, 650),
    (700, 1585, 225, 915, 1170, 640),
    (1150, 1565, 230, 910, 1185, 620),
    (1500, 1530, 235, 900, 1200, 590),
    (1700, 1420, 250, 880, 1190, 570),
    (1880, 1200, 270, 840, 1130, 600),
    (1990, 960, 290, 780, 920, 600),
    (2030, 700, 320, 640, 680, 470),
]


def ring(top, bottom, half, belt, roof):
    """Half a section, from the bottom centre round the right side (+Y) to the roof centre, as (y, z) in mm."""
    body = belt - bottom
    upper = max(top - belt, 20)
    return [
        (0, bottom),
        (0.78 * half, bottom),
        (0.97 * half, bottom + 0.12 * body),
        (half, bottom + 0.55 * body),
        (0.985 * half, belt - 0.04 * body),
        (0.95 * half, belt),
        (roof, belt + 0.80 * upper),
        (0.72 * roof, top - 0.01 * upper),
        (0, top),
    ]


def cage():
    """The right half of the body as a quad cage, open along the centre plane, with both ends capped."""
    mesh = bpy.data.meshes.new("BodyCage")
    bm = bmesh.new()
    rows = []
    for x, top, bottom, half, belt, roof in STATIONS:
        rows.append([bm.verts.new((x * MM, y * MM, z * MM)) for y, z in ring(top, bottom, half, belt, roof)])
    for a, b in zip(rows, rows[1:]):
        for j in range(len(a) - 1):
            bm.faces.new((a[j], a[j + 1], b[j + 1], b[j]))
    bm.faces.new(rows[0])
    bm.faces.new(list(reversed(rows[-1])))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


# --- materials ------------------------------------------------------------------------------------------------


def material(name, color, rough=0.5, metal=0.0, coat=0.0, emit=None, alpha=1.0, trans=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Coat Roughness"].default_value = 0.05
    p.inputs["Transmission Weight"].default_value = trans
    if emit:
        p.inputs["Emission Color"].default_value = (*emit[0], 1)
        p.inputs["Emission Strength"].default_value = emit[1]
    if alpha < 1:
        p.inputs["Alpha"].default_value = alpha
    return m


def srgb(hex_):
    c = [int(hex_[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


MAT = {}


def materials():
    MAT["paint"] = material("Paint", srgb("#8fb8d6"), rough=0.35, metal=0.35, coat=1.0)
    MAT["liner"] = material("ArchLiner", srgb("#141518"), rough=0.9)
    MAT["black"] = material("GlossBlack", srgb("#08090b"), rough=0.12, coat=1.0)
    MAT["glass"] = material("Glass", srgb("#1a2228"), rough=0.03, metal=0.0, coat=1.0)
    MAT["trim"] = material("MatteTrim", srgb("#1d1f22"), rough=0.7)
    MAT["chrome"] = material("Chrome", srgb("#d8dde2"), rough=0.08, metal=1.0)
    MAT["drl"] = material("DRL", srgb("#f4f8ff"), rough=0.2, emit=(srgb("#eef4ff"), 4.0))
    MAT["lamp"] = material("HeadLamp", srgb("#1b1e22"), rough=0.05, metal=0.6)
    MAT["tail"] = material("TailLamp", srgb("#5a0608"), rough=0.1, emit=(srgb("#ff1a1a"), 1.5))
    MAT["tyre"] = material("Tyre", srgb("#151515"), rough=0.85)
    MAT["rim"] = material("Rim", srgb("#2a2d31"), rough=0.3, metal=0.8)
    MAT["rimlip"] = material("RimLip", srgb("#c9ced3"), rough=0.15, metal=1.0)
    MAT["cut"] = material("_cut", (1, 0, 1))  # faces a cutter leaves behind, deleted after each boolean
    MAT["plate"] = material("Plate", srgb("#e8ecef"), rough=0.6)


# --- helpers --------------------------------------------------------------------------------------------------


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def apply_all(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    for mod in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.select_set(False)


def prism(name, outline, axis, lo, hi, mat):
    """A closed prism: a 2-D outline extruded from lo to hi along X, Y or Z (metres).

    For axis 'Y' the outline is (x, z), for 'X' it is (y, z), for 'Z' it is (x, y).
    """
    bm = bmesh.new()
    place = {
        "Y": lambda a, b, t: (a, t, b),
        "X": lambda a, b, t: (t, a, b),
        "Z": lambda a, b, t: (a, b, t),
    }[axis]
    low = [bm.verts.new(place(a, b, lo)) for a, b in outline]
    high = [bm.verts.new(place(a, b, hi)) for a, b in outline]
    bm.faces.new(low)
    bm.faces.new(list(reversed(high)))
    n = len(outline)
    for i in range(n):
        bm.faces.new((low[i], low[(i + 1) % n], high[(i + 1) % n], high[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = link(bpy.data.objects.new(name, mesh))
    obj.data.materials.append(mat)
    return obj


def boolean(target, cutter, op):
    mod = target.modifiers.new("bool", "BOOLEAN")
    mod.operation = op
    mod.solver = "EXACT"
    mod.object = cutter
    mod.material_mode = "TRANSFER"
    apply_all(target)


def slot(obj, mat):
    """The index of mat in obj's material slots, adding it if missing."""
    names = [m.name for m in obj.data.materials]
    if mat.name not in names:
        obj.data.materials.append(mat)
        names.append(mat.name)
    return names.index(mat.name)


def paint_region(body, cutter, mat):
    """Repaint the part of body's skin inside cutter with mat, along the cutter's exact intersection curve.

    The body is split by two booleans (difference and intersect) on copies, the walls the cutter leaves are
    dropped, and the two skins are joined back with their seam welded.
    """
    inside = body.copy()
    inside.data = body.data.copy()
    link(inside)
    n0 = len(body.data.polygons)
    boolean(body, cutter, "DIFFERENCE")
    boolean(inside, cutter, "INTERSECT")
    print(f"REGION {cutter.name} {mat.name}: body {n0} -> diff {len(body.data.polygons)}, inter {len(inside.data.polygons)}")
    keep = slot(inside, mat)
    cut_b, cut_i = slot(body, MAT["cut"]), slot(inside, MAT["cut"])
    for obj, cut in ((body, cut_b), (inside, cut_i)):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index == cut], context="FACES")
        if obj is inside:
            for f in bm.faces:
                f.material_index = keep
        bm.to_mesh(obj.data)
        bm.free()
    bpy.ops.object.select_all(action="DESELECT")
    inside.select_set(True)
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    bm = bmesh.new()
    bm.from_mesh(body.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0005)
    bm.to_mesh(body.data)
    bm.free()
    bpy.data.objects.remove(cutter)


def mm(points):
    return [(a * MM, b * MM) for a, b in points]


# --- the body -------------------------------------------------------------------------------------------------


def body():
    obj = link(bpy.data.objects.new("Body", cage()))
    # Every material in one fixed order, so the boolean copies and the body agree on slot indices when joined.
    for m in MAT.values():
        obj.data.materials.append(m)
    mirror = obj.modifiers.new("mirror", "MIRROR")
    mirror.use_axis = (False, True, False)
    mirror.use_clip = True
    mirror.use_mirror_merge = True
    sub = obj.modifiers.new("subdiv", "SUBSURF")
    sub.levels = sub.render_levels = 3
    apply_all(obj)

    # Wheel arches: cylinders across the car. Their walls stay as the arch liners, in matte black.
    for x in AXLES:
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=TYRE_R + 0.045, depth=2.4,
                                            location=(x, 0, TYRE_R + 0.01), rotation=(pi / 2, 0, 0))
        cyl = bpy.context.active_object
        cyl.data.materials.append(MAT["liner"])
        boolean(obj, cyl, "DIFFERENCE")
        bpy.data.objects.remove(cyl)
    # Inner wheel-well walls: an inset box across the arch so the arch does not see through the car.
    for x in AXLES:
        cut = prism("well", mm([(x * 1000 - 420, 260), (x * 1000 + 420, 260), (x * 1000 + 420, 760),
                                (x * 1000 - 420, 760)]), "Y", -0.66, 0.66, MAT["liner"])
        # keep only the walls: the box's ends become the well's inner sides
        boolean(obj, cut, "UNION")
        bpy.data.objects.remove(cut)

    # Material regions, all from exact intersections with prisms seen from the side (Y), front or back (X).
    greenhouse = mm([(-1000, 1120), (-280, 1560), (1460, 1545), (1620, 1450), (1760, 1175), (1500, 1180),
                     (300, 1150), (-950, 1110)])
    paint_region(obj, prism("gh", greenhouse, "Y", -1.2, 1.2, MAT["cut"]), MAT["black"])
    roof = mm([(-300, 1490), (1440, 1490), (1440, 1700), (-300, 1700)])
    roof_c = prism("roof", roof, "Y", -0.60, 0.60, MAT["cut"])
    paint_region(obj, roof_c, MAT["paint"])
    windshield = mm([(-990, 1150), (-320, 1520), (-320, 1700), (-1100, 1700)])
    paint_region(obj, prism("ws", windshield, "Y", -0.72, 0.72, MAT["cut"]), MAT["glass"])
    for poly in (
        [(-860, 1175), (-380, 1450), (40, 1470), (40, 1185)],  # front door window
        [(150, 1470), (980, 1465), (1180, 1290), (1180, 1200), (150, 1190)],  # rear door window
        [(1260, 1240), (1260, 1440), (1380, 1440), (1440, 1290)],  # quarter glass
    ):
        paint_region(obj, prism("win", mm(poly), "Y", -1.2, 1.2, MAT["cut"]), MAT["glass"])
    rear = mm([(-560, 1200), (560, 1200), (500, 1470), (-500, 1470)])
    paint_region(obj, prism("rw", rear, "X", 1.55, 2.2, MAT["cut"]), MAT["glass"])

    # Front: LED bar along the hood's leading edge, headlamps below it, a dark lower intake and the plate.
    bar = mm([(-760, 925), (760, 925), (760, 950), (-760, 950)])
    paint_region(obj, prism("drl", bar, "X", -2.3, -1.6, MAT["cut"]), MAT["drl"])
    for s in (1, -1):
        lamp = mm([(s * 520, 760), (s * 800, 790), (s * 830, 900), (s * 470, 900)])
        paint_region(obj, prism("lamp", lamp, "X", -2.3, -1.7, MAT["cut"]), MAT["lamp"])
        led = mm([(s * 480, 885), (s * 820, 885), (s * 830, 900), (s * 470, 900)])
        paint_region(obj, prism("led", led, "X", -2.3, -1.7, MAT["cut"]), MAT["drl"])
    intake = mm([(-600, 300), (600, 300), (650, 470), (-650, 470)])
    paint_region(obj, prism("intake", intake, "X", -2.3, -1.85, MAT["cut"]), MAT["trim"])
    plate = mm([(-260, 480), (260, 480), (260, 600), (-260, 600)])
    paint_region(obj, prism("plate", plate, "X", -2.3, -1.85, MAT["cut"]), MAT["plate"])

    # Rear: a red light bar across the tailgate and a black diffuser.
    tail = mm([(-830, 1000), (830, 1000), (800, 1080), (-800, 1080)])
    paint_region(obj, prism("tail", tail, "X", 1.8, 2.3, MAT["cut"]), MAT["tail"])
    diff = mm([(-800, 300), (800, 300), (800, 420), (-800, 420)])
    paint_region(obj, prism("diff", diff, "X", 1.9, 2.3, MAT["cut"]), MAT["trim"])

    # Side: the black door strip with its chrome lower line.
    strip = mm([(-760, 330), (1000, 330), (950, 400), (-700, 400)])
    paint_region(obj, prism("strip", strip, "Y", -1.2, 1.2, MAT["cut"]), MAT["trim"])
    line = mm([(-740, 315), (980, 315), (1000, 330), (-760, 330)])
    paint_region(obj, prism("line", line, "Y", -1.2, 1.2, MAT["cut"]), MAT["chrome"])

    # Door shut lines (5 mm, gloss black) and the two handles.
    for seam in (
        [(-1000, 360), (-995, 360), (-955, 1112), (-960, 1112)],  # front door, front edge
        [(95, 330), (100, 330), (100, 1180), (95, 1180)],  # between the doors
        [(1060, 360), (1065, 360), (1245, 1195), (1240, 1195)],  # rear door, back edge
    ):
        paint_region(obj, prism("seam", mm(seam), "Y", -1.2, 1.2, MAT["cut"]), MAT["black"])
    for hx in (-130, 1050):
        handle = mm([(hx, 1015), (hx + 150, 1015), (hx + 150, 1040), (hx, 1040)])
        paint_region(obj, prism("handle", handle, "Y", -1.2, 1.2, MAT["cut"]), MAT["chrome"])

    for p in obj.data.polygons:
        p.use_smooth = True
    obj.data.set_sharp_from_angle(angle=radians(40))
    return obj


# --- details: mirrors and handles -----------------------------------------------------------------------------


def mirrors():
    out = []
    for s in (1, -1):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.84, s * 0.99, 1.17))
        m = bpy.context.active_object
        m.scale = (0.11, 0.17, 0.09)
        m.data.materials.append(MAT["paint"])
        sub = m.modifiers.new("s", "SUBSURF")
        sub.levels = 2
        apply_all(m)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.80, s * 0.92, 1.14))
        arm = bpy.context.active_object
        arm.scale = (0.08, 0.08, 0.04)
        arm.data.materials.append(MAT["black"])
        out += [m, arm]
    return out


# --- wheels ---------------------------------------------------------------------------------------------------


def tyre():
    """A tyre revolved from its cross-section with a Screw modifier: tread, rounded shoulders, sidewalls."""
    r, w, rim = TYRE_R, TYRE_W, 17 * 25.4 / 2 * MM
    half = w / 2
    prof = [(rim, -half * 0.92), (rim + 0.02, -half), (r - 0.03, -half), (r - 0.008, -half * 0.85),
            (r, -half * 0.55), (r, half * 0.55), (r - 0.008, half * 0.85), (r - 0.03, half), (rim + 0.02, half),
            (rim, half * 0.92)]
    bm = bmesh.new()
    vs = [bm.verts.new((0, z, rr)) for rr, z in prof]
    for a, b in zip(vs, vs[1:]):
        bm.edges.new((a, b))
    mesh = bpy.data.meshes.new("Tyre")
    bm.to_mesh(mesh)
    bm.free()
    obj = link(bpy.data.objects.new("Tyre", mesh))
    obj.data.materials.append(MAT["tyre"])
    screw = obj.modifiers.new("screw", "SCREW")
    screw.axis = "Y"
    screw.steps = screw.render_steps = 64
    screw.use_smooth_shade = True
    apply_all(obj)
    return obj


def rim():
    """A rim: a dished barrel, a hub and five twisted spokes repeated by an Array modifier around an empty."""
    rr = 17 * 25.4 / 2 * MM
    parts = []
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=rr, depth=TYRE_W * 0.9, rotation=(pi / 2, 0, 0))
    barrel = bpy.context.active_object
    barrel.data.materials.append(MAT["rim"])
    parts.append(barrel)
    bpy.ops.mesh.primitive_torus_add(major_radius=rr - 0.012, minor_radius=0.012, major_segments=64,
                                     location=(0, -TYRE_W * 0.45, 0), rotation=(pi / 2, 0, 0))
    lip = bpy.context.active_object
    lip.data.materials.append(MAT["rimlip"])
    parts.append(lip)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -TYRE_W * 0.47, rr * 0.55))
    spoke = bpy.context.active_object
    spoke.scale = (0.06, 0.03, rr * 0.85)
    spoke.rotation_euler = (0, radians(12), 0)
    spoke.data.materials.append(MAT["rim"])
    bev = spoke.modifiers.new("bevel", "BEVEL")
    bev.width, bev.segments = 0.008, 3
    apply_all(spoke)
    # The array copies around the object's origin with the pivot's transform relative to the spoke, so the
    # spoke's own scale and rotation are baked in first and its origin sits on the axle.
    spoke.select_set(True)
    bpy.context.view_layer.objects.active = spoke
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    spoke.select_set(False)
    pivot = link(bpy.data.objects.new("pivot", None))
    pivot.rotation_euler = (0, radians(72), 0)
    arr = spoke.modifiers.new("array", "ARRAY")
    arr.count = 5
    arr.use_relative_offset = False
    arr.use_object_offset = True
    arr.offset_object = pivot
    apply_all(spoke)
    bpy.data.objects.remove(pivot)
    parts.append(spoke)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.07, depth=0.04, location=(0, -TYRE_W * 0.47, 0),
                                        rotation=(pi / 2, 0, 0))
    hub = bpy.context.active_object
    hub.data.materials.append(MAT["rimlip"])
    parts.append(hub)
    return parts


def wheel(name, x, side):
    parts = [tyre()] + rim()
    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    w = bpy.context.active_object
    w.name = name
    for p in w.data.polygons:
        p.use_smooth = True
    w.data.set_sharp_from_angle(angle=radians(35))
    # Built facing -Y (the left side); the right-hand wheels are turned round so the spokes face out.
    w.location = (x, side * TRACK_HALF, TYRE_R)
    if side > 0:
        w.rotation_euler = (0, 0, pi)
    return w


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    materials()
    b = body()
    for m in mirrors():
        m.parent = b
    wheels = [wheel(f"Wheel_{n}", x, s) for n, (x, s) in
              {"FL": (AXLES[0], -1), "FR": (AXLES[0], 1), "RL": (AXLES[1], -1), "RR": (AXLES[1], 1)}.items()]
    for p in b.data.materials:
        pass
    # Drop the unused cutter material.
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            names = [m.name for m in obj.data.materials]
            assert "_cut" not in [obj.data.materials[p.material_index].name for p in obj.data.polygons], obj.name
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.data.objects if o.type == "MESH")
    bpy.ops.wm.save_as_mainfile(filepath=f"{OUT}/box-blender.blend")
    bpy.ops.export_scene.gltf(filepath=f"{OUT}/box-blender.glb", export_format="GLB", export_apply=True,
                              export_yup=True)
    print(f"RESULT tris={tris} objects={len(bpy.data.objects)} wheels={len(wheels)}")


main()
