"""The Dongfeng BOX modelled in Blender, headless: a spike to compare against the manifoldx loft in box.py.

Run: blender -b --python box_blender.py -- OUT_DIR
Writes OUT_DIR/box-blender.glb and OUT_DIR/box-blender.blend.

Axes: metres, X along the car with the front at -X, Z up, Y across (+Y is the car's right). The glTF exporter
turns Z up into Y up, which gives the game's axes (x along the car, front at -x, y up).

The body is a Catmull-Clark surface. Its shape is given as points that must lie on the surface (read off the
photos and the spec sheet), and the control cage is solved so that the limit surface passes through them.
Colour regions (glass, lamps, trims) are cut into the skin with exact booleans against prisms drawn in side,
front or top view, so every region has a crisp edge and no texture is needed.
"""

import sys
from math import cos, pi, radians, sin

import bmesh
import bpy
import numpy as np

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "/tmp"
MM = 0.001
AXLES = (-1.302, 1.358)  # wheel centres along X, wheelbase 2660 mm
TYRE_R = 0.3184  # 205/50 R17
TYRE_W = 0.205
RIM_R = 17 * 25.4 / 2 * MM
TRACK_HALF = 0.775  # tyre centre plane

# --- the body's shape: points on the surface, from shape.py (fitted values when BOX_SHAPE names a JSON) ---------

sys.path.insert(0, __import__("os").path.dirname(__file__))
import shape  # noqa: E402

SECTIONS, FACTORS = shape.load()


def targets():
    return shape.targets(SECTIONS, FACTORS)


def cage_mesh(points, name="BodyCage"):
    """A closed quad mesh: rings joined section to section, both ends capped with one n-gon."""
    S, R, _ = points.shape
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    vs = [[bm.verts.new(tuple(points[i, j])) for j in range(R)] for i in range(S)]
    for a, b in zip(vs, vs[1:]):
        for j in range(R):
            bm.faces.new((a[j], a[(j + 1) % R], b[(j + 1) % R], b[j]))
    bm.faces.new(vs[0])
    bm.faces.new(list(reversed(vs[-1])))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def limit_points(cage_obj, n):
    """Where the cage's vertices land on the limit surface: the first n vertices of a level-1 limit evaluation."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = cage_obj.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    return co.reshape(-1, 3)[:n]


def fit_cage(iterations=30, step=1.5):
    """Solve the control cage whose Catmull-Clark limit surface passes through the target points.

    The limit position of each vertex is a fixed weighted average of its neighbours, so the error can be pushed back
    onto the cage and the evaluation repeated; it converges because the averaging weights are positive.
    """
    T = targets()
    S, R, _ = T.shape
    flat_t = T.reshape(-1, 3)
    cage = flat_t.copy()
    obj = link(bpy.data.objects.new("Body", cage_mesh(T)))
    sub = obj.modifiers.new("fit", "SUBSURF")
    sub.levels = 1
    sub.use_limit_surface = True
    for it in range(iterations):
        obj.data.vertices.foreach_set("co", cage.ravel())
        obj.data.update()
        bpy.context.view_layer.update()
        lim = limit_points(obj, len(flat_t))
        err = flat_t - lim
        if it == 0:
            print(f"FIT start rms {np.sqrt((err ** 2).sum(1).mean()) * 1000:.1f} mm, max {np.abs(err).max() * 1000:.1f} mm")
            # The first evaluation must keep the cage's vertex order; check it before trusting the update.
            assert np.abs(err).max() < 0.25, f"limit vertices do not follow cage order: {np.abs(err).max()}"
        cage += step * err
        cage[np.abs(flat_t[:, 1]) < 1e-9, 1] = 0.0  # the centre line stays on the mirror plane
    print(f"FIT rms {np.sqrt((err ** 2).sum(1).mean()) * 1000:.1f} mm, max {np.abs(err).max() * 1000:.1f} mm")
    obj.data.vertices.foreach_set("co", cage.ravel())
    obj.data.update()
    obj.modifiers.remove(sub)
    final = obj.modifiers.new("subdiv", "SUBSURF")
    final.levels = final.render_levels = 2
    apply_all(obj)
    return obj


# --- materials ------------------------------------------------------------------------------------------------


def srgb(hex_):
    c = [int(hex_[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


def material(name, color, rough=0.5, metal=0.0, coat=0.0, emit=None, alpha=1.0, cull=True):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*srgb(color), 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Coat Roughness"].default_value = 0.04
    if emit:
        p.inputs["Emission Color"].default_value = (*srgb(emit[0]), 1)
        p.inputs["Emission Strength"].default_value = emit[1]
    if alpha < 1:
        p.inputs["Alpha"].default_value = alpha
        m.surface_render_method = "BLENDED"
    m.use_backface_culling = cull
    return m


MAT = {}


def materials():
    # Fitted against the photos' paint in CIELAB (match/render_matched.py, Khronos PBR Neutral, exposure +1.1).
    MAT["paint"] = material("Paint", "#96c0cf", rough=0.3, metal=0.15, coat=1.0)
    MAT["liner"] = material("ArchLiner", "#121315", rough=0.95)
    MAT["black"] = material("GlossBlack", "#060708", rough=0.1, coat=1.0)
    MAT["glass"] = material("Glass", "#20292f", rough=0.02, coat=1.0, alpha=0.42)
    MAT["trim"] = material("MatteTrim", "#18191b", rough=0.75)
    MAT["chrome"] = material("Chrome", "#dfe4e8", rough=0.06, metal=1.0)
    MAT["drl"] = material("DRL", "#f6f9ff", rough=0.2, emit=("#eef4ff", 6.0))
    MAT["lens"] = material("Lens", "#14171b", rough=0.04, metal=0.5, coat=1.0)
    MAT["tail"] = material("TailLED", "#ff2a1a", rough=0.2, emit=("#ff2410", 5.0))
    MAT["taillens"] = material("TailLens", "#3a0507", rough=0.05, coat=1.0)
    MAT["plate"] = material("Plate", "#dfe8ec", rough=0.6)
    MAT["cut"] = material("_cut", "#ff00ff")  # faces a cutter leaves behind, deleted after each boolean
    MAT["cabin"] = material("Cabin", "#1d1e21", rough=0.9)
    MAT["seat"] = material("Seat", "#2b2d31", rough=0.75)
    MAT["tyre"] = material("Tyre", "#141414", rough=0.88)
    MAT["rim"] = material("Rim", "#2a2c30", rough=0.45, metal=0.3)
    MAT["machined"] = material("Machined", "#c8cdd2", rough=0.18, metal=1.0)
    MAT["disc"] = material("Disc", "#6d7075", rough=0.45, metal=1.0)


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


def prism(name, outline, axis, lo, hi, mat=None):
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
    obj.data.materials.append(mat or MAT["cut"])
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
    boolean(body, cutter, "DIFFERENCE")
    boolean(inside, cutter, "INTERSECT")
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


def mirrored(points):
    """A right-hand (+Y) outline in front or rear view, and the same outline mirrored to the left."""
    return [points, [(-y, z) for y, z in reversed(points)]]


def band(curve, thickness):
    """A thin strip in a 2-D view: the polyline curve and the same curve shifted up by thickness."""
    return curve + [(a, b + thickness) for a, b in reversed(curve)]


def circle(cy, cz, r, n=24):
    return [(cy + r * cos(2 * pi * k / n), cz + r * sin(2 * pi * k / n)) for k in range(n)]


def rounded_rect(y0, z0, y1, z1, r, n=6):
    pts = []
    for cy, cz, a0 in ((y1 - r, z0 + r, -90), (y1 - r, z1 - r, 0), (y0 + r, z1 - r, 90), (y0 + r, z0 + r, 180)):
        pts += [(cy + r * cos(radians(a0 + 90 * k / n)), cz + r * sin(radians(a0 + 90 * k / n))) for k in range(n + 1)]
    return pts


# --- the body -------------------------------------------------------------------------------------------------


def body():
    obj = fit_cage()
    # Every material in one fixed order, so the boolean copies and the body agree on slot indices when joined.
    obj.data.materials.clear()
    for m in MAT.values():
        obj.data.materials.append(m)
    for p in obj.data.polygons:
        p.material_index = 0

    # Wheel arches: cylinders across the car. Their walls stay as the arch liners, in matte black.
    for x in AXLES:
        bpy.ops.mesh.primitive_cylinder_add(vertices=72, radius=TYRE_R + 0.05, depth=2.4,
                                            location=(x, 0, TYRE_R + 0.015), rotation=(pi / 2, 0, 0))
        cyl = bpy.context.active_object
        cyl.data.materials.append(MAT["liner"])
        boolean(obj, cyl, "DIFFERENCE")
        bpy.data.objects.remove(cyl)
    # Inner wheel-well walls, so the arch does not look into the cabin.
    for x in AXLES:
        well = prism("well", mm([(x * 1000 - 430, 270), (x * 1000 + 430, 270), (x * 1000 + 430, 760),
                                 (x * 1000 - 430, 760)]), "Y", -0.66, 0.66, MAT["liner"])
        boolean(obj, well, "UNION")
        bpy.data.objects.remove(well)

    side = lambda poly, mat, half=1.2: paint_region(obj, prism("s", mm(poly), "Y", -half, half), mat)
    front = lambda poly, mat, lo=-2.3, hi=-1.8: paint_region(obj, prism("f", mm(poly), "X", lo, hi), mat)
    rear = lambda poly, mat, lo=1.85, hi=2.3: paint_region(obj, prism("r", mm(poly), "X", lo, hi), mat)
    top = lambda poly, mat, lo, hi: paint_region(obj, prism("t", mm(poly), "Z", lo, hi), mat)

    # Greenhouse: gloss black from the belt up, the roof back in paint, then the glass. With BOX_GREENHOUSE the
    # black band and the side windows are the outlines traced off the side photo (match/greenhouse.py).
    gh_path = __import__("os").environ.get("BOX_GREENHOUSE")
    G = __import__("json").load(open(gh_path)) if gh_path else None
    if G:
        side([tuple(q) for q in G["dlo"]], MAT["black"])
    else:
        side([(-1060, 1100), (-330, 1600), (1500, 1600), (1960, 1200), (2060, 1010), (2060, 960), (1900, 1180),
              (1300, 1168), (-1000, 1112)], MAT["black"])
    rear([(-620, 1000), (620, 1000), (620, 1600), (-620, 1600)], MAT["black"], lo=1.6)
    top([(-330, -560), (1590, -520), (1590, 520), (-330, 560)], MAT["paint"], 1.49, 1.8)
    # Windscreen: between the cowl (where the hood ends) and the roof, the faces that look forward or up rather than
    # sideways. The A-pillar is where the skin turns to face the side, so the cut follows any fitted shape.
    glass = slot(obj, MAT["glass"])
    for p in obj.data.polygons:
        c, n = p.center, p.normal
        if FACTORS["hood_end"] * MM <= c.x <= -0.30 and c.z > 0.95 and abs(n.y) < 0.45:
            p.material_index = glass
    if G:
        for pane in G["windows"]:
            side([tuple(q) for q in pane], MAT["glass"])
    else:
        side([(-900, 1165), (-420, 1468), (60, 1500), (60, 1178)], MAT["glass"])
        side([(150, 1500), (1000, 1496), (1180, 1300), (1180, 1185), (150, 1180)], MAT["glass"])
        side([(1270, 1250), (1270, 1470), (1400, 1470), (1500, 1300)], MAT["glass"])
    rear([(-540, 1210), (540, 1210), (500, 1470), (-500, 1470)], MAT["glass"], lo=1.6)

    # Front: the LED bar across the nose bending down into C-shaped headlamps, a wide black grille, silver-framed
    # corner intakes, the plate and the badge.
    lamps_path = __import__("os").environ.get("BOX_LAMPS")
    if lamps_path:
        # Traced off the front photo where the LEDs are lit (match/lamps.py): the bar, then per side the eye in
        # LED white, its lens inset by one LED stroke, and the projector.
        L = __import__("json").load(open(lamps_path))
        bar = np.array(L["bar"])
        z0, z1, half = bar[:, 1].min(), bar[:, 1].max(), L["bar_half_length"]
        front([(-half, z0), (half, z0), (half, z1), (-half, z1)], MAT["drl"])
        for key, mat in (("eye", MAT["drl"]), ("eye_inner", MAT["lens"]), ("projector", MAT["drl"])):
            if L[key]:
                for poly in mirrored([tuple(q) for q in L[key]]):
                    front(poly, mat, hi=-1.70)
    else:
        ys = np.linspace(-865, 865, 61)
        front(band([(y, 930 - 115 * (abs(y) / 865) ** 4) for y in ys], 16), MAT["drl"])
        for lens in mirrored([(530, 905), (610, 912), (710, 905), (800, 890), (858, 862), (872, 815), (845, 778),
                              (765, 768), (680, 782), (600, 830)]):
            front(lens, MAT["lens"], hi=-1.75)
        for drl in mirrored(band([(575, 838), (650, 800), (760, 784), (838, 790), (866, 815)], 13)):
            front(drl, MAT["drl"], hi=-1.75)
        for lamp in (circle(770, 840, 28), circle(-770, 840, 28)):
            front(lamp, MAT["drl"], hi=-1.75)
    front([(-650, 290), (650, 290), (705, 425), (-705, 425)], MAT["trim"])
    for frame in mirrored([(600, 300), (805, 332), (845, 440), (640, 448)]):
        front(frame, MAT["chrome"], hi=-1.75)
    for hole in mirrored([(628, 318), (792, 344), (818, 425), (655, 430)]):
        front(hole, MAT["trim"], hi=-1.75)
    front([(-260, 470), (260, 470), (260, 590), (-260, 590)], MAT["plate"])
    front(circle(0, 985, 32), MAT["chrome"], hi=-1.85)

    # Rear: round-cornered lamps at the corners of the black band, a red LED ring, the bumper's black lip.
    for y0, y1 in ((540, 850), (-850, -540)):
        rear(rounded_rect(y0, 920, y1, 1100, 60), MAT["tail"], lo=1.7)
        rear(rounded_rect(y0 + 22, 942, y1 - 22, 1078, 42), MAT["taillens"], lo=1.7)
        rear(rounded_rect(y0 + 70, 990, y1 - 70, 1028, 14), MAT["plate"], lo=1.7)
    rear([(-820, 290), (820, 290), (820, 380), (-820, 380)], MAT["trim"])

    # Side: shut lines, handles and the black door strip with its chrome line.
    for seam in ([(-962, 360), (-957, 360), (-940, 1118), (-945, 1118)], [(300, 300), (305, 300), (305, 1150),
                 (300, 1150)], [(1185, 700), (1190, 700), (1305, 950), (1425, 1175), (1420, 1175), (1300, 950)]):
        side(seam, MAT["black"])
    for hx in (60, 1150):
        side([(hx, 915), (hx + 150, 915), (hx + 150, 937), (hx, 937)], MAT["chrome"])
    side([(-760, 340), (1000, 340), (960, 450), (-700, 450)], MAT["trim"])
    side([(-770, 325), (1010, 325), (1000, 340), (-760, 340)], MAT["chrome"])

    for p in obj.data.polygons:
        p.use_smooth = True
    obj.data.set_sharp_from_angle(angle=radians(40))
    return obj


def cabin(body_obj):
    """The inside: the body's skin without its glass, pulled in 25 mm and turned inside out, plus seats, dash
    and wheel, so the glass shows a cabin instead of the far side of the paint."""
    shell = body_obj.copy()
    shell.data = body_obj.data.copy()
    link(shell)
    glass = slot(shell, MAT["glass"])
    bm = bmesh.new()
    bm.from_mesh(shell.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index == glass], context="FACES")
    bm.normal_update()
    for v in bm.verts:
        v.co -= v.normal * 0.025
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    cab = slot(shell, MAT["cabin"])
    for f in bm.faces:
        f.material_index = cab
    bm.to_mesh(shell.data)
    bm.free()
    shell.name = "CabinShell"
    parts = [shell]

    def block(loc, size, mat, rot=(0, 0, 0), level=2):
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
        b = bpy.context.active_object
        b.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        b.data.materials.append(mat)
        sub = b.modifiers.new("s", "SUBSURF")
        sub.levels = level
        bev = b.modifiers.new("b", "BEVEL")
        bev.width, bev.segments = 0.01, 2
        b.modifiers.move(1, 0)
        apply_all(b)
        parts.append(b)

    block((-0.75, 0, 0.98), (0.42, 1.44, 0.22), MAT["cabin"])  # dashboard
    for y in (-0.37, 0.37):
        block((-0.12, y, 0.56), (0.52, 0.50, 0.14), MAT["seat"])
        block((0.16, y, 0.90), (0.14, 0.48, 0.62), MAT["seat"], rot=(0, radians(-14), 0))
        block((0.24, y, 1.28), (0.10, 0.26, 0.16), MAT["seat"])
    block((0.78, 0, 0.56), (0.50, 1.26, 0.14), MAT["seat"])
    block((1.04, 0, 0.90), (0.14, 1.26, 0.60), MAT["seat"], rot=(0, radians(-12), 0))
    bpy.ops.mesh.primitive_torus_add(major_radius=0.18, minor_radius=0.018, location=(-0.56, -0.37, 1.02),
                                     rotation=(0, radians(-65), 0))
    wheel_ = bpy.context.active_object
    wheel_.data.materials.append(MAT["cabin"])
    parts.append(wheel_)
    return parts


# --- details: mirrors -----------------------------------------------------------------------------------------


def mirrors():
    out = []
    for s in (1, -1):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.86, s * 1.0, 1.17))
        m = bpy.context.active_object
        m.scale = (0.17, 0.24, 0.14)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        m.data.materials.append(MAT["paint"])
        sub = m.modifiers.new("s", "SUBSURF")
        sub.levels = 3
        apply_all(m)
        # The back of the housing (facing the driver, +X) and its base are black.
        m.data.materials.append(MAT["black"])
        for p in m.data.polygons:
            if p.center.z < -0.035:  # local coordinates: the housing is centred on its origin
                p.material_index = 1
            p.use_smooth = True
        bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.84, s * 0.91, 1.13))
        arm = bpy.context.active_object
        arm.scale = (0.09, 0.12, 0.035)
        arm.data.materials.append(MAT["black"])
        out += [m, arm]
    return out


# --- wheels ---------------------------------------------------------------------------------------------------


def revolve(name, profile, mat, steps=96):
    """Revolve a (radius, y) polyline about the Y axis with a Screw modifier."""
    bm = bmesh.new()
    vs = [bm.verts.new((0, y, r)) for r, y in profile]
    for a, b in zip(vs, vs[1:]):
        bm.edges.new((a, b))
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = link(bpy.data.objects.new(name, mesh))
    obj.data.materials.append(mat)
    screw = obj.modifiers.new("screw", "SCREW")
    screw.axis = "Y"
    screw.steps = screw.render_steps = steps
    screw.use_smooth_shade = True
    screw.use_merge_vertices = True
    apply_all(obj)
    return obj


def spokes(count=5, face=-0.094, twist=12.0):
    """Five broad spokes, twisted and dished: each a tapered outline in the wheel plane, extruded 28 mm."""
    bm = bmesh.new()
    rs = np.linspace(0.058, RIM_R - 0.03, 9)
    for k in range(count):
        base = 2 * pi * k / count
        left, right = [], []
        for r in rs:
            t = (r - rs[0]) / (rs[-1] - rs[0])
            a = base + radians(twist) * t
            half = (0.046 - 0.008 * t + 0.03 * t ** 3) / r  # narrow in the middle, flaring into the rim
            y = face + 0.032 * (1 - t) ** 1.5  # dish: the hub sits deeper than the rim
            left.append((r * sin(a - half), y, r * cos(a - half)))
            right.append((r * sin(a + half), y, r * cos(a + half)))
        front_ = [bm.verts.new(p) for p in left + list(reversed(right))]
        f = bm.faces.new(front_)
        ext = bmesh.ops.extrude_face_region(bm, geom=[f])
        bmesh.ops.translate(bm, vec=(0, 0.028, 0),
                            verts=[v for v in ext["geom"] if isinstance(v, bmesh.types.BMVert)])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("Spokes")
    bm.to_mesh(mesh)
    bm.free()
    obj = link(bpy.data.objects.new("Spokes", mesh))
    obj.data.materials.append(MAT["rim"])
    obj.data.materials.append(MAT["machined"])
    bev = obj.modifiers.new("bevel", "BEVEL")
    bev.width, bev.segments, bev.limit_method = 0.004, 2, "ANGLE"
    apply_all(obj)
    return obj


def wheel(name, x, side_):
    r, h = TYRE_R, TYRE_W / 2
    parts = [
        revolve("Tyre", [(RIM_R + 0.004, -h * 0.93), (RIM_R + 0.03, -h * 1.03), (r - 0.04, -h * 1.02),
                         (r - 0.013, -h * 0.92), (r - 0.002, -h * 0.70), (r, -h * 0.35), (r, h * 0.35),
                         (r - 0.002, h * 0.70), (r - 0.013, h * 0.92), (r - 0.04, h * 1.02), (RIM_R + 0.03, h * 1.03),
                         (RIM_R + 0.004, h * 0.93)], MAT["tyre"]),
        # Rim: machined outer face, a dark barrel and a dark back plate so the openings read as depth.
        revolve("Lip", [(RIM_R - 0.035, -h * 0.90), (RIM_R + 0.004, -h * 0.93), (RIM_R + 0.004, -h * 0.80),
                        (RIM_R - 0.012, -h * 0.80), (RIM_R - 0.012, h * 0.80)], MAT["machined"]),
        revolve("Back", [(0.0, 0.03), (RIM_R - 0.012, 0.03)], MAT["liner"], steps=48),
        revolve("Disc", [(0.06, -0.005), (0.145, -0.005), (0.145, 0.015), (0.06, 0.015)], MAT["disc"], steps=48),
        revolve("Cap", [(0.0, -0.075), (0.04, -0.073), (0.052, -0.064), (0.058, -0.05)], MAT["machined"], steps=32),
        spokes(),
    ]
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.0, 0.0, 0.12))
    caliper = bpy.context.active_object
    caliper.scale = (0.11, 0.05, 0.05)
    caliper.data.materials.append(MAT["rim"])
    parts.append(caliper)
    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    w = bpy.context.active_object
    w.name = name
    w.data.set_sharp_from_angle(angle=radians(35))
    # Built facing -Y (the left side); the right-hand wheels are turned round so the spokes face out.
    w.location = (x, side_ * TRACK_HALF, TYRE_R)
    if side_ > 0:
        w.rotation_euler = (0, 0, pi)
    return w


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    materials()
    b = body()
    inside = cabin(b)
    for m in mirrors() + inside:
        m.parent = b
    wheels = [wheel(f"Wheel_{n}", x, s) for n, (x, s) in
              {"FL": (AXLES[0], -1), "FR": (AXLES[0], 1), "RL": (AXLES[1], -1), "RR": (AXLES[1], 1)}.items()]
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            used = {obj.data.materials[p.material_index].name for p in obj.data.polygons}
            assert "_cut" not in used, obj.name
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.data.objects if o.type == "MESH")
    bpy.ops.wm.save_as_mainfile(filepath=f"{OUT}/box-blender.blend")
    bpy.ops.export_scene.gltf(filepath=f"{OUT}/box-blender.glb", export_format="GLB", export_apply=True,
                              export_yup=True)
    print(f"RESULT tris={tris} objects={len(bpy.data.objects)} wheels={len(wheels)}")


main()
