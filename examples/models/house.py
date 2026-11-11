"""A small two-room house, built from a floor plan, four facades and a close-up of its door.

The plan gives the footprint and the two rooms; the front facade gives the door, its windows and the roof. The
rooms are white holes in the black walls, so the build stands a box around each room instead of cutting the plan
with a boolean, which leaves open edges. Its drawings are committed beside it, so build and lint run offline:

    uv run python examples/models/house.py
"""

import numpy as np

from manifoldx.asset import Asset
from manifoldx.modeling import primitives, trace
from manifoldx.modeling.lint import Finding

_PLAN = (
    "A flat 2D floor plan of a small two-room house, seen straight from above, orthographic, no perspective. "
    "On a pure white page, draw thick solid black (0,0,0) walls: one rectangular outer wall and one straight "
    "black wall across the middle that divides the inside into two rooms. In the middle of the front (bottom) "
    "outer wall, mark the front door with a solid red (220,40,40) gap through the wall. Everything else is pure "
    "white. No floor texture, furniture, labels, text, dimensions, arrows, shadows or outlines; only flat white, "
    "black and red."
)

_FACADE = (
    "The {side} facade of the small two-room house in the reference floor plan, as a flat 2D elevation seen "
    "straight on, orthographic, no perspective. Flat colours only, no shading, gradients or texture: the walls "
    "are solid orange (255,140,0); a solid yellow (255,220,0) roof band runs across the top above the walls, a "
    "little wider than them; {openings}; everything else is pure white. Sharp straight edges, no outlines, no "
    "text, no furniture, no trees, no ground line, no sky."
)

_OPENINGS = {
    "front": ("a solid red (220,40,40) front door stands on the ground in the middle of the wall, and two solid "
              "blue (0,110,255) square windows sit one on each side of it"),
    "back": "two solid blue (0,110,255) square windows sit on the wall, one on each side of the middle",
    "left": ("one solid blue (0,110,255) square window sits in the middle of one plain orange wall: no door, "
             "no interior wall, no black lines, and do not copy the plan's layout into this elevation"),
    "right": "one solid blue (0,110,255) square window sits in the middle of the wall",
}

_DOOR = (
    "A close-up of the front door of the house in the reference picture, straight on, orthographic: one tall "
    "solid red (220,40,40) rectangle with a simple dark grey frame and a small dark grey handle, on a pure white "
    "background. Flat colours, no shading, no text."
)


def _bounds(mask):
    """The pixel box ``(x0, y0, x1, y1)`` of a non-empty mask, x1 and y1 exclusive."""
    rows, cols = np.nonzero(np.asarray(mask, bool))
    if not len(rows):
        raise ValueError("an empty mask has no bounds")
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _main(mask):
    """The box of a mask's largest piece, so one stray antialiased pixel cannot place a part."""
    labels, count = trace.components(np.asarray(mask, bool))
    if not count:
        return None
    sizes = [(labels == label).sum() for label in range(1, count + 1)]
    return _bounds(labels == 1 + int(np.argmax(sizes)))


def _rooms(walls):
    """The boxes of the plan's enclosed white rooms, largest first; the page around the house is not a room."""
    labels, count = trace.components(~np.asarray(walls, bool))
    outside = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    rooms = [_bounds(labels == label) for label in range(1, count + 1)
             if label not in outside and (labels == label).sum() >= 0.005 * labels.size]
    return sorted(rooms, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)


def _slab(x0, x1, y0, y1, z0, z1):
    """A box between two corners on each axis."""
    return primitives.box(x1 - x0, y1 - y0, z1 - z0).translate([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])


class House(Asset):
    height = 2.75            # walls plus roof, in metres
    wall_height = 2.6
    model = "google/gemini-2.5-flash-image"

    def draw(self, c):
        c.image("plan", _PLAN)
        for side in ("front", "left", "back", "right"):
            c.image(f"facade-{side}", _FACADE.format(side=side, openings=_OPENINGS[side]), refs=["plan"])
        c.image("door", _DOOR, refs=["facade-front"])

    # --- build -----------------------------------------------------------------------------------------------

    def build(self, c):
        scene = c.scene()
        facade = c.load("facade-front")
        image_h, image_w = facade.shape[:2]
        masks = {colour: c.mask("facade-front", colour) for colour in ("orange", "yellow", "red", "blue")}
        wall = _main(masks["orange"])
        if wall is None:
            raise ValueError("the front facade shows no orange wall to measure")
        scale = self.wall_height / (wall[3] - wall[1])

        def x_of(col):
            return float(c.metres(col - (wall[0] + wall[2]) / 2, scale))

        def y_of(row):
            return float(c.metres(wall[3] - row, scale))

        plan = c.mask("plan", "black") | c.mask("plan", "red")   # the red door gap, closed for tracing
        outer, rooms = _bounds(plan), _rooms(plan)
        if not rooms:
            raise ValueError("the plan has no enclosed room to trace walls around")
        plan_scale = (x_of(wall[2]) - x_of(wall[0])) / (outer[2] - outer[0])

        def px(col):
            return float(c.metres(col - (outer[0] + outer[2]) / 2, plan_scale))

        def pz(row):
            return float(c.metres(row - (outer[1] + outer[3]) / 2, plan_scale))

        x0, x1, z0, z1 = px(outer[0]), px(outer[2]), pz(outer[1]), pz(outer[3])
        left = max(0.08, px(min(room[0] for room in rooms)) - x0)
        right = max(0.08, x1 - px(max(room[2] for room in rooms)))
        back = max(0.08, pz(min(room[1] for room in rooms)) - z0)
        front = max(0.08, z1 - pz(max(room[3] for room in rooms)))

        # Four outer walls, then a wall in every gap between two rooms: the traced rooms lay out the plan.
        walls = [_slab(x0, x1, 0.0, self.wall_height, z0, z0 + back),
                 _slab(x0, x0 + left, 0.0, self.wall_height, z0 + back, z1 - front),
                 _slab(x1 - right, x1, 0.0, self.wall_height, z0 + back, z1 - front)]
        for i, a in enumerate(rooms):
            for b in rooms[i + 1:]:
                if a[2] <= b[0]:                                  # a left of b
                    lo, hi = a[2], b[0]
                    zlo, zhi = pz(max(a[1], b[1])), pz(min(a[3], b[3]))
                    if px(hi) - px(lo) > 0.02 and zhi - zlo > 0.02:
                        walls.append(_slab(px(lo), px(hi), 0.0, self.wall_height, zlo, zhi))
                elif b[2] <= a[0]:                                # b left of a
                    lo, hi = b[2], a[0]
                    zlo, zhi = pz(max(a[1], b[1])), pz(min(a[3], b[3]))
                    if px(hi) - px(lo) > 0.02 and zhi - zlo > 0.02:
                        walls.append(_slab(px(lo), px(hi), 0.0, self.wall_height, zlo, zhi))
                elif a[3] <= b[1]:                                # a behind b
                    lo, hi = pz(a[3]), pz(b[1])
                    xlo, xhi = px(max(a[0], b[0])), px(min(a[2], b[2]))
                    if hi - lo > 0.02 and xhi - xlo > 0.02:
                        walls.append(_slab(xlo, xhi, 0.0, self.wall_height, lo, hi))
                elif b[3] <= a[1]:                                # b behind a
                    lo, hi = pz(b[3]), pz(a[1])
                    xlo, xhi = px(max(a[0], b[0])), px(min(a[2], b[2]))
                    if hi - lo > 0.02 and xhi - xlo > 0.02:
                        walls.append(_slab(xlo, xhi, 0.0, self.wall_height, lo, hi))

        # The front wall is split around the door the facade shows, with a lintel above it.
        red = _main(masks["red"])
        door_x0, door_x1 = (x_of(red[0]), x_of(red[2])) if red else (-0.45, 0.45)
        door_y0, door_y1 = (y_of(red[3]), y_of(red[1])) if red else (0.0, 2.0)
        door_x0, door_x1 = max(x0, door_x0), min(x1, door_x1)
        door_y0, door_y1 = max(0.0, door_y0), min(self.wall_height, door_y1)
        front_z = (z1 - front, z1)
        if door_x0 > x0 + 0.02:
            walls.append(_slab(x0, door_x0, 0.0, self.wall_height, *front_z))
        if door_x1 < x1 - 0.02:
            walls.append(_slab(door_x1, x1, 0.0, self.wall_height, *front_z))
        if door_y1 < self.wall_height - 0.02:
            walls.append(_slab(door_x0, door_x1, door_y1, self.wall_height, *front_z))

        for index, mesh in enumerate(walls, start=1):
            scene.add(f"wall{index}", mesh)
        scene.add("door", _slab(door_x0 + 0.01, door_x1 - 0.01, door_y0 + 0.01, door_y1 - 0.01,
                                z1 - front - 0.01, z1 + 0.01))

        # Windows sit on the facade where the blue regions are, as shallow boxes in the front wall.
        labels, count = trace.components(masks["blue"])
        window = 0
        for label in range(1, count + 1):
            mask = labels == label
            if mask.sum() < 0.001 * mask.size:
                continue
            box = _bounds(mask)
            width, tall = (box[2] - box[0]) * scale, (box[3] - box[1]) * scale
            if width < 0.25 or tall < 0.25:
                continue
            window += 1
            scene.add(f"window{window}", _slab(x_of(box[0]), x_of(box[2]), y_of(box[3]), y_of(box[1]),
                                               z1 - 0.05, z1 + 0.01))

        # A flat roof over the whole footprint, its width and thickness read from the yellow band on the facade.
        roof = _main(masks["yellow"])
        roof_x0 = min(x_of(roof[0]), x0 - 0.1) if roof else x0 - 0.1
        roof_x1 = max(x_of(roof[2]), x1 + 0.1) if roof else x1 + 0.1
        roof_tall = max(0.12, (roof[3] - roof[1]) * scale) if roof else 0.15
        scene.add("roof", _slab(roof_x0, roof_x1, self.wall_height - 0.02,
                                self.wall_height - 0.02 + roof_tall, z0 - 0.15, z1 + 0.15))

        shown = {"wall": masks["orange"], "roof": masks["yellow"], "door": masks["red"],
                 "window": masks["blue"]}
        c.expect("front", {name: mask for name, mask in shown.items() if mask.any()},
                 frame=(x_of(0), x_of(image_w), y_of(image_h), y_of(0)))
        return scene

    # --- check -----------------------------------------------------------------------------------------------

    def check(self, c, scene):
        """The roof must cover every wall, and the door must stand inside the walls."""
        walls = [mesh for name, mesh in scene.meshes.items() if name.startswith("wall")]
        corners = np.vstack([np.asarray(mesh.positions, float) for mesh in walls])
        low, high = corners.min(axis=0), corners.max(axis=0)
        roof = np.asarray(scene["roof"].positions, float)
        roof_low, roof_high = roof.min(axis=0), roof.max(axis=0)
        door = np.asarray(scene["door"].positions, float)
        centre = (door.min(axis=0) + door.max(axis=0)) / 2
        findings = []
        if (roof_low[[0, 2]] > low[[0, 2]]).any() or (roof_high[[0, 2]] < high[[0, 2]]).any():
            findings.append(Finding("A-COVER", "error", part="roof",
                                    message="the roof does not cover every wall",
                                    hint="widen the roof or redraw the facade"))
        if not (low[0] <= centre[0] <= high[0] and low[2] <= centre[2] <= high[2]
                and 0.0 <= centre[1] <= self.wall_height):
            findings.append(Finding("A-DOOR", "error", part="door",
                                    message="the door stands outside the walls",
                                    hint="check where the front facade shows the door"))
        return findings


if __name__ == "__main__":
    result = House().run(stages=("build", "lint"))
    print(result.report.text() or "no findings")
    triangles = sum(len(mesh.faces) for mesh in result.scene.meshes.values())
    print(f"\n{len(result.scene)} parts, {triangles} triangles")
