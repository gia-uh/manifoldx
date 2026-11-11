"""An object of named parts, built from its blueprints by choosing a builder and its options for each part.

Layer 2 of the asset tools, for the common case the prototype covered: one object that the image model drew in
the same code colours in the front, back and left views. A subclass says what the object is (``desc`` and
``style``), how tall it is, which part centres the views (``core``), which parts the views are scaled by
(``align``), and then one line per part in :meth:`Parts.parts`, each naming a builder:

    class Hydrant(Parts):
        height, core = 0.8, "barrel"

        def parts(self, p):
            p.add("barrel", "orange", "the round barrel")
            p.add("side", "blue", "each side hose outlet", build=two_view(power=6))

The build traces every blueprint, aligns the views to the same rows, and hands each part's masks and the frame
to its builder. A builder is a small frozen dataclass that makes one thin call into
:mod:`manifoldx.modeling.views`; the options they share (``views``, ``attach``, ``keep``, ``grid``, ``mirror``)
are handled once, here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np

from manifoldx.asset.base import Asset
from manifoldx.asset.context import CODE_COLOURS
from manifoldx.image import nearest_aspect, pad_to_aspect, read_png, resize
from manifoldx.modeling import paint as painting
from manifoldx.modeling.lint import Finding, closest_offset, gap
from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling.skeleton import fit_radii, skin
from manifoldx.modeling.trace import classify, components, contours, iou, open_mask, simplify
from manifoldx.modeling.views import (
    extrude,
    fill_rows,
    lathe_solid,
    reflect_x,
    row_centres,
    swept_tube,
    template,
    two_view_solid,
    view_atlas,
)

# The rows an aligned view spans: the scale the prototype's drawings were made at.
N = 480

# The views a blueprint may give, and how the view drawing asks for each one.
VIEWS = ("front", "back", "left")
_HOW = {
    "front": "seen exactly from the FRONT, facing the viewer straight on, perfectly symmetric left to right",
    "back": "seen exactly from BEHIND, perfectly symmetric left to right",
    "left": "seen exactly from its LEFT SIDE, in strict profile, its front facing the LEFT edge of the image",
}

# The prototype's prompts, kept verbatim: they were tuned on five objects.
_REF_PROMPT = ("Reference for a 3D model: {desc}, the whole object, front three-quarter view from "
               "the left. {style} Plain pure white background, no ground shadow, no text.")
_VIEW_PROMPT = ("The images show {desc}. Draw the same object, with the same style, colours "
                "and proportions, {how}. Orthographic view as in a model sheet for 3D modelling, no perspective, no "
                "three-quarter angle. Draw ONE single view of ONE object: not a sheet of several views. {note} "
                "The whole object visible and centred, filling about 85% of the image height. Plain "
                "pure white background, no ground shadow, no text.")
_BP_PROMPT = ("Image 1 is a drawing of an object. Make a BLUEPRINT of exactly this drawing for "
              "building a 3D model: keep the same view and the same outline of every part, pixel for pixel, and paint "
              "each part as one flat silhouette in its code colour, with hard edges, no outlines, no shading, no "
              "gradients, no details, no lines, no text:\n{code_list}\n{note}\n"
              "A part in front of another covers it, as in the drawing. Output one image, 1:1.")

# The style a Parts subclass uses when it names none: the hydrant's, the prototype's cartoon default.
DEFAULT_STYLE = "Clean cel-shaded cartoon style with flat colours and thin dark outlines, even light."

# How a paint call names the view it is painting, and the two finishes the prototype asked for.
_VIEW_NAME = {"front": "front view", "back": "back view", "left": "left side view", "right": "right side view"}
_FLAT = ("Flat colours: no shading, no highlights, no shadows, no outlines round the silhouette, no text.")
_REAL = ("Real materials as in the photos, but under even, flat light: no cast shadows, no sky reflections, "
         "no text.")
_PAINT_PROMPT = (
    "Image 1 is a flat-colour map of the {name} of a 3D model of the object in images 2 and 3 ({desc}): each "
    "colour is one part. {codes}, white = background.\n\nRepaint image 1 as the colour TEXTURE of that object "
    "seen from the {name}, as in images 2 and 3: {details}. Keep the canvas, the position, the size and the "
    "outline of every coloured region of image 1 exactly, pixel for pixel, because the result is wrapped back "
    "onto the 3D model; the parts' code colours do not appear in the result. {finish} Keep the background pure "
    "white. Output one image with the same aspect ratio as image 1.")
# An answer whose outline moved or shrank is asked again, at most this many times.
_PAINT_TRIES = 3
_PAINT_KEPT = 0.85

# A tube's defaults, from the prototype: a third of two_view's rings, half its segments.
_TUBE_SAMPLES = 10
_TUBE_SEGMENTS = 8

# What a dropped piece is measured against: one full N by N view, so the prototype's 0.001 is a thousandth.


# --- what a builder makes ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Built:
    """One builder's answer: a mesh per instance, the rows it filled from the other view, the pieces it dropped.

    ``dropped`` holds the area of each piece ``keep`` left out, as a fraction of N by N, so the A-STRAY finding
    can say how big what it threw away was.
    """

    meshes: tuple[Mesh, ...] = ()
    filled: int = 0
    dropped: tuple[float, ...] = ()
    skeleton: tuple | None = None                 # a skeleton's (names, joints, bones, radii), for inspect


@dataclass(frozen=True)
class Frame:
    """The metre frame every aligned view of one asset shares.

    ``rows`` rows span ``top`` metres; ``scale`` is metres per mask pixel; ``x`` and ``z`` are the core part's
    centre column in the front and the left view, which makes those two columns the world's x = 0 and z = 0.
    """

    rows: int
    scale: float
    top: float
    x: float
    z: float
    n: int = N

    def x_range(self, width: int) -> tuple[float, float]:
        """The metres the front view's columns cover, with the core's axis at x = 0."""
        return (-self.x * self.scale, (width - self.x) * self.scale)

    def z_range(self, width: int) -> tuple[float, float]:
        """The metres the left view's columns cover; its first column is the object's front, z_max."""
        return ((self.z - width) * self.scale, self.z * self.scale)


# --- the builders -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Builder:
    """What every builder shares: which views to trust, a part to attach to, what to keep, repeats and mirrors.

    ``views`` says which drawings say something about this part; the back view is often drawn mirrored, so a
    part whose back disagrees leaves it out. ``attach`` names a part this one must touch: after the build it is
    moved the shortest way until it does, and how far it moved is reported. ``keep`` drops stray pieces:
    ``"all"`` keeps them, ``"largest"`` keeps only the biggest, and a number keeps the pieces of at least that
    fraction of an N by N drawing (0.001 by default, the prototype's threshold). ``grid`` pairs the instances a
    part shows two from the front and two from the side (four legs). ``mirror`` builds the instances seen on the
    near side of the core and reflects them across x = 0, for a part whose side view shows only the near one.
    """

    views: tuple[str, ...] = field(default=VIEWS, kw_only=True)
    attach: str | None = field(default=None, kw_only=True)
    keep: str | float = field(default=0.001, kw_only=True)
    grid: bool = field(default=False, kw_only=True)
    mirror: bool = field(default=False, kw_only=True)

    requires: ClassVar[tuple[str, ...]] = ()
    bare_name: ClassVar[bool] = False

    def __post_init__(self) -> None:
        views = tuple(self.views)
        unknown = [view for view in views if view not in VIEWS]
        if unknown:
            raise ValueError(f"unknown view {unknown[0]!r}; choose from {', '.join(VIEWS)}")
        if not views:
            raise ValueError("a builder needs at least one view")
        missing = [view for view in self.requires if view not in views]
        if missing:
            raise ValueError(f"{type(self).__name__} needs the {missing[0]} view, which is not in views={views}")
        if self.attach is not None and not isinstance(self.attach, str):
            raise ValueError("attach must be the name of a part")
        if not isinstance(self.keep, str) and not isinstance(self.keep, (int, float)):
            raise ValueError("keep must be 'all', 'largest' or a fraction of a drawing")
        if isinstance(self.keep, str) and self.keep not in ("all", "largest"):
            raise ValueError("keep must be 'all', 'largest' or a fraction of a drawing")
        if isinstance(self.keep, (int, float)) and not 0.0 <= float(self.keep) <= 1.0:
            raise ValueError("keep as a fraction must be between 0 and 1")
        object.__setattr__(self, "views", views)

    def build(self, masks: dict[str, np.ndarray], frame: Frame, anchor=None) -> Built:
        """The part's meshes from its aligned masks; each builder says what it needs from them."""
        raise NotImplementedError

    # --- what the builders share -------------------------------------------------------------------------

    def _pairs(self, masks, frame: Frame):
        """One mask pair per instance, the rows filled from the other view, and the pieces keep dropped."""
        front, side = masks["front"], masks["left"]
        fronts, dropped = _pieces(front, self.keep, frame)
        if self.grid:
            sides, more = _pieces(side, self.keep, frame)
            dropped += more
        else:
            sides = [side]
        if not fronts or not sides:
            return [], 0, dropped
        if self.grid and len(fronts) == 2 * len(sides) and len(sides) > 1:
            # The front view also shows the rear instances between the front ones: the outer pair goes with the
            # side instance nearest the front (leftmost in the side view), the next pair with the next one.
            fronts = sorted(fronts, key=lambda m: -abs(_mean_column(m) - frame.x))
            sides = sorted(sides, key=_mean_column)
            pairs = [(fronts[i], sides[i // 2]) for i in range(len(fronts))]
        else:
            pairs = [(comp, piece) for comp in fronts for piece in sides]
        fill = getattr(self, "fill", "narrow")
        reach = int(0.12 * frame.n)
        out, filled = [], 0
        for comp, piece in pairs:
            if fill == "off":
                near, far = comp, piece
            else:
                near = fill_rows(comp, piece, reach, fill)
                far = fill_rows(piece, comp, reach, fill)
            rows = int((near & ~comp).any(axis=1).sum() + (far & ~piece).any(axis=1).sum())
            filled = max(filled, rows)
            out.append((near, far))
        return out, filled, dropped

    def _finish(self, meshes, filled=0, dropped=()) -> Built:
        meshes = list(meshes)
        if self.mirror and meshes:
            meshes = _mirrored(meshes)
        return Built(meshes=tuple(meshes), filled=int(filled), dropped=tuple(dropped))


@dataclass(frozen=True)
class two_view(Builder):
    """A rounded or boxy part seen whole from the front and the side: a head, a body, a seat, a pedestal.

    ``power`` sets how square the sections are (2 is an ellipse, 6 a rounded box); ``rings`` and ``segments``
    the mesh's density; ``fill`` says how a row only one view draws is completed: ``"narrow"`` (taper it as the
    other view narrows), ``"copy"`` (the other view's span exactly) or ``"off"``.
    """

    requires: ClassVar[tuple[str, ...]] = ("front", "left")
    power: float = 2.0
    rings: int = 30
    segments: int = 16
    fill: str = "narrow"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fill not in ("narrow", "copy", "off"):
            raise ValueError("fill must be 'narrow', 'copy' or 'off'")
        if float(self.power) <= 0.0:
            raise ValueError("power must be positive")
        if int(self.rings) < 2 or int(self.segments) < 3:
            raise ValueError("two_view needs rings >= 2 and segments >= 3")

    def build(self, masks, frame, anchor=None) -> Built:
        pairs, filled, dropped = self._pairs(masks, frame)
        meshes = [two_view_solid(a, b, (0.0, frame.top), n=self.rings, segments=self.segments,
                                 power=self.power, x_range=frame.x_range(a.shape[1]),
                                 z_range=frame.z_range(b.shape[1]))
                  for a, b in pairs]
        return self._finish(meshes, filled, dropped)


@dataclass(frozen=True)
class tube(Builder):
    """A limb or frame that reaches across and forward at once: an arm, an armrest on its post, an arch.

    ``anchor`` names the part whose centre orients the path (the core by default); ``radius`` is the two views'
    smaller half-width (``"min"``), their mean (``"mean"``) or a number of metres; ``samples`` is how many rings
    the path gets; ``section="drape"`` hangs cloth under the limb, as tall as both views draw it and only the
    limb's thickness across (a robed arm), where the default ``"round"`` would swell it into a cone.
    """

    requires: ClassVar[tuple[str, ...]] = ("front", "left")
    anchor: str | None = None
    radius: str | float = "min"
    samples: int | None = None
    section: str = "round"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.anchor is not None and not isinstance(self.anchor, str):
            raise ValueError("anchor must be the name of a part")
        if isinstance(self.radius, str) and self.radius not in ("min", "mean"):
            raise ValueError("radius must be 'min', 'mean' or a number of metres")
        if not isinstance(self.radius, str) and float(self.radius) <= 0.0:
            raise ValueError("radius must be positive")
        if self.samples is not None and int(self.samples) < 2:
            raise ValueError("samples must be at least 2")
        if self.section not in ("round", "drape"):
            raise ValueError("section must be 'round' or 'drape'")

    def build(self, masks, frame, anchor=None) -> Built:
        pairs, filled, dropped = self._pairs(masks, frame)
        samples = _TUBE_SAMPLES if self.samples is None else int(self.samples)
        origin = np.zeros(3) if anchor is None else np.asarray(anchor, float)
        meshes = [swept_tube(a, b, (0.0, frame.top), samples=samples, segments=_TUBE_SEGMENTS,
                             anchor=tuple(origin), radius=self.radius, section=self.section,
                             x_range=frame.x_range(a.shape[1]), z_range=frame.z_range(b.shape[1]))
                  for a, b in pairs]
        return self._finish(meshes, filled, dropped)


@dataclass(frozen=True)
class skeleton(Builder):
    """One connected solid grown from a tree of joints and cut to the drawings: a figure, an animal, a plant.

    ``joints`` maps a name to ``(x, y, z)`` in the asset's metres (x right, y up from the ground, z toward the
    front), or ``(x, y, z, radius)`` to fix a radius; ``bones`` pairs joint names. Each joint's radius is read
    from how wide the drawings are across its bones, the bones melt into each other over ``blend`` metres, the
    solid is cut to the outline the front and left drawings show (edges rounded over ``round``), meshed every
    ``step`` metres and smoothed ``smooth`` times. ``covers`` names other parts of the drawing the same solid
    makes (a head and arms grown with the body): their colours join this part's outline and they get no mesh
    of their own. Head, neck and limbs come out as one piece, so nothing floats.
    """

    joints: dict = field(default_factory=dict)
    bones: tuple = ()
    covers: tuple[str, ...] = ()
    step: float = 0.015
    blend: float = 0.05
    round: float = 0.05
    smooth: int = 10

    requires: ClassVar[tuple[str, ...]] = ("front", "left")
    bare_name: ClassVar[bool] = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.joints) < 2 or not self.bones:
            raise ValueError("a skeleton needs at least two joints and one bone")
        for a, b in self.bones:
            for name in (a, b):
                if name not in self.joints:
                    raise ValueError(f"bone ({a!r}, {b!r}) names a joint the skeleton does not have: {name!r}")
        for name, joint in self.joints.items():
            if len(joint) not in (3, 4):
                raise ValueError(f"joint {name!r} must be (x, y, z) or (x, y, z, radius)")
        object.__setattr__(self, "bones", tuple(tuple(bone) for bone in self.bones))
        object.__setattr__(self, "covers", tuple(self.covers))

    def views_of(self, masks, frame: Frame):
        """The front and left masks as (right, up, mask, frame) views in the asset's metres."""
        front, side = masks["front"], masks["left"]
        x0, x1 = frame.x_range(front.shape[1])
        z0, z1 = frame.z_range(side.shape[1])
        return [((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), front, (x0, x1, 0.0, frame.top)),
                ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), side, (-z1, -z0, 0.0, frame.top))]

    def solve(self, masks, frame: Frame):
        """The joints as an array, the bones as index pairs and every joint's radius, fitted where not given."""
        names = list(self.joints)
        joints = np.array([self.joints[name][:3] for name in names], float)
        bones = [(names.index(a), names.index(b)) for a, b in self.bones]
        radii = fit_radii(joints, bones, self.views_of(masks, frame))
        for i, name in enumerate(names):
            if len(self.joints[name]) == 4:
                radii[i] = float(self.joints[name][3])
        return names, joints, bones, radii

    def build(self, masks, frame, anchor=None) -> Built:
        solved = self.solve(masks, frame)
        _, joints, bones, radii = solved
        mesh = skin(joints, radii, bones, step=self.step, blend=self.blend, hull=self.views_of(masks, frame),
                    round=self.round, smooth=self.smooth)
        return replace(self._finish([mesh]), skeleton=solved)


@dataclass(frozen=True)
class plate(Builder):
    """A flat part seen whole from one side: a tail, a fin, a sign.

    The side view's outline is extruded along x by ``thickness`` metres — the front view's median width by
    default, clipped to 0.015-0.05 m — and centred where the front view's centre is. ``lean`` names a view
    whose centre per row bends the plate, so a tail sweeps back with height.
    """

    requires: ClassVar[tuple[str, ...]] = ("left",)
    bare_name: ClassVar[bool] = True
    thickness: float | None = None
    lean: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.thickness is not None and float(self.thickness) <= 0.0:
            raise ValueError("thickness must be positive")
        if self.lean is not None and self.lean not in VIEWS:
            raise ValueError(f"lean must name a view: {', '.join(VIEWS)}")

    def build(self, masks, frame, anchor=None) -> Built:
        side = masks["left"]
        if not side.any():
            return Built()
        shapes = [(poly, abs(_area(poly))) for poly in contours(side)]
        if not shapes:
            return Built()
        minimum = 0.0 if self.keep == "all" else float(self.keep) * frame.n * frame.n
        kept = [poly for poly, area in shapes if area >= minimum]
        dropped = [area / (float(frame.n) * frame.n) for _poly, area in shapes if area < minimum]
        if not kept:
            return Built(dropped=tuple(dropped))
        if self.keep != "all":
            kept = [max(kept, key=lambda poly: abs(_area(poly)))]
        meshes = [self._one(poly, masks, frame) for poly in kept]
        return self._finish(meshes, 0, dropped)

    def _one(self, polygon, masks, frame: Frame) -> Mesh:
        front = masks.get("front")
        poly = simplify(polygon, 1.0)
        profile = np.column_stack([(frame.z - poly[:, 0]) * frame.scale,
                                   frame.top + poly[:, 1] * frame.scale])
        if self.thickness is not None:
            thickness = float(self.thickness)
        elif front is not None and front.any():
            widths = front.sum(axis=1)
            thickness = float(np.clip(np.median(widths[widths > 0]) * frame.scale, 0.015, 0.05))
        else:
            thickness = 0.03
        if front is not None and front.any():
            centre = float((np.flatnonzero(front.any(axis=0)).mean() - frame.x) * frame.scale)
        else:
            centre = 0.0
        prism = extrude(profile, thickness, centre=centre)
        # The extrusion runs along z in extrude's frame: swap the axes so it runs along x, and turn the faces.
        mesh = Mesh(positions=np.asarray(prism.positions)[:, [2, 1, 0]].astype(np.float32),
                    faces=np.asarray(prism.faces)[:, ::-1].copy()).recompute_normals()
        if self.lean is not None and self.lean in masks and masks[self.lean].any():
            mesh = _lean(mesh, masks[self.lean], frame, axis="x" if self.lean == "front" else "z")
        return mesh


@dataclass(frozen=True)
class lathe(Builder):
    """A turned part: a vase, a column, a lamp. The front view's half-profile, revolved about its own centre."""

    requires: ClassVar[tuple[str, ...]] = ("front",)
    segments: int = 16
    rings: int = 30

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.rings) < 2 or int(self.segments) < 3:
            raise ValueError("lathe needs rings >= 2 and segments >= 3")

    def build(self, masks, frame, anchor=None) -> Built:
        front = masks["front"]
        if not front.any():
            return Built()
        kept, dropped = _pieces(front, self.keep, frame)
        if not kept:
            return Built(dropped=tuple(dropped))
        mask = kept[0]
        for piece in kept[1:]:
            mask = mask | piece
        mesh = lathe_solid(mask, (0.0, frame.top), frame.x_range(mask.shape[1]),
                           samples=self.rings, segments=self.segments)
        return self._finish([mesh], 0, dropped)


# --- what the builders share ------------------------------------------------------------------------------


def _mean_column(mask) -> float:
    cols = np.flatnonzero(np.asarray(mask, bool).any(axis=0))
    return float(cols.mean()) if cols.size else 0.0


def _area(polygon) -> float:
    points = np.asarray(polygon, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _pieces(mask, keep, frame: Frame):
    """The drawn pieces of one view a part keeps, and the area of each it drops as a fraction of N by N."""
    labels, count = components(mask)
    pieces = [labels == i for i in range(1, count + 1)]
    if not pieces:
        return [], []
    if keep == "all":
        return pieces, []
    unit = float(frame.n) * frame.n
    if keep == "largest":
        areas = [int(piece.sum()) for piece in pieces]
        best = int(np.argmax(areas))
        return [pieces[best]], [area / unit for i, area in enumerate(areas) if i != best]
    minimum = float(keep) * unit
    kept = [piece for piece in pieces if int(piece.sum()) >= minimum]
    dropped = [int(piece.sum()) / unit for piece in pieces if int(piece.sum()) < minimum]
    return kept, dropped


def _mirrored(meshes) -> list[Mesh]:
    """Keep the instances seen on the near side of the core (x >= 0) and reflect them across x = 0."""
    def span(mesh):
        x = np.asarray(mesh.positions, float)[:, 0]
        return float(x.min()), float(x.max())

    near = [mesh for mesh in meshes if sum(span(mesh)) / 2.0 >= 0.0]
    if not near:                                    # only the far one was drawn: move it to the near side
        near = [reflect_x(mesh) for mesh in meshes]
    out = []
    for mesh in near:
        out.append(mesh)
        if span(mesh)[0] > 1e-9:                    # wholly on the near side: add its reflection
            out.append(reflect_x(mesh))
    return out


def _lean(mesh: Mesh, mask, frame: Frame, axis: str = "x") -> Mesh:
    """Bend a plate so each row's centre follows a view's centre at that row."""
    centres = row_centres(mask)
    rows = np.flatnonzero(~np.isnan(centres))
    if rows.size == 0:
        return mesh
    positions = np.asarray(mesh.positions, float).copy()
    row = np.clip(np.round((frame.top - positions[:, 1]) / frame.scale - 0.5), 0, frame.rows - 1).astype(int)
    wanted = np.interp(row, rows, centres[rows] - centres[rows].mean()) * frame.scale
    if axis == "x":
        positions[:, 0] += wanted
    else:                                           # the left view's columns run the other way in z
        positions[:, 2] -= wanted
    return mesh.with_positions(positions).recompute_normals()


def _centre_column(mask) -> float:
    cols = np.flatnonzero(np.asarray(mask, bool).any(axis=0))
    return float((cols.min() + cols.max() + 1) / 2.0) if cols.size else 0.0


def _resize_mask(mask, width: int, height: int) -> np.ndarray:
    """A bool mask resized with nearest neighbours, as the prototype did: 255, resized, then above 127."""
    pixels = (np.asarray(mask, bool) * 255).astype(np.uint8)[..., None].repeat(3, axis=2)
    return resize(pixels, (int(width), int(height)), nearest=True)[..., 0] > 127


def _pad(mask, rows: int) -> np.ndarray:
    """Pad a view's masks at the bottom to ``rows``, so every view's first row is the same height."""
    return np.pad(np.asarray(mask, bool), ((0, rows - mask.shape[0]), (0, 0)))


# --- the declarations a Parts subclass makes ---------------------------------------------------------------


@dataclass(frozen=True)
class _Added:
    """One ``p.add`` line: a part's name, code colour, description and builder."""

    name: str
    colour: str
    what: str
    build: Builder


class _Parts:
    """The ``p`` of ``parts(self, p)``: one :meth:`add` call per named part."""

    def __init__(self, palette=None) -> None:
        self.palette = dict(CODE_COLOURS if palette is None else palette)
        self.items: list[_Added] = []

    def add(self, name: str, colour: str, what: str, build: Builder | None = None) -> _Added:
        """Declare one part: its name, its code colour, what it is, and the builder that makes it."""
        if not name or not str(name).isidentifier():
            raise ValueError(f"a part's name must be a Python identifier, not {name!r}")
        if str(name)[-1].isdigit():
            raise ValueError(f"the part name {name!r} ends in a digit; instances are numbered {name}1, {name}2...")
        if any(item.name == name for item in self.items):
            raise ValueError(f"the asset already has a part named {name!r}")
        if colour not in self.palette:
            raise ValueError(f"unknown code colour {colour!r}; choose from {', '.join(self.palette)}")
        if not what:
            raise ValueError(f"the part {name!r} needs a line saying what it is")
        builder = two_view() if build is None else build
        if not isinstance(builder, Builder):
            raise TypeError(f"build for {name!r} is a {type(builder).__name__}, not a builder")
        item = _Added(str(name), str(colour), str(what), builder)
        self.items.append(item)
        return item


@dataclass
class _Record:
    """What the build made for one part, kept for the lint."""

    item: _Added
    names: list[str]
    built: Built


class _Views:
    """The blueprints, aligned: cropped to the figure, scaled so the align parts span N rows, padded to R."""

    def __init__(self, c, items, palette, align, height, core) -> None:
        covered = _covered(items)
        trusted = {view for item in items if item.name not in covered for view in item.build.views}
        indexes = {name: index for index, name in enumerate(palette)}
        self.masks: dict[str, dict[str, np.ndarray]] = {}   # view -> part -> aligned bool mask
        self.drawn: dict[str, dict[str, np.ndarray]] = {}   # the blueprint as drawn (the back not mirrored)
        self.centres: dict[str, float] = {}                 # view -> the core's centre column, as drawn
        for view in VIEWS:
            if view not in trusted:
                continue
            image = classify(c.load(f"bp-{view}"), palette)
            masks = {name: open_mask(image == indexes[name], 2) for name in indexes if name != "white"}
            figure = np.any(list(masks.values()), axis=0)
            if not figure.any():
                raise ValueError(f"the {view} blueprint has no part in a code colour")
            rows, cols = np.nonzero(figure)
            r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
            chosen = np.any([masks[name] for name in align], axis=0) if align else figure
            arows = np.flatnonzero(chosen.any(axis=1))
            if arows.size == 0:
                raise ValueError(f"the align parts are not drawn in the {view} view")
            a0, a1 = int(arows.min()), int(arows.max()) + 1
            k = N / (a1 - a0)
            r0 = min(r0, a0)
            width, tall = max(1, round((c1 - c0) * k)), max(1, round((r1 - r0) * k))
            crop = {name: _resize_mask(mask[r0:r1, c0:c1], width, tall) for name, mask in masks.items()}
            self.drawn[view] = crop
            self.masks[view] = ({name: mask[:, ::-1] for name, mask in crop.items()}
                                if view == "back" else crop)
            self.centres[view] = _centre_column((crop if view == "back" else self.masks[view])[core])
        if not self.masks:
            raise ValueError("no blueprint was read")
        self.widths = {view: next(iter(masks.values())).shape[1] for view, masks in self.masks.items()}
        self.rows = max(next(iter(masks.values())).shape[0] for masks in self.masks.values())
        self.scale = float(height) / N
        self.top = self.rows * self.scale
        self.masks = {view: {name: _pad(mask, self.rows) for name, mask in masks.items()}
                      for view, masks in self.masks.items()}

    def frame(self, core: str) -> Frame:
        """The metre frame of this asset, from the core part's centre column in the front and left views."""
        for view in ("front", "left"):
            if view in self.masks and not self.masks[view][core].any():
                raise ValueError(f"the core part {core!r} is not drawn in the {view} view")
        x = _centre_column(self.masks["front"][core]) if "front" in self.masks else 0.0
        z = _centre_column(self.masks["left"][core]) if "left" in self.masks else 0.0
        return Frame(rows=self.rows, scale=self.scale, top=self.top, x=x, z=z)

    def frame_of(self, view: str, centre: float) -> tuple[float, float, float, float]:
        """The rectangle a view's drawing covers, with ``centre`` the core's column in it."""
        return (-centre * self.scale, (self.widths[view] - centre) * self.scale, 0.0, self.top)


# --- the subclass -----------------------------------------------------------------------------------------


class Parts(Asset):
    """One object of named parts, built from the blueprints of its views.

    ``desc`` says what the object is, in the words the drawings were asked with; ``style`` how it is drawn.
    ``photos`` are pictures, relative to the asset file, that the view drawings are asked from, used instead of
    a reference drawing. ``core`` is the part that centres the views (the first part by default), and ``align``
    the parts the views are scaled by (None: the whole figure). ``notes`` adds a line per view to that view's
    drawing prompt and ``details`` a line per view to its texture prompt; ``base`` is the colour the paint
    starts from.
    """

    desc: str = ""
    style: str = DEFAULT_STYLE
    photos: list[str] = []
    core: str | None = None
    align: list[str] | None = None
    notes: dict[str, str] = {}
    details: dict[str, str] = {}
    base: tuple[int, int, int] = (170, 170, 170)
    photoreal: bool = False
    lod: str | None = None
    atlas_size: int = 2048

    def parts(self, p: _Parts) -> None:
        """Declare the object's parts, one ``p.add`` per part."""
        raise NotImplementedError(f"{type(self).__name__} has no parts() method")

    # --- draw --------------------------------------------------------------------------------------------

    def draw(self, c) -> None:
        """The reference (unless there are photos), one drawing per view, and a blueprint of each."""
        items, _core = self._collect(c.palette)
        if self.photos:
            refs = [_rgb(read_png(self.cache_dir().parent / photo)) for photo in self.photos]
        else:
            c.image("ref", _REF_PROMPT.format(desc=self.desc, style=self.style), aspect="1:1")
            refs = ["ref"]
        codes = "\n".join(["- pure white (255,255,255) = background;"]
                          + [f"- {item.colour} {tuple(c.palette[item.colour])} = {item.what};" for item in items])
        for view in VIEWS:
            note = self.notes.get(view, "")
            c.image(f"view-{view}", _VIEW_PROMPT.format(desc=self.desc, how=_HOW[view], note=note),
                    refs=refs, aspect="1:1")
            c.image(f"bp-{view}", _BP_PROMPT.format(code_list=codes, note=note), refs=[f"view-{view}"],
                    aspect="1:1")

    # --- build -------------------------------------------------------------------------------------------

    def build(self, c):
        """Trace the blueprints, align the views, build every part, attach, and record what each view expects."""
        items, core = self._collect(c.palette)
        palette = {"white": c.palette["white"], **{item.name: c.palette[item.colour] for item in items}}
        loaded = _Views(c, items, palette, self.align, self.height, core)
        frame = loaded.frame(core)
        scene = c.scene()
        records: list[_Record] = []
        covered = _covered(items)
        for item in items:
            if item.name in covered:
                continue                                     # a skeleton part grows it with its own solid
            masks = {view: _union(loaded.masks[view], item) for view in item.build.views if view in loaded.masks}
            if any(view not in masks or not masks[view].any() for view in item.build.requires):
                continue                                     # the prototype skips a part its side view misses
            anchor = self._anchor(item.build, items, loaded, frame)
            builder = self._coarse(item.build)
            built = builder.build(masks, frame, anchor=anchor)
            names = _instance_names(item, built.meshes)
            for name, mesh in zip(names, built.meshes):
                scene.add(name, mesh)
            records.append(_Record(item=item, names=names, built=built))
        self._attach(scene, records)
        for view in VIEWS:
            if view not in loaded.masks:
                continue
            drawn = self._drawn(loaded, items, view)
            if drawn:
                centre = (loaded.centres[view] if view == "back"
                          else (frame.x if view == "front" else frame.z))
                c.expect(view, drawn, frame=loaded.frame_of(view, centre))
        self._built, self._frame = records, frame
        return scene

    def marks(self) -> list[tuple[str, tuple, tuple]]:
        """Every skeleton's bones from the last build, ``(label, start, end)``, labelled with the joints' radii."""
        out = []
        for record in getattr(self, "_built", []):
            solved = getattr(record.built, "skeleton", None)
            if solved is None:
                continue
            names, joints, bones, radii = solved
            for i, j in bones:
                label = f"{names[i]} r={radii[i]:.3f} - {names[j]} r={radii[j]:.3f}"
                out.append((label, tuple(float(v) for v in joints[i]), tuple(float(v) for v in joints[j])))
        return out

    def _coarse(self, builder: Builder) -> Builder:
        """The builder a ``--lod low`` run uses: fewer rings and segments, still textured from the same answers."""
        if self.lod is None:
            return builder
        if self.lod != "low":
            raise ValueError(f"lod must be 'low', not {self.lod!r}")
        fields = {}
        if hasattr(builder, "rings"):
            fields["rings"] = 10
        if hasattr(builder, "segments"):
            fields["segments"] = 8
        if isinstance(builder, skeleton):
            fields["step"] = 2 * builder.step
        return replace(builder, **fields) if fields else builder

    # --- paint -------------------------------------------------------------------------------------------

    def paint(self, c, scene) -> dict:
        """A template and one albedo call per view (right from the mirrored left), assembled into an atlas.

        The call asks for the colour texture of the template's flat regions, with the view drawing and the
        photos as references. An answer that keeps under 0.85 of the template's outline is asked again, up to
        three times, and the best answer is kept; the code colours it left are replaced by ``base``. A normal
        map is computed from the finished albedo. Returns the albedo and normal arrays, the atlas's UVs and its
        meshes (split where a vertex carries two UVs).
        """
        items, _core = self._collect(c.palette)
        colours = self._colours(c, scene, items)
        size = int(self.atlas_size)
        atlas = view_atlas(scene.meshes, ("front", "back", "left", "right"), (size, size))
        views = {island.view: _island_template(scene.meshes, atlas, colours, island)
                 for island in atlas.islands}
        answers = {}
        for view in ("front", "back", "left"):
            flat, regions, aspect, offset = views[view]
            refs = [flat]
            if (c.images / f"view-{view}.png").exists():
                refs.append(f"view-{view}")
            refs.extend(self._paint_refs(c))
            prompt = self._paint_prompt(items, view)
            kept, picture = 0.0, np.full_like(flat, 255)
            for attempt in range(_PAINT_TRIES):
                name = f"albedo-{view}" if attempt == 0 else f"albedo-{view}-{attempt + 1}"
                text = prompt if attempt == 0 else f"{prompt} (attempt {attempt + 1})"
                answer = c.image(name, text, refs=refs, aspect=aspect)
                candidate = resize(answer, (flat.shape[1], flat.shape[0]))
                score = painting.kept_outline(flat, candidate)
                if score > kept:
                    kept, picture = score, candidate
                if score >= _PAINT_KEPT:
                    break
            answers[view] = self._strip_codes(c, items, view, picture, regions, offset)
            if kept < _PAINT_KEPT:
                c.findings.append(Finding(
                    "A-OUTLINE", "warn", view=view, value=round(kept, 3),
                    message=f"the {view} albedo keeps only {kept:.0%} of its template's outline",
                    hint="a note for that view; another model"))
        answers["right"] = answers["left"][:, ::-1].copy()

        albedo = np.full((size, size, 3), 255, np.uint8)
        covered = np.zeros((size, size), bool)
        bases = {name: self.base for name in scene.meshes}
        for island in atlas.islands:
            flat, regions, _aspect, offset = views[island.view]
            picture = resize(answers[island.view], (flat.shape[1], flat.shape[0]))
            x0, y0, x1, y1 = island.rect
            ox, oy = offset
            crop = picture[oy:oy + (y1 - y0), ox:ox + (x1 - x0)]
            region = regions[:y1 - y0, :x1 - x0]
            albedo[y0:y1, x0:x1] = painting.clean(crop, region, bases)
            covered[y0:y1, x0:x1] = region > 0
        albedo = painting.bleed(albedo, covered, steps=12)
        return {"albedo": albedo, "normal": painting.normal_from_albedo(albedo),
                "uvs": dict(atlas.uvs), "meshes": dict(atlas.meshes)}

    def _colours(self, c, scene, items) -> dict:
        """The code colour of every mesh the scene holds, by instance name."""
        colours = {}
        for record in getattr(self, "_built", []):
            for name in record.names:
                if name in scene.meshes:
                    colours[name] = c.palette[record.item.colour]
        for name in scene.meshes:
            if name in colours:
                continue
            for item in items:
                suffix = name[len(item.name):] if name.startswith(item.name) else None
                if name == item.name or (suffix and suffix.isdigit()):
                    colours[name] = c.palette[item.colour]
                    break
        missing = [name for name in scene.meshes if name not in colours]
        if missing:
            raise ValueError(f"paint does not know which part {missing[0]!r} is; build the scene with build()")
        return colours

    def _paint_refs(self, c) -> list:
        """The reference pictures of the paid call: the asset's first two photos, else its reference drawing."""
        refs = []
        folder = self.cache_dir().parent
        for photo in self.photos[:2]:
            path = folder / photo
            if path.exists():
                refs.append(_rgb(read_png(path)))
        if not refs and (c.images / "ref.png").exists():
            refs.append("ref")
        return refs

    def _paint_prompt(self, items, view: str) -> str:
        """The text of one view's albedo call, from the parts, the object and the view's details."""
        codes = ", ".join(f"{item.colour} = {item.what.split(':')[0]}" for item in items)
        return _PAINT_PROMPT.format(name=_VIEW_NAME[view], desc=self.desc, codes=codes,
                                    details=self.details.get(view, ""), finish=_REAL if self.photoreal else _FLAT)

    def _strip_codes(self, c, items, view, picture, regions, offset):
        """Replace the code colours a model left in an answer with ``base``, and report what is still near one."""
        code_rgb = np.array([c.palette[item.colour] for item in items], float)
        out = np.array(picture, copy=True)
        distance = np.linalg.norm(out[:, :, None, :].astype(float) - code_rgb, axis=3)
        out[(distance < 60).any(axis=2)] = self.base
        leftover = (distance < 120).any(axis=2)
        height, width = regions.shape
        ox, oy = offset
        inside = regions > 0
        fraction = (float(leftover[oy:oy + height, ox:ox + width][inside].mean())
                    if inside.any() else 0.0)
        if fraction > 0.005:
            c.findings.append(Finding(
                "A-CODE", "warn", view=view, value=round(fraction, 4),
                message=f"{fraction:.1%} of the object is still near a part's code colour",
                hint="paint --again VIEW"))
        return out

    def forget(self, c, name: str) -> list:
        """A view forgets its drawing, blueprint and albedos; a part forgets the views its builder trusts."""
        if name in VIEWS:
            return self._forget_views(c, (name,))
        try:
            items, _core = self._collect(c.palette)
        except ValueError:
            items = []
        for item in items:
            if item.name == name:
                return self._forget_views(c, item.build.views)
        return super().forget(c, name)

    def _forget_views(self, c, views) -> list:
        removed = []
        for view in views:
            for stem in (f"view-{view}", f"bp-{view}", f"albedo-{view}"):
                removed.extend(c.forget(stem))
        return removed

    def _anchor(self, builder: Builder, items, loaded: _Views, frame: Frame):
        """The point that orients a tube: the middle of the part it names, or of the core by default."""
        name = getattr(builder, "anchor", None) or self.core or items[0].name
        if "front" not in loaded.masks:
            return None
        mask = loaded.masks["front"][name]
        if not mask.any():
            return None
        rows = np.flatnonzero(mask.any(axis=1))
        y = frame.top - (rows.mean() + 0.5) * frame.scale
        x = (frame.x - _centre_column(mask)) * frame.scale
        z = 0.0
        if "left" in loaded.masks and loaded.masks["left"][name].any():
            z = (frame.z - _centre_column(loaded.masks["left"][name])) * frame.scale
        return (x, float(y), float(z))

    def _attach(self, scene, records) -> None:
        """Move each part that names ``attach`` the shortest way until it touches that part."""
        for record in records:
            target = record.item.build.attach
            if target is None:
                continue
            if target == record.item.name:
                raise ValueError(f"the part {record.item.name!r} cannot attach to itself")
            targets = [mesh for other in records if other.item.name == target for mesh in other.built.meshes]
            if not targets:
                raise ValueError(f"{record.item.name!r} attaches to {target!r}, which was not built")
            moved = []
            for name, mesh in zip(record.names, record.built.meshes):
                nearest = min(targets, key=lambda other: gap(mesh, other))
                offset = closest_offset(mesh, nearest)
                distance = float(np.linalg.norm(offset))
                if distance > 0.0:
                    mesh = mesh.translate(offset)
                    scene.meshes[name] = mesh
                    moved.append((name, distance))
            if moved:
                names = ", ".join(name for name, _ in moved)
                distance = max(value for _, value in moved)
                self._moved.append(Finding(
                    "A-ATTACH", "info", part=record.item.name, value=round(distance, 4),
                    message=f"moved {names} {distance:.3f} m to touch {target}",
                    hint="none, unless the move is larger than the gap it was meant to close"))
            record.built = Built(meshes=tuple(scene.meshes[name] for name in record.names),
                                 filled=record.built.filled, dropped=record.built.dropped)

    def _drawn(self, loaded: _Views, items, view: str) -> dict[str, np.ndarray]:
        """The blueprint's masks for the parts that trust ``view``; the back view's as drawn, not mirrored."""
        masks = loaded.drawn[view]
        covered = _covered(items)
        return {item.name: _pad(_union(masks, item), loaded.rows) for item in items
                if item.name not in covered and view in item.build.views and _union(masks, item).any()}

    def _collect(self, palette):
        p = _Parts(palette)
        self.parts(p)
        if not p.items:
            raise ValueError(f"{type(self).__name__}.parts() added no part")
        names = [item.name for item in p.items]
        if "white" in names:
            raise ValueError("'white' is the background's name, not a part's")
        core = names[0] if self.core is None else self.core
        if core not in names:
            raise ValueError(f"core names {core!r}, which parts() never adds")
        unknown = [name for name in (self.align or []) if name not in names]
        if unknown:
            raise ValueError(f"align names {unknown[0]!r}, which parts() never adds")
        for item in p.items:
            for option in ("anchor", "attach"):
                target = getattr(item.build, option, None)
                if target is not None and target not in names:
                    raise ValueError(f"{item.name!r} names {target!r} as its {option}, "
                                     "which parts() never adds")
        colours = [p.palette[item.colour] for item in p.items]
        for index, item in enumerate(p.items):
            if colours[index] in colours[:index]:
                other = names[colours.index(colours[index])]
                raise ValueError(f"{item.name!r} and {other!r} share the code colour {colours[index]!r}; "
                                 "a blueprint cannot tell them apart")
        self._moved = []
        return p.items, core

    # --- check -------------------------------------------------------------------------------------------

    def check(self, c, scene) -> list[Finding]:
        """A-ROWS, A-STRAY, A-ATTACH, A-KEEP and A-SHEET: what the drawings and the build disagree about."""
        records = getattr(self, "_built", [])
        frame = getattr(self, "_frame", None)
        findings = list(getattr(self, "_moved", []))
        for record in records:
            name = record.item.name
            if frame is not None and record.built.filled > 0.2 * frame.n:
                findings.append(Finding(
                    "A-ROWS", "warn", part=name, value=record.built.filled,
                    message=(f"{record.built.filled} rows ({record.built.filled / frame.n:.0%} of the height) "
                             "filled from the other view"),
                    hint=("the blueprints disagree on what the part is: say it in the part's text, "
                          "or fill=\"off\"")))
            if record.built.dropped:
                sizes = ", ".join(f"{area:.2%}" for area in record.built.dropped)
                findings.append(Finding(
                    "A-STRAY", "info", part=name, value=len(record.built.dropped),
                    message=f"dropped {len(record.built.dropped)} stray piece(s) ({sizes} of the drawing)",
                    hint="none, or keep=\"all\""))
        findings.extend(self._kept(c))
        findings.extend(self._sheets(c))
        return findings

    def _kept(self, c) -> list[Finding]:
        """A-KEEP: a blueprint that lost the outline of its view drawing, when that drawing is cached."""
        findings = []
        for view in VIEWS:
            drawing = c.images / f"view-{view}.png"
            blueprint = c.images / f"bp-{view}.png"
            if not drawing.exists() or not blueprint.exists():
                continue
            overlap = float(iou(_figure(read_png(blueprint)), _figure(read_png(drawing))))
            if overlap < 0.85:
                findings.append(Finding(
                    "A-KEEP", "warn", view=view, value=round(overlap, 3),
                    message=f"the {view} blueprint does not keep the outline of the {view} drawing",
                    hint="draw --again, or a note for that view"))
        return findings

    def _sheets(self, c) -> list[Finding]:
        """A-SHEET: a drawing that holds more than one figure of comparable size, a sheet of views."""
        findings = []
        for view in VIEWS:
            for kind in ("view", "bp"):
                path = c.images / f"{kind}-{view}.png"
                if not path.exists():
                    continue
                figure = (np.asarray(read_png(path))[..., :3] < 235).any(axis=2)
                figures = _sheet_figures(figure)
                if figures > 1:
                    findings.append(Finding(
                        "A-SHEET", "error", view=view, value=figures,
                        message=f"the {kind} drawing for {view} holds {figures} figures of comparable size",
                        hint="draw --again VIEW"))
        return findings


# --- helpers ----------------------------------------------------------------------------------------------


def _instance_names(item: _Added, meshes) -> list[str]:
    """A part's instance names: ``name1``, ``name2``... (a lone plate keeps its own name, as the prototype did)."""
    if not meshes:
        return []
    if len(meshes) == 1 and item.build.bare_name:
        return [item.name]
    return [f"{item.name}{k}" for k in range(1, len(meshes) + 1)]


def _figure(image) -> np.ndarray:
    """The drawing's figure as a square mask: every non-white pixel, cropped to its box, resized to 256."""
    mask = (np.asarray(image)[..., :3] < 235).any(axis=2)
    rows, cols = np.nonzero(mask)
    if not rows.size:
        return np.zeros((256, 256), bool)
    mask = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return _resize_mask(mask, 256, 256)


def _sheet_figures(figure) -> int:
    """How many figures of comparable size a drawing holds: two big, tall and similar is a sheet of views."""
    figure = np.asarray(figure, bool)
    rows, cols = np.nonzero(figure)
    if not rows.size:
        return 0
    figure = figure[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    small = _resize_mask(figure, 128, 128)
    labels, count = components(small)
    if count < 2:
        return 0
    total = float(small.sum())
    if total <= 0:
        return 0
    big = []
    for index in range(1, count + 1):
        piece = labels == index
        area = float(piece.sum())
        rows = np.flatnonzero(piece.any(axis=1))
        if area < 0.1 * total or rows.size < 0.6 * small.shape[0]:
            continue
        big.append(area)
    if len(big) < 2:
        return 0
    largest = max(big)
    return len(big) if all(area >= 0.5 * largest for area in big) else 0


def _rgb(image) -> np.ndarray:
    """An image as uint8 RGB, dropping any alpha, for a paid call's reference."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("a photo must be an (H, W, 3) or (H, W, 4) image")
    return np.ascontiguousarray(array[..., :3], dtype=np.uint8)


def _island_template(parts, atlas, colours, island):
    """One island's whole picture, its region map, the aspect it is padded to and the padding's offset."""
    image, regions = template(parts, atlas, colours, island.view, whole=True)
    x0, y0, x1, y1 = island.rect
    crop, region = image[y0:y1, x0:x1], regions[y0:y1, x0:x1]
    aspect = nearest_aspect(crop.shape[1], crop.shape[0])
    padded, offset = pad_to_aspect(crop, aspect)
    return padded, region, aspect, offset


def _covered(items) -> set[str]:
    """The parts another part's skeleton grows, so they get no mesh of their own."""
    return {name for item in items for name in getattr(item.build, "covers", ())}


def _union(masks: dict, item) -> np.ndarray:
    """A part's mask in one view, joined with those of the parts its skeleton covers."""
    out = masks[item.name]
    for name in getattr(item.build, "covers", ()):
        if name not in masks:
            raise ValueError(f"{item.name}'s skeleton covers {name!r}, which is not a part")
        out = out | masks[name]
    return out
