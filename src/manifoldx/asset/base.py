"""An asset: a class whose stages draw, build, paint and check one model, and the run that puts them in order.

The build stage returns a :class:`~manifoldx.asset.scene.Scene`; the lint stage is the standard checks of
:mod:`manifoldx.modeling.lint` over that scene plus whatever the asset adds in ``check``. Every paid call goes
through the :class:`~manifoldx.asset.context.Context`, so a run of ``build`` and ``lint`` alone needs no network.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from manifoldx.asset.context import Context
from manifoldx.asset.report import Report
from manifoldx.asset.scene import Scene
from manifoldx.modeling.lint import (Finding, check_contact, check_meshes, check_seen, check_triangles)
from manifoldx.modeling.views import _VIEWS

STAGES = ("draw", "build", "lint", "paint")

# A drawing without a frame covers the scene's silhouette with this much room around it.
_FRAME_PAD = 0.02


@dataclass
class Result:
    """What a run gives back: the scene, the textures, the report of what is wrong, and the context it used."""

    scene: Scene | None = None
    textures: dict = field(default_factory=dict)
    report: Report = field(default_factory=Report)
    context: Context | None = None

    @property
    def exit_code(self) -> int:
        return self.report.exit_code


def _instances(parts: dict, name: str) -> list:
    """The scene's meshes for one declared part: its one mesh, or every ``name1``, ``name2``... instance."""
    if name in parts:
        return [parts[name]]
    return [mesh for key, mesh in parts.items()
            if key.startswith(name) and key[len(name):].isdigit() and key[len(name):]]


def standard_findings(context: Context, scene: Scene, tol: float, max_tris: int) -> list[Finding]:
    """The checks every asset gets: closed meshes, contact, triangle count and what each drawing expects."""
    parts = scene.meshes
    findings = [*check_meshes(parts), *check_contact(parts, tol), *check_triangles(parts, max_tris)]
    for expectation in context.expectations:
        named = {}
        for name in expectation.names:
            meshes = _instances(parts, name)
            if not meshes:
                raise ValueError(f"expect({expectation.view!r}) names parts the scene does not have: {name}")
            named[name] = meshes if len(meshes) > 1 else meshes[0]
        frame = expectation.frame if expectation.frame is not None else _frame(parts, expectation.view)
        findings.extend(check_seen(named, expectation.view, expectation.masks, frame))
    return findings


def _frame(parts: dict, view: str) -> tuple[float, float, float, float]:
    """The rectangle of metres a drawing of the whole scene in ``view`` covers, with a little room around it."""
    if view not in _VIEWS:
        raise ValueError(f"unknown view {view!r}; choose from {', '.join(_VIEWS)}")
    _, right, up = _VIEWS[view]
    right, up = np.asarray(right, float), np.asarray(up, float)
    positions = np.concatenate([np.asarray(mesh.positions, float) for mesh in parts.values()])
    u, v = positions @ right, positions @ up
    pad_u = _FRAME_PAD * max(float(u.max() - u.min()), 1e-6)
    pad_v = _FRAME_PAD * max(float(v.max() - v.min()), 1e-6)
    return (float(u.min()) - pad_u, float(u.max()) + pad_u,
            float(v.min()) - pad_v, float(v.max()) + pad_v)


class Asset:
    """One model as a pipeline: what to draw, how to build it, how to paint it, what else to check.

    ``name`` is the cache directory's stem and defaults to the class name lowercased. ``model`` is an image
    model id to reach through OpenRouter, unless a model is passed to :meth:`run`. ``height`` is the object's
    height in metres, which sets the contact tolerance and the default pixel-to-metre scale.
    """

    name = "asset"
    model: str | None = None
    height: float = 1.0
    max_tris: int = 20000

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = cls.__name__.lower()

    # --- the stages ---------------------------------------------------------------------------------------

    def draw(self, c: Context) -> None:
        """Ask the image model for every picture the build needs. Cached, so a second run sends nothing."""

    def build(self, c: Context) -> Scene:
        """Build the scene from the drawings: named meshes, placed, with their materials."""
        raise NotImplementedError(f"{type(self).__name__} has no build stage")

    def paint(self, c: Context, scene: Scene) -> dict:
        """The textures of a built scene, per mesh. Nothing by default."""
        return {}

    def check(self, c: Context, scene: Scene) -> list[Finding]:
        """The asset's own findings, on top of the standard lint. Empty by default."""
        return []

    def marks(self) -> list[tuple[str, tuple, tuple]]:
        """Lines the last build wants drawn over its drawings, ``(label, start, end)`` in metres; none by default.

        ``inspect`` draws them on every overlay and lists them in the summary, so an agent can see where a
        skeleton's joints and bones sit against the blueprints.
        """
        return []

    def forget(self, c: Context, name: str) -> list[Path]:
        """Delete cached pictures before a run; a stage may widen ``name`` to mean more than one image."""
        return c.forget(name)

    # --- the run ------------------------------------------------------------------------------------------

    def cache_dir(self) -> Path:
        """``NAME.asset`` next to the file that defines the class."""
        return Path(inspect.getfile(type(self))).resolve().parent / f"{self.name}.asset"

    def run(self, cache=None, stages=STAGES, model=None, budget=None, max_tris=None, again=()) -> Result:
        """Run the stages in order and return the scene, the textures and the report.

        ``cache`` is the asset's directory, ``NAME.asset`` next to this file by default; ``stages`` picks and
        orders them (``build`` runs when ``lint`` or ``paint`` needs a scene); ``model`` overrides the class's
        image model, ``budget`` caps the spend in dollars, or the calls when the model reports no cost, and
        ``max_tris`` the A-TRIS threshold. ``again`` names cached images to forget first, as ``--again`` does.
        """
        stages = tuple(stages)
        unknown = [stage for stage in stages if stage not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage {unknown[0]!r}; choose from {', '.join(STAGES)}")
        c = Context(self.cache_dir() if cache is None else cache, model=self.model if model is None else model,
                    budget=budget, height=self.height)
        for name in again:
            self.forget(c, name)
        if "draw" in stages:
            self.draw(c)
        scene = None
        if any(stage in stages for stage in ("build", "lint", "paint")):
            scene = self.build(c)
            if not isinstance(scene, Scene):
                raise TypeError(f"{type(self).__name__}.build returned {type(scene).__name__}, not a Scene")
        report = Report()
        if "lint" in stages:
            tol = max(0.01 * float(self.height), 1e-6)
            limit = self.max_tris if max_tris is None else int(max_tris)
            report.extend(standard_findings(c, scene, tol=tol, max_tris=limit))
            report.extend(self.check(c, scene))
        textures = dict(self.paint(c, scene) or {}) if "paint" in stages else {}
        if "paint" in stages:
            report.extend(c.findings)
        return Result(scene=scene, textures=textures, report=report, context=c)
