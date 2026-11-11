"""Assets as a pipeline: drawings turned into a scene, a report of what is wrong, and textures.

Layer 2 of the asset tools. A class inherits from :class:`Asset`, overrides ``draw``, ``build``, ``paint`` and
``check``, and is run from Python with :meth:`Asset.run`; every paid call is cached under ``NAME.asset/``.
:class:`Parts` is the ready subclass for one object of named parts, with the builders :func:`two_view`,
:func:`tube`, :func:`plate` and :func:`lathe`. Layer 1 is :mod:`manifoldx.modeling` (trace, views, paint, lint)
and :mod:`manifoldx.imagegen`; this package imports those and nothing imports it back.
"""

from manifoldx.asset.base import Asset, Result, standard_findings
from manifoldx.asset.context import CODE_COLOURS, Context, Expectation
from manifoldx.asset.inspect import inspect
from manifoldx.asset.parts import Parts, lathe, plate, tube, two_view
from manifoldx.asset.report import Report
from manifoldx.asset.scene import Scene
from manifoldx.asset.write import save

__all__ = ["Asset", "CODE_COLOURS", "Context", "Expectation", "Parts", "Report", "Result", "Scene",
           "inspect", "lathe", "plate", "save", "standard_findings", "tube", "two_view"]
