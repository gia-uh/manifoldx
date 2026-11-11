"""Assets as a pipeline: drawings turned into a scene, a report of what is wrong, and textures.

Layer 2 of the asset tools. A class inherits from :class:`Asset`, overrides ``draw``, ``build``, ``paint`` and
``check``, and is run from Python with :meth:`Asset.run`; every paid call is cached under ``NAME.asset/``. Layer
1 is :mod:`manifoldx.modeling` (trace, views, paint, lint) and :mod:`manifoldx.imagegen`; this package imports
those and nothing imports it back.
"""

from manifoldx.asset.base import Asset, Result, standard_findings
from manifoldx.asset.context import CODE_COLOURS, Context, Expectation
from manifoldx.asset.report import Report
from manifoldx.asset.scene import Scene

__all__ = ["Asset", "CODE_COLOURS", "Context", "Expectation", "Report", "Result", "Scene", "standard_findings"]
