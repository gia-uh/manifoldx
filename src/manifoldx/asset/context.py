"""The tools a stage gets: paid calls paid once, cached drawings, masks, the scene and what a drawing expects.

Every call that costs money goes through :meth:`Context.image`, which names its answer and stores it under the
asset's cache directory next to the hash of everything that shaped it. A later run with the same prompt, model,
aspect and references reads the file and sends nothing; a changed prompt re-requests only that image. The build
stage reads answers with :meth:`Context.load`, which never reaches the network.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifoldx.asset.scene import Scene
from manifoldx.image import read_png, write_png
from manifoldx.imagegen import BudgetExceeded
from manifoldx.modeling.trace import classify

# The flat colours a drawing may code its regions in, white being the paper. A stage may pass its own palette.
# The prototype's drawings, and the committed blueprints, were made with these values; a pixel is matched to the
# nearest entry, so they must be the ones the image model was asked for.
CODE_COLOURS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (130, 130, 130),
    "red": (220, 40, 40),
    "orange": (255, 140, 0),
    "yellow": (255, 220, 0),
    "green": (0, 200, 0),
    "cyan": (0, 200, 255),
    "blue": (0, 110, 255),
    "magenta": (255, 0, 200),
}

# Pixels per metre a drawing is taken to have when a stage gives no scale and the asset no height.
_DRAWING_PIXELS = 480.0


@dataclass(frozen=True)
class Expectation:
    """What one drawing says the camera should see: a view, each part's mask, and an optional frame.

    ``masks`` holds one bool mask per part name: a blueprint with a colour per part is checked part by part,
    never as one mask for all. Every mask has the same shape, the drawing's pixels.
    """

    view: str
    masks: dict[str, np.ndarray]                        # part name -> (H, W) bool
    frame: tuple[float, float, float, float] | None = None  # u_min, u_max, v_min, v_max in metres

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.masks)


def _rgb(image) -> np.ndarray:
    """An image as contiguous uint8 RGB, refusing anything else."""
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("an image must be a uint8 array of shape (H, W, 3)")
    return np.ascontiguousarray(a)


def _key(model: str, prompt: str, aspect: str, refs: list[np.ndarray]) -> str:
    """The hash of everything that shapes an answer: the model, the prompt, the aspect and the reference pixels."""
    digest = hashlib.sha256()
    digest.update(b"manifoldx.asset/1\n")
    for text in (model, prompt, aspect):
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    for ref in refs:
        digest.update(f"{ref.shape}\0".encode("ascii"))
        digest.update(np.ascontiguousarray(ref, np.uint8).tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


class Context:
    """The tools a stage is given, and the cache that makes a second run free.

    ``cache`` is the asset's directory (``NAME.asset/``), holding ``images/<name>.png`` and the sidecar
    ``images/<name>.json`` with the prompt, references, model, aspect and key that produced it. ``model`` is an
    :class:`~manifoldx.imagegen.ImageModel` or the id of one to reach through OpenRouter. ``budget`` caps the
    spend in dollars when the model reports a cost per call, else the number of calls.
    """

    def __init__(self, cache, model=None, budget: float | None = None, height: float | None = None,
                 palette=None, scale: float | None = None) -> None:
        self.cache = Path(cache)
        self.images = self.cache / "images"
        self.model = model
        self.budget = budget
        self.height = height
        self.palette = dict(CODE_COLOURS if palette is None else palette)
        self.scale = float(scale) if scale is not None else (
            float(height) / _DRAWING_PIXELS if height else 2.0 / _DRAWING_PIXELS)
        self.spent = 0.0        # dollars when the model reports a cost, else calls
        self.calls = 0          # requests actually sent
        self.expectations: list[Expectation] = []
        self._instance = None   # the model once built from an id

    # --- the paid step ------------------------------------------------------------------------------------

    def image(self, name: str, prompt: str, refs=(), aspect: str = "1:1") -> np.ndarray:
        """The picture called ``name``: from the cache when its key matches, else from the model, once.

        ``refs`` are names of cached pictures or images themselves; they are part of the key, so changing one
        re-requests the picture. With no model configured and no matching cached answer this raises.
        """
        path, sidecar = self.images / f"{name}.png", self.images / f"{name}.json"
        arrays, labels = [], []
        for ref in refs:
            if isinstance(ref, str):
                arrays.append(self.load(ref))
                labels.append(ref)
            else:
                arrays.append(_rgb(ref))
                labels.append("sha256:" + hashlib.sha256(arrays[-1].tobytes()).hexdigest())
        meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None
        model_name = self._model_name() or (meta or {}).get("model") or ""
        key = _key(model_name, prompt, aspect, arrays)
        if path.exists() and meta and meta.get("key") == key:
            return read_png(path)
        model = self._resolve_model(name)
        if self.budget is not None and self.spent >= self.budget:
            raise BudgetExceeded(f"the budget of {self.budget} is spent; {name!r} was not drawn")
        drawn = _rgb(model.generate(prompt, images=arrays, aspect=aspect))
        self.calls += 1
        cost = getattr(model, "cost", None)
        self.spent += float(cost) if cost is not None else 1.0
        self.images.mkdir(parents=True, exist_ok=True)
        write_png(path, drawn)
        sidecar.write_text(json.dumps({"prompt": prompt, "refs": labels, "model": model_name,
                                       "aspect": aspect, "key": key}, indent=1), encoding="utf-8")
        return drawn

    def load(self, name: str) -> np.ndarray:
        """A cached picture by name, with no request; raises when it has not been drawn."""
        path = self.images / f"{name}.png"
        if not path.exists():
            raise FileNotFoundError(
                f"no cached image {name!r} at {path}; run the draw stage first, or the asset that draws it")
        return read_png(path)

    def mask(self, name: str, colour, palette=None) -> np.ndarray:
        """The pixels of a cached picture in one code colour, as a bool mask.

        ``colour`` is a name in the palette (white by default) or an RGB tuple; a pixel belongs to the nearest
        colour of the palette, so a drawing's antialiased edges fall to the region they surround.
        """
        picture = self.load(name)
        if isinstance(colour, str):
            palette = dict(self.palette if palette is None else palette)
            if colour not in palette:
                raise ValueError(f"unknown colour {colour!r}; choose from {', '.join(palette)}")
            return classify(picture, palette) == list(palette).index(colour)
        palette = {"background": (255, 255, 255), "code": tuple(int(v) for v in np.asarray(colour).reshape(3))}
        return classify(picture, palette) == 1

    # --- the scene -----------------------------------------------------------------------------------------

    def scene(self) -> Scene:
        """A new, empty scene to add named meshes to."""
        return Scene()

    def expect(self, view: str, mask, names=None, frame=None) -> Expectation:
        """Record what a drawing says the camera sees in ``view``, for the standard A-SEEN check.

        ``mask`` is one bool mask shared by every name, or a mapping from each part name to its own mask, so a
        blueprint with a colour per part is checked part by part. ``names`` lists the parts a single mask
        shows; it is left out with a mapping.
        """
        if hasattr(mask, "items"):
            masks = {str(name): np.asarray(m, bool) for name, m in mask.items()}
            if names is not None:
                missing = [name for name in names if name not in masks]
                if missing:
                    raise ValueError(f"expect({view!r}) names parts its masks do not have: {', '.join(missing)}")
                masks = {str(name): masks[name] for name in names}
        else:
            if names is None:
                raise ValueError("one mask for every part needs names=")
            single = np.asarray(mask, bool)
            masks = {str(name): single for name in names}
        if not masks:
            raise ValueError(f"expect({view!r}) has no masks")
        shapes = {mask.shape for mask in masks.values()}
        if len(shapes) > 1:
            raise ValueError(f"expect({view!r}) needs every mask to have the same shape")
        expectation = Expectation(view=view, masks=masks,
                                  frame=None if frame is None else tuple(float(v) for v in frame))
        self.expectations.append(expectation)
        return expectation

    def asset(self, asset_class) -> Scene:
        """Build another asset, with its own cache next to this one, and return its scene.

        The other asset's drawings must already be on disk: building never reaches the network.
        """
        other = asset_class() if isinstance(asset_class, type) else asset_class
        remaining = None if self.budget is None else max(0.0, self.budget - self.spent)
        sub = Context(self.cache.parent / f"{other.name}.asset", model=self._instance or self.model,
                      budget=remaining, height=other.height, palette=self.palette)
        try:
            return other.build(sub)
        finally:
            self.spent += sub.spent
            self.calls += sub.calls

    def metres(self, value, scale: float | None = None):
        """Pixels to metres at ``scale`` (metres per pixel), by default the stage's ``scale``."""
        factor = self.scale if scale is None else float(scale)
        if np.isscalar(value):
            return float(value) * factor
        return np.asarray(value, float) * factor

    # --- the model behind a paid call ------------------------------------------------------------------------

    def _model_name(self) -> str | None:
        if isinstance(self.model, str):
            return self.model
        return getattr(self.model, "name", None)

    def _resolve_model(self, name: str):
        """The model to ask, building one from an id on first use; raises when there is none."""
        if self._instance is not None:
            return self._instance
        if isinstance(self.model, str):
            from manifoldx.imagegen import OpenRouterImage

            self._instance = OpenRouterImage(self.model, log=self.cache / "cost.csv")
            return self._instance
        if self.model is not None:
            self._instance = self.model
            return self._instance
        raise RuntimeError(
            f"no image model configured, so the image {name!r} cannot be drawn; give the asset a model, "
            "or pass model= to run()")
