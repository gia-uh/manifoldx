"""Blender stills of a saved asset, when Blender is installed.

``toon.py`` draws two-band cel shading with an inverted-hull outline; ``real.py`` draws a photo-like still with
Cycles. Both take a ``model.glb`` and, when it exists, the ``albedo.png`` beside it. Nothing here runs unless a
``blender`` executable is on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SHOTS = ("three-quarter", "front", "side", "back", "above")


class BlenderMissing(RuntimeError):
    """No ``blender`` executable is on PATH."""


def available() -> bool:
    """Whether a ``blender`` executable is on PATH."""
    return shutil.which("blender") is not None


def render(model, out_dir, style: str = "toon", albedo=None, ground: str = "sea") -> list[Path]:
    """Render the five stills of ``model`` into ``out_dir`` and return the PNGs Blender wrote.

    ``style`` is ``"toon"`` or ``"photo"``; ``albedo`` is a texture to wrap the model in, if there is one;
    ``ground`` is ``"sea"`` or ``"floor"`` for the photo still.
    """
    executable = shutil.which("blender")
    if executable is None:
        raise BlenderMissing("no blender executable is on PATH; install Blender to render stills")
    script = Path(__file__).resolve().parent / ("real.py" if style in ("photo", "real") else "toon.py")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texture = str(albedo) if albedo is not None and Path(albedo).exists() else "-"
    command = [executable, "-b", "--python", str(script), "--", str(model), texture, str(out), str(ground)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError("blender failed: " + (" / ".join(tail) or f"exit {done.returncode}"))
    return [out / f"{shot}.png" for shot in SHOTS if (out / f"{shot}.png").exists()]
