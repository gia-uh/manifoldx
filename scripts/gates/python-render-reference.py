"""Capture the FR-02 flat sRGB material oracle from an installed Python engine."""

import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import FlatMaterial

os.environ["MANIFOLDX_CORE"] = "python"
assert sys.flags.isolated and sys.prefix != sys.base_prefix
package = Path(mx.__file__).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve())

engine = mx.Engine("FR-02 flat sRGB", width=32, height=32)
engine.camera.set_pose(position=(0, 0, 3), target=(0, 0, 0))
try:
    if "--without-material" not in sys.argv:
        engine.spawn(
            Mesh(mx.geometry.plane(10, 10)),
            Material(FlatMaterial("#cc2222")),
            Transform(pos=(0, 0, 0)),
        )
    rgb = engine.render_frame(supersample=1, dt=1 / 30)
    assert rgb.shape == (32, 32, 3) and rgb.dtype.name == "uint8"
    pixel = rgb[16, 16].tolist()
    payload = rgb.tobytes()
    ppm = b"P6\n32 32\n255\n" + payload
    Path(sys.argv[1]).write_bytes(ppm)
    print(
        json.dumps(
            {
                "module": str(package),
                "adapter": dict(engine._adapter.info),
                "target": str(engine._texture_format),
                "center_rgb": pixel,
                "ppm_sha256": hashlib.sha256(ppm).hexdigest(),
            },
            indent=2,
        )
    )
    assert all(abs(value - ref) <= 1 for value, ref in zip(pixel, (204, 34, 34))), pixel
finally:
    engine.close()
