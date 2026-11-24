"""Record wheel metadata and assert the expected interpreter and import provenance."""

import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import sysconfig
from pathlib import Path

wheel = Path(sys.argv[1])
dist = importlib.metadata.distribution("manifold-gfx")
prefix = Path(sys.prefix).resolve()
assert sys.flags.isolated and sys.prefix != sys.base_prefix
gil_before = sys._is_gil_enabled()
origins = {}
for name in (
    "manifoldx",
    "manifoldx.cli",
    "manifoldx._core_native",
    "numpy",
    "wgpu",
    "rendercanvas",
):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(prefix), (name, path, prefix)
    origins[name] = str(path)
gil_after = sys._is_gil_enabled()
with wheel.open("rb") as stream:
    sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
print(
    json.dumps(
        {
            "wheel": wheel.name,
            "sha256": sha256,
            "compressed_bytes": wheel.stat().st_size,
            "wheel_tags": [
                line.removeprefix("Tag: ")
                for line in dist.read_text("WHEEL").splitlines()
                if line.startswith("Tag: ")
            ],
            "python": sys.version,
            "platform": platform.platform(),
            "free_threaded_abi": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            "gil_before_imports": gil_before,
            "gil_after_imports": gil_after,
            "origins": origins,
            "dependencies": {
                name: importlib.metadata.version(name) for name in ("numpy", "wgpu", "rendercanvas")
            },
        },
        indent=2,
    ),
    flush=True,
)
assert gil_after == gil_before, "wheel imports changed the GIL state"
