"""Assemble the site: index.html, three.js r170 (vendored, minified) and one GLB per car under cars/."""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
THREE = Path("/home/apiad/Workspace/repos/webfair/viewer/vendor/three")
CARS = {"box": "box/box-poster.glb", "friday": "friday/box-photo.glb", "voyah": "voyah/box-photo.glb",
        "rich7": "rich7/box-photo.glb", "m5": "m5/box-photo.glb"}
out = Path(sys.argv[1])
(out / "cars").mkdir(parents=True, exist_ok=True)
shutil.copy(HERE / "game.html", out / "index.html")
for rel in ("three.module.js", "addons/loaders/GLTFLoader.js", "addons/utils/BufferGeometryUtils.js"):
    (out / "vendor/three" / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(THREE / rel, out / "vendor/three" / rel)
for key, rel in CARS.items():
    shutil.copy(HERE.parent / "out" / rel, out / "cars" / f"{key}.glb")
print(sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) // 1024, "KiB in", out)
