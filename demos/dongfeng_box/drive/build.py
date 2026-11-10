"""Bundle game.html with three.js, its GLTF loader and the cars' GLBs into one offline HTML file."""

import base64
import sys
from pathlib import Path

HERE = Path(__file__).parent
THREE = Path("/home/apiad/Workspace/repos/webfair/viewer/vendor/three")  # r170, minified
b64 = lambda p: base64.b64encode(Path(p).read_bytes()).decode()
page = (HERE / "game.html").read_text()
for key, path in {
    "__THREE__": THREE / "three.module.js",
    "__BGU__": THREE / "addons/utils/BufferGeometryUtils.js",
    "__GLTF__": THREE / "addons/loaders/GLTFLoader.js",
    "__CAR_BOX__": HERE.parent / "out/box/box-poster.glb",
    "__CAR_FRIDAY__": HERE.parent / "out/friday/box-photo.glb",
}.items():
    page = page.replace(key, b64(path))
out = Path(sys.argv[1])
out.write_text(page)
print(out.stat().st_size // 1024, "KiB")
