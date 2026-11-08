import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
VALIDATE = ("const v=require('gltf-validator');const fs=require('fs');"
            "v.validateBytes(new Uint8Array(fs.readFileSync(process.argv[1])))"
            ".then(r=>console.log(JSON.stringify(r.issues)))")


def sample(tmp_path):
    """Every writer feature in one file."""
    from PIL import Image

    from manifoldx.camera import Camera
    from manifoldx.gltf import Node, export_gltf
    from manifoldx.ibl import EnvironmentMap
    from manifoldx.resources import (BasicMaterial, DirectionalLight, FlatMaterial, PointLight, SpotLight,
                                     StandardMaterial, cube)
    from manifoldx.textures import TextureHandle

    img = tmp_path / "t.png"
    Image.new("RGB", (8, 8), (10, 200, 30)).save(img)
    tex = TextureHandle(id=1, texture=None, view=None, sampler=None, size=(8, 8), source=img)
    geo = cube(1, 1, 1)
    geo = geo | {"uvs": geo["positions"][:, :2].copy()}
    out = tmp_path / "all.glb"
    export_gltf(out, [Node(geo, StandardMaterial("#ffffff", albedo_map=tex), name="tex", extras={"pick": "a"}),
                      Node(geo, FlatMaterial("#ff0000"), pos=(2, 0, 0)),
                      Node(geo, BasicMaterial((1, 1, 1, 0.5)), pos=(4, 0, 0))],
                camera=Camera(position=(0, 3, 8)),
                lights=[DirectionalLight("#ffffff", 2.0, (0, -1, -1)), PointLight("#ffffff", 3, (0, 4, 0)),
                        SpotLight("#ffffff", 3, (0, 4, 4), (0, -1, -1), 0.2, 0.4, distance=30)],
                environment=EnvironmentMap.from_sky((0.4, 0.5, 0.7), (0.6, 0.6, 0.7), (0.3, 0.3, 0.2)),
                scene_extras={"fog": {"start": 1.0, "end": 9.0}})
    return out


@pytest.mark.skipif(shutil.which("npm") is None or shutil.which("node") is None, reason="needs node and npm")
def test_khronos_validator_finds_no_errors(tmp_path):
    out = sample(tmp_path)
    tools = tmp_path / "validator"
    install = subprocess.run(["npm", "install", "--silent", "--prefix", str(tools), "gltf-validator"],
                             capture_output=True, text=True, timeout=300)
    if install.returncode != 0:
        pytest.skip(f"npm install gltf-validator failed: {install.stderr[-300:]}")
    res = subprocess.run(["node", "-e", VALIDATE, str(out)], cwd=tools, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    issues = json.loads(res.stdout)
    errors = [m for m in issues["messages"] if m["severity"] == 0]
    assert issues["numErrors"] == 0, errors


@pytest.mark.skipif(shutil.which("blender") is None, reason="needs blender")
def test_blender_imports_the_example(tmp_path):
    out = tmp_path / "terrain.glb"
    subprocess.run(["uv", "run", "python", str(ROOT / "examples" / "gltf_export.py"), str(out)], check=True,
                   cwd=ROOT, capture_output=True, timeout=300)
    script = ("import bpy;bpy.ops.wm.read_factory_settings(use_empty=True);"
              f"bpy.ops.import_scene.gltf(filepath={str(out)!r});"
              "print('COUNTS', len(bpy.data.meshes), len(bpy.data.materials))")
    res = subprocess.run(["blender", "-b", "--factory-startup", "--python-expr", script],
                         capture_output=True, text=True, timeout=300)
    line = next(ln for ln in res.stdout.splitlines() if ln.startswith("COUNTS"))
    assert line == "COUNTS 1 1", res.stdout[-2000:]
