import numpy as np
from PIL import Image

from manifoldx.cli import main
from manifoldx.gltf import Node, build_glb
from manifoldx.resources import StandardMaterial
from manifoldx.trace.scenes import box


def _glb(tmp_path):
    data, _ = build_glb([Node(box((0, 0, 0), (1, 1, 1)), StandardMaterial(color="#c08040"), name="cube")])
    path = tmp_path / "cube.glb"
    path.write_bytes(data)
    return path


def test_trace_writes_a_png_of_the_requested_size(tmp_path, device, capsys):
    out = tmp_path / "out.png"
    code = main(["trace", str(_glb(tmp_path)), "-o", str(out), "--size", "24x16", "--spp", "2",
                 "--camera", "2,1.5,2:0,0,0:50"])
    assert code == 0
    img = np.asarray(Image.open(out))
    assert img.shape == (16, 24, 3) and img.std() > 0
    assert "2 spp" in capsys.readouterr().out


def test_a_bad_camera_is_refused(tmp_path, capsys):
    code = main(["trace", str(_glb(tmp_path)), "-o", str(tmp_path / "x.png"), "--camera", "1,2:0,0,0"])
    assert code == 2
    assert "--camera eye needs 3" in capsys.readouterr().err
