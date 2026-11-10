"""How well the model's side silhouette matches the photo's: IoU in the side photo's pixel frame."""

import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import box  # noqa: E402
from mask import car_mask  # noqa: E402

sp = box.P["side"]
s = sp["scale_mm_per_px"]
to_px = lambda p: np.column_stack([np.interp(p[:, 0] / box.MM, sp["anchors_mm"], sp["anchors_px"]),
                                   sp["ground"] - p[:, 1] / box.MM / s])
_, photo = car_mask(f"{box.HERE}/../../../briefings/dongfeng/modelos/box-430/imagenes/04-dongfeng-box-vista-lateral.webp")
sil = Image.new("1", (photo.shape[1], photo.shape[0]), 0)
d = ImageDraw.Draw(sil)
mesh = box.body()
tri = to_px(mesh.positions)[mesh.faces]
for t in tri:
    d.polygon([tuple(v) for v in t], fill=1)
for w in sp["wheels"]:
    cx, cy = to_px(np.array([[w["x"] * box.MM, box.TYRE_R, 0]]))[0]
    r = box.TYRE_R / box.MM / s
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=1)
model = np.asarray(sil, dtype=bool)
iou = (model & photo).sum() / (model | photo).sum()
diff = np.zeros(photo.shape + (3,), np.uint8) + 235
diff[model & photo] = (150, 170, 190)
diff[photo & ~model] = (220, 60, 60)
diff[model & ~photo] = (40, 120, 220)
Image.fromarray(diff).save(sys.argv[1])
cut = int(sp["ground"] - (box.TYRE_R / box.MM * 2 + 90) / s)  # above the arches' tops: no shadow, no arch insides
upper = (model[:cut] & photo[:cut]).sum() / (model[:cut] | photo[:cut]).sum()
print(f"above the arches IoU {upper:.3f}")
print(f"side silhouette IoU {iou:.3f}; photo only {(photo & ~model).sum()} px, model only {(model & ~photo).sum()} px")
