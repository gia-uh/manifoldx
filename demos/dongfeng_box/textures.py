"""Textures for the BOX, cut from its photos into out/tex-*.png, plus the paint colour.

Side and front textures are the photos themselves, in their own pixel frame; box.py projects UVs into that frame.
Pixels off the car become paint, so a mesh slightly larger than the photo's silhouette does not show the backdrop.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mask import car_mask

HERE = Path(__file__).parent
PHOTOS = "/home/apiad/Workspace/repos/briefings/dongfeng/modelos/box-430/imagenes"
GLASS = (22, 26, 32)


def paint_colour(img, mask):
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.float32)
    paint = mask & (hsv[..., 1] > 38) & (hsv[..., 2] > 75)
    return np.median(img[paint], axis=0).astype(np.uint8)


def posterize(img, mask, paint):
    """The sober variant: paint becomes one flat colour, dark becomes glass, bright trim stays as it is."""
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.float32)
    out = img.copy()
    out[mask & (hsv[..., 1] > 38) & (hsv[..., 2] > 75)] = paint
    out[mask & (hsv[..., 2] <= 75)] = GLASS
    out[~mask] = paint
    return out


def main(out):
    P = json.load(open(HERE / "profile.json"))
    out = Path(out)
    side, smask = car_mask(f"{PHOTOS}/04-dongfeng-box-vista-lateral.webp")
    paint = paint_colour(side, smask)
    # The tyres and the dark inside the arches: the mesh's arches are drawn, so whatever of them shows in the photo
    # would land on the body as dark smears. Paint over a disc round each wheel.
    sp = P["side"]
    s = sp["scale_mm_per_px"]
    yy, xx = np.mgrid[: side.shape[0], : side.shape[1]]
    for w in sp["wheels"]:
        cx = np.interp(w["x"], sp["anchors_mm"], sp["anchors_px"])
        cy = sp["ground"] - P["tyre_d"] / 2 / s
        smask &= (xx - cx) ** 2 + (yy - cy) ** 2 > ((P["tyre_d"] / 2 + 85) / s) ** 2
    photo = side.copy()
    photo[~smask] = paint
    Image.fromarray(photo).save(out / "tex-side-photo.png")
    Image.fromarray(posterize(side, smask, paint)).save(out / "tex-side-poster.png")
    front = np.asarray(Image.open(f"{PHOTOS}/03-dongfeng-box-vista-frontal.webp").convert("RGB").resize((1400, 933)))
    Image.fromarray(front).save(out / "tex-front-photo.png")
    fmask = np.zeros(front.shape[:2], bool)
    fmask[60:870, 140:1260] = True
    Image.fromarray(posterize(front, fmask, paint)).save(out / "tex-front-poster.png")
    # The wheel face: the front wheel's rim, cut from the side photo as a square.
    cx = np.interp(sp["wheels"][0]["x"], sp["anchors_mm"], sp["anchors_px"])
    cy = sp["ground"] - P["tyre_d"] / 2 / s
    r = 17 * 25.4 / 2 / s
    Image.fromarray(side).crop((cx - r, cy - r, cx + r, cy + r)).resize((512, 512)).save(out / "tex-wheel.png")
    # Rear and top have no photo: a flat livery in paint, with the glass drawn where the profiles put it.
    tex = Image.new("RGB", (1024, 1024), tuple(int(c) for c in paint))
    # Top frame: u along the car (front at 0), v across it. Glass from the cowl to where the roof levels off, and
    # from where the roof drops to the tail.
    top = Image.new("RGB", (1024, 1024), tuple(int(c) for c in paint))
    L = P["dims"][0]
    xs, ys = np.array(sp["x"]), np.array(sp["top"])
    flat = np.where(ys > 0.97 * ys.max())[0]
    u = lambda x: (x + L / 2) / L * 1024
    t = ImageDraw.Draw(top)
    t.rectangle([u(sp["belt"][0][0]), 150, u(xs[flat[0]]), 874], fill=GLASS)
    t.rectangle([u(xs[flat[-1]]), 200, 1024, 824], fill=GLASS)
    top.save(out / "tex-top.png")
    d = ImageDraw.Draw(tex)
    d.rectangle([180, 120, 844, 420], fill=GLASS)        # rear window, in the rear frame (see box.py)
    d.rectangle([120, 470, 904, 500], fill=(150, 20, 24))  # the light bar across the tailgate
    tex.save(out / "tex-rear.png")
    json.dump({"paint": [int(c) for c in paint]}, open(out / "paint.json", "w"))
    print("paint", "#%02x%02x%02x" % tuple(paint))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "out")
