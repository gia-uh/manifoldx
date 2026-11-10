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


def top_livery(P, paint, out):
    # Top frame: u along the car (front at 0), v across it. Glass from the cowl to where the roof levels off, and
    # from where the roof drops to the tail.
    top = Image.new("RGB", (1024, 1024), tuple(int(c) for c in paint))
    L = P["dims"][0]
    xs, ys = np.array(P["side"]["x"]), np.array(P["side"]["top"])
    flat = np.where(ys > 0.97 * ys.max())[0]
    u = lambda x: (x + L / 2) / L * 1024
    t = ImageDraw.Draw(top)
    t.rectangle([u(P["side"]["belt"][0][0]), 150, u(xs[flat[0]]), 874], fill=GLASS)
    t.rectangle([u(xs[flat[-1]]), 200, 1024, 824], fill=GLASS)
    top.save(out / "tex-top.png")


def main(out):
    P = json.load(open(HERE / "profile-box.json"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
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
    top_livery(P, paint, out)
    d = ImageDraw.Draw(tex)
    d.rectangle([180, 120, 844, 420], fill=GLASS)        # rear window, in the rear frame (see box.py)
    d.rectangle([120, 470, 904, 500], fill=(150, 20, 24))  # the light bar across the tailgate
    tex.save(out / "tex-rear.png")
    json.dump({"paint": [int(c) for c in paint]}, open(out / "paint.json", "w"))
    print("paint", "#%02x%02x%02x" % tuple(paint))


def friday(out):
    """The Friday's photos have no usable mask: the side photo goes on as it is, and the front photo, of a green
    car, has its green paint swapped for the side photo's white."""
    P = json.load(open(HERE / "profile-friday.json"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    photos = "/home/apiad/Workspace/repos/briefings/dongfeng/modelos/friday-ev/imagenes"
    load = lambda name: np.asarray(Image.open(next(Path(photos).glob(name))).convert("RGB").resize((1400, 788)))
    side, front = load("03-*"), load("02-*")
    paint = np.median(side[400:450, 700:800].reshape(-1, 3), axis=0).astype(np.uint8)
    Image.fromarray(side).save(out / "tex-side-photo.png")
    hsv = np.asarray(Image.fromarray(front).convert("HSV")).astype(np.float32)
    green = (hsv[..., 0] > 70) & (hsv[..., 0] < 140) & (hsv[..., 1] > 60)
    shade = hsv[..., 2:3] / max(1.0, np.percentile(hsv[..., 2][green], 90))
    fixed = front.copy()
    fixed[green] = np.clip(paint * (0.55 + 0.45 * shade[green]), 0, 255).astype(np.uint8)
    Image.fromarray(fixed).save(out / "tex-front-photo.png")
    for name in ("side", "front"):
        Image.open(out / f"tex-{name}-photo.png").save(out / f"tex-{name}-poster.png")
    sp = P["side"]
    s = sp["scale_mm_per_px"]
    cx = np.interp(sp["wheels"][0]["x"], sp["anchors_mm"], sp["anchors_px"])
    cy = sp["ground"] - P["tyre_d"] / 2 / s
    r = 19 * 25.4 / 2 / s
    Image.fromarray(side).crop((cx - r, cy - r, cx + r, cy + r)).resize((512, 512)).save(out / "tex-wheel.png")
    top_livery(P, paint, out)
    rear = Image.new("RGB", (1024, 1024), tuple(int(c) for c in paint))
    d = ImageDraw.Draw(rear)
    d.rectangle([200, 170, 824, 360], fill=GLASS)
    d.rectangle([110, 420, 914, 440], fill=(150, 20, 24))
    rear.save(out / "tex-rear.png")
    print("paint", "#%02x%02x%02x" % tuple(paint))


if __name__ == "__main__":
    car = sys.argv[1] if len(sys.argv) > 1 else "box"
    (friday if car == "friday" else main)(HERE / "out" / car)
