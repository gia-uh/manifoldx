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


def livery(key):
    """Textures for a car without usable side and front photos (or with a side photo only, the Voyah), drawn from its
    profile: paint, glass above the beltline, pillars, lights, dark cladding, and a drawn wheel face."""
    P = json.load(open(HERE / f"profile-{key}.json"))
    out = HERE / "out" / key
    out.mkdir(parents=True, exist_ok=True)
    sp, fp = P["side"], P["front"]
    L, W, H = P["dims"]
    paint = tuple(int(P["paint"][i:i + 2], 16) for i in (1, 3, 5))
    dark, light, red, trim = GLASS, (235, 240, 245), (170, 20, 26), (28, 30, 34)
    s = sp["scale_mm_per_px"]
    X = lambda x: float(np.interp(x, sp["anchors_mm"], sp["anchors_px"]))
    Y = lambda y: sp["ground"] - y / s
    xs, top = np.array(sp["x"]), np.array(sp["top"])
    if P["designed"]:
        im = Image.new("RGB", (1400, sp["photo_h"]), paint)
        d = ImageDraw.Draw(im)
        (bx0, by0), (bx1, by1) = sp["belt"][0], sp["belt"][-1]
        w0, w1 = sp["windows"]
        # Glass: between the beltline and 70 mm under the roof line, split by pillars.
        span = np.linspace(w0, w1, 60)
        roof = np.interp(span, xs, top) - 70
        belt = np.interp(span, [bx0, bx1], [by0, by1]) + 20
        d.polygon([(X(x), Y(y)) for x, y in zip(span, roof)] + [(X(x), Y(y)) for x, y in zip(span[::-1], belt[::-1])],
                  fill=dark)
        for k in range(1, 3 if key == "rich7" else 2):
            px = X(w0 + (w1 - w0) * k / (3 if key == "rich7" else 2))
            d.rectangle([px - 6, 0, px + 6, Y(by0)], fill=trim if key == "rich7" else paint)
        d.rectangle([0, Y(0.30 * H), 1400, Y(0)], fill=trim if key == "rich7" else paint)  # lower cladding
        d.rectangle([0, Y(0.22 * H), 1400, Y(0.20 * H)], fill=trim)
        for wx in (w["x"] for w in sp["wheels"]):
            r = (P["tyre_d"] / 2 + 140) / s
            d.ellipse([X(wx) - r, Y(P["tyre_d"] / 2) - r, X(wx) + r, Y(P["tyre_d"] / 2) + r], fill=trim)
        d.rectangle([X(-L / 2) - 5, Y(0.62 * H), X(-L / 2 + 160), Y(0.55 * H)], fill=light)   # headlight
        d.rectangle([X(L / 2 - 120), Y(0.66 * H), X(L / 2) + 5, Y(0.52 * H)], fill=red)       # tail light
        for frac in ((0.36, 0.56) if key == "rich7" else (0.36, 0.62)):                       # door seams
            d.line([(X(-L / 2 + frac * L), Y(0.23 * H)), (X(-L / 2 + frac * L), Y(by1))], fill=(90, 96, 104), width=2)
    else:
        photo = np.asarray(Image.open(next(Path(
            "/home/apiad/Workspace/repos/briefings/dongfeng/modelos/voyah-free/imagenes").glob("03-*"))).convert("RGB")
            .resize((1400, sp["photo_h"]))).copy()
        hsv = np.asarray(Image.fromarray(photo).convert("HSV")).astype(np.float32)
        photo[(hsv[..., 2] > 150) & (hsv[..., 1] < 40)] = paint  # the white backdrop becomes paint
        im = Image.fromarray(photo)
    im.save(out / "tex-side-photo.png"); im.save(out / "tex-side-poster.png")
    # Front: windshield, grille, headlights and bumper, in the front frame (centre 700, ground fp["ground"]).
    fy = lambda y: fp["ground"] - y / fp["scale_y"]
    fx = lambda z: 700 + z / fp["scale_x"]
    f = Image.new("RGB", (1400, fp["photo_h"]), paint)
    d = ImageDraw.Draw(f)
    hw = lambda y: float(np.interp(y, fp["y"], fp["half_width"]))
    belt_h = sp["belt"][0][1]
    d.polygon([(fx(-hw(belt_h) * 0.86), fy(belt_h + 40)), (fx(hw(belt_h) * 0.86), fy(belt_h + 40)),
               (fx(hw(H * 0.95) * 0.86), fy(H * 0.95)), (fx(-hw(H * 0.95) * 0.86), fy(H * 0.95))], fill=dark)
    gh = (0.30, 0.48) if key != "m5" else (0.22, 0.38)
    d.rectangle([fx(-W * 0.26), fy(gh[1] * H), fx(W * 0.26), fy(gh[0] * H)], fill=trim)
    for side in (-1, 1):
        x0, x1 = sorted((fx(side * W * 0.30), fx(side * W * 0.44)))
        d.rectangle([x0, fy(0.56 * H), x1, fy(0.50 * H)], fill=light)
    if key == "voyah":
        d.rectangle([fx(-W * 0.44), fy(0.555 * H), fx(W * 0.44), fy(0.545 * H)], fill=light)  # the light bar
    d.rectangle([fx(-W * 0.48), fy(0.22 * H), fx(W * 0.48), fy(0.0)], fill=trim if key != "m5" else (200, 204, 208))
    f.save(out / "tex-front-photo.png"); f.save(out / "tex-front-poster.png")
    # Wheel face: a dark rim with spokes and a light lip.
    wimg = Image.new("RGB", (512, 512), (24, 25, 28))
    d = ImageDraw.Draw(wimg)
    d.ellipse([6, 6, 506, 506], fill=(150, 154, 160) if key == "m5" else (40, 42, 46), outline=(190, 194, 200), width=10)
    for k in range(5 if key != "m5" else 8):
        a = k * 2 * np.pi / (5 if key != "m5" else 8)
        d.line([(256, 256), (256 + 230 * np.cos(a), 256 + 230 * np.sin(a))], fill=(120, 124, 130) if key != "voyah" else (30, 31, 34), width=34)
    d.ellipse([200, 200, 312, 312], fill=(60, 62, 66))
    if key == "voyah":
        d.rectangle([60, 300, 150, 420], fill=(190, 20, 24))  # red callipers behind the spokes
    wimg.save(out / "tex-wheel.png")
    top_livery(P, np.array(paint), out)
    rear = Image.new("RGB", (1024, 1024), paint)
    d = ImageDraw.Draw(rear)
    d.rectangle([200, 140, 824, 380], fill=GLASS)
    d.rectangle([90, 430, 934, 452], fill=red)
    rear.save(out / "tex-rear.png")
    print(key, "livery done")


if __name__ == "__main__":
    car = sys.argv[1] if len(sys.argv) > 1 else "box"
    if car in ("voyah", "rich7", "m5"):
        livery(car)
    else:
        (friday if car == "friday" else main)(HERE / "out" / car)
