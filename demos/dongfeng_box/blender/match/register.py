"""Measure a car GLB against the catalogue photos: silhouette, lamp shapes and paint colour.

Run: uv run --with trimesh --with scipy --with pillow --with numpy python register.py MODEL.glb OUT_DIR [--render-dir D]

For each photo the model is drawn with that photo's camera: the side photo's camera comes from its wheels (calib.py)
and is held fixed, so the side score is pure shape. The other photos start from a rough camera and are registered
by maximising silhouette IoU over the 7 camera parameters; a camera only bends the whole image (perspective,
scale, turn), so a local difference of shape, such as a flatter nose, survives registration and shows in the score.

Outputs per photo: IoU above the sills, contour distance (mean and 95th percentile, in px and mm), an overlay
(green: photo only, magenta: model only), the model's material regions outlined on the photo, and the regions
score (regions.py): IoU of paint and of non-paint where both silhouettes show car, with its overlay. The outline
alone says nothing about where windows, pillars and lamps sit, and a fit on it alone moves them freely. With
--render-dir also the internal edges (edge_stats): panel lines, creases and lamp edges of the matched render
against the photo's, as a mean distance both ways, with an overlay (green photo, magenta render, white both).
With --render-dir, the matched Blender renders (render_matched.py) give the paint colour comparison in CIELAB.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import minimize

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import calib  # noqa: E402
import regions  # noqa: E402

import os

PHOTOS = Path(os.environ.get("BOX_PHOTOS", "/home/apiad/Workspace/repos/briefings/dongfeng/modelos/box-430/imagenes"))
SCALE = 0.5  # work at 1400 px wide
W, H = int(calib.W * SCALE), int(calib.H * SCALE)
VIEWS = {
    "04-dongfeng-box-vista-lateral": "fixed",
    "02-dongfeng-box-vista-exterior-tres-cuartos": "register",
    "03-dongfeng-box-vista-frontal": "register",
}
FRONT_GUESS = [-6.0, 0.0, 1.0, 90.0, -3.0, 0.0, 4500.0]


def load(glb):
    """Triangles in Blender axes (X along the car, front -X; Y across; Z up) and a material name per triangle."""
    scene = trimesh.load(glb, force="scene")
    tris, mats = [], []
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph[node]
        g = scene.geometry[gname]
        v = trimesh.transform_points(g.vertices, tf)
        v = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])  # glTF Y-up back to Z-up
        tris.append(v[g.faces])
        name = getattr(getattr(g.visual, "material", None), "name", None) or gname
        mats += [name] * len(g.faces)
    return np.concatenate(tris), np.array(mats)


def project(params, pts):
    cx, cy, cz, yaw, pitch, roll, f = params
    right, up, fwd = calib.basis(yaw, pitch, roll)
    d = pts - np.array([cx, cy, cz])
    z = d @ fwd
    return np.column_stack([W / 2 + f * SCALE * (d @ right) / z, H / 2 - f * SCALE * (d @ up) / z]), z


def raster(params, tris, ids=None):
    """Silhouette (ids None) or an id image drawn far to near (painter's algorithm)."""
    uv, z = project(params, tris.reshape(-1, 3))
    uv, z = uv.reshape(-1, 3, 2), z.reshape(-1, 3)
    ok = (z > 0.1).all(1)
    img = Image.new("I" if ids is not None else "1", (W, H), 0)
    d = ImageDraw.Draw(img)
    order = np.argsort(-z.mean(1)) if ids is not None else np.arange(len(uv))
    for i in order:
        if ok[i]:
            d.polygon([tuple(p) for p in uv[i]], fill=int(ids[i]) if ids is not None else 1)
    return np.asarray(img)


def sill_cut(params):
    """Image rows below the projected sill line (z = 0.24 m at the car's middle) hold ground shadow: ignore them."""
    uv, _ = project(params, np.array([[x, y, 0.24] for x in np.linspace(-2, 2, 9) for y in (-0.9, 0.9)]))
    return int(uv[:, 1].max())


def iou(a, b, cut):
    a, b = a[:cut], b[:cut]
    return (a & b).sum() / max((a | b).sum(), 1)


def contour_stats(photo, model, cut, params):
    edge = lambda m: m[:cut] & ~ndimage.binary_erosion(m[:cut])
    ep, em = edge(photo), edge(model)
    dist = ndimage.distance_transform_edt(~em)[ep]
    centre = np.linalg.norm(np.array(params[:3]))  # the car's centre is at the origin
    mm_per_px = centre / (params[6] * SCALE) * 1000
    return {"mean_px": float(dist.mean()), "p95_px": float(np.percentile(dist, 95)),
            "mean_mm": float(dist.mean() * mm_per_px), "p95_mm": float(np.percentile(dist, 95) * mm_per_px)}


def edges(rgb, region, top=0.12):
    """The strongest internal lines inside region: pixels whose grey-level gradient is in the top fraction there."""
    g = ndimage.gaussian_filter(rgb.astype(float).mean(-1), 1.5)
    mag = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    return region & (mag >= np.quantile(mag[region], 1 - top))


def edge_stats(photo_rgb, render_rgb, pmask, model, cut, params):
    """Panel lines, creases, lamp and window edges: mean distance between the photo's and the render's internal
    lines, both ways, in mm. Silhouette and region scores cannot see a fender crease or a lamp's volume; this can."""
    region = ndimage.binary_erosion(pmask & model, iterations=6)
    region[cut:] = False
    ep, er = edges(photo_rgb, region), edges(render_rgb, region)
    d_pr = ndimage.distance_transform_edt(~er)[ep].mean()
    d_rp = ndimage.distance_transform_edt(~ep)[er].mean()
    mm_per_px = np.linalg.norm(np.array(params[:3])) / (params[6] * SCALE) * 1000
    ov = (photo_rgb * 0.35).astype(np.uint8)
    ov[ep] = (40, 220, 60)
    ov[er] = (230, 40, 200)
    ov[ep & er] = (255, 255, 255)
    return {"mean_px": round(float((d_pr + d_rp) / 2), 1), "mean_mm": round(float((d_pr + d_rp) / 2 * mm_per_px), 1)}, ov


def lab(rgb):
    c = rgb / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[:, 1] - 16
    a, b = 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])
    return np.column_stack([L, a, b, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360])


def colour_stats(pixels):
    q = lab(pixels.reshape(-1, 3).astype(float))
    return {k: [round(float(np.percentile(q[:, i], p)), 1) for p in (10, 50, 90)]
            for i, k in enumerate(("L", "a", "b", "chroma", "hue"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("out")
    ap.add_argument("--render-dir")
    ap.add_argument("--cams", help="reuse the cameras of an earlier report.json instead of registering")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tris, mats = load(args.glb)
    names = sorted(set(mats))
    ids = np.array([names.index(m) + 1 for m in mats])
    cams = json.load(open(HERE / "cameras.json"))
    report = {"model": args.glb, "triangles": int(len(tris)), "views": {}}
    for photo, mode in VIEWS.items():
        img, pmask, cls = regions.classes(PHOTOS / f"{photo}.webp", width=W)
        if photo in cams:
            c = cams[photo]
            p0 = c["position"] + [c["yaw"], c["pitch"], c["roll"], c["focal_px"]]
        else:
            p0 = FRONT_GUESS
        params = np.array(p0, float)
        if args.cams and mode != "fixed":
            params, mode = np.array(json.load(open(args.cams))["views"][photo]["params"]), "reused"
        if mode == "register":
            # Coarse to fine: Powell on 1 - IoU, with steps scaled to each parameter's sensitivity.
            scale = np.array([0.2, 0.2, 0.1, 2.0, 1.0, 1.0, 200.0])
            cost = lambda s: 1 - iou(raster(params + s * scale, tris).astype(bool), pmask, sill_cut(params + s * scale))
            res = minimize(cost, np.zeros(7), method="Powell", options={"xtol": 0.02, "ftol": 1e-4, "maxfev": 400})
            params = params + res.x * scale
        model = raster(params, tris).astype(bool)
        cut = sill_cut(params)
        view = {"camera": "wheels (fixed)" if mode == "fixed" else "registered on silhouette",
                "params": [round(float(v), 3) for v in params], "iou_above_sills": round(float(iou(model, pmask, cut)), 4),
                "contour": {k: round(v, 1) for k, v in contour_stats(pmask, model, cut, params).items()}}
        # Overlay: green photo only, magenta model only, grey both.
        ov = (img * 0.45).astype(np.uint8)
        ov[pmask & ~model] = (40, 220, 60)
        ov[model & ~pmask] = (230, 40, 200)
        ov[cut:] = (ov[cut:] * 0.4).astype(np.uint8)
        Image.fromarray(ov).save(out / f"sil-{photo[:2]}.png")
        # Material regions of the model drawn as outlines over the photo.
        idimg = raster(params, tris, ids)
        view["regions"] = regions.score(cls, idimg, names, cut)
        Image.fromarray(regions.overlay(img, cls, idimg, names, cut)).save(out / f"paint-{photo[:2]}.png")
        lines = img.copy()
        palette = {"DRL": (255, 255, 0), "Lens": (0, 255, 255), "TailLED": (255, 0, 0), "TailLens": (255, 120, 0),
                   "Glass": (0, 120, 255), "GlossBlack": (255, 0, 255), "MatteTrim": (0, 255, 0), "Chrome": (255, 255, 255)}
        for n, col in palette.items():
            if n in names:
                m = idimg == names.index(n) + 1
                lines[m & ~ndimage.binary_erosion(m)] = col
        Image.fromarray(lines).save(out / f"regions-{photo[:2]}.png")
        # Paint colour: photo pixels where the model shows paint, both masks agreeing, minus specular highlights.
        if "Paint" in names:
            paint = (idimg == names.index("Paint") + 1) & pmask
            paint = ndimage.binary_erosion(paint, iterations=3)
            hsv = np.asarray(Image.fromarray(img).convert("HSV"))
            sel = paint & (hsv[..., 2] < 245)
            view["paint_photo"] = colour_stats(img[sel])
            if args.render_dir:
                r = Path(args.render_dir) / f"matched-{photo[:2]}.png"
                if r.exists():
                    rimg = np.asarray(Image.open(r).convert("RGB").resize((W, H)))
                    view["paint_render"] = colour_stats(rimg[sel])
                    view["edges"], ov = edge_stats(img, rimg, pmask, model, cut, params)
                    Image.fromarray(ov).save(out / f"edges-{photo[:2]}.png")
        report["views"][photo] = view
        print(photo[:2], json.dumps({k: v for k, v in view.items() if k != "params"}), flush=True)
    json.dump(report, open(out / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
