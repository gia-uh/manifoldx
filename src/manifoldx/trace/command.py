"""`manifoldx trace scene.glb -o out.png`: render a glb with the path tracer."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def add_parser(sub):
    p = sub.add_parser("trace", help="render a .glb with the path tracer and write a PNG")
    p.add_argument("scene", metavar="scene.glb")
    p.add_argument("-o", "--output", required=True, metavar="out.png")
    p.add_argument("--camera", metavar="PX,PY,PZ:TX,TY,TZ[:FOV]",
                   help="eye, target and vertical field of view in degrees (default: the glb's camera)")
    p.add_argument("--size", default="1024x640", metavar="WxH")
    p.add_argument("--spp", type=int, default=64, help="samples per pixel (default 64)")
    p.add_argument("--seconds", type=float, help="instead of --spp, trace for this long")
    p.add_argument("--sun", metavar="X,Y,Z", help="direction towards the sun, y up (default 0.4,0.7,0.3)")
    p.add_argument("--turbidity", type=float, default=3.0, help="haze of the sky, 2 (clear) to 10")
    p.add_argument("--hdr", metavar="sky.hdr", help="light the scene with this equirect .hdr instead of a sky")
    p.add_argument("--scene-environment", action="store_true",
                   help="use the environment map stored in the glb instead of a sky")
    p.add_argument("--overrides", metavar="JSON|file.json",
                   help='material fields by node or material name, e.g. \'{"window-*": {"transmission": 1}}\'')
    p.add_argument("--bounces", type=int, default=6)
    p.add_argument("--exposure", default="auto", help='a multiplier, or "auto" (default)')
    p.add_argument("--clamp", type=float, default=0.0, help="cap each indirect sample's radiance (0: off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-denoise", action="store_true")
    return p


def _vec(text, n, name):
    parts = [float(v) for v in text.split(",")]
    if len(parts) != n:
        raise ValueError(f"{name} needs {n} comma-separated numbers, got {text!r}")
    return tuple(parts)


def parse_camera(text):
    from manifoldx.trace.scene import Camera

    fields = text.split(":")
    if len(fields) not in (2, 3):
        raise ValueError(f"--camera is PX,PY,PZ:TX,TY,TZ[:FOV], got {text!r}")
    fov = float(fields[2]) if len(fields) == 3 else 60.0
    return Camera(_vec(fields[0], 3, "--camera eye"), _vec(fields[1], 3, "--camera target"), fov=fov)


def run(args) -> int:
    from PIL import Image

    from manifoldx.trace import Environment, Scene, Tracer

    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
        camera = parse_camera(args.camera) if args.camera else None
        sun = _vec(args.sun, 3, "--sun") if args.sun else (0.4, 0.7, 0.3)
        overrides = None
        if args.overrides:
            text = Path(args.overrides).read_text() if args.overrides.endswith(".json") else args.overrides
            overrides = {k: {f: (tuple(v) if isinstance(v, list) else v) for f, v in fields.items()}
                         for k, fields in json.loads(text).items()}
        exposure = args.exposure if args.exposure == "auto" else float(args.exposure)
    except ValueError as e:
        print(f"manifoldx trace: {e}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    scene = Scene.from_glb(args.scene, overrides)
    if args.hdr:
        env = Environment.from_hdr(args.hdr)
    elif args.scene_environment:
        if scene.environment is None:
            print("manifoldx trace: the glb stores no environment map", file=sys.stderr)
            return 2
        env = scene.environment
    else:
        env = Environment.sky(sun, turbidity=args.turbidity)
    tracer = Tracer(scene, w, h, camera=camera, environment=env, max_bounces=args.bounces, seed=args.seed,
                    clamp=args.clamp)
    ready = time.perf_counter()
    if args.seconds:
        traced = 0.0
        while traced < args.seconds:
            traced += tracer.step(4)
    else:
        traced = tracer.step(args.spp)
    t1 = time.perf_counter()
    rgb = tracer.image(denoise=not args.no_denoise, exposure=exposure)
    Image.fromarray(rgb).save(args.output)
    t2 = time.perf_counter()
    print(f"{scene.triangle_count} triangles, BVH {tracer.build_seconds:.2f} s; {w}x{h}, {tracer.samples} spp in "
          f"{traced:.2f} s ({tracer.samples / traced:.1f} spp/s); load {ready - t0:.2f} s, image {t2 - t1:.2f} s "
          f"-> {args.output}")
    return 0
