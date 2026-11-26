"""Compare diffuse, specular and skybox regions with installed reference presets."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import StandardMaterial, sphere
from PIL import Image

WIDTH, HEIGHT = 180, 116
CASES = ('off', 'studio', 'sky', 'dark', 'sky-hidden')
REGIONS = {'diffuse': (61, 53, 78, 68), 'specular': (101, 53, 118, 68),
           'sky': (76, 12, 104, 30)}


def frame(case, disabled):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), mx.__file__
    e = mx.Engine('Offscreen IBL', width=WIDTH, height=HEIGHT, max_entities=8)
    if os.environ['MANIFOLDX_CORE'] == 'rust':
        assert type(e).__name__ == '_NativeEngine', type(e)
    e.background_color = (0.02, 0.033, 0.065)
    e.camera.set_pose(position=(0, 0.1, 4.8), target=(0, 0, 0))
    e.camera.fov = 45
    if case != 'off' and disabled != case:
        e.set_environment('sky' if case == 'sky-hidden' else case)
        e.environment.show_skybox = case != 'sky-hidden'
    geo = sphere(0.62, 48)
    e.spawn(Mesh(geo), Material(StandardMaterial('#ced3d2', roughness=0.73, metallic=0.0)), Transform(pos=(-0.70, 0, 0)))
    e.spawn(Mesh(geo), Material(StandardMaterial('#d9e0f0', roughness=0.20, metallic=1.0)), Transform(pos=(0.70, 0, 0)))
    return e.render_frame(supersample=1)


def region(im, bounds):
    x0, y0, x1, y1 = bounds
    return im[y0:y1, x0:x1].astype('i2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--disable', choices=CASES[1:])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = {name: frame(name, args.disable) for name in CASES}
    results = {}
    for name, image in images.items():
        path = args.output / f'{name}.png'
        Image.fromarray(image).save(path)
        result = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'mean_delta': {tag: float(np.abs(region(image, box) - region(images['off'], box)).mean())
                                 for tag, box in REGIONS.items()}}
        if args.reference:
            ref = np.asarray(Image.open(args.reference / path.name))
            error = np.abs(ref.astype('i2') - image)
            Image.fromarray(error.astype('u1')).save(args.output / f'{name}-diff.png')
            result['roi_mae'] = {tag: float(region(error, box).mean()) for tag, box in REGIONS.items()}
        results[name] = result
    (args.output / 'ibl.json').write_text(json.dumps({'roi': REGIONS, 'cases': results}, indent=2) + '\n')
    print(json.dumps(results))
    for name in ('studio', 'sky'):
        delta = results[name]['mean_delta']
        assert min(delta.values()) > 12, f'{name} environment did not light its three regions'
    assert results['dark']['mean_delta']['diffuse'] < results['studio']['mean_delta']['diffuse'] - 12
    assert results['sky-hidden']['mean_delta']['sky'] == 0
    assert np.array_equal(region(images['sky'], REGIONS['specular']),
                          region(images['sky-hidden'], REGIONS['specular']))
    if args.reference:
        assert all(max(result['roi_mae'].values()) <= 2 for result in results.values()), results


if __name__ == '__main__':
    main()
