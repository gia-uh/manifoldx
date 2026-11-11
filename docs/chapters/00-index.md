ManifoldX draws 3D scenes and simulations from Python. You keep your data in
numpy arrays, one row per thing in the world, and the engine draws every row
on the GPU through [wgpu](https://github.com/pygfx/wgpu-py). There is no scene
graph to walk and no shader to write before the first picture.

This guide follows one project from start to finish: a small town plaza, forty
metres on a side, with a fountain, benches, the buildings around it and the
people who cross it. Each chapter starts from a job the plaza needs done
and solves it with ManifoldX, bringing in the parts of the library the job
needs, simplest first. Chapter 14 puts the whole plaza together and adds
nothing new. The chapter after it builds the objects the plaza still lacks,
from drawings.

Every code block in this guide ran when the guide was built, against the
version of ManifoldX named at the top of the page. The frames you see were
drawn by the block above them, offscreen, on a software GPU. Each chapter's
finished program is in
[`docs/examples/`](https://github.com/gia-uh/manifoldx/tree/main/docs/examples),
written out from the chapter itself.

## Install

```bash
pip install "manifold-gfx[desktop]"   # a window on your desktop
pip install "manifold-gfx[offline]"   # frames and videos without a window
```

ManifoldX needs Python 3.13 and a GPU that speaks Vulkan, Metal or Direct3D 12.
On a machine without one, a software Vulkan driver such as Mesa's lavapipe is
enough for still frames.

## The chapters

1. A first frame: the engine, a camera, and something to look at.
2. People who walk: components and systems, the engine's data model.
3. Reacting to the visitor: events, input and commands.
4. Light and surface: materials, lights and shadows.
5. Sky, water and textures.
6. Modelling the buildings: meshes built in code.
7. Bodies and poses: people with a skeleton.
8. Measuring the plaza: charts, labels and fields in the scene.
9. A control panel: widgets inside the window.
10. Crowds on the GPU: compute kernels written in Python.
11. Assets in and out: OBJ, glTF and streaming.
12. A still worth printing: the path tracer.
13. Working with an agent: hot reload, the dev channel and the web page.
14. The plaza, end to end.
15. Models from drawings: assets built from an image model's drawings, linted and corrected.
