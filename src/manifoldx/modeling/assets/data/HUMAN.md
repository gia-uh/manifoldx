# The humanoid section tables

`human_female.npz` and `human_male.npz` are the two base bodies every humanoid is built from
(`manifoldx.modeling.assets.human_sections`). Each was read once from a mesh an image-to-mesh model made from
drawings; the meshes themselves are not kept here (about 4.7 MB each). This is how to make them again.

## 1. Drawings

FLUX.2 klein 4B, 896 x 1152, 4 steps. The front view, from text alone (for the male, "an adult man of
average build"):

```text
Full-body orthographic view exactly from the front, perfectly symmetric, of an adult woman of average build.
A smooth matte uniform light grey clay mannequin, an artist's neutral base mesh for a 3D game character:
completely bald smooth head, no hair, no clothes, no underwear lines, no shoes, smooth simplified anatomy,
no visible muscles, no abs, no genitals, no nipples, simple mitten-like hands with the fingers together,
simple feet, a plain smooth face with only a soft hint of nose, brow and lips. A-pose: both arms held
straight out diagonally downward at 40 degrees from the torso, hands well away from the hips, a wide gap of
white background between each arm and the torso, legs apart with a gap of white background between them,
looking straight ahead. Normal average adult proportions, slightly soft, not athletic, not a caricature.
The whole figure from head to feet visible with margin, centred. Plain flat white background, no shadow,
no ground, no text, even soft light
```

The back and the profile are drawn from the chosen front as the only reference: "This same clay figure seen
exactly from behind ..." and "This same clay figure turned 90 degrees to stand in strict side profile, nose
pointing at the left edge of the image ...", each followed by "smooth matte uniform light grey clay mannequin,
completely bald, no hair, no clothes, no visible muscles, mitten-like hands, simple feet. Same proportions,
same size, same A-pose as the reference image. One single view of one figure, not a sheet, orthographic, the
whole figure from head to feet visible with margin, centred. Plain flat white background, no shadow, no
ground, no text, even soft light".

Seeds kept: female front 11, male front 12, both backs 22, both profiles 21. The profile looks left; mirror it
so it looks right before the next step.

## 2. Mesh

Pixal3D multi-view (front at azimuth 0, the mirrored profile at 90, the back at 180, the profile at 270), with
the background removed, at resolution 1024, decimated to 150 000 faces. The result carries an inner shell and
loose bits; `read_sections` keeps the outer surface.

## 3. Table

```bash
uv run python scripts/human_sections.py female.glb src/manifoldx/modeling/assets/data/human_female.npz --stature 1.63 --body female
uv run python scripts/human_sections.py male.glb src/manifoldx/modeling/assets/data/human_male.npz --stature 1.75 --body male
```

Each takes about 20 s.
