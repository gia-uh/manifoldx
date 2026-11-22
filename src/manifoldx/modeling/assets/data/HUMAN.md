# The humanoid section tables

`human_female.npz` and `human_male.npz` are the two base bodies every humanoid is built from
(`manifoldx.modeling.assets.human_sections`); `human_<body>_<variant>.npz` are the same bodies drawn heavy,
muscular and old, which `manifoldx.modeling.assets.human` blends in as blend shapes. Each was read once from a mesh an image-to-mesh model made from
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

## 4. Variants

Each variant starts from its base body's chosen front, edited with klein (the base front as the only
reference, 4 steps, 896 x 1152). The edit prompt is one sentence for the variant, followed by "Keep exactly the
same A-pose, both arms held straight out diagonally downward at 40 degrees with a wide gap of white background
between each arm and the torso, legs apart with a gap between them, the same framing, size, camera and lighting.
Same smooth matte uniform light grey clay, completely bald, no hair, no clothes, no genitals, no nipples,
mitten-like hands, simple feet. Orthographic view exactly from the front, symmetric. Plain flat white
background, no shadow, no ground, no text":

- heavy: "Edit this image: make this same clay figure overweight, with a large round belly that sticks out,
  thicker waist and love handles, fuller thighs and upper arms, a double chin and a fuller face; the shoulders
  and the height stay the same."
- muscular: "Edit this image: make this same clay figure very athletic and muscular, with broad shoulders, big
  rounded deltoids and biceps, a wide chest, a narrow waist, muscular thighs and calves; the height stays the
  same."
- elder: "Edit this image: make this same clay figure an old person of about eighty: a hunched upper back with
  the head pushed forward and lowered, rounded slumped shoulders, a slightly sagging belly, thinner arms and
  legs, a little shorter."

The back and the profile are drawn from the edited front with the base body's back and profile prompts, the
variant's shape inserted after the pose ("... exactly as in the reference: an overweight person with a large round
belly sticking out far in front and a big rounded backside"; "a very muscular athletic person with a big chest,
broad back and thick limbs"; "an old person with a strongly hunched upper back, the head pushed forward and down,
rounded shoulders"), and "clearly visible in profile" added to the profile.

Seeds kept:

| variant | front | back | profile |
|---|---|---|---|
| female heavy | 32 | 41 | 41 |
| female muscular | 32 | 42 | 41 |
| female elder | 32 | 42 | 41 |
| male heavy | 32 | 41 | 41 |
| male muscular | 31 | 42 | 42 |
| male elder | 31 | 41 | 41 |

Pixal3D as for the base bodies. A variant keeps its base body's crotch and axilla heights: touching thighs and
arms resting on a belly fool the scans for them.

```bash
uv run python scripts/human_sections.py female_heavy.glb src/manifoldx/modeling/assets/data/human_female_heavy.npz \
    --stature 1.63 --body female --landmarks-from src/manifoldx/modeling/assets/data/human_female.npz
```
