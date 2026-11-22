# People on the street

The plaza's walkers from chapter 2 are capsules on a skeleton. A street wants people: short and tall, thin and
heavy, young and old, each one different and all of them cheap enough to draw by the hundred.
`manifoldx.modeling.assets.human` builds them. A person is a `HumanParams`, a handful of numbers with ranges like
a car's, and `build_human` turns it into a `Figure`: one closed mesh, its rest joints and their parents.

## Where the bodies come from

A figure is never modelled by hand. Two base bodies, a woman and a man in an A-pose, were drawn by an image model
from four sides, turned into meshes by an image-to-mesh model, and read once into section tables: rings of radii
along each limb and the torso, and a grid of radii for the head. Each base body was then drawn again, edited to
look heavy, muscular and old, and those drawings were read into variant tables with the same rings. The tables
ship with the package; `load_table` reads any of them.

```python {run}
from manifoldx.modeling.assets.human_sections import load_table

for name in ("female", "male", "male_heavy", "female_elder"):
    t = load_table(name)
    print(f"{name:13s} stature {t.stature:.2f} m, crotch {t.crotch_y:.3f} m, axilla {t.axilla_y:.3f} m")
```

A table is a `SectionTable`: its stature, the crotch, axilla and acromion heights, the head grid, the rest joints
and one `Chain` per limb and the torso, an axis with the centre and radii of a ring at each station.
`human_skin.skin` lofts a table into a `Skin`, one closed mesh with each vertex's region and its rings:

```python {run continue}
from manifoldx.modeling.assets.human_skin import skin

torso = t.chains["torso"]
print("torso:", len(torso.t), "rings of", torso.r.shape[1], "radii")
s = skin(load_table("female"))
print("skin:", len(s.positions), "vertices,", len(s.faces), "faces, rings for", ", ".join(s.ring))
```

Every table has the same number of rings and radii, so any weighted sum of tables is a table too, and every one
lofts into the same faces. `build` runs from the female table (0) to the male one (1). `weight`, `muscle` and
`age` are blend weights of the variants: a weight of 1.5 is the whole heavy body, added to the base as its
difference ring by ring. A belly therefore grows forward and an old back bends where the drawings put them.

## A person

```python {run continue}
from manifoldx.modeling.assets.human import HumanParams, build_human

person = HumanParams(stature=1.68, build=0.3)
figure = build_human(person)
mesh = figure.mesh
print(len(mesh.positions), "vertices,", len(mesh.faces), "faces, height",
      round(float(mesh.positions[:, 1].max()), 3), "m")
print("joints:", ", ".join(list(figure.joints)[:6]), "...")
print("the knee hangs from", figure.parents["knee_l"])
```

The figure stands on y = 0 and faces +Z, its left side at +X, like every model in this guide. Its vertices carry
one skin tone from `SKIN_TONES`.

## Changing three numbers

`weight`, `age` and `shoulders` change the body in different ways. The waist's girth shows the first, the head's
lead over the pelvis the second, and the shoulder joint's distance from the middle the third:

```python {run continue}
import numpy as np

def waist(f):
    y = f.joints["spine"][1]
    band = f.mesh.positions[np.abs(f.mesh.positions[:, 1] - y) < 0.01]
    return float(np.ptp(band[:, 2]))

def lead(f):
    return float(f.joints["head"][2] - f.joints["pelvis"][2])

for label, p in [("as built", person), ("weight 1.4", HumanParams(stature=1.68, build=0.3, weight=1.4)),
                 ("elder", HumanParams(stature=1.68, build=0.3, age="elder")),
                 ("shoulders 1.15", HumanParams(stature=1.68, build=0.3, shoulders=1.15))]:
    f = build_human(p)
    print(f"{label:15s} waist depth {waist(f):.3f} m, head lead {lead(f):+.3f} m, "
          f"shoulder at x {f.joints['shoulder_l'][0]:.3f} m")
```

`stature` scales the whole body, `leg_ratio` moves the crotch up or down with the legs stretching below it, and
`hips` widens the pelvis the way `shoulders` widens the chest, both on the finished skin so the faces never change.

## A crowd from seeds

`HumanParams.sample(seed)` draws a plausible person: a stature around 1.63 m or 1.75 m by build, one person in
four old, every other number within the middle of its range. The same seed always gives the same person, so a
crowd is a list of seeds.

```python {run continue}
crowd = [HumanParams.sample(seed) for seed in range(6)]
for seed, p in enumerate(crowd):
    print(seed, f"{p.stature:.2f} m, build {p.build:.2f}, weight {p.weight:.2f}, {p.age}, {p.skin}")
```

## Clothes, hair and a face

`build_human(params, outfit)` dresses the figure in an `Outfit`: a top with sleeves as a fraction of the arm,
trousers, shorts or a skirt down a fraction of the leg, shoes, sandals or boots, a hair style, a cap or a brimmed
hat, their colours and an optional stripe across the top. Up close a garment is the body's own vertices over its
span, pushed out by its looseness and painted, so the skin under it is gone and the faces stay the same; a skirt,
a bun and a brim are closed shells of their own. The face is painted: eyes, brows and a mouth a darker shade of
the skin.

```python {run continue}
from manifoldx.modeling.assets.human_dress import Outfit

outfit = Outfit(top="shirt", sleeves=1.0, bottom="skirt", bottom_length=0.55, hair="bun",
                top_color="#2e86c1", bottom_color="#5d4037")
dressed = build_human(person, outfit)
print(len(dressed.mesh.faces), "faces, garments:", {k: len(v) for k, v in dressed.parts.items()})
far = build_human(person, outfit, level=1)
print("level 1:", len(far.mesh.faces), "faces, garments with shape:", sorted(far.parts))
```

Every vertex of a dressed figure still has two joints and a weight, so it poses the same way: the skirt follows
the hips at its waist and the knees at its hem. Farther away the garments are only painted on the skin; hair and
hats keep their shape at level 1 and become paint at level 2.

## Three levels

A person near the camera needs more faces than one across the street. `level` picks one of three resolutions,
all from the same table: 16 radii per ring and every station at level 0, 8 radii and every other station at
level 1, 6 radii and every fourth station at level 2. `detail` refines or coarsens level 0 alone.

```python {run continue}
p = crowd[0]
for level in (0, 1, 2):
    print("level", level, len(build_human(p, level=level).mesh.faces), "faces")
print("level 0 at detail 2:", len(build_human(HumanParams(detail=2.0)).mesh.faces), "faces")
```
