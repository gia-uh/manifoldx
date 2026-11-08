import manifoldx as mx
import manifoldx.viz as mxv
import numpy as np
from manifoldx.components import Material, Transform
from manifoldx.resources import PointLight, StandardMaterial, cube
from PIL import Image

HALF = 20.0
rng = np.random.default_rng(7)
N = 500
spots = np.column_stack(
    [
        rng.uniform(-19.0, 19.0, N),
        np.full(N, 0.10),
        rng.uniform(-19.0, 19.0, N),
    ]
).astype(np.float32)


def hotspot(cx, cz, sx, sz):
    """A smooth bump over the square, in metres."""
    dx = (spots[:, 0] - cx) / sx
    dz = (spots[:, 2] - cz) / sz
    return np.exp(-(dx**2 + dz**2))


visits = 0.4 + 4.5 * hotspot(-3, 2, 4, 3) + 2.2 * hotspot(10, -8, 3, 3) + 0.8 * hotspot(0, 14, 8, 2)
visits = np.clip(visits + rng.normal(0.0, 0.15, N), 0.0, None).astype(np.float32)

dwell = 30.0 + 90.0 * hotspot(-3, 2, 5, 4) + rng.normal(0.0, 5.0, N)
radius = (0.30 + np.clip(dwell, 10.0, 150.0) / 350.0).astype(np.float32)

visits_color = mxv.color(visits, cmap="turbo", domain=(0.0, 6.0), title="Visits / min")

chart = (
    mxv.points(
        positions=mxv.position(spots),
        color=visits_color,
        size=mxv.size(radius),
    )
    + mxv.mesh(
        geometry=cube(40, 0.2, 40),
        material=StandardMaterial("#5c5242", roughness=0.9),
        position=(0.0, -0.12, 0.0),
    )
    + mxv.mesh(
        geometry=cube(4, 1, 4),
        material=StandardMaterial("#9aa8b8", roughness=0.6),
        position=(0.0, 0.5, 0.0),
    )
    + mxv.lights(
        [
            PointLight(color="#fff3dd", intensity=1300.0, position=(0.0, 16.0, 4.0)),
            PointLight(color="#bcd4ff", intensity=800.0, position=(-14.0, 10.0, -12.0)),
        ]
    )
)

engine = chart.engine
engine.background_color = (0.32, 0.38, 0.48)
engine.camera.fit(25.0, center=(0.0, 1.5, 0.0), azimuth=45, elevation=32, margin=0.95)
print(type(chart).__name__, type(chart.marks[0]).__name__)
print(
    type(mxv.position(spots)).__name__,
    type(visits_color).__name__,
    type(mxv.size(radius)).__name__,
)
frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
cam = engine.camera
right = cam.get_right()
pixels, _ = cam.project([cam.target, cam.target + 5.0 * right], engine.w, engine.h)
five_metres = abs(pixels[1, 0] - pixels[0, 0]) * 2.0 / engine.w

chart = (
    chart
    + mxv.axes(extent=10.0)
    + mxv.legend(visits_color, position="top-right")
    + mxv.scale_bar(ndc_length=five_metres, label="5 m", position="bottom-left")
)
engine = chart.engine
engine.background_color = (0.32, 0.38, 0.48)
engine.camera.fit(25.0, center=(0.0, 1.5, 0.0), azimuth=45, elevation=32, margin=0.95)
frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
atlas = engine.get_label_atlas()
places = [
    (0.0, 2.4, 0.0, "Fountain"),
    (10.0, 2.8, -9.0, "Market"),
    (-11.0, 2.2, 9.0, "Bench row"),
]
slots = [atlas.get_or_create(place[3]) for place in places]
print("slots", slots, "of", atlas.slice_count)

for (x, y, z, _), slot in zip(places, slots):
    engine.spawn(
        Material(mxv.LabelMaterial(pixel_width=240, pixel_height=60)),
        Transform(pos=(x, y, z)),
        mxv.TextLabel(index=slot),
        n=1,
    )

frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
NZ, NY, NX = 32, 12, 32
zs = np.linspace(-HALF, HALF, NZ, dtype=np.float32)
ys = np.linspace(0.0, 10.0, NY, dtype=np.float32)
xs = np.linspace(-HALF, HALF, NX, dtype=np.float32)
Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")

noise = (
    44.0
    + 30.0 * np.exp(-(((X - 9.0) ** 2 + (Z + 6.0) ** 2) / 25.0))
    + 24.0 * np.exp(-(((X + 4.0) ** 2 + (Z - 2.0) ** 2) / 16.0))
) * np.exp(-Y / 5.0) + 44.0 * (1.0 - np.exp(-Y / 5.0))
noise = np.ascontiguousarray(noise.astype(np.float32))

handle = engine.register_volume(noise, name="noise")
engine.spawn(
    mxv.Volume(volume_id=handle),
    Material(
        mxv.VolumeMaterial(
            cmap="inferno",
            vmin=49.0,
            vmax=82.0,
            opacity_stops=[(0.0, 0.0), (0.35, 0.02), (0.7, 0.08), (1.0, 0.22)],
            density_scale=5.0,
            max_steps=128,
        )
    ),
    Transform(pos=(0.0, 5.0, 0.0), scale=(40.0, 10.0, 40.0)),
    n=1,
)

frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
