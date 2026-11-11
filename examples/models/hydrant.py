"""A classic red cartoon fire hydrant, built from its three blueprints.

A round barrel with a domed bonnet, a wide base flange, two side outlets and a front pumper. Its drawings are
committed beside it, so the build and the lint run with no network and no image model:

    uv run python examples/models/hydrant.py
"""

from manifoldx.asset import Parts, two_view


class Hydrant(Parts):
    desc = ("a classic red cartoon fire hydrant: a round barrel on a wide base flange, a domed bonnet on top "
            "with a pentagon nut, one short hose outlet on each side with a round cap, and one big pumper outlet "
            "with a round cap facing the front")
    style = "Clean cel-shaded cartoon style with flat colours and thin dark outlines, even light."
    height = 0.8
    model = "google/gemini-3.1-flash-image"
    core = "barrel"
    base = (200, 40, 35)

    notes = {
        "front": ("In this view the big pumper outlet points straight at the viewer, a round cap in the middle "
                  "of the barrel, and the two side outlets stick out to the left and to the right."),
        "back": ("In this view the pumper outlet is hidden behind the barrel and does not show; the two side "
                 "outlets stick out to the left and to the right."),
        "left": ("In this view the big pumper outlet sticks out toward the LEFT edge of the image, seen from the "
                 "side; one side outlet points straight at the viewer, a round cap in the middle of the barrel; "
                 "the other side outlet is hidden behind the barrel."),
    }
    details = {
        "front": ("glossy fire-engine red paint on the barrel, bonnet and base flange, a row of small bolts round "
                  "the base flange, the caps of the outlets silver-grey with a nut in the middle, a thin darker "
                  "red ring where the bonnet meets the barrel"),
        "back": ("glossy fire-engine red paint, a row of small bolts round the base flange, the side outlet caps "
                 "silver-grey seen from behind, a little worn paint near the bottom"),
        "left": ("glossy fire-engine red paint, the side outlet cap facing the viewer as a silver-grey disc with a "
                 "nut in the middle, the front pumper outlet sticking out to the left with a silver-grey cap, "
                 "bolts round the base flange"),
    }

    def parts(self, p):
        p.add("barrel", "orange", "the round barrel, the main body")
        p.add("bonnet", "yellow", "the domed bonnet on top, with its nut and the ring under it")
        p.add("base", "magenta", "the wide base flange at the bottom")
        p.add("side", "blue", "each side hose outlet with its cap",
              build=two_view(power=6.0, views=("front", "left")))
        p.add("pumper", "green", "the big front pumper outlet with its cap",
              build=two_view(power=6.0, keep="largest"))


if __name__ == "__main__":
    result = Hydrant().run(stages=("build", "lint"))
    print(result.report.text() or "no findings")
    triangles = sum(len(mesh.faces) for mesh in result.scene.meshes.values())
    print(f"\n{len(result.scene)} parts, {triangles} triangles")
