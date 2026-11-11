"""A mid-century modern armchair, built from its three blueprints.

The asset of the prototype's armchair (`.playground/assetgen/armchair.json`): a walnut frame with four tapered
legs and flat armrests, a mustard-yellow seat cushion and a back cushion. Its drawings are committed beside it,
so the build and the lint run with no network and no image model:

    uv run python examples/models/armchair.py
"""

from manifoldx.asset import Parts, two_view


class Armchair(Parts):
    desc = ("a mid-century modern armchair: a walnut wooden frame with four tapered legs and flat wooden "
            "armrests, a thick mustard-yellow fabric seat cushion and a mustard-yellow fabric back cushion "
            "leaning back")
    style = "Photorealistic product photograph, soft studio light."
    height = 0.85
    model = "google/gemini-3.1-flash-image"
    core = "seat"
    base = (196, 150, 40)

    notes = {
        "front": ("In this view the armchair faces the viewer: the seat cushion in front, the back cushion behind "
                  "it, an armrest on each side, the two front legs in front of the two back legs."),
        "back": ("In this view the armchair is seen from behind: the back of the back cushion and frame, an "
                 "armrest on each side, the two back legs in front of the two front legs."),
        "left": ("In this view the armchair is seen from its left side, its front toward the LEFT edge of the "
                 "image: the back leaning back toward the right, one armrest in front, the front leg at the left "
                 "and the back leg at the right."),
    }
    details = {
        "front": ("mustard-yellow woven fabric on both cushions with piping along their edges and a few buttons on "
                  "the back cushion, warm walnut wood with visible grain on the armrests, frame and legs"),
        "back": ("the back of the frame in walnut wood with visible grain, the mustard-yellow fabric of the back "
                 "cushion showing at the top, walnut legs"),
        "left": ("walnut wood with visible grain on the armrest, its post, the side of the seat frame and the legs, "
                 "the mustard-yellow fabric of the cushions with piping along their edges"),
    }

    # The back view paints the back cushion over the seat, the arms and the legs, so those three trust only the
    # front and the left drawings; the arms and legs are moved the 2 cm their drawings leave to meet the frame.
    FRONT_LEFT = ("front", "left")

    def parts(self, p):
        p.add("seat", "orange", "the seat cushion with the wooden seat frame under it",
              build=two_view(power=5.0, views=self.FRONT_LEFT))
        p.add("back", "yellow", "the back: the back cushion and the whole wooden back frame, including its two "
                                "upright side posts from the seat up to the top", build=two_view(power=5.0))
        p.add("arms", "blue", "each flat wooden armrest with the single short post under its front end, and "
                              "nothing above the armrest",
              build=two_view(power=6.0, views=self.FRONT_LEFT, attach="back"))
        p.add("legs", "magenta", "each tapered wooden leg",
              build=two_view(power=2.0, grid=True, views=self.FRONT_LEFT, attach="seat"))


if __name__ == "__main__":
    result = Armchair().run(stages=("build", "lint"))
    print(result.report.text() or "no findings")
    triangles = sum(len(mesh.faces) for mesh in result.scene.meshes.values())
    print(f"\n{len(result.scene)} parts, {triangles} triangles")
