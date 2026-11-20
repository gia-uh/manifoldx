import numpy as np

# Export core modules and types
from manifoldx.engine import Engine
from manifoldx import backends
from manifoldx.types import Vector3, Vector4, Float, Color
from manifoldx.ecs import EntityStore, ComponentView
from manifoldx.systems import Query
from manifoldx.components import Transform, Mesh, Material, Colors, Layer, Tint, Fade
from manifoldx.camera import Camera
from manifoldx.input import KeyEvent, PointerEvent, WheelEvent, ResizeEvent
from manifoldx.resources import cube, sphere, plane, basic, phong, standard
from manifoldx.assets import load_obj
from manifoldx.textures import load_texture, texture_from_array, texture_from_svg, write_texture
from manifoldx.skinning import MAX_JOINTS
from manifoldx import compute  # noqa: E402, F401
from manifoldx import physics  # noqa: E402, F401
from manifoldx import random  # noqa: E402, F401
from manifoldx import viz  # noqa: E402, F401


# =============================================================================
# Module Proxies for mx.types, mx.components, mx.geometry, etc.
# =============================================================================


class _TypesProxy:
    """Proxy for mx.types module."""

    Vector3 = Vector3
    Vector4 = Vector4
    Float = Float
    Color = Color


class _ComponentsProxy:
    """Proxy for mx.components module."""

    Transform = Transform
    Mesh = Mesh
    Material = Material
    Colors = Colors


class _GeometryProxy:
    """Proxy for mx.geometry module."""

    def cube(self, w=1, h=1, d=1):
        return cube(w, h, d)

    def sphere(self, r=1, segments=32):
        return sphere(r, segments)

    def plane(self, w=1, h=1):
        return plane(w, h)


class _MaterialProxy:
    """Proxy for mx.material module."""

    def basic(self, color):
        return basic(color)

    def phong(self, color, shininess=32):
        return phong(color, shininess)

    def standard(self, color, roughness=0.5, metallic=0.0):
        return standard(color, roughness, metallic)


class _ColorsProxy:
    """Proxy for mx.colors module."""

    RED = "#ff0000"
    GREEN = "#00ff00"
    BLUE = "#0000ff"
    WHITE = "#ffffff"
    BLACK = "#000000"
    YELLOW = "#ffff00"
    CYAN = "#00ffff"
    MAGENTA = "#ff00ff"


# Create module instances
types = _TypesProxy()
components = _ComponentsProxy()
geometry = _GeometryProxy()
material = _MaterialProxy()
colors = _ColorsProxy()


__all__ = [
    "Engine",
    "backends",
    "Camera",
    "Vector3",
    "Vector4",
    "Float",
    "Color",
    "EntityStore",
    "ComponentView",
    "Query",
    "Transform",
    "Layer",
    "Tint",
    "Fade",
    "Mesh",
    "Material",
    "Colors",
    "cube",
    "sphere",
    "plane",
    "basic",
    "phong",
    "standard",
    "load_obj",
    "load_texture",
    "texture_from_array",
    "texture_from_svg",
    "write_texture",
    "MAX_JOINTS",
    "component",
    "types",
    "components",
    "geometry",
    "material",
    "colors",
    "compute",
    "physics",
    "random",
    "viz",
    "KeyEvent",
    "PointerEvent",
    "WheelEvent",
    "ResizeEvent",
]
