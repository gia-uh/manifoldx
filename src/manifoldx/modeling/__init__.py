"""Procedural geometric modeling: numpy-first Mesh value type + operator pipeline."""

from manifoldx.modeling.mesh import Mesh
from manifoldx.modeling.sculpt import Falloff
from manifoldx.modeling.ffd import FFD
from manifoldx.modeling.fields import Field
from manifoldx.modeling.gradient import Gradient
from manifoldx.modeling.gpu_fields import field_to_wgsl, WGSL_NOISE_PRELUDE, FieldNotTranspilable
from manifoldx.modeling.shape import Layer, Shape
from manifoldx.modeling import fields, noise, shape, trace

__all__ = ["Mesh", "Falloff", "FFD", "Field", "Gradient", "fields", "noise",
           "shape", "trace", "Layer", "Shape",
           "field_to_wgsl", "WGSL_NOISE_PRELUDE", "FieldNotTranspilable"]
