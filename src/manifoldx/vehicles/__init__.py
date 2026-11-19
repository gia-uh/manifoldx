"""Vehicles as rigid bodies on ray wheels: suspension, tyres that grip and slide, engine, gearbox and brakes,
with driving assists on top. Numpy only, vectorised over N vehicles; no engine or ECS coupling."""

from .spec import Engine, Gearbox, Tyre, VehicleSpec, Wheel, box_inertia
from .assists import ASSISTS, apply_assists, strengths_for
from .fleet import Vehicles
from .presets import car_spec
from .ground import FlatGround, Ground, ShapeGround
from .state import Packed, pack

__all__ = ["car_spec", "ASSISTS", "apply_assists", "strengths_for", "Engine", "FlatGround", "Gearbox", "Ground", "ShapeGround", "Vehicles", "Packed", "Tyre", "VehicleSpec", "Wheel", "box_inertia", "pack"]
