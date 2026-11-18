"""Vehicles as rigid bodies on ray wheels: suspension, tyres that grip and slide, engine, gearbox and brakes,
with driving assists on top. Numpy only, vectorised over N vehicles; no engine or ECS coupling."""

from .spec import Engine, Gearbox, Tyre, VehicleSpec, Wheel, box_inertia
from .state import Packed, pack

__all__ = ["Engine", "Gearbox", "Packed", "Tyre", "VehicleSpec", "Wheel", "box_inertia", "pack"]
