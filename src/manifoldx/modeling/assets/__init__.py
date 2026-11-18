"""Procedural assets: objects built from named, measurable parameters (cars first, every vehicle built alike)."""

from manifoldx.modeling.assets.car import Car, CarParams, Part, build_car
from manifoldx.modeling.assets.vehicle import Vehicle

__all__ = ["Car", "CarParams", "Part", "Vehicle", "build_car"]
