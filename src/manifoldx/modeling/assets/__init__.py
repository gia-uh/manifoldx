"""Procedural assets: objects built from named, measurable parameters (cars first, every vehicle built alike)."""

from manifoldx.modeling.assets.bus import Bus, BusParams, build_bus
from manifoldx.modeling.assets.car import Car, CarParams, Part, build_car
from manifoldx.modeling.assets.cycle import (
    Cycle,
    CycleParams,
    bicycle,
    build_cycle,
    cargo_tricycle,
    cycle_rickshaw,
    electric_moto,
    motorcycle,
)
from manifoldx.modeling.assets.vehicle import Vehicle

__all__ = ["Bus", "BusParams", "Car", "CarParams", "Cycle", "CycleParams", "Part", "Vehicle", "bicycle", "build_bus",
           "build_car", "build_cycle", "cargo_tricycle", "cycle_rickshaw", "electric_moto", "motorcycle"]
